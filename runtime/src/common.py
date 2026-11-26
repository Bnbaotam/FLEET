#!/usr/bin/env python3
"""Shared helpers of the runtime pipeline: model input from tracker results, data loading and drawing of trajectories."""

import os
import time
import cv2
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import deque
import pickle
import tempfile

# ── Light imports only at module level ──────────────────────────────────────
# torch / ultralytics / Build_Model etc. are imported lazily inside functions.
import torch
import torch.nn.functional as F

from data.TP.environment import Environment, Scene, Node, derivative_of
from yacs.config import CfgNode

# ── Constants ────────────────────────────────────────────────────────────────
dt = 0.1

standardization = {
    "PEDESTRIAN": {
        "position": {
            "x": {"mean": 17.82, "std": 5.52},
            "y": {"mean":  8.93, "std": 1.44},
        },
        "velocity": {
            "x": {"mean":  0.25, "std": 2.14},
            "y": {"mean":  0.07, "std": 0.72},
        },
        "acceleration": {
            "x": {"mean":  0.04, "std": 6.93},
            "y": {"mean":  0.12, "std": 3.10},
        },
    }
}

# Single canonical definition (duplicate removed)
data_columns = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration"], ["x", "y"]]
)


# ── Perception helpers ────────────────────────────────────────────────────────

def make_df_from_results(results, frame_id: int, scene_id: str = "Georgie_NW") -> pd.DataFrame:
    """
    Turn a YOLO track() result into a per-frame DataFrame.

    Args:
        results:   Ultralytics Results object from model.track()
        frame_id:  Pipeline frame index (emitted_idx from camera_processor)
        scene_id:  Scene label written to the 'sceneId' column.
                   Defaults to "Georgie_NW"; pass the actual intersection name
                   when running on a different video.

    Returns:
        pd.DataFrame with columns:
            frame, trackId, x, y, bbox, cls_name, sceneId, metaId
        Empty DataFrame if no tracked boxes.
    """
    recs = []
    if results.boxes is None or results.boxes.id is None:
        return pd.DataFrame()

    class_names = results.names
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, tid, conf, cls_id = box
        recs.append({
            "frame":    frame_id,
            "trackId":  int(tid),
            "x":        (x1 + x2) / 2,
            "y":        (y1 + y2) / 2,
            "bbox":     [int(x1), int(y1), int(x2), int(y2)],
            "cls_name": class_names[int(cls_id)],
            "sceneId":  scene_id,
            "metaId":   int(tid),
        })
    return pd.DataFrame(recs)


def interpolate_window(df_window: pd.DataFrame) -> pd.DataFrame:
    """Interpolate gaps in tracking data within the sliding window."""
    out = []
    for tid, grp in df_window.groupby("trackId"):
        grp = grp.set_index("frame").sort_index()
        if grp.empty:
            continue
        full_idx = range(grp.index.min(), grp.index.max() + 1)
        grp_full = grp.reindex(full_idx)
        grp_full["sceneId"]  = grp_full["sceneId"].ffill().bfill()
        grp_full["metaId"]   = grp_full["metaId"].ffill().bfill()
        grp_full[["x", "y"]] = grp_full[["x", "y"]].interpolate()
        grp_full[["bbox", "cls_name"]] = grp_full[["bbox", "cls_name"]].ffill()
        grp_full["trackId"]  = tid
        grp_full = grp_full.reset_index().rename(columns={"index": "frame"})
        out.append(grp_full)

    if not out:
        return pd.DataFrame()

    return (
        pd.concat(out, ignore_index=True)
        .drop_duplicates(["frame", "trackId"])
        .sort_values(["frame", "trackId"])
        .reset_index(drop=True)
    )


def create_flowchain_enviroment(df: pd.DataFrame) -> "Environment":
    """Create the Environment object required by the prediction model."""
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=standardization)
    attention_radius = {
        (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 15.0
    }
    env.attention_radius = attention_radius
    scenes = []

    for scene_name, data in df.groupby("sceneId"):
        data = data.copy()
        data["frame"]    = pd.to_numeric(data["frame"],   downcast="integer")
        data["trackId"]  = pd.to_numeric(data["trackId"], downcast="integer")
        data["node_type"] = "PEDESTRIAN"
        data["node_id"]   = data["trackId"].astype(str)
        data["x"]         = data["x"] / 50
        data["y"]         = data["y"] / 50
        max_timesteps     = data["frame"].max()

        if len(data) > 0:
            scene = Scene(
                timesteps=max_timesteps + 1, dt=dt, name=scene_name, aug_func=None
            )
            for node_id in pd.unique(data["node_id"]):
                node_df = data[data["node_id"] == node_id]
                if len(node_df) <= 1:
                    continue
                node_values = node_df[["x", "y"]].values
                if node_values.shape[0] < 2:
                    continue
                new_first_idx = node_df["frame"].iloc[0]
                x  = node_values[:, 0]
                y  = node_values[:, 1]
                vx = derivative_of(x,  scene.dt)
                vy = derivative_of(y,  scene.dt)
                ax = derivative_of(vx, scene.dt)
                ay = derivative_of(vy, scene.dt)
                data_dict = {
                    ("position",     "x"): x,
                    ("position",     "y"): y,
                    ("velocity",     "x"): vx,
                    ("velocity",     "y"): vy,
                    ("acceleration", "x"): ax,
                    ("acceleration", "y"): ay,
                }
                node_data = pd.DataFrame(data_dict, columns=data_columns)
                node = Node(
                    node_type=env.NodeType.PEDESTRIAN,
                    node_id=node_id,
                    data=node_data,
                )
                node.first_timestep = new_first_idx
                scene.nodes.append(node)
            scenes.append(scene)

    env.scenes = scenes
    return env


# ── DataLoader (lazy heavy imports) ──────────────────────────────────────────

def custom_unified_loader(
    cfg: CfgNode,
    env: "Environment",
    rand: bool = True,
    split: str = "test",
    batch_size: int = None,
    data_fraction: float = 1.0,
):
    """Creates a DataLoader for the prediction model (offline/batch evaluation)."""
    # Lazy imports — only needed in offline eval, not in the live pipeline
    from torch.utils.data import DataLoader, Subset
    from data.TP.trajectron_dataset import EnvironmentDataset, hypers
    from data.TP.preprocessing import dict_collate as seq_collate

    dataset = EnvironmentDataset(
        env,
        state=hypers[cfg.DATA.TP.STATE],
        pred_state=hypers[cfg.DATA.TP.PRED_STATE],
        node_freq_mult=hypers["scene_freq_mult_train"],
        scene_freq_mult=hypers["node_freq_mult_train"],
        hyperparams=hypers,
        min_history_timesteps=(
            1
            if cfg.DATA.TP.ACCEPT_NAN and split == "train"
            else cfg.DATA.OBSERVE_LENGTH - 1
        ),
        min_future_timesteps=cfg.DATA.PREDICT_LENGTH,
    )

    pedestrian_dataset = None
    for node_type_dataset in dataset:
        if node_type_dataset.node_type == "PEDESTRIAN":
            pedestrian_dataset = node_type_dataset
            break

    if pedestrian_dataset is None:
        return None

    if data_fraction < 1.0:
        total_length  = len(pedestrian_dataset)
        subset_length = max(1, int(total_length * data_fraction))
        pedestrian_dataset = Subset(pedestrian_dataset, list(range(subset_length)))

    return DataLoader(
        pedestrian_dataset,
        batch_size=batch_size or cfg.DATA.BATCH_SIZE,
        shuffle=rand,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=seq_collate,
        drop_last=(split == "train"),
        pin_memory=True,
    )


# ── Visualization helpers ─────────────────────────────────────────────────────

def draw_circle_safe(img, pt, radius: int, color, thickness: int):
    """Draw a circle only if the coordinates are finite."""
    if (
        isinstance(pt, (tuple, list, np.ndarray))
        and len(pt) == 2
        and not (np.isnan(pt[0]) or np.isnan(pt[1]))
    ):
        cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), radius, color, thickness)


def draw_trajectories_standard(frame, vis_data, boxes_data):
    """
    Draw observed (green) and predicted (red) trajectories on the frame,
    plus bounding boxes and ID labels.
    """
    annotated_frame = frame.copy()
    color_pred = (0,   0, 255)   # red
    color_obs  = (0, 255,   0)   # green
    color_box  = (255, 100,  0)  # blue

    for obj_info in vis_data:
        obj_id  = obj_info["object_id"]
        obs_xy  = obj_info["obs_xy"].cpu().numpy()
        pred_seq = obj_info["pred_seq"].cpu().numpy()

        for pt in obs_xy:
            draw_circle_safe(annotated_frame, pt, 3, color_obs, -1)
        if len(obs_xy) > 1:
            cv2.polylines(annotated_frame, [obs_xy.astype(np.int32)], False, color_obs, 2)

        for pt in pred_seq:
            draw_circle_safe(annotated_frame, pt, 3, color_pred, -1)
        if len(pred_seq) > 1:
            cv2.polylines(annotated_frame, [pred_seq.astype(np.int32)], False, color_pred, 2)

        if obj_id in boxes_data:
            box_info = boxes_data[obj_id]
            x1, y1, x2, y2 = box_info["bbox"]
            cls_name = box_info["cls_name"]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_box, 2)
            label = f"ID: {obj_id} - {cls_name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), color_box, -1)
            cv2.putText(
                annotated_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

    return annotated_frame

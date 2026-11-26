#!/usr/bin/env python3
"""Prediction stage of the runtime pipeline: forecasts the trajectories of tracked objects with a pool of FlowChain workers."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_STDERR_THRESHOLD", "3")

import sys
import json
import time
import traceback
import argparse
import multiprocessing as mp
from collections import deque
from copy import deepcopy
from queue import Empty, Full
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from data.TP.environment import Environment, Node, Scene, derivative_of
from data.TP.preprocessing import dict_collate as seq_collate
from data.TP.trajectron_dataset import EnvironmentDataset, hypers
from utils import load_config
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics


T_O = 8
T_F = 12
QUEUE_GET_TIMEOUT = 0.1

KEEP_CLASSES = {0, 1, 2, 3, 5, 7}
KEEP_CLASS_NAMES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

# FALLBACK_STANDARDIZATION = {
#     "PEDESTRIAN": {
#         "position": {
#             "x": {"mean": 0, "std": 11.1232},
#             "y": {"mean": 1.6302, "std": 9.3534},
#         },
#         "velocity": {
#             "x": {"mean": 0, "std": 1.1417},
#             "y": {"mean": 0, "std": 0.8948},
#         },
#         "acceleration": {
#             "x": {"mean": 0.0384, "std": 1.5816},
#             "y": {"mean": 0, "std": 1.2824},
#         },
#     }
# }

FALLBACK_STANDARDIZATION = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 17.82, 'std': 5.52}, 
            'y': {'mean': 8.93, 'std': 1.44}
        }, 
        'velocity': {
        'x': {'mean': 0.25, 'std': 2.14},
        'y': {'mean': 0.07, 'std': 0.72}
        }, 
        'acceleration': {
        'x': {'mean': 0.04, 'std': 6.93}, 
        'y': {'mean': 0.12, 'std': 3.10}
        }
    }
}

class SlidingState:
    """Legacy stub — kept for backward-compat imports only."""
    def __init__(self, buffer_proxy=None, lock=None, last_emitted_end=None):
        self.buffer = buffer_proxy
        self.lock = lock
        self.last_emitted_end = last_emitted_end


class _RollingFPS:
    def __init__(self, window: int = 50):
        self._ts: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._ts.append(time.perf_counter())
        if len(self._ts) < 2:
            return 0.0
        dt = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / dt if dt > 0 else 0.0


def _filter_kept_classes(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    out = df.copy()

    if "cls_id" in out.columns:
        out = out[out["cls_id"].isin(KEEP_CLASSES)]
    elif "clsid" in out.columns:
        out = out[out["clsid"].isin(KEEP_CLASSES)]
    elif "cls_name" in out.columns:
        out = out[out["cls_name"].isin(KEEP_CLASS_NAMES)]
    elif "clsname" in out.columns:
        out = out[out["clsname"].isin(KEEP_CLASS_NAMES)]

    return out if not out.empty else None


def _get_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _to_int_track_id(v) -> Optional[int]:
    try:
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _get_standardization(metrics) -> dict:
    train_env = getattr(metrics, "env", None)
    if train_env is not None and getattr(train_env, "standardization", None):
        return deepcopy(train_env.standardization)
    return deepcopy(FALLBACK_STANDARDIZATION)


def _process_track_data(
    arr_xy: np.ndarray,
    scale: float,
    dt: float,
    data_columns: pd.MultiIndex,
) -> Optional[pd.DataFrame]:
    try:
        arr = np.asarray(arr_xy, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None

        x = arr[:, 0].astype(float) / scale
        y = arr[:, 1].astype(float) / scale

        vx = derivative_of(x, dt) if len(x) > 1 else np.zeros_like(x)
        vy = derivative_of(y, dt) if len(y) > 1 else np.zeros_like(y)
        ax = derivative_of(vx, dt) if len(vx) > 1 else np.zeros_like(vx)
        ay = derivative_of(vy, dt) if len(vy) > 1 else np.zeros_like(vy)

        data_dict = {
            ("position", "x"): x,
            ("position", "y"): y,
            ("velocity", "x"): vx,
            ("velocity", "y"): vy,
            ("acceleration", "x"): ax,
            ("acceleration", "y"): ay,
        }
        return pd.DataFrame(data_dict, columns=data_columns)
    except Exception:
        return None


def _build_track_histories(
    df_window: pd.DataFrame,
    latest_df: pd.DataFrame,
    latest_fidx: int,
    obs_len: int,
) -> Dict[int, np.ndarray]:
    """
    Build fixed-length x,y histories for tracks active in the latest frame.
    Missing timesteps inside the observation window are linearly interpolated.
    """
    track_col = _get_col(df_window, "trackId")
    frame_col = _get_col(df_window, "frame", "frame_idx")
    x_col = _get_col(df_window, "x")
    y_col = _get_col(df_window, "y")

    if not all([track_col, frame_col, x_col, y_col]):
        return {}

    latest_track_col = _get_col(latest_df, "trackId")
    if latest_track_col is None:
        return {}

    active_ids = []
    for v in latest_df[latest_track_col].tolist():
        tid = _to_int_track_id(v)
        if tid is not None:
            active_ids.append(tid)

    if not active_ids:
        return {}

    frame_range = np.arange(latest_fidx - obs_len + 1, latest_fidx + 1, dtype=int)
    histories: Dict[int, np.ndarray] = {}

    dfw = df_window.copy()
    dfw[track_col] = pd.to_numeric(dfw[track_col], errors="coerce")
    dfw[frame_col] = pd.to_numeric(dfw[frame_col], errors="coerce")
    dfw[x_col] = pd.to_numeric(dfw[x_col], errors="coerce")
    dfw[y_col] = pd.to_numeric(dfw[y_col], errors="coerce")
    dfw = dfw.dropna(subset=[track_col, frame_col, x_col, y_col])

    for tid in sorted(set(active_ids)):
        g = dfw[dfw[track_col] == tid].copy()
        if g.empty:
            continue

        g = g.sort_values(frame_col)
        g = g.drop_duplicates(subset=[frame_col], keep="last")
        g = g[[frame_col, x_col, y_col]].rename(
            columns={frame_col: "frame", x_col: "x", y_col: "y"}
        )
        g["frame"] = g["frame"].astype(int)
        g = g.set_index("frame").reindex(frame_range)

        if latest_fidx not in g.index:
            continue

        if pd.isna(g.loc[latest_fidx, "x"]) or pd.isna(g.loc[latest_fidx, "y"]):
            continue

        g[["x", "y"]] = g[["x", "y"]].interpolate(
            method="linear",
            limit_direction="both",
        )

        if g[["x", "y"]].isna().any().any():
            continue

        histories[tid] = g[["x", "y"]].to_numpy(dtype=float)

    return histories


def _create_env_from_histories(
    track_histories: Dict[int, np.ndarray],
    dt: float,
    scale: float,
    standardization: dict,
) -> Optional[Environment]:
    if not track_histories:
        return None

    env = Environment(
        node_type_list=["PEDESTRIAN"],
        standardization=standardization,
    )
    env.attention_radius = {
        (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 15.0,
    }

    scene = Scene(
        timesteps=T_O,
        dt=dt,
        name="intersection",
    )

    data_columns = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]]
    )

    for track_id, arr_xy in track_histories.items():
        node_data = _process_track_data(
            arr_xy=arr_xy,
            scale=scale,
            dt=dt,
            data_columns=data_columns,
        )
        if node_data is None:
            continue

        node = Node(
            node_type=env.NodeType.PEDESTRIAN,
            node_id=str(track_id),
            data=node_data,
        )
        node.first_timestep = 0
        scene.nodes.append(node)

    if not scene.nodes:
        return None

    env.scenes = [scene]
    return env


def _feeder_loop(q_phi: mp.Queue, q_snapshot: mp.Queue, stop: mp.Event, out_dir: Optional[str] = None):
    buf: deque = deque(maxlen=T_O)
    last_end = -1
    n_in, n_out, dropped = 0, 0, []                   # added
    print("[FEEDER] Started.")
    try:
        while not stop.is_set():
            try:
                pkt = q_phi.get(timeout=QUEUE_GET_TIMEOUT)
            except Empty:
                continue

            n_in += 1
            buf.append(pkt)
            if len(buf) < T_O:
                continue

            end = int(pkt["frame_idx"])
            if end <= last_end:
                continue

            last_end = end
            snapshot = list(buf)
            pkt["meas"]["t_window_ready"] = time.perf_counter()   # added
            try:
                q_snapshot.put(snapshot, timeout=QUEUE_GET_TIMEOUT)
                n_out += 1
            except Full:                                  # added: was a silent drop; now counted
                dropped.append(end)
                print(f"[FEEDER] q_snapshot full: dropped frame {end} (total {len(dropped)})", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if out_dir is not None:
            with open(os.path.join(out_dir, "feeder.json"), "w") as f:
                json.dump({"packets_in": n_in, "windows_out": n_out, "q_snapshot_dropped_frames": dropped}, f)

    print("[FEEDER] Stopped.")


def _latest_df_with_predictions(
    window_packets: List[Dict[str, Any]],
    cfg,
    model,
    metrics,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[str, float]]:
    timing: Dict[str, float] = {"prep_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}

    latest_pkt = window_packets[-1]
    latest_fidx = int(latest_pkt["frame_idx"])
    latest_df = _filter_kept_classes(latest_pkt.get("perception_df"))

    def _empty_return():
        out = latest_df.copy() if isinstance(latest_df, pd.DataFrame) else pd.DataFrame()
        if "pred_xy12" not in out.columns:
            out["pred_xy12"] = None
        return out, {}, timing

    if latest_df is None or latest_df.empty:
        return _empty_return()

    t0 = time.perf_counter()

    frames = []
    for p in window_packets:
        dfp = _filter_kept_classes(p.get("perception_df"))
        if dfp is not None and not dfp.empty:
            frames.append(dfp)

    if not frames:
        return _empty_return()

    df_window = pd.concat(frames, ignore_index=True)
    if df_window.empty:
        return _empty_return()

    scale = float(getattr(cfg.DATA, "SCALE", 50.0))
    dt = float(getattr(cfg.DATA, "DT", 0.1))
    standardization = _get_standardization(metrics)

    track_histories = _build_track_histories(
        df_window=df_window,
        latest_df=latest_df,
        latest_fidx=latest_fidx,
        obs_len=T_O,
    )
    if not track_histories:
        return _empty_return()

    env = _create_env_from_histories(
        track_histories=track_histories,
        dt=dt,
        scale=scale,
        standardization=standardization,
    )
    if env is None:
        return _empty_return()

    dataset = EnvironmentDataset(
        env,
        state=hypers[cfg.DATA.TP.STATE],
        pred_state=hypers[cfg.DATA.TP.PRED_STATE],
        node_freq_mult=hypers["scene_freq_mult_train"],
        scene_freq_mult=hypers["node_freq_mult_train"],
        hyperparams=hypers,
        min_history_timesteps=T_O - 1,
        min_future_timesteps=0,
    )

    pedestrian_dataset = None
    for node_type_dataset in dataset:
        if getattr(node_type_dataset, "node_type", None) == "PEDESTRIAN":
            pedestrian_dataset = node_type_dataset
            break
        if len(node_type_dataset) > 0 and pedestrian_dataset is None:
            pedestrian_dataset = node_type_dataset

    if pedestrian_dataset is None or len(pedestrian_dataset) == 0:
        return _empty_return()

    loader = DataLoader(
        pedestrian_dataset,
        batch_size=min(50, max(1, len(pedestrian_dataset))),
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate,
        drop_last=False,
        pin_memory=False,
    )

    try:
        batch = next(iter(loader))
    except StopIteration:
        return _empty_return()

    data_dict = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }

    timing["prep_ms"] = (time.perf_counter() - t0) * 1e3

    t1 = time.perf_counter()
    with torch.no_grad():
        result_dict = model.predict(deepcopy(data_dict), return_prob=False)
        result_denorm = metrics.denormalize([result_dict])[0]
    timing["infer_ms"] = (time.perf_counter() - t1) * 1e3

    t2 = time.perf_counter()
    pred_map: Dict[int, np.ndarray] = {}
    for i, pred in enumerate(result_denorm[("pred", 0)]):
        raw_tid = data_dict["index"][i][2].split("/")[-1]
        tid = _to_int_track_id(raw_tid)
        if tid is None:
            continue
        pred_map[tid] = pred.detach().cpu().numpy() * scale

    out = latest_df.copy()
    if "pred_xy12" not in out.columns:
        out["pred_xy12"] = None
    out["trackId"] = pd.to_numeric(out["trackId"], errors="coerce").astype("Int64")
    out["pred_xy12"] = out["trackId"].map(
        lambda tid: pred_map.get(int(tid), None) if pd.notna(tid) else None
    )

    if "frame" in out.columns:
        out = out[pd.to_numeric(out["frame"], errors="coerce") == latest_fidx].copy()

    timing["post_ms"] = (time.perf_counter() - t2) * 1e3
    return out, pred_map, timing


def _prediction_worker_loop(
    worker_id: int,
    cfg_path: str,
    ckpt_path: str,
    q_snapshot: mp.Queue,
    q_pred: mp.Queue,
    stop: mp.Event,
    pred_device: str = "cuda:0",
    save_frames: bool = False,
    perf_log_every: int = 50,
    ready_q: Optional[mp.Queue] = None,
    site: str = "",
    out_dir: Optional[str] = None,
):
    device = torch.device(pred_device if torch.cuda.is_available() else "cpu")
    stats: Dict[str, Any] = {"site": site, "worker": worker_id, "n_pred": 0,
                             "error_frames": [], "q_pred_dropped_frames": []}   # added

    try:
        cfg = load_config(
            argparse.Namespace(
                config_file=cfg_path,
                mode="test",
                gpu=str(device.index) if device.type == "cuda" else "",
            )
        )

        model = Build_Model(cfg).to(device).eval()

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        known = ("model_state_dict", "state_dict", "model", "state", "net")
        if isinstance(ckpt, dict):
            state_dict = None
            for k in known:
                if k in ckpt:
                    candidate = ckpt[k]
                    if isinstance(candidate, dict) and any(kk in candidate for kk in known):
                        for kk in known:
                            if kk in candidate:
                                candidate = candidate[kk]
                                break
                    state_dict = candidate
                    break
            if state_dict is None:
                state_dict = ckpt
        else:
            state_dict = ckpt

        if isinstance(state_dict, dict):
            state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

        model.load_state_dict(state_dict, strict=True)
        metrics = Build_Metrics(cfg)

        print(f"[P{worker_id}] Ready on {device}. save_frames={save_frames}")
        if ready_q is not None:                           # added: common start
            stats["t_ready"] = time.perf_counter()
            ready_q.put({"proc": f"worker{worker_id}", "site": site, "t": stats["t_ready"]})

        fps_tracker = _RollingFPS(window=perf_log_every)
        lat_prep = []
        lat_infer = []
        lat_total = []
        pred_count = 0

        while not stop.is_set():
            try:
                snapshot = q_snapshot.get(timeout=QUEUE_GET_TIMEOUT)
            except Empty:
                continue

            t_start = time.perf_counter()
            try:
                df_out, pred_map, timing = _latest_df_with_predictions(
                    snapshot, cfg, model, metrics, device
                )
            except Exception as e:
                print(f"[P{worker_id}] Prediction error: {e}")
                traceback.print_exc()
                stats["error_frames"].append(int(snapshot[-1]["frame_idx"]))   # added: counted
                continue

            total_ms = (time.perf_counter() - t_start) * 1e3
            latest_pkt = snapshot[-1]
            fidx = int(latest_pkt["frame_idx"])

            df_out = df_out.copy()
            df_out["frame_idx"] = fidx
            df_out["frame"] = fidx
            if "trackId" in df_out.columns:
                df_out = df_out.drop_duplicates(subset=["trackId"], keep="last")

            t_done = time.perf_counter()
            meas = dict(latest_pkt["meas"])               # added: carry the frame's stamps on
            meas.update(t_pred_start=t_start, t_pred_done=t_done, worker=worker_id,
                        n_pred=len(pred_map), prep_ms=timing["prep_ms"],
                        infer_ms=timing["infer_ms"], post_ms=timing["post_ms"])
            packet_out: Dict[str, Any] = {
                "frame_idx": fidx,
                "perception_df": df_out,
                "pred_t_emit": t_done,
                "cam_t_emit": latest_pkt.get("cam_t_emit", None),
                "frame_bgr": latest_pkt.get("frame_bgr", None) if save_frames else None,
                "meas": meas,
            }

            try:
                q_pred.put_nowait(packet_out)
            except Full:                                  # added: drop-oldest kept, but counted
                try:
                    stats["q_pred_dropped_frames"].append(int(q_pred.get_nowait()["frame_idx"]))
                except Empty:
                    pass
                try:
                    q_pred.put_nowait(packet_out)
                except Full:
                    stats["q_pred_dropped_frames"].append(fidx)

            pred_count += 1
            stats["n_pred"] = pred_count
            fps_tracker.tick()
            lat_prep.append(timing["prep_ms"])
            lat_infer.append(timing["infer_ms"])
            lat_total.append(total_ms)

            if pred_count % perf_log_every == 0 and lat_total:
                pa = np.array(lat_prep)
                ia = np.array(lat_infer)
                ta = np.array(lat_total)
                print(
                    f"[P{worker_id} PERF] pred {pred_count:6d} | "
                    f"FPS={fps_tracker.tick():5.2f} | "
                    f"prep mean={pa.mean():.1f}ms p95={np.percentile(pa,95):.1f}ms | "
                    f"infer mean={ia.mean():.1f}ms p95={np.percentile(ia,95):.1f}ms | "
                    f"total mean={ta.mean():.1f}ms p95={np.percentile(ta,95):.1f}ms"
                )
                lat_prep.clear()
                lat_infer.clear()
                lat_total.clear()

        print(f"[P{worker_id}] Stopped.")

    except KeyboardInterrupt:
        pass
    finally:                                              # added: errors now end the process with a non-zero code
        if out_dir is not None:
            with open(os.path.join(out_dir, f"worker{worker_id}.json"), "w") as f:
                json.dump(stats, f)


def start_prediction_pool(
    cfg_path: str,
    ckpt_path: str,
    q_phi: mp.Queue,
    q_snapshot: mp.Queue,
    q_pred: mp.Queue,
    shared=None,
    n_workers: int = 4,
    pred_device: str = "cuda:0",
    save_frames: bool = False,
    perf_log_every: int = 50,
    ready_q: Optional[mp.Queue] = None,
    site: str = "",
    out_dir: Optional[str] = None,
):
    stop = mp.Event()
    all_procs: List[mp.Process] = []

    feeder = mp.Process(
        target=_feeder_loop,
        args=(q_phi, q_snapshot, stop, out_dir),
        daemon=True,
        name="feeder",
    )
    feeder.start()
    all_procs.append(feeder)

    for wid in range(n_workers):
        p = mp.Process(
            target=_prediction_worker_loop,
            args=(
                wid,
                cfg_path,
                ckpt_path,
                q_snapshot,
                q_pred,
                stop,
                pred_device,
                save_frames,
                perf_log_every,
                ready_q,
                site,
                out_dir,
            ),
            daemon=True,
            name=f"worker-{wid}",
        )
        p.start()
        all_procs.append(p)

    return all_procs, stop
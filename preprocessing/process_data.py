#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data.TP.environment import Environment, Node, Scene, derivative_of  # noqa: E402  (the module path the pickles reference)
from summarize import clients_file  # noqa: E402

NODE = "PEDESTRIAN"
STATES = [("position", "x"), ("position", "y"), ("velocity", "x"), ("velocity", "y"), ("acceleration", "x"), ("acceleration", "y")]
DT = 0.1
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


# ---- the processing of the paper (unchanged) ----
def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


def process_file(df: pd.DataFrame, standardization, is_train: bool, rotations: bool = True):
    """The Environment of one raw file. standardization: what goes into the env (None for the constants pass of
    compute_standardization.py); rotations=False skips the 24 rotated copies of a training scene (constants pass only: they do
    not change the nodes)."""
    dt = DT
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 15.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")
    for scene_id, data in group:
        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')
        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['trackId'].astype(str)

        # scale as PECNet
        data['x'] = data['x'] / 50
        data['y'] = data['y'] / 50

        max_timesteps = data['frame'].max()

        if len(data) > 0:
            scene = Scene(
                timesteps=max_timesteps + 1,
                dt=dt,
                name=scene_id,
                aug_func=augment if is_train else None
            )

            for node_id in pd.unique(data['node_id']):
                node_df = data[data['node_id'] == node_id]

                if len(node_df) > 1:
                    if not np.all(np.diff(node_df['frame']) == 1):
                        print(f"  WARNING: Non-consecutive frames for node {node_id} in scene {scene_id}, skipping.")
                        continue

                    node_values = node_df[['x', 'y']].values
                    if node_values.shape[0] < 2:
                        continue

                    new_first_idx = node_df['frame'].iloc[0]
                    x = node_values[:, 0]
                    y = node_values[:, 1]
                    vx = derivative_of(x, scene.dt)
                    vy = derivative_of(y, scene.dt)
                    ax = derivative_of(vx, scene.dt)
                    ay = derivative_of(vy, scene.dt)

                    data_dict = {
                        ('position', 'x'): x, ('position', 'y'): y,
                        ('velocity', 'x'): vx, ('velocity', 'y'): vy,
                        ('acceleration', 'x'): ax, ('acceleration', 'y'): ay
                    }

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    node = Node(
                        node_type=env.NodeType.PEDESTRIAN,
                        node_id=node_id,
                        data=node_data
                    )
                    node.first_timestep = new_first_idx
                    scene.nodes.append(node)

            if is_train and rotations:
                scene.augmented = list()
                angles = np.arange(0, 360, 15)
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, angle))

            scenes.append(scene)

    env.scenes = scenes
    assert scenes, "no scenes"
    return env
# ---- end of the processing of the paper ----


def read_raw(raw_dir: Path, camera: str, split: str) -> pd.DataFrame:
    """<raw_dir>/<camera>_<split>.pkl, a raw trajectory file of preprocess_tracks.py."""
    with open(Path(raw_dir) / f"{camera}_{split}.pkl", "rb") as f:
        df = pickle.load(f)
    assert isinstance(df, pd.DataFrame) and list(df.columns) == ["frame", "trackId", "x", "y", "sceneId", "metaId"], (camera, split)
    assert len(df) and np.isfinite(df[["x", "y"]].to_numpy()).all(), (camera, split)
    return df


def check_standardization(std: dict) -> dict:
    """std = {entity: {dim: {"mean", "std"}}} for the six STATES, finite, std > 0, mean >= 0; returns it as env.standardization."""
    assert sorted(std) == sorted({e for e, _ in STATES}) and all(sorted(std[e]) == ["x", "y"] for e in std), std
    for e, d in STATES:
        v = std[e][d]
        assert sorted(v) == ["mean", "std"] and all(isinstance(v[k], float) and np.isfinite(v[k]) for k in v), (e, d, v)
        assert v["std"] > 0 and v["mean"] >= 0, (e, d, v)
    return {NODE: std}


def camera_standardization(constants: dict, cameras: list[str]) -> dict:
    """camera -> env.standardization, from a JSON of compute_standardization.py (pooled or per camera)."""
    assert sorted(constants) in (["cameras", "n_rows", "standardization"], ["per_camera"]), sorted(constants)
    if "per_camera" in constants:
        missing = sorted(set(cameras) - set(constants["per_camera"]))
        assert not missing, f"the per-camera constants lack {missing}"
        return {c: check_standardization(constants["per_camera"][c]["standardization"]) for c in cameras}
    std = check_standardization(constants["standardization"])
    return {c: std for c in cameras}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True, help="folder with <camera>_{train,test}.pkl of step 1")
    ap.add_argument("--clients", type=Path, required=True, help="clients file: the cameras to process")
    ap.add_argument("--standardization", type=Path, required=True, help="JSON of compute_standardization.py")
    ap.add_argument("--out_dir", type=Path, required=True)
    a = ap.parse_args()
    assert augment.__module__ == "__main__", "run as a script: dill stores aug_func by value only from __main__"
    np.random.seed(123)
    cameras = clients_file(a.clients)
    std = camera_standardization(json.loads(a.standardization.read_text()), cameras)
    missing = [f"{c}_{s}.pkl" for c in cameras for s in ("train", "test") if not (a.raw_dir / f"{c}_{s}.pkl").is_file()]
    assert not missing, f"missing in {a.raw_dir}: {missing}"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        for split in ("train", "test"):
            env = process_file(read_raw(a.raw_dir, cam, split), std[cam], split == "train")
            with open(a.out_dir / f"{cam}_{split}.pkl", "wb") as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print(f"{a.out_dir / f'{cam}_{split}.pkl'}: {sum(len(s.nodes) for s in env.scenes)} tracks", flush=True)
            del env


if __name__ == "__main__":
    main()

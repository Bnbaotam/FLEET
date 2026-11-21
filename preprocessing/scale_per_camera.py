#!/usr/bin/env python3
"""Data of Appendix A.4: the processed files of the 23-view setting with one scale factor per camera view.

    python preprocessing/scale_per_camera.py --clients configs/clients_23.txt              # all 23 views
    python preprocessing/scale_per_camera.py --clients configs/clients_23_chattanooga.txt  # the 10 distributed views

Rule: k_i = sqrt((SD_i(vx)^2 + SD_i(vy)^2) / (SD_pool(vx)^2 + SD_pool(vy)^2)), with SD_i / SD_pool the population SDs (ddof 0) of the
velocity rows (x and y) of camera i's / of all 23 cameras' pooled processed TRAINING files (the nodes of the scenes, not their rotated
copies). configs/scale_per_camera_23.csv holds the factors (columns camera, sd_vx, sd_vy, sd_pool_vx, sd_pool_vy, k). Each output
<camera>_{train,test}.pkl is a copy of the processed file in which only the velocity and acceleration stds (x and y) and the attention
radius are multiplied by k_i; the means and the position stds stay. The model input of camera i is then that of its frame shrunk by
k_i; the metric de-standardizes with the same constants, so errors stay in pixels. The scene graph uses the attention radius, so the
neighbour radius of camera i becomes 15 k_i (15 units of the rescaled frame).

Checks, before any file is written: for every camera of the clients file, SD_i(vx) and SD_i(vy) recomputed from its training file
equal the table; when the clients file lists all cameras of the table, the recomputed pooled SDs and every recomputed k equal the
table as well (the table is reproduced from the data). While writing: every input file carries the same standardization and the
attention radius 15. The factors applied are those of the table, so the files of the distributed views can be built without the
other views.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import dill
import numpy as np
import pandas as pd

from process_data import NODE, ROOT, STATES, clients_file

COLUMNS = ["camera", "sd_vx", "sd_vy", "sd_pool_vx", "sd_pool_vy", "k"]
VELOCITY = [STATES.index(("velocity", d)) for d in ("x", "y")]
SCALED = [("velocity", "x"), ("velocity", "y"), ("acceleration", "x"), ("acceleration", "y")]
RADIUS = 15.0  # the attention radius process_data.py stores
RTOL = 1e-12  # recomputed SDs and factors vs the table


def read_env(path: Path):
    with open(path, "rb") as f:  # as src/data/unified_loader.load_env
        return dill.load(f, encoding="latin1")


def write_env(env, path: Path) -> None:
    with open(path, "wb") as f:
        dill.dump(env, f, protocol=4)  # the protocol of the files used in the paper


def check_files(d: Path, cameras: list[str]) -> None:
    """Raise unless <d>/<camera>_{train,test}.pkl exist for every camera, listing the missing ones."""
    missing = [f"{c}_{s}.pkl" for c in cameras for s in ("train", "test") if not (d / f"{c}_{s}.pkl").is_file()]
    assert not missing, f"{len(missing)} of the {2 * len(cameras)} processed files are missing in {d}: {missing}"


def velocity_rows(env) -> np.ndarray:
    """Velocity x and y of every node row of the env's scenes (not the rotated copies), [rows, 2]."""
    nodes = [n for s in env.scenes for n in s.nodes]
    assert nodes and all(list(n.data.header) == STATES for n in nodes), "no nodes or an unexpected node state header"
    rows = np.concatenate([n.data.data[:, VELOCITY] for n in nodes])
    assert rows.ndim == 2 and rows.shape[1] == 2 and np.isfinite(rows).all(), rows.shape
    return rows


def scale_factor(sd_vx, sd_vy, sd_pool_vx, sd_pool_vy):
    k = np.sqrt((np.square(sd_vx) + np.square(sd_vy)) / (np.square(sd_pool_vx) + np.square(sd_pool_vy)))
    assert np.isfinite(k).all() and (k > 0).all(), k
    return k


def read_table(path: Path) -> pd.DataFrame:
    """The factor table indexed by camera, checked: its columns, unique cameras, one pooled SD pair, k = scale_factor of its SDs."""
    t = pd.read_csv(path, float_precision="round_trip")
    assert list(t.columns) == COLUMNS and t.camera.is_unique and len(t), (path, list(t.columns))
    assert t.sd_pool_vx.nunique() == 1 and t.sd_pool_vy.nunique() == 1, f"{path}: the pooled SDs differ between rows"
    np.testing.assert_allclose(t.k, scale_factor(t.sd_vx, t.sd_vy, t.sd_pool_vx, t.sd_pool_vy), rtol=RTOL, atol=0, err_msg=str(path))
    return t.set_index("camera")


def check_table(processed_dir: Path, cameras: list[str], table: pd.DataFrame) -> None:
    """The checks of the module docstring on the training files (raises on a mismatch)."""
    rows = {}
    for cam in cameras:
        rows[cam] = velocity_rows(read_env(processed_dir / f"{cam}_train.pkl"))
        np.testing.assert_allclose(rows[cam].std(axis=0), table.loc[cam, ["sd_vx", "sd_vy"]].to_numpy(float), rtol=RTOL, atol=0,
                                   err_msg=f"{cam}: velocity SDs of the training file differ from the table")
        print(f"[scale_per_camera] {cam}: {len(rows[cam])} training rows, SDs as in the table", flush=True)
    if set(cameras) != set(table.index):
        print(f"[scale_per_camera] {len(cameras)} of the {len(table)} cameras of the table: pooled SDs not recomputed", flush=True)
        return
    pool = np.concatenate([rows[c] for c in table.index]).std(axis=0)
    np.testing.assert_allclose(pool, table[["sd_pool_vx", "sd_pool_vy"]].to_numpy(float)[0], rtol=RTOL, atol=0,
                               err_msg="pooled velocity SDs differ from the table")
    sd = np.stack([rows[c].std(axis=0) for c in table.index])
    np.testing.assert_allclose(scale_factor(sd[:, 0], sd[:, 1], *pool), table.k.to_numpy(float), rtol=RTOL, atol=0,
                               err_msg="recomputed factors differ from the table")
    print(f"[scale_per_camera] all {len(table)} cameras: pooled SDs and factors as in the table", flush=True)


def scaled_env(env, k: float):
    """Copy of env with the velocity and acceleration stds (x and y) and the attention radius multiplied by k."""
    assert list(env.attention_radius) == [(NODE, NODE)] and env.attention_radius[(NODE, NODE)] == RADIUS, env.attention_radius
    out = copy.deepcopy(env)
    for e, d in SCALED:
        out.standardization[NODE][e][d]["std"] *= k
    out.attention_radius = {key: r * k for key, r in env.attention_radius.items()}  # keyed by env's node types, as in the paper's files
    for e, d in STATES:
        want = env.standardization[NODE][e][d]["std"] * (k if (e, d) in SCALED else 1.0)
        assert out.standardization[NODE][e][d] == {"mean": env.standardization[NODE][e][d]["mean"], "std": want}, (e, d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", type=Path, default=ROOT / "data" / "processed_23", help="processed files of the 23-view setting")
    ap.add_argument("--clients", type=Path, required=True, help="clients file: the cameras to scale")
    ap.add_argument("--table", type=Path, default=ROOT / "configs" / "scale_per_camera_23.csv")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "data" / "processed_23_scaled")
    a = ap.parse_args()
    assert a.out_dir.resolve() != a.processed_dir.resolve(), "--out_dir must differ from --processed_dir"
    table = read_table(a.table)
    cameras = clients_file(a.clients)
    assert set(cameras) <= set(table.index), f"cameras missing from {a.table}: {sorted(set(cameras) - set(table.index))}"
    check_files(a.processed_dir, cameras)
    check_table(a.processed_dir, cameras, table)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stored = None
    for cam in cameras:
        k = float(table.k[cam])
        for split in ("train", "test"):
            env = read_env(a.processed_dir / f"{cam}_{split}.pkl")
            stored = env.standardization if stored is None else stored
            assert env.standardization == stored, f"{cam}_{split}.pkl: standardization differs from {cameras[0]}_train.pkl"
            write_env(scaled_env(env, k), a.out_dir / f"{cam}_{split}.pkl")
        print(f"[scale_per_camera] wrote {cam} (k = {k:.4f}) to {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()

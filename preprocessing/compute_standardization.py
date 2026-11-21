#!/usr/bin/env python3
"""Step 2 of the data preparation: the standardization constants (mean, SD) that step 3 (process_data.py) stores in the files.

    python preprocessing/compute_standardization.py --raw_dir data/raw_trajectories --clients configs/clients_06.txt \
        --out configs/standardization_06.json
    python preprocessing/compute_standardization.py --raw_dir data/raw_trajectories --clients configs/clients_06.txt \
        --per_camera --out configs/standardization_per_camera.json

Inputs: the TRAINING raw trajectory files <raw_dir>/<camera>_train.pkl of the cameras of the clients file (the test files are
not read).
Rule: over every node row of these files, built as process_data.py builds the nodes (tracks with consecutive frames and more
than one row; positions = pixels / 50), the mean and the population SD (ddof 0) of position, velocity and acceleration in x
and y, with velocity = np.gradient(position, 0.1) and acceleration = np.gradient(velocity, 0.1) per node; a negative mean is
set to 0; every value is rounded to 6 decimals. Pooled (default): one set of constants over all listed cameras.
--per_camera: one set per camera, each from that camera's own training file.
Output (JSON): pooled {"cameras": [...], "n_rows": N, "standardization": {"position": {"x": {"mean": .., "std": ..}, "y": ..},
"velocity": .., "acceleration": ..}}; per camera {"per_camera": {"<camera>": {"n_rows": N, "standardization": {...}}, ...}}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from process_data import NODE, STATES, clients_file, process_file, read_raw


def pooled_constants(envs: list) -> pd.DataFrame:
    """The rule of the module docstring over every node row of envs: one row per state (entity, dim) with the raw mean / SD and the
    stored values (negative mean -> 0, rounded to 6 decimals)."""
    cols = {s: [] for s in STATES}
    for env in envs:
        for scene in env.scenes:
            for node in scene.nodes:
                h = list(node.data.header)
                assert h == STATES, h
                for d in ("x", "y"):
                    p = node.data.data[:, h.index(("position", d))]
                    assert p.size >= 2 and np.isfinite(p).all()
                    v = np.gradient(p, scene.dt)
                    cols["position", d].append(p)
                    cols["velocity", d].append(v)
                    cols["acceleration", d].append(np.gradient(v, scene.dt))
    rows = []
    for (entity, dim), parts in cols.items():
        a = np.concatenate(parts)
        mean, sd = float(a.mean()), float(a.std(ddof=0))
        rows.append({"entity": entity, "dim": dim, "n_rows": a.size, "mean_raw": mean, "std_raw": sd, "mean_set_to_0": mean < 0,
                     "mean": 0.0 if mean < 0 else round(mean, 6), "std": round(sd, 6)})
    return pd.DataFrame(rows)


def as_standardization(c: pd.DataFrame) -> dict:
    """The env.standardization dict of one camera's (or set's) constants table."""
    assert [tuple(r) for r in c[["entity", "dim"]].itertuples(index=False)] == STATES, c[["entity", "dim"]]
    out = {}
    for r in c.itertuples():
        out.setdefault(r.entity, {})[r.dim] = {"mean": float(r.mean), "std": float(r.std)}
    return {NODE: out}


def constants(raw_dir: Path, cameras: list[str]) -> dict:
    """{"n_rows", "standardization"} over the training files of `cameras`."""
    c = pooled_constants([process_file(read_raw(raw_dir, cam, "train"), None, is_train=True, rotations=False) for cam in cameras])
    assert c.n_rows.nunique() == 1, c
    return {"n_rows": int(c.n_rows.iloc[0]), "standardization": as_standardization(c)[NODE]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", type=Path, required=True, help="folder with <camera>_train.pkl of step 1")
    ap.add_argument("--clients", type=Path, required=True, help="clients file: the cameras whose training files are used")
    ap.add_argument("--out", type=Path, required=True, help="JSON file to write")
    ap.add_argument("--per_camera", action="store_true", help="one set of constants per camera instead of one pooled set")
    a = ap.parse_args()
    cameras = clients_file(a.clients)
    missing = [c for c in cameras if not (a.raw_dir / f"{c}_train.pkl").is_file()]
    assert not missing, f"no <camera>_train.pkl in {a.raw_dir} for {missing}"
    out = {"per_camera": {c: constants(a.raw_dir, [c]) for c in cameras}} if a.per_camera else {"cameras": cameras, **constants(a.raw_dir, cameras)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.out}: {len(cameras)} cameras" + (" (per camera)" if a.per_camera else f", {out['n_rows']} rows"))


if __name__ == "__main__":
    main()

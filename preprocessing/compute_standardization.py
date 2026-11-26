#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from process_data import NODE, STATES, clients_file, process_file, read_raw


def pooled_constants(envs: list) -> pd.DataFrame:
    """The standardization constants over every node row of envs: one row per state (entity, dim) with the raw mean / SD and the
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
    ap = argparse.ArgumentParser()
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

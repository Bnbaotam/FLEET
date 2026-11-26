#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from process_data import ROOT, clients_file
from scale_per_camera import check_files, read_env, write_env


def federation_constants(d: Path) -> tuple[dict, dict, set[str]]:
    """The standardization and attention radius every <view>_{train,test}.pkl of d holds (asserted identical), and the views of d."""
    names = sorted(p.name for p in d.glob("*.pkl"))
    views = {n.removesuffix("_train.pkl") for n in names if n.endswith("_train.pkl")}
    assert views and names == sorted(f"{v}_{s}.pkl" for v in views for s in ("train", "test")), f"{d}: not train/test pairs: {names}"
    st = radius = None
    for n in names:
        env = read_env(d / n)
        st, radius = (env.standardization, env.attention_radius) if st is None else (st, radius)
        assert env.standardization == st and env.attention_radius == radius, f"{n}: standardization or attention radius differs from {names[0]}"
    return st, radius, views


def unseen_env(env, st: dict, radius: dict):
    """Copy of env with its standardization replaced by st."""
    assert env.attention_radius == radius, (env.attention_radius, radius)
    assert env.standardization != st, "the input file already carries the federation's constants"
    out = copy.deepcopy(env)
    out.standardization = copy.deepcopy(st)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=ROOT / "data" / "processed_23", help="processed files of the 23-view setting")
    ap.add_argument("--constants_dir", type=Path, default=ROOT / "data" / "processed_06", help="processed files of the federation")
    ap.add_argument("--clients", type=Path, default=ROOT / "configs" / "clients_17.txt", help="clients file: the views to write")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "data" / "processed_17_unseen")
    a = ap.parse_args()
    assert a.out_dir.resolve() not in (a.processed_dir.resolve(), a.constants_dir.resolve()), "--out_dir must differ from the input folders"
    views = clients_file(a.clients)
    check_files(a.processed_dir, views)
    st, radius, members = federation_constants(a.constants_dir)
    assert not set(views) & members, f"views of the federation in {a.clients}: {sorted(set(views) & members)}"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        for split in ("train", "test"):
            write_env(unseen_env(read_env(a.processed_dir / f"{view}_{split}.pkl"), st, radius), a.out_dir / f"{view}_{split}.pkl")
        print(f"[unseen_views] wrote {view} to {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()

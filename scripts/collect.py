#!/usr/bin/env python3
"""Concatenate the server-written metrics.csv / rounds.csv of every run in runs/index.jsonl.

    python scripts/collect.py [--include_smoke] [--out data/derived]

Writes <out>/all_runs.csv (one row per run x round x phase x split x intersection) and <out>/all_rounds.csv, with the
columns tag and git_commit from the index added. Runs tagged smoke-* are skipped unless --include_smoke; a listed run
without metrics.csv or rounds.csv is an error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NAMES = ("metrics", "rounds")


def collect(runs_dir: Path, out_dir: Path, include_smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {name: [] for name in NAMES}
    for rec in (json.loads(l) for l in (runs_dir / "index.jsonl").read_text().splitlines() if l.strip()):
        if rec["tag"].startswith("smoke-") and not include_smoke:
            continue
        run_dir = runs_dir / rec["run_id"]
        missing = [n for n in NAMES if not (run_dir / f"{n}.csv").is_file()]
        if missing:
            raise FileNotFoundError(f"run {rec['run_id']} (tag {rec['tag']}) has no {missing} in {run_dir}")
        for name in NAMES:
            df = pd.read_csv(run_dir / f"{name}.csv")
            assert (df.run_id == rec["run_id"]).all(), (run_dir, name)
            frames[name].append(df.assign(tag=rec["tag"], git_commit=rec["git_commit"]))
    out = []
    for name in NAMES:
        assert frames[name], f"no runs with {name}.csv found in {runs_dir}"
        assert all(list(f.columns) == list(frames[name][0].columns) for f in frames[name]), f"{name}.csv column sets differ between runs"
        out.append(pd.concat(frames[name], ignore_index=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    out[0].to_csv(out_dir / "all_runs.csv", index=False)
    out[1].to_csv(out_dir / "all_rounds.csv", index=False)
    return out[0], out[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include_smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "derived")
    a = ap.parse_args()
    runs, rounds = collect(ROOT / "runs", a.out, a.include_smoke)
    print(f"[collect] {runs.run_id.nunique()} runs, {len(runs)} metric rows, {len(rounds)} round rows -> {a.out}")


if __name__ == "__main__":
    main()

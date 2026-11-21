#!/usr/bin/env python3
"""Centralized baseline: one model trained on the pooled train splits of all intersections (not federated).

    python scripts/run_central.py --clients 06 --seed 1
    python scripts/run_central.py --clients 06 --seed 0 --rounds 2 --max_batches 2 --eval_every 1   # smoke test

One process (strategies/central.py) on --gpu. "Round" = 1 epoch over the pooled set (batch 128, shuffled, drop_last) with
fl_local_train, so --rounds 20 matches FedAvg's compute. Same init as the FL runs. The model is evaluated on every site's test
split every --eval_every epochs and at the last epoch (evaluating all sites is expensive), so metrics.csv / rounds.csv have only
those rounds; the final one is --rounds. Output format = run_fl.py's (strategy "central", tag central-<NN>c); every site folder
holds a copy of the one final_model.pth and its own per_sample_final.csv. phase_wall_s = that epoch's training + all-site
evaluation wall time; aggregate_time_s = 0. run.json's note records the pooled number of train windows.
--data_dir as in run_fl.py (default data/processed_<NN>).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_fl import CONFIG, add_data_dir_arg, check_data, data_dir, finish, new_stage, spawn, wait_all, worker_env  # noqa: E402
from summarize import read_clients  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", required=True, choices=["06", "23"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=20, help="epochs over the pooled set")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_batches", type=int, default=None, help="smoke tests only; tags the run 'smoke-...'")
    add_data_dir_arg(ap)
    ap.add_argument("--note", default="")
    return ap


def main() -> int:
    a = parser().parse_args()

    clients = read_clients(a.clients)
    data = data_dir(a)
    check_data(data, clients, "Build them with preprocessing/process_data.py or pass --data_dir.")
    tag = f"central-{a.clients}c"
    tag = tag if a.max_batches is None else f"smoke-{tag}"
    run_id, stage = new_stage(tag, a.seed)
    smoke = [] if a.max_batches is None else ["--max_batches", str(a.max_batches)]
    start = time.time()
    procs = {"central": spawn(stage, worker_env(a.gpu), "strategies.central", "central", [
        "--config_file", str(ROOT / CONFIG), "--data_dir", str(data), "--run_dir", str(stage), "--run_id", run_id,
        "--sites", *clients, "--seed", str(a.seed), "--rounds", str(a.rounds), "--eval_every", str(a.eval_every), *smoke])}
    print(f"[run_central] {tag} seed={a.seed}: started; log in {stage}/central/stdout.log", flush=True)
    if not wait_all(procs, stage):
        return 1
    metrics = pd.read_csv(stage / "metrics.csv")
    pooled = int(metrics[metrics["round"] == a.rounds].n_train_windows.sum())  # sites' train windows are disjoint
    finish(stage, run_id, tag, a.seed, a.rounds, clients, start, f"{a.note} pooled_train_windows={pooled} eval_every={a.eval_every}".strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())

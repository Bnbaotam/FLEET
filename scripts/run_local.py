#!/usr/bin/env python3
"""Local-only baseline: every intersection trains its own model on its own train split, no communication.

    python scripts/run_local.py --clients 06 --seed 1
    python scripts/run_local.py --clients 06 --seed 1 --per_camera                       # each camera with its own constants
    python scripts/run_local.py --clients 06 --seed 0 --rounds 2 --max_batches 2         # smoke test
    python scripts/run_local.py --clients 06 --seed 1 --init_from <fedavg run id> --rounds 5   # global model + local fine-tuning

One process per intersection (strategies/local.py), all on --gpu. "Round" r = 1 local epoch (the FedAvg client's
fl_local_train) + test evaluation, so 20 rounds match FedAvg's compute. Same init as the FL runs (the server-built model),
client i seeded 100 * seed + i. Output format = run_fl.py's (strategy "local", tag local-<NN>c[-r<rounds> if --rounds is not 20]),
with only evaluate-phase rows; aggregate_time_s = 0 and phase_wall_s = max over sites of the round's wall time (train + eval).
Data: data/processed_<NN>.
--per_camera: the data are data/processed_per_camera/ instead, where every camera is standardized with constants computed
from its own training file (built by preprocessing/compute_standardization.py --per_camera and preprocessing/process_data.py);
tag local-<NN>c[-r<rounds>]-pc, hparams {"per_camera": True, "data": "processed_per_camera"}. Nothing else changes: the same
workers, initialization, seeds and evaluation.
--init_from <run id>: every site starts from that FL run's final global model (the identical final_model.pth all its sites
saved) instead of the shared random init; the run must be a finished run of fedavg, fedprox, fedadam or fedadambn on the same
clients (a smoke run only if this run is a smoke run too); strategy "<strategy>ft", tag <strategy>ft-<NN>c,
hparams {"init_from": run id}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_fl import CONFIG, check_data, finish, new_stage, spawn, wait_all, worker_env  # noqa: E402
from strategies.baselines import merge_site_parts  # noqa: E402
from summarize import TAG, read_clients  # noqa: E402

GLOBAL = ("fedavg", "fedprox", "fedadam", "fedadambn")  # strategies whose sites all end with the same (global) model
ROUNDS = 20
PER_CAMERA = ROOT / "data" / "processed_per_camera"


def run_tag(a: argparse.Namespace, strategy: str) -> str:
    """<strategy>-<NN>c, plus -r<rounds> for a plain local run with a non-default number of epochs (so it never pools with the
    20-epoch ones) and -pc for per-camera constants."""
    tag = f"{strategy}-{a.clients}c" + (f"-r{a.rounds}" if a.init_from is None and a.rounds != ROUNDS else "") + ("-pc" if a.per_camera else "")
    return tag if a.max_batches is None else f"smoke-{tag}"


def run_hparams(a: argparse.Namespace) -> dict | None:
    """Recorded in run.json / index: None for a plain local run, {"per_camera", "data"} for --per_camera, {"init_from"} for fine-tuning."""
    if a.init_from is None:
        return {"per_camera": True, "data": PER_CAMERA.name} if a.per_camera else None
    return {"init_from": a.init_from}


def index_record(run_id: str) -> dict:
    recs = [r for r in map(json.loads, (ROOT / "runs" / "index.jsonl").read_text().splitlines()) if r["run_id"] == run_id]
    assert len(recs) == 1, f"{len(recs)} index records for {run_id}"
    return recs[0]


def run_sites(run_id: str) -> list[str]:
    return sorted(pd.read_csv(ROOT / "runs" / run_id / "metrics.csv").intersection.unique())


def global_strategy(run_id: str, clients: list[str], smoke: bool) -> str:
    """Strategy of run_id, which must be a single, finished run of a GLOBAL strategy on exactly `clients`, and not a smoke run
    unless the fine-tuning run is a smoke run itself (smoke = True)."""
    tag = index_record(run_id)["tag"]
    m = TAG.fullmatch(tag)
    assert m and (smoke or not m.group(1)) and m["strategy"] in GLOBAL, \
        f"{run_id}: tag {tag} is not a finished run of {GLOBAL}" + ("" if smoke else " (a smoke source needs --max_batches)")
    sites = run_sites(run_id)
    assert sites == sorted(clients), (sites, clients)
    return m["strategy"]


def global_model(run_dir: Path, sites: list[str]) -> Path:
    """Path of the final_model.pth every site of run_dir holds; raises unless all sites' checkpoints are identical."""
    paths = [run_dir / s / "final_model.pth" for s in sites]
    ref = torch.load(paths[0], map_location="cpu")
    for p in paths[1:]:
        sd = torch.load(p, map_location="cpu")
        assert sd.keys() == ref.keys() and all(torch.equal(sd[k], ref[k]) for k in ref), f"{p} differs from {paths[0]}"
    return paths[0]


def build_hint(nn: str) -> str:
    json_path = "configs/standardization_per_camera.json"
    return ("Build them with\n"
            f"  python preprocessing/compute_standardization.py --raw_dir data/raw_trajectories --clients configs/clients_{nn}.txt "
            f"--per_camera --out {json_path}\n"
            f"  python preprocessing/process_data.py --raw_dir data/raw_trajectories --clients configs/clients_{nn}.txt "
            f"--standardization {json_path} --out_dir data/processed_per_camera")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", required=True, choices=["06", "23"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_batches", type=int, default=None, help="smoke tests only; tags the run 'smoke-...'")
    ap.add_argument("--init_from", default=None, help="run id of a finished FL run of " + ", ".join(GLOBAL) + ": fine-tune its global model")
    ap.add_argument("--per_camera", action="store_true", help="local-only training on data/processed_per_camera (per-camera "
                    "standardization constants); tags the run '...-pc'")
    ap.add_argument("--note", default="")
    return ap


def main() -> int:
    a = parser().parse_args()
    assert a.rounds > 0, a.rounds
    assert not (a.per_camera and a.init_from), "--per_camera is local-only training: it does not go with --init_from"

    clients = read_clients(a.clients)
    data_dir = PER_CAMERA if a.per_camera else ROOT / "data" / f"processed_{a.clients}"
    check_data(data_dir, clients, build_hint(a.clients) if a.per_camera else "Build them with preprocessing/process_data.py.")
    strategy = "local" if a.init_from is None else global_strategy(a.init_from, clients, a.max_batches is not None) + "ft"
    tag = run_tag(a, strategy)
    init = [] if a.init_from is None else ["--init_model", str(global_model(ROOT / "runs" / a.init_from, clients))]
    run_id, stage = new_stage(tag, a.seed)
    env = worker_env(a.gpu)
    smoke = [] if a.max_batches is None else ["--max_batches", str(a.max_batches)]
    start = time.time()
    procs = {c: spawn(stage, env, "strategies.local", c, [
        "--config_file", str(ROOT / CONFIG), "--data_dir", str(data_dir), "--intersection", c, "--first_intersection", clients[0],
        "--init_seed", str(a.seed), "--seed", str(100 * a.seed + i), "--rounds", str(a.rounds), *init, *smoke])
        for i, c in enumerate(clients)}
    print(f"[run_local] {tag} seed={a.seed}: {len(clients)} processes started; logs in {stage}", flush=True)
    if not wait_all(procs, stage):
        return 1
    merge_site_parts(stage, run_id, strategy, a.seed, a.rounds, clients)
    finish(stage, run_id, tag, a.seed, a.rounds, clients, start, a.note, run_hparams(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Local-only baseline: every intersection trains its own model on its own train split, no communication.

    python scripts/run_local.py --clients 06 --seed 1
    python scripts/run_local.py --clients 06 --seed 1 --per_camera                       # each camera with its own constants
    python scripts/run_local.py --clients 06 --seed 0 --rounds 2 --max_batches 2         # smoke test
    python scripts/run_local.py --clients 06 --seed 1 --init_from <fedavg run id> --rounds 5   # global model + local fine-tuning
    python scripts/run_local.py --clients 17 --seed 1 --init_from <fedrep run id> --rounds 0 --head Broad   # Appendix A.5, zero-shot

One process per intersection (strategies/local.py), all on --gpu. "Round" r = 1 local epoch (the FedAvg client's
fl_local_train) + test evaluation, so 20 rounds match FedAvg's compute. Same init as the FL runs (the server-built model),
client i seeded 100 * seed + i. Output format = run_fl.py's (strategy "local", tag local-<NN>c[-r<rounds> if --rounds is not 20]),
with only evaluate-phase rows; aggregate_time_s = 0 and phase_wall_s = max over sites of the round's wall time (train + eval).
Data: data/processed_<NN>. --data_dir replaces the default folder of every mode (the modes below have their own); a missing file
stops the launch before any process starts.
--per_camera: the data are data/processed_per_camera/ instead, where every camera is standardized with constants computed
from its own training file (built by preprocessing/compute_standardization.py --per_camera and preprocessing/process_data.py);
tag local-<NN>c[-r<rounds>]-pc, hparams {"per_camera": True, "data": <name of the data folder>}. Nothing else changes: the same
workers, initialization, seeds and evaluation.
--init_from <run id>: every site starts from that FL run's final global model (the identical final_model.pth all its sites
saved) instead of the shared random init; the run must be a finished run of fedavg, fedprox, fedadam or fedadambn on the same
clients (a smoke run only if this run is a smoke run too); strategy "<strategy>ft", tag <strategy>ft-<NN>c,
hparams {"init_from": run id}.
--clients 17 (Appendix A.5): the six-intersection models at the 17 camera views outside the federation (configs/clients_17.txt).
Data: data/processed_17_unseen (built by preprocessing/unseen_views.py: the files of the 23-view setting with the constants of the
six intersections). Needs --init_from <a finished six-intersection run (tag <src>-06c) of fedavg, fedrep or fedper> and --seed = the
seed of that run (a smoke source run only if this run is a smoke run too):
    --rounds 0: zero-shot, one test evaluation (round 0), no training, no final_model.pth. FedAvg applies its global model (strategy
               fedavgzs, tag fedavgzs-17c); FedRep and FedPer need --head <a member intersection of clients_06> and apply the shared
               encoder with that intersection's head, i.e. its final_model.pth as is (strategy <src>zs, tag <src>zs-17c-d<intersection>).
    --rounds R: adaptation, R x (1 local epoch on the view's training split + test evaluation). FedAvg fine-tunes the whole global
               model (strategy fedavgft, tag fedavgft-17c); FedRep and FedPer need --head mean --head_only and train a new head on the
               frozen shared encoder, starting from the element-wise mean of the six trained heads (mean_head, saved as
               init_head_mean.pth in the run folder; strategy <src>ad, tag <src>ad-17c).
    hparams {"init_from", "mode": "zeroshot" | "adapt", "head": None | "mean" | <intersection>, "head_only", "data": <name of the data folder>}.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from models.build_model import Build_Model  # noqa: E402
from run_fl import CONFIG, add_data_dir_arg, check_data, data_dir, finish, new_stage, spawn, wait_all, worker_env  # noqa: E402
from strategies.baselines import merge_site_parts  # noqa: E402
from strategies.keys import exchanged_keys  # noqa: E402
from summarize import DONOR_STRATEGIES, TAG, read_clients  # noqa: E402
from utils import load_config  # noqa: E402

GLOBAL = ("fedavg", "fedprox", "fedadam", "fedadambn")  # strategies whose sites all end with the same (global) model
SOURCES = ("fedavg", "fedrep", "fedper")  # six-intersection models evaluated at the 17 views outside the federation (--clients 17)
HEADED = ("fedrep", "fedper")  # sources with a personal head per intersection: --head required
ROUNDS = 20
PER_CAMERA = "processed_per_camera"
UNSEEN = "processed_17_unseen"


def run_tag(a: argparse.Namespace, strategy: str) -> str:
    """<strategy>-<NN>c, plus -r<rounds> for a plain local run with a non-default number of epochs (so it never pools with the
    20-epoch ones), -d<intersection> for a zero-shot run with a donor head and -pc for per-camera constants."""
    tag = f"{strategy}-{a.clients}c" + (f"-r{a.rounds}" if a.init_from is None and a.rounds != ROUNDS else "") \
        + (f"-d{a.head}" if strategy in DONOR_STRATEGIES else "") + ("-pc" if a.per_camera else "")
    return tag if a.max_batches is None else f"smoke-{tag}"


def local_data_dir(a: argparse.Namespace) -> Path:
    """--data_dir, else data/processed_per_camera (--per_camera), data/processed_17_unseen (--clients 17) or data/processed_<NN>."""
    return data_dir(a, PER_CAMERA if a.per_camera else UNSEEN if a.clients == "17" else f"processed_{a.clients}")


def run_hparams(a: argparse.Namespace) -> dict | None:
    """Recorded in run.json / index: None for a plain local run, {"per_camera", "data"} for --per_camera, {"init_from"} for
    fine-tuning, for --clients 17 also mode, head, head_only and data."""
    if a.init_from is None:
        return {"per_camera": True, "data": local_data_dir(a).name} if a.per_camera else None
    if a.clients != "17":
        return {"init_from": a.init_from}
    return {"init_from": a.init_from, "mode": "zeroshot" if a.rounds == 0 else "adapt", "head": a.head, "head_only": a.head_only,
            "data": local_data_dir(a).name}


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


def source_strategy(run_id: str, seed: int, smoke: bool) -> str:
    """--clients 17: the strategy of run_id, which must be a single, finished six-intersection run (tag <src>-06c) of SOURCES with
    this seed, whose sites are clients_06 and none of the 17, and not a smoke run unless this run is a smoke run itself (smoke = True)."""
    rec = index_record(run_id)
    tag = rec["tag"].removeprefix("smoke-") if smoke else rec["tag"]
    src = tag.removesuffix("-06c")
    assert tag == f"{src}-06c" and src in SOURCES, f"{run_id}: tag {rec['tag']} is not a finished six-intersection run of {SOURCES}" \
        + ("" if smoke else " (a smoke source needs --max_batches)")
    assert rec["seed"] == seed, f"--seed {seed} differs from the seed {rec['seed']} of the source run {run_id}"
    sites = run_sites(run_id)
    assert sites == sorted(read_clients("06")) and not set(sites) & set(read_clients("17")), (sites, run_id)
    return src


def unseen_strategy(a: argparse.Namespace, src: str) -> str:
    """--clients 17: checks the flags against the source strategy (module docstring) and returns the run's strategy."""
    assert a.rounds >= 0, a.rounds
    assert (a.head is not None) == (src in HEADED), f"--head is required for a {HEADED} source and forbidden for {src}"
    assert a.head_only == (src in HEADED and a.rounds > 0), "--head_only goes with (and is required for) adapting a fedrep / fedper source"
    if a.head is not None:
        assert (a.head == "mean") if a.rounds > 0 else (a.head in read_clients("06")), \
            f"--head {a.head}: zero-shot (--rounds 0) takes a donor intersection of clients_06, adaptation --head mean"
    return f"{src}zs" if a.rounds == 0 else "fedavgft" if src == "fedavg" else f"{src}ad"


def mean_head(states: list[dict], shared: list[str], model: torch.nn.Module) -> dict:
    """State dict with the shared keys of `states` (asserted identical across them) and, for every other key, the element-wise mean of
    the floating tensors or the max of an integer buffer (num_batches_tracked, unused because every BatchNorm has a momentum)."""
    keys = list(model.state_dict())
    assert len(states) > 1 and all(list(sd) == keys for sd in states), "the state dicts' keys differ from the model's"
    assert shared and set(shared) < set(keys), shared
    assert all(m.momentum is not None for m in model.modules() if "BatchNorm" in type(m).__name__), "a BatchNorm without momentum"
    out = {}
    for k in keys:
        vals = [sd[k] for sd in states]
        assert all(v.shape == vals[0].shape and v.dtype == vals[0].dtype for v in vals), k
        if k in shared:
            assert all(torch.equal(v, vals[0]) for v in vals), f"shared key {k} differs between the intersections"
            out[k] = vals[0].clone()
        elif vals[0].is_floating_point():
            out[k] = torch.stack(vals).mean(0)
        else:
            assert k.endswith("num_batches_tracked"), k
            out[k] = torch.stack(vals).max(0).values
        assert not out[k].is_floating_point() or torch.isfinite(out[k]).all(), k
    return out


def write_mean_head(run_dir: Path, src: str, stage: Path, data: Path, seed: int) -> Path:
    """mean_head of the six intersections' final_model.pth of run_dir (shared keys = exchanged_keys(src)), saved as
    <stage>/init_head_mean.pth."""
    states = [torch.load(run_dir / s / "final_model.pth", map_location="cpu") for s in read_clients("06")]
    with tempfile.TemporaryDirectory() as tmp:
        model = Build_Model(load_config(str(ROOT / CONFIG), read_clients("17")[0], str(data), tmp, seed))
    model.load_state_dict(states[0])  # strict: the checkpoints fit the architecture
    path = stage / "init_head_mean.pth"
    torch.save(mean_head(states, exchanged_keys(src, model), model), path)
    return path


def global_model(run_dir: Path, sites: list[str]) -> Path:
    """Path of the final_model.pth every site of run_dir holds; raises unless all sites' checkpoints are identical."""
    paths = [run_dir / s / "final_model.pth" for s in sites]
    ref = torch.load(paths[0], map_location="cpu")
    for p in paths[1:]:
        sd = torch.load(p, map_location="cpu")
        assert sd.keys() == ref.keys() and all(torch.equal(sd[k], ref[k]) for k in ref), f"{p} differs from {paths[0]}"
    return paths[0]


def build_hint(a: argparse.Namespace) -> str:
    """How to build the default data folder of the run (shown with the missing files)."""
    if a.clients == "17":
        return "Build them with preprocessing/unseen_views.py or pass --data_dir."
    if not a.per_camera:
        return "Build them with preprocessing/process_data.py or pass --data_dir."
    json_path = "configs/standardization_per_camera.json"
    return ("Build them with\n"
            f"  python preprocessing/compute_standardization.py --raw_dir data/raw_trajectories --clients configs/clients_{a.clients}.txt "
            f"--per_camera --out {json_path}\n"
            f"  python preprocessing/process_data.py --raw_dir data/raw_trajectories --clients configs/clients_{a.clients}.txt "
            f"--standardization {json_path} --out_dir data/{PER_CAMERA}\nor pass --data_dir.")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", required=True, choices=["06", "17", "23"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_batches", type=int, default=None, help="smoke tests only; tags the run 'smoke-...'")
    ap.add_argument("--init_from", default=None, help="run id of a finished FL run of " + ", ".join(GLOBAL) + ": fine-tune its global "
                    "model (--clients 17: a finished six-intersection run of " + ", ".join(SOURCES) + ")")
    ap.add_argument("--head", default=None, help="--clients 17 with a fedrep / fedper source: a donor intersection of clients_06 "
                    "(zero-shot) or mean (adaptation)")
    ap.add_argument("--head_only", action="store_true", help="--clients 17 adaptation of a fedrep / fedper source: train the head only")
    ap.add_argument("--per_camera", action="store_true", help="local-only training on data/processed_per_camera (per-camera "
                    "standardization constants); tags the run '...-pc'")
    add_data_dir_arg(ap)
    ap.add_argument("--note", default="")
    return ap


def main() -> int:
    a = parser().parse_args()
    assert not (a.per_camera and a.init_from), "--per_camera is local-only training: it does not go with --init_from"
    if a.clients == "17":
        assert a.init_from is not None, "--clients 17 evaluates a six-intersection model: --init_from is required"
    else:
        assert a.head is None and not a.head_only and a.rounds > 0, "--head, --head_only and --rounds 0 go with --clients 17 only"

    clients = read_clients(a.clients)
    data = local_data_dir(a)
    check_data(data, clients, build_hint(a))
    smoke_run = a.max_batches is not None
    if a.clients == "17":
        src = source_strategy(a.init_from, a.seed, smoke_run)
        strategy = unseen_strategy(a, src)
    else:
        strategy = "local" if a.init_from is None else global_strategy(a.init_from, clients, smoke_run) + "ft"
    tag = run_tag(a, strategy)
    src_dir = None if a.init_from is None else ROOT / "runs" / a.init_from
    if a.init_from is None or a.head == "mean":
        init = None  # the mean head is written into the run folder below
    elif a.head is None:
        init = global_model(src_dir, read_clients("06") if a.clients == "17" else clients)
    else:
        init = src_dir / a.head / "final_model.pth"
        assert init.is_file(), init
    run_id, stage = new_stage(tag, a.seed)
    if a.head == "mean":
        init = write_mean_head(src_dir, src, stage, data, a.seed)
    env = worker_env(a.gpu)
    init_args = [] if init is None else ["--init_model", str(init)]
    head_only = ["--head_only", src] if a.head_only else []
    smoke = [] if a.max_batches is None else ["--max_batches", str(a.max_batches)]
    start = time.time()
    procs = {c: spawn(stage, env, "strategies.local", c, [
        "--config_file", str(ROOT / CONFIG), "--data_dir", str(data), "--intersection", c, "--first_intersection", clients[0],
        "--init_seed", str(a.seed), "--seed", str(100 * a.seed + i), "--rounds", str(a.rounds), *init_args, *head_only, *smoke])
        for i, c in enumerate(clients)}
    print(f"[run_local] {tag} seed={a.seed}: {len(clients)} processes started; logs in {stage}", flush=True)
    if not wait_all(procs, stage):
        return 1
    merge_site_parts(stage, run_id, strategy, a.seed, a.rounds, clients)
    finish(stage, run_id, tag, a.seed, a.rounds, clients, start, a.note, run_hparams(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())

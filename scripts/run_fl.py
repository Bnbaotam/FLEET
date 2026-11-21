#!/usr/bin/env python3
"""Run one federated experiment on this machine: 1 Flower server + one client process per intersection.

    python scripts/run_fl.py --strategy fedrep --clients 06 --seed 1
    python scripts/run_fl.py --strategy fedavg --clients 23 --seed 1 --port 8081 --gpu 1
    python scripts/run_fl.py --strategy fedavg --clients 06 --seed 0 --rounds 2 --max_batches 2   # smoke test
    python scripts/run_fl.py --strategy fedprox --clients 06 --seed 0 --val_frac 0.1 --fedprox_mu 0.001   # sweep point, tag fedprox-06c-val10-mu0.001

--fedadam_eta / --fedadambn_eta / --fedprox_mu / --fedcross_alpha override the default value (strategies/keys.py SWEEP) and only
go with their own --strategy; the tag gets the value only if it differs from the default. The value is recorded in run.json
("hparams"). --data_dir (default data/processed_<NN>) holds <intersection>_{train,test}.pkl for every client; a missing file
stops the launch before any process starts.

The server writes metrics.csv and rounds.csv (every client's ADE/FDE per round) into runs/_staging/<name>/; each client
writes final_model.pth and per_sample_final.csv into its own subfolder. On success the folder
is registered with scripts/log_run.py and moved to runs/<run_id>/; on failure it stays in runs/_staging/.
Seeds: server = seed, client i = 100 * seed + i.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from strategies.keys import SWEEP, STRATEGIES, add_sweep_args, sweep_value  # noqa: E402
from summarize import read_clients  # noqa: E402
from log_run import git_state, log_run, new_run_id  # noqa: E402

CONFIG = "configs/flowchain.yml"


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_server(port: int, server: subprocess.Popen, timeout: float = 300.0) -> None:
    start = time.time()
    while port_is_free(port):
        if server.poll() is not None:
            raise RuntimeError(f"server exited with code {server.returncode} before listening")
        if time.time() - start > timeout:
            raise TimeoutError(f"server not listening on port {port} after {timeout}s")
        time.sleep(1)


def check_data(data: Path, clients: list[str], hint: str) -> None:
    """Raise unless <data>/<client>_{train,test}.pkl exist for every client (called before any process is started)."""
    missing = [f"{c}_{s}.pkl" for c in clients for s in ("train", "test") if not (data / f"{c}_{s}.pkl").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of the {2 * len(clients)} processed files are missing in {data}:\n  "
                                + "\n  ".join(missing) + f"\n{hint}")


def new_stage(tag: str, seed: int) -> tuple[str, Path]:
    """Draw the run id, create runs/_staging/<time>-<tag>-s<seed>/ and record the git state the run starts from."""
    stage = ROOT / "runs" / "_staging" / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{tag}-s{seed}"
    stage.mkdir(parents=True)
    (stage / "git_state.json").write_text(json.dumps(git_state(), indent=2) + "\n")
    return new_run_id(tag, seed), stage


WORKER_THREADS = "2"  # CPU threads per worker process


def worker_env(gpu: str) -> dict:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": WORKER_THREADS, "MKL_NUM_THREADS": WORKER_THREADS}


def spawn(stage: Path, env: dict, module: str, name: str, args: list[str]) -> subprocess.Popen:
    """Start `python -m module args --output_dir <stage>/<name>` with its stdout in <stage>/<name>/stdout.log."""
    out = stage / name
    out.mkdir()
    log = open(out / "stdout.log", "w")
    cmd = [sys.executable, "-m", module, *args, "--output_dir", str(out)]
    (out / "command.txt").write_text(" ".join(cmd) + "\n")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_all(procs: dict, stage: Path) -> bool:
    """Wait for every process (terminating the rest on interrupt); prints a message and returns False if any failed."""
    try:
        codes = {name: p.wait() for name, p in procs.items()}
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
    failed = {n: c for n, c in codes.items() if c != 0}
    if failed:
        print(f"[run] FAILED {failed}; see {stage}/<name>/stdout.log", file=sys.stderr)
    return not failed


def finish(stage: Path, run_id: str, tag: str, seed: int, rounds: int, clients: list[str], start: float, note: str,
           hparams: dict | None = None) -> None:
    """Check the staged run, register it with log_run (index entry, run.json) and move it to runs/<run_id>/."""
    table = pd.read_csv(stage / "rounds.csv")
    final = table[(table.phase == "evaluate") & (table.split == "test") & (table["round"] == rounds)]
    assert len(final) == 1 and final.n_reporting.iloc[0] == len(clients) and (table.run_id == run_id).all(), final
    metrics = {"num_rounds": rounds,
               **{f"final_{k}_px": float(final[f"{k}_mean_px"].iloc[0]) for k in ("ade", "fde", "ade_legacy", "fde_legacy")},
               "wall_clock_s": round(time.time() - start, 1)}
    for c in clients:
        assert (stage / c / "final_model.pth").is_file() and (stage / c / "per_sample_final.csv").is_file(), c
    assert (stage / "git_state.json").is_file(), stage
    git = json.loads((stage / "git_state.json").read_text())
    log_run(tag=tag, seed=seed, metrics=metrics, config=CONFIG, note=note, command=" ".join(sys.argv), run_id=run_id,
            hparams=hparams, git=git)
    for item in stage.iterdir():
        shutil.move(str(item), ROOT / "runs" / run_id / item.name)
    stage.rmdir()
    print(f"[run] done: runs/{run_id} {metrics}")


def add_data_dir_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data_dir", type=Path, default=None, help="folder with <intersection>_{train,test}.pkl (default data/processed_<NN>)")


def data_dir(a: argparse.Namespace) -> Path:
    """--data_dir as an absolute path (the workers run in the repository root), default data/processed_<NN>."""
    return ROOT / "data" / f"processed_{a.clients}" if a.data_dir is None else a.data_dir.resolve()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--clients", required=True, choices=["06", "23"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for every process of this run")
    ap.add_argument("--max_batches", type=int, default=None, help="smoke tests only; tags the run 'smoke-...'")
    ap.add_argument("--val_frac", type=float, default=0.0, help="carve this fraction of each client's train tracks into a validation set (0 = off)")
    add_sweep_args(ap)
    add_data_dir_arg(ap)
    ap.add_argument("--note", default="")
    return ap


def run_tag(a: argparse.Namespace) -> str:
    """<strategy>-<NN>c[-val<pct>][-<hyperparameter><value>, only if not the default], with a smoke- prefix for --max_batches runs."""
    tag = f"{a.strategy}-{a.clients}c" + (f"-val{a.val_frac * 100:g}" if a.val_frac > 0 else "")
    value = sweep_value(a)
    if value is not None and value != SWEEP[a.strategy][1]:
        tag += f"-{SWEEP[a.strategy][0]}{value:g}"
    return tag if a.max_batches is None else f"smoke-{tag}"


def run_hparams(a: argparse.Namespace) -> dict:
    """Recorded in run.json / index: val_frac and the sweep value if the strategy has one."""
    value = sweep_value(a)
    return {"val_frac": a.val_frac, **({} if value is None else {f"{a.strategy}_{SWEEP[a.strategy][0]}": value})}


def main() -> int:
    a = parser().parse_args()
    value = sweep_value(a)
    hparams = run_hparams(a)

    clients = read_clients(a.clients)
    tag = run_tag(a)
    data = data_dir(a)
    check_data(data, clients, "Build them with preprocessing/process_data.py or pass --data_dir.")
    assert port_is_free(a.port), f"port {a.port} is in use; pick another --port"

    run_id, stage = new_stage(tag, a.seed)
    env = worker_env(a.gpu)
    common = ["--strategy", a.strategy, "--config_file", str(ROOT / CONFIG), "--data_dir", str(data)]

    def launch(module: str, name: str, extra: list[str]) -> subprocess.Popen:
        return spawn(stage, env, module, name, [*common, *extra])

    start = time.time()
    procs = {}
    try:
        procs["server"] = launch("strategies.server", "server", [
            "--intersection", clients[0], "--seed", str(a.seed), "--num_clients", str(len(clients)),
            "--num_rounds", str(a.rounds), "--port", str(a.port), "--run_dir", str(stage), "--run_id", run_id,
            "--val_frac", str(a.val_frac), *([] if value is None else ["--sweep_value", str(value)])])
        wait_for_server(a.port, procs["server"])
        smoke = [] if a.max_batches is None else ["--max_batches", str(a.max_batches)]
        for i, c in enumerate(clients):
            procs[c] = launch("strategies.client", c, [
                "--intersection", c, "--seed", str(100 * a.seed + i), "--server_address", f"127.0.0.1:{a.port}",
                "--val_frac", str(a.val_frac), *smoke])
            time.sleep(1)
        print(f"[run_fl] {tag} seed={a.seed}: {len(clients)} clients started; logs in {stage}", flush=True)
        ok = wait_all(procs, stage)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
    if not ok:
        return 1
    finish(stage, run_id, tag, a.seed, a.rounds, clients, start, a.note, hparams)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Server-side logger of the metrics every client reports in each round of a run, per client and averaged over clients."""

import csv
import math
import os
from typing import Dict, List

import numpy as np

from utils import PIXELS_PER_UNIT

METRICS_COLUMNS = ["run_id", "strategy", "n_clients", "seed", "round", "phase", "split", "intersection", "n_train_windows",
                   "n_test_windows", "ade_px", "fde_px", "ade_legacy_px", "fde_legacy_px", "train_loss", "train_time_s",
                   "eval_time_s", "inference_time_s"]
ROUNDS_COLUMNS = ["run_id", "strategy", "n_clients", "seed", "round", "phase", "split", "n_reporting", "ade_mean_px",
                  "fde_mean_px", "ade_legacy_mean_px", "fde_legacy_mean_px", "ade_weighted_px", "fde_weighted_px",
                  "phase_wall_s", "aggregate_time_s"]
ERRORS = ("ade", "fde", "ade_legacy", "fde_legacy")
EVAL_COLUMNS = ["n_test_windows", "eval_time_s", "inference_time_s", *(f"{k}_px" for k in ERRORS)]  # empty in unevaluated fit rows
STAT_COLUMNS = [c for c in ROUNDS_COLUMNS if c.endswith(("_mean_px", "_weighted_px"))]


class RoundLogger:
    def __init__(self, run_dir: str, run_id: str, strategy: str, n_clients: int, seed: int, num_rounds: int, val_frac: float):
        self.num_rounds, self.val = num_rounds, val_frac > 0
        self.ident = dict(run_id=run_id, strategy=strategy, n_clients=n_clients, seed=seed)
        self.files = {}
        for name, columns in (("metrics", METRICS_COLUMNS), ("rounds", ROUNDS_COLUMNS)):
            path = os.path.join(run_dir, f"{name}.csv")
            assert not os.path.exists(path), path
            f = open(path, "w", newline="")
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            f.flush()
            self.files[name] = (f, writer)
        self.intersections = None

    def write(self, name: str, rows: List[Dict]) -> None:
        f, writer = self.files[name]
        writer.writerows(rows)
        f.flush()

    def log(self, phase: str, server_round: int, results: list, phase_wall_s: float, aggregate_time_s: float) -> None:
        """`results`: the FitRes / EvaluateRes of every client of this round (the failures were asserted empty)."""
        n = self.ident["n_clients"]
        metrics = [r.metrics for r in results]
        names = sorted(m["intersection"] for m in metrics)
        assert len(names) == n and len(set(names)) == n, f"round {server_round} {phase}: reports from {names}, expected {n} distinct clients"
        self.intersections = self.intersections or names
        assert names == self.intersections, (names, self.intersections)

        evaluated = phase == "evaluate" or server_round == self.num_rounds
        splits = [("", "test")] + ([("val_", "val")] if self.val and phase == "evaluate" else [])
        rows, rounds = [], []
        for prefix, split in splits:
            split_rows = []
            for m in sorted(metrics, key=lambda m: m["intersection"]):
                assert ("val_ade" in m) == (self.val and phase == "evaluate"), (phase, sorted(m))
                assert (phase == "fit") == ("loss" in m) == ("train_time_s" in m), (phase, sorted(m))
                assert ("ade" in m) == evaluated, (server_round, phase, sorted(m))
                measured = dict.fromkeys(EVAL_COLUMNS)
                if evaluated:
                    measured.update(n_test_windows=m[prefix + "n_test_windows"], eval_time_s=m[prefix + "eval_time_s"],
                                    inference_time_s=m[prefix + "inference_time"], **{f"{k}_px": m[prefix + k] * PIXELS_PER_UNIT for k in ERRORS})
                row = dict(self.ident, round=server_round, phase=phase, split=split, intersection=m["intersection"],
                           n_train_windows=m["n_train_windows"], train_loss=m.get("loss"), train_time_s=m.get("train_time_s"), **measured)
                assert all(v is None or (not isinstance(v, float)) or math.isfinite(v) for v in row.values()), row
                split_rows.append(row)
            rows += split_rows
            stat = dict.fromkeys(STAT_COLUMNS)
            if evaluated:
                weights = np.array([r["n_test_windows"] for r in split_rows], dtype=float)
                stat = {f"{k}_mean_px": np.mean([r[f"{k}_px"] for r in split_rows]) for k in ERRORS}
                stat.update({f"{k}_weighted_px": np.average([r[f"{k}_px"] for r in split_rows], weights=weights) for k in ("ade", "fde")})
            rounds.append(dict(self.ident, round=server_round, phase=phase, split=split, n_reporting=len(split_rows),
                               phase_wall_s=phase_wall_s, aggregate_time_s=aggregate_time_s, **stat))
        self.write("metrics", rows)
        self.write("rounds", rounds)

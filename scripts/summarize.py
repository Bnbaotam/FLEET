#!/usr/bin/env python3
"""Turn runs into the paper's Table 2 / Table 4 format, from the server-written metrics (see scripts/collect.py).

All numbers are in pixels. ade/fde are best-of-20 and corrected; --legacy uses ade_legacy/fde_legacy (the original
evaluation code's metric). Per run and intersection one value is selected from the rounds of one phase and split (--select):
  final      : round = num_rounds (default; no test-set model selection; the last round is always evaluated)
  best       : lowest ADE and, independently, lowest FDE over the rounds (selects on the test set)
  same_round : the round of the lowest ADE, with that round's FDE (the chosen rounds are printed)
--phase evaluate (default) is the model the client holds after receiving the server's update, fit the model right after
local training (evaluated at the final round only, so with fit every --select gives the final round). --split test (default) or val (runs made with --val_frac only).
Across runs (seeds): mean ± std (ddof=1; "(n=1)" for a single run) per intersection; Average = per-run unweighted mean
over intersections, then mean ± std over runs; Wins = intersections where the strategy has the lowest mean.

CLI:
    python scripts/summarize.py --clients 06 [--phase evaluate] [--split test] [--select final] [--legacy] [--strategies fedavg fedrep ...]
Strategies (table columns): the federated ones, fedavgft (FedAvg + local fine-tuning, run_local.py --init_from), and the
non-federated baselines "local" (run_local.py), "local-pc" (Local (per-camera): run_local.py --per_camera, each camera
standardized with its own constants) and "central" (run_central.py); central evaluates only every few epochs, so best /
same_round select among fewer rounds for it than for the others. --clients 23 also shows "fedavg-scaled" / "fedrep-scaled"
(Appendix A.4: run_fl.py --scaled, one scale factor per camera). --clients 17 (Appendix A.5: the six-intersection models at the
17 views outside the federation, run_local.py --clients 17) shows the zero-shot runs "fedavgzs" (FedAvg's global model) and
"fedrepzs-d<intersection>" / "fedperzs-d<intersection>" (the shared encoder with the head of that member intersection, one
column per donor head), and the adapted models "fedavgft" (whole model fine-tuned), "fedrepad" / "fedperad" (new head on the frozen
encoder, from the mean of the six heads). Zero-shot runs have the single round 0; the runs of one strategy must have the same
number of rounds.
Uses every run in runs/index.jsonl whose tag is "<strategy>-<NN>c", a tuned "<strategy>-<NN>c-<hp><value>", a local run with
a non-default number of epochs "local-<NN>c-r<epochs>" (reported as its own strategy "local-r<epochs>"), a per-camera local
run "local-<NN>c[-r<epochs>]-pc" (reported as "local[-r<epochs>]-pc"), a scaled run "<fedavg|fedrep>-23c-scaled" (reported as
"<strategy>-scaled") or a zero-shot run with a donor head "<fedrepzs|fedperzs>-17c-d<intersection>" (reported as
"<strategy>-d<intersection>"); one configuration per strategy. The runs are re-collected into data/derived/ on every call, or
--input reads a CSV written by collect.py; --include_smoke also uses smoke-<strategy>-<NN>c runs.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from collect import collect

ROOT = Path(__file__).resolve().parent.parent
SCALED_STRATEGIES = ("fedavg", "fedrep")  # run with run_fl.py --scaled (Appendix A.4)
DONOR_STRATEGIES = ("fedrepzs", "fedperzs")  # zero-shot of a FedRep / FedPer model with the head of one member intersection (Appendix A.5)
# "<strategy>-<NN>c" or a tuned final run "<strategy>-<NN>c-<hyperparameter><value>" (e.g. fedprox-06c-mu0.001);
# or a local run with a non-default number of epochs "local-<NN>c-r<epochs>" and/or per-camera constants "...-pc" (run_local.run_tag);
# or a scaled run "<fedavg|fedrep>-23c-scaled" (run_fl.run_tag); or a zero-shot run with a donor head "<fedrepzs|fedperzs>-17c-d<intersection>"
# (run_local.run_tag); validation (sweep) runs such as fedadam-06c-val10[-eta0.01] do not match and stay out
TAG = re.compile(r"(smoke-)?(?P<strategy>[a-z]+)-(?P<nn>\d\d)c(-(?:eta|mu|alpha)[0-9.e+-]+)?(?P<epochs>-r\d+)?(?P<pc>-pc)?"
                 r"(?P<scaled>-scaled)?(?P<donor>-d[A-Za-z0-9_]+)?")


def run_key(m: re.Match) -> str:
    """Strategy name of a TAG match in the tables: the tag's strategy, plus -r<epochs> for a non-default-epoch local run (local-r25),
    -pc for a per-camera local run (local-pc), -scaled for a scaled run (fedavg-scaled) and -d<intersection> for a zero-shot run with
    that donor's head (fedrepzs-dBroad)."""
    assert m["epochs"] is None or m["strategy"] == "local", m.group(0)
    assert m["pc"] is None or m["strategy"] == "local", m.group(0)
    assert m["scaled"] is None or (m["strategy"] in SCALED_STRATEGIES and m["nn"] == "23"), m.group(0)
    assert (m["donor"] is not None) == (m["strategy"] in DONOR_STRATEGIES and m["nn"] == "17"), m.group(0)
    return m["strategy"] + (m["epochs"] or "") + (m["pc"] or "") + (m["scaled"] or "") + (m["donor"] or "")


def clients_file(path: Path) -> list[str]:
    """Camera views listed in a clients file (one per line; '#' starts a comment line)."""
    clients = [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert clients and len(set(clients)) == len(clients), (path, clients)
    return clients


def read_clients(nn: str) -> list[str]:
    clients = clients_file(ROOT / "configs" / f"clients_{nn}.txt")
    assert len(clients) == int(nn), clients
    return clients


# table columns, in this order; for the 17 views FedRep and FedPer have one zero-shot column per donor head (member intersection)
ORDER = ("fedadam", "fedadambn", "fedavg", "fedavg-scaled", "fedavgzs", "fedavgft", "fedcross", "fedprox",
         "fedper", *(f"fedperzs-d{d}" for d in read_clients("06")), "fedperad",
         "fedrep", "fedrep-scaled", *(f"fedrepzs-d{d}" for d in read_clients("06")), "fedrepad",
         "local", "local-r25", "local-pc", "central")


def select_rounds(df: pd.DataFrame, how: str, legacy: bool) -> pd.DataFrame:
    """df: all rows of one phase and split. Returns one row per (run_id, intersection): columns ade, fde, round (NaN for best)."""
    ade, fde = ("ade_legacy_px", "fde_legacy_px") if legacy else ("ade_px", "fde_px")
    g = df.groupby(["run_id", "intersection"])
    last = g["round"].transform("max")
    n_rounds = df.groupby("run_id")["round"].transform("nunique")  # central runs evaluate only every --eval_every epochs (sparse rounds)
    assert (g["round"].transform("count") == n_rounds).all() and (g["round"].transform("nunique") == n_rounds).all(), \
        "rounds are not 1..R (or the same sparse rounds at every intersection)"
    per_strategy = last.groupby(df.strategy).unique()
    assert per_strategy.map(len).eq(1).all(), f"runs of one strategy differ in their number of rounds: {per_strategy.to_dict()}"
    cols = ["run_id", "strategy", "intersection", "round", "ade", "fde"]
    if how == "best":
        return g.agg(strategy=("strategy", "first"), ade=(ade, "min"), fde=(fde, "min")).assign(round=np.nan).reset_index()[cols]
    rows = df[df["round"] == last] if how == "final" else df.loc[g[ade].idxmin()]
    assert len(rows) == g.ngroups
    return rows.rename(columns={ade: "ade", fde: "fde"})[cols]


def pm(x: np.ndarray, fmt: str = "{:.3f}") -> str:
    return fmt.format(x.mean()) + (" (n=1)" if len(x) == 1 else " ± " + fmt.format(x.std(ddof=1)))


def table(df: pd.DataFrame, nn: str, clients: list[str], phase: str, split: str, how: str, legacy: bool, strategies: list[str]) -> str:
    key = df.tag.map(lambda t: run_key(m) if (m := TAG.fullmatch(t)) and m["nn"] == nn else None)
    df = df[key.isin(strategies)].assign(strategy=key)
    df = df[(df.phase == phase) & (df.split == split)]
    assert phase == "fit" or df.ade_px.notna().all(), "evaluate rows without ADE"
    df = df.dropna(subset=["ade_px"])  # fit rows of non-final rounds have only train_loss / train_time_s
    assert len(df), f"no runs found for {nn} clients, phase={phase}, split={split}"
    sel = select_rounds(df, how, legacy)
    assert set(sel.strategy) <= set(ORDER), f"strategies missing from ORDER: {set(sel.strategy) - set(ORDER)}"
    cols = [s for s in ORDER if s in set(sel.strategy)]
    lines = [f"# {nn}-intersection results, phase={phase}, split={split}, select={how}" + (", LEGACY metric (velocity mean not added back)" if legacy else "")]
    arr, rnd = {}, {}
    for s in cols:
        runs = sorted(sel[sel.strategy == s].run_id.unique())
        tags = sorted(df[df.strategy == s].tag.unique())
        assert len(tags) == 1, f"{s}: runs with different configurations would be pooled: {tags}"
        lines.append(f"# {s} [{tags[0]}]: n={len(runs)} runs: {', '.join(runs)}")
        wide = {m: sel[sel.strategy == s].pivot(index="run_id", columns="intersection", values=m) for m in ("ade", "fde", "round")}
        assert set(wide["ade"].columns) == set(clients), (s, sorted(wide["ade"].columns))
        arr[s] = {m: wide[m][clients].to_numpy() for m in ("ade", "fde")}  # runs x clients
        rnd[s] = wide["round"][clients].to_numpy()
    for m in ("ade", "fde"):
        lines.append(f"\n{m.upper()}\t" + "\t".join(cols))
        for j, c in enumerate(clients):
            lines.append(c + "\t" + "\t".join(pm(arr[s][m][:, j]) for s in cols))
        lines.append("Average\t" + "\t".join(pm(arr[s][m].mean(axis=1)) for s in cols))
        means = np.stack([arr[s][m].mean(axis=0) for s in cols])  # strategies x clients
        lines.append("Wins\t" + "\t".join(str(w) for w in np.bincount(means.argmin(axis=0), minlength=len(cols))))
    if how == "same_round":
        lines.append("\nCHOSEN ROUND (of the lowest ADE, one per run, in run order)\t" + "\t".join(cols))
        for j, c in enumerate(clients):
            lines.append(c + "\t" + "\t".join(",".join(str(int(r)) for r in rnd[s][:, j]) for s in cols))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", required=True, choices=["06", "17", "23"])
    ap.add_argument("--phase", default="evaluate", choices=["evaluate", "fit"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--select", default="final", choices=["final", "best", "same_round"])
    ap.add_argument("--legacy", action="store_true", help="use ade_legacy/fde_legacy (original evaluation code's metric, biased by -velocity mean)")
    ap.add_argument("--strategies", nargs="+", default=list(ORDER))
    ap.add_argument("--include_smoke", action="store_true")
    ap.add_argument("--input", type=Path, help="all_runs.csv from collect.py (default: collect runs/ into data/derived/ now)")
    a = ap.parse_args()
    df = pd.read_csv(a.input) if a.input else collect(ROOT / "runs", ROOT / "data" / "derived", a.include_smoke)[0]
    print(table(df, a.clients, read_clients(a.clients), a.phase, a.split, a.select, a.legacy, a.strategies))


if __name__ == "__main__":
    main()

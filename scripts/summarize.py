#!/usr/bin/env python3
"""Summarizes the runs into tables of forecasting errors per camera and method, with mean and standard deviation over seeds."""

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
    ap = argparse.ArgumentParser(description=__doc__)
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

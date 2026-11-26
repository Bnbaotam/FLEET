import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

from models.build_model import Build_Model
from strategies.metrics_log import RoundLogger
from utils import load_config, seed_everything

PART = "rounds_part.jsonl"  # per-site rows of run_local workers, merged (and deleted) by merge_site_parts


def fl_initial_state(config_file: str, data_dir: str, first_site: str, seed: int, output_dir: str):
    """The model strategies/server.py builds: seed_everything(seed), then Build_Model on clients[0]. `output_dir` only
    receives a config.yaml, which the caller overwrites with its own."""
    seed_everything(seed)
    return Build_Model(load_config(config_file, first_site, data_dir, output_dir, seed)).state_dict()


def assert_same_standardization(envs: List) -> None:
    """Pooling sites into one training set is only consistent if every site standardizes inputs identically."""
    assert envs and all(e.standardization == envs[0].standardization for e in envs), "sites differ in env.standardization"


def merge_site_parts(run_dir: Path, run_id: str, strategy: str, seed: int, rounds: int, sites: List[str]) -> None:
    """Merge <run_dir>/<site>/PART (one JSON line per round: {round, wall_s, metrics}) into metrics.csv / rounds.csv through
    RoundLogger, phase "evaluate" only. phase_wall_s = max over sites of that round's wall time (train + eval);
    aggregate_time_s = 0 (nothing is aggregated). rounds = 0 (zero-shot evaluation): the single round 0. The part files are deleted."""
    expected = list(range(1, rounds + 1)) if rounds else [0]
    parts = {}
    for s in sites:
        parts[s] = [json.loads(l) for l in (run_dir / s / PART).read_text().splitlines()]
        assert [p["round"] for p in parts[s]] == expected, (s, [p["round"] for p in parts[s]])
        assert all(p["metrics"]["intersection"] == s for p in parts[s]), s
    logger = RoundLogger(str(run_dir), run_id, strategy, len(sites), seed, rounds, 0.0)
    for i, r in enumerate(expected):
        row = [parts[s][i] for s in sites]
        logger.log("evaluate", r, [SimpleNamespace(metrics=p["metrics"]) for p in row], max(p["wall_s"] for p in row), 0.0)
    for s in sites:
        (run_dir / s / PART).unlink()

#!/usr/bin/env python3
"""Records an experiment run with its configuration, seed, git state and metrics in an append-only log of runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
INDEX = RUNS / "index.jsonl"


class RunLogError(RuntimeError):
    pass


GIT_KEYS = ("git_commit", "git_dirty", "git_branch", "git_code_diff", "git_code_untracked")
CODE_DIRS = ("src", "scripts", "preprocessing", "configs", "tests")


def in_git_work_tree() -> bool:
    return shutil.which("git") is not None and subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "true"


def git_state() -> dict:
    """HEAD, dirty flag, branch, the uncommitted diff and the untracked files of the code dirs; launchers capture it when a
    run starts. Every value is None outside a git work tree; inside one, a failing git command raises."""
    if not in_git_work_tree():
        return dict.fromkeys(GIT_KEYS)

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "git_branch": git("branch", "--show-current"), "git_code_diff": git("diff", "HEAD", "--", *CODE_DIRS),
            "git_code_untracked": git("ls-files", "--others", "--exclude-standard", "--", *CODE_DIRS)}


def new_run_id(tag: str, seed: int) -> str:
    """Run id of the form <utc time>-<tag>-s<seed>-<4 hex>; run_fl.py draws it before launching so the server can log it."""
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "-" for c in tag)
    return f"{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}-{safe_tag}-s{seed}-{secrets.token_hex(2)}"


def log_run(*, tag: str, seed: int, metrics: dict, config: str | None = None, note: str = "", command: str | None = None,
            run_id: str | None = None, hparams: dict | None = None, git: dict | None = None) -> str:
    """`git` is the git_state() taken when the run started; None takes it now."""
    if not isinstance(metrics, dict) or not metrics:
        raise RunLogError("metrics must be a non-empty dict")

    git = git_state() if git is None else git
    assert set(git) == set(GIT_KEYS), git.keys()
    no_git = all(v is None for v in git.values())
    assert no_git or (all(isinstance(git[k], str) for k in GIT_KEYS if k != "git_dirty") and isinstance(git["git_dirty"], bool)), git
    now = dt.datetime.now(dt.timezone.utc)
    run_id = run_id or new_run_id(tag, seed)
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_copy = None
    if config:
        src = (ROOT / config) if not Path(config).is_absolute() else Path(config)
        if not src.is_file():
            raise RunLogError(f"config not found: {config}")
        config_copy = f"runs/{run_id}/{src.name}"
        shutil.copy2(src, run_dir / src.name)

    diff_copy = None
    if not no_git and git["git_code_diff"]:
        diff_copy = f"runs/{run_id}/git_code_diff.patch"
        (run_dir / "git_code_diff.patch").write_text(git["git_code_diff"] + "\n")

    record = {
        "run_id": run_id,
        "time_utc": now.isoformat(timespec="seconds"),
        "tag": tag,
        "seed": seed,
        "metrics": metrics,
        "config": config,
        "config_copy": config_copy,
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "git_branch": git["git_branch"],
        "git_code_dirty": None if no_git else bool(git["git_code_diff"] or git["git_code_untracked"]),
        "git_code_diff": diff_copy,
        "git_code_untracked": None if no_git else git["git_code_untracked"].splitlines(),
        "command": command or " ".join(sys.argv),
        "hparams": hparams,
        "note": note,
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    with INDEX.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return run_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="experiment name, e.g. baseline, lr-sweep")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--metrics", required=True, help="JSON object, or @file.json")
    ap.add_argument("--config")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    raw = Path(a.metrics[1:]).read_text() if a.metrics.startswith("@") else a.metrics
    try:
        run_id = log_run(tag=a.tag, seed=a.seed, metrics=json.loads(raw), config=a.config, note=a.note)
    except (RunLogError, json.JSONDecodeError) as e:
        print(f"log_run: {e}", file=sys.stderr)
        return 2
    print(run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())

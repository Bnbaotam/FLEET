"""Unit tests of the run logging and the git state it records."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import log_run  # noqa: E402

GIT = {"git_commit": "a" * 40, "git_dirty": True, "git_branch": "feat", "git_code_diff": "", "git_code_untracked": ""}
HAS_GIT = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """runs/ (and the git lookups) in tmp_path, which is not inside a git repository."""
    monkeypatch.setattr(log_run, "ROOT", tmp_path)
    monkeypatch.setattr(log_run, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(log_run, "INDEX", tmp_path / "runs" / "index.jsonl")
    return tmp_path / "runs"


def _log(**kw) -> tuple[dict, Path]:
    run_id = log_run.log_run(tag="t", seed=0, metrics={"m": 1.0}, **kw)
    record = json.loads((log_run.INDEX).read_text().splitlines()[-1])
    assert record == json.loads((log_run.RUNS / run_id / "run.json").read_text())
    return record, log_run.RUNS / run_id


def test_outside_a_git_repository_the_git_fields_are_null(runs, tmp_path):
    assert not log_run.in_git_work_tree() and log_run.git_state() == dict.fromkeys(log_run.GIT_KEYS)
    (tmp_path / "cfg.yml").write_text("a: 1\n")
    record, run_dir = _log(config="cfg.yml", hparams={"x": 1}, command="python run.py")
    assert all(record[k] is None for k in ("git_commit", "git_dirty", "git_branch", "git_code_dirty", "git_code_diff", "git_code_untracked"))
    assert record["metrics"] == {"m": 1.0} and record["hparams"] == {"x": 1} and record["command"] == "python run.py"
    assert record["config_copy"] == f"runs/{run_dir.name}/cfg.yml" and (run_dir / "cfg.yml").read_text() == "a: 1\n"
    assert list(record) == ["run_id", "time_utc", "tag", "seed", "metrics", "config", "config_copy", "git_commit", "git_dirty", "git_branch",
                            "git_code_dirty", "git_code_diff", "git_code_untracked", "command", "hparams", "note"]
    _log(git=dict.fromkeys(log_run.GIT_KEYS))  # the state a launcher recorded at the start of a run outside git
    assert len(log_run.INDEX.read_text().splitlines()) == 2


@HAS_GIT
def test_inside_a_git_repository_the_state_is_recorded(runs, tmp_path):
    git = lambda *a: subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=", *a], cwd=tmp_path, check=True, capture_output=True, text=True)
    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", "src/a.py")
    git("commit", "-q", "-m", "init")
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    (tmp_path / "src" / "b.py").write_text("y = 1\n")
    state = log_run.git_state()
    assert state["git_commit"] == git("rev-parse", "HEAD").stdout.strip() and state["git_dirty"] is True
    assert "+x = 2" in state["git_code_diff"] and state["git_code_untracked"] == "src/b.py"
    record, run_dir = _log()
    assert record["git_code_dirty"] is True and (run_dir / "git_code_diff.patch").is_file() and record["git_code_untracked"] == ["src/b.py"]


def test_clean_code_records_given_state_and_no_patch(runs):
    record, run_dir = _log(git=GIT)
    assert {k: record[k] for k in ("git_commit", "git_dirty", "git_branch")} == {"git_commit": "a" * 40, "git_dirty": True, "git_branch": "feat"}
    assert record["git_code_dirty"] is False and record["git_code_diff"] is None and record["git_code_untracked"] == []
    assert not (run_dir / "git_code_diff.patch").exists()


def test_dirty_code_writes_patch(runs):
    diff = "diff --git a/src/x.py b/src/x.py\n-a\n+b"
    record, run_dir = _log(git={**GIT, "git_code_diff": diff})
    assert record["git_code_dirty"] is True
    assert record["git_code_diff"] == f"runs/{run_dir.name}/git_code_diff.patch"
    assert (run_dir / "git_code_diff.patch").read_text() == diff + "\n"
    assert "diff --git" not in log_run.INDEX.read_text()


def test_untracked_code_marks_dirty(runs):
    record, run_dir = _log(git={**GIT, "git_code_untracked": "src/new.py\ntests/test_new.py"})
    assert record["git_code_dirty"] is True and record["git_code_untracked"] == ["src/new.py", "tests/test_new.py"]
    assert record["git_code_diff"] is None and not (run_dir / "git_code_diff.patch").exists()


@pytest.mark.parametrize("git", [{k: v for k, v in GIT.items() if k != "git_code_diff"}, {**GIT, "extra": 1}, {**GIT, "git_commit": None}])
def test_wrong_keys_or_a_partial_state_raise(runs, git):
    with pytest.raises(AssertionError):
        log_run.log_run(tag="t", seed=0, metrics={"m": 1.0}, git=git)

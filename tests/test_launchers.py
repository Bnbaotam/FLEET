"""Unit tests of the launchers' options: sweep flags and run tags (run_fl.py), the server's hyperparameters, the missing-file check
that stops a launch, and the local-only / fine-tuning options of run_local.py (tags, hparams, the global-model check and the
source-run check)."""
import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_local  # noqa: E402
from run_fl import check_data, data_dir, parser, run_hparams, run_tag  # noqa: E402
from strategies.keys import SWEEP, sweep_value  # noqa: E402
from strategies.server import build_strategy  # noqa: E402
from summarize import TAG, read_clients  # noqa: E402


# ---- run_fl.py: sweep flags, tags, hparams, data dir --------------------------------------------------------------------
def _args(*words):
    return parser().parse_args(["--clients", "06", "--seed", "0", *words])


@pytest.mark.parametrize("words, value, tag", [
    (["--strategy", "fedadam", "--val_frac", "0.1"], 0.1, "fedadam-06c-val10"),                       # default: no suffix
    (["--strategy", "fedadam", "--val_frac", "0.1", "--fedadam_eta", "0.01"], 0.01, "fedadam-06c-val10-eta0.01"),
    (["--strategy", "fedprox", "--fedprox_mu", "0.001"], 0.001, "fedprox-06c-mu0.001"),               # tuned final run: no val_frac
    (["--strategy", "fedprox", "--fedprox_mu", "0.01"], 0.01, "fedprox-06c"),                          # explicit default
    (["--strategy", "fedcross", "--fedcross_alpha", "0.5", "--max_batches", "2"], 0.5, "smoke-fedcross-06c-alpha0.5"),
    (["--strategy", "fedavg"], None, "fedavg-06c"),
    (["--strategy", "fedadambn", "--fedadambn_eta", "0.1"], 0.1, "fedadambn-06c"),
    (["--strategy", "fedadambn", "--val_frac", "0.1", "--fedadambn_eta", "0.3"], 0.3, "fedadambn-06c-val10-eta0.3"),
])
def test_sweep_value_tag_and_hparams(words, value, tag):
    a = _args(*words)
    assert sweep_value(a) == value and run_tag(a) == tag
    assert run_hparams(a) == {"val_frac": a.val_frac, **({} if value is None else {f"{a.strategy}_{SWEEP[a.strategy][0]}": value})}


def test_sweep_flags_only_with_their_strategy_and_in_range():
    for words in (["--strategy", "fedavg", "--fedprox_mu", "0.1"], ["--strategy", "fedadam", "--fedprox_mu", "0.01"],
                  ["--strategy", "fedcross", "--fedcross_alpha", "1.5"], ["--strategy", "fedadam", "--fedadam_eta", "0"],
                  ["--strategy", "fedadambn", "--fedadam_eta", "0.3"], ["--strategy", "fedadam", "--fedadambn_eta", "0.3"]):
        with pytest.raises(AssertionError):
            sweep_value(_args(*words))


def test_sweep_tags_stay_out_of_the_main_tables():
    for tag in ("fedadam-06c", "local-23c", "smoke-fedavg-06c", "fedprox-06c-mu0.001", "fedcross-06c-alpha0.5"):  # final runs
        assert TAG.fullmatch(tag)
    for tag in ("fedadam-06c-val10", "fedadam-06c-val10-eta0.01"):  # validation sweep runs do not
        assert not TAG.fullmatch(tag)


def test_server_gets_the_cli_hyperparameter():
    model = torch.nn.Linear(2, 2)
    build = lambda name, v: build_strategy(name, model, 2, 3, None, v)
    assert build("fedadam", 0.01).eta == 0.01 and build("fedprox", 0.1).proximal_mu == 0.1 and build("fedcross", 0.5).cross_alpha == 0.5
    with pytest.raises(AssertionError):
        build("fedavg", 0.1)
    with pytest.raises(AssertionError):
        build("fedprox", None)


def test_data_dir_default_and_relative_path(tmp_path, monkeypatch):
    assert data_dir(_args("--strategy", "fedavg"), "processed_06") == ROOT / "data" / "processed_06"
    monkeypatch.chdir(tmp_path)
    assert data_dir(_args("--strategy", "fedavg", "--data_dir", "mine"), "processed_06") == tmp_path / "mine"


def test_check_data_lists_every_missing_file(tmp_path):
    check_data(ROOT / "data" / "processed_06", read_clients("06"), "")
    with pytest.raises(FileNotFoundError) as e:
        check_data(ROOT / "data" / "processed_23", read_clients("23"), "hint")
    msg = str(e.value)
    assert "26 of the 46 processed files are missing" in msg and msg.endswith("hint")
    assert all(f"{c}_{s}.pkl" in msg for c in read_clients("23")[10:] for s in ("train", "test"))
    assert "Broad_2_train.pkl" not in msg
    (tmp_path / "A_train.pkl").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="1 of the 2 processed files are missing"):
        check_data(tmp_path, ["A"], "")


# ---- run_local.py ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rounds, init_from, max_batches, per_camera, tag", [
    (20, None, None, False, "local-06c"), (25, None, None, False, "local-06c-r25"), (3, None, 2, False, "smoke-local-06c-r3"),
    (5, "some-run", None, False, "fedavgft-06c"), (20, "some-run", 2, False, "smoke-fedavgft-06c"),
    (20, None, None, True, "local-06c-pc"), (25, None, None, True, "local-06c-r25-pc"), (2, None, 2, True, "smoke-local-06c-r2-pc"),
])
def test_local_tag_and_hparams(rounds, init_from, max_batches, per_camera, tag):
    a = Namespace(clients="06", rounds=rounds, init_from=init_from, max_batches=max_batches, per_camera=per_camera, data_dir=None)
    assert run_local.run_tag(a, "local" if init_from is None else "fedavgft") == tag
    assert run_local.run_hparams(a) == ({"init_from": init_from} if init_from else
                                        {"per_camera": True, "data": "processed_per_camera"} if per_camera else None)


SITES = ["a", "b", "c"]


def _write(tmp_path, states):
    for s, sd in zip(SITES, states):
        (tmp_path / s).mkdir()
        torch.save(sd, tmp_path / s / "final_model.pth")


def _state(w=0.0):
    return {"w": torch.full((2, 3), w), "b": torch.arange(3.0)}


def test_global_model_accepts_identical(tmp_path):
    _write(tmp_path, [_state() for _ in SITES])
    assert run_local.global_model(tmp_path, SITES) == tmp_path / "a" / "final_model.pth"


def test_global_model_rejects_one_differing_tensor(tmp_path):
    _write(tmp_path, [_state(), _state(), _state(1e-6)])
    with pytest.raises(AssertionError, match="differs"):
        run_local.global_model(tmp_path, SITES)


def test_global_model_rejects_different_keys(tmp_path):
    _write(tmp_path, [_state(), _state(), {"w": _state()["w"]}])
    with pytest.raises(AssertionError):
        run_local.global_model(tmp_path, SITES)


def test_global_strategy_checks_the_source_run(tmp_path, monkeypatch):
    runs = {"avg": "fedavg-06c", "prox": "fedprox-06c-mu0.001", "loc": "local-06c", "rep": "fedrep-06c", "smk": "smoke-fedavg-06c"}
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "index.jsonl").write_text("".join(json.dumps({"run_id": r, "tag": t}) + "\n" for r, t in runs.items()))
    for r in runs:
        (tmp_path / "runs" / r).mkdir()
        pd.DataFrame({"intersection": SITES}).to_csv(tmp_path / "runs" / r / "metrics.csv", index=False)
    monkeypatch.setattr(run_local, "ROOT", tmp_path)
    assert run_local.global_strategy("avg", SITES, smoke=False) == "fedavg"
    assert run_local.global_strategy("prox", SITES, smoke=False) == "fedprox"
    for r in ("loc", "rep"):
        with pytest.raises(AssertionError, match="not a finished run"):
            run_local.global_strategy(r, SITES, smoke=True)
    with pytest.raises(AssertionError, match="smoke source"):  # a smoke source only for a smoke fine-tuning run
        run_local.global_strategy("smk", SITES, smoke=False)
    assert run_local.global_strategy("smk", SITES, smoke=True) == "fedavg"
    with pytest.raises(AssertionError):  # the source run must have exactly these sites
        run_local.global_strategy("avg", SITES[:-1], smoke=False)
    with pytest.raises(AssertionError, match="0 index records"):
        run_local.global_strategy("missing", SITES, smoke=False)

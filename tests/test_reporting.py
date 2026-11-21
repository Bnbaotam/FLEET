"""Unit tests (known input -> known output) for the reporting path: server-side CSV logging, per-sample errors,
collect.py / summarize.py selection and statistics, and the opt-in validation carve."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collect import collect  # noqa: E402
from data.unified_loader import load_env, unified_loader, val_track_ids  # noqa: E402
from fl_train_utils import evaluate_model, window_table  # noqa: E402
from models.build_model import Build_Model  # noqa: E402
from strategies.baselines import PART, assert_same_standardization, merge_site_parts  # noqa: E402
from strategies.metrics_log import METRICS_COLUMNS, ROUNDS_COLUMNS, RoundLogger  # noqa: E402
from summarize import TAG, clients_file, pm, run_key, select_rounds, table  # noqa: E402
from utils import PIXELS_PER_UNIT, load_config  # noqa: E402


# ---- RoundLogger ----------------------------------------------------------------------------------------
def _result(name, ade, n, phase, val=None, evaluated=True):
    """Client report; evaluated=False is a fit report of a non-final round (no test evaluation, only the training numbers)."""
    m = {"intersection": name, "n_train_windows": 100}
    if evaluated:
        m.update({"n_test_windows": n, "ade": ade, "fde": 2 * ade, "ade_legacy": 3 * ade, "fde_legacy": 4 * ade,
                  "inference_time": 0.5, "eval_time_s": 7.0})
    if phase == "fit":
        m.update(loss=1.5, train_time_s=9.0)
    if val:
        m.update({"val_" + k: v for k, v in {**m, "ade": val, "fde": 2 * val, "n_test_windows": 5}.items() if k != "intersection"})
    return SimpleNamespace(metrics=m)


def test_round_logger_rows_and_means(tmp_path):
    logger = RoundLogger(str(tmp_path), "rid", "fedavg", 2, 7, 3, val_frac=0.1)
    logger.log("fit", 3, [_result("B", 0.04, 30, "fit"), _result("A", 0.02, 10, "fit")], 12.0, 0.5)  # final round: evaluated
    logger.log("evaluate", 1, [_result("A", 0.02, 10, "evaluate", val=0.1), _result("B", 0.04, 30, "evaluate", val=0.2)], 8.0, 0.25)
    metrics, rounds = pd.read_csv(tmp_path / "metrics.csv"), pd.read_csv(tmp_path / "rounds.csv")
    assert list(metrics.columns) == METRICS_COLUMNS and list(rounds.columns) == ROUNDS_COLUMNS
    assert len(metrics) == 2 + 2 * 2 and len(rounds) == 1 + 2  # fit: 2 clients; evaluate: 2 clients x (test, val)
    fit_a = metrics[(metrics.phase == "fit") & (metrics.intersection == "A")].iloc[0]
    assert fit_a.ade_px == pytest.approx(0.02 * PIXELS_PER_UNIT) and fit_a.fde_legacy_px == pytest.approx(0.08 * PIXELS_PER_UNIT)
    assert fit_a.train_loss == 1.5 and fit_a.train_time_s == 9.0 and fit_a.seed == 7 and fit_a.run_id == "rid"
    ev = metrics[metrics.phase == "evaluate"]
    assert ev.train_loss.isna().all() and ev.train_time_s.isna().all() and ev.n_train_windows.eq(100).all()
    val_b = ev[(ev.split == "val") & (ev.intersection == "B")].iloc[0]
    assert val_b.ade_px == pytest.approx(0.2 * PIXELS_PER_UNIT) and val_b.n_test_windows == 5
    r = rounds[(rounds.phase == "evaluate") & (rounds.split == "test")].iloc[0]
    assert r.ade_mean_px == pytest.approx(0.03 * PIXELS_PER_UNIT)  # unweighted: (0.02 + 0.04) / 2
    assert r.ade_weighted_px == pytest.approx(0.035 * PIXELS_PER_UNIT)  # (10 * 0.02 + 30 * 0.04) / 40
    assert r.fde_legacy_mean_px == pytest.approx(0.12 * PIXELS_PER_UNIT) and r.n_reporting == 2
    assert (r.phase_wall_s, r.aggregate_time_s) == (8.0, 0.25)


def test_round_logger_fit_rows_of_non_final_rounds_have_only_training_numbers(tmp_path):
    logger = RoundLogger(str(tmp_path), "rid", "fedavg", 2, 0, 3, val_frac=0.0)
    logger.log("fit", 1, [_result("B", 0, 0, "fit", evaluated=False), _result("A", 0, 0, "fit", evaluated=False)], 4.0, 0.5)
    logger.log("evaluate", 1, [_result("A", 0.02, 10, "evaluate"), _result("B", 0.04, 30, "evaluate")], 8.0, 0.25)
    metrics, rounds = pd.read_csv(tmp_path / "metrics.csv"), pd.read_csv(tmp_path / "rounds.csv")
    assert list(metrics.columns) == METRICS_COLUMNS and list(rounds.columns) == ROUNDS_COLUMNS
    fit, ev = metrics[metrics.phase == "fit"], metrics[metrics.phase == "evaluate"]
    assert len(fit) == 2 and fit.train_loss.eq(1.5).all() and fit.train_time_s.eq(9.0).all() and fit.n_train_windows.eq(100).all()
    assert fit[["n_test_windows", "eval_time_s", "inference_time_s", "ade_px", "fde_px", "ade_legacy_px", "fde_legacy_px"]].isna().all().all()
    assert ev[["n_test_windows", "ade_px", "eval_time_s"]].notna().all().all()
    r = rounds[rounds.phase == "fit"].iloc[0]
    assert r.n_reporting == 2 and r.phase_wall_s == 4.0 and r.aggregate_time_s == 0.5
    assert r[[c for c in ROUNDS_COLUMNS if c.endswith(("_mean_px", "_weighted_px"))]].isna().all()
    assert rounds[rounds.phase == "evaluate"].ade_mean_px.notna().all()


def test_round_logger_rejects_evaluation_at_the_wrong_rounds(tmp_path):
    logger = RoundLogger(str(tmp_path), "rid", "fedavg", 1, 0, 3, val_frac=0.0)
    with pytest.raises(AssertionError):  # evaluated fit report before the final round
        logger.log("fit", 2, [_result("A", 0.02, 10, "fit")], 1.0, 1.0)
    with pytest.raises(AssertionError):  # final-round fit report without evaluation
        logger.log("fit", 3, [_result("A", 0, 0, "fit", evaluated=False)], 1.0, 1.0)
    with pytest.raises(AssertionError):  # evaluate rows are always evaluated
        logger.log("evaluate", 1, [_result("A", 0, 0, "evaluate", evaluated=False)], 1.0, 1.0)


def test_round_logger_rejects_missing_or_duplicate_clients(tmp_path):
    logger = RoundLogger(str(tmp_path), "rid", "fedavg", 2, 0, 3, val_frac=0.0)
    with pytest.raises(AssertionError):
        logger.log("fit", 3, [_result("A", 0.02, 10, "fit"), _result("A", 0.02, 10, "fit")], 1.0, 1.0)
    with pytest.raises(AssertionError):
        logger.log("fit", 3, [_result("A", 0.02, 10, "fit")], 1.0, 1.0)


# ---- window_table ---------------------------------------------------------------------------------------
def test_window_table_known_geometry():
    B = 2
    obs = torch.zeros(B, 8, 6)
    obs[:, -1, :2] = torch.tensor([1.0, 1.0])
    obs[:, -1, 2:4] = torch.tensor([0.3, 0.4])  # speed 0.5 units/s
    steps = torch.zeros(B, 12, 2)
    steps[0, 0] = torch.tensor([3.0, 4.0])      # window 0: one 5-unit step, then standing still
    steps[1, :, 0] = 1.0                        # window 1: 12 steps of 1 along x
    gt = obs[:, -1:, :2] + steps.cumsum(dim=1)
    t = window_table({"obs": obs, "gt": gt, "index": [["s", np.int64(4), "PEDESTRIAN/9"], ["s", np.int64(5), "PEDESTRIAN/10"]]},
                     {"ade": np.array([1.0, 2.0]), "fde": np.array([3.0, 4.0]), "ade_legacy": np.ones(B), "fde_legacy": np.ones(B)})
    assert list(t.node_id) == ["9", "10"] and list(t.t) == [4, 5]
    np.testing.assert_allclose(t.gt_endpoint_disp_px, np.array([5.0, 12.0]) * PIXELS_PER_UNIT, rtol=1e-6)
    np.testing.assert_allclose(t.gt_path_len_px, np.array([5.0, 12.0]) * PIXELS_PER_UNIT, rtol=1e-6)
    np.testing.assert_allclose(t.obs_last_speed_px_s, 0.5 * PIXELS_PER_UNIT, rtol=1e-6)
    np.testing.assert_allclose(t.ade_px, [1.0 * PIXELS_PER_UNIT, 2.0 * PIXELS_PER_UNIT])


def test_window_table_path_longer_than_endpoint_when_turning():
    obs = torch.zeros(1, 8, 6)
    steps = torch.zeros(1, 12, 2)
    steps[0, 0], steps[0, 1] = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])  # out and back
    t = window_table({"obs": obs, "gt": steps.cumsum(dim=1), "index": [["s", 0, "PEDESTRIAN/1"]]},
                     {k: np.zeros(1) for k in ("ade", "fde", "ade_legacy", "fde_legacy")})
    assert t.gt_endpoint_disp_px[0] == 0 and t.gt_path_len_px[0] == pytest.approx(2 * PIXELS_PER_UNIT)


# ---- per-sample == aggregate on real data (GPU) ----------------------------------------------------------
@CUDA
def test_per_sample_mean_equals_aggregate(tmp_path):
    cfg = load_config(str(ROOT / "configs/flowchain.yml"), "Broad_2", str(ROOT / "data/processed_06"), str(tmp_path), 0)
    torch.manual_seed(0)
    model = Build_Model(cfg)
    loader = unified_loader(cfg, rand=False, split="test")
    result, table_ = evaluate_model(cfg, model, loader, max_batches=3, return_per_sample=True)  # asserts rtol 1e-5 inside
    assert len(table_) == result["n_windows"] == 3 * cfg.DATA.BATCH_SIZE
    for k in ("ade", "fde", "ade_legacy", "fde_legacy"):
        assert table_[f"{k}_px"].mean() / PIXELS_PER_UNIT == pytest.approx(result[k], rel=1e-5)
    assert (table_.gt_path_len_px >= table_.gt_endpoint_disp_px - 1e-4).all()
    assert (table_.fde_px >= 0).all() and table_.scene.nunique() >= 1
    # same windows, same order as the loader's (unshuffled) index
    expected = [(w[0].name, w[1], w[2].id) for w in loader.dataset.index[:len(table_)]]
    assert list(zip(table_.scene, table_.t, table_.node_id)) == expected


# ---- collect / summarize --------------------------------------------------------------------------------
def _write_run(runs, run_id, tag, strategy, ade, fde, extra_rows=True, round_numbers=None):
    """ade/fde: {intersection: [value per round]} for phase=evaluate, split=test; other phase/split rows hold decoys."""
    d = runs / run_id
    d.mkdir(parents=True)
    rows = []
    for phase, split, scale in (("evaluate", "test", 1.0), ("fit", "test", 100.0), ("evaluate", "val", 1000.0)):
        if phase == "evaluate" and split == "val" and not extra_rows:
            continue
        for c in ade:
            rounds_ = list(round_numbers or range(1, len(ade[c]) + 1))
            for r, a, f in zip(rounds_, ade[c], fde[c]):
                if phase == "fit" and r != rounds_[-1]:  # fit rows are evaluated at the final round only
                    rows.append(dict.fromkeys(METRICS_COLUMNS) | dict(run_id=run_id, strategy=strategy, n_clients=len(ade), seed=0,
                                round=r, phase=phase, split=split, intersection=c, train_loss=1.0, train_time_s=1.0))
                    continue
                rows.append(dict.fromkeys(METRICS_COLUMNS, 0.0) | dict(run_id=run_id, strategy=strategy, n_clients=len(ade), seed=0,
                              round=r, phase=phase, split=split, intersection=c, ade_px=a * scale, fde_px=f * scale,
                              ade_legacy_px=a * scale + 1, fde_legacy_px=f * scale + 1))
    pd.DataFrame(rows, columns=METRICS_COLUMNS).to_csv(d / "metrics.csv", index=False)
    pd.DataFrame([dict.fromkeys(ROUNDS_COLUMNS, 0.0) | dict(run_id=run_id)], columns=ROUNDS_COLUMNS).to_csv(d / "rounds.csv", index=False)
    return {"run_id": run_id, "tag": tag, "git_commit": "abc"}


@pytest.fixture
def runs_dir(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    index = [
        _write_run(runs, "avg1", "fedavg-06c", "fedavg", {"A": [5, 3, 4], "B": [6, 7, 2]}, {"A": [10, 12, 9], "B": [8, 8, 8]}),
        _write_run(runs, "avg2", "fedavg-06c", "fedavg", {"A": [7, 5, 6], "B": [4, 5, 6]}, {"A": [14, 10, 11], "B": [6, 9, 7]}),
        _write_run(runs, "rep1", "fedrep-06c", "fedrep", {"A": [3, 3, 3], "B": [3, 3, 3]}, {"A": [1, 1, 1], "B": [1, 1, 1]}),
        _write_run(runs, "val1", "fedavg-06c-val10", "fedavg", {"A": [0, 0, 0], "B": [0, 0, 0]}, {"A": [0, 0, 0], "B": [0, 0, 0]}),
        _write_run(runs, "smk1", "smoke-fedavg-06c", "fedavg", {"A": [0, 0, 0], "B": [0, 0, 0]}, {"A": [0, 0, 0], "B": [0, 0, 0]}),
        _write_run(runs, "lr1", "local-06c-r25", "local", {"A": [0, 0, 0], "B": [0, 0, 0]}, {"A": [0, 0, 0], "B": [0, 0, 0]}),
        _write_run(runs, "pc1", "local-06c-pc", "local", {"A": [1, 1, 1], "B": [1, 1, 1]}, {"A": [2, 2, 2], "B": [2, 2, 2]}),
    ]
    (runs / "index.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n")
    return runs


def _sel(df, how, legacy=False):
    df = df[(df.run_id.isin(["avg1", "avg2"])) & (df.phase == "evaluate") & (df.split == "test")]
    return select_rounds(df, how, legacy).set_index(["run_id", "intersection"])


def test_collect_columns_and_skips(runs_dir, tmp_path):
    runs, rounds = collect(runs_dir, tmp_path / "out", include_smoke=False)
    assert set(runs.run_id) == {"avg1", "avg2", "rep1", "val1", "lr1", "pc1"}  # smoke run skipped
    assert {"tag", "git_commit"} <= set(runs.columns) and set(rounds.run_id) == set(runs.run_id)
    assert (tmp_path / "out" / "all_runs.csv").is_file() and (tmp_path / "out" / "all_rounds.csv").is_file()
    assert "smk1" in set(collect(runs_dir, tmp_path / "out2", include_smoke=True)[0].run_id)


def test_non_default_epoch_and_per_camera_local_runs_are_their_own_strategy(runs_dir, tmp_path):
    df = collect(runs_dir, tmp_path / "out", include_smoke=False)[0]
    out = table(df, "06", ["A", "B"], "evaluate", "test", "final", False, ["fedavg", "fedrep", "local"])
    assert "n=2 runs: avg1, avg2" in out and "local" not in out  # local-06c-r25 is local-r25 and local-06c-pc local-pc, never pooled with local
    assert "# local-r25 [local-06c-r25]: n=1 runs: lr1" in table(df, "06", ["A", "B"], "evaluate", "test", "final", False, ["local-r25"])
    pc = table(df, "06", ["A", "B"], "evaluate", "test", "final", False, ["fedavg", "local-pc"]).splitlines()
    assert "# local-pc [local-06c-pc]: n=1 runs: pc1" in pc and pc[pc.index("ADE\tfedavg\tlocal-pc") + 1] == "A\t5.000 ± 1.414\t1.000 (n=1)"
    for tag in ("diag-fedavg-06c", "local-06c-val10", "fedavg-23c-val10-scaled"):
        assert not TAG.fullmatch(tag), tag
    assert [run_key(TAG.fullmatch(t)) for t in ("local-06c-r25", "local-23c-r3", "smoke-local-06c-r3", "local-06c", "fedprox-06c-mu0.001",
                                                "local-23c-pc", "local-06c-r25-pc", "smoke-local-06c-r2-pc", "smoke-fedavgft-06c")] == \
        ["local-r25", "local-r3", "local-r3", "local", "fedprox", "local-pc", "local-r25-pc", "local-r2-pc", "fedavgft"]
    for tag in ("fedavg-06c-r25", "fedavg-06c-pc"):  # only local runs carry an epoch or per-camera suffix
        with pytest.raises(AssertionError):
            run_key(TAG.fullmatch(tag))


def test_collect_fails_on_old_format_run_and_column_mismatch(runs_dir, tmp_path):
    (runs_dir / "avg2" / "metrics.csv").unlink()
    with pytest.raises(FileNotFoundError, match="avg2"):
        collect(runs_dir, tmp_path / "out", include_smoke=False)
    (runs_dir / "avg2" / "metrics.csv").write_text("run_id,round\navg2,1\n")
    with pytest.raises(AssertionError, match="column sets differ"):
        collect(runs_dir, tmp_path / "out", include_smoke=False)


def test_select_final_best_same_round(runs_dir, tmp_path):
    df = collect(runs_dir, tmp_path / "out", include_smoke=False)[0]
    final = _sel(df, "final")
    assert final.loc[("avg1", "A"), ["ade", "fde"]].tolist() == [4, 9] and final.loc[("avg1", "B"), ["ade", "fde"]].tolist() == [2, 8]
    assert final.loc[("avg2", "A"), ["ade", "fde"]].tolist() == [6, 11] and final.loc[("avg2", "B"), ["ade", "fde"]].tolist() == [6, 7]
    best = _sel(df, "best")  # min ADE and min FDE independently
    assert best.loc[("avg1", "A"), ["ade", "fde"]].tolist() == [3, 9] and best.loc[("avg1", "B"), ["ade", "fde"]].tolist() == [2, 8]
    assert best.loc[("avg2", "A"), ["ade", "fde"]].tolist() == [5, 10] and best.loc[("avg2", "B"), ["ade", "fde"]].tolist() == [4, 6]
    same = _sel(df, "same_round")  # round of min ADE, FDE of that round
    assert same.loc[("avg1", "A"), ["round", "ade", "fde"]].tolist() == [2, 3, 12]
    assert same.loc[("avg1", "B"), ["round", "ade", "fde"]].tolist() == [3, 2, 8]
    assert same.loc[("avg2", "A"), ["round", "ade", "fde"]].tolist() == [2, 5, 10]
    assert same.loc[("avg2", "B"), ["round", "ade", "fde"]].tolist() == [1, 4, 6]
    assert _sel(df, "final", legacy=True).loc[("avg1", "A"), "ade"] == 5  # decoy-free legacy column = ade + 1


def test_select_rejects_incomplete_rounds(runs_dir, tmp_path):
    df = collect(runs_dir, tmp_path / "out", include_smoke=False)[0]
    df = df[(df.phase == "evaluate") & (df.split == "test") & ~((df.run_id == "avg1") & (df.intersection == "A") & (df["round"] == 2))]
    with pytest.raises(AssertionError, match="rounds are not 1..R"):
        select_rounds(df, "final", False)


def test_table_phase_fit_uses_the_evaluated_final_round(runs_dir, tmp_path):
    df = collect(runs_dir, tmp_path / "out", include_smoke=False)[0]
    assert df[(df.phase == "fit") & (df["round"] < 3)].ade_px.isna().all() and df[(df.phase == "fit") & (df["round"] == 3)].ade_px.notna().all()
    for how in ("final", "best", "same_round"):  # one evaluated round only: every selection is the final round
        out = table(df, "06", ["A", "B"], "fit", "test", how, False, ["fedavg"]).splitlines()
        ade = out[out.index("ADE\tfedavg"):]
        assert ade[1] == "A\t500.000 ± 141.421" and ade[2] == "B\t400.000 ± 282.843", how  # fit decoys = 100 x the evaluate values [4, 6], [2, 6]
    assert df[df.phase == "evaluate"].ade_px.notna().all()  # evaluate rows are complete


def test_pm_marks_single_run():
    assert pm(np.array([2.0, 4.0])) == "3.000 ± 1.414"
    assert pm(np.array([2.5])) == "2.500 (n=1)"


def test_table_average_std_and_wins(runs_dir, tmp_path):
    df = collect(runs_dir, tmp_path / "out", include_smoke=False)[0]
    out = table(df, "06", ["A", "B"], "evaluate", "test", "final", False, ["fedavg", "fedrep"]).splitlines()
    ade, fde = out[out.index("ADE\tfedavg\tfedrep"):], out[out.index("FDE\tfedavg\tfedrep"):]
    assert "# fedavg [fedavg-06c]: n=2 runs: avg1, avg2" in out and "# fedrep [fedrep-06c]: n=1 runs: rep1" in out  # val and smoke runs excluded
    assert ade[1] == "A\t5.000 ± 1.414\t3.000 (n=1)"        # A: [4, 6]
    assert ade[2] == "B\t4.000 ± 2.828\t3.000 (n=1)"        # B: [2, 6]
    assert ade[3] == "Average\t4.500 ± 2.121\t3.000 (n=1)"  # per-run means 3 and 6
    assert ade[4].startswith("Wins\t0\t2")                  # fedrep has the lowest mean at both sites
    assert fde[3] == "Average\t8.750 ± 0.354\t1.000 (n=1)"  # per-run means 8.5 and 9
    same = table(df, "06", ["A", "B"], "evaluate", "test", "same_round", False, ["fedavg"])
    assert "A\t2,2" in same and "B\t3,1" in same  # chosen rounds per run
    assert "best" in table(df, "06", ["A", "B"], "evaluate", "test", "best", False, ["fedavg"]).splitlines()[0]


# ---- non-federated baselines ----------------------------------------------------------------------------
def _part(site, ades, walls):
    rows = [{"round": r, "wall_s": w, "metrics": _result(site, a, 10 * r, "evaluate").metrics}
            for r, (a, w) in enumerate(zip(ades, walls), start=1)]
    return "".join(json.dumps(r) + "\n" for r in rows)


def _site_parts(tmp_path, rounds_of_a=2):
    for site, ades, walls in (("A", [0.02, 0.04], [5.0, 6.0]), ("B", [0.04, 0.08], [7.0, 3.0])):
        (tmp_path / site).mkdir()
        lines = _part(site, ades, walls).splitlines(True)
        (tmp_path / site / PART).write_text("".join(lines[:rounds_of_a] if site == "A" else lines))


def test_merge_site_parts_known_rows_and_means(tmp_path):
    _site_parts(tmp_path)
    merge_site_parts(tmp_path, "rid", "local", 3, 2, ["A", "B"])
    metrics, rounds = pd.read_csv(tmp_path / "metrics.csv"), pd.read_csv(tmp_path / "rounds.csv")
    assert list(metrics.columns) == METRICS_COLUMNS and list(rounds.columns) == ROUNDS_COLUMNS
    assert len(metrics) == 4 and len(rounds) == 2 and set(metrics.phase) == {"evaluate"} and set(metrics.strategy) == {"local"}
    assert metrics.train_loss.isna().all() and metrics.train_time_s.isna().all() and (metrics.seed == 3).all()
    b2 = metrics[(metrics.intersection == "B") & (metrics["round"] == 2)].iloc[0]
    assert b2.ade_px == pytest.approx(0.08 * PIXELS_PER_UNIT) and b2.n_test_windows == 20
    r1, r2 = rounds.iloc[0], rounds.iloc[1]
    assert r1.ade_mean_px == pytest.approx(0.03 * PIXELS_PER_UNIT) and r2.ade_mean_px == pytest.approx(0.06 * PIXELS_PER_UNIT)
    assert r2.ade_weighted_px == pytest.approx(0.06 * PIXELS_PER_UNIT) and r1.n_reporting == 2
    assert (r1.phase_wall_s, r2.phase_wall_s) == (7.0, 6.0) and (rounds.aggregate_time_s == 0).all()  # max over sites
    assert not (tmp_path / "A" / PART).exists() and not (tmp_path / "B" / PART).exists()


def test_merge_site_parts_rejects_missing_round(tmp_path):
    _site_parts(tmp_path, rounds_of_a=1)
    with pytest.raises(AssertionError):
        merge_site_parts(tmp_path, "rid", "local", 3, 2, ["A", "B"])


def test_assert_same_standardization():
    env = lambda std: SimpleNamespace(standardization={"PEDESTRIAN": {"velocity": {"x": {"mean": 0.0, "std": std}}}})
    assert_same_standardization([env(1.0), env(1.0)])
    with pytest.raises(AssertionError, match="standardization"):
        assert_same_standardization([env(1.0), env(2.0)])


@pytest.mark.parametrize("nn, clients", [("06", "06"), ("23", "23_chattanooga")])
def test_real_sites_share_standardization(nn, clients, tmp_path):
    sites = clients_file(ROOT / "configs" / f"clients_{clients}.txt")
    cfgs = [load_config(str(ROOT / "configs/flowchain.yml"), s, str(ROOT / f"data/processed_{nn}"), str(tmp_path), 0) for s in sites]
    assert_same_standardization([load_env(c, "train") for c in cfgs])


def test_summarize_sparse_central_rounds_and_new_strategies(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    index = [  # central evaluated at rounds 2 and 4 of 4 (--eval_every 2); local and fedavg are dense
        _write_run(runs, "c1", "central-06c", "central", {"A": [3.0, 1.0], "B": [2.0, 5.0]}, {"A": [6.0, 2.0], "B": [4.0, 9.0]}, round_numbers=[2, 4]),
        _write_run(runs, "l1", "local-06c", "local", {"A": [9.0] * 4, "B": [9.0] * 4}, {"A": [9.0] * 4, "B": [9.0] * 4}),
        _write_run(runs, "a1", "fedavg-06c", "fedavg", {"A": [7.0] * 4, "B": [7.0] * 4}, {"A": [7.0] * 4, "B": [7.0] * 4})]
    (runs / "index.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n")
    df = collect(runs, tmp_path / "out", include_smoke=False)[0]
    sub = df[(df.phase == "evaluate") & (df.split == "test")]
    chosen = lambda how: select_rounds(sub[sub.run_id == "c1"], how, False).set_index("intersection")
    assert chosen("final").loc["B", ["round", "ade", "fde"]].tolist() == [4, 5.0, 9.0]  # final = last evaluated round
    assert chosen("same_round").loc["A", ["round", "ade", "fde"]].tolist() == [4, 1.0, 2.0]
    assert chosen("same_round").loc["B", ["round", "ade", "fde"]].tolist() == [2, 2.0, 4.0]
    assert chosen("best").loc["B", ["ade", "fde"]].tolist() == [2.0, 4.0]
    out = table(df, "06", ["A", "B"], "evaluate", "test", "final", False, ["fedavg", "local", "central"]).splitlines()
    ade = out[out.index("ADE\tfedavg\tlocal\tcentral"):]
    assert ade[1] == "A\t7.000 (n=1)\t9.000 (n=1)\t1.000 (n=1)" and ade[2] == "B\t7.000 (n=1)\t9.000 (n=1)\t5.000 (n=1)"
    assert ade[4] == "Wins\t0\t0\t2"  # central has the lowest mean at both sites
    hole = sub[~((sub.run_id == "c1") & (sub.intersection == "A") & (sub["round"] == 2))]  # one site misses an evaluated round
    with pytest.raises(AssertionError, match="rounds are not 1..R"):
        select_rounds(hole, "final", False)


# ---- validation carve -----------------------------------------------------------------------------------
def _cfg(tmp_path, val_frac):
    return load_config(str(ROOT / "configs/flowchain.yml"), "Broad_2", str(ROOT / "data/processed_06"), str(tmp_path), 0, val_frac)


def _ids(loader):
    return {w[2].id for w in loader.dataset.index}


def test_val_carve_disjoint_deterministic_and_test_style(tmp_path):
    cfg = _cfg(tmp_path, 0.1)
    train_env = load_env(cfg, "train")
    all_ids = {n.id for s in train_env.scenes for n in s.nodes}
    test_ids = {n.id for s in load_env(cfg, "test").scenes for n in s.nodes}
    val_ids = val_track_ids(train_env, 0.1)
    assert val_ids == val_track_ids(train_env, 0.1) and len(val_ids) == round(0.1 * len(all_ids))
    train, val = unified_loader(cfg, split="train"), unified_loader(cfg, rand=False, split="val")
    assert _ids(val) and _ids(val) <= val_ids  # a val track may have no window with full history
    assert not _ids(train) & val_ids and not _ids(val) & test_ids and not _ids(train) & test_ids
    assert _ids(train) | val_ids == all_ids
    assert torch.isfinite(next(iter(val))["obs"]).all()  # test-style windows: full history, no NaN padding
    base = unified_loader(_cfg(tmp_path, 0.0), split="train")  # val off: every track trains
    assert _ids(base) == all_ids
    assert len(train.dataset) + sum(w[2].id in val_ids for w in base.dataset.index) == len(base.dataset)


def test_val_split_requires_val_frac(tmp_path):
    with pytest.raises(AssertionError):
        unified_loader(_cfg(tmp_path, 0.0), split="val")


def test_summarize_uses_tuned_final_run_and_refuses_mixed_configs(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    ade, fde = {"A": [5, 4], "B": [3, 2]}, {"A": [9, 8], "B": [7, 6]}
    index = [_write_run(runs, "tuned", "fedadam-06c-eta0.01", "fedadam", ade, fde),
             _write_run(runs, "sweep", "fedadam-06c-val10-eta0.01", "fedadam", {"A": [0, 0], "B": [0, 0]}, {"A": [0, 0], "B": [0, 0]})]
    (runs / "index.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n")
    df = collect(runs, tmp_path / "out", include_smoke=False)[0]
    out = table(df, "06", ["A", "B"], "evaluate", "test", "final", False, ["fedadam"])
    assert "fedadam [fedadam-06c-eta0.01]: n=1 runs: tuned" in out and "sweep" not in out  # validation sweep run stays out
    index.append(_write_run(runs, "paper", "fedadam-06c", "fedadam", ade, fde))
    (runs / "index.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n")
    df = collect(runs, tmp_path / "out2", include_smoke=False)[0]
    with pytest.raises(AssertionError, match="different configurations"):
        table(df, "06", ["A", "B"], "evaluate", "test", "final", False, ["fedadam"])

"""Unit tests of the scaling with one factor per camera view and of the runs on the scaled data."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "preprocessing"), str(ROOT / "tests")]

import process_data as pdata  # noqa: E402
import run_fl  # noqa: E402
import scale_per_camera as spc  # noqa: E402
from data.unified_loader import unified_loader  # noqa: E402
from env_compare import compare_envs, load_env  # noqa: E402
from fl_train_utils import to_cuda  # noqa: E402
from metrics.build_metrics import Build_Metrics  # noqa: E402
from summarize import TAG, clients_file, read_clients, run_key, table  # noqa: E402
from utils import PIXELS_PER_UNIT, load_config  # noqa: E402

TABLE = ROOT / "configs" / "scale_per_camera_23.csv"
CHATTANOOGA = clients_file(ROOT / "configs" / "clients_23_chattanooga.txt")
SMALL = "Lindsay_2"  # the smallest distributed view of the 23-view setting


# ---- factor and table -----------------------------------------------------------------------------------------------------
def test_scale_factor_by_hand():
    assert spc.scale_factor(3.0, 4.0, 1.0, 2.0) == pytest.approx(np.sqrt(5), rel=1e-15)  # sqrt((9 + 16) / (1 + 4))
    np.testing.assert_allclose(spc.scale_factor(np.array([1.0, 0.0]), np.array([0.0, 2.0]), 2.0, 0.0), [0.5, 1.0], rtol=1e-15)
    with pytest.raises(AssertionError):
        spc.scale_factor(0.0, 0.0, 1.0, 1.0)


def test_tracked_table():
    t = spc.read_table(TABLE)  # columns, one pooled pair, k = the formula of its SDs
    assert list(t.index) == read_clients("23")
    assert t.k.idxmin() == "C042" and t.k.idxmax() == "Lindsay_2" and 0.58 < t.k.min() < 0.59 and 1.55 < t.k.max() < 1.56


def test_read_table_refuses_an_inconsistent_table(tmp_path):
    t = pd.read_csv(TABLE)
    for col, row in (("k", 5), ("sd_vx", 0), ("sd_pool_vy", 3)):
        bad = t.copy()
        bad.loc[row, col] *= 1 + 1e-9
        bad.to_csv(tmp_path / "t.csv", index=False)
        with pytest.raises(AssertionError):
            spc.read_table(tmp_path / "t.csv")
    t.rename(columns={"camera": "site"}).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(AssertionError):
        spc.read_table(tmp_path / "t.csv")


# ---- the check of the table on synthetic training files --------------------------------------------------------------------
def _write_camera(d: Path, cam: str, xs: list) -> None:
    """Processed files of one synthetic camera: one track per list of x positions (pixels), y constant, train = test."""
    rows = [{"frame": i, "trackId": t, "x": x, "y": 10.0 * t, "sceneId": "s", "metaId": t} for t, x_t in enumerate(xs) for i, x in enumerate(x_t)]
    df = pd.DataFrame(rows).sort_values(["frame", "trackId"], kind="stable").reset_index(drop=True)
    std = pdata.check_standardization({e: {d_: {"mean": 1.0, "std": 2.0} for d_ in ("x", "y")} for e in ("position", "velocity", "acceleration")})
    for split in ("train", "test"):
        spc.write_env(pdata.process_file(df.copy(), std, is_train=False), d / f"{cam}_{split}.pkl")


def _table(sd: dict, pool: tuple) -> pd.DataFrame:
    t = pd.DataFrame([{"camera": c, "sd_vx": v[0], "sd_vy": v[1], "sd_pool_vx": pool[0], "sd_pool_vy": pool[1]} for c, v in sd.items()])
    return t.assign(k=spc.scale_factor(t.sd_vx, t.sd_vy, t.sd_pool_vx, t.sd_pool_vy))


def test_check_table_by_hand(tmp_path):
    # stored velocities = derivative_of at 0.1 s (backward differences, the first one forward):
    # A: x = 0, 50, 150 px = 0, 1, 3 units -> vx = 10, 10, 20, vy = 0; B: x = 100, 50 px and 0, 50 px -> vx = -10, -10, 10, 10, vy = 0
    _write_camera(tmp_path, "A", [[0, 50, 150]])
    _write_camera(tmp_path, "B", [[100, 50], [0, 50]])
    va, vb = np.array([10.0, 10, 20]), np.array([-10.0, -10, 10, 10])
    np.testing.assert_allclose(spc.velocity_rows(load_env(tmp_path / "A_train.pkl")), np.c_[va, np.zeros(3)], rtol=1e-12)
    sd, pool = {"A": (va.std(), 0.0), "B": (vb.std(), 0.0)}, (np.r_[va, vb].std(), 0.0)
    assert sd["B"][0] == 10.0
    t = _table(sd, pool).set_index("camera")
    np.testing.assert_allclose(t.k, [va.std() / pool[0], 10.0 / pool[0]], rtol=1e-15)
    spc.check_table(tmp_path, ["A", "B"], t)  # all cameras of the table: pooled SDs and factors checked
    spc.check_table(tmp_path, ["B"], t)
    with pytest.raises(AssertionError, match="pooled"):
        spc.check_table(tmp_path, ["A", "B"], _table(sd, (pool[0] * (1 + 1e-9), 0.0)).set_index("camera"))
    spc.check_table(tmp_path, ["B"], _table(sd, (pool[0] * (1 + 1e-9), 0.0)).set_index("camera"))  # a subset: pooled SDs not recomputed
    with pytest.raises(AssertionError, match="B: velocity SDs"):
        spc.check_table(tmp_path, ["B"], _table({**sd, "B": (10.0 * (1 + 1e-9), 0.0)}, pool).set_index("camera"))


def test_scaled_env_by_hand(tmp_path):
    _write_camera(tmp_path, "A", [[0, 50, 150]])
    env = load_env(tmp_path / "A_test.pkl")
    out = spc.scaled_env(env, 0.5)
    st = out.standardization["PEDESTRIAN"]
    assert {(e, d): st[e][d]["std"] for e, d in pdata.STATES} == {
        ("position", "x"): 2.0, ("position", "y"): 2.0, ("velocity", "x"): 1.0, ("velocity", "y"): 1.0,
        ("acceleration", "x"): 1.0, ("acceleration", "y"): 1.0}
    assert all(st[e][d]["mean"] == 1.0 for e, d in pdata.STATES) and list(out.attention_radius.values()) == [7.5]
    assert env.standardization["PEDESTRIAN"]["velocity"]["x"]["std"] == 2.0 and list(env.attention_radius.values()) == [15.0]  # input untouched
    env.attention_radius = {k: 10.0 for k in env.attention_radius}
    with pytest.raises(AssertionError):
        spc.scaled_env(env, 0.5)


# ---- the scaled files of the distributed views -------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scaled_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("scaled")
    subprocess.run([sys.executable, str(ROOT / "preprocessing/scale_per_camera.py"), "--clients", str(ROOT / "configs/clients_23_chattanooga.txt"),
                    "--out_dir", str(out)], check=True, capture_output=True)
    return out


def test_scaled_files_of_the_distributed_views(scaled_dir):
    """Every scaled file = the processed file with the velocity and acceleration stds and the attention radius multiplied by the
    table's k, and equal to it in everything else (compare_envs after restoring the two)."""
    k = spc.read_table(TABLE).k
    assert sorted(p.name for p in scaled_dir.iterdir()) == sorted(f"{c}_{s}.pkl" for c in CHATTANOOGA for s in ("train", "test"))
    assert k["Broad"] < 1 < k["Lindsay_2"]
    for cam in CHATTANOOGA:
        for split in ("train", "test"):
            scaled, raw = load_env(scaled_dir / f"{cam}_{split}.pkl"), load_env(ROOT / f"data/processed_23/{cam}_{split}.pkl")
            st, raw_st = scaled.standardization["PEDESTRIAN"], raw.standardization["PEDESTRIAN"]
            for e, d in pdata.STATES:
                assert st[e][d] == {"mean": raw_st[e][d]["mean"], "std": raw_st[e][d]["std"] * (1.0 if e == "position" else k[cam])}, (cam, e, d)
            assert list(scaled.attention_radius) == list(raw.attention_radius) and list(raw.attention_radius.values()) == [15.0]
            assert list(scaled.attention_radius.values()) == [15.0 * k[cam]], cam
            scaled.standardization, scaled.attention_radius = raw.standardization, raw.attention_radius
            stats = compare_envs(scaled, raw)
            assert stats["nodes"] > 0 and stats["augmented_scenes"] == (24 * stats["scenes"] if split == "train" else 0), (cam, split, stats)


def test_scale_per_camera_lists_missing_files(tmp_path):
    clients = tmp_path / "clients.txt"
    clients.write_text("Broad\nC042\n")
    r = subprocess.run([sys.executable, str(ROOT / "preprocessing/scale_per_camera.py"), "--clients", str(clients), "--out_dir", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "2 of the 4 processed files are missing" in r.stderr and "C042_train.pkl" in r.stderr
    assert not (tmp_path / "o").exists()


@pytest.fixture(scope="module")
def loaders(scaled_dir, tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("cfg"))
    cfgs = {name: load_config(str(ROOT / "configs/flowchain.yml"), SMALL, str(d), tmp, 0)
            for name, d in (("raw", ROOT / "data/processed_23"), ("scaled", scaled_dir))}
    return {name: (cfg, unified_loader(cfg, rand=False, split="test")) for name, cfg in cfgs.items()}


def test_test_windows_are_the_same(loaders):
    keys = {name: [(scene.name, int(t), node.id) for scene, t, node in loader.dataset.index] for name, (_, loader) in loaders.items()}
    assert len(keys["raw"]) == len(set(keys["raw"])) > 0 and keys["raw"] == keys["scaled"]


def test_model_input_is_the_raw_input_scaled_by_1_over_k(loaders):
    raw, scaled = (next(iter(loader)) for _, loader in loaders.values())
    k = float(spc.read_table(TABLE).k[SMALL])
    assert raw["index"] == scaled["index"]
    for key in ("obs", "gt"):
        assert torch.equal(raw[key], scaled[key]), key
    for key in ("obs_st", "gt_st"):  # all 6 input dims (relative position / R, velocity, acceleration) and the velocity target
        torch.testing.assert_close(scaled[key] * k, raw[key], rtol=1e-5, atol=1e-5, equal_nan=True)


@CUDA
def test_metric_is_the_same_in_pixels(loaders):
    """For each data folder: a 'prediction' equal to the standardized target de-standardizes (TP_metrics.denormalize, as in
    fl_train_utils.evaluate_model) to the true future positions; these and the prediction are the same pixels in both folders;
    a +1 px x offset of every future position, made in pixel space and re-standardized with the file's constants, gives ADE = FDE = 1 px."""
    out = {}
    for name, (cfg, loader) in loaders.items():
        metrics = Build_Metrics(cfg)
        batch = to_cuda(next(iter(loader)))
        exact = {**batch, ("pred_st", 0): batch["gt_st"].clone()}
        vel = batch["gt"].clone()  # future velocities, units of 50 px / s; +1 px on every future position = +1 px / dt on the first velocity
        vel[:, 0, 0] += 1 / PIXELS_PER_UNIT / batch["dt"]
        shifted = {**batch, ("pred_st", 0): (vel - metrics.mean) / metrics.std}
        exact, shifted = metrics.denormalize([exact])[0], metrics.denormalize([shifted])[0]
        assert exact["gt"].shape[1:] == (12, 2), exact["gt"].shape
        torch.testing.assert_close(exact[("pred", 0)] * PIXELS_PER_UNIT, exact["gt"] * PIXELS_PER_UNIT, rtol=0, atol=1e-4)
        per = metrics([shifted])["per_sample"]
        np.testing.assert_allclose(per["ade"] * PIXELS_PER_UNIT, 1.0, rtol=0, atol=1e-4)
        np.testing.assert_allclose(per["fde"] * PIXELS_PER_UNIT, 1.0, rtol=0, atol=1e-4)
        out[name] = (exact["gt"] * PIXELS_PER_UNIT, exact[("pred", 0)] * PIXELS_PER_UNIT, shifted[("pred", 0)] * PIXELS_PER_UNIT)
    for a, b in zip(out["raw"], out["scaled"]):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-4)


# ---- run_fl.py --scaled -------------------------------------------------------------------------------------------------------
def _args(*words):
    return run_fl.parser().parse_args(["--seed", "0", *words])


@pytest.mark.parametrize("strategy", ["fedavg", "fedrep"])
def test_run_fl_scaled_tag_and_hparams(strategy):
    a = _args("--strategy", strategy, "--clients", "23", "--scaled")
    assert run_fl.run_tag(a) == f"{strategy}-23c-scaled" and run_fl.run_hparams(a) == {"val_frac": 0.0, "scaled": True}
    assert run_fl.run_tag(_args("--strategy", strategy, "--clients", "23", "--scaled", "--max_batches", "2")) == f"smoke-{strategy}-23c-scaled"
    c = _args("--strategy", strategy, "--clients", "23")  # without the flag: unchanged
    assert run_fl.run_tag(c) == f"{strategy}-23c" and run_fl.run_hparams(c) == {"val_frac": 0.0}


@pytest.mark.parametrize("words", [["--strategy", "fedprox", "--clients", "23"], ["--strategy", "fedper", "--clients", "23"],
                                   ["--strategy", "fedavg", "--clients", "06"], ["--strategy", "fedrep", "--clients", "06"]])
def test_run_fl_scaled_refuses_other_strategies_and_06(words):
    with pytest.raises(AssertionError, match="--scaled needs"):
        run_fl.run_tag(_args(*words, "--scaled"))


def test_run_fl_scaled_stops_on_missing_files(tmp_path, monkeypatch):
    """Default folder data/processed_23_scaled (here under a monkeypatched root, empty), or --data_dir; every missing file is listed
    with the hint, before any process starts."""
    monkeypatch.setattr(run_fl, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_fl.py", "--strategy", "fedrep", "--clients", "23", "--seed", "0", "--scaled"])
    with pytest.raises(FileNotFoundError, match="46 of the 46 processed files are missing") as e:
        run_fl.main()
    assert str(tmp_path / "data" / "processed_23_scaled") in str(e.value) and str(e.value).endswith("preprocessing/scale_per_camera.py or pass --data_dir.")
    d = tmp_path / "mine"
    d.mkdir()
    for c in CHATTANOOGA:
        for s in ("train", "test"):
            (d / f"{c}_{s}.pkl").write_bytes(b"")
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--data_dir", str(d)])
    with pytest.raises(FileNotFoundError, match="26 of the 46 processed files are missing") as e:
        run_fl.main()
    assert "C042_test.pkl" in str(e.value) and "Broad_train.pkl" not in str(e.value)
    assert not (tmp_path / "runs").exists()


# ---- summarize ------------------------------------------------------------------------------------------------------------------
def test_scaled_runs_are_their_own_strategy():
    tags = ("fedavg-23c-scaled", "smoke-fedrep-23c-scaled", "fedrep-23c", "fedavg-23c")
    assert [run_key(TAG.fullmatch(t)) for t in tags] == ["fedavg-scaled", "fedrep-scaled", "fedrep", "fedavg"]
    for bad in ("fedprox-23c-scaled", "fedavg-06c-scaled", "local-23c-scaled"):
        with pytest.raises(AssertionError):
            run_key(TAG.fullmatch(bad))
    df = pd.DataFrame([{"run_id": r, "tag": tag, "strategy": tag.split("-")[0], "phase": "evaluate", "split": "test", "intersection": c,
                        "round": 1, "ade_px": v, "fde_px": 2 * v, "ade_legacy_px": v, "fde_legacy_px": v}
                       for r, tag, vals in (("rep", "fedrep-23c", (1, 2)), ("rs", "fedrep-23c-scaled", (5, 7)))
                       for c, v in zip("AB", vals)])
    out = table(df, "23", ["A", "B"], "evaluate", "test", "final", False, ["fedrep", "fedrep-scaled"])
    assert "# fedrep [fedrep-23c]: n=1 runs: rep" in out and "# fedrep-scaled [fedrep-23c-scaled]: n=1 runs: rs" in out
    assert "Average\t1.500 (n=1)\t6.000 (n=1)" in out

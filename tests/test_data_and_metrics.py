import hashlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.unified_loader import load_env, unified_loader  # noqa: E402
from fl_train_utils import to_cuda  # noqa: E402
from metrics.build_metrics import Build_Metrics  # noqa: E402
from metrics.TP_metrics import TP_metrics, displacement_error, evaluate_helper, final_displacement_error  # noqa: E402
from data.TP.trajectron_dataset import hypers  # noqa: E402
from summarize import clients_file  # noqa: E402
from utils import load_config  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CLIENTS = {"06": clients_file(ROOT / "configs/clients_06.txt"), "23": clients_file(ROOT / "configs/clients_23_chattanooga.txt")}
SETS = [(nn, c) for nn in ("06", "23") for c in CLIENTS[nn]]  # 23: the distributed (Chattanooga) views


def test_data_matches_checksums():
    """Every distributed data file is byte-identical to the one recorded in data/CHECKSUMS.txt (sha256), and every file is listed."""
    data = ROOT / "data"
    entries = [l.split() for l in (data / "CHECKSUMS.txt").read_text().splitlines() if l.strip()]
    assert len(entries) == 52, len(entries)  # 20 raw trajectory files + 12 in processed_06 + 20 in processed_23
    assert sorted(rel for _, rel in entries) == sorted(str(p.relative_to(data)) for d in ("raw_trajectories", "processed_06", "processed_23")
                                                       for p in (data / d).glob("*.pkl"))
    for digest, rel in entries:
        assert hashlib.sha256((data / rel).read_bytes()).hexdigest() == digest, rel


def _cfg(nn, name, tmp_path):
    return load_config(str(ROOT / "configs/flowchain.yml"), name, str(ROOT / f"data/processed_{nn}"), str(tmp_path), 0)


# ---- metric ---------------------------------------------------------------------------------------------
def test_ade_fde_constant_offset():
    gt = torch.zeros(4, 12, 2)
    pred = gt + torch.tensor([3.0, 4.0])  # every step is 5 away
    assert torch.allclose(displacement_error(pred, gt), torch.full((4,), 5.0))
    assert torch.allclose(final_displacement_error(pred[:, -1], gt[:, -1]), torch.full((4,), 5.0))


def test_ade_is_mean_over_time():
    gt = torch.zeros(1, 12, 2)
    pred = gt.clone()
    pred[0, -1] = torch.tensor([12.0, 0.0])  # only the last step is wrong, by 12
    assert displacement_error(pred, gt).item() == pytest.approx(1.0)
    assert final_displacement_error(pred[:, -1], gt[:, -1]).item() == pytest.approx(12.0)


def test_best_of_n_takes_per_sample_min_then_sums():
    # 3 trials x 2 samples
    errors = [torch.tensor([1.0, 9.0]), torch.tensor([4.0, 2.0]), torch.tensor([3.0, 5.0])]
    assert evaluate_helper(errors).item() == pytest.approx(1.0 + 2.0)


def _denormalize_and_score(metrics, data_dict, n_trials=2):
    dict_list = metrics.denormalize([{k: v.clone() if torch.is_tensor(v) else v for k, v in data_dict.items()}
                                     for _ in range(n_trials)])
    return metrics(dict_list)


@CUDA
def test_velocity_metric_known_values():
    """pred_st = gt_st: corrected ADE/FDE are 0; legacy (velocity mean not added back) is off by |m|*dt*k at step k."""
    B, T, dt = 3, 12, 0.1
    mean, std = torch.tensor([0.3, -0.4]).cuda(), torch.tensor([2.0, 0.5]).cuda()  # |mean| = 0.5
    gen = torch.Generator().manual_seed(0)
    gt = torch.randn(B, T, 2, generator=gen).cuda()
    metrics = TP_metrics.__new__(TP_metrics)  # skip __init__: it loads a real env
    metrics.state, metrics.mean, metrics.std = hypers["state_v"]["PEDESTRIAN"], mean, std
    out = _denormalize_and_score(metrics, {"obs": torch.randn(B, 8, 6, generator=gen).cuda(), "gt": gt,
                                           "gt_st": (gt - mean) / std, ("pred_st", 0): (gt - mean) / std,
                                           "dt": torch.full((B,), dt).cuda()})
    n = out["nsample"]
    assert n == B
    assert out["ade"] / n == pytest.approx(0.0, abs=1e-5) and out["fde"] / n == pytest.approx(0.0, abs=1e-5)
    assert out["ade_legacy"] / n == pytest.approx(0.5 * dt * (T + 1) / 2, abs=1e-5)
    assert out["fde_legacy"] / n == pytest.approx(0.5 * dt * T, abs=1e-5)


@CUDA
def test_velocity_metric_real_batch(tmp_path):
    """Real Broad test batch with pred_st = gt_st: corrected ADE ~ 0, legacy ADE > 0."""
    cfg = _cfg("06", "Broad", tmp_path)
    batch = to_cuda(next(iter(unified_loader(cfg, rand=False, split="test"))))
    batch[("pred_st", 0)] = batch["gt_st"].clone()
    out = _denormalize_and_score(Build_Metrics(cfg), batch)
    assert out["ade"] / out["nsample"] < 1e-5 and out["fde"] / out["nsample"] < 1e-5
    assert out["ade_legacy"] / out["nsample"] > 1e-3


# ---- data -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nn,name", SETS)
def test_train_test_tracks_disjoint(nn, name, tmp_path):
    cfg = _cfg(nn, name, tmp_path)
    ids = {split: {n.id for s in load_env(cfg, split).scenes for n in s.nodes} for split in ("train", "test")}
    assert ids["train"] and ids["test"]
    assert not ids["train"] & ids["test"], f"{name}: tracks in both splits"


@pytest.mark.parametrize("nn,name", [SETS[0], SETS[-1]])
def test_loader_batch(nn, name, tmp_path):
    cfg = _cfg(nn, name, tmp_path)
    for split in ("train", "test"):
        batch = next(iter(unified_loader(cfg, rand=False, split=split)))
        B = batch["obs"].shape[0]
        assert B == cfg.DATA.BATCH_SIZE
        assert batch["obs"].shape == (B, cfg.DATA.OBSERVE_LENGTH, 6)  # state_pva: x, y, vx, vy, ax, ay
        assert batch["gt"].shape == (B, cfg.DATA.PREDICT_LENGTH, 2)
        assert batch["obs_st"].shape == batch["obs"].shape and batch["gt_st"].shape == batch["gt"].shape
        assert torch.isfinite(batch["gt"]).all()
        assert torch.allclose(batch["dt"].double(), torch.full((B,), 0.1, dtype=torch.float64))
        if split == "test":  # train allows NaN-padded history (ACCEPT_NAN); test requires full history
            assert torch.isfinite(batch["obs"]).all()


@pytest.mark.parametrize("nn", ["06", "23"])
def test_positions_are_pixels_over_50(nn, tmp_path):
    """Preprocessing divided pixel coordinates by 50; frames are at most ~1920 px wide."""
    cfg = _cfg(nn, CLIENTS[nn][0], tmp_path)
    batch = next(iter(unified_loader(cfg, rand=False, split="test")))
    pos = batch["obs"][..., :2]
    assert 0 <= pos.min() and pos.max() <= 1920 / 50 + 1


def test_standardization_matches_provenance(tmp_path):
    """processed_23 carries the constants of the 23 views, processed_06 those of the six intersections."""
    expected = {"23": (18.091689, 11.023442), "06": (19.882223, 11.292678)}  # position.x mean, std
    for nn, clients in CLIENTS.items():
        for name in clients:
            std = load_env(_cfg(nn, name, tmp_path), "train").standardization["PEDESTRIAN"]["position"]["x"]
            assert (std["mean"], std["std"]) == pytest.approx(expected[nn]), (nn, name, std)

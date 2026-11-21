import itertools
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from yacs.config import CfgNode
from copy import deepcopy
from tqdm import tqdm

from metrics.build_metrics import Build_Metrics
from utils import PIXELS_PER_UNIT


def to_cuda(data_dict: Dict) -> Dict:
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}


def limited(loader, max_batches: Optional[int]):
    """Iterate over `loader`, stopping after `max_batches` batches (smoke tests only)."""
    n = len(loader) if max_batches is None else min(max_batches, len(loader))
    return itertools.islice(loader, n), n


def fl_local_train(cfg: CfgNode, model: torch.nn.Module, train_loader, num_epochs: int = 1,
                   max_batches: Optional[int] = None) -> float:
    """Local training loop shared by FedAvg, FedAdam, FedProx, FedCross and FedRep clients.

    Note (kept on purpose to match the published results):
    `model.update()` already runs zero_grad/backward/step on the model's own Adam, so the
    `optimizer.step()` below applies a second Adam update from a fresh optimizer every round.
    """
    assert not hasattr(model, "model"), "expected a bare fastpredNF_TP model"
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    losses: List[float] = []

    for epoch in range(num_epochs):
        batches, n = limited(train_loader, max_batches)
        pbar = tqdm(batches, total=n, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data_dict in pbar:
            data_dict = to_cuda(data_dict)
            optimizer.zero_grad()
            loss_info = model.update(data_dict)
            assert not isinstance(loss_info["loss"], torch.Tensor), "update() is expected to step internally"
            optimizer.step()
            losses.append(loss_info["loss"])
            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_info.items()})

    assert losses, "no training batches"
    return sum(losses) / len(losses)


def window_table(data_dict: Dict, per_sample: Dict) -> pd.DataFrame:
    """One row per evaluated window, in pixels. `data_dict` is a denormalized trial dict (obs = x, y, vx, vy, ax, ay of the
    8 observed steps, gt = absolute positions of the 12 future steps), `per_sample` the best-of-N_TRIAL errors.
    With p0 = last observed position and g1..g12 the future positions:
      gt_endpoint_disp_px  = |g12 - p0|
      gt_path_len_px       = |g1 - p0| + sum_{k=2..12} |gk - g(k-1)|
      obs_last_speed_px_s  = |(vx, vy)| at the last observed step (the velocity state is the position difference / dt)
    """
    obs, gt = data_dict["obs"], data_dict["gt"]
    p0 = obs[:, -1, :2]
    path = torch.cat([p0[:, None], gt], dim=1)
    index = data_dict["index"]
    assert len(index) == len(obs) == len(gt) and gt.shape[1:] == (12, 2), (len(index), obs.shape, gt.shape)
    scale = PIXELS_PER_UNIT
    table = pd.DataFrame({
        "scene": [i[0] for i in index], "t": [int(i[1]) for i in index], "node_id": [i[2].split("/", 1)[1] for i in index],
        **{f"{k}_px": v * scale for k, v in per_sample.items()},
        "gt_endpoint_disp_px": (scale * (gt[:, -1] - p0).norm(dim=1)).cpu().numpy(),
        "gt_path_len_px": (scale * (path[:, 1:] - path[:, :-1]).norm(dim=2).sum(dim=1)).cpu().numpy(),
        "obs_last_speed_px_s": (scale * obs[:, -1, 2:4].norm(dim=1)).cpu().numpy()})
    assert np.isfinite(table.select_dtypes("number").to_numpy()).all()
    return table


def evaluate_model(cfg: CfgNode, model: torch.nn.Module, data_loader, max_batches: Optional[int] = None,
                   return_per_sample: bool = False):
    """Best-of-N_TRIAL ADE/FDE on `data_loader` (denormalized), plus mean inference time per call.

    `ade_legacy`/`fde_legacy` are the original evaluation code's values (velocity mean not added back), same samples.
    `n_windows` is the number of windows evaluated. With return_per_sample, returns (metrics, window_table): the
    per-window errors come from the same forward passes, and their mean equals the aggregate ADE/FDE.
    """
    model.eval()
    metrics = Build_Metrics(cfg)
    inference_times = []

    with torch.no_grad():
        result_list, tables = [], []
        batches, _ = limited(data_loader, max_batches)
        for data_dict in batches:
            data_dict = to_cuda(data_dict)
            dict_list = []
            for _ in range(cfg.TEST.N_TRIAL):
                start = time.time()
                result_dict = model.predict(deepcopy(data_dict), return_prob=False)
                inference_times.append(time.time() - start)
                dict_list.append(deepcopy(result_dict))
            dict_list = metrics.denormalize(dict_list)
            result = metrics(dict_list)
            per_sample = result.pop("per_sample")
            if return_per_sample:
                tables.append(window_table(dict_list[0], per_sample))
            result_list.append(deepcopy(result))

    assert result_list, "no evaluation batches"
    d = aggregate(result_list)
    result_info = {k: float(d[k]) for k in d if d[k] != 0.0}
    for k in ("ade", "fde", "ade_legacy", "fde_legacy"):
        assert np.isfinite(result_info[k]), (k, result_info[k])
    result_info["inference_time"] = float(np.mean(inference_times))
    result_info["n_windows"] = int(sum(r["nsample"] for r in result_list))

    model.train()
    if not return_per_sample:
        return result_info
    table = pd.concat(tables, ignore_index=True)
    assert len(table) == result_info["n_windows"], (len(table), result_info["n_windows"])
    for k in ("ade", "fde", "ade_legacy", "fde_legacy"):
        np.testing.assert_allclose(table[f"{k}_px"].mean() / PIXELS_PER_UNIT, result_info[k], rtol=1e-5, err_msg=k)
    return result_info, table


def aggregate(dict_list: List[Dict]) -> Dict:
    """Sample-weighted average of per-batch metric sums."""
    assert "nsample" in dict_list[0]
    total = np.sum([d["nsample"] for d in dict_list])
    return {k: np.sum([d[k] for d in dict_list], axis=0) / total for k in dict_list[0].keys()}

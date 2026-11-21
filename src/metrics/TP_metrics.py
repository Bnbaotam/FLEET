from copy import deepcopy
import numpy as np
import torch
import ot
from yacs.config import CfgNode
from typing import List, Dict
from data.TP.trajectron_dataset import hypers
from data.unified_loader import load_env
import warnings
warnings.simplefilter('ignore')


_CONSTANTS = {}  # (train pkl, PRED_STATE) -> (mean, std); the train env is read once per process, not per evaluation


def _standardization(cfg: CfgNode, state: Dict, node: str):
    key = (cfg.DATA.DIR, cfg.DATA.DATASET_NAME, cfg.DATA.TP.PRED_STATE)
    if key not in _CONSTANTS:
        env = load_env(cfg, 'train')
        mean, std = env.get_standardize_params(state, node)
        if cfg.DATA.TP.PRED_STATE == 'state_p':
            std = env.attention_radius[('PEDESTRIAN', 'PEDESTRIAN')]
        _CONSTANTS[key] = (np.array(mean), np.array(std))
    return _CONSTANTS[key]


class TP_metrics(object):
    def __init__(self, cfg: CfgNode):
        node = 'PEDESTRIAN'
        self.state = hypers[cfg.DATA.TP.PRED_STATE][node]

        # TODO: params for other nodes
        mean, std = _standardization(cfg, self.state, node)
        self.mean = torch.Tensor(mean).cuda()
        self.std = torch.Tensor(std).cuda()

        self.dataset = cfg.DATA.DATASET_NAME

    # TODO : calculate metrics on updated trajectory e.g. ("pred", 1), ("pred", 2)
    def __call__(self, dict_list: List) -> Dict:
        ade, fde, ade_legacy, fde_legacy, emd, log_prob = [], [], [], [], [], []
        for data_dict in dict_list:
            ade.append(displacement_error(
                data_dict[('pred', 0)][:, -12:], data_dict['gt'][:, -12:]))
            fde.append(final_displacement_error(
                data_dict[('pred', 0)][:, -1], data_dict['gt'][:, -1]))
            ade_legacy.append(displacement_error(
                data_dict[('pred_legacy', 0)][:, -12:], data_dict['gt'][:, -12:]))
            fde_legacy.append(final_displacement_error(
                data_dict[('pred_legacy', 0)][:, -1], data_dict['gt'][:, -1]))
            emd.append(self.emd(data_dict))
            log_prob.append(self.log_prob(data_dict))

        # best-of-N error per window; the batch sums below are what the aggregate ADE/FDE are built from
        per_sample = {"ade": best_of_n(ade), "fde": best_of_n(fde),
                      "ade_legacy": best_of_n(ade_legacy), "fde_legacy": best_of_n(fde_legacy)}
        ade, fde, ade_legacy, fde_legacy = (per_sample[k].sum(dim=0) for k in ("ade", "fde", "ade_legacy", "fde_legacy"))
        emd = evaluate_helper(emd)
        log_prob = evaluate_helper(log_prob)

        return {"score": ade.cpu().numpy(),
                "ade": ade.cpu().numpy(),
                "fde": fde.cpu().numpy(),
                "ade_legacy": ade_legacy.cpu().numpy(),
                "fde_legacy": fde_legacy.cpu().numpy(),
                "emd": emd.cpu().numpy(),
                "log_prob": log_prob.cpu().numpy(),
                "nsample": len(data_dict[('pred', 0)]),
                "per_sample": {k: v.cpu().numpy() for k, v in per_sample.items()}}

    def denormalize(self, dict_list: List) -> List:
        for data_dict in dict_list:
            if not ("pred", 0) in data_dict.keys():
                data_dict = self.unstandardize(data_dict)
                data_dict = self.output_to_trajectories(data_dict)
        return dict_list

    def output_to_trajectories(self, data_dict: Dict) -> Dict:
        # see preprocessing.py
        assert 'acceleration' not in self.state.keys()
        if 'position' in self.state.keys():
            for key in (('pred', 0), ('pred_legacy', 0)):
                data_dict[key] += data_dict['obs'][:, -1:, 0:2]
            for k in data_dict.keys():
                if k[0] == "prob" and type(data_dict[k]) == torch.Tensor:
                    offset = data_dict['obs'][:, None, -1:,
                                              0:2] if k[1] == 0 else data_dict['gt'][:, None, k[1]-1:k[1], 0:2]
                    data_dict[k][..., :2] += offset

        elif 'velocity' in self.state.keys():
            for key in (('pred', 0), ('pred_legacy', 0)):
                data_dict[key] = torch.cumsum(data_dict[key], dim=1) * \
                    data_dict['dt'][:, None, None] + data_dict['obs'][:, -1:, 0:2]
            data_dict['gt'] = torch.cumsum(
                data_dict['gt'], dim=1) * data_dict['dt'][:, None, None] + data_dict['obs'][:, -1:, 0:2]
            for k in data_dict.keys():
                if k[0] == "prob" and type(data_dict[k]) == torch.Tensor:
                    offset = data_dict['obs'][:, None, -1:,
                                              0:2] if k[1] == 0 else data_dict['gt'][:, None, k[1]-1:k[1], 0:2]
                    data_dict[k][..., :2] = torch.cumsum(
                        data_dict[k][..., :2], dim=2) * data_dict['dt'][:, None, None] + offset

        return data_dict

    def unstandardize(self, data_dict: Dict) -> Dict:
        # assume we observed positions
        assert 'acceleration' not in self.state.keys()
        if 'position' in self.state.keys():
            assert torch.all(data_dict['gt_st'] * self.std +
                             data_dict['obs'][:, -1:, 0:2] - data_dict['gt'] < 1e-5)
        elif 'velocity' in self.state.keys():
            # velocity targets are standardized with the env's (non-zero) mean, see preprocessing.py
            assert torch.all(torch.abs(data_dict['gt_st'] * self.std +
                                       self.mean - data_dict['gt']) < 1e-5)

        # 'pred_legacy' omits the velocity mean (as the original evaluation code did), from the SAME sample as 'pred'
        shift = self.mean if 'velocity' in self.state.keys() else 0.
        data_dict[('pred_legacy', 0)] = data_dict[('pred_st', 0)] * self.std
        data_dict[('pred', 0)] = data_dict[('pred_legacy', 0)] + shift

        for k in list(data_dict.keys()):
            if k[0] == "prob_st" and type(data_dict[k]) == torch.Tensor:
                data_dict[("prob", k[1])] = data_dict[k].clone()
                data_dict[("prob", k[1])][..., :2] = data_dict[("prob", k[1])][..., :2] * self.std + shift
                data_dict[("prob", k[1])][..., -
                                          1] = torch.exp(data_dict[("prob", k[1])][..., -1])

        return data_dict

    def emd(self, data_dict):
        if ("prob", 0) not in data_dict or ("gt_prob") not in data_dict:
            return torch.Tensor([0.0])

        key = None
        for k in data_dict.keys():
            if k[0] == "prob":
                key = k
        prob = torch.Tensor(data_dict[key]).cuda() + 1e-12
        prob /= prob.sum(dim=1, keepdim=True)
        gt_prob = torch.Tensor(data_dict["gt_prob"])[:, :, key[1]:]
        gt_prob /= gt_prob.sum(dim=1, keepdim=True)

        X, Y = data_dict["grid"]
        coords = torch.Tensor(np.array([X.flatten(), Y.flatten()]).T).cuda()
        coordsSqr = torch.sum(coords**2, dim=1)
        M = coordsSqr[:, None] + coordsSqr[None, :] - 2*coords.matmul(coords.T)
        M[M < 0] = 0
        M = torch.sqrt(M)
        emd = []
        for b in range(prob.shape[0]):
            emd_ = []
            for t in range(prob.shape[-1]):
                emd__ = ot.sinkhorn2(
                    prob[0, :, t], gt_prob[0, :, t], M, 1.0, warn=False)
                print(emd__)
                emd_.append(emd__)
            emd.append(torch.Tensor(emd_).mean(dim=-1))
        emd = torch.Tensor(emd)
        print(emd)
        return emd

    def log_prob(self, data_dict):
        # we assume batch_size = 1
        if ("gt_traj_log_prob", 0) in data_dict:
            """
            target_frame = 5
            gt_traj_log_prob = torch.zeros_like(data_dict[("gt_traj_log_prob", 0)])[:, target_frame:]
            for k in data_dict.keys():
                if k[0] == "gt_traj_log_prob":
                    gt_traj_log_prob[:, k[1]:] = data_dict[k][:, target_frame:]
            print(gt_traj_log_prob)
            return gt_traj_log_prob
            """
            return data_dict[("gt_traj_log_prob", 0)].mean(dim=-1)

        else:
            return torch.Tensor([0.0])


def best_of_n(error):
    """List of N per-trial error tensors (batch,) -> per-sample minimum over trials (batch,)."""
    return torch.min(torch.stack(error, dim=0), dim=0)[0]


def evaluate_helper(error):
    return best_of_n(error).sum(dim=0)


def displacement_error(pred_traj, gt_traj):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - gt_traj: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    Output:
    - loss: gives the eculidian displacement error
    """
    return torch.norm(pred_traj - gt_traj, dim=2).mean(dim=1)


def final_displacement_error(pred_pos, gt_pos):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - gt_pos: Tensor of shape (batch, 2). Groud truth
    last pos
    Output:
    - loss: gives the euclidian displacement error
    """
    return torch.norm(pred_pos - gt_pos, dim=1)

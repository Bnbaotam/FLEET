# Adapted from FlowChain (Maeda & Ukita, 2023), which builds on Trajectron++ and MID.
from pathlib import Path
import dill
import numpy as np
from yacs.config import CfgNode
from torch.utils.data import DataLoader

from .TP.trajectron_dataset import EnvironmentDataset, hypers
from .TP.preprocessing import dict_collate


def env_path(cfg: CfgNode, split: str) -> Path:
    assert split in ("train", "test"), split
    path = Path(cfg.DATA.DIR) / f"{cfg.DATA.DATASET_NAME}_{split}.pkl"
    assert path.is_file(), f"missing processed data file: {path}"
    return path


def load_env(cfg: CfgNode, split: str):
    with open(env_path(cfg, split), 'rb') as f:
        return dill.load(f, encoding='latin1')


def val_track_ids(env, val_frac: float) -> set:
    """VAL_FRAC of the train tracks (node ids). The carve seed is fixed at 0, independent of the run seed,
    so every strategy and seed gets the same split."""
    ids = sorted({n.id for s in env.scenes for n in s.nodes})
    k = round(val_frac * len(ids))
    assert 0 < k < len(ids), (val_frac, len(ids))
    return {ids[i] for i in np.random.default_rng(0).choice(len(ids), size=k, replace=False)}


def make_loader(cfg: CfgNode, dataset, rand: bool, drop_last: bool, batch_size=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.DATA.BATCH_SIZE if batch_size is None else batch_size,
        shuffle=rand,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=dict_collate,
        drop_last=drop_last,
        pin_memory=True)


def unified_loader(cfg: CfgNode, rand=True, split="train", batch_size=None) -> DataLoader:
    """split "train" | "test" | "val". "val" (only with cfg.DATA.VAL_FRAC > 0) holds the carved train tracks, with
    windows built like test windows (full 8-step history, no ACCEPT_NAN padding), because that is what the model is
    evaluated on; "train" then excludes those tracks. With VAL_FRAC = 0 the loaders are unchanged."""
    assert cfg.DATA.TASK == "TP", cfg.DATA.TASK
    assert split in ("train", "val", "test"), split
    assert split != "val" or cfg.DATA.VAL_FRAC > 0
    # The original loader shortened OBSERVE_LENGTH by (PREDICT_LENGTH - 12) for eval splits; that is a no-op at 12.
    assert cfg.DATA.PREDICT_LENGTH == 12, cfg.DATA.PREDICT_LENGTH

    env = load_env(cfg, "test" if split == "test" else "train")
    # NOTE: node/scene freq-mult arguments are swapped relative to their names; kept as in the original code.
    dataset = EnvironmentDataset(env,
                                 state=hypers[cfg.DATA.TP.STATE],
                                 pred_state=hypers[cfg.DATA.TP.PRED_STATE],
                                 node_freq_mult=hypers['scene_freq_mult_train'],
                                 scene_freq_mult=hypers['node_freq_mult_train'],
                                 hyperparams=hypers,
                                 min_history_timesteps=1 if cfg.DATA.TP.ACCEPT_NAN and split == 'train' else cfg.DATA.OBSERVE_LENGTH - 1,
                                 min_future_timesteps=cfg.DATA.PREDICT_LENGTH)

    # only the 'PEDESTRIAN' node type exists in these environments
    node_datasets = [d for d in dataset if d.node_type == 'PEDESTRIAN']
    assert len(node_datasets) == 1, [d.node_type for d in dataset]
    if split != "test" and cfg.DATA.VAL_FRAC > 0:
        val_ids = val_track_ids(env, cfg.DATA.VAL_FRAC)
        node_datasets[0].index = [w for w in node_datasets[0].index if (w[2].id in val_ids) == (split == "val")]
        node_datasets[0].len = len(node_datasets[0].index)
        assert node_datasets[0].len > 0

    return make_loader(cfg, node_datasets[0], rand, drop_last=split == 'train', batch_size=batch_size)

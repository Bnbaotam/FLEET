import os
import random
import numpy as np
import torch
from yacs.config import CfgNode

# Preprocessing divides pixel coordinates by 50 (preprocessing/process_data.py), so raw ADE/FDE,
# positions and velocities are in units of 50 px. Every reported number is in pixels (= raw * PIXELS_PER_UNIT).
PIXELS_PER_UNIT = 50.0


def load_config(config_file: str, dataset_name: str, data_dir: str, output_dir: str, seed: int, val_frac: float = 0.0) -> CfgNode:
    """Default params <- shared yml <- per-run fields set by the launcher. val_frac = 0.0 means no validation carve."""
    from default_params import _C
    assert os.path.isfile(config_file), config_file
    assert os.path.isdir(data_dir), data_dir
    cfg = _C.clone()
    cfg.merge_from_file(config_file)
    cfg.DATA.DATASET_NAME = dataset_name
    cfg.DATA.DIR = data_dir
    cfg.DATA.NUM_WORKERS = 0
    cfg.OUTPUT_DIR = output_dir
    cfg.SEED = seed
    assert 0.0 <= val_frac < 1.0, val_frac
    cfg.DATA.VAL_FRAC = val_frac
    cfg.freeze()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimizer_to_cuda(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

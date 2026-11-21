from yacs.config import CfgNode
import torch.nn as nn
from models.TP.TP_models import Build_TP_model


def Build_Model(cfg: CfgNode) -> nn.Module:
    assert cfg.DATA.TASK == "TP", cfg.DATA.TASK
    return Build_TP_model(cfg)

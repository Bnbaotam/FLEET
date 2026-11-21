from yacs.config import CfgNode
from typing import Callable


def Build_Metrics(cfg: CfgNode) -> Callable:
    assert cfg.DATA.TASK == "TP", cfg.DATA.TASK
    from metrics.TP_metrics import TP_metrics
    return TP_metrics(cfg)

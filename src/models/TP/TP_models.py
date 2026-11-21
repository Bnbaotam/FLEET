from yacs.config import CfgNode


def Build_TP_model(cfg: CfgNode):
    assert "fastpredNF" in cfg.MODEL.TYPE, f"only FlowChain (fastpredNF*) is kept in this repo: {cfg.MODEL.TYPE}"
    from models.TP.fastpredNF import fastpredNF_TP
    return fastpredNF_TP(cfg).cuda()

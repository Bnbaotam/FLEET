from yacs.config import CfgNode as CN

_C = CN()

_C.SEED = 12345

_C.OUTPUT_DIR = ""  # set by the launcher (runs/<run>/<intersection>)

_C.DATA = CN()
_C.DATA.DIR = ""  # dir holding <DATASET_NAME>_{train,test}.pkl, set by the launcher
_C.DATA.TASK = "TP"
_C.DATA.DATASET_NAME = "zara1"
_C.DATA.OBSERVE_LENGTH = 8
_C.DATA.PREDICT_LENGTH = 12
_C.DATA.SKIP = 1
_C.DATA.BATCH_SIZE = 128
_C.DATA.NUM_WORKERS = 8
_C.DATA.VAL_FRAC = 0.0  # fraction of TRAIN tracks carved out as validation set; 0 = off (set by the launcher)


_C.DATA.TP = CN()

_C.DATA.TP.STATE = 'state_pva'  # should be fixed
_C.DATA.TP.PRED_STATE = 'state_p'
_C.DATA.TP.ACCEPT_NAN = False

_C.MODEL = CN()
_C.MODEL.ENCODER_TYPE = "transformer"
_C.MODEL.TYPE = "COPY_LAST"
_C.MODEL.FLOW = CN()
_C.MODEL.FLOW.ARCHITECTURE = 'realNVP'
_C.MODEL.FLOW.N_BLOCKS = 3
_C.MODEL.FLOW.N_HIDDEN = 2
_C.MODEL.FLOW.HIDDEN_SIZE = 64
_C.MODEL.FLOW.CONDITIONING_LENGTH = 16

_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER= "adam"
_C.SOLVER.LR = 5e-4
_C.SOLVER.ITER = 20 #100
_C.SOLVER.SAVE_EVERY = 10
_C.SOLVER.USE_SCHEDULER = False
_C.SOLVER.VALIDATION = True
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.DEQUANTIZE = False

_C.TEST = CN()
_C.TEST.N_TRIAL = 20
_C.TEST.KDE = False


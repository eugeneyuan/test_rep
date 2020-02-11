import os
import time
from yacs.config import CfgNode as CN

from src.utils.miscs import IS_ON_SERVER


_C = CN()
_C.EXP_NAME = ""
_C.DEVICE = "cuda"  # "cpu"

_C.DATA = CN()
_C.DATA.DATA_DIR = r'/mnt/lustre/yuanjing1/Datasets/cbct2ct/relative_zyx_200108' if IS_ON_SERVER \
            else r'/home/SENSETIME/yuanjing1/Datasets/cbct2ct/relative_zyx_200108'
_C.DATA.CBCT_DIR = os.path.join(_C.DATA.DATA_DIR, 'cbct')
_C.DATA.CBCT_SEG_DIR = os.path.join(_C.DATA.DATA_DIR, 'cbct_seg')
_C.DATA.CT_DIR = os.path.join(_C.DATA.DATA_DIR, 'ct')
_C.DATA.CT_SEG_DIR = os.path.join(_C.DATA.DATA_DIR, 'ct_seg')
# set data dir for train
_C.DATA.SRC_DIR = _C.DATA.CBCT_DIR
_C.DATA.DST_DIR = _C.DATA.CT_DIR
_C.DATA.MASK_DIR = _C.DATA.CBCT_SEG_DIR
# set dataset property
_C.DATA.INPUT_SP = [0.5, 0.5, 1.0]
_C.DATA.INPUT_SHAPE = [3, 384, 384]
_C.DATA.AXIS = "z"
_C.DATA.RANDOM_SEED = 30
_C.DATA.TRAIN_RATIO = 0.6
_C.DATA.VAL_RATIO = 0.2
# modify data value
_C.DATA.MODIFY_SRC = True
_C.DATA.MODIFY_A_RANGE = 0.4
_C.DATA.MODIFY_B_RANGE = 1.0
_C.DATA.NUM_WORKERS = 8

_C.NET = CN()
_C.NET.NET_NAME = ""
_C.NET.USE_DP = True
_C.NET.USE_NORM = True
_C.NET.NUM_CLASS = 1
_C.NET.RESUME = ""

_C.TRAIN = CN()
_C.TRAIN.LR = 1e-3
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_E = 1

_C.VAL = CN()
_C.VAL.BATCH_SIZE = _C.TRAIN.BATCH_SIZE

cfg = _C


def _set_log_dir(c):
    experiment_name = '{}'.format('{}_'.format(c.EXP_NAME) if c.EXP_NAME != "" else '') + \
                      time.strftime(str("%y%m%d-%H%M%S"), time.localtime())
    save_dir = os.path.join('./logs', experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        return save_dir
    else:
        raise Exception("Log path {} already exists!".format(save_dir))

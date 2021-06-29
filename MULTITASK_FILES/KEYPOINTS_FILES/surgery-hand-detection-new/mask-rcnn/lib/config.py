from detectron2.config import CfgNode as CN


def add_aug_config(cfg: CN):
    _C = cfg
    _C.DATALOADER.AUG = ''

def add_freeze_config(cfg: CN):
	_C = cfg
	_C.MODEL.FREEZE = False

def add_keypoint_alphas(cfg: CN):
	_C = cfg
	_C.TEST.KEYPOINT_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
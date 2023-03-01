from .build import build_loader as _build_loader
from .build import build_loader_init_stitch
from .data_simmim_pt import build_loader_simmim
from .data_simmim_ft import build_loader_finetune


def build_loader(config, simmim=False, is_pretrain=False, init_stitch=False):
    if init_stitch:
        return build_loader_init_stitch(config)
    if not simmim:
        return _build_loader(config)
    if is_pretrain:
        return build_loader_simmim(config)
    else:
        return build_loader_finetune(config)

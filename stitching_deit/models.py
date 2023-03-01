import os

import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, VisionTransformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from params import args

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224'
]

class ViTAnchor(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

    def extract_block_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        outs = {}
        outs[-1] = x.detach()
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            outs[i] = x.detach()
        return outs

    def forward_patch_embed(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        return x

    def forward_norm_head(self, x):
        x = self.norm(x)
        x = self.forward_head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def deit_tiny_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViTAnchor(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_tiny_patch16_224-a1311bcf.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViTAnchor(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_small_patch16_224-cd65a155.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, pretrained_cfg=None,  **kwargs):
    model = ViTAnchor(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_base_patch16_224-b5f2ef4d.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model



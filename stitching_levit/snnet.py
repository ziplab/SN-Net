import os.path

import torch.nn as nn
import torch
from collections import defaultdict
from timm.models import create_model
from utils import get_stitch_configs, ps_inv
from timm.models.registry import register_model
import numpy as np


class StitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None):
        super().__init__()
        self.transform = nn.Linear(in_features, out_features)

    def init_stitch_weights_bias(self, weight, bias):
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)

    def forward(self, x):
        x = self.transform(x)
        return x


class SNNet(nn.Module):
    '''
    Stitchable Neural Networks
    '''

    def __init__(self, anchors):
        super(SNNet, self).__init__()

        self.anchors = nn.ModuleList(anchors)
        stage_depths = [mod.depth for mod in self.anchors]

        total_configs = []
        self.num_stitches = []
        self.stitch_layers = nn.ModuleList()
        self.stitching_map_id = {}

        for i in range(len(self.anchors)):
            total_configs.append({
                'comb_id': [i],
                'stitch_cfgs': [],
                'stitch_layers': []
            })

        for i in range(3):
            if i == 2:
                break
            cur_depths = [stage_depths[mod_id][i] for mod_id in range(len(self.anchors))]
            stage_configs, stage_stitches = get_stitch_configs(cur_depths, i)
            self.num_stitches.append(stage_stitches)
            total_configs += stage_configs
            stage_stitching_layers = nn.ModuleList()
            for j, (num_s, comb) in enumerate(stage_stitches):
                front, end = comb
                stage_stitching_layers.append(nn.ModuleList(
                    [StitchingLayer(self.anchors[front].embed_dim[i], self.anchors[end].embed_dim[i]) for _ in range(num_s)]))
                self.stitching_map_id[f'{i}-{front}-{end}'] = j
            self.stitch_layers.append(stage_stitching_layers)

        self.stitch_configs = {i: cfg for i, cfg in enumerate(total_configs)}
        self.num_configs = len(total_configs)
        self.stitch_config_id = 0

    def reset_stitch_id(self, stitch_config_id):
        self.stitch_config_id = stitch_config_id


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def initialize_stitching_weights(self, x):
        vit_features = []
        with torch.no_grad():
            for mod in self.anchors:
                vit_features.append(mod.extract_block_features(x))

        for stage_id in range(3):
            if stage_id == 2:
                break
            stage_stitches = self.num_stitches[stage_id]

            for j, (num_s, comb) in enumerate(stage_stitches):
                front, end = comb
                stitching_dicts = defaultdict(set)
                for id, config in self.stitch_configs.items():
                    if config['comb_id'] == comb and stage_id == config['stage_id']:
                        stitching_dicts[config['stitch_layers'][0]].add(config['stitch_cfgs'][0])

                for stitch_layer_id, stitch_positions in stitching_dicts.items():
                    weight_candidates = []
                    bias_candidates = []
                    for front_id, end_id in stitch_positions:
                        front_blk_feat = vit_features[front][stage_id][front_id]
                        end_blk_feat = vit_features[end][stage_id][end_id - 1]
                        w, b = ps_inv(front_blk_feat, end_blk_feat)
                        weight_candidates.append(w)
                        bias_candidates.append(b)
                    weights = torch.stack(weight_candidates).mean(dim=0)
                    bias = torch.stack(bias_candidates).mean(dim=0)

                    self.stitch_layers[stage_id][j][stitch_layer_id].init_stitch_weights_bias(weights, bias)
                    print(f'Initialized Stitching Model {front} to Model {end}, Stage {stage_id}, Layer {stitch_layer_id}')


    def forward(self, x):
        if self.training:
            stitch_cfg_id = np.random.randint(0, self.num_configs)
        else:
            stitch_cfg_id = self.stitch_config_id

        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']
        if len(comb_id) == 1:
            return self.anchors[comb_id[0]](x)

        stitch_cfgs = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']
        stitch_stage_id = self.stitch_configs[stitch_cfg_id]['stage_id']
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layers']

        cfg = stitch_cfgs[0]

        x = self.anchors[comb_id[0]].forward_until(x, stage_id=stitch_stage_id, blk_id=cfg[0])

        sl_id = stitch_layer_ids[0]
        key = f'{stitch_stage_id}-{comb_id[0]}-{comb_id[1]}'
        stitch_projection_id = self.stitching_map_id[key]
        x = self.stitch_layers[stitch_stage_id][stitch_projection_id][sl_id](x)

        x = self.anchors[comb_id[1]].forward_from(x, stage_id=stitch_stage_id, blk_id=cfg[1])

        return x

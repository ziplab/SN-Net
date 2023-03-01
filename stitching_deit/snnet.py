import torch
import torch.nn as nn
import numpy as np

from logger import logger
from collections import defaultdict
from itertools import combinations
from utils import get_stitch_configs, ps_inv

class StitchingLayer(nn.Module):
    def __init__(self, in_dim=None, out_dim=None):
        super().__init__()
        self.transform = nn.Linear(in_dim, out_dim)

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

    def __init__(self, anchors, kernel_size=2, stride=1, nearest_stitching=True):
        super(SNNet, self).__init__()

        self.anchors = nn.ModuleList(anchors)
        self.kernel_size = kernel_size
        self.stride = stride

        self.anchor_depths = [len(anchor.blocks) for anchor in self.anchors]
        blk_stitch_cfgs, num_stitches = get_stitch_configs(self.anchor_depths[0], kernel_size, stride, num_models=len(anchors), nearest_stitching=nearest_stitching)
        self.num_stitches = num_stitches

        candidate_combinations = list(combinations(list(range(len(anchors))), 2))
        if nearest_stitching:
            candidate_combinations.pop(candidate_combinations.index((0, 2)))
        self.candidate_combinations = candidate_combinations


        self.stitch_layers = nn.ModuleList()
        self.stitching_map_id = {}
        for i, cand in enumerate(candidate_combinations):
            front, end = cand
            self.stitch_layers.append(nn.ModuleList([StitchingLayer(self.anchors[front].embed_dim, self.anchors[end].embed_dim) for _ in range(num_stitches)]))
            self.stitching_map_id[f'{front}-{end}'] = i

        self.stitch_configs = {i: cfg for i, cfg in enumerate(blk_stitch_cfgs)}
        self.num_configs = len(blk_stitch_cfgs)
        self.stitch_config_id = 0

    def reset_stitch_id(self, stitch_config_id):
        self.stitch_config_id = stitch_config_id


    def initialize_stitching_weights(self, x):
        stitching_dicts = defaultdict(set)
        for id, config in self.stitch_configs.items():
            if len(config['comb_id']) == 1:
                continue

            # each stitching layer is shared among neighboring blocks, thus it handles different stitching path.
            stitching_dicts[config['stitch_layers'][0]].add(config['stitch_cfgs'][0])

        for i, combo in enumerate(self.candidate_combinations):
            front, end = combo

            # extract feature maps from the blocks of anchors
            with torch.no_grad():
                front_features = self.anchors[front].extract_block_features(x)
                end_features = self.anchors[end].extract_block_features(x)

            for stitch_layer_id, stitch_positions in stitching_dicts.items():
                weight_candidates = []
                bias_candidates = []
                for front_id, end_id in stitch_positions:
                    front_blk_feat = front_features[front_id]
                    end_blk_feat = end_features[end_id-1]

                    # solve the least square problem to get the weights and bias
                    w, b = ps_inv(front_blk_feat, end_blk_feat)
                    weight_candidates.append(w)
                    bias_candidates.append(b)

                # since each stitching layer is shared among different stitching paths, we average the weights and bias
                weights = torch.stack(weight_candidates).mean(dim=0)
                bias = torch.stack(bias_candidates).mean(dim=0)

                self.stitch_layers[i][stitch_layer_id].init_stitch_weights_bias(weights, bias)
                logger.info(f'Initialized Stitching Model {front} to Model {end}, Layer {stitch_layer_id}')


    def forward(self, x):
        if self.training:
            stitch_cfg_id = np.random.randint(0, self.num_configs) # random sampling during training
        else:
            stitch_cfg_id = self.stitch_config_id

        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']
        stitch_cfgs = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layers']

        if len(comb_id) == 1:
            # simply forward the anchor
            return self.anchors[comb_id[0]](x)

        x = self.anchors[comb_id[0]].forward_patch_embed(x)

        front_id = 0
        for i, cfg in enumerate(stitch_cfgs):
            end_id = cfg[0] + 1

            for blk in self.anchors[comb_id[i]].blocks[front_id:end_id]:
                x = blk(x)

            front_id = cfg[1]
            sl_id = stitch_layer_ids[i]
            key = str(comb_id[i]) + '-' + str(comb_id[i+1])
            stitch_projection_id = self.stitching_map_id[key]
            x = self.stitch_layers[stitch_projection_id][sl_id](x)

        for blk in self.anchors[comb_id[-1]].blocks[front_id:]:
            x = blk(x)

        x = self.anchors[comb_id[-1]].forward_head(x)

        return x


    def get_model_size(self, stitch_cfg_id):

        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']
        stitch_cfgs = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layers']

        if len(comb_id) == 1:
            return sum(p.numel() for p in self.anchors[comb_id[0]].parameters())

        total_params = 0
        total_params += sum(p.numel() for p in self.anchors[comb_id[0]].patch_embed.parameters())

        front_id = 0

        for i, cfg in enumerate(stitch_cfgs):
            end_id = cfg[0] + 1
            for blk in self.anchors[comb_id[i]].blocks[front_id:end_id]:
                total_params += sum(p.numel() for p in blk.parameters())

            front_id = cfg[1]
            sl_id = stitch_layer_ids[i]
            key = str(comb_id[i]) + '-' + str(comb_id[i+1])
            stitch_projection_id = self.stitching_map_id[key]
            total_params += sum(p.numel() for p in self.stitch_layers[stitch_projection_id][sl_id].parameters())

        for blk in self.anchors[comb_id[-1]].blocks[front_id:]:
            total_params += sum(p.numel() for p in blk.parameters())

        total_params += sum(p.numel() for p in self.anchors[comb_id[-1]].head.parameters())
        total_params += sum(p.numel() for p in self.anchors[comb_id[-1]].norm.parameters())
        return total_params

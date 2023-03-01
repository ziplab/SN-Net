import matplotlib.pyplot as plt
import json
import torch
from collections import defaultdict
import numpy as np
from matplotlib import rcParams
import seaborn as sns
sns.set_style("ticks")
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['legend.title_fontsize'] = 'small'
plt.rcParams['axes.facecolor'] = '#F4F4F4'


if __name__ == '__main__':
    my_colors = ['#8CA0CA', '#FAA17E', '#66C2A4', '#CCCCCC']
    total_configs = []
    stitching_map_id = {}
    data = {}
    with open('results/stitches_res.txt', 'r') as f:
        for line in f.readlines():
            epoch_stat = json.loads(line.strip())
            data[epoch_stat['cfg_id']] = epoch_stat

    plt.figure(dpi=400, figsize=(6, 3.5))

    acc1 = np.array([value['acc1'] for k, value in data.items()])
    flops = np.array([value['flops'] / 1e9 for k, value in data.items()])
    params = np.array([value['params'] / 1e6 for k, value in data.items()])
    anchor_acc1 = acc1[:3]
    anchor_flops = flops[:3]

    sorted_id = np.argsort(flops)

    acc1 = acc1[sorted_id].tolist()
    flops = flops[sorted_id].tolist()
    params = params[sorted_id].tolist()

    sns.scatterplot(flops, acc1, edgecolor='white', linewidth=0.1, facecolor=my_colors[0],
                    size=params, sizes=(30, 100))

    plt.scatter(anchor_flops, anchor_acc1, marker='*', s=200, edgecolor='black', facecolor='orange',
                linewidth=0.5)

    anchor_names = ['Swin-Ti', 'Swin-S', 'Swin-B']
    for i, name in enumerate(anchor_names):
        if name == 'Swin-B':
            plt.text(anchor_flops[i] - 0.5, anchor_acc1[i] - 0.5, name, fontsize=12,
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2})
        else:
            plt.text(anchor_flops[i] + 0.5, anchor_acc1[i] - 0.2, name, fontsize=12,
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2})

    plt.legend(title='#Params (M)', fontsize=11)
    plt.ylabel('Top-1(%)')
    plt.xlabel('FLOPs (G)')
    plt.grid(color='#ffffff', linestyle='--', linewidth=0.2)
    plt.tight_layout()
    plt.show()
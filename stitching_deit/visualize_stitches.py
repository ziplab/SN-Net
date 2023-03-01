import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from matplotlib import rcParams
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
plt.rcParams['axes.axisbelow'] = True
my_border_color = 'white'
plt.rcParams['legend.title_fontsize'] = 'small'
sns.set_style("ticks")
plt.rcParams['axes.facecolor'] = '#F4F4F4'

if __name__ == '__main__':
    my_colors = ['#8CA0CA', '#FAA17E', '#66C2A4', '#CCCCCC']
    data = {}
    with open('results/stitches_res.txt', 'r') as f:
        for line in f.readlines():
            epoch_stat = json.loads(line.strip())
            data[epoch_stat['cfg_id']] = epoch_stat

    acc1 = np.array([value['acc1'] for k, value in data.items()])
    flops = np.array([value['flops'] / 1e9 for k, value in data.items()])
    params = np.array([value['params'] / 1e6 for k, value in data.items()])

    plt.figure(dpi=400, figsize=(6, 3.5))
    sns.scatterplot(flops, acc1, size=params, sizes=(15, 90), edgecolor=my_border_color, facecolor=my_colors[0])
    anchor_acc1 = acc1[:3]
    anchor_flops = flops[:3]

    plt.scatter(anchor_flops, anchor_acc1, marker='*', s=200, edgecolor='black', facecolor='orange',
                linewidth=0.5)

    labels = ['DeiT-Ti', 'DeiT-S', 'DeiT-B']
    for i, label in enumerate(labels):
        if label == 'DeiT-B':
            plt.text(flops[i] - 2, acc1[i] - 1, label, fontsize=10,
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2})

        else:
            plt.text(flops[i] + 0.5, acc1[i] - 0.1, label, fontsize=10,
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2})

    plt.ylabel('Top-1 (%)')
    plt.legend(title='#Params (M)', fontsize=11)
    plt.grid(color='white', linewidth=0.5)
    plt.xlabel('FLOPs (G)')
    plt.tight_layout()
    plt.show()

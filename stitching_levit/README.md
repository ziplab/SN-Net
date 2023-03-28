# Stitchable Neural Networks 🪡

This directory contains the training and evaluation scripts for stitching LeViT-192/256.


## Requirements

### Prepare Python Environment

* PyTorch 1.10.1+
* CUDA 11.1+
* fvcore 0.1.5

### Prepare Pretrained Weights

Download the pretrained weights of LeViT-192/256 from [here](https://github.com/facebookresearch/LeViT) and put them in the `pretrained/` directory.
The following commands can be helpful.

```bash
cd pretrained/
wget https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth
wget https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth
```

## Training

To stitch LeViT-192/256 on ImageNet with 8 GPUs, run the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model stitch_levits \
    --data-path [path/to/imagenet] \
    --output_dir ./exp_levit_192_256 \
    --epochs 100 \
    --batch-size 128 \
    --lr 5e-5 \
    --warmup-lr 1e-7 \
    --min-lr 1e-6
```

## Evaluation

You can download our trained weights from [here](https://github.com/ziplab/SN-Net/releases/download/v1.2/levit_192_256_release.pth). Next,

```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model stitch_levits \
    --data-path [path/to/imagenet] \
    --output_dir ./eval_levit_192_256 \
    --batch-size 128 \
    --resume [path/to/checkpoint.pth] --eval
```

After evaluation, you can find a `stitches_res.txt` under the `output_dir` directory which contains the results for all stitches. Our evaluation results can be found at `results/stitches_res.txt`.


## Acknowledgement

This code is based on [LeViT](https://github.com/facebookresearch/LeViT). We thank the authors for their released code.
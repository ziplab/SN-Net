# Stitchable Neural Networks 🪡

This directory contains the training and evaluation scripts for stitching ResNet-18/50 and ResNet-18/Swin-Ti.



## Requirements

* PyTorch 1.10.1+
* CUDA 11.1+
* fvcore 0.1.5



## Training

Inside this directory, you can use the following commands to replicate our results on ImageNet.

To stitch a ResNet-18 with ResNet-50 with 8 GPUs on ImageNet.
```bash
./distributed_train.sh 8 \
[path/to/imagenet] \
-b 128 \
--stitch_config configs/resnet18_resnet50.json \
--sched cosine \
--epochs 30 \
--lr 0.05 \
--amp --remode pixel \
--reprob 0.6 \
--aa rand-m9-mstd0.5-inc1 \
--resplit --split-bn -j 10 --dist-bn reduce
```


To stitch a ResNet-18 with Swin-Ti with 8 GPUs on ImageNet.
```bash
./distributed_train.sh 8 \
[path/to/imagenet] \
-b 128 \
--stitch_config configs/resnet18_swin_ti.json \
--sched cosine \
--epochs 30 \
--lr 0.05 \
--amp --remode pixel \
--reprob 0.6 \
--aa rand-m9-mstd0.5-inc1 \
--resplit --split-bn -j 10 --dist-bn reduce
```



## Evaluation

Download our trained wegihts

| Stitched Models      | Checkpoint                                                   |
| -------------------- | ------------------------------------------------------------ |
| ResNet-18, ResNet-50 | [link](https://github.com/ziplab/SN-Net/releases/download/v1.1/resnet_18_50.pth) |
| ResNet-18, Swin-Ti   | [link](https://github.com/ziplab/SN-Net/releases/download/v1.1/resnet18_swin_ti.pth) |



To evaluate, 


```bash
./distributed_train.sh 1 \
[path/to/imagenet] \
-b 128 \
--stitch_config configs/resnet18_swin_ti.json \
--sched cosine \
--epochs 30 \
--lr 0.05 \
--amp --remode pixel \
--reprob 0.6 \
--aa rand-m9-mstd0.5-inc1 \
--resplit --split-bn -j 10 --dist-bn reduce \
--resume [path/to/weights] --eval
```

After evaluation, you can find a `stitches_res.txt` under the `outputs/[name]/default/` directory which contains the results for all stitches.  You can find our evaluation results at `results/`

**Please note that** our experiments involving the stitching of ResNet-18/50 and ResNet-18/Swin-Ti did not extensively explore optimal experiment settings. Hyperparameters were barely tuned during these experiments. Therefore, if you encounter poor performance with some stitched models, selecting the networks on the Pareto frontier will again give you a smooth FLOPs-accuracy curve.



### Acknowledgement

This implementation adopts code from [timm](https://github.com/huggingface/pytorch-image-models). We thank the authors for their released code.

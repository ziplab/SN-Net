python -m torch.distributed.launch --nproc_per_node 1 \
    --master_port 12345  main.py \
    --cfg configs/snnet/stitch_swin_ti_s_b.yaml \
    --eval

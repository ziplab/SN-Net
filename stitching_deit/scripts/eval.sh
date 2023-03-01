python -m torch.distributed.launch --nproc_per_node=1 \
      --master_port 12345 \
      --use_env main.py \
      --config config/deit_stitching.json --dist-eval --eval
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate, evaluate_snnet, initialize_model_stitching_layer
from losses import DistillationLoss, StitchDistillationLoss
from samplers import RASampler
import models
from snnet import SNNet
import utils
from params import args
from logger import logger
import warnings
warnings.filterwarnings("ignore")

@torch.no_grad()
def throughput(data_loader, model):
    model.eval()
    # update latency level
    if hasattr(model, 'module'):
        model.module.reset_latency_level(args.latency_level)
    else:
        model.reset_latency_level(args.latency_level)

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def main():
    utils.init_distributed_mode(args)
    if utils.get_rank() != 0:
        logger.disabled = True
    logger.info(str(args))

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    logger.info(f"Creating model: {args.model}")


    if args.model == 'snnet':
        anchors = []
        for i, anchor_name in enumerate(args.anchors):
            logger.info(f"Creating model: {anchor_name}")
            anchor = create_model(
                anchor_name,
                pretrained=True,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
            )
            anchors.append(anchor)
        model = SNNet(anchors=anchors, kernel_size=args.stitch_kernel_size, stride=args.stitch_stride, nearest_stitching=args.nearest_stitching)

        if not args.fulltune:
            for name, param in model.named_parameters():
                if 'stitch_layers' in name or 'cls' in name:
                    # cls token has to be included for backpropagation
                    param.requires_grad = True
                    logger.info(f'learnable param: {name}')
                else:
                    param.requires_grad = False

        model.to(device)

        if args.ls_init:
            temp_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.stitch_init_bs,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )

            # solve by least square
            initialize_model_stitching_layer(model, mixup_fn, temp_loader_train, device)
            logger.info('Stitching Layer Initialized')
            del temp_loader_train
    else:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
        model.to(device)

    logger.info(str(model))


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: ' + str(n_parameters))

    args.lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    optimizer = create_optimizer(args, model_without_ddp)

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        logger.info(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=True,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom StitchDistillationLoss
    if args.model == 'snnet':
        criterion = StitchDistillationLoss(criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau)
    else:
        criterion = DistillationLoss(criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau)


    output_dir = Path(args.output_dir)
    if args.resume:
        logger.info(f'resume from {args.resume}')
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        if args.model == 'snnet':
            evaluate_snnet(data_loader_val, model, device, os.path.join(args.output_dir, 'stitches_res.txt'))
        else:
            evaluate(data_loader_val, model, device)
        return


    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if args.model == 'snnet':
        evaluate_snnet(data_loader_val, model, device, os.path.join(args.output_dir, 'stitches_res.txt'))
    else:
        evaluate(data_loader_val, model, device)

if __name__ == '__main__':
    main()

"""
Train and eval functions used in main.py
"""
import math
import os.path
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from logger import logger
import time
from utils import save_on_master_eval_res
import json

from fvcore.nn import FlopCountAnalysis

def initialize_model_stitching_layer(model, mixup_fn, data_loader,  device):
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        model.initialize_stitching_weights(samples)

        break


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_snnet(data_loader, model, device, output_dir):
    # check last config:
    last_cfg_id = -1
    if os.path.exists(output_dir):
        with open(output_dir, 'r') as f:
            for line in f.readlines():
                epoch_stat = json.loads(line.strip())
                last_cfg_id = epoch_stat['cfg_id']

    criterion = torch.nn.CrossEntropyLoss()

    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if hasattr(model, 'module'):
        num_configs = model.module.num_configs
    else:
        num_configs = model.num_configs

    for cfg_id in range(last_cfg_id+1, num_configs):
        if hasattr(model, 'module'):
            model.module.reset_stitch_id(cfg_id)
        else:
            model.reset_stitch_id(cfg_id)

        logger.info(f'------------- Evaluting stitch config {cfg_id}/{num_configs} -------------')

        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).cuda())
        flops = flops.total()
        converted = flops / 1e9
        converted = round(converted, 2)
        logger.info(f'FLOPs = {converted}')

        metric_logger = utils.MetricLogger(delimiter="  ")

        for images, target in metric_logger.log_every(data_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info('cfg_id = ' + str(
            cfg_id) + '  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                    .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        log_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats['cfg_id'] = cfg_id
        log_stats['flops'] = flops
        log_stats['params'] = model.module.get_model_size(cfg_id)
        save_on_master_eval_res(log_stats, output_dir)


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

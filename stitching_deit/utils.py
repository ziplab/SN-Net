"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import json
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
from logger import logger
import numpy as np
from itertools import combinations

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def save_on_master_eval_res(log_stats, output_dir):
    if is_main_process():
        with open(output_dir, 'a') as f:
            f.write(json.dumps(log_stats) + "\n")


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        logger.info('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    logger.info('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url))
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def paired_stitching(depth=12, kernel_size=2, stride=1):
    blk_id = list(range(depth))
    i = 0
    stitch_cfgs = []
    stitch_id = -1
    stitching_layers_mappings = []

    while i < depth:
        ids = blk_id[i:i + kernel_size]
        has_new_stitches = False
        for j in ids:
            for k in ids:
                if (j, k) not in stitch_cfgs:
                    has_new_stitches = True
                    stitch_cfgs.append((j, k))
                    stitching_layers_mappings.append(stitch_id + 1)

        if has_new_stitches:
            stitch_id += 1

        i += stride

    num_stitches = stitch_id + 1
    return stitch_cfgs, stitching_layers_mappings, num_stitches


def unpaired_stitching(front_depth=12, end_depth=24):
    num_stitches = front_depth

    block_ids = torch.tensor(list(range(front_depth)))
    block_ids = block_ids[None, None, :].float()
    end_mapping_ids = torch.nn.functional.interpolate(block_ids, end_depth)
    end_mapping_ids = end_mapping_ids.squeeze().long().tolist()
    front_mapping_ids = block_ids.squeeze().long().tolist()

    stitch_cfgs = []
    for idx in front_mapping_ids:
        for i, e_idx in enumerate(end_mapping_ids):
            if idx != e_idx or idx >= i:
                continue
            else:
                stitch_cfgs.append((idx, i))
    return stitch_cfgs, end_mapping_ids, num_stitches



def get_stitch_configs(depth=12, kernel_size=2, stride=1, num_models=3, nearest_stitching=True):
    '''This function assumes the two model have the same depth, for demonstrating DeiT stitching.

    Args:
        depth: number of blocks in the model
        kernel_size: size of the stitching sliding window
        stride: stride of the stitching sliding window
        num_models: number of models to be stitched
        nearest_stitching: whether to use nearest stitching
    '''

    stitch_cfgs, layers_mappings, num_stitches = paired_stitching(depth, kernel_size, stride)

    model_combinations = []
    candidates = list(range(num_models))
    for i in range(1, num_models + 1):
        model_combinations += list(combinations(candidates, i))

    if nearest_stitching:
        # remove tiny-base
        model_combinations.pop(model_combinations.index((0, 2)))

        # remove three model settings
        model_combinations.pop(model_combinations.index((0, 1, 2)))

    total_configs = []

    for comb in model_combinations:
        if len(comb) == 1:
            total_configs.append({
                'comb_id': comb,
                'stitch_cfgs': [],
                'stitch_layers': []
            })
            continue

        for cfg, layer_mapping_id in zip(stitch_cfgs, layers_mappings):
            if len(comb) == 2:
                total_configs.append({
                    'comb_id': comb,
                    'stitch_cfgs': [cfg],
                    'stitch_layers': [layer_mapping_id]
                })
            else:
                last_out_id = cfg[1]
                for second_cfg, second_layer_mapping_id in zip(stitch_cfgs, layers_mappings):
                    middle_id, end_id = second_cfg
                    if middle_id < last_out_id:
                        continue
                    total_configs.append({
                        'comb_id': comb,
                        'stitch_cfgs': [cfg, second_cfg],
                        'stitch_layers': [layer_mapping_id, second_layer_mapping_id]
                    })

    return total_configs, num_stitches


def rearrange_activations(activations):
    n_channels = activations.shape[-1]
    activations = activations.reshape(-1, n_channels)
    return activations


def ps_inv(x1, x2):
    '''Least-squares solver given feature maps from two anchors.
    
    Source: https://github.com/renyi-ai/drfrankenstein/blob/main/src/comparators/compare_functions/ps_inv.py
    '''
    x1 = rearrange_activations(x1)
    x2 = rearrange_activations(x2)

    if not x1.shape[0] == x2.shape[0]:
        raise ValueError('Spatial size of compared neurons must match when ' \
                         'calculating psuedo inverse matrix.')

    # Get transformation matrix shape
    shape = list(x1.shape)
    shape[-1] += 1

    # Calculate pseudo inverse
    x1_ones = torch.ones(shape)
    x1_ones[:, :-1] = x1
    A_ones = torch.matmul(torch.linalg.pinv(x1_ones), x2.to(x1_ones.device)).T

    # Get weights and bias
    w = A_ones[..., :-1]
    b = A_ones[..., -1]

    return w, b
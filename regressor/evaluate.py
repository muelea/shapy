import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
#try:
#    os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]
#except:
#    pass

import sys
import os.path as osp

import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist

import resource

from threadpoolctl import threadpool_limits
from loguru import logger

from human_shape.utils.checkpointer import Checkpointer
from human_shape.data import build_all_data_loaders
from human_shape.models.build import build_model
from human_shape.config import parse_args
from human_shape.evaluation import build as build_evaluator

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

DEFAULT_FORMAT = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |'
                  ' <level>{level: <8}</level> |'
                  ' <cyan>{name}</cyan>:<cyan>{function}</cyan>:'
                  '<cyan>{line}</cyan> - <level>{message}</level>')

DIST_FORMAT = ('<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> |'
               ' <level>{{level: <8}}</level> |'
               ' <red><bold>Rank {rank: <3} </bold></red> |'
               ' <cyan>{{name}}</cyan>:<cyan>{{function}}</cyan>:'
               '<cyan>{{line}}</cyan> - <level>{{message}}</level>')


def main():
    exp_cfg = parse_args()
    eval_on_val_split = exp_cfg.get('run_final_evaluation_on_validation_set')

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    num_gpus = exp_cfg.get('num_gpus', 1)
    distributed = num_gpus > 1
    local_rank = exp_cfg.get('local_rank', 0)

    logger_format = (DIST_FORMAT.format(rank=local_rank)
                     if distributed else DEFAULT_FORMAT)
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''),
               level=exp_cfg.logger_level.upper(),
               format=logger_format,
               colorize=True)

    logger.info(f'Distributed {distributed}: {local_rank} / {num_gpus}')
    output_folder = osp.expandvars(exp_cfg.output_folder)

    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()

        backend = exp_cfg.get('backend', 'nccl')
        logger.info(f'Using backend: {backend}')
        torch.distributed.init_process_group(
            backend,
            init_method='env://',
            world_size=num_gpus,
            rank=local_rank,
        )
        #########################################################

    # Set up a seed manually so that all device share the same seed when
    # initializing the parameters
    if distributed:
        torch.manual_seed(0)
    #  os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.local_rank}'
    logger.info(f'Rank = {local_rank}: device = {torch.cuda.device_count()}')

    if distributed and local_rank == 0:
        log_file = osp.join(output_folder, 'info.log')
        logger.add(log_file, level=exp_cfg.logger_level.upper(), colorize=True)

    model_dict = build_model(exp_cfg)
    model = model_dict['network']
    if local_rank == 0:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

    use_sync_bn = exp_cfg.get('network', {}).get('use_sync_bn', False)
    # Convert the Batch Normalization modules to their synchronized equivalent
    if distributed and use_sync_bn:
        logger.info(f'Use sync Batch Normalization: {use_sync_bn}')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        dist.barrier()

    # Copy the model to the correct device
    model = model.to(device=device)
    if distributed:
        torch.manual_seed(int(time.time()) % 1024)

    checkpoint_folder = osp.join(output_folder, exp_cfg.checkpoint_folder)
    os.makedirs(checkpoint_folder, exist_ok=True)

    checkpointer = Checkpointer(
        model, save_dir=checkpoint_folder, pretrained=exp_cfg.pretrained,
        distributed=distributed, rank=local_rank)

    code_folder = osp.join(output_folder, exp_cfg.code_folder)
    os.makedirs(code_folder, exist_ok=True)

    # Set the model to evaluation mode
    dataset_split = 'val' if eval_on_val_split else 'test'
    logger.info(f'Loading {dataset_split} split for evaluation.')
    data_loaders = build_all_data_loaders(exp_cfg, split=dataset_split)

    arguments = {'iteration': 0}
    extra_checkpoint_data = checkpointer.load_checkpoint()
    for key in arguments:
        if key in extra_checkpoint_data:
            arguments[key] = extra_checkpoint_data[key]
    model.eval()

    evaluator = build_evaluator(
        exp_cfg, rank=local_rank, distributed=distributed)

    with evaluator:
        evaluator.run(model, data_loaders, exp_cfg, device,
                        step=arguments['iteration'])


if __name__ == '__main__':

    with threadpool_limits(limits=1):
        main()

from typing import NewType, List, Tuple, Union
import sys
import os.path as osp

from loguru import logger

import time
import numpy as np
import functools
import torch
import torch.utils.data as dutils
from omegaconf import DictConfig

from copy import deepcopy
from . import datasets
from .samplers import EqualSampler, ShapeSampler

from .structures import (AbstractStructure, StructureList, to_image_list,
                         ImageList, ImageListPacked)
from .transforms import build_transforms
from human_shape.utils import Tensor, TensorList

DEFAULT_NUM_WORKERS = {
    'train': 0,
    'val': 0,
    'test': 0
}


def wif(id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


def build_data_sampler(
    dataset,
    is_train: bool = True,
    shuffle: bool = True,
    is_distributed: bool = False,
):
    if is_train:
        sampler = dutils.RandomSampler(dataset)
    else:
        sampler = dutils.SequentialSampler(dataset)
    return sampler


def build_batch_sampler(
    datasets: List[dutils.Dataset],
    batch_size: int = 32,
    shuffle: bool = True,
    ratio_2d: float = 0.5
):
    if part_key == 'pose':
        sampler = dutils.RandomSampler(dataset)
        batch_sampler = EqualSampler(
            datasets, batch_size=batch_size, shuffle=shuffle, ratio_2d=ratio_2d)
    else:
        sampler = dutils.SequentialSampler(dataset)
    return sampler


def build_shape_dataset(
    name: str,
    dataset_cfg: DictConfig,
    transforms,
    **kwargs
) -> None:

    if name == 'model_agencies':
        obj = datasets.ModelAgency
    elif name == 'weight_height':
        obj = datasets.WeightHeightDataset
    elif name == 'ssp3d':
        obj = datasets.SSP3D
    elif name == 'hbw':
        obj = datasets.HumanBodyInTheWild
    else:
        raise ValueError(f'Unknown dataset: {name}')

    args = dict(**dataset_cfg[name])
    args.update(kwargs)

    vertex_flip_correspondences = osp.expandvars(dataset_cfg.get(
        'vertex_flip_correspondences', ''))
    dset_obj = obj(transforms=transforms,
                   vertex_flip_correspondences=vertex_flip_correspondences,
                   **args)

    logger.info('Created dataset: {}', dset_obj.name())
    return dset_obj


def build_pose_dataset(name, dataset_cfg, transforms, **kwargs):
    if name == 'ehf':
        obj = datasets.EHF
    elif name == 'totalcapture':
        obj = datasets.TotalCapture
    elif name == 'curated_fits':
        obj = datasets.CuratedFittings
    elif name == 'eft':
        obj = datasets.EFTDataset
    elif name == 'coco':
        obj = datasets.COCO
    elif name == 'agora':
        obj = datasets.Agora
    elif name == 'threedpw':
        obj = datasets.ThreeDPW
    elif name == 'human36m':
        obj = datasets.Human36M
    elif name == 'human36mx':
        obj = datasets.Human36MX
    elif name == 'spin':
        obj = datasets.SPIN
    elif name == 'spinx':
        obj = datasets.SPINX
    elif name == 'lsp_test':
        obj = datasets.LSPTest
    elif name == 'openpose':
        obj = datasets.OpenPose
    elif name == 'tracks':
        obj = datasets.OpenPoseTracks
    else:
        raise ValueError(f'Unknown dataset: {name}')

    args = dict(**dataset_cfg[name])
    args.update(kwargs)

    vertex_flip_correspondences = osp.expandvars(dataset_cfg.get(
        'vertex_flip_correspondences', ''))
    dset_obj = obj(transforms=transforms,
                   vertex_flip_correspondences=vertex_flip_correspondences,
                   **args)

    logger.info('Created dataset: {}', dset_obj.name())
    return dset_obj


class MemoryPinning(object):
    def __init__(
        self,
        full_img_list: Union[ImageList, List[Tensor]],
        images: Tensor,
        targets: StructureList
    ):
        super(MemoryPinning, self).__init__()
        self.img_list = full_img_list
        self.images = images
        self.targets = targets

    def pin_memory(
            self
    ) -> Tuple[Union[ImageList, ImageListPacked, TensorList],
               Tensor, StructureList]:
        if self.img_list is not None:
            if isinstance(self.img_list, (ImageList, ImageListPacked)):
                self.img_list.pin_memory()
            elif isinstance(self.img_list, (list, tuple)):
                self.img_list = [x.pin_memory() for x in self.img_list]
        return (
            self.img_list,
            self.images.pin_memory(),
            self.targets,
        )


def collate_batch(
    batch,
    use_shared_memory=False,
    return_full_imgs=False,
    pin_memory=False
):
    if return_full_imgs:
        images, cropped_images, targets, _ = zip(*batch)
    else:
        _, cropped_images, targets, _ = zip(*batch)

    out_targets = []
    for t in targets:
        if t is None:
            continue
        if type(t) == list:
            out_targets += t
        else:
            out_targets.append(t)
    out_cropped_images = []
    for img in cropped_images:
        if img is None:
            continue
        if torch.is_tensor(img):
            if len(img.shape) < 4:
                img.unsqueeze_(dim=0)
            out_cropped_images.append(img)
        elif isinstance(img, (list, tuple)):
            for d in img:
                d.unsqueeze_(dim=0)
                out_cropped_images.append(d)

    if len(out_cropped_images) < 1:
        return None, None, None

    full_img_list = None
    if return_full_imgs:
        full_img_list = images
    #  out = None
    #  if use_shared_memory:
        #  numel = sum([x.numel() for x in out_cropped_images if x is not None])
        #  storage = out_cropped_images[0].storage()._new_shared(numel)
        #  out = out_cropped_images[0].new(storage)

    out_cropped_images = torch.cat(out_cropped_images)
    #  del targets, batch
    if pin_memory:
        return MemoryPinning(
            full_img_list,
            #  torch.cat(out_cropped_images, 0, out=out),
            out_cropped_images,
            out_targets
        )
    else:
        #  return full_img_list, torch.cat(
        #  out_cropped_images, 0, out=out), out_targets
        return full_img_list, out_cropped_images, out_targets


def build_sampler(
    datasets: List[dutils.Dataset],
    part_key: str = 'pose',
    batch_size: int = 32,
    shuffle: bool = True,
    ratio_2d: float = 0.5,
    importance_key: str = 'weight',
    balance_genders: bool = True,
    use_equal_sampling: bool = True,
    is_train: bool = True,
):
    if part_key == 'pose':
        # Equal sampling should only be used during training and only if there
        # are multiple datasets
        if use_equal_sampling and is_train:
            batch_sampler = EqualSampler(
                datasets, batch_size=batch_size, shuffle=shuffle,
                ratio_2d=ratio_2d)
            out_dsets_lst = [
                dutils.ConcatDataset(datasets) if len(datasets) > 1 else
                datasets[0]]
        else:
            return None, datasets
    else:
        if is_train:
            batch_sampler = ShapeSampler(
                datasets,
                batch_size=batch_size,
                shuffle=shuffle,
                importance_key=importance_key,
                balance_genders=balance_genders,
            )
            out_dsets_lst = [
                dutils.ConcatDataset(datasets) if len(datasets) > 1 else
                datasets[0]]
        else:
            batch_sampler, out_dsets_lst = None, datasets

    return batch_sampler, out_dsets_lst


def build_data_loader(dataset, batch_size=32, num_workers=0,
                      is_train=True, sampler=None, collate_fn=None,
                      shuffle=True, is_distributed=False,
                      batch_sampler=None, pin_memory=False):

    if batch_sampler is None:
        sampler = build_data_sampler(
            dataset, is_train=is_train,
            shuffle=shuffle, is_distributed=is_distributed)

    if batch_sampler is None:
        assert sampler is not None, (
            'Batch sampler and sampler can\'t be "None" at the same time')
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True and is_train,
            pin_memory=pin_memory,
            worker_init_fn=wif,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            pin_memory=pin_memory,
            worker_init_fn=wif,
        )
    return data_loader


def build_all_data_loaders(
    exp_cfg,
    split='train',
    start_iter=0,
    rank=0,
    distributed=False,
    return_full_imgs=False,
    shuffle=True,
    enable_augment=True,
    **kwargs
):
    is_train = 'train' in split
    dataset_cfg = exp_cfg.get('datasets', {})

    if not is_train and distributed and rank > 0:
        return {
            'pose': {},
            'shape': {},
        }

    #  return_full_imgs = 'expose' in net_type or return_full_imgs
    logger.info(f'Return full resolution images: {return_full_imgs}')

    # Hard-coded for now
    shuffle = is_train and shuffle
    is_distributed = False

    batch_size = dataset_cfg.batch_size
    logger.info(f'Total batch size: {batch_size}')
    pose_shape_ratio = dataset_cfg.pose_shape_ratio
    logger.info(f'Pose shape ration: {pose_shape_ratio}')
    
    batch_sizes = {'pose': int(round(batch_size * pose_shape_ratio)),
                   'shape': int(round(batch_size * (1 - pose_shape_ratio))),
                   }

    data_loaders = {'pose': [], 'shape': []}
    build_dset_foo = {'pose': build_pose_dataset,
                      'shape': build_shape_dataset}

    for part_key in ['pose', 'shape']:
        dsets_cfg = dataset_cfg.get(part_key, {})
        dset_names = dsets_cfg.get('splits', {})[split]
        if len(dset_names) < 1:
            continue

        transfs_cfg = dsets_cfg.get('transforms', {})
        transforms = build_transforms(transfs_cfg, is_train=is_train,
                                      enable_augment=enable_augment,
                                      return_full_imgs=return_full_imgs)
        logger.info(
            'Body transformations: \n{}',
            '\n'.join(list(map(str, transforms))))

        datasets = []
        for dataset_name in dset_names:
            dset = build_dset_foo[part_key](
                dataset_name, dsets_cfg, transforms=transforms,
                is_train=is_train, split=split, **kwargs)
            datasets.append(dset)

        if len(datasets) < 1:
            continue

        batch_size = batch_sizes[part_key]

        num_workers = dsets_cfg.get(
            'num_workers', DEFAULT_NUM_WORKERS).get(split, 0)
        logger.info(f'{split.upper()} Body num workers: {num_workers}')

        collate_fn = functools.partial(
            collate_batch, use_shared_memory=num_workers > 0,
            return_full_imgs=return_full_imgs)

        sampling_cfg = dsets_cfg.get('sampler', {})
        logger.info(sampling_cfg)
        batch_sampler, datasets = build_sampler(
            datasets,
            part_key=part_key,
            batch_size=batch_size,
            is_train=is_train,
            **sampling_cfg)

        for dataset in datasets:
            data_loaders[part_key].append(
                build_data_loader(dataset, batch_size=batch_size,
                                  num_workers=num_workers,
                                  is_train=is_train,
                                  batch_sampler=batch_sampler,
                                  collate_fn=collate_fn,
                                  shuffle=shuffle,
                                  is_distributed=is_distributed))
    return data_loaders

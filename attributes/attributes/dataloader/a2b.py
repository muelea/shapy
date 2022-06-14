import torch
import sys
import json
import torch.functional as nn
import pickle
import joblib
from loguru import logger
import numpy as np
import os.path as osp
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from collections import defaultdict
import attributes.utils.constants as constants

MAX_WEIGHT = 500
MIN_WEIGHT = 20

MIN_HEIGHT = 0.546
MAX_HEIGHT = 2.72


class MIXEDAB_DATASET(Dataset):
    def __init__(
        self,
        datasets,
        ds_gender,
        model_gender,
        model_type,
        set='train',
        sample_raw: bool = False,
        num_samples: int = 10,
        normalize: bool = False,
        sample_agg_foo: str = 'mean',
        bodytalk_meas_preprocess: bool = False,
    ):
        super(MIXEDAB_DATASET, self).__init__()

        self.dataset_list = datasets
        self.datasets = [AB_Dataset(
            x, ds_gender, model_gender, model_type,
            set,
            normalize=normalize,
            sample_agg_foo=sample_agg_foo,
            sample_raw=sample_raw,
            num_samples=num_samples,
            bodytalk_meas_preprocess=bodytalk_meas_preprocess,
        )
            for x in datasets]
        self.length = max([len(x) for x in self.datasets])
        self.lengths = [len(x) for x in self.datasets]
        self.cumsum = np.cumsum(np.array([0] + self.lengths))
        self.total_length = sum([len(ds) for ds in self.datasets])

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dsidx = [(i, x) for i, x in enumerate(self.cumsum) if idx >= x][-1]
        data = self.datasets[dsidx[0]][idx - dsidx[1]]
        return data


class AB_Dataset(Dataset):
    """
    Dataset class for VAE.
    """

    def __init__(
        self,
        dataset,
        ds_gender,
        model_gender,
        model_type,
        set='train',
        sample_raw: bool = False,
        num_samples: int = 10,
        sample_agg_foo: str = 'mean',
        normalize: bool = False,
        bodytalk_meas_preprocess: bool = False,
    ):
        super(AB_Dataset, self).__init__()

        self.dataset = dataset
        self.ds_gender = ds_gender
        self.model_gender = model_gender
        self.model_type = model_type
        self.bodytalk_meas_preprocess = bodytalk_meas_preprocess

        db_path = osp.join(f'../data/dbs/{dataset}_{self.ds_gender}_{set}.pt')

        self.db = joblib.load(db_path)

        self.has_gt_height = 'heights' in self.db.keys()
        self.has_gt_weight = 'weights' in self.db.keys()
        self.has_bg_height = 'guess_heights' in self.db.keys()
        self.has_bg_weight = 'guess_weights' in self.db.keys()

        self.normalize = normalize

        self.sample_agg_foo = sample_agg_foo
        assert self.sample_agg_foo in ['mean', 'median'], (
            'We support only mean/median for rating aggregation')
        self.num_samples = num_samples
        self.sample_raw = sample_raw
        self.is_train = 'train' in set
        # logger.info(
        #     f'{set}: sample from raw={sample_raw},'
        #     f' num_samples={self.num_samples}')
        self.set = set

    def __repr__(self) -> str:
        msg = [
            f'Split: {self.set}',
            f'Dataset: {self.dataset}',
            f'Dataset gender: {self.ds_gender}',
            f'Model type: {self.model_type}',
            f'Has GT height: {self.has_gt_height}',
            f'Has GT weight: {self.has_gt_weight}',
            f'Has BG height: {self.has_bg_height}',
            f'Has BG weight: {self.has_bg_weight}',
            f'Normalize: {self.normalize}',
        ]
        if self.is_train and self.sample_raw:
            msg.append(f'Sample raw ratings: {self.sample_raw}')
            msg.append(f'Number of samples: {self.num_samples}')
            msg.append(f'Sample aggregation function: {self.sample_agg_foo}')
        return '\n'.join(msg)

    def process_mmt(self, value):
        if value != '':
            return True, torch.tensor(float(value) / 100).float()
        else:
            return False, torch.tensor(0).float()

    def __len__(self):
        return len(self.db['ids'])

    def __getitem__(self, idx):
        has_gt_betas = True if self.dataset in ['caesar'] else False
        has_gt_mmts = True if self.dataset in ['models'] else False

        betas_key = f'betas_{self.model_type}_{self.model_gender}'

        rating_label = self.db['rating_label']

        if self.sample_raw and self.is_train:
            raw_ratings = self.db['ratings_raw'][idx]
            valid_ratings_mask = raw_ratings.sum(axis=-1) > 0
            valid_raw_ratings = raw_ratings[valid_ratings_mask]

            num_valid = valid_ratings_mask.sum()
            if num_valid <= self.num_samples:
                rating_choice = valid_raw_ratings
            else:
                sample_idxs = np.random.choice(
                    num_valid, self.num_samples, replace=False)
                rating_choice = valid_raw_ratings[sample_idxs]

            if self.sample_agg_foo == 'mean':
                rating = rating_choice.mean(axis=0)
            elif self.sample_agg_foo == 'median':
                rating = np.median(rating_choice, axis=0)
        else:
            rating = self.db['ratings'][idx]

        height_gt = torch.tensor(self.db['heights'][idx]).float()
        weight_gt = torch.tensor(self.db['weights'][idx]).float()
        if self.bodytalk_meas_preprocess:
            height_gt *= 100
            weight_gt = weight_gt.pow(1.0 / 3)

        # Normalizes the ratings to [0, 1]
        if self.normalize:
            # before = rating
            rating /= 5
            # logger.info(f'{before} -> {rating}')

            # before = height_gt
            height_gt = (height_gt - MIN_HEIGHT) / (MAX_HEIGHT - MIN_HEIGHT)
            # logger.info(f'{before} -> {height_gt}')

            # before = weight_gt
            weight_gt = (weight_gt - MIN_WEIGHT) / (MAX_WEIGHT - MIN_WEIGHT)
            # logger.info(f'{before} -> {weight_gt}')

        #mask = self.db['ratings_raw'][idx].sum(1) > 0
        # print(max(abs(self.db['ratings'][idx] - \
        # self.db['ratings_raw'][idx][mask].mean(0))))
        item = {
            'id': self.db['ids'][idx],
            'rating': torch.from_numpy(rating).float(),
            'rating_raw': torch.from_numpy(self.db['ratings_raw'][idx]).float(),
            'has_gt_betas': has_gt_betas,
            'betas': torch.from_numpy(self.db[betas_key][idx]).float(),
            'chest': torch.tensor(0).float(),
            'hips': torch.tensor(0).float(),
            'height_gt': height_gt,
            'weight_gt': weight_gt,
            'waist': torch.tensor(0).float(),
            'height_bg': torch.tensor(0).float(),
            'weight_bg': torch.tensor(0).float(),
            'rating_label': rating_label,
        }

        # if available add best guess for height and weight
        if self.has_bg_weight:
            item['weight_bg'] = torch.tensor(
                self.db['guess_weights'][idx]).float()
        if self.has_bg_height:
            item['height_bg'] = torch.tensor(
                self.db['guess_heights'][idx]).float()

        # check and load other measurements
        if self.dataset in ['models']:
            # get ground truth measurements for model data
            has_bust, item['chest'] = self.process_mmt(self.db['bust'][idx])
            has_hips, item['hips'] = self.process_mmt(self.db['hips'][idx])
            has_waist, item['waist'] = self.process_mmt(self.db['waist'][idx])
        elif self.dataset in ['caesar']:
            item['chest'] = torch.tensor(self.db['chest'][idx]).float()
            item['waist'] = torch.tensor(self.db['waist'][idx]).float()
            item['hips'] = torch.tensor(self.db['hips'][idx]).float()

        return item


def collate_fn(batch):
    output = defaultdict(lambda: [])
    for b_item in batch:
        for k, v in b_item.items():
            output[k].append(v)

    for k in output:
        if torch.is_tensor(output[k][0]):
            output[k] = torch.stack(output[k])

    return dict(output)


class AB_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        dataset=[],
        bodytalk_meas_preprocess: bool = False,
        ds_gender='female',
        model_gender='female',
        model_type='smplx',
        sample_raw: bool = False,
        num_samples: int = 10,
        normalize: bool = False,
        sample_agg_foo: str = 'mean',
    ):
        super().__init__()

        self.batch_size = batch_size

        self.train_sets = MIXEDAB_DATASET(
            datasets=dataset,
            set='train',
            ds_gender=ds_gender,
            model_gender=model_gender,
            model_type=model_type,
            sample_raw=sample_raw,
            num_samples=num_samples,
            normalize=normalize,
            sample_agg_foo=sample_agg_foo,
            bodytalk_meas_preprocess=bodytalk_meas_preprocess,
        )

        self.val_sets = MIXEDAB_DATASET(
            datasets=dataset,
            set='val',
            ds_gender=ds_gender,
            model_gender=model_gender,
            model_type=model_type,
            normalize=normalize,
            sample_agg_foo=sample_agg_foo,
            bodytalk_meas_preprocess=bodytalk_meas_preprocess,
        )

        self.test_sets = AB_Dataset(
            dataset=dataset[0],
            set='test',
            ds_gender=ds_gender,
            model_gender=model_gender,
            model_type=model_type,
            normalize=normalize,
            sample_agg_foo=sample_agg_foo,
            bodytalk_meas_preprocess=bodytalk_meas_preprocess,
        )

    def train_dataloader(self):
        ab_train = DataLoader(self.train_sets, batch_size=self.batch_size,
                              shuffle=True, drop_last=True, collate_fn=collate_fn)
        return ab_train

    def val_dataloader(self):
        ab_val = DataLoader(self.val_sets, batch_size=self.batch_size,
                            drop_last=False, collate_fn=collate_fn)
        return ab_val

    def test_dataloader(self):
        ab_test = DataLoader(self.test_sets, batch_size=self.batch_size,
                             drop_last=False, collate_fn=collate_fn)
        return ab_test

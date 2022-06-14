from typing import List, Union, Optional
from itertools import cycle, islice
import sys

import numpy as np
import torch
import torch.utils.data as dutils
from loguru import logger
from collections import defaultdict


class EqualSampler(dutils.Sampler):

    ''' Form a batch by sampling an equal amount of images from N datasets
    '''

    def __init__(self, datasets, batch_size=1, ratio_2d=0.5, shuffle=False):
        super(EqualSampler, self).__init__(datasets)
        self.num_datasets = len(datasets)
        self.ratio_2d = ratio_2d

        self.shuffle = shuffle
        self.dset_sizes = {}
        self.elements_per_index = {}
        self.only_2d = {}
        self.offsets = {}
        start = 0
        for dset in datasets:
            self.dset_sizes[dset.name()] = len(dset)
            self.offsets[dset.name()] = start
            self.only_2d[dset.name()] = dset.only_2d()
            self.elements_per_index[
                dset.name()] = dset.get_elements_per_index()

            start += len(dset)

        if ratio_2d < 1.0 and sum(self.only_2d.values()) == len(self.only_2d):
            raise ValueError(
                f'Invalid 2D ratio value: {ratio_2d} with only 2D data')

        self.length = sum(map(lambda x: len(x), datasets))

        self.batch_size = batch_size
        self._can_reuse_batches = False
        logger.info(self)

    def __repr__(self):
        msg = 'EqualSampler(batch_size={}, shuffle={}, ratio_2d={}\n'.format(
            self.batch_size, self.shuffle, self.ratio_2d)
        for dset_name in self.dset_sizes:
            msg += '\t{}: {}, only 2D is {}\n'.format(
                dset_name, self.dset_sizes[dset_name],
                self.only_2d[dset_name])

        return msg + ')'

    def _prepare_batches(self):
        batch_idxs = []

        dset_idxs = {}
        for dset_name, dset_size in self.dset_sizes.items():
            if self.shuffle:
                dset_idxs[dset_name] = cycle(
                    iter(torch.randperm(dset_size).tolist()))
            else:
                dset_idxs[dset_name] = cycle(range(dset_size))

        num_batches = int(round(self.length / self.batch_size))
        for bidx in range(num_batches):
            curr_idxs = []
            num_samples = 0
            num_2d_only = 0
            max_num_2d = int(self.batch_size * self.ratio_2d)
            idxs_add = defaultdict(lambda: 0)
            while num_samples < self.batch_size:
                for dset_name in dset_idxs:
                    # If we already have self.ratio_2d * batch_size items with
                    # 2D annotations then ignore this dataset for now
                    if num_2d_only >= max_num_2d and self.only_2d[dset_name]:
                        continue
                    try:
                        curr_idxs.append(
                            next(dset_idxs[dset_name]) +
                            self.offsets[dset_name])
                        num_samples += self.elements_per_index[dset_name]
                        # If the dataset has only 2D annotations increase the
                        # count
                        num_2d_only += (self.elements_per_index[dset_name] *
                                        self.only_2d[dset_name])
                        idxs_add[dset_name] += (
                            self.elements_per_index[dset_name])
                    finally:
                        pass
                    if num_samples >= self.batch_size:
                        break

            curr_idxs = np.array(curr_idxs)
            if self.shuffle:
                np.random.shuffle(curr_idxs)
            batch_idxs.append(curr_idxs)

        if self.shuffle:
            np.random.shuffle(batch_idxs)

        return batch_idxs

    def __len__(self):
        if not hasattr(self, '_batch_idxs'):
            self._batch_idxs = self._prepare_batches()
            self._can_reuse_bathces = True
        return len(self._batch_idxs)

    def __iter__(self):
        if self._can_reuse_batches:
            batch_idxs = self._batch_idxs
            self._can_reuse_batches = False
        else:
            batch_idxs = self._prepare_batches()

        self._batch_idxs = batch_idxs
        return iter(batch_idxs)


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def weights_to_probabilities(
    importance_weights,
    bins: Union[str, int] = 'auto',
    prob_type: str = 'inverse',
):
    ''' Converts relative weights to probabilities
    '''
    # Create a histogram of weights/bmi values
    hist, bin_edges = np.histogram(importance_weights, bins=bins)

    # Find the bin in which every element belongs
    indices = np.searchsorted(bin_edges, importance_weights, 'left')

    if prob_type == 'inverse':
        p = 1 / hist[indices - 1]
        p /= p.sum()
    elif prob_type == 'proportional':
        p = hist[indices - 1]
        p /= p.sum()
    else:
        raise ValueError(
            f'Unknown probability type: {prob_type}. Expected one of'
            ' ["proportional", "inverse"]')

    return p


class ShapeSampler(dutils.Sampler):

    ''' Form a batch with importance sampling from the weights
    '''

    def __init__(
        self,
        datasets: List[dutils.Dataset],
        batch_size: int = 1,
        importance_key: str = 'weight',
        shuffle: bool = False,
        balance_genders: bool = True,
    ) -> None:
        ''' Importance sample a body measurement dataset

            Parameters
            ----------
            datasets: List[dutils.Dataset]
                The list of datasets used for sampling
            batch_size: int = 1
                The size of the batch that will be sampled
            key: str = 'bmi'
                The key that is used to choose how to form the importance
                weights. Valid valus are 'bmi' and 'weight'.
            shuffle: bool = False
                Whether to shuffle the indices
            balance_genders: bool = True
                Equally balance all genders in the batch
        '''
        super(ShapeSampler, self).__init__(datasets)
        self.num_datasets = len(datasets)

        assert importance_key in ['bmi', 'weight'], (
            'We only support importance sampling from bmi/weight values, not:'
            f' {importance_key}'
        )

        self.balance_genders = balance_genders
        self.importance_key = importance_key

        self.shuffle = shuffle
        self.dset_sizes = {}
        self.elements_per_index = {}
        self.only_2d = {}
        self.offsets = {}
        self.importance_weights = {}
        self.importance_probs = {}
        self.genders = {}
        self.indices = {}
        start = 0

        for dset in datasets:
            self.dset_sizes[dset.name()] = len(dset)
            self.offsets[dset.name()] = start
            self.elements_per_index[
                dset.name()] = dset.get_elements_per_index()
            self.only_2d[dset.name()] = dset.only_2d()
            self.importance_weights[dset.name()] = getattr(
                dset, importance_key)
            self.importance_probs[dset.name()] = weights_to_probabilities(
                getattr(dset, importance_key))
            self.genders[dset.name()] = dset.gender

            start += len(dset)
            # logger.info(f'{dset}: {len(dset)}')

        self.length = sum(map(lambda x: len(x), datasets))

        flat_genders = []
        for value in self.genders.values():
            flat_genders += value.tolist()
        self.flat_genders = np.array(flat_genders)
        self.gender_labels = np.unique(self.flat_genders)

        self.batch_size = batch_size
        self._can_reuse_batches = False

    def __repr__(self):
        msg = ['ShapeSampler(',
               f'batch_size={self.batch_size}',
               f'shuffle={self.shuffle}',
               f'balance_genders={self.balance_genders}',
               ]
        msg
        for dset_name in self.dset_sizes:
            msg.append(
                f'{dset_name}: {self.dset_sizes[dset_name]},' +
                f' only 2D is {self.only_2d[dset_name]}'
            )

        msg.append(')')
        return '\n'.join(msg)

    def _prepare_batches(self):
        batch_idxs = []

        # Store a dict that contains the indices of every gender for every
        # dataset
        gender_idxs = {}
        gender_importance = {}
        for gender in self.gender_labels:
            gender_idxs[gender] = {}
            gender_importance[gender] = {}
            for dset_name, gender_array in self.genders.items():
                gender_idxs[gender][dset_name] = np.where(
                    gender_array == gender)[0]
                importance_probs = self.importance_probs[
                    dset_name][gender_idxs[gender][dset_name]]
                importance_probs /= importance_probs.sum()
                gender_importance[gender][dset_name] = importance_probs
                #  logger.info(
                #  f'{dset_name}, {gender}: {gender_importance[dset_name][gender].shape}')

        #  for dset_name, dset_size in self.dset_sizes.items():
            #  if self.shuffle:
                #  dset_idxs[dset_name] = cycle(
                #  iter(torch.randperm(dset_size).tolist()))
            #  else:
                #  dset_idxs[dset_name] = cycle(range(dset_size))

        num_batches = int(round(self.length / self.batch_size))
        num_genders = len(self.gender_labels)
        #  gender_per_batch = self.batch_size // num_genders

        # Sampling once from each dataset should give us `els` elements
        els = 0
        for dset_name, els_per_index in self.elements_per_index.items():
            els += els_per_index

        # We will sample `num_sampling` times from each dataset
        num_sampling = int(round(self.batch_size / els))

        # Build each batch
        for bidx in range(num_batches):

            curr_idxs = []
            # Go over all genders
            for gender, data in gender_importance.items():
                # Now iterate over the datasets
                for dset_name, importance_weights in data.items():
                    # From each dataset sample `num_sampling // 2` indices, so
                    # that we
                    ii = np.random.choice(
                        gender_idxs[gender][dset_name],
                        #  self.elements_per_index[dset_name],
                        num_sampling // num_genders,
                        replace=False,
                        p=importance_weights,
                    ) + self.offsets[dset_name]

                    curr_idxs.append(ii)

            # Merge the indices into a single list
            curr_idxs = list(roundrobin(*curr_idxs))

            # Convert to a numpy array
            curr_idxs = np.array(curr_idxs)
            # Shuffle if necessary
            if self.shuffle:
                np.random.shuffle(curr_idxs)
            batch_idxs.append(curr_idxs)

        #  for x in batch_idxs:
            #  logger.info(len(x))
        #  logger.info(num_batches)
        #  logger.info(len(batch_idxs))
        #  sys.exit(0)
        # Shuffle the order of the sampled batches
        if self.shuffle:
            np.random.shuffle(batch_idxs)
        return batch_idxs

    def __len__(self):
        return int(round(self.length / self.batch_size))

    def __iter__(self):
        #  if self._can_reuse_batches:
        #  batch_idxs = self._batch_idxs
        #  self._can_reuse_batches = False
        #  else:
        batch_idxs = self._prepare_batches()
        self._batch_idxs = batch_idxs

        #  self._batch_idxs = batch_idxs
        return iter(batch_idxs)

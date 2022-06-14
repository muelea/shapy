import sys
import math

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit


def build_discriminator(disc_cfg, num_joints=21):
    disc_type = disc_cfg.type
    if disc_type == 'hmr':
        return HMRDiscriminator(num_joints=num_joints)
    else:
        raise ValueError('Unknown discriminator type')


class PartDiscriminators(nn.Module):
    def __init__(self, num_input, num_parts, bias=True,
                 use_spectral_norm=True):
        super(PartDiscriminators, self).__init__()
        self.num_parts = num_parts

        #  weights = torch.zeros([num_parts, num_input])
        weights = []
        for pidx in range(num_parts):
            weight = torch.zeros([1, num_input])
            nninit.kaiming_uniform_(weight, a=math.sqrt(5))
            weights.append(weight)
        weights = torch.cat(weights, dim=0)
        self.register_parameter('weights',
                                nn.Parameter(weights, requires_grad=True))
        if bias:
            biases = torch.zeros(num_parts)
            self.register_parameter('bias',
                                    nn.Parameter(biases, requires_grad=True))

    def forward(self, part_input):
        return (torch.einsum('bcj,jc->bj', [part_input, self.weights]) +
                self.bias.unsqueeze(dim=0))


class HMRDiscriminator(nn.Module):
    def __init__(self, num_joints=21, num_channels=32, nzfeat=1024,
                 use_spectral_norm=True):
        super(HMRDiscriminator, self).__init__()
        self.num_joints = num_joints
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(9, num_channels, 1)
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 1)
        if use_spectral_norm:
            self.conv2 = nn.utils.spectral_norm(self.conv2)

        self.part_disc = nn.Linear(num_channels, 1)
        if use_spectral_norm:
            self.part_disc = nn.utils.spectral_norm(self.part_disc)

        linear1 = nn.Linear(num_channels * num_joints, nzfeat)
        if use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
        linear2 = nn.Linear(nzfeat, nzfeat)
        if use_spectral_norm:
            linear2 = nn.utils.spectral_norm(linear2)
        linear3 = nn.Linear(nzfeat, 1)
        if use_spectral_norm:
            linear3 = nn.utils.spectral_norm(linear3)
        self.all_poses = nn.Sequential(
            linear1,
            nn.ReLU(),
            linear2,
            nn.ReLU(),
            linear3,
        )

    def forward(self, pose):
        batch_size, num_joints = pose.shape[:2]
        pose = pose.view(batch_size, -1, 1, 9).permute(0, 3, 1, 2)

        pose = F.relu(self.conv2(F.relu(self.conv1(pose)))).reshape(
            batch_size, self.num_channels, self.num_joints)

        part_probs = self.part_disc(pose.permute(0, 2, 1)).reshape(
            batch_size, num_joints).reshape(batch_size, num_joints)
        full_prob = self.all_poses(pose.view(batch_size, -1))
        return torch.cat([part_probs, full_prob], dim=1)

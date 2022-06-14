from typing import Optional, List, Dict, Callable
import pickle
import os
import os.path as osp
import sys
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from loguru import logger

class VertexEdgeLoss(nn.Module):
    def __init__(self, norm_type='l2',
                 gt_edges=None,
                 gt_edge_path='',
                 est_edges=None,
                 est_edge_path='',
                 robustifier=None,
                 edge_thresh=0.0, epsilon=1e-8,
                 reduction='sum',
                 **kwargs):
        super(VertexEdgeLoss, self).__init__()

        assert norm_type in ['l1', 'l2'], 'Norm type must be [l1, l2]'
        self.norm_type = norm_type
        self.epsilon = epsilon
        self.reduction = reduction
        assert self.reduction in ['sum', 'mean']
        logger.info(f'Building edge loss with'
                    f' norm_type={norm_type},'
                    f' reduction={reduction},'
                    )

        gt_edge_path = osp.expandvars(gt_edge_path)
        est_edge_path = osp.expandvars(est_edge_path)
        assert osp.exists(gt_edge_path) or gt_edges is not None, (
            'gt_edges must not be None or gt_edge_path must exist'
        )
        assert osp.exists(est_edge_path) or est_edges is not None, (
            'est_edges must not be None or est_edge_path must exist'
        )
        if osp.exists(gt_edge_path) and gt_edges is None:
            gt_edges = np.load(gt_edge_path)
        if osp.exists(est_edge_path) and est_edges is None:
            est_edges = np.load(est_edge_path)

        self.register_buffer(
            'gt_connections', torch.tensor(gt_edges, dtype=torch.long))
        self.register_buffer(
            'est_connections', torch.tensor(est_edges, dtype=torch.long))

    def extra_repr(self):
        msg = [
            f'Norm type: {self.norm_type}',
        ]
        if self.has_connections:
            msg.append(
                f'GT Connections shape: {self.gt_connections.shape}'
            )
            msg.append(
                f'Est Connections shape: {self.est_connections.shape}'
            )
        return '\n'.join(msg)

    def compute_edges(self, points, connections):
        edge_points = torch.index_select(
            points, 1, connections.view(-1)).reshape(points.shape[0], -1, 2, 3)
        return edge_points[:, :, 1] - edge_points[:, :, 0]

    def forward(self, gt_vertices, est_vertices, weights=None):
        gt_edges = self.compute_edges(
            gt_vertices, connections=self.gt_connections)
        est_edges = self.compute_edges(
            est_vertices, connections=self.est_connections)

        raw_edge_diff = (gt_edges - est_edges)

        batch_size = gt_vertices.shape[0]
        if self.norm_type == 'l2':
            edge_diff = raw_edge_diff.pow(2)
        elif self.norm_type == 'l1':
            edge_diff = raw_edge_diff.abs()
        else:
            raise NotImplementedError(
                f'Loss type not implemented: {self.loss_type}')
        #  if self.reduction == 'sum':
        return edge_diff.reshape(batch_size, -1).sum(dim=-1).mean(dim=0)

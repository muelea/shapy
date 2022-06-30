from logging import error
from typing import List, Optional, Dict, Union, Tuple
import sys
import torch
import smplx
import os
import os.path as osp
import math

from collections import defaultdict

import pickle
import trimesh
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cmap

from tqdm import tqdm
from loguru import logger
from torch.optim import Adam
from body_measurements import BodyMeasurements
import PIL.Image as pil_img
from omegaconf import DictConfig

import attributes.utils.constants as constants
from attributes.utils.renderer import Renderer
from attributes.utils.sparse import sparse_batch_mm
from attributes.attributes_betas.models import build_network
from attributes.attributes_betas.prob import build_distr_regressor
from attributes.utils.losses import VertexEdgeLoss
from attributes.utils.config import get_features_from_config
from attributes.utils.sampling import sample_in_sphere
from attributes.utils.typing import Tensor, Array

MAX_SUMMARY_IMGS = 16


def build_loss(loss_metric):
    if loss_metric == 'l2':
        return nn.MSELoss()
    elif loss_metric == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError(
            f'Unknown metric for vertex loss: {loss_metric}, expected [l1, l2]!')


def v2v_func(
    x: Union[Tensor, Array], y: Union[Array, Tensor]
) -> Union[Array, Tensor]:
    if torch.is_tensor(x):
        return (x - y).pow(2).sum(dim=-1).sqrt()
    else:
        return np.sqrt(np.power(x - y, 2).sum(-1))


def get_translation(gt_vertices, pred_vertices, align=True):
    if align:
        t = gt_vertices.mean(1) - pred_vertices.mean(1)
    else:
        B = len(gt_vertices)
        t = torch.zeros(
            [B, 3], dtype=gt_vertices.dtype, device=gt_vertices.device)
    return t


def v2v_error(gt_vertices, pred_vertices, align=True):
    t = get_translation(gt_vertices, pred_vertices, align=align)
    loss = torch.norm(pred_vertices + t.reshape(-1, 1, 3) -
                      gt_vertices, dim=2).mean(1)

    return loss.item()


def v2v_hd_error(
        gt_vertices: Tensor,
        pred_vertices: Tensor,
        hd_operator: Tensor,
        align: bool = True
):

    hd_gt_verts = sparse_batch_mm(hd_operator, gt_vertices.double())
    hd_pred_verts = sparse_batch_mm(
        hd_operator, pred_vertices.double())

    t = get_translation(hd_gt_verts, hd_pred_verts, align=align)
    loss = torch.norm(hd_pred_verts + t.reshape(-1, 1, 3) -
                      hd_gt_verts, dim=2).mean(1)

    return loss.item()


class A2B(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig
    ):
        super().__init__()

        self.save_hyperparameters()

        self.cfg = cfg

        self.output_dir = cfg.get('output_dir')

        self.summary_steps = cfg.get('summary_steps', 100)
        self.render_summary_steps = cfg.get('render_summary_steps', 10000)

        self.batch_size = cfg.get('batch_size', 32)
        self.betas_size = cfg.get('num_shape_comps', 10)
        self.model_gender = cfg.get('model_gender', 'female')

        self.ds_gender = cfg.get('ds_gender', 'female')

        self.bodytalk_meas_preprocess = cfg.get(
            'bodytalk_meas_preprocess', False)
       
        logger.info('BodyTalk measurement pre-processing: '
                    f'{self.bodytalk_meas_preprocess}')

        self.eval_val = cfg.get('eval_val', True)
        logger.info(f'Eval val: {self.eval_val}')
        self.eval_test = cfg.get('eval_test', False)
        logger.info(f'Eval test: {self.eval_test}')

        self.selected_attr, self.selected_attr_idx, self.selected_mmts = (
            get_features_from_config(cfg))
        self.input_feature_size = len(
            self.selected_attr) + len(self.selected_mmts)
        self.attr_feat_names = []
        if len(self.selected_attr_idx) > 0:
            self.attr_feat_names = np.array(self.selected_attr)[
                self.selected_attr_idx]
        self.feature_names = np.concatenate(
            [self.attr_feat_names, np.array(self.selected_mmts)])

        self.align = cfg.get('align', True)
        self.cmap = cfg.get('cmap', 'viridis')
        self.max_val = cfg.get('max_val', 20)

        model_type = cfg.get('model_type', 'smplx')
        self.model_type = model_type

        network_cfg = cfg.get('network', {})
        self.a2b = build_network(
            network_cfg,
            self.input_feature_size,
            self.betas_size
        )
        logger.info(self.a2b)

        # Self-report bias added for noise computation
        self.use_srb_noise = cfg.get('use_srb_noise', False)
        self.srb = constants.SELF_REPORT_BIAS

        reg_cfg = cfg.get('regression', {})
        self.use_loo = reg_cfg.get('use_loo_cross_val', False)
        self.whw2s_model = reg_cfg.get(
            'use_whw2s_setting', True)
        logger.info(f'Use BodyTalk data pre-processing: {self.whw2s_model}')
        self.add_noise = reg_cfg.get('add_noise', False)
        logger.info(f'Add noise to input: {self.add_noise}')

        self.v2v_weight = cfg.get('v2v_weight', 0.0)
        logger.info(f'V2V weight: {self.v2v_weight}')
        self.v2v_hd_weight = cfg.get('v2v_hd_weight', 0.0)
        logger.info(f'V2V-HD weight: {self.v2v_hd_weight}')
        self.vertex_loss = build_loss(cfg.get('vertex_loss_metric', 'l2'))

        self.betas_weight = cfg.get('betas_weight', 0.0)
        logger.info(f'Betas weight: {self.betas_weight}')
        self.betas_loss = build_loss(cfg.get('betas_loss_metric', 'l2'))

        self.val_metric = cfg.get('val_metric', 'v2v_hd')
        
        #hd_operator = torch.load(constants.HD.get(model_type))
        #self.register_buffer('hd_operator', hd_operator)

        with open(constants.HD.get(model_type), 'rb') as f:
            hd_operator_pkl = pickle.load(f).tocoo()
            indices = np.vstack(
            (hd_operator_pkl.row, hd_operator_pkl.col)
            )
            values = hd_operator_pkl.data
            shape = hd_operator_pkl.shape
            hd_operator = torch.sparse_coo_tensor(
                indices, values, shape
            )
            self.register_buffer('hd_operator', hd_operator) 

        self.height_weight = cfg.get('height_weight', 0.0)
        self.height_loss = nn.MSELoss()
        self.chest_weight = cfg.get('chest_weight', 0.0)
        self.chest_loss = nn.MSELoss()
        self.waist_weight = cfg.get('waist_weight', 0.0)
        self.waist_loss = nn.MSELoss()
        self.hips_weight = cfg.get('hips_weight', 0.0)
        self.hips_loss = nn.MSELoss()

        self.compute_train_meas = (
            self.height_weight > 0 or self.chest_weight > 0 or self.waist_weight > 0 or
            self.hips_weight > 0
        )

        self.edge_weight = cfg.get('edge_weight', 0.0)
        if self.edge_weight > 0:
            self.edge_loss = VertexEdgeLoss(
                gt_edge_path=constants.MODEL_EDGES.get(model_type),
                est_edge_path=constants.MODEL_EDGES.get(model_type),
            )

        # Body Measurements Module for evaluation
        meas_cfg = {
            'meas_definition_path': constants.MEAS.get('definition'),
            'meas_vertices_path': constants.MEAS.get(model_type),
        }

        self.meas_module = BodyMeasurements(meas_cfg).to(self.device)

        self.compute_train_verts = (
            self.compute_train_meas or self.v2v_weight > 0 or self.v2v_hd_weight > 0 or
            self.edge_weight > 0
        )

        # Body Module for evaluation and v2v loss
        self.model = smplx.build_layer(
            gender=self.model_gender,
            model_type=self.model_type,
            num_betas=self.betas_size,
            flat_hand_mean=True,
            model_path=constants.MODEL_PATH,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.use_attr_noise = cfg.get('use_attr_noise', False)
        logger.info(f'Add attribute noise: {self.use_attr_noise}')
        self.attr_noise_range = cfg.get('attr_noise_range', 0.2)
        logger.info(f'Sample attribute noise from U({-self.attr_noise_range},'
                    f' {self.attr_noise_range})')

        self.use_betas_noise = cfg.get('use_betas_noise', False)
        logger.info(f'Add noise to GT betas: {self.use_betas_noise}')
        self.noise_rmse = cfg.get('noise_rmse', 3)
        logger.info(
            f'RMSE Upper bound for perturbed meshes in mm: {self.noise_rmse}')
        shapedirs_fro = self.model.shapedirs.pow(2).sum().item()
        constant = (
            (self.noise_rmse / 1000) ** 2 / shapedirs_fro
            * self.model.get_num_verts())
        self.noise_constant = np.sqrt(constant)

        self.renderer = Renderer(
            is_registration=False
        )

        self.eval_output = {
            'mass': [],
            'height': [],
            'chest': [],
            'waist': [],
            'hips': [],
            'RE': [],
            'RE_HD': []
        }

        self.colormap = mpl_cmap.get_cmap(self.cmap)
        self.norm = mpl_colors.Normalize(vmin=0, vmax=self.max_val)
        self.scalar_mappable = mpl_cmap.ScalarMappable(
            norm=self.norm, cmap=self.colormap,
        )

    def forward(self, x):
        return self.a2b(x)

    def to_eval_mode(self, cmap, max_val, align, dataset, prefix, render_result):
        self.cmap = cmap
        self.max_val = max_val
        self.align = align
        self.dataset = dataset
        self.prefix = prefix
        self.render_result = render_result

    def configure_optimizers(self):
        logger.info(
            f'Building Adam optimizer with: lr={self.cfg.lr:.6f},'
            f' weight decay={self.cfg.weight_decay:.6f}')
        optimizer = Adam(
            self.a2b.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        output = {
            'optimizer': optimizer,
        }

        use_scheduler = self.cfg.get('use_scheduler', False)
        if use_scheduler:
            sched_cfg = self.cfg.get('scheduler', {})
            sched_type = sched_cfg.get('type', 'exponential')
            if sched_type == 'exponential':
                exp_cfg = sched_cfg.get('exponential', {})
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, **exp_cfg)
            else:
                raise ValueError(f'Unknown scheduler type: {sched_type}')

            output['scheduler'] = {
                # REQUIRED: The scheduler instance
                'scheduler': lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                'interval': sched_cfg.get('interval', 'epoch'),
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                'frequency': sched_cfg.get('frequency', 1),
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                'monitor': 'Loss/RE',
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                'strict': True,
            }
        return output

    def get_input_output_dim(self) -> Tuple[int, int]:
        return self.num_features, self.num_shape_comps

    def fit_raw(self, data):
       
        self.rating_label = data.db['labels']

        train_data, val_data, test_data = self.get_tvt_data_raw(data)
        self.fit_tvt(self.a2b, train_data, val_data, test_data)

    def fit(self, data):
       
        self.rating_label = data.db['labels']

        if self.use_loo:
            input, output = self.get_loo_data(data)
            self.fit_loo(self.model, input, output)
        else:
            train_data, val_data, test_data = self.get_tvt_data(data)
            self.fit_tvt(self.a2b, train_data, val_data, test_data)

    def to_whw2s(self, data, noise):

        logger.info('Converting input features to BodyTalk setup.')

        data = data.clone()

        height_idx = np.where(self.feature_names == 'height_gt')[0]
        weight_idx = np.where(self.feature_names == 'weight_gt')[0]

        # change height unit to cm to fit the noise unit.
        data[:, height_idx] = data[:, height_idx] * 100

        if noise is None:
            data[:, weight_idx] = data[:, weight_idx].sqrt()
        else:
            # for weight we must first add the noise (noise unit is kg).
            # The noise unit of height is cm, so it can be added later.
            weight_data = data[:, weight_idx] + noise[:, weight_idx]
            data[:, weight_idx] = weight_data.sqrt()
            noise[:, weight_idx] = 0.0
            data = data + noise

        return data

    def fit_loo(self, model, input, output):

        from sklearn.model_selection import LeaveOneOut

        # leave one out cross validation
        loo = LeaveOneOut()

        # unpack input
        input, noise = input

        # in each step, fit model to all but one data samples,
        # then evaluate the leaft-out sample.
        gt_betas_eval = []
        pred_betas_eval = []

        for train_idx, test_idx in loo.split(input):
            # fit model
            train_input = self.to_whw2s(input[train_idx, :], None) \
                if self.whw2s_model else input[train_idx, :]
            fitted_model = model.fit(train_input, output[train_idx, :])

            # make prediction
            test_input = self.to_whw2s(input[test_idx, :], noise[test_idx, :]) \
                if self.whw2s_model else input[test_idx, :]
            pred_betas = fitted_model.predict(test_input)
            gt_betas = output[test_idx, :]

            gt_betas_eval += [gt_betas]
            pred_betas_eval += [pred_betas]

        logger.info('Reporting results from leave-one-out cross validation.')
        val_output = self.validate(
            np.stack(gt_betas_eval).squeeze(),
            np.stack(pred_betas_eval).squeeze()
        )

        self.print_result(val_output)

    def fit_tvt(self, model, train_data, val_data, test_data):

        (train_input, train_noise), train_output = train_data
        (val_input, val_noise), val_output = val_data
        (test_input, test_noise), test_output = test_data

        train_input = self.to_whw2s(train_input, None) \
            if self.whw2s_model else train_input
        logger.info('Fit model ...')
        fitted_model = model.fit(train_input, train_output)

        # predict and eval betas val / test
        if self.eval_val:
            logger.info('Reporting results on validation set.')
            val_input = self.to_whw2s(val_input, None) \
                if self.whw2s_model else val_input
            prediction = fitted_model.predict(val_input)
            val_output = self.run_batch_validation(val_output, prediction)
            self.print_result(val_output)

        if self.eval_test:
            logger.info('Reporting results on test set.')
            test_input = self.to_whw2s(test_input, None) \
                if self.whw2s_model else test_input
            prediction = fitted_model.predict(test_input)
            test_output = self.run_batch_validation(test_output, prediction)
            self.print_result(test_output)

    def get_tvt_data(self, data):
        beta_key = f'betas_{self.model_type}_{self.model_gender}'
        train_data = (
            self.create_input_feature_vec(data.db['train']),
            data.db['train'][beta_key][:, :self.betas_size]
        )
        val_data = (
            self.create_input_feature_vec(data.db['val']),
            data.db['val'][beta_key][:, :self.betas_size]
        )
        test_data = (
            self.create_input_feature_vec(data.db['test']),
            data.db['test'][beta_key][:, :self.betas_size]
        )
        return train_data, val_data, test_data

    def run_batch_validation(self, gt_betas, pred_betas):

        val_output = defaultdict(lambda: [])

        if not torch.is_tensor(gt_betas):
            gt_betas = torch.from_numpy(gt_betas).to(
                dtype=torch.float32, device=self.device)
        if not torch.is_tensor(pred_betas):
            pred_betas = torch.from_numpy(pred_betas).to(
                dtype=torch.float32, device=self.device)

        gt_verts = self.model.forward_shape(betas=gt_betas).vertices
        pred_verts = self.model.forward_shape(
            betas=pred_betas).vertices

        for gt_vert, pred_vert in zip(gt_verts, pred_verts):
            gt_vert = gt_vert.unsqueeze(0)
            pred_vert = pred_vert.unsqueeze(0)
            # Reconstruction errors
            val_output['RE'] += [v2v_error(gt_vert, pred_vert, self.align)]
            val_output['RE_HD'] += [
                v2v_hd_error(gt_vert, pred_vert, self.hd_operator, self.align)]

        for key in ['mass', 'height', 'chest', 'waist', 'hips']:
            val_output[key] = []

        for pred_beta, gt_beta in zip(pred_betas, gt_betas):
            meas_error_dict = self.mmts_mae(pred_beta.unsqueeze(0), gt_beta.unsqueeze(0))
            for key, val in meas_error_dict.items():
                if torch.is_tensor(val):
                    val = val.detach().abs().cpu().numpy()
                val_output[key] += val.tolist()

        for key in val_output:
            if isinstance(val_output[key], (list, tuple)):
                val_output[key] = np.array(val_output[key])

        return val_output

    def get_loo_data(self, data):
        input = self.create_input_feature_vec(data.db)
        beta_key = f'betas_{self.model_type}_{self.model_gender}'
        output = data.db[beta_key][:, :self.betas_size]
        return input, output

    def get_tvt_data_raw(self, data):
        num_ratings = data.db['train']['rating_raw'].shape[1]

        beta_key = f'betas_{self.model_type}_{self.model_gender}'
        train_data = (
            self.create_input_feature_vec_raw(data.db['train']),
            torch.repeat_interleave(
                torch.tensor(data.db['train'][beta_key][:, :self.betas_size], dtype=torch.float32, device=self.device), num_ratings, dim=0)
        )
        val_data = (
            self.create_input_feature_vec_raw(data.db['val']),
            torch.repeat_interleave(
                torch.tensor(data.db['val'][beta_key][:, :self.betas_size], dtype=torch.float32, device=self.device), num_ratings, dim=0)
        )
        test_data = (
            self.create_input_feature_vec_raw(data.db['test']),
            torch.repeat_interleave(
                torch.tensor(data.db['test'][beta_key][:, :self.betas_size], dtype=torch.float32, device=self.device), num_ratings, dim=0)
        )
        return train_data, val_data, test_data

    def get_tvt_data(self, data):
        beta_key = f'betas_{self.model_type}_{self.model_gender}'
        train_data = (
            self.create_input_feature_vec(data.db['train']),
            data.db['train'][beta_key][:, :self.betas_size]
        )
        val_data = (
            self.create_input_feature_vec(data.db['val']),
            data.db['val'][beta_key][:, :self.betas_size]
        )
        test_data = (
            self.create_input_feature_vec(data.db['test']),
            data.db['test'][beta_key][:, :self.betas_size]
        )
        return train_data, val_data, test_data

    def create_input_feature_vec_raw(self, batch):
        
        feature_vec = batch['rating_raw'][:, :, self.selected_attr_idx]
        num_ratings = batch['rating_raw'].shape[1]

        if not torch.is_tensor(feature_vec):
            feature_vec = torch.from_numpy(feature_vec).to(dtype=torch.float32)
        feature_vec = torch.flatten(feature_vec, start_dim=0, end_dim=1)

        noise = torch.zeros_like(feature_vec)
        

        for feature_name in self.selected_mmts:
            meas = batch[feature_name].reshape(-1, 1)
            if not torch.is_tensor(meas):
                meas = torch.from_numpy(meas).to(dtype=torch.float32)
            
            meas = torch.repeat_interleave(meas, num_ratings, dim=0)
            if self.bodytalk_meas_preprocess:
                if 'height' in feature_name:
                    meas *= 100
                if 'mass' in feature_name or 'weight' in feature_name:
                    meas = meas.pow(1.0 / 3.0)

            feature_vec = torch.hstack((feature_vec, meas))
            
        return feature_vec, noise

    def create_input_feature_vec(self, batch):
        
        feature_vec = batch['rating'][:, self.selected_attr_idx]

        if not torch.is_tensor(feature_vec):
            feature_vec = torch.from_numpy(feature_vec).to(dtype=torch.float32)

        noise = torch.zeros_like(feature_vec)
        if self.use_attr_noise:
            noise = (torch.rand_like(feature_vec) * 2 *
                     self.attr_noise_range - self.attr_noise_range)

        for feature_name in self.selected_mmts:
            meas = batch[feature_name].reshape(-1, 1)
            if not torch.is_tensor(meas):
                meas = torch.from_numpy(meas).to(dtype=torch.float32)

            if self.bodytalk_meas_preprocess:
                if 'height' in feature_name:
                    meas *= 100
                if 'mass' in feature_name or 'weight' in feature_name:
                    meas = meas.pow(1.0 / 3.0)

            feature_vec = torch.hstack((feature_vec, meas))

            if (self.use_srb_noise and
                    feature_name in ['height_gt', 'weight_gt']):
                logger.info('Problem!')
                std = self.srb[self.ds_gender][feature_name.split('_')[0]][1]
                feature_noise = torch.randn(
                    [len(feature_vec), 1], device=feature_vec.device) * std
                noise = torch.cat([noise, feature_noise], dim=-1)

        return feature_vec, noise

    def compute_losses(
        self,
        gt_output,
        pred_output,
        gt_betas=None,
        pred_betas=None,
        gt_mmts=None
    ) -> Dict[str, Tensor]:

        losses = {}

        if self.v2v_weight > 0:
            losses['v2v'] = self.v2v_weight * self.vertex_loss(
                pred_output.vertices, gt_output.vertices)

        if self.betas_weight > 0:
            losses['betas'] = self.betas_weight * self.betas_loss(
                pred_betas, gt_betas)

        if self.v2v_hd_weight > 0:
            hd_pred_verts = sparse_batch_mm(self.hd_operator,
                                            pred_output.vertices.double())
            hd_gt_verts = sparse_batch_mm(self.hd_operator,
                                          gt_output.vertices.double())
            losses['v2v_hd'] = self.v2v_hd_weight * self.vertex_loss(
                hd_pred_verts, hd_gt_verts)

        if self.edge_weight > 0:
            losses['edge'] = self.edge_weight * self.edge_loss(
                pred_output.vertices, gt_output.vertices)

        if self.compute_train_meas:
            pred_tris = pred_output.vertices[:, self.model.faces_tensor]
            pred_mmts = self.meas_module(pred_tris)['measurements']

            # gt_tris = gt_output.vertices[:, self.model.faces_tensor]
            # gt_mmts = self.meas_module(gt_tris)['measurements']

            if self.height_weight > 0:
                losses['height'] = self.height_weight * \
                    self.height_loss(
                        gt_mmts['height'], pred_mmts['height']['tensor'])

            if self.chest_weight > 0:
                losses['chest'] = self.chest_weight * \
                    self.chest_loss(
                        gt_mmts['chest'], pred_mmts['chest']['tensor'])

            if self.waist_weight > 0:
                losses['waist'] = self.waist_weight * \
                    self.waist_loss(
                        gt_mmts['waist'], pred_mmts['waist']['tensor'])

            if self.hips_weight > 0:
                losses['hips'] = self.hips_weight * \
                    self.hips_loss(
                        gt_mmts['hips'], pred_mmts['hips']['tensor'])

        return losses

    def training_step(self, batch, idx):

        # unpack batch
        gt_betas = batch['betas'][:, :self.betas_size]
        gt_mmts = {
            'height': batch['height_gt'],
            'chest': batch['chest'],
            'waist': batch['waist'],
            'hips': batch['hips']
        }

        # stack selected input variables
        input, noise = self.create_input_feature_vec(batch)

        # make prediction
        pred_betas = self.forward(input)

        betas_noise = torch.zeros_like(gt_betas)
        if self.use_betas_noise:
            noise = sample_in_sphere(gt_betas.shape[1], len(input),
                                     self.noise_constant)
            betas_noise = torch.from_numpy(noise).to(
                dtype=gt_betas.dtype, device=gt_betas.device)

        gt_output = None
        pred_output = None
        if self.compute_train_verts:
            gt_output = self.model.forward_shape(betas=gt_betas + betas_noise)
            pred_output = self.model.forward_shape(betas=pred_betas)

            with torch.no_grad():
                train_v2v = (pred_output.vertices -
                             gt_output.vertices).pow(2).sum(dim=-1).sqrt().mean()
                self.log('Train/v2v', train_v2v * 1000)

        losses = self.compute_losses(
            gt_output,
            pred_output,
            gt_betas,
            pred_betas,
            gt_mmts,
        )

        for k, v in losses.items():
            self.log(f'Loss/{k}', v)

        loss = sum(losses.values())
        self.log('Loss/train', loss)

        if self.global_step % self.summary_steps == 0:
            tensorboard = self.logger.experiment
            num_betas = pred_betas.shape[1]
            for bii in range(num_betas):
                tensorboard.add_histogram(
                    f'BetasMAE_{bii + 1:03d}',
                    (pred_betas[:, bii] - gt_betas[:, bii]).abs().mean(),
                    global_step=self.global_step,
                )

            self.render_pred_gt(batch,
                                pred_betas,
                                gt_betas,
                                gt_output=gt_output,
                                pred_output=pred_output,
                                max_summaries=MAX_SUMMARY_IMGS,
                                prefix='Train',
                                log=True,
                                save_to_disk=False,
                                )

        return loss

    def render_pred_gt(
        self,
        batch: Dict[str, Tensor],
        pred_betas: Tensor,
        gt_betas: Tensor,
        gt_output=None,
        pred_output=None,
        max_summaries: int = -1,
        prefix: str = '',
        log: bool = True,
        save_to_disk: bool = False,
    ) -> None:
        if not (self.global_step % self.render_summary_steps == 0):
            return

        if not (log or save_to_disk):
            return

        tensorboard = self.logger.experiment
        # tensorboard.add_scalars('Loss', losses, global_step=self.global_step)

        # get mesh if not computed already
        if gt_output is None:
            gt_output = self.model.forward_shape(betas=gt_betas)
        if pred_output is None:
            pred_output = self.model.forward_shape(betas=pred_betas)

        pred_verts = pred_output.vertices.detach().cpu().numpy()
        gt_verts = gt_output.vertices.detach().cpu().numpy()
        if self.align:
            t = gt_verts.mean(1) - pred_verts.mean(1)
        else:
            t = torch.zeros([len(gt_output.vertices), 3],
                            dtype=gt_output.vertices.dtype,
                            device=gt_output.vertices.device)

        verts_it = zip(pred_verts, gt_verts)

        ratings_labels = batch.get('rating_label', None)
        for ii, (pred_verts, gt_verts) in enumerate(
                tqdm(verts_it, leave=False, desc='Rendering summary')):
            if max_summaries > 0 and ii >= max_summaries:
                break

            pred_mesh = trimesh.Trimesh(pred_verts, self.model.faces)
            pred_img = self.renderer.render(pred_mesh)

            gt_mesh = trimesh.Trimesh(gt_verts, self.model.faces)
            gt_img = self.renderer.render(gt_mesh)
            diff = v2v_func(pred_verts + t[ii], gt_verts) * 1000

            colors = self.colormap(self.norm(diff))[:, :3]
            error_mesh = trimesh.Trimesh(
                gt_verts, self.model.faces,
                vertex_colors=colors, process=False)
            error_img = self.renderer.render(
                error_mesh, vertex_colors=colors)

            imgs_comb = np.concatenate(
                [pred_img, gt_img, error_img], axis=1)[:, :, :3]

            fig = self.images_plus_attributes(
                imgs_comb, batch['rating'][ii],
                ratings_labels[ii] if ratings_labels is not None else None,
                scalar_mappable=self.scalar_mappable)
            if log:
                tensorboard.add_figure(f'{prefix}/{ii:03d}',
                                       fig,
                                       global_step=self.global_step)
            if save_to_disk:
                img_id = batch['id'][ii]
                out_fname = osp.join(self.output_dir, f'{prefix}_imgs',
                                     self.dataset, f'{img_id}_pred_gt.png')
                plt.savefig(out_fname, dpi=80, bbox_inches='tight')

    def images_plus_attributes(
        self,
        img,
        attributes,
        names: Optional[List[str]] = None,
        scalar_mappable: Optional[mpl_cmap.ScalarMappable] = None,
    ):
        """Create a pyplot plot and save to buffer."""
        if torch.is_tensor(attributes):
            attributes = attributes.detach().cpu().numpy()
        fig = plt.figure(num=0, dpi=80, figsize=(40, 20))
        fig.clear()
        gs = fig.add_gridspec(1, 2)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1])
        ]

        for ax in axes:
            ax.set_axis_off()
        axes[0].imshow(img)
        axes[0].set_axis_off()
        axes[0].set_title('Prediction, ground-truth, vertex-to-vertex error')
        if scalar_mappable is not None:
            cbar = fig.colorbar(scalar_mappable, ax=axes[0],
                                orientation='vertical')
            cbar.ax.tick_params(labelsize=20)

        y = np.arange(len(attributes)) * 4
        # axes[1].barh(y, attributes, align='center')
        axes[1].barh(y, attributes, align='center', height=2.0)

        axes[1].set_xlim([0, 5])
        #  axes[1].set_yticklabels(attribute_names)
        if names is not None:
            for i, v in enumerate(y):
                axes[1].text(
                    s=f'{names[i].replace("Answer.", "")}: {attributes[i]:.2f}',
                    x=0.1, y=v, color="black", verticalalignment="center",
                    size=16)

        axes[1].set_xticks(np.arange(0, 6))
        axes[1].set_yticks([])

        axes[1].invert_yaxis()
        axes[1].set_xlabel(f'Attribute ratings for {self.ds_gender}')
        fig.tight_layout()

        return fig

    @torch.no_grad()
    def validation_step(self, batch, idx):

        output = {}

        input, noise = self.create_input_feature_vec(batch)

        pred_betas = self.forward(input)
        gt_betas = batch['betas'][:, :self.betas_size]

        # compute RE (used to save checkpoint)
        # pred_output = self.model(betas=pred_betas, return_verts=True)
        # gt_output = self.model(betas=gt_betas, return_verts=True)
        pred_output = self.model.forward_shape(betas=pred_betas)
        gt_output = self.model.forward_shape(betas=gt_betas)

        v2v_loss2 = 1000 * v2v_func(
            pred_output.vertices, gt_output.vertices).mean()
        output['RE'] = v2v_loss2
        self.log('Validation/v2v', v2v_loss2)

        output['v2v_array'] = v2v_func(
            pred_output.vertices, gt_output.vertices)

        if self.val_metric == 'betas':
            betas_error = (pred_betas - gt_betas).pow(2).sum(dim=-1).sqrt()
            output['betas'] = betas_error
            self.log('Validation/betas', betas_error)

        if self.val_metric == 'v2v':
            v2v_mse = 1000 * self.vertex_loss(
                pred_output.vertices, gt_output.vertices)
            output['v2v_mse'] = v2v_mse

        if self.val_metric == 'v2v_hd':
            hd_pred_verts = sparse_batch_mm(self.hd_operator,
                                            pred_output.vertices.double())
            hd_gt_verts = sparse_batch_mm(self.hd_operator,
                                          gt_output.vertices.double())
            v2v_hd_error = v2v_func(hd_pred_verts, hd_gt_verts) * 1000
            output['v2v_hd'] = v2v_hd_error
            self.log('Validation/v2v_hd', v2v_hd_error)

        pred_tris = pred_output.vertices[:, self.model.faces_tensor]
        pred_mmts = self.meas_module(pred_tris)['measurements']

        gt_tris = gt_output.vertices[:, self.model.faces_tensor]
        gt_mmts = self.meas_module(gt_tris)['measurements']

        height_error = abs(gt_mmts['height']['tensor'] - pred_mmts['height']['tensor']).mean() * 1000
        self.log('Validation/height', height_error)

        chest_error = abs(gt_mmts['chest']['tensor'] - pred_mmts['chest']['tensor']).mean() * 1000 
        self.log('Validation/chest', chest_error)

        waist_error = abs(gt_mmts['waist']['tensor'] - pred_mmts['waist']['tensor']).mean() * 1000
        self.log('Validation/waist', waist_error)

        hips_error = abs(gt_mmts['hips']['tensor'] - pred_mmts['hips']['tensor']).mean() * 1000 
        self.log('Validation/hips', hips_error)

        if self.val_metric == 'v2v_hd':
            self.log('Loss/val', v2v_hd_error.mean())
        elif self.val_metric == 'v2v':
            self.log('Loss/val', v2v_mse.mean())
        elif self.val_metric == 'betas':
            self.log('Loss/val', betas_error.mean())
        elif self.val_metric == 'measurements':
            mmts_avg_error = (height_error + chest_error \
                + waist_error + hips_error) / 4
            self.log('Loss/val', mmts_avg_error)

        return output

    def validation_epoch_end(self, outputs):
        all_v2v = np.concatenate(
            [lst['v2v_array'].detach().cpu().numpy()
             for lst in outputs], axis=0)

        v2v_mesh = all_v2v.mean(0) * 1000
        colors = self.colormap(self.norm(v2v_mesh))[:, :3]
        error_mesh = trimesh.Trimesh(
            self.model.v_template.detach().cpu().numpy(),
            self.model.faces,
            vertex_colors=colors, process=False)
        error_img = self.renderer.render(error_mesh, vertex_colors=colors)
        error_img = np.asarray(error_img)

        tensorboard = self.logger.experiment

        tensorboard.add_image('Validation/V2VMean', error_img,
                              global_step=self.global_step, dataformats='HWC'
                              )

        v2v_mesh_median = np.median(all_v2v, axis=0) * 1000
        colors = self.colormap(self.norm(v2v_mesh_median))[:, :3]
        error_mesh = trimesh.Trimesh(
            self.model.v_template.detach().cpu().numpy(),
            self.model.faces,
            vertex_colors=colors, process=False)
        error_img = self.renderer.render(error_mesh, vertex_colors=colors)
        error_img = np.asarray(error_img)

        tensorboard.add_image('Validation/V2VMedian', error_img,
                              global_step=self.global_step, dataformats='HWC'
                              )

    def test_step(self, batch, idx):

        input, noise = self.create_input_feature_vec(batch)

        gt_betas = batch['betas'][:, :self.betas_size]

        pred_betas = self.forward(input)

        # compute measurements
        mmts = self.mmts_mae(pred_betas, gt_betas)
        # for k, v in mmts.items():
        #     self.eval_output[k] += [v]

        pred_output = self.model.forward_shape(betas=pred_betas)
        gt_output = self.model.forward_shape(betas=gt_betas)

        if self.align:
            t = gt_output.vertices.mean(1) - pred_output.vertices.mean(1)
        else:
            t = torch.zeros([len(gt_output.vertices), 3],
                            dtype=gt_output.vertices.dtype,
                            device=gt_output.vertices.device)

        # compute reconstruction error (RE) BodyTalk
        REALL = torch.norm(pred_output.vertices + t.reshape(-1, 1, 3) -
                           gt_output.vertices, dim=2)
        RE = REALL.mean(1)
        if idx == 0:
            self.all_verts_v2v = []
        self.all_verts_v2v += [REALL]
        self.eval_output['RE'] += [RE]

        # compute reconstruction error of HD vertices (RE_HD)
        hd_pred_verts = sparse_batch_mm(self.hd_operator,
                                        pred_output.vertices.double())
        hd_gt_verts = sparse_batch_mm(self.hd_operator,
                                      gt_output.vertices.double())
        RE_HD = torch.norm(hd_pred_verts + t.reshape(-1, 1, 3) -
                           hd_gt_verts, dim=2).mean(1)
        self.eval_output['RE_HD'] += [RE_HD]

        if self.render_result:
            self.render_pred_gt(batch,
                                pred_betas,
                                gt_betas,
                                gt_output=gt_output,
                                pred_output=pred_output,
                                max_summaries=-1,
                                log=False,
                                prefix=self.prefix,
                                save_to_disk=True,
                                )

        out = {}
        # for k, v in self.eval_output.items():
        #     out[k] = torch.stack(v).flatten().mean()
        for k, v in mmts.items():
            out[k] = v
        out['RE'] = RE
        out['RE_HD'] = RE_HD

        return out

    def make_error_hist(self, name, values):
        fig = plt.figure(num=0, dpi=80, figsize=(40, 20))
        fig.clear()

        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()

        ax = fig.add_subplot(1, 1, 1)
        ax.hist(values, bins='auto')

        ax.set_xlabel(f'(GT - Pred) for {name.title()}')

        return fig

    def print_result(self, result_dict):

        row_format = "{:>15}" * 5
        logger.info(row_format.format('Measure', 'Mean', 'SD', 'MIN', 'MAX'))

        for k, v in result_dict.items():
            f = 1000 if k != 'mass' else 1
            v = f * np.array(v)
            mean = f'{np.round(v.mean(), 2):.2f}'
            sd = f'{np.round(v.std(), 2):.2f}'
            min = f'{np.round(v.min(), 2):.2f}'
            max = f'{np.round(v.max(), 2):.2f}'
            out = [k, mean, sd, min, max]
            logger.info(row_format.format(*out))

    def test_epoch_end(self, outputs):

        all_metrics = defaultdict(lambda: [])
        # Gather the metrics from all the runs in
        for metric_lst in outputs:
            for key, val in metric_lst.items():
                all_metrics[key].append(val)
        # Stack all metric values into a single tensor
        for key in all_metrics:
            all_metrics[key] = torch.cat(all_metrics[key]).flatten()

        all_metrics = dict(all_metrics)

        hist_output_dir = osp.join(self.output_dir, 'hists', self.dataset)
        os.makedirs(hist_output_dir, exist_ok=True)
        # Print the values of the metrics
        for mm, mm_res in all_metrics.items():
            mm_mean = mm_res.abs().mean().item()
            mm_std = mm_res.abs().std().item()
            cv = 1000 if mm != 'mass' else 1
            logger.info(f'{mm}: {cv * mm_mean:.2f} (SD={cv * mm_std:.2f})')

            hist_fig = self.make_error_hist(mm, mm_res * cv)

            plt.savefig(osp.join(hist_output_dir,
                                 f'{mm}.png'), dpi=80, bbox_inches='tight')

        # Dump the results into a file
        with open(osp.join(self.output_dir, 'test_result.pkl'), 'wb') as f:
            pickle.dump(all_metrics, f)

        per_vertex_v2v = torch.cat(self.all_verts_v2v, axis=0)
        if torch.is_tensor(per_vertex_v2v):
            per_vertex_v2v = per_vertex_v2v.detach().cpu().numpy()
        per_vertex_v2v = per_vertex_v2v.reshape(
            -1, self.model.get_num_verts()).mean(0) * 1000
        colors = self.colormap(self.norm(per_vertex_v2v))[:, :3]
        error_mesh = trimesh.Trimesh(
            self.model.v_template.detach().cpu().numpy(),
            self.model.faces,
            vertex_colors=colors, process=False)
        error_img = self.renderer.render(error_mesh, vertex_colors=colors)
        error_img = np.asarray(error_img)

        IMG = pil_img.fromarray(error_img)
        IMG.save(f'{self.output_dir}/{self.cfg.dataset}_testset_all_meshes.png')

    def mmts_mae(self, pred_betas, gt_betas):
        """
            Function to compute body measurements from vertices.
        """

        output = {
            'mass': 0,
            'height': 0,
            'chest': 0,
            'waist': 0,
            'hips': 0
        }

        pred_output = self.model.forward_shape(betas=pred_betas)
        gt_output = self.model.forward_shape(betas=gt_betas)

        cuda_device = torch.device('cuda')
        if not (next(self.meas_module.buffers()) == cuda_device):
            self.meas_module = self.meas_module.to(device=cuda_device)

        # pred_output = self.model(betas=pred_betas, return_verts=True)
        pred_tris = pred_output.vertices[:, self.model.faces_tensor]
        if not (pred_tris.device == cuda_device):
            pred_tris = pred_tris.to(device=cuda_device)

        pred_mmts = self.meas_module(pred_tris)['measurements']

        # gt_output = self.model(betas=gt_betas, return_verts=True)
        gt_tris = gt_output.vertices[:, self.model.faces_tensor]
        if not (gt_tris.device == cuda_device):
            gt_tris = gt_tris.to(device=cuda_device)
        gt_mmts = self.meas_module(gt_tris)['measurements']

        for mm in ['mass', 'height']:
            output[mm] = gt_mmts[mm]['tensor'] - pred_mmts[mm]['tensor']

        for mm in ['chest', 'waist', 'hips']:
            output[mm] = gt_mmts[mm]['tensor'] - pred_mmts[mm]['tensor']

        return output

    '''
    def log_mmts(self, mmts_dict, set):
        for k, v in mmts_dict.items():
            if k != 'mass':
                self.log('{}/mmts/val_mm_{}'.format(set, k),
                         torch.stack(v).mean() * 1000)
            else:
                self.log('{}/mmts/val_kg_{}'.format(set, k),
                         torch.stack(v).mean())

    # comment in to visualize validation set and eval measurements
    def validation_epoch_end(self, val_step_outputs):

        output = {
            'mass': [],
            'height': [],
            'chest': [],
            'waist': [],
            'hips': [],
        }
        os.makedirs(f'outdebug/a2b/{self.current_epoch}', exist_ok=True)
        for idx, pred in enumerate(val_step_outputs):
            if pred['pred_betas'].shape[0] == self.batch_size:
                pred_betas = pred['pred_betas']
                gt_betas = pred['gt_betas']

                mmts = self.mmts_mae(pred_betas, gt_betas)
                for k, v in mmts.items():
                    output[k] += [v]

                # render fit
                pred_output = self.model(betas=pred_betas, return_verts=True)
                gt_output = self.model(betas=gt_betas, return_verts=True)
                count = 0
                for pred_verts, gt_verts, id in zip(pred_output.vertices,
                                                gt_output.vertices,
                                                pred['id']):
                    count += 1
                    if count < 100:
                        pred_mesh = trimesh.Trimesh(pred_verts.detach() \
                            .cpu().numpy(), self.model.faces)
                        pred_img = self.renderer.render(pred_mesh)

                        gt_mesh = trimesh.Trimesh(gt_verts.detach() \
                            .cpu().numpy(), self.model.faces)
                        gt_img = self.renderer.render(gt_mesh)

                        imgs_comb = np.hstack((np.asarray(i) \
                                       for i in [pred_img, gt_img]))[:,:,:3]
                        IMG = pil_img.fromarray(imgs_comb)
                        IMG.save(f'outdebug/a2b/{self.current_epoch}/{id}_pred_gt.png')

        self.log_mmts(output, 'val')

        return output
        '''


class A2BProbabilistic(A2B):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(A2BProbabilistic, self).__init__(cfg)

        prob_cfg = cfg.get('probabilistic', {})
        self.a2b = build_distr_regressor(
            cfg, self.input_feature_size, self.betas_size)

        self.num_samples = prob_cfg.get('num_samples', 32)
        self.noise_std = prob_cfg.get('noise_std', 0.0)

        self.ll_weight = cfg.get('ll_weight', 0.0)
        logger.info(f'Negative log-likelihood weight: {self.ll_weight}')

    def forward(
        self,
        attributes: Tensor,
        betas: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        return self.a2b(cond=attributes, values=betas)

    def log_likelihood(
        self,
        attributes: Tensor,
        betas: Tensor
    ) -> Tensor:
        nll = self.a2b.neg_log_likelihood(attributes, betas)
        return nll

    def training_step(self, batch, idx):

        gt_betas = batch['betas'][:, :self.betas_size]
        gt_mmts = {
            'height': batch['height_gt'],
            'chest': batch['chest'],
            'waist': batch['waist'],
            'hips': batch['hips']
        }

        # stack selected input variables
        input, noise = self.create_input_feature_vec(batch)
        if self.noise_std > 0:
            input += torch.randn_like(input) * self.noise_std

        B = len(input)

        gt_betas = batch['betas'][:, :self.betas_size]

        model_output_dict = self.a2b.samples_and_loglikelihood(
            self.num_samples, cond=input, values=gt_betas,
            return_mean=True,
        )

        samples = model_output_dict['samples']
        ll = model_output_dict['neg_log_likelihood']

        pred_output = self.model.forward_shape(betas=samples.reshape(
            B * self.num_samples, -1))
        pred_output.vertices = pred_output.vertices.reshape(
            B, self.num_samples, -1, 3)
        gt_output = self.model.forward_shape(betas=gt_betas)

        gt_vertices = gt_output.vertices.reshape(B, 1, -1, 3).expand(
            -1, self.num_samples, -1, -1)
        gt_output.vertices = gt_vertices

        losses = {}

        gt_mmts = {
            'height': batch['height_gt'],
            'chest': batch['chest'],
            'waist': batch['waist'],
            'hips': batch['hips']
        }
        losses = super(A2BProbabilistic, self).compute_losses(
            gt_output,
            pred_output,
            gt_betas=gt_betas,
            pred_betas=model_output_dict['mean'],
            gt_mmts=gt_mmts,
        )
        if self.ll_weight > 0:
            losses['nll'] = self.ll_weight * ll.mean()

        loss = sum(losses.values())
        self.log('Loss/train', loss)

        train_v2v = v2v_func(pred_output.vertices, gt_output.vertices).mean()
        self.log('Train/V2V', train_v2v * 1000)

        tensorboard = self.logger.experiment
        tensorboard.add_scalars('Loss', losses, global_step=self.global_step)

        self.render_pred_gt(batch,
                            model_output_dict['mean'],
                            gt_betas,
                            gt_output=None,
                            pred_output=None,
                            max_summaries=MAX_SUMMARY_IMGS,
                            prefix='Train',
                            )

        return loss

    def validation_step(self, batch, idx):

        input, noise = self.create_input_feature_vec(batch)

        gt_betas = batch['betas'][:, :self.betas_size]

        model_output_dict = self.a2b(values=gt_betas, cond=input)
        if 'nll' not in model_output_dict:
            nll = self.a2b.neg_log_likelihood(values=gt_betas, cond=input)

        # Use the mean as the point prediction
        pred_betas = model_output_dict['mean']
        pred_output = self.model.forward_shape(betas=pred_betas)
        gt_output = self.model.forward_shape(betas=gt_betas)

        output = {}

        v2v_loss2 = 1000 * v2v_func(
            pred_output.vertices, gt_output.vertices) .mean()

        output['RE'] = v2v_loss2
        self.log('Loss/RE', v2v_loss2)

        if self.val_metric == 'betas':
            betas_error = (pred_betas - gt_betas).pow(2).sum(dim=-1).sqrt()
            output['betas'] = betas_error
            self.log('Loss/betas', betas_error)

        if self.val_metric == 'v2v':
            v2v_mse = 1000 * self.vertex_loss(
                pred_output.vertices, gt_output.vertices)
            output['v2v_mse'] = v2v_mse

        if self.val_metric == 'v2v_hd':
            hd_pred_verts = sparse_batch_mm(self.hd_operator,
                                            pred_output.vertices.double())
            hd_gt_verts = sparse_batch_mm(self.hd_operator,
                                          gt_output.vertices.double())
            v2v_hd_error = v2v_func(hd_pred_verts, hd_gt_verts) * 1000
            output['v2v_hd'] = v2v_hd_error
            self.log('Loss/v2v_hd', v2v_hd_error)

        if self.val_metric == 'v2v_hd':
            self.log('Loss/val', v2v_hd_error.mean())
        elif self.val_metric == 'v2v':
            self.log('Loss/val', v2v_mse.mean())
        elif self.val_metric == 'betas':
            self.log('Loss/val', betas_error.mean())

        return output

    def test_step(self, batch, idx):
        raise NotImplementedError
        input, noise = self.create_input_feature_vec(batch)

        gt_betas = batch['betas'][:, :self.betas_size]

        model_output_dict = self.a2b(input)
        nll = self.a2b.neg_log_likelihood(
            input, gt_betas, mean=model_output_dict['mean'],
            L=model_output_dict['L']
        )

        # Use the mean as the point prediction
        pred_betas = model_output_dict['mean']
        pred_output = self.model(betas=pred_betas, return_verts=True)
        gt_output = self.model(betas=gt_betas, return_verts=True)

        # compute measurements
        mmts = self.mmts_mae(pred_betas, gt_betas)
        # for k, v in mmts.items():
        #     self.eval_output[k] += [v]

        pred_output = self.model(betas=pred_betas, return_verts=True)
        gt_output = self.model_eval(betas=gt_betas, return_verts=True)

        if self.align:
            t = gt_output.vertices.mean(1) - pred_output.vertices.mean(1)
        else:
            t = torch.zeros([len(gt_output.vertices), 3],
                            dtype=gt_output.vertices.dtype,
                            device=gt_output.vertices.device)

        # compute reconstruction error (RE) BodyTalk
        RE = torch.norm(pred_output.vertices + t.reshape(-1, 1, 3) -
                        gt_output.vertices, dim=2).mean(1)
        self.eval_output['RE'] += [RE]

        # compute reconstruction error of HD vertices (RE_HD)
        # hd_pred_verts = torch.matmul(self.hd_operator, pred_output.vertices)
        # hd_gt_verts = torch.matmul(self.hd_operator, gt_output.vertices)
        hd_pred_verts = sparse_batch_mm(self.hd_operator,
                                        pred_output.vertices.double())
        hd_gt_verts = sparse_batch_mm(self.hd_operator,
                                      gt_output.vertices.double())
        RE_HD = torch.norm(hd_pred_verts + t.reshape(-1, 1, 3) -
                           hd_gt_verts, dim=2).mean(1)
        self.eval_output['RE_HD'] += [RE_HD]

        if idx == 0:
            self.all_verts_v2v = []

        ii = 0
        B = len(gt_betas)
        self.render_pred_gt(batch,
                            pred_betas,
                            gt_betas,
                            gt_output=gt_output,
                            pred_output=pred_output,
                            prefix=self.prefix,
                            log=False,
                            save_to_disk=True,
                            )

        out = {}
        # for k, v in self.eval_output.items():
        #     out[k] = torch.stack(v).flatten().mean()
        for k, v in mmts.items():
            out[k] = v
        out['RE'] = RE
        out['RE_HD'] = RE_HD
        return out

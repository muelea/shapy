import torch
import smplx
import os
import os.path as osp
import pickle
import trimesh
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cmap

from loguru import logger
from torch.optim import Adam
from body_measurements import BodyMeasurements
import PIL.Image as pil_img

from omegaconf import DictConfig

from attributes.attributes_betas.models import build_network
from attributes.utils.config import get_features_from_config


class B2A(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.batch_size = cfg.get('batch_size', 32)
        self.betas_size = cfg.get('num_shape_comps', 10)
        self.output_dir = cfg.get('output_dir')
        self.model_type = cfg.get('model_type', 'smplx')
        self.model_gender = cfg.get('model_gender', 'female')
        self.ds_gender = cfg.get('ds_gender', 'female')

        self.selected_attr, self.selected_attr_idx, \
            self.selected_mmts = get_features_from_config(cfg)
        self.output_feature_size = len(
            self.selected_attr) + len(self.selected_mmts)

        network_cfg = cfg.get('network', {})
        self.b2a = build_network(
            network_cfg, self.betas_size, self.output_feature_size)

        self.loss = nn.MSELoss()

        self.eval_output = {
            'diff': [],
            'classification_error': []
        }

    def fit(self, data):

        self.rating_label = data.db['labels']
        train_data, val_data, test_data = self.get_tvt_data(data)
        self.fit_tvt(self.b2a, train_data, val_data, test_data)

    def get_tvt_data(self, data):
        beta_key = f'betas_{self.model_type}_{self.model_gender}'
        train_data = (
            data.db['train'][beta_key][:, :self.betas_size],
            data.db['train']['rating']
        )
        val_data = (
            data.db['val'][beta_key][:, :self.betas_size],
            data.db['val']['rating']
        )
        test_data = (
            data.db['test'][beta_key][:, :self.betas_size],
            data.db['test']['rating']
        )
        return train_data, val_data, test_data


    def fit_tvt(self, model, train_data, val_data, test_data):
        
        train_input, train_output = train_data
        val_input, val_output = val_data
        test_input, test_output = test_data

        fitted_model = model.fit(train_input, train_output)

        # predict and eval betas val / test
        logger.info('Reporting results on validation set.')
        val_prediction = fitted_model.predict(val_input)

        mean, std = self.metric_mean_std(val_output, val_prediction)
        ccp = self.metric_classification(val_output, val_prediction)

        # print result for each attribute
        output_names = self.selected_attr + self.selected_mmts
        for i, name in enumerate(output_names):
            l1m = mean[i].item()
            l1std = std[i].item()
            acc = ccp[i].item() * 100
            print(f'{name:20s} &   ${l1m:.2f} \pm {l1std:.2f}$   &   ${acc:.2f}\%$   &   &   \\\\')

    def metric_mean_std(self, gt_ratings, pred_ratings):
        # mean and std absolute error
        mean = np.absolute(gt_ratings-pred_ratings).mean(0)
        std = np.absolute(gt_ratings-pred_ratings).std(0)
        return mean, std
    
    def metric_classification(self, gt_ratings, pred_ratings):
        # classification error
        gt_ratings_class = np.round(gt_ratings)
        pred_ratings_class = np.round(pred_ratings)
        correct_class = (gt_ratings_class == pred_ratings_class)
        correct_class_prec = correct_class.sum(0) / correct_class.shape[0]
        return correct_class_prec

    def forward(self, x):
        return self.b2a(x)

    def configure_optimizers(self):
        # summary when starting training shows all params as trainable ???
        optimizer = Adam(
            self.b2a.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        return optimizer

    def create_output_feature_vec(self, batch):
        feature_vec = batch['rating'][:, self.selected_attr_idx]
        for feature_name in self.selected_mmts:
            feature_vec = torch.hstack(
                (feature_vec, batch[feature_name].view(-1, 1))
            )
        return feature_vec

    def to_eval_mode(self, *args):
        pass

    def training_step(self, batch, idx):

        input = batch['betas'][:, :self.betas_size]
        pred_ratings = self.forward(input)

        gt_ratings = self.create_output_feature_vec(batch)

        loss = self.loss(pred_ratings, gt_ratings)
        self.log('Loss/train', loss)

        return loss

    def validation_step(self, batch, idx):

        input = batch['betas'][:, :self.betas_size]
        pred_ratings = self.forward(input)

        gt_ratings = self.create_output_feature_vec(batch)

        loss = self.loss(pred_ratings, gt_ratings)
        self.log('Loss/val', loss)

        # classification error
        gt_ratings_class = torch.round(gt_ratings)
        pred_ratings_class = torch.round(pred_ratings)
        correct_class = (gt_ratings_class == pred_ratings_class)
        correct_class_prec = correct_class.sum(0) / correct_class.shape[0]
        self.log('Loss/correct_class', correct_class_prec * 100)

    def test_step(self, batch, idx):

        input = batch['betas'][:, :self.betas_size]
        pred_ratings = self.forward(input)

        gt_ratings = self.create_output_feature_vec(batch)

        # mean and std absolute error
        error = gt_ratings - pred_ratings
        self.eval_output['diff'] += [error]

        # classification error
        gt_ratings_class = torch.round(gt_ratings)
        pred_ratings_class = torch.round(pred_ratings)
        self.eval_output['classification_error'] += \
            [(gt_ratings_class == pred_ratings_class)]

    def test_epoch_end(self, output):

        for k, v in self.eval_output.items():
            if k == 'diff':
                l1_mean = torch.cat(v, dim=0).abs().mean(0)
                l1_std = torch.cat(v, dim=0).abs().std(0)
            if k == 'classification_error':
                ces = torch.cat(v, dim=0)
                avg_correct_class = 100 * ces.sum(0) / ces.shape[0]

        # print result for each attribute
        output_names = self.selected_attr + self.selected_mmts
        for i, name in enumerate(output_names):
            l1m = l1_mean[i].item()
            l1std = l1_std[i].item()
            acc = avg_correct_class[i].item()
            print(f'{name:20s} : {l1m:.2f} (SD={l1std:.2f}) : {acc:.2f}%')

        # Average all ratings
        acc = avg_correct_class.mean().item()
        print(f'Correct class overall average: {acc:.2f}%')

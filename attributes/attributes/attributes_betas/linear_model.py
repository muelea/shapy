from typing import Tuple
import sys
import smplx
import torch
import numpy as np
from loguru import logger
from collections import defaultdict
from body_measurements import BodyMeasurements
from attributes.utils import constants
from attributes.utils.config import get_features_from_config
from attributes.utils.sparse import sparse_batch_mm


def build(cfg, eval_bs, device):
    network_type = cfg.get('type', 'a2b')
    if network_type == 'a2b':
        return FitterA2S(cfg, eval_bs, device=device)
    elif network_type == 'b2a':
        return FitterS2A(cfg, eval_bs, device=device)
    else:
        raise ValueError(f'Unknown model: {network_type}')


class FitterA2S(object):
    def __init__(
        self,
        cfg,
        eval_bs=1,
        device='cuda',
    ):

        self.eval_val = cfg.get('eval_val', True)
        logger.info(f'Eval val: {self.eval_val}')
        self.eval_test = cfg.get('eval_test', False)
        logger.info(f'Eval test: {self.eval_test}')

        self.device = device
        self.align = cfg.align
        self.model_type = cfg.model_type
        self.model_gender = cfg.model_gender
        self.num_shape_comps = cfg.num_shape_comps
        self.use_loo = cfg.regression.use_loo_cross_val
        self.eval_bs = eval_bs
        self.whw2s_model = cfg.regression.use_whw2s_setting
        logger.info(f'Use BodyTalk data pre-processing: {self.whw2s_model}')
        self.add_noise = cfg.regression.get('add_noise', False)
        logger.info(f'Add noise to input: {self.add_noise}')
        self.srb = constants.SELF_REPORT_BIAS

        self.selected_attr, self.selected_attr_idx, \
            self.selected_mmts = get_features_from_config(cfg)
        self.num_features = len(self.selected_attr_idx) + \
            len(self.selected_mmts)

        self.attr_feat_names = []
        if len(self.selected_attr_idx) > 0:
            self.attr_feat_names = np.array(self.selected_attr)[
                self.selected_attr_idx]
        self.feature_names = np.concatenate(
            [self.attr_feat_names, np.array(self.selected_mmts)])

        # create body model
        logger.info(f'Creating {self.model_gender} {self.model_type} '
                    f'model with {self.num_shape_comps} shape components.')
        self.body_model = smplx.create(
            gender=self.model_gender,
            model_type=self.model_type,
            num_betas=self.num_shape_comps,
            flat_hand_mean=True,
            model_path=constants.MODEL_PATH,
            batch_size=self.eval_bs,
        ).to(device=device)

        hd_operator_path = f'../data/hd/{self.model_type}/HD_{self.model_type.upper()}_NEUTRAL_vert_regressor_sparse.pt'

        # load measurements module
        cfg = {
            'meas_definition_path': '../data/measurements/measurement_defitions.yaml',
            'meas_vertices_path': '../data/measurements/smpl_measurement_vertices.yaml',
        }
        if self.model_type == 'smplx':
            cfg['meas_vertices_path'] = '../data/measurements/smplx_measurements.yaml'
        self.meas_module = BodyMeasurements(cfg).to(self.device)

        self.hd_operator = torch.load(hd_operator_path).to(self.device)

    def get_input_output_dim(self) -> Tuple[int, int]:
        return self.num_features, self.num_shape_comps

    def fit(self, model, data):
        self.rating_label = data.db['labels']

        if self.use_loo:
            input, output = self.get_loo_data(data)
            self.fit_loo(model, input, output)
        else:
            train_data, val_data, test_data = self.get_tvt_data(data)
            self.fit_tvt(model, train_data, val_data, test_data)

    def to_whw2s(self, data, noise):

        logger.info('Converting input features to BodyTalk setup.')

        data = data.copy()

        height_idx = np.where(self.feature_names == 'height_gt')[0]
        weight_idx = np.where(self.feature_names == 'weight_gt')[0]

        # change height unit to cm to fit the noise unit.
        data[:, height_idx] = data[:, height_idx] * 100

        if noise is None:
            data[:, weight_idx] = np.sqrt(data[:, weight_idx])
        else:
            # for weight we must first add the noise (noise unit is kg).
            # The noise unit of height is cm, so it can be added later.
            weight_data = data[:, weight_idx] + noise[:, weight_idx]
            data[:, weight_idx] = np.sqrt(weight_data)
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
        fitted_model = model.fit(train_input, train_output)

        # predict and eval betas val / test
        if self.eval_val:
            logger.info('Reporting results on validation set.')
            val_input = self.to_whw2s(val_input, None) \
                if self.whw2s_model else val_input
            predition = fitted_model.predict(val_input)
            val_output = self.validate(val_output, predition)
            self.print_result(val_output)

        if self.eval_test:
            logger.info('Reporting results on test set.')
            test_input = self.to_whw2s(test_input, None) \
                if self.whw2s_model else test_input
            predition = fitted_model.predict(test_input)
            test_output = self.validate(test_output, predition)
            self.print_result(test_output)

        # visu model coefficients
        # create_coef_heatmap(fitted_model.coef_, feature_names,
        #    np.arange(num_betas), plot_name)

    def get_translation(self, gt_vertices, pred_vertices):
        if self.align:
            t = gt_vertices.mean(1) - pred_vertices.mean(1)
        else:
            t = torch.zeros([len(gt_vertices), 3],
                            dtype=gt_vertices.dtype,
                            device=gt_vertices.device)
        return t

    def v2v_error(self, gt_vertices, pred_vertices):
        t = self.get_translation(gt_vertices, pred_vertices)
        loss = torch.norm(pred_vertices + t.reshape(-1, 1, 3) -
                          gt_vertices, dim=2).mean(1)

        return loss.item()

    def v2v_hd_error(self, gt_vertices, pred_vertices):

        hd_gt_verts = sparse_batch_mm(self.hd_operator, gt_vertices.double())
        hd_pred_verts = sparse_batch_mm(
            self.hd_operator, pred_vertices.double())

        t = self.get_translation(hd_gt_verts, hd_pred_verts)
        loss = torch.norm(hd_pred_verts + t.reshape(-1, 1, 3) -
                          hd_gt_verts, dim=2).mean(1)

        return loss.item()

    def mmts_error(self, gt_vertices, pred_vertices):
        output = {}

        pred_tris = pred_vertices[:, self.body_model.faces_tensor]
        pred_mmts = self.meas_module(pred_tris)['measurements']

        gt_tris = gt_vertices[:, self.body_model.faces_tensor]
        gt_mmts = self.meas_module(gt_tris)['measurements']

        for mm in ['mass']:
            output[f'{mm}'] = [abs(gt_mmts[mm]['tensor'] -
                                   pred_mmts[mm]['tensor']).item()]

        for mm in ['chest', 'waist', 'hips', 'height']:
            output[f'{mm}'] = [abs(gt_mmts[mm]['tensor'] -
                                   pred_mmts[mm]['tensor']).item()]

        return output

    def validate(self, gt_betas, pred_betas):

        val_output = defaultdict(lambda: [])

        for gt_beta, pred_beta in zip(gt_betas, pred_betas):
            if not torch.is_tensor(gt_beta):
                gt_beta = torch.from_numpy(gt_beta)
            if not torch.is_tensor(pred_beta):
                pred_beta = torch.from_numpy(pred_beta)

            pred_beta = pred_beta.unsqueeze(0).to(self.device,
                                                  dtype=torch.float32)
            gt_beta = gt_beta.unsqueeze(0).to(self.device,
                                              dtype=torch.float32)

            gt_verts = self.body_model.forward_shape(betas=gt_beta).vertices
            pred_verts = self.body_model.forward_shape(
                betas=pred_beta).vertices

            # Reconstruction errors
            val_output['RE'] += [self.v2v_error(gt_verts, pred_verts)]
            val_output['RE_HD'] += [self.v2v_hd_error(gt_verts, pred_verts)]

            # Measurements errors
            mmts_errors = self.mmts_error(gt_verts, pred_verts)
            for k, v in mmts_errors.items():
                val_output[k] += v

        return val_output

    def get_noise_mat(self, num_samples, ds_gender):

        noise = np.zeros((num_samples, self.num_features))

        if not self.add_noise:
            return noise

        # create a noise vector. Keep same order as features to make
        # sure the noise is added to the corresponding feature.
        for idx, feature_name in enumerate(self.selected_mmts):
            feature_idx = len(self.selected_attr_idx) + idx
            if feature_name in ['height_gt', 'weight_gt']:
                std = self.srb[ds_gender][feature_name.split('_')[0]][1]
                feature_noise = np.random.normal(0.0, std, num_samples)
                noise[:, feature_idx] = feature_noise

        return noise

    def create_input_feature_vec(self, data, ds_gender):

        num_samples = data['rating'].shape[0]

        # create ratings matrix
        if len(self.selected_attr_idx) == 0:
            feature_vec = np.empty((num_samples, 0))
        else:
            feature_vec = data['rating'][:, self.selected_attr_idx]

        # stack ratings with measurements
        for feature_name in self.selected_mmts:
            feature_vec = np.hstack(
                (feature_vec, data[feature_name].reshape(-1, 1))
            )

        # get noise matrix
        noise = self.get_noise_mat(num_samples, ds_gender)

        return feature_vec, noise

    def get_loo_data(self, data):
        input = self.create_input_feature_vec(data.db, data.ds_gender)
        beta_key = f'betas_{self.model_type}_{self.model_gender}'
        output = data.db[beta_key][:, :self.num_shape_comps]
        return input, output

    def get_tvt_data(self, data):
        beta_key = f'betas_{self.model_type}_{self.model_gender}'
        train_data = (
            self.create_input_feature_vec(data.db['train'], data.ds_gender),
            data.db['train'][beta_key][:, :self.num_shape_comps]
        )
        val_data = (
            self.create_input_feature_vec(data.db['val'], data.ds_gender),
            data.db['val'][beta_key][:, :self.num_shape_comps]
        )
        test_data = (
            self.create_input_feature_vec(data.db['test'], data.ds_gender),
            data.db['test'][beta_key][:, :self.num_shape_comps]
        )
        return train_data, val_data, test_data

    def print_result(self, result_dict):

        row_format = "{:>15}" * 5
        logger.info(row_format.format('Measure', 'Mean', 'SD', 'MIN', 'MAX'))

        for k, v in result_dict.items():
            f = 1000 if k != 'mass' else 1
            v = f * np.array(v)
            mean = np.round(v.mean(), 2)
            sd = np.round(v.std(), 2)
            min = np.round(v.min(), 2)
            max = np.round(v.max(), 2)
            out = [k, mean, sd, min, max]
            logger.info(row_format.format(*out))


class FitterS2A(object):
    def __init__(
        self,
        cfg,
        eval_bs=1,
        device='cuda',
    ):
        super(FitterS2A, self).__init__()

        self.eval_val = cfg.get('eval_val', True)
        logger.info(f'Eval val: {self.eval_val}')
        self.eval_test = cfg.get('eval_test', False)
        logger.info(f'Eval test: {self.eval_test}')

        self.device = device
        self.align = cfg.align
        self.model_type = cfg.model_type
        self.model_gender = cfg.model_gender
        self.num_shape_comps = cfg.num_shape_comps
        self.use_loo = cfg.regression.use_loo_cross_val
        self.eval_bs = eval_bs
        self.whw2s_model = cfg.regression.use_whw2s_setting
        logger.info(f'Use BodyTalk data pre-processing: {self.whw2s_model}')
        self.add_noise = cfg.regression.get('add_noise', False)
        logger.info(f'Add noise to input: {self.add_noise}')
        self.srb = constants.SELF_REPORT_BIAS

        self.selected_attr, self.selected_attr_idx, \
            self.selected_mmts = get_features_from_config(cfg)
        self.num_features = len(self.selected_attr_idx) + \
            len(self.selected_mmts)
        self.feature_names = np.concatenate(
            (np.array(self.selected_attr)[self.selected_attr_idx],
             np.array(self.selected_mmts)))

        # create body model
        logger.info(f'Creating {self.model_gender} {self.model_type} '
                    f'model with {self.num_shape_comps} shape components.')

    def get_input_output_dim(self) -> Tuple[int, int]:
        return self.num_shape_comps, self.num_features

    def fit(self, model, data):
        if self.use_loo:
            input, output = self.get_loo_data(data)
            self.fit_loo(model, input, output)
        else:
            train_data, val_data, test_data = self.get_tvt_data(data)
            self.fit_tvt(model, train_data, val_data, test_data)

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
            train_input = input[train_idx, :]
            fitted_model = model.fit(train_input, output[train_idx, :])

            # make prediction
            test_input = input[test_idx, :]
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

    def get_noise_mat(self, num_samples, ds_gender):

        noise = np.zeros((num_samples, self.num_features))

        if not self.add_noise:
            return noise

        # TODO: Support Gaussian noise
        return noise

    def fit_tvt(self, model, train_data, val_data, test_data):
        (train_input, train_noise), train_output = train_data
        (val_input, val_noise), val_output = val_data
        (test_input, test_noise), test_output = test_data

        fitted_model = model.fit(train_input, train_output)

        # predict and eval betas val / test
        if self.eval_val:
            logger.info('Reporting results on validation set.')
            predition = fitted_model.predict(val_input)
            val_output = self.validate(val_output, predition)
            self.print_result(val_output)

        if self.eval_test:
            logger.info('Reporting results on test set.')
            predition = fitted_model.predict(test_input)
            test_output = self.validate(test_output, predition)
            self.print_result(test_output)

        # visu model coefficients
        # create_coef_heatmap(fitted_model.coef_, feature_names,
        #    np.arange(num_betas), plot_name)

    def validate(self, gt_attr_scores, pred_attr_scores):

        val_output = defaultdict(lambda: [])

        for gt_attr_score, pred_attr_score in zip(
                gt_attr_scores, pred_attr_scores):

            # Reconstruction errors
            val_output['mse'] += [
                np.power(gt_attr_score - pred_attr_score, 2).mean()]

            val_output['classification_error'].append(
                np.equal(np.round(pred_attr_score), np.round(gt_attr_score)))
            # logger.info(np.round(pred_attr_score))
            # logger.info(np.round(gt_attr_score))

        return val_output

    def create_data(self, data, ds_gender, split='train'):
        beta_key = f'betas_{self.model_type}_{self.model_gender}'

        betas = data.db[split][beta_key][:, :self.num_shape_comps]

        num_samples = len(betas)
        # create ratings matrix
        if len(self.selected_attr_idx) == 0:
            feature_vec = np.empty((num_samples, 0))
        else:
            feature_vec = data.db[split]['ratings'][:, self.selected_attr_idx]

        noise = self.get_noise_mat(num_samples, ds_gender)

        return (betas, noise), feature_vec

    def get_loo_data(self, data):
        input = self.create_input_feature_vec(data.db, data.ds_gender)
        beta_key = f'betas_{self.model_type}_{self.model_gender}'
        output = data.db[beta_key][:, :self.num_shape_comps]
        return input, output

    def get_tvt_data(self, data):
        beta_key = f'betas_{self.model_type}_{self.model_gender}'
        self.rating_label = data.db['labels']

        train_data = self.create_data(data, data.ds_gender, split='train')
        val_data = self.create_data(data, data.ds_gender, split='val')
        test_data = self.create_data(data, data.ds_gender, split='test')
        return train_data, val_data, test_data

    def print_result(self, result_dict):

        row_format = "{:>15}" * 5
        logger.info(row_format.format('Measure', 'Mean', 'SD', 'MIN', 'MAX'))

        for k, v in result_dict.items():
            f = 100 if k == 'classification_error' else 1.0
            v = f * np.stack(v)
            mean = np.round(v.mean(), 2)
            sd = np.round(v.std(), 2)
            min = np.round(v.min(), 2)
            max = np.round(v.max(), 2)
            out = [k, mean, sd, min, max]
            logger.info(row_format.format(*out))

        classification_error = result_dict.get('classification_error')
        if classification_error is not None:
            classification_error = np.stack(classification_error, axis=0)
            f = 100
            for ii, label in enumerate(self.rating_label):
                v = f * classification_error[:, ii]
                mean = np.round(v.mean(), 2)
                sd = np.round(v.std(), 2)
                min = np.round(v.min(), 2)
                max = np.round(v.max(), 2)
                out = [label, mean, sd, min, max]
                logger.info(row_format.format(*out))

import sys

from typing import Dict, List, Union, Optional, Tuple
import os
import os.path as osp

from copy import deepcopy
from collections import defaultdict, OrderedDict

import time
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.utils import make_grid
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from loguru import logger

#  from .utils.metrics import (mpjpe, vertex_to_vertex_error,
#  ProcrustesAlignmentMPJPE,
#  PelvisAlignmentMPJPE,
#  RootAlignmentMPJPE)
from .data.structures import to_image_list, StructureList
from .data.utils import (
    KEYPOINT_NAMES_DICT, targets_to_array_and_indices,
    map_keypoints,
)
from .models.body_models import KeypointTensor

from .models import BODY_HEAD_REGISTRY
from .utils import (Tensor, Array,
                    FloatList,
                    PointError, build_alignment,
                    undo_img_normalization,
                    OverlayRenderer,
                    GTRenderer,
                    COLORS,
                    keyp_target_to_image, create_skel_img,
                    v2vhdError
                    )


def build(exp_cfg, distributed=False, rank=0):
    return Evaluator(exp_cfg, rank=rank, distributed=distributed)


class Evaluator(object):
    def __init__(self, exp_cfg, rank=0, distributed=False):
        super(Evaluator, self).__init__()
        self.rank = rank
        self.distributed = distributed

        self.imgs_per_row = exp_cfg.get('imgs_per_row', 2)
        self.exp_cfg = deepcopy(exp_cfg)
        self.output_folder = osp.expandvars(exp_cfg.output_folder)

        self.summary_folder = osp.join(
            self.output_folder, exp_cfg.summary_folder)
        os.makedirs(self.summary_folder, exist_ok=True)
        self.summary_steps = exp_cfg.summary_steps

        self.results_folder = osp.join(
            self.output_folder, exp_cfg.results_folder)
        os.makedirs(self.results_folder, exist_ok=True)

        self.means = np.array(self.exp_cfg.datasets.pose.transforms.mean)
        self.std = np.array(self.exp_cfg.datasets.pose.transforms.std)

        self.degrees = exp_cfg.get('degrees', tuple())
        crop_size = exp_cfg.get('datasets', {}).get('pose', {}).get(
            'crop_size', 256)
        self.renderer = OverlayRenderer(img_size=crop_size)
        self.render_gt_meshes = exp_cfg.get('render_gt_meshes', True)
        if self.render_gt_meshes:
            self.gt_renderer = GTRenderer(img_size=crop_size)

        self.J14_regressor = None
        self.metrics = self.build_metric_utilities(exp_cfg, part_key='body')

    @torch.no_grad()
    def __enter__(self):
        if self.rank == 0:
            self.filewriter = SummaryWriter(self.summary_folder, max_queue=1)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.rank == 0:
            self.filewriter.close()

    def _compute_mpjpe(
        self,
        model_output,
        targets: StructureList,
        metric_align_dicts: Dict,
        mpjpe_root_joints_names: Optional[Array] = None,
    ) -> Dict[str, Array]:
        # keypoint annotations.
        gt_joints_3d_indices = np.array(
            [ii for ii, t in enumerate(targets)
             if t.has_field('keypoints3d')], dtype=np.int)
        output = {}
        # Get the number of valid instances
        num_instances = len(gt_joints_3d_indices)
        if num_instances < 1:
            return output
        # Get the data from the output of the model
        est_joints = model_output.get('joints', None)
        joints_from_model_np = est_joints.detach().cpu().numpy()

        for t in targets:
            if not t.has_field('keypoints3d'):
                continue
            gt_source = t.get_field('keypoints3d').source
            break

        # Get the indices that map the estimated joints to the order of the
        # ground truth joints
        target_names = KEYPOINT_NAMES_DICT.get(gt_source)
        target_indices, source_indices, target_dim = map_keypoints(
            target_dataset=gt_source,
            source_dataset=est_joints.source,
            names_dict=KEYPOINT_NAMES_DICT,
            source_names=est_joints.keypoint_names,
            target_names=target_names,
        )

        # Create the final array
        est_joints_np = np.zeros(
            [num_instances, target_dim, 3], dtype=np.float32)
        # Map the estimated joints to the order used by the ground truth joints
        for ii in gt_joints_3d_indices:
            est_joints_np[ii, target_indices] = joints_from_model_np[
                ii, source_indices]

        # Stack all 3D joint tensors
        gt_joints3d = np.stack(
            [t.get_field('keypoints3d').as_array()
             for t in targets if t.has_field('keypoints3d')])
        for alignment_name, alignment in metric_align_dicts.items():
            # Update the root joint for the current dataset
            if hasattr(alignment, 'set_root'):
                root_indices = [
                    target_names.index(name)
                    for name in mpjpe_root_joints_names
                ]
                alignment.set_root(root_indices)
            metric_value = alignment(
                est_joints_np[gt_joints_3d_indices],
                gt_joints3d[:, :, :-1])
            name = f'{alignment_name}_mpjpe'
            output[name] = metric_value
        return output

    def _compute_mpjpe14(
        self,
        model_output,
        targets: StructureList,
        metric_align_dicts: Dict,
        J14_regressor: Array,
        **extra_args,
    ) -> Dict[str, Array]:
        output = {}
        gt_joints_3d_indices = np.array(
            [ii for ii, t in enumerate(targets)
             if t.has_field('joints14')], dtype=np.long)
        if len(gt_joints_3d_indices) < 1:
            return output
        # Stack all 3D joint tensors
        gt_joints3d = np.stack(
            [t.get_field('joints14').joints.detach().cpu().numpy()
             for t in targets if t.has_field('joints14')])

        # Get the data from the output of the model
        est_vertices = model_output.get('vertices', None)
        est_vertices_np = est_vertices.detach().cpu().numpy()
        est_joints_np = np.einsum(
            'jv,bvn->bjn', J14_regressor, est_vertices_np)
        for alignment_name, alignment in metric_align_dicts.items():
            metric_value = alignment(est_joints_np[gt_joints_3d_indices],
                                     gt_joints3d)
            name = f'{alignment_name}_mpjpe14'
            output[name] = metric_value
        return output

    def _compute_v2v(
        self,
        model_output,
        targets: StructureList,
        metric_align_dicts: Dict,
        vertex_key: str = 'vertices',
        metric_name: str = 'v2v',
        **extra_args,
    ) -> Dict[str, Array]:
        ''' Computes the Vertex-to-Vertex error for the current input
        '''
        output = {}
        # Ground truth vertices
        gt_verts_indices = np.array(
            [ii for ii, t in enumerate(targets)
             if t.has_field(vertex_key)], dtype=np.int)
        if len(gt_verts_indices) < 1:
            return output

        # Stack all vertices
        gt_vertices = np.stack(
            [t.get_field(vertex_key).vertices.detach().cpu().numpy()
             for t in targets if t.has_field(vertex_key)])

        # Get the data from the output of the model
        est_vertices = model_output.get(vertex_key, None)
        est_vertices_np = est_vertices.detach().cpu().numpy()

        for alignment_name, alignment in metric_align_dicts.items():
            metric_value = alignment(est_vertices_np, gt_vertices)
            name = f'{alignment_name}_{metric_name}'
            output[name] = metric_value
        return output


    def _compute_p2p(
        self,
        model_output,
        targets: StructureList,
        metric: Dict,
        vertex_key: str = 'v_shaped',
        metric_name: str = 'p2p_t',
        **extra_args,
    ) -> Dict[str, Array]:
        ''' Computes the Vertex-to-Vertex error for the current input
        '''

        output = {}
        # Ground truth vertices
        gt_verts_indices = np.array(
            [ii for ii, t in enumerate(targets)
             if t.has_field(vertex_key)], dtype=np.int)
        if len(gt_verts_indices) < 1:
            return output

        # Stack all vertices
        gt_vertices = torch.stack(
            [t.get_field(vertex_key).vertices
             for t in targets if t.has_field(vertex_key)])

        # Get the data from the output of the model
        est_vertices = model_output.get(vertex_key, None)

        pin = est_vertices.cpu().double()
        pta = gt_vertices.cpu().double()
        v2vhd_diff, _ = metric(pin, pta)
        output[metric_name] = v2vhd_diff

        #v2vhds_diff.append(v2vhd_diff)
        #for alignment_name, alignment in metric_align_dicts.items():
        #    metric_value = alignment(est_vertices_np, gt_vertices)
        #    name = f'{alignment_name}_{metric_name}'
        #    output[name] = metric_value
        return output


    def _compute_measurement_error(
        self,
        model_output,
        targets: StructureList,
    ) -> Dict[str, Array]:

        est_measurements = model_output.get('measurements', {})
        if len(est_measurements) < 1:
            return {}

        measurement_errors = {}
        for meas_name, val in est_measurements.items():
            indices = []
            gt_values = []
            for ii, t in enumerate(targets):
                if t.has_field(meas_name):
                    meas_value = t.get_field(meas_name)
                    if meas_value > 0:
                        gt_values.append(t.get_field(meas_name))
                        indices.append(ii)

            if len(indices) < 1:
                continue
            gt_values = np.asarray(gt_values)

            est_values = val[indices]
            if torch.is_tensor(est_values):
                est_values = est_values.detach().cpu().numpy()
            measurement_errors[meas_name] = np.abs(gt_values - est_values)

        return measurement_errors

    '''def _compute_attributes_error(
        self
    ):
    '''

    def compute_metric(
        self,
        model_output,
        targets: StructureList,
        metrics: Dict,
        mpjpe_root_joints_names: Optional[Array] = None,
        **extra_args,
    ):
        
        output_metric_values = {}
        for metric_name, metric in metrics.items():
            if metric_name == 'mpjpe':
                curr_vals = self._compute_mpjpe(
                    model_output, targets, metric,
                    mpjpe_root_joints_names=mpjpe_root_joints_names)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'mpjpe14':
                curr_vals = self._compute_mpjpe14(
                    model_output, targets, metric, **extra_args)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'v2v':
                curr_vals = self._compute_v2v(
                    model_output, targets, metric,
                    vertex_key='vertices',
                    **extra_args)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'v2v_t':
                curr_vals = self._compute_v2v(
                    model_output, targets, metric,
                    metric_name='v2v_t',
                    vertex_key='v_shaped',
                    **extra_args)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'measurements':
                meas_errors = self._compute_measurement_error(
                    model_output, targets)
                for key, val in meas_errors.items():
                    output_metric_values[key] = val
            elif metric_name == 'p2p_t':
                curr_vals = self._compute_p2p(
                    model_output, targets, metric,
                    metric_name='p2p_t',
                    vertex_key='v_shaped',
                    **extra_args)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            else:
                raise ValueError(f'Unsupported metric: {metric_name}')
        return output_metric_values

    def _create_keypoint_images(
        self,
        images: Tensor,
        targets: StructureList,
        model_output: Dict,
        draw_text: bool = True
    ) -> Tuple[Array, Array]:
        gt_keyp_imgs = []
        est_keyp_imgs = []

        proj_joints = model_output.get('proj_joints', None)
        if proj_joints is not None:
            keypoint_names = proj_joints.keypoint_names
            est_connections = proj_joints._connections
            if torch.is_tensor(proj_joints):
                proj_joints = proj_joints.detach().cpu().numpy()
            elif isinstance(proj_joints, KeypointTensor,):
                proj_joints = proj_joints._t.detach().cpu().numpy()

        # Scale the predicted keypoints to image coordinates
        crop_size = images.shape[-1]
        proj_joints = (proj_joints + 1) * 0.5 * crop_size

        for ii, (img, target) in enumerate(zip(images, targets)):
            gt_keyp_imgs.append(keyp_target_to_image(
                img, target, draw_text=draw_text))
            if proj_joints is not None:
                est_keyp_imgs.append(create_skel_img(
                    img, proj_joints[ii],
                    draw_text=draw_text,
                    names=keypoint_names,
                    connections=est_connections,
                ))

        gt_keyp_imgs = np.transpose(np.stack(gt_keyp_imgs), [0, 3, 1, 2])
        est_keyp_imgs = np.transpose(np.stack(est_keyp_imgs), [0, 3, 1, 2])
        return gt_keyp_imgs, est_keyp_imgs

    def render_mesh_overlay(self, bg_imgs,
                            vertices, faces,
                            camera_scale,
                            camera_translation, genders=None,
                            flip=False, renderer=None,
                            degrees=None,
                            body_color=None,
                            scale_first: bool = False,
                            ):
        if degrees is None:
            degrees = []
        body_imgs = renderer(
            vertices,
            faces,
            camera_scale,
            camera_translation,
            bg_imgs=bg_imgs,
            genders=genders,
            body_color=body_color,
            scale_first=scale_first,
        )
        if flip:
            body_imgs = body_imgs[:, :, :, ::-1]

        out_imgs = [body_imgs]
        # Add the rendered meshes
        for deg in degrees:
            body_imgs = renderer(
                vertices, faces,
                camera_scale, camera_translation,
                bg_imgs=None,
                genders=genders,
                deg=deg,
                return_with_alpha=False,
                body_color=body_color,
                scale_first=scale_first,
            )
            body_imgs = body_imgs[:, :, :, ::-1]
            out_imgs.append(body_imgs)
        return np.concatenate(out_imgs, axis=-1)

    def create_image_summaries(
        self,
        step: int,
        dset_name: str,
        images: Tensor,
        targets: StructureList,
        model_output: Dict,
        degrees: Optional[FloatList] = None,
        renderer: Optional[OverlayRenderer] = None,
        gt_renderer: Optional[GTRenderer] = None,
        render_gt_meshes: bool = True,
        prefix: str = '',
        draw_text: bool = True,
        draw_keyps: bool = True,
    ) -> None:
        if not hasattr(self, 'filewriter'):
            return
        if degrees is None:
            degrees = []

        crop_size = images.shape[-1]
        images = np.stack([
            undo_img_normalization(img, self.means, self.std)
            for img in images])
        _, _, crop_size, _ = images.shape

        summary_imgs = OrderedDict()
        summary_imgs['rgb'] = images

        # Create the keypoint images
        if draw_keyps:
            gt_keyp_imgs, est_keyp_imgs = self._create_keypoint_images(
                images, targets, model_output=model_output,
                draw_text=draw_text)
            summary_imgs['gt_keypoint_images'] = gt_keyp_imgs
            summary_imgs['est_keypoint_images'] = est_keyp_imgs

        render_gt_meshes = (render_gt_meshes and self.render_gt_meshes and
                            any([t.has_field('vertices') for t in targets]))
        stage_keys = model_output.get('stage_keys', [])
        last_stage = stage_keys[-1]
        if render_gt_meshes:
            gt_mesh_imgs = []
            faces = model_output[last_stage]['faces']
            for bidx, t in enumerate(targets):
                if (not t.has_field('vertices') or
                        not t.has_field('intrinsics')):
                    gt_mesh_imgs.append(np.zeros_like(images[bidx]))
                    continue

                curr_gt_vertices = t.get_field(
                    'vertices').vertices.detach().cpu().numpy().squeeze()
                intrinsics = t.get_field('intrinsics')

                mesh_img = gt_renderer(
                    curr_gt_vertices[np.newaxis], faces=faces,
                    intrinsics=intrinsics[np.newaxis],
                    bg_imgs=images[[bidx]])
                gt_mesh_imgs.append(mesh_img.squeeze())

            gt_mesh_imgs = np.stack(gt_mesh_imgs)
            B, C, H, W = gt_mesh_imgs.shape
            row_pad = (crop_size - H) // 2
            gt_mesh_imgs = np.pad(
                gt_mesh_imgs,
                [[0, 0], [0, 0], [row_pad, row_pad], [row_pad, row_pad]])
            summary_imgs['gt_meshes'] = gt_mesh_imgs

        camera_params = model_output.get('camera_parameters', None)
        scale = camera_params.scale
        translation = camera_params.translation
        scale_first = camera_params.scale_first

        for stage_key in stage_keys:
            if stage_key not in model_output:
                continue
            curr_stage_output = model_output.get(stage_key)
            vertices = curr_stage_output.get('vertices', None)
            if vertices is None:
                continue
            vertices = vertices.detach().cpu().numpy()
            faces = curr_stage_output['faces']

            body_color = COLORS.get(stage_key, COLORS['default'])
            overlays = self.render_mesh_overlay(
                images,
                vertices, faces,
                scale, translation,
                degrees=degrees if stage_key == stage_keys[-1] else None,
                renderer=renderer,
                body_color=body_color,
                scale_first=scale_first,
            )
            summary_imgs[f'overlays_{stage_key}'] = overlays

        albedo_images = model_output.get('albedo_images', None)
        if albedo_images is not None:
            summary_imgs['albedo'] = albedo_images.detach(
            ).cpu().numpy()
            #  self.filewriter.add_image(
            #  f'{prefix}/AlbedoImages', albedo_images_grid, step)

        normal_images = model_output.get('normal_images', None)
        if normal_images is not None:
            normal_images = (normal_images / normal_images.norm(
                dim=1, keepdim=True) + 1) * 0.5
            summary_imgs['normals'] = normal_images.detach(
            ).cpu().numpy()
            #  normal_images_grid = make_grid(
            #  normal_images, nrow=self.imgs_per_row)
            #  self.filewriter.add_image(
            #  f'{prefix}/NormalImages', normal_images_grid, step)

        predicted_images = model_output.get('predicted_images', None)
        if predicted_images is not None:
            predicted_images = predicted_images.detach().cpu().numpy()
            summary_imgs['predicted'] = predicted_images
            #  predicted_images = np.concatenate([
            #  images, predicted_images], axis=-1)
            #  predicted_images_grid = make_grid(
            #  torch.from_numpy(predicted_images), nrow=self.imgs_per_row)
            #  self.filewriter.add_image(
            #  f'{prefix}/PredictedImages', predicted_images_grid, step)

        summary_imgs = np.concatenate(
            list(summary_imgs.values()), axis=3)
        img_grid = make_grid(
            torch.from_numpy(summary_imgs), nrow=self.imgs_per_row)
        img_tab_name = (f'{dset_name}/{prefix}/Images' if len(prefix) > 0 else
                        f'{dset_name}/Images')
        self.filewriter.add_image(img_tab_name, img_grid, step)
        return

    def build_metric_utilities(self, exp_cfg, part_key):
        eval_cfg = exp_cfg.get('evaluation', {}).get(part_key, {})
        fscores_thresh = eval_cfg.get('fscores_thresh', None)
        v2v_cfg = eval_cfg.get('v2v', {})
        # Vertex-to-Vertex at T-Pose
        v2v_t_cfg = eval_cfg.get('v2v_t', {})

        p2p_t_cfg = eval_cfg.get('p2p_t', {})

        mpjpe_cfg = eval_cfg.get('mpjpe', {})
        mpjpe_alignments = mpjpe_cfg.get('alignments', [])
        mpjpe_root_joints_names = mpjpe_cfg.get('root_joints', [])

        model_name = exp_cfg.get(f'{part_key}_model', {}).get('type', 'smplx')
        keypoint_names = KEYPOINT_NAMES_DICT[model_name]
        #  self.mpjpe_root_joints_names = mpjpe_root_joints_names
        if not hasattr(self, 'mpjpe_root_joints_names'):
            self.mpjpe_root_joints_names = {}
        self.mpjpe_root_joints_names[part_key] = mpjpe_root_joints_names

        mpjpe_root_joints = [
            keypoint_names.index(name)
            for name in mpjpe_root_joints_names]

        v2v = {
            name: PointError(build_alignment(name))
            for name in v2v_cfg
        }
        v2v_t = {
            name: PointError(build_alignment(name))
            for name in v2v_t_cfg
        }
        mpjpe = {
            name: PointError(build_alignment(name, root=mpjpe_root_joints))
            for name in mpjpe_alignments
        }

        v2vhd = v2vhdError(**p2p_t_cfg)

        metrics = {'v2v': v2v, 'mpjpe': mpjpe,
                   'v2v_t': v2v_t,
                   'measurements': None,
                   'p2p_t': v2vhd}

        if part_key == 'body':
            mpjpe14 = {
                name: PointError(build_alignment(name, root=[2, 3]))
                for name in mpjpe_alignments
            }
            metrics['mpjpe14'] = mpjpe14
            self.J14_regressor = None
            j14_regressor_path = osp.expandvars(exp_cfg.j14_regressor_path)
            if osp.exists(osp.expandvars(exp_cfg.j14_regressor_path)):
                if j14_regressor_path.endswith('pkl'):
                    with open(j14_regressor_path, 'rb') as f:
                        J14_regressor = pickle.load(f, encoding='latin1')
                elif j14_regressor_path.endswith('npy'):
                    J14_regressor = np.load(j14_regressor_path)
                else:
                    _, ext = osp.splitext(j14_regressor_path)
                    logger.error(
                        f'Unknown extension {ext} for path:'
                        f' {j14_regressor_path}')

                self.J14_regressor = J14_regressor[:14]

        return metrics

    @torch.no_grad()
    def run(self, model, dataloaders, exp_cfg, device, step=0):
        if self.rank > 0:
            return
        model.eval()
        assert not (model.training), 'Model is in training mode!'

        # Copy the model to avoid deadlocks and convert to float
        if self.distributed:
            eval_model = deepcopy(model.module).float()
        else:
            eval_model = deepcopy(model).float()
        eval_model.eval()
        assert not (eval_model.training), 'Model is in training mode!'

        #  part_dataloader = dataloaders.get(self.part_key)
        for dataloader in (dataloaders['pose'] + dataloaders['shape']):

            dset = dataloader.dataset

            dset_name = dset.name()
            dset_metrics = dset.metrics
            if len(dset_metrics) < 1:
                continue
                
            metric_values = defaultdict(lambda: [])
            desc = f'Evaluating dataset: {dset_name}'
            logger.info(f'Starting evaluation for: {dset_name}')

            # save results for BMI groups
            metric_histograms = {}
            plot_dict = {}
            bins = np.array([20,25,30,35,40])
            bins_names = ['<20', '20-25', '25-30', '30-35', '35-40', '>40']

            save_v_shaped = []
            save_labels = []

            for ii, batch in enumerate(
                    tqdm(dataloader, desc=desc,
                         leave=False, dynamic_ncols=True)):

                _, images, targets = batch

                # Transfer to the device
                images = images.to(device=device)
                targets = [target.to(device) for target in targets]

                model_output = eval_model(images, targets, device=device)
                num_stages = model_output.get('num_stages', 1)
                stage_n_out = model_output.get(
                    f'stage_{num_stages - 1:02d}', {})
                save_v_shaped.append(stage_n_out['v_shaped'].cpu().numpy())
                save_label = '/'.join(targets[0].get_field('filename').split('/')[-4:])
                save_labels.append(save_label)
                if ii == 0:
                    self.create_image_summaries(
                        step, dset_name,
                        images,
                        targets,
                        model_output,
                        degrees=self.degrees,
                        renderer=self.renderer,
                        gt_renderer=self.gt_renderer,
                    )
                curr_metrics = self.compute_metric(
                    stage_n_out, targets,
                    metrics={metric: self.metrics[metric]
                             for metric in dset_metrics},
                    J14_regressor=self.J14_regressor,
                    mpjpe_root_joints_names=self.mpjpe_root_joints_names[
                        'body']
                )
                
                for key, value in curr_metrics.items():
                    metric_values[key].append(value)

                    # add this to get validation errors for males and females
                    for jj, vv in enumerate(value):
                        if targets[jj].has_field('gender'):
                            gg = targets[jj].get_field('gender')
                            metric_values[f'{key}_{gg}'].append(np.array([vv]))

                # add error histogram based on BMI
                for key, value in curr_metrics.items():
                    for jj, vv in enumerate(value):
                        if targets[jj].has_field('height') and targets[jj].has_field('weight'):
                            height_m = targets[jj].get_field('height')
                            weight_kg = targets[jj].get_field('weight')
                            bmi = weight_kg / height_m**2
                            bmi_group = np.digitize(bmi, bins)
                            if key not in metric_histograms.keys():
                                metric_histograms[key] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
                            metric_histograms[key][bmi_group].append(vv)
            
            for metric_key, metric_val in metric_histograms.items():
                plot_dict[metric_key] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
                for bmi_group, bmi_vals in metric_val.items():
                    mean_metric = 0
                    metric_array = None
                    if len(bmi_vals) > 0:
                        metric_array = np.array(bmi_vals)
                        mean_metric = metric_array.mean() * 1000
                    plot_dict[metric_key][bmi_group] = mean_metric
                fig = plt.figure()
                plt.ylim([0, 300])
                plt.bar(
                    plot_dict[metric_key].keys(),
                    plot_dict[metric_key].values(),
                    tick_label = bins_names,

                )
                self.filewriter.add_figure(f'bmi histogram {metric_key}', fig, step)
                plt.close('all')

            for metric_name in metric_values:
                metric_array = np.concatenate(
                        metric_values[metric_name], axis=0)
                mean_metric_value = np.mean(metric_array)
                mean_metric_value *= 1000
                logger.info('[{:06d}] {}, {}: {:.4f} (mm)',
                            step, dset_name, metric_name,
                            mean_metric_value,
                            )
                summary_name = f'{dset_name}/{metric_name}'
                self.filewriter.add_scalar(
                    summary_name, mean_metric_value, step)

        return

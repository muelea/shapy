import sys
import os
import os.path as osp

from typing import Dict, List, Union, Optional, Tuple
import time
import numpy as np
from loguru import logger
from copy import deepcopy
from collections import OrderedDict

import yaml
import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from .data.structures import (
    to_image_list, ImageList, ImageListPacked, StructureList)
from .optimizers import build_optimizer, build_scheduler
from .evaluation import build as build_evaluator
from .models.body_models import KeypointTensor
from .models import BODY_HEAD_REGISTRY


from human_shape.utils import (Tensor, Array,
                               Timer,
                               tensor_scalar_dict_to_float,
                               FloatList,
                               DataLoader,
                               cfg_to_dict,
                               undo_img_normalization,
                               keyp_target_to_image,
                               create_skel_img,
                               OverlayRenderer,
                               create_bbox_img,
                               GTRenderer,
                               COLORS,
                               )


def build_trainer(exp_cfg, distributed=False, rank=0):
    evaluator = build_evaluator(exp_cfg, rank=rank, distributed=distributed)
    return Trainer(
        exp_cfg, evaluator, rank=rank, distributed=distributed)


class Trainer(object):

    def __init__(self, exp_cfg, evaluator, distributed=False, rank=0):
        super(Trainer, self).__init__()

        self.rank = rank
        self.distributed = distributed

        self.exp_cfg = deepcopy(exp_cfg)
        self.max_duration = exp_cfg.max_duration
        self.max_iters = exp_cfg.max_iters
        self.checkpoint_steps = exp_cfg.checkpoint_steps
        self.use_half_precision = exp_cfg.use_half_precision
        logger.warning(f'Mixed precision training: {self.use_half_precision}')

        self.alpha_blend = exp_cfg.get('alpha_blend', 0.7)

        self.output_folder = osp.expandvars(exp_cfg.output_folder)

        if rank == 0:
            exp_conf_fn = osp.join(self.output_folder, 'conf.yaml')
            with open(exp_conf_fn, 'w') as conf_file:
                exp_cfg_dict = cfg_to_dict(exp_cfg)
                yaml.safe_dump(exp_cfg_dict, conf_file)

        self.summary_folder = osp.join(
            self.output_folder, exp_cfg.summary_folder)
        if rank == 0:
            os.makedirs(self.summary_folder, exist_ok=True)
        self.summary_steps = exp_cfg.summary_steps
        self.img_summary_steps = exp_cfg.img_summary_steps
        self.hd_img_summary_steps = exp_cfg.hd_img_summary_steps
        self.imgs_per_row = exp_cfg.get('imgs_per_row', 1)

        self.eval_steps = exp_cfg.eval_steps
        self.evaluator = evaluator

        self.use_adv_training = exp_cfg.use_adv_training

        self.means = np.array(self.exp_cfg.datasets.pose.transforms.mean)
        self.std = np.array(self.exp_cfg.datasets.pose.transforms.std)

        self.degrees = exp_cfg.get('degrees', tuple())
        crop_size = exp_cfg.get('datasets', {}).get('pose', {}).get(
            'crop_size', 256)
        self.renderer = OverlayRenderer(img_size=crop_size)
        self.render_gt_meshes = exp_cfg.get('render_gt_meshes', True)
        if self.render_gt_meshes:
            self.gt_renderer = GTRenderer(img_size=crop_size)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if hasattr(self, 'filewriter'):
            self.filewriter.close()

    def save_checkpoint(self, checkpointer, arguments=None) -> None:
        if self.rank == 0:
            step = arguments['iteration']
            logger.info(f'Saving checkpoint at step {step:06d}')
            ckpt_name = f'model_{step:07d}'
            checkpointer.save_checkpoint(ckpt_name, **arguments)
        if self.distributed:
            dist.barrier()
        return

    def create_histogram(
        self,
        key: str,
        value: Optional[Tensor] = None,
        prefix: str = '',
        step: int = 0
    ) -> None:
        if value is None:
            return
        if torch.is_tensor(value):
            value = value.detach()
            summary_key = f'{prefix}/{key}' if prefix else f'{key}'
            self.filewriter.add_histogram(summary_key, value, step)

    def check_timeout(
        self,
        checkpointer,
        arguments=None,
        device=None,
    ) -> bool:
        ''' Check if the maximum duration has elapsed
        '''
        max_duration_reached = False
        timeout_tensor = torch.tensor(0.0, dtype=torch.float, device=device)

        timeout = time.time() - self.start_time > self.max_duration
        if self.distributed:
            timeout_tensor += float(timeout)
            dist.all_reduce(timeout_tensor)
        else:
            timeout_tensor = torch.tensor(timeout)

        if timeout_tensor.item() > 0:
            logger.warning(
                f'[{self.rank}]: Running time exceeded, exiting!')
            if self.distributed:
                #  # Sync all jobs
                dist.barrier()
            self.save_checkpoint(checkpointer, arguments)
            max_duration_reached = True
            if self.distributed:
                # Sync all jobs
                dist.barrier()

        return max_duration_reached

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

    @torch.no_grad()
    def create_summaries(
        self,
        step: int,
        targets: StructureList,
        model_output: Dict,
        lr: Optional[Union[float, Tensor]] = None,
        extra_output=None,
        prefix: str = '',
        **kwargs,
    ) -> None:
        ''' Build scalar and histogram summaries
        '''
        # Only rank zero should save summaries
        if (
            self.rank > 0 or not (step % self.summary_steps == 0) or
                model_output is None or not hasattr(self, 'filewriter')
        ):
            return

        if lr is not None:
            self.filewriter.add_scalar('LearningRate', deepcopy(lr), step)

        if self.use_adv_training:
            for key, val in extra_output.items():
                if val is None:
                    continue
                self.filewriter.add_histogram(key, val.detach(), step)
            if 'wasserstein_distance' in extra_output:
                self.filewriter.add_scalar(
                    'WassersteinDistance',
                    extra_output['wasserstein_distance'], step)

        camera_params = model_output.get('camera_parameters', None)
        if camera_params is not None:
            # Add the camera parameters to the histograms
            camera_scale = camera_params['scale'].detach()
            if len(camera_scale) > 0:
                self.filewriter.add_histogram(
                    f'{prefix}/Camera/Scale' if prefix else 'Camera/Scale',
                    camera_scale.cpu(), step)

            camera_translation = camera_params['translation'].detach()
            self.filewriter.add_histogram(
                f'{prefix}/Camera/TranslationX' if prefix else
                'Camera/TranslationX',
                camera_translation.cpu()[:, 0], step)
            self.filewriter.add_histogram(
                f'{prefix}/Camera/TranslationY' if prefix else
                'Camera/TranslationY',
                camera_translation.cpu()[:, 1], step)

        stage_keys = model_output.get('stage_keys', [])
        for stage_key in stage_keys:
            if stage_key not in model_output:
                continue
            for key, val in model_output[stage_key].items():
                if ('vertices' in key or key == 'v_shaped' or
                        'faces' in key or 'joints' in key or
                        key == 'proj_joints'):
                    continue
                self.create_histogram(key, val, prefix=prefix, step=step)

    def run(self, model, data_loader, val_loaders, device, exp_cfg,
            checkpointer,
            discriminator=None,
            discriminator_loss=None,
            ):
        model.train(True)
        if self.distributed:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.rank], output_device=self.rank,
                find_unused_parameters=False)
            if discriminator is not None:
                discriminator = nn.parallel.DistributedDataParallel(
                    discriminator, device_ids=[self.rank],
                    output_device=self.rank,
                    find_unused_parameters=True)

        logger.debug('Building optimizer...')
        optimizer = build_optimizer(
            model, exp_cfg.optim, exclude='discriminator')
        logger.debug('Building scheduler...')
        scheduler = build_scheduler(optimizer, exp_cfg.optim.scheduler)

        adv_optimizer = None
        if self.use_adv_training:
            assert discriminator is not None
            # Make sure that the model has a discriminator module
            #  assert hasattr(model, 'discriminator')
            logger.info('Building optimizer for discriminator...')
            adv_optimizer = build_optimizer(
                discriminator, exp_cfg.optim.discriminator)
            checkpointer.adv_optimizer = adv_optimizer

        checkpointer.optimizer = optimizer
        checkpointer.scheduler = scheduler

        arguments = {'iteration': 0, 'epoch_number': 0}
        extra_checkpoint_data = checkpointer.load_checkpoint()

        # Sync all distributed jobs
        if self.distributed:
            dist.barrier()

        for key in arguments:
            if key in extra_checkpoint_data:
                arguments[key] = extra_checkpoint_data[key]

        if not hasattr(self, 'filewriter') and self.rank == 0:
            self.filewriter = SummaryWriter(
                self.summary_folder, max_queue=1,
                purge_step=arguments['iteration'] + 1)

        extra_checkpoint_data.clear()
        start_iter = arguments.get('iteration')
        start_epoch = arguments.get('epoch_number')

        logger.info(f'Start epoch, iteration: {start_epoch}, {start_iter}')

        model.train(True)
        if self.distributed:
            dist.barrier()
        self.start_time = time.time()

        scaler = None
        if self.use_half_precision:
            # Create the gradient scaler
            scaler = torch.cuda.amp.GradScaler()

        max_duration_reached = False
        num_epochs = exp_cfg.optim.num_epochs
        with tqdm.tqdm(
                range(start_epoch, num_epochs),
                dynamic_ncols=True) as pbar:
            for epoch_number in pbar:
                pbar.set_description('Epoch: {:04d}'.format(epoch_number))
                arguments['epoch_number'] = epoch_number
                max_duration_reached = self.train_one_epoch(
                    model, data_loader,
                    val_loaders, optimizer, scheduler,
                    device, exp_cfg, checkpointer, arguments,
                    adv_optimizer=adv_optimizer,
                    discriminator=discriminator,
                    discriminator_loss=discriminator_loss,
                    scaler=scaler,
                )
                if max_duration_reached:
                    return max_duration_reached

                if scheduler is not None:
                    scheduler.step()

                if arguments['iteration'] >= self.max_iters:
                    logger.warning(
                        'Max iterations reached at: {:06d}, exiting!',
                        arguments['iteration'])
                    if self.distributed:
                        dist.barrier()
                    return False
        return max_duration_reached

    def render_mesh_overlay(self, bg_imgs,
                            vertices,
                            faces,
                            camera_scale,
                            camera_translation, genders=None,
                            gt_faces=None,
                            flip=False, renderer=None,
                            degrees=None,
                            body_color=None,
                            scale_first=False,
                            ):
        if degrees is None:
            degrees = []
        if gt_faces is None:
            gt_faces = faces
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

    @torch.no_grad()
    def create_image_summaries(
        self,
        step: int,
        images: Tensor,
        targets: StructureList,
        model_output: Dict,
        degrees: Optional[FloatList] = None,
        lr: Optional[Tensor] = None,
        extra_output=None,
        renderer: Optional[OverlayRenderer] = None,
        gt_renderer: Optional[GTRenderer] = None,
        render_gt_meshes: bool = True,
        keyp_idxs=None,
        prefix: str = '',
        draw_text: bool = True,
        draw_keyps: bool = True,
    ):

        if not (step % self.img_summary_steps == 0):
            return

        images = np.stack(
            [undo_img_normalization(img, self.means, self.std)
             for img in images])
        _, _, crop_size, _ = images.shape

        stage_keys = model_output.get('stage_keys', [])
        summary_imgs = OrderedDict(rgb=images)
        if draw_keyps:
            gt_keyp_imgs, est_keyp_imgs = self._create_keypoint_images(
                images, targets, model_output=model_output,
                draw_text=draw_text)
            summary_imgs['gt_keypoint_images'] = gt_keyp_imgs
            summary_imgs['est_keypoint_images'] = est_keyp_imgs

        render_gt_meshes = (render_gt_meshes and self.render_gt_meshes and
                            any([t.has_field('vertices') for t in targets]))

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
                curr_stage_output.get('render_faces', None),
                degrees=degrees if stage_key == stage_keys[-1] else None,
                renderer=renderer,
                body_color=body_color,
                scale_first=scale_first,
            )
            summary_imgs[f'overlays_{stage_key}'] = overlays

        albedo = model_output.get('albedo', None)
        if albedo is not None:
            albedo_grid = make_grid(albedo, nrow=self.imgs_per_row,
                                    padding=10, pad_value=1.0)
            self.filewriter.add_image(f'{prefix}/Albedo', albedo_grid, step)

        albedo_images = model_output.get('albedo_images', None)
        if albedo_images is not None:
            albedo_images_grid = make_grid(
                albedo_images, nrow=self.imgs_per_row)
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
            list(summary_imgs.values()), axis=3).astype(np.float32)
        img_grid = make_grid(
            torch.from_numpy(summary_imgs), nrow=self.imgs_per_row)
        self.filewriter.add_image(f'{prefix}/Images', img_grid, step)
        return

    def train_one_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        val_loaders: Dict[str, DataLoader],
        optimizer, scheduler,
        device: torch.device,
        exp_cfg,
        checkpointer,
        arguments: Dict[str, Union[int, float]],
        adv_optimizer=None,
        discriminator: Optional[nn.Module] = None,
        discriminator_loss: Optional[nn.Module] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> float:
        #  dloader = data_loader.get(self.part_key)

        pose_dloader = data_loader.get('pose')[0]
        shape_dloader = data_loader.get('shape')

        load_shape_data = len(shape_dloader) > 0
        if load_shape_data:
            shape_dloader = shape_dloader[0]

        num_steps = max(len(pose_dloader), len(shape_dloader))

        pose_dloader_iter = iter(pose_dloader)

        if load_shape_data:
            shape_dloader_iter = iter(shape_dloader)

        adv_loader_iter = None
        if self.use_adv_training:
            adv_loader_iter = iter(data_loader['adv_loader'])

        scaler = None
        if self.use_half_precision:
            # Create the gradient scaler
            scaler = torch.cuda.amp.GradScaler()

        max_duration_reached = False
        model.train(True)
        with tqdm.tqdm(range(num_steps)) as pbar:
            for _ in pbar:
                pbar.set_description(
                    f' Iteration: {arguments["iteration"]:07d}')
                pbar.update()
                arguments['iteration'] += 1

                try:
                    _, pose_imgs, pose_targets = next(pose_dloader_iter)
                except StopIteration:
                    pose_dloader_iter = iter(pose_dloader)
                    _, pose_imgs, pose_targets = next(pose_dloader_iter)

                pose_imgs = pose_imgs.to(device=device)
                pose_imgs.clamp(min=-3, max=3)
                pose_targets = [target.to(device) for target in pose_targets]
                for t in pose_targets:
                    t.add_field('is_shape', False)

                imgs = pose_imgs
                targets = pose_targets
                if load_shape_data:
                    try:
                        _, shape_imgs, shape_targets = next(shape_dloader_iter)
                    except StopIteration:
                        shape_dloader_iter = iter(shape_dloader)
                        _, shape_imgs, shape_targets = next(shape_dloader_iter)

                    shape_imgs = shape_imgs.to(device=device)
                    shape_imgs.clamp(min=-3, max=3)
                    shape_targets = [
                        target.to(device) for target in shape_targets]
                    for t in shape_targets:
                        t.add_field('is_shape', True)
                #  logger.info(
                    #  f'Pose={pose_imgs.shape}, shape={shape_imgs.shape}')

                    imgs = torch.cat([pose_imgs, shape_imgs], dim=0)
                    targets = pose_targets + shape_targets

                max_duration_reached = self.train_one_step(
                    model,
                    optimizer,
                    imgs, targets,
                    arguments=arguments,
                    device=device,
                    adv_loader_iter=adv_loader_iter,
                    adv_optimizer=adv_optimizer,
                    discriminator=discriminator,
                    discriminator_loss=discriminator_loss,
                    scaler=scaler,
                )

                if (arguments['iteration'] % self.checkpoint_steps == 0 and
                        arguments['iteration'] > 0):
                    self.save_checkpoint(checkpointer, arguments)

                if arguments['iteration'] % self.eval_steps == 0:
                    step = arguments['iteration']
                    model.eval()
                    if self.rank == 0:
                        with self.evaluator as evaluator:
                            evaluator.run(model, val_loaders, exp_cfg, device,
                                          step)
                    model.train(True)
                    if self.distributed:
                        dist.barrier()

                max_duration_reached = self.check_timeout(
                    checkpointer, arguments=arguments, device=device)
                if max_duration_reached:
                    break

                if arguments['iteration'] >= self.max_iters:
                    logger.warning(
                        'Max iterations reached at: {:06d}, exiting!',
                        arguments['iteration'])
                    if self.distributed:
                        dist.barrier()
                    return False

        return max_duration_reached

    def train_one_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        imgs: Tensor,
        targets: Optional[StructureList] = None,
        arguments: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        adv_loader_iter=None,
        discriminator: Optional[nn.Module] = None,
        discriminator_loss: Optional[nn.Module] = None,
        adv_optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        ''' Runs one training step for the part predictor
        '''
        if device is None:
            device = next(model.parameters()).device
        if self.use_half_precision:
            assert scaler is not None, (
                'Use half precision is true, but scaler is None'
            )

        optimizer.zero_grad()

        if self.use_half_precision:
            with torch.cuda.amp.autocast():
                model_output = model(imgs, targets=targets, device=device)
                loss_dict = model_output['losses']

                if self.use_adv_training:
                    body_poses = []
                    for n in range(self.num_stages):
                        stage_key = f'stage_{n:02d}'
                        body_poses.append(model_output[stage_key]['body_pose'])
                    body_poses = torch.cat(body_poses, dim=0)
                    curr_disc_loss, _ = discriminator_loss(
                        body_poses, update_gen=True)
                    loss_dict['regressor_disc_loss'] = curr_disc_loss
                losses_sum = sum(loss for loss in loss_dict.values())

            # Scales the loss and call backward
            scaler.scale(losses_sum).backward()

            # Perform one optimization step
            scaler.step(optimizer)

            # Update the scales for the next iteration
            scaler.update()
        else:
            model_output = model(imgs, targets=targets, device=device)
            loss_dict = model_output['losses']

            if self.use_adv_training:
                body_poses = []
                for n in range(self.num_stages):
                    stage_key = f'stage_{n:02d}'
                    body_poses.append(model_output[stage_key]['body_pose'])
                body_poses = torch.cat(body_poses, dim=0)
                curr_disc_loss, _ = discriminator_loss(
                    body_poses, update_gen=True)
                loss_dict['regressor_disc_loss'] = curr_disc_loss

            losses_sum = sum(loss for loss in loss_dict.values())
            losses_sum.backward()

            optimizer.step(lambda: losses_sum.item())

        # Check that all values that require grad, have a valid gradient field
        for name, value in model.named_parameters():
            if value.requires_grad:
                assert value.grad is not None, (
                    f'{name} as a grad attribute that is None')

        extra_output = None
        if self.use_adv_training:
            # Train the discriminator
            real_poses = next(adv_loader_iter)['real'].to(device=device)

            adv_optimizer.zero_grad()

            curr_disc_loss, extra_output = discriminator_loss(
                body_poses.detach(), real_poses, update_gen=False)
            curr_disc_loss.backward()
            loss_dict['discriminator_loss'] = curr_disc_loss

            adv_optimizer.step(lambda: curr_disc_loss.item())

        if arguments['iteration'] % self.summary_steps == 0:
            if self.rank == 0:
                self.filewriter.add_scalar(
                    'TotalLoss', losses_sum,
                    global_step=arguments['iteration'])

                step = arguments['iteration']
                tokens = []
                for key, val in loss_dict.items():
                    tokens.append(
                        f'{key.replace("_", " ").title()}: {val:.4f}')
                loss_str = ', '.join(tokens)
                logger.info(f'[{step:06d}]: {loss_str}')

                self.filewriter.add_scalars(
                    'Losses',
                    {key: val.detach() if torch.is_tensor(val) else val
                     for key, val in loss_dict.items()},
                    arguments['iteration'])

                self.create_summaries(arguments['iteration'],
                                      targets,
                                      model_output,
                                      lr=optimizer.param_groups[0]['lr'],
                                      extra_output=extra_output)
                self.create_image_summaries(
                    step, imgs, targets, model_output=model_output,
                    extra_output=extra_output,
                    renderer=self.renderer,
                    gt_renderer=self.gt_renderer,
                    degrees=self.degrees,
                )

            if self.distributed:
                dist.barrier()
        return

from typing import NewType, Union, Tuple

import numpy as np
import random

import time
from copy import deepcopy
from loguru import logger
import cv2

import PIL.Image as pil_img
import torch
import torchvision
from torchvision.transforms import functional as F

from ..structures import AbstractStructure
from human_shape.utils.transf_utils import crop, get_transform
from human_shape.utils import Timer


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        self.timers = {}
        for t in self.transforms:
            self.timers[str(t)] = Timer(str(t))

    def __call__(self, image, target, **kwargs):
        next_input = (image, target)
        for t in self.transforms:
            #  with self.timers[str(t)]:
            output = t(*next_input, **kwargs)
            next_input = output
        return next_input

    def __iter__(self):
        return iter(self.transforms)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ExtremeBodyCrop(object):
    def __init__(self, prob=0.0, torso_upper_body_prob=0.5):
        self.prob = prob
        self.torso_upper_body_prob = torso_upper_body_prob
        logger.info(
            f'ExtremeBodyCrop: {self.prob}, {self.torso_upper_body_prob}')

    def __str__(self) -> str:
        return f'ExtremeBodyCrop (p={self.prob:.2f})'

    def __call__(self, image, target, **kwargs):
        if self.prob == 0:
            return image, target
        crop = random.random() < self.prob
        if not crop:
            return image, target

        torso_crop = random.random() < self.torso_upper_body_prob
        # Choose between an upper body or a head crop
        if torso_crop:
            new_center, new_scale, new_bbox_size = target.torso_crop()
        else:
            new_center, new_scale, new_bbox_size = target.upper_body_crop()
        if new_center is None or new_scale is None or new_bbox_size is None:
            return image, target
        target.add_field('center', new_center)
        target.add_field('scale', new_scale)
        target.add_field('bbox_size', new_bbox_size)
        return image, target


class MotionBlur(object):
    def __init__(
        self,
        prob: float = 0.0,
        kernel_size_min: int = 7,
        kernel_size_max: int = 7,
    ) -> None:
        #  assert kernel_size >= 3 and kernel_size <= 30, (
            #  f'Kernel must be between 3 and 20, got {kernel_size}'
        #  )
        self.prob = np.clip(prob, 0.0, 1.0)
        self.kernel_size_min = kernel_size_min
        self.kernel_size_max = kernel_size_max
        logger.info(str(self))

    def __str__(self):
        ksize_str = f'[{self.kernel_size_min}, {self.kernel_size_max}]'
        return (
            f'MotionBlur(prob={self.prob}, kernel_size={ksize_str})')

    def __call__(self, image, target, **kwargs):
        if not (self.prob > 0):
            return image, target

        blur = random.random() < self.prob
        if not blur:
            return image, target

        kernel_size = np.random.randint(
            self.kernel_size_min, self.kernel_size_max)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)

        #  Sample the start and end x-coordinates
        xs, xe = np.random.randint(0, kernel_size, size=2)
        if xs == xe:
            indices = np.arange(0, kernel_size)
            ys, ye = np.random.choice(indices, 2, replace=False)
        else:
            ys, ye = np.random.randint(0, kernel_size, size=2)

        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
        kernel = kernel.astype(np.float32) / kernel.sum()
        blurred_img = cv2.filter2D(image, -1, kernel,
                                   borderType=cv2.BORDER_CONSTANT)

        xshift = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        xshift = np.tile(xshift[None, :], (kernel_size, 1))

        yshift = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        yshift = np.tile(yshift[:, None], (1, kernel_size))

        vector = np.array([(kernel * xshift).sum(),
                           (kernel * yshift).sum()
                           ], dtype=np.float32)

        # Shift the targets
        shifted_target = target.shift(-vector)

        return blurred_img, shifted_target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        logger.info(str(self))

    def __str__(self):
        return 'RandomHorizontalFlip({:.03f})'.format(self.prob)

    def _flip(self, img):
        if img is None:
            return None
        if 'numpy.ndarray' in str(type(img)):
            return np.ascontiguousarray(img[:, ::-1, :]).copy()
        else:
            return F.hflip(img)

    def __call__(self, image, target, force_flip=False, **kwargs):
        flip = random.random() < self.prob
        target.add_field('is_flipped', flip)
        if flip or force_flip:
            output_image = self._flip(image)
            flipped_target = target.transpose(0)

            _, W, _ = output_image.shape

            left_hand_bbox, right_hand_bbox = None, None
            if flipped_target.has_field('left_hand_bbox'):
                left_hand_bbox = flipped_target.get_field('left_hand_bbox')
            if target.has_field('right_hand_bbox'):
                right_hand_bbox = flipped_target.get_field('right_hand_bbox')
            if left_hand_bbox is not None:
                flipped_target.add_field('right_hand_bbox', left_hand_bbox)
            if right_hand_bbox is not None:
                flipped_target.add_field('left_hand_bbox', right_hand_bbox)

            width = target.size[1]
            center = target.get_field('center')
            TO_REMOVE = 1
            center[0] = width - center[0] - TO_REMOVE

            #  if target.has_field('keypoints_hd'):
                #  keypoints_hd = target.get_field('keypoints_hd')
                #  flipped_keypoints_hd = keypoints_hd.copy()
                #  flipped_keypoints_hd[:, 0] = (
                    #  width - flipped_keypoints_hd[:, 0] - TO_REMOVE)
                #  flipped_keypoints_hd = flipped_keypoints_hd[
                    #  target.flip_indices]
                #  flipped_target.add_field('keypoints_hd', flipped_keypoints_hd)

            # Update the center
            flipped_target.add_field('center', center)
            if target.has_field('orig_center'):
                orig_center = target.get_field('orig_center').copy()
                orig_center[0] = width - orig_center[0] - TO_REMOVE
                flipped_target.add_field('orig_center', orig_center)

            # Flip the mask
            if target.has_field('mask'):
                mask = target.get_field('mask')
                output_mask = self._flip(mask)
                flipped_target.add_field('mask', output_mask)

            if target.has_field('translation'):
                translation = target.get_field('translation')
                # Flip the translation vector
                new_translation = deepcopy(translation)
                #  new_translation[0] *= (-1)
                flipped_target.add_field('translation', new_translation)

            if target.has_field('intrinsics'):
                intrinsics = target.get_field('intrinsics')
                cam_center = intrinsics[:2, 2].copy()
                cam_center[0] = width - cam_center[0] - TO_REMOVE
                intrinsics[:2, 2] = cam_center
                flipped_target.add_field('intrinsics', intrinsics)
            # Expressions are not symmetric, so we remove them from the targets
            # when the image is flipped
            if flipped_target.has_field('expression'):
                flipped_target.delete_field('expression')

            return output_image, flipped_target
        else:
            return image, target


class BBoxCenterJitter(object):
    def __init__(
        self,
        factor=0.0,
        dist='uniform'
    ):
        super(BBoxCenterJitter, self).__init__()
        self.factor = factor
        self.dist = dist
        assert self.dist in ['normal', 'uniform'], (
            f'Distribution must be normal or uniform, not {self.dist}')

    def __str__(self):
        return f'BBoxCenterJitter({self.factor:0.2f})'

    def __call__(self, image, target, **kwargs):
        if self.factor <= 1e-3:
            return image, target

        bbox_size = target.get_field('bbox_size')

        jitter = bbox_size * self.factor

        if self.dist == 'normal':
            center_jitter = np.random.randn(2) * jitter
        elif self.dist == 'uniform':
            center_jitter = np.random.rand(2) * 2 * jitter - jitter

        center = target.get_field('center')
        H, W, _ = target.size
        new_center = center + center_jitter
        new_center[0] = np.clip(new_center[0], 0, W)
        new_center[1] = np.clip(new_center[1], 0, H)

        target.add_field('center', new_center)
        target.add_field('jitter', center_jitter)

        return image, target


class SimulateLowRes(object):
    def __init__(
        self,
        dist: str = 'categorical',
        factor: float = 1.0,
        cat_factors: Tuple[float] = (1.0,),
        factor_min: float = 1.0,
        factor_max: float = 1.0
    ) -> None:
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.dist = dist
        self.cat_factors = cat_factors
        assert dist in ['uniform', 'categorical']

    def __str__(self) -> str:
        if self.dist == 'uniform':
            dist_str = (
                f'{self.dist.title()}: [{self.factor_min}, {self.factor_max}]')
        else:
            dist_str = (
                f'{self.dist.title()}: [{self.cat_factors}]')
        return f'SimulateLowResolution({dist_str})'

    def _sample_low_res(
        self,
        image: Union[np.ndarray, pil_img.Image]
    ) -> np.ndarray:
        '''
        '''
        if self.dist == 'uniform':
            downsample = self.factor_min != self.factor_max
            if not downsample:
                return image
            factor = np.random.rand() * (
                self.factor_max - self.factor_min) + self.factor_min
        elif self.dist == 'categorical':
            if len(self.cat_factors) < 2:
                return image
            idx = np.random.randint(0, len(self.cat_factors))
            factor = self.cat_factors[idx]

        H, W, _ = image.shape
        downsampled_image = cv2.resize(
            image, (int(W // factor), int(H // factor)), cv2.INTER_NEAREST
        )
        resized_image = cv2.resize(
            downsampled_image, (W, H), cv2.INTER_LINEAR_EXACT)
        return resized_image

    def __call__(
        self,
        image: Union[np.ndarray, pil_img.Image],
        cropped_image: Union[np.ndarray, pil_img.Image],
        target: AbstractStructure,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, AbstractStructure]:
        '''
        '''
        if torch.is_tensor(cropped_image):
            raise NotImplementedError
        elif isinstance(cropped_image, (pil_img.Image, np.ndarray)):
            resized_image = self._sample_low_res(cropped_image)

        return image, resized_image, target


class ChannelNoise(object):
    def __init__(self, noise_scale=0.0):
        self.noise_scale = noise_scale

    def __str__(self):
        return 'ChannelNoise: {:.4f}'.format(self.noise_scale)

    def __call__(
        self,
        image: Union[np.ndarray, pil_img.Image],
        cropped_image: Union[np.ndarray, pil_img.Image],
        target: AbstractStructure,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, AbstractStructure]:
        '''
        '''
        if self.noise_scale > 0:
            # Each channel is multiplied with a number
            # in the area [1 - self.noise_scale,1 + self.noise_scale]
            pn = np.random.uniform(
                1 - self.noise_scale,
                1 + self.noise_scale, 3).astype(np.float32)
            if not isinstance(image, (np.ndarray, )) and image is not None:
                image = np.asarray(image)
            if not isinstance(cropped_image, (np.ndarray,)):
                cropped_image = np.asarray(cropped_image)

            # Add the noise
            output_image = image
            if image is not None:
                output_image = np.clip(
                    image * pn[np.newaxis, np.newaxis], 0, 1.0)
            output_cropped_image = np.clip(
                cropped_image * pn[np.newaxis, np.newaxis], 0, 1.0)

            return output_image, output_cropped_image, target
        else:
            return image, cropped_image, target


class RandomRotation(object):
    def __init__(self, is_train: bool = True,
                 rotation_factor: float = 0):
        self.is_train = is_train
        self.rotation_factor = rotation_factor

    def __str__(self):
        return f'RandomRotation(rotation_factor={self.rotation_factor})'

    def __repr__(self):
        msg = [
            f'Training: {self.is_training}',
            f'Rotation factor: {self.rotation_factor}'
        ]
        return '\n'.join(msg)

    def __call__(self, image, target, **kwargs):
        rot = 0.0
        if not self.is_train:
            return image, target
        if self.is_train:
            rot = min(2 * self.rotation_factor,
                      max(-2 * self.rotation_factor,
                          np.random.randn() * self.rotation_factor))
            if np.random.uniform() <= 0.6:
                rot = 0
        if rot == 0.0:
            return image, target

        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        rotated_image = cv2.warpAffine(image, M, (nW, nH))

        new_target = target.rotate(rot=rot)

        center = target.get_field('center')
        center = np.dot(M[:2, :2], center) + M[:2, 2]
        new_target.add_field('center', center)

        if target.has_field('mask'):
            mask = target.get_field('mask')
            rotated_mask = cv2.warpAffine(mask, M, (nW, nH))
            new_target.add_field('mask', rotated_mask)

        #  if target.has_field('keypoints_hd'):
            #  keypoints_hd = target.get_field('keypoints_hd')
            #  rotated_keyps = (
                #  np.dot(keypoints_hd[:, :2], M[:2, :2].T) + M[:2, 2] +
                #  1).astype(np.int)
            #  rotated_keyps = np.concatenate(
                #  [rotated_keyps, keypoints_hd[:, [2]]], axis=-1)
            #  new_target.add_field('keypoints_hd', rotated_keyps)

        if target.has_field('intrinsics'):
            intrinsics = target.get_field('intrinsics').copy()

            cam_center = intrinsics[:2, 2]
            intrinsics[:2, 2] = (
                np.dot(M[:2, :2], cam_center) + M[:2, 2])
            new_target.add_field('intrinsics', intrinsics)

        # Fix the translation
        if target.has_field('translation') and target.has_field('pelvis'):
            translation = target.get_field('translation')
            is_tensor = False
            if torch.is_tensor(translation):
                translation = translation.detach().cpu().numpy()
                is_tensor = True

            R = np.array([[np.cos(np.deg2rad(-rot)),
                           -np.sin(np.deg2rad(-rot)), 0],
                          [np.sin(np.deg2rad(-rot)),
                           np.cos(np.deg2rad(-rot)), 0],
                          [0, 0, 1]], dtype=np.float32)

            pelvis = target.get_field('pelvis')
            if torch.is_tensor(pelvis):
                pelvis = pelvis.detach().cpu().numpy()

            #  new_translation = np.dot(translation, R.T)
            new_translation = np.dot(R, translation + pelvis) - pelvis
            if is_tensor:
                new_translation = torch.tensor(
                    new_translation, dtype=torch.float32)

            new_target.add_field('translation', new_translation)

        return rotated_image, new_target


class Crop(object):
    def __init__(self, is_train=True,
                 crop_size=224,
                 scale_factor_min=0.00,
                 scale_factor_max=0.00,
                 scale_factor=0.0,
                 scale_dist='uniform',
                 rotation_factor=0,
                 min_hand_bbox_dim=20,
                 min_head_bbox_dim=20,
                 return_full_imgs=True,
                 ):
        super(Crop, self).__init__()
        self.crop_size = crop_size
        self.return_full_imgs = return_full_imgs

        self.is_train = is_train
        self.scale_factor_min = scale_factor_min
        self.scale_factor_max = scale_factor_max
        self.scale_factor = scale_factor
        self.scale_dist = scale_dist

        self.rotation_factor = rotation_factor
        self.min_hand_bbox_dim = min_hand_bbox_dim
        self.min_head_bbox_dim = min_head_bbox_dim

    def __str__(self):
        msg = [
            'Crop(',
            f'Training: {self.is_train}',
            f'Crop size: {self.crop_size}',
            f'Scale factor: {self.scale_factor}',
            f'Return full images: {self.return_full_imgs}',
            f')',
        ]
        return '\n'.join(msg)

    def __repr__(self):
        msg = [
            f'Training: {self.is_train}',
            f'Crop size: {self.crop_size}',
            f'Scale factor: {self.scale_factor}',
            f'Return full images: {self.return_full_imgs}',
        ]
        return '\n'.join(msg)

    def __call__(self, image, target, **kwargs):
        sc = 1.0
        if self.is_train:
            if self.scale_dist == 'normal':
                sc = min(1 + self.scale_factor,
                         max(1 - self.scale_factor,
                             np.random.randn() * self.scale_factor + 1))
            elif self.scale_dist == 'uniform':
                if self.scale_factor_max == 0.0 and self.scale_factor_min == 0:
                    sc = 1.0
                else:
                    sc = (np.random.rand() *
                          (self.scale_factor_max - self.scale_factor_min) +
                          self.scale_factor_min)

        scale = target.get_field('scale') * sc
        center = target.get_field('center')
        orig_bbox_size = target.get_field('bbox_size')
        bbox_size = orig_bbox_size * sc

        np_image = np.asarray(image)
        cropped_image = crop(
            np_image, center, scale, [self.crop_size, self.crop_size])
        cropped_target = target.crop(
            center, scale, crop_size=self.crop_size)

        transf = get_transform(
            center, scale, [self.crop_size, self.crop_size])

        if target.has_field('mask'):
            mask = target.get_field('mask')
            cropped_mask = crop(
                mask, center, scale, [self.crop_size, self.crop_size])
            cropped_mask = cropped_mask.reshape(
                cropped_mask.shape[0], cropped_mask.shape[1], 1)
            cropped_target.add_field('mask', cropped_mask)

        cropped_target.add_field('crop_transform', transf)
        cropped_target.add_field('bbox_size', bbox_size)

        if target.has_field('intrinsics'):
            intrinsics = target.get_field('intrinsics').copy()
            fscale = cropped_image.shape[0] / orig_bbox_size
            intrinsics[0, 0] *= (fscale / sc)
            intrinsics[1, 1] *= (fscale / sc)

            cam_center = intrinsics[:2, 2]
            intrinsics[:2, 2] = (
                np.dot(transf[:2, :2], cam_center) + transf[:2, 2])
            cropped_target.add_field('intrinsics', intrinsics)

        return (np_image if self.return_full_imgs else None,
                cropped_image, cropped_target)


class ColorJitter(object):
    def __init__(self, brightness=0.0, contrast=0, saturation=0, hue=0):
        super(ColorJitter, self).__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.transform = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue)

    def __repr__(self):
        msg = [
            'ColorJitter(',
            f'brightness={self.brightness:.2f}',
            f'contrast={self.contrast:.2f}',
            f'saturation={self.saturation:.2f}',
            f'hue={self.hue:.2f}',
            ')',
        ]
        return '\n'.join(msg)

    def __call__(self, image, target, **kwargs):
        return self.transform(image), target


class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()

    def __repr__(self):
        return 'ToTensor()'

    def __str__(self):
        return 'ToTensor()'

    def __call__(self, image, cropped_image, target, **kwargs):
        target.to_tensor()
        full_img = F.to_tensor(image) if image is not None else image

        if target.has_field('mask'):
            mask = target.get_field('mask')
            mask_tensor = torch.from_numpy(mask)

            mask_tensor = mask_tensor.permute(2, 0, 1)
            target.add_field('mask', mask_tensor)

        return full_img, F.to_tensor(cropped_image), target


class Resize(object):
    def __init__(self, max_size=1080):
        self.max_size = max_size

    def __str__(self):
        msg = [f'Resize(Maximum size: {self.max_size})']
        return '\n'.join(msg)

    def __repr__(self):
        msg = [f'Resize(Maximum size: {self.max_size})']
        return '\n'.join(msg)

    def __call__(self, image, target, **kwargs):
        if self.max_size < 0:
            return image, target

        H, W, _ = image.shape
        max_dim = max(H, W)
        # If the image size is already smaller than the max size
        if max_dim < self.max_size:
            return image, target

        ratio = self.max_size / max_dim
        dest_size = (int(ratio * H), int(ratio * W), 3)

        output_image = cv2.resize(image, (int(ratio * W), int(ratio * H)),
                                  interpolation=cv2.INTER_AREA)

        resized_target = target.resize(size=dest_size)

        if target.has_field('mask'):
            mask = target.get_field('mask')
            resized_mask = cv2.resize(mask, (int(ratio * W), int(ratio * H)),
                                      interpolation=cv2.INTER_AREA)
            resized_target.add_field('mask', resized_mask)

        center = target.get_field('center')
        bbox_size = target.get_field('bbox_size')
        scale = target.get_field('scale')

        resized_target.add_field('center', center * ratio)
        resized_target.add_field('bbox_size', bbox_size * ratio)
        resized_target.add_field('scale', scale * ratio)

        resized_target.add_field(
            'orig_center', target.get_field('orig_center') * ratio)
        resized_target.add_field(
            'orig_bbox_size', target.get_field('orig_bbox_size') * ratio)
        if target.has_field('intrinsics'):
            intrinsics = target.get_field('intrinsics').copy()
            intrinsics[0, 0] *= (ratio)
            intrinsics[1, 1] *= (ratio)

            intrinsics[:2, 2] *= ratio
            resized_target.add_field('intrinsics', intrinsics)

        return output_image, resized_target


class Normalize(object):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __str__(self):
        msg = [
            'Normalize(',
            f'Mean: {self.mean}',
            f'Standard deviation: {self.std}',
            ')',
        ]
        return '\n'.join(msg)

    def __repr__(self):
        msg = [
            'Normalize(',
            f'Mean: {self.mean}',
            f'Standard deviation: {self.std}',
            ')',
        ]
        return '\n'.join(msg)

    def __call__(self, image, cropped_image, target, **kwargs):
        output_image = image
        if image is not None:
            # Clamp the image to  [0, 1]
            if torch.is_tensor(image):
                image = torch.clamp(image, 0, 1)
            elif isinstance(image, (np.ndarray, np.array)):
                image = np.clip(image, 0, 1)
            output_image = F.normalize(
                image, mean=self.mean, std=self.std)

        # Clamp the cropped image to [0, 1]
        if torch.is_tensor(cropped_image):
            cropped_image = torch.clamp(cropped_image, 0, 1)
        elif isinstance(image, (np.ndarray, np.array)):
            cropped_image = np.clip(cropped_image, 0, 1)

        output_cropped_image = F.normalize(
            cropped_image, mean=self.mean, std=self.std)
        norm_target = target.normalize()
        # Store the normalization parameters
        norm_target.add_field('mean', self.mean)
        norm_target.add_field('std', self.std)
        return output_image, output_cropped_image, norm_target

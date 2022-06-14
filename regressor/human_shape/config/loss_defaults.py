from typing import Tuple
from copy import deepcopy
from loguru import logger
from dataclasses import dataclass, make_dataclass
from omegaconf import OmegaConf


@dataclass
class Normal:
    reduction: str = 'none'


@dataclass
class GenderShapePrior:
    female_stats_path: str = ''
    male_stats_path: str = ''
    prior_type: str = 'normal'
    female_normal: Normal = Normal()
    male_normal: Normal = Normal()


@dataclass
class Prior:
    type: str = 'l2'
    weight: float = 1.0
    margin: float = 1.0
    norm: str = 'l2'
    use_vector: bool = True
    barrier: str = 'log'
    epsilon: float = 1e-07
    path: str = ''
    num_gaussians: int = 8
    reduction: str = 'mean'
    gender_shape: GenderShapePrior = GenderShapePrior()


# TODO: Break down into parts
@dataclass
class Loss:
    type: str = 'l2'
    weight: float = 0.0
    robustifier: str = 'none'
    norm_type: str = 'l1'
    rho: float = 100.0
    beta: float = 5.0 / 100 * 2
    size_average: bool = True
    enable: int = 0
    normalize: str = 'none'
    use_conf_weight: bool = False
    division: str = 'batch'


@dataclass
class EdgeLoss(Loss):
    gt_edge_path: str = ''
    est_edge_path: str = ''


@dataclass
class IdentityLoss:
    weight: float = 0.0
    checkpoint: str = ''


@dataclass
class LossWithPrior(Loss):
    prior: Prior = Prior()


@dataclass
class CouplingLoss(Loss):
    key: str = 'v_shaped'


@dataclass
class MultiStageLosses:
    stages_to_penalize: Tuple[str] = ('stage_00',)
    stages_to_regularize: Tuple[str] = ('stage_00',)


@dataclass
class BodyLossConfig(MultiStageLosses):
    body_joints_2d: Loss = Loss(type='keypoints', norm_type='l1')
    face_joints_2d: Loss = Loss(type='keypoints', norm_type='l1')
    left_hand_joints_2d: Loss = Loss(type='keypoints', norm_type='l1')
    right_hand_joints_2d: Loss = Loss(type='keypoints', norm_type='l1')

    body_joints_3d: Loss = Loss(type='keypoints', norm_type='l1')
    face_joints_3d: Loss = Loss(type='keypoints', norm_type='l1')
    left_hand_joints_3d: Loss = Loss(type='keypoints', norm_type='l1')
    right_hand_joints_3d: Loss = Loss(type='keypoints', norm_type='l1')

    body_edge_2d: EdgeLoss = EdgeLoss(type='keypoint-edge', norm_type='l1')
    face_edge_2d: EdgeLoss = EdgeLoss(type='keypoint-edge', norm_type='l1')
    left_hand_edge_2d: EdgeLoss = EdgeLoss(
        type='keypoint-edge', norm_type='l1')
    right_hand_edge_2d: EdgeLoss = EdgeLoss(
        type='keypoint-edge', norm_type='l1')

    body_edge_3d: EdgeLoss = EdgeLoss(type='keypoint-edge', norm_type='l1')
    face_edge_3d: EdgeLoss = EdgeLoss(type='keypoint-edge', norm_type='l1')
    left_hand_edge_3d: EdgeLoss = EdgeLoss(
        type='keypoint-edge', norm_type='l1')
    right_hand_edge_3d: EdgeLoss = EdgeLoss(
        type='keypoint-edge', norm_type='l1')

    shape: LossWithPrior = LossWithPrior()
    expression: LossWithPrior = LossWithPrior()
    global_rot: Loss = Loss(type='rotation')
    body_pose: LossWithPrior = LossWithPrior(type='rotation')
    left_hand_pose: LossWithPrior = LossWithPrior(type='rotation')
    right_hand_pose: LossWithPrior = LossWithPrior(type='rotation')
    jaw_pose: LossWithPrior = LossWithPrior(type='rotation')

    edge: EdgeLoss = EdgeLoss(type='vertex-edge')

    vertex: Loss = Loss(type='l2')

    mass: Loss = Loss(type='l2')
    height: Loss = Loss(type='l2')
    chest: Loss = Loss(type='l2')
    waist: Loss = Loss(type='l2')
    hips: Loss = Loss(type='l2')

    identity: Loss = Loss(type='l2')

    attributes: Loss = Loss(type='l2')

    beta_refined: Loss = Loss(type='l2')
    vertex_refined: Loss = Loss(type='l2')


@dataclass
class LossConfig:
    body: BodyLossConfig = BodyLossConfig()


conf = OmegaConf.structured(LossConfig)

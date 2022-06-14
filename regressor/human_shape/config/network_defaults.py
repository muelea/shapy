from typing import Tuple
from copy import deepcopy
from loguru import logger
from dataclasses import dataclass, make_dataclass, field
from omegaconf import OmegaConf


@dataclass
class LeakyReLU:
    negative_slope: float = 0.01


@dataclass
class ELU:
    alpha: float = 1.0


@dataclass
class PReLU:
    num_parameters: int = 1
    init: float = 0.25


@dataclass
class Activation:
    type: str = 'relu'
    inplace: bool = True

    leaky_relu: LeakyReLU = LeakyReLU()
    prelu: PReLU = PReLU()
    elu: ELU = ELU()


@dataclass
class BatchNorm:
    eps: float = 1e-05
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True


@dataclass
class GroupNorm:
    num_groups: int = 32
    eps: float = 1e-05
    affine: bool = True


@dataclass
class LayerNorm:
    eps: float = 1e-05
    elementwise_affine: bool = True


@dataclass
class Normalization:
    type: str = 'batch-norm'
    batch_norm: BatchNorm = BatchNorm()
    layer_norm = LayerNorm = LayerNorm()
    group_norm: GroupNorm = GroupNorm()


@dataclass
class WeakPerspective:
    regress_scale: bool = True
    regress_translation: bool = True
    mean_scale: float = 0.9
    scale_first: bool = False


@dataclass
class Perspective:
    regress_translation: bool = False
    regress_rotation: bool = False
    regress_focal_length: bool = False
    focal_length: float = 5000.0


@dataclass
class Camera:
    type: str = 'weak-persp'
    pos_func: str = 'softplus'
    weak_persp: WeakPerspective = WeakPerspective()
    perspective: Perspective = Perspective()


@dataclass
class ResNet:
    replace_stride_with_dilation: Tuple[bool] = (False, False, False)


@dataclass
class HRNet:
    @dataclass
    class Stage:
        num_modules: int = 1
        num_branches: int = 1
        num_blocks: Tuple[int] = (4,)
        num_channels: Tuple[int] = (64,)
        block: str = 'BOTTLENECK'
        fuse_method: str = 'SUM'

    @dataclass
    class SubSample:
        num_layers: int = 3
        num_filters: Tuple[int] = (512,) * num_layers
        kernel_size: int = 3
        norm_type: str = 'bn'
        activ_type: str = 'relu'
        dim: int = 2
        kernel_sizes = [kernel_size] * len(num_filters)
        stride: int = 2
        strides: Tuple[int] = (stride,) * len(num_filters)
        padding: int = 1

    use_old_impl: bool = False
    pretrained_layers: Tuple[str] = ('*',)
    pretrained_path: str = (
        '../data/hrnet_v2/hrnetv2_w48_imagenet_pretrained.pth'
    )
    stage1: Stage = Stage()
    stage2: Stage = Stage(num_branches=2, num_blocks=(4, 4),
                          num_channels=(48, 96), block='BASIC')
    stage3: Stage = Stage(num_modules=4, num_branches=3,
                          num_blocks=(4, 4, 4),
                          num_channels=(48, 96, 192),
                          block='BASIC')
    stage4: Stage = Stage(num_modules=3, num_branches=4,
                          num_blocks=(4, 4, 4, 4,),
                          num_channels=(48, 96, 192, 384),
                          block='BASIC',
                          )


@dataclass
class Backbone:
    type: str = 'resnet50'
    pretrained: bool = True

    resnet: ResNet = ResNet()
    hrnet: HRNet = HRNet()


@dataclass
class MLP:
    layers: Tuple[int] = (1024, 1024)
    activation: Activation = Activation()
    normalization: Normalization = Normalization()
    preactivated: bool = False
    dropout: float = 0.0
    init_type: str = 'xavier'
    gain: float = 0.01
    bias_init: float = 0.0


@dataclass
class FCN:
    layers: Tuple[int] = (1024, 1024)
    activation: Activation = Activation()
    normalization: Normalization = Normalization()
    preactivated: bool = False
    dropout: float = 0.0
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1


@dataclass
class SPIN:
    pass


@dataclass
class EFT:
    pass


@dataclass
class ExPose:
    pass


@dataclass
class HMRLike:
    type: str = 'mlp'
    feature_key: str = 'avg_pooling'
    append_params: bool = True
    num_stages: int = 3
    pose_last_stage: bool = True
    detach_mean: bool = False
    learn_mean: bool = False

    backbone: Backbone = Backbone(type='resnet50')
    camera: Camera = Camera()
    mlp: MLP = MLP()


@dataclass
class SMPL(HMRLike):
    compute_measurements: bool = True
    meas_definition_path: str = ''
    meas_vertices_path: str = ''

    use_b2a: bool = True
    b2a_males_checkpoint: str = ''
    b2a_females_checkpoint: str = ''

    use_a2b: bool = True
    num_attributes: int = 15
    a2b_males_checkpoint: str = ''
    a2b_females_checkpoint: str = ''

    groups: Tuple[str] = (
        (
            'betas',
            'global_rot',
            'body_pose',
            'camera'
        ),
    )
    joints_to_exclude: Tuple[str] = tuple()


@dataclass
class SMPLH(SMPL):
    predict_hands: bool = True
    groups: Tuple[str] = (
        (
            'betas',
            'global_rot',
            'body_pose',
            'left_hand_pose',
            'right_hand_pose',
            'camera',
        ),
    )


@dataclass
class SMPLX(SMPLH):
    predict_face: bool = True
    groups: Tuple[str] = (
        (
            'betas',
            'expression',
            'global_rot',
            'body_pose',
            'left_hand_pose',
            'right_hand_pose',
            'jaw_pose',
            'camera',
        ),
    )

@dataclass
class Network:
    type: str = 'expose'
    use_sync_bn: bool = True

    #  expose: ExPose = ExPose()
    hmr: HMRLike = HMRLike()
    smpl: SMPL = SMPL()
    smplh: SMPLH = SMPLH()
    smplx: SMPLX = SMPLX()
    expose: SMPLX = SMPLX()


conf = OmegaConf.structured(Network)

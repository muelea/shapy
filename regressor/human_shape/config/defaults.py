from typing import Tuple, Optional
from loguru import logger
from copy import deepcopy
from dataclasses import dataclass
from omegaconf import OmegaConf
from .network_defaults import conf as network_cfg, Network
from .optim_defaults import conf as optim_cfg, OptimConfig
from .loss_defaults import conf as loss_cfg, LossConfig
from .datasets_defaults import (
    pose_conf as pose_data_conf, PoseConfig,
    shape_conf as shape_data_conf, ShapeConfig,
)
from .body_model import (
    body_conf, BodyModel,
)
from .utils import FScores


@dataclass
class MPJPE:
    alignments: Tuple[str] = ('root', 'procrustes')
    root_joints: Tuple[str] = tuple()

@dataclass
class P2P_T:
    input_point_regressor_path: str = ''
    target_point_regressor_path: str = ''
    align: bool = True

@dataclass
class Metrics:
    v2v: Tuple[str] = ('procrustes', 'scale', 'translation')
    v2v_t: Tuple[str] = ('scale', 'translation')
    mpjpe: MPJPE = MPJPE()
    fscores_thresh: Optional[Tuple[float]] = (5.0 / 1000, 15.0 / 1000)
    p2p_t: P2P_T = P2P_T()

@dataclass
class Evaluation:
    body: Metrics = Metrics(
        mpjpe=MPJPE(
            root_joints=('left_hip', 'right_hip')),
        fscores_thresh=(10.0 / 1000, 20.0 / 1000,
                        50.0 / 1000,
                        75.0 / 1000,
                        100.0 / 1000,
                        )
    )



@dataclass
class Config:
    num_gpus: int = 1
    local_rank: int = 0
    use_cuda: bool = True
    is_training: bool = True
    logger_level: str = 'info'
    use_half_precision: bool = False

    output_folder: str = 'output'
    summary_folder: str = 'summaries'
    results_folder: str = 'results'
    code_folder: str = 'code'

    summary_steps: int = 100
    img_summary_steps: int = 100
    hd_img_summary_steps: int = 1000
    imgs_per_row: int = 2
    backend: str = 'nccl'

    part_key: str = 'pose'

    degrees: Tuple[float] = (90, 180, 270)

    j14_regressor_path: str = ''
    pretrained: str = ''

    use_adv_training: bool = False

    checkpoint_folder: str = 'checkpoints'
    checkpoint_steps: int = 1000

    eval_steps: int = 500

    float_dtype: str = 'float32'
    max_duration: float = float('inf')
    max_iters: float = float('inf')

    body_vertex_ids_path: str = ''

    network: Network = network_cfg
    optim: OptimConfig = optim_cfg

    body_model: BodyModel = body_conf

    @dataclass
    class Datasets:
        batch_size: int = 64
        pose_shape_ratio: float = 0.5
        use_equal_sampling: bool = True
        use_packed: bool = False
        pose: PoseConfig = pose_data_conf
        shape: ShapeConfig = shape_data_conf

    datasets: Datasets = Datasets()
    losses: LossConfig = LossConfig()

    evaluation: Evaluation = Evaluation()
    run_final_evaluation_on_validation_set: bool = False

conf = OmegaConf.structured(Config)

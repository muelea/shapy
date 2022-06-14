import sys
from typing import Tuple, Optional
from loguru import logger
from copy import deepcopy
from dataclasses import dataclass
from omegaconf import OmegaConf
from human_shape.utils.typing import StringTuple, FloatTuple


############################## DATASETS ##############################

@dataclass
class Sampler:
    ratio_2d: float = 0.5
    use_equal_sampling: bool = True
    importance_key: str = 'weight'
    balance_genders: bool = True


@dataclass
class Transforms:
    flip_prob: float = 0.0
    max_size: float = 1080
    downsample_dist: str = 'categorical'
    downsample_factor_min: float = 1.0
    downsample_factor_max: float = 1.0
    downsample_cat_factors: Tuple[float] = (1.0,)
    center_jitter_factor: float = 0.0
    center_jitter_dist: str = 'uniform'
    crop_size: int = 256
    scale_factor_min: float = 1.0
    scale_factor_max: float = 1.0
    scale_factor: float = 0.0
    scale_dist: str = 'uniform'
    noise_scale: float = 0.0
    rotation_factor: float = 0.0
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)
    brightness: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    contrast: float = 0.0
    extreme_crop_prob: float = 0.0
    torso_upper_body_prob: float = 0.5
    motion_blur_prob: float = 0.0
    motion_blur_kernel_size_min: int = 3
    motion_blur_kernel_size_max: int = 21


@dataclass
class NumWorkers:
    train: int = 8
    val: int = 2
    test: int = 2


@dataclass
class Splits:
    train: StringTuple = tuple()
    val: StringTuple = tuple()
    test: StringTuple = tuple()


@dataclass
class Dataset:
    data_folder: str = 'data/'
    metrics: StringTuple = ('mpjpe14',)


@dataclass
class DatasetWithKeypoints(Dataset):
    binarization = True
    body_thresh: float = 0.05
    hand_thresh: float = 0.2
    head_thresh: float = 0.3
    keyp_folder: str = 'keypoints'
    keyp_format: str = 'openpose25_v1'
    use_face_contour: bool = True


@dataclass
class ParameterOptions:
    return_params: bool = True
    return_shape: bool = False
    return_expression: bool = False
    return_full_pose: bool = False
    return_vertices: bool = False


@dataclass
class OpenPose(DatasetWithKeypoints):
    data_folder: str = 'data/openpose'
    img_folder: str = 'images'


@dataclass
class Tracks(OpenPose):
    data_folder: str = 'data/tracks'


@dataclass
class Human36m(DatasetWithKeypoints, ParameterOptions):
    data_folder = 'data/h36m'
    img_folder = 'images'
    annotations_fn: str = 'data/human36m/train.npz'


@dataclass
class Human36mX(Human36m):
    data_folder: str = 'data/h36mx'
    annotations_fn: str = 'data/human36mx/train.npz'


@dataclass
class EHF(DatasetWithKeypoints):
    data_folder: str = 'data/EHF'
    img_folder: str = 'images'
    alignments_folder: str = 'alignments'
    metrics: StringTuple = ('v2v',)


@dataclass
class COCO:
    ann_file: str = 'data/coco/train.json'
    root: str = 'data/coco/'
    remove_images_without_annotations: bool = True
    val_ann_file: str = 'data/coco/val.json'
    min_keypoints_per_subject: int = 8
    min_max_height: float = 60
    return_segm_masks: bool = False
    dset_scale_factor: float = 1.2
    param_folder: str = ''
    load_params: bool = True
    is_right: bool = True


@dataclass
class CuratedFits(DatasetWithKeypoints, ParameterOptions):
    data_folder: str = 'data/curated_fits'
    img_folder: str = 'images'
    metrics: StringTuple = ('v2v',)
    body_thresh: float = 0.1
    hand_thresh: float = 0.2
    face_thresh: float = 0.4
    min_hand_keypoints: int = 8
    min_head_keypoints: int = 8
    return_gender: bool = True


@dataclass
class WeightHeight(DatasetWithKeypoints):
    data_folder: str = 'data/weight_height'
    data_fname: str = 'weight_height.npz'
    img_folder: str = 'images'
    metrics: StringTuple = tuple()
    body_thresh: float = 0.1
    hand_thresh: float = 0.2
    face_thresh: float = 0.4
    min_hand_keypoints: int = 8
    min_head_keypoints: int = 8
    return_gender: bool = True


@dataclass
class PoseTrack(COCO):
    root: str = 'data/posetrack'
    metrics: StringTuple = ('keyp2d_error',)


@dataclass
class ThreeDPW(DatasetWithKeypoints):
    data_folder: str = 'data/3dpw'
    img_folder: str = ''
    param_folder: str = ''
    vertex_folder: str = ''
    seq_folder: str = 'sequenceFiles'
    metrics: StringTuple = ('mpjpe14', 'v2v')
    body_thresh: float = 0.05


@dataclass
class Agora(DatasetWithKeypoints):
    metrics: StringTuple = ('v2v',)
    train_image_folders: Tuple[str] = (
        'archviz_5_10', 'brushifyforest_5_15',
        'brushifygrasslands_5_15', 'construction_5_15',
        'flowers_5_15', 'hdri_50mm_5_10', 'multiview_2'
    )
    train_param_folder: str = 'data/agora/parameters/train'
    val_image_folders: Tuple[str] = (
        'archviz_5_10', 'brushifyforest_5_15',
        'brushifygrasslands_5_15', 'construction_5_15',
        'flowers_5_15', 'hdri_50mm_5_10', 'multiview_2'
    )
    val_param_folder: str = 'data/agora/parameters/val'

    test_image_folders: Tuple[str] = (
        'archviz_5_10', 'brushifyforest_5_15',
        'brushifygrasslands_5_15', 'construction_5_15',
        'flowers_5_15', 'hdri_50mm_5_10', 'multiview_2'
    )
    test_param_folder: str = 'data/agora/parameters/test'

    body_dset_factor: float = 1.2
    hand_dset_factor: float = 2.0
    head_dset_factor: float = 2.0

    use_high_res_only: bool = True
    occlusion_per: float = 90


@dataclass
class SPIN(DatasetWithKeypoints, ParameterOptions):
    img_folder: str = 'data/spin/images'
    vertex_folder: str = 'data/spin/vertices'
    npz_files: StringTuple = ('mpii.npz', 'lsp.npz', 'lspet.npz', 'coco.npz')
    body_thresh: float = 0.1
    hand_thresh: float = 0.2
    face_thresh: float = 0.4
    min_hand_keypoints: int = 8
    min_head_keypoints: int = 8


@dataclass
class SPINX(SPIN):
    img_folder = 'data/spinx/images'
    vertex_folder: str = 'data/spinx/vertices'
    return_shape: bool = True
    return_full_pose: bool = True
    return_expression: bool = True
    return_vertices: bool = True
    metrics: StringTuple = tuple()


@dataclass
class DatasetConfig:
    use_packed: bool = True
    use_face_contour: bool = True
    vertex_flip_correspondences: str = ''
    transforms: Transforms = Transforms()
    splits: Splits = Splits()
    num_workers: NumWorkers = NumWorkers()


@dataclass
class EFT(DatasetWithKeypoints):
    ''' EFT dataset configuration
    '''
    data_folder: str = 'data/eft'
    img_folders: Tuple[str] = ('data/eft/coco/',)
    json_files: Tuple[str] = ('data/eft/coco.json',)
    pass


@dataclass
class Agencies(DatasetWithKeypoints):
    data_folder: str = 'data/model_agencies'
    metrics: StringTuple = ('measurements',)
    img_folder: str = 'images'
    annot_fname: str = 'cleaned_model_data.json'
    keypoint_fname: str = 'keypoints.json'
    weight_fname: str = 'weights.json'
    splits_fname: str = 'splits.json'
    identity_fname: str = 'final_identities.pkl'
    betas_fname: str = 'betas.json'
    attributes_fname: str = 'attributes.json'
    return_params: bool = True
    param_folder: str = 'data/model_agencies/parameters'
    #  agencies: Optional[StringTuple] = None
    openpose_format: str = 'coco25'
    body_thresh: float = 0.1
    binarization: bool = False
    keep_only_with_reg: bool = False
    only_data_with_attributes: bool = False


@dataclass
class SSP3D(DatasetWithKeypoints):
    data_folder: str = 'data/ssp3d'
    metrics: StringTuple = ('v2v', 'v2v_t')
    img_folder: str = 'images'
    silh_folder: str = 'silhouettes'
    label_fname: str = 'data/ssp3d/labels.npz'

@dataclass
class HBW(DatasetWithKeypoints):
    data_folder: str = 'data/hbw'
    img_folder: str = 'photos'
    keyp_folder: str = 'keypoints/keypoints'
    imgs_minimal: str = ''
    keyps_minimal: str = ''
    annot_fname: str = 'annotations.yaml'
    gender_fname: str = 'genders.yaml'
    mesh_folder: str = 'v_templates/smplx_with_optimization'
    meas_definition_path: str = ''
    meas_vertices_path: str = ''
    body_model_folder: str = ''
    metrics: StringTuple =  ('v2v_t', 'p2p_t', 'measurements')

@dataclass
class PoseConfig(DatasetConfig):
    ''' Configuration for the body pose datasets
    '''
    sampler: Sampler = Sampler()
    splits: Splits = Splits(train=('curated_fits', 'spin'))
    eft: EFT = EFT()
    openpose: OpenPose = OpenPose()
    tracks: Tracks = Tracks()
    human36m: Human36m = Human36m()
    human36mx: Human36mX = Human36mX()
    ehf: EHF = EHF()
    curated_fits: CuratedFits = CuratedFits()
    threedpw: ThreeDPW = ThreeDPW()
    spin: SPIN = SPIN()
    spinx: SPINX = SPINX()
    agora: Agora = Agora()


@dataclass
class ShapeConfig(DatasetConfig):
    ''' Configuration for the body shape datasets
    '''
    balanced_genders: bool = True
    key: str = 'bmi'
    model_agencies: Agencies = Agencies()
    ssp3d: SSP3D = SSP3D()
    weight_height: WeightHeight = WeightHeight()
    hbw: HBW = HBW()
    sampler: Sampler = Sampler()


pose_conf = OmegaConf.structured(PoseConfig)
shape_conf = OmegaConf.structured(ShapeConfig)

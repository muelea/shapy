import sys
import argparse
import glob

import pickle

import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

import smplx

from body_measurements import BodyMeasurements


Array = np.ndarray


PARENT_FOLDER, _ = osp.split(osp.abspath(__file__))

DEFAULT_HBW_FOLDER = osp.join(
    PARENT_FOLDER, '../..', 'datasets', 'HBW'
)
DEFAULT_POINT_REG_SMPLX = osp.join(
    PARENT_FOLDER, '../..', 'data', 'utility_files', 'evaluation', 'eval_point_set',
    'HD_SMPLX_from_SMPL.pkl',
)
DEFAULT_POINT_REG_SMPL = osp.join(
    PARENT_FOLDER, '../..', 'data', 'utility_files', 'evaluation', 'eval_point_set',
    'HD_SMPL_sparse.pkl'
)
DEFAULT_BODY_MODEL_FOLDER = osp.join(
    PARENT_FOLDER, '../..', 'data', 'body_models'
)
DEFAULT_BODY_MEASUREMENT_FOLDER = osp.join(
    PARENT_FOLDER, '../..', 'data', 'utility_files', 'measurements'
)


def point_error(
    x: Array,
    y: Array,
    align: bool = True,
) -> float:
    
    t = 0.0
    if align:
        t = x.mean(0, keepdims=True) - y.mean(0, keepdims=True)

    x_hat = x - t

    error = np.sqrt(np.power(x_hat - y, 2).sum(axis=-1))

    return error.mean().item()


def main(
    input_npz_file: str,
    hbw_folder: str,
    model_type: str = 'smplx',
    point_reg_gt: str = DEFAULT_POINT_REG_SMPLX,
    point_reg_fit: str = DEFAULT_POINT_REG_SMPLX,
    body_measurement_folder: str = DEFAULT_BODY_MEASUREMENT_FOLDER,
    body_model_folder: str = DEFAULT_BODY_MODEL_FOLDER
) -> None:

    # read submitted npz file
    new_method_result = np.load(input_npz_file)
    labels = new_method_result['image_name']
    fits = new_method_result['v_shaped']

    # load files to compute P2P-20K Error
    with open(point_reg_gt, 'rb') as f:
        point_regressor_gt = pickle.load(f)

    with open(point_reg_fit, 'rb') as f:
        point_regressor_fit = pickle.load(f)

    # load files to compute Measurements Error
    meas_def_path = osp.join(body_measurement_folder, 'measurement_defitions.yaml')
    meas_verts_path_gt = osp.join(body_measurement_folder, 'smplx_measurements.yaml')
    body_measurements_gt = BodyMeasurements(
        {'meas_definition_path': meas_def_path,
            'meas_vertices_path': meas_verts_path_gt},
    ).to('cuda')
    model_measurements_file_fit = f'{model_type}_measurement_vertices.yaml' \
        if model_type == 'smpl' else f'{model_type}_measurements.yaml'
    meas_verts_path_fit = osp.join(body_measurement_folder, model_measurements_file_fit)
    body_measurements_fit = BodyMeasurements(
        {'meas_definition_path': meas_def_path,
            'meas_vertices_path': meas_verts_path_fit},
    ).to('cuda')

    # create SMPL model
    #body_model = smplx.create(
    #    model_path=body_model_folder,
    #    model_type=model_type
    #)
    #faces_tensor = body_model.faces_tensor

    # create ground-truth (SMPL-X) model
    body_model_smplx = smplx.create(
        model_path=body_model_folder,
        model_type='smplx'
    )
    faces_tensor_smplx = body_model_smplx.faces_tensor

    # create fir (SMPL or SMPL-X) model
    body_model_fit = smplx.create(
        model_path=body_model_folder,
        model_type=model_type
    )
    faces_tensor_fit = body_model_fit.faces_tensor
    
    v2v_t_errors = []
    point_t_errors = []
    measurement_errors = {
        'height': [],
        'chest': [],
        'waist': [],
        'hips': [],
        'mass': []
    }

    # Evaluate
    for label, v_shaped_fit in tqdm(zip(labels, fits), total=len(fits)):

        # load ground truth shape of subject
        split, subject, _, img_fn = label.split('/')
        subject_id_npy = subject.split('_')[0] + '.npy'
        v_shaped_gt_path = osp.join(hbw_folder, 'smplx', split, subject_id_npy)
        v_shaped_gt = np.load(v_shaped_gt_path)

        # cast v-shaped
        v_shaped_gt = v_shaped_gt.astype(np.float32)
        v_shaped_fit = v_shaped_fit.astype(np.float32)

        # compute vertex-to-vertex error (SMPL-X only)
        if model_type == 'smplx':
            v2v_error = point_error(v_shaped_fit, v_shaped_gt, align=True)
            v2v_t_errors.append(v2v_error)

        # compute P2P-20k error
        points_gt = point_regressor_gt.dot(v_shaped_gt)
        points_fit = point_regressor_fit.dot(v_shaped_fit)
        p2p_error = point_error(points_gt, points_fit, align=True)
        point_t_errors.append(p2p_error)

        # compute height/chest/waist/hip error
        #shaped_triangles_gt = v_shaped_gt[faces_tensor]
        shaped_triangles_gt = v_shaped_gt[faces_tensor_smplx]
        shaped_triangles_gt = torch.from_numpy(shaped_triangles_gt) \
            .unsqueeze(0).to('cuda')
        measurements_gt = body_measurements_gt(
            shaped_triangles_gt)['measurements']

        #shaped_triangles_fit = v_shaped_fit[faces_tensor]
        shaped_triangles_fit = v_shaped_fit[faces_tensor_fit]
        shaped_triangles_fit = torch.from_numpy(shaped_triangles_fit) \
            .unsqueeze(0).to('cuda')
        measurements_fit = body_measurements_fit(
            shaped_triangles_fit)['measurements']
        
        for k in measurement_errors.keys():
            error = abs(measurements_gt[k]['tensor'].item() - \
                measurements_fit[k]['tensor'].item())
            measurement_errors[k].append(error)

    # print result 
    if model_type == 'smplx': 
        final_v2v_t_error = np.array(v2v_t_errors).mean() * 1000
        print(f'V2V Error: {final_v2v_t_error:.0f} mm')

    final_point_t_error = np.array(point_t_errors).mean() * 1000
    print(f'P2P-20k Error: {final_point_t_error:.0f} mm')

    for k, v in measurement_errors.items():
        if k in ['chest', 'waist', 'hips', 'height']:
            final_mmts_error = np.array(v).mean() * 1000
            print(f'{k} Error: {final_mmts_error:.0f} mm')
        else:
            final_mmts_error = np.array(v).mean()
            print(f'{k} Error: {final_mmts_error:.0f} kg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-npz-file',
                        dest='input_npz_file', type=str, required=True,
                        help='npz containing labels and body shape parameters.')
    parser.add_argument('--hbw-folder',
                        dest='hbw_folder', type=str,
                        default=DEFAULT_HBW_FOLDER,
                        help='folder with ground truth bodies.')
    parser.add_argument('--model-type', choices=['smpl', 'smplx'], type=str,
                        default='smplx',
                        help='The model type used for body shape prediction. ')
    parser.add_argument('--point-reg-gt',
                        dest='point_reg_gt',
                        default=DEFAULT_POINT_REG_SMPLX,
                        type=str,
                        help='The path to the point regressor for the ground-truth.'
                        )
    parser.add_argument('--point-reg-fit',
                        dest='point_reg_fit',
                        default=DEFAULT_POINT_REG_SMPLX,
                        type=str,
                        help='The path to the point regressor for the predictions.'
                        )
    parser.add_argument('--body-model-folder',
                        dest='body_model_folder',
                        default=DEFAULT_BODY_MODEL_FOLDER,
                        type=str,
                        help='The path to the smpl/body model folder.'
                        )
    parser.add_argument('--body-measurement-folder',
                        dest='body_measurement_folder',
                        default=DEFAULT_BODY_MEASUREMENT_FOLDER,
                        type=str,
                        help='The path to the smpl/body model folder.'
                        )

    args = parser.parse_args()

    main(
        input_npz_file=args.input_npz_file,
        hbw_folder=args.hbw_folder,
        model_type=args.model_type,
        point_reg_gt=args.point_reg_gt,
        point_reg_fit=args.point_reg_fit,
    )
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import sys
import os.path as osp
import argparse

import numpy as np
import torch

import smplx
import open3d as o3d
import time
import cv2
from scipy.spatial import ConvexHull

import trimesh
#  from meas_definitions import measurements_definitions, measures_vertex
from loguru import logger
from star.pytorch.star import STAR
from star.config import cfg as star_cfg

from mesh_mesh_intersection import MeshMeshIntersection
from body_measurements import ChestWaistHipsMeasurements, BodyMeasurements


def get_plane_at_height(h):
    verts = np.array([[-1., h, -1], [1, h, -1], [1, h, 1], [-1, h, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    normal = np.array([0.0, 1.0, 0.0])
    return verts, faces, (verts[0], normal)


def my_bool_arg(x):
    return x.lower() in ['true', '1']


def to_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def main(
    model_folder,
    model_type='smplx',
    ext='npz',
    gender='neutral',
    num_betas=10,
    sample_shape=False,
    num_expression_coeffs=10,
    num_samples=1,
    meas_definition_path: str = 'data/measurement_defitions.yaml',
    meas_vertices_path: str = 'data/smpl_measurement_vertices.yaml',
    use_cuda: bool = True,
):

    device = torch.device('cuda' if use_cuda else 'cpu')
    dtype = torch.float32

    meas_module = BodyMeasurements(dict(
        meas_definition_path=meas_definition_path,
        meas_vertices_path=meas_vertices_path,
    )
    )

    meas_module = meas_module.to(device=device)

    trans, pose = None, None
    if model_type == 'star':
        star_cfg.path_male_star = osp.expandvars(
            '$HOME/workspace/body_models/star/STAR_MALE.npz')
        star_cfg.path_female_star = osp.expandvars(
            '$HOME/workspace/body_models/star/STAR_FEMALE.npz')
        model = STAR(gender=gender, num_betas=num_betas)
        trans = torch.zeros([num_samples, 3], dtype=dtype, device=device)
        pose = torch.zeros([num_samples, 72], dtype=dtype, device=device)
    else:
        model = smplx.build_layer(
            model_folder, model_type=model_type, gender=gender,
            num_betas=num_betas, num_expression_coeffs=num_expression_coeffs,
            ext=ext)

    model = model.to(device=device)

    if sample_shape:
        np.random.seed(0)
        betas = np.random.rand(num_samples, model.num_betas)
        betas = torch.from_numpy(betas).to(device=device, dtype=dtype)
    else:
        betas = torch.zeros(
            [num_samples, model.num_betas],
            dtype=torch.float32, device=device)

    if model_type == 'star':
        vertices = model(pose=pose, trans=trans, betas=betas)
        model_tris = vertices[:, model.faces]
        faces = to_numpy(model.faces)
        faces_tensor = model.faces
    else:
        output = model(betas=betas, return_verts=True)
        vertices = output.vertices
        faces = model.faces
        faces_tensor = model.faces_tensor[None].expand(len(vertices), -1, -1)
        model_tris = output.vertices[:, model.faces_tensor]

    output = meas_module(
        model_tris, vertices=vertices, faces=faces_tensor,
        compute_mass=False, compute_height=False,)['measurements']

    vertices = to_numpy(vertices)

    for n in range(num_samples):

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices[n])
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        colors = np.ones_like(vertices[n]) * [0.3, 0.3, 0.3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        geometry = []
        geometry.append(mesh)

        meas_values = [
            f'{meas_name}: {output[meas_name]["value"][n].item() * 100:.3f} (cm)'
            for meas_name in output]
        logger.info(f'{n}: {meas_values}')

        for meas_name in output:
            pcl = o3d.geometry.PointCloud()
            if 'points' not in output[meas_name]:
                continue

            meas_value = output[meas_name]['value'][n]
            points = output[meas_name]['points'][n]
            if isinstance(points, (tuple, list)):
                points = torch.stack(points)
            if torch.is_tensor(points):
                points = points.detach().cpu().numpy()
            points = points.reshape(-1, 3)

            pcl.points = o3d.utility.Vector3dVector(points)
            pcl.paint_uniform_color([1.0, 0.0, 0.0])
            geometry.append(pcl)

            lineset = o3d.geometry.LineSet()
            line_ids = np.arange(len(points)).reshape(-1, 2)
            lineset.points = o3d.utility.Vector3dVector(points)
            lineset.lines = o3d.utility.Vector2iVector(line_ids)
            lineset.paint_uniform_color([0.0, 0.0, 0.0])
            geometry.append(lineset)

        o3d.visualization.draw_geometries(geometry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'star', ],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--sample-shape', default=False,
                        dest='sample_shape', type=my_bool_arg,
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=my_bool_arg,
                        help='Sample a random expression')
    parser.add_argument('--num-samples', default=1, type=int,
                        dest='num_samples',
                        help='Number of samples to draw.')
    parser.add_argument('--meas-definition-path',
                        dest='meas_definition_path',
                        default='data/measurement_defitions.yaml',
                        type=str,
                        help='The definitions of the measurements')
    parser.add_argument('--meas-vertices-path', dest='meas_vertices_path',
                        type=str,
                        default='data/smpl_measurement_vertices.yaml',
                        help='The indices of the vertices used for the'
                        ' the measurements')
    parser.add_argument('--use-cuda', dest='use_cuda', default=False,
                        type=my_bool_arg, help='Use CUDA for computations.')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    gender = args.gender
    ext = args.ext
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression
    num_samples = args.num_samples
    meas_definition_path = args.meas_definition_path
    meas_vertices_path = args.meas_vertices_path
    use_cuda = args.use_cuda

    main(model_folder, model_type, ext=ext,
         gender=gender,
         num_betas=num_betas,
         num_samples=num_samples,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         meas_definition_path=meas_definition_path,
         meas_vertices_path=meas_vertices_path,
         use_cuda=use_cuda,
         )

import sys
import os
import os.path as osp

os.environ['PYOPENGL_PLATFORM'] = 'egl'

from threadpoolctl import threadpool_limits
from tqdm import tqdm
import torch
import argparse
import trimesh
from loguru import logger
import numpy as np
import smplx
from body_measurements import BodyMeasurements
from attributes.utils.renderer import Renderer
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont


@torch.no_grad()
def main(
    demo_input_folder: os.PathLike = 'demo_input',
    demo_output_folder: os.PathLike = 'demo_output',
    meas_definition_path:  os.PathLike = '',
    meas_vertices_path:  os.PathLike = '',
    smpl_model_path:  os.PathLike = 'data/body_models/smpl',
    gender: str = 'neutral',
    num_betas: int = 10,
    render: bool = True,
) -> None:

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    os.makedirs(demo_output_folder, exist_ok=True)

    npz_files = sorted(os.listdir(demo_input_folder))
    npz_files = [x for x in npz_files if x.endswith('npz')]

    body_measurements = BodyMeasurements(
        {'meas_definition_path': meas_definition_path,
            'meas_vertices_path': meas_vertices_path},
    ).to(device)

    smpl= smplx.create(
        model_path=smpl_model_path,
        gender=gender,
        num_betas=num_betas,
        model_type='smplx'
    ).to(device)

    if render:
        renderer = Renderer(
            is_registration=False
        )

    for npz_file in npz_files:
        print(f'Processing: {npz_file}')

        # read betas
        data = np.load(osp.join(demo_input_folder, npz_file))
        betas = torch.from_numpy(data['betas']).to(device).unsqueeze(0)

        # smpl function & shaped body
        body = smpl(betas=betas)
        shaped_vertices = body['v_shaped']
        shaped_triangles = shaped_vertices[:,smpl.faces_tensor]

        # Compute the measurements on the body
        measurements = body_measurements(
            shaped_triangles)['measurements']
        
        # render shaped body
        if render:
            pred_mesh = trimesh.Trimesh(shaped_vertices.cpu().numpy()[0], smpl.faces)
            pred_img = renderer.render(pred_mesh)
            

        # print result
        mmts_str = '    Virtual measurements: '
        for k, v in measurements.items():
            value = v['tensor'].item()
            unit = 'kg' if k == 'mass' else 'm'
            mmts_str += f'    {k}: {value:.2f} {unit}'
        print(mmts_str)

        # add measurements to image and save image
        if render:
            font = ImageFont.truetype("../samples/OpenSans-Regular.ttf", size=24)
            ImageDraw.Draw(pred_img).text(
                (0, 10),  mmts_str, (0, 0, 0), font=font
            )
            pred_img.save(osp.join(demo_output_folder, npz_file.replace('npz', 'png')))


if __name__ == '__main__':
    #  torch.multiprocessing.set_start_method('fork')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'PyTorch SMPL-X Regressor Demo'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--output-folder', dest='output_folder',
                        default='demo_output', type=str,
                        help='The folder where the demo renderings will be saved')
    parser.add_argument('--input-folder', dest='input_folder',
                        default='demo_input', type=str,
                        help='The folder where the demo npz files are stored')
    parser.add_argument('--meas_definition_path', dest='meas_definition_path',
                        default='../data/utility_files/measurements/measurement_defitions.yaml', 
                        type=str, help='Path to measurement definitions')
    parser.add_argument('--meas_vertices_path', dest='meas_vertices_path',
                        default='../data/utility_files/measurements/smplx_measurements.yaml', type=str,
                        help='Path to measurement vertices')
    parser.add_argument('--smpl_model_path', dest='smpl_model_path',
                        default='../data/body_models', type=str,
                        help='Path to smpl model folder')
    parser.add_argument('--num_betas', dest='num_betas',
                        default=10, type=int,
                        help='number of betas smpl model uses')
    parser.add_argument('--gender', dest='gender',
                        default='neutral', type=str,
                        help='gender of smpl model')
                        

    args = parser.parse_args()

    main( 
        demo_input_folder=args.input_folder,
        demo_output_folder=args.output_folder, 
        meas_definition_path=args.meas_definition_path,
        meas_vertices_path=args.meas_vertices_path,
        smpl_model_path=args.smpl_model_path,
        gender=args.gender,
        num_betas=args.num_betas
    )
import os
import json
import os.path as osp
import argparse

from loguru import logger

from attributes.utils.config import parse_args
from attributes.dataloader.demo import DEMO_S2A, DEMO_A2S
from attributes.utils.config import default_conf
from omegaconf import OmegaConf

from attributes.attributes_betas.build import MODEL_DICT


def main(args, demo_output_folder, smpl_model_path, render=True):

    os.makedirs(args.output_dir, exist_ok=True)

    network_type = args.get('type', 'a2b')

    output_dir = osp.expandvars(args.get('output_dir', 'output'))
    checkpoint_path = osp.join(output_dir, 'last.ckpt')
    ds_gender = args.get('ds_gender', '')
    model_gender = args.get('model_gender', '')
    model_type = args.get('model_type', 'smplx')

    os.makedirs(demo_output_folder, exist_ok=True)

    if network_type == 'a2b':
        if render:
            import trimesh
            import torch
            import smplx
            import sys
            from attributes.utils.renderer import Renderer

            renderer = Renderer(
                is_registration=False
            )

            device = torch.device('cuda')
            if not torch.cuda.is_available():
                logger.error('CUDA is not available!')
                sys.exit(3)

            smpl= smplx.create(
                model_path=smpl_model_path,
                gender=model_gender,
                num_betas=10,
                model_type=model_type
            ).to(device)

        loaded_model = MODEL_DICT[network_type].load_from_checkpoint(
            checkpoint_path=checkpoint_path)

        dataset = DEMO_A2S(
            ds_gender=ds_gender,
            model_gender=model_gender,
            model_type=model_type,
            rating_folder='../samples/attributes/'
        )

        test_input, _ = loaded_model.create_input_feature_vec(dataset.db)

        test_input = loaded_model.to_whw2s(test_input, None) \
            if loaded_model.whw2s_model else test_input
        prediction = loaded_model.a2b.predict(test_input)

        logger.info(f'Estimated shape for demo dataset.')
        for idx, betas in enumerate(prediction):
            model_name = dataset.db['ids'][idx]
            print(f'Predicted bestas for {model_name}')
            print(betas.detach().cpu().numpy())

        if render:
            for idx, betas in enumerate(prediction):
                body = smpl(betas=betas.unsqueeze(0).to(device))
                shaped_vertices = body['v_shaped']
                pred_mesh = trimesh.Trimesh(shaped_vertices.detach().cpu().numpy()[0], smpl.faces)
                pred_img = renderer.render(pred_mesh)
                pred_img.save(osp.join(demo_output_folder, dataset.db['ids'][idx]+'.png'))


    elif network_type == 'b2a':

        loaded_model = MODEL_DICT[network_type].load_from_checkpoint(
            checkpoint_path=checkpoint_path)

        dataset = DEMO_S2A(
            betas_folder='../samples/shapy_fit/',
            ds_genders_path='../samples/genders.yaml',
            model_gender=model_gender,
            model_type=model_type,            
        )

        logger.info(f'Estimated attribute ratings for demo dataset for {ds_gender}')

        dataset.create_db(ds_gender)

        test_input = dataset.db[dataset.betas_key][:, :loaded_model.betas_size]
        if 'rating' in dataset.db.keys():
            test_output = dataset.db['rating']
        else:
            test_output = None

        test_prediction = loaded_model.b2a.predict(test_input)
        
        if test_output is not None:
            mean, std = loaded_model.metric_mean_std(test_output, test_prediction)
            ccp = loaded_model.metric_classification(test_output, test_prediction)

        # print result for each attribute
        output_names = loaded_model.selected_attr + loaded_model.selected_mmts
        if test_output is not None:
            for i, name in enumerate(output_names):
                l1m = mean[i].item()
                l1std = std[i].item()
                acc = ccp[i].item() * 100
                print(f'{name:20s}: {l1m:.2f} +/- {l1std:.2f},  {acc:.2f}\%')

        # print attribute predition for each shape
        for img_idx, img_id in enumerate(dataset.db['filename']):
            print(f'\n Results for image {img_id}')
            for name, estimate in zip(output_names, test_prediction[img_idx]):
                print(f'{name:20s}: {estimate:.2f}')


if __name__ == "__main__":

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter

    description = 'A2S and S2A regressor'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfgs',
                        nargs='+', default=[],
                        help='The configuration of the experiment')
    parser.add_argument('--demo_output_folder', default='../samples/a2s_fit',
                        help='folder where the fitted mesh is stored.')
    parser.add_argument('--smpl_model_path', default='../data/body_models',
                        help='Body model folder.')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*',
                        help='The configuration of the Detector')
    cmd_args = parser.parse_args()
    demo_output_folder = cmd_args.demo_output_folder
    smpl_model_path = cmd_args.smpl_model_path

    cfg = default_conf.copy()
    for exp_cfg in cmd_args.exp_cfgs:
        if exp_cfg:
            cfg.merge_with(OmegaConf.load(exp_cfg))
    if cmd_args.exp_opts:
        cfg.merge_with(OmegaConf.from_cli(cmd_args.exp_opts))

    main(cfg, demo_output_folder, smpl_model_path)

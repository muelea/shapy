import os
import os.path as osp

import torch
from loguru import logger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from attributes.utils.config import parse_args
from attributes.dataloader.linear_regression import REGRESSION_DATASET

from attributes.attributes_betas.build import build, MODEL_DICT
from attributes.utils.checkpoint import get_checkpoint_filename


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_name = args.get('dataset', 'sythetic')[0]

    os.makedirs(args.output_dir, exist_ok=True)

    network_type = args.get('type', 'a2b')

    output_dir = osp.expandvars(args.get('output_dir', 'output'))
    checkpoint_path = osp.join(output_dir, 'last.ckpt')

    if args.train:
        logger.add(f'{output_dir}/train_out.log')

        # create dataset
        dataset = REGRESSION_DATASET(
            ds_name=ds_name,
            ds_gender=args.ds_gender,
            model_gender=args.model_gender,
            model_type=args.model_type,
            use_loo=args.regression.use_loo_cross_val,
            is_train=args.train
        )

        # get regression model
        fitter = build(
            OmegaConf.to_container(args),
            type=network_type,
        )
        fitter.fit(dataset)

        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename=get_checkpoint_filename(network_type),
            save_last=True,
        )
        trainer = pl.Trainer(
            gpus=1,
            default_root_dir=args.output_dir,
            callbacks=[checkpoint_callback],
            max_epochs=args.max_epochs,
        )

        # trainer.model_connector.copy_trainer_model_properties(fitter)
        # trainer.callback_connector._attach_model_callbacks(fitter, trainer)
        # trainer.checkpoint_connector.restore_weights()
        # trainer.checkpoint_connector.has_trained = True
        trainer.model = fitter
        # trainer.setup(fitter, 'validate')

        trainer.save_checkpoint(checkpoint_path)
    else:
        if network_type == 'a2b':
            logger.add(f'{output_dir}/val_out.log')

            loaded_model = MODEL_DICT[network_type].load_from_checkpoint(
                checkpoint_path=checkpoint_path)

            # evaluate on caesar data
            for val_dataset_name in ['caesar', 'models']:
                dataset = REGRESSION_DATASET(
                    ds_name=val_dataset_name,
                    ds_gender=args.ds_gender,
                    model_gender=args.model_gender,
                    model_type=args.model_type,
                    use_loo=args.regression.use_loo_cross_val,
                    is_train=args.train
                )

                train_data, val_data, test_data = loaded_model.get_tvt_data(dataset)
                (train_input, train_noise), train_output = train_data
                (val_input, val_noise), val_output = val_data
                (test_input, test_noise), test_output = test_data

                train_input = loaded_model.to_whw2s(train_input, None) \
                    if loaded_model.whw2s_model else train_input
                val_input = loaded_model.to_whw2s(val_input, None) \
                    if loaded_model.whw2s_model else val_input
                prediction = loaded_model.a2b.predict(val_input)
                val_output = loaded_model.run_batch_validation(val_output, prediction)
                logger.info(f'Reporting results on {val_dataset_name} validation set.')
                loaded_model.print_result(val_output)

                if args.eval_test:
                    test_input = loaded_model.to_whw2s(test_input, None) \
                        if loaded_model.whw2s_model else test_input
                    prediction = loaded_model.a2b.predict(test_input)
                    test_output = loaded_model.run_batch_validation(test_output, prediction)
                    logger.info(f'Reporting results on {val_dataset_name} test set.')
                    loaded_model.print_result(test_output)

        elif network_type == 'b2a':
            data = REGRESSION_DATASET(
                    ds_name='caesar',
                    ds_gender=args.ds_gender,
                    model_gender=args.model_gender,
                    model_type=args.model_type,
                    use_loo=args.regression.use_loo_cross_val,
                    is_train=args.train
            )

            loaded_model = MODEL_DICT[network_type].load_from_checkpoint(
                checkpoint_path=checkpoint_path)

            rating_label = data.db['labels']
            train_data, val_data, test_data = loaded_model.get_tvt_data(data)
            train_input, train_output = train_data
            val_input, val_output = val_data
            test_input, test_output = test_data

            # Eval validation set
            print(f'Reporting results on CAESAR validation set')
            val_prediction = loaded_model.b2a.predict(val_input)
            mean, std = loaded_model.metric_mean_std(val_output, val_prediction)
            ccp = loaded_model.metric_classification(val_output, val_prediction)

            # print result for each attribute
            output_names = loaded_model.selected_attr + loaded_model.selected_mmts
            for i, name in enumerate(output_names):
                l1m = mean[i].item()
                l1std = std[i].item()
                acc = ccp[i].item() * 100
                print(f'{name:20s} &   ${l1m:.2f} \pm {l1std:.2f}$   &   ${acc:.2f}\%$   &   &   \\\\')

            if args.eval_test:
                print(f'Reporting results on CAESAR test set')
                # Eval validation set
                test_prediction = loaded_model.b2a.predict(test_input)
                mean, std = loaded_model.metric_mean_std(test_output, test_prediction)
                ccp = loaded_model.metric_classification(test_output, test_prediction)

                # print result for each attribute
                output_names = loaded_model.selected_attr + loaded_model.selected_mmts
                for i, name in enumerate(output_names):
                    l1m = mean[i].item()
                    l1std = std[i].item()
                    acc = ccp[i].item() * 100
                    print(f'{name:20s} &   ${l1m:.2f} \pm {l1std:.2f}$   &   ${acc:.2f}\%$   &   &   \\\\')


if __name__ == "__main__":

    args = parse_args()

    # check arguments
    assert len(args.dataset) == 1, \
        'Max. one dataset allwod for fitting'

    # check if model params are correct when synthetic data is selected.
    if args.dataset[0] == 'synthetic':
        assert args.model_type == 'smpl' and args.num_shape_comps == 8 \
            and args.regression.use_loo_cross_val, 'Settings not possible. '\
            'Please use SMPL, 8 betas, and loo cross validation for synthetic data.'

    main(args)

import sys
import os
import joblib
import numpy as np
from loguru import logger


class REGRESSION_DATASET():
    def __init__(
        self,
        ds_name,
        ds_gender,
        model_gender,
        model_type,
        use_loo,
        is_train,
    ):
        self.ds_name = ds_name
        self.ds_gender = ds_gender
        self.model_gender = model_gender
        self.model_type = model_type
        self.use_loo = use_loo
        self.is_train = is_train
        self.eval_bs = 1
        self.betas_key = f'betas_{self.model_type}_{self.model_gender}'

        self.create_db()

        if not self.use_loo:
            set = 'val' if self.is_train else 'test'
            self.eval_bs = self.db[set]['rating'].shape[0]

    def create_db(self):
        if self.ds_name == 'synthetic':
            self.create_sythetic_db()
        elif self.ds_name == 'caesar':
            self.create_caesar_db()
        elif self.ds_name == 'models':
            self.create_model_db()

    def create_sythetic_db(self):

        pt_path = f'../data/dbs/synthetic_{self.ds_gender}.pt'
        logger.info(f'Loading data from {pt_path}')
        db = joblib.load(pt_path)

        self.db = {
            self.betas_key: np.array(db[self.betas_key]),
            'rating': np.array(db['rating']),
            'height_gt': np.array(db['heights']),
            'weight_gt': np.array(db['weights'])
        }

    def create_caesar_db(self):

        self.db = {}

        # read data
        pt_path = '../data/dbs/caesar_{}_{}.pt'

        path = pt_path.format(self.ds_gender, 'train')
        train_data = joblib.load(path)
        logger.info(f'Load train_data data from {path}')

        path = pt_path.format(self.ds_gender, 'val')
        val_data = joblib.load(pt_path.format(self.ds_gender, 'val'))
        logger.info(f'Load val_data data from {path}')

        path = pt_path.format(self.ds_gender, 'test')
        test_data = joblib.load(pt_path.format(self.ds_gender, 'test'))
        logger.info(f'Load test_data data from {path}')

        # labels are equal for each set so we only add them once
        self.db['labels'] = np.array([x.split('.')[1]
                                      for x in train_data['rating_label']])
        
        if not self.use_loo:
            for set in ['train', 'val', 'test']:
                self.db[set] = {}
                self.db[set]['rating'] = eval(f'{set}_data').get('rating')
                if self.db[set]['rating'] is None:
                    self.db[set]['rating'] = eval(f'{set}_data').get('ratings')
                    self.db[set]['rating_raw'] = eval(f'{set}_data').get('ratings_raw')

                self.db[set]['height_gt'] = eval(f'{set}_data')['heights']
                self.db[set]['weight_gt'] = eval(f'{set}_data')['weights']
                self.db[set][self.betas_key] = eval(f'{set}_data')[
                    self.betas_key]

                self.db[set]['chest'] = eval(f'{set}_data')['chest']
                self.db[set]['waist'] = eval(f'{set}_data')['waist']
                self.db[set]['hips'] = eval(f'{set}_data')['hips']
        else:
            logger.info('Concatenate train, val and test data for loo.')
            # concatenate train, val and test data for leave-one-out cross validation
            self.db['rating'] = np.concatenate(
                (train_data['rating'], val_data['rating'],
                 test_data['rating']), 0
            )
            self.db[self.betas_key] = np.concatenate(
                (train_data[self.betas_key], val_data[self.betas_key],
                 test_data[self.betas_key]), 0
            )
            self.db['height_gt'] = np.concatenate(
                (train_data['heights'], val_data['heights'],
                 test_data['heights']), 0
            )
            self.db['weight_gt'] = np.concatenate(
                (train_data['weights'], val_data['weights'],
                 test_data['weights']), 0
            )

    def create_model_db(self):
        self.db = {}
        # read data
        pt_path = '../data/dbs/models_{}_{}.pt'

        path = pt_path.format(self.ds_gender, 'train')
        train_data = joblib.load(path)
        logger.info(f'Load train_data data from {path}')

        path = pt_path.format(self.ds_gender, 'val')
        val_data = joblib.load(pt_path.format(self.ds_gender, 'val'))
        logger.info(f'Load val_data data from {path}')

        path = pt_path.format(self.ds_gender, 'test')
        test_data = joblib.load(pt_path.format(self.ds_gender, 'test'))
        logger.info(f'Load test_data data from {path}')

        # labels are equal for each set so we only add them once
        self.db['labels'] = np.array([x.split('.')[1]
                                      for x in train_data['rating_label']])

        if not self.use_loo:
            for set in ['train', 'val', 'test']:
                self.db[set] = {}
                self.db[set]['rating'] = eval(f'{set}_data').get('rating')
                if self.db[set]['rating'] is None:
                    self.db[set]['rating'] = eval(f'{set}_data').get('ratings')
                self.db[set]['height_gt'] = eval(f'{set}_data')['heights']
                self.db[set]['weight_gt'] = eval(f'{set}_data')['weights']
                self.db[set]['ids'] = eval(f'{set}_data')['ids']

                def to_float(input_lst):
                    return list(map(
                        lambda x: float(x) if len(x) > 0 else -1,
                        input_lst))

                chest = to_float(eval(f'{set}_data')['bust'])
                chest = np.asarray(chest) / 100

                mean_chest = np.mean(chest[chest > 0])
                chest[chest <= 0] = mean_chest

                self.db[set]['chest'] = chest

                waist = to_float(eval(f'{set}_data')['waist'])
                waist = np.asarray(waist) / 100
                mean_waist = np.mean(waist[waist > 0])
                waist[waist <= 0] = mean_waist

                self.db[set]['waist'] = waist

                hips = to_float(eval(f'{set}_data')['hips'])
                hips = np.asarray(hips) / 100
                mean_hips = np.mean(hips[hips > 0])
                hips[hips <= 0] = mean_hips

                self.db[set]['hips'] = hips

                self.db[set][self.betas_key] = eval(f'{set}_data')[
                    self.betas_key]
        else:
            logger.info('Concatenate train, val and test data for loo.')
            # concatenate train, val and test data for leave-one-out cross validation
            self.db['rating'] = np.concatenate(
                (train_data['rating'], val_data['rating'],
                 test_data['rating']), 0
            )
            self.db[self.betas_key] = np.concatenate(
                (train_data[self.betas_key], val_data[self.betas_key],
                 test_data[self.betas_key]), 0
            )
            self.db['height_gt'] = np.concatenate(
                (train_data['heights'], val_data['heights'],
                 test_data['heights']), 0
            )
            self.db['weight_gt'] = np.concatenate(
                (train_data['weights'], val_data['weights'],
                 test_data['weights']), 0
            )

        # for key, value in self.db.items():
        #     logger.info(f'{key}: {value.shape}')
        # logger.info(self.db['weight_gt'].shape)
        # import sys
        # sys.exit(0)

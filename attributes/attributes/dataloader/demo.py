import os
import os.path as osp
import yaml
import numpy as np
import pickle
import joblib

LABELS = {
    'female': np.array([
                'Big', 'Broad Shoulders', 'Feminine', 'Large Breasts', 'Long Legs',
                'Long Neck', 'Long Torso', 'Muscular', 'Pear Shaped', 'Petite', 
                'Short', 'Short Arms', 'Skinny Legs', 'Slim Waist', 'Tall']),

    'male': np.array([
                'Average', 'Big', 'Broad Shoulders', 'Delicate Build', 'Long Legs', 
                'Long Neck', 'Long Torso', 'Masculine', 'Muscular', 'Rectangular', 
                'Short', 'Short Arms', 'Skinny Arms', 'Soft Body', 'Tall'])
}


class DEMO_S2A():
    def __init__(
        self,
        betas_folder='../samples/shapy_fit/',
        ds_genders_path='../samples/genders.yaml',
        model_gender='neutral',
        model_type='smplx',
    ):
        self.ds_gender = yaml.safe_load(open(ds_genders_path, 'r')) # even if the smpl betas are neutral,
        # the gender of the person on the image is required, because S2A are gender specific.
        
        self.model_gender = model_gender
        self.model_type = model_type
        self.eval_bs = 1
        self.betas_key = f'betas_{self.model_type}_{self.model_gender}'

        self.npz_files_ = sorted(os.listdir(betas_folder))
        self.npz_files_ = [x for x in self.npz_files_ if x.endswith('npz')]

        self.npz_files = {'male': [], 'female': []}
        self.betas = {'male': [], 'female': []}

        for npz_file in self.npz_files_:
            npz_file_id = npz_file.split('.')[0]
            gender = self.ds_gender[npz_file_id]
            data = np.load(osp.join(betas_folder, npz_file))      
            self.betas[gender].append(data['betas'])
            self.npz_files[gender].append(npz_file_id)
        
        self.betas['male'] = np.array(self.betas['male'])
        self.betas['female'] = np.array(self.betas['female'])
        
    def create_db(self, ds_gender):
        """Bring data into structure that is compatible with model class"""
        self.db = {}
        self.db['labels'] = LABELS[ds_gender]
        self.db[self.betas_key] = self.betas[ds_gender]
        self.db['filename'] = self.npz_files[ds_gender]

class DEMO_A2S():
    def __init__(
        self,
        #betas_folder='../samples/shapy_fit/',
        ds_gender='female',
        model_gender='neutral',
        model_type='smplx',
        rating_folder='../samples/attributes/'
    ):
        self.ds_gender = ds_gender # even if the smpl betas are neutral,
        # the gender of the person on the image is required, 
        # because A2S and its variations are gender specific.
        
        self.model_gender = model_gender
        self.model_type = model_type
        self.eval_bs = 1
        #self.betas_key = f'betas_{self.model_type}_{self.model_gender}'


        self.rating_path = osp.join(rating_folder, f'modeldata_for_a2s_{self.ds_gender}.pt')
        self.db = joblib.load(self.rating_path)
        
        if 'rating' not in self.db.keys():
            self.db['rating'] = self.db['ratings']

        self.db['height_gt'] = self.db['heights'].astype(np.float32)
        self.db['chest'] = self.db['bust'].astype(np.float32) / 100
        self.db['waist'] = self.db['waist'].astype(np.float32) / 100
        self.db['hips'] = self.db['hips'].astype(np.float32) / 100

    def create_db(self, ds_gender):
        """Bring data into structure that is compatible with model class"""
        self.db = {}
        self.db['labels'] = LABELS[ds_gender]
        #self.db[self.betas_key] = self.betas[ds_gender]
        #self.db['filename'] = self.npz_files[ds_gender]
        self.db['rating'] = self.ratings[ds_gender]
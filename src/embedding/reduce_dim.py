import os
import json
import umap
import joblib
import numpy as np
from tqdm import tqdm
from time import time

class Reducer:
    """Dimensionality reduction"""

    """
    Constructor
    """
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path

        self.model_code_to_file_path = {}

        self.X = None
        self.X_all = None
        self.idx2id = {}
        self.idx2id_all = {}
        self.num_instances = 0
        self.base_model_nickname = ''
    
        self.reducer = None
        self.emb = {}
        self.emb2d = {}


    """
    Wrapper function called by main.py
    """
    def reduce_dim(self):
        self.init_reducer()
        self.load_embedding()
        self.run_dim_reduction()
        self.save_results()


    """
    Initial setting
    """
    def init_reducer(self):
        if self.args.dim_reduction == 'UMAP':
            self.reducer = umap.UMAP(n_components=2)

    
    def load_embedding(self):
        # Assign model code and load embedding
        for dirpath, dnames, fnames in os.walk(self.args.emb_set_dir):
            for f in fnames:
                if '-log-' in f:
                    continue
                if 'emb2d' in f:
                    continue
                
                file_path = os.path.join(dirpath, f)
                if f.startswith('neuron_emb'):
                    # Base model
                    model_nickname = f.split('-')[1]
                    self.model_code_to_file_path[model_nickname] = file_path
                    self.emb[model_nickname] = self.load_json(file_path)
                    self.num_instances += len(self.emb[model_nickname])
                    self.base_model_nickname = model_nickname

                elif f.startswith('proj_neuron_emb'):
                    # Other models
                    model_nickname = f.split('-')[1]
                    self.model_code_to_file_path[model_nickname] = file_path
                    self.emb[model_nickname] = self.load_json(file_path)
                    self.num_instances += len(self.emb[model_nickname])

                elif f.startswith('img_emb'):
                    # Image embedding
                    self.model_code_to_file_path['img'] = file_path
                    self.emb['img'] = np.loadtxt(file_path)
                    self.num_instances += len(self.emb['img'])
                    
                else:
                    log = f'Err. Unknown file type: {f}'
                    raise ValueError(log)

        if len(self.base_model_nickname) == 0:
            log = 'A base model file is not given, starting with "neuron_emb"'
            raise ValueError(log)

        # Generate matrix X for all neurons' vectors
        num_base_instances = len(self.emb[self.base_model_nickname])
        self.X = np.zeros((num_base_instances, self.args.dim))
        self.X_all = np.zeros((self.num_instances, self.args.dim))
        idx = 0
        for model_code in self.emb:
            if 'img' in model_code:
                num_imgs = len(self.emb[model_code])
                for i in range(num_imgs):
                    self.idx2id_all[idx] = '{}-{}'.format(model_code, i)
                    self.X_all[idx] = self.emb[model_code][i]
                    idx += 1
            else:
                for neuron_i, neuron in enumerate(self.emb[model_code]):
                    neuron_id = '{}-{}'.format(model_code, neuron)
                    self.idx2id_all[idx] = neuron_id
                    self.X_all[idx] = self.emb[model_code][neuron]
                    idx += 1
                    if model_code == self.base_model_nickname:
                        self.X[neuron_i] = self.emb[model_code][neuron]
                        self.idx2id[neuron_i] = neuron_id

    
    """
    Project the embdding to 2D
    """
    def run_dim_reduction(self):
        self.write_first_log()

        # Fit reducer and get all 2d embeddings
        tic = time()
        if self.args.model_for_emb_space == 'base':
            fitted_emb2d = self.reducer.fit_transform(self.X)
            emb2d = self.reducer.transform(self.X_all)
        elif self.args.model_for_emb_space == 'all':
            emb2d = self.reducer.fit_transform(self.X_all)
        toc = time()
        log = 'Fit and transform: {:.2f} sec'.format(toc - tic)
        self.write_log(log)

        # Save the reducer
        tic = time()
        self.save_reducer()
        toc = time()
        log = 'Save reducer: {:.2f} sec'.format(toc - tic)
        self.write_log(log)

        # Parse embeddings into self.emb2d
        tic = time()
        with tqdm(total = len(emb2d)) as pbar:
            for i, emb in enumerate(emb2d):
                emb_arr = emb.tolist()
                instance_id = self.idx2id_all[i]
                model_code = instance_id.split('-')[0]
                if model_code != 'img':
                    instance_id = '-'.join(instance_id.split('-')[1:])
                if model_code not in self.emb2d:
                    self.emb2d[model_code] = {}
                self.emb2d[model_code][instance_id] = emb_arr
                pbar.update(1)
        toc = time()
        log = '2D Projection: {:.2f} sec'.format(toc - tic)
        self.write_log(log)

    
    def sample_points(self):
        if self.args.model_for_emb_space == 'base':
            num_base_instances = len(self.X)
            rand_indices = np.random.choice(
                num_base_instances, 
                size=int(self.args.sample_rate * num_base_instances), 
                replace=False
            )
            sampled_X = self.X[rand_indices]
        elif self.args.model_for_emb_space == 'all':
            num_base_instances = self.num_instances
            rand_indices = np.random.choice(
                num_base_instances, 
                size=int(self.args.sample_rate * num_base_instances), 
                replace=False
            )
            sampled_X = self.X_all[rand_indices]
        return sampled_X

    
    def save_results(self):
        # Save 2d embedding
        dir_path = self.data_path.get_path('dim_reduction-dir')
        apdx = self.data_path.get_path('dim_reduction-apdx')
        for model_code in self.emb2d:
            file_path = os.path.join(
                dir_path, 'emb2d-{}-{}.json'.format(model_code, apdx)
            )
            self.save_json(
                self.emb2d[model_code], 
                file_path
            )
        
        # Save model code
        self.save_json(
            self.model_code_to_file_path, 
            self.data_path.get_path('dim_reduction-model_code')
        )


    """
    Handle external files (e.g., output, log, ...)
    """
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data


    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)


    def save_reducer(self):
        file_path = self.data_path.get_path('dim_reduction-reducer')
        joblib.dump(self.reducer, file_path)

    
    def load_reducer(self):
        file_path = self.data_path.get_path('dim_reduction-reducer')
        self.reducer = joblib.load(filename)


    def write_first_log(self):
        hyperpara_setting = self.data_path.gen_act_setting_str(
            'dim_reduction', '\n'
        )
        
        log = 'Dimensionality reduction\n\n'
        log += 'emb_set_dir: {}\n'.format(self.args.emb_set_dir)
        log += hyperpara_setting + '\n\n'
        self.write_log(log, False)


    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('dim_reduction-log'), log_opt) as f:
            f.write(log + '\n')
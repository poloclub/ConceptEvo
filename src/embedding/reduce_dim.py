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

        self.X = None
        self.X_all = None
        self.idx2id = {}
        self.idx2id_all = {}
        self.num_instances = 0
        self.base_model_nickname = ''
    
        self.is_reducer_given = False    
        self.reducer = None

        self.emb = {}
        self.emb2d = {}

    """
    Wrapper function called by main.py
    """
    def reduce_dim(self):
        self.emb_set_dir = self.data_path.get_path('emb_nd-dir')
        self.init_reducer()
        self.load_embedding()
        self.run_dim_reduction()
        self.save_results()

    """
    Initial setting
    """
    def init_reducer(self):
        self.is_reducer_given = os.path.exists(
            self.data_path.get_path('emb2d-reducer')
        )
        if self.is_reducer_given:
            self.load_reducer()
        elif self.args.dim_reduction == 'UMAP':
            self.reducer = umap.UMAP(n_components=2)
        else:
            err = 'An invalid dimensionality reduction method is given: '
            err += str(self.args.dim_reduction)
            raise ValueError(err)

    def load_embedding(self):
        # Collect file paths for all models' embedding
        proj_emb_dir = self.data_path.get_path('emb_nd-dir')
        proj_emb_files = os.listdir(proj_emb_dir)
        proj_emb_files = [
            os.path.join(proj_emb_dir, f)
            for f in proj_emb_files if 'proj_emb' in f
        ]
        file_paths = [self.data_path.get_path('neuron_emb')] \
            + sorted(proj_emb_files)
        
        # Load embedding of each model
        for file_path in file_paths:
            # Model nickname
            file_name = os.path.basename(file_path)
            if file_name == 'emb.json':
                model_nickname = self.args.basemodel_nickname
            else:
                model_nickname = file_name[9:-5]
            model_nickname = model_nickname.replace('-', '_')

            # Load embedding
            self.emb[model_nickname] = self.load_json(file_path)
            self.num_instances += len(self.emb[model_nickname])

        # Initialize matrix X for all neurons' vector
        num_base_instances = len(self.emb[self.args.basemodel_nickname])
        self.X = np.zeros((num_base_instances, self.args.dim))
        self.X_all = np.zeros((self.num_instances, self.args.dim))

        # Generate X
        idx = 0
        for model_nickname in self.emb:
            for neuron_i, neuron in enumerate(self.emb[model_nickname]):
                neuron_id = '{}-{}'.format(model_nickname, neuron)
                self.idx2id_all[idx] = neuron_id
                self.X_all[idx] = self.emb[model_nickname][neuron]
                if model_nickname == self.base_model_nickname:
                    self.X[neuron_i] = self.emb[model_nickname][neuron]
                    self.idx2id[neuron_i] = neuron_id
                idx += 1

        self.write_log('Embedding loaded')
    
    """
    Project the embdding to 2D
    """
    def run_dim_reduction(self):
        self.write_first_log()

        # Fit the reducer if it is not given
        if not self.is_reducer_given:
            # Fit the reducer and get all 2d embeddings
            tic = time()
            fitted_emb2d = self.reducer.fit_transform(self.X)
            emb2d = self.reducer.transform(self.X_all)
            toc = time()
            log = 'Fit and transform: {:.2f} sec'.format(toc - tic)
            self.write_log(log)

            # Save the reducer
            tic = time()
            self.save_reducer()
            toc = time()
            log = 'Save reducer: {:.2f} sec'.format(toc - tic)
            self.write_log(log)
        else:
            # Get embedding
            emb2d = self.reducer.transform(self.X_all)

        # Parse embeddings into self.emb2d
        tic = time()
        with tqdm(total = len(emb2d)) as pbar:
            for i, emb in enumerate(emb2d):
                emb_arr = emb.tolist()
                instance_id = self.idx2id_all[i]
                model_nickname = instance_id.split('-')[0]
                neuron_id = '-'.join(instance_id.split('-')[1:])
                if model_nickname not in self.emb2d:
                    self.emb2d[model_nickname] = {}
                self.emb2d[model_nickname][neuron_id] = emb_arr
                pbar.update(1)
        toc = time()
        log = '2D Projection: {:.2f} sec'.format(toc - tic)
        self.write_log(log)
    
    def save_results(self):
        dir_path = self.data_path.get_path('emb2d-dir')
        for model_nickname in self.emb2d:
            file_path = os.path.join(
                dir_path, 'emb2d-{}.json'.format(model_nickname)
            )
            self.save_json(self.emb2d[model_code], file_path)

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
        file_path = self.data_path.get_path('emb2d-reducer')
        joblib.dump(self.reducer, file_path)

    def load_reducer(self):
        file_path = self.data_path.get_path('emb2d-reducer')
        self.reducer = joblib.load(file_path)

    def write_first_log(self):
        log = 'Dimensionality reduction\n'
        self.write_log(log, False)

    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('emb2d-log'), log_opt) as f:
            f.write(log + '\n')

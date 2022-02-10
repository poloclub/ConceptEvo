import os
import json
import umap
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
        self.idx2id = {}
        self.num_instances = 0
    
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
        i = 0
        for dirpath, dnames, fnames in os.walk(self.args.emb_set_dir):
            for f in fnames:
                if '-log-' in f:
                    continue
                if 'emb2d' in f:
                    continue
                
                file_path = os.path.join(dirpath, f)
                if f.endswith('.json'):
                    # Assign model code
                    model_code = 'model_{}'.format(i)
                    self.model_code_to_file_path[model_code] = file_path
                    i += 1

                    # Load embedding
                    self.emb[model_code] = self.load_json(file_path)
                    self.num_instances += len(self.emb[model_code])

                elif f.endswith('.txt') and ('img_emb' in f):
                    # Assign model code
                    model_code = 'img'
                    self.model_code_to_file_path[model_code] = file_path

                    # Load embedding
                    self.emb[model_code] = np.loadtxt(file_path)
                    self.num_instances += len(self.emb[model_code])

        # Generate matrix X for all neurons' vectors
        self.X = np.zeros((self.num_instances, self.args.dim))
        idx = 0
        for model_code in self.emb:
            if 'img' in model_code:
                num_imgs = len(self.emb[model_code])
                for i in range(num_imgs):
                    self.idx2id[idx] = '{}-{}'.format(model_code, i)
                    idx += 1
            else:
                for neuron in self.emb[model_code]:
                    self.idx2id[idx] = '{}-{}'.format(model_code, neuron)
                    idx += 1

    
    """
    Project the embdding to 2D
    """
    def run_dim_reduction(self):
        self.write_first_log()

        # Fit reducer
        tic = time()
        sampled_X = self.sample_points()
        self.reducer.fit_transform(sampled_X)
        toc = time()
        log = 'Fit reducer: {:.2f} sec'.format(toc - tic)
        self.write_log(log)

        # Get 2D vectors of all neurons for all epochs
        tic = time()
        emb2d = self.reducer.transform(self.X).tolist()
        for i, emb_arr in enumerate(emb2d):
            instance_id = self.idx2id[i]
            self.emb2d[instance_id] = emb_arr
        toc = time()
        log = '2D Projection: {:.2f} sec'.format(toc - tic)
        self.write_log(log)

    
    def sample_points(self):
        rand_indices = np.random.choice(
            self.num_instances, 
            size=int(self.args.sample_rate * self.num_instances), 
            replace=False
        )
        sampled_X = self.X[rand_indices]
        return sampled_X

    
    def save_results(self):
        self.save_json(
            self.emb2d, 
            self.data_path.get_path('dim_reduction')
        )
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
import json
import umap
import numpy as np
from tqdm import tqdm
from time import time

class Reducer:
    """Dimensionality reduction"""

    # TODO: Need to update this class later

    """
    Constructor
    """
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path
    
        self.reducer = None
        self.emb = {}
        self.emb2d = {}
        self.X = None
        self.idx2id = {}

        self.num_neurons = -1


    """
    Wrapper function called by main.py
    """
    def reduce_dim(self):
        self.init_reducer()
        self.load_embedding()
        self.run_dim_reduction()
        self.save_json(self.emb2d, self.data_path.get_path('emb2d'))


    """
    Initial setting
    """
    def init_reducer(self):
        if self.args.dim_reduction == 'UMAP':
            self.reducer = umap.UMAP(n_components=2)

    
    def load_embedding(self):
        # Load embedding vectors from files
        for epoch in range(self.args.num_epochs):
            self.emb[epoch] = self.load_json(
                self.data_path.get_path('emb', epoch)
            )

        # Generate matrix X for all neurons' vectors
        self.num_neurons = len(self.emb[0])
        self.X = np.zeros((
            self.num_neurons * self.args.num_epochs, 
            self.args.dim
        ))
        for epoch in range(self.args.num_epochs):
            for i, neuron in enumerate(self.emb[epoch]):
                idx = (epoch * self.num_neurons) + i
                self.X[idx] = self.emb[epoch][neuron][:]
                self.idx2id[idx] = neuron

    
    """
    Project the embdding to 2D
    """
    def run_dim_reduction(self):
        # Fit reducer
        tic = time()
        sampled_X = self.sample_points()
        self.reducer.fit_transform(sampled_X)
        toc = time()
        log = 'Fit reducer: {:.2f} sec'.format(toc - tic)
        self.write_log(log)

        # Get 2D vectors of all neurons for all epochs
        total = self.args.num_emb_epochs * self.num_neurons
        with tqdm(total=total) as pbar:
            for epoch in range(self.args.num_epochs):
                self.emb2d[epoch] = {}
                f, t = self.num_neurons * epoch, self.num_neurons * (epoch + 1)
                emb2d_epoch = self.reducer.transform(self.X[f: t]).tolist()
                for i, emb_arr in enumerate(emb2d_epoch):
                    neuron_id = self.idx2id[f + i]
                    self.emb2d[epoch][neuron_id] = emb_arr
                    pbar.update(1)
    

    def sample_points(self):
        num_points = self.num_neurons * self.args.num_epochs
        rand_indices = np.random.choice(
            num_points, 
            size=int(self.args.sample_rate * num_points), 
            replace=False
        )
        sampled_X = self.X[rand_indices]
        return sampled_X


    """
    Handle external files (e.g., output, log, ...)
    """
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('dim-reduction'), log_opt) as f:
            f.write(log + '\n')


    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data


    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)
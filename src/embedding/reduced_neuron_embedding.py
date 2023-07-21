import os
import json
import umap
import random
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches


class ReducedNeuronEmb:
    """Dimensionality reduction"""

    """
    Constructor
    """
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path
        self.basemodel_nickname = ''

        self.X = {}
        self.X_2d = {}
        self.emb = {}
        self.emb_2d = {}
        self.id2idx = {}
        self.idx2id = {}
        
        self.reducer = None

    """
    A wrapper function called by main.py
    """
    def compute_reduced_embedding(self):
        self.write_first_log()
        self.load_base_embedding()
        self.load_proj_embedding()
        self.compute_reducer()
        self.reduce_embedding()
        self.save_result()

    """
    Load embedding data
    """
    def load_base_embedding(self):
        p = self.data_path.get_path('neuron_embedding_path')
        self.load_embedding(p)

        name = self.get_model_name_from_emb_file(p)
        self.basemodel_nickname = name

    def load_proj_embedding(self):
        dir_p = self.data_path.get_path('proj_embedding_dir_path')
        for fn in os.listdir(dir_p):
            if fn.endswith('.json'):
                p = os.path.join(dir_p, fn)
                self.load_embedding(p)

    def get_model_name_from_emb_file(self, p):
        prefix = 'neuron_emb_' if 'neuron_emb' in p else 'proj_emb_'
        suffix = '.json'
        fn = os.path.basename(p)
        return fn[len(prefix): -len(suffix)]

    def load_embedding(self, emb_path):
        emb = self.load_json(emb_path)
        X, id2idx, idx2id = self.convert_emb_to_X(emb)

        model_nickname = self.get_model_name_from_emb_file(emb_path)

        self.emb[model_nickname] = emb
        self.X[model_nickname] = X
        self.id2idx[model_nickname] = id2idx
        self.idx2id[model_nickname] = idx2id

    def convert_emb_to_X(self, emb):
        dim = len(emb[list(emb.keys())[0]])
        X, id2idx, idx2id = np.zeros((len(emb), dim)), {}, {}
        for i, neuron_id in enumerate(emb):
            vec = emb[neuron_id]
            X[i] = vec
            id2idx[neuron_id] = i
            idx2id[i] = neuron_id
        return X, id2idx, idx2id

    """
    UMAP reducer
    """
    def compute_reducer(self):
        tic = time()
        X = self.X[self.basemodel_nickname]
        self.reducer = umap.UMAP(n_components=2)
        self.reducer = self.reducer.fit(X)
        toc = time()
        log = f'Compute reducer: {toc - tic:.2f} sec'
        self.write_log(log)

    """
    Compute reduced embedding
    """
    def reduce_embedding(self):
        with tqdm(total=len(self.X)) as pbar:
            for model_nickname in self.X:
                tic = time()
                
                X = self.X[model_nickname]
                X_2d = self.reducer.transform(X)
                idx2id = self.idx2id[model_nickname]
                emb_2d = self.convert_X_to_emb(X_2d, idx2id)
                self.X_2d[model_nickname] = X_2d
                self.emb_2d[model_nickname] = emb_2d
                
                toc = time()
                log = f'Reduce embedding, {model_nickname}: '
                log += f'{toc - tic:.2f} sec'
                self.write_log(log)
                
                pbar.update(1)

    def convert_X_to_emb(self, X_2d, idx2id):
        emb_2d = {}
        for i, x_2d in enumerate(X_2d):
            neuron_id = idx2id[i]
            vec = x_2d.tolist()
            emb_2d[neuron_id] = vec
        return emb_2d

    """
    Save the result
    """
    def save_result(self):
        dir_path = self.data_path.get_path('reduced_emb')
        for model_nickname in self.emb_2d:
            # Save 2d embedding
            p = os.path.join(
                dir_path, f'reduced_emb_{model_nickname}.json'
            )
            self.save_json(self.emb_2d[model_nickname], p)

            # Save 2d embedding vis
            self.save_embedding_vis(model_nickname)
            
    def save_embedding_vis(self, model_nickname):
        # Show scatter plot
        X_2d = self.X_2d[model_nickname]
        plt.scatter(
            X_2d[:, 0], 
            X_2d[:, 1], 
            s=1, 
            color='lightgray'
        )

        # Load color mapping for sample neurons
        p = self.data_path.get_path('color_map')
        color_map = {}
        if p is not None:
            color_map = self.load_json(p)

        # Show example neurons
        p = self.data_path.get_path('sample_neuron')
        id2idx = self.id2idx[model_nickname]
        if p is not None:
            sample_neuron = self.load_json(p)
            if model_nickname in sample_neuron:
                # Highlight sample neurons
                sample_neuron_data = sample_neuron[model_nickname]
                for key in sample_neuron_data:
                    neurons = sample_neuron_data[key]
                    X_2d_sample = np.zeros((len(neurons), 2))
                    for i, neuron in enumerate(neurons):
                        if neuron in id2idx:
                            X_2d_sample[i] = X_2d[id2idx[neuron]]
                        else:
                            log = f'{neuron} not in {model_nickname}\n'
                            self.write_log(log)

                    if key in color_map:
                        color = color_map[key]
                    else:
                        color = self.generate_random_color_hex()
                        color_map[key] = color
                    plt.scatter(
                        X_2d_sample[:, 0], 
                        X_2d_sample[:, 1], 
                        s=10, 
                        color=color
                    )

                # Legend
                handles = [mpatches.Patch(color=color_map[key]) for key in color_map]
                labels = list(color_map.keys())
                plt.legend(handles, labels)
        
        # Save the figure
        p = os.path.join(
            self.data_path.get_path('reduced_emb'),
            f'reduced_emb_{model_nickname}.pdf'
        )
        plt.savefig(p)
        plt.clf()

    def generate_random_color_hex(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hex_code = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        return hex_code

    """
    Utils
    """
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def write_first_log(self):
        log = 'Dimensionality reduction\n\n'
        log += f'neuron_embedding_path: {self.args.neuron_embedding_path}\n'
        log += f'proj_embedding_dir_path: {self.args.proj_embedding_dir_path}\n'
        log += f'reduced_embedding_sub_dir_name: {self.args.reduced_embedding_sub_dir_name}\n'
        self.write_log(log, False)

    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('reduced_emb_log'), log_opt) as f:
            f.write(log + '\n')

import json
import umap
import random
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

class ProjNeuronEmb:
    """
    Generate neuron embeddings projected on the unified semantic space
    """

    """
    Constructor
    """
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path

        if 'pretrained' in self.args.model_nickname:
            self.model_nickname = self.args.model_nickname
        else:
            self.model_nickname = f'{self.args.model_nickname}_{self.args.epoch}'

        self.stimulus = {}
        self.base_stimulus = {}
        self.vocab = {}
        self.img_emb = None
        self.neuron_emb = {}
    
    """
    A wrapper function called by main.py
    """
    def compute_projected_neuron_embedding(self):
        self.img_emb_path = self.data_path.get_path('img_emb')
        self.load_img_emb()
        self.load_stimulus()
        self.gen_vocab()
        self.init_neuron_embedding()
        self.project_neuron_embedding()
        self.save_embedding()
        self.save_embedding_vis()

    """
    Load data
    """
    def load_img_emb(self):
        self.img_emb = np.loadtxt(self.img_emb_path)

    def load_stimulus(self):
        stimulus_path = self.data_path.get_path('stimulus')
        self.stimulus = self.load_json(stimulus_path)
        base_stimulus_path = self.data_path.get_path('base_stimulus')
        self.base_stimulus = self.load_json(base_stimulus_path)

    def gen_vocab(self):
        for layer in self.base_stimulus:
            for imgs in self.base_stimulus[layer]:
                for img in imgs:
                    if img not in self.vocab:
                        self.vocab[img] = 0
                    self.vocab[img] += 1

    """
    Compute projected neuron embedding
    """
    def init_neuron_embedding(self):
        for layer in self.stimulus:
            num_neurons = len(self.stimulus[layer])
            for i in range(num_neurons):
                neuron = f'{layer}-{i}'
                self.neuron_emb[neuron] = np.zeros(self.args.dim)
                # self.neuron_emb[neuron] = np.random.rand(self.args.dim) - 0.5

    def get_stimulus_of_neuron(self, neuron):
        layer, neuron_idx = neuron.split('-')
        neuron_idx = int(neuron_idx)
        return self.stimulus[layer][neuron_idx]

    def compute_approx_neuron_vec(self, X_n):
        vec_sum = np.zeros(self.args.dim)
        num_imgs_in_vocab = 0
        for x in X_n:
            if x in self.vocab:
                vec_sum += self.img_emb[x]
                num_imgs_in_vocab += 1
        if num_imgs_in_vocab > 0:
            return vec_sum / num_imgs_in_vocab, num_imgs_in_vocab
        else:
            return vec_sum, num_imgs_in_vocab
    
    def project_neuron_embedding(self):
        self.write_first_log()
        tic = time()
        no_vocab_neurons = 0
        for neuron in self.neuron_emb:
            stimulus = self.get_stimulus_of_neuron(neuron)
            v_neuron, num_imgs_in_vocab = self.compute_approx_neuron_vec(stimulus)
            self.neuron_emb[neuron] = v_neuron
        toc = time()
        self.write_log('running_time: {} sec'.format(toc - tic))
        self.write_log(f'# no_vocab_neurons: {no_vocab_neurons}')
        self.write_log(f'# total neurons: {len(self.neuron_emb)}')

    def save_embedding(self):
        for neuron in self.neuron_emb:
            self.neuron_emb[neuron] = [
                round(x, 3) for x in self.neuron_emb[neuron].tolist()
            ]

        self.save_json(
            self.neuron_emb, self.data_path.get_path('proj_emb')
        )

    def get_emb_arr(self):
        X = np.zeros((len(self.neuron_emb), self.args.dim))
        id2idx = {}
        for i, neuron in enumerate(self.neuron_emb):
            X[i] = self.neuron_emb[neuron]
            id2idx[neuron] = i
        return X, id2idx

    def save_embedding_vis(self):
        # Get 2d embedding
        X, id2idx = self.get_emb_arr()
        reducer = umap.UMAP(n_components=2)
        reducer = reducer.fit(X)
        X_2d = reducer.transform(X)

        # Show scatter plot
        plt.scatter(X_2d[:, 0], X_2d[:, 1], s=1, color='lightgray')

        # Load color mapping for sample neurons
        p = self.data_path.get_path('color_map')
        color_map = {}
        if p is not None:
            color_map = self.load_json(p)

        # Show example neurons
        p = self.data_path.get_path('sample_neuron')
        if p is not None:
            sample_neuron = self.load_json(p)
            if self.model_nickname in sample_neuron:
                # Highlight sample neurons
                sample_neuron_data = sample_neuron[self.model_nickname]
                for key in sample_neuron_data:
                    neurons = sample_neuron_data[key]
                    X_2d_sample = np.zeros((len(neurons), 2))
                    for i, neuron in enumerate(neurons):
                        X_2d_sample[i] = X_2d[id2idx[neuron]]

                    if key in color_map:
                        color = color_map[key]
                    else:
                        color = self.generate_random_color_hex()
                        color_map[key] = color
                    plt.scatter(X_2d_sample[:, 0], X_2d_sample[:, 1], s=10, color=color)

                # Legend
                handles = [mpatches.Patch(color=color_map[key]) for key in color_map]
                labels = list(color_map.keys())
                plt.legend(handles, labels)

        plt.savefig(self.data_path.get_path('proj_emb_vis'))

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
        log = 'Project Embedding\n\n'
        log += f'model_nickname: {self.args.model_nickname}\n'
        log += f'epoch: {self.args.epoch}\n'
        log += f'dim: {self.args.dim}\n'
        log += f'img_embedding_path: {self.args.img_embedding_path}\n'
        log += f'stimulus path: {self.data_path.get_path("stimulus")}\n'
        log += f'proj_embedding_sub_dir_name: {self.args.proj_embedding_sub_dir_name}'
        self.write_log(log, False)
   
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('proj_emb_log'), log_opt) as f:
            f.write(log + '\n')

import json
import umap
import random
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

class Emb:
    """Generate neuron embeddings"""

    """
    Constructor
    """
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path

        self.model_nickname = f'{self.args.model_nickname}_{self.args.epoch}'

        self.stimulus = {}
        self.co_act_neurons = {}
        self.emb = {}
        self.neuron_sample_pool = []

        self.num_total_neurons = 0

    """
    A wrapper function called by main.py
    """
    def compute_neuron_embedding(self):
        self.compute_co_activated_neurons()
        self.compute_embedding_of_neurons()
        self.save_embedding_vis()

    """
    Find co-activated neurons
    """
    def compute_co_activated_neurons(self):
        # Load stimulus
        self.load_stimulus()

        # Find neurons that are highly activated by each input
        tic, co_act_neurons = time(), {}
        for layer_name in self.stimulus:
            for i, neuron_stimulus in enumerate(self.stimulus[layer_name]):
                neuron_id = f'{layer_name}-{i}'
                for img in neuron_stimulus:
                    if img not in co_act_neurons:
                        co_act_neurons[img] = []
                    co_act_neurons[img].append(neuron_id)

        # Keep only images that activate multiple neurons
        for img in co_act_neurons:
            if len(co_act_neurons[img]) > 1:
                self.co_act_neurons[img] = co_act_neurons[img]

        # Save co-activated neurons
        self.save_co_activated_neurons()

    def load_stimulus(self):
        stimulus_path = self.data_path.get_path('stimulus')
        stimulus = self.load_json(stimulus_path)
        self.stimulus = stimulus

    def save_co_activated_neurons(self):
        file_path = self.data_path.get_path('co_act')
        self.save_json(self.co_act_neurons, file_path)
    
    def load_co_activated_neurons(self):
        if len(self.co_act_neurons) <= 0:
            file_path = self.data_path.get_path('co_act')
            self.co_act_neurons = self.load_json(file_path)

    """
    Compute neuron embedding
    """
    def compute_embedding_of_neurons(self):
        # Write log header
        self.write_first_log()

        # Initialize neuron embedding
        self.init_neuron_embedding()

        # Load co-activated neurons
        self.load_co_activated_neurons()
        co_act_neurons = [
            self.co_act_neurons[img]
            for img in self.co_act_neurons
            if len(self.co_act_neurons[img]) > 1
        ]

        # Learn neuron embedding
        tic, total = time(), self.args.num_emb_epochs * len(self.co_act_neurons)
        with tqdm(total=total) as pbar:
            for emb_epoch in range(self.args.num_emb_epochs):
                grad_l2 = 0
                for neurons in co_act_neurons:
                    # Shuffle neurons
                    np.random.shuffle(neurons)

                    # Compute neuron embedding
                    for i, neuron in enumerate(neurons[:-1]):
                        next_neuron = neurons[i + 1]
                        v_n = self.emb[neuron]
                        v_m = self.emb[next_neuron]

                        # 1 - sigma(v_n \dot v_m)
                        coeff = 1 - self.sigmoid(v_n.dot(v_m))

                        # Update gradients for v_n
                        g_n = coeff * v_m
                        for neg_i in range(self.args.num_emb_negs):
                            neg_neuron = self.sample_neg_neuron()
                            v_r = self.emb[neg_neuron]
                            g_n -= self.sigmoid(v_n.dot(v_r)) * v_r

                        # Update gradients for v_m
                        g_m = coeff * v_n
                        for neg_i in range(self.args.num_emb_negs):
                            neg_neuron = self.sample_neg_neuron()
                            v_r = self.emb[neg_neuron]
                            g_m -= self.sigmoid(v_m.dot(v_r)) * v_r
                        
                        # Regularization
                        # lb = 0.1
                        # g_n -= lb * v_n
                        # g_m -= lb * v_m

                        # Update embedding
                        self.emb[neuron] += self.args.lr_emb * g_n
                        self.emb[next_neuron] += self.args.lr_emb * g_m

                        grad_l2 += g_n.dot(g_n) + g_m.dot(g_m)

                    pbar.update(1)

                if grad_l2 <= 0.01:
                    break

        # Save neuron embedding
        self.save_embedding()

        # Write embedding log
        log = 'runnig_time: {}sec\n'.format(time() - tic)
        log += 'grad_l2: {}\n'.format(grad_l2)
        self.write_log(log)

    def compute_err(self):
        err = 0
        imgs = list(self.co_act_neurons.keys())
        
        for img in imgs:
            neurons = self.co_act_neurons[img]
            for i, neuron in enumerate(neurons[:-1]):
                next_neuron = neurons[i + 1]
                v_neuron = self.emb[neuron]
                v_next_neuron = self.emb[next_neuron]
                err += -np.log(self.sigmoid(v_neuron.dot(v_next_neuron)))
        return err

    def get_num_samples(self, n):
        num_samples = int(self.args.t * np.sqrt(n))
        num_samples = np.min([num_samples, n]) 
        return num_samples

    def init_neuron_embedding(self):
        for layer in self.stimulus:
            for i, neuron_stimulus in enumerate(self.stimulus[layer]):
                neuron_id = f'{layer}-{i}'
                self.emb[neuron_id] = np.random.rand(self.args.dim) - 0.5
   
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_neg_neuron(self):
        if self.num_total_neurons == 0:
            self.neuron_sample_pool = list(self.emb.keys())
            self.num_total_neurons = len(self.neuron_sample_pool)

        n = np.random.randint(self.num_total_neurons)
        neuron = self.neuron_sample_pool[n]

        return neuron

    def save_embedding(self, epoch=None):
        path = self.data_path.get_path('neuron_emb')
        emb_to_save = {}
        for neuron in self.emb:
            emb_to_save[neuron] = self.emb[neuron].tolist()
        self.save_json(emb_to_save, path)

    def get_emb_arr(self):
        X = np.zeros((len(self.emb), self.args.dim))
        id2idx = {}
        for i, neuron in enumerate(self.emb):
            X[i] = self.emb[neuron]
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


        plt.savefig(self.data_path.get_path('neuron_emb_vis'))

    def generate_random_color_hex(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hex_code = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        return hex_code

    """
    Handle external files (e.g., output, log, ...)
    """
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('neuron_emb_log'), log_opt) as f:
            f.write(log + '\n')

    def write_first_log(self):
        log = 'Compute neuron Embedding\n\n'
        log += f'model_nickname: {self.model_nickname}\n'
        log += f'model_path: {self.data_path.get_path("model_path")}\n'
        log += self.data_path.data_path_neuron_embedding.para_info
        log += '\n'
        self.write_log(log, False)
            
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)

import os
import json
from time import time

import numpy as np
from tqdm import tqdm


class ImageEmbMatFac:
    """Generate image embeddings that are not represented by the base model"""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path

        self.num_imgs = model.num_training_imgs
        self.num_neurons = -1

        self.layer_act = None
        self.stimulus = {}
        self.vocab = {}

        self.W = None

        self.img_emb = None

    """
    A wrapper function called in main.py
    """
    def compute_img_embedding_mat_fac(self):
        self.write_first_log()
        self.gen_vocab()
        self.load_img_emb()
        self.load_layer_act()
        self.normalize_layer_act()
        self.init_weight()
        self.compute_weight()
        self.compute_img_emb()
        self.save_weight()
        self.save_img_emb()

    """
    Utils
    """
    def gen_vocab(self):
        self.load_stimulus()
        for layer in self.stimulus:
            for neuron_stimuli in self.stimulus[layer]:
                for img in neuron_stimuli:
                    if img not in self.vocab:
                        self.vocab[img] = 0
                    self.vocab[img] += 1
        self.write_log(f'Number of imgs in vocab: {len(self.vocab)}')

    def load_stimulus(self):
        stimulus_path = self.data_path.get_path('stimulus')
        self.stimulus = self.load_json(stimulus_path)
        for layer in self.stimulus:
            for neuron, neuron_imgs in enumerate(self.stimulus[layer]):
                self.stimulus[layer][neuron] = neuron_imgs[:self.args.k]

    def load_img_emb(self):
        p = self.data_path.get_path('img_emb')
        self.img_emb = np.loadtxt(p)

    def load_layer_act(self):
        log = 'Load layer activation'
        print(log)
        tic = time()
        p = self.data_path.get_path('layer_act')
        self.layer_act = np.loadtxt(p)
        self.num_neurons = self.layer_act.shape[-1]
        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')

    def normalize_layer_act(self):
        log = 'Normalize layer activation'
        m = np.min(self.layer_act)
        M = np.max(self.layer_act)
        a = 2 / (M - m)
        b = -(m + M) / (M - m)
        self.layer_act = 0.5 * (self.layer_act * a + b)

    """
    Compute and save image embedding
    """
    def init_weight(self):
        self.W = np.random.random((self.args.dim, self.num_neurons)) - 0.5

    def compute_weight(self):
        tic = time()
        total = self.args.num_epochs_mat_fac * self.args.dim * len(self.vocab)
        prev_rmse = 100000
        with tqdm(total=total) as pbar:
            for epoch in range(self.args.num_epochs_mat_fac):
                avg_rmse = 0
                for d in range(self.args.dim):
                    # grad = np.zeros(self.num_neurons)
                    rmse = 0
                    for img in self.vocab:
                        l_img = self.layer_act[img]
                        dot = l_img.dot(self.W[d])
                        loss = dot - self.img_emb[img][d]
                        grad = loss * l_img
                        rmse += loss ** 2
                        self.W[d] -= self.args.lr_mat_fac * grad
                        pbar.update(1)
                    rmse = np.sqrt(rmse / len(self.vocab))
                    avg_rmse += rmse
                avg_rmse = avg_rmse / self.args.dim

                if avg_rmse <= 0.001:
                    break
                if prev_rmse < avg_rmse:
                    break
                prev_rmse = avg_rmse

                toc = time()
                log = f'epoch={epoch}, '
                log += f'avg_rmse={avg_rmse}, '
                log += f'cum_time={toc - tic:.2f} sec'
                self.write_log(log)

    def compute_img_emb(self):
        W_t = self.W.transpose()
        with tqdm(total=self.num_imgs) as pbar:
            for img in range(self.num_imgs):
                if img not in self.vocab:
                    est_v_img = np.matmul(self.layer_act[img], W_t)
                    self.img_emb[img] = est_v_img
                pbar.update(1)

    def save_weight(self):
        file_path = self.data_path.get_path('weight_mat_fac')
        np.savetxt(file_path, self.W, fmt='%.3f')

    def save_img_emb(self):
        file_path = self.data_path.get_path('img_emb_mat_fac')
        np.savetxt(file_path, self.img_emb, fmt='%.3f')

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
        hyp = self.data_path.gen_act_setting_str('img_emb_mat_fac', '\n')
        log = 'Image embedding with matrix factorization\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += hyp + '\n\n'
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        p = self.data_path.get_path('img_emb_mat_fac-log')
        with open(p, log_opt) as f:
            f.write(log + '\n')

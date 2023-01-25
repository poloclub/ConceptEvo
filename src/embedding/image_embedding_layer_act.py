import os
import json
from time import time

import numpy as np
from tqdm import tqdm


class ImageEmbLayerAct:
    """Generate image embeddings that are not represented by the base model"""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path

        # TODO: How to set layers?
        # self.layers = [model.layers[-1]]
        # self.layers = [layer['name'] for layer in self.layers]
        self.layers = ['Sequential_0_Conv2d_34']
        self.num_imgs = model.num_training_imgs

        self.layer_acts = {}
        self.stimulus = {}
        self.vocab = {}
        self.added_vocab = {}

        self.img_emb = None
        self.top_neurons_by_img = {}
        self.co_activating_imgs_by_neuron = {}

        self.img_pairs = {}

    """
    A wrapper function called in main.py
    """
    def compute_img_embedding_with_layer_act(self):
        self.write_first_log()
        self.gen_vocab()
        self.load_img_emb()
        self.load_layer_act()
        self.find_top_neurons_by_img()
        self.find_co_activating_imgs_by_neuron()
        self.sample_img_pairs()
        self.init_img_emb()
        self.compute_img_emb()
        self.save_img_emb()
        self.save_added_vocab()

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
        dir_path = self.data_path.get_path('layer_act_dir')
        for layer in self.layers:
            p = os.path.join(dir_path, layer)
            p = os.path.join(p, 'img_emb.txt')
            self.layer_acts[layer] = np.loadtxt(p)
        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')

    def find_top_neurons_by_img(self):
        log = f'Find top {self.args.k} activated neurons by image'
        print(log)
        tic = time()
        total = len(self.layers) * self.num_imgs
        with tqdm(total=total) as pbar:
            for layer in self.layers:
                self.top_neurons_by_img[layer] = {}
                for img_i, img_v in enumerate(self.layer_acts[layer]):
                    neuron_idxs = np.argsort(-img_v)[:self.args.k]
                    neuron_ids = [f'{layer}-{idx}' for idx in neuron_idxs]
                    self.top_neurons_by_img[layer][img_i] = neuron_ids
                    pbar.update(1)
        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')
    
    def find_co_activating_imgs_by_neuron(self):
        log = 'Find co-activating images by neurons'
        print(log)
        tic = time()
        total = len(self.layers) * self.num_imgs
        with tqdm(toal=total) as pbar:
            for layer in self.layers:
                for i in self.top_neurons_by_img[layer]:
                    for neuron_id in self.top_neurons_by_img[layer][i]:
                        if neuron_id not in self.co_activating_imgs_by_neuron:
                            self.co_activating_imgs_by_neuron[neuron_id] = []
                        self.co_activating_imgs_by_neuron[neuron_id].append(i)
                    pbar.update(1)
        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')
    
    def sample_img_pairs(self):
        log = 'Sample image pairs'
        print(log)
        tic = time()
        total = len(self.co_activating_imgs_by_neuron)
        with tqdm(total=total) as pbar:
            for neuron_id in self.co_activating_imgs_by_neuron:
                imgs = self.co_activating_imgs_by_neuron[neuron_id]
                for i, img_i in enumerate(imgs):
                    if i == len(imgs) - 1:
                        break
                    img_j = imgs[i + 1]
                    key = '-'.join(list(map(str, sorted([img_i, img_j]))))
                    if key not in self.img_pairs:
                        self.img_pairs[key] = 0
                    self.img_pairs[key] += 1
                pbar.update(1)
        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')

    def sample_rand_imgs(self):
        n = self.args.num_emb_negs_layer_act
        sampled_imgs = np.random.choice(self.num_imgs, n, replace=False)
        return sampled_imgs

    """
    Compute and save image embedding
    """
    def init_img_emb(self):
        self.img_emb = np.random.random((self.num_imgs, self.args.dim)) - 0.5
        # if self.start_from_pre_computed:
        #     file_path = self.data_path.get_path('img_emb_layer_act_from')
        #     self.img_emb = np.loadtxt(file_path)
        # else:
        #     self.img_emb \
        #         = np.random.random((self.num_imgs, self.args.dim)) - 0.5

    def compute_img_emb(self):
        tic = time()
        total = self.args.num_emb_epochs_layer_act * len(self.img_pairs)
        with tqdm(total=total) as pbar:
            for epoch in range(self.args.num_emb_epochs_layer_act):
                for pair in self.img_pairs:
                    img_i, img_j = pair.split('-')
                    img_i, img_j = int(img_i), int(img_j)
                    cnt = self.img_pairs[pair]
                    pbar.update(1)

                    if (img_i not in self.vocab) and (img_j not in self.vocab):
                        continue
                
                    # Get image vectors
                    v_i = self.img_emb[img_i]
                    v_j = self.img_emb[img_j]
                    coeff = 1 - sigmoid(v_i.dot(v_j))

                    # Update gradients for v_i
                    if img_i not in self.vocab:
                        g_i = coeff * cnt * v_j
                        rand_imgs = self.sample_rand_imgs()
                        for img_r in rand_imgs:
                            v_r = self.img_emb[img_r]
                            g_i -= self.sigmoid(v_i.dot(v_r)) * v_r
                        self.img_emb[img_i] += lr * g_i

                        if img_i not in self.added_vocab:
                            self.added_vocab[img_i] = 0
                        self.added_vocab[img_i] += 1

                    # Update gradients for v_j
                    if img_j not in self.vocab:
                        g_j = coeff * cnt * v_i
                        rand_imgs = self.sample_rand_imgs()
                        for img_r in rand_imgs:
                            v_r = self.img_emb[img_r]
                            g_j -= self.sigmoid(v_j.dot(v_r)) * v_r
                        self.img_emb[img_j] += lr * g_j

                        if img_j not in self.added_vocab:
                            self.added_vocab[img_j] = 0
                        self.added_vocab[img_j] += 1
                    break
                break

    def save_img_emb(self):
        file_path = self.data_path.get_path('img_emb_layer_act')
        np.savetxt(file_path, self.img_emb, fmt='%.3f')

    def save_added_vocab(self):
        p = self.data_path.get_path('added_vocab_layer_act')
        self.save_json(self.added_vocab, p)

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
        hyp = self.data_path.gen_act_setting_str('img_emb_layer_act', '\n')
        log = 'Image embedding with layer activation\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'from_iter_img_emb: {}\n'.format(self.args.from_iter_img_emb)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hyp + '\n\n'
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        p = self.data_path.get_path('img_emb_layer_act-log')
        with open(p, log_opt) as f:
            f.write(log + '\n')

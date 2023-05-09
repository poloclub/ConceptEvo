import os
import json
from time import time

import numpy as np
from tqdm import tqdm

class ImagePairs:
    """Generate image pairs"""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path
        self.layer = self.args.layer
        self.layer_act = {}
        self.top_neurons_by_img = {}
        self.co_imgs_of_neuron = {}
        self.img_pairs = {}
        self.sorted_img_pairs = {}

    """
    A wrapper function called in main.py
    """
    def compute_img_pairs(self):
        self.write_first_log()
        self.load_layer_act()
        self.find_top_neurons_by_img()
        self.find_co_activating_imgs_by_neuron()
        self.find_img_pairs()
        self.sort_img_pairs()
        self.save_sorted_img_pairs()
    
    """
    Utils
    """
    def load_layer_act(self):
        tic = time()

        p = self.data_path.get_path('layer_act')
        self.layer_act = np.loadtxt(p)

        toc = time()
        log = 'Load layer activation'
        self.write_log(f'{log}: {toc - tic:.2f} sec')

    """
    Compute image pairs
    """
    def find_top_neurons_by_img(self):
        tic = time()

        total = len(self.layer_act)
        with tqdm(total=total) as pbar:
            for img_i, img_v in enumerate(self.layer_act):
                neuron_idxs = np.argsort(-img_v)[:self.args.k]
                neuron_ids = [f'{self.layer}-{idx}' for idx in neuron_idxs]
                self.top_neurons_by_img[img_i] = neuron_ids
                pbar.update(1)
        
        toc = time()
        log = f'Find top {self.args.k} activated neurons by each image'
        self.write_log(f'{log}: {toc - tic:.2f} sec')
    
    def find_co_activating_imgs_by_neuron(self):
        tic = time()

        total = len(self.top_neurons_by_img)
        with tqdm(total=total) as pbar:
            for i in self.top_neurons_by_img:
                for neuron_id in self.top_neurons_by_img[i]:
                    if neuron_id not in self.co_imgs_of_neuron:
                        self.co_imgs_of_neuron[neuron_id] = []
                    self.co_imgs_of_neuron[neuron_id].append(i)
                pbar.update(1)

        toc = time()
        log = 'Find co-activating images by neurons'
        self.write_log(f'{log}: {toc - tic:.2f} sec')

    def find_img_pairs(self):
        tic = time()

        total = len(self.co_imgs_of_neuron) * self.args.num_epochs_co_act
        with tqdm(total=total) as pbar:
            for e in range(self.args.num_epochs_co_act):
                for neuron_id in self.co_imgs_of_neuron:
                    imgs = self.co_imgs_of_neuron[neuron_id]
                    np.random.shuffle(imgs)
                    for i, img_i in enumerate(imgs):
                        if i == len(imgs) - 1:
                            break
                        img_j = imgs[i + 1]

                        if img_i not in self.img_pairs:
                            self.img_pairs[img_i] = {}
                        if img_j not in self.img_pairs:
                            self.img_pairs[img_j] = {}

                        if img_j not in self.img_pairs[img_i]:
                            self.img_pairs[img_i][img_j] = 0
                        if img_i not in self.img_pairs[img_j]:
                            self.img_pairs[img_j][img_i] = 0
                        
                        self.img_pairs[img_i][img_j] += 1
                        self.img_pairs[img_j][img_i] += 1

                    pbar.update(1)

        toc = time()
        log = 'Find img_pairs'
        self.write_log(f'{log}: {toc - tic:.2f} sec')  

    def sort_img_pairs(self):
        for img_i in self.img_pairs:
            self.sorted_img_pairs[img_i] = self.sort_dict_by_val(
                self.img_pairs[img_i], reverse=True
            )
                
    def sort_dict_by_val(self, d, reverse=True):
        s = [
            [k, v] for k, v in 
            sorted(d.items(), key=lambda x: x[1], reverse=reverse)
        ]
        return s

    def save_sorted_img_pairs(self):
        p = self.data_path.get_path('img_pairs')
        self.save_json(self.sorted_img_pairs, p)

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
        log = 'Save image pairs\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'layer: {}\n'.format(self.args.layer)
        log += 'num_epochs_co_act: {}\n'.format(self.args.num_epochs_co_act)
        log += 'k: {}\n'.format(self.args.k)
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('img_pairs-log'), log_opt) as f:
            f.write(log + '\n')
    
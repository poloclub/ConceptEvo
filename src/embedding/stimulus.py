import json
from time import time

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.utils import *


class Stimulus:
    """
    Compute stimulus for each neuron
    """

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path

        self.model = model
        self.layers = []
        self.target_layers = []
        self.num_neurons = {}

        self.device = model.device
        self.training_dataset = None
        self.data_loader = None

        self.stimulus = {}

    """
    A wrapper function called by main.py
    """
    def compute_stimulus(self):
        self.init_setting()
        self.get_layer_info()
        self.find_stimulus()
        self.save_stimulus()

    """
    Initial setting
    """
    def init_setting(self):
        S = self.model.get_input_size()
        data_transform = transforms.Compose([
            transforms.Resize((S, S)),
            transforms.ToTensor(),
            transforms.Normalize(*self.model.input_normalization)
        ])

        self.training_dataset = datasets.ImageFolder(
            self.data_path.get_path('stimulus_image_path'),
            data_transform
        )

        self.data_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )

    def get_layer_info(self):
        self.layers = self.model.layers[:]
        self.target_layers = self.model.layers_for_ex_patch[:]
        self.num_neurons = self.model.num_neurons

    """
    Find stimulus (i.e., inputs that activate neurons the most) for each neuron
    """
    def find_stimulus(self):
        self.write_first_log()
        self.init_stimulus()
        tic, total = time(), len(self.data_loader)

        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                # Get input images in a batch and their labels
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Update stimulus for the first layer
                f_map = self.model.forward_one_layer(0, imgs)
                self.update_stimulus(
                    self.layers[0]['name'], f_map, batch_idx
                )

                # Update stimulus for remaining layers
                for i in range(1, len(self.layers) - 1):
                    try:
                        f_map = self.model.forward_one_layer(i, f_map)
                        self.update_stimulus(
                            self.layers[i]['name'], f_map, batch_idx
                        )

                    except RuntimeError:
                        log = f'Error in find_stimulus for '
                        log += self.layers[i]['name']
                        #  self.write_log(log)

                pbar.update(1)

        log_str = 'cumulative_time_sec: {:.2f}\n'.format(time() - tic)
        self.write_log(log_str)

    def init_stimulus(self):
        for layer_name in self.target_layers:
            self.stimulus[layer_name] = [
                TopKKeeper(self.args.topk_s) 
                for neuron in range(self.num_neurons[layer_name])
            ]
            
    def compute_feature_map(self, layer, prev_f_map, res_f_map=None):
        # Compute feature map. feature_map: [B, N, W, H]
        # where B is batch size and N is the number of neurons
        feature_map = layer(prev_f_map)
        if res_f_map is not None:
            feature_map = feature_map + res_f_map
        return feature_map

    def compute_max_act_of_feature_map(self, feature_map):
        # Get the maximum activation of the feature map. max_act: [B, N]
        # where B is batch size and N is the number of neurons
        return torch.max(torch.max(feature_map, dim=2).values, dim=2).values

    def update_stimulus(self, layer_name, feature_map, batch_idx):
        # Check if the layer is one of what we want
        if layer_name not in self.target_layers:
            return
        
        # Get maximum activation values
        N = self.num_neurons[layer_name]
        max_act = self.compute_max_act_of_feature_map(feature_map)
        top_k = min(max_act.shape[0], self.args.topk_s)
        act_vals, img_indices = torch.topk(max_act, k=top_k, dim=0)
        act_vals = act_vals.cpu().data.numpy()
        img_indices = img_indices.cpu().data.numpy()

        for k in range(top_k):
            for neuron in range(N):
                img_idx = batch_idx * self.args.batch_size + img_indices[k, neuron]
                act_val = act_vals[k, neuron]
                self.stimulus[layer_name][neuron].insert(act_val, key=img_idx)

    def save_stimulus(self):
        for layer in self.stimulus:
            for neuron, neuron_imgs in enumerate(self.stimulus[layer]):
                imgs = neuron_imgs.keys
                imgs = list(map(int, imgs))
                self.stimulus[layer][neuron] = imgs
        file_path = self.data_path.get_path('stimulus')
        self.save_json(self.stimulus, file_path)

    """
    Handle external files (e.g., output, log, ...)
    """
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('stimulus_log'), log_opt) as f:
            f.write(log + '\n')
        
    def write_first_log(self):
        log = 'Compute stimulus\n\n'
        log += f'model_nickname: {self.args.model_nickname}\n'
        log += f'model_path: {self.data_path.get_path("model_path")}\n'
        self.write_log(log, False)

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
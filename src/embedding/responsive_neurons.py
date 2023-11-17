import json
from time import time

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.utils import *

class ResponsiveNeurons:
    """
    Compute most responsive neurons for each image
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
        self.num_imgs = self.get_total_number_of_images()

        self.device = model.device
        self.training_dataset = None
        self.data_loader = None

        self.responsive_neurons = {}

    """
    A wrapper function called by main.py
    """
    def compute_responsive_neurons(self):
        self.init_setting()
        self.get_layer_info()
        self.find_responsive_neurons()
        self.save_responsive_neurons()

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
            self.data_path.get_path('responsive_neurons_image_path'),
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

    def get_total_number_of_images(self):
        root_directory = self.data_path.get_path('responsive_neurons_image_path')
        total_images = 0
        for dirpath, dirnames, filenames in os.walk(root_directory):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    total_images += 1
        return total_images

    """
    Find most responsive neurons for each input
    """
    def find_responsive_neurons(self):
        self.write_first_log()
        self.init_responsive_neurons()
        tic, total = time(), len(self.data_loader)

        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                # Get input images in a batch and their labels
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Update responsive neurons for the first layer
                f_map = self.model.forward_one_layer(0, imgs)
                self.update_responsive_neurons(
                    self.layers[0]['name'], f_map, batch_idx
                )

                # Update responsive neurons for remaining layers
                for i in range(1, len(self.layers) - 1):
                    try:
                        f_map = self.model.forward_one_layer(i, f_map)
                        self.update_responsive_neurons(
                            self.layers[i]['name'], f_map, batch_idx
                        )

                    except RuntimeError:
                        log = f'Error in find_stimulus for '
                        log += self.layers[i]['name']
                        #  self.write_log(log)

                pbar.update(1)

        log_str = 'cumulative_time_sec: {:.2f}\n'.format(time() - tic)
        self.write_log(log_str)

    def init_responsive_neurons(self):
        for i in range(self.num_imgs):
            self.responsive_neurons[i] = TopKKeeper(self.args.topk_i)
    
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

    def update_responsive_neurons(self, layer_name, feature_map, batch_idx):
        # Check if the layer is one of what we want
        if layer_name not in self.target_layers:
            return
        
        # Get maximum activation values
        max_act = self.compute_max_act_of_feature_map(feature_map)
        top_k = min(max_act.shape[1], self.args.topk_i)
        act_vals, neuron_indices = torch.topk(max_act, k=top_k, dim=1)
        act_vals = act_vals.cpu().data.numpy()
        neuron_indices = neuron_indices.cpu().data.numpy()
        n_imgs = neuron_indices.shape[0]

        for i in range(n_imgs):
            img_idx = batch_idx * self.args.batch_size + i
            for k in range(top_k):
                neuron_idx = neuron_indices[i, k]
                act_val = act_vals[i, k]
                neuron_id = f'{layer_name}-{neuron_idx}'
                self.responsive_neurons[img_idx].insert(
                    act_val, key=neuron_id
                )

    def save_responsive_neurons(self):
        for i in range(self.num_imgs):
            neurons = self.responsive_neurons[i].keys
            self.responsive_neurons[i] = neurons
        file_path = self.data_path.get_path('responsive_neurons')
        save_json(self.responsive_neurons, file_path)

    """
    Handle external files (e.g., output, log, ...)
    """
    def write_first_log(self):
        log = 'Compute responsive neurons\n\n'
        log += f'model_nickname: {self.args.model_nickname}\n'
        log += f'model_path: {self.data_path.get_path("model_path")}\n'
        log += f'responsive_neurons_image_path: {self.data_path.get_path("responsive_neurons_image_path")}\n'
        log += f'topk_i: {self.data_path.get_path("topk_i")}\n'
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('responsive_neurons_log'), log_opt) as f:
            f.write(log + '\n')
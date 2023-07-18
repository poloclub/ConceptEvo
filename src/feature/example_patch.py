import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from utils.utils import *

class ExamplePatch:
    """Generate example patches for each neuron"""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path
        self.data_path_example_patch = data_path.data_path_example_patch

        self.model = model
        self.S = model.get_input_size()
        self.R = self.args.ex_patch_size_ratio
        
        self.layers = []
        self.layers_for_ex_patch = []
        self.num_neurons = {}

        self.device = model.device
        self.raw_training_datasets = None
        self.data_loader = None

        self.ex_patch = {}
        self.stimulus = {}

    """
    A wrapper function called by main.py
    """
    def generate_example_patch(self):
        self.init_setting()
        self.get_layer_info()
        self.compute_neuron_example_patches()
        self.save_neuron_example_patches()

    """
    Initial setting
    """
    def init_setting(self):
        data_transform = transforms.Compose([
            transforms.Resize((self.S, self.S)),
            transforms.ToTensor(),
            transforms.Normalize(*self.model.input_normalization)
        ])

        training_dataset = datasets.ImageFolder(
            self.data_path.get_path('train_data'),
            data_transform
        )

        self.data_loader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )

        raw_data_transform = transforms.Compose([
            transforms.Resize((self.S, self.S)),
            transforms.ToTensor()
        ])

        self.raw_training_datasets = datasets.ImageFolder(
            self.data_path.get_path('train_data'),
            raw_data_transform
        )

    def get_layer_info(self):
        self.layers = self.model.layers[:]
        self.layers_for_ex_patch = self.model.layers_for_ex_patch[:]
        self.num_neurons = self.model.num_neurons
        layer_names =  [layer['name'] for layer in self.layers]
        last_layer_name = self.layers_for_ex_patch[-1]
        self.last_layer_idx = layer_names.index(last_layer_name)

    """
    Compute example patches
    """
    def compute_neuron_example_patches(self):
        self.write_first_log()
        self.init_example_patches()

        tic, total, total_num_so_far = time(), len(self.data_loader), 0
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                # Get input images in a batch and their labels
                f_map = imgs.to(self.device)
                labels = labels.to(self.device)

                # Update example patches
                for i in range(self.last_layer_idx + 1):
                    f_map = self.layers[i]['layer'](f_map)
                    self.update_ex_patch(
                        self.layers[i]['name'], f_map, batch_idx
                    )

                pbar.update(1)

        self.write_log(f'Running time for computing: {time() - tic:.2f} sec')

    def init_example_patches(self):
        for layer_name in self.layers_for_ex_patch:
            self.ex_patch[layer_name] = [
                TopKKeeper(self.args.topk_e)
                for neuron_i in range(self.num_neurons[layer_name])
            ]

    def compute_feature_map(self, layer, imgs):
        # Compute feature map. feature_map: [B, N, W, H]
        # where B is batch size and N is the number of neurons
        feature_map = layer(imgs)
        return feature_map

    def compute_max_act_of_feature_map(self, feature_map):
        # Get the maximum activation of the feature map. max_act: [B, N]
        # where B is batch size and N is the number of neurons
        return torch.max(torch.max(feature_map, dim=2).values, dim=2).values
    
    def update_ex_patch(self, layer_name, feature_map, batch_idx):
        # Check if the layer is a convolutional layer
        if layer_name not in self.layers_for_ex_patch:
            return

        # Get top-k images that induce the higest maximum activation
        B, N, H, W = feature_map.shape
        top_k, H, W = min(B, self.args.topk_e), int(H), int(W)
        linearized_feature_map = feature_map.view(B, N, -1)
        max_pixel_vals, max_pixel_indices = torch.max(linearized_feature_map, dim=-1)
        topk_pixel_vals, topk_img_indices = torch.topk(max_pixel_vals, k=top_k, dim=0)
        topk_pixel_vals = topk_pixel_vals.detach().cpu().numpy()
        topk_img_indices = topk_img_indices.detach().cpu().numpy()
        max_pixel_indices = max_pixel_indices.detach().cpu().numpy()
        
        # Update ex_patch information
        for k in range(top_k):
            for neuron in range(N):  
                # Get img index
                b = topk_img_indices[k, neuron]
                img_idx = int(b + self.args.batch_size * batch_idx)
                
                # Get row and column
                pixel = max_pixel_indices[b, neuron]
                row = int(pixel // W)
                col = int(pixel % W)
                
                # Get the highest activation value
                act_val = float(topk_pixel_vals[k, neuron])

                # Save the information
                self.ex_patch[layer_name][neuron].insert(
                    act_val, 
                    key=img_idx,
                    content={
                        'img_idx': img_idx, 
                        'act_val': act_val, 
                        'row': row, 
                        'col': col, 
                        'W': W, 
                        'H': H
                    }
                )

    def save_neuron_example_patches(self):
        # Convert the data types in self.ex_patch to standard data types
        # for json serialization
        ex_patch_json = {}
        for layer in self.ex_patch:
            ex_patch_json[layer] = []
            N = len(self.ex_patch[layer])
            for n in range(N):
                neuron_ex_patch_info = []
                for e in self.ex_patch[layer][n].contents:
                    new_e = {}
                    for key in e:
                        if key in ['img_idx', 'row', 'col', 'W', 'H']:
                            new_e[key] = int(e[key])
                        elif key in ['act_val']:
                            new_e[key] = float(e[key])
                        else:
                            new_e[key] = e[key]
                    neuron_ex_patch_info.append(new_e)
                ex_patch_json[layer].append(neuron_ex_patch_info)

        # Save example patch information
        p = self.data_path.get_path('example_patch_info')
        self.save_json(ex_patch_json, p)

        # Save example patches
        total = np.sum([len(ex_patch_json[layer]) for layer in ex_patch_json])
        for key in ['crop', 'mask', 'inverse_mask']:
            tic = time()
            with tqdm(total=total) as pbar:
                for layer in ex_patch_json:
                    for neuron, topk_patch_infos in enumerate(ex_patch_json[layer]):
                        patches = self.get_patches(topk_patch_infos, option=key)
                        for i, patch in enumerate(patches):
                            file_name = f'{layer}-{neuron}-{i}.jpg'
                            dir_path = self.data_path.get_path(f'example_patch_{key}')
                            file_path = os.path.join(dir_path, file_name)
                            cv2.imwrite(file_path, patch)
                        pbar.update(1)
            self.write_log(f'Running time for saving {key} patch: {time() - tic:.2f} sec')

    def get_patches(self, topk_patch_infos, option='crop'):
        patches = []
        for i, patch_info in enumerate(topk_patch_infos):
            # Get coord information
            img_idx, row, col, W, H = [
                patch_info['img_idx'],
                patch_info['row'],
                patch_info['col'],
                patch_info['W'],
                patch_info['H']
            ]

            # Get coordinates
            r1 = int((row * self.S / W) - (self.S * self.R / 2))
            r2 = int((row * self.S / W) + (self.S * self.R / 2))
            c1 = int((col * self.S / H) - (self.S * self.R / 2))
            c2 = int((col * self.S / H) + (self.S * self.R / 2))
            r1 = max(0, r1)
            r2 = min(r2, self.S)
            c1 = max(0, c1)
            c2 = min(c2, self.S)

            # Get cropped patch
            img, _ = self.raw_training_datasets[img_idx]
            if option == 'crop':
                patch = img[:, r1: r2, c1: c2] * 255

            # Get masked patch
            if option == 'mask':
                patch = img.clone() * 255
                patch[:, r1: r2, c1: c2] = 0

            # Get inverse-masked patch
            if option == 'inverse_mask':
                patch = torch.zeros(img.size())
                patch[:, r1: r2, c1: c2] = img[:, r1: r2, c1: c2] * 255

            patch = patch.cpu().data.numpy()
            patch = np.einsum('kij->ijk', patch)
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            patches.append(patch)

        return patches

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

    def write_log(self, log):
        with open(self.data_path.get_path('example_patch_log'), 'a') as f:
            f.write(log + '\n')

    def write_first_log(self):
        # First line of the log
        log = 'Example patches\n'

        # Model specification
        log += self.args.model_nickname + '\n'
        if self.data_path_example_patch.util.is_arg_given(self.args.epoch):
            log += f'Epoch = {self.args.epoch}\n'

        # Hyperparameters
        hypara = self.data_path_example_patch.hypara
        apdx = [f'{arg}={val}' for arg, val in hypara]
        apdx = '\n'.join(apdx)
        log += apdx + '\n'

        # Write the log
        self.write_log(log)

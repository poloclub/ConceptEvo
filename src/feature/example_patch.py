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

        self.model = model
        self.S = model.input_size
        
        self.layers = []
        self.conv_layers = []
        self.num_neurons = {}

        self.device = model.device
        self.raw_training_datasets = None
        self.data_loader = None

        self.ex_patch = {}
        self.stimulus = {}

    """
    A wrapper function called by main.py
    """
    def compute_neuron_feature(self):
        self.init_setting()
        self.get_layer_info()
        self.load_stimulus()
        self.compute_example_patches()

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
        self.conv_layers = self.model.conv_layers[:]
        self.num_neurons = self.model.num_neurons


    """
    Compute example patches
    """
    def compute_example_patches(self):
        self.write_first_log()
        self.init_example_patches()

        tic, total, total_num_so_far = time(), len(self.data_loader), 0
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):

                # Get input images in a batch and their labels
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Update example patches for the first layer
                f_map = self.compute_feature_map(self.layers[0]['layer'], imgs)
                self.update_ex_patch(self.layers[0]['name'], f_map, batch_idx)

                # # Update stimulus for remaining layers
                # for i in range(1, len(self.layers) - 1):
                #     try:
                #         f_map = self.layers[i]['layer'](f_map)
                #         self.update_ex_patch(
                #             self.layers[i]['name'], f_map, batch_idx
                #         )
                #     except RuntimeError as e:
                #         log = 'Error in compute_example_patches(): '
                #         log += self.layers[i]['name']
                #         #  self.write_log(log)

                pbar.update(1)

        self.write_log('running_time_for_computing: {}sec'.format(time() - tic))

        # Save example patches
        tic = time()
        self.save_example_patches()
        self.write_log('running_time_for_saving: {}sec'.format(time() - tic))

    def init_example_patches(self):
        for layer_name in self.conv_layers:
            self.ex_patch[layer_name] = []
            for neuron_i in range(self.num_neurons[layer_name]):
                self.ex_patch[layer_name].append({
                    'img_idxs': [],
                    'coords': []
                })

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
        if layer_name not in self.conv_layers:
            return
        
        # Iterate through all neurons to update information of example patches 
        ra = self.args.ex_patch_size_ratio / 2
        max_act = self.compute_max_act_of_feature_map(feature_map)
        feature_map_np = feature_map.cpu().data.numpy()
        B, N, w, h = feature_map.shape

        for neuron in range(N):
            stimulating_imgs = self.stimulus[layer_name][neuron]
            for img_idx in stimulating_imgs:
                # Check if img_idx is in the current batch
                if img_idx < self.args.batch_size * batch_idx:
                    continue

                if img_idx >= self.args.batch_size * (batch_idx + 1):
                    continue

                # Maximum activation value
                img_i = img_idx - self.args.batch_size * batch_idx
                act_val = max_act[img_i, neuron]

                # Pixel coordinate of maximum activation of feature maps
                highest_pixel = np.argmax(feature_map_np[img_i, neuron, :, :])
                r, c = highest_pixel // w, highest_pixel % w
                r1 = np.max([r - int(h * ra), 0])
                c1 = np.max([c - int(w * ra), 0])
                r2 = np.min([r + int(h * ra), self.S])
                c2 = np.min([c + int(w * ra), self.S])

                # Pixel coordinate for input
                r1 = int(r1 * self.S / h)
                c1 = int(c1 * self.S / w)
                r2 = int(r2 * self.S / h)
                c2 = int(c2 * self.S / w)

                # Insert patch
                self.ex_patch[layer_name][neuron]['img_idxs'].append(img_idx)
                self.ex_patch[layer_name][neuron]['coords'].append(
                    [r1, c1, r2, c2]
                )

            # act_vals, img_indices = torch.sort(
            #     max_act[:, neuron], descending=True
            # )
            # img_indices = img_indices.cpu().data.numpy()
            # for k in range(self.args.num_features):
            #     act_val = act_vals[k].cpu().data.numpy()
            #     img_idx_in_all_data = \
            #         batch_idx * self.args.batch_size + img_indices[k]
            #     if self.ex_patch[layer_name][neuron].will_insert(act_val):     
            #         # Pixel coordinate of maximum activation of feature maps
            #         img_i = img_indices[k]
            #         highest_idx = np.argmax(
            #             feature_map_np[img_i, neuron, :, :]
            #         )
            #         r, c = highest_idx // w, highest_idx % w
            #         r1 = np.max([r - int(h * ra), 0])
            #         c1 = np.max([c - int(w * ra), 0])
            #         r2 = np.min([r + int(h * ra), self.S])
            #         c2 = np.min([c + int(w * ra), self.S])

            #         # Pixel coordinate for input
            #         r1 = int(r1 * self.S / h)
            #         c1 = int(c1 * self.S / w)
            #         r2 = int(r2 * self.S / h)
            #         c2 = int(c2 * self.S / w)

            #         # Insert patch
            #         self.ex_patch[layer_name][neuron].insert(
            #             act_val, 
            #             key=img_idx_in_all_data,
            #             content=[r1, c1, r2, c2]
            #         )

    """
    Utils
    """
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def load_stimulus(self):
        stimulus_path = self.data_path.get_path('stimulus')
        self.stimulus = self.load_json(stimulus_path)

    def write_log(self, log):
        with open(self.data_path.get_path('neuron_feature-log'), 'a') as f:
            f.write(log + '\n')

    def write_first_log(self):
        hyperpara_setting = self.data_path.gen_act_setting_str(
            'neuron_feature', '\n'
        )
        log = 'Example patches\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hyperpara_setting + '\n\n'
        self.write_log(log)

    def save_example_patches(self):
        total = len(self.ex_patch)
        for layer in self.ex_patch:
            total += len(self.ex_patch[layer])

        with tqdm(total=total) as pbar:
            for layer in self.ex_patch:
                for neuron, patches in enumerate(self.ex_patch[layer]):
                    img_idxs = patches['img_idxs']
                    coords = patches['coords']
                    num_fail = 0
                    for i, (idx, coord) in enumerate(zip(img_idxs, coords)):                        
                        try:
                            # Get patch
                            r1, c1, r2, c2 = coord
                            patch, _ = self.raw_training_datasets[idx]
                            patch = patch[:, r1: r2, c1: c2] * 255
                            patch = patch.cpu().data.numpy()
                            patch = np.einsum('kij->ijk', patch)
                            
                            # Save patch
                            file_name = f'{layer}-{neuron}-{i - num_fail}.jpg'
                            file_path = os.path.join(
                                self.data_path.get_path('neuron_feature'),
                                file_name
                            )   
                            cv2.imwrite(
                                file_path, 
                                cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                            )
                        except Exception as e:
                            num_fail += 1
                    pbar.update(1)

    def test_raw_example(self):
        for idx in range(5):
            img, _ = self.raw_training_datasets[idx]
            img = img * 255
            img = img.cpu().data.numpy()
            img = np.einsum('kij->ijk', img)
            file_path = f'b-{idx}.jpg'
            cv2.imwrite(
                file_path, 
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
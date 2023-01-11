import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from utils.utils import *

class StimulusActMap:
    """Generate activation maps for stimulus"""

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

        self.act_map = {}
        self.stimulus = {}

    """
    A wrapper function called by main.py
    """
    def compute_act_map(self):
        self.init_setting()
        self.get_layer_info()
        self.load_stimulus()
        self.compute_act_maps()

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
    def compute_act_maps(self):
        self.write_first_log()
        self.init_act_maps()

        tic, total, total_num_so_far = time(), len(self.data_loader), 0
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):

                # Get input images in a batch and their labels
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Update activation maps for the first layer
                f_map = self.compute_feature_map(self.layers[0]['layer'], imgs)
                self.update_act_map(self.layers[0]['name'], f_map, batch_idx)

                # Update activation maps for remaining layers
                for i in range(1, len(self.layers) - 1):
                    try:
                        f_map = self.layers[i]['layer'](f_map)
                        self.update_act_map(
                            self.layers[i]['name'], f_map, batch_idx
                        )
                    except RuntimeError as e:
                        log = 'Error in compute_act_maps(): '
                        log += self.layers[i]['name']
                        #  self.write_log(log)

                pbar.update(1)

        self.write_log('running_time_for_computing: {}sec'.format(time() - tic))

        # Save example patches
        # tic = time()
        # self.save_act_maps()
        # self.write_log('running_time_for_saving: {}sec'.format(time() - tic))

    def init_act_maps(self):
        for layer_name in self.conv_layers:
            self.act_map[layer_name] = []
            for neuron_i in range(self.num_neurons[layer_name]):
                self.act_map[layer_name].append({
                    'img_idxs': [],
                    'act_maps': []
                })

    def compute_feature_map(self, layer, imgs):
        # Compute feature map. feature_map: [B, N, W, H]
        # where B is batch size and N is the number of neurons
        feature_map = layer(imgs)
        return feature_map
    
    def update_act_map(self, layer_name, feature_map, batch_idx):
        # Check if the layer is a convolutional layer
        if layer_name not in self.conv_layers:
            return
        
        # Iterate through all neurons to update information of example patches 
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
                neuron_act_map_np = feature_map_np[img_i, neuron, :, :]
                # print(neuron_act_map_np.shape)
                
                # Save activation map
                # self.act_map[layer_name][neuron]['img_idxs'].append(img_idx)
                # a_map = np.einsum('kij->ijk', neuron_act_map_np)
                file_name = f'{layer_name}-{neuron}-{img_idx}.jpg'
                file_path = os.path.join(
                    self.data_path.get_path('act_map'),
                    file_name
                )
                plt.imshow(neuron_act_map_np)
                plt.savefig(file_path)
                # cv2.imwrite(
                #     file_path, 
                #     cv2.cvtColor(a_map, cv2.COLOR_RGB2BGR)
                # )
                
                # self.act_map[layer_name][neuron]['act_maps'].append(
                #     neuron_act_map_np
                # )
                # self.act_map[layer_name][neuron].append(neuron_act_map_np)

            # act_vals, img_indices = torch.sort(
            #     max_act[:, neuron], descending=True
            # )
            # img_indices = img_indices.cpu().data.numpy()
            # for k in range(self.args.num_features):
            #     act_val = act_vals[k].cpu().data.numpy()
            #     img_idx_in_all_data = \
            #         batch_idx * self.args.batch_size + img_indices[k]
            #     if self.act_map[layer_name][neuron].will_insert(act_val):     
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
            #         self.act_map[layer_name][neuron].insert(
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
        with open(self.data_path.get_path('act_map-log'), 'a') as f:
            f.write(log + '\n')

    def write_first_log(self):
        hyperpara_setting = self.data_path.gen_act_setting_str(
            'act_map', '\n'
        )
        log = 'Activation maps for given stimulus\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hyperpara_setting + '\n\n'
        self.write_log(log)

    def save_act_maps(self):
        total = 0
        for layer in self.act_map:
            total += len(self.act_map[layer])

        with tqdm(total=total) as pbar:
            for layer in self.act_map:
                for neuron, neuron_a_maps in enumerate(self.act_map[layer]):
                    img_idxs = neuron_a_maps['img_idxs']
                    a_maps = neuron_a_maps['act_maps']
                    for idx, am in zip(img_idxs, a_maps):
                        neuron_act_map = np.einsum('kij->ijk', am)
                        file_name = f'{layer}-{neuron}-{idx}.jpg'
                        file_path = os.path.join(
                            self.data_path.get_path('act_map'),
                            file_name
                        )   
                        cv2.imwrite(
                            file_path, 
                            cv2.cvtColor(neuron_act_map, cv2.COLOR_RGB2BGR)
                        )
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
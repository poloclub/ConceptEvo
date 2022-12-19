import re
import json
from time import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.utils import *


class Stimulus:
    """Find stimulus for each neuron"""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path

        self.model = model
        self.layers = []
        self.conv_layers = []
        self.num_neurons = {}

        self.device = model.device
        self.training_dataset = None
        self.data_loader = None

        self.relu = nn.ReLU(inplace=True)

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
        data_transform = transforms.Compose([
            transforms.Resize((self.model.input_size, self.model.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(*self.model.input_normalization)
        ])

        self.training_dataset = datasets.ImageFolder(
            self.data_path.get_path('train_data'),
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
        self.conv_layers = self.model.conv_layers[:]
        self.num_neurons = self.model.num_neurons

    """
    Find stimulus (i.e., inputs that activate neurons the most) for each neuron
    """
    def find_stimulus(self):
        self.write_first_log()
        self.init_stimulus()
        f_map_res_input = None
        tic, total = time(), len(self.data_loader)

        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                # Get input images in a batch and their labels
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Update stimulus for the first layer
                f_map = self.compute_feature_map(
                    self.layers[0]['layer'], imgs, None
                )
                self.update_stimulus(
                    self.layers[0]['name'], f_map, batch_idx
                )

                # Check if the output of the first layer
                # can be used as a residual input for later layers
                if self.model.layer_is_res_input(0):
                    f_map_res_input = f_map.clone()
                    
                # Update stimulus for remaining layers
                for i in range(1, len(self.layers) - 1):
                    try:
                        if self.model.layer_is_downsample(i):
                            # Downsample input (for ResNets)
                            res_input = self.compute_feature_map(
                                self.layers[i]['layer'], f_map_res_input, None
                            )

                            # Add residual input
                            f_map = f_map + res_input

                            # ReLU
                            f_map = self.relu(f_map)
                            
                            # Update stimulus
                            self.update_stimulus(
                                self.layers[i]['name'], f_map, batch_idx
                            )
                        else:
                            # Residual input: res_input is not None for 
                            # some models such as ResNets and ConvNeXt
                            res_input = None
                            if self.model.layer_take_res_input(i):
                                res_input = f_map_res_input

                            # Compute feature map of the layer
                            f_map = self.compute_feature_map(
                                self.layers[i]['layer'], f_map, res_input
                            )
                            self.update_stimulus(
                                self.layers[i]['name'], f_map, batch_idx
                            )

                        # Update residual input
                        if self.model.layer_is_res_input(i):
                            f_map_res_input = f_map.clone()
                        
                    except RuntimeError:
                        log = f'Error in find_stimulus for '
                        log += self.layers[i]['name']
                        #  self.write_log(log)

                pbar.update(1)

        log_str = 'cumulative_time_sec: {:.2f}\n'.format(time() - tic)
        self.write_log(log_str)

    def init_stimulus(self):
        for layer_name in self.conv_layers:
            self.stimulus[layer_name] = [
                TopKKeeper(self.args.topk_s) 
                for neuron in range(self.num_neurons[layer_name])
            ]

    def prof_to_dict(self, prof_str):

        def extract_cells_from_one_row(s, widths, white_spaces):
            columns = []
            acc_width = 0
            for i, width in enumerate(widths):
                columns.append(s[acc_width: acc_width + width])
                acc_width += width

                if i < len(white_spaces) - 1:
                    acc_width += white_spaces[i]
                    
            columns = list(map(lambda x: x.strip(), columns))
            return columns

        def is_new_row(s, widths):
            if_all_whitespace = s[:widths[0]].isspace()
            return not if_all_whitespace

        def is_blank_row(s):
            return s.isspace() or len(s) == 0

        lines = prof_str.split('\n')
        widths = list(map(len, lines[0].split()))
        white_spaces = list(map(len, re.findall(r'[ ]{1,}', lines[0])))
        columns = extract_cells_from_one_row(lines[1], widths, white_spaces)

        rows = []
        row_idx = -1
        for line in lines[3:-4]:
            if is_blank_row(line):
                continue
            if is_new_row(line, widths):
                rows.append([''] * len(columns))
                row_idx += 1
            columns_in_row = extract_cells_from_one_row(
                line, widths, white_spaces
            )
            
            for c, column_in_row in enumerate(columns_in_row):
                if len(column_in_row) > 0:
                    rows[row_idx][c] += column_in_row

        prof_dict = {}
        for r, row in enumerate(rows):
            d = {}
            for c, column in enumerate(columns):
                d[column] = row[c]
            name = d['Name']
            prof_dict[name] = d

        return prof_dict

    def convert_time_to_second(self, s):
        try:
            t = float(re.findall('\d*\.*\d*', s)[0])
        except:
            t = 0

        if 'ms' in s:
            t = t / 1000
        elif 'us' in s:
            t = t / 1000000
        elif 's' in s:
            t = t
        else:
            t = 0
            
        return t
        
    def add_time(self, t1, t2):
        return '{}s'.format(
            self.convert_time_to_second(t1) + self.convert_time_to_second(t2)
        )

    def agg_prof_dict(self, prof_dict, d):
        for name in d:
            if name not in prof_dict:
                keys = [
                    '# of Call', '# of Calls', 'Self CPU', \
                    'CPU total', 'CPU time avg', 'Self CUDA', \
                    'CUDA total', 'CUDA time avg'
                ]
                prof_dict[name] = dict()
                for key in keys:
                    if key in d[name]:
                        prof_dict[name][key] = d[name][key]
                    else:
                        prof_dict[name][key] = 0
            else:
                # Simple addition
                keys = ['# of Call', '# of Calls']
                for key in keys:
                    if key not in d[name]:
                        continue
                    prof_dict[name][key] = \
                        int(prof_dict[name][key]) + int(d[name][key])
                    
                # Add times
                keys = [
                    'Self CPU', 'CPU total', 'CPU time avg', \
                    'Self CUDA', 'CUDA total', 'CUDA time avg'
                ]
                for key in keys:
                    if key not in d[name]:
                        continue
                    prof_dict[name][key] = self.add_time(
                        prof_dict[name][key], d[name][key]
                    )
                
        return prof_dict
            
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
        # Check if the layer is a convolutional layer
        if layer_name not in self.conv_layers:
            return
        
        # Iterate through all neurons to update stimulus for each neuron
        N = self.num_neurons[layer_name]
        max_act = self.compute_max_act_of_feature_map(feature_map)
        for neuron in range(N):
            # torch.sort -> torch.topk
            act_vals, img_indices = torch.sort(
                max_act[:, neuron], descending=True
            )
            
            act_vals = act_vals.cpu().data.numpy()
            img_indices = img_indices.cpu().data.numpy()

            for k in range(self.args.topk_s):
                img_idx = batch_idx * self.args.batch_size + img_indices[k]
                act_val = act_vals[k]
                self.stimulus[layer_name][neuron].insert(act_val, key=img_idx)


    def save_stimulus(self):
        for layer in self.stimulus:
            for neuron, neuron_imgs in enumerate(self.stimulus[layer]):
                imgs = neuron_imgs.keys
                imgs = list(map(int, imgs))
                self.stimulus[layer][neuron] = imgs
        file_path = self.data_path.get_path('stimulus')
        self.save_json(self.stimulus, file_path)
        

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)


    """
    Handle external files (e.g., output, log, ...)
    """
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('stimulus-log'), log_opt) as f:
            f.write(log + '\n')

        
    def write_first_log(self):
        hypara_setting = self.data_path.gen_act_setting_str('stimulus', '\n')
        log = 'Find Stimulus\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hypara_setting + '\n\n'
        self.write_log(log, False)


    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data


    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)

        
import os
import json
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

class ImportantNeuronActMap:
    """Get activation map of important neurons"""

    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path
        self.model = model

        self.layers = []
        self.conv_layers = []

        self.device = model.device

        self.start_idx = 0
        self.end_idx = 0

        self.important_neurons = {}

    """
    A wrapper function called by main.py
    """
    def compute_important_neuron_act_map(self):
        self.init_setting()
        self.get_layer_info()
        self.load_important_neurons()
        self.compute_and_save_important_neuron_act_map()

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

        self.find_idx_training_dataset_for_class()
        self.class_dataset = torch.utils.data.Subset(
            self.training_dataset, 
            range(self.start_idx, self.end_idx)
        )
        self.data_loader = torch.utils.data.DataLoader(
            self.class_dataset, 
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
    
    def find_idx_training_dataset_for_class(self):
        total = len(self.training_dataset)
        unit = int(total / self.model.num_classes)
        start = max(0, unit * (self.args.label - 1))
        end = min(total, unit * (self.args.label + 2))

        start_idx, end_idx = -1, -1
        with tqdm(total=(end - start)) as pbar:
            for i in range(start, end):
                img, label = self.training_dataset[i]
                if (self.args.label == label) and (start_idx == -1):
                    start_idx = i
                    end_idx = -2
                if (self.args.label < label) and (end_idx == -2):
                    end_idx = i
                    break
                pbar.update(1)

        if (start_idx != -1) and (end_idx < 0):
            end_idx = end

        self.start_idx = start_idx
        self.end_idx = end_idx
        print(self.start_idx, self.end_idx)

    def get_layer_info(self):
        self.layers = self.model.layers[:]
        self.conv_layers = self.model.conv_layers[:]

    """
    Compute activation maps of important neurons
    """
    def compute_and_save_important_neuron_act_map(self):
        self.write_first_log()
        
        tic, total = time(), self.end_idx - self.start_idx
        
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                # Get input images in a batch and their labels
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # First layer
                f_map = self.model.forward_one_layer(0, imgs)
                if self.layers[0]['name'] == self.args.layer:
                    self.save_act_maps(f_map, batch_idx, pbar)
                
                # Remaining layers
                for i in range(1, len(self.layers) - 1):
                    try:
                        f_map = self.model.forward_one_layer(i, f_map)
                        if self.layers[i]['name'] == self.args.layer:
                            self.save_act_maps(f_map, batch_idx, pbar)
                    except RuntimeError:
                        log = f'Error in find_stimulus for '
                        log += self.layers[i]['name']
                        #  self.write_log(log)

    def save_act_maps(self, f_map, batch_idx, pbar):
        layer_name = self.args.layer
        feature_map_np = f_map.cpu().data.numpy()
        B, N, W, H = feature_map_np.shape

        for img_i in range(B):
            # Image index
            img_idx = self.start_idx
            img_idx += (batch_idx * self.args.batch_size) + img_i
            img_idx = int(img_idx)

            for n in range(N):
                # Skip if the current image is not important for the neuron
                neuron_id = f'{layer_name}-{n}'
                img_idxs = self.important_neurons[neuron_id]['img_idxs']
                if img_idx not in img_idxs:
                    continue

                # Save activation map
                file_name = f'{layer_name}-{n}-{img_idx}.jpg'
                file_path = os.path.join(
                    self.data_path.get_path('important_neuron_act_map'),
                    file_name
                )
                neuron_act_map_np = feature_map_np[img_i, n, :, :]
                ax = plt.subplot()
                im = ax.imshow(neuron_act_map_np)
                plt.colorbar(im)
                plt.savefig(file_path)

            # Update pbar
            pbar.update(1)

    """
    Utils
    """
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def load_important_neurons(self):
        p = self.data_path.get_path('important_neuron')
        self.important_neurons = self.load_json(p)

    def write_log(self, log):
        file_path = self.data_path.get_path('important_neuron_act_map-log')
        with open(file_path, 'a') as f:
            f.write(log + '\n')
    
    def write_first_log(self):
        hyperpara_setting = self.data_path.gen_act_setting_str(
            'important_neuron_act_map', '\n'
        )
        log = 'Find activation maps of important neurons\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hyperpara_setting + '\n\n'
        self.write_log(log)
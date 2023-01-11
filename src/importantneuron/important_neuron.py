import os
import json
import numpy as np
from tqdm import tqdm
from time import time

import torch
from torchvision import datasets, transforms


class ImportantNeuron:
    """Finds important neurons for a class"""

    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path
        self.model = model

        self.layers = []
        self.conv_layers = []
        self.num_neurons = {}

        self.device = model.device

        self.start_idx = 0
        self.end_idx = 0

        self.important_neurons = {}
        
    """
    A wrapper function called by main.py
    """
    def compute_important_neuron(self):
        self.init_setting()
        self.get_layer_info()
        self.find_important_neurons()
        self.save_important_neurons()

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
        self.num_neurons = self.model.num_neurons

    """
    Find important neurons
    """
    def find_important_neurons(self):
        self.write_first_log()
        self.init_important_neurons()
        
        tic, total = time(), len(self.data_loader)

        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                # Get input images in a batch and their labels
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # First layer
                f_map = self.model.forward_one_layer(0, imgs)
                if self.layers[0]['name'] == self.args.layer:
                    self.update_important_neurons(f_map, batch_idx)
                
                # Remaining layers
                for i in range(1, len(self.layers) - 1):
                    try:
                        f_map = self.model.forward_one_layer(i, f_map)
                        if self.layers[i]['name'] == self.args.layer:
                            self.update_important_neurons(f_map, batch_idx)
                    except RuntimeError:
                        log = f'Error in find_stimulus for '
                        log += self.layers[i]['name']
                        #  self.write_log(log)
                pbar.update(1)

    def init_important_neurons(self):
        if self.args.layer not in self.num_neurons:
            return
        
        for neuron_idx in range(self.num_neurons[self.args.layer]):
            neuron_id = f'{self.args.layer}-{neuron_idx}'
            self.important_neurons[neuron_id] = {
                'img_idxs': [],
                'max_acts': []
            }

    def update_important_neurons(self, f_map, batch_idx):
        layer_name = self.args.layer
        max_act = self.compute_max_act_of_feature_map(f_map)
        B, N = max_act.shape

        # Initialize self.important_neurons 
        # if its number of neurons are not available
        if self.args.layer not in self.num_neurons:
            if len(self.important_neurons) == 0:
                for neuron_idx in range(N):
                    neuron_id = f'{self.args.layer}-{neuron_idx}'
                    self.important_neurons[neuron_id] = {
                        'img_idxs': [],
                        'max_acts': []
                    }

        for img_i in range(B):
            # Image index
            img_idx = self.start_idx
            img_idx += (batch_idx * self.args.batch_size) + img_i
            img_idx = int(img_idx)

            # Get max pool activation values and images indices
            act_vals, neuron_idxs = torch.sort(
                max_act[img_i, :], descending=True
            )
            act_vals = act_vals.cpu().data.numpy()
            neuron_idxs = neuron_idxs.cpu().data.numpy()

            # Get most activating neurons for the image
            for k in range(self.args.topk_n):
                n = neuron_idxs[k]
                neuron_id = f'{layer_name}-{n}'
                act_val = act_vals[k]
                self.important_neurons[neuron_id]['img_idxs'].append(img_idx)
                self.important_neurons[neuron_id]['max_acts'].append(act_val)

    def compute_max_act_of_feature_map(self, feature_map):
        # Get the maximum activation of the feature map. max_act: [B, N]
        # where B is batch size and N is the number of neurons
        return torch.max(torch.max(feature_map, dim=2).values, dim=2).values

    def save_important_neurons(self):
        # Sort img indices by the activation values
        for neuron_id in self.important_neurons:
            # Get image indices and activation values
            img_idxs = self.important_neurons[neuron_id]['img_idxs']
            max_acts = self.important_neurons[neuron_id]['max_acts']

            # Sort image indices and activation values
            sorted_idxs = np.argsort(max_acts)[::-1]
            img_idxs = np.array(img_idxs)[sorted_idxs]
            max_acts = np.array(max_acts)[sorted_idxs]

            # Save the sorted values
            self.important_neurons[neuron_id]['img_idxs'] = [
                int(img_idx) for img_idx in img_idxs
            ]
            self.important_neurons[neuron_id]['max_acts'] = [
                round(float(max_act), 3) for max_act in max_acts
            ]

        # Save important neurons into a file
        p = self.data_path.get_path('important_neuron')
        self.save_json(self.important_neurons, p)

    """
    Utils
    """
    def write_log(self, log):
        file_path = self.data_path.get_path('important_neuron-log')
        with open(file_path, 'a') as f:
            f.write(log + '\n')
    
    def write_first_log(self):
        hyperpara_setting = self.data_path.gen_act_setting_str(
            'important_neuron', '\n'
        )
        log = 'Find important neurons\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hyperpara_setting + '\n\n'
        self.write_log(log)

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)
    
    
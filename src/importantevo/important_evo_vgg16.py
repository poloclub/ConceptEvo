import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

from model.vgg16 import *
from model.inception_v3 import *

import torch
from torch import autograd
from torchvision import datasets, transforms

class ImportantEvoVgg16:
    """Find important evolution."""

    """
    Constructor
    """
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path

        self.device = None
        self.from_model = None
        self.to_model = None

        self.input_size = -1        
        self.num_classes = 1000

        self.label_to_synset = {}
        self.sensitivity = {}


    """
    A wrapper function called by main.py
    """
    def find_important_evolution(self):
        self.write_first_log()
        self.init_setting()
        self.find_imp_evo()
        self.save_sensitivity()


    """
    Initial setting
    """
    def init_setting(self):
        self.set_input_size()
        self.init_device()
        self.get_synset_info()
        self.load_models()

        data_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225]
            )
        ])

        self.training_dataset = datasets.ImageFolder(
            self.data_path.get_path('train_data'), data_transform
        )

        self.class_training_dataset = []
        self.gen_training_dataset_of_class()

        self.data_loader = torch.utils.data.DataLoader(
            self.class_training_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )

    
    def set_input_size(self):
        if self.args.model_name == 'inception_v3':
            self.input_size = 299
        elif self.args.model_name == 'vgg16':
            self.input_size = 224


    def init_device(self):
        self.device = torch.device(
            'cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() 
            else 'cpu'
        )
        print('Run on {}'.format(self.device))

    
    def load_models(self):
        # Initialize self.from_model and self.to_model
        if self.args.model_name == 'vgg16':
            self.from_model = Vgg16(self.args, self.data_path)
            self.to_model = Vgg16(self.args, self.data_path)
        elif self.args.model_name == 'inception_v3':
            self.from_model = InceptionV3(self.args, self.data_path)
            self.to_model = InceptionV3(self.args, self.data_path)
        else:
            raise ValueError(f'Error: unkonwn model {self.args.model_name}')
        
        # Set both models need checkpoints
        self.from_model.need_loading_a_saved_model = True
        self.to_model.need_loading_a_saved_model = True

        # Load checkpoints
        self.from_model.ckpt = torch.load(self.args.from_model_path)
        self.to_model.ckpt = torch.load(self.args.to_model_path)

        # Initialize the models
        self.from_model.device = self.device
        self.to_model.device = self.device
        self.from_model.init_model()
        self.to_model.init_model()

        # Set the training setting
        self.from_model.init_training_setting()
        self.to_model.init_training_setting()


    def get_synset_info(self):
        df = pd.read_csv(self.args.data_label_path, sep='\t')
        for synset, label in zip(df['synset'], df['training_label']):
            self.label_to_synset[int(label) - 1] = synset


    def gen_training_dataset_of_class(self):
        tic = time()
        total = len(self.training_dataset)
        unit = int(total / self.num_classes)
        start = max(0, unit * (self.args.label - 1))
        end = min(total, unit * (self.args.label + 2))

        with tqdm(total=(end - start)) as pbar:
            for i in range(start, end):
                img, label = self.training_dataset[i]
                if label == self.args.label:
                    self.class_training_dataset.append([img, label])
                elif label > self.args.label:
                    break
                pbar.update(1)
        toc = time()
        log = 'Filter images for the label: {} sec'.format(toc - tic)
                

    """
    Find important evolution
    """
    def find_imp_evo(self):
        tic, total = time(), len(self.class_training_dataset)
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                if self.args.model_name == 'vgg16':
                    self.find_imp_evo_one_batch_vgg16(imgs, labels)
                elif self.args.model_name == 'inception_v3':
                    self.find_imp_evo_one_batch_inception_v3(imgs, labels)
                pbar.update(self.args.batch_size)
        toc = time()
        log = 'Find important evo: {:.2f} sec'.format(toc - tic)
        self.write_log(log)


    def find_imp_evo_one_batch_vgg16(self, imgs, labels):
        # Send input images and their labels to GPU
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        # Forward
        from_model_children = list(self.from_model.model.children())
        to_model_children = list(self.to_model.model.children())
        from_f_map, to_f_map = imgs, imgs
        f_maps, layer_info = {'from': [], 'to': []}, []
        for i, child in enumerate(from_model_children):
            if type(child) == nn.Sequential:
                for j, from_layer in enumerate(child.children()):
                    to_layer = to_model_children[i][j]
                    from_f_map = from_layer(from_f_map)
                    to_f_map = to_layer(to_f_map)
                    f_maps['from'].append(from_f_map)
                    f_maps['to'].append(to_f_map)
                    layer_name = '{}_{}_{}_{}'.format(
                        type(child).__name__, i,
                        type(from_layer).__name__, j
                    )
                    layer_info.append({
                        'name': layer_name,
                        'num_neurons': from_f_map.shape[1]
                    })
            else:
                to_layer = to_model_children[i]
                from_f_map = child(from_f_map)
                to_f_map = to_layer(to_f_map)
                if type(child) == nn.AdaptiveAvgPool2d:
                    from_f_map = torch.flatten(from_f_map, 1)
                    to_f_map = torch.flatten(to_f_map, 1)
                f_maps['from'].append(from_f_map)
                f_maps['to'].append(to_f_map)
                child_name = type(child).__name__
                layer_name = '{}_{}'.format(child_name, i)
                layer_info.append({
                    'name': layer_name,
                    'num_neurons': from_f_map.shape[1]
                })

        # Compute the sensitivity
        num_layers = len(layer_info)
        for img_idx, img in enumerate(imgs):
            for layer_idx in range(num_layers - 1):
                # Gradient (N, W, H)
                grad = autograd.grad(
                    f_maps['from'][-1][img_idx, self.args.label], 
                    f_maps['from'][layer_idx],
                    retain_graph=True
                )
                grad = grad[0][img_idx]

                layer_name = layer_info[layer_idx]['name']
                num_neurons = layer_info[layer_idx]['num_neurons']
                if layer_name not in self.sensitivity:
                    self.sensitivity[layer_name] = {}
                for neuron_idx in range(num_neurons):
                    neuron_grad = grad[neuron_idx]
                    from_f_map = f_maps['from'][layer_idx][img_idx]
                    to_f_map = f_maps['to'][layer_idx][img_idx]
                    delta_f_map = to_f_map - from_f_map
                    delta_f_map = delta_f_map[neuron_idx]
                    sens = torch.mul(neuron_grad, delta_f_map)
                    sens = torch.sum(sens).item()
                    neuron_id = '{}-{}'.format(layer_name, neuron_idx)
                    if neuron_id not in self.sensitivity[layer_name]:
                        self.sensitivity[layer_name][neuron_id] = []
                    self.sensitivity[layer_name][neuron_id].append(sens)


    def find_imp_evo_one_batch_inception_v3(self, imgs, labels):
        # Send input images and their labels to GPU
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        # Forward
        from_model_layers = list(self.from_model.model.children())
        to_model_layers = list(self.to_model.model.children())
        num_layers = len(from_model_layers)
        from_f_map, to_f_map = imgs, imgs
        f_maps, layer_info = {'from': [], 'to': []}, []
        for i in range(num_layers):
            from_layer = from_model_layers[i]
            to_layer = to_model_layers[i]
            child_name = type(from_layer).__name__
            if 'Aux' in child_name:
                continue
            if i == num_layers - 1:
                from_f_map = torch.flatten(from_f_map, 1)
                to_f_map = torch.flatten(to_f_map, 1)
            from_f_map = from_layer(from_f_map)
            to_f_map = to_layer(to_f_map)
            f_maps['from'].append(from_f_map)
            f_maps['to'].append(to_f_map)
            layer_name = '{}_{}'.format(child_name, i)
            layer_info.append({
                'name': layer_name,
                'num_neurons': from_f_map.shape[1]
            })

        # Compute the sensitivity
        for img_idx, img in enumerate(imgs):
            for layer_idx in range(num_layers - 1):
                # Gradient (N, W, H)
                grad = autograd.grad(
                    f_maps['from'][-1][img_idx, self.args.label], 
                    f_maps['from'][layer_idx],
                    retain_graph=True
                )
                grad = grad[0][img_idx]

                layer_name = layer_info[layer_idx]['name']
                num_neurons = layer_info[layer_idx]['num_neurons']
                if layer_name not in self.sensitivity:
                    self.sensitivity[layer_name] = {}
                for neuron_idx in range(num_neurons):
                    neuron_grad = grad[neuron_idx]
                    from_f_map = f_maps['from'][layer_idx][img_idx]
                    to_f_map = f_maps['to'][layer_idx][img_idx]
                    delta_f_map = to_f_map - from_f_map
                    delta_f_map = delta_f_map[neuron_idx]
                    sens = torch.mul(neuron_grad, delta_f_map)
                    sens = torch.sum(sens).item()
                    neuron_id = '{}-{}'.format(layer_name, neuron_idx)
                    if neuron_id not in self.sensitivity[layer_name]:
                        self.sensitivity[layer_name][neuron_id] = []
                    self.sensitivity[layer_name][neuron_id].append(sens)


    def save_sensitivity(self):
        path = self.data_path.get_path('find_important_evo-sensitivity')
        self.save_json(self.sensitivity, path)

    
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
        hyperpara_setting = self.data_path.gen_act_setting_str(
            'find_important_evo', '\n'
        )
        
        log = 'Find important evolution\n\n'
        log += 'from_model_nickname: {}\n'.format(self.args.from_model_nickname)
        log += 'from_model_path: {}\n'.format(self.args.from_model_path)
        log += 'to_model_nickname: {}\n'.format(self.args.to_model_nickname)
        log += 'to_model_path: {}\n'.format(self.args.to_model_path)
        log += 'label: {}\n'.format(self.args.label)
        log += hyperpara_setting + '\n\n'
        self.write_log(log, False)

    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        path = self.data_path.get_path('find_important_evo-log')
        with open(path, log_opt) as f:
            f.write(log + '\n')


    """
    Utils
    """
    def test_input(self):
        for batch_idx, (imgs, labels) in enumerate(self.data_loader):
            for idx in range(5):
                img = imgs[idx] * 255
                img = np.einsum('kij->ijk', img)
                file_path = f'img-{idx}.jpg'
                cv2.imwrite(
                    file_path, 
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )
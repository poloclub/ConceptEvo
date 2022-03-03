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

class ImportantEvo:
    """Find important evolution."""

    """
    Constructor
    """
    def __init__(self, args, data_path, from_model, to_model):
        self.args = args
        self.data_path = data_path

        self.device = None
        self.from_model = from_model
        self.to_model = to_model
        self.from_model.model.eval()
        self.to_model.model.eval()
        
        self.start_idx = -1
        self.end_idx = -1
        self.input_size = self.from_model.input_size
        self.num_classes = self.from_model.num_classes
        self.data_loader = None
        
        self.sensitivity = {}
        self.importance_score = {}


    """
    A wrapper function called by main.py
    """
    def find_important_evolution(self):
        self.write_first_log()
        self.init_setting()
        self.find_imp_evo()
        self.save_results()


    """
    Initial setting
    """
    def init_setting(self):
        self.init_device()
        self.init_data_loader()

    
    def init_device(self):
        self.device = torch.device(
            'cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() 
            else 'cpu'
        )
        print('Run on {}'.format(self.device))

    
    def init_data_loader(self):
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
        self.find_idx_training_dataset_for_class()
        self.class_dataset = torch.utils.data.Subset(
            self.training_dataset, 
            range(self.start_idx, self.end_idx)
        )
        self.data_loader = torch.utils.data.DataLoader(
            self.class_dataset, 
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )


    def find_idx_training_dataset_for_class(self):
        total = len(self.training_dataset)
        unit = int(total / self.num_classes)
        start = max(0, unit * (self.args.label - 2))
        end = min(total, unit * (self.args.label + 4))

        start_idx, end_idx = -1, -1
        with tqdm(total=(end - start)) as pbar:
            for i in range(start, end):
                img, label = self.training_dataset[i]
                if (self.args.label == label) and (start_idx == -1):
                    start_idx = i
                elif (self.args.label == label) and (end_idx == -1):
                    end_idx = -2
                if (self.args.label < label) and (end_idx == -2):
                    end_idx = i
                    break
                pbar.update(1)
        self.start_idx = start_idx
        self.end_idx = end_idx
        print(self.start_idx, self.end_idx)


    """
    Find important evolution
    """
    def find_imp_evo(self):
        tic, total = time(), len(self.data_loader.dataset)
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                
                # Evaluate important evolution for given label
                # if self.args.label not in labels:
                #     continue
                # nec_idxs = labels == self.args.label
                # nec_labels = labels[nec_idxs]
                # nec_imgs = imgs[nec_idxs]

                # Find important evolution for one batch
                # self.find_imp_evo_one_batch(nec_imgs, nec_labels)
                self.find_imp_evo_one_batch(imgs, labels)
                pbar.update(self.args.batch_size)

                total_num_so_far = (batch_idx + 1) * self.args.batch_size
                if total_num_so_far >= self.args.find_num_sample_imgs:
                    break

        self.compute_importance_score()
        toc = time()
        log = 'Find important evo: {:.2f} sec'.format(toc - tic)
        self.write_log(log)

    
    def find_imp_evo_one_batch(self, imgs, labels):
        # Forward
        from_f_maps, layer_info = self.from_model.forward(imgs)
        to_f_maps, layer_info = self.to_model.forward(imgs)
        f_maps = {'from': from_f_maps, 'to': to_f_maps}
        
        # Compute importance score
        self.compute_sensitivity(imgs, f_maps, layer_info)


    def compute_sensitivity(self, imgs, f_maps, layer_info):
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

                # Compute sensitivity
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

    
    def compute_importance_score(self):
        # Load sensitivity 
        if len(self.sensitivity) == 0:
            path = self.data_path.get_path('find_important_evo-sensitivity')
            self.sensitivity = self.load_json(path)

        # Compute score
        for layer_name in self.sensitivity:
            self.importance_score[layer_name] = []
            for neuron_id in self.sensitivity[layer_name]:
                sensitivities = self.sensitivity[layer_name][neuron_id]
                num_pos = len([s for s in sensitivities if s > 0])
                total_num = len(sensitivities)
                self.importance_score[layer_name].append({
                    'neuron': neuron_id,
                    'score': num_pos / total_num,
                    'total_num': total_num,
                    'num_positives': num_pos
                })

        # Sort score
        for layer_name in self.sensitivity:
            sorted_scores = sorted(
                self.importance_score[layer_name],
                key=lambda score_info: score_info['score'],
                reverse=True
            )
            self.importance_score[layer_name] = sorted_scores


    def save_results(self):
        if self.args.find_important_evo:
            # Save sensitivity score
            path = self.data_path.get_path('find_important_evo-sensitivity')
            self.save_json(self.sensitivity, path)

        # Save importance score
        path = self.data_path.get_path('find_important_evo-score')
        self.save_json(self.importance_score, path)

    
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
        if self.args.find_important_evo:
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
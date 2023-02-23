import os
import json
import numpy as np
from tqdm import tqdm
from time import time

import torch
from torchvision import datasets, transforms

class LayerAct:
    """Get layer activation"""

    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path
        self.model = model

        self.device = model.device
        self.layer_act = None

    """
    A wrapper function called by main.py
    """
    def compute_layer_act(self):
        self.init_setting()
        self.compute_and_save_layer_act()

    """
    Utils
    """    
    def init_setting(self):
        data_transform = transforms.Compose([
            transforms.Resize((self.model.input_size, self.model.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(*self.model.input_normalization)
        ])

        self.dataset = datasets.ImageFolder(
            self.data_path.get_path('train_data'),
            data_transform
        )

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )

        self.num_imgs = len(self.dataset)

    def write_first_log(self):
        log = 'Layer activation\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += 'layer: {}\n\n'.format(self.args.layer)
        self.write_log(log, False)

    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('layer_act-log'), log_opt) as f:
            f.write(log + '\n')
    
    """
    Compute layer activation
    """
    def compute_and_save_layer_act(self):
        self.write_first_log()
        tic, total = time(), len(self.data_loader)

        n = self.model.num_neurons[self.args.layer]
        self.layer_act = np.zeros((self.num_imgs, n))

        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                imgs = imgs.to(self.device)
                f_map = self.model.forward_until_given_layer(
                    self.args.layer, imgs
                )
                if len(f_map.shape) == 2:
                    f_map = f_map.cpu().data.numpy()
                    start_idx = batch_idx * self.args.batch_size
                    end_idx = start_idx + self.args.batch_size
                    self.layer_act[start_idx: end_idx] = f_map
                else:
                    max_act = torch.max(
                        torch.max(f_map, dim=2).values, 
                        dim=2
                    ).values.cpu().data.numpy()
                    B, N = max_act.shape
                    for i in range(B):
                        img_idx = (batch_idx * self.args.batch_size) + i
                        self.layer_act[img_idx] = max_act[i, :]
                pbar.update(1)

        toc = time()
        log = f'Compute layer act: {toc - tic:.2f} sec'
        self.write_log(log)
    
        p = self.data_path.get_path('layer_act')
        np.savetxt(p, self.layer_act)
                
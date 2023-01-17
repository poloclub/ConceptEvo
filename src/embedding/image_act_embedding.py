import json
from time import time

import umap

import torch
from torchvision import datasets, transforms

import numpy as np
from tqdm import tqdm


class ImageActEmb:
    """Generate image embeddings from activation of the base model"""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path
        self.model = model 

        self.num_imgs = model.num_training_imgs
        self.imgs = []

        self.stimuluated_neurons_by = {}
        self.img_max_act = None
        self.img_emb = None

        self.device = model.device
        self.training_dataset = None
        self.data_loader = None

    """
    A wrapper function called in main.py
    """
    def compute_img_embedding_from_activation(self):
        self.init_setting()
        self.get_layer_info()
        self.compute_img_emb_max_act()
        # self.reduce_dim()
        self.save_img_emb()

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
        self.layers_for_stimulus = self.model.layers_for_stimulus[:]
        self.num_neurons = self.model.num_neurons

    """
    Compute image embedding from max activation of neurons
    """
    def compute_img_emb_max_act(self):
        self.write_first_log()
        tic, total = time(), self.num_imgs

        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                # Get input images in a batch and their labels
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Update max activation embedding for the first layer
                f_map = self.model.forward_one_layer(0, imgs)
                layer_name = self.layers[0]['name']

                if layer_name == self.args.layer:
                    max_act = self.compute_max_act_of_feature_map(f_map)
                    self.update_img_emb(max_act, batch_idx, pbar)

                # Update max activation embedding for remaining layers
                for i in range(1, len(self.layers) - 1):
                    try:
                        f_map = self.model.forward_one_layer(i, f_map)
                        layer_name = self.layers[i]['name']
                        if layer_name == self.args.layer:
                            max_act = self.compute_max_act_of_feature_map(f_map)
                            self.update_img_emb(max_act, batch_idx, pbar)

                    except RuntimeError:
                        log = f'Error in find_stimulus for '
                        log += self.layers[i]['name']
                        #  self.write_log(log)
        toc = time()
        log = '{} sec'.format(toc - tic)
        self.write_log(log)

    def compute_max_act_of_feature_map(self, feature_map):
        # Get the maximum activation of the feature map. max_act: [B, N]
        # where B is batch size and N is the number of neurons
        return torch.max(torch.max(feature_map, dim=2).values, dim=2).values

    def update_img_emb(self, max_act, batch_idx, pbar):
        max_act = max_act.cpu().data.numpy()
        B, N = max_act.shape
        if self.img_max_act is None:
            self.img_max_act = np.zeros((self.num_imgs, N))
        
        for i in range(B):
            img_idx = (batch_idx * self.args.batch_size) + i
            self.img_max_act[img_idx] = max_act[i, :]
            pbar.update(1)

    def reduce_dim(self):
        reducer = umap.UMAP(n_components=self.args.dim)
        reducer = reducer.fit(self.img_max_act)
        self.img_emb = reducer.transform(self.img_max_act)

    def save_img_emb(self):
        file_path = self.data_path.get_path('img_act_emb')
        # np.savetxt(file_path, self.img_emb, fmt='%.3f')
        np.savetxt(file_path, self.img_max_act, fmt='%.3f')

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
        log = 'Image Embedding from activation of a base model\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += 'layer: {}\n\n'.format(self.args.layer)
        log += 'dim: {}\n\n'.format(self.args.dim)
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('img_act_emb-log'), log_opt) as f:
            f.write(log + '\n')

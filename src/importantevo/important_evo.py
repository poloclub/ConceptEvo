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
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path

        self.device = None
        self.from_model = None
        self.to_model = None

        self.input_size = -1        
        self.num_classes = 1000

        self.label_to_synset = {}


    """
    A wrapper function called by main.py
    """
    def find_important_evolution(self):
        self.init_setting()
        self.get_gradients()


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
        for synset, label in zip(df['synset'], df['tfrecord_label']):
            self.label_to_synset[int(label) - 1] = synset


    def gen_training_dataset_of_class(self):
        total = len(self.training_dataset)
        unit = int(total / self.num_classes)
        # start = max(0, unit * (self.args.label - 1))
        start = max(0, unit * (self.args.label))
        end = min(total, unit * (self.args.label + 2))

        num = 0
        with tqdm(total=(end - start)) as pbar:
            for i in range(start, end):
                img, label = self.training_dataset[i]
                if label == self.args.label:
                    self.class_training_dataset.append([img, label])
                    num += 1
                    if num == 10:
                        break
                elif label > self.args.label:
                    break
                pbar.update(1)
                

    """
    Get gradient for each layer
    """
    def get_gradients(self):
        for batch_idx, (imgs, labels) in enumerate(self.data_loader):

            # Send input images and their labels to GPU
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            # Forward
            from_model_layers = list(self.from_model.model.children())
            to_model_layers = list(self.to_model.model.children())
            num_layers = len(from_model_layers)
            from_f_map, to_f_map = imgs, imgs
            f_maps, layer_names = {'from': [], 'to': []}, []
            for i in range(num_layers):
                from_layer = from_model_layers[i]
                to_layer = to_model_layers[i]
                child_name = type(from_layer).__name__
                if 'Aux' in child_name:
                    continue
                layer_name = '{}_{}'.format(child_name, i)
                layer_names.append(layer_name)
                if i == num_layers - 1:
                    from_f_map = torch.flatten(from_f_map, 1)
                    to_f_map = torch.flatten(to_f_map, 1)
                from_f_map = from_layer(from_f_map)
                to_f_map = to_layer(to_f_map)
                f_maps['from'].append(from_f_map)
                f_maps['to'].append(to_f_map)

            # Gradient of all layers
            grads = []
            for img_idx, img in enumerate(imgs):
                for i in range(num_layers):
                    # Gradient of the layer (B, N, W, H)
                    # where B = batch size and N = the number of neurons
                    grad = autograd.grad(
                        f_maps['from'][-1][img_idx, self.args.label], 
                        f_maps['from'][i]
                    )
                    grad = grad[0]
                    grads.append(grad)
    
    
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


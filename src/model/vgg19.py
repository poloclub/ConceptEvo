import os
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.utils.data as data_utils
from torchvision import datasets, transforms

class Vgg19:
    def __init__(self, args, data_path, pretrained=False, from_to=None):
        self.args = args
        self.data_path = data_path

        self.input_size = 256
        self.num_classes = 1000
        self.num_training_imgs = -1

        self.model = None
        self.pretrained = pretrained
        self.from_to = from_to
        self.layers = []
        self.conv_layers = []
        self.num_neurons = {}

        self.need_loading_a_saved_model = len(self.args.model_path) > 0
        self.ckpt = None

        self.device = None
        self.training_dataset = None
        self.test_dataset = None
        self.training_data_loader = None
        self.test_data_loader = None
        self.optimizer = None
        self.criterion = None


    def init_basic_setting(self):
        self.init_device()
        self.init_training_datasets_and_loader()
        self.load_checkpoint()


    def init_model(self):
        # Initialize an empty model
        self.model = models.vgg19(pretrained=self.pretrained)

        # Load a saved model
        if self.need_loading_a_saved_model:
            self.model.load_state_dict(self.ckpt['model_state_dict'])
        
        # Set all parameters learnable
        self.set_all_parameter_requires_grad()

        # Send the model to GPU
        self.model.to(self.device)

        # Update layer info
        self.get_layer_info()


    def init_training_setting(self):
        self.init_optimizer()
        self.init_criterion()

    
    def load_checkpoint(self):
        if self.need_loading_a_saved_model:
            if self.from_to == 'from':
                self.ckpt = torch.load(self.args.from_model_path)
            elif self.from_to == 'to':
                self.ckpt = torch.load(self.args.to_model_path)
            else:
                self.ckpt = torch.load(self.args.model_path)


    def set_all_parameter_requires_grad(self):
        for param in self.model.parameters():
            param.requires_grad = True


    def init_device(self):
        self.device = torch.device(
            'cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() 
            else 'cpu'
        )
        print('Run on {}'.format(self.device))


    def init_training_datasets_and_loader(self):
        data_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225]
            )
        ])

        self.training_dataset = datasets.ImageFolder(
            self.data_path.get_path('train_data'),
            data_transform
        )
        self.num_training_imgs = len(self.training_dataset)

        self.test_dataset = datasets.ImageFolder(
            self.data_path.get_path('test_data'),
            data_transform
        )

        self.training_data_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )

        self.test_data_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )

    
    def init_optimizer(self):
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.args.lr, 
            momentum=self.args.momentum
        )
        if self.need_loading_a_saved_model:
            self.optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])
            for param_group in self.optimizer.state_dict()['param_groups']:
                param_group['lr'] = self.args.lr
                param_group['momentum'] = self.args.momentum


    def init_criterion(self):
        if self.need_loading_a_saved_model:
            self.criterion = self.ckpt['loss']
        else:
            self.criterion = nn.CrossEntropyLoss()


    def get_layer_info(self):
        model_children = list(self.model.children())
        for i, child in enumerate(model_children):
            if type(child) == nn.Sequential:
                for j, layer in enumerate(child.children()):
                    layer_name = '{}_{}_{}_{}'.format(
                        type(child).__name__, i,
                        type(layer).__name__, j
                    )
                    self.update_layer_info(layer_name, layer)
            else:
                child_name = type(child).__name__
                layer_name = '{}_{}'.format(child_name, i)
                self.update_layer_info(layer_name, child)


    def update_layer_info(self, layer_name, layer):
        self.layers.append({
            'name': layer_name,
            'layer': layer
        })
        if type(layer) == nn.Conv2d:
            self.conv_layers.append(layer_name)
            self.num_neurons[layer_name] = layer.out_channels


    def test_model(self):
        # Make the first log
        self.write_test_first_log()

        # Get ready to train the model
        tic = time()
        total = len(self.test_data_loader.dataset)

        # Variables to evaluate the training performance
        top1_test_corrects, topk_test_corrects = 0, 0

        # Measure test set accuracy
        with tqdm(total=total) as pbar:
            for test_imgs, test_labels in self.test_data_loader:

                top1_corrects, topk_corrects = \
                    self.test_one_batch(test_imgs, test_labels)

                top1_test_corrects += top1_corrects
                topk_test_corrects += topk_corrects

                pbar.update(self.args.batch_size)

        toc = time()

        # Save log
        log = 'total = {}\n'.format(total)
        log += 'top1 test accuracy = {} / {} = {}\n'.format(
            top1_test_corrects, total, top1_test_corrects / total
        )
        log += 'topk test accuracy = {} / {} = {}\n'.format(
            topk_test_corrects, total, topk_test_corrects / total
        )
        log += 'time: {} sec\n'.format(toc - tic)
        print(log)
        self.write_log(log, append=True, test=True)


    def test_one_batch(self, test_imgs, test_labels):
        # Get test images and labels
        test_imgs = test_imgs.to(self.device)
        test_labels = test_labels.to(self.device)

        # Prediction
        outputs = self.model(test_imgs)
        _, topk_test_preds = outputs.topk(
            k=self.args.topk, dim=1
        )
        top1_test_preds = topk_test_preds[:, 0]
        topk_test_preds = topk_test_preds.t()

        # Number of correct top-k prediction in test set
        topk_test_corrects = 0
        for k in range(self.args.topk):
            topk_test_corrects += torch.sum(
                topk_test_preds[k] == test_labels.data
            )

        # Number of correct top-1 prediction in training set
        top1_test_corrects = torch.sum(
            top1_test_preds == test_labels.data
        )

        return top1_test_corrects, topk_test_corrects


    def forward(self, imgs):
        # Initialize feature maps
        imgs = imgs.to(self.device)
        f_map, f_maps = imgs, []

        # Layer information
        layers = list(self.model.children())
        layer_info = []

        # Forward and save feature map for each layer
        for i, child in enumerate(layers):
            if type(child) == nn.Sequential:
                for j, layer in enumerate(child.children()):
                    # Compute and save f_map
                    f_map = layer(f_map)
                    f_maps.append(f_map)

                    # Save layer info
                    layer_name = '{}_{}_{}_{}'.format(
                        type(child).__name__, i,
                        type(layer).__name__, j
                    )
                    layer_info.append({
                        'name': layer_name,
                        'num_neurons': f_map.shape[1],
                    })
            else:
                # Compute f_map
                layer = layers[i]
                f_map = layer(f_map)

                # Flatten before fully connected layer
                if type(child) == nn.AdaptiveAvgPool2d:
                    f_map = torch.flatten(f_map, 1)

                # Save f_map
                f_maps.append(f_map)

                # Save layer info
                child_name = type(child).__name__
                layer_name = '{}_{}'.format(child_name, i)
                layer_info.append({
                    'name': layer_name,
                    'num_neurons': f_map.shape[1],
                })

        return f_maps, layer_info


    def write_log(self, log, append=True, test=False):
        log_opt = 'a' if append else 'w'
        key = 'test-log' if test else 'train-log'
        path = self.data_path.get_path(key)
        with open(path, log_opt) as f:
            f.write(log + '\n')

    
    def write_log_with_log_info(self, log_info):
        log = ', '.join([f'{key}={log_info[key]}' for key in log_info])
        self.write_log(log)
    

    def write_training_first_log(self):
        log_param_sets = {
            'batch_size': self.args.batch_size,
            'lr': self.args.lr,
            'momentum': self.args.momentum,
            'k': self.args.topk,
            'start_model_path': self.args.model_path
        }
        first_log = ', '.join(
            [f'{p}={log_param_sets[p]}' for p in log_param_sets]
        )
        self.write_log(first_log, append=True)


    def write_training_epoch_log(self, tic, epoch, stats):
        num_training_data = len(self.training_data_loader.dataset)

        running_loss, top1_train_corrects, topk_train_corrects = stats

        epoch_loss = running_loss / num_training_data
        epoch_top1_train_acc = \
            top1_train_corrects.double() / num_training_data
        epoch_topk_train_acc = \
            topk_train_corrects.double() / num_training_data

        self.write_log_with_log_info({
            'epoch': epoch,
            'cumulative_time_sec': '{:.2f}'.format(time() - tic),
            'loss': '{:.4f}'.format(epoch_loss),
            'top1_train_acc': '{:.4f}'.format(epoch_top1_train_acc),
            'topk_train_acc': '{:.4f}'.format(epoch_topk_train_acc)
        })

    def write_test_first_log(self):
        log_param_sets = {
            'model_nickname': self.args.model_nickname,
            'model_path': self.args.model_path,
            'k': self.args.topk
        }
        first_log = '\n'.join(
            [f'{p}={log_param_sets[p]}' for p in log_param_sets]
        )
        self.write_log(first_log, append=True, test=True)
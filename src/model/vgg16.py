import copy
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from tqdm import tqdm


class Vgg16:
    """Defines Vgg16 model"""

    def __init__(self, args, data_path, pretrained=False, from_to=None):
        self.args = args
        self.data_path = data_path 

        self.input_size = 224
        self.num_classes = 1000
        self.num_training_imgs = -1
        self.input_normalization = [
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        ]

        self.model = None
        self.pretrained = pretrained
        self.from_to = from_to
        self.layers = []
        self.conv_layers = []
        self.num_neurons = {}

        self.need_loading_a_saved_model = None
        self.ckpt = None
        self.training_start_epoch = 0

        self.device = None
        self.training_dataset = None
        self.test_dataset = None
        self.training_data_loader = None
        self.class_dataset = None
        self.test_data_loader = None
        self.optimizer = None
        self.criterion = None

        self.init()

    """
    Initialize the model and training settings
    """
    def init(self):
        self.init_device()
        self.init_model()
        self.init_training_setting()

    """
    Initialize device
    """
    def init_device(self):
        self.device = torch.device(
            'cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() 
            else 'cpu'
        )
        print('Run on {}'.format(self.device))

    """
    Initialize model
    """
    def init_model(self):
        # Load checkpoint if necessary
        self.check_if_need_to_load_model()
        self.load_checkpoint()

        # Create an empty model
        if self.pretrained:
            self.model = models.vgg16(weights='DEFAULT')
        else:
            self.model = models.vgg16()

        # Load a saved model
        self.load_saved_model()

        # Reset the final layer
        self.reset_final_layer()
        
        # Set all parameters learnable
        self.set_all_parameter_requires_grad()

        # Send the model to GPU
        self.model.to(self.device)

        # Update layer info
        self.get_layer_info()
        self.save_layer_info()

        # Set criterion
        self.init_criterion()

    def check_if_need_to_load_model(self):
        check1 = len(self.args.model_path) > 0
        check2 = self.args.model_path != 'DO_NOT_NEED_CURRENTLY'
        check3 = not self.pretrained
        self.need_loading_a_saved_model = check1 and check2 and check3

        if self.need_loading_a_saved_model and self.args.train:
            last_epoch = int(self.args.model_path.split('-')[-1].split('.')[0])
            self.training_start_epoch = last_epoch + 1

    def load_checkpoint(self):
        if self.need_loading_a_saved_model:
            if self.from_to == 'from':
                self.ckpt = torch.load(
                    self.args.from_model_path,
                    map_location=self.device
                )
            elif self.from_to == 'to':
                self.ckpt = torch.load(
                    self.args.to_model_path,
                    map_location=self.device
                )
            else:
                self.ckpt = torch.load(
                    self.args.model_path,
                    map_location=self.device
                )

    def load_saved_model(self):
        if self.need_loading_a_saved_model:
            if 'model_state_dict' in self.ckpt:
                self.model.load_state_dict(self.ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(self.ckpt)

    def reset_final_layer(self):
        if self.args.train:
            num_feautres = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(
                num_feautres, 
                self.num_classes
            )

    def set_all_parameter_requires_grad(self):
        for param in self.model.parameters():
            param.requires_grad = True

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
    
    def save_layer_info(self):
        if self.args.train:
            # Save model information
            s = str(self.model)
            p = self.data_path.get_path('model-info')
            with open(p, 'a') as f:
                f.write(s + '\n')

            # Save layer names
            p = self.data_path.get_path('layer-info')
            for layer in self.layers:
                with open(p, 'a') as f:
                    f.write(layer['name'] + '\n')

    def init_criterion(self):
        if self.need_loading_a_saved_model and ('loss' in self.ckpt):
            self.criterion = self.ckpt['loss']
        else:
            self.criterion = nn.CrossEntropyLoss()

    """
    Initialize training settings
    """
    def init_training_setting(self):
        self.init_training_datasets_and_loader()
        self.init_optimizer()

    def init_training_datasets_and_loader(self):
        data_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(*self.input_normalization)
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
            if 'optimizer_state_dict' in self.ckpt:
                self.optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])
                for param_group in self.optimizer.state_dict()['param_groups']:
                    param_group['lr'] = self.args.lr
                    param_group['momentum'] = self.args.momentum

    """
    Train the model
    """
    def train_model(self):
        # Make the first log
        self.write_training_first_log()
    
        # Get ready to train the model
        tic = time()
        total = self.args.num_epochs * len(self.training_data_loader.dataset)

        # Train the model
        with tqdm(total=total) as pbar:
            for epoch in range(self.args.num_epochs):
                # Update parameters with one epoch's data
                running_loss, top1_train_corrects, topk_train_corrects = \
                    self.train_one_epoch(pbar)
                
                # Measure test accuracy
                test_total, top1_test_corrects, topk_test_corrects = \
                    self.test_model(write_log=False, test_on='test')

                # Save the model
                self.save_model(self.training_start_epoch + epoch)

                # Save log
                self.write_training_epoch_log(
                    tic, self.training_start_epoch + epoch,
                    [
                        running_loss, top1_train_corrects, topk_train_corrects,
                        test_total, top1_test_corrects, topk_test_corrects
                    ]
                )

    def train_one_epoch(self, pbar):
        # Set model to training mode
        self.model.train()

        # Variables to evaluate the training performance
        running_loss = 0.0
        top1_train_corrects, topk_train_corrects = 0, 0

        # Update parameters with one epoch's data
        for imgs, labels in self.training_data_loader:

            # Get input image and its label
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            # Forward and backward
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                # Forward
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                # Prediction
                _, topk_train_preds = outputs.topk(
                    k=self.args.topk, 
                    dim=1
                )
                top1_train_preds = topk_train_preds[:, 0]
                topk_train_preds = topk_train_preds.t()

                # Backward
                loss.backward()
                self.optimizer.step()

            # Number of correct top-k prediction in training set
            for k in range(self.args.topk):
                topk_train_corrects += torch.sum(
                    topk_train_preds[k] == labels.data
                )
            
            # Number of correct top-1 prediction in training set
            top1_train_corrects += torch.sum(
                top1_train_preds == labels.data
            )

            # Loss
            running_loss += loss.item() * imgs.size(0)
            
            # Update pbar
            pbar.update(self.args.batch_size)

        top1_train_corrects = top1_train_corrects.double()
        topk_train_corrects = topk_train_corrects.double()

        return running_loss, top1_train_corrects, topk_train_corrects

    """
    Test model
    """
    def test_model(self, write_log=True, test_on='test'):
        # Make the first log
        if write_log:
            self.write_test_first_log()

        # Test model
        if test_on == 'training':
            total, top1_corrects, topk_corrects = \
                self.measure_acc(self.training_data_loader)
        elif test_on == 'test':
            total, top1_corrects, topk_corrects = \
                self.measure_acc(self.test_data_loader)
        else:
            err = 'Unknown option for test_on={} in test_model'.format(test_on)
            raise ValueError(err) 

        # Save log
        if write_log:
            acc_log = self.gen_acc_log([total, top1_corrects, topk_corrects])
            log = ('-' * 10) + test_on + '\n' + acc_log + ('-' * 10) + '\n'
            self.write_test_log(log)

        return total, top1_corrects, topk_corrects

    def measure_acc(self, data_loader):
        total = len(data_loader.dataset)
        final_top1_corrects, final_topk_corrects = 0, 0
        for imgs, labels in data_loader:
            top1_corrects, topk_corrects = self.test_one_batch(imgs, labels)
            final_top1_corrects += top1_corrects
            final_topk_corrects += topk_corrects
        return total, final_top1_corrects, final_topk_corrects

    def gen_acc_log(self, stats):
        total, top1_corrects, topk_corrects = stats
        log = 'total = {}\n'.format(total)
        log += 'top1 accuracy = {} / {} = {}\n'.format(
            top1_corrects, total, top1_corrects / total
        )
        log += 'topk accuracy = {} / {} = {}\n'.format(
            topk_corrects, total, topk_corrects / total
        )
        return log
    
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

        top1_test_corrects = top1_test_corrects.double()
        top1_test_corrects = top1_test_corrects.double()

        return top1_test_corrects, topk_test_corrects
    
    """
    Save model
    """
    def save_model(self, epoch):
        path = self.data_path.get_model_path_during_training(epoch)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion,
                'epoch': epoch
            }, 
            path
        )

    """
    Forward
    """
    def forward_one_layer(self, layer_idx, prev_f_map):
        f_map = self.layers[layer_idx]['layer'](prev_f_map)
        return f_map

    """
    Log for training the model
    """
    def write_training_first_log(self):
        log_param_sets = {
            'batch_size': self.args.batch_size,
            'lr': self.args.lr,
            'momentum': self.args.momentum,
            'k': self.args.topk,
            'model_path': self.args.model_path
        }
        first_log = ', '.join(
            [f'{p}={log_param_sets[p]}' for p in log_param_sets]
        )
        self.write_training_log(first_log)

    def write_training_epoch_log(self, tic, epoch, stats):
        num_training_data = len(self.training_data_loader.dataset)

        running_loss, top1_train_corrects, topk_train_corrects, \
            test_total, top1_test_corrects, topk_test_corrects = stats

        epoch_loss = running_loss / num_training_data
        epoch_top1_train_acc = top1_train_corrects / num_training_data
        epoch_topk_train_acc = topk_train_corrects / num_training_data
        epoch_top1_test_acc = top1_test_corrects / test_total
        epoch_topk_test_acc = topk_test_corrects / test_total

        log = self.gen_training_epoch_log({
            'epoch': epoch,
            'cumulative_time_sec': '{:.2f}'.format(time() - tic),
            'loss': '{:.4f}'.format(epoch_loss),
            'top1_train_acc': '{:.4f}'.format(epoch_top1_train_acc),
            'topk_train_acc': '{:.4f}'.format(epoch_topk_train_acc),
            'top1_test_acc': '{:.4f}'.format(epoch_top1_test_acc),
            'topk_test_acc': '{:.4f}'.format(epoch_topk_test_acc),
        })

        self.write_training_log(log)

    def gen_training_epoch_log(self, stats):
        log = ', '.join([f'{key}={stats[key]}' for key in stats])
        return log

    def write_training_log(self, log):
        path = self.data_path.get_path('train-log')
        with open(path, 'a') as f:
            f.write(log + '\n')

    """
    Log for testing the model
    """
    def write_test_first_log(self):
        log_param_sets = {
            'model_nickname': self.args.model_nickname,
            'model_path': self.args.model_path,
            'k': self.args.topk
        }
        first_log = '\n'.join(
            [f'{p}={log_param_sets[p]}' for p in log_param_sets]
        )
        self.write_test_log(first_log)

    def write_test_log(self, log):
        path = self.data_path.get_path('test-log')
        with open(path, 'a') as f:
            f.write(log + '\n')

    """
    """
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
        
    def load_model(self, epoch):
        path = self.data_path.get_model_path_during_training(epoch)
        self.load_model_from_path(path)


    def load_model_from_path(self, path):
        self.init_model()
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.set_all_parameter_requires_grad()

    
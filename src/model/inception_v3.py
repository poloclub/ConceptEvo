import os
import copy
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms

class InceptionV3:
    """Defines InceptionV3 model"""

    def __init__(self, args, data_path, pretrained=False):
        self.args = args
        self.data_path = data_path

        self.input_size = 299
        self.num_classes = 1000
        self.num_training_imgs = -1

        self.model = None
        self.pretrained = pretrained
        self.layers = []
        self.conv_layers = []
        self.num_neurons = {}

        self.continuing_from_a_model = len(self.args.model_path) > 0
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
        self.model = models.inception_v3(pretrained=self.pretrained)

        # Load a saved model if we continue the training from it
        if self.continuing_from_a_model:
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
        if self.continuing_from_a_model:
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
        if self.continuing_from_a_model:
            self.optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])
            for param_group in self.optimizer.state_dict()['param_groups']:
                param_group['lr'] = self.args.lr
                param_group['momentum'] = self.args.momentum


    def init_criterion(self):
        if self.continuing_from_a_model:
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
                if self.is_inceptionV3_aux(child_name):
                    continue
                layer_name = '{}_{}'.format(child_name, i)
                self.update_layer_info(layer_name, child)

    
    def is_inceptionV3_aux(self, blk_name):
        return 'Aux' in blk_name

    
    def update_layer_info(self, layer_name, layer, next_layer=None):
        self.layers.append({
            'name': layer_name,
            'layer': layer
        })
        if 'Conv2d' in layer_name:
            conv_layer = layer.__dict__['_modules']['conv']
            self.conv_layers.append(layer_name)
            self.num_neurons[layer_name] = conv_layer.out_channels
        elif 'Inception' in layer_name:

            self.conv_layers.append(layer_name)

            # Count the number of neurons in blocks to be concatenated
            children = layer.__dict__['_modules']
            children_names = list(children.keys())
            num_neurons_concat_blk = {}
            for child_name in children_names:
                
                # The number of neurons of the child
                child = children[child_name]
                child_conv = child.__dict__['_modules']['conv']
                num_neurons = child_conv.out_channels

                # The number of neurons of all branches
                name_tokens = child_name.split('_')
                if len(name_tokens) == 1:
                    num_neurons_concat_blk[child_name] = {
                        'apdx': None,
                        'num': num_neurons
                    }
                else:
                    apdx = child_name.split('_')[-1]
                    apdx = ''.join(filter(str.isdigit, apdx))

                    if len(apdx) == 0:
                        num_neurons_concat_blk[child_name] = {
                            'apdx': None,
                            'num': num_neurons
                        }
                    else:
                        apdx = int(apdx)
                        blk = '_'.join(child_name.split('_')[:-1])
                        if blk not in num_neurons_concat_blk:
                            num_neurons_concat_blk[blk] = {
                                'apdx': apdx,
                                'num': num_neurons
                            }
                        else:
                            prev_apdx = num_neurons_concat_blk[blk]['apdx']
                            if prev_apdx < apdx:
                                num_neurons_concat_blk[blk] = {
                                    'apdx': apdx,
                                    'num': num_neurons
                                }
                            elif prev_apdx == apdx:
                                prev_sum = num_neurons_concat_blk[blk]['num']
                                num_neurons_concat_blk[blk] = {
                                    'apdx': apdx,
                                    'num': prev_sum + num_neurons
                                }

            # The total number of neurons in the concatenated layer
            num_out_channels = 0
            for child in num_neurons_concat_blk:
                num_out_channels += num_neurons_concat_blk[child]['num']

            if ('InceptionB' in layer_name) or ('InceptionD' in layer_name):
                fst_child = list(children.values())[0]
                fst_child_conv_layer = fst_child.__dict__['_modules']['conv']
                num_in_channel = fst_child_conv_layer.in_channels
                num_out_channels += num_in_channel
            
            # Save the total number of neurons
            self.num_neurons[layer_name] = num_out_channels
            

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

                # Save the model
                self.save_model(epoch)

                # Save log
                self.write_training_epoch_log(
                    tic, epoch,
                    [running_loss, top1_train_corrects, topk_train_corrects,]
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

            # Forward
            outputs = self.model(imgs).logits
            loss = self.criterion(outputs, labels)

            # Prediction
            _, topk_train_preds = outputs.topk(
                k=self.args.topk, 
                dim=1
            )
            top1_train_preds = topk_train_preds[:, 0]
            topk_train_preds = topk_train_preds.t()

            # Backward
            self.optimizer.zero_grad()
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

        return running_loss, top1_train_corrects, topk_train_corrects

    
    def measure_test_accuracy(self):
        # Variables to evaluate the training performance
        top1_test_corrects, topk_test_corrects = 0, 0

        # Measure test set accuracy
        for test_imgs, test_labels in self.test_data_loader:

            # Get test images and labels
            test_imgs = test_imgs.to(self.device)
            test_labels = test_labels.to(self.device)

            # Prediction
            outputs = self.model(test_imgs).logits
            _, topk_test_preds = outputs.topk(
                k=self.args.topk, dim=1
            )
            top1_test_preds = topk_test_preds[:, 0]
            topk_test_preds = topk_test_preds.t()

            # Number of correct top-k prediction in test set
            for k in range(self.args.topk):
                topk_test_corrects += torch.sum(
                    topk_test_preds[k] == test_labels.data
                )

            # Number of correct top-1 prediction in training set
            top1_test_corrects += torch.sum(
                top1_test_preds == test_labels.data
            )

        return top1_test_corrects, topk_test_corrects


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
        
        
    def load_model(self, epoch):
        path = self.data_path.get_model_path_during_training(epoch)
        self.load_model_from_path(path)


    def load_model_from_path(self, path):
        self.init_model()
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.set_all_parameter_requires_grad()
        

    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('train-log'), log_opt) as f:
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
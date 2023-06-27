import os
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.models as models
from torchvision import datasets, transforms
from tqdm import tqdm


class Vgg19:
    def __init__(self, args, data_path, pretrained=False, from_to=None):
        self.args = args
        self.data_path = data_path

        self.input_size = 256
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
        self.layers_for_ex_patch = []
        self.num_neurons = {}

        self.need_loading_a_saved_model = None
        self.ckpt = None
        self.training_start_epoch = 0

        self.device = None
        self.training_dataset = None
        self.test_dataset = None
        self.training_data_loader = None
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
            self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        else:
            self.model = models.vgg19()

        # Load a saved model
        self.load_saved_model()
        
        # Set all parameters learnable
        self.set_all_parameter_requires_grad()

        # Send the model to GPU
        self.model.to(self.device)

        # Update layer info
        self.get_layer_info()
        self.save_layer_info()

        # Update the number of neurons for each layer
        self.get_num_neurons()

        # Set criterion
        self.init_criterion()

    def check_if_need_to_load_model(self):
        if self.from_to is None:
            self.need_loading_a_saved_model = \
                self.data_path.get_path('model_path') is not None
        elif self.from_to == 'from':
            self.need_loading_a_saved_model = \
                self.data_path.get_path('from_model_path') is not None
        elif self.from_to == 'to':
            self.need_loading_a_saved_model = \
                self.data_path.get_path('to_model_path') is not None
        else:
            raise ValueError(f'Unknown from_to is given: "{self.from_to}"')

        if self.args.train and self.need_loading_a_saved_model:
            last_epoch = int(self.args.model_path.split('-')[-1].split('.')[0])
            self.training_start_epoch = last_epoch + 1
    
    def load_checkpoint(self):
        if self.need_loading_a_saved_model:
            if self.from_to == 'from':
                model_path = self.data_path.get_path('from_model_path')
            elif self.from_to == 'to':
                model_path = self.data_path.get_path('to_model_path')
            else:
                model_path = self.data_path.get_path('model_path')
            self.ckpt = torch.load(
                model_path,
                map_location=self.device
            )

    def load_saved_model(self):
        if self.need_loading_a_saved_model:
            if 'model_state_dict' in self.ckpt:
                self.model.load_state_dict(self.ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(self.ckpt)

    def set_all_parameter_requires_grad(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_layer_info(self):
        for module_name, module in self.model.named_children():
            if module_name == 'features':
                for i, layer in enumerate(module.children()):
                    layer_type = type(layer).__name__
                    layer_name = f'{module_name}_{layer_type}_{i}'
                    self.update_layer_info(layer_name, layer)
            elif module_name == 'avgpool':
                self.update_layer_info(module_name, module)
                self.update_layer_info('flatten', nn.Flatten(start_dim=1, end_dim=-1))
            else:
                self.update_layer_info(module_name, module)

    def update_layer_info(self, layer_name, layer):
        self.layers.append({
            'name': layer_name,
            'layer': layer
        })
        if type(layer) == nn.Conv2d:
            self.layers_for_ex_patch.append(layer_name)

    def save_layer_info(self):
        # Save model information
        s = str(self.model)
        p = self.data_path.get_path('model_info')
        with open(p, 'w') as f:
            f.write(s + '\n')

        # Save layer names
        p = self.data_path.get_path('layer_info')
        log = '\n'.join([layer['name'] for layer in self.layers])
        with open(p, 'w') as f:
            f.write(log + '\n')

    def get_num_neurons(self):
        dummy_input = torch.zeros(1, 3, self.input_size, self.input_size)
        f_map = dummy_input.to(self.device)
        for i, layer in enumerate(self.layers):
            layer_name = layer['name']
            f_map = layer['layer'](f_map)
            self.num_neurons[layer_name] = f_map.shape[1]

    def init_criterion(self):
        if self.need_loading_a_saved_model and ('loss' in self.ckpt):
            self.criterion = self.ckpt['loss']
        else:
            self.criterion = nn.CrossEntropyLoss()

    """
    Initialize training settings
    """
    def init_training_setting(self):
        if self.data_path is not None:
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

    def test_model_by_class(self, write_log=True, test_on='test'):
        # Make the first log
        if write_log:
            self.write_test_by_class_first_log()

        # Test model
        if test_on == 'training':
            total_dict, top1_corrects_dict, topk_corrects_dict = \
                self.measure_acc_by_class(self.training_data_loader)
        elif test_on == 'test':
            total_dict, top1_corrects_dict, topk_corrects_dict = \
                self.measure_acc_by_class(self.test_data_loader)
        else:
            err = 'Unknown option for test_on={} in test_model'.format(test_on)
            raise ValueError(err) 

        # Save log
        if write_log:
            log = ('-' * 10) + test_on + '\n'
            log += 'label, top1, topk, total\n'
            max_label = max(list(total_dict.keys()))
            for label in range(max_label):
                total, t1, tk = 0, 0, 0
                if label in total_dict:
                    total = total_dict[label]
                if label in top1_corrects_dict:
                    t1 = top1_corrects_dict[label]
                if label in topk_corrects_dict:
                    tk = topk_corrects_dict[label]
                
                if total > 0:
                    log += f'{label}, {t1 / total:.2f}, {tk / total:.2f}, {total}\n'
                else:
                    log += f'{label}, {0:.2f}, {0:.2f}, {total}\n'
            self.write_test_by_class_log(log)

        return total_dict, top1_corrects_dict, topk_corrects_dict

    def measure_acc(self, data_loader):
        total = len(data_loader.dataset)
        final_top1_corrects, final_topk_corrects = 0, 0
        for imgs, labels in data_loader:
            top1_corrects, topk_corrects = self.test_one_batch(imgs, labels)
            final_top1_corrects += top1_corrects
            final_topk_corrects += topk_corrects
        return total, final_top1_corrects, final_topk_corrects

    def measure_acc_by_class(self, data_loader):
        final_top1_corrects_dict, final_topk_corrects_dict = {}, {}
        total_dict = {}
        for imgs, labels in data_loader:
            top1_test_corrects_dict, topk_test_corrects_dict, total_num_dict = \
                self.test_by_class_one_batch(imgs, labels)

            for label in total_num_dict:
                n = total_num_dict[label]
                if label not in total_dict:
                    total_dict[label] = 0
                total_dict[label] += n    
            
            for label in top1_test_corrects_dict:
                n = top1_test_corrects_dict[label]
                if label not in final_top1_corrects_dict:
                    final_top1_corrects_dict[label] = 0
                final_top1_corrects_dict[label] += n

            for label in topk_test_corrects_dict:
                n = topk_test_corrects_dict[label]
                if label not in final_topk_corrects_dict:
                    final_topk_corrects_dict[label] = 0
                final_topk_corrects_dict[label] += n
            
        return total_dict, final_top1_corrects_dict, final_topk_corrects_dict

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

    def test_by_class_one_batch(self, test_imgs, test_labels):
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

        # Number of correct prediction in the test set
        topk_test_corrects_dict = {}
        top1_test_corrects_dict = {}
        total_num_dict = {}
        for k in range(self.args.topk):
            test_results = topk_test_preds[k] == test_labels.data
            correct_labels = test_labels[test_results].cpu().numpy()
            for label in correct_labels:
                # Total number of data for each class
                if label not in total_num_dict:
                    total_num_dict[label] = 0
                total_num_dict[label] += 1

                # top-1 prediction
                if k == 0:
                    if label not in top1_test_corrects_dict:
                        top1_test_corrects_dict[label] = 0
                    top1_test_corrects_dict[label] += 1

                # top-k prediction
                if label not in topk_test_corrects_dict:
                    topk_test_corrects_dict[label] = 0
                topk_test_corrects_dict[label] += 1
    
        return top1_test_corrects_dict, topk_test_corrects_dict, total_num_dict

    """
    Forward
    """
    def forward_one_layer(self, layer_idx, prev_f_map):
        # Compute a forward pass for a single layer, 
        # given the layer index and previous feature map
        f_map = self.layers[layer_idx]['layer'](prev_f_map)
        return f_map

    def forward(self, imgs):
        # Perform a forward pass through the entire layer,
        # obtaining feature map for each layer

        # Initialize feature maps
        imgs = imgs.to(self.device)
        f_map, f_maps = imgs, []

        # Forward
        for i, layer in enumerate(self.layers):
            f_map = self.forward_one_layer(i, f_map)
            f_maps.append(f_map)
        return f_maps

    def forward_until_the_end(self, layer_idx, prev_f_map):
        num_layers = len(self.layers)
        f_map = prev_f_map.clone().detach()
        for i in range(layer_idx, num_layers):
            f_map = self.forward_one_layer(i, f_map)
        return f_map

    def forward_until_given_layer(self, layer_name, imgs):
        f_map = imgs.clone().detach()
        for i, layer in enumerate(self.layers):
            f_map = self.forward_one_layer(i, f_map)
            if layer['name'] == layer_name:
                break
        return f_map
        
    """
    Load and save model
    """
    def load_model(self, epoch):
        path = self.data_path.get_model_path_during_training(epoch)
        self.load_model_from_path(path)

    def load_model_from_path(self, path):
        self.init_model()
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.set_all_parameter_requires_grad()

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
    Log for training the model
    """
    def write_training_first_log(self):
        log_param_sets = {
            'batch_size': self.args.batch_size,
            'lr': self.args.lr,
            'momentum': self.args.momentum,
            'k': self.args.topk,
            'model_path': self.data_path.get_path('model_path'),
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
        epoch_top1_train_acc = \
            top1_train_corrects.double() / num_training_data
        epoch_topk_train_acc = \
            topk_train_corrects.double() / num_training_data
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

    def gen_training_epoch_log(self, log_info):
        log = ', '.join([f'{key}={log_info[key]}' for key in log_info])
        return log

    def write_training_log(self, log):
        path = self.data_path.get_path('train_log')
        with open(path, 'a') as f:
            f.write(log + '\n')

    """
    Log for testing the model
    """
    def write_test_first_log(self):
        first_log = self.param_str_for_log()
        self.write_test_log(first_log)

    def write_test_by_class_first_log(self):
        first_log = self.param_str_for_log()
        self.write_test_by_class_log(first_log)

    def param_str_for_log(self):
        log_param_sets = {
            'model_nickname': self.args.model_nickname,
            'model_path': self.data_path.get_path('model_path'),
            'k': self.args.topk
        }
        log = '\n'.join(
            [f'{p}={log_param_sets[p]}' for p in log_param_sets]
        )
        return log

    def write_test_log(self, log):
        path = self.data_path.get_path('test_log')
        with open(path, 'a') as f:
            f.write(log + '\n')

    def write_test_by_class_log(self, log):
        path = self.data_path.get_path('test_by_class_log')
        with open(path, 'a') as f:
            f.write(log + '\n')
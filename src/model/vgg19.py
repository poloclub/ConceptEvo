from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from model.model import Model


class Vgg19(Model):
    """
    Defines Vgg19 model
    """
    def __init__(self, args, data_path, pretrained=False, from_to=None):
        super(Vgg19, self).__init__(args, data_path, pretrained, from_to)

    """
    Initialize model
    """
    def create_empty_model(self):
        if self.pretrained:
            print("Load a pretrained VGG19")
            self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        else:
            self.model = models.vgg19()

    def get_layer_info(self):
        for module_name, module in self.model.named_children():
            if module_name in ['features', 'classifier']:
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

    def get_input_size(self):
        return 224

    def init_criterion(self):
        if self.need_loading_a_saved_model and ('loss' in self.ckpt):
            self.criterion = self.ckpt['loss']
        else:
            self.criterion = nn.CrossEntropyLoss()

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
    Get the output
    """
    def get_output(self, imgs):
        return self.model(imgs)

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
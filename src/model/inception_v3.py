from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from model.model import Model

class InceptionV3(Model):
    """
    Defines InceptionV3 model
    """
    def __init__(self, args, data_path, pretrained=False, from_to=None):
        super(InceptionV3, self).__init__(args, data_path, pretrained, from_to)

    """
    Initialize the model
    """
    def create_empty_model(self):
        if self.pretrained:
            # self.model = models.inception_v3(weights='DEFAULT')
            self.model = torch.hub.load(
                'pytorch/vision:v0.10.0', 
                'inception_v3', 
                pretrained=True
            )
            print('Load a pretrained InceptionV3')
        else:
            self.model = models.inception_v3()

    def get_layer_info(self):
        for module_name, module in self.model.named_children():
            if module_name == 'AuxLogits':
                continue
            elif module_name == 'dropout':
                self.update_layer_info(module_name, module)
                self.update_layer_info('flatten', nn.Flatten(start_dim=1, end_dim=-1))
            else:
                self.update_layer_info(module_name, module)

    def update_layer_info(self, layer_name, layer):
        self.layers.append({
            'name': layer_name,
            'layer': layer
        })
        if layer_name not in ['avgpool', 'dropout', 'flatten', 'fc']:
            self.layers_for_ex_patch.append(layer_name)

    def init_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
        if not self.pretrained and self.need_loading_a_saved_model:
            if 'loss' in self.ckpt:
                self.criterion = self.ckpt['loss']

    def get_input_size(self):
        return 299

    def init_optimizer(self):
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.args.lr, 
            momentum=self.args.momentum
        )
        # self.optimizer = optim.RMSprop(
        #     self.model.parameters(), 
        #     lr=self.args.lr, 
        #     eps=self.args.learning_eps,
        #     momentum=self.args.momentum,
        #     weight_decay=self.args.weight_decay
        # )
        if not self.pretrained and self.need_loading_a_saved_model:
            if 'optimizer_state_dict' in self.ckpt:
                self.optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])
                for param_group in self.optimizer.state_dict()['param_groups']:
                    param_group['lr'] = self.args.lr
                    param_group['eps'] = self.args.learning_eps
                    param_group['weight_decay'] = self.args.weight_decay
                    param_group['momentum'] = self.args.momentum

    """ 
    Get the output
    """
    def get_output(self, imgs):
        if self.model.training:
            return self.model(imgs).logits
        else:
            return self.model(imgs)    

    def write_training_first_log(self):
        log_param_sets = {
            'model_nickname': self.args.model_nickname,
            'batch_size': self.args.batch_size,
            'lr': self.args.lr,
            'momentum': self.args.momentum,
            # 'weight_decay': self.args.weight_decay,
            # 'eps': self.args.learning_eps,
            'k': self.args.topk,
            'model_path': self.data_path.get_path('model_path')
        }
        first_log = ', '.join(
            [f'{p}={log_param_sets[p]}' for p in log_param_sets]
        )
        self.write_training_log(first_log, append=False)

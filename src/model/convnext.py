from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from model.model import Model


class ConvNeXt(Model):
    """
    Defines ConvNeXt model
    """
    def __init__(self, args, data_path, pretrained=False, from_to=None):
        super(ConvNeXt, self).__init__(args, data_path, pretrained, from_to)

    """ 
    Initialize model
    """
    def create_empty_model(self):
        if self.pretrained:
            print("Load a pretrained ConvNeXt")
            self.model = models.convnext_tiny(weights="DEFAULT")
        else:
            self.model = models.convnext_tiny()

    def get_layer_info(self):
        # Features
        layer_idx = 0
        feature_layers = list(self.model.features.children())
        for i, layer in enumerate(feature_layers):
            if i == 0:
                # Stem
                self.update_layer_info("features_stem", layer, layer_idx)
            else:
                # Sequence of inverted blocks (CNBlocks)
                blks = list(layer.children())
                for j, blk in enumerate(blks):
                    blk_name = f"features_{i}_blk_{j}"
                    self.update_layer_info(blk_name, blk, layer_idx)
                    layer_idx += 1

        # Avgpool
        self.update_layer_info("avgpool", self.model.avgpool, layer_idx)
        layer_idx += 1

        # Classifier
        self.update_layer_info("classifier", self.model.classifier, layer_idx)

    def update_layer_info(self, layer_name, layer, layer_idx):
        self.layers.append({"name": layer_name, "layer": layer})
        if "features" in layer_name:
            self.layers_for_ex_patch.append(layer_name)

    def init_criterion(self):
        if self.need_loading_a_saved_model and ("loss" in self.ckpt):
            self.criterion = self.ckpt["loss"]
        else:
            self.criterion = nn.CrossEntropyLoss()

    def get_input_size(self):
        return 256

    def init_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

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
            'weight_decay': self.args.weight_decay,
            'k': self.args.topk,
            'model_path': self.args.model_path
        }
        first_log = ', '.join(
            [f'{p}={log_param_sets[p]}' for p in log_param_sets]
        )
        self.write_training_log(first_log)

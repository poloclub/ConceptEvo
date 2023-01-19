import cv2
import numpy as np
import json
from tqdm import tqdm
from time import time

from model.vgg16 import *
from model.inception_v3 import *

import torch
from torchvision import datasets, transforms

class EvalImportantEvo:
    """Evaluate how well our method finds important evolutions"""

    """
    Constructor
    """
    def __init__(self, args, data_path, from_model, to_model):
        self.args = args
        self.data_path = data_path

        self.from_model = from_model
        self.to_model = to_model
        self.from_model.model.eval()
        self.to_model.model.eval()

        self.input_size = self.from_model.input_size
        self.num_classes = self.from_model.num_classes
        self.data_loader = None

        self.imp_evo = {}
        self.pred = {}

        self.start_idx = -1
        self.end_idx = -1

    """
    A wrapper function called by main.py
    """
    def eval_important_evolution(self):
        self.write_first_log()
        self.init_setting()
        self.eval_imp_evo()
        self.save_results()
    
    """
    Initial setting
    """
    def init_setting(self):
        self.init_pred_dict()
        self.init_device()
        self.init_data_loader()

    def init_device(self):
        self.device = torch.device(
            'cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() 
            else 'cpu'
        )
        print(f'Run on {self.device}')
    
    def init_pred_dict(self):
        for train_test in ['train', 'test']:
            self.pred[train_test] = {
                'from': {
                    'top1': {'correct': 0, 'incorrect': 0},
                    'topk': {'correct': 0, 'incorrect': 0}
                },
                'to': {
                    'top1': {'correct': 0, 'incorrect': 0},
                    'topk': {'correct': 0, 'incorrect': 0}
                },
                'important': {},
                'random': {}
            }
            for layer in self.to_model.layers:
                layer_name = layer['name']
                self.pred[train_test]['important'][layer_name] = {
                    'top1': {
                        'correct': [0] * self.args.num_bins, 
                        'incorrect': [0] * self.args.num_bins
                    },
                    'topk': {
                        'correct': [0] * self.args.num_bins, 
                        'incorrect': [0] * self.args.num_bins
                    }
                }

                self.pred[train_test]['random'][layer_name] = {
                    'top1': {'correct': 0, 'incorrect': 0},
                    'topk': {'correct': 0, 'incorrect': 0}
                }

    def init_data_loader(self):
        data_transform = transforms.Compose([
            transforms.Resize(
                (self.from_model.input_size, self.from_model.input_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(*self.from_model.input_normalization)
        ])
        self.training_dataset = datasets.ImageFolder(
            self.data_path.get_path('train_data'),
            data_transform
        )
        self.test_dataset = datasets.ImageFolder(
            self.data_path.get_path('test_data'),
            data_transform
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )

        self.find_idx_training_dataset_for_class()
        self.class_dataset = torch.utils.data.Subset(
            self.training_dataset, 
            range(self.start_idx, self.end_idx)
        )
        self.data_loader = torch.utils.data.DataLoader(
            self.class_dataset, 
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )

    def find_idx_training_dataset_for_class(self):
        if len(self.args.label_img_idx_path) > 0:
            d = self.load_json(self.args.label_img_idx_path)
            img_idx_data = {}
            for label in d:
                img_idx_data[int(label)] = d[label]
            self.start_idx, self.end_idx = img_idx_data[self.args.label]
        else:
            total = len(self.training_dataset)
            unit = int(total / self.num_classes)
            start = max(0, unit * (self.args.label - 6))
            end = min(total, unit * (self.args.label + 5))

            start_idx, end_idx = -1, -1
            with tqdm(total=(end - start)) as pbar:
                for i in range(start, end):
                    img, label = self.training_dataset[i]
                    if (self.args.label == label) and (start_idx == -1):
                        start_idx = i
                        end_idx = -2
                    if (self.args.label < label) and (end_idx == -2):
                        end_idx = i
                        break
                    pbar.update(1)

            if (start_idx != -1) and (end_idx < 0):
                end_idx = end

            self.start_idx = start_idx
            self.end_idx = end_idx

        print(f'Label={self.args.label}, [{self.start_idx}, {self.end_idx}]')

    """
    Evaluate important evolution
    """
    def eval_imp_evo(self):
        tic, total = time(), len(self.data_loader.dataset)
        self.load_imp_evo()

        print('test on training data')
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                self.eval_imp_evo_one_batch(imgs, labels, if_test=False)
                pbar.update(self.args.batch_size)

        print('test on test data')
        total = len(self.test_loader.dataset)
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.test_loader):
                idxs_of_given_label = labels == self.args.label
                if True in idxs_of_given_label:
                    imgs_of_label = imgs[idxs_of_given_label]
                    labels_of_label = labels[idxs_of_given_label]
                    self.eval_imp_evo_one_batch(
                        imgs_of_label, labels_of_label, if_test=True
                    )
                pbar.update(self.args.batch_size)

        toc = time()
        log = f'Evaluate important evo: {toc - tic:.2f} sec'
        self.write_log(log)

    def load_imp_evo(self):
        path = self.data_path.get_path('find_important_evo-score')
        self.imp_evo = self.load_json(path)

    def eval_imp_evo_one_batch(self, imgs, labels, if_test=False):
        # Forward
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        from_f_maps = self.from_model.forward(imgs)
        to_f_maps = self.to_model.forward(imgs)
        f_maps = {'from': from_f_maps, 'to': to_f_maps}

        # Measure the prediction accuracy before reverting evolutions
        self.eval_model_before_reversion(f_maps, labels, if_test=if_test)
        
        # Measure the prediction accuracy after reverting evolutions
        self.eval_model_reverting_important_evo(f_maps, labels, if_test=if_test)
        self.eval_model_reverting_random_evo(f_maps, labels, if_test=if_test)
    
    def eval_model_before_reversion(self, f_maps, labels, if_test=False):
        train_test = 'test' if if_test else 'train'
        for key in ['from', 'to']:
            vals = self.eval_prediction(f_maps[key][-1], labels)
            top1_correct, top1_incorr, topk_correct, topk_incorr = vals
            self.pred[train_test][key]['top1']['correct'] += top1_correct
            self.pred[train_test][key]['top1']['incorrect'] += top1_incorr
            self.pred[train_test][key]['topk']['correct'] += topk_correct
            self.pred[train_test][key]['topk']['incorrect'] += topk_incorr
        
    def eval_model_reverting_important_evo(self, f_maps, labels, if_test=False):
        for layer_idx, layer in enumerate(self.to_model.layers):
            # Ignore the final layer (classifier layer)
            if layer_idx == len(self.to_model.layers) - 1:
                break

            layer_name = layer['name']
            for bin_idx in range(self.args.num_bins):
                # Get neurons in the current bin
                binned_neurons = self.get_neurons_in_a_bin(layer_name, bin_idx)

                # Measure accuracy after reverting evolutions
                final_layer_f_map = self.forward_after_reversion(
                    f_maps, layer_idx, binned_neurons, labels
                )
                pred_vals = self.eval_prediction(final_layer_f_map, labels)
            
                # Record the prediction accuracy
                self.record_eval_by_bin(
                    layer_name, pred_vals, bin_idx, if_test=if_test
                )

    def eval_model_reverting_random_evo(self, f_maps, labels, if_test=False):
        for layer_idx, layer in enumerate(self.to_model.layers):
            # Ignore the final layer (classifier layer)
            if layer_idx == len(self.to_model.layers) - 1:
                break

            # Sample neurons
            layer_name = layer['name']
            sampled_neurons = self.sample_neurons(layer_name)

            # Measure accuracy after reverting evolutions
            final_layer_f_map = self.forward_after_reversion(
                f_maps, layer_idx, sampled_neurons, labels
            )
            pred_vals = self.eval_prediction(final_layer_f_map, labels)
        
            # Record the prediction accuracy
            self.record_eval_random(layer_name, pred_vals, if_test=if_test)

    def eval_prediction(self, outputs, labels):
        # Prediction
        _, topk_preds = outputs.topk(k=self.args.topk, dim=1)

        # Top-1 prediction: (B,)
        labels = labels.to(self.device)
        num_inputs = outputs.shape[0]
        top1_preds = topk_preds[:, 0]
        top1_correct = torch.sum(top1_preds == labels.data).item()
        top1_incorr = num_inputs - top1_correct
        
        # Top-k prediction: (k, B)
        topk_preds = topk_preds.t()
        topk_correct = 0
        for k in range(self.args.topk):
            num_correct = torch.sum(topk_preds[k] == labels.data).item()
            topk_correct += num_correct
        topk_incorr = num_inputs - topk_correct

        return top1_correct, top1_incorr, topk_correct, topk_incorr

    def get_neurons_in_a_bin(self, layer_name, bin_idx):
        num_neurons = self.to_model.num_neurons[layer_name]
        num_neurons_in_a_bin = int(num_neurons / self.args.num_bins)
        start_idx = bin_idx * num_neurons_in_a_bin
        end_idx = min(num_neurons, (bin_idx + 1) * num_neurons_in_a_bin)
        binned_neurons = self.imp_evo[layer_name][start_idx: end_idx]
        binned_neurons = [neuron['neuron'] for neuron in binned_neurons]
        return binned_neurons

    def sample_neurons(self, layer_name):
        num_neurons = self.to_model.num_neurons[layer_name]
        num_samples = int(num_neurons / self.args.num_bins)
        idxs = np.random.choice(num_neurons, num_samples, replace=False)
        sampled_neurons = [self.imp_evo[layer_name][r]['neuron'] for r in idxs]
        return sampled_neurons

    def forward_after_reversion(self, f_maps, layer_idx, neurons_to_revert, labels):
        # Revert binned neurons' evolution
        from_f_map = f_maps['from'][layer_idx].clone().detach()
        to_f_map = f_maps['to'][layer_idx].clone().detach()
        for neuron_id in neurons_to_revert:
            neuron_i = int(neuron_id.split('-')[-1])
            to_f_map[:, neuron_i, :, :] = from_f_map[:, neuron_i, :, :]

        # Forward pass
        to_f_map = self.to_model.forward_until_the_end(layer_idx + 1, to_f_map)
        
        return to_f_map

    def record_eval_by_bin(self, layer_name, vals, bin_idx, if_test):
        train_test = 'test' if if_test else 'train'
        key = 'important'
        top1_correct, top1_incorr, topk_correct, topk_incorr = vals
        self.pred[train_test][key][layer_name]['top1']['correct'][bin_idx] += top1_correct
        self.pred[train_test][key][layer_name]['top1']['incorrect'][bin_idx] += top1_incorr
        self.pred[train_test][key][layer_name]['topk']['correct'][bin_idx] += topk_correct
        self.pred[train_test][key][layer_name]['topk']['incorrect'][bin_idx] += topk_incorr

    def record_eval_random(self, layer_name, vals, if_test):
        train_test = 'test' if if_test else 'train'
        key = 'random'
        top1_correct, top1_incorr, topk_correct, topk_incorr = vals
        self.pred[train_test][key][layer_name]['top1']['correct'] += top1_correct
        self.pred[train_test][key][layer_name]['top1']['incorrect'] += top1_incorr
        self.pred[train_test][key][layer_name]['topk']['correct'] += topk_correct
        self.pred[train_test][key][layer_name]['topk']['incorrect'] += topk_incorr
    
    """
    Utils
    """
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def save_results(self):
        path = self.data_path.get_path('eval_important_evo')
        self.save_json(self.pred, path)    
    
    def write_first_log(self):
        hyperpara_setting = self.data_path.gen_act_setting_str(
            'eval_important_evo', '\n'
        )
        
        log = 'Evaluate important evolution\n\n'
        log += 'from_model_nickname: {}\n'.format(self.args.from_model_nickname)
        log += 'from_model_path: {}\n'.format(self.args.from_model_path)
        log += 'to_model_nickname: {}\n'.format(self.args.to_model_nickname)
        log += 'to_model_path: {}\n'.format(self.args.to_model_path)
        log += 'label: {}\n'.format(self.args.label)
        log += hyperpara_setting + '\n\n'
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        path = self.data_path.get_path('eval_important_evo-log')
        with open(path, log_opt) as f:
            f.write(log + '\n')

    

    # def forward_reverting(self, f_maps, layer_idx, sampled_neurons, labels, num_skips=0, layer_name=None):
    #     # Ignore if 'AuxLogits' layer in InceptionV3 is given 
    #     if self.args.model_name == 'inception_v3':
    #         if 'Aux' in layer_name:
    #             return f_maps, False

    #     # Ignore if 'AdaptiveAvgPool2d' and 'Linear' layer in Vgg16 is given
    #     if self.args.model_name == 'vgg16':
    #         if 'AdaptiveAvgPool2d' in layer_name:
    #             return f_maps, False
    #         if 'Linear' in layer_name:
    #             return f_maps, False
    #         if len(f_maps['from']) <= layer_idx - num_skips:
    #             return f_maps, False
    #         tensor_shape = f_maps['from'][layer_idx - num_skips].shape
    #         if len(tensor_shape) < 4:
    #             return f_maps, False

    #     # Revert sampled neurons' activation map
    #     from_f_map = f_maps['from'][layer_idx - num_skips]
    #     to_f_map = f_maps['to'][layer_idx - num_skips]
    #     for neuron_info in sampled_neurons:
    #         neuron_id = neuron_info['neuron']
    #         neuron_i = int(neuron_id.split('-')[-1])
    #         layer = neuron_id.split('-')[0]
    #         to_f_map[:, neuron_i, :, :] = from_f_map[:, neuron_i, :, :]
    #     f_maps['to'][layer_idx - num_skips] =  to_f_map

    #     # Forward after reverting_by_layer
    #     if self.args.model_name == 'inception_v3':
    #         to_f_map = self.forward_inception_v3_at_layer_one_step(layer_idx, to_f_map)
    #     elif self.args.model_name == 'vgg16':
    #         to_f_map = self.forward_vgg16_at_layer_one_step(layer_idx, to_f_map)
    #     else:
    #         log = f'Error: unkonwn model {self.args.model_name}'
    #         raise ValueError(log)
    #     f_maps['to'][layer_idx - num_skips + 1] = to_f_map
        
    #     return f_maps, True


    # def forward_inception_v3_at_layer(self, layer_idx, to_f_map):
    #     to_model_children = list(self.to_model.model.children())
    #     num_layers = len(to_model_children)
    #     for next_layer_idx in range(layer_idx + 1, num_layers):
    #         next_layer = to_model_children[next_layer_idx]
    #         next_layer_name = type(next_layer).__name__
    #         if 'Aux' in next_layer_name:
    #             continue
    #         if next_layer_idx == num_layers - 1:
    #             to_f_map = torch.flatten(to_f_map, 1)
    #         to_f_map = next_layer(to_f_map)
    #     return to_f_map


    # def forward_inception_v3_at_layer_one_step(self, layer_idx, to_f_map):
    #     to_model_children = list(self.to_model.model.children())
    #     num_layers = len(to_model_children)
    #     for next_layer_idx in range(layer_idx + 1, num_layers):
    #         next_layer = to_model_children[next_layer_idx]
    #         next_layer_name = type(next_layer).__name__
    #         if 'Aux' in next_layer_name:
    #             continue
    #         if next_layer_idx == num_layers - 1:
    #             to_f_map = torch.flatten(to_f_map, 1)
    #         to_f_map = next_layer(to_f_map)
    #         break
    #     return to_f_map


    # def forward_vgg16_at_layer(self, layer_idx, to_f_map):
    #     to_model_children = list(self.to_model.model.children())
    #     num_layers = len(to_model_children)
    #     layer_i = 0
    #     for child in to_model_children:
    #         if type(child) == nn.Sequential:
    #             for layer in child.children():
    #                 if layer_i > layer_idx:
    #                     to_f_map = layer(to_f_map)
    #                 layer_i += 1
    #         else:
    #             if layer_i > layer_idx:
    #                 to_f_map = child(to_f_map)
    #             if type(child) == nn.AdaptiveAvgPool2d:
    #                 to_f_map = torch.flatten(to_f_map, 1)
    #             layer_i += 1
    #     return to_f_map

    
    # def forward_vgg16_at_layer_one_step(self, layer_idx, to_f_map):
    #     to_model_children = list(self.to_model.model.children())
    #     num_layers = len(to_model_children)
    #     layer_i = 0
    #     one_step_end = False

    #     for child in to_model_children:
    #         if one_step_end:
    #             break
    #         if type(child) == nn.Sequential:
    #             for layer in child.children():
    #                 if layer_i > layer_idx:
    #                     to_f_map = layer(to_f_map)
    #                     one_step_end = True
    #                 layer_i += 1
    #                 if one_step_end:
    #                     break
    #             if one_step_end:
    #                     break
    #         else: 
    #             if layer_i > layer_idx:
    #                 to_f_map = child(to_f_map)
    #                 one_step_end = True
    #             if type(child) == nn.AdaptiveAvgPool2d:
    #                 to_f_map = torch.flatten(to_f_map, 1)
    #                 one_step_end = True
    #             layer_i += 1
    #             if one_step_end:
    #                 break

    #     return to_f_map

    # def record_eval(self, layer_name, vals, key, if_test):
    #     train_test = 'test' if if_test else 'train'
    #     top1_correct, top1_incorr, topk_correct, topk_incorr = vals
    #     self.pred[train_test][key][layer_name]['top1']['correct'] += top1_correct
    #     self.pred[train_test][key][layer_name]['top1']['incorrect'] += top1_incorr
    #     self.pred[train_test][key][layer_name]['topk']['correct'] += topk_correct
    #     self.pred[train_test][key][layer_name]['topk']['incorrect'] += topk_incorr
 

    
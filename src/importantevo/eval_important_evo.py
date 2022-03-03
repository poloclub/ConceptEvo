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
        self.init_global_vars()
        self.init_device()
        self.init_data_loader()


    def init_device(self):
        self.device = torch.device(
            'cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() 
            else 'cpu'
        )
        print('Run on {}'.format(self.device))

    
    def init_global_vars(self):
        self.pred = {
            'from': {
                'top1': {'correct': 0, 'incorrect': 0},
                'topk': {'correct': 0, 'incorrect': 0}
            },
            'to': {
                'top1': {'correct': 0, 'incorrect': 0},
                'topk': {'correct': 0, 'incorrect': 0}
            },
            'important': {},
            'least-important': {},
            'random': {}
        }


    def init_data_loader(self):
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
        total = len(self.training_dataset)
        unit = int(total / self.num_classes)
        start = max(0, unit * (self.args.label - 1))
        end = min(total, unit * (self.args.label + 2))

        start_idx, end_idx = -1, -1
        with tqdm(total=(end - start)) as pbar:
            for i in range(start, end):
                img, label = self.training_dataset[i]
                if (self.args.label == label) and (start_idx == -1):
                    start_idx = i
                elif (self.args.label == label) and (end_idx == -1):
                    end_idx = -2
                if (self.args.label < label) and (end_idx == -2):
                    end_idx = i
                    break
                pbar.update(1)
        self.start_idx = start_idx
        self.end_idx = end_idx
        print(self.start_idx, self.end_idx)


    """
    Evaluate important evolution
    """
    def eval_imp_evo(self):
        tic, total = time(), len(self.data_loader.dataset)
        self.load_imp_evo()
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                
                # Evaluate important evolution for given label
                # if self.args.label not in labels:
                    # continue
                # nec_idxs = labels == self.args.label
                # nec_labels = labels[nec_idxs]
                # nec_imgs = imgs[nec_idxs]

                # Evaluate important evolution for one batch
                # self.eval_imp_evo_one_batch(nec_imgs, nec_labels)
                self.eval_imp_evo_one_batch(imgs, labels)
                pbar.update(self.args.batch_size)

        toc = time()
        log = 'Evaluate important evo: {:.2f} sec'.format(toc - tic)
        self.write_log(log)

    
    def load_imp_evo(self):
        path = self.data_path.get_path('find_important_evo-score')
        self.imp_evo = self.load_json(path)


    def eval_imp_evo_one_batch(self, imgs, labels):
        # Forward
        from_f_maps, layer_info = self.from_model.forward(imgs)
        to_f_maps, layer_info = self.to_model.forward(imgs)
        f_maps = {'from': from_f_maps, 'to': to_f_maps}

        # Measure prediction accuracy before prevention
        self.eval_prediction_before_prevention_one_batch(f_maps)
        
        # Evaluate evolutions
        self.eval_evol_with_baseline_one_epoch(f_maps, layer_info)

    
    def eval_prediction_before_prevention_one_batch(self, f_maps):
        for key in ['from', 'to']:
            top1_correct, top1_incorr, topk_correct, topk_incorr = \
                self.eval_prediction(f_maps[key][-1])
            self.pred[key]['top1']['correct'] += top1_correct
            self.pred[key]['top1']['incorrect'] += top1_incorr
            self.pred[key]['topk']['correct'] += topk_correct
            self.pred[key]['topk']['incorrect'] += topk_incorr


    def eval_prediction(self, outputs):

        num_inputs = outputs.shape[0]
        _, topk_preds = outputs.topk(
            k=self.args.topk, dim=1
        )

        # Top-1 prediction (B,)
        top1_preds = topk_preds[:, 0]
        top1_correct = torch.sum(top1_preds == self.args.label).item()
        top1_incorr = num_inputs - top1_correct

        # Top-k prediction (k, B)
        topk_preds = topk_preds.t()
        topk_correct = 0
        for k in range(self.args.topk):
            num_correct = torch.sum(topk_preds[k] == self.args.label).item()
            topk_correct += num_correct
        topk_incorr = num_inputs - topk_correct

        return top1_correct, top1_incorr, topk_correct, topk_incorr


    def eval_evol_with_baseline_one_epoch(self, f_maps, layer_info):
        to_model_children = list(self.to_model.model.children())
        num_layers = len(to_model_children)
        
        for method_type in ['important', 'least-important', 'random']:
            num_layer_skip = 0
            # Ignore the last linear layer
            for layer_idx in range(num_layers -1):
                # Get ready and sample neurons
                s_neurons = self.get_ready_eval(
                    method_type, layer_info, layer_idx - num_layer_skip
                )

                # Evaluate predictions after prevention
                option = self.args.eval_important_evo
                if option == 'perturbation':
                    vals = self.eval_perturb(
                        f_maps, layer_idx, s_neurons, num_layer_skip
                    )
                elif option == 'freezing':
                    vals = self.eval_freezing(
                        f_maps, layer_idx, s_neurons, num_layer_skip
                    )
                else:
                    log = f'Error: unkonwn option {option}'
                    raise ValueError(log)

                # Record the prediction performance
                if vals is None:
                    num_layer_skip += 1
                else:
                    layer_name = layer_info[layer_idx - num_layer_skip]['name']
                    self.record_eval(layer_name, vals, method_type)

    
    def get_ready_eval(self, key, layer_info, layer_idx):
        layer_name = layer_info[layer_idx]['name']
        num_neurons = layer_info[layer_idx]['num_neurons']
        num_sampled_neurons = int(num_neurons * self.args.eval_sample_ratio)
        if key == 'important':
            sampled_neurons = self.imp_evo[layer_name][:num_sampled_neurons]
        elif key == 'least-important':
            sampled_neurons = self.imp_evo[layer_name][-num_sampled_neurons:]
        elif key == 'random':
            rand_idxs = np.random.choice(
                num_neurons, num_sampled_neurons, replace=False
            )
            sampled_neurons = [self.imp_evo[layer_name][r] for r in rand_idxs]
        if layer_name not in self.pred[key]:
            self.pred[key][layer_name] = {
                'top1': {'correct': 0, 'incorrect': 0},
                'topk': {'correct': 0, 'incorrect': 0}
            }
        return sampled_neurons

    
    def eval_perturb(self, f_maps, layer_idx, sampled_neurons, num_skips=0):
        # Ignore if 'AuxLogits' layer in InceptionV3 is given 
        to_model_children = list(self.to_model.model.children())
        curr_layer_name = type(to_model_children[layer_idx]).__name__
        if self.args.model_name == 'inception_v3':
            if 'Aux' in curr_layer_name:
                return None

        # Apply perturbation on sampled neurons
        from_f_map = f_maps['from'][layer_idx - num_skips]
        to_f_map = f_maps['to'][layer_idx - num_skips]
        
        # delta_f_map = to_f_map - from_f_map
        # eps = torch.tensor(self.args.eps).to(self.device)

        noise_shape = to_f_map[:, 0, :, :].shape

        for neuron_info in sampled_neurons:
            neuron_id = neuron_info['neuron']
            neuron_i = int(neuron_id.split('-')[-1])

            # neuron_delta = delta_f_map[:, neuron_i, :, :]
            # norm_delta = torch.norm(neuron_delta)
            # noise = torch.rand(neuron_delta.shape) - 0.5
            # noise = noise.to(self.device)
            # norm_noise = torch.norm(noise)
            # coeff = (norm_delta * eps / norm_noise).to(self.device)
            # noise = coeff * noise

            noise = self.args.eps * 2 * (torch.rand(noise_shape) - 0.5) 
            noise = noise.to(self.device)

            # print(noise)
            to_f_map[:, neuron_i, :, :] = to_f_map[:, neuron_i, :, :] + noise

        # Forward after perturbation
        if self.args.model_name == 'inception_v3':
            to_f_map = self.forward_inception_v3_at_layer(layer_idx, to_f_map)
        elif self.args.model_name == 'vgg16':
            to_f_map = self.forward_vgg16_at_layer(layer_idx, to_f_map)
        else:
            log = f'Error: unkonwn model {self.args.model_name}'
            raise ValueError(log)

        # Measure accuracy after perturbation
        pred_vals = self.eval_prediction(to_f_map)
        
        return pred_vals


    def eval_freezing(self, f_maps, layer_idx, sampled_neurons, num_skips=0):
        # Ignore if 'AuxLogits' layer in InceptionV3 is given 
        to_model_children = list(self.to_model.model.children())
        curr_layer_name = type(to_model_children[layer_idx]).__name__
        if self.args.model_name == 'inception_v3':
            if 'Aux' in curr_layer_name:
                return None

        # Freeze sampled neurons
        from_f_map = f_maps['from'][layer_idx - num_skips]
        to_f_map = f_maps['to'][layer_idx - num_skips]
        delta_f_map = to_f_map - from_f_map
        eps = torch.tensor(self.args.eps).to(self.device)
        for neuron_info in sampled_neurons:
            neuron_id = neuron_info['neuron']
            neuron_i = int(neuron_id.split('-')[-1])
            to_f_map[:, neuron_i, :, :] = from_f_map[:, neuron_i, :, :]

        # Forward after freezing
        if self.args.model_name == 'inception_v3':
            to_f_map = self.forward_inception_v3_at_layer(layer_idx, to_f_map)
        elif self.args.model_name == 'vgg16':
            to_f_map = self.forward_vgg16_at_layer(layer_idx, to_f_map)
        else:
            log = f'Error: unkonwn model {self.args.model_name}'
            raise ValueError(log)

        # Measure accuracy after freezing
        pred_vals = self.eval_prediction(to_f_map)
        
        return pred_vals


    def forward_inception_v3_at_layer(self, layer_idx, to_f_map):
        to_model_children = list(self.to_model.model.children())
        num_layers = len(to_model_children)
        for next_layer_idx in range(layer_idx + 1, num_layers):
            next_layer = to_model_children[next_layer_idx]
            next_layer_name = type(next_layer).__name__
            if 'Aux' in next_layer_name:
                continue
            if next_layer_idx == num_layers - 1:
                to_f_map = torch.flatten(to_f_map, 1)
            to_f_map = next_layer(to_f_map)
        return to_f_map


    def forward_vgg16_at_layer(self, layer_idx, to_f_map):
        to_model_children = list(self.to_model.model.children())
        num_layers = len(to_model_children)
        layer_i = 0
        for child in to_model_children:
            if type(child) == nn.Sequential:
                for layer in child.children():
                    if layer_i > layer_idx:
                        to_f_map = layer(to_f_map)
                    layer_i += 1
            else:
                if layer_i > layer_idx:
                    to_f_map = child(to_f_map)
                if type(child) == nn.AdaptiveAvgPool2d:
                    to_f_map = torch.flatten(to_f_map, 1)
                layer_i += 1
        return to_f_map


    def record_eval(self, layer_name, vals, key):
        if layer_name not in self.pred[key]:
            self.pred[key][layer_name] = {
                'top1': {'correct': 0, 'incorrect': 0},
                'topk': {'correct': 0, 'incorrect': 0}
            }
        top1_correct, top1_incorr, topk_correct, topk_incorr = vals
        self.pred[key][layer_name]['top1']['correct'] += top1_correct
        self.pred[key][layer_name]['top1']['incorrect'] += top1_incorr
        self.pred[key][layer_name]['topk']['correct'] += topk_correct
        self.pred[key][layer_name]['topk']['incorrect'] += topk_incorr


    def save_results(self):
        path = self.data_path.get_path('eval_important_evo')
        self.save_json(self.pred, path)
    

    """
    Handle external files (e.g., output, log, ...)
    """
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data


    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    
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
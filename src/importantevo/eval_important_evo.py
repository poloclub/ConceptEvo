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

        self.input_size = -1        
        self.num_classes = 1000
        self.label_to_synset = {}

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
        self.set_input_size()
        self.init_device()
        self.get_synset_info()
        # self.load_models()
        self.data_loader = self.from_model.training_data_loader
        self.start_idx = self.from_model.start_idx
        self.end_idx = self.from_model.end_idx

    
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
        
        # Set attributes
        self.from_model.need_loading_a_saved_model = True
        self.to_model.need_loading_a_saved_model = True
        self.from_model.args.model_path = self.args.from_model_path
        self.to_model.args.model_path = self.args.to_model_path

        # Initialize model
        self.from_model.init_basic_setting()
        self.to_model.init_basic_setting()

        # Initialize device of self.from_model and self.to_model
        # self.from_model.init_device()
        # self.to_model.init_device()
        self.from_model.device = self.device
        self.to_model.device = self.device

        # Load checkpoints
        self.from_model.ckpt = torch.load(self.args.from_model_path)
        self.to_model.ckpt = torch.load(self.args.to_model_path)

        # Initialize the models
        self.from_model.init_model()
        self.to_model.init_model()

        # Set the training setting
        self.from_model.init_training_setting()
        self.to_model.init_training_setting()


    def get_synset_info(self):
        df = pd.read_csv(self.args.data_label_path, sep='\t')
        for synset, label in zip(df['synset'], df['training_label']):
            self.label_to_synset[int(label) - 1] = synset

    
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


    """
    Evaluate important evolution
    """
    def eval_imp_evo(self):
        tic, total = time(), len(self.data_loader.dataset)
        self.load_imp_evo()
        with tqdm(total=total) as pbar:
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):
                # Check if the current batch includes the label
                img_start_idx = batch_idx * self.args.batch_size
                img_end_idx = (batch_idx + 1) * self.args.batch_size
                if img_end_idx < self.start_idx:
                    pbar.update(self.args.batch_size)
                    continue
                if self.end_idx < img_start_idx:
                    break
                
                # Evaluate important evolution for one batch
                self.eval_imp_evo_one_batch(imgs, labels)
                pbar.update(self.args.batch_size)
        toc = time()
        log = 'Evaluate important evo: {:.2f} sec'.format(toc - tic)
        self.write_log(log)

    
    def load_imp_evo(self):
        path = self.data_path.get_path('find_important_evo-score')
        self.imp_evo = self.load_json(path)


    def eval_imp_evo_one_batch(self, imgs, labels):
        # Send input images and their labels to GPU
        # imgs = imgs.to(self.device)
        # labels = labels.to(self.device)

        # Forward
        # if self.args.model_name == 'vgg16':
        #     f_maps, layer_info = self.forward_vgg16(imgs)
        # elif self.args.model_name == 'inception_v3':
        #     f_maps, layer_info = self.forward_inception_v3(imgs)
        # else:
        #     raise ValueError(f'Error: unkonwn model {self.args.model_name}')

        # Measure prediction accuracy before prevention
        f_maps = {}
        self.eval_prediction_before_prevention_one_batch(f_maps, imgs, labels)
        
        # Evaluate evolutions
        self.eval_evol_with_baseline_one_epoch(f_maps, layer_info)

    
    def forward_vgg16(self, imgs):
        from_model_children = list(self.from_model.model.children())
        to_model_children = list(self.to_model.model.children())
        from_f_map, to_f_map = imgs, imgs
        f_maps, layer_info = {'from': [], 'to': []}, []
        for i, child in enumerate(from_model_children):
            if type(child) == nn.Sequential:
                for j, from_layer in enumerate(child.children()):
                    to_layer = to_model_children[i][j]
                    from_f_map = from_layer(from_f_map)
                    to_f_map = to_layer(to_f_map)
                    f_maps['from'].append(from_f_map)
                    f_maps['to'].append(to_f_map)
                    layer_name = '{}_{}_{}_{}'.format(
                        type(child).__name__, i,
                        type(from_layer).__name__, j
                    )
                    layer_info.append({
                        'name': layer_name,
                        'num_neurons': from_f_map.shape[1],
                    })
            else:
                to_layer = to_model_children[i]
                from_f_map = child(from_f_map)
                to_f_map = to_layer(to_f_map)
                if type(child) == nn.AdaptiveAvgPool2d:
                    from_f_map = torch.flatten(from_f_map, 1)
                    to_f_map = torch.flatten(to_f_map, 1)
                f_maps['from'].append(from_f_map)
                f_maps['to'].append(to_f_map)
                child_name = type(child).__name__
                layer_name = '{}_{}'.format(child_name, i)
                layer_info.append({
                    'name': layer_name,
                    'num_neurons': from_f_map.shape[1],
                })
        
        return f_maps, layer_info

    
    def forward_inception_v3(self, imgs):
        from_model_layers = list(self.from_model.model.children())
        to_model_layers = list(self.to_model.model.children())
        num_layers = len(from_model_layers)
        from_f_map, to_f_map = imgs, imgs
        f_maps, layer_info = {'from': [], 'to': []}, []
        for i in range(num_layers):
            from_layer = from_model_layers[i]
            to_layer = to_model_layers[i]
            child_name = type(from_layer).__name__
            if 'Aux' in child_name:
                continue
            if i == num_layers - 1:
                from_f_map = torch.flatten(from_f_map, 1)
                to_f_map = torch.flatten(to_f_map, 1)
            from_f_map = from_layer(from_f_map)
            to_f_map = to_layer(to_f_map)
            f_maps['from'].append(from_f_map)
            f_maps['to'].append(to_f_map)
            layer_name = '{}_{}'.format(child_name, i)
            layer_info.append({
                'name': layer_name,
                'num_neurons': from_f_map.shape[1],
            })

        return f_maps, layer_info


    def eval_prediction_before_prevention_one_batch(self, f_maps, imgs=None, labels=None):
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        print(labels)
        outputs = self.to_model.model(imgs).logits
        _, topk_preds = outputs.topk(
            k=self.args.topk, dim=1
        )
        print(topk_preds)
        print(topk_preds.shape)
        sdfsss

        for key in ['from', 'to']:
            top1_correct, top1_incorr, topk_correct, topk_incorr = \
                self.eval_prediction(f_maps[key][-1])
            self.pred[key]['top1']['correct'] += top1_correct
            self.pred[key]['top1']['incorrect'] += top1_incorr
            self.pred[key]['topk']['correct'] += topk_correct
            self.pred[key]['topk']['incorrect'] += topk_incorr


    def eval_prediction(self, outputs):
        print(outputs.shape)

        num_inputs = outputs.shape[0]
        _, topk_preds = outputs.topk(
            k=self.args.topk, dim=1
        )

        print(topk_preds)

        # Top-1 prediction (B,)
        top1_preds = topk_preds[:, 0]
        top1_correct = torch.sum(top1_preds == self.args.label).item()
        print(top1_correct)
        top1_incorr = num_inputs - top1_correct
        sdf

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
        delta_f_map = to_f_map - from_f_map
        eps = torch.tensor(self.args.eps).to(self.device)
        for neuron_info in sampled_neurons:
            neuron_id = neuron_info['neuron']
            neuron_i = int(neuron_id.split('-')[-1])
            neuron_delta = delta_f_map[:, neuron_i, :, :]
            norm_delta = torch.norm(neuron_delta)
            noise = torch.rand(neuron_delta.shape) - 0.5
            noise = noise.to(self.device)
            norm_noise = torch.norm(noise)
            coeff = (norm_delta * eps / norm_noise).to(self.device)
            noise = coeff * noise
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


    def eval_freezing(self, f_maps, layer_idx, sampled_neurons):
        # Ignore if 'AuxLogits' layer in InceptionV3 is given 
        to_model_children = list(self.to_model.model.children())
        curr_layer_name = type(to_model_children[layer_idx]).__name__
        if self.args.model_name == 'inception_v3':
            if 'Aux' in curr_layer_name:
                return None

        # Freeze sampled neurons
        from_f_map = f_maps['from'][layer_idx]
        to_f_map = f_maps['to'][layer_idx]
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
                for to_f_map in child.children():
                    if layer_i > layer_idx:
                        to_f_map = to_f_map(to_f_map)
                    layer_i += 1
            else:
                if layer_i > layer_idx:
                    to_f_map = to_f_map(to_f_map)
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

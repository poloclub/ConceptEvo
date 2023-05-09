import os
import json
from time import time

import numpy as np
from tqdm import tqdm


class ImageEmbCoAct:
    """Generate image embeddings"""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path

        self.start_from_pre_computed = self.args.from_iter_img_emb >= 0
        self.num_imgs = model.num_training_imgs
        self.imgs = []

        self.stimulus = {}

        self.layer = self.args.layer
        self.layer_act = {}
        self.top_neurons_by_img = {}
        self.co_imgs_of_neuron = {}
        self.img_pairs = {}

        self.neuron_emb = {}
        self.num_neurons = None

        self.stimuluated_neurons_by = {}
        self.img_emb = None

    """
    A wrapper function called in main.py
    """
    def compute_img_embedding(self):
        self.write_first_log()
        self.load_stimulus()
        self.gen_neurons_activated_by_stimulus()
        self.load_layer_act()
        self.find_top_neurons_by_img()
        self.find_co_activating_imgs_by_neuron()
        # self.find_img_pairs()
        self.load_neuron_emb()
        self.init_img_emb()
        self.compute_img_emb()
        self.save_img_emb()

    """
    Load neuron embedding
    """
    def load_neuron_emb(self):
        log = 'Load neuron embedding'
        print(log)
        tic = time()

        self.neuron_emb = self.load_json(self.data_path.get_path('neuron_emb'))
        for neuron in self.neuron_emb:
            self.neuron_emb[neuron] = np.array(self.neuron_emb[neuron])
        self.num_neurons = len(self.neuron_emb)

        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')

    """
    Get stimulus data
    """
    def load_stimulus(self):
        log = 'Load stimulus'
        print(log)
        tic = time()

        stimulus_path = self.data_path.get_path('stimulus')
        self.stimulus = self.load_json(stimulus_path)
        for layer in self.stimulus:
            for neuron, neuron_imgs in enumerate(self.stimulus[layer]):
                self.stimulus[layer][neuron] = neuron_imgs[:self.args.k]
        
        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')
    
    def gen_neurons_activated_by_stimulus(self):
        for layer in self.stimulus:
            for neuron, neuron_imgs in enumerate(self.stimulus[layer]):
                neuron_id = f'{layer}-{neuron}'
                for img in neuron_imgs:
                    if img not in self.stimuluated_neurons_by:
                        self.stimuluated_neurons_by[img] = []
                    self.stimuluated_neurons_by[img].append(neuron_id)
    
    """
    Get image pairs that activate the same neurons
    """
    def load_layer_act(self):
        log = 'Load layer activation'
        print(log)
        tic = time()

        p = self.data_path.get_path('layer_act')
        self.layer_act = np.loadtxt(p)

        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')

    def find_top_neurons_by_img(self):
        log = f'Find top {self.args.k} activated neurons by each image'
        print(log)
        tic = time()

        total = self.num_imgs
        with tqdm(total=total) as pbar:
            for img_i, img_v in enumerate(self.layer_act):
                neuron_idxs = np.argsort(-img_v)[:self.args.k]
                neuron_ids = [f'{self.layer}-{idx}' for idx in neuron_idxs]
                self.top_neurons_by_img[img_i] = neuron_ids
                pbar.update(1)
        
        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')
    
    def find_co_activating_imgs_by_neuron(self):
        log = 'Find co-activating images by neurons'
        print(log)
        tic = time()

        total = self.num_imgs
        with tqdm(total=total) as pbar:
            for i in self.top_neurons_by_img:
                for neuron_id in self.top_neurons_by_img[i]:
                    if neuron_id not in self.co_imgs_of_neuron:
                        self.co_imgs_of_neuron[neuron_id] = []
                    self.co_imgs_of_neuron[neuron_id].append(i)
                pbar.update(1)

        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')

    def find_img_pairs(self):
        log = 'Find img_pairs'
        print(log)
        tic = time()

        total = len(self.co_imgs_of_neuron) * self.args.num_epochs_co_act
        with tqdm(total=total) as pbar:
            for e in range(self.args.num_epochs_co_act):
                for neuron_id in self.co_imgs_of_neuron:
                    imgs = self.co_imgs_of_neuron[neuron_id]
                    np.random.shuffle(imgs)
                    for i, img_i in enumerate(imgs):
                        if i == len(imgs) - 1:
                            break
                        img_j = imgs[i + 1]
                        key = '-'.join(list(map(str, sorted([img_i, img_j]))))
                        if key not in self.img_pairs:
                            self.img_pairs[key] = 0
                        self.img_pairs[key] += 1
                    pbar.update(1)

        toc = time()
        self.write_log(f'{log}: {toc - tic:.2f} sec')    

    def random_sample_imgs(self):
        if len(self.imgs) == 0:
            self.imgs = list(self.stimuluated_neurons_by.keys())
        num_sample = int(self.args.sample_rate_img_emb * self.num_imgs)
        sampled_imgs = np.random.choice(self.imgs, num_sample, replace=False)
        return sampled_imgs

    """
    Compute image embedding
    """
    def init_img_emb(self):
        if self.start_from_pre_computed:
            file_path = self.data_path.get_path('img_emb_from')
            self.img_emb = np.loadtxt(file_path)
        else:
            self.img_emb \
                = np.random.random((self.num_imgs, self.args.dim)) - 0.5

    def compute_img_emb(self):
        print('Compute image embedding')
        tic = time()
        total = len(self.stimuluated_neurons_by) + len(self.co_imgs_of_neuron)
        total = total * self.args.max_iter_img_emb
        with tqdm(total=total) as pbar:
            for i in range(total):
                # Update image embedding
                self.update_img_embedding(pbar) 
                pbar.update(1)

                # Check convergence
                err = self.compute_rmse()
                if err < self.args.thr_img_emb:
                    break
                if i % 10 == 0:
                    toc = time()
                    iter_num = i
                    if self.start_from_pre_computed:
                        iter_num += self.args.from_iter_img_emb
                    self.write_log(
                        f'iter={iter_num}, rmse={err}, cum_time={toc - tic}sec'
                    )
                    self.save_img_emb(iter_num)

    def update_img_embedding(self, pbar):
        for img in self.stimuluated_neurons_by:
            grad_img = self.compute_gradient(img)
            self.img_emb[img] -= self.args.lr_img_emb * grad_img
            pbar.update(1)

        for neuron_id in self.co_imgs_of_neuron:
            imgs = self.co_imgs_of_neuron[neuron_id]
            np.random.shuffle(imgs)
            print(imgs)
            print(neuron_id)
            asdf
            for i, img_i in enumerate(imgs):
                if i == len(imgs) - 1:
                    break
                img_j = imgs[i + 1]
                v_i = self.img_emb[img_i]
                v_j = self.img_emb[img_j]
                self.img_emb[img_i] -= self.args.lr_img_emb * (v_i - v_j)
                self.img_emb[img_j] -= self.args.lr_img_emb * (v_j - v_i)
            pbar.update(1)

    def compute_gradient(self, img):
        grad = np.zeros(self.args.dim)
        for neuron in self.stimuluated_neurons_by[img]:
            X_n = self.get_stimulus_of_neuron(neuron)
            v_n = self.neuron_emb[neuron]
            v_n_p = self.compute_approx_neuron_vec(X_n)
            grad += (v_n_p - v_n) / len(X_n)
        return grad

    def get_stimulus_of_neuron(self, neuron):
        layer, neuron_idx = neuron.split('-')
        neuron_idx = int(neuron_idx)
        return self.stimulus[layer][neuron_idx]
        
    def compute_approx_neuron_vec(self, X_n):
        vec_sum = np.zeros(self.args.dim)
        for x in X_n:
            vec_sum += self.img_emb[x]
        return vec_sum / len(X_n)

    def compute_vec_approx_err(self, v_n_p, v_n):
        diff = v_n_p - v_n
        err = diff.dot(diff)
        return err
    
    def compute_rmse(self):
        err = 0
        for neuron in self.neuron_emb:
            v_n = self.neuron_emb[neuron]
            X_n = self.get_stimulus_of_neuron(neuron)
            v_n_p = self.compute_approx_neuron_vec(X_n)
            err += self.compute_vec_approx_err(v_n_p, v_n)
        err = np.sqrt(err / self.num_neurons)
        return err

    def save_img_emb(self, epoch=None):
        file_path = self.data_path.get_path('img_emb_co_act')
        if epoch is not None:
            file_path = file_path.replace('img_emb.txt', f'img_emb-{epoch}.txt')
        np.savetxt(file_path, self.img_emb, fmt='%.3f')

    """
    Handle external files (e.g., output, log, ...)
    """
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def write_first_log(self):
        hyperpara_setting = self.data_path.gen_act_setting_str('img_emb', '\n')
        log = 'Image Embedding with Image Co-activation\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'from_iter_img_emb: {}\n'.format(self.args.from_iter_img_emb)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hyperpara_setting + '\n\n'
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('img_emb_co_act-log'), log_opt) as f:
            f.write(log + '\n')
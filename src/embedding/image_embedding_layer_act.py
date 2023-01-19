import json
from time import time

import numpy as np
from tqdm import tqdm


class ImageEmb:
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
        self.neuron_emb = {}
        self.num_neurons = None
        self.norm_coeff = -1

        self.label_img_idx = {}
        self.img_emb_layer_act = {}

        self.stimuluated_neurons_by_img = {}
        self.img_emb = None

    """
    A wrapper function called in main.py
    """
    def compute_img_embedding(self):
        self.load_label_img_idx()
        self.load_neuron_emb()
        self.load_img_emb_layer_act()
        self.init_img_emb()
        self.load_stimulus()
        self.gen_neurons_activated_by_stimulus()
        self.compute_img_emb()
        self.save_img_emb()

    """
    Utils
    """
    def load_label_img_idx(self):
        path = self.data_path.get_path('label_img_idx')
        data = self.load_json(path)
        for key in data:
            self.label_img_idx[int(key)] = data[key]
    
    def load_neuron_emb(self):
        self.neuron_emb = self.load_json(self.data_path.get_path('neuron_emb'))
        for neuron in self.neuron_emb:
            self.neuron_emb[neuron] = np.array(self.neuron_emb[neuron])
            norm = np.linalg.norm(self.neuron_emb[neuron])
            self.norm_coeff = max(self.norm_coeff, norm)
        self.num_neurons = len(self.neuron_emb)
    
    def load_img_emb_layer_act(self):
        path = self.data_path.get_path('img_act_emb')
        self.img_emb_layer_act = np.loadtxt(path)
        max_norm = 0
        for img_vec in self.img_emb_layer_act:
            max_norm = max(max_norm, np.linalg.norm(img_vec))
        self.img_emb_layer_act = self.img_emb_layer_act / max_norm * self.norm_coeff

    def load_stimulus(self):
        stimulus_path = self.data_path.get_path('stimulus')
        self.stimulus = self.load_json(stimulus_path)
        for layer in self.stimulus:
            for neuron, neuron_imgs in enumerate(self.stimulus[layer]):
                self.stimulus[layer][neuron] = neuron_imgs[:self.args.k]

    def gen_neurons_activated_by_stimulus(self):
        for layer in self.stimulus:
            for neuron, neuron_imgs in enumerate(self.stimulus[layer]):
                neuron_id = f'{layer}-{neuron}'
                for img in neuron_imgs:
                    if img not in self.stimuluated_neurons_by_img:
                        self.stimuluated_neurons_by_img[img] = []
                    self.stimuluated_neurons_by_img[img].append(neuron_id)

    def get_stimulus_of_neuron(self, neuron):
        layer, neuron_idx = neuron.split('-')
        neuron_idx = int(neuron_idx)
        return self.stimulus[layer][neuron_idx]

    def random_sample_imgs(self):
        if len(self.imgs) == 0:
            self.imgs = list(self.stimuluated_neurons_by_img.keys())
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
            self.img_emb = self.img_emb * self.norm_coeff

    def compute_img_emb(self):
        self.write_first_log()

        tic, total = time(), self.args.max_iter_img_emb * self.num_imgs
        with tqdm(total=total) as pbar:
            for i in range(total):
                # Update image embedding
                self.img_emb_one_iter()
                for label in self.label_img_idx:
                    self.img_emb_layer_act_one_iter(label, pbar)

                # Check convergence
                err = self.compute_rmse()
                if err < self.args.thr_img_emb:
                    break
                if i % 2 == 0:
                    toc = time()
                    iter_num = i
                    if self.start_from_pre_computed:
                        iter_num += self.args.from_iter_img_emb
                    self.write_log(
                        f'iter={iter_num}, rmse={err}, cum_time={toc - tic}sec'
                    )

    def img_emb_one_iter(self):
        for img in self.stimuluated_neurons_by_img:
            grad_img = self.compute_gradient(img)
            self.img_emb[img] -= self.args.lr_img_emb * grad_img

    def compute_gradient(self, img):
        grad = np.zeros(self.args.dim)
        for neuron in self.stimuluated_neurons_by_img[img]:
            X_n = self.get_stimulus_of_neuron(neuron)
            v_n = self.neuron_emb[neuron]
            v_n_p = self.compute_approx_neuron_vec(X_n)
            grad += (v_n_p - v_n) / len(X_n)
        # grad = grad / self.num_neurons
        return grad

    def sample_imgs(self, start_img_idx, end_img_idx, ratio=0.1):
        n_label_imgs = end_img_idx - start_img_idx
        sampled_imgs = start_img_idx + \
            np.random.choice(n_label_imgs, int(n_label_imgs * ratio))
        return sampled_imgs

    def img_emb_layer_act_one_iter(self, label, pbar):
        start_img_idx, end_img_idx = self.label_img_idx[label]
        sampled_imgs_i = self.sample_imgs(start_img_idx, end_img_idx, 0.1)
        sampled_imgs_j = self.sample_imgs(start_img_idx, end_img_idx, 0.1)
        for img_i in sampled_imgs_i:
            grad = np.zeros(self.args.dim)
            for img_j in sampled_imgs_j:
                if img_i == img_j:
                    continue
                v_i = self.img_emb[img_i]
                v_j = self.img_emb[img_j]
                l_i = self.img_emb_layer_act[img_i]
                l_j = self.img_emb_layer_act[img_j]
                grad += (v_i.dot(v_j) - l_i.dot(l_j)) * v_j
            self.img_emb[img_i] -= self.args.lr_img_emb * grad
            pbar.update(10)

    def compute_rmse(self):
        err = 0
        for neuron in self.neuron_emb:
            v_n = self.neuron_emb[neuron]
            X_n = self.get_stimulus_of_neuron(neuron)
            v_n_p = self.compute_approx_neuron_vec(X_n)
            err += self.compute_vec_approx_err(v_n_p, v_n)
        err = np.sqrt(err / self.num_neurons)
        return err

    def compute_approx_neuron_vec(self, X_n):
        vec_sum = np.zeros(self.args.dim)
        for x in X_n:
            vec_sum += self.img_emb[x]
        return vec_sum / len(X_n)

    def compute_vec_approx_err(self, v_n_p, v_n):
        diff = v_n_p - v_n
        err = diff.dot(diff)
        return err

    def save_img_emb(self):
        file_path = self.data_path.get_path('img_emb')
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
        log = 'Image Embedding\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'from_iter_img_emb: {}\n'.format(self.args.from_iter_img_emb)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hyperpara_setting + '\n\n'
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('img_emb-log'), log_opt) as f:
            f.write(log + '\n')
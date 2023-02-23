import json
from time import time

import numpy as np
from tqdm import tqdm


class ImageEmbWithLayerAct:
    """Generate image embeddings"""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path

        self.num_imgs = model.num_training_imgs
        self.imgs = []
        self.layer = self.args.layer
        self.num_neurons = None

        self.layer_act = {}
        self.stimulus = {}
        self.neuron_emb = {}
        self.S = {}

        self.stimuluated_neurons_by = {}
        self.img_emb = None

    """
    A wrapper function called in main.py
    """
    def compute_img_embedding_with_layer_act(self):
        self.write_first_log()
        self.load_stimulus()
        self.load_neuron_emb()
        self.load_layer_act()
        self.gen_neurons_activated_by_stimulus()
        self.gen_similar_img_dict()
        self.init_img_emb()
        self.compute_img_emb()
        self.save_img_emb()

    """
    Utils
    """
    def load_stimulus(self):
        stimulus_path = self.data_path.get_path('stimulus')
        self.stimulus = self.load_json(stimulus_path)
        for layer in self.stimulus:
            for neuron, neuron_imgs in enumerate(self.stimulus[layer]):
                self.stimulus[layer][neuron] = neuron_imgs[:self.args.k]

    def load_neuron_emb(self):
        task = 'Load neuron_emb'
        print(task)
        tic = time()
        self.neuron_emb = self.load_json(self.data_path.get_path('neuron_emb'))
        for neuron in self.neuron_emb:
            self.neuron_emb[neuron] = np.array(self.neuron_emb[neuron])
        self.num_neurons = len(self.neuron_emb)
        toc = time()
        self.write_log(f'{task}: {toc - tic:.2f} sec')

    def load_layer_act(self):
        task = 'Load layer_act'
        print(task)
        tic = time()
        p = self.data_path.get_path('layer_act')
        self.layer_act = np.loadtxt(p)
        toc = time()
        self.write_log(f'{task}: {toc - tic:.2f} sec')

    def gen_neurons_activated_by_stimulus(self):
        for layer in self.stimulus:
            for neuron, neuron_imgs in enumerate(self.stimulus[layer]):
                neuron_id = f'{layer}-{neuron}'
                for img in neuron_imgs:
                    if img not in self.stimuluated_neurons_by:
                        self.stimuluated_neurons_by[img] = []
                    self.stimuluated_neurons_by[img].append(neuron_id)

    def gen_similar_img_dict(self):
        # Find top neurons by image (based on the layer activation)
        task = f'Find top {self.args.k} activated neurons by image'
        print(task)
        tic = time()
        top_neurons_by_img = {}
        with tqdm(total=self.num_imgs) as pbar:
            for img_i, img_v in enumerate(self.layer_act):
                neuron_idxs = np.argsort(-img_v)[:self.args.k]
                top_neurons_by_img[img_i] = neuron_idxs
                pbar.update(1)
        toc = time()
        self.write_log(f'{task}: {toc - tic:.2f} sec')

        # Find strongly activating images by neurons
        task = 'Find co-activating images by neurons'
        print(task)
        tic = time()
        total = self.num_imgs
        co_activating_imgs_by_neuron = {}
        with tqdm(total=total) as pbar:
            for i in top_neurons_by_img:
                for neuron_idx in top_neurons_by_img[i]:
                    if neuron_idx not in co_activating_imgs_by_neuron:
                        co_activating_imgs_by_neuron[neuron_idx] = []
                    co_activating_imgs_by_neuron[neuron_idx].append(i)
                pbar.update(1)
        toc = time()
        self.write_log(f'{task}: {toc - tic:.2f} sec')

        # Find image neighbors
        task = 'Find image neighbors'
        print(task)
        tic = time()
        total = len(co_activating_imgs_by_neuron)
        with tqdm(total=total) as pbar:
            for neuron_idx in co_activating_imgs_by_neuron:
                imgs = co_activating_imgs_by_neuron[neuron_idx]
                for i, img_i in enumerate(imgs):
                    if i == len(imgs) - 1:
                        break
                    img_j = imgs[i + 1]

                    if img_i not in self.S:
                        self.S[img_i] = []
                    if img_j not in self.S:
                        self.S[img_j] = []
                    
                    self.S[img_i].append(img_j)
                    self.S[img_j].append(img_i)
                pbar.update(1)
        toc = time()
        self.write_log(f'{task}: {toc - tic:.2f} sec')

    def get_stimulus_of_neuron(self, neuron):
        layer, neuron_idx = neuron.split('-')
        neuron_idx = int(neuron_idx)
        return self.stimulus[layer][neuron_idx]

    """
    Compute image embedding
    """
    def init_img_emb(self):
        self.img_emb = np.random.random((self.num_imgs, self.args.dim)) - 0.5
        
    def compute_approx_neuron_vec(self, xs):
        vec_sum = np.zeros(self.args.dim)
        for x in xs:
            vec_sum += self.img_emb[x]
        return vec_sum / len(xs)

    def compute_vec_approx_err(self, v_n_p, v_n):
        diff = v_n_p - v_n
        err = diff.dot(diff)
        return err
    
    def update_img_embedding(self, pbar):
        for img in self.stimuluated_neurons_by:
            # Update image embedding to minimize the gap between 
            # neuron embedding and the approximated neuron embedding
            for neuron in self.stimuluated_neurons_by[img]:
                X_n = self.get_stimulus_of_neuron(neuron)
                v_n = self.neuron_emb[neuron]
                v_n_p = self.compute_approx_neuron_vec(X_n)
                grad = (v_n_p - v_n) / self.args.k
                self.img_emb[img] -= self.args.lr_img_emb * grad
            
            # Update image embedding to make similar images to be closer
            # (for images that activate similar neurons) 
            v_img = self.img_emb[img]
            for j in self.S[img]:
                v_j = self.img_emb[j]
                g = (1 - self.sigmoid(v_img.dot(v_j))) * v_j
                if self.args.num_emb_negs_layer_act > 0:
                    sampled_imgs = self.random_sample_imgs(img)
                    for r in sampled_imgs:
                        if j == r:
                            continue
                        v_r = self.img_emb[r]
                        g -= self.sigmoid(v_img.dot(v_r)) * v_r
                self.img_emb[img] += self.args.lr_img_emb * g

            pbar.update(1)

    # def compute_gradient(self, img):
    #     # Gradient to minimize the gap between neuron embedding
    #     # and the approximated neuron embedding
    #     grad = np.zeros(self.args.dim)
    #     for neuron in self.stimuluated_neurons_by[img]:
    #         X_n = self.get_stimulus_of_neuron(neuron)
    #         v_n = self.neuron_emb[neuron]
    #         v_n_p = self.compute_approx_neuron_vec(X_n)
    #         grad += (v_n_p - v_n) / len(X_n)
        
    #     # Gradient to make similar images to be closer
    #     # (for images that activate similar neurons) 
    #     v_img = self.img_emb[img]
    #     for j in self.S[img]:
    #         v_j = self.img_emb[j]
    #         g = (1 - self.sigmoid(v_img.dot(v_j))) * v_j
    #         sampled_imgs = self.random_sample_imgs(img)
    #         for r in sampled_imgs:
    #             if j == r:
    #                 continue
    #             v_r = self.img_emb[r]
    #             g -= self.sigmoid(v_img.dot(v_r)) * v_r
    #         grad -= g
    #     return grad

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def random_sample_imgs(self, img):
        n = self.args.num_emb_negs_layer_act
        sampled_imgs = np.random.choice(self.num_imgs, n, replace=False)
        return sampled_imgs


    def compute_rmse(self):
        err = 0
        for neuron in self.neuron_emb:
            v_n = self.neuron_emb[neuron]
            X_n = self.get_stimulus_of_neuron(neuron)
            v_n_p = self.compute_approx_neuron_vec(X_n)
            err += self.compute_vec_approx_err(v_n_p, v_n)
        err = np.sqrt(err / self.num_neurons)
        return err

    def compute_img_emb(self):
        tic = time()
        total = self.args.max_iter_img_emb * len(self.stimuluated_neurons_by)
        prev_rmse = 10000
        with tqdm(total=total) as pbar:
            for i in range(self.args.max_iter_img_emb):
                # Update image embedding
                self.update_img_embedding(pbar)

                # Check convergence
                rmse = self.compute_rmse()
                if rmse < self.args.thr_img_emb:
                    self.write_log(
                        f'iter={i}, rmse={rmse}, cum_time={toc - tic}sec'
                    )
                    self.write_log(
                        'Terminate earlier since rmse < thr_img_emb'
                    )
                    break
                if rmse > prev_rmse:
                    self.write_log(
                        f'iter={i}, rmse={rmse}, cum_time={toc - tic}sec'
                    )
                    self.write_log(
                        'Terminate earlier since rmse > prev_remse'
                    )
                    break
                prev_rmse = rmse
                if i % 100 == 0:
                    toc = time()
                    self.write_log(
                        f'iter={i}, rmse={rmse}, cum_time={toc - tic}sec'
                    )

    def save_img_emb(self):
        file_path = self.data_path.get_path('img_emb_with_layer_act')
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
        hyperpara_setting = self.data_path.gen_act_setting_str(
            'img_emb_with_layer_act', '\n'
        )
        log = 'Image embedding with layer activation (altogether)\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += hyperpara_setting + '\n\n'
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('img_emb_with_layer_act-log'), log_opt) as f:
            f.write(log + '\n')
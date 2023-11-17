from time import time

import numpy as np
from tqdm import tqdm

from utils.utils import *


class IndirectImageEmb:
    """Generate image embeddings"""

    """
    Constructor
    """
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path

        self.stimulus = {}
        self.responsive_neurons = {}
        self.co_act_imgs = {}

        self.vocab = {}
        self.img_emb = None
        self.num_total_imgs = 0

    """
    A wrapper function called in main.py
    """
    def compute_indirect_img_embedding(self):
        self.load_stimulus()
        self.compute_vocab()
        self.load_img_emb()
        self.load_responsive_neurons()
        self.compute_co_activating_imgs()
        self.compute_embedding_of_images()
        self.save_embedding_of_images()

    """
    Compute image embedding
    """
    def compute_co_activating_imgs(self):
        # For each neuron, find images that strongly activate it
        tic, co_act_imgs = time(), {}
        for img in self.responsive_neurons:
            for neurons in self.responsive_neurons[img]:
                for neuron in neurons:
                    if neuron not in co_act_imgs:
                        co_act_imgs[neuron] = []
                    co_act_imgs[neuron].append(img)
        
        # Keep only neurons that get strongly activated by multiple images
        for neuron in co_act_imgs:
            if len(co_act_imgs[neuron]) > 1:
                self.co_act_imgs[neuron] = co_act_imgs[neuron]

    def compute_embedding_of_images(self):
        # Write log header
        self.write_first_log()

        # Co-activating images
        co_act_imgs = [
            self.co_act_imgs[neuron]
            for neuron in self.co_act_imgs
            if len(self.co_act_imgs[neuron]) > 1
        ]

        # Learn image embedding
        tic = time()
        if self.args.num_indirect_img_emb_pairs == -1:
            total = np.sum([len(imgs) for imgs in co_act_imgs])
        else:
            total = self.args.num_indirect_img_emb_pairs * len(co_act_imgs)
        total *= self.args.num_indirect_img_emb_epochs
        
        with tqdm(total=total) as pbar:
            for emb_epoch in range(self.args.num_indirect_img_emb_epochs):
                grad_l2 = 0

                for imgs in co_act_imgs:
                    # Shuffle images
                    np.random.shuffle(imgs)

                    # Compute image embedding
                    iter_imgs = imgs[:self.args.num_indirect_img_emb_pairs]
                    for i, img in enumerate(iter_imgs):

                        next_img = imgs[i + 1]
                        v_i = self.img_emb[img]
                        v_j = self.img_emb[next_img]

                        # 1 - sigma(v_i \dot v_j)
                        coeff = 1 - self.sigmoid(v_i.dot(v_j))

                        # Update v_i
                        if img not in self.vocab:
                            g_i = coeff * v_j
                            for neg_i in range(self.args.num_indirect_img_emb_negs):
                                neg_img = self.sample_neg_img()
                                v_r = self.img_emb[neg_img]
                                g_i -= self.sigmoid(v_i.dot(v_r)) * v_r
                            self.img_emb[img] += self.args.lr_indirect_img_emb * g_i
                            grad_l2 += g_i.dot(g_i)

                        # Update v_j
                        if next_img not in self.vocab:
                            g_j = coeff * v_i
                            for neg_i in range(self.args.num_indirect_img_emb_negs):
                                neg_img = self.sample_neg_img()
                                v_r = self.img_emb[neg_img]
                                g_j -= self.sigmoid(v_j.dot(v_r)) * v_r
                            self.img_emb[next_img] += self.args.lr_indirect_img_emb * g_j
                            grad_l2 += g_j.dot(g_j)
                    
                        pbar.update(1)

                if grad_l2 <= self.args.thr_indirect_img_emb:
                    break

        # Write embedding log
        log = 'runnig_time: {}sec\n'.format(time() - tic)
        log += 'grad_l2: {}\n'.format(grad_l2)
        self.write_log(log)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def save_embedding_of_images(self):
        p = self.data_path.get_path('indirect_img_emb')
        np.savetxt(p, self.img_emb, fmt='%.3f')

    def sample_neg_img(self):
        if self.num_total_imgs == 0:
            self.num_total_imgs = len(self.img_emb)

        img = np.random.randint(self.num_total_imgs)

        return img

    """
    Utils
    """
    def load_stimulus(self):
        stimulus_path = self.data_path.get_path('stimulus')
        self.stimulus = load_json(stimulus_path)

    def compute_vocab(self):
        for layer in self.stimulus:
            for neuron_stimuli in self.stimulus[layer]:
                for img in neuron_stimuli:
                    if img not in self.vocab:
                        self.vocab[img] = 0
                    self.vocab[img] += 1

    def load_img_emb(self):
        img_emb_path = self.data_path.get_path('img_emb')
        self.img_emb = np.loadtxt(img_emb_path)

    def load_responsive_neurons(self):
        p = self.data_path.get_path('responsive_neurons')
        responsive_neurons = load_json(p)
        for img in responsive_neurons:
            self.responsive_neurons[int(img)] = responsive_neurons[img]

    """
    Handle external files (e.g., output, log, ...)
    """
    def write_first_log(self):
        log = 'Indirect Image Embedding\n'
        log += f'model_nickname: {self.args.model_nickname}\n'
        log += f'model_path: {self.data_path.get_path("model_path")}\n'
        log += self.data_path.data_path_indirect_image_embedding.para_info
        self.write_log(log, False)
    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('indirect_img_emb_log'), log_opt) as f:
            f.write(log + '\n')
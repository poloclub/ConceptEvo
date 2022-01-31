import json
import numpy as np
from tqdm import tqdm
from time import time

class Emb:
    """Generate neuron embeddings."""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path

        self.stimulus = {}
        self.co_act_neurons = {}
        self.emb = {}
        self.neuron_sample_pool = []

        self.num_total_neurons = 0


    """
    A wrapper function called by main.py
    """
    def compute_neuron_embedding(self):
        self.compute_co_activated_neurons()
        self.compute_embedding_of_neurons()


    """
    Find co-activated neurons
    """
    def compute_co_activated_neurons(self):
        # Load stimulus
        self.load_stimulus()

        # Find neurons that are highly activated by each input
        tic, co_act_neurons = time(), {}
        for layer_name in self.stimulus:
            for i, neuron_stimulus in enumerate(self.stimulus[layer_name]):
                neuron_id = f'{layer_name}-{i}'
                for img in neuron_stimulus:
                    if img not in co_act_neurons:
                        co_act_neurons[img] = []
                    co_act_neurons[img].append(neuron_id)

        # Keep only images that activate multiple neurons
        for img in co_act_neurons:
            if len(co_act_neurons[img]) > 1:
                self.co_act_neurons[img] = co_act_neurons[img]

        # Save co-activated neurons
        self.save_co_activated_neurons()


    def load_stimulus(self):
        stimulus_path = self.data_path.get_path('stimulus')
        stimulus = self.load_json(stimulus_path)
        self.stimulus = stimulus


    def save_co_activated_neurons(self):
        file_path = self.data_path.get_path('co_act')
        self.save_json(self.co_act_neurons, file_path)

    
    def load_co_activated_neurons(self):
        if len(self.co_act_neurons) <= 0:
            file_path = self.data_path.get_path('co_act')
            self.co_act_neurons = self.load_json(file_path)


    """
    Compute neuron embedding
    """
    def compute_embedding_of_neurons(self):
        # Write log header
        self.write_first_log()

        # Initialize neuron embedding
        self.init_neuron_embedding()

        # Load co-activated neurons
        self.load_co_activated_neurons()

        # Learn neuron embedding
        tic, total = time(), self.args.num_emb_epochs * len(self.co_act_neurons)
        with tqdm(total=total) as pbar:
            for emb_epoch in range(self.args.num_emb_epochs):
                imgs = list(self.co_act_neurons.keys())
                np.random.shuffle(imgs)
                for img in imgs:
                    # Shuffle neurons
                    neurons = self.co_act_neurons[img]
                    np.random.shuffle(neurons)

                    # Compute neuron embedding
                    for i, neuron in enumerate(neurons[:-1]):
                        next_neuron = neurons[i + 1]
                        v_neuron = self.emb[neuron]
                        v_next_neuron = self.emb[next_neuron]

                        # 1 - sigma(V_u \dot V_v)
                        coeff = 1 - self.sigmoid(
                            v_neuron.dot(v_next_neuron)
                        )

                        # Update gradients through negative sampling
                        g_u = -coeff * v_next_neuron
                        g_v = -coeff * v_neuron
                        for neg_i in range(self.args.num_emb_negs):
                            neg_neuron = self.sample_neg_neuron()
                            v_neg_neuron = self.emb[neg_neuron]
                            dot_u = v_neg_neuron.dot(v_neuron)
                            dot_v = v_neg_neuron.dot(v_next_neuron)
                            g_u += self.sigmoid(dot_u) * v_neg_neuron
                            g_v += self.sigmoid(dot_v) * v_neg_neuron

                        # Update embedding
                        self.emb[neuron] -= self.args.lr_emb * g_u
                        self.emb[next_neuron] -= self.args.lr_emb * g_v

                    pbar.update(1)

        err = self.compute_err()

        # Save neuron embedding
        self.save_embedding()

        # Write embedding log
        log = 'runnig_time: {}sec\n'.format(time() - tic)
        log += 'err(?): {}\n'.format(err)
        self.write_log(log)


    def compute_err(self):
        err = 0
        imgs = list(self.co_act_neurons.keys())
        
        for img in imgs:
            neurons = self.co_act_neurons[img]
            for i, neuron in enumerate(neurons[:-1]):
                next_neuron = neurons[i + 1]
                v_neuron = self.emb[neuron]
                v_next_neuron = self.emb[next_neuron]
                err += -np.log(self.sigmoid(v_neuron.dot(v_next_neuron)))
        return err


    def get_num_samples(self, n):
        num_samples = int(self.args.t * np.sqrt(n))
        num_samples = np.min([num_samples, n]) 
        return num_samples


    def init_neuron_embedding(self):
        for layer in self.stimulus:
            for i, neuron_stimulus in enumerate(self.stimulus[layer]):
                neuron_id = f'{layer}-{i}'
                self.emb[neuron_id] = np.random.rand(self.args.dim) - 0.5
   

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sample_neg_neuron(self):
        if self.num_total_neurons == 0:
            self.neuron_sample_pool = list(self.emb.keys())
            self.num_total_neurons = len(self.neuron_sample_pool)

        n = np.random.randint(self.num_total_neurons)
        neuron = self.neuron_sample_pool[n]

        return neuron


    def save_embedding(self):
        for neuron in self.emb:
            self.emb[neuron] = self.emb[neuron].tolist()
        self.save_json(self.emb, self.data_path.get_path('neuron_emb'))


    """
    Handle external files (e.g., output, log, ...)
    """
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('neuron_emb-log'), log_opt) as f:
            f.write(log + '\n')


    def write_first_log(self):
        hypara_setting = self.data_path.gen_act_setting_str('neuron_emb', '\n')
        log = 'Neuron Embedding\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hypara_setting + '\n\n'
        self.write_log(log, False)
            

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data


    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)

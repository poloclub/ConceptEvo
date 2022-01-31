import json
import numpy as np
from tqdm import tqdm
from time import time

class ProjNeuronEmb:
    """Generate neuron embeddings that projected on shared space."""

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Constructor
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path

        self.stimulus = {}
        self.img_emb = None
        self.neuron_emb = {}

    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    A wrapper function called by main.py
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def compute_projected_neuron_emb(self):
        self.load_img_emb()
        self.load_stimulus()
        self.init_neuron_embedding()
        self.project_neuron_embedding()
        self.save_projected_neuron_embedding()


    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Utils
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def load_img_emb(self):
        file_path = self.data_path.get_path('img_emb')
        self.img_emb = np.loadtxt(file_path)

    
    def load_stimulus(self):
        stimulus_path = self.data_path.get_path('stimulus')
        self.stimulus = self.load_json(stimulus_path)

    
    def save_projected_neuron_embedding(self):
        for neuron in self.neuron_emb:
            self.neuron_emb[neuron] = [
                round(x, 3) for x in self.neuron_emb[neuron].tolist()
            ]

        self.save_json(
            self.neuron_emb, 
            self.data_path.get_path('proj_neuron_emb')
        )


    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Compute projected neuron embedding
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def init_neuron_embedding(self):
        for layer in self.stimulus:
            num_neurons = len(self.stimulus[layer])
            for i in range(num_neurons):
                neuron = f'{layer}-{i}'
                self.neuron_emb[neuron] = np.random.rand(self.args.dim) - 0.5


    def get_stimulus_of_neuron(self, neuron):
        layer, neuron_idx = neuron.split('-')
        neuron_idx = int(neuron_idx)
        return self.stimulus[layer][neuron_idx][:self.args.k]

    
    def compute_approx_neuron_vec(self, X_n):
        vec_sum = np.zeros(self.args.dim)
        for x in X_n:
            vec_sum += self.img_emb[x]
        return vec_sum / len(X_n)

    
    def project_neuron_embedding(self):
        self.write_first_log()
        tic = time()
        for neuron in self.neuron_emb:
            stimulus = self.get_stimulus_of_neuron(neuron)
            v_neuron = self.compute_approx_neuron_vec(stimulus)
            self.neuron_emb[neuron] = v_neuron
        toc = time()
        self.write_log('running_time: {}sec'.format(toc - tic))


    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Handle external files (e.g., output, log, ...)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data


    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)

    
    def write_first_log(self):
        hyperpara_setting = self.data_path.gen_act_setting_str(
            'proj_neuron_emb', '\n'
        )

        with open(self.data_path.get_path('img_emb-log'), 'r') as f:
            space_model_info = f.readlines()
        
        log = 'Projected Neuron Embedding\n\n'
        log += 'Information of the model for the shared semantic space\n'
        log += space_model_info[1]
        log += space_model_info[2]
        log += '\nInformation of the projected model\n'
        log += 'model_nickname: {}\n'.format(self.args.model_nickname)
        log += 'model_path: {}\n\n'.format(self.args.model_path)
        log += hyperpara_setting + '\n\n'
        self.write_log(log, False)

    
    def write_log(self, log, append=True):
        log_opt = 'a' if append else 'w'
        with open(self.data_path.get_path('img_emb-log'), log_opt) as f:
            f.write(log + '\n')
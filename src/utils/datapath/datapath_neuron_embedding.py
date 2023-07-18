import os
from utils.datapath.datapath_util import DataPathUtil

class DataPathNeuronEmbedding:
    """
    Manage paths for neuron embedding
    """

    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}

        self.para = [
            ['dim', self.args.dim],
            ['lr_emb', self.args.lr_emb],
            ['topk_n', self.args.topk_n],
            ['num_emb_epochs', self.args.num_emb_epochs],
            ['num_emb_negs', self.args.num_emb_negs],
            ['stimulus_sub_dir_name', self.args.stimulus_sub_dir_name],
            ['neuron_embedding_sub_dir_name', self.args.neuron_embedding_sub_dir_name]
        ]

        self.para_info = '\n'.join([
            f'{name}: {val}' 
            for name, val in self.para
        ])

        self.actions_requiring_paths = [
            self.args.neuron_embedding,
            self.args.image_embedding
        ]

        self.util = DataPathUtil(args.output_dir)
        self.gen_data_paths()

    def gen_data_paths(self):
        # Check if data paths for example patches are necessary
        if True not in self.actions_requiring_paths:
            return

        # Check if model_nickname is given
        if not self.util.is_arg_given(self.args.model_nickname):
            log = 'Model nickname is not given.'
            raise ValueError(log)

        # Check if epoch is given
        if 'pretrained' not in self.args.model_nickname:
            if not self.util.is_arg_given(self.args.epoch):
                log = 'Epoch is not given.'
                raise ValueError(log)

        # Check if hyperparameters are given
        for arg, val in self.para:
            log = ''
            if not self.util.is_arg_given(val):
                log += f'{arg} is not given or invalid (the value is {val}).\n'
        if len(log) > 0:
            raise ValueError(log)

        # Generate data and log directory
        data_dir_path, log_dir_path = self.util.gen_sub_directories([
            'neuron_embedding', 
            self.args.neuron_embedding_sub_dir_name
        ])

        # Log path (key: ['neuron_emb_log'])
        if 'pretrained' in self.args.model_nickname:
            model_nickname_epoch = self.args.model_nickname
        else:
            model_nickname_epoch = f'{self.args.model_nickname}_{self.args.epoch}'
        self.path['neuron_emb_log'] = os.path.join(
            log_dir_path, f'neuron_emb_log_{model_nickname_epoch}.txt'
        )

        # Data path 
        # keys: [
        #    'neuron_emb', 
        #    'co_act', 
        #    'neuron_emb_vis',
        #    'sample_neuron',
        #    'color_map'
        # ])
        self.path['neuron_emb'] = os.path.join(
            data_dir_path, f'neuron_emb_{model_nickname_epoch}.json'
        )
        self.path['co_act'] = os.path.join(
            data_dir_path, f'co_act_{model_nickname_epoch}.json'
        )
        self.path['neuron_emb_vis'] = os.path.join(
            data_dir_path, f'neuron_emb_vis_{model_nickname_epoch}.pdf'
        )
        self.path['sample_neuron'] = None
        p = os.path.join(
            self.args.output_dir, 'neuron_embedding', 'sample_neuron.json'
        )
        if os.path.exists(p):
            self.path['sample_neuron'] = p
        
        self.path['color_map'] = None
        p = os.path.join(
            self.args.output_dir, 'neuron_embedding', 'color_map.json'
        )
        if os.path.exists(p):
            self.path['color_map'] = p

        # Save setting information
        setting_info_path = os.path.join(log_dir_path, 'setting.txt')
        with open(setting_info_path, 'w') as f:
            f.write(self.para_info)


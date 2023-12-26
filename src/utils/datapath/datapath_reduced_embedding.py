import os
from utils.datapath.datapath_util import DataPathUtil

class DataPathReducedEmbedding:
    """
    Manage paths for reduceed embedding
    """

    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}

        self.para = [
            ['reduced_embedding_sub_dir_name', self.args.reduced_embedding_sub_dir_name],
            ['neuron_embedding_path', self.args.neuron_embedding_path],
            ['proj_embedding_dir_path', self.args.proj_embedding_dir_path]
        ]

        self.para_info = '\n'.join([
            f'{name}: {val}' 
            for name, val in self.para
        ])

        self.actions_requiring_paths = [
            self.args.reduced_embedding
        ]

        self.util = DataPathUtil(args.output_dir)
        self.gen_data_paths()

    def gen_data_paths(self):
        # Check if data paths for reducing the dimensions of embeddings are necessary
        if True not in self.actions_requiring_paths:
            return

        # Check if hyperparameters are given
        log = ''
        for arg, val in self.para:
            if not self.util.is_arg_given(val):
                log += f'{arg} is not given or invalid (the value is {val}).\n'
        if len(log) > 0:
            raise ValueError(log)

        # Generate data and log directory
        data_dir_path, log_dir_path = self.util.gen_sub_directories([
            'reduced_embedding', 
            self.args.reduced_embedding_sub_dir_name
        ])

        # Log path (key: ['img_emb_log'])
        self.path['reduced_emb_log'] = os.path.join(
            log_dir_path, 'reduced_emb_log.txt'
        )

        # Data path 
        # keys: [
        #    'reduced_emb', 
        #    'neuron_embedding_path',
        #    'proj_embedding_dir_path',
        #    'sample_neuron'
        #    'color_map'
        # ]
        self.path['reduced_emb'] = data_dir_path
        self.path['neuron_embedding_path'] = self.args.neuron_embedding_path
        self.path['proj_embedding_dir_path'] = self.args.proj_embedding_dir_path

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
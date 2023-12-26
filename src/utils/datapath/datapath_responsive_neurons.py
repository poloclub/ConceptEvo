import os
from utils.datapath.datapath_util import DataPathUtil

class DataPathResponsiveNeurons:
    """
    Manage paths for responsive neurons
    """

    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}

        self.setting = [
            [
                'topk_i', 
                self.args.topk_i
            ],
            [
                'responsive_neurons_sub_dir_name', 
                self.args.responsive_neurons_sub_dir_name
            ],
            [
                'responsive_neurons_image_path', 
                self.args.responsive_neurons_image_path
            ],
        ]
        
        self.actions_requiring_paths = [
            self.args.responsive_neurons,
            self.args.indirect_image_embedding
        ]

        self.util = DataPathUtil(args.output_dir)
        self.gen_data_paths()

    def gen_data_paths(self):
        # Check if data paths for responsive neurons are necessary
        if True not in self.actions_requiring_paths:
            return

        # Check if model_name is given
        if self.args.responsive_neurons:
            if not self.util.is_arg_given(self.args.model_name):
                log = 'Model name is not given.'
                raise ValueError(log)

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
        log = ''
        for arg, val in self.setting:
            if not self.util.is_arg_given(val):
                if arg == 'responsive_neurons_image_path':
                    if self.args.responsive_neurons:
                        log += f'{arg} is not given or invalid (the value is {val}).\n'
                else:
                    log += f'{arg} is not given or invalid (the value is {val}).\n'
        if len(log) > 0:
            raise ValueError(log)

        # Generate data and log directory
        data_dir_path, log_dir_path = self.util.gen_sub_directories([
            'responsive_neurons', 
            self.args.responsive_neurons_sub_dir_name
        ])

        # Log path (key: ['responsive_neurons_log'])
        if 'pretrained' in self.args.model_nickname:
            model_nickname_epoch = self.args.model_nickname
        else:
            model_nickname_epoch = f'{self.args.model_nickname}_{self.args.epoch}'
        self.path['responsive_neurons_log'] = os.path.join(
            log_dir_path, f'responsive_neurons_log_{model_nickname_epoch}.txt'
        )

        # Data path (key: ['responsive_neurons', 'responsive_neurons_image_path'])
        self.path['responsive_neurons'] = os.path.join(
            data_dir_path, f'responsive_neurons_{model_nickname_epoch}.json'
        )
        self.path['responsive_neurons_image_path'] = \
            self.args.responsive_neurons_image_path

        # Save setting information
        setting_info_path = os.path.join(log_dir_path, 'setting.txt')
        with open(setting_info_path, 'w') as f:
            for name, val in self.setting:
                f.write(f'{name}: {val}\n')

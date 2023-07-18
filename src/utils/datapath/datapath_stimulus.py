import os
from utils.datapath.datapath_util import DataPathUtil

class DataPathStimulus:
    """
    Manage paths for stimulus
    """

    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}

        self.setting = [
            ['topk_s', self.args.topk_s],
            ['stimulus_sub_dir_name', self.args.stimulus_sub_dir_name],
            ['stimulus_image_path', self.args.stimulus_image_path],
        ]
        
        self.actions_requiring_paths = [
            self.args.stimulus,
            self.args.neuron_embedding,
            self.args.image_embedding
        ]

        self.util = DataPathUtil(args.output_dir)
        self.gen_data_paths()

    def gen_data_paths(self):
        # Check if data paths for example patches are necessary
        if True not in self.actions_requiring_paths:
            return

        # Check if model_name is given
        if self.args.stimulus:
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
        for arg, val in self.setting:
            log = ''
            if not self.util.is_arg_given(val):
                if arg == 'stimulus_image_path':
                    if self.args.stimulus:
                        log += f'{arg} is not given or invalid (the value is {val}).\n'
                else:
                    log += f'{arg} is not given or invalid (the value is {val}).\n'
        if len(log) > 0:
            raise ValueError(log)

        # Generate data and log directory
        data_dir_path, log_dir_path = self.util.gen_sub_directories([
            'stimulus', 
            self.args.stimulus_sub_dir_name
        ])

        # Log path (key: ['stimulus_log'])
        if 'pretrained' in self.args.model_nickname:
            model_nickname_epoch = self.args.model_nickname
        else:
            model_nickname_epoch = f'{self.args.model_nickname}_{self.args.epoch}'
        self.path['stimulus_log'] = os.path.join(
            log_dir_path, f'stimulus_log_{model_nickname_epoch}.txt'
        )

        # Data path (key: ['stimulus'])
        self.path['stimulus'] = os.path.join(
            data_dir_path, f'stimulus_{model_nickname_epoch}.json'
        )

        # Save setting information
        setting_info_path = os.path.join(log_dir_path, 'setting.txt')
        with open(setting_info_path, 'w') as f:
            for name, val in self.setting:
                f.write(f'{name}: {val}\n')

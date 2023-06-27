import os
from utils.datapath_util import DataPathUtil

class DataPathExamplePatch:
    """
    Manage paths for example patch of neurons
    """

    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}

        self.hypara = [
            ['topk_s', self.args.topk_s],
            ['ex_patch_size_ratio', self.args.ex_patch_size_ratio]
        ]
        self.actions_requiring_paths = [
            self.args.example_patch,
            self.args.neuron_embedding
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

        # Check if hyperparameters are given
        for arg, val in self.hypara:
            log = ''
            if not self.util.is_arg_given(val):
                log += f'{arg} is not given or invalid (the value is {val}).\n'
        if len(log) > 0:
            raise ValueError(log)

        # Generate data and log directory
        if 'pretrained' in self.args.model_nickname:
            data_dir_path, log_dir_path = self.util.gen_sub_directories([
                'example_patch', 
                self.args.model_nickname
            ])
        else:
            if not self.util.is_arg_given(self.args.epoch):
                log = 'Epoch nickname is not given.'
                raise ValueError(log)
            data_dir_path, log_dir_path = self.util.gen_sub_directories([
                'example_patch', 
                self.args.model_nickname,
                f'epoch_{self.args.epoch}'
            ])

        # Log paths for neuron features (key: ['example_patch_log'])
        apdx = [f'{arg}={val}' for arg, val in self.hypara]
        apdx = '-'.join(apdx)
        self.path['example_patch_log'] = os.path.join(
            log_dir_path, f'example_patch_log_{apdx}.txt'
        )

        # Data paths for neuron features 
        # key: [
        #   'example_patch_info', 
        #   'example_patch_crop',
        #   'example_patch_mask',
        #   'example_patch_inverse_mask'
        # ])
        data_dir_path = os.path.join(data_dir_path, apdx)
        self.util.make_dir(data_dir_path)
        for key in ['crop', 'mask', 'inverse_mask']:
            sub_dir_path = os.path.join(data_dir_path, key)
            self.path[f'example_patch_{key}'] = sub_dir_path
            self.util.make_dir(sub_dir_path)
        self.path['example_patch_info'] = os.path.join(data_dir_path, 'example_patch_info.json')
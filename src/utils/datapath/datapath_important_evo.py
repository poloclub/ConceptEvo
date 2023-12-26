import os
from utils.datapath.datapath_util import DataPathUtil

class DataPathImportantEvo:
    """
    Manage paths for finding and evaluating important evolutions
    """

    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}

        self.util = DataPathUtil(args.output_dir)

        self.path['from_model_path'] = self.args.from_model_path
        self.path['to_model_path'] = self.args.to_model_path

        self.init_for_finding()
        self.gen_data_paths_for_finding()
    
        self.init_for_evaluating()
        self.gen_data_paths_for_evaluating()

    def init_for_finding(self):
        self.para_for_finding = [
            ['label', self.args.label],
            ['model_name', self.args.model_name],
            ['from_model_path', self.args.from_model_path],
            ['to_model_path', self.args.to_model_path],
            ['important_evo_sub_dir_name', self.args.important_evo_sub_dir_name],
            ['label_img_idx_path', self.args.label_img_idx_path],
            ['num_sampled_imgs', self.args.num_sampled_imgs],
            ['idx', self.args.idx]
        ]

        self.para_info_for_finding = '\n'.join([
            f'{name}: {val}' 
            for name, val in self.para_for_finding
        ])

    def init_for_evaluating(self):
        self.para_for_evaluating = [
            ['label', self.args.label],
            ['model_name', self.args.model_name],
            ['from_model_path', self.args.from_model_path],
            ['to_model_path', self.args.to_model_path],
            ['important_evo_sub_dir_name', self.args.important_evo_sub_dir_name],
            ['label_img_idx_path', self.args.label_img_idx_path],
            ['num_sampled_imgs', self.args.num_sampled_imgs],
            ['num_bins', self.args.num_bins],
            ['idx', self.args.idx]
        ]

        self.para_info_for_evaluating = '\n'.join([
            f'{name}: {val}' 
            for name, val in self.para_for_evaluating
        ])

    def check_if_hyperpameters_given(self, para):
        # Check if hyperparameters are given
        log = ''
        for arg, val in para:
            if not self.util.is_arg_given(val):
                log += f'{arg} is not given or invalid (the given value is {val}).\n'
        if len(log) > 0:
            raise ValueError(log)
        return len(log) == 0
        
    def gen_data_paths_for_finding(self):
        # Check the necessity of data paths for ConceptEvo
        # to quantify the importance of concept evolutions
        necessary_cases = [
            self.args.find_important_evo,
            self.args.eval_important_evo
        ]
        if True not in necessary_cases:
            return
        
        # Check if hyperparameters are given
        self.check_if_hyperpameters_given(self.para_for_finding)
        
        # Generate data and log directory
        data_dir_path, log_dir_path = self.util.gen_sub_directories([
            'important_evo',
            self.args.important_evo_sub_dir_name,
            f'label={self.args.label}',
            f'num_sampled_imgs={self.args.num_sampled_imgs}'
        ])

        # Data path
        # keys: [
        #    'find_important_evo_sensitivity', 
        #    'find_important_evo_score',
        # ]
        self.path['find_important_evo_sensitivity'] = os.path.join(
            data_dir_path,
            f'sensitivity_idx={self.args.idx}.json'
        )
        self.path['find_important_evo_score'] = os.path.join(
            data_dir_path,
            f'score_idx={self.args.idx}.json'
        )

        # Log path (key: ['find_important_evo_log')
        self.path['find_important_evo_log'] = os.path.join(
            log_dir_path, 
            f'find_important_evo_log_idx={self.args.idx}.txt'
        )

        # Save setting information
        setting_info_path = os.path.join(log_dir_path, 'setting.txt')
        with open(setting_info_path, 'w') as f:
            f.write('\nFinding')
            f.write(self.para_info_for_finding)

    def gen_data_paths_for_evaluating(self):
        # Check the necessity of data paths for ConceptEvo's evaluation
        # on quantifying the importance of concept evolutions
        if not self.args.eval_important_evo:
            return

        # Check if hyperparameters are given
        self.check_if_hyperpameters_given(self.para_for_evaluating)

        # Generate data and log directory
        data_dir_path, log_dir_path = self.util.gen_sub_directories([
            'important_evo',
            self.args.important_evo_sub_dir_name,
            f'label={self.args.label}',
            f'num_sampled_imgs={self.args.num_sampled_imgs}'
        ])

        # Data path (keys: ['eval_important_evo'])
        self.path['eval_important_evo'] = os.path.join(
            data_dir_path,
            f'eval_important_evo_idx={self.args.idx}.json'
        )

        # Log path (key: ['eval_important_evo_log')
        self.path['eval_important_evo_log'] = os.path.join(
            log_dir_path, 
            f'eval_important_evo_log_idx={self.args.idx}.txt'
        )

        # Save setting information
        setting_info_path = os.path.join(log_dir_path, 'setting.txt')
        with open(setting_info_path, 'a') as f:
            f.write('\nEvaluating')
            f.write(self.para_info_for_evaluating)
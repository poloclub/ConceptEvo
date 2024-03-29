import os
from utils.datapath.datapath_util import DataPathUtil

class DataPathModel:
    """
    Manage paths for models
    """

    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}

        self.not_need_model_path = [
            self.args.sample_images,
            self.args.reduced_embedding,
            self.args.find_important_evo,
            self.args.eval_important_evo,
        ]

        self.util = DataPathUtil(args.output_dir)
        self.gen_data_paths()

    def gen_data_paths(self):
        if True in self.not_need_model_path:
            self.path['model_path'] = None
            return
        
        # Check if model_nickname is given
        if not self.util.is_arg_given(self.args.model_nickname):
            log = 'model_nickname is not given.'
            raise ValueError(log)

        # Generate data and log directory
        data_dir_path, log_dir_path = self.util.gen_sub_directories([
            'model', 
            self.args.model_nickname \
                if hasattr(self.args, 'model_nickname') else None,
        ])

        # Data paths for a model (keys: ['model_path'])
        self.gen_model_path(
            self.args.model_path \
                if hasattr(self.args, 'model_path') else None,
            self.args.model_nickname \
                if hasattr(self.args, 'model_nickname') else None,
            self.args.epoch \
                if hasattr(self.args, 'epoch') else None,
        )

        # Model and layer information (keys: ['layer_info', 'model_info'])
        self.path['layer_info'] = os.path.join(log_dir_path, 'layer_info.txt')
        self.path['model_info'] = os.path.join(log_dir_path, 'model_info.txt')

        # Data paths for model training (keys: ['model_dir', 'train_log'])
        self.path['model_dir'] = data_dir_path
        self.path['train_log'] = os.path.join(log_dir_path, 'train_log.txt')

        # Data paths for model testing (keys: ['test_log', 'test_by_class_log'])
        if 'pretrained' in self.args.model_nickname:
            for key in ['test_log', 'test_by_class_log']:
                self.path[key] = os.path.join(log_dir_path, f'{key}.txt')
        elif not self.args.train:
            if not self.util.is_arg_given(self.args.epoch):
                log = 'Epoch is not given.'
                raise ValueError(log)
            for key in ['test_log', 'test_by_class_log']:
                log_file_name = f'{key}_{self.args.epoch}.txt'        
                self.path[key] = os.path.join(log_dir_path, log_file_name)

    def gen_model_path(self, model_path, model_nickname, epoch):
        # Check if model_nickname is given
        if not self.util.is_arg_given(model_nickname):
            log = 'model_nickname is not given.'
            raise ValueError(log)

        # Save model path
        key = 'model_path'
        if self.util.is_arg_given(model_path):
            self.path[key] = model_path
        elif not self.util.is_arg_given(model_nickname):
            log = 'model_nickname is not given.'
            raise ValueError(log)  
        elif 'pretrained' in model_nickname:
            self.path[key] = None
        elif not self.util.is_arg_given(epoch):
            self.path[key] = None
        else:
            # If model_nickname and epoch are given
            model_dir_path = os.path.join(
                self.args.output_dir, 'model', model_nickname, 'data'
            )
            model_file_path = os.path.join(
                model_dir_path, f'model-{epoch}.pth'
            )
            if self.args.train and os.path.exists(model_file_path):
                time_stamp = self.util.get_time_stamp()
                model_file_path = os.path.join(
                    model_dir_path,
                    f'model-{epoch}-{time_stamp}.pth'
                )
            self.path[key] = model_file_path

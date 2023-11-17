import os
from utils.datapath.datapath_util import DataPathUtil

class DataPathIndirectImageEmbedding:
    """
    Manage paths for indirect image embedding
    """

    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}

        self.para = [
            ['num_indirect_img_emb_epochs', self.args.num_indirect_img_emb_epochs],
            ['num_indirect_img_emb_negs', self.args.num_indirect_img_emb_negs],
            ['lr_indirect_img_emb', self.args.lr_indirect_img_emb],
            ['thr_indirect_img_emb', self.args.thr_indirect_img_emb],
        ]

        self.para_info = '\n'.join([
            f'{name}: {val}' 
            for name, val in self.para
        ])

        self.actions_requiring_paths = [
            self.args.indirect_image_embedding
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
            'indirect_image_embedding', 
            self.args.indirect_image_embedding_sub_dir_name
        ])

        # Log path (key: ['indirect_img_emb_log'])
        if 'pretrained' in self.args.model_nickname:
            model_nickname_epoch = self.args.model_nickname
        else:
            model_nickname_epoch = f'{self.args.model_nickname}_{self.args.epoch}'
        self.path['indirect_img_emb_log'] = os.path.join(
            log_dir_path, f'indirect_img_emb_log_{model_nickname_epoch}.txt'
        )

        # Data path (key: ['indirect_img_emb'])
        self.path['indirect_img_emb'] = os.path.join(
            data_dir_path, f'indirect_img_emb_{model_nickname_epoch}.txt'
        )
        
        # Save setting information
        setting_info_path = os.path.join(log_dir_path, 'setting.txt')
        with open(setting_info_path, 'w') as f:
            f.write(self.para_info)
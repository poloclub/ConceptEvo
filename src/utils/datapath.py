import os
from datetime import datetime

class DataPath:
    """Manage data paths.

    This class manages path of input and output data. Some paths are 
    auto-generated. Details of the generated data paths are documented in 
    `../../docs`.
    """
    
    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}

        self.path_keys = [
            'train-data', 'test-data', 'label-data',
            'stimulus', 'co_act', 'neuron_emb', 'img_emb',
            'proj_neuron_emb', 'dim_reduction', 'neuron_feature', 
            'find_important_evo'
        ]

        self.path_key_to_actions = {}
        self.action_to_args = {}


    """
    A wrapper function called in main.py
    """
    def gen_data_dirs(self):
        """Generate data directories"""
        self.make_dir(self.args.output_dir)
        self.find_actions_and_necessary_paths()
        self.map_action_to_args()
        self.set_input_data_path()
        self.set_model_path()
        self.set_stimulus_path()
        self.set_co_act_neurons_path()
        self.set_neuron_emb_path()
        self.set_img_emb_path()
        self.set_proj_emb_path()
        self.set_emb2d_path()
        self.set_neuron_feature_path()

    
    """
    Utils
    """
    def get_path(self, path_key):
        """Return path for given path_key"""
        if path_key in self.path:
            return self.path[path_key]
        else:
            return None


    def find_actions_and_necessary_paths(self):
        """Find actions and necessary data paths for the actions.

        As defined in `./args.py` > `parse_action_setting()`, there are types of
        actions that can be run. They are:
            - `train`: whether to train a model
            - `stimulus`: whether to find stimulus
            - `neuron_emb`: whether to compute neuron embedding
            - `img_emb`: whether to compute image embedding
            - `proj_neuron_emb`: whether to compute projected neuron embedding
            - `dim_reduction`: a dimensionality redurtion method
            - `neuron_feature`: a neuron feature visualization method
        For each path key in self.path_keys, this function finds which actions
        would need data path for the corresponding key. Then it stores those 
        actions' argument value in self.path_key_to_actions.
        
        For example, four actions `stimulus`, `neuron_emb`, `img_emb`, 
        `proj_neuron_emb` need path for `stimulus` data, thus we will have:
            self.path_key_to_actions['stimulus'] = [
                self.args.stimulus,
                self.args.neuron_emb,
                self.args.img_emb,
                self.args.proj_neuron_emb
            ]
        """

        self.path_key_to_actions['train-data'] = [
            self.args.train,
            self.args.stimulus,
            self.args.neuron_feature
        ]

        self.path_key_to_actions['test-data'] = [
            self.args.train
        ]

        self.path_key_to_actions['stimulus'] = [
            self.args.stimulus,
            self.args.neuron_emb,
            self.args.img_emb,
            self.args.proj_neuron_emb
        ]

        self.path_key_to_actions['co_act'] = [
            self.args.neuron_emb
        ]

        self.path_key_to_actions['neuron_emb'] = [
            self.args.neuron_emb,
            self.args.img_emb,
            self.args.dim_reduction != 'None'
        ]

        self.path_key_to_actions['img_emb'] = [
            self.args.img_emb,
            self.args.proj_neuron_emb
        ]

        self.path_key_to_actions['proj_neuron_emb'] = [
            self.args.proj_neuron_emb
        ]

        self.path_key_to_actions['dim_reduction'] = [
            self.args.dim_reduction
        ]

        self.path_key_to_actions['neuron_feature'] = [
            self.args.neuron_feature
        ]

        self.path_key_to_actions['find_important_evo'] = [
            self.args.find_important_evo
        ]


    def map_action_to_args(self):
        """Find arguments necessary for an action.

        As defined in `./args.py` > `parse_action_setting()`, there are types of
        actions that can be run. They are:
            - `train`: whether to train a model
            - `stimulus`: whether to find stimulus
            - `neuron_emb`: whether to compute neuron embedding
            - `img_emb`: whether to compute image embedding
            - `proj_neuron_emb`: whether to compute projected neuron embedding
            - `dim_reduction`: a dimensionality redurtion method
            - `neuron_feature`: a neuron feature visualization method
        For each action, this function finds input arguments necessary for the 
        corresponding action. Then it stores the mapping of each action and 
        neccessary arguments in self.action_to_args. The necessary arguments
        are represented by a dictionary that maps the name of the argument and
        the value of the argument.
        
        For example, an action 'train' needs four input arguments: `lr`, 
        `momentum`, `num_epochs`, `top_k`. Thus, we will have:
            self.action_to_args['train'] = {
                'lr': self.args.lr,
                'momentum': self.args.momentum,
                'num_epochs': self.args.num_epochs,
                'topk': self.args.topk
            }
        """

        self.action_to_args['train'] = [
            ['lr', self.args.lr],
            ['momentum', self.args.momentum],
            ['num_epochs', self.args.num_epochs],
            ['topk', self.args.topk]
        ]
        
        self.action_to_args['stimulus'] = [
            ['topk_s', self.args.topk_s]
        ]

        self.action_to_args['neuron_emb'] = [
            ['dim', self.args.dim],
            ['lr_emb', self.args.lr_emb],
            ['num_emb_epochs', self.args.num_emb_epochs],
            ['num_emb_negs', self.args.num_emb_negs]
        ]

        self.action_to_args['img_emb'] = [
            ['dim', self.args.dim],
            ['lr_img_emb', self.args.lr_img_emb],
            ['thr_img_emb', self.args.thr_img_emb],
            ['max_iter_img_emb', self.args.max_iter_img_emb],
            ['k', self.args.k]
        ]

        self.action_to_args['proj_neuron_emb'] = [
            ['dim', self.args.dim]
        ]

        self.action_to_args['dim_reduction'] = [
            ['dim', self.args.dim],
            ['model_for_emb_space', self.args.model_for_emb_space]
        ]

        self.action_to_args['neuron_feature'] = [
            ['method', self.args.neuron_feature],
            ['num_features', self.args.num_features],
            ['ex_patch_size_ratio', self.args.ex_patch_size_ratio]
        ]

        self.action_to_args['find_important_evo'] = [
        ]


    def check_need_to_gen_path(self, path_key):
        """Check if data paths of a given path_key needs to be generated.

        Args:
            - path_key: a path key (one in self.path_keys())
        
        Returns:
            - need_or_not: True if data paths for the given path_key should be 
                generated, False otherwise.
        """

        actions = self.path_key_to_actions[path_key]
        need_or_not = False
        for action in actions:
            if self.check_if_arg_given(action):
                need_or_not = True
                break
        return need_or_not


    def make_dir(self, dir_path):
        """Generate a directory for a given path, if it does not exist.

        Args:
            - dir_path: a directory path to generate
        
        Returns:
            - N/A
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    def gen_data_log_sub_dir(self, dir_name, inner_dirname=None):
        """Generate sub-directories for data and log for a given directory name.

        For a given dir_path, it generates 
            self.args.output_dir
                └── dir_name
                    └── inner_dirname
                        ├── data
                        └── log

        Args:
            - dir_name: a directory name

        Returns:
            - data_dir_path: data sub directory path 
                (self.args.output_dir/self.args.model_nickname/dir_name/data)
            - log_dir_path: data sub directory path 
                (self.args.output_dir/self.args.model_nickname/dir_name/log)
        """

        if inner_dirname is None:
            inner_dirname = self.args.model_nickname

        dir_path = os.path.join(self.args.output_dir, dir_name, inner_dirname)
        data_dir_path = os.path.join(dir_path, 'data')
        log_dir_path = os.path.join(dir_path, 'log')
        self.make_dir(dir_path)
        self.make_dir(data_dir_path)
        self.make_dir(log_dir_path)
        
        return data_dir_path, log_dir_path


    def check_if_arg_given(self, arg):
        """Check if a given argument is given by a user.

        Args:
            - arg: an argument
        
        Returns:
            - given_or_not: True if it is given by the user, False otherwise.
        """
        
        given_or_not = False
        if 'bool' in str(type(arg)):
            given_or_not = arg
        elif 'str' in str(type(arg)):
            if (len(arg) > 0) and (arg.lower() != 'none'):
                given_or_not = True

        return given_or_not


    def auto_fill_model_nickname_and_model_path(self):
        """Automatically fill model_nickname in the input argument.

        Automatically fill model_nickname in the input argument, when the 
        argument is not given by the user and a pretrained model provided by
        pytorch is used.
        """
        if not self.check_if_arg_given(self.args.model_nickname):
            if 'pretrained' in self.args.model_name:
                self.args.model_nickname = self.args.model_name
                # self.args.model_path = self.args.model_name


    def check_model_nickname_and_path(self):
        """Check if model nickname and model path is given."""

        self.auto_fill_model_nickname_and_model_path()
        self.raise_err_for_ungiven_arg(
            self.args.model_nickname, 'model_nickname'
        )

        use_pretrained = (self.args.model_nickname == self.args.model_name)
        use_pretrained = use_pretrained and 'pretrained' in self.args.model_name
        if not use_pretrained:
            self.raise_err_for_ungiven_arg(
                self.args.model_path, 'model_path'
            )


    def raise_err_for_ungiven_arg(self, arg, arg_name):
        """Raise a ValueError when an argument is not given by the user.

        Args:
            - arg: an input argument
            - arg_name: the name of arg
        
        Returns:
            - N/A

        Raises:
            - ValueError when arg is not given by the user
        """
        if not self.check_if_arg_given(arg):
            log = f'An argument {arg_name} is not given.'
            raise ValueError(log)


    def get_time_stamp(self):
        """Get current time stamp.
        
        Args: 
            - N/A

        Returns:
            - time_stamp: a formatted string of current time stamp.
                The format is: "%Y%m%d_%H:%M:%S". i.e., yyyymmdd_hhmmss

        """
        now = datetime.now()
        time_stamp = now.strftime('%Y%m%d_%H:%M:%S')
        return time_stamp

    
    def gen_act_setting_str(self, action, delimiter='-'):
        """Generate a string for showing input argument settings for an action.

        Args:
            - action: a type of action defined in args.py. It should be one of 
                ['train', 'stimulus', 'neuron_emb', 'img_emb, 
                'proj_neuron_emb', 'dim_reduction'].
            - delimiter: delimiter between settings

        Returns:
            - arg_s: a string that shows the input argument settings for action.
                The format of this string is 
                    '<arg1>=<val1><delimiter><arg2>=<val2><delimiter>...'.

        Example:
            If action is 'train', the input arguments that need to be given
            are ['lr', 'momentum', 'num_epochs', 'topk'], as defined in
            self.action_to_args['train]. An exampe of the output is
                `lr=0.01-momentum=0.9-num_epochs=300-topk=5`.
        """

        setting_info = self.action_to_args[action]
        setting_list = ['{}={}'.format(arg, val) for arg, val in setting_info]
        arg_s = delimiter.join(setting_list)

        return arg_s


    """
    Setting paths for input data
    """
    def set_input_data_path(self):
        """Set paths for image input data."""
        self.path['train_data'] = self.args.training_data
        self.path['test_data'] = self.args.test_data


    """
    Setting paths for a model to use or train
    """
    def set_model_path(self):
        """Set paths for models."""

        if not self.check_if_arg_given(self.args.model_path):
            if self.args.train:
                self.args.model_path = 'DO_NOT_NEED_CURRENTLY'
            elif self.check_if_arg_given(self.args.dim_reduction):
                self.args.model_path = 'DO_NOT_NEED_CURRENTLY'
                self.args.model_nickname = 'DO_NOT_NEED_CURRENTLY'

        self.check_model_nickname_and_path()

        if self.args.train:
            data_dir_path, log_dir_path = self.gen_data_log_sub_dir('model')
            train_log_path = os.path.join(log_dir_path, 'training-log.txt')
            self.path['model-dir'] = data_dir_path
            self.path['train-log'] = train_log_path
        
        self.path['model-file'] = self.args.model_path
    

    def get_model_path_during_training(self, epoch):
        """Get Path for a model during training.

        Args:
            - epoch: current epoch
        
        Returns:
            - model_path: the path of model for the current epoch
        """

        model_path = os.path.join(
            self.path['model-dir'], 'model-{}.pth'.format(epoch)
        )
        if os.path.exists(model_path):
            time_stamp = self.get_time_stamp()
            model_path = 'model-{}-{}.pth'.format(time_stamp, epoch)

        return model_path

    
    """
    Setting paths for finding stimulus
    """
    def set_stimulus_path(self):
        """Set paths for stimulus."""

        if not self.check_need_to_gen_path('stimulus'):
            return

        self.check_model_nickname_and_path()

        data_dir_path, log_dir_path = self.gen_data_log_sub_dir('stimulus')
        apdx = self.gen_act_setting_str('stimulus')
        file_path = os.path.join(
            data_dir_path, 'stimulus-{}.json'.format(apdx)
        )
        log_path = os.path.join(
            log_dir_path, 'stimulus-log-{}.txt'.format(apdx)
        )

        self.path['stimulus'] = file_path
        self.path['stimulus-log'] = log_path


    """
    Setting paths for neuron embedding
    """
    def set_co_act_neurons_path(self):
        """Set paths for finding co-activated neurons."""

        if not self.check_need_to_gen_path('co_act'):
            return

        self.check_model_nickname_and_path()

        data_dir_path, log_dir_path = self.gen_data_log_sub_dir('co_act')
        file_path = os.path.join(data_dir_path, 'co-activated-neurons.json')
        log_path = os.path.join(log_dir_path, 'co-activated-neurons.txt')

        self.path['co_act'] = file_path
        self.path['co-act-log'] = log_path


    def set_neuron_emb_path(self):
        """Set paths for neuron embedding."""

        if not self.check_need_to_gen_path('neuron_emb'):
            return

        self.check_model_nickname_and_path()

        data_dir_path, log_dir_path = self.gen_data_log_sub_dir('embedding')
        apdx = self.gen_act_setting_str('neuron_emb')
        file_path = os.path.join(
            data_dir_path, 'neuron_emb-{}-{}.json'.format(
                self.args.model_nickname.replace('-', '_'),
                apdx
            )
        )
        log_path = os.path.join(
            log_dir_path, 'neuron_emb-log-{}-{}.txt'.format(
                self.args.model_nickname.replace('-', '_'),
                apdx
            )
        )

        self.path['neuron_emb'] = file_path
        self.path['neuron_emb-log'] = log_path


    """
    Setting paths for image embedding
    """
    def set_img_emb_path(self):
        if not self.check_need_to_gen_path('img_emb'):
            return

        self.check_model_nickname_and_path()

        data_dir_path, log_dir_path = self.gen_data_log_sub_dir('embedding')
        apdx = self.gen_act_setting_str('img_emb')
        file_path = os.path.join(
            data_dir_path, 'img_emb-{}.txt'.format(apdx)
        )
        log_path = os.path.join(
            log_dir_path, 'img_emb-log-{}.txt'.format(apdx)
        )
        self.path['img_emb'] = file_path
        self.path['img_emb-log'] = log_path


    """
    Setting paths for approximate projected neuron embedding
    """
    def set_proj_emb_path(self):
        if not self.check_need_to_gen_path('proj_neuron_emb'):
            return

        self.check_model_nickname_and_path()
        self.raise_err_for_ungiven_arg(self.args.img_emb_path, 'img_emb_path')

        data_dir_path, log_dir_path = self.gen_data_log_sub_dir(
            'embedding', inner_dirname=self.args.model_nickname
        )
        apdx = self.gen_act_setting_str('proj_neuron_emb')
        file_name = 'proj_neuron_emb-{}-{}.json'.format(
            self.args.model_nickname.replace('-', '_'),
            apdx
        )
        file_path = os.path.join(data_dir_path, file_name)
        log_path = os.path.join(
            log_dir_path, 'proj_neuron_emb-log-{}-{}.txt'.format(
                self.args.model_nickname.replace('-', '_'),
                apdx
            )
        )
        self.path['proj_neuron_emb'] = file_path
        self.path['proj_neuron_emb-log'] = log_path

        if self.check_if_arg_given(self.args.emb_store_dirname):
            dir_path = os.path.join(
                self.args.output_dir, 'embedding', self.args.emb_store_dirname
            )
            self.make_dir(dir_path)
            self.path['proj_neuron_emb-store'] = os.path.join(
                dir_path, file_name
            )


    """
    Setting paths for dimensionality reduction of embeddings
    """
    def set_emb2d_path(self):
        if not self.check_need_to_gen_path('dim_reduction'):
            return

        self.raise_err_for_ungiven_arg(self.args.emb_set_dir, 'emb_set_dir')
        data_dir_path, log_dir_path = self.gen_data_log_sub_dir(
            'embedding', inner_dirname='emb2d'
        )
        
        emb_set_dir_name = self.args.emb_set_dir.split('/')[-1]
        data_dir_path = os.path.join(data_dir_path, emb_set_dir_name)
        self.make_dir(data_dir_path)

        apdx = self.gen_act_setting_str('dim_reduction')
        log_path = os.path.join(
            log_dir_path, 
            'emb2d-log-{}-{}.txt'.format(apdx, emb_set_dir_name)
        )
        idx_path = os.path.join(
            data_dir_path, 'emb2d-idx2id-{}.json'.format(apdx)
        )
        code_path = os.path.join(
            data_dir_path, 'model_code-{}.json'.format(apdx)
        )
        reducer_path = os.path.join(
            data_dir_path, 'reducer-{}.sav'.format(apdx)
        )
        self.path['dim_reduction-dir'] = data_dir_path
        self.path['dim_reduction-apdx'] = apdx
        self.path['dim_reduction-log'] = log_path
        self.path['dim_reduction-idx2id'] = idx_path
        self.path['dim_reduction-model_code'] = code_path
        self.path['dim_reduction-reducer'] = reducer_path


    """
    Setting paths for generating neurons' feature
    """
    def set_neuron_feature_path(self):
        if not self.check_need_to_gen_path('neuron_feature'):
            return

        self.auto_fill_model_nickname_and_model_path()
        self.raise_err_for_ungiven_arg(
            self.args.model_nickname, 'model_nickname'
        )

        d_dir_path, l_dir_path = self.gen_data_log_sub_dir('neuron_feature')
        apdx = self.gen_act_setting_str('neuron_feature')
        log_path = os.path.join(
            l_dir_path, 'neuron_feature-log-{}.txt'.format(apdx)
        )
        self.make_dir(d_dir_path)

        self.path['neuron_feature'] = d_dir_path
        self.path['neuron_feature-log'] = log_path

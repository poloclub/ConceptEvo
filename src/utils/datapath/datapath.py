import os
from datetime import datetime

from utils.datapath.datapath_model import DataPathModel
from utils.datapath.datapath_stimulus import DataPathStimulus
from utils.datapath.datapath_neuron_embedding import DataPathNeuronEmbedding
from utils.datapath.datapath_image_embedding import DataPathImageEmbedding
from utils.datapath.datapath_proj_embedding import DataPathProjEmbedding
from utils.datapath.datapath_reduced_embedding import DataPathReducedEmbedding
from utils.datapath.datapath_example_patch import DataPathExamplePatch

class DataPath:
    """
    Manage data paths

    This class manages path of input and output data. 
    Some paths are auto-generated.
    """
    
    """
    Constructor
    """
    def __init__(self, args):
        self.args = args
        self.path = {}
        self.gen_data_dirs()

    """
    A wrapper function called in main.py
    """
    def gen_data_dirs(self):
        # Generate the output root directory
        self.make_dir(self.args.output_dir)

        # Generate input image path
        self.set_input_image_path()

        # Generate paths for model data
        self.data_path_model = DataPathModel(self.args)
        self.path = {**self.path, **self.data_path_model.path}

        # Generate paths for example patch
        self.data_path_example_patch = DataPathExamplePatch(self.args)
        self.path = {**self.path, **self.data_path_example_patch.path}

        # Generate paths for stimulus
        self.data_path_stimulus = DataPathStimulus(self.args)
        self.path = {**self.path, **self.data_path_stimulus.path}

        # Generate paths for neuron embedding
        self.data_path_neuron_embedding = DataPathNeuronEmbedding(self.args)
        self.path = {**self.path, **self.data_path_neuron_embedding.path}

        # Generate paths for image embedding
        self.data_path_image_embedding = DataPathImageEmbedding(self.args)
        self.path = {**self.path, **self.data_path_image_embedding.path}

        # Generate paths for projected embedding
        self.data_path_proj_embedding = DataPathProjEmbedding(self.args)
        self.path = {**self.path, **self.data_path_proj_embedding.path}

        # Generate paths for reduced embedding
        self.data_path_reduced_embedding = DataPathReducedEmbedding(self.args)
        self.path = {**self.path, **self.data_path_reduced_embedding.path}

        # self.find_actions_and_necessary_paths()
        # self.map_action_to_args()
        # self.set_model_path()
        # self.set_layer_act_path()
        # self.set_stimulus_path()
        # self.set_co_act_neurons_path()
        # self.set_neuron_emb_path()
        # self.set_img_emb_path()
        # self.set_img_emb_co_act_path()
        # self.set_img_pairs_path()
        # self.set_img_emb_layer_co_act()
        # self.set_proj_emb_path()
        # self.set_emb2d_path()
        # self.set_neuron_feature_path()
        # self.set_act_map_path()
        # self.set_important_evolution_path()
        # self.set_eval_important_evolution_path()
        # self.set_important_neuron_path()
        # self.set_important_neuron_act_map_path()

    
    def get_path(self, path_key):
        """Return path for given path_key"""
        if path_key in self.path:
            return self.path[path_key]
        else:
            return None

    def set_input_image_path(self):
        """Set paths for image input data."""
        self.path['train_data'] = self.args.training_data
        self.path['test_data'] = self.args.test_data













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

        self.path_key_to_actions['layer_act'] = [
            self.args.layer_act,
            self.args.img_pairs,
            self.args.img_emb_co_act
        ]

        self.path_key_to_actions['stimulus'] = [
            self.args.stimulus,
            self.args.neuron_emb,
            self.args.img_emb,
            self.args.img_emb_co_act,
            self.args.proj_neuron_emb,
            self.args.neuron_feature,
            self.args.act_map
        ]

        self.path_key_to_actions['co_act'] = [
            self.args.neuron_emb
        ]

        self.path_key_to_actions['neuron_emb'] = [
            self.args.neuron_emb,
            self.args.img_emb,
            self.args.img_emb_co_act,
            self.args.dim_reduction != 'None'
        ]

        self.path_key_to_actions['img_emb'] = [
            self.args.img_emb,
            self.args.img_emb_co_act,
            self.args.proj_neuron_emb
        ]

        self.path_key_to_actions['img_pairs'] = [
            self.args.img_pairs,
            self.args.img_emb_co_act,
        ]

        self.path_key_to_actions['img_emb_co_act'] = [
            self.args.img_emb_co_act,
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

        self.path_key_to_actions['act_map'] = [
            self.args.act_map
        ]

        self.path_key_to_actions['find_important_evo'] = [
            self.args.find_important_evo,
            self.args.eval_important_evo
        ]

        self.path_key_to_actions['eval_important_evo'] = [
            self.args.eval_important_evo
        ]

        self.path_key_to_actions['important_neuron'] = [
            self.args.important_neuron,
            self.args.important_neuron_act_map
        ]

        self.path_key_to_actions['important_neuron_act_map'] = [
            self.args.important_neuron_act_map
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

        self.action_to_args['layer_act'] = [
            ['model_nickname', self.args.model_nickname],
            ['layer', self.args.layer]
        ]
        
        self.action_to_args['stimulus'] = [
            ['topk_s', self.args.topk_s]
        ]

        self.action_to_args['neuron_emb'] = [
            ['topk_s', self.args.topk_s],
            ['dim', self.args.dim],
            ['lr_emb', self.args.lr_emb],
            ['num_emb_epochs', self.args.num_emb_epochs],
            ['num_emb_negs', self.args.num_emb_negs]
        ]

        max_iter = self.args.max_iter_img_emb
        if self.is_given_arg(self.args.from_iter_img_emb):
            max_iter += self.args.from_iter_img_emb
        self.action_to_args['img_emb'] = [
            ['dim', self.args.dim],
            ['lr_img_emb', self.args.lr_img_emb],
            ['thr_img_emb', self.args.thr_img_emb],
            ['max_iter_img_emb', max_iter],
            ['k', self.args.k]
        ]

        self.action_to_args['img_pairs'] = [
            ['layer', self.args.layer],
            ['num_epochs_co_act', self.args.num_epochs_co_act],
            ['k', self.args.k]
        ]

        self.action_to_args['img_emb_co_act'] = [
            ['dim', self.args.dim],
            ['layer', self.args.layer],
            ['lr_img_emb', self.args.lr_img_emb],
            ['k', self.args.k]
        ]

        self.action_to_args['img_emb_from'] = [
            ['dim', self.args.dim],
            ['lr_img_emb', self.args.lr_img_emb],
            ['thr_img_emb', self.args.thr_img_emb],
            ['max_iter_img_emb', self.args.from_iter_img_emb],
            ['k', self.args.k]
        ]

        self.action_to_args['proj_neuron_emb'] = [
            ['model_nickname', self.args.model_nickname],
            ['dim', self.args.dim]
        ]

        self.action_to_args['dim_reduction'] = [
            # ['dim', self.args.dim],
            # ['model_for_emb_space', self.args.model_for_emb_space]
        ]

        self.action_to_args['neuron_feature'] = [
            ['topk_s', self.args.topk_s],
            ['ex_patch_size_ratio', self.args.ex_patch_size_ratio]
        ]

        self.action_to_args['act_map'] = [
            ['topk_s', self.args.topk_s],
        ]

        self.action_to_args['find_important_evo'] = [
            ['from', self.args.from_model_nickname.replace('-', '_')],
            ['to', self.args.to_model_nickname.replace('-', '_')],
            ['label', self.args.label],
            ['num_sampled_imgs', self.args.num_sampled_imgs],
            ['idx', self.args.idx]
        ]

        self.action_to_args['eval_important_evo'] = [
            ['from', self.args.from_model_nickname.replace('-', '_')],
            ['to', self.args.to_model_nickname.replace('-', '_')],
            ['label', self.args.label],
            ['num_bins', self.args.num_bins],
            ['num_sampled_imgs', self.args.num_sampled_imgs],
            ['idx', self.args.idx]
        ]

        self.action_to_args['important_neuron'] = [
            ['label', self.args.label],
            ['layer', self.args.layer],
            ['topk_n', self.args.topk_n],
        ]
        self.action_to_args['important_neuron_act_map'] = [
            ['label', self.args.label],
            ['layer', self.args.layer],
            ['topk_n', self.args.topk_n],
        ]


    def need_to_gen_path(self, path_key):
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
            if self.is_given_arg(action):
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


    def gen_data_log_sub_dir(self, dir_name, inner_dirname=None, mkdir=True):
        """Generate sub-directories for data and log for a given directory name.

        For a given dir_path, it generates 
            self.args.output_dir
                └── dir_name
                    └── inner_dirname (model_nickname if it's not given)
                        ├── data
                        └── log

        Args:
            - dir_name: a directory name
            - inner_dirname: the name of inner directory
            - mkdir: whether to make the directories

        Returns:
            - data_dir_path: data sub directory path 
                (self.args.output_dir/dir_name/inner_dirname/data)
            - log_dir_path: data sub directory path 
                (self.args.output_dir/dir_name/inner_dirname/log)
        """

        if inner_dirname is None:
            inner_dirname = self.args.model_nickname

        dir_path = os.path.join(self.args.output_dir, dir_name, inner_dirname)
        data_dir_path = os.path.join(dir_path, 'data')
        log_dir_path = os.path.join(dir_path, 'log')

        if mkdir:
            self.make_dir(dir_path)
            self.make_dir(data_dir_path)
            self.make_dir(log_dir_path)
            
        return data_dir_path, log_dir_path


    def is_given_arg(self, arg):
        """Check if a given argument is given by a user.

        Args:
            - arg: an argument
        
        Returns:
            - given_or_not: True if it is given by the user, False otherwise.
        """
        
        given_or_not = False
        arg_type = str(type(arg))
        if 'bool' in arg_type:
            given_or_not = arg
        elif 'str' in arg_type:
            if (len(arg) > 0) and (arg.lower() != 'none'):
                given_or_not = True
        elif 'int' in arg_type:
            if arg >= 0:
                given_or_not = True
        else:
            given_or_not = True

        return given_or_not


    def auto_fill_model_nickname_and_model_path(self):
        """Automatically fill model_nickname in the input argument.

        Automatically fill model_nickname in the input argument, when the 
        argument is not given by the user and a pretrained model provided by
        pytorch is used.
        """
        if not self.is_given_arg(self.args.model_nickname):
            if 'pretrained' in self.args.model_name:
                self.args.model_nickname = self.args.model_name
                self.args.model_path = self.args.model_name


    def check_model_nickname_and_path(self):
        """Check if model nickname and model path is given."""

        self.auto_fill_model_nickname_and_model_path()
        self.raise_err_for_ungiven_arg(
            self.args.model_nickname, 'model_nickname'
        )
        
        use_pretrained = 'pretrained' in self.args.model_name
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
        if not self.is_given_arg(arg):
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
    Setting paths for a model to use or train
    """
    def set_model_path(self):
        """Set paths for models."""

        if not self.is_given_arg(self.args.model_path):
            when_to_skip = [
                self.args.train,
                self.is_given_arg(self.args.proj_neuron_emb),
                self.is_given_arg(self.args.img_pairs),
                self.is_given_arg(self.args.dim_reduction),
                self.is_given_arg(self.args.find_important_evo),
                self.is_given_arg(self.args.eval_important_evo)
            ]
            if True in when_to_skip:
                self.args.model_path = 'DO_NOT_NEED_CURRENTLY'
            if self.is_given_arg(self.args.dim_reduction):
                self.args.model_nickname = 'DO_NOT_NEED_CURRENTLY'

        self.check_model_nickname_and_path()

        data_dir_path, log_dir_path = self.gen_data_log_sub_dir('model')
        model_info_path = os.path.join(log_dir_path, 'model-info.txt')
        layer_info_path = os.path.join(log_dir_path, 'layer-info.txt')
        self.path['model-info'] = model_info_path
        self.path['layer-info'] = layer_info_path

        if self.args.train:
            train_log_path = os.path.join(log_dir_path, 'training-log.txt')
            self.path['model-dir'] = data_dir_path
            self.path['train-log'] = train_log_path
        elif self.args.test:
            filename = 'test.txt'
            if 'pretrained' not in self.args.model_nickname:
                self.raise_err_for_ungiven_arg(self.args.epoch, 'epoch')
                filename = 'test-epoch={}.txt'.format(self.args.epoch)
                
            test_log_path = os.path.join(log_dir_path, filename)
            train_log_path = os.path.join(log_dir_path, 'training-test-log.txt')
            self.path['test-log'] = test_log_path
            self.path['train-log'] = train_log_path
        elif self.args.test_by_class:
            filename = 'test_by_class.txt'
            if 'pretrained' not in self.args.model_nickname:
                self.raise_err_for_ungiven_arg(self.args.epoch, 'epoch')
                filename = 'test-by-class-epoch={}.txt'.format(self.args.epoch)
                
            data_dir_path, log_dir_path = self.gen_data_log_sub_dir('model')
            test_log_path = os.path.join(log_dir_path, filename)
            train_log_path = os.path.join(log_dir_path, 'training-test-log.txt')
            model_info_path = os.path.join(log_dir_path, 'model-info.txt')
            layer_info_path = os.path.join(log_dir_path, 'layer-info.txt')
            self.path['test-by-class-log'] = test_log_path
            self.path['train-log'] = train_log_path
            self.path['model-info'] = model_info_path
            self.path['layer-info'] = layer_info_path
        
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
            model_path = os.path.join(
                self.path['model-dir'],
                'model-{}-{}.pth'.format(time_stamp, epoch)
            )

        return model_path

    """
    Setting paths for computing layer act
    """
    def set_layer_act_path(self):
        if not self.need_to_gen_path('layer_act'):
            return

        if self.args.layer_act:
            self.check_model_nickname_and_path()

        d_path, l_path = self.gen_data_log_sub_dir('layer_act')
        self.make_dir(d_path)

        d_path = os.path.join(d_path, self.args.layer)
        self.make_dir(d_path)

        f_path = os.path.join(d_path, 'img_emb.txt')
        self.path['layer_act'] = f_path

        apdx = self.gen_act_setting_str('layer_act')
        l_path = os.path.join(
            l_path,
            f'log-layer_act-{self.args.model_nickname}-{apdx}.txt'
        )
        self.path['layer_act-log'] = l_path

    
    """
    Setting paths for finding stimulus
    """
    def set_stimulus_path(self):
        """Set paths for stimulus."""
        if not self.need_to_gen_path('stimulus'):
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
        if not self.need_to_gen_path('co_act'):
            return

        self.check_model_nickname_and_path()

        data_dir_path, log_dir_path = self.gen_data_log_sub_dir('co_act')
        file_path = os.path.join(data_dir_path, 'co-activated-neurons.json')
        log_path = os.path.join(log_dir_path, 'co-activated-neurons.txt')

        self.path['co_act'] = file_path
        self.path['co-act-log'] = log_path


    def set_neuron_emb_path(self):
        """Set paths for neuron embedding."""
        if not self.need_to_gen_path('neuron_emb'):
            return

        self.check_model_nickname_and_path()

        apdx = self.gen_act_setting_str('neuron_emb')

        model_nickname = self.args.model_nickname
        if self.is_given_arg(self.args.dim_reduction):
            model_nickname = self.args.basemodel_nickname

        data_dir_path, log_dir_path = self.gen_data_log_sub_dir(
            'embedding', 
            inner_dirname='emb-{}-{}'.format(model_nickname, apdx),
            mkdir=self.args.neuron_emb
        )
        data_dir_path = os.path.join(data_dir_path, 'emb')
        self.make_dir(data_dir_path)

        file_path = os.path.join(data_dir_path, 'emb.json')
        vis_path = os.path.join(data_dir_path, 'emb.png')
        log_path = os.path.join(log_dir_path, 'log.txt')

        self.path['neuron_emb'] = file_path
        self.path['neuron_emb-vis'] = vis_path
        self.path['neuron_emb-log'] = log_path


    # """
    # Setting paths for image embedding
    # """
    # def set_img_emb_path(self):
    #     if not self.need_to_gen_path('img_emb'):
    #         return

    #     self.check_model_nickname_and_path()

    #     # Basemodel nickname
    #     if self.is_given_arg(self.args.basemodel_nickname):
    #         basemodel_nickname = self.args.basemodel_nickname
    #     else:
    #         basemodel_nickname = self.args.model_nickname

    #     # Directory for neuron embedding
    #     neuron_emb_apdx = self.gen_act_setting_str('neuron_emb')
    #     root_data_dir_path, log_dir_path = self.gen_data_log_sub_dir(
    #         'embedding', 
    #         inner_dirname='emb-{}-{}'.format(
    #             basemodel_nickname, neuron_emb_apdx
    #         )
    #     )

    #     # Parent directory for image embedding
    #     img_emb_apdx = self.gen_act_setting_str('img_emb')
    #     data_dir_path = os.path.join(
    #         root_data_dir_path, 'emb-set-{}'.format(img_emb_apdx)
    #     )
    #     self.make_dir(data_dir_path)

    #     # Parent directory for image embedding 
    #     # from which further iterations starts
    #     if self.is_given_arg(self.args.from_iter_img_emb):
    #         img_emb_from_apdx = self.gen_act_setting_str('img_emb_from')
    #         from_data_dir_path = os.path.join(
    #             root_data_dir_path, 'emb-set-{}'.format(img_emb_from_apdx)
    #         )

    #     # Directory for image embedding
    #     data_dir_path = os.path.join(data_dir_path, 'emb_nd')
    #     self.make_dir(data_dir_path)

    #     # Directory for image embedding from which further iterations starts
    #     if self.is_given_arg(self.args.from_iter_img_emb):
    #         from_data_dir_path = os.path.join(from_data_dir_path, 'emb_nd')

    #     # Files
    #     file_path = os.path.join(data_dir_path, 'img_emb.txt')
    #     log_path = os.path.join(
    #         log_dir_path, 'img_emb-log-{}.txt'.format(img_emb_apdx)
    #     )
    #     self.path['img_emb'] = file_path
    #     self.path['img_emb-log'] = log_path
    #     self.path['label_img_idx'] = os.path.join(
    #         self.args.output_dir, 'ILSVRC2012_label_img_idx.json'
    #     )

    #     if self.is_given_arg(self.args.from_iter_img_emb):
    #         from_file_path = os.path.join(from_data_dir_path, 'img_emb.txt')
    #         self.path['img_emb_from'] = from_file_path

    """
    Setting paths for image pairs
    """
    def set_img_pairs_path(self):
        if not self.need_to_gen_path('img_pairs'):
            return
        
        # Basemodel nickname
        if self.is_given_arg(self.args.basemodel_nickname):
            basemodel_nickname = self.args.basemodel_nickname
        else:
            self.raise_err_for_ungiven_arg(
                self.args.model_nickname, 'model_nickname'
            )
            basemodel_nickname = self.args.model_nickname

        # Check user-given parameters
        self.raise_err_for_ungiven_arg(
            self.args.layer, 'layer'
        )
        self.raise_err_for_ungiven_arg(
            self.args.num_epochs_co_act, 'num_epochs_co_act'
        )
        self.raise_err_for_ungiven_arg(
            self.args.k, 'k'
        )

        # Image pairs path
        apdx = self.gen_act_setting_str('img_pairs')
        d_path, l_path = self.gen_data_log_sub_dir('img_pairs')
        d_path = os.path.join(d_path, self.args.layer)
        self.make_dir(d_path)
        d_path = os.path.join(d_path, f'img_pairs-{apdx}.json')

        # Log path
        log_path = os.path.join(l_path, f'img_pairs-log-{apdx}.txt')

        # File paths
        self.path['img_pairs'] = d_path
        self.path['img_pairs-log'] = log_path

    """
    Setting paths for image embedding with co-activation
    """
    def set_img_emb_co_act_path(self):
        if not self.need_to_gen_path('img_emb_co_act'):
            return

        # Basemodel nickname
        if self.is_given_arg(self.args.basemodel_nickname):
            basemodel_nickname = self.args.basemodel_nickname
        else:
            basemodel_nickname = self.args.model_nickname

        # Layer activation path
        d_path, l_path = self.gen_data_log_sub_dir('layer_act')
        self.path['layer_act_dir'] = d_path

        # Directory for neuron embedding
        neuron_emb_apdx = self.gen_act_setting_str('neuron_emb')
        root_data_dir_path, log_dir_path = self.gen_data_log_sub_dir(
            'embedding', 
            inner_dirname='emb-{}-{}'.format(
                basemodel_nickname, neuron_emb_apdx
            )
        )

        # Parent directory for image embedding
        apdx2 = self.gen_act_setting_str('img_emb')
        apdx3 = self.gen_act_setting_str('img_emb_co_act')
        data_dir_path = os.path.join(
            root_data_dir_path, 
            f'emb-set-{apdx2}-{apdx3}'
        )
        self.make_dir(data_dir_path)

        # Directory for image embedding
        data_dir_path = os.path.join(data_dir_path, 'emb_nd')
        self.make_dir(data_dir_path)

        # Files
        file_path = os.path.join(data_dir_path, 'img_emb.txt')
        log_path = os.path.join(
            log_dir_path, 'img_emb_co_act-log-{}.txt'.format(apdx3)
        )
        self.path['img_emb_co_act'] = file_path
        self.path['img_emb_co_act-log'] = log_path

    """
    Setting paths for image embedding with image co-activation
    """
    def set_img_emb_layer_co_act(self):
        if not self.need_to_gen_path('img_emb_co_act'):
            return

        # Basemodel nickname
        if self.is_given_arg(self.args.basemodel_nickname):
            basemodel_nickname = self.args.basemodel_nickname
        else:
            basemodel_nickname = self.args.model_nickname

        # Layer activation path
        d_path, l_path = self.gen_data_log_sub_dir('layer_act')
        p = os.path.join(d_path, self.args.layer)
        p = os.path.join(p, 'img_emb.txt')
        self.path['layer_act'] = p

        # Directory for neuron embedding
        neuron_emb_apdx = self.gen_act_setting_str('neuron_emb')
        root_data_dir_path, log_dir_path = self.gen_data_log_sub_dir(
            'embedding', 
            inner_dirname='emb-{}-{}'.format(
                basemodel_nickname, neuron_emb_apdx
            )
        )

        # Parent directory for image embedding
        apdx2 = self.gen_act_setting_str('img_emb')
        apdx3 = self.gen_act_setting_str('img_emb_co_act')
        data_dir_path = os.path.join(
            root_data_dir_path, 
            f'emb-set-co_act-{apdx2}-{apdx3}'
        )
        self.make_dir(data_dir_path)

        # Directory for image embedding
        data_dir_path = os.path.join(data_dir_path, 'emb_nd')
        self.make_dir(data_dir_path)

        # Files
        file_path = os.path.join(data_dir_path, 'img_emb.txt')
        log_path = os.path.join(
            log_dir_path, 'img_emb_co_act-log-{}.txt'.format(apdx3)
        )
        self.path['img_emb_co_act'] = file_path
        self.path['img_emb_co_act-log'] = log_path


    # """
    # Setting paths for approximate projected neuron embedding
    # """
    # def set_proj_emb_path(self):
    #     if not self.need_to_gen_path('proj_neuron_emb'):
    #         return

    #     # Check if the nicknames of the base model and model are given
    #     self.check_model_nickname_and_path()
    #     self.raise_err_for_ungiven_arg(
    #         self.args.basemodel_nickname, 'basemodel_nickname'
    #     )

    #     # Directory
    #     neuron_emb_apdx = self.gen_act_setting_str('neuron_emb')
    #     data_dir_path, log_dir_path = self.gen_data_log_sub_dir(
    #         'embedding', 
    #         inner_dirname='emb-{}-{}'.format(
    #             self.args.basemodel_nickname, neuron_emb_apdx
    #         )
    #     )
    #     img_emb_apdx = self.gen_act_setting_str('img_emb')
    #     data_dir_path = os.path.join(
    #         data_dir_path, 'emb-set-{}'.format(img_emb_apdx)
    #     )
    #     self.make_dir(data_dir_path)

    #     data_dir_path = os.path.join(data_dir_path, 'emb_nd')
    #     self.make_dir(data_dir_path)

    #     # Files
    #     file_name = 'proj_emb-{}.json'.format(self.args.model_nickname)
    #     file_path = os.path.join(data_dir_path, file_name)
    #     log_path = os.path.join(
    #         log_dir_path, 'proj_emb-log-{}-{}.txt'.format(
    #             self.args.model_nickname, img_emb_apdx
    #         )
    #     )

    #     # Path
    #     self.path['proj_neuron_emb'] = file_path
    #     self.path['proj_neuron_emb-log'] = log_path

    """
    Setting paths for dimensionality reduction of embeddings
    """
    def set_emb2d_path(self):
        if not self.need_to_gen_path('dim_reduction'):
            return

        # Directory
        if self.is_given_arg(self.args.basemodel_nickname):
            basemodel_nickname = self.args.basemodel_nickname
        else:
            basemodel_nickname = self.args.model_nickname

        neuron_emb_apdx = self.gen_act_setting_str('neuron_emb')
        data_dir_path, log_dir_path = self.gen_data_log_sub_dir(
            'embedding', 
            inner_dirname='emb-{}-{}'.format(
                basemodel_nickname, neuron_emb_apdx
            )
        )

        img_emb_apdx = self.gen_act_setting_str('img_emb')
        data_dir_path = os.path.join(
            data_dir_path, 'emb-set-{}'.format(img_emb_apdx)
        )
        self.make_dir(data_dir_path)
        self.path['emb_nd-dir'] = os.path.join(data_dir_path, 'emb_nd')
        data_dir_path = os.path.join(data_dir_path, 'emb_2d')
        self.make_dir(data_dir_path)

        # Files
        log_path = os.path.join(
            log_dir_path, 'emb2d-log-{}.txt'.format(img_emb_apdx)
        )
        reducer_path = os.path.join(data_dir_path, 'reducer.sav')
        
        # Paths
        self.path['emb2d-dir'] = data_dir_path
        self.path['emb2d-log'] = log_path
        self.path['emb2d-reducer'] = reducer_path

    """
    Setting paths for generating neurons' feature
    """
    def set_neuron_feature_path(self):
        if not self.need_to_gen_path('neuron_feature'):
            return

        # self.auto_fill_model_nickname_and_model_path()
        self.raise_err_for_ungiven_arg(
            self.args.model_nickname, 'model_nickname'
        )

        d_dir_path, l_dir_path = self.gen_data_log_sub_dir('neuron_feature')
        self.make_dir(d_dir_path)
        apdx = self.gen_act_setting_str('neuron_feature')
        d_dir_path = os.path.join(d_dir_path, apdx)
        self.make_dir(d_dir_path)
        img_d_dir_path = os.path.join(d_dir_path, 'img')
        self.make_dir(img_d_dir_path)

        data_path = os.path.join(
            d_dir_path, 'neuron_feature-{}.json'.format(apdx)
        )
        log_path = os.path.join(
            l_dir_path, 'neuron_feature-log-{}.txt'.format(apdx)
        )

        self.path['neuron_feature-info'] = data_path
        self.path['neuron_feature'] = img_d_dir_path
        self.path['neuron_feature-log'] = log_path

    """
    Setting paths for generating activation maps for given stimulus
    """
    def set_act_map_path(self):
        if not self.need_to_gen_path('act_map'):
            return

        self.raise_err_for_ungiven_arg(
            self.args.model_nickname, 'model_nickname'
        )

        d_dir_path, l_dir_path = self.gen_data_log_sub_dir('act_map')
        self.make_dir(d_dir_path)
        apdx = self.gen_act_setting_str('act_map')
        d_dir_path = os.path.join(d_dir_path, apdx)
        self.make_dir(d_dir_path)

        log_path = os.path.join(
            l_dir_path, 'act_map-log-{}.txt'.format(apdx)
        )

        self.path['act_map'] = d_dir_path
        self.path['act_map-log'] = log_path


    """
    Setting paths for finding important evolution
    """
    def set_important_evolution_path(self):
        if not self.need_to_gen_path('find_important_evo'):
            return

        self.auto_fill_model_nickname_and_model_path()
        self.raise_err_for_ungiven_arg(
            self.args.model_name, 'model_name'
        )
        self.raise_err_for_ungiven_arg(
            self.args.model_nickname, 'model_nickname'
        )
        self.raise_err_for_ungiven_arg(
            self.args.from_model_nickname, 'from_model_nickname'
        )
        self.raise_err_for_ungiven_arg(
            self.args.from_model_path, 'from_model_path'
        )
        self.raise_err_for_ungiven_arg(
            self.args.to_model_nickname, 'to_model_nickname'
        )
        self.raise_err_for_ungiven_arg(
            self.args.to_model_path, 'to_model_path'
        )

        d_dir_path, l_dir_path = self.gen_data_log_sub_dir('find_important_evo')
        apdx = self.gen_act_setting_str('find_important_evo')
        log_path = os.path.join(
            l_dir_path, 
            'find_important_evo-log-{}.txt'.format(apdx)
        )
        self.make_dir(d_dir_path)

        self.path['find_important_evo-sensitivity'] = os.path.join(
            d_dir_path, 'sensitivity-{}.json'.format(apdx)
        )
        self.path['find_important_evo-score'] = os.path.join(
            d_dir_path, 'score-{}.json'.format(apdx)
        )
        self.path['find_important_evo-log'] = log_path

    """
    Setting paths for evaluating important evolution
    """
    def set_eval_important_evolution_path(self):
        if not self.need_to_gen_path('eval_important_evo'):
            return

        self.auto_fill_model_nickname_and_model_path()
        self.raise_err_for_ungiven_arg(
            self.args.model_name, 'model_name'
        )
        self.raise_err_for_ungiven_arg(
            self.args.model_nickname, 'model_nickname'
        )
        self.raise_err_for_ungiven_arg(
            self.args.from_model_nickname, 'from_model_nickname'
        )
        self.raise_err_for_ungiven_arg(
            self.args.from_model_path, 'from_model_path'
        )
        self.raise_err_for_ungiven_arg(
            self.args.to_model_nickname, 'to_model_nickname'
        )
        self.raise_err_for_ungiven_arg(
            self.args.to_model_path, 'to_model_path'
        )

        d_dir_path, l_dir_path = self.gen_data_log_sub_dir('eval_important_evo')
        apdx = self.gen_act_setting_str('eval_important_evo')
        log_path = os.path.join(
            l_dir_path, 
            'eval_important_evo-log-{}.txt'.format(apdx)
        )
        self.make_dir(d_dir_path)

        data_path = os.path.join(d_dir_path, f'eval_evo-{apdx}.json')
        self.path['eval_important_evo'] = data_path
        self.path['eval_important_evo-log'] = log_path
       
    """
    Setting paths for finding important neuron
    """
    def set_important_neuron_path(self):
        if not self.need_to_gen_path('important_neuron'):
            return

        self.auto_fill_model_nickname_and_model_path()
        self.raise_err_for_ungiven_arg(self.args.model_name, 'model_name')
        self.raise_err_for_ungiven_arg(
            self.args.model_nickname, 'model_nickname'
        )
        self.raise_err_for_ungiven_arg(self.args.layer, 'layer')
        self.raise_err_for_ungiven_arg(self.args.label, 'label')
        self.raise_err_for_ungiven_arg(self.args.topk_n, 'topk_n')
        
        # Directory
        d_dir_path, l_dir_path = self.gen_data_log_sub_dir('important_neuron')
        self.make_dir(d_dir_path)
        d_dir_path = os.path.join(d_dir_path, f'label={self.args.label}')
        self.make_dir(d_dir_path)

        # Data path
        d_path = os.path.join(
            d_dir_path, 
            f'{self.args.layer}-topk_n={self.args.topk_n}.json'
        )
        self.path['important_neuron'] = d_path

        # Log path
        apdx = self.gen_act_setting_str('important_neuron')
        log_path = os.path.join(
            l_dir_path, 
            'important_neuron-log-{}.txt'.format(apdx)
        )
        self.path['important_neuron-log'] = log_path
        
    def set_important_neuron_act_map_path(self):
        if not self.need_to_gen_path('important_neuron_act_map'):
            return

        self.auto_fill_model_nickname_and_model_path()
        self.raise_err_for_ungiven_arg(self.args.model_name, 'model_name')
        self.raise_err_for_ungiven_arg(
            self.args.model_nickname, 'model_nickname'
        )
        self.raise_err_for_ungiven_arg(self.args.layer, 'layer')
        self.raise_err_for_ungiven_arg(self.args.label, 'label')
        self.raise_err_for_ungiven_arg(self.args.topk_n, 'topk_n')
        
        # Directory
        d_dir_path, l_dir_path = self.gen_data_log_sub_dir(
            'important_neuron_act_map'
        )
        self.make_dir(d_dir_path)
        d_dir_path = os.path.join(d_dir_path, f'label={self.args.label}')
        self.make_dir(d_dir_path)
        d_dir_path = os.path.join(d_dir_path, self.args.layer)
        self.make_dir(d_dir_path)
        d_dir_path = os.path.join(d_dir_path, f'topk_n={self.args.topk_n}')
        self.make_dir(d_dir_path)

        # Data directory path
        self.path['important_neuron_act_map'] = d_dir_path

        # Log path
        apdx = self.gen_act_setting_str('important_neuron_act_map')
        log_path = os.path.join(
            l_dir_path, 
            'important_neuron_act_map-log-{}.txt'.format(apdx)
        )
        self.path['important_neuron_act_map-log'] = log_path
        
        
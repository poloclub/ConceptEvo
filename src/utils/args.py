import argparse

class ArgParser:
    """Parse input arguments.

    This class parses input arguments for user-given settings, such as setting
    hyperparameters or input data paths. Details of setting arguments are
    documented in `../../docs`.
    """

    """
    Constructor
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='NeuEvo')
        self.parse_basic_setting()
        self.parse_model_setting()
        self.parse_action_setting()
        self.parse_data_path_setting()
        self.parse_model_training_setting()
        self.parse_stimulus_setting()
        self.parse_embedding_setting()
        self.parse_neuron_embedding_setting()
        self.parse_image_embedding_setting()
        self.parse_proj_neuron_embedding_setting()
        self.parse_dim_reduction_setting()
        self.parse_neuron_feature_setting()
        self.parse_important_evolution_setting()


    """
    A wrapper function called in main.py
    """
    def get_args(self):
        """Returns parsed arguments"""
        return self.parser.parse_args()


    """
    Utils
    """
    def parse_bool_arg(self, arg):
        """Parse boolean argument
        
        Args:
            - arg: argument

        Returns:
            - bool_arg: booleanized arg, either True or False

        Raises:
            - ArgumentTypeError when an invalid input is given.
        """

        true_vals = ['yes', 'true', 't', 'y', '1']
        false_vals = ['no', 'false', 'f', 'n', '0']
        valid_true_vals = self.gen_valid_bool_options(true_vals)
        valid_false_vals = self.gen_valid_bool_options(false_vals)
        
        if isinstance(arg, bool):
            bool_arg = arg
        if arg in valid_true_vals:
            bool_arg = True
        elif arg in valid_false_vals:
            bool_arg = False
        else:
            log = f'Boolean value expected for {arg}. '
            log += f'Available options for {arg}=True: {valid_true_vals}. '
            log += f'Available options for {arg}=False: {valid_false_vals}. '
            raise argparse.ArgumentTypeError(log)
        return bool_arg


    def gen_valid_bool_options(self, opts):
        """Generate a list of the whole valid options for a boolean argument.

        A boolean argument can have two options, either True or False. Each 
        option can be given by either (i) lower-case letters, (ii) upper-case
        letters, or (iii) letters starting with an upper-case letter and 
        continuing with lower-case letters. For example, an argument A can have 
        value True for options `true`, `TRUE`, or `True`.

        Args: 
            - opts: a list of options for an argument.

        Returns:
            - all_valid_opts: a list of all valid options for given opts.

        Example:
            >>> opts = ['true', 'FALSE']
            >>> all_valid_opts = gen_valid_bool_options(opts)
            >>> all_valid_opts
            ['true', 'True', 'TRUE', 'false', 'False', 'FALSE']
        """

        all_valid_opts = []
        for opt in opts:
            opt_lower = opt.lower()
            opt_upper = opt.upper()
            opt_lower_with_fst_upper = opt_upper[0] + opt_lower[1:]
            all_valid_opts += [
                opt_lower, opt_upper, opt_lower_with_fst_upper
            ]
        return all_valid_opts

    
    """
    Basic settings
    """
    def parse_basic_setting(self):
        """Parse arguments for basic settings."""

        self.parser.add_argument(
            '--gpu', 
            default='0', 
            type=str,
            help='GPU number'
        )

        self.parser.add_argument(
            '--batch_size', 
            default=512, 
            type=int,
            help='batch size'
        )


    """
    Settings for models to use
    """
    def parse_model_setting(self):
        """Parse arguments for a model's name to use or its metadata."""

        self.parser.add_argument(
            '--model_name', 
            default='vgg16', 
            choices=[
                'vgg16', 
                'vgg19',
                'inception_v3', 
                'vgg16_no_dropout',
                'vgg16_pretrained', 
                'vgg19_pretrained',
                'inception_v1_pretrained',
                'inception_v3_pretrained',
                'convnext',
                'vgg16_cifar10' # remove the harcode later
            ],
            type=str,                
            help='Neural network model name'
        )

        self.parser.add_argument(
            '--model_path', 
            default='', 
            type=str,                
            help=f'Path of the neural network model'
        )

        self.parser.add_argument(
            '--model_nickname', 
            default='', 
            type=str,                
            help=f'Nickname of the neural network model'
        )

    
    """
    Settings for whether to do an action
    """
    def parse_action_setting(self):
        """Parse arguments for setting which actions to do."""

        self.parser.add_argument(
            '--train', 
            default=False, 
            type=self.parse_bool_arg,
            help='Whether to train a model'
        )

        self.parser.add_argument(
            '--test', 
            default=False, 
            type=self.parse_bool_arg,
            help='Whether to test a model'
        )

        self.parser.add_argument(
            '--stimulus', 
            default=False, 
            type=self.parse_bool_arg,
            help='Whether to find stimulus'
        )

        self.parser.add_argument(
            '--neuron_emb', 
            default=False, 
            type=self.parse_bool_arg,
            help='Whether to compute neuron embedding'
        )

        self.parser.add_argument(
            '--img_emb', 
            default=False, 
            type=self.parse_bool_arg,
            help='Whether to compute image embedding'
        )

        self.parser.add_argument(
            '--proj_neuron_emb', 
            default=False, 
            type=self.parse_bool_arg,
            help='Whether to compute projected neuron embedding'
        )

        self.parser.add_argument(
            '--dim_reduction', 
            default='None', 
            choices=['None', 'UMAP'],
            type=str,
            help='Dimensionality reduction method'
        )

        self.parser.add_argument(
            '--neuron_feature', 
            default='None', 
            choices=['None', 'example_patch'],
            type=str,
            help='Method to compute visualized features for neurons'
        )

        self.parser.add_argument(
            '--find_important_evo', 
            default=False, 
            type=self.parse_bool_arg,
            help='Whether to find concept evolution for a class prediction'
        )

        self.parser.add_argument(
            '--eval_important_evo', 
            default='None', 
            choices=[
                'None', 
                'perturbation', 
                'reverting',
                'reverting_by_layer', 
                'reverting_by_layer_by_bin'
            ],
            type=str,
            help='Option to evaluate important evolution'
        )
    

    """
    Settings for data path
    """
    def parse_data_path_setting(self):
        """Parse arguments for path of data (both input and output)."""

        self.parser.add_argument(
            '--training_data', 
            default='../../ILSVRC2012/train', 
            type=str,                
            help='Training data path'
        )

        self.parser.add_argument(
            '--data_label_path', 
            default='../../ILSVRC2012/imagenet-labels.txt', 
            type=str,                
            help='Label data path'
        )

        self.parser.add_argument(
            '--test_data', 
            default='../../ILSVRC2012/val-by-class', 
            type=str,
            help='Test data path'
        )

        self.parser.add_argument(
            '--output_dir', 
            default='../data',
            type=str,                
            help='Where to save output (trained models, log, ...)'
        )


    """
    Settings for training models
    """
    def parse_model_training_setting(self):
        """Parse arguments for training models."""

        self.parser.add_argument(
            '--lr',
            default=0.01, 
            type=float,
            help='Learning rate for training'
        )

        self.parser.add_argument(
            '--momentum',
            default=0.9, 
            type=float,
            help='Momentum for training (SGD)'
        )

        self.parser.add_argument(
            '--weight_decay',
            default=0.05, 
            type=float,
            help='weight_decay for training (AdamW)'
        )

        self.parser.add_argument(
            '--num_epochs', 
            default=300, 
            type=int,
            help='Number of epochs'
        )

        self.parser.add_argument(
            '--topk', 
            default=5, 
            type=int,
            help='k for evaluating top-k accuracy'
        )


    """
    Settings for finding stimulus
    """
    def parse_stimulus_setting(self):
        """Parse arguments for finding stimulus."""

        self.parser.add_argument(
            '--topk_s', 
            default=10, 
            type=int,
            help='k for top-k most stimulating inputs for a neuron'
        )


    """
    Settings for embedding for general
    """
    def parse_embedding_setting(self):
        """Parse arguments for embedding in general."""

        self.parser.add_argument(
            '--dim', 
            default=30, 
            type=int,
            help='Embedding dimension'
        )


    """
    Settings for neuron embedding
    """
    def parse_neuron_embedding_setting(self):
        """Parse arguments for neuron embedding."""

        self.parser.add_argument(
            '--lr_emb', 
            default=0.05, 
            type=float,
            help='Learning rate for neuron embedding'
        )

        self.parser.add_argument(
            '--num_emb_epochs', 
            default=10000, 
            type=int,
            help='Number of epochs for neuron embedding'
        )

        self.parser.add_argument(
            '--num_emb_negs', 
            default=3, 
            type=int,
            help='Number of negative sampling for neuron embedding'
        )


    """
    Settings for image embedding
    """
    def parse_image_embedding_setting(self):
        """Parse arguments for image embedding."""

        self.parser.add_argument(
            '--img_emb_set_dir', 
            default='', 
            type=str,
            help='Name of a directory for a set of image embeddings'
        )

        self.parser.add_argument(
            '--lr_img_emb', 
            default=10, 
            type=float,
            help='Learning rate for image embedding'
        )

        self.parser.add_argument(
            '--thr_img_emb', 
            default=0.05, 
            type=float,
            help='Threshold for convergence in image embedding'
        )

        self.parser.add_argument(
            '--max_iter_img_emb', 
            default=10000, 
            type=int,
            help='The number of maximum iteration for image embedding'
        )

        self.parser.add_argument(
            '--sample_rate_img_emb', 
            default=0.01, 
            type=float,
            help='Sampling rate for image embedding (It is not used)'
        )

        self.parser.add_argument(
            '--k', 
            default=10, 
            type=int,
            help='The number of stimulating images used in image embedding.'
        )


    """
    Settings for approximate projected embedding
    """
    def parse_proj_neuron_embedding_setting(self):
        """Parse arguments for approximate projected embedding."""

        self.parser.add_argument(
            '--img_emb_path', 
            default='', 
            type=str,
            help='Path of img embedding file'
        )

        self.parser.add_argument(
            '--emb_store_dirname', 
            default='', 
            type=str,
            help='Directory name to store the embedding'
        )

        self.parser.add_argument(
            '--model_for_emb_space', 
            default='base', 
            choices=['all', 'base'],
            type=str,
            help='Models to generate the embedding space'
        )

    
    """
    Settings for dimensionality reduction of embeddings
    """
    def parse_dim_reduction_setting(self):
        """Parse arguments for dimensionality reduction."""

        self.parser.add_argument(
            '--emb_set_dir', 
            default='', 
            type=str,
            help='Name of directory for a set of embeddings of multiple models'
        )

        self.parser.add_argument(
            '--reducer_path', 
            default='', 
            type=str,
            help='Path of reducer'
        )


    """
    Settings for neurons' feature visualization
    """
    def parse_neuron_feature_setting(self):
        """Parse arguments for neurons' feature visualization."""

        self.parser.add_argument(
            '--num_features', 
            default=15, 
            type=int,
            help='Number of features for each neuron'
        )

        self.parser.add_argument(
            '--ex_patch_size_ratio', 
            default=0.3, 
            type=float,
            help='Ratio of the size of example patches to input size'
        )


    """
    Settings for finding important concept evolution
    """
    def parse_important_evolution_setting(self):
        """Parse arguments for finding important concept evolution."""

        self.parser.add_argument(
            '--label', 
            default=1, 
            type=int,
            help='Class label'
        )

        self.parser.add_argument(
            '--from_model_nickname', 
            default='', 
            type=str,
            help='Nickname of model before evolution'
        )

        self.parser.add_argument(
            '--from_model_path', 
            default='', 
            type=str,
            help='Path of model before evolution'
        )

        self.parser.add_argument(
            '--to_model_nickname', 
            default='', 
            type=str,
            help='Nickname of model after evolution'
        )

        self.parser.add_argument(
            '--to_model_path', 
            default='', 
            type=str,
            help='Path of model after evolution'
        )

        self.parser.add_argument(
            '--eps', 
            default=0.3, 
            type=float,
            help='Perturbation strength'
        )

        self.parser.add_argument(
            '--eval_sample_ratio', 
            default=0.3, 
            type=float,
            help='Ratio of neurons to be evaluated for given layer'
        )

        self.parser.add_argument(
            '--find_num_sample_imgs', 
            default=250, 
            type=int,
            help='Number of sampled images to find important evolution'
        )

        self.parser.add_argument(
            '--idx', 
            default=0, 
            type=int,
            help='Index of run'
        )
        
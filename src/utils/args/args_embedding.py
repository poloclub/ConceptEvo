import argparse
from utils.args.args_util import parse_bool_arg

class EmbeddingArgs:
    def __init__(self, parser):
        self.parser = parser
        self.create_image_sampling_parser()
        self.create_stimulus_parser()
        self.create_neuron_embedding_parser()
        self.create_image_embedding_parser()

    def create_image_sampling_parser(self):
        # Whether to do
        self.parser.add_argument(
            '--sample_images', 
            default=False, 
            type=parse_bool_arg,
            help='Whether to sample images'
        )

        # Hyperparameter
        self.parser.add_argument(
            '--image_sampling_ratio', 
            default=0.1, 
            type=float,
            help='Ratio of sampling images'
        )

        # Data path
        self.parser.add_argument(
            '--input_image_path', 
            default='../../ILSVRC2012/train', 
            type=str,
            help='Directory path of original input images'
        )

        self.parser.add_argument(
            '--output_image_path', 
            default='', 
            type=str,
            help='Directory path to save the sampled images'
        )

    def create_stimulus_parser(self):
        # Whether to do
        self.parser.add_argument(
            '--stimulus', 
            default=False, 
            type=parse_bool_arg,
            help='Whether to compute stimulus'
        )

        # Hyperparameter
        self.parser.add_argument(
            '--topk_s', 
            default=20, 
            type=int,
            help='The number of most stimulating inputs \
                to consider for a neuron'
        )
        
        # Data path
        self.parser.add_argument(
            '--stimulus_sub_dir_name', 
            default='', 
            type=str,
            help='Sub-directory name for output'
        )

        self.parser.add_argument(
            '--stimulus_image_path', 
            default='', 
            type=str,
            help='Directory path of input images \
                for computing stimulus'
        )

    def create_neuron_embedding_parser(self):
        # Whether to do
        self.parser.add_argument(
            '--neuron_embedding', 
            default=False, 
            type=parse_bool_arg,
            help='Whether to compute neuron embedding'
        )

        # Hyperparameter
        self.parser.add_argument(
            '--dim', 
            default=10, 
            type=int,
            help='Embedding dimension'
        )

        self.parser.add_argument(
            '--lr_emb', 
            default=0.05, 
            type=float,
            help='Learning rate for neuron embedding'
        )

        self.parser.add_argument(
            '--topk_n', 
            default=20, 
            type=int,
            help='The number of most stimulating inputs \
                to consider for a neuron in neuron embedding'
        )

        self.parser.add_argument(
            '--num_emb_epochs', 
            default=20, 
            type=int,
            help='Number of epochs for neuron embedding'
        )

        self.parser.add_argument(
            '--num_emb_negs', 
            default=3,
            type=int,
            help='Number of negative sampling for neuron embedding'
        )

        # Data path
        self.parser.add_argument(
            '--neuron_embedding_sub_dir_name', 
            default='', 
            type=str,
            help='Sub-directory name for output'
        )

    def create_image_embedding_parser(self):
        # Whether to do
        self.parser.add_argument(
            '--image_embedding', 
            default=False, 
            type=parse_bool_arg,
            help='Whether to compute image embedding'
        )

        # Hyperparameter
        self.parser.add_argument(
            '--lr_img_emb', 
            default=0.05, 
            type=float,
            help='Learning rate for neuron embedding'
        )

        self.parser.add_argument(
            '--thr_img_emb', 
            default=0.05, 
            type=float,
            help='Threshold for convergence in image embedding'
        )

        self.parser.add_argument(
            '--num_img_emb_epochs', 
            default=10000, 
            type=int,
            help='Maximum epoch for image embedding'
        )

        # Data path
        self.parser.add_argument(
            '--image_embedding_sub_dir_name', 
            default='', 
            type=str,
            help='Sub-directory name for output'
        )
        
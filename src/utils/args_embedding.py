import argparse
from utils.args_util import parse_bool_arg

class EmbeddingArgs:
    def __init__(self, parser):
        self.parser = parser
        self.create_image_sampling_parser()
        self.create_stimulus_parser()

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
            default='../../ILSVRC2012/train_0.1', 
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
            default=10, 
            type=int,
            help='The number of most stimulating inputs to consider \
                for a neuron'
        )
        
        # Data path
        self.parser.add_argument(
            '--stimulus_image_path', 
            default='../../ILSVRC2012/train_0.1', 
            type=str,
            help='Directory path of input images for computing stimulus'
        )
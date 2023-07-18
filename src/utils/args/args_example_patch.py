import argparse
from utils.args.args_util import parse_bool_arg

class ExamplePatchArgs:
    def __init__(self, parser):
        self.parser = parser
        self.create_parser()

    def create_parser(self):
        # Whether to compute example patches
        self.parser.add_argument(
            '--example_patch', 
            default=False, 
            type=parse_bool_arg,
            help='Method to compute example patches of neurons'
        )

        # Hyperparameter
        self.parser.add_argument(
            '--ex_patch_size_ratio', 
            default=0.3, 
            type=float,
            help='The ratio of the size of example patches \
                to the input size'
        )

        self.parser.add_argument(
            '--topk_e', 
            default=20, 
            type=int,
            help='The number of example patches for a neuron'
        )

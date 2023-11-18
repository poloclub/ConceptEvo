import argparse
from utils.args.args_util import parse_bool_arg

class ImportantEvoArgs:
    def __init__(self, parser):
        self.parser = parser
        self.create_parser()

    def create_parser(self):
        # Whether to do
        self.parser.add_argument(
            '--find_important_evo', 
            default=False, 
            type=parse_bool_arg,
            help='Whether to find important evolution for a class prediction'
        )
        self.parser.add_argument(
            '--eval_important_evo', 
            default=False,
            type=self.parse_bool_arg,
            help='Option to evaluate important evolution'
        )

        # From model, representing the state of the model 
        # before undergoing the targeted evolutionary changes
        self.parser.add_argument(
            '--from_model_nickname', 
            default='', 
            type=str,
            help='Nickname of model before the target evolution'
        )
        self.parser.add_argument(
            '--from_model_path', 
            default='', 
            type=str,
            help='Path of model before the target evolution'
        )

        # To model, representing the state of the model 
        # after undergoing the targeted evolutionary changes
        self.parser.add_argument(
            '--to_model_nickname', 
            default='', 
            type=str,
            help='Nickname of model after the target evolution'
        )
        self.parser.add_argument(
            '--to_model_path', 
            default='', 
            type=str,
            help='Path of model after the target evolution'
        )

        # Hyperparameter
        self.parser.add_argument(
            '--label_img_idx_path', 
            default='', 
            type=str,
            help='The path of file specifying the range of image indices for each class label'
        )
        self.parser.add_argument(
            '--label', 
            default=1, 
            type=int,
            help='Class label to analyze for the evolution importance'
        )
        self.parser.add_argument(
            '--num_sampled_imgs', 
            default=250, 
            type=int,
            help='The number of images to sample for finding important evolutions'
        )
        self.parser.add_argument(
            '--num_bins', 
            default=4, 
            type=int,
            help='The number of bins to split neurons'
        )

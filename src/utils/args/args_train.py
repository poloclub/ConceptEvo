import argparse
from utils.args.args_util import parse_bool_arg

class TrainArgs:
    def __init__(self, parser):
        self.parser = parser
        self.create_action_parser()
        self.create_data_path_parser()
        self.create_hyperparameter_parser()

    def create_action_parser(self):
        self.parser.add_argument(
            '--train', 
            default=False, 
            type=parse_bool_arg,
            help='Whether to train a model'
        )

    def create_data_path_parser(self):
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

    def create_hyperparameter_parser(self):
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
            help='Momentum for training'
        )

        self.parser.add_argument(
            '--weight_decay',
            default=0.05, 
            type=float,
            help='weight_decay for training'
        )

        self.parser.add_argument(
            '--learning_eps',
            default=0.05, 
            type=float,
            help='eps (can be used in RMSProp)'
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
            help='The number of predicted outcomes \
                with the highest probabilities to consider \
                when evaluating top-k accuracy'
        )
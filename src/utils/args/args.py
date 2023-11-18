import argparse

from utils.args.args_train import TrainArgs
from utils.args.args_test import TestArgs
from utils.args.args_embedding import EmbeddingArgs
from utils.args.args_important_evo import ImportantEvoArgs
from utils.args.args_example_patch import ExamplePatchArgs

class ArgParser:
    """Parse input arguments.

    This class efficiently handles the parsing of input arguments 
    provided by the user, enabling them to specify various settings, 
    such as hyperparameters or input data paths.
    """

    """
    Constructor
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='ConceptEvo')
        
        self.parse_general_setting()
        self.parse_model_setting()

        self.train_args = TrainArgs(self.parser)
        self.test_args = TestArgs(self.parser)
        self.embedding_args = EmbeddingArgs(self.parser)
        self.important_evo_args = ImportantEvoArgs(self.parser)
        self.example_patch_args = ExamplePatchArgs(self.parser)
        

    """
    A wrapper function called in main.py, which returns the parsed arguments.
    """
    def get_args(self):
        return self.parser.parse_args()

    """
    Basic settings
    """
    def parse_general_setting(self):
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

        self.parser.add_argument(
            '--output_dir', 
            default='../data',
            type=str,                
            help='Where to save output (trained models, log, ...)'
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
                'vgg16_no_dropout',
                'inception_v3', 
                'convnext',
                'resnet18',
                'resnet18_dropout',
                'resnet50',
                'resnext50_32x4d'
            ],
            type=str,                
            help='Name of the neural network model'
        )

        self.parser.add_argument(
            '--model_nickname', 
            default='', 
            type=str,                
            help='Nickname of the neural network model. \
                Include "pretrained" in the model_nickname \
                to load a pretrained model.'
        )

        self.parser.add_argument(
            '--epoch', 
            default=-1, 
            type=int,
            help='Specific epoch of the neural network model'
        )

        self.parser.add_argument(
            '--model_path', 
            default='', 
            type=str,                
            help=f'Path of the neural network model'
        )

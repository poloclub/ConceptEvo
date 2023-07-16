import argparse
from utils.args_util import parse_bool_arg

class TestArgs:
    def __init__(self, parser):
        self.parser = parser
        self.create_parser()

    def create_parser(self):
        self.parser.add_argument(
            '--test', 
            default=False, 
            type=parse_bool_arg,
            help='Whether to test a model'
        )

        self.parser.add_argument(
            '--test_by_class', 
            default=False, 
            type=parse_bool_arg,
            help='Whether to test a model by class'
        )

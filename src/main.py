"""A main module to run python codes.

It parses arguments given by the users and run the codes based on the 
argumensts. An instruction for setting arguments can be found in 
<TODO: Add a link to an instruction for setting args>.
"""

# CNN Models
# from vgg16.vgg16 import *
# from inception_v3.inception_v3 import *

# Utils
from utils.args import *
from utils.datapath import *

# Embedding
from embedding.stimulus import *
from embedding.reduce_dim import *
from embedding.image_embedding import *
from embedding.neuron_embedding import *
from embedding.proj_neuron_embedding import *

# # Concept images of neurons
# from feature.example_patch import *


def main():
    args = ArgParser().get_args()

if __name__ == '__main__':
    main()
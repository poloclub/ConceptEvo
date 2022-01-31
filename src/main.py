"""A main module to run python codes.

It parses arguments given by the users and run the codes based on the 
argumensts. An instruction for setting arguments can be found in 
<TODO: Add a link to an instruction for setting args>.
"""

# CNN Models
from model.vgg16 import *
from model.inception_v3 import *

# Utils
from utils.args import *
from utils.datapath import *

# Embedding
from embedding.stimulus import *
from embedding.reduce_dim import *
from embedding.image_embedding import *
from embedding.neuron_embedding import *
from embedding.proj_neuron_embedding import *

# Concept images of neurons
from feature.example_patch import *


def main():
    # Parse input arguments
    args = ArgParser().get_args()

    # Generate data directories
    data_path = DataPath(args)
    data_path.gen_data_dirs()

    # Load model
    model = load_model(args, data_path)

    # Find stimulus
    if args.stimulus:
        compute_stimulus(args, data_path, model)

    # Compute neuron embedding
    if args.neuron_emb:
        compute_neuron_embedding(args, data_path, model)

    # Compute image embedding
    if args.img_emb:
        compute_image_embedding(args, data_path, model)

    # Compute projected neuron embedding
    if args.proj_neuron_emb:
        compute_proj_neuron_emb(args, data_path, model)

    # Project neuron embedding to 2D space
    if args.dim_reduction != 'None':
        reduce_embedding_dim(args, data_path)

    # Compute visualized features for neurons
    if args.neuron_feature != 'None':
        compute_neuron_feature(args, data_path, model)


def load_model(args, data_path):
    if args.model_name == 'vgg16':
        model = Vgg16(args, data_path)
    elif args.model_name == 'inception_v3':
        model = InceptionV3(args, data_path)
    elif args.model_name == 'vgg16_pretrained':
        model = Vgg16(args, data_path, pretrained=True)
    elif args.model_name == 'inception_v3_pretrained':
        model = InceptionV3(args, data_path, pretrained=True)
    else:
        raise ValueError(f'Error: unkonwn model {args.model_name}')

    model.init_basic_setting()
    model.init_model()
    model.init_training_setting()
    return model
    

def train_model(model):
    model.train_model()


def compute_stimulus(args, data_path, model):
    stimulus = Stimulus(args, data_path, model)
    stimulus.compute_stimulus()


def compute_neuron_embedding(args, data_path, model):
    neuron_emb = Emb(args, data_path, model)
    neuron_emb.compute_neuron_embedding()


def compute_image_embedding(args, data_path, model):
    img_emb = ImageEmb(args, data_path, model)
    img_emb.compute_img_embedding()


def compute_proj_neuron_emb(args, data_path, model):
    proj_neuron_emb = ProjNeuronEmb(args, data_path, model)
    proj_neuron_emb.compute_projected_neuron_emb()


def reduce_embedding_dim(args, data_path):
    reducer = Reducer(args, data_path)
    reducer.reduce_dim()


def compute_neuron_feature(args, data_path, model):
    if args.neuron_feature == 'example_patch':
        exPatch = ExamplePatch(args, data_path, model)
        exPatch.compute_neuron_feature()



if __name__ == '__main__':
    main()
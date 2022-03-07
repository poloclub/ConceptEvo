"""A main module to run python codes.

It parses arguments given by the users and run the codes based on the 
argumensts. An instruction for setting arguments can be found in 
<TODO: Add a link to an instruction for setting args>.
"""

# CNN Models
from model.vgg16 import *
from model.vgg19 import *
from model.inception_v1 import *
from model.inception_v3 import *
from model.vgg16_no_dropout import *

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

# Find and evaluate concept evolution for class predictions
from importantevo.important_evo import *
from importantevo.eval_important_evo import *


def main():
    # Parse input arguments
    args = ArgParser().get_args()

    # Generate data directories
    data_path = DataPath(args)
    data_path.gen_data_dirs()

    # Load model
    model = load_model(args, data_path)

    # Train model
    if args.train:
        train_model(model)

    # Test model
    if args.test:
        test_model(model)

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

    # Find concept evolution for class predictions
    if args.find_important_evo:
        find_important_evolution(args, data_path)

    # Evaluate important concept evolution for class predictions
    if args.eval_important_evo != 'None':
        eval_important_evolution(args, data_path)


def load_model(args, data_path):

    when_to_skip_loading_model = [
        args.dim_reduction != 'None',
        args.find_important_evo,
        args.eval_important_evo != 'None'
    ]

    if True in when_to_skip_loading_model:
        return

    if args.model_name == 'vgg16':
        model = Vgg16(args, data_path)
    elif args.model_name == 'vgg19':
        model = Vgg19(args, data_path)
    elif args.model_name == 'inception_v3':
        model = InceptionV3(args, data_path)
    elif args.model_name == 'vgg16_pretrained':
        model = Vgg16(args, data_path, pretrained=True)
    elif args.model_name == 'vgg19_pretrained':
        model = Vgg19(args, data_path, pretrained=True)
    elif args.model_name == 'vgg16_no_dropout':
        model = Vgg16NoDropout(args, data_path)
    elif args.model_name == 'inception_v3_pretrained':
        model = InceptionV3(args, data_path, pretrained=True)
    elif args.model_name == 'inception_v1_pretrained':
        model = InceptionV1(args, data_path, pretrained=True)
    else:
        raise ValueError(f'Error: unkonwn model {args.model_name}')

    model.init_basic_setting()
    model.init_model()
    model.init_training_setting()
    return model


def load_models(args, data_path):
    when_to_load_model = [
        args.find_important_evo,
        args.eval_important_evo != 'None'
    ]
    if True not in when_to_load_model:
        return

    if args.model_name == 'vgg16':
        from_model = Vgg16(args, data_path, from_to='from')
        to_model = Vgg16(args, data_path, from_to='to')
    elif args.model_name == 'inception_v3':
        from_model = InceptionV3(args, data_path, from_to='from')
        to_model = InceptionV3(args, data_path, from_to='to')
    else:
        raise ValueError(f'Error: unkonwn model {args.model_name}')
    
    from_model.init_basic_setting()
    from_model.init_model()
    from_model.init_training_setting()

    to_model.init_basic_setting()
    to_model.init_model()
    to_model.init_training_setting()

    return from_model, to_model


def train_model(model):
    model.train_model()


def test_model(model):
    model.test_model()


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
        ex_patch = ExamplePatch(args, data_path, model)
        ex_patch.compute_neuron_feature()


def find_important_evolution(args, data_path):
    from_model, to_model = load_models(args, data_path)
    imp_evo = ImportantEvo(args, data_path, from_model, to_model)
    imp_evo.find_important_evolution()
    

def eval_important_evolution(args, data_path):
    from_model, to_model = load_models(args, data_path)
    eval_evo = EvalImportantEvo(args, data_path, from_model, to_model)
    eval_evo.eval_important_evolution()



if __name__ == '__main__':
    main()
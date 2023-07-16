"""A main module to run codes"""

# Embedding
from embedding.sample_images import *
from embedding.image_embedding import *
from embedding.neuron_embedding import *
from embedding.proj_neuron_embedding import *
from embedding.reduce_dim import *
from embedding.stimulus import *

# Concept images of neurons
from feature.example_patch import *
from feature.stimulus_act_map import *

# Find and evaluate concept evolution for class predictions
from importantevo.eval_important_evo import *
from importantevo.find_important_evo import *

# Find important neurons for class predictions
from importantneuron.important_neuron import *
from importantneuron.important_neuron_act_map import *

# CNN Models
from model.convnext import *
from model.inception_v3 import *
from model.vgg16 import *
from model.vgg16_no_dropout import *
from model.vgg19 import *
from model.resnet18 import *
from model.resnet50 import *

# Utils
from utils.args import *
from utils.datapath import *

def main():
    # Parse input arguments
    args = ArgParser().get_args()

    # Generate data directories
    data_path = DataPath(args)

    # Load model
    model = load_model(args, data_path)
    if model is not None:
        model.save_layer_info()

    # Train model
    if args.train:
        train_model(model)

    # Test model
    if args.test:
        test_model(model)

    # Test model by class
    if args.test_by_class:
        test_model_by_class(model)

    # Generate example patches of neurons
    if args.example_patch:
        generate_example_patch(args, data_path, model)

    # Sample images
    if args.sample_images:
        sample_images(args)

    # Find stimulus
    if args.stimulus:
        compute_stimulus(args, data_path, model)




    # Compute neuron embedding
    if args.neuron_embedding:
        compute_neuron_embedding(args, data_path, model)

    # Compute image embedding
    if args.img_emb:
        compute_image_embedding(args, data_path, model)

    if args.img_pairs:
        compute_img_pairs(args, data_path, model)

    if args.img_emb_co_act:
        compute_image_embedding_co_act(args, data_path, model)

    # Compute projected neuron embedding
    if args.proj_neuron_emb:
        compute_proj_neuron_emb(args, data_path, model)

    # Project neuron embedding to 2D space
    if args.dim_reduction != 'None':
        reduce_embedding_dim(args, data_path)

    
    
    if args.act_map:
        compute_act_maps(args, data_path, model)

    # Find concept evolution for class predictions
    if args.find_important_evo:
        find_important_evolution(args, data_path)

    # Evaluate important concept evolution for class predictions
    if args.eval_important_evo:
        eval_important_evolution(args, data_path)

    # Find important neurons for class predictions
    if args.important_neuron:
        find_important_neuron(args, data_path, model)

    # Compute activation map of important neurons for class predictions
    if args.important_neuron_act_map:
        compute_important_neuron_act_map(args, data_path, model)

def load_model(args, data_path):

    when_to_skip_loading_model = [
        args.sample_images,
        args.dim_reduction != 'None',
        args.find_important_evo,
        args.eval_important_evo,
    ]

    if True in when_to_skip_loading_model:
        return None

    pretrained = 'pretrained' in args.model_nickname

    if args.model_name == 'vgg16':
        model = Vgg16(args, data_path, pretrained=pretrained)
    elif args.model_name == 'vgg19':
        model = Vgg16(args, data_path, pretrained=pretrained)
    elif args.model_name == 'inception_v3':
        model = InceptionV3(args, data_path, pretrained=pretrained)
    elif args.model_name == 'vgg16_no_dropout':
        model = Vgg16NoDropout(args, data_path, pretrained=pretrained)
    elif args.model_name == 'convnext':
        model = ConvNeXt(args, data_path, pretrained=pretrained)
    elif args.model_name == 'resnet18':
        model = ResNet18(args, data_path, pretrained=pretrained)
    elif args.model_name == 'resnet50':
        model = ResNet50(args, data_path, pretrained=pretrained)
    else:
        raise ValueError(f'Error: unknown model given "{args.model_name}"')

    return model

def load_models(args, data_path):
    when_to_load_model = [
        args.find_important_evo,
        args.eval_important_evo
    ]
    if True not in when_to_load_model:
        return

    if args.model_name == 'vgg16':
        from_model = Vgg16(args, data_path, from_to='from')
        to_model = Vgg16(args, data_path, from_to='to')
    elif args.model_name == 'inception_v3':
        from_model = InceptionV3(args, data_path, from_to='from')
        to_model = InceptionV3(args, data_path, from_to='to')
    elif args.model_name == 'convnext':
        from_model = ConvNeXt(args, data_path, from_to='from')
        to_model = ConvNeXt(args, data_path, from_to='to')
    else:
        raise ValueError(f'Error: unknown model {args.model_name}')
    
    return from_model, to_model

def train_model(model):
    model.train_model()

def test_model(model):
    model.test_model(write_log=True, test_on='training')
    model.test_model(write_log=True, test_on='test')

def test_model_by_class(model):
    model.test_model_by_class(write_log=True, test_on='training')
    model.test_model_by_class(write_log=True, test_on='test')

def generate_example_patch(args, data_path, model):
    ex_patch = ExamplePatch(args, data_path, model)
    ex_patch.generate_example_patch()

def sample_images(args):
    img_sample = SampleImages(args)
    img_sample.sample_images()



def compute_layer_act(args, data_path, model):
    layer_act = LayerAct(args, data_path, model)
    layer_act.compute_layer_act()

def compute_stimulus(args, data_path, model):
    stimulus = Stimulus(args, data_path, model)
    stimulus.compute_stimulus()

def compute_neuron_embedding(args, data_path, model):
    neuron_emb = Emb(args, data_path, model)
    neuron_emb.compute_neuron_embedding()

def compute_image_embedding(args, data_path, model):
    img_emb = ImageEmb(args, data_path, model)
    img_emb.compute_img_embedding()

def compute_img_pairs(args, data_path, model):
    img_pairs = ImagePairs(args, data_path, model)
    img_pairs.compute_img_pairs()

def compute_image_embedding_co_act(args, data_path, model):
    img_emb_layer_act = ImageEmbCoAct(args, data_path, model)
    img_emb_layer_act.compute_img_embedding()

def compute_proj_neuron_emb(args, data_path, model):
    proj_neuron_emb = ProjNeuronEmb(args, data_path, model)
    proj_neuron_emb.compute_projected_neuron_emb()

def reduce_embedding_dim(args, data_path):
    reducer = Reducer(args, data_path)
    reducer.reduce_dim()



def compute_act_maps(args, data_path, model):
    act_map = StimulusActMap(args, data_path, model)
    act_map.compute_act_map()

def find_important_evolution(args, data_path):
    from_model, to_model = load_models(args, data_path)
    find_evo = FindImportantEvo(args, data_path, from_model, to_model)
    find_evo.find_important_evolution()
    
def eval_important_evolution(args, data_path):
    from_model, to_model = load_models(args, data_path)
    eval_evo = EvalImportantEvo(args, data_path, from_model, to_model)
    eval_evo.eval_important_evolution()

def find_important_neuron(args, data_path, model):
    imp_neuron = ImportantNeuron(args, data_path, model)
    imp_neuron.compute_important_neuron()

def compute_important_neuron_act_map(args, data_path, model):
    imp_neuron_act_map = ImportantNeuronActMap(args, data_path, model)
    imp_neuron_act_map.compute_important_neuron_act_map()

if __name__ == '__main__':
    main()
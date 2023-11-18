"""A main module to run codes"""

# Neuron embedding for the base model
from embedding.stimulus import *
from embedding.neuron_embedding import *

# Image embedding
from embedding.sample_images import *
from embedding.image_embedding import *
from embedding.responsive_neurons import *
from embedding.indirect_image_embedding import *

# Approximated neuron embedding
from embedding.proj_neuron_embedding import *
from embedding.reduced_neuron_embedding import *

# Concept images of neurons
from feature.example_patch import *

# Find and evaluate concept evolution for class predictions
from importantevo.eval_important_evo import *
from importantevo.find_important_evo import *

# CNN Models
from model.vgg16 import *
from model.vgg16_no_dropout import *
from model.vgg19 import *
from model.inception_v3 import *
from model.convnext import *

# Utils
from utils.args.args import *
from utils.datapath.datapath import *

def main():
    # Parse input arguments
    args = ArgParser().get_args()

    # Generate data directories
    data_path = DataPath(args)

    # Load model
    model = load_model(args, data_path)
    if model is not None:
        model.save_layer_info()

    #######################################################
    # Train or test a DNN model
    #######################################################
    # Train model
    if args.train:
        # model.model.train()
        train_model(model)
    
    # Set evaluation mode
    if model is not None:
        model.model.eval()
        print(f'training mode: {model.model.training}')

    # Test model
    if args.test:
        test_model(model)

    # Test model by class
    if args.test_by_class:
        test_model_by_class(model)

    #######################################################
    # Neuron and image embedding
    #######################################################
    # Sample images
    if args.sample_images:
        sample_images(args)

    # Find stimulus
    if args.stimulus:
        compute_stimulus(args, data_path, model)
    
    # Find most responsive neurons
    if args.responsive_neurons:
        compute_responsive_neurons(args, data_path, model)

    # Compute neuron embedding
    if args.neuron_embedding:
        compute_neuron_embedding(args, data_path)

    # Compute image embedding
    if args.image_embedding:
        compute_image_embedding(args, data_path)

    # Compute indirect image embedding
    if args.indirect_image_embedding:
        compute_indirect_image_embedding(args, data_path)

    # Compute projected neuron embedding
    if args.proj_embedding:
        compute_proj_neuron_embedding(args, data_path)

    # Project neuron embedding to 2D space
    if args.reduced_embedding:
        compute_reduced_embedding(args, data_path)

    #######################################################
    # Generate example patches
    #######################################################    
    # Find concept evolution for class predictions
    if args.find_important_evo:
        find_important_evolution(args, data_path)

    # Evaluate important concept evolution for class predictions
    if args.eval_important_evo:
        eval_important_evolution(args, data_path)
        
    #######################################################
    # Generate example patches
    #######################################################
    # Generate example patches of neurons
    if args.example_patch:
        generate_example_patch(args, data_path, model)


def load_model(args, data_path):

    when_to_skip_loading_model = [
        args.sample_images,
        args.neuron_embedding,
        args.image_embedding,
        args.indirect_image_embedding,
        args.proj_embedding,
        args.reduced_embedding,
        args.find_important_evo,
        args.eval_important_evo,
    ]

    if True in when_to_skip_loading_model:
        return None

    pretrained = 'pretrained' in args.model_nickname

    if args.model_name == 'vgg16':
        model = Vgg16(args, data_path, pretrained=pretrained)
    elif args.model_name == 'vgg19':
        model = Vgg19(args, data_path, pretrained=pretrained)
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
    model.test_model(write_log=True, train_or_test='train')
    model.test_model(write_log=True, train_or_test='test')

def test_model_by_class(model):
    model.test_model_by_class(write_log=True, train_or_test='train')
    model.test_model_by_class(write_log=True, train_or_test='test')

def generate_example_patch(args, data_path, model):
    ex_patch = ExamplePatch(args, data_path, model)
    ex_patch.generate_example_patch()

def sample_images(args):
    img_sample = SampleImages(args)
    img_sample.sample_images()

def compute_stimulus(args, data_path, model):
    stimulus = Stimulus(args, data_path, model)
    stimulus.compute_stimulus()

def compute_responsive_neurons(args, data_path, model):
    responsive_neurons = ResponsiveNeurons(args, data_path, model)
    responsive_neurons.compute_responsive_neurons()

def compute_neuron_embedding(args, data_path):
    neuron_emb = Emb(args, data_path)
    neuron_emb.compute_neuron_embedding()

def compute_image_embedding(args, data_path):
    img_emb = ImageEmb(args, data_path)
    img_emb.compute_img_embedding()

def compute_indirect_image_embedding(args, data_path):
    indirect_img_emb = IndirectImageEmb(args, data_path)
    indirect_img_emb.compute_indirect_img_embedding()

def compute_proj_neuron_embedding(args, data_path):
    proj_neuron_emb = ProjNeuronEmb(args, data_path)
    proj_neuron_emb.compute_projected_neuron_embedding()

def compute_reduced_embedding(args, data_path):
    reducer = ReducedNeuronEmb(args, data_path)
    reducer.compute_reduced_embedding()

def find_important_evolution(args, data_path):
    from_model, to_model = load_models(args, data_path)
    find_evo = FindImportantEvo(args, data_path, from_model, to_model)
    find_evo.find_important_evolution()
    
def eval_important_evolution(args, data_path):
    from_model, to_model = load_models(args, data_path)
    eval_evo = EvalImportantEvo(args, data_path, from_model, to_model)
    eval_evo.eval_important_evolution()

if __name__ == '__main__':
    main()
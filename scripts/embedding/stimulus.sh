###############################################################################
# stimulus.sh
# 
# Find the most stimulating images for each neuron in the base model.
# Run this script at `../../src` where `main.py` exists.
# 
# The result will be saved at
# ../../data
#     └── stimulus
#           └── <stimulus_sub_dir_name>
#                 ├── data
#                 │     └── stimulus_<model_nickname_in_file*>.json
#                 └── log
#                       ├── setting.txt
#                       └── stimulus_log_<model_nickname_in_file*>.txt
#
# <model_nickname_in_file*> depends on the model's training status:
# - For models trained up to a specific epoch, <model_nickname>_<epoch>
# - For pretrained models, <model_nickname>
###############################################################################

###############################################################################
# Hyperparameters: Modify them below as you want

# 1. GPU selection
# Specify the GPU device index you want to use (usually 0, 1, etc)
gpu=0

# 2. Batch size
# Set the batch size
batch_size=512

# 3. Model selection
# Choose the name of the model architecture from the available options
# Available models: [
#    'vgg16', 
#    'vgg19', 
#    'inception_v3', 
#    'vgg16_no_dropout', 
#    'convnext', 
#    'resnet18', 
#    'resnet50']
model_name=vgg16

# 4. Model nickname
# Assign a unique nickname to the model for easy identification
# among other models. If you intend to use a pre-trained model, consider
# using the format <model_name>_pretrained as the model nickname.
model_nickname=vgg16_0.01
# model_nickname=vgg19_pretrained

# 5. Epoch
# Specify the epoch of the model you want to use
epoch=207
# epoch=-1 # if you are using a pretrained model

# 6. Model path
# You have two options to specify the model path:
# (1) Directly provide the file path to the model
# (2) Indirectly set the epoch, allowing automatic model path 
#   determination based on the model nickname and epoch.
# 
# If you choose (2) the automatic search method by providing only 
# model nickname and epoch, ensure that the model nickname here matches 
# the one used during model training or found in the directory where 
# trained models are saved:
#   ../../data/model/<model_nickname>/data/model-<epoch>.pth
# 
# If you are using a pretrained model, you don't need to provide any of 
# these. Simply set model_nickname to <model_name>_pretrained.
#
# (1) Directly provide the file path to the model
model_path=../data/model/vgg16_0.01/data/model-207.pth
#
# (2) Automatic model path determination
# model_path=None
#
# If you are usign a pretrained model:
# model_path=None

# 6. The number of images used as stimulus for each neuron
topk_s=20

# 7. The file path for image pools used as stimulus sources, which may 
# include the subset of images selected by the './sample_images.sh' script.
stimulus_image_path=../../ILSVRC2012/train_0.1

# 8. Name of the output directory
stimulus_sub_dir_name=train_0.1

###############################################################################
python main.py \
    --stimulus True \
    --gpu $gpu \
    --batch_size $batch_size \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --epoch $epoch \
    --model_path $model_path \
    --topk_s $topk_s \
    --stimulus_image_path $stimulus_image_path \
    --stimulus_sub_dir_name $stimulus_sub_dir_name
###############################################################################

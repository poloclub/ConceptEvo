###############################################################################
# embedding_img_pairs.sh
# 
# Create similar image pairs.
# Run this script at `../src` where `main.py` exists.
###############################################################################

###############################################################################
# File structure:
# 
# ../data
#     └── img_pairs
#             └── <basemodel_nickname>
#                   └── data
#                   │     └── <layer>
#                   │          └── img_pairs-<apdx>.json 
#                   └── log
# <apdx>: 
# ```
# '-'.join([
#     layer=<layer>,
#     num_epochs_co_act=<num_epochs_co_act>,
#     k=<k>
# ])
# ```
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_nickname=?
# model_path=?
# layer=?
# num_epochs_co_act
# k=?
# 
# For example:
gpu=0

model_nickname=vgg19_pretrained
layer=Sequential_0_Conv2d_34
num_epochs_co_act=10
k=10
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --img_pairs T \
    --model_nickname $model_nickname \
    --layer $layer \
    --num_epochs_co_act $num_epochs_co_act \
    --k $k
###############################################################################

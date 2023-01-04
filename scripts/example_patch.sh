###############################################################################
# example_patch.sh
# 
# Create example patches for each neuron in a model.
# Run this script at `../src` where `main.py` exists.
# 
# The result will be saved at
# `../data/neuron_feature/<model_nickname>/data/DIR_NAME*/`, where
# DIR_NAME* is
# '-'.join(
#   'topk_s=<topk_s>',
#   'ex_patch_size_ratio=<ex_patch_size_ratio>',
# )
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# model_path=?
# topk_s=?
# ex_patch_size_ratio=?
# batch_size=?
# 
# For example:
gpu=0
model_name=convnext
lr=0.004
epoch=96
model_nickname=convnext-"$lr"-"$epoch"
model_path=../data/model/"$model_name"_"$lr"/data/model-"$epoch".pth
topk_s=20
ex_patch_size_ratio=0.3
batch_size=512
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --neuron_feature example_patch \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --topk_s $topk_s \
    --ex_patch_size_ratio $ex_patch_size_ratio \
    --batch_size $batch_size 
###############################################################################

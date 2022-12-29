###############################################################################
# stimulus.sh
# 
# Find the most stimulating images for each neuron in the base model.
# Run this script at `../src` where `main.py` exists.
# 
# The result will be saved at
# `../data/stimulus/<model_nickname>/data/stimulus-topk_s=<topk_s>.json``.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# model_path=?
# batch_size=?
# topk_s=?
# 
# For example:
gpu=0
model_name=vgg16
model_nickname=$model_name-0.01-207
model_path=../data/model/$model_name-512-0.01-0.9/data/model-207.pth
batch_size=512
topk_s=20
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --stimulus T \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --batch_size $batch_size \
    --topk_s $topk_s
###############################################################################

###############################################################################
# find_important_evo.sh
# 
# Find the most important evolutions.
# Run this script at `../src` where `main.py` exists.
# 
# The result will be saved at
# `../data/find_important_evo/<model_nickname>/data/`.
# 
# Two files will be saved:
#  - 1) `sensitivity-<apdx*>.json`:
#       Sensitivity of each neuron's evolution to given sampled images
#       
#  - 2) `score-<apdx*>.json`:
#       Score of each neurons' evolution for the label prediction
# 
#  where apdx* is:
#  '-'.join([
#      label=<label>,
#      num_sampled_imgs=<num_sampled_imgs>
#      from=<from_model_nickname>
#      to=<to_model_nickname>
#      idx=<idx>
#  ])      
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# label=?
# batch_size=?
# model_name=?
# model_nickname=?
# from_model_nickname=?
# from_model_path=?
# to_model_nickname=?
# to_model_path=?
# num_sampled_imgs=?
# idx=?
# 
# For example:
gpu=0
label=457
batch_size=50
model_name=convnext
model_nickname=convnext_0.004
from_epoch=3
from_model_nickname="$model_nickname"_"$from_epoch"
from_model_path=../data/model/"$model_nickname"/data/model-"$from_epoch".pth 
to_epoch=96
to_model_nickname="$model_nickname"_"$to_epoch"
to_model_path=../data/model/"$model_nickname"/data/model-"$to_epoch".pth 
label_img_idx_path=../data/ILSVRC2012_label_img_idx.json
num_sampled_imgs=250
idx=0
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --find_important_evo T \
    --label $label \
    --batch_size $batch_size \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --from_model_nickname $from_model_nickname \
    --from_model_path $from_model_path \
    --to_model_nickname $to_model_nickname \
    --to_model_path $to_model_path \
    --label_img_idx_path $label_img_idx_path \
    --num_sampled_imgs $num_sampled_imgs \
    --idx $idx
###############################################################################
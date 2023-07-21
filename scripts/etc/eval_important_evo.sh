###############################################################################
# eval_important_evo.sh
# 
# Evaluate the important evolutions
# Run this script at `../src` where `main.py` exists.
# 
# The result will be saved at
# `../data/eval_important_evo/<model_nickname>/data/eval_evo-<apdx>.json`,
#
# where apdx is
# '-'.join([
#     from=<from_model_nickname>,
#     to=<to_model_nickname>,
#     label=<label>,
#     num_bins=<num_bins>,
#     num_sampled_imgs=<num_sampled_imgs>,
#     idx=<idx>
# ])
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# batch_size=?
# label=?
# num_bins=?
# model_name=?
# model_nickname=?
# from_model_nickname=?
# from_model_path=?
# to_model_nickname=?
# to_model_path=?
# label_img_idx_path=?
# num_sampled_imgs=?
# idx=?
# 
# For example:
gpu=0
batch_size=128
label=457
num_bins=4
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
    --eval_important_evo T \
    --batch_size $batch_size \
    --label $label \
    --num_bins $num_bins \
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

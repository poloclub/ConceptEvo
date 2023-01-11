###############################################################################
# act_map.sh
# 
# Create activation maps for each neuron for most stimulating images.
# Run this script at `../src` where `main.py` exists.
# 
# The result will be saved at
# `../data/act_map/<model_nickname>/data/topk_s=<topk_s>/`.
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
lr=0.004
epoch=96
model_name=convnext
model_nickname=convnext-"$lr"-"$epoch"
model_path=../data/model/"$model_name"_"$lr"/data/model-"$epoch".pth
topk_s=20
batch_size=512
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --act_map True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --topk_s $topk_s \
    --batch_size $batch_size 
###############################################################################

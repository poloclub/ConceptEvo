###############################################################################
# important_neuron.sh
# 
# Find important neurons for a class prediction.
# Run this script at `../src` where `main.py` exists.
# 
# The results will be saved at
# `../data/important_neuron/<model_nickname>/data/label=<label>/<layer>/`.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# model_path=?
# topk_s=?
# batch_size=?
# label=?
# layer=?
# 
# For example:
gpu=7
model_name=convnext_pretrained
model_nickname=convnext_pretrained
model_path=DO_NOT_NEED_CURRENTLY
topk_s=20
batch_size=512
label=457
layer=Conv2d_144
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --important_neuron True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --topk_s $topk_s \
    --batch_size $batch_size \
    --label $label \
    --layer $layer
###############################################################################

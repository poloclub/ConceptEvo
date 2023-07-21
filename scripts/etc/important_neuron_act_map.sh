###############################################################################
# important_neuron_act_map.sh
# 
# Compute activation maps of important neurons for a class prediction.
# Run this script at `../src` where `main.py` exists.
# 
# The results will be saved at
# ```
# ../data/important_neuron_act_map/<model_nickname>/data/label=<label>/ \
# layer=<layer>/topk_n=<topk_n>/`.
# ```
# 
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# model_path=?
# topk_n=?
# batch_size=?
# label=?
# layer=?
# 
# For example:
gpu=0
model_name=convnext_pretrained
model_nickname=convnext_pretrained
model_path=DO_NOT_NEED_CURRENTLY
topk_n=10
batch_size=512
label=457
layer=Conv2d_0
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --important_neuron_act_map True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --topk_n $topk_n \
    --batch_size $batch_size \
    --label $label \
    --layer $layer
###############################################################################

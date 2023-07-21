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
#                 │     └── stimulus_<model_nickname>_<epoch>.json
#                 └── log
#                       ├── setting.txt
#                       └── stimulus_log_<model_nickname>_<epoch>.txt
#
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# epoch=?
# model_path=? (optional)
# batch_size=?
# topk_s=?
# stimulus_image_path=?
# stimulus_sub_dir_name=?
# 
# For example:
gpu=0
model_name=vgg16
lr=0.01
model_nickname="$model_name"_"$lr"
epoch=207
batch_size=512
topk_s=20
stimulus_image_path=../../ILSVRC2012/train_0.1
stimulus_sub_dir_name=train_0.1
#
# For example, the setting for a pretrained model would be:
# gpu=0
# model_name=vgg19
# model_nickname="$model_name"_pretrained
# epoch=-1
# model_path=None
# batch_size=512
# topk_s=20
# stimulus_image_path=../../ILSVRC2012/train_0.1
# stimulus_sub_dir_name=train_0.1
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --stimulus True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --epoch $epoch \
    --batch_size $batch_size \
    --topk_s $topk_s \
    --stimulus_image_path $stimulus_image_path \
    --stimulus_sub_dir_name $stimulus_sub_dir_name
###############################################################################

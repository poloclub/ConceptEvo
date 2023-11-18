###############################################################################
# responsive_neurons.sh
# 
# Identify neurons that are most responsive to each sampled image.
# Run this script at `../../src` where `main.py` exists.
# 
# The result will be saved at
# ../../data
#     └── responsive_neurons
#           └── <responsive_neurons_sub_dir_name>
#                 ├── data
#                 │     └── responsive_neurons_<model_nickname_in_file*>.json
#                 └── log
#                       ├── setting.txt
#                       └── responsive_neurons_log_<model_nickname_in_file*>.txt
#
# <model_nickname_in_file*> depends on the model's training status:
# - For models trained up to a specific epoch, <model_nickname>_<epoch>
# - For pretrained models, <model_nickname>
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
# topk_i=?
# responsive_neurons_image_path=?
# responsive_neurons_sub_dir_name=?
# 
# For example:
gpu=0
model_name=vgg16
lr=0.01
model_nickname="$model_name"_"$lr"
epoch=207
batch_size=512
topk_i=20
responsive_neurons_image_path=../../ILSVRC2012/train_0.1
responsive_neurons_sub_dir_name=train_0.1
#
# For another example, a setting for a pretrained model would be:
# gpu=0
# model_name=vgg19
# model_nickname="$model_name"_pretrained
# epoch=-1
# model_path=None
# batch_size=512
# topk_i=20
# responsive_neurons_image_path=../../ILSVRC2012/train_0.1
# responsive_neurons_sub_dir_name=train_0.1
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --responsive_neurons True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --epoch $epoch \
    --batch_size $batch_size \
    --topk_i $topk_i \
    --responsive_neurons_image_path $responsive_neurons_image_path \
    --responsive_neurons_sub_dir_name $responsive_neurons_sub_dir_name
###############################################################################

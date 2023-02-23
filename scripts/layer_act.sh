###############################################################################
# layer_act.sh
# 
# Compute layer activation.
# Run this script at `../src` where `main.py` exists.
# 
# The result will be saved at `../data/layer_act/<model_nickname>`.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# training_data=?
# model_name=?
# model_nickname=?
# model_path=?
# batch_size=?
# layer=?
# 
# For example:
gpu=0
training_data=../../Broden/dataset/broden1_227/parsed_images

model_name=vgg16
lr=0.01
epoch=207
model_nickname="$model_name"_"$lr"_"$epoch"_broden
model_path=../data/model/"$model_name"_"$lr"/data/model-"$epoch".pth

batch_size=512
layer=Sequential_2_Linear_6

###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --layer_act T \
    --training_data $training_data \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --batch_size $batch_size \
    --layer $layer 
###############################################################################

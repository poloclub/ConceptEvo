###############################################################
# train_model.sh
# 
# Train a model from scratch or a pre-trained one.
# Run this script at `../src` where `main.py` exists.
# 
# The trained model of an epoch will be saved at
# `../data/model/<model_nickname>/data/model-<epoch>.pth`.
# 
# The training logs will be saved at
# `../data/model/<model_nickname>/log/training-log.pth`.
# 
# The description of model will be saved at
# `../data/model/<model_nickname>/log/model-info.txt`.

# The description of the layers will be saved at
# `../data/model/<model_nickname>/log/layer-info.txt`. 
###############################################################

###############################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# training_data=?
# test_data=?
# model_name=?
# model_nickname=?
# batch_size=?
# lr=?
# weight_decay=?
# momentum=?
# learning_eps=?
# topk=?
# num_epochs=?

# For example:
gpu=0
training_data=../../ILSVRC2012/train
test_data=../../ILSVRC2012/val-by-class
model_name=convnext
model_nickname=convnext_0.004
batch_size=512
lr=0.004
weight_decay=0.05
momentum=0.9
learning_eps=0.05
topk=5
num_epochs=300
###############################################################

###############################################################
# To train a model from a pre-trained one,
# provide a correct path of the pre-trained model:
# 
# model_path=?

# For example:
model_path=../data/model/convnext_0.004/data/model-34.pth
###############################################################

###############################################################
# Train a model from scratch
python main.py \
    --gpu $gpu \
    --train T \
    --training_data $training_data \
    --test_data $test_data \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --batch_size $batch_size \
    --lr $lr \
    --weight_decay $weight_decay \
    --momentum $momentum \
    --learning_eps $learning_eps \
    --topk $topk \
    --num_epochs $num_epochs
###############################################################

###############################################################
# Train a model from a pre-trained one
python main.py \
    --gpu $gpu \
    --train T \
    --training_data $training_data \
    --test_data $test_data \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --lr $lr \
    --weight_decay $weight_decay \
    --momentum $momentum \
    --learning_eps $learning_eps \
    --topk $topk \
    --num_epochs $num_epochs
###############################################################

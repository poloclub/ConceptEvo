###############################################################
# test_model.sh
# 
# Measure the training and test accuracy of a model.
# Run this script at `../src` where `main.py` exists.
# 
# The result will be saved at 
# `../data/model/<model_nickname>/log/test-epoch=<epoch>.txt`.
###############################################################

###############################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# training_data=?
# test_data=?
# model_name=?
# model_nickname=?
# epoch=?
# model_path=?
# batch_size=?
# topk=?

# For example:
gpu=0
training_data=../../ILSVRC2012/train
test_data=../../ILSVRC2012/val-by-class
model_name=convnext
model_nickname=convnext_0.004
epoch=7
model_path=../data/model/convnext_0.004/data/model-7.pth
batch_size=512
topk=5
###############################################################

###############################################################
python main.py \
    --gpu $gpu \
    --test T \
    --training_data $training_data \
    --test_data $test_data \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --epoch $epoch \
    --model_path $model_path \
    --batch_size $batch_size \
    --topk $topk
 ###############################################################
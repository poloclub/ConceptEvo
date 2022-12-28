# Provide a correct value for each "?" below:
# 
# gpu=?
# training_data=?
# test_data=?
# model_name=?
# model_nickname=?
# lr=?
# weight_decay=?
# momentum=?
# learning_eps=?
# topk=5

# For example:
gpu=0
training_data=../../ILSVRC2012/train
test_data=../../ILSVRC2012/val-by-class
model_name=convnext
model_nickname=convnext_0.004
lr=0.004
weight_decay=0.05
momentum=0.9
learning_eps=0.05
num_epochs=300

# Provide a correct value for each "?" below to train a model
# from a pre-trained model
# model_path=?

# For example:
model_path=../data/model/convnext_4e-3/data/model-34.pth


# Train a model from scratch
python main.py \
    --gpu $gpu \
    --train T \
    --training_data $training_data \
    --test_data $test_data \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --lr $lr \
    --weight_decay $weight_decay \
    --momentum $momentum \
    --learning_eps $learning_eps \
    --topk $topk \
    --num_epochs $num_epochs

# Train a model from a pre-trained model
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


# InceptionV3
python main.py \
    --gpu $gpu \
    --train T \
    --training_data ../../ILSVRC2012/train \
    --test_data ../../ILSVRC2012/val-by-class \
    --batch_size 64 \
    --lr 0.045 \
    --weight_decay 0.9 \
    --learning_eps 1.0 \
    --num_epochs 100 \
    --model_name inception_v3 \
    --model_nickname inception_v3_0.045_0.9_1.0

# ResNet18
python main.py \
    --gpu $gpu \
    --train T \
    --training_data ../../ILSVRC2012/train \
    --test_data ../../ILSVRC2012/val-by-class \
    --batch_size 256 \
    --lr 0.1 \
    --weight_decay 0.0001 \
    --momentum 0.9 \
    --num_epochs 300 \
    --model_name resnet18 \
    --model_nickname resnet18_0.1
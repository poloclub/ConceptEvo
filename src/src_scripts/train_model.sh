gpu=1
python main.py \
    --gpu $gpu \
    --train T \
    --training_data ../../cifar10/train \
    --test_data ../../cifar10/test \
    --model_name vgg16_cifar10 \
    --model_nickname vgg16_cifar10
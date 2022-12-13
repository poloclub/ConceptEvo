gpu=5

# ConvNeXt
python main.py \
    --gpu $gpu \
    --train T \
    --training_data ../../ILSVRC2012/train \
    --test_data ../../ILSVRC2012/val-by-class \
    --lr 4e-3 \
    --weight_decay 0.05 \
    --num_epochs 300 \
    --model_name convnext \
    --model_nickname convnext_4e-3

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
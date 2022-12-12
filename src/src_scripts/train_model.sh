gpu=5

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
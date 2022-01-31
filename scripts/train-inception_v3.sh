# Run the script in `../src` where `main.py` exists.
python main.py \
    --gpu 0 \
    --train T \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-0.5-0.9 \
    --batch_size 512 \
    --lr 0.5 \
    --momentum 0.9 \
    --num_epochs 300 \
    --topk 5 
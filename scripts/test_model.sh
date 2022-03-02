# Run the script in `../src` where `main.py` exists.
python main.py \
    --gpu 0 \
    --test T \
    --batch_size 256 \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-0.5-0.9-121 \
    --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth \
    --topk 5 
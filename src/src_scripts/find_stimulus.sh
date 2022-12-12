gpu=7
label=457
batch_size=512
eps=0.5
ratio=0.3

echo "Find stimulus"
python main.py \
    --gpu $gpu \
    --stimulus True \
    --batch_size $batch_size \
    --label $label \
    --eps $eps \
    --eval_sample_ratio $ratio \
    --model_name vgg16 \
    --model_nickname vgg16-512-0.01-0.9-207 \
    --model_path ../data/model/vgg16-512-0.01-0.9/data/model-207.pth
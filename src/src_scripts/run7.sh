gpu=7

batch_size=512
model_name=vgg16_no_dropout
model_nickname=vgg16_no_dropout-256-0.01-0.9

for epoch in 0 1 2 3 5 30
do
    echo "VGG16-no-dropout-$epoch stimulus"
    python main.py \
        --gpu $gpu \
        --stimulus T \
        --model_name $model_name \
        --model_nickname $model_nickname-$epoch \
        --model_path ../data/model/$model_nickname/data/model-$epoch.pth \
        --batch_size $batch_size \
        --topk_s 10 
done
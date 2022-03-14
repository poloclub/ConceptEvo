gpu=6

batch_size=512
model_name=vgg16
model_nickname=vgg16-512-0.05-0.9

for epoch in 14 54
do
    echo "VGG16-0.05-$epoch stimulus"
    python main.py \
        --gpu $gpu \
        --batch_size $batch_size
        --stimulus T \
        --model_name $model_name \
        --model_nickname $model_nickname-$epoch \
        --model_path ../data/model/$model_nickname/data/model-$epoch.pth \
        --batch_size $batch_size \
        --topk_s 10 
done

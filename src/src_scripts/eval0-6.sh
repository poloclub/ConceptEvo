gpu=0
batch_size=10
model_name=vgg16
find_num_sample_imgs=260

for label in 277 27 59 394 830
do
    model_nickname=vgg16-512-0.01-0.9
    from_model_nickname=vgg16-512-0.01-0.9-5
    from_model_path='../data/model/vgg16-512-0.01-0.9/data/model-5.pth'
    to_model_nickname=vgg16-512-0.01-0.9-21
    to_model_path='../data/model/vgg16-512-0.01-0.9/data/model-21.pth'
    
    echo "Find Vgg16 5->21 label=$label gpu=$gpu N=$find_num_sample_imgs"
    python main.py \
        --gpu $gpu \
        --find_important_evo T \
        --batch_size $batch_size \
        --label $label \
        --model_name $model_name \
        --model_nickname $model_nickname \
        --from_model_nickname $from_model_nickname \
        --from_model_path $from_model_path \
        --to_model_nickname $to_model_nickname \
        --to_model_path $to_model_path \
        --find_num_sample_imgs $find_num_sample_imgs

    echo "Eval Vgg16 5->21 label=$label gpu=$gpu"
    for ratio in 0.1 0.3 0.5 0.7
    do
        python main.py \
            --gpu $gpu \
            --eval_important_evo freezing \
            --batch_size $batch_size \
            --label $label \
            --eval_sample_ratio $ratio \
            --model_name $model_name \
            --model_nickname $model_nickname \
            --from_model_nickname $from_model_nickname \
            --from_model_path $from_model_path \
            --to_model_nickname $to_model_nickname \
            --to_model_path $to_model_path \
            --find_num_sample_imgs $find_num_sample_imgs
    done

    from_model_nickname=vgg16-512-0.01-0.9-21
    from_model_path='../data/model/vgg16-512-0.01-0.9/data/model-21.pth'
    to_model_nickname=vgg16-512-0.01-0.9-207
    to_model_path='../data/model/vgg16-512-0.01-0.9/data/model-207.pth'

    echo "Find Vgg16 21->207 label=$label gpu=$gpu N=$find_num_sample_imgs"
    python main.py \
        --gpu $gpu \
        --find_important_evo T \
        --batch_size $batch_size \
        --label $label \
        --model_name $model_name \
        --model_nickname $model_nickname \
        --from_model_nickname $from_model_nickname \
        --from_model_path $from_model_path \
        --to_model_nickname $to_model_nickname \
        --to_model_path $to_model_path \
        --find_num_sample_imgs $find_num_sample_imgs

    echo "Eval Vgg16 21->207 label=$label gpu=$gpu"
    for ratio in 0.1 0.3 0.5 0.7
    do
        python main.py \
            --gpu $gpu \
            --eval_important_evo freezing \
            --batch_size $batch_size \
            --label $label \
            --eval_sample_ratio 0.1 \
            --model_name $model_name \
            --model_nickname $model_nickname \
            --from_model_nickname $from_model_nickname \
            --from_model_path $from_model_path \
            --to_model_nickname $to_model_nickname \
            --to_model_path $to_model_path \
            --find_num_sample_imgs $find_num_sample_imgs
    done

done
gpu=3
batch_size=20
eps=0.5

for label in 494 353 10 913 436 34 324 735 857 20 910 500 524 286 707 819 897 42 520 582 346 867 156 435 800 378 301 104 994 842 820
do
    echo "find Vgg16 label $label 5->21"
    python main.py \
        --gpu $gpu \
        --find_important_evo T \
        --batch_size $batch_size \
        --label $label \
        --model_name vgg16 \
        --model_nickname vgg16-512-0.01-0.9 \
        --from_model_nickname vgg16-512-0.01-0.9-5 \
        --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-5.pth \
        --to_model_nickname vgg16-512-0.01-0.9-21 \
        --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth 

    for ratio in 0.1 0.3 0.5 0.7 1
    do 
        echo "eval Vgg16 label $label 5->21 freezing ratio $ratio"
        python main.py \
            --gpu $gpu \
            --eval_important_evo freezing \
            --batch_size $batch_size \
            --label $label \
            --eps $eps \
            --eval_sample_ratio $ratio \
            --model_name vgg16 \
            --model_nickname vgg16-512-0.01-0.9 \
            --from_model_nickname vgg16-512-0.01-0.9-5 \
            --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-5.pth \
            --to_model_nickname vgg16-512-0.01-0.9-21 \
            --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth 
    done

    echo "find Vgg16 label $label 21->107"
    python main.py \
        --gpu $gpu \
        --find_important_evo T \
        --batch_size $batch_size \
        --label $label \
        --model_name vgg16 \
        --model_nickname vgg16-512-0.01-0.9 \
        --from_model_nickname vgg16-512-0.01-0.9-21 \
        --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth \
        --to_model_nickname vgg16-512-0.01-0.9-107 \
        --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-107.pth 

    for ratio in 0.1 0.3 0.5 0.7 1
    do 
        echo "eval Vgg16 label $label 21->107 freezing ratio $ratio"
        python main.py \
            --gpu $gpu \
            --eval_important_evo freezing \
            --batch_size $batch_size \
            --label $label \
            --eps $eps \
            --eval_sample_ratio $ratio \
            --model_name vgg16 \
            --model_nickname vgg16-512-0.01-0.9 \
            --from_model_nickname vgg16-512-0.01-0.9-21 \
            --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth \
            --to_model_nickname vgg16-512-0.01-0.9-107 \
            --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-107.pth 
    done
done


# # for label in 388 394 400 406 415 435 436 446 469 475 481 494 500
# for label in 475 481 494 500
# do
#     echo "label $label 3->11"
#     python main.py \
#         --gpu $gpu \
#         --find_important_evo T \
#         --batch_size 128 \
#         --label $label \
#         --model_name inception_v3 \
#         --model_nickname inception_v3-512-0.5-0.9 \
#         --from_model_nickname inception_v3-512-0.5-0.9-3 \
#         --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-3.pth \
#         --to_model_nickname inception_v3-512-0.5-0.9-11 \
#         --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth 

#     echo "label $label 11->121"
#     python main.py \
#         --gpu $gpu \
#         --find_important_evo T \
#         --batch_size 128 \
#         --label $label \
#         --model_name inception_v3 \
#         --model_nickname inception_v3-512-0.5-0.9 \
#         --from_model_nickname inception_v3-512-0.5-0.9-11 \
#         --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
#         --to_model_nickname inception_v3-512-0.5-0.9-121 \
#         --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 
# done

# for label in 388 394 400 406 415 435 436 446 469 475 481 494 500
# do
#     echo "vgg16 label $label 5->21"
#     python main.py \
#         --gpu $gpu \
#         --find_important_evo T \
#         --batch_size 128 \
#         --label $label \
#         --model_name vgg16 \
#         --model_nickname vgg16-512-0.01-0.9 \
#         --from_model_nickname vgg16-512-0.01-0.9-5 \
#         --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-5.pth \
#         --to_model_nickname vgg16-512-0.01-0.9-21 \
#         --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth

#     echo "vgg16 label $label 21->207"
#     python main.py \
#         --gpu $gpu \
#         --find_important_evo T \
#         --batch_size 128 \
#         --label $label \
#         --model_name vgg16 \
#         --model_nickname vgg16-512-0.01-0.9 \
#         --from_model_nickname vgg16-512-0.01-0.9-21 \
#         --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth \
#         --to_model_nickname vgg16-512-0.01-0.9-207 \
#         --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-207.pth
# done
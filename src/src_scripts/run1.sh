gpu=2

for epoch in `seq 0 173`;
do
    echo "epoch=$epoch"
    python main.py \
        --gpu $gpu \
        --test T \
        --batch_size 256 \
        --model_name inception_v3 \
        --model_nickname inception_v3-512-0.05-0.9-$epoch \
        --model_path ../data/model/inception_v3-512-0.05-0.9/data/model-512-0.05-0.9-$epoch.pt \
        --topk 5 
done

# batch_size=10
# model_name=vgg16

# for label in 800 801 820 842 867 903 943 957 963 994
# do
#     model_nickname=vgg16-512-0.01-0.9
#     from_model_nickname=vgg16-512-0.01-0.9-5
#     from_model_path='../data/model/vgg16-512-0.01-0.9/data/model-5.pth'
#     to_model_nickname=vgg16-512-0.01-0.9-21
#     to_model_path='../data/model/vgg16-512-0.01-0.9/data/model-21.pth'

#     python main.py \
#         --gpu $gpu \
#         --find_important_evo T \
#         --batch_size $batch_size \
#         --label $label \
#         --model_name $model_name \
#         --model_nickname $model_nickname \
#         --from_model_nickname $from_model_nickname \
#         --from_model_path $from_model_path \
#         --to_model_nickname $to_model_nickname \
#         --to_model_path $to_model_path

#     from_model_nickname=vgg16-512-0.01-0.9-21
#     from_model_path='../data/model/vgg16-512-0.01-0.9/data/model-21.pth'
#     to_model_nickname=vgg16-512-0.01-0.9-207
#     to_model_path='../data/model/vgg16-512-0.01-0.9/data/model-207.pth'

#     python main.py \
#         --gpu $gpu \
#         --find_important_evo T \
#         --batch_size $batch_size \
#         --label $label \
#         --model_name $model_name \
#         --model_nickname $model_nickname \
#         --from_model_nickname $from_model_nickname \
#         --from_model_path $from_model_path \
#         --to_model_nickname $to_model_nickname \
#         --to_model_path $to_model_path

# done


# for label in 494 353 10 913 436 34 324 735 857 20 910 500 524 286 707 819 897 42 520 582 346 867 156 435 800 378 301 104 994 842 820
# do
#     echo "find InceptionV3 label $label 3->11"
#     python main.py \
#         --gpu $gpu \
#         --find_important_evo T \
#         --batch_size $batch_size \
#         --label $label \
#         --model_name inception_v3 \
#         --model_nickname inception_v3-512-0.5-0.9 \
#         --from_model_nickname inception_v3-512-0.5-0.9-3 \
#         --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-3.pth \
#         --to_model_nickname inception_v3-512-0.5-0.9-11 \
#         --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth 

#     for ratio in 0.1 0.3 0.5 0.7 1
#     do 
#         echo "eval InceptionV3 label $label 3->11 freezing ratio $ratio"
#         python main.py \
#             --gpu $gpu \
#             --eval_important_evo freezing \
#             --batch_size $batch_size \
#             --label $label \
#             --eps $eps \
#             --eval_sample_ratio $ratio \
#             --model_name inception_v3 \
#             --model_nickname inception_v3-512-0.5-0.9 \
#             --from_model_nickname inception_v3-512-0.5-0.9-3 \
#             --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-3.pth \
#             --to_model_nickname inception_v3-512-0.5-0.9-11 \
#             --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth 
#     done

#     echo "find InceptionV3 label $label 11->121"
#     python main.py \
#         --gpu $gpu \
#         --find_important_evo T \
#         --batch_size $batch_size \
#         --label $label \
#         --model_name inception_v3 \
#         --model_nickname inception_v3-512-0.5-0.9 \
#         --from_model_nickname inception_v3-512-0.5-0.9-11 \
#         --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
#         --to_model_nickname inception_v3-512-0.5-0.9-121 \
#         --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 

#     for ratio in 0.1 0.3 0.5 0.7 1
#     do 
#         echo "eval InceptionV3 label $label 11->121 freezing ratio $ratio"
#         python main.py \
#             --gpu $gpu \
#             --eval_important_evo freezing \
#             --batch_size $batch_size \
#             --label $label \
#             --eps $eps \
#             --eval_sample_ratio $ratio \
#             --model_name inception_v3 \
#             --model_nickname inception_v3-512-0.5-0.9 \
#             --from_model_nickname inception_v3-512-0.5-0.9-11 \
#             --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
#             --to_model_nickname inception_v3-512-0.5-0.9-121 \
#             --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 
#     done
# done

# # gpu=1
# # for label in 97 104 133 156 159 175 230 242 262 263 268 273
# # do
# #     echo "label $label 3->11"
# #     python main.py \
# #         --gpu $gpu \
# #         --find_important_evo T \
# #         --batch_size 256 \
# #         --label $label \
# #         --model_name inception_v3 \
# #         --model_nickname inception_v3-512-0.5-0.9 \
# #         --from_model_nickname inception_v3-512-0.5-0.9-3 \
# #         --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-3.pth \
# #         --to_model_nickname inception_v3-512-0.5-0.9-11 \
# #         --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth 

# #     echo "label $label 11->121"
# #     python main.py \
# #         --gpu $gpu \
# #         --find_important_evo T \
# #         --batch_size 256 \
# #         --label $label \
# #         --model_name inception_v3 \
# #         --model_nickname inception_v3-512-0.5-0.9 \
# #         --from_model_nickname inception_v3-512-0.5-0.9-11 \
# #         --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
# #         --to_model_nickname inception_v3-512-0.5-0.9-121 \
# #         --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 
# # done
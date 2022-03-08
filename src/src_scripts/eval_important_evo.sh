gpu=1
label=1
eps=2
ratio=0.3
batch_size=128

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

echo "eval Vgg16 label $label 11->121 perturbation"
python main.py \
    --gpu $gpu \
    --eval_important_evo perturbation \
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


echo "eval Vgg16 label $label 11->121 freezing"
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


# for label in 1 10 11 20 27 34 42 48 53 59 67 85 97 104 133 156 159 175 230 242 262 263 268 273 277 280 282 286 301 304 315 324 346 353 371 378 384 388 394 400 406 415 435 436 446 469 475 481 494 500 513 514 520 524 538 548 553 555 578 582 592 622 638 658 663 690 694 703 707 712 721 732 735 758 762 789 800 801 819 820 830 842 856 857 867 897 903 906 910 913 943 944 957 963 964 972 975 979 993 994
# for label in 1 10 11 20 27 
# for label in 801 957
# do
#     for eps in 0.1 0.3 0.5 0.7 0.9 
#     do
#         for ratio in 0.1 0.3 0.5 0.7 0.9
#         do
#             echo "eval inceptionV3 3->11, $label, $eps, $ratio"
#             python main.py \
#                 --gpu $gpu \
#                 --eval_important_evo perturbation \
#                 --batch_size 10 \
#                 --label $label \
#                 --eps $eps \
#                 --eval_sample_ratio $ratio \
#                 --model_name inception_v3 \
#                 --model_nickname inception_v3-512-0.5-0.9 \
#                 --from_model_nickname inception_v3-512-0.5-0.9-11 \
#                 --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
#                 --to_model_nickname inception_v3-512-0.5-0.9-121 \
#                 --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 

#             echo "eval inceptionV3 label $label 11->121"
#             python main.py \
#                 --gpu $gpu \
#                 --eval_important_evo perturbation \
#                 --batch_size 10 \
#                 --label $label \
#                 --eps $eps \
#                 --eval_sample_ratio $ratio \
#                 --model_name inception_v3 \
#                 --model_nickname inception_v3-512-0.5-0.9 \
#                 --from_model_nickname inception_v3-512-0.5-0.9-11 \
#                 --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
#                 --to_model_nickname inception_v3-512-0.5-0.9-121 \
#                 --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 
#         done
#     done
# done

# label=1
# eps=2
# ratio=0.3
# echo "eval inceptionV3 label $label 11->121"
# python main.py \
#     --gpu $gpu \
#     --eval_important_evo perturbation \
#     --batch_size 512 \
#     --label $label \
#     --eps $eps \
#     --eval_sample_ratio $ratio \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-0.5-0.9 \
#     --from_model_nickname inception_v3-512-0.5-0.9-11 \
#     --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
#     --to_model_nickname inception_v3-512-0.5-0.9-121 \
#     --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 
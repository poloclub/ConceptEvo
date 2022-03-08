gpu=5
label=10
batch_size=256

# echo "find InceptionV3 label $label 11->121"
# python main.py \
#     --gpu $gpu \
#     --find_important_evo T \
#     --batch_size $batch_size \
#     --label $label \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-0.5-0.9 \
#     --from_model_nickname inception_v3-512-0.5-0.9-11 \
#     --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
#     --to_model_nickname inception_v3-512-0.5-0.9-121 \
#     --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 
    
eps=0.5
ratio=0.3
echo "eval InceptionV3 label $label 11->121"
python main.py \
    --gpu $gpu \
    --eval_important_evo perturbation \
    --batch_size $batch_size \
    --label $label \
    --eps $eps \
    --eval_sample_ratio $ratio \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-0.5-0.9 \
    --from_model_nickname inception_v3-512-0.5-0.9-11 \
    --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
    --to_model_nickname inception_v3-512-0.5-0.9-121 \
    --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 
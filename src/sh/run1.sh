python main.py \
    --gpu 0 \
    --neuron_emb T \
    --model_name inception_v3_pretrained \
    --dim 30 \
    --lr_emb 0.1 \
    --num_emb_epochs 1000 \
    --num_emb_negs 3 

# echo "example_patch inception_v3-512-1.5-0.9-69"
# python main.py \
#     --gpu 1 \
#     --neuron_feature example_patch \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-1.5-0.9-69 \
#     --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-69.pth \
#     --num_features 15 \
#     --ex_patch_size_ratio 0.3

# echo "example_patch vgg16-512-0.05-0.9-11"
# python main.py \
#     --gpu 1 \
#     --neuron_feature example_patch \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.05-0.9-11 \
#     --model_path ../data/model/vgg16-512-0.05-0.9/data/model-11.pth \
#     --num_features 15 \
#     --ex_patch_size_ratio 0.3

# echo "example_patch inception_v3-512-1.5-0.9-71"
# python main.py \
#     --gpu 1 \
#     --neuron_feature example_patch \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-1.5-0.9-71 \
#     --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-71.pth \
#     --num_features 15 \
#     --ex_patch_size_ratio 0.3

# echo "example_patch inception_v3-512-1.5-0.9-4"
# python main.py \
#     --gpu 1 \
#     --neuron_feature example_patch \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-1.5-0.9-4 \
#     --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-4.pth \
#     --num_features 15 \
#     --ex_patch_size_ratio 0.3

# echo "stimulus vgg16-512-0.01-0.9-207"
# python main.py \
#     --gpu 1 \
#     --stimulus T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.01-0.9-207 \
#     --model_path ../data/model/vgg16-512-0.01-0.9/data/model-207.pth \
#     --batch_size 512 \
#     --topk_s 10 
python main.py \
    --gpu 0 \
    --neuron_emb T \
    --model_name inception_v3_pretrained \
    --dim 30 \
    --lr_emb 0.05 \
    --num_emb_epochs 4000 \
    --num_emb_negs 10 

# echo "example_patch vgg16-512-0.01-0.9-5"
# python main.py \
#     --gpu 2 \
#     --neuron_feature example_patch \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.01-0.9-5 \
#     --model_path ../data/model/vgg16-512-0.01-0.9/data/model-5.pth \
#     --num_features 15 \
#     --ex_patch_size_ratio 0.3

# echo "example_patch vgg16-512-0.01-0.9-21"
# python main.py \
#     --gpu 2 \
#     --neuron_feature example_patch \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.01-0.9-21 \
#     --model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth \
#     --num_features 15 \
#     --ex_patch_size_ratio 0.3

# echo "example_patch vgg16-512-0.05-0.9-3"
# python main.py \
#     --gpu 2 \
#     --neuron_feature example_patch \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.05-0.9-3 \
#     --model_path ../data/model/vgg16-512-0.05-0.9/data/model-3.pth \
#     --num_features 15 \
#     --ex_patch_size_ratio 0.3

# echo "example_patch vgg16-512-0.05-0.9-13"
# python main.py \
#     --gpu 2 \
#     --neuron_feature example_patch \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.05-0.9-13 \
#     --model_path ../data/model/vgg16-512-0.05-0.9/data/model-13.pth \
#     --num_features 15 \
#     --ex_patch_size_ratio 0.3

# echo "stimulus vgg16-512-0.05-0.9-11"
# python main.py \
#     --gpu 2 \
#     --stimulus T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.05-0.9-11 \
#     --model_path ../data/model/vgg16-512-0.05-0.9/data/model-11.pth \
#     --batch_size 512 \
#     --topk_s 10 

# echo "stimulus vgg16-512-0.01-0.9-207"
# python main.py \
#     --gpu 2 \
#     --stimulus T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.01-0.9-207 \
#     --model_path ../data/model/vgg16-512-0.01-0.9/data/model-207.pth \
#     --batch_size 512 \
#     --topk_s 10 
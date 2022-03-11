gpu=2

python main.py \
    --gpu $gpu \
    --neuron_feature example_patch \
    --model_name vgg19_pretrained \
    --batch_size 512 \
    --num_features 15 \
    --ex_patch_size_ratio 0.3

# Test Vgg19_pretrained
# python main.py \
#     --gpu 0 \
#     --test T \
#     --batch_size 256 \
#     --model_name vgg19_pretrained \
#     --topk 5 

# # Test Vgg16-0.005-189
# epoch=189
# python main.py \
#     --gpu 0 \
#     --test T \
#     --batch_size 256 \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.005-0.9-$epoch \
#     --model_path ../data/model/vgg16-512-0.005-0.9/data/model-512-0.005-0.9-$epoch.pt \
#     --topk 5 

# # Test Vgg16-0.001-133
# epoch=133
# python main.py \
#     --gpu 0 \
#     --test T \
#     --batch_size 256 \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.001-0.9-$epoch \
#     --model_path ../data/model/vgg16-512-0.001-0.9/data/model-512-0.001-0.9-$epoch.pt \
#     --topk 5 

# # Test InceptionV3-0.005-149
# epoch=149
# python main.py \
#     --gpu 0 \
#     --test T \
#     --batch_size 256 \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-0.005-0.9-$epoch \
#     --model_path ../data/model/inception_v3-512-0.005-0.9/data/model-512-0.005-0.9-$epoch.pt \
#     --topk 5 

# # Test InceptionV3-0.1-149
# epoch=100
# python main.py \
#     --gpu 0 \
#     --test T \
#     --batch_size 256 \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-0.1-0.9-$epoch \
#     --model_path ../data/model/inception_v3-512-0.1-0.9/data/model-512-0.1-0.9-$epoch.pt \
#     --topk 5 

# # Test InceptionV3-0.5-299
# epoch=299
# python main.py \
#     --gpu 0 \
#     --test T \
#     --batch_size 256 \
#     --model_name vgg16 \
#     --model_nickname inception_v3-512-0.005-0.9-$epoch \
#     --model_path ../data/model/vgg16-512-0.005-0.9/data/model-512-0.005-0.9-$epoch.pt \
#     --topk 2 

# Run image embedding (Base model Vgg16-0.01-207)
# python main.py \
#     --gpu $gpu \
#     --img_emb T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.01-0.9-207 \
#     --model_path ../data/model/vgg16-512-0.01-0.9/data/model-207.pth \
#     --dim 30 \
#     --lr_emb 0.05 \
#     --num_emb_epochs 10000 \
#     --num_emb_negs 3 \
#     --lr_img_emb 10 \
#     --thr_img_emb 0.001 \
#     --max_iter_img_emb 10000 \
#     --k 10

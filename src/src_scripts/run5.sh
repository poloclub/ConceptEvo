gpu=5

base_model_name=vgg19_pretrained
img_emb_path='../data/embedding/vgg19_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=10000-k=10.txt' 
emb_store_dirname='emb-5-base-vgg19_pretrained-10.0'


# Run neuron embedding (Base model Vgg19-pretrained)
# python main.py \
#     --gpu $gpu \
#     --neuron_emb T \
#     --model_name $model_name \
#     --dim 30 \
#     --lr_emb 0.05 \
#     --num_emb_epochs 10000 \
#     --num_emb_negs 3 

# Run image embedding (Base model Vgg19-pretrained)
# echo "Run image embedding (Base model Vgg19-pretrained)"
# python main.py \
#     --gpu $gpu \
#     --img_emb T \
#     --model_name $model_name \
#     --dim 30 \
#     --lr_emb 0.05 \
#     --num_emb_epochs 10000 \
#     --num_emb_negs 3 \
#     --lr_img_emb 10 \
#     --thr_img_emb 0.001 \
#     --max_iter_img_emb 10000 \
#     --k 10


# Run projection
# for model in 'inception_v3_pretrained' 'vgg16_pretrained' 'vgg19_pretrained'
# for model in 'inception_v3_pretrained' 'vgg16_pretrained'
# do
#     echo "Run projection $model" 
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name $model \
#         --model_nickname $model \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done

# model_name=inception_v3
# model_nickname=inception_v3-512-0.5-0.9
# for epoch in 3 11 121
# do
#     echo "Run projection $model_nickname-$epoch" 
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name $model_name \
#         --model_nickname $model_nickname-$epoch \
#         --model_path ../data/model/$model_nickname/data/model-$epoch.pth \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done

# model_name=inception_v3
# model_nickname=inception_v3-512-1.5-0.9
# for epoch in 4 69 70 71 100
# do
#     echo "Run projection $model_nickname-$epoch" 
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name $model_name \
#         --model_nickname $model_nickname-$epoch \
#         --model_path ../data/model/$model_nickname/data/model-$epoch.pth \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done

# model_name=vgg16
# model_nickname=vgg16-512-0.05-0.9
# # for epoch in 3 11 12 13 14 54
# for epoch in 14 54
# do
#     echo "Run projection $model_nickname-$epoch" 
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name $model_name \
#         --model_nickname $model_nickname-$epoch \
#         --model_path ../data/model/$model_nickname/data/model-$epoch.pth \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done

# model_name=vgg16
# model_nickname_wo_epoch=vgg16-512-0.01-0.9
# for epoch in 5 7 21 207
# do
#     echo "Run projection $model_nickname-$epoch" 
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name $model_name \
#         --model_nickname $model_nickname-$epoch \
#         --model_path ../data/model/$model_nickname/data/model-$epoch.pth \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done 

model_name=vgg16_no_dropout
model_nickname=$model_name-256-0.01-0.9
# # for epoch in 0 1 2 3 5 30
for epoch in 30
# for epoch in 0 1 2 3
do
    echo "Run projection $model_nickname-$epoch" 
    python main.py \
        --gpu $gpu \
        --proj_neuron_emb T \
        --model_name $model_name \
        --model_nickname $model_nickname-$epoch \
        --model_path ../data/model/$model_nickname/data/model-$epoch.pth \
        --img_emb_path $img_emb_path \
        --dim 30 \
        --k 10 \
        --emb_store_dirname $emb_store_dirname
done 

# Emb 2d
echo "Emb2d based on pretrained $base_model_name" 
python main.py \
    --gpu $gpu \
    --dim_reduction UMAP \
    --emb_set_dir ../data/embedding/$emb_store_dirname \
    --dim 30 \
    --model_for_emb_space base
gpu=4

# Run neuron embedding for a base model

# Project other models' neurons
img_emb_path='../data/embedding/vgg16-512-0.01-0.9-207/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=10000-k=10.txt'

model_names=(
    
)

model_nicknames=(

)

model_paths=(

)

for index in ${!model_names[*]}
do
    echo "${model_names[$index]}, ${model_nicknames[$index]}"
done
# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-0.5-0.9-3 \
#     --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-3.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname
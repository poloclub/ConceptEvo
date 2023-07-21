###############################################################################
# image_embedding.sh
# 
# Create image embeddings from base model's neuron embeddings.
# Run this script at `../src` where `main.py` exists.
###############################################################################

###############################################################################
# File structure:
# 
# ../data
#     └── image_embedding
#             └── <image_embedding_sub_dir_name>
#                   ├── data
#                   │     └── img_emb_<model_nickname>.json
#                   └── log
#                         ├── setting.txt
#                         └── img_emb_log_<model_nickname>.txt
#
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# model_path=?
# dim=?
# topk_n=?
# lr_img_emb=?
# thr_img_emb=?
# num_img_emb_epochs=?
# stimulus_sub_dir_name=?
# neuron_embedding_sub_dir_name=?
# image_embedding_sub_dir_name=?
# 
# For example:
gpu=0
model_name=vgg19
model_nickname=vgg19_pretrained
model_path=None
dim=30
topk_n=20
lr_img_emb=0.1
thr_img_emb=0.01
num_img_emb_epochs=10000
stimulus_sub_dir_name=train_0.1
neuron_embedding_sub_dir_name=train_0.1
image_embedding_sub_dir_name=train_0.1
stimulus_image_path=../../ILSVRC2012/train_0.1
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --image_embedding True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --dim $dim \
    --topk_n $topk_n \
    --lr_img_emb $lr_img_emb \
    --thr_img_emb $thr_img_emb \
    --num_img_emb_epochs $num_img_emb_epochs \
    --stimulus_sub_dir_name $stimulus_sub_dir_name \
    --neuron_embedding_sub_dir_name $neuron_embedding_sub_dir_name \
    --image_embedding_sub_dir_name $image_embedding_sub_dir_name \
    --stimulus_image_path $stimulus_image_path
###############################################################################

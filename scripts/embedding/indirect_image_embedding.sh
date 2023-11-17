###############################################################################
# indirect_image_embedding.sh
# 
# Create image embeddings from base model's neuron embeddings.
# Run this script at `../../src` where `main.py` exists.
###############################################################################

###############################################################################
# File structure:
# 
# ../../data
#     └── indirect_image_embedding
#             └── <indirect_image_embedding_sub_dir_name>
#                   ├── data
#                   │     └── indirect_img_emb_<model_nickname>.json
#                   └── log
#                         ├── setting.txt
#                         └── indirect_img_emb_log_<model_nickname>.txt
#
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# model_path=?
# lr_indirect_img_emb=0.05
# thr_indirect_img_emb=0.01
# num_indirect_img_emb_epochs=10000
# stimulus_sub_dir_name=?
# image_embedding_sub_dir_name=?
# responsive_neurons_sub_dir_name=?
# indirect_image_embedding_sub_dir_name=?
# stimulus_image_path=?
# 
# For example:
gpu=0
model_name=vgg19
model_nickname=vgg19_pretrained
model_path=None
lr_indirect_img_emb=0.01
thr_indirect_img_emb=0.01
num_indirect_img_emb_epochs=1
num_indirect_img_emb_negs=2
num_indirect_img_emb_pairs=500
stimulus_sub_dir_name=train_0.1
image_embedding_sub_dir_name=train_0.1
responsive_neurons_sub_dir_name=train_0.1
indirect_image_embedding_sub_dir_name=train_0.1
stimulus_image_path=../../ILSVRC2012/train_0.1
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --indirect_image_embedding True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --lr_indirect_img_emb $lr_indirect_img_emb \
    --thr_indirect_img_emb $thr_indirect_img_emb \
    --num_indirect_img_emb_negs $num_indirect_img_emb_negs \
    --num_indirect_img_emb_epochs $num_indirect_img_emb_epochs \
    --num_indirect_img_emb_pairs $num_indirect_img_emb_pairs \
    --stimulus_sub_dir_name $stimulus_sub_dir_name \
    --responsive_neurons_sub_dir_name $responsive_neurons_sub_dir_name \
    --image_embedding_sub_dir_name $image_embedding_sub_dir_name \
    --indirect_image_embedding_sub_dir_name $indirect_image_embedding_sub_dir_name \
    --stimulus_image_path $stimulus_image_path
###############################################################################

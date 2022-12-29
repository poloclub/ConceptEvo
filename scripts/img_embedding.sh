###############################################################################
# img_embedding.sh
# 
# Create image embeddings from base model's neuron embeddings.
# Run this script at `../src` where `main.py` exists.
###############################################################################

###############################################################################
# File structure:
# 
# ../data
#     └── embedding
#             └── emb-<model_nickname>-<apdx1>
#                   └── data
#                   │     └── emb
#                   │     │    ├── emb.json     
#                   │     │    └── img_emb-<apdx2>.txt
#                   │     └── emb2d
#                   └── log
# <apdx1>: 
# ```
# '-'.join([
#     topk_s=<topk_s>,
#     dim=<dim>,
#     lr_emb=<lr_emb>,
#     num_emb_epochs=<num_emb_epochs>,
#     num_emb_negs=<num_emb_negs>
# ])
# ```
# 
# <apdx2>:
# ```
# '-'.join([
#     dim=<dim>,
#     lr_img_emb=<lr_img_emb>,
#     thr_img_emb=<thr_img_emb>,
#     max_iter_img_emb=<max_iter_img_emb>,
#     k=<k>
# ])
# ```
###############################################################################

###############################################################################
# It loads the base model's neuron embedding at `emb.json` 
# in the above file structure.
# 
# It saves the image embeddings at `img_emb-<apdx2>.txt`
# in the above file structure.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# model_path=?
# topk_s=?
# dim=?
# lr_emb=?
# num_emb_epochs=?
# num_emb_negs=?
# lr_img_emb=?
# thr_img_emb=?
# max_iter_img_emb=?
# k=?
# 
# For example:
gpu=0
model_name=vgg19_pretrained
model_nickname=vgg19_pretrained
model_path='DO_NOT_NEED_CURRENTLY' # Use this for a pytorch pretrained model
topk_s=20
dim=30
lr_emb=0.01
num_emb_epochs=100
num_emb_negs=3
lr_img_emb=1.0
thr_img_emb=0.01
max_iter_img_emb=10000
k=$topk_s
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --img_emb T \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --dim $dim \
    --lr_emb $lr_emb \
    --num_emb_epochs $num_emb_epochs \
    --num_emb_negs $num_emb_negs \
    --topk_s $topk_s \
    --lr_img_emb $lr_img_emb \
    --thr_img_emb $thr_img_emb \
    --max_iter_img_emb $max_iter_img_emb \
    --k $k
###############################################################################

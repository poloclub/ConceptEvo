###############################################################################
# embedding_img_layer_act.sh
# 
# Create image embeddings with layer activation.
# Run this script at `../src` where `main.py` exists.
###############################################################################

###############################################################################
# File structure:
# 
# ../data
#     └── embedding
#             └── emb-<basemodel_nickname>-<apdx1>
#                   ├── data
#                   │     ├── emb
#                   │     │    └── emb.json 
#                   │     └── emb_with_layer_act-<apdx2>-<apdx3>
#                   │          ├── emb_nd
#                   │          │      ├── img_emb.txt
#                   │          │      └── proj_emb-<model_nickname>.json
#                   │          └── emb_2d
#                   │                 ├── reducer.sav
#                   │                 ├── idx2id.json
#                   │                 ├── model_code.json
#                   │                 ├── emb_2d-<basemodel_nickname>.json
#                   │                 └── emb_2d-<model_nickname>.json
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
# 
# <apdx3>:
# ```
# '-'.join([
#     layer=<layer>,
#     lr_img_emb=<lr_img_emb>,
#     max_iter_img_emb=<max_iter_img_emb>,
#     num_emb_negs_layer_act=<num_emb_negs_layer_act>,
#     thr_img_emb=<thr_img_emb>,
#     k=<k>,
#     dim=<dim>
# ])
# ```
###############################################################################

###############################################################################
# It loads the base model's neuron embedding at `emb.json` 
# in the above file structure.
# 
# It saves the image embeddings at `img_emb.txt`
# in the above file structure.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# topk_s=?
# dim=?
# lr_emb=?
# num_emb_epochs=?
# num_emb_negs=?
# layer=?
# lr_img_emb=?
# max_iter_img_emb=?
# num_emb_negs_layer_act=?
# thr_img_emb=?
# k=?
# 
# For example:
gpu=0

model_name=vgg19_pretrained
model_nickname=vgg19_pretrained

topk_s=20
dim=30
lr_emb=0.05
num_emb_epochs=10000
num_emb_negs=3

layer=Sequential_0_Conv2d_34
lr_img_emb=0.01
max_iter_img_emb=100
num_emb_negs_layer_act=0
thr_img_emb=0.001
k=10


###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --img_emb_with_layer_act T \
    --model_nickname $model_nickname \
    --dim $dim \
    --lr_emb $lr_emb \
    --num_emb_epochs $num_emb_epochs \
    --num_emb_negs $num_emb_negs \
    --topk_s $topk_s \
    --layer $layer \
    --lr_img_emb $lr_img_emb \
    --thr_img_emb $thr_img_emb \
    --max_iter_img_emb $max_iter_img_emb \
    --num_emb_negs_layer_act $num_emb_negs_layer_act \
    --k $k
###############################################################################

###############################################################################
# embedding_img_co_act.sh
# 
# Create image embeddings with matrix factorization.
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
#                   │     └── emb-set-co_act-<apdx2>-<apdx3>
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
#     dim=<dim>,
#     layer=<layer>,
#     num_epochs_co_act=<num_epochs_co_act>,
#     k=<k>
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
# model_path=?
# topk_s=?
# dim=?
# lr_emb=?
# num_emb_epochs=?
# num_emb_negs=?
# lr_img_emb=?
# thr_img_emb=?
# from_iter_img_emb=?
# max_iter_img_emb=?
# k=?
# layer=?
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

lr_img_emb=0.005
thr_img_emb=0.001
from_iter_img_emb=-1
max_iter_img_emb=10000

layer='Sequential_0_Conv2d_34'

k=10
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --img_emb_co_act T \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --dim $dim \
    --lr_emb $lr_emb \
    --num_emb_epochs $num_emb_epochs \
    --num_emb_negs $num_emb_negs \
    --topk_s $topk_s \
    --lr_img_emb $lr_img_emb \
    --thr_img_emb $thr_img_emb \
    --from_iter_img_emb $from_iter_img_emb \
    --max_iter_img_emb $max_iter_img_emb \
    --layer $layer \
    --k $k
###############################################################################

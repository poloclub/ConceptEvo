###############################################################################
# embedding_2d.sh
# 
# Create 2D embedding of the base model and non-base models 
# on the unified semantic space.
# 
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
#                   │     └── emb-set-<apdx2>
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
###############################################################################

###############################################################################
# The result will be saved at `emb2d/` in the above file structure.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# basemodel_nickname=?
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
basemodel_nickname=vgg19_pretrained
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
    --dim_reduction UMAP \
    --basemodel_nickname $basemodel_nickname \
    --topk_s $topk_s \
    --dim $dim \
    --lr_emb $lr_emb \
    --num_emb_epochs $num_emb_epochs \
    --num_emb_negs $num_emb_negs \
    --lr_img_emb $lr_img_emb \
    --thr_img_emb $thr_img_emb \
    --max_iter_img_emb $max_iter_img_emb \
    --k $k
###############################################################################

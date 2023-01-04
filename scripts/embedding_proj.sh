###############################################################################
# embedding_proj.sh
# 
# Create (approximated) neuron embeddings of a non-base model.
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
# The result will be saved at `proj_emb-<model_nickname>-<apdx2>.json` 
# in the above file structure.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# basemodel_nickname=?
# model_nickname=?
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
model_nickname=vgg16-0.01-207
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
    --proj_neuron_emb T \
    --basemodel_nickname $basemodel_nickname \
    --model_nickname $model_nickname \
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

###############################################################################
# If you have multiple models, uncomment BLOCK1 below and run it.
# 
# If you get an error saying `Syntax error: "(" unexpected`,  
# run this script with `bash` command: `bash embedding_proj.sh`.
###############################################################################

###############################################################################
# # BLOCK1: if you have multiple models
# model_nicknames=( 
#     nickname_0
#     nickname_1
#     nickname_2
# )
# 
# For example:
# model_nicknames=( 
#     convnext-0.02-0
#     convnext-0.02-3
#     convnext-0.02-6
#     convnext-0.02-8
#     convnext-0.02-9
#     convnext-0.02-10
#     convnext-0.02-11
#     convnext-0.02-12
#     convnext-0.02-13
#     convnext-0.02-14
#     convnext-0.02-15
#     convnext-0.02-17
#     convnext-0.02-19
#     convnext-0.004-0
#     convnext-0.004-1
#     convnext-0.004-3
#     convnext-0.004-91
#     convnext-0.004-lambda0-0
#     convnext-0.004-lambda0-1
#     convnext-0.004-lambda0-2
#     convnext-0.004-lambda0-7
#     convnext-0.004-lambda0-32
#     inception_v3-1.5-0
#     inception_v3-1.5-4
#     inception_v3-1.5-70
#     inception_v3-1.5-71
#     inception_v3-1.5-72
#     inception_v3-1.5-73
#     inception_v3-1.5-100
#     vgg16_no_dropout-0.01-0
#     vgg16_no_dropout-0.01-1
#     vgg16_no_dropout-0.01-3
#     vgg16_no_dropout-0.01-30
#     vgg16-0.01-0
#     vgg16-0.01-5
#     vgg16-0.01-27
#     vgg16-0.01-21
#     vgg16-0.01-207
#     vgg16-0.05-0
#     vgg16-0.05-3
#     vgg16-0.05-12
#     vgg16-0.05-13
#     vgg16-0.05-14
#     vgg16-0.05-54
# )
# 
# for model_nickname in "${model_nicknames[@]}"
# do
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --basemodel_nickname $basemodel_nickname \
#         --model_nickname $model_nickname \
#         --topk_s $topk_s \
#         --dim $dim \
#         --lr_emb $lr_emb \
#         --num_emb_epochs $num_emb_epochs \
#         --num_emb_negs $num_emb_negs \
#         --lr_img_emb $lr_img_emb \
#         --thr_img_emb $thr_img_emb \
#         --max_iter_img_emb $max_iter_img_emb \
#         --k $k
# done
###############################################################################
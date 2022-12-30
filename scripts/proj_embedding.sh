###############################################################################
# proj_embedding.sh
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
#                   │          │      ├── emb.json
#                   │          │      ├── img_emb.txt
#                   │          │      └── proj_emb-<model_nickname>.json
#                   │          └── emb_2d
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
# model_name=?
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
model_name=vgg16
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
    --model_name $model_name \
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
# run this script with `bash` command: `bash proj_embedding.sh`.
###############################################################################

###############################################################################
# # BLOCK1: if you have multiple models
# 
# model_names=( 
#     model_0
#     model_1
#     model_2
# )

# model_nicknames=( 
#     nickname_0
#     nickname_1
#     nickname_2
# )

# for i in "${!model_names[@]}"
# do
#     model_name=${model_names[i]}
#     model_nickname=${model_nicknames[i]}
#     echo $model_name, $model_nickname

#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --basemodel_nickname $basemodel_nickname \
#         --model_name $model_name \
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
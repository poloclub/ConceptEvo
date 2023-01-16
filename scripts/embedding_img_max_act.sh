###############################################################################
# embedding_img_max_act.sh
# 
# Create image embeddings from base model's neuron activation.
# Run this script at `../src` where `main.py` exists.
###############################################################################

###############################################################################
# File structure:
# XXXX ../data/img_act_emb/vgg19_pretrained/data/Sequential_0_Conv2d_34/img_emb-dim=30.txt
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
# dim=?
# layer=?
# 
# For example:
gpu=0
model_name=vgg19_pretrained
model_nickname=vgg19_pretrained
model_path='DO_NOT_NEED_CURRENTLY' # Use this for a pytorch pretrained model
dim=30
layer=Sequential_0_Conv2d_34
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --img_act_emb T \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --model_path $model_path \
    --dim $dim \
    --layer $layer 
###############################################################################

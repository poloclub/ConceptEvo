###############################################################################
# proj_embedding.sh
# 
# Create (approximated) neuron embeddings of a non-base model.
# Run this script at `../../src` where `main.py` exists.
###############################################################################

###############################################################################
# File structure:
# 
# ../../data
#     └── proj_embedding
#             └── proj_embedding_sub_dir_name
#                   ├── data
#                   │     ├── proj_emb_<model_nickname_in_file*>.json
#                   │     └── proj_emb_vis_<model_nickname_in_file*>.pdf
#                   └── log
#                         ├── setting.txt
#                         └── proj_emb_log_<model_nickname_in_file*>.txt
#
# <model_nickname_in_file*> depends on the model's training status:
# - For models trained up to a specific epoch, <model_nickname>_<epoch>
# - For pretrained models, <model_nickname>
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# epoch=?
# dim=?
# stimulus_sub_dir_name=?
# proj_embedding_sub_dir_name=?
# img_embedding_path=?
# base_stimulus_path=?
# 
# For example:
gpu=0
model_name=vgg16
model_nickname=vgg16_0.01
epoch=207
dim=30
stimulus_sub_dir_name=train_0.1
# proj_embedding_sub_dir_name=train_0.1
# img_embedding_path=../data/image_embedding/train_0.1/data/img_emb_vgg19_pretrained.txt
proj_embedding_sub_dir_name=train_0.1_indirect
img_embedding_path=../data/indirect_image_embedding/train_0.1/data/indirect_img_emb_vgg19_pretrained.txt
base_stimulus_path=../data/stimulus/train_0.1/data/stimulus_vgg19_pretrained.json
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --proj_embedding True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --epoch $epoch \
    --dim $dim \
    --stimulus_sub_dir_name $stimulus_sub_dir_name \
    --proj_embedding_sub_dir_name $proj_embedding_sub_dir_name \
    --img_embedding_path $img_embedding_path \
    --base_stimulus_path $base_stimulus_path
###############################################################################
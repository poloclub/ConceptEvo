###############################################################################
# neuron_embedding.sh
# 
# Create neuron embeddings of a base model.
# Run this script at `../../src` where `main.py` exists.
###############################################################################

###############################################################################
# The results will be saved at:
# 
# ../../data
#     └── neuron_embedding
#             └── <neuron_embedding_sub_dir_name>
#                   ├── data
#                   │     ├── neuron_emb_<model_nickname_in_file*>.json
#                   │     ├── neuron_emb_vis_<model_nickname_in_file*>.pdf
#                   │     └── co_act_<model_nickname_in_file*>.json
#                   └── log
#                         ├── setting.txt
#                         └── neuron_emb_log_<model_nickname_in_file*>.txt
#
# <model_nickname_in_file*> depends on the model's training status:
# - For models trained up to a specific epoch, <model_nickname>_<epoch>
# - For pretrained models, <model_nickname>
###############################################################################

###############################################################################
# The result will be saved at `emb.json` in the above file structure.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_nickname=?
# epoch=?
# model_path=? (optional)
# dim=?
# lr_emb=?
# topk_n=?
# num_emb_epochs=?
# num_emb_negs=?
# neuron_embedding_sub_dir_name=?
# stimulus_sub_dir_name=?
# 
# For example:
gpu=0
model_name=vgg16
lr=0.01
epoch=4
model_nickname="$model_name"_"$lr"
dim=30
lr_emb=0.01
topk_n=20
num_emb_epochs=20
num_emb_negs=3
neuron_embedding_sub_dir_name=train_0.1
stimulus_sub_dir_name=train_0.1
#
# For example, the setting for a pretrained model would be:
# gpu=0
# model_name=vgg19
# model_nickname="$model_name"_pretrained
# epoch=-1
# model_path=None
# dim=30
# lr_emb=0.01
# topk_n=20
# num_emb_epochs=20
# num_emb_negs=3
# neuron_embedding_sub_dir_name=train_0.1
# stimulus_sub_dir_name=train_0.1
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --neuron_embedding True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --epoch $epoch \
    --dim $dim \
    --lr_emb $lr_emb \
    --topk_n $topk_n \
    --num_emb_epochs $num_emb_epochs \
    --num_emb_negs $num_emb_negs \
    --neuron_embedding_sub_dir_name $neuron_embedding_sub_dir_name \
    --stimulus_sub_dir_name $stimulus_sub_dir_name
###############################################################################

###############################################################################
# reduced_embedding.sh
# 
# Create 2D neuron embedding for both the base and non-base models.
# 
# Run this script at `../../src` where `main.py` exists.
###############################################################################

###############################################################################
# File structure:
# 
# ../../data
#     └── reduced_embedding
#             └── <reduced_embedding_sub_dir_name>
#                   ├── data
#                   │     ├── reduced_emb_<model_nickname_in_file*>.json
#                   │     └── reduced_emb_vis_<model_nickname_in_file*>.pdf
#                   └── log
#                         ├── setting.txt
#                         └── reduced_emb_log.txt
#
# <model_nickname_in_file*> depends on the model's training status:
# - For models trained up to a specific epoch, <model_nickname>_<epoch>
# - For pretrained models, <model_nickname>
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# neuron_embedding_path=?
# proj_embedding_dir_path=?
# reduced_embedding_sub_dir_name=?
# 
# For example:
gpu=0
neuron_embedding_path=../data/neuron_embedding/train_0.1/data/neuron_emb_vgg19_pretrained.json
# proj_embedding_dir_path=../data/proj_embedding/train_0.1/data/
# reduced_embedding_sub_dir_name=train_0.1
proj_embedding_dir_path=../data/proj_embedding/train_0.1_indirect/data/
reduced_embedding_sub_dir_name=train_0.1_indirect
###############################################################################

###############################################################################
python main.py \
    --gpu $gpu \
    --reduced_embedding True \
    --neuron_embedding_path $neuron_embedding_path \
    --proj_embedding_dir_path $proj_embedding_dir_path \
    --reduced_embedding_sub_dir_name $reduced_embedding_sub_dir_name 
###############################################################################

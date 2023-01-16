###############################################################################
# embedding_img_max_act.sh
# 
# Create image embeddings from base model's layer activation.
# Run this script at `../src` where `main.py` exists.
###############################################################################

###############################################################################
# The result will be saved at
# ../data/img_act_emb/<model_nickname>/data/<layer>/img_emb-dim=<dim>.txt
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

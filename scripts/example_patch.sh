###############################################################################
# example_patch.sh
# 
# Create example patches for each neuron in a model.
# Run this script at `../src` where `main.py` exists.
# 
# For a pretrained model, the result will be saved at
# ../data
#     └── example_patch
#           └── <model_nickname>
#                 ├── data
#                 │     └── topk_s=<topk_s>-ex_patch_ratio=<ex_patch_ratio>
#                 │          ├── crop/
#                 │          ├── mask/
#                 │          ├── inverse_mask/
#                 │          └── example_patch_info.json
#                 └── log
#
# Otherwise, the result will be saved at
# ../data
#     └── example_patch
#           └── <model_nickname>
#                 └── epoch_<epoch>
#                     ├── data
#                     │     └── topk_s=<topk_s>-ex_patch_ratio=<ex_patch_ratio>
#                     │          ├── crop/
#                     │          ├── mask/
#                     │          ├── inverse_mask/
#                     │          └── example_patch_info.json
#                     └── log
#
#
# In `.../crop`, it saves cropped example patches.
# In `.../mask`, it saves images with example patches being masked.
# In `.../inverse_mask`, it saves images with highlighted example patches 
#     and masked background.
# In `.../example_patch_info.json`, it saves the coordinate of example patches.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# gpu=?
# model_name=?
# model_nickname=?
# epoch=? (if it is not a pretrained model)
# topk_s=?
# ex_patch_size_ratio=?
# batch_size=?
# 
# For example:
gpu=0
model_name=convnext
lr=0.004
epoch=96
model_nickname="$model_name"_"$lr"
topk_s=20
ex_patch_size_ratio=0.3
batch_size=128

############################################################################### 
python main.py \
    --gpu $gpu \
    --example_patch True \
    --model_name $model_name \
    --model_nickname $model_nickname \
    --epoch $epoch \
    --topk_s $topk_s \
    --ex_patch_size_ratio $ex_patch_size_ratio \
    --batch_size $batch_size 
###############################################################################


###############################################################################
# eval_important_evo.sh
# 
# Evaluate if ConceptEvo well identifies important evolutions
# Run this script at `../../src` where `main.py` exists.
# 
# The result will be saved at
#
# ../../data
#     └── important_evo
#             └── <important_evo_sub_dir_name>
#                   └── label=<label>
#                         └── num_sampled_imgs=<num_sampled_imgs>
#                             ├── data
#                             │    └── eval_important_evo_idx=<idx>.json
#                             └── log
#                                  ├── setting.txt
#                                  └── eval_important_evo_log_idx=<idx>.txt
#
###############################################################################

###############################################################################
# Hyperparameters: Modify them below as you want

# 0. GPU Selection
# Specify the GPU device index you want to use (usually 0, 1, etc.)
gpu=0

# 1. Batch Size
# Set the batch size
batch_size=50

# 2. Target Label
# Define the classification label for which you want to find important evolutions
label=457

# 3. Model name
# The name of the DNN model. 
# Currently ['vgg16', 'inception_v3', 'convnext'] are available.
model_name=convnext

# 4. From model (Before target evolution)
# Provide the file path of the model, whose important evolutions you want to analyze. 
# This model should represent the state "before" the target evolution.
from_model_path=../data/model/convnext_0.004/data/model-3.pth

# 5. To model (After target evolution)
# Provide the file path of the model, whose important evolutions you want to analyze. 
# This model should represent the state "after" the target evolution.
to_model_path=../data/model/convnext_0.004/data/model-91.pth

# 6. The name of subdirectory for output
# Specify a subdirectory name for storing the important evolutions,
# which should be unique for each (from_model, to_model) pair. 
important_evo_sub_dir_name=convnext_0.004_3_91

# 7. (Optional) The file path of image index ranges by class
# Provide the file path that contains mappings between each class and 
# the class' image index ranges. 
#
# Each entry in the mapping consists of:
# - 'key': a class
# - 'val': a list representing the range of image indices for the class,
#          that looks like [start_idx, end_idx]
#
# This mapping is used to create a training dataset subset with images belonging to 
# a specific class. See how `label_subset` uses the `start_idx` and `end_idx`):
#
# ```
# import json
# from torchvision import datasets, transforms
#
# label_img_idx_path = a file path given by the user
# label = a label given by the user
#
# label_idx_dict = json.load(label_img_idx_path)
# start_idx, end_idx = label_idx_dict[label]
#
# training_dataset = datasets.ImageFolder(
#     train_data_path, 
#     data_transform
# )
# label_subset = torch.utils.data.Subset(
#     training_dataset, 
#     range(start_idx, end_idx)
# )
# ```
label_img_idx_path=../data/ILSVRC2012_label_img_idx.json

# 8. The number of sampled images
# Define the number of images to be sampled from the entire set of images 
# associated with the user-specified label.
# This should be the same as `num_sampled_imgs` used in `./find_important_evo.sh`.
num_sampled_imgs=250

# 9. The number of bins
# Define the number of bins to categorize the levels of importance based on 
# percentiles. For example, we use 4 bins to represent the following ranges:
# 1) 0-25th percentiles (Most important)
# 2) 25-50th percentiles
# 3) 50-75th percentiles
# 4) 75-100th percentiles
num_bins=4

# 10. Run index
# This index determines the order of execution among multiple runs of the same 
# process, which is aimed at finding important evolutions. It helps track the 
# progression of each run.
idx=0

###############################################################################
python main.py \
    --eval_important_evo True \
    --gpu $gpu \
    --batch_size $batch_size \
    --label $label \
    --model_name $model_name \
    --from_model_path $from_model_path \
    --to_model_path $to_model_path \
    --important_evo_sub_dir_name $important_evo_sub_dir_name \
    --label_img_idx_path $label_img_idx_path \
    --num_sampled_imgs $num_sampled_imgs \
    --num_bins $num_bins \
    --idx $idx
###############################################################################

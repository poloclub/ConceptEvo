###############################################################################
# sample_images.sh
# 
# Select a random subset of training images.
# Run this script at `../../src` where `main.py` exists.
#
# This takes training images from <input_image_path> and 
# stores the randomly sampled images at <output_image_path>.
###############################################################################

###############################################################################
# Provide a correct value for each "?" below:
# 
# image_sampling_ratio=?
# input_image_path=?
# output_image_path=?
# 
# For example:
image_sampling_ratio=0.1
input_image_path="../../ILSVRC2012/train"
output_image_path="../../ILSVRC2012/train"_"$image_sampling_ratio"
###############################################################################

###############################################################################
python main.py \
    --sample_images True \
    --image_sampling_ratio $image_sampling_ratio \
    --input_image_path $input_image_path \
    --output_image_path $output_image_path
###############################################################################

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
#             └── emb-<model_nickname>-<apdx1>
#                   └── data
#                   │     └── emb
#                   │     │    ├── emb.json     
#                   │     │    └── img_emb-<apdx2>.txt
#                   │     └── emb2d
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



python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name vgg16_pretrained \
    --model_nickname vgg16_pretrained \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-0.5-0.9-121 \
    --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-0.5-0.9-11 \
    --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-0.5-0.9-3 \
    --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-3.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-1.5-0.9-69 \
    --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-69.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-1.5-0.9-71 \
    --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-71.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-1.5-0.9-4 \
    --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-4.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name vgg16 \
    --model_nickname vgg16-512-0.05-0.9-11 \
    --model_path ../data/model/vgg16-512-0.05-0.9/data/model-11.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name vgg16 \
    --model_nickname vgg16-512-0.05-0.9-3 \
    --model_path ../data/model/vgg16-512-0.05-0.9/data/model-3.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name vgg16 \
    --model_nickname vgg16-512-0.05-0.9-13 \
    --model_path ../data/model/vgg16-512-0.05-0.9/data/model-13.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name vgg16 \
    --model_nickname vgg16-512-0.01-0.9-5 \
    --model_path ../data/model/vgg16-512-0.01-0.9/data/model-5.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name vgg16 \
    --model_nickname vgg16-512-0.01-0.9-21 \
    --model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'


python main.py \
    --gpu 1 \
    --proj_neuron_emb T \
    --model_name vgg16 \
    --model_nickname vgg16-512-0.01-0.9-207 \
    --model_path ../data/model/vgg16-512-0.01-0.9/data/model-207.pth \
    --img_emb_path '../data/embedding/inception_v3_pretrained/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=20000-k=10.txt' \
    --dim 30 \
    --k 10 \
    --emb_store_dirname 'emb-1'

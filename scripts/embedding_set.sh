gpu=4
model_name=vgg16
model_nickname=vgg16-512-0.01-0.9-207
model_path='../data/model/vgg16-512-0.01-0.9/data/model-207.pth'
img_emb_path='../data/embedding/vgg16-512-0.01-0.9-207/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=10000-k=10.txt' 
emb_store_dirname='emb-3-base-vgg16-207'







# batch_size=512
# basemodel_name='vgg19_pretrained'
# basemodel_nickname='vgg19_pretrained'

# dim=30
# topk=20
# lr_emb=0.01
# num_emb_epochs=100
# num_emb_negs=3

# python main.py \
#     --gpu $gpu \
#     --neuron_emb True \
#     --model_name $basemodel_name \
#     --model_nickname $basemodel_nickname \
#     --topk_s $topk \
#     --dim $dim \
#     --lr_emb $lr_emb \
#     --num_emb_epochs $num_emb_epochs \
#     --num_emb_negs $num_emb_negs

# Run neuron embedding (Base model Vgg16-0.01-207)
# python main.py \
#     --gpu $gpu \
#     --neuron_emb T \
#     --model_name $model_name \
#     --model_nickname $model_nickname \
#     --model_path $model_path \
#     --dim 30 \
#     --lr_emb 0.05 \
#     --num_emb_epochs 10000 \
#     --num_emb_negs 3 

# Run image embedding (Base model Vgg16-0.01-207)
# python main.py \
#     --gpu $gpu \
#     --img_emb T \
#     --model_name $model_name \
#     --model_nickname $model_nickname \
#     --model_path $model_path \
#     --dim 30 \
#     --lr_emb 0.05 \
#     --num_emb_epochs 10000 \
#     --num_emb_negs 3 \
#     --lr_img_emb 1 \
#     --thr_img_emb 0.001 \
#     --max_iter_img_emb 10000 \
#     --k 10


# Run projection
# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name inception_v3_pretrained \
#     --model_nickname inception_v3_pretrained \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name vgg16_pretrained \
#     --model_nickname vgg16_pretrained \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-0.5-0.9-3 \
#     --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-3.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-0.5-0.9-11 \
#     --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-0.5-0.9-121 \
#     --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-1.5-0.9-4 \
#     --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-4.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-1.5-0.9-69 \
#     --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-69.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name inception_v3 \
#     --model_nickname inception_v3-512-1.5-0.9-71 \
#     --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-71.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.05-0.9-3 \
#     --model_path ../data/model/vgg16-512-0.05-0.9/data/model-3.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.05-0.9-11 \
#     --model_path ../data/model/vgg16-512-0.05-0.9/data/model-11.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.05-0.9-13 \
#     --model_path ../data/model/vgg16-512-0.05-0.9/data/model-13.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.01-0.9-5 \
#     --model_path ../data/model/vgg16-512-0.01-0.9/data/model-5.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.01-0.9-21 \
#     --model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname

# python main.py \
#     --gpu $gpu \
#     --proj_neuron_emb T \
#     --model_name vgg16 \
#     --model_nickname vgg16-512-0.01-0.9-207 \
#     --model_path ../data/model/vgg16-512-0.01-0.9/data/model-207.pth \
#     --img_emb_path $img_emb_path \
#     --dim 30 \
#     --k 10 \
#     --emb_store_dirname $emb_store_dirname


# Emb 2d
# python main.py \
#     --gpu $gpu \
#     --dim_reduction UMAP \
#     --emb_set_dir ../data/embedding/$emb_store_dirname \
#     --dim 30 \
#     --model_for_emb_space base
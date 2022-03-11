# gpu=4
# model_name=vgg16-512-0.01-0.9-207
# img_emb_path='../data/embedding/vgg16-512-0.01-0.9-207/data/img_emb-dim=30-lr_img_emb=10.0-thr_img_emb=0.001-max_iter_img_emb=10000-k=10.txt' 
# emb_store_dirname='emb-5-base-vgg19_pretrained-10.0'

# # Run projection
# for model in 'inception_v3_pretrained' 'vgg16_pretrained'
# do
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name $model \
#         --model_nickname $model \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done

# for epoch in 3 11 121
# do
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name inception_v3 \
#         --model_nickname inception_v3-512-0.5-0.9-$epoch \
#         --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-$epoch.pth \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done

# for epoch in 4 69 70 71 100 298
# for epoch in 71 100 298
# do
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name inception_v3 \
#         --model_nickname inception_v3-512-1.5-0.9-$epoch \
#         --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-$epoch.pth \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done

# for epoch in 3 11 12 13
# do
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name vgg16 \
#         --model_nickname vgg16-512-0.05-0.9-$epoch \
#         --model_path ../data/model/vgg16-512-0.05-0.9/data/model-$epoch.pth \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done

# for epoch in 5 7 21 207
# do
#     python main.py \
#         --gpu $gpu \
#         --proj_neuron_emb T \
#         --model_name vgg16 \
#         --model_nickname vgg16-512-0.01-0.9-$epoch \
#         --model_path ../data/model/vgg16-512-0.01-0.9/data/model-$epoch.pth \
#         --img_emb_path $img_emb_path \
#         --dim 30 \
#         --k 10 \
#         --emb_store_dirname $emb_store_dirname
# done 

# Emb 2d
# python main.py \
#     --gpu $gpu \
#     --dim_reduction UMAP \
#     --emb_set_dir ../data/embedding/$emb_store_dirname \
#     --dim 30 \
#     --model_for_emb_space base \
#     --reducer_path ../data/embedding/emb2d/data/emb-5-base-vgg19_pretrained-10.0-0305/reducer-dim=30-model_for_emb_space=base.sav


# # batch_size=10

# # for label in 27 59 85 104 156 230 273 277 282 301 315 378 194 400 
# # do
# #     model_nickname=vgg16-512-0.01-0.9
# #     from_model_nickname=vgg16-512-0.01-0.9-5
# #     from_model_path='../data/model/vgg16-512-0.01-0.9/data/model-5.pth'
# #     to_model_nickname=vgg16-512-0.01-0.9-21
# #     to_model_path='../data/model/vgg16-512-0.01-0.9/data/model-21.pth'

# #     python main.py \
# #         --gpu $gpu \
# #         --find_important_evo T \
# #         --batch_size $batch_size \
# #         --label $label \
# #         --model_name $model_name \
# #         --model_nickname $model_nickname \
# #         --from_model_nickname $from_model_nickname \
# #         --from_model_path $from_model_path \
# #         --to_model_nickname $to_model_nickname \
# #         --to_model_path $to_model_path

# #     from_model_nickname=vgg16-512-0.01-0.9-21
# #     from_model_path='../data/model/vgg16-512-0.01-0.9/data/model-21.pth'
# #     to_model_nickname=vgg16-512-0.01-0.9-207
# #     to_model_path='../data/model/vgg16-512-0.01-0.9/data/model-207.pth'

# #     python main.py \
# #         --gpu $gpu \
# #         --find_important_evo T \
# #         --batch_size $batch_size \
# #         --label $label \
# #         --model_name $model_name \
# #         --model_nickname $model_nickname \
# #         --from_model_nickname $from_model_nickname \
# #         --from_model_path $from_model_path \
# #         --to_model_nickname $to_model_nickname \
# #         --to_model_path $to_model_path

# # done
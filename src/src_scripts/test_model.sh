gpu=1
python main.py \
    --gpu $gpu \
    --test T \
    --batch_size 256 \
    --model_name vgg19_pretrained \
    --model_nickname vgg19_pretrained \
    --topk 5 

# for epoch in 3 11 121
# do
#     python main.py \
#         --gpu 0 \
#         --test T \
#         --batch_size 256 \
#         --model_name inception_v3 \
#         --model_nickname inception_v3-512-0.5-0.9-$epoch \
#         --model_path ../data/model/inception_v3-512-0.5-0.9/data/model-$epoch.pth \
#         --topk 5 
# done

# for epoch in 4 69 71
# do
#     python main.py \
#         --gpu 0 \
#         --test T \
#         --batch_size 256 \
#         --model_name inception_v3 \
#         --model_nickname inception_v3-512-1.5-0.9-$epoch \
#         --model_path ../data/model/inception_v3-512-1.5-0.9/data/model-$epoch.pth \
#         --topk 5 
# done



# for epoch in 5 21 207
# do
#     python main.py \
#         --gpu 0 \
#         --test T \
#         --batch_size 256 \
#         --model_name vgg16 \
#         --model_nickname vgg16-512-0.01-0.9-$epoch \
#         --model_path ../data/model/vgg16-512-0.01-0.9/data/model-$epoch.pth \
#         --topk 5 
# done

# for epoch in 3 11 13
# do
#     python main.py \
#         --gpu 0 \
#         --test T \
#         --batch_size 256 \
#         --model_name vgg16 \
#         --model_nickname vgg16-512-0.05-0.9-$epoch \
#         --model_path ../data/model/vgg16-512-0.05-0.9/data/model-$epoch.pth \
#         --topk 5 
# done
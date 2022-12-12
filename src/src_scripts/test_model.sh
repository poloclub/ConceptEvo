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

# for epoch in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99
# do
#     python main.py \
#         --gpu $gpu \
#         --test T \
#         --training_data ../../cifar10/train \
#         --test_data ../../cifar10/test \
#         --model_name vgg16_cifar10 \
#         --model_nickname vgg16_cifar10 \
#         --model_path ../data/model/vgg16_cifar10/data/model-$epoch.pth \
#         --topk 5 
# done
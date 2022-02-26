# Run the script in `../src` where `main.py` exists.
# To get image embeddings from a specific model,
# please give values for --model_nickname and --model_path.
python main.py \
    --gpu 2 \
    --find_important_evo T \
    --batch_size 128 \
    --label 1 \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-0.5-0.9 \
    --from_model_nickname inception_v3-512-0.5-0.9-11 \
    --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
    --to_model_nickname inception_v3-512-0.5-0.9-121 \
    --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth \
    

# Run the script in `../src` where `main.py` exists.
# To get image embeddings from a specific model,
# please give values for --model_nickname and --model_path.
python main.py \
    --gpu 7 \
    --find_important_evo T \
    --batch_size 128 \
    --label 1 \
    --model_name vgg16 \
    --model_nickname vgg16-512-0.01-0.9 \
    --from_model_nickname vgg16-512-0.01-0.9-5 \
    --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-5.pth \
    --to_model_nickname vgg16-512-0.01-0.9-21 \
    --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-21.pth
    
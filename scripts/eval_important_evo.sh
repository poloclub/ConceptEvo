# Run the script in `../src` where `main.py` exists.
# To get image embeddings from a specific model,
# please give values for --model_nickname and --model_path.
python main.py \
    --gpu 0 \
    --eval_important_evo perturbation \
    --batch_size 128 \
    --label 1 \
    --eps 0.1 \
    --eval_sample_ratio 0.1 \
    --model_name inception_v3 \
    --model_nickname inception_v3-512-0.5-0.9 \
    --from_model_nickname inception_v3-512-0.5-0.9-11 \
    --from_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-11.pth \
    --to_model_nickname inception_v3-512-0.5-0.9-121 \
    --to_model_path ../data/model/inception_v3-512-0.5-0.9/data/model-121.pth 
    

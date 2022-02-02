# Run the script in `../src` where `main.py` exists.
# Get image embeddings from a pretrained InceptionV3.
# To get image embeddings from a specific model,
# please give values for --model_nickname and --model_path.
python main.py \
    --gpu 2 \
    --img_emb T \
    --model_name inception_v3_pretrained \
    --dim 30 \
    --lr_emb 0.01 \
    --num_emb_epochs 30000 \
    --num_emb_negs 3 \
    --lr_img_emb 10 \
    --thr_img_emb 0.05 \
    --max_iter_img_emb 10000 \
    --k 10
# Run the script in `../src` where `main.py` exists.
# Dimensionality reduction
python main.py \
    --gpu 3 \
    --dim_reduction UMAP \
    --emb_set_dir ../data/embedding/emb-1 \
    --dim 30 \
    --model_for_emb_space base


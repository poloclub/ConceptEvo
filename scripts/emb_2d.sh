# Run the script in `../src` where `main.py` exists.
# Dimensionality reduction
python main.py \
    --gpu 3 \
    --dim_reduction UMAP \
    --emb_set_dir ../data/embedding/emb-0 \
    --dim 30 \
    --sample_rate 1
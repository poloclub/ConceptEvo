# Run the script in `../src` where `main.py` exists.
# Dimensionality reduction
python main.py \
    --gpu 0 \
    --dim_reduction UMAP \
    --emb_set_dir ../data/embedding \
    --dim 30 \
    --sample_rate 0.5 
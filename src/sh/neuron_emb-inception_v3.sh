# Run the script in `../src` where `main.py` exists.
# To get image embeddings from a specific model,
# please give values for --model_nickname and --model_path.
python main.py \
    --gpu 0 \
    --neuron_emb T \
    --model_name inception_v3_pretrained \
    --dim 30 \
    --lr_emb 0.05 \
    --num_emb_epochs 1000 \
    --num_emb_negs 10 
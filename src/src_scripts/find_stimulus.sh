gpu=7
batch_size=512
topk=10
profiling=False

echo "Find stimulus for VGG16-0.01-207"
python main.py \
    --gpu $gpu \
    --stimulus True \
    --batch_size $batch_size \
    --topk_s $topk \
    --profiling_stimulus $profiling \
    --model_name vgg16 \
    --model_nickname vgg16-512-0.01-0.9-207 \
    --model_path ../data/model/vgg16-512-0.01-0.9/data/model-207.pth


echo "Find stimulus for ConvNeXt-0.004-1"
python main.py \
    --gpu $gpu \
    --stimulus True \
    --batch_size $batch_size \
    --topk_s $topk \
    --profiling_stimulus $profiling \
    --model_name convnext \
    --model_nickname convnext-0.004-1 \
    --model_path ../data/model/convnext_4e-3/data/model-1.pth

gpu=6
batch_size=128
find_num_sample_imgs=128


for label in 622
# 514 520 524 538 548 553 555 578 582 592 622 638 
do
    echo "======================================================================="
    from_epoch=5
    to_epoch=21

    for idx in 0 1 2 3 4
    do
        msg="find important evo, Vgg16, label=$label, epoch=$from_epoch->$to_epoch, find_num_sample_imgs=$find_num_sample_imgs, idx=$idx"
        echo $msg
        python main.py \
            --gpu $gpu \
            --find_important_evo T \
            --batch_size $batch_size \
            --label $label \
            --find_num_sample_imgs $find_num_sample_imgs \
            --idx $idx \
            --model_name vgg16 \
            --model_nickname vgg16-512-0.01-0.9 \
            --from_model_nickname vgg16-512-0.01-0.9-$from_epoch \
            --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-$from_epoch.pth \
            --to_model_nickname vgg16-512-0.01-0.9-$to_epoch \
            --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-$to_epoch.pth

        msg="eval important evo, Vgg16, label=$label, epoch=$from_epoch->$to_epoch, find_num_sample_imgs=$find_num_sample_imgs, idx=$idx, freezing" 
        echo $msg
        for ratio in 0.1 0.3 0.5 0.7 0.9
        do
            echo "-------------------------------------------------------------------"
            echo "[start] (ratio=$ratio) $msg"
            python main.py \
                --gpu $gpu \
                --eval_important_evo freezing \
                --batch_size $batch_size \
                --label $label \
                --find_num_sample_imgs $find_num_sample_imgs \
                --idx $idx \
                --eval_sample_ratio $ratio \
                --model_name vgg16 \
                --model_nickname vgg16-512-0.01-0.9 \
                --from_model_nickname vgg16-512-0.01-0.9-$from_epoch \
                --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-$from_epoch.pth \
                --to_model_nickname vgg16-512-0.01-0.9-$to_epoch \
                --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-$to_epoch.pth 
            echo "[end] (ratio=$ratio) $msg"
        done
    done
    echo "======================================================================="

    from_epoch=21
    to_epoch=207
    for idx in 0 1 2 3 4
    do
        msg="find important evo, Vgg16, label=$label, epoch=$from_epoch->$to_epoch, idx=$idx"
        echo $msg
        python main.py \
            --gpu $gpu \
            --find_important_evo T \
            --batch_size $batch_size \
            --label $label \
            --find_num_sample_imgs $find_num_sample_imgs \
            --idx $idx \
            --model_name vgg16 \
            --model_nickname vgg16-512-0.01-0.9 \
            --from_model_nickname vgg16-512-0.01-0.9-$from_epoch \
            --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-$from_epoch.pth \
            --to_model_nickname vgg16-512-0.01-0.9-$to_epoch \
            --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-$to_epoch.pth 

        msg="eval important evo, Vgg16, label=$label, epoch=$from_epoch->$to_epoch, idx=$idx, freezing" 
        echo $msg
        for ratio in 0.1 0.3 0.5 0.7 0.9
        do
            echo "-------------------------------------------------------------------"
            echo "[start] (ratio=$ratio) $msg"
            python main.py \
                --gpu $gpu \
                --eval_important_evo freezing \
                --batch_size $batch_size \
                --label $label \
                --find_num_sample_imgs $find_num_sample_imgs \
                --idx $idx \
                --eval_sample_ratio $ratio \
                --model_name vgg16 \
                --model_nickname vgg16-512-0.01-0.9 \
                --from_model_nickname vgg16-512-0.01-0.9-$from_epoch \
                --from_model_path ../data/model/vgg16-512-0.01-0.9/data/model-$from_epoch.pth \
                --to_model_nickname vgg16-512-0.01-0.9-$to_epoch \
                --to_model_path ../data/model/vgg16-512-0.01-0.9/data/model-$to_epoch.pth 
            echo "[end] (ratio=$ratio) $msg"
        done
    done

done
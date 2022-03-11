gpu=6
batch_size=10
model_name=vgg16
find_num_sample_imgs=50

for label in 457 856 53 262 304 242 578 469 514 282 538 690 658 175 415 48 263 732 789 67 475 964 388 801 273 268 315 406 592 400 555 622 481 85 712 638 553 944 159 371 758 384 280 694 762 663 975 972 494 721 943 513 957 446 963 230 703 903 277 27 353 10 913 436 34 324 735 857 20 910 500 524 286 59 394 830 1 979 548 133 97 11 993 906 707 819 897 42 520 582 346 867 156 435 800 378 301 104 994 842 
do
    model_nickname=vgg16-512-0.01-0.9
    from_model_nickname=vgg16-512-0.01-0.9-21
    from_model_path='../data/model/vgg16-512-0.01-0.9/data/model-21.pth'
    to_model_nickname=vgg16-512-0.01-0.9-207
    to_model_path='../data/model/vgg16-512-0.01-0.9/data/model-207.pth'

    for ratio in 0.1
    do
        echo "Eval Vgg16 21->207 label=$label gpu=$gpu ratio=$ratio"
        python main.py \
            --gpu $gpu \
            --eval_important_evo freezing \
            --batch_size $batch_size \
            --label $label \
            --eval_sample_ratio $ratio \
            --model_name $model_name \
            --model_nickname $model_nickname \
            --from_model_nickname $from_model_nickname \
            --from_model_path $from_model_path \
            --to_model_nickname $to_model_nickname \
            --to_model_path $to_model_path \
            --find_num_sample_imgs $find_num_sample_imgs
    done

done
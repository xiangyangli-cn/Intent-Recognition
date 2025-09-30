#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 1
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'misa_t_a' \
        --method 'misa_t_a' \
        --data_mode 'multi-class' \
        --save_results \
        --save_model  \
        --seed $seed \
        --gpu_id '0' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'lxy.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'misa_bert' \
        --results_file_name 'misa_bert_t_a.csv'
    done
done

#--audio_feats_path 'lxy.pkl' \
#--audio_feats_path 'audio_feats_0.pkl' \

#--train \

#--train \
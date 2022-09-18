# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# SEEDS=(171 354 550 667 985)
SEEDS=(171)
N=(1)
K=1
#mode=train

for seed in ${SEEDS[@]}; do
    for n in ${N[@]}; do
        # python3 main.py \
        #     --dataset=Domain \
        #     --gpu_device=0 \
        #     --seed=${seed} \
        #     --mode=train \
        #     --N=${n} \
        #     --K=${K} \
        #     --similar_k=10 \
        #     --eval_every_meta_steps=100 \
        #     --name=10-k_100_2_4_3_max_loss_2_5_BIOES \
        #     --train_mode=span \
        #     --inner_steps=2 \
        #     --inner_size=4 \
        #     --max_ft_steps=3 \
        #     --lambda_max_loss=2 \
        #     --inner_lambda_max_loss=5 \
        #     --tagging_scheme=BIOES \
        #     --viterbi=hard \
        #     --concat_types=None \
        #     --top_top_dir=cross${n} \
        #     --ignore_eval_test   
                    

        python3 main.py \
            --dataset=Domain \
            --seed=${seed} \
            --gpu_device=0 \
            --lr_inner=1e-4 \
            --lr_meta=1e-4 \
            --mode=train \
            --N=${n} \
            --K=${K} \
            --similar_k=10 \
            --inner_similar_k=10 \
            --eval_every_meta_steps=100 \
            --name=10-k_100_type_2_4_3_10_10 \
            --train_mode=type \
            --inner_steps=2 \
            --inner_size=1 \
            --max_ft_steps=3 \
            --concat_types=None \
            --top_top_dir=cross${n} \
            --lambda_max_loss=2.0 \
            --debug \
            --mask_proto \
            --proto_cL_Loss
            # --instance_mask_cL_Loss
            

            # --instance_context_cL_Loss \
            
            # --adv_proto

        # cp models-${n}-${K}-train-cross${n}/bert-base-uncased-innerSteps_2-innerSize_4-lrInner_0.0001-lrMeta_0.0001-maxSteps_5001-seed_${seed}-name_10-k_100_type_2_4_3_10_10/en_type_pytorch_model.bin models-${n}-${K}-train-cross${n}/bert-base-uncased-innerSteps_2-innerSize_4-lrInner_3e-05-lrMeta_3e-05-maxSteps_5001-seed_${seed}-name_10-k_100_2_4_3_max_loss_2_5_BIOES

        # python3 main.py \
        #     --gpu_device=0 \
        #     --dataset=Domain \
        #     --seed=${seed} \
        #     --N=${n} \
        #     --K=${K} \
        #     --mode=test \
        #     --similar_k=10 \
        #     --name=10-k_100_2_4_3_max_loss_2_5_BIOES \
        #     --concat_types=None \
        #     --test_only \
        #     --eval_mode=two-stage \
        #     --inner_steps=2 \
        #     --inner_size=4 \
        #     --max_ft_steps=3 \
        #     --max_type_ft_steps=3 \
        #     --lambda_max_loss=2.0 \
        #     --inner_lambda_max_loss=5.0 \
        #     --inner_similar_k=10 \
        #     --viterbi=hard \
        #     --top_top_dir=cross${n} \
        #     --tagging_scheme=BIOES
        #     # --mask_proto
    done
done
SEEDS=(171)
N=5
K=1
mode=inter
for seed in ${SEEDS[@]}; do
    python3 main.py \
            --gpu_device=0 \
            --seed=${seed} \
            --N=${N} \
            --K=${K} \
            --mode=${mode} \
            --similar_k=10 \
            --name=10-k_100_2_4_3_max_loss_2_5_BIOES \
            --concat_types=None \
            --test_only \
            --eval_mode=two-stage \
            --inner_steps=2 \
            --inner_size=4 \
            --max_ft_steps=3 \
            --max_type_ft_steps=3 \
            --lambda_max_loss=2.0 \
            --inner_lambda_max_loss=5.0 \
            --inner_similar_k=10 \
            --viterbi=hard \
            --tagging_scheme=BIOES
done
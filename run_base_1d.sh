# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Pretrain_1d --mode stream \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Oracle_1d --mode stream --lr 0.001  \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Vanilla_1d --mode stream --C 0.5 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
--method Vanilla-win_1d --mode stream --C 1.0 \
--pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
--data_path ../data/criteo/data.txt \
--data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method ES-DFM_1d --mode stream  \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
# --pretrain_esdfm_model_ckpt_path ./delayed_feedback_release/ckpts/esdfm_mask30d_1d/esdfm_mask30d_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method FNC_1d --mode stream --lr 0.0005 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method FNW_1d --mode stream  --lr 0.0005 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

#DEFER_1d
# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method DEFER --mode stream --C 1.0 --W 1 --lr 0.0005 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
# --pretrain_defer_model_ckpt_path ./delayed_feedback_release/ckpts/defer_test/defer_test \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache
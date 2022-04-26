# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Pretrain --mode pretrain \
# --model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Pretrain_1d --mode pretrain \
# --model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

## rn_dp pretrain
# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method ES-DFM --mode pretrain --C 1 \
# --model_ckpt_path ./delayed_feedback_release/ckpts/esdfm_mask30d/esdfm_mask30d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

## DEFER pretrain
CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
--method DEFER --mode pretrain \
--model_ckpt_path ./delayed_feedback_release/ckpts/defer/defer \
--data_path ../data/criteo/data.txt \
--data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method DEFER_1d --mode pretrain \
# --model_ckpt_path ./delayed_feedback_release/ckpts/defer_1d/defer_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

## Bi-DEFUSE pretrain
# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Bi-DEFUSE_1d --mode pretrain --C 0.5 --lr 0.0005 \
# --model_ckpt_path ./delayed_feedback_release/ckpts/bidefuse_1d/bidefuse_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache
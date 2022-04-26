# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method DEFUSE --mode stream --C 1.0 --lr 0.0005 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --pretrain_esdfm_model_ckpt_path ./delayed_feedback_release/ckpts/esdfm_mask30d/esdfm_mask30d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
--method Bi-DEFUSE --mode stream --C 0.25 --lr 0.0005 \
--pretrain_defuse_model_ckpt_path ./pretrain/ckpts/pretrain/pretrain \
--data_path ../data/criteo/data.txt \
--data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Bi-DEFUSE_1d --mode stream --C 0.5 --lr 0.0005 \
# --pretrain_defuse_model_ckpt_path ./delayed_feedback_release/ckpts/DEFUSE_MTL_1d/DEFUSE_MTL_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method DEFUSE_1d --mode stream --C 0.5 --lr 0.0005 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain_1d/pretrain_1d \
# --pretrain_esdfm_model_ckpt_path ./delayed_feedback_release/ckpts/esdfm_mask30d_1d/esdfm_mask30d_1d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method FNW_unbiased --mode stream --C 0.5 --lr 0.0005 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method DEFER_unbiased --mode stream --C 1.0 --W 30 --lr 0.0005 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/bidefuse/bidefuse \
# --pretrain_defer_model_ckpt_path ./delayed_feedback_release/ckpts/defer_test/defer_test \
# --pretrain_esdfm_model_ckpt_path ./delayed_feedback_release/ckpts/esdfm_mask30d/esdfm_mask30d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache
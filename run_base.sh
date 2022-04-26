# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Pretrain --mode stream \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Oracle --mode stream --lr 0.0005 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Vanilla --mode stream \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method Vanilla-win --mode stream \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method ES-DFM --mode stream --C 0.25 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --pretrain_esdfm_model_ckpt_path ./delayed_feedback_release/ckpts/esdfm_mask30d/esdfm_mask30d \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method FNC --mode stream --C 0.5 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
# --method FNW --mode stream --C 0.5 \
# --pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
# --data_path ../data/criteo/data.txt \
# --data_cache_path ./delayed_feedback_release/data_cache

#DEFER 30d
CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
--method DEFER --mode stream --C 0.5 --W 30 --lr 0.0005 \
--pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
--pretrain_defer_model_ckpt_path ./delayed_feedback_release/ckpts/defer/defer \
--data_path ../data/criteo/data.txt \
--data_cache_path ./delayed_feedback_release/data_cache
set -x

# 切换到正确的目录
cd "$(dirname "$0")/../../.." || exit 1

GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export CUDA_VISIBLE_DEVICES=6

export PYTHONPATH="${PYTHONPATH}:$(pwd)/internvl_chat_gpt_oss"
export MASTER_PORT=37886
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL

pretrained_model_path='/mnt/jianghc1995/model/InternVL3_5-8B/'
OUTPUT_DIR='/mnt/jianghc1995/train_debug/internvl3_5_single_gpu_test'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl_chat_gpt_oss/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path ${pretrained_model_path} \
  --conv_style "internvl3_5_gpt_oss" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "internvl_chat_gpt_oss/shell/data/data4debug.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_steps 20 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_num_frame 48 \
  --max_seq_length 13000 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size False \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
  # --deepspeed "internvl_chat_gpt_oss/zero_stage1_config.json" \

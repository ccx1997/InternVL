set -x
CUDA_VISIBLE_DEVICES=0
GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34228
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

pretrained_model_path='/mnt/models/InternVL3-VideoChat-Merged-2B'
OUTPUT_DIR='work_dirs/internvl_videochatflash/internvl2b_videochatflash2b_sft_s1_debug'
META_PATH='shell/data/vsi_meta_env.json'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
# nohup torchrun \
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path $pretrained_model_path \
  --video_encoder_tokenizer_path /mnt/models/VideoChat-Flash-Qwen2_5-2B_res448 \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path $META_PATH \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --freeze_video_encoder True \
  --vision_select_layer -1 \
  --dataloader_num_workers 0 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config_34b.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
  # > "${OUTPUT_DIR}/training_log.txt" 2>&1 &

#!/bin/bash
set -x

# NCCL and communication settings
# export NCCL_TIMEOUT=1800  # 30 minutes
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30 minutes
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# # Suppress video decoding warnings
# export OPENCV_FFMPEG_LOGLEVEL=-8
# export FFMPEG_LOG_LEVEL=quiet

# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
# Check if PER_DEVICE_BATCH_SIZE is supported
if [ "$PER_DEVICE_BATCH_SIZE" != "1" ]; then
    echo "Error: Current training only supports PER_DEVICE_BATCH_SIZE=1, but got PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
    echo "Please set PER_DEVICE_BATCH_SIZE=1 or remove the environment variable to use the default value."
    exit 1
fi
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=39879
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# pretrained_model_path='work_dirs/internvl_chat_dual_encoder/internvl_chat_dual_encoder_8b_mix_stage1/checkpoint-10600'
# pretrained_model_path='/mnt/chensenda/codes/VLN/InternVL_video/internvl_chat/work_dirs/internvl_chat_dual_encoder/internvl_chat_dual_encoder_8b_mix_stage2_videocompression_s1.3/checkpoint-1200'
pretrained_model_path='/mnt/models/InternVL3-8B'
vision_path2='/mnt/models/VGGT-1B/model.pt'
OUTPUT_DIR='work_dirs/internvl_chat_dual_compressor/internvl_chat_dual_compressor_8b_mix_s1'

# copy current script to output directory
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 1
# gradient accumulation steps: 8
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path ${pretrained_model_path} \
  --vision_path2 ${vision_path2} \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/vsi_mix_meta_s1.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 1 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --freeze_mlp2 False \
  --freeze_vision2 True \
  --freeze_vision_compressor False \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_num_frame 48 \
  --max_seq_length 7000 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size False \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

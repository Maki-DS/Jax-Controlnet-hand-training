#!/bin/bash

export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/mnt/disks/persist/train_output/controlnet-encoded-hands-{timestamp}" # location to save trained model
export DATASET_DIR="/mnt/disks/persist/hand-enc-250k-v2" # Dataset directory
export DISK_DIR="/mnt/disks/persist" 
export HUB_MODEL_ID="controlnet-encoded-hands" # Model Name

python3 train_controlnet_flax.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATASET_DIR \
 --cache_dir=$DISK_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --validation_image "./val1.png" "./val2.png" \
 --validation_prompt "a man in a colorful shirt giving a peace sign in front of a rallying crowd" "a police officer signaling someone to stop in a park" \
 --validation_steps=500 \
 --revision="non-ema" \
 --from_pt \
 --report_to="wandb" \
 --tracker_project_name=$HUB_MODEL_ID \
 --checkpointing_steps=5000 \
 --dataloader_num_workers=16 \
 --max_train_steps=80000 \
 --gradient_accumulation_steps=1 \
 --push_to_hub
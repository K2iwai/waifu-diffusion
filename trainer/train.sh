#!/bin/bash

# Just an example of how to run the training script.

export HF_API_TOKEN="your_token"
BASE_MODEL="runwayml/stable-diffusion-v1-5"
RUN_NAME="bdjp"
DATASET="/workspace/waifu-diffusion/mydataset"
N_GPU=1
N_EPOCHS=50
BATCH_SIZE=1
RESUME_MODEL="/workspace/waifu-diffusion/trainer/output/XXXX"

python3 -m torch.distributed.run --nproc_per_node=$N_GPU diffusers_trainer.py --model=$BASE_MODEL --run_name=$RUN_NAME --dataset=$DATASET --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=$BATCH_SIZE --fp16=True --image_log_steps=10000 --epochs=$N_EPOCHS --resolution=512 --use_ema=False --clip_penultimate=False --wandb=False --resume=$RESUME_MODEL --save_steps=10000

# and to resume... just add the --resume flag and supply it with the path to the checkpoint.

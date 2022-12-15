#!/bin/bash

# Just an example of how to run the training script.

export HF_API_TOKEN="your_token"
BASE_MODEL="runwayml/stable-diffusion-v1-5"
RUN_NAME="bdjp"
DATASET="/workspace/waifu-diffusion/mydataset"
N_GPU=1
N_EPOCHS=5
BATCH_SIZE=2
RESUME_MODEL="/workspace/waifu-diffusion/trainer/output/XXXX"
N_SAVESTEP=10000
N_LOGIMAGESTEP=1000
python3 -m torch.distributed.run --nproc_per_node=$N_GPU diffusers_trainer.py --model=$BASE_MODEL --run_name=$RUN_NAME --dataset=$DATASET --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=$BATCH_SIZE --fp16=True --image_log_steps=$N_LOGIMAGESTEP --epochs=$N_EPOCHS --resolution=512 --use_ema=False --clip_penultimate=False --wandb=False --save_steps=$N_SAVESTEP --lr_scheduler=constant
# python3 -m torch.distributed.run --nproc_per_node=$N_GPU diffusers_trainer.py --model=$BASE_MODEL --run_name=$RUN_NAME --dataset=$DATASET --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=$BATCH_SIZE --fp16=True --image_log_steps=$N_LOGIMAGESTEP --epochs=$N_EPOCHS --resolution=512 --use_ema=False --clip_penultimate=False --wandb=False --resume=$RESUME_MODEL --save_steps=$N_SAVESTEP --lr_scheduler=constant

# and to resume... just add the --resume flag and supply it with the path to the checkpoint.

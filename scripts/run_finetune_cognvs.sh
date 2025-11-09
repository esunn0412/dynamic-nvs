#!/bin/bash
#SBATCH --job-name=cognvs_finetune
#SBATCH --output=/scratch/tkim462/vision/scripts/logs/ft_rodtang_%j.out
#SBATCH --error=/scratch/tkim462/vision/scripts/logs/ft_rodtang_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=l40s-8-gm384-c192-m1536
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=300GB
#SBATCH --gpus=6


set -euo pipefail

echo "======================================"
echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "======================================"

# Option 3: module-based Python
# module load python/3.10
# module load cuda/11.8

# Print environment info
echo "======================================"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Number of GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo "======================================"

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model Configuration
MODEL_ARGS=(
    --model_path "/scratch/tkim462/vision/models/CogVideoX-5b-I2V"
    --transformer_id "/scratch/tkim462/vision/models/checkpoints/cognvs_ckpt_inpaint"
    --model_name "cogvideox-v2v"
    --model_type "v2v"
    --training_type "sft"
)

# Output Configuration (per-seq)
OUTPUT_ARGS=(
    --output_dir "/scratch/tkim462/vision/models/checkpoints/cognvs_ckpt_finetuned_rodtang"
    --report_to "wandb"
)

# Data Configuration (per-seq)
DATA_ARGS=(
    --json_file ""
    --base_dir_input "/scratch/tkim462/vision/demo_data/rodtang"
    --base_dir_target ""
    --train_resolution "49x480x720"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 200
    --seed 42
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 2
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 200
    --checkpointing_limit 5
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation False
    --validation_dir ""
    --validation_steps 200
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
)

# Launch training
echo "======================================"
echo "Starting training..."
echo "======================================"

accelerate launch \
    --main_process_port 29501 \
    --config_file /scratch/tkim462/vision/finetune/accelerate_config.yaml \
    /scratch/tkim462/vision/finetune/train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"

EXIT_CODE=$?

echo "======================================"
echo "Job finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================"

exit ${EXIT_CODE}

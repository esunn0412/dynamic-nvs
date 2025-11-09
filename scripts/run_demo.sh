#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --output=/scratch/tkim462/vision/scripts/logs/demo_%j.out
#SBATCH --error=/scratch/tkim462/vision/scripts/logs/demo_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100-8-gm320-c96-m1152
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=100G

set -euo pipefail

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"

# Activate your Python environment (uncomment and edit as needed)
# source ~/venv/bin/activate
# or
# module load anaconda
# conda activate myenv

# Run demo
python /scratch/tkim462/vision/demo.py \
    --model_path "/scratch/tkim462/vision/models/CogVideoX-5b-I2V" \
    --cognvs_ckpt_path "/scratch/tkim462/vision/models/checkpoints/cognvs_ckpt_finetuned_rodtang/my_checkpoint-200_transformer" \
    --data_path "/scratch/tkim462/vision/demo_data/rodtang" \
    --mp4_name "eval_render2.mp4"

python /scratch/tkim462/vision/demo.py \
    --model_path "/scratch/tkim462/vision/models/CogVideoX-5b-I2V" \
    --cognvs_ckpt_path "/scratch/tkim462/vision/models/checkpoints/cognvs_ckpt_finetuned_rodtang/my_checkpoint-200_transformer" \
    --data_path "/scratch/tkim462/vision/demo_data/rodtang" \
    --mp4_name "eval_render3.mp4"

python /scratch/tkim462/vision/demo.py \
    --model_path "/scratch/tkim462/vision/models/CogVideoX-5b-I2V" \
    --cognvs_ckpt_path "/scratch/tkim462/vision/models/checkpoints/cognvs_ckpt_finetuned_rodtang/my_checkpoint-200_transformer" \
    --data_path "/scratch/tkim462/vision/demo_data/rodtang" \
    --mp4_name "eval_render4.mp4"

echo "Job finished: $(date)"

#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --output=/scratch/tkim462/vision/scripts/logs/demo_%j.out
#SBATCH --error=/scratch/tkim462/vision/scripts/logs/demo_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=l4-4-gm96-c48-m192
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
    --cognvs_ckpt_path "/scratch/tkim462/vision/models/checkpoints/cognvs_ckpt_finetuned_davis_bear/my_checkpoint-200_transformer" \
    --data_path "/scratch/tkim462/vision/demo_data/davis_bear" \
    --mp4_name "example_eval_render.mp4"

echo "Job finished: $(date)"

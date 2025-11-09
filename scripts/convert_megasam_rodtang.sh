#!/bin/bash
#SBATCH --job-name=convert_megasam
#SBATCH --output=/scratch/tkim462/vision/scripts/logs/convert_megasam_%j.out
#SBATCH --error=/scratch/tkim462/vision/scripts/logs/convert_megasam_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=l4-4-gm96-c48-m192
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50GB
#SBATCH --gpus=1


set -euo pipefail

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"

# Activate your Python environment
# conda activate mega_sam

# Convert MegaSAM outputs to CogNVS format
python /scratch/tkim462/vision/toolbox/convert_megasam_outputs.py \
    --megasam_npz_path "/scratch/tkim462/vision/mega-sam/output/rodtang/megasam/sgd_cvd_hr.npz" \
    --output_path "/scratch/tkim462/vision/demo_data/rodtang"

echo "Job finished: $(date)"

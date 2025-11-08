#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --output=/scratch/tkim462/vision/scripts/logs/data_gen_%j.out
#SBATCH --error=/scratch/tkim462/vision/scripts/logs/data_gen_%j.err
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

# Run data generation
# Any arguments after `--` when calling sbatch will be forwarded to data_gen.py via "$@"
python /scratch/tkim462/vision/data_gen.py \
    --device "cuda:0" \
    --data_path "demo_data/davis_bear" \
    --mode "train" \
    --intrinsics_file "cam_info/megasam_intrinsics.npy" \
    --extrinsics_file "cam_info/megasam_c2ws.npy"

echo "Job finished: $(date)"

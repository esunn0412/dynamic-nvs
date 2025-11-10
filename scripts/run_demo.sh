#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --output=/scratch/tkim462/vision/scripts/logs/demo_%j.out
#SBATCH --error=/scratch/tkim462/vision/scripts/logs/demo_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=l40s-8-gm384-c192-m1536
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
# python /scratch/tkim462/vision/data_gen.py \
#     --device "cuda:0" \
#     --data_path "/scratch/tkim462/vision/demo_data/rodtang" \
#     --mode "eval" \
#     --intrinsics_file "cam_info/megasam_intrinsics.npy" \
#     --extrinsics_file "cam_info/megasam_c2ws.npy"

# Process eval_render1.mp4 through eval_render11.mp4
for i in {1..11}; do
    echo "Processing eval_render${i}.mp4..."
    python /scratch/tkim462/vision/demo.py \
        --model_path "/scratch/tkim462/vision/models/CogVideoX-5b-I2V" \
        --cognvs_ckpt_path "/scratch/tkim462/vision/models/checkpoints/cognvs_ckpt_finetuned_rodtang/my_checkpoint2-200_transformer" \
        --data_path "/scratch/tkim462/vision/demo_data/rodtang" \
        --mp4_name "eval_render${i}.mp4" \
        --frame_offset 49
done

echo "Job finished: $(date)"

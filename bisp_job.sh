#!/bin/bash
#SBATCH --job-name=bisp_gpu
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=gpu_conda_%j.out
#SBATCH --error=gpu_conda_%j.err
echo "=== GPU Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
# Load required modules
module load miniconda3/4.12.0
module load cuda/11.7
# Initialize conda
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
# Activate environment with GPU support
conda activate pytorch_gpu
# Verify GPU access
echo "Python: $(which python)"
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
# GPU-accelerated script
python Train.py --batch_size 1 --epochs 10

echo "=== GPU Job Completed ==="

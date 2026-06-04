#!/bin/bash
#SBATCH --job-name=dns_lr001
#SBATCH --account=ealloc_ati-1-neur
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=train_lr001_%j.log
#SBATCH --error=train_lr001_%j.log

echo "========================================"
echo "DNS Training Job — lr=0.01"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "========================================"

# load CUDA
module load cuda/11.7.0

# activate venv
source "$HOME/nn_proj/venv_gpu/bin/activate"

# go to project folder
cd "$HOME/nn_proj"

# set learning rate via environment variable
export LEARNING_RATE=0.01

# run training — -u flag forces unbuffered output (live log updates)
python -u train.py

echo ""
echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"

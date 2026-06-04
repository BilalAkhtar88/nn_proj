#!/bin/bash
#SBATCH --job-name=dns_block3_search
#SBATCH --account=ealloc_ati-1-neur
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4
#SBATCH --nodelist=falcon2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/helios/home/etais/hpc_bilal/nn_proj/logs/block3_search_%j.log
#SBATCH --no-requeue

echo "========================================================================"
echo "DNS Block 3 — Hyperparameter Search"
echo "========================================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Started      : $(date)"
echo "========================================================================"
echo ""

cd ~/nn_proj
source ~/nn_proj/venv_gpu/bin/activate

echo "Python       : $(which python)"
echo "Python ver   : $(python --version)"
echo ""

# ── hyperparameters for this search run ──────────────────────────────────────
export LEARNING_RATE=0.005
export HIDDEN_SIZE_A=322    # GRU hidden size
export HIDDEN_SIZE_B=322    # CRN bottleneck hidden size
export TRAIN_SAMPLES=500
export NUM_EPOCHS=20
export RUN_TYPE=search

echo "LEARNING_RATE  : $LEARNING_RATE"
echo "HIDDEN_SIZE_A  : $HIDDEN_SIZE_A"
echo "HIDDEN_SIZE_B  : $HIDDEN_SIZE_B"
echo "TRAIN_SAMPLES  : $TRAIN_SAMPLES"
echo "NUM_EPOCHS     : $NUM_EPOCHS"
echo "RUN_TYPE       : $RUN_TYPE"
echo ""

python block3.py

echo ""
echo "========================================================================"
echo "Block 3 search finished : $(date)"
echo "Job ID                  : $SLURM_JOB_ID"
echo "========================================================================"

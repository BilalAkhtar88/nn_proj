#!/bin/bash
#SBATCH --job-name=dns_full_s2000
#SBATCH --account=ealloc_ati-1-neur
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4
#SBATCH --nodelist=falcon4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=/gpfs/helios/home/etais/hpc_bilal/nn_proj/logs/block3_full_samp2000_%j.log
#SBATCH --no-requeue

echo "========================================================================"
echo "DNS Block 3 — Full Training  (TRAIN_SAMPLES=2000)"
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

export LEARNING_RATE=0.005
export HIDDEN_SIZE_A=161
export HIDDEN_SIZE_B=161
export TRAIN_SAMPLES=2000
export NUM_EPOCHS=100
export RUN_TYPE=full_train

echo "LEARNING_RATE  : $LEARNING_RATE"
echo "HIDDEN_SIZE_A  : $HIDDEN_SIZE_A  (GRU)"
echo "HIDDEN_SIZE_B  : $HIDDEN_SIZE_B  (CRN)"
echo "TRAIN_SAMPLES  : $TRAIN_SAMPLES"
echo "NUM_EPOCHS     : $NUM_EPOCHS"
echo "RUN_TYPE       : $RUN_TYPE"
echo ""

python block3.py

echo ""
echo "========================================================================"
echo "Full training finished : $(date)"
echo "Job ID                 : $SLURM_JOB_ID"
echo "========================================================================"

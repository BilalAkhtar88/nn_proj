#!/bin/bash
#SBATCH --job-name=dns_eval
#SBATCH --account=ealloc_ati-1-neur
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4
#SBATCH --nodelist=falcon2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/helios/home/etais/hpc_bilal/nn_proj/logs/block4and5_%j.log

echo "========================================================================"
echo "DNS Block 4 + Block 5 — Enhancement and DNSMOS Evaluation"
echo "========================================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "========================================================================"

source /gpfs/helios/home/etais/hpc_bilal/nn_proj/venv_gpu/bin/activate

cd /gpfs/helios/home/etais/hpc_bilal/nn_proj

export CHALLENGE_DIR=/gpfs/helios/home/etais/hpc_bilal/nn_proj
python3 block4and5.py

echo "========================================================================"
echo "Job finished : $(date)"
echo "========================================================================"

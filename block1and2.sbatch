#!/bin/bash
#SBATCH --job-name=dns_block1and2
#SBATCH --account=ealloc_ati-1-neur
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/helios/home/etais/hpc_bilal/nn_proj/logs/block1and2_%j.log
#SBATCH --no-requeue

# ── job info ──────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "DNS Project — Block 1 (Data Loader) + Block 2 (Preprocessing)"
echo "========================================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Started      : $(date)"
echo "Working dir  : $SLURM_SUBMIT_DIR"
echo "========================================================================"
echo ""

# ── environment ───────────────────────────────────────────────────────────────
cd ~/nn_proj
source ~/nn_proj/venv_gpu/bin/activate

echo "Python       : $(which python)"
echo "Python ver   : $(python --version)"
echo ""

# ── ensure logs folder exists ─────────────────────────────────────────────────
mkdir -p ~/nn_proj/logs

# ── run block1and2.py ─────────────────────────────────────────────────────────
python block1and2.py

# ── finish ────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "Block 1 + Block 2 finished : $(date)"
echo "Job ID                     : $SLURM_JOB_ID"
echo "========================================================================"

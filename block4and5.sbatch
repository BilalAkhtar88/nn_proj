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

set -uo pipefail

PROJECT_DIR=/gpfs/helios/home/etais/hpc_bilal/nn_proj
VENV_DIR="$PROJECT_DIR/venv_gpu"
PY_FILE="$PROJECT_DIR/block4and5.py"

# Instructor-provided files live here:
#   baseline/enhance.py
#   baseline/dec-baseline-model-icassp2022*.onnx
#   DNSMOS/model_v8.onnx
#   DNSMOS/sig_bak_ovr.onnx
#   dnsmos_local.py
# In your current setup this appears to be the same as PROJECT_DIR.
# If you unzipped nnp_track02-master inside PROJECT_DIR, change this to:
# INSTRUCTOR_DIR="$PROJECT_DIR/nnp_track02-master"
INSTRUCTOR_DIR="$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs"

echo "========================================================================"
echo "DNS Block 4 + Block 5 — Enhancement and DNSMOS Evaluation"
echo "========================================================================"
echo "Job ID        : ${SLURM_JOB_ID:-interactive}"
echo "Node          : ${SLURMD_NODENAME:-local}"
echo "Project dir   : $PROJECT_DIR"
echo "Challenge dir : $INSTRUCTOR_DIR"
echo "Python file   : $PY_FILE"
echo "Started       : $(date)"
echo "========================================================================"

if [[ ! -f "$PY_FILE" ]]; then
    echo "ERROR: Python file not found: $PY_FILE"
    echo "Copy the downloaded block4and5.py into PROJECT_DIR, then run: sbatch block4and5.sbatch"
    exit 2
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found: $VENV_DIR"
    exit 2
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

export PROJECT_DIR="$PROJECT_DIR"
export EVAL_DIR="$PROJECT_DIR/eval_dataset"
export TEST_PT_DIR="$PROJECT_DIR/eval_dataset/test/pt_files"
export TEST_WAV_DIR="$PROJECT_DIR/eval_dataset/test/wav_files"
export RUNS_DIR="$PROJECT_DIR/runs/full_train"
export RESULTS_DIR="$PROJECT_DIR/results"
export CHALLENGE_DIR="$INSTRUCTOR_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Override at submission time if needed, e.g.:
#   sbatch --export=ALL,N_SCORE=200 block4and5.sbatch
export N_SCORE="${N_SCORE:-2000}"
export WAV_SAVE_COUNT="${WAV_SAVE_COUNT:-20}"

python3 -u "$PY_FILE"
status=$?

echo "========================================================================"
echo "Job finished : $(date)"
echo "Exit status  : $status"
echo "========================================================================"
exit $status

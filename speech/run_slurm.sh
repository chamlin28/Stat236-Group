#!/bin/bash
#SBATCH -J crec
#SBATCH -p shared
#SBATCH -t 24:00:00
#SBATCH --mem=16G
#SBATCH -c 1
#SBATCH --array=114-119%3        # Congresses 114 to 119, at most 3 at a time
#SBATCH -o logs/crec_%a.out

# Before submitting, create the logs folder in the same directory:
#   mkdir -p logs
# Submit with:  sbatch run_slurm.sh

# EDIT THIS: a folder with room for about 15 GB (lab or group storage).
export CREC_DATA_DIR=/n/home08/mquintero/crec_data

set -e   # stop if the download script itself crashes
cd "$SLURM_SUBMIT_DIR"
source ~/crec_env/bin/activate
mkdir -p "$CREC_DATA_DIR"

echo "Congress $SLURM_ARRAY_TASK_ID started $(date)"
python -u download_crec.py $SLURM_ARRAY_TASK_ID --passes 3
python -u parse_crec.py $SLURM_ARRAY_TASK_ID
echo "Congress $SLURM_ARRAY_TASK_ID finished $(date)"

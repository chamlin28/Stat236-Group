#!/bin/bash
#SBATCH -J emb
#SBATCH -p shared
#SBATCH -t 04:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -o logs/emb_%j.out

# Usage (from the crec_pipeline folder, after: mkdir -p logs):
#   sbatch -t 1:00:00 run_embeddings.sh 118 --pilot 3000 --graph-repo ~/Stat236-Group
#   sbatch -t 8:00:00 run_embeddings.sh 118 --graph-repo ~/Stat236-Group
# Everything after the script name is passed to speech_embeddings.py.
# The model must be cached first (see instructions), because this job runs offline.

set -e
export CREC_DATA_DIR=/n/home08/mquintero/crec_data
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # use all requested cores for PyTorch

cd "$SLURM_SUBMIT_DIR"
source ~/crec_env/bin/activate
echo "started $(date): speech_embeddings.py $@"
python -u speech_embeddings.py "$@"
echo "finished $(date)"

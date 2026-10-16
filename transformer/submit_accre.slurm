#!/bin/bash
# Transformer tagger on ACCRE (Rocky 9), one GPU, three seeds as a job array.
#
#   sbatch submit_accre.slurm                  # 3 seeds, full training file
#   sbatch --array=0 submit_accre.slurm        # a single run
#
# EDIT BEFORE FIRST USE: --account, and the DATA/CODE paths below.

#SBATCH --job-name=jet-transformer
#SBATCH --output=logs/transformer_%A_%a.log
#SBATCH --account=CHANGEME_acc            # your group name + "_acc"
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1     # or nvidia_a100_80gb:1, nvidia_l40s:1, ...
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue                         # resubmit automatically if preempted
##SBATCH --mail-type=END,FAIL             # uncomment these two to get emails
##SBATCH --mail-user=you@vanderbilt.edu
#SBATCH --array=0-2                       # three seeds

set -e

CODE=$HOME/JetVLAD_vs_transformer/transformer                        # code on ACCRE
DATA=$HOME/JetVLAD_vs_transformer/JetVLAD/NetVLAD-tagger-pytorch      # ROOT files
VENV=$HOME/jets-venv                                                  # python env (created below)

setup_accre_software_stack
module load cuda/12.6 python/3.12.4

# One-time environment setup, done automatically on the first run:
if [ ! -d "$VENV" ]; then
    echo "Creating $VENV ..."
    python -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install torch uproot awkward scikit-learn numpy matplotlib
fi
source "$VENV/bin/activate"

cd "$CODE"
mkdir -p logs

SEED=${SLURM_ARRAY_TASK_ID:-0}
VARIABLES=trkvtx
SIGNAL=hf_vs_l
DEPTH=2
WIDTH=64
HEADS=8
EPOCHS=60
JOBID=transformer_${SIGNAL}_depth_${DEPTH}_width_${WIDTH}_seed${SEED}

echo "=== $(date) | node $(hostname) | seed $SEED ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

RESUME=""
CKPT=$(ls -t output_files/$JOBID/transformer_best_val_loss_*.pth 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    echo "Found existing checkpoint, resuming from it: $CKPT"
    RESUME="--resume $CKPT"
fi

python train.py \
    $RESUME \
    --train-data "$DATA/training_data/output_training_all_data.root" \
    --val-data   "$DATA/validation_data/output_5to10_validation_all.root" \
    --test-data  "$DATA/testing_data/output_5to10_test_all.root" \
    --variables $VARIABLES \
    --signal $SIGNAL \
    --depth $DEPTH \
    --width $WIDTH \
    --heads $HEADS \
    --epochs $EPOCHS \
    --batch-size 1024 \
    --eval-every 1 \
    --seed $SEED \
    --jobid $JOBID \
    --output-dir output_files

echo "=== done $(date) | outputs in output_files/$JOBID/ ==="

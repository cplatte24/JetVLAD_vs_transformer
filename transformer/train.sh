#!/bin/bash
#SBATCH --job-name=transformer-train
#SBATCH --output=transformer-train-%j.log
#SBATCH --time=168:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
cd "$(dirname "$0")"

# --- configuration (same knobs as the JetVLAD train.sh, plus width/heads) ---
VARIABLES=full          # trk | vtx | trkvtx | trkfrag | trkvtxfrag | full
SIGNAL=lcb              # lcb | hf_vs_l | c_vs_b | c_vs_all | b_vs_all
DEPTH=4                 # transformer encoder layers
WIDTH=64                # embedding dimension
HEADS=8                 # attention heads
JOBID=transformer_${VARIABLES}_${SIGNAL}_depth_${DEPTH}_width_${WIDTH}

python train.py \
                --train-data "training_data/output_5to10_9_fixed.root,training_data/output_10to15_train.root,training_data/output_15to20_train.root" \
                --val-data   "validation_data/output_5to10_validation_all.root,validation_data/output_10to15_validation.root,validation_data/output_15to20_validation.root" \
                --test-data  "testing_data/output_5to10_test_all.root,testing_data/output_10to15_test.root,testing_data/output_15to20_test.root" \
                --variables $VARIABLES \
                --signal $SIGNAL \
                --depth $DEPTH \
                --width $WIDTH \
                --heads $HEADS \
                --jobid $JOBID

echo "Training complete. Outputs saved under training_runs/$JOBID/"

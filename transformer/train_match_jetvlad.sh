#!/bin/bash
# Apples-to-apples run against the JetVLAD baseline: same files, same feature
# set (trkvtx) and the same binary task (hf_vs_l). The only difference left is
# the model itself.
set -e
cd "$(dirname "$0")"

python train.py \
                --train-data training_data/output_5to10_9_fixed.root \
                --val-data   validation_data/output_5to10_validation_all.root \
                --test-data  testing_data/output_5to10_test_all.root \
                --variables trkvtx \
                --signal hf_vs_l \
                --depth 4 \
                --width 64 \
                --heads 8 \
                --jobid transformer_trkvtx_hf_vs_l

echo "Training complete. Outputs saved under training_runs/transformer_trkvtx_hf_vs_l/"

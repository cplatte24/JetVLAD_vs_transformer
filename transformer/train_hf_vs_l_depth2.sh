xy
#!/bin/bash
# Transformer, matched to the JetVLAD run
#   --variables trkvtx --signal hf_vs_l --depth 2
# on the large 5-10 GeV validation/testing files.
#
# Set MAXJETS to a number (e.g. 1000000) for a quick check; leave empty to use
# all 4.97M training jets.
set -e
cd "$(dirname "$0")"

NV=/Users/plattecw/JetVLAD_JupyterNotebook/NetVLAD-tagger-pytorch

VARIABLES=trkvtx
SIGNAL=hf_vs_l
DEPTH=2
WIDTH=64
HEADS=8
EPOCHS=10
MAXJETS=1000000
OUTDIR=output_files   # all results go under this folder
JOBID=transformer_run_${SIGNAL}_depth_${DEPTH}_width_${WIDTH}

python train.py \
                --train-data $NV/training_data/output_training_all_data.root \
                --val-data   $NV/validation_data/output_5to10_validation_all.root \
                --test-data  $NV/testing_data/output_5to10_test_all.root \
                --variables $VARIABLES \
                --signal $SIGNAL \
                --depth $DEPTH \
                --width $WIDTH \
                --heads $HEADS \
                --epochs $EPOCHS \
                ${MAXJETS:+--max-train-jets $MAXJETS} \
                --jobid $JOBID \
                --output-dir $OUTDIR

echo "Training complete. Outputs saved under $OUTDIR/$JOBID/"

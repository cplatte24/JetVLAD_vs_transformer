# JetVLAD vs. Transformer

Two jet-flavour taggers on the same simulated data, kept in separate folders so
neither one's code affects the other.

```
JetVLAD/NetVLAD-tagger-pytorch/   the NetVLAD baseline (unmodified upstream code)
transformer/                      the transformer tagger
```

The ROOT files live in `JetVLAD/NetVLAD-tagger-pytorch/{training,validation,testing}_data/`
and the transformer reaches them through symlinks, so the data is stored once.

## Running them

Both use the same conda environment:

```
conda create -n jets python=3.11 -y
conda activate jets
pip install torch uproot awkward scikit-learn numpy matplotlib notebook
```

**JetVLAD:**

```
cd JetVLAD/NetVLAD-tagger-pytorch
python train.py --train-data training_data/output_5to10_9_fixed.root \
                --val-data validation_data/output_5to10_validation_all.root \
                --test-data testing_data/output_5to10_test_all.root \
                --variables trkvtx --clusters 33 --depth 2 --signal hf_vs_l \
                --jobid jetvlad_trkvtx_hf_vs_l
```

**Transformer:**

```
cd transformer
python smoke_test.py            # first time only
bash train_match_jetvlad.sh     # same files, features and task as above
```

Both write checkpoints and ROC/PR curves to `training_runs/<jobid>/`, in the
same file format.

## Comparing

The transformer accepts the same `--variables` and `--signal` values as
JetVLAD, so a like-for-like comparison is a matter of matching those two
options. Compare JetVLAD's `Final ROC-AUC` with the transformer's
`auc_<signal>` (or `auc_hf_vs_l` when running the 3-class `lcb` mode).

See [transformer/DIFFERENCES_VS_JETVLAD.md](transformer/DIFFERENCES_VS_JETVLAD.md)
for what the models actually do differently, and
[transformer/README.md](transformer/README.md) for the full configuration
reference.

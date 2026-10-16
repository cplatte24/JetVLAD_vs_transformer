# Transformer jet-flavour tagger

A Particle-Transformer-style tagger for light / charm / bottom jets, written to
be a drop-in comparison for the NetVLAD tagger in `../JetVLAD/`. It reads the
same ROOT files, accepts the same `--variables` and `--signal` options, and
reports AUCs in the same form, so any difference in performance comes from the
model rather than from the data or the task.

See [DIFFERENCES_VS_JETVLAD.md](DIFFERENCES_VS_JETVLAD.md) for what the model
does differently and why.

## Layout

```
jet_features.py        feature definitions, variable sets, signal modes, ROOT loading (numpy only)
model/transformer.py   the model
utils.py               dataset, LR schedule, training loops, metrics
train.py               entry point
smoke_test.py          correctness checks -- run this first
train.sh               configurable run script (all pT bins)
train_match_jetvlad.sh apples-to-apples run against the JetVLAD baseline
training_data/ validation_data/ testing_data/   symlinks into ../JetVLAD/NetVLAD-tagger-pytorch/
```

## Setup

```
conda create -n jets python=3.11 -y
conda activate jets
pip install torch uproot awkward scikit-learn numpy matplotlib notebook
```

Installing torch with pip rather than conda is deliberate: the `pytorch` conda
channel no longer ships current macOS builds, and asking conda for it makes the
solver hang. `environment.yml` is set up the same way (conda for Python,
pip for the rest).

Then check the setup:

```
python smoke_test.py
```

This builds a dataset from a real file and verifies that padding is zeroed,
that the mask has the right length, that adding extra padding does not change
the model's output, that shuffling the tracks does not change it either, and
that a short training run converges. It takes a couple of minutes.

## Training

```
bash train.sh                 # all three pT bins, 3-class output, all features
bash train_match_jetvlad.sh   # same file/features/task as the JetVLAD baseline
```

or directly:

```
python train.py --train-data training_data/output_5to10_9_fixed.root \
                --val-data   validation_data/output_5to10_validation_all.root \
                --test-data  testing_data/output_5to10_test_all.root \
                --variables trkvtx \
                --signal hf_vs_l \
                --depth 4 \
                --jobid transformer_trkvtx_hf_vs_l
```

Data arguments accept comma-separated lists and globs, so several pT bins can
be trained on at once.

## Configuration

Shared with the JetVLAD tagger, same meanings:

| Option | Values | Default | Meaning |
|---|---|---|---|
| `--train-data`, `--val-data`, `--test-data` | paths | required | ROOT files; lists and globs allowed |
| `--variables` | `trk`, `vtx`, `trkvtx`, `trkfrag`, `trkvtxfrag`, `full` | `full` | per-track feature set |
| `--signal` | `lcb`, `hf_vs_l`, `c_vs_b`, `c_vs_all`, `b_vs_all` | `lcb` | classification task |
| `--depth` | int | `4` | model size: encoder layers here, residual blocks in JetVLAD |
| `--jobid` | string | `transformer_run_0` | names `training_runs/<jobid>/` |
| `--epochs`, `--patience` | int | `50`, `8` | training length and early stopping |

Transformer-specific:

| Option | Default | Meaning |
|---|---|---|
| `--width` | `64` | embedding dimension `d_model` (the size knob, as `--clusters` is for JetVLAD) |
| `--heads` | `8` | attention heads per layer |
| `--n-max` | `16` | track slots per jet; jets are pT-sorted and truncated to this (16 covers every jet in the current data, whose largest has 15 tracks) |
| `--dropout` | `0.1` | dropout in the encoder and head |
| `--lr` | `2e-4` | peak learning rate (AdamW) |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--warmup-frac` | `0.05` | fraction of steps spent warming the LR up linearly before cosine decay |
| `--batch-size` | `512` | |
| `--grad-clip` | `1.0` | gradient-norm clipping |
| `--class-weighting` | `sqrt` | `sqrt`, `inverse` or `none` re-weighting of the loss |
| `--seed` | `0` | |
| `--max-train-jets`, `--max-eval-jets` | none | cap jets loaded, for memory or quick tests |

### Feature sets

The first five match the NetVLAD sets, using the transformer's jet-relative
versions of the track kinematics:

| Set | Features |
|---|---|
| `trk` | log pT, Δη, Δφ |
| `vtx` | DCA_z, DCA_xy |
| `trkvtx` | log pT, Δη, Δφ, DCA_z, DCA_xy |
| `trkfrag` | log pT, Δη, Δφ, z, ΔR, m |
| `trkvtxfrag` | log pT, Δη, Δφ, z, ΔR, m, DCA_z, DCA_xy |
| `full` | all of the above plus log(pT/jet pT) and SIP3D |

Δη and Δφ are measured relative to the jet axis (Δφ wrapped into (−π, π]);
`m` is fZ·ΔR², the same proxy NetVLAD calls `fM`. DCA_xy uses `log1p`, and
DCA_z and SIP3D use a signed `log1p`, to stop their long tails (SIP3D reaches
±1200) from dominating after z-scoring. Every feature is z-scored using
statistics computed once on the training set; those statistics are stored in
the checkpoint so inference matches training exactly.

### Signal modes

`lcb` is the native 3-class output: one model gives P(light), P(charm),
P(bottom), and every binary question can be asked of it afterwards. The other
four modes reproduce the NetVLAD binary tasks, including dropping light jets
for `c_vs_b`, and train a 2-class model.

## Training procedure

- **Optimizer:** AdamW, peak LR 2e-4, weight decay 0.01.
- **Schedule:** linear warmup over the first 5% of steps, then cosine decay to
  zero, stepped once per batch. Warmup matters: without it, transformers on
  small datasets often diverge or collapse onto the majority class.
- **Gradient clipping** at norm 1.0.
- **Class weighting:** the labels are roughly 97.5% light, 2.2% charm, 0.3%
  bottom, so the loss is re-weighted by 1/√frequency by default.
- **Early stopping:** the run keeps the checkpoint with the best validation
  loss and stops after `--patience` epochs without improvement.

## Outputs

Everything lands in `training_runs/<jobid>/`:

- `transformer_best_val_loss_<train file>_<jobid>.pth` — best checkpoint, which
  also stores the normalization statistics, the feature set, the signal mode
  and the full argument list.
- ROC and PR curves as two-row text files, the same format the NetVLAD code
  writes, so existing plotting scripts can read them.

Metrics printed each epoch:

- **Binary signals:** a single `auc_<signal>`, directly comparable to JetVLAD's
  `Final ROC-AUC` for the same `--signal`.
- **`lcb`:** per-class one-vs-rest AUCs, `auc_hf_vs_l` (score = P(c) + P(b)),
  which is comparable to a JetVLAD `hf_vs_l` run, and `auc_b_vs_c`.

## Memory and runtime

The padded training tensor is `n_jets × n_max × n_features × 4` bytes: about
350 MB for the three pT bins together (537k jets, `full`, `--n-max 16`), and
about 2.9 GB for a 5M-jet file. Loading is streamed and vectorized, roughly 25
seconds for 5M jets. Use `--max-train-jets` if memory is tight.

Training on CPU is slower per epoch than NetVLAD (attention compares every pair
of tracks), but the model is small — around 210k parameters at the default
depth and width.

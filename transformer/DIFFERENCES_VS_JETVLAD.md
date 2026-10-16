# What this model does differently from JetVLAD

Both taggers read the same jets, use the same per-track quantities (with
`--variables`), and answer the same question (with `--signal`). They differ in
how a variable-length set of tracks becomes one jet-level decision.

## The short version

1. **Tracks see each other.** JetVLAD describes every track on its own. The
   transformer lets every track attend to every other track, four times over,
   so it can use relations between tracks — for example that three displaced
   tracks sit close together and look like one decay, rather than three
   unrelated displaced tracks.
2. **Processing happens before pooling, not after.** JetVLAD pools almost
   immediately (one cluster assignment, then a sum) and does its heavy work on
   the pooled summary. The transformer builds rich per-track representations
   first and pools at the end, with a learned, jet-dependent weighting.
3. **Padding is masked.** Jets have different track counts, so both models pad.
   JetVLAD has no mask and its padding leaks into the output. The transformer
   passes an explicit mask, and padded slots get exactly zero weight.

## 1. Aggregation

**JetVLAD.** Each track is softly assigned to 33 learned cluster centres. The
assignment weights come from that track alone; nothing about the other tracks
in the jet enters. The jet becomes the sum of the tracks' residuals per
cluster — a 33 × n_features vector, effectively a learned histogram.

**Transformer.** Each track is embedded into a `--width`-dimensional vector,
then `--depth` encoder layers let the tracks exchange information. In each
layer every track forms a query, compares it against every other track's key,
and takes a weighted blend of their values. A learnable class token takes part
in the same attention and collects what matters for classification; its final
vector is the jet summary.

Consequence: JetVLAD can count how many displaced tracks a jet has, but the
sum discards which tracks belong together. The transformer can express the
joint conditions — displaced, angularly close to another displaced track, with
consistent kinematics — that distinguish a real heavy-flavour decay from a
light jet containing a V0 decay, a conversion or a mismeasured track.

The limits are worth stating: with no signed impact parameter, no
point-of-closest-approach position and no track uncertainties in the input,
the transformer judges *consistency* with a common origin. It does not
reconstruct a vertex.

## 2. Where the model's capacity sits

| | JetVLAD | Transformer |
|---|---|---|
| Before pooling | one linear cluster assignment + softmax | embedding MLP, then `--depth` attention + feed-forward layers |
| Pooling | fixed rule (cluster-residual sums), identical for every jet | learned, jet-dependent (the class token decides what to read) |
| After pooling | 2 residual blocks at width 165, ~111k of its ~112k parameters | Linear → GELU → Dropout → Linear, a few thousand parameters |

Pooling is lossy: a variable number of tracks becomes a fixed-size summary.
Whatever is not captured before the squeeze cannot be recovered afterwards.

## 3. Padding

`collate_fn` in JetVLAD pads each jet to the largest jet **in its batch** with
rows of zeros, and the model gets no mask. Because the features are z-scored, a
zero row looks exactly like a track with the average value of every feature, so
those rows act as phantom "average tracks": the softmax always gives them
cluster weights, which enter the `a_sum` term and shift the jet summary. Two
consequences follow. The summary is blurred, and the amount of blurring depends
on which jets happened to share the batch, so the same jet can score slightly
differently in different batches.

Measured on `output_5to10_9_fixed.root`: jets average 3.2 real tracks and about
4 padding rows, and 61% of jets have more padding rows than real tracks.

The transformer pads to a fixed `--n-max` and passes a boolean
`src_key_padding_mask`. Padded slots score −∞ before the softmax, hence exactly
zero weight, both in track-to-track attention and in the class token's pooling.
Padded rows are also zeroed before the embedding as a second safeguard, and
LayerNorm (unlike JetVLAD's BatchNorm) never mixes statistics across slots,
jets or batches. `smoke_test.py` verifies this by padding a jet with random
garbage and confirming the logits move by less than 1e-5.

This one is about the implementation rather than the idea — JetVLAD could be
given a mask too — but as the code stands it affects how reproducible its
scores are.

## 4. Input features

With the same `--variables` set, both models receive the same physics. The
transformer's versions differ in preparation, because attention is far more
sensitive to input scale than cluster assignment is:

- **Jet-relative kinematics.** Δη and Δφ measured from the jet axis (Δφ wrapped
  into (−π, π]) instead of absolute `fEta` and `fPhi`. Absolute φ ranges over
  0–2π and carries no flavour information, and its wrap-around makes two nearby
  tracks look maximally far apart.
- **Log scaling.** log pT instead of raw pT.
- **Tail compression.** `log1p` on DCA_xy and a signed `log1p` on DCA_z and
  SIP3D, whose tails reach ±1200 and would otherwise dominate the z-scoring.
- **Normalization from the training set only.** JetVLAD's dataset class
  recomputes means and standard deviations separately for train, validation and
  test, so the three are normalized differently. The transformer computes them
  once on train, reuses them for val and test, and stores them in the
  checkpoint so inference matches training.
- **`full` set.** Adds log(pT / jet pT) and SIP3D, which the NetVLAD variable
  sets never included.

## 5. Output

JetVLAD always produces 2 logits for the task chosen with `--signal`, so each
question needs its own trained model. The transformer defaults to `--signal
lcb`, 3 logits for light/charm/bottom, from which every binary question can be
derived: hf-vs-light as P(c) + P(b), b-vs-c from their ratio. The binary
`--signal` modes are still available for a like-for-like comparison.

## 6. Training recipe

NetVLAD's hyperparameters do not transfer to a transformer.

| | JetVLAD | Transformer |
|---|---|---|
| Optimizer | SGD, lr 0.013, momentum 0.9, weight decay 1e-6 | AdamW, lr 2e-4, weight decay 0.01 |
| LR schedule | cosine with warm restarts (T₀=1, T_mult=3) | linear warmup over 5% of steps, then a single cosine decay, stepped per batch |
| Gradient clipping | none | norm 1.0 |
| Loss | plain cross-entropy | class-weighted cross-entropy (1/√frequency by default) |
| Batch size | 256 | 512 |
| Dropout | 0.5 | 0.1 |
| Normalization layers | BatchNorm | LayerNorm, pre-norm (`norm_first=True`) |

Warmup is not optional: without it, transformers on small datasets tend to
diverge or collapse onto the majority class, and with 97.5% light jets that
collapse is an easy local minimum. Class weighting addresses the same
imbalance from the loss side.

## How to compare fairly

Run both models on the same files, the same `--variables` and the same
`--signal`:

```
# JetVLAD
cd ../JetVLAD/NetVLAD-tagger-pytorch
python train.py --train-data training_data/output_5to10_9_fixed.root \
                --val-data validation_data/output_5to10_validation_all.root \
                --test-data testing_data/output_5to10_test_all.root \
                --variables trkvtx --clusters 33 --depth 2 --signal hf_vs_l \
                --jobid jetvlad_trkvtx_hf_vs_l

# Transformer
cd ../../transformer
bash train_match_jetvlad.sh
```

Compare JetVLAD's `Final ROC-AUC` with the transformer's `auc_hf_vs_l`. Both
are computed on the same test file.

Two caveats. The feature *preparation* still differs even within one
`--variables` set (jet-relative, log-scaled), so the comparison is
"architecture plus its natural input preparation" rather than architecture
alone. And the b-jet class is small — around 730 b-jets in the 5–10 GeV
training file — which limits how much any model can learn about the relations
that distinguish them.

## A useful ablation

If you want to isolate how much the track-to-track attention contributes,
compare against a model that is identical except that tracks cannot interact:
same features, same embedding, same head, with a masked mean over the real
tracks instead of attention (a "Deep Sets" model). If the transformer beats it,
that gap is what attention is buying on this data. It is not implemented here
yet, but it is a small addition to `model/`.

"""Smoke test for the transformer migration. Run locally:

    python smoke_test.py

Checks, in order:
  1. Dataset builds from a real ROOT file (padding, mask, labels, norm stats).
  2. Mask invariance: adding extra padding to a jet must not change its logits.
  3. Track-order invariance sanity (permutation of real tracks -> same logits,
     since there is no positional encoding).
  4. A few hundred training steps on a small subset: loss must drop.

Exits non-zero on failure.
"""

import sys
import numpy as np
import torch

from utils import (JetSequenceDataset, make_warmup_cosine_scheduler,
                   train_epoch)
from model.transformer import TransformerTagger

TRAIN_FILE = 'training_data/output_5to10_9_fixed.root'
VARIABLES = 'full'
SIGNAL = 'lcb'


def check(cond, msg):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {msg}')
    if not cond:
        sys.exit(1)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 1. dataset ------------------------------------------------------
    data = JetSequenceDataset(TRAIN_FILE, n_max=16, variables=VARIABLES, signal=SIGNAL)
    X, mask, y = data[0]
    check(X.shape == (16, data.n_features), f'feature tensor shape {tuple(X.shape)}')
    check(mask.dtype == torch.bool, 'mask is boolean')
    check(bool((X[mask] == 0).all()), 'padded rows are exactly zero')
    check(int(y) in range(data.n_classes), f'labels in range for signal {SIGNAL}')
    counts = data.label_counts()
    check(counts.sum() == len(data), f'label counts {counts.tolist()} cover dataset')
    mean, std = data.norm_stats
    check(np.all(np.isfinite(mean)) and np.all(std > 0), 'norm stats finite, std>0')
    # real tracks should be roughly z-scored
    real = data.X[~data.mask]
    check(abs(float(real.mean())) < 0.05 and 0.5 < float(real.std()) < 2.0,
          f'real tracks ~z-scored (mean {float(real.mean()):.3f}, std {float(real.std()):.3f})')

    # ---- 2. mask invariance ---------------------------------------------
    model = TransformerTagger(feature_size=data.n_features, d_model=32,
                              n_heads=4, n_layers=2, dropout=0.0,
                              n_classes=data.n_classes)
    model.eval()

    # take a jet with few tracks, present it with N=16 and N=32 padding
    idx = int(np.argmax(data.mask.sum(dim=1).numpy()))  # most-padded jet
    X1, m1 = data.X[idx:idx + 1], data.mask[idx:idx + 1]
    X2 = torch.cat([X1, torch.randn(1, 16, data.n_features)], dim=1)  # garbage padding
    m2 = torch.cat([m1, torch.ones(1, 16, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        out1, out2 = model(X1, m1), model(X2, m2)
    diff = float((out1 - out2).abs().max())
    check(diff < 1e-5, f'mask invariance: extra garbage padding changes logits by {diff:.2e}')

    # ---- 3. permutation invariance --------------------------------------
    n_real = int((~m1[0]).sum())
    perm = torch.randperm(n_real)
    Xp = X1.clone()
    Xp[0, :n_real] = X1[0, perm]
    with torch.no_grad():
        outp = model(Xp, m1)
    pdiff = float((out1 - outp).abs().max())
    check(pdiff < 1e-5, f'permutation invariance: track shuffle changes logits by {pdiff:.2e}')

    # ---- 4. short training run ------------------------------------------
    n_sub = min(20000, len(data))
    sub = torch.utils.data.Subset(data, list(range(n_sub)))
    loader = torch.utils.data.DataLoader(sub, batch_size=256, shuffle=True,
                                         drop_last=True, num_workers=0)
    model = TransformerTagger(feature_size=data.n_features, d_model=64,
                              n_heads=8, n_layers=4, dropout=0.1,
                              n_classes=data.n_classes)
    weights = data.loss_class_weights()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    n_epochs = 3
    scheduler = make_warmup_cosine_scheduler(optimizer, len(loader) * n_epochs)

    first = train_epoch(loader, model, criterion, optimizer, scheduler, 0)
    last = first
    for e in range(1, n_epochs):
        last = train_epoch(loader, model, criterion, optimizer, scheduler, e)
    print(f'loss epoch0={first:.4f} -> epoch{n_epochs-1}={last:.4f}')
    check(last < first, 'training loss decreases')
    check(np.isfinite(last), 'loss is finite (no divergence)')

    print('\nAll smoke tests passed.')


if __name__ == '__main__':
    main()

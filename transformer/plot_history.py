"""Plot training vs validation loss (and the validation AUC) for a run.

Reads either the history CSV written by train.py, or an existing text training
log, so runs from before the CSV existed can still be plotted.

    python plot_history.py --csv output_files/<jobid>/history_<jobid>.csv
    python plot_history.py --log output_files/run_log.txt
"""

import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def from_csv(path):
    rows = np.genfromtxt(path, delimiter=',', names=True)
    auc_col = [n for n in rows.dtype.names if n.startswith('auc')]
    return (np.atleast_1d(rows['epoch']), np.atleast_1d(rows['train_loss']),
            np.atleast_1d(rows['val_loss']),
            np.atleast_1d(rows[auc_col[0]]) if auc_col else None)


def from_log(path):
    """Parse a training log: last progress line per epoch + the val lines."""
    train, val, aucs = {}, [], []
    for line in open(path):
        m = re.match(r'Epoch: \[(\d+)\]\[\s*\d+/\d+\]\s+Time.*Loss\s+\S+\s+\(([\d.e+-]+)\)', line)
        if m:
            train[int(m.group(1))] = float(m.group(2))
        m = re.match(r' \* Val loss ([\d.]+)', line)
        if m:
            val.append(float(m.group(1)))
        m = re.match(r'Val metrics: auc_\S+=([\d.]+)', line)
        if m:
            aucs.append(float(m.group(1)))
    epochs = sorted(train)
    n = min(len(epochs), len(val))
    return (np.array(epochs[:n]), np.array([train[e] for e in epochs[:n]]),
            np.array(val[:n]), np.array(aucs[:n]) if len(aucs) >= n else None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--csv', help='history_<jobid>.csv written by train.py')
    src.add_argument('--log', help='a text training log to parse instead')
    p.add_argument('--out', default=None, help='output PDF (default: beside the input)')
    p.add_argument('--title', default='Transformer tagger')
    args = p.parse_args()

    path = args.csv or args.log
    epochs, train_loss, val_loss, auc = (from_csv(args.csv) if args.csv
                                         else from_log(args.log))
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(path)),
                                   'loss_history.pdf')

    ncols = 2 if auc is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5), dpi=110)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.plot(epochs, train_loss, 'o-', color='crimson', lw=2.5, label='training')
    ax.plot(epochs, val_loss, 's--', color='tab:blue', lw=2.5, label='validation')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(f'{args.title}: loss', fontsize=14)
    ax.grid(alpha=.25)
    ax.legend(fontsize=12, frameon=False)

    if auc is not None:
        ax = axes[1]
        ax.plot(epochs, auc, 'o-', color='seagreen', lw=2.5)
        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel('Validation AUC', fontsize=13)
        ax.set_title(f'{args.title}: validation AUC', fontsize=14)
        ax.grid(alpha=.25)

    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print(f'Wrote {out}')

    gap = train_loss[-1] - val_loss[-1]
    print(f'\nFinal: train {train_loss[-1]:.4f}  val {val_loss[-1]:.4f}  '
          f'(train - val = {gap:+.4f})')
    print('  val loss still falling' if len(val_loss) > 2 and
          val_loss[-1] < val_loss[-3] - 1e-4 else '  val loss has plateaued')
    if gap > 0.01:
        print('  train loss above val loss: still underfitting, more capacity or epochs may help')
    elif gap < -0.01:
        print('  train loss well below val loss: check for overfitting')


if __name__ == '__main__':
    main()

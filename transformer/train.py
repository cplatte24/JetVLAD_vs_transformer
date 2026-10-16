"""Train the transformer jet-flavour tagger.

The command-line configuration mirrors the NetVLAD tagger: --variables selects
the per-track feature set, --signal the classification task, --depth the model
size, and --jobid names the output directory. Transformer-specific knobs
(--width, --heads, --n-max, ...) have defaults that work out of the box.

Example:

    python train.py --train-data training_data/output_5to10_9_fixed.root \
                    --val-data   validation_data/output_5to10_validation_all.root \
                    --test-data  testing_data/output_5to10_test_all.root \
                    --variables trkvtx --signal hf_vs_l --depth 4 \
                    --jobid transformer_trkvtx_hf_vs_l
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from jet_features import VARIABLE_SETS, SIGNALS
from model.transformer import TransformerTagger
import utils
from utils import (JetSequenceDataset, make_warmup_cosine_scheduler,
                   train_epoch, validate, evaluate)

parser = argparse.ArgumentParser(description="Train transformer-tagger on specified dataset")
# --- same options as the NetVLAD tagger -------------------------------------
parser.add_argument('--train-data', type=str, required=True,
                    metavar='/path/to/train/dataset/*.root',
                    help='path to the train dataset (comma-separated list and globs allowed)')
parser.add_argument('--test-data', type=str, required=True,
                    metavar='/path/to/test/dataset/*.root', help='path to the test dataset')
parser.add_argument('--val-data', type=str, required=True,
                    metavar='/path/to/val/dataset/*.root', help='path to the val dataset')
parser.add_argument('--variables', type=str, default='full', choices=sorted(VARIABLE_SETS),
                    help='input variable set to use (trk/vtx/trkvtx/trkfrag/trkvtxfrag match '
                         'the NetVLAD sets; full adds log(pT/jetpT) and SIP3D)')
parser.add_argument('--signal', type=str, default='lcb', choices=SIGNALS,
                    help='classification task; lcb is the native 3-class light/charm/bottom '
                         'output, the rest match the NetVLAD binary signals')
parser.add_argument('--depth', type=int, default=4,
                    help='number of transformer encoder layers')
parser.add_argument('--jobid', type=str, metavar='run index (can be SLURM job id)',
                    default='transformer_run_0')
parser.add_argument('--output-dir', type=str, default='training_runs',
                    help='directory that holds the per-run output folders; '
                         'results go to <output-dir>/<jobid>/')
parser.add_argument('--epochs', type=int, default=50, help='maximum number of training epochs')
parser.add_argument('--patience', type=int, default=8,
                    help='epochs without val improvement before early stopping')
# --- transformer-specific ---------------------------------------------------
parser.add_argument('--width', type=int, default=64,
                    help='embedding dimension d_model (NetVLAD analogue: --clusters)')
parser.add_argument('--heads', type=int, default=8, help='attention heads per layer')
parser.add_argument('--n-max', type=int, default=16,
                    help='track slots per jet; jets are pT-sorted and truncated to this')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=2e-4, help='peak learning rate')
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--schedule', type=str, choices=['cosine', 'constant'], default='cosine',
                    help="LR after warmup: 'cosine' decay to zero (training), "
                         "'constant' (overfitting studies)")
parser.add_argument('--warmup-frac', type=float, default=0.05,
                    help='fraction of total steps used for linear LR warmup')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--grad-clip', type=float, default=1.0)
parser.add_argument('--class-weighting', type=str, choices=['sqrt', 'inverse', 'none'],
                    default='sqrt', help='loss re-weighting for the class imbalance')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--raw-features', action='store_true',
                    help='ablation: skip the log / log1p transforms and feed raw pT and '
                         'impact parameters (z-scoring still applied)')
parser.add_argument('--resume', type=str, default=None,
                    help='continue from this checkpoint instead of random init '
                         '(its normalization statistics are reused, so the inputs match)')
parser.add_argument('--device', type=str, default='auto',
                    help="'auto' (cuda > mps > cpu), or force 'cpu' / 'mps' / 'cuda'")
parser.add_argument('--eval-every', type=int, default=1,
                    help='run validation every N epochs (the val pass costs real time on CPU)')
parser.add_argument('--max-train-jets', type=int, default=None,
                    help='cap the number of training jets (memory: n_jets x n_max x n_features x 4 bytes)')
parser.add_argument('--max-eval-jets', type=int, default=None, help='cap the number of val/test jets')
args = parser.parse_args()

device = utils.pick_device(args.device)
utils.device = device      # the training loops read utils.device


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f'Device: {device}  |  variables={args.variables}  signal={args.signal}')

    resume_stats = None
    if args.resume:
        _prev = torch.load(args.resume, map_location='cpu', weights_only=False)
        resume_stats = (_prev['norm_mean'], _prev['norm_std'])
        print(f'Reusing normalization statistics from {args.resume}')

    print("Loading train dataset...", flush=True)
    train_data = JetSequenceDataset(args.train_data, n_max=args.n_max,
                                    variables=args.variables, signal=args.signal,
                                    norm_stats=resume_stats,
                                    max_jets=args.max_train_jets,
                                    log_transform=not args.raw_features)
    print(f"  {len(train_data)} jets, classes {train_data.class_names}, "
          f"counts {train_data.label_counts().tolist()}")

    print("Loading val dataset...", flush=True)
    val_data = JetSequenceDataset(args.val_data, n_max=args.n_max,
                                  variables=args.variables, signal=args.signal,
                                  norm_stats=train_data.norm_stats,
                                  max_jets=args.max_eval_jets,
                                    log_transform=not args.raw_features)
    print(f"  {len(val_data)} jets")

    print("Loading test dataset...", flush=True)
    test_data = JetSequenceDataset(args.test_data, n_max=args.n_max,
                                   variables=args.variables, signal=args.signal,
                                   norm_stats=train_data.norm_stats,
                                   max_jets=args.max_eval_jets,
                                    log_transform=not args.raw_features)
    print(f"  {len(test_data)} jets")

    # Data already lives in memory as tensors: workers would only add overhead.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)

    run_path = Path(args.output_dir).resolve() / args.jobid
    run_path.mkdir(parents=True, exist_ok=True)
    train_name = Path(args.train_data.split(',')[0]).stem
    save_path = run_path / f'transformer_best_val_loss_{train_name}_{args.jobid}.pth'

    model = TransformerTagger(feature_size=train_data.n_features,
                              d_model=args.width, n_heads=args.heads,
                              n_layers=args.depth, dropout=args.dropout,
                              n_classes=train_data.n_classes).to(device)
    if args.resume:
        prev = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(prev['model_state_dict'])
        print(f'Resumed weights from {args.resume}')
    print(model)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters()):,}')

    weights = train_data.loss_class_weights(scheme=args.class_weighting).to(device)
    print(f'Class weights {train_data.class_names}: {weights.cpu().numpy().round(3).tolist()}')
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_warmup_cosine_scheduler(optimizer, len(train_loader) * args.epochs,
                                             args.warmup_frac, schedule=args.schedule)

    def save_checkpoint(path):
        mean, std = train_data.norm_stats
        torch.save({
            'model_state_dict': model.state_dict(),
            'norm_mean': mean, 'norm_std': std,
            'n_max': args.n_max, 'variables': args.variables, 'signal': args.signal,
            'log_transform': not args.raw_features,
            'class_names': train_data.class_names,
            'config': {'d_model': args.width, 'n_heads': args.heads,
                       'n_layers': args.depth, 'dropout': args.dropout,
                       'feature_size': train_data.n_features,
                       'n_classes': train_data.n_classes},
            'args': vars(args),
        }, path.as_posix())

    history_path = run_path / f'history_{args.jobid}.csv'
    history = open(history_path, 'w', buffering=1)
    history.write('epoch,train_loss,val_loss,val_acc,lr,' +
                  'auc\n' if args.signal != 'lcb' else
                  'epoch,train_loss,val_loss,val_acc,lr,auc_hf_vs_l,auc_b_vs_c\n')

    best_val, best_acc, best_metrics, early_stop = np.inf, 0, {}, 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(train_loader, model, criterion, optimizer,
                                 scheduler, epoch, grad_clip=args.grad_clip)
        last_epoch = (epoch == args.epochs - 1)
        if not (last_epoch or (epoch + 1) % args.eval_every == 0):
            continue
        val_loss, acc = validate(val_loader, model, criterion)
        metrics = evaluate(val_loader, model, signal=args.signal)
        print('Val metrics: ' + ', '.join(f'{k}={v:.4f}' for k, v in metrics.items()))

        lr_now = optimizer.param_groups[0]['lr']
        aucs = ([metrics.get(f'auc_{args.signal}', float('nan'))] if args.signal != 'lcb'
                else [metrics.get('auc_hf_vs_l', float('nan')),
                      metrics.get('auc_b_vs_c', float('nan'))])
        history.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},{acc:.4f},{lr_now:.3e},'
                      + ','.join(f'{a:.6f}' for a in aucs) + '\n')

        if best_val - val_loss > 0.001:
            best_val, best_acc, best_metrics, early_stop = val_loss, acc, metrics, 0
            save_checkpoint(save_path)
        else:
            early_stop += 1

        print(f'Best val loss: {best_val:.4f}  best acc: {best_acc:.2f}')
        if best_metrics:
            print('Best val metrics: ' + ', '.join(f'{k}={v:.4f}' for k, v in best_metrics.items()))
        if early_stop == args.patience:
            print(f'Early stopping after {epoch + 1} epochs.')
            break

    # weights_only=False: torch >= 2.6 defaults to True, which refuses the
    # numpy normalization arrays we store alongside the weights. This file is
    # written by this script, so it is trusted.
    history.close()
    print(f'Per-epoch history: {history_path}')

    checkpoint = torch.load(save_path.as_posix(), map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    log_path = run_path / f'{train_name}_{args.jobid}'
    final_metrics = evaluate(test_loader, model, signal=args.signal,
                             log_path=log_path, generate_curve=True)
    print('Final test metrics: ' + ', '.join(f'{k}={v:.4f}' for k, v in final_metrics.items()))
    print(f'Checkpoint: {save_path}')


if __name__ == "__main__":
    main()

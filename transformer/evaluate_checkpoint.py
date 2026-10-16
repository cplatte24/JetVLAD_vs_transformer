"""Evaluate a saved transformer checkpoint on a test set.

Use this to finish a run whose training completed but whose final evaluation
failed, or to score an existing checkpoint on a different file. It reads the
feature set, signal mode, model shape and normalization statistics straight
from the checkpoint, so nothing needs to be re-specified.

    python evaluate_checkpoint.py \
        --checkpoint output_files/<jobid>/transformer_best_val_loss_<...>.pth \
        --test-data /path/to/test.root
"""

import argparse
from pathlib import Path

import torch

from model.transformer import TransformerTagger
from utils import JetSequenceDataset, evaluate, predict_probs, device

parser = argparse.ArgumentParser(description="Evaluate a saved transformer checkpoint")
parser.add_argument('--checkpoint', type=str, required=True, help='path to the .pth file')
parser.add_argument('--test-data', type=str, required=True,
                    help='ROOT file(s) to evaluate on; lists and globs allowed')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--max-eval-jets', type=int, default=None)
parser.add_argument('--no-curves', action='store_true',
                    help='skip writing the ROC/PR text files')
args = parser.parse_args()


def main():
    # weights_only=False: the checkpoint holds numpy normalization arrays as
    # well as the weights, and torch >= 2.6 refuses those in its strict mode.
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    cfg = ck['config']
    variables = ck.get('variables', 'full')
    signal = ck.get('signal', 'lcb')
    n_max = ck.get('n_max', 16)
    print(f'Checkpoint: variables={variables}  signal={signal}  n_max={n_max}')
    print(f'  model: {cfg}')

    print('Loading test dataset...', flush=True)
    data = JetSequenceDataset(args.test_data, n_max=n_max, variables=variables,
                              signal=signal,
                              norm_stats=(ck['norm_mean'], ck['norm_std']),
                              max_jets=args.max_eval_jets)
    print(f'  {len(data)} jets, classes {data.class_names}, '
          f'counts {data.label_counts().tolist()}')

    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)

    model = TransformerTagger(feature_size=cfg['feature_size'],
                              d_model=cfg['d_model'], n_heads=cfg['n_heads'],
                              n_layers=cfg['n_layers'], dropout=cfg['dropout'],
                              n_classes=cfg.get('n_classes', 3)).to(device)
    model.load_state_dict(ck['model_state_dict'])

    ckpt_path = Path(args.checkpoint)
    log_path = ckpt_path.parent / ckpt_path.stem.replace('transformer_best_val_loss_', '')

    # unweighted cross-entropy + accuracy, so the same numbers can be compared
    # between the training file and the validation file
    probs, y_true = predict_probs(loader, model)
    import numpy as np
    p_true = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    loss = float(-np.log(p_true).mean())
    acc = float((probs.argmax(1) == y_true).mean() * 100)
    print(f'Loss (unweighted CE): {loss:.5f}   accuracy: {acc:.2f}%')

    metrics = evaluate(loader, model, signal=signal, log_path=log_path,
                       generate_curve=not args.no_curves)
    print('Test metrics: ' + ', '.join(f'{k}={v:.4f}' for k, v in metrics.items()))
    if not args.no_curves:
        print(f'Curves written next to: {log_path}_roc.txt / _pr.txt')


if __name__ == '__main__':
    main()

"""Dataset, training loops and metrics for the transformer tagger.

Self-contained: nothing here imports from the NetVLAD repository.
"""

import math
import time

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score

from jet_features import (build_dataset, compute_norm_stats, apply_norm,
                          class_weights, apply_signal, class_names, n_features)

def pick_device(preferred=None):
    """cuda > mps (Apple GPU) > cpu, unless `preferred` names one."""
    if preferred and preferred != 'auto':
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = pick_device()


class JetSequenceDataset(torch.utils.data.Dataset):
    """Fixed-length padded track sequences.

    Every jet is padded/truncated to `n_max` tracks (descending-pT order) and
    returned with a boolean key_padding_mask (True = padding).

    `variables` selects the feature set and `signal` the classification task,
    using the same names as the NetVLAD tagger (plus 'full' and 'lcb').

    Pass `norm_stats=None` for the training set (statistics are computed from
    it); pass the training set's `.norm_stats` for val/test so normalization is
    identical everywhere. `max_jets` caps how many jets are loaded.
    """

    def __init__(self, file_paths, n_max=16, variables='full', signal='lcb',
                 norm_stats=None, tree_name='T', max_jets=None,
                 log_transform=True):
        X, mask, y3 = build_dataset(file_paths, n_max=n_max, variables=variables,
                                    tree_name=tree_name, max_jets=max_jets,
                                    log_transform=log_transform)

        y, keep = apply_signal(y3, signal)
        if not keep.all():                       # only c_vs_b drops jets
            X, mask = X[keep], mask[keep]

        if norm_stats is None:
            mean, std = compute_norm_stats(X, mask)
        else:
            mean, std = norm_stats
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
        apply_norm(X, mask, mean, std)

        self.X = torch.from_numpy(X)              # [n_jets, n_max, F] float32
        self.mask = torch.from_numpy(mask)        # [n_jets, n_max] bool
        self.y = torch.from_numpy(np.ascontiguousarray(y))   # [n_jets] int64
        self.n_max = n_max
        self.variables = variables
        self.signal = signal
        self.class_names = class_names(signal)
        self.n_classes = len(self.class_names)
        self.n_features = n_features(variables)
        self.norm_stats = (mean, std)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]

    def label_counts(self):
        return np.bincount(self.y.numpy(), minlength=self.n_classes)

    def loss_class_weights(self, scheme='sqrt'):
        return torch.from_numpy(class_weights(self.y.numpy(), self.n_classes,
                                              scheme=scheme))


def make_warmup_cosine_scheduler(optimizer, total_steps, warmup_frac=0.05,
                                 schedule='cosine'):
    """Linear warmup, then either cosine decay to zero or a constant LR.

    Stepped once per BATCH (not per epoch). Use schedule='constant' when
    studying overfitting: a cosine decay to zero suppresses late-epoch
    memorization, which is the very thing such a study is looking for.
    """
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == 'constant':
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- small logging helpers (adapted from the pytorch imagenet example) ------

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = '[{:' + str(len(str(num_batches))) + 'd}/' + str(num_batches) + ']'
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)


def accuracy(output, target):
    with torch.no_grad():
        return 100.0 * (output.argmax(dim=1) == target).float().mean().item()


# --- training / evaluation --------------------------------------------------

def train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch,
                grad_clip=1.0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, acc,
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (X, mask, y) in enumerate(train_loader):
        X = X.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        output = model(X, mask)
        loss = criterion(output, y)

        losses.update(loss.item(), X.size(0))
        acc.update(accuracy(output, y), X.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            progress.print(i)

    return losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')

    model.eval()
    with torch.no_grad():
        for X, mask, y in val_loader:
            X = X.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            output = model(X, mask)
            losses.update(criterion(output, y).item(), X.size(0))
            acc.update(accuracy(output, y), X.size(0))

    print(' * Val loss {:.4f}  acc {:.2f}'.format(losses.avg, acc.avg), flush=True)
    return losses.avg, acc.avg


def predict_probs(loader, model):
    """Softmax probabilities and labels for a whole loader."""
    probs, targets = [], []
    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        for X, mask, y in loader:
            X = X.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            probs.append(softmax(model(X, mask)).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def _save_curves(log_path, suffix, y_bin, score):
    fpr, tpr, _ = roc_curve(y_bin, score)
    pr, rc, _ = precision_recall_curve(y_bin, score)
    base = log_path.with_suffix('').name
    np.savetxt((log_path.with_name(base + suffix + '_roc.txt')).as_posix(), (fpr, tpr))
    np.savetxt((log_path.with_name(base + suffix + '_pr.txt')).as_posix(), (pr, rc))
    return auc(fpr, tpr)


def evaluate(loader, model, signal='lcb', log_path=None, generate_curve=False):
    """AUCs for the configured task.

    Binary signals report a single AUC on P(class 1), exactly like the NetVLAD
    baseline. The 3-class 'lcb' mode reports per-class one-vs-rest AUCs plus
    `auc_hf_vs_l` (score = P(charm) + P(bottom)), which is directly comparable
    to a NetVLAD run with --signal hf_vs_l, and `auc_b_vs_c`.

    ROC/PR curves are written in the same two-row format as the NetVLAD code.
    """
    probs, y_true = predict_probs(loader, model)
    results = {}

    if probs.shape[1] == 2:
        score = probs[:, 1]
        if 0 < y_true.sum() < len(y_true):
            fpr, tpr, _ = roc_curve(y_true, score)
            results[f'auc_{signal}'] = auc(fpr, tpr)
            if generate_curve and log_path is not None:
                _save_curves(log_path, '', y_true, score)
        return results

    names = class_names('lcb')
    for k, name in enumerate(names):
        y_bin = (y_true == k).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            results[f'auc_{name}_vs_rest'] = roc_auc_score(y_bin, probs[:, k])

    y_hf = (y_true != 0).astype(int)
    hf_score = probs[:, 1] + probs[:, 2]
    if 0 < y_hf.sum() < len(y_hf):
        fpr, tpr, _ = roc_curve(y_hf, hf_score)
        results['auc_hf_vs_l'] = auc(fpr, tpr)
        if generate_curve and log_path is not None:
            _save_curves(log_path, '_hf_vs_l', y_hf, hf_score)

    hf_sel = y_true != 0
    if hf_sel.sum() > 0:
        y_b = (y_true[hf_sel] == 2).astype(int)
        if 0 < y_b.sum() < len(y_b):
            b_score = probs[hf_sel, 2] / (probs[hf_sel, 1] + probs[hf_sel, 2] + 1e-12)
            results['auc_b_vs_c'] = roc_auc_score(y_b, b_score)
            if generate_curve and log_path is not None:
                _save_curves(log_path, '_b_vs_c', y_b, b_score)

    return results

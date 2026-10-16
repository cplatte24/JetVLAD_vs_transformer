"""Background-rejection and purity plots from saved ROC / PR curves.

Plots the transformer run and, when available, overlays the JetVLAD baseline
for the same task. Paths are resolved relative to this script, so it can be run
from anywhere, and it needs only numpy + matplotlib.

    python make_plots.py                                  # defaults below
    python make_plots.py --run-dir output_files/<jobid>   # a different run
    python make_plots.py --no-baseline                    # transformer only

Curve files are the two-row text format written by train.py /
evaluate_checkpoint.py (and by the NetVLAD code):
    *_roc.txt : row 0 = false-positive rate, row 1 = true-positive rate
    *_pr.txt  : row 0 = precision,           row 1 = recall
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')                      # no display needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN = os.path.join(HERE, 'output_files',
                           'transformer_run_hf_vs_l_depth_2_width_64')
DEFAULT_BASELINE = os.path.join(
    HERE, '..', 'JetVLAD', 'NetVLAD-tagger-pytorch', 'training_runs', 'jetvlad_test_run')

mean_fpr = np.logspace(-6, 0, 6000)        # log grid: rejection lives at small FPR
mean_rec = np.linspace(1, 1e-4, 8000)      # grid for averaging PR curves

_auc = getattr(np, 'trapezoid', None) or np.trapz     # numpy 2 renamed trapz


def area(x, y):
    """Area under y(x), with x sorted ascending."""
    order = np.argsort(x)
    return float(_auc(np.asarray(y)[order], np.asarray(x)[order]))


def load_roc(folder, pattern='*_roc.txt'):
    """Mean TPR on `mean_fpr`, its AUC, and the spread over matching files."""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        return None
    tprs, aucs, raw = [], [], []
    for path in files:
        fpr, tpr = np.loadtxt(path)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(area(fpr, tpr))
        raw.append((fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    return dict(files=files, mean=mean_tpr, auc=area(mean_fpr, mean_tpr),
                auc_std=float(np.std(aucs)), raw=raw,
                upper=np.minimum(mean_tpr + std_tpr, 1),
                lower=np.maximum(mean_tpr - std_tpr, 0))


def load_pr(folder, pattern='*_pr.txt'):
    """Mean precision on `mean_rec`, its AUC, and the spread over files."""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        return None
    precs, aucs, raw = [], [], []
    for path in files:
        prec, rec = np.loadtxt(path)
        precs.append(np.interp(mean_rec[::-1], rec[::-1], prec[::-1])[::-1])
        aucs.append(area(rec, prec))
        raw.append((prec, rec))
    mean_prec = np.mean(precs, axis=0)
    std_prec = np.std(precs, axis=0)
    return dict(files=files, mean=mean_prec, auc=area(mean_rec, mean_prec),
                auc_std=float(np.std(aucs)), raw=raw,
                upper=np.minimum(mean_prec + std_prec, 1),
                lower=np.maximum(mean_prec - std_prec, 0))


def working_points(roc, pr, name):
    """Rejection and purity at fixed signal efficiencies.

    Interpolated on the RAW curves, not on the averaging grid: at 50% signal
    efficiency the false-positive rate is ~1e-4, where any coarse grid would
    quantize the answer.
    """
    print(f'\n{name}')
    print(f'  ROC AUC: {roc["auc"]:.4f}' + (f' +/- {roc["auc_std"]:.4f}'
                                            if len(roc['files']) > 1 else ''))
    for eff in (0.5, 0.8, 0.9):
        rejs = []
        for fpr, tpr in roc['raw']:
            f_at = float(np.interp(eff, tpr, fpr))          # tpr is ascending
            rejs.append(1.0 / f_at if f_at > 0 else np.inf)
        rej = float(np.mean(rejs))
        line = f'  eff {eff:.0%}:  rejection {rej:10.1f}'
        if pr is not None:
            purs = [float(np.interp(eff, rec[::-1], prec[::-1]))
                    for prec, rec in pr['raw']]
            line += f'   purity {np.mean(purs):.3f}'
        print(line)


def legend_title(args, model_line):
    return (f'{model_line}\n'
            f'Pythia {args.pythia}, cross-section weighted\n'
            r'$\sqrt{s} = %s$ GeV, anti-$k_{T}$ jets R = %s' % (args.sqrts, args.R)
            + '\n'
            + r'$%s < p_{T,jet} < %s$ GeV/$c$, $|\eta_{jet}| < %s$'
              % (args.pt_min, args.pt_max, args.eta_max))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=DEFAULT_RUN, help='folder with the transformer curves')
    p.add_argument('--baseline-dir', default=DEFAULT_BASELINE, help='folder with the JetVLAD curves')
    p.add_argument('--no-baseline', action='store_true', help='plot the transformer alone')
    p.add_argument('--baseline-pattern', default='output_training_all_data_*_roc.txt',
                   help='which baseline ROC files to use (the PR files are matched to them)')
    p.add_argument('--label', default='Transformer (tracking + vertexing)')
    p.add_argument('--baseline-label', default='JetVLAD (tracking + vertexing)')
    p.add_argument('--out-dir', default=None, help='where to write the PDFs (default: run dir)')
    p.add_argument('--tag', default=None, help='filename suffix (default: the run folder name)')
    # figure annotation
    p.add_argument('--pt-min', default='5')
    p.add_argument('--pt-max', default='10')
    p.add_argument('--eta-max', default='0.6')
    p.add_argument('--sqrts', default='200')
    p.add_argument('--R', default='0.4')
    p.add_argument('--pythia', default='8.314')
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or run_dir)
    tag = args.tag or os.path.basename(run_dir.rstrip('/'))
    os.makedirs(out_dir, exist_ok=True)

    roc = load_roc(run_dir)
    pr = load_pr(run_dir)
    if roc is None:
        raise SystemExit(f'No *_roc.txt found in {run_dir}\n'
                         'Run train.py or evaluate_checkpoint.py first.')
    print(f'Transformer curves: {[os.path.basename(f) for f in roc["files"]]}')

    base_roc = base_pr = None
    if not args.no_baseline:
        bdir = os.path.abspath(args.baseline_dir)
        base_roc = load_roc(bdir, args.baseline_pattern)
        base_pr = load_pr(bdir, args.baseline_pattern.replace('_roc.txt', '_pr.txt'))
        if base_roc is None:
            print(f'(no baseline curves matching {args.baseline_pattern} in {bdir}; '
                  'plotting the transformer alone)')
        else:
            print(f'Baseline curves:    {[os.path.basename(f) for f in base_roc["files"]]}')

    # ---- background rejection vs signal efficiency -------------------------
    plt.figure(figsize=(11, 10), dpi=80, facecolor='w', edgecolor='k')
    if base_roc is not None:
        plt.plot(base_roc['mean'], 1 / mean_fpr, color='tab:blue', lw=4, alpha=.85,
                 linestyle='--', label=args.baseline_label)
        if len(base_roc['files']) > 1:
            plt.fill_betweenx(1 / mean_fpr, base_roc['lower'], base_roc['upper'],
                              color='tab:blue', alpha=.2)
    plt.plot(roc['mean'], 1 / mean_fpr, color='crimson', lw=4, alpha=.85,
             linestyle='-', label=args.label)
    if len(roc['files']) > 1:
        plt.fill_betweenx(1 / mean_fpr, roc['lower'], roc['upper'],
                          color='crimson', alpha=.2)

    plt.xlim([0, 1])
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=23)
    plt.yscale('log')
    plt.ylim([1e0, 1e5])
    plt.yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5], fontsize=23)
    plt.grid(axis='x', color='0.95')
    plt.xlabel('Signal (Heavy-Flavor Jets) Efficiency', fontsize=23)
    plt.ylabel('Background (Light-Flavor Jets) Rejection', fontsize=23)
    plt.legend(loc='upper right', fancybox=True, framealpha=0., fontsize=20,
               title=legend_title(args, 'HF-jet vs udsg-jet'), title_fontsize=20)
    rej_path = os.path.join(out_dir, f'rejection_{tag}.pdf')
    plt.savefig(rej_path, bbox_inches='tight')
    plt.close()

    # ---- purity vs signal efficiency --------------------------------------
    if pr is not None:
        plt.figure(figsize=(11, 10), dpi=80, facecolor='w', edgecolor='k')
        if base_pr is not None:
            plt.plot(mean_rec, base_pr['mean'], color='tab:blue', lw=4, alpha=.85,
                     linestyle='--', label=args.baseline_label)
            if len(base_pr['files']) > 1:
                plt.fill_between(mean_rec, base_pr['lower'], base_pr['upper'],
                                 color='tab:blue', alpha=.2)
        plt.plot(mean_rec, pr['mean'], color='crimson', lw=4, alpha=.85,
                 linestyle='-', label=args.label)
        if len(pr['files']) > 1:
            plt.fill_between(mean_rec, pr['lower'], pr['upper'],
                             color='crimson', alpha=.2)

        plt.xlim([0.2, 1])
        plt.ylim([-0.05, 1.05])
        plt.xticks(np.arange(0.2, 1.1, 0.1), fontsize=23)
        plt.yticks(fontsize=23)
        plt.grid(axis='x', color='0.95')
        plt.xlabel('Signal (Heavy-Flavor Jets) Efficiency', fontsize=23)
        plt.ylabel('Signal (Heavy-Flavor Jets) Purity', fontsize=23)
        plt.legend(loc=(0.025, 0.12), fancybox=True, framealpha=0., fontsize=20,
                   title=legend_title(args, 'HF-jet vs udsg-jet'), title_fontsize=20)
        pur_path = os.path.join(out_dir, f'purity_{tag}.pdf')
        plt.savefig(pur_path, bbox_inches='tight')
        plt.close()
    else:
        pur_path = None

    working_points(roc, pr, args.label)
    if base_roc is not None:
        working_points(base_roc, base_pr, args.baseline_label)

    print(f'\nWrote {rej_path}')
    if pur_path:
        print(f'Wrote {pur_path}')


if __name__ == '__main__':
    main()

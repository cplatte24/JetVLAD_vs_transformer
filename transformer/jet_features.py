"""Feature engineering and ROOT loading for the transformer tagger.

Pure numpy + uproot/awkward (no torch), so the data pipeline can be tested on
its own. utils.py wraps these into a torch Dataset.

The configuration mirrors the NetVLAD tagger: the same `--variables` set names
select which per-track quantities are used, and the same `--signal` names
select the classification task.

Per-track quantities available (all measured relative to the jet axis where
that makes sense, and z-scored with training-set statistics):

    log_pt       log(fPt)
    log_pt_rel   log(fPt / fJetPt)
    d_eta        fEta - fJetEta
    d_phi        wrap(fPhi - fJetPhi) into (-pi, pi]
    dca_xy       log1p(fDCA_xy)              tail-compressed
    dca_z        signed log1p(fDCA_z)        tail-compressed
    sip3d        signed log1p(fSIP3D)        tail-compressed
    delta_r      fDeltaR
    z            fZ
    m            fZ * fDeltaR^2  (the NetVLAD 'fM' proxy)

Labels come from mTag (1/2/3 = light/charm/bottom); `--signal` maps them.
"""

import glob

import numpy as np

# Every quantity the loader can compute, in fixed column order.
FEATURE_NAMES = [
    'log_pt', 'log_pt_rel', 'd_eta', 'd_phi',
    'dca_xy', 'dca_z', 'sip3d', 'delta_r', 'z', 'm',
]
N_ALL_FEATURES = len(FEATURE_NAMES)

# Named feature sets. The first five mirror the NetVLAD --variables sets
# (with the transformer's jet-relative versions of pT/eta/phi); 'full' adds
# the two quantities NetVLAD never used: pT relative to the jet, and SIP3D.
VARIABLE_SETS = {
    'trk':        ['log_pt', 'd_eta', 'd_phi'],
    'vtx':        ['dca_z', 'dca_xy'],
    'trkvtx':     ['log_pt', 'd_eta', 'd_phi', 'dca_z', 'dca_xy'],
    'trkfrag':    ['log_pt', 'd_eta', 'd_phi', 'z', 'delta_r', 'm'],
    'trkvtxfrag': ['log_pt', 'd_eta', 'd_phi', 'z', 'delta_r', 'm', 'dca_z', 'dca_xy'],
    'full':       list(FEATURE_NAMES),
}

# Classification tasks. 'lcb' is the transformer's native 3-class output; the
# four binary modes match the NetVLAD --signal options exactly.
SIGNALS = ['lcb', 'hf_vs_l', 'c_vs_b', 'c_vs_all', 'b_vs_all']
CLASS_NAMES = ['light', 'charm', 'bottom']

_JAGGED = ['fPt', 'fEta', 'fPhi', 'fDCA_z', 'fDCA_xy', 'fDeltaR', 'fZ', 'fSIP3D']
_SCALAR = ['mTag', 'fJetPt', 'fJetEta', 'fJetPhi']
_BRANCHES = _JAGGED + _SCALAR


def feature_indices(variables):
    """Column indices (into FEATURE_NAMES) for a named feature set."""
    if variables not in VARIABLE_SETS:
        raise ValueError(f'unknown variable set {variables!r}; '
                         f'choose from {sorted(VARIABLE_SETS)}')
    return [FEATURE_NAMES.index(n) for n in VARIABLE_SETS[variables]]


def n_features(variables):
    return len(VARIABLE_SETS[variables])


def class_names(signal):
    """Output class names for a signal mode (index = model output index)."""
    if signal == 'lcb':
        return list(CLASS_NAMES)
    if signal == 'hf_vs_l':
        return ['light', 'heavy_flavour']
    if signal == 'c_vs_b':
        return ['charm', 'bottom']
    if signal == 'c_vs_all':
        return ['not_charm', 'charm']
    if signal == 'b_vs_all':
        return ['not_bottom', 'bottom']
    raise ValueError(f'unknown signal {signal!r}; choose from {SIGNALS}')


def apply_signal(y3, signal):
    """Map 3-class labels (0/1/2 = l/c/b) onto a task.

    Returns (y, keep) where `keep` is a boolean selection of jets to use
    (only c_vs_b drops jets, exactly as the NetVLAD dataset does).
    """
    if signal == 'lcb':
        return y3, np.ones(len(y3), dtype=bool)
    if signal == 'hf_vs_l':
        return (y3 != 0).astype(np.int64), np.ones(len(y3), dtype=bool)
    if signal == 'c_vs_all':
        return (y3 == 1).astype(np.int64), np.ones(len(y3), dtype=bool)
    if signal == 'b_vs_all':
        return (y3 == 2).astype(np.int64), np.ones(len(y3), dtype=bool)
    if signal == 'c_vs_b':
        keep = y3 != 0
        return (y3[keep] == 2).astype(np.int64), keep
    raise ValueError(f'unknown signal {signal!r}; choose from {SIGNALS}')


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _wrap_phi(dphi):
    """Wrap angle differences into (-pi, pi]."""
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def expand_paths(spec):
    """Expand a comma-separated list of paths/globs into a sorted file list."""
    if isinstance(spec, (list, tuple)):
        parts = list(spec)
    else:
        parts = [p for p in str(spec).split(',') if p]
    files = []
    for part in parts:
        matches = sorted(glob.glob(part.strip()))
        files.extend(matches if matches else [part.strip()])
    return files


def _chunk_to_arrays(a, n_max, cols, log_transform=True):
    """Vectorized [n, n_max, len(cols)] features + mask + 3-class labels."""
    import awkward as ak

    pt = a['fPt']
    order = ak.argsort(pt, axis=1, ascending=False)   # hardest tracks first

    def padded(name):
        v = a[name][order]
        v = ak.pad_none(v, n_max, axis=1, clip=True)
        return ak.to_numpy(ak.fill_none(v, 0.0)).astype(np.float64)

    pt_p = padded('fPt')
    eta_p = padded('fEta')
    phi_p = padded('fPhi')
    dcaz_p = padded('fDCA_z')
    dcaxy_p = padded('fDCA_xy')
    dr_p = padded('fDeltaR')
    z_p = padded('fZ')
    sip_p = padded('fSIP3D')

    counts = ak.to_numpy(ak.num(pt, axis=1)).astype(np.int64)
    n = counts.shape[0]
    mask = np.arange(n_max)[None, :] >= np.minimum(counts, n_max)[:, None]

    jet_pt = ak.to_numpy(a['fJetPt']).astype(np.float64)[:, None]
    jet_eta = ak.to_numpy(a['fJetEta']).astype(np.float64)[:, None]
    jet_phi = ak.to_numpy(a['fJetPhi']).astype(np.float64)[:, None]

    full = np.empty((n, n_max, N_ALL_FEATURES), dtype=np.float32)
    if log_transform:
        full[:, :, 0] = np.log(np.clip(pt_p, 1e-6, None))
        full[:, :, 1] = np.log(np.clip(pt_p / np.clip(jet_pt, 1e-6, None), 1e-6, None))
        full[:, :, 4] = np.log1p(np.clip(dcaxy_p, 0.0, None))
        full[:, :, 5] = _signed_log1p(dcaz_p)
        full[:, :, 6] = _signed_log1p(sip_p)
    else:                      # ablation: raw values, z-scoring only
        full[:, :, 0] = pt_p
        full[:, :, 1] = pt_p / np.clip(jet_pt, 1e-6, None)
        full[:, :, 4] = dcaxy_p
        full[:, :, 5] = dcaz_p
        full[:, :, 6] = sip_p
    full[:, :, 2] = eta_p - jet_eta
    full[:, :, 3] = _wrap_phi(phi_p - jet_phi)
    full[:, :, 7] = dr_p
    full[:, :, 8] = z_p
    full[:, :, 9] = z_p * dr_p * dr_p                 # NetVLAD's 'fM' proxy

    X = full[:, :, cols].copy()
    del full
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    X[mask] = 0.0                                     # padded rows exactly zero

    y3 = ak.to_numpy(a['mTag']).astype(np.int64) - 1
    return X, mask, y3


def count_jets(file_paths, tree_name='T'):
    """Total number of jets across the given files (reads headers only)."""
    import uproot
    return sum(uproot.open(p)[tree_name].num_entries for p in expand_paths(file_paths))


def build_dataset(file_paths, n_max=16, variables='full', tree_name='T',
                  max_jets=None, chunk_size=200_000, verbose=True,
                  log_transform=True):
    """Stream one or more ROOT files into padded arrays.

    Output arrays are preallocated and filled in place, so peak memory is about
    one copy of X: n_jets * n_max * n_features * 4 bytes.

    Returns
    -------
    X    : float32 [n_jets, n_max, n_features]  zero-padded features
    mask : bool    [n_jets, n_max]              True where PADDING
    y3   : int64   [n_jets]                     0/1/2 = light/charm/bottom
    """
    import uproot

    cols = feature_indices(variables)
    paths = expand_paths(file_paths)
    n_total = count_jets(paths, tree_name)
    if max_jets is not None:
        n_total = min(n_total, int(max_jets))

    X = np.zeros((n_total, n_max, len(cols)), dtype=np.float32)
    mask = np.ones((n_total, n_max), dtype=bool)
    y3 = np.zeros(n_total, dtype=np.int64)
    if verbose:
        print(f'    allocating {X.nbytes / 1e9:.2f} GB for {n_total} jets '
              f'({variables}: {len(cols)} features)', flush=True)

    filled = 0
    for path in paths:
        if filled >= n_total:
            break
        tree = uproot.open(path)[tree_name]
        if verbose:
            print(f'    {path}: {tree.num_entries} jets', flush=True)
        for a in tree.iterate(_BRANCHES, step_size=chunk_size, library='ak'):
            cx, cm, cy = _chunk_to_arrays(a, n_max, cols, log_transform)
            take = min(len(cy), n_total - filled)
            X[filled:filled + take] = cx[:take]
            mask[filled:filled + take] = cm[:take]
            y3[filled:filled + take] = cy[:take]
            filled += take
            del cx, cm, cy
            if filled >= n_total:
                break

    if filled == 0:
        raise ValueError(f'no jets loaded from {file_paths!r}')
    if filled < n_total:
        X, mask, y3 = X[:filled], mask[:filled], y3[:filled]

    if y3.min() < 0 or y3.max() > 2:
        raise ValueError(f'unexpected mTag values (expected 1/2/3): {np.unique(y3 + 1)}')
    return X, mask, y3


def compute_norm_stats(X, mask):
    """Per-feature mean/std over REAL tracks only (padding excluded)."""
    valid = ~mask
    n_real = int(valid.sum())
    n_feat = X.shape[2]
    mean = np.zeros(n_feat, dtype=np.float64)
    sq = np.zeros(n_feat, dtype=np.float64)
    step = max(1, 2_000_000 // max(1, X.shape[1]))
    for i in range(0, X.shape[0], step):
        blk = X[i:i + step][valid[i:i + step]].astype(np.float64)
        mean += blk.sum(axis=0)
        sq += (blk ** 2).sum(axis=0)
    mean /= n_real
    std = np.sqrt(np.maximum(sq / n_real - mean ** 2, 0.0))
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X, mask, mean, std):
    """Z-score in place; padded rows stay exactly zero."""
    X -= mean[None, None, :]
    X /= std[None, None, :]
    X[mask] = 0.0
    return X


def class_weights(y, n_classes, scheme='sqrt'):
    """Class weights for imbalanced labels, normalized to mean 1.

    scheme='sqrt'    : w ~ 1/sqrt(freq)  (default; softer, more stable when the
                       imbalance is extreme, ~97.5/2.2/0.3 for l/c/b here)
    scheme='inverse' : w ~ 1/freq        (full re-balancing)
    scheme='none'    : all ones
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    if scheme == 'none':
        w = np.ones(n_classes)
    elif scheme == 'inverse':
        w = 1.0 / counts
    elif scheme == 'sqrt':
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f'unknown class weighting scheme: {scheme!r}')
    return (w / w.mean()).astype(np.float32)

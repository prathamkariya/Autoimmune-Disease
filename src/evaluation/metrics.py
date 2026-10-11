"""
Shared evaluation helpers (identical in behaviour to the inline helpers of
04_cross_validation.ipynb). Used from notebook 05 onwards.
"""
import numpy as np
from scipy.stats import binomtest


def macro_f1_fast(y, p, k=7):
    """Macro-F1 from the confusion matrix (verified against sklearn in 04)."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    return np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1))


def per_class_prf(y, p, k=7):
    """Per-class precision, recall, F1 (arrays of length k)."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return prec, rec, f1


def brier_multiclass(y, P):
    """Mean over patients of sum_c (P_c - 1[y==c])^2. 0 = perfect, 2 = worst."""
    return float(((P - np.eye(P.shape[1])[y]) ** 2).sum(1).mean())


def ece_top_label(y, P, bins=10):
    """Expected calibration error on the top-class confidence, equal-width bins."""
    conf, pred = P.max(1), P.argmax(1)
    hit = (pred == y)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(ece)


def mcnemar_exact(y, pa, pb):
    """Exact McNemar test on correctness. Returns (b, c, p): b = only A right, c = only B right."""
    ca, cb = (pa == y), (pb == y)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return b, c, (1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue))


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def bootstrap_f1_matrix(y, preds, B=2000, seed=42):
    """
    Paired bootstrap: resample patients B times (same resample for every
    configuration) and return (keys, F) where F[b, j] is the macro-F1 of
    configuration keys[j] on resample b. Differences of columns of F give
    paired bootstrap distributions of macro-F1 differences.
    """
    keys = list(preds)
    n = len(y)
    rng = np.random.default_rng(seed)
    F = np.empty((B, len(keys)))
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for j, k in enumerate(keys):
            F[b, j] = macro_f1_fast(yb, preds[k][idx])
    return keys, F

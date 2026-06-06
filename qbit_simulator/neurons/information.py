"""Information-theoretic neural analysis: MI and transfer entropy.

Mutual information:
    I(X; Y) = sum_{x,y} p(x,y) log(p(x,y) / (p(x) p(y)))

Transfer entropy (Schreiber 2000):
    TE(X → Y) = sum p(y_{t+1}, y_t, x_t) log(p(y_{t+1} | y_t, x_t)
                                              / p(y_{t+1} | y_t))

Both are estimated from histograms over discrete (binned) values. For
real-valued signals, bin the data first (e.g. by histogram).
"""

from __future__ import annotations

import numpy as np


def _joint_probs(*arrays: np.ndarray, n_bins: int = 8
                  ) -> tuple[np.ndarray, list]:
    """Joint histogram of K discrete arrays. Returns (P_joint, edges)."""
    arrs = [np.asarray(a).ravel() for a in arrays]
    edges = []
    binned = []
    for a in arrs:
        a_min, a_max = float(a.min()), float(a.max())
        if a_max == a_min:
            a_max = a_min + 1e-9
        e = np.linspace(a_min, a_max, n_bins + 1)
        edges.append(e)
        b = np.clip(np.digitize(a, e) - 1, 0, n_bins - 1)
        binned.append(b)
    shape = tuple(n_bins for _ in arrs)
    P = np.zeros(shape)
    for idx in zip(*binned):
        P[idx] += 1
    P /= max(P.sum(), 1)
    return P, edges


def mutual_information(X: np.ndarray, Y: np.ndarray,
                        n_bins: int = 8) -> float:
    """Empirical MI of two scalar time series."""
    P, _ = _joint_probs(X, Y, n_bins=n_bins)
    P_x = P.sum(axis=1, keepdims=True)
    P_y = P.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(P > 0, P / (P_x * P_y + 1e-12), 1.0)
        log_term = np.where(P > 0, np.log(ratio + 1e-12), 0.0)
        return float((P * log_term).sum())


def entropy(X: np.ndarray, n_bins: int = 8) -> float:
    P, _ = _joint_probs(X, n_bins=n_bins)
    p = P[P > 0]
    return float(-(p * np.log(p)).sum())


def transfer_entropy(X: np.ndarray, Y: np.ndarray,
                      n_bins: int = 4, lag: int = 1) -> float:
    """TE(X → Y) with history length 1.

    Estimates how much knowing X_t reduces uncertainty about Y_{t+1}
    beyond what is already known from Y_t.
    """
    Y_future = Y[lag:]
    Y_past   = Y[:-lag]
    X_past   = X[:-lag]
    # Joint P(Y_+, Y_p, X_p).
    P_yyx, _ = _joint_probs(Y_future, Y_past, X_past, n_bins=n_bins)
    P_yy,  _ = _joint_probs(Y_future, Y_past,           n_bins=n_bins)
    P_yx,  _ = _joint_probs(Y_past,   X_past,           n_bins=n_bins)
    P_y, _   = _joint_probs(Y_past, n_bins=n_bins)
    te = 0.0
    nb = n_bins
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                p_yyx = P_yyx[i, j, k]
                if p_yyx <= 0:
                    continue
                p_yx = P_yx[j, k]
                p_yy = P_yy[i, j]
                p_y  = P_y[j]
                if p_yx <= 0 or p_yy <= 0 or p_y <= 0:
                    continue
                te += p_yyx * np.log((p_yyx * p_y) / (p_yy * p_yx))
    return float(te)

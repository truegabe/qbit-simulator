"""Compressed sensing relay -- random projection + ISTA recovery.

Unlike SparseRelay (which needs to know WHICH k dimensions are large),
compressed sensing works by:

  1. ENCODER:  y = Phi @ x    (m x n random Gaussian matrix, m << n)
               Send the m-dimensional measurement vector y.

  2. DECODER:  recover x from y using ISTA (Iterative Shrinkage-Thresholding)
               Exploits the fact that most natural signals are sparse in
               some basis (pixels -> wavelets, spikes -> identity, etc.)

Why this is powerful
--------------------
- No codebook training: the random matrix IS the codebook.
- Exact recovery is guaranteed (RIP theorem) when k < m / (2 log n/k),
  where k = true sparsity of x.
- Recovery works even without knowing which dimensions are active.

Compression ratio: n/m   (e.g. 1024 -> 128 = 8x, no training needed)

Classes
-------
  RandomProjector        -- builds Phi, projects, reconstructs shape
  ISTASolver             -- sparse recovery via proximal gradient descent
  CompressedSensingRelay -- full pipeline with stats
  BasisPursuit           -- alternate L1 minimisation using gradient descent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# RandomProjector
# ---------------------------------------------------------------------------

class RandomProjector:
    """Gaussian random measurement matrix Phi (m x n).

    The matrix is fixed once built and shared between encoder and decoder.
    Columns are normalised so that ||Phi x||_2 ~ ||x||_2.

    Parameters
    ----------
    n_dims : original signal length
    m_meas : number of measurements to take (m < n for compression)
    seed   : reproducible matrix generation
    """

    def __init__(self, n_dims: int, m_meas: int,
                 seed: int = 0) -> None:
        if m_meas >= n_dims:
            raise ValueError(f"m_meas ({m_meas}) must be < n_dims ({n_dims}) "
                             "for compression")
        self.n_dims = n_dims
        self.m_meas = m_meas
        rng         = np.random.default_rng(seed)
        raw         = rng.standard_normal((m_meas, n_dims))
        # Normalise columns so each measurement is an isometry in expectation.
        self.Phi: np.ndarray = raw / np.sqrt(m_meas)

    # ------------------------------------------------------------------
    def project(self, x: np.ndarray) -> np.ndarray:
        """y = Phi @ x  :  (n,) -> (m,)"""
        return self.Phi @ np.asarray(x, dtype=np.float64).ravel()

    def pseudo_inv(self, y: np.ndarray) -> np.ndarray:
        """Naive pseudo-inverse (baseline, not sparse-aware).  (m,) -> (n,)"""
        return self.Phi.T @ np.asarray(y, dtype=np.float64).ravel()

    @property
    def compression_ratio(self) -> float:
        return self.n_dims / self.m_meas

    def __repr__(self) -> str:
        return (f"RandomProjector(n={self.n_dims}, m={self.m_meas}, "
                f"cr={self.compression_ratio:.1f}x)")


# ---------------------------------------------------------------------------
# ISTASolver  (Iterative Shrinkage-Thresholding Algorithm)
# ---------------------------------------------------------------------------

class ISTASolver:
    """Recover a sparse signal from compressed measurements.

    Solves:  min_x  (1/2)||y - Phi x||^2  +  lam * ||x||_1

    Algorithm (ISTA):
        x <- S_lam(x + step * Phi.T @ (y - Phi @ x))

    where S_lam is the soft-thresholding operator.

    Parameters
    ----------
    n_iters   : number of gradient steps
    lam       : L1 regularisation weight
    step      : gradient step size (use 1 / ||Phi||^2 for convergence)
    tol       : early stop when ||x_new - x_old||_2 < tol
    warm_start: if True, previous solution is reused as initialisation
    """

    def __init__(self, n_iters: int = 200, lam: float = 0.05,
                 step: Optional[float] = None, tol: float = 1e-6,
                 warm_start: bool = False) -> None:
        self.n_iters    = n_iters
        self.lam        = lam
        self.step       = step     # None -> auto-compute from Phi
        self.tol        = tol
        self.warm_start = warm_start
        self._x_prev: Optional[np.ndarray] = None

    @staticmethod
    def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

    def solve(self, y: np.ndarray,
              projector: RandomProjector) -> tuple[np.ndarray, dict]:
        """Recover x from y = Phi @ x.

        Returns
        -------
        x_hat  : recovered signal (n_dims,)
        info   : dict with n_iters_run, residual, sparsity
        """
        Phi   = projector.Phi
        PhiT  = Phi.T
        y     = np.asarray(y, dtype=np.float64).ravel()

        # Step size: 1/L where L = Lipschitz constant of grad f = ||Phi||_F^2
        # (Frobenius norm^2 is a valid upper bound on the spectral norm^2).
        if self.step is None:
            L    = np.linalg.norm(Phi, 'fro') ** 2
            step = 1.0 / max(L, 1e-12)
        else:
            step = self.step

        # Initialisation.
        if self.warm_start and self._x_prev is not None:
            x = self._x_prev.copy()
        else:
            x = PhiT @ y

        thresh = self.lam * step
        n_run  = 0
        for i in range(self.n_iters):
            residual = y - Phi @ x
            grad     = PhiT @ residual
            x_new    = self._soft_threshold(x + step * grad, thresh)
            if np.linalg.norm(x_new - x) < self.tol:
                x     = x_new
                n_run = i + 1
                break
            x     = x_new
            n_run = i + 1

        if self.warm_start:
            self._x_prev = x.copy()

        residual_norm = float(np.linalg.norm(y - Phi @ x))
        sparsity      = float(np.mean(np.abs(x) < 1e-6))

        return x, {"n_iters_run": n_run,
                   "residual":    residual_norm,
                   "sparsity":    sparsity}


# ---------------------------------------------------------------------------
# CompressedSensingRelay
# ---------------------------------------------------------------------------

class CompressedSensingRelay:
    """Full random-projection encode -> channel -> ISTA decode pipeline.

    Parameters
    ----------
    n_dims    : original signal dimensionality
    m_meas    : number of measurements (< n_dims); controls compression
    seed      : random matrix seed
    n_iters   : ISTA iterations
    lam       : ISTA L1 weight
    noise_std : additive Gaussian channel noise on y
    warm_start: carry solver state between calls (faster on slow signals)
    """

    def __init__(self, n_dims: int, m_meas: Optional[int] = None,
                 seed: int = 0, n_iters: int = 200, lam: float = 0.05,
                 noise_std: float = 0.0, warm_start: bool = False) -> None:
        self.n_dims    = n_dims
        self.m_meas    = m_meas or max(1, n_dims // 4)
        self.noise_std = noise_std

        self.projector = RandomProjector(n_dims, self.m_meas, seed=seed)
        self.solver    = ISTASolver(n_iters=n_iters, lam=lam,
                                    warm_start=warm_start)
        self._n_sent:  int   = 0
        self._err_acc: float = 0.0

    # ------------------------------------------------------------------
    def transmit(self, x: np.ndarray,
                 rng: Optional[np.random.Generator] = None
                 ) -> tuple[np.ndarray, dict]:
        """Compress -> (noise) -> reconstruct.

        Returns
        -------
        x_rec  : reconstructed signal (n_dims,)
        stats  : dict with compression_ratio, reconstruction_error,
                 ista_iters, ista_residual, sparsity, bits_sent
        """
        x   = np.asarray(x, dtype=np.float64).ravel()
        y   = self.projector.project(x)

        if self.noise_std > 0:
            rng = rng or np.random.default_rng()
            y   = y + rng.standard_normal(len(y)) * self.noise_std

        x_rec, info = self.solver.solve(y, self.projector)
        x_rec       = x_rec[:self.n_dims]
        if len(x_rec) < self.n_dims:
            x_rec = np.pad(x_rec, (0, self.n_dims - len(x_rec)))

        rec_err = float(np.linalg.norm(x - x_rec) /
                         (np.linalg.norm(x) + 1e-12))

        self._n_sent  += 1
        self._err_acc += rec_err

        bits_dense  = self.n_dims * 32
        bits_sparse = self.m_meas * 32   # m float32 measurements

        stats = {
            "compression_ratio":    bits_dense / max(bits_sparse, 1),
            "reconstruction_error": rec_err,
            "mean_error_so_far":    self._err_acc / self._n_sent,
            "ista_iters":           info["n_iters_run"],
            "ista_residual":        info["residual"],
            "sparsity":             info["sparsity"],
            "bits_sent":            bits_sparse,
            "bits_dense":           bits_dense,
            "m_meas":               self.m_meas,
            "n_dims":               self.n_dims,
        }
        return x_rec, stats

    def reset(self) -> None:
        """Reset solver warm-start state and accumulators."""
        self.solver._x_prev = None
        self._n_sent        = 0
        self._err_acc       = 0.0

    def __repr__(self) -> str:
        return (f"CompressedSensingRelay(n={self.n_dims}, m={self.m_meas}, "
                f"cr={self.projector.compression_ratio:.1f}x, "
                f"lam={self.solver.lam})")


# ---------------------------------------------------------------------------
# BasisPursuit (gradient descent L1 via FISTA -- faster variant)
# ---------------------------------------------------------------------------

class FISTASolver(ISTASolver):
    """Fast ISTA (FISTA) -- Nesterov momentum, same interface as ISTASolver.

    Convergence: O(1/k^2) vs O(1/k) for plain ISTA.
    """

    def solve(self, y: np.ndarray,
              projector: RandomProjector) -> tuple[np.ndarray, dict]:
        Phi   = projector.Phi
        PhiT  = Phi.T
        y     = np.asarray(y, dtype=np.float64).ravel()

        if self.step is None:
            L    = np.linalg.norm(Phi, 'fro') ** 2
            step = 1.0 / max(L, 1e-12)
        else:
            step = self.step

        if self.warm_start and self._x_prev is not None:
            x = self._x_prev.copy()
        else:
            x = PhiT @ y

        thresh = self.lam * step
        z      = x.copy()
        t      = 1.0
        n_run  = 0

        for i in range(self.n_iters):
            x_new = self._soft_threshold(
                z + step * (PhiT @ (y - Phi @ z)), thresh)
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z     = x_new + ((t - 1) / t_new) * (x_new - x)
            if np.linalg.norm(x_new - x) < self.tol:
                x     = x_new
                n_run = i + 1
                break
            x, t  = x_new, t_new
            n_run = i + 1

        if self.warm_start:
            self._x_prev = x.copy()

        residual_norm = float(np.linalg.norm(y - Phi @ x))
        sparsity      = float(np.mean(np.abs(x) < 1e-6))
        return x, {"n_iters_run": n_run,
                   "residual":    residual_norm,
                   "sparsity":    sparsity}

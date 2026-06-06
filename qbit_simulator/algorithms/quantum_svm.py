"""End-to-end Quantum Support Vector Machine.

Builds on the existing `quantum_kernel.py` (ZZ feature map) to provide:

  1. **Kernel-based SVM training**: solve the SVM dual problem using
     the precomputed quantum kernel matrix → support vectors + bias.
  2. **Prediction**: given trained α's, classify new points by
     evaluating the quantum kernel against the support vectors.
  3. **Datasets**: small classification toys (XOR, moons, circles)
     for end-to-end demos.

SVM dual problem (soft-margin, C-regularized):

    max_α   sum_i α_i  −  (1/2) sum_{ij} α_i α_j y_i y_j K_ij
    s.t.    0 ≤ α_i ≤ C
            sum_i α_i y_i = 0

Solved via scipy.optimize.minimize (SLSQP) for small datasets. The
"quantumness" enters through K_ij = |⟨φ(x_i) | φ(x_j)⟩|² built via the
quantum-kernel feature map.

This is one of the canonical applications of NISQ-era quantum kernels
(Havlíček-Córcoles-Temme et al. 2019).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .quantum_kernel import quantum_kernel_matrix, quantum_kernel_value


# ----------------------------------------------------------------------------
# Toy datasets
# ----------------------------------------------------------------------------

def xor_dataset(n_per_class: int = 10,
                 noise: float = 0.1,
                 rng: np.random.Generator | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """4-cluster XOR problem in 2D.

    Class +1: clusters at (+1, +1) and (-1, -1).
    Class -1: clusters at (+1, -1) and (-1, +1).
    """
    rng = rng or np.random.default_rng()
    centers_pos = [(1.0, 1.0), (-1.0, -1.0)]
    centers_neg = [(1.0, -1.0), (-1.0, 1.0)]
    X, y = [], []
    for c in centers_pos:
        X.append(np.array(c) + noise * rng.normal(size=(n_per_class, 2)))
        y.extend([+1] * n_per_class)
    for c in centers_neg:
        X.append(np.array(c) + noise * rng.normal(size=(n_per_class, 2)))
        y.extend([-1] * n_per_class)
    return np.vstack(X), np.array(y)


def two_moons_dataset(n_per_class: int = 10,
                       noise: float = 0.05,
                       rng: np.random.Generator | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-moon clusters."""
    rng = rng or np.random.default_rng()
    t = np.linspace(0, np.pi, n_per_class)
    X_pos = np.column_stack([np.cos(t), np.sin(t)])
    X_neg = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    X = np.vstack([X_pos, X_neg]) + noise * rng.normal(size=(2 * n_per_class, 2))
    y = np.concatenate([+np.ones(n_per_class, dtype=int),
                        -np.ones(n_per_class, dtype=int)])
    return X, y


def circles_dataset(n_per_class: int = 10,
                     noise: float = 0.05,
                     rng: np.random.Generator | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Concentric circles: inner = class +1, outer = class -1."""
    rng = rng or np.random.default_rng()
    t = np.linspace(0, 2 * np.pi, n_per_class, endpoint=False)
    X_inner = 0.5 * np.column_stack([np.cos(t), np.sin(t)])
    X_outer = 1.0 * np.column_stack([np.cos(t), np.sin(t)])
    X = np.vstack([X_inner, X_outer]) + noise * rng.normal(size=(2 * n_per_class, 2))
    y = np.concatenate([+np.ones(n_per_class, dtype=int),
                         -np.ones(n_per_class, dtype=int)])
    return X, y


# ----------------------------------------------------------------------------
# Trained SVM model
# ----------------------------------------------------------------------------

@dataclass
class QuantumSVM:
    """A trained kernel SVM with quantum feature map.

    Attributes:
        X_train:        training inputs.
        y_train:        training labels (±1).
        alpha:          optimized Lagrange multipliers.
        bias:           SVM bias term.
        reps:           feature-map repetitions (ZZ feature map).
        C:              soft-margin parameter.
        support_idx:    indices with α > tol (the support vectors).
    """
    X_train:     np.ndarray
    y_train:     np.ndarray
    alpha:       np.ndarray
    bias:        float
    reps:        int
    C:           float
    support_idx: np.ndarray

    def decision_function(self, x: np.ndarray) -> float:
        """f(x) = sum_i α_i y_i K(x_i, x) + bias."""
        x = np.asarray(x, dtype=np.float64)
        total = self.bias
        for i in self.support_idx:
            k_val = quantum_kernel_value(self.X_train[i], x, reps=self.reps)
            total += self.alpha[i] * self.y_train[i] * k_val
        return float(total)

    def predict(self, x: np.ndarray) -> int:
        return int(np.sign(self.decision_function(x))) or 1


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_quantum_svm(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    reps: int = 2,
    tol: float = 1e-6,
) -> QuantumSVM:
    """Fit a quantum SVM by solving the dual problem.

    Args:
        X:    training inputs, shape (n_samples, n_features).
        y:    labels in {-1, +1}, shape (n_samples,).
        C:    soft-margin parameter.
        reps: ZZ-feature-map repetitions.
        tol:  threshold for "support vector" (α > tol).

    Returns:
        a fitted `QuantumSVM`.
    """
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be (n_samples, n_features)")
    if set(np.unique(y)) - {-1.0, 1.0}:
        raise ValueError("y must contain only -1 and +1")

    n = len(y)
    # Build the quantum kernel matrix.
    K = quantum_kernel_matrix(X, reps=reps)
    # The dual objective: minimize 0.5 α^T Q α - 1^T α
    # where Q[i,j] = y_i y_j K_ij.
    YY = np.outer(y, y)
    Q = YY * K

    def neg_dual(alpha):
        return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

    def neg_dual_grad(alpha):
        return Q @ alpha - np.ones(n)

    # Constraint: sum_i α_i y_i = 0.
    constraint = {"type": "eq",
                  "fun":  lambda a: a @ y,
                  "jac":  lambda a: y}
    bounds = [(0.0, C)] * n

    # Warm start.
    alpha0 = np.full(n, 0.5 * C)
    res = minimize(neg_dual, alpha0, jac=neg_dual_grad,
                    method="SLSQP", bounds=bounds, constraints=[constraint],
                    options={"maxiter": 200, "ftol": 1e-9})
    alpha = res.x

    # Support vectors.
    support_idx = np.where(alpha > tol)[0]

    # Compute bias: pick a margin vector (0 < α < C) and use
    # b = y_i - sum_j α_j y_j K(x_j, x_i).
    margin_idx = [i for i in support_idx if alpha[i] < C - tol]
    if not margin_idx:
        margin_idx = list(support_idx)
    b_values = []
    for i in margin_idx:
        b_i = y[i] - sum(alpha[j] * y[j] * K[j, i] for j in support_idx)
        b_values.append(b_i)
    bias = float(np.mean(b_values)) if b_values else 0.0

    return QuantumSVM(
        X_train=X, y_train=y.astype(int), alpha=alpha, bias=bias,
        reps=reps, C=C, support_idx=support_idx,
    )


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def accuracy(model: QuantumSVM, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    correct = sum(1 for xi, yi in zip(X, y) if model.predict(xi) == yi)
    return correct / len(y)


def train_test_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.3,
                       rng: np.random.Generator | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple random train/test split."""
    rng = rng or np.random.default_rng()
    n = len(y)
    indices = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

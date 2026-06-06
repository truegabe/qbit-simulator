"""Quantum Boltzmann Machine (QBM): generative model on quantum spins.

A classical Boltzmann machine assigns probability

    p(s) = exp(-β E(s)) / Z,    E(s) = -sum_i h_i s_i - sum_{i<j} J_ij s_i s_j

to each binary configuration s ∈ {-1, +1}^N. Training adjusts (h, J) so
the marginal distribution matches a target dataset.

A **quantum Boltzmann machine** (Amin et al. 2018) generalizes this:
classical spins become Pauli operators, and the energy becomes a
Hamiltonian:

    H(θ) = -sum_i h_i Z_i - sum_{ij} J_ij Z_i Z_j - sum_i Γ_i X_i

with the last term (transverse field) the new "quantum" ingredient.
The "Boltzmann state" is the GIBBS state:

    rho_β(θ) = exp(-β H(θ)) / Tr(exp(-β H(θ)))

Marginal on the COMPUTATIONAL basis (visible units) is what we
optimize against the dataset.

Training: gradient descent on the KL divergence

    D(p_data || p_model) = -sum_v p_data(v) log p_model(v)

where p_model(v) = ⟨v | rho_β(θ) | v⟩ for visible-only QBM (no hidden
units, the simplest case).

This module implements:

  - `gibbs_state(H, beta)`: exact computation of exp(-βH)/Z for small N.
  - `qbm_energy_pauli(weights, n_visible, hidden_dim)`: assemble the QBM
    Hamiltonian from (h, J, Γ).
  - `qbm_log_likelihood(p_model, data)`: average log-prob of data
    samples under p_model.
  - `train_qbm_visible(data, n_visible, beta, lr, n_iter)`: gradient
    descent training of a visible-only QBM.

For tractable simulation, we restrict to N ≤ 8 visible qubits (i.e.
dim = 256). For larger systems you'd use Monte Carlo / variational
approximation.
"""

from __future__ import annotations

import numpy as np


# Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _pauli_kron(N: int, q: int, op: np.ndarray) -> np.ndarray:
    """Embed a 2x2 op on qubit q (MSB-first) into the full 2^N Hilbert space."""
    out = np.array([[1.0 + 0j]])
    for k in range(N):
        out = np.kron(out, op if k == q else _I)
    return out


def _pauli_two_body(N: int, i: int, j: int, op_i: np.ndarray, op_j: np.ndarray
                     ) -> np.ndarray:
    return _pauli_kron(N, i, op_i) @ _pauli_kron(N, j, op_j)


# ----------------------------------------------------------------------------
# Hamiltonian builder
# ----------------------------------------------------------------------------

def qbm_hamiltonian(
    h: np.ndarray,        # field on each spin (length N)
    J: np.ndarray,        # coupling matrix (N × N, upper-triangular used)
    Gamma: np.ndarray,    # transverse field (length N) — quantum ingredient
) -> np.ndarray:
    """Build the QBM Hamiltonian:

        H = -sum_i h_i Z_i - sum_{i<j} J_ij Z_i Z_j - sum_i Γ_i X_i
    """
    N = len(h)
    if J.shape != (N, N):
        raise ValueError(f"J must be ({N}, {N}), got {J.shape}")
    if len(Gamma) != N:
        raise ValueError(f"Gamma must have length {N}")
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(N):
        H -= h[i] * _pauli_kron(N, i, _Z)
        H -= Gamma[i] * _pauli_kron(N, i, _X)
    for i in range(N):
        for j in range(i + 1, N):
            H -= J[i, j] * _pauli_two_body(N, i, j, _Z, _Z)
    return H


# ----------------------------------------------------------------------------
# Gibbs state and marginals
# ----------------------------------------------------------------------------

def gibbs_state(H: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """ρ_β = exp(-β H) / Tr exp(-β H) via eigendecomposition."""
    eigs, V = np.linalg.eigh(H)
    eigs = eigs - eigs[0]   # shift so largest weight = 1 (numerical stability)
    weights = np.exp(-beta * eigs)
    weights /= weights.sum()
    return (V * weights) @ V.conj().T


def computational_basis_distribution(rho: np.ndarray) -> np.ndarray:
    """p(v) = ⟨v | ρ | v⟩ — the diagonal of ρ in the computational basis."""
    return np.real(np.diag(rho))


# ----------------------------------------------------------------------------
# Data fitting
# ----------------------------------------------------------------------------

def empirical_distribution(samples: list[int], n_bits: int) -> np.ndarray:
    """Histogram samples into a probability over 2^n_bits states."""
    p = np.zeros(2 ** n_bits, dtype=np.float64)
    for s in samples:
        p[s] += 1.0
    p /= len(samples)
    return p


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) = sum_v p_v log(p_v / q_v)."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    return float(np.sum(p * np.log(p / q)))


def train_qbm_visible(
    data_distribution: np.ndarray,
    n_visible: int,
    beta: float = 1.0,
    lr: float = 0.1,
    n_iter: int = 100,
    include_transverse: bool = True,
    rng: np.random.Generator | None = None,
) -> dict:
    """Train a visible-only QBM by gradient descent on KL divergence.

    Args:
        data_distribution: target distribution as a length-2^N array.
        n_visible:          number of visible spins.
        beta:               inverse temperature (fixed).
        lr:                 learning rate.
        n_iter:             gradient-descent iterations.
        include_transverse: if False, fix Γ_i = 0 (classical BM limit).
        rng:                generator (for init).

    Returns:
        dict with optimized h, J, Γ, final KL, KL history.
    """
    rng = rng or np.random.default_rng()
    h = 0.1 * rng.normal(size=n_visible)
    J = 0.1 * rng.normal(size=(n_visible, n_visible))
    J = np.triu(J, k=1)   # use only upper triangle
    Gamma = (0.1 * rng.normal(size=n_visible) if include_transverse
              else np.zeros(n_visible))

    history = []
    for it in range(n_iter):
        H = qbm_hamiltonian(h, J, Gamma)
        rho = gibbs_state(H, beta)
        p_model = computational_basis_distribution(rho)
        kl = kl_divergence(data_distribution, p_model)
        history.append(kl)

        # Numerical gradient via finite differences (small problem).
        eps = 1e-4
        for i in range(n_visible):
            h[i] += eps
            H_p = qbm_hamiltonian(h, J, Gamma)
            p_p = computational_basis_distribution(gibbs_state(H_p, beta))
            h[i] -= 2 * eps
            H_m = qbm_hamiltonian(h, J, Gamma)
            p_m = computational_basis_distribution(gibbs_state(H_m, beta))
            h[i] += eps
            grad = (kl_divergence(data_distribution, p_p)
                     - kl_divergence(data_distribution, p_m)) / (2 * eps)
            h[i] -= lr * grad

        for i in range(n_visible):
            for j in range(i + 1, n_visible):
                J[i, j] += eps
                H_p = qbm_hamiltonian(h, J, Gamma)
                p_p = computational_basis_distribution(gibbs_state(H_p, beta))
                J[i, j] -= 2 * eps
                H_m = qbm_hamiltonian(h, J, Gamma)
                p_m = computational_basis_distribution(gibbs_state(H_m, beta))
                J[i, j] += eps
                grad = (kl_divergence(data_distribution, p_p)
                         - kl_divergence(data_distribution, p_m)) / (2 * eps)
                J[i, j] -= lr * grad

        if include_transverse:
            for i in range(n_visible):
                Gamma[i] += eps
                H_p = qbm_hamiltonian(h, J, Gamma)
                p_p = computational_basis_distribution(gibbs_state(H_p, beta))
                Gamma[i] -= 2 * eps
                H_m = qbm_hamiltonian(h, J, Gamma)
                p_m = computational_basis_distribution(gibbs_state(H_m, beta))
                Gamma[i] += eps
                grad = (kl_divergence(data_distribution, p_p)
                         - kl_divergence(data_distribution, p_m)) / (2 * eps)
                Gamma[i] -= lr * grad

    H_final = qbm_hamiltonian(h, J, Gamma)
    p_final = computational_basis_distribution(gibbs_state(H_final, beta))
    return {
        "h":              h,
        "J":              J,
        "Gamma":          Gamma,
        "p_model":        p_final,
        "kl_final":       kl_divergence(data_distribution, p_final),
        "kl_history":     history,
    }


def sample_qbm(rho: np.ndarray, n_samples: int = 100,
                rng: np.random.Generator | None = None) -> list[int]:
    """Sample bit-string outcomes from the Gibbs state."""
    rng = rng or np.random.default_rng()
    p = computational_basis_distribution(rho)
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return list(rng.choice(len(p), size=n_samples, p=p))

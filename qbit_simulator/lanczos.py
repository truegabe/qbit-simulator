"""Lanczos / Davidson eigensolvers for large sparse Hermitian operators.

For an exponentially-large Hilbert space (say 2^20 ≈ 10⁶), direct
diagonalization via `np.linalg.eigh` is impossible (O(d³) time, O(d²)
memory). Iterative Krylov methods recover the LOWEST few eigenvalues
in O(d · k) time using only matrix-vector products.

Two flavors:

  * **Lanczos** (1950): for Hermitian operators. Build the Krylov
    subspace K_k(H, v) = span{v, Hv, H²v, ..., H^{k-1}v}, then
    diagonalize the small tridiagonal projection T_k. Eigenvalues of
    T_k converge to extreme eigenvalues of H exponentially fast.

  * **Davidson** (1975): generalizes Lanczos with a preconditioner.
    Standard in quantum chemistry. We implement a simple unprecondi-
    tioned variant that's still much faster than full diagonalization.

Both methods only need a function `matvec(v) -> H @ v`, never the full
matrix H. This makes them ideal for:
  - Large Pauli-sum Hamiltonians (apply Pauli strings one by one).
  - Sparse matrices in scipy.sparse format.
  - Tensor-network Hamiltonians.

This module provides:

  - `lanczos_ground_state(matvec, n, k, tol, rng)`: Lanczos for the
    minimum eigenvalue + eigenvector.
  - `lanczos_lowest_k(matvec, n, n_eigs, k_dim, ...)`: get the lowest
    n_eigs eigenvalues.
  - `davidson_ground_state(matvec, diag, n, ...)`: Davidson with a
    diagonal preconditioner.
  - `matvec_from_pauli_op(op, n_qubits)`: build a matvec closure from
    a `PauliOp`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ----------------------------------------------------------------------------
# Lanczos for ground state
# ----------------------------------------------------------------------------

def lanczos_iterate(
    matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    k_dim: int = 50,
    v0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build a Krylov subspace via Lanczos. Returns (alphas, betas, basis).

    The tridiagonal matrix is:
        T = diag(alphas) + off-diag(betas).

    The Q matrix (columns are basis vectors) approximately satisfies
    H @ Q ≈ Q @ T (where H @ v is given by matvec).
    """
    rng = rng or np.random.default_rng()
    if v0 is None:
        v0 = rng.normal(size=n) + 1j * rng.normal(size=n)
    v0 = v0.astype(np.complex128)
    v0 = v0 / np.linalg.norm(v0)

    basis: list[np.ndarray] = [v0]
    alphas: list[float] = []
    betas: list[float] = []
    w = matvec(v0)
    alpha = float(np.real(np.vdot(v0, w)))
    alphas.append(alpha)
    w = w - alpha * v0

    for j in range(1, k_dim):
        beta = float(np.linalg.norm(w))
        if beta < 1e-12:
            break
        betas.append(beta)
        v_next = w / beta
        # Full re-orthogonalization to combat numerical drift.
        for prev in basis:
            v_next = v_next - np.vdot(prev, v_next) * prev
        v_next = v_next / np.linalg.norm(v_next)
        basis.append(v_next)
        w = matvec(v_next)
        alpha = float(np.real(np.vdot(v_next, w)))
        alphas.append(alpha)
        w = w - alpha * v_next - beta * basis[j - 1]

    return np.array(alphas), np.array(betas), basis


def lanczos_ground_state(
    matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    k_dim: int = 50,
    tol: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> dict:
    """Return the lowest eigenvalue + eigenvector of H via Lanczos.

    Args:
        matvec:  callable v → H @ v on a length-n vector.
        n:       Hilbert-space dimension.
        k_dim:   max Krylov subspace size.
        tol:     convergence tolerance.
        rng:     generator.

    Returns:
        dict with energy, eigenvector, n_iter, converged.
    """
    alphas, betas, basis = lanczos_iterate(matvec, n, k_dim, rng=rng)
    k = len(alphas)
    # Diagonalize the small tridiagonal T.
    T = np.diag(alphas)
    for i, b in enumerate(betas):
        T[i, i + 1] = b
        T[i + 1, i] = b
    eigvals, eigvecs = np.linalg.eigh(T)
    # Recover the corresponding eigenvector of the original H.
    coeffs = eigvecs[:, 0]      # ground-state coefficients in Krylov basis
    eigvec = np.zeros(n, dtype=np.complex128)
    for i, b in enumerate(basis):
        eigvec += coeffs[i] * b
    eigvec /= np.linalg.norm(eigvec)

    # Residual check: how close is H · eigvec − E · eigvec to zero?
    H_v = matvec(eigvec)
    E = float(eigvals[0])
    residual = float(np.linalg.norm(H_v - E * eigvec))

    return {
        "energy":      E,
        "eigenvector": eigvec,
        "n_iter":      k,
        "converged":   residual < tol,
        "residual":    residual,
        "all_eigs":    eigvals,
    }


def lanczos_lowest_k(
    matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    n_eigs: int = 3,
    k_dim: int = 50,
    rng: np.random.Generator | None = None,
) -> dict:
    """Lowest n_eigs eigenvalues + eigenvectors via Lanczos."""
    alphas, betas, basis = lanczos_iterate(matvec, n, k_dim, rng=rng)
    k = len(alphas)
    T = np.diag(alphas)
    for i, b in enumerate(betas):
        T[i, i + 1] = b
        T[i + 1, i] = b
    eigvals, eigvecs = np.linalg.eigh(T)
    actual_n = min(n_eigs, k)
    out_eigs = eigvals[:actual_n]
    out_vecs = []
    for r in range(actual_n):
        coeffs = eigvecs[:, r]
        v = np.zeros(n, dtype=np.complex128)
        for i, b in enumerate(basis):
            v += coeffs[i] * b
        v /= np.linalg.norm(v)
        out_vecs.append(v)
    return {
        "eigenvalues":  out_eigs,
        "eigenvectors": out_vecs,
        "k_dim":        k,
    }


# ----------------------------------------------------------------------------
# Davidson with diagonal preconditioning
# ----------------------------------------------------------------------------

def davidson_ground_state(
    matvec: Callable[[np.ndarray], np.ndarray],
    diag: np.ndarray,
    n: int,
    max_subspace: int = 30,
    tol: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> dict:
    """Davidson algorithm for the ground state.

    Args:
        matvec:        v → H @ v.
        diag:          length-n array of diagonal entries of H (used
                       as a preconditioner).
        n:             Hilbert dimension.
        max_subspace:  max subspace size before restart.
        tol:           residual-norm tolerance.
        rng:           generator.

    Returns:
        dict with energy, eigenvector, n_iter, converged.
    """
    rng = rng or np.random.default_rng()
    # Start from a random vector.
    v0 = rng.normal(size=n) + 1j * rng.normal(size=n)
    v0 = v0.astype(np.complex128) / np.linalg.norm(v0)
    V = [v0]
    HV = [matvec(v0)]
    eigvec = v0.copy()
    E = float(np.real(np.vdot(v0, HV[0])))
    converged = False
    for it in range(max_subspace):
        # Build the small subspace matrix.
        k = len(V)
        M = np.zeros((k, k), dtype=np.complex128)
        for i in range(k):
            for j in range(k):
                M[i, j] = np.vdot(V[i], HV[j])
        M = 0.5 * (M + M.conj().T)
        eigvals, eigvecs = np.linalg.eigh(M)
        y = eigvecs[:, 0]
        E = float(eigvals[0])
        # Form the Ritz vector + residual.
        v_ritz = sum(y[i] * V[i] for i in range(k))
        Hv_ritz = sum(y[i] * HV[i] for i in range(k))
        r = Hv_ritz - E * v_ritz
        res_norm = float(np.linalg.norm(r))
        eigvec = v_ritz
        if res_norm < tol:
            converged = True
            break
        # Davidson correction (preconditioned residual).
        denom = diag - E
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        delta = -r / denom
        # Orthogonalize against the existing subspace.
        for u in V:
            delta = delta - np.vdot(u, delta) * u
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-12:
            break
        delta = delta / delta_norm
        V.append(delta)
        HV.append(matvec(delta))

    return {
        "energy":      E,
        "eigenvector": eigvec / np.linalg.norm(eigvec),
        "n_iter":      it + 1,
        "converged":   converged,
        "residual":    res_norm,
    }


# ----------------------------------------------------------------------------
# Pauli-Op matvec adapter
# ----------------------------------------------------------------------------

def matvec_from_pauli_op(op, n_qubits: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a closure `v → H @ v` from a PauliOp without ever building
    the full matrix.

    For each Pauli string in the op, we apply it directly to the state
    vector via efficient bit-shifting (avoiding the 2^n × 2^n dense
    representation).
    """
    # Pre-process: turn each Pauli string into per-qubit operations.
    terms = list(op.terms)

    def apply(v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(v)
        for coef, s in terms:
            out += coef * _apply_pauli_string_to_vec(s, v, n_qubits)
        return out
    return apply


def _apply_pauli_string_to_vec(s: str, v: np.ndarray, n: int) -> np.ndarray:
    """Apply a Pauli string to a state vector (MSB-first qubits).

    Uses index-bit manipulation: each Pauli flips/phases certain bits.
    """
    out = v.copy().astype(np.complex128)
    # Apply each single-qubit Pauli to the corresponding qubit.
    for q, p in enumerate(s):
        if p == "I":
            continue
        # Reshape, apply 2x2 matrix on axis q, reshape back.
        bit_mask = 1 << (n - 1 - q)
        new_out = np.zeros_like(out)
        if p == "X":
            for idx in range(2 ** n):
                new_out[idx] = out[idx ^ bit_mask]
        elif p == "Y":
            for idx in range(2 ** n):
                bit = (idx >> (n - 1 - q)) & 1
                new_out[idx] = (-1j if bit == 0 else 1j) * out[idx ^ bit_mask]
        elif p == "Z":
            for idx in range(2 ** n):
                bit = (idx >> (n - 1 - q)) & 1
                new_out[idx] = (1 if bit == 0 else -1) * out[idx]
        out = new_out
    return out

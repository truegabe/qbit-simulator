"""Linear Combination of Unitaries (LCU).

The fundamental primitive behind block-encodings, QSVT, and Hamiltonian
simulation by Taylor series (Berry-Childs-Cleve-Kothari-Somma 2015):
implement a non-unitary linear combination

    A  =  Σ_k  α_k U_k          (α_k ≥ 0, U_k unitary)

as the "system" part of a unitary on system + ancilla register.

Construction
------------
Let `s = Σ_k α_k` and prepare on an n_anc-qubit ancilla the state

    PREP |0>  =  (1/√s) · Σ_k √(α_k) |k>.

Then build a controlled-SELECT operator that applies `U_k` to the
system register conditional on the ancilla being `|k>`. The combined
"PREP† · SELECT · PREP" unitary acting on |0>_anc ⊗ |ψ>_sys satisfies

    (⟨0|_anc ⊗ I_sys) [PREP† · SELECT · PREP] (|0>_anc ⊗ |ψ>_sys)
        =  (1/s) · A |ψ>

so a measurement of the ancilla outcome 0 *post-selects* the desired
non-unitary action A on the system, with success probability
‖A|ψ>‖² / s².

This module gives dense state-vector implementations of:

  - `lcu_unitary(alphas, U_list)`: return the full unitary on
    (n_anc + n_sys) qubits (compose PREP, SELECT, PREP†).
  - `apply_lcu(alphas, U_list, psi)`: apply LCU to system state and
    return (post-selected system state, success probability).
  - `taylor_hamiltonian_simulation(H, t, K, psi)`: simulate exp(-iHt)|ψ⟩
    by truncating the Taylor series at order K and using LCU to
    implement the polynomial in H.

The amplitude amplification version (oblivious amplitude amplification,
OAA) is implemented separately in `generalized_amplitude_amplification.py`.
"""

from __future__ import annotations

import math

import numpy as np


def _next_pow_two(k: int) -> int:
    p = 1
    while p < k:
        p *= 2
    return p


def prep_state(alphas: np.ndarray) -> np.ndarray:
    """Build PREP|0>: ancilla state with amplitudes ∝ √α_k.

    `alphas` is padded to the next power of two with zeros.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    if (alphas < 0).any():
        raise ValueError("All α_k must be non-negative.")
    d = _next_pow_two(len(alphas))
    padded = np.zeros(d)
    padded[: len(alphas)] = alphas
    s = padded.sum()
    if s <= 0:
        raise ValueError("Σ α_k must be positive.")
    state = np.sqrt(padded / s).astype(np.complex128)
    return state


def select_unitary(U_list: list[np.ndarray], n_anc: int) -> np.ndarray:
    """Block-diagonal SELECT operator: applies U_k when ancilla = |k>.

    Padded slots (k ≥ K) apply identity on the system.
    Returns a 2^(n_anc + n_sys) × 2^(n_anc + n_sys) unitary.
    """
    K = len(U_list)
    d_anc = 2 ** n_anc
    d_sys = U_list[0].shape[0]
    out = np.zeros((d_anc * d_sys, d_anc * d_sys), dtype=np.complex128)
    I_sys = np.eye(d_sys, dtype=np.complex128)
    for k in range(d_anc):
        Uk = U_list[k] if k < K else I_sys
        # Block at (k, k) of size d_sys.
        out[k * d_sys:(k + 1) * d_sys, k * d_sys:(k + 1) * d_sys] = Uk
    return out


def lcu_unitary(alphas: np.ndarray,
                 U_list: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """Build the full LCU unitary W = (PREP† ⊗ I) · SELECT · (PREP ⊗ I).

    Returns (W, n_anc, n_sys).
    """
    if len(alphas) != len(U_list):
        raise ValueError("len(alphas) must equal len(U_list).")
    K = len(alphas)
    d_anc = _next_pow_two(K)
    n_anc = int(np.log2(d_anc))
    d_sys = U_list[0].shape[0]
    n_sys = int(np.log2(d_sys))
    if 2 ** n_sys != d_sys:
        raise ValueError("U_k must be 2^n_sys × 2^n_sys.")
    # PREP: a unitary that takes |0>_anc -> prep_state. Build the smallest
    # unitary with that first column via Gram-Schmidt.
    prep = prep_state(alphas)
    PREP = _unitary_with_first_column(prep)
    PREP_full = np.kron(PREP, np.eye(d_sys, dtype=np.complex128))
    SELECT = select_unitary(U_list, n_anc=n_anc)
    PREP_dag_full = PREP_full.conj().T
    W = PREP_dag_full @ SELECT @ PREP_full
    return W, n_anc, n_sys


def _unitary_with_first_column(v: np.ndarray) -> np.ndarray:
    """Return a unitary whose first column equals v (||v|| = 1)."""
    d = len(v)
    M = np.eye(d, dtype=np.complex128)
    M[:, 0] = v
    Q, R = np.linalg.qr(M)
    # Fix the global-phase sign so Q[:, 0] points along v.
    if abs(R[0, 0]) > 1e-14:
        Q = Q * (R[0, 0] / abs(R[0, 0]))
    return Q


def apply_lcu(alphas: np.ndarray,
              U_list: list[np.ndarray],
              psi_sys: np.ndarray) -> dict:
    """Apply LCU to a system state. Returns dict with:
      - 'psi_out':    system state on the |0>_anc branch (un-normalized).
      - 'prob':       probability of successful post-selection.
      - 'psi_norm':   normalized system state after success.
      - 'A_psi':      the *exact* (Σ α_k U_k) ψ, for verification.
    """
    W, n_anc, n_sys = lcu_unitary(alphas, U_list)
    d_anc = 2 ** n_anc
    d_sys = 2 ** n_sys
    # Initial joint state: |0>_anc ⊗ ψ.
    psi0 = np.zeros(d_anc, dtype=np.complex128); psi0[0] = 1.0
    joint = np.kron(psi0, psi_sys)
    out = W @ joint
    # Project onto |0>_anc subspace.
    psi_out = out[:d_sys]      # ancilla index 0 → first d_sys entries
    prob = float(np.real(psi_out.conj() @ psi_out))
    # Exact reference: A psi = Σ α_k U_k psi.
    A_psi = np.zeros_like(psi_sys, dtype=np.complex128)
    for a, U in zip(alphas, U_list):
        A_psi = A_psi + a * (U @ psi_sys)
    psi_norm = psi_out / np.sqrt(prob) if prob > 0 else psi_out
    return {"psi_out": psi_out, "prob": prob,
            "psi_norm": psi_norm, "A_psi": A_psi}


# ----------------------------------------------------------------------------
# Hamiltonian simulation by Taylor series (BCCKS 2015)
# ----------------------------------------------------------------------------

def taylor_hamiltonian_simulation(H_terms: list[tuple[float, np.ndarray]],
                                    t: float,
                                    psi: np.ndarray,
                                    K: int = 4,
                                    n_segments: int = 1) -> dict:
    """Simulate exp(-i H t) |ψ⟩ where H = Σ_j h_j U_j with U_j unitary
    (and h_j real, possibly negative).

    Strategy: write
        exp(-i H τ) ≈ Σ_{k=0}^{K} (-iτ)^k / k!  H^k
                    = Σ_{k=0}^{K} Σ_{j1..jk} (1/k!) · (-iτ h_j1 ... h_jk)
                                · U_j1 U_j2 ... U_jk
    Each term is a unitary up to a (possibly complex) prefactor. We
    group prefactor magnitudes into α and complex phases into U.

    Splits the total time t into n_segments small steps; applies LCU
    + post-selection per segment.

    Args:
        H_terms:  list of (h_j, U_j) where each U_j is unitary 2^n × 2^n.
        t:        total simulation time.
        psi:      initial system state.
        K:        Taylor-series truncation order.
        n_segments: number of time slices.

    Returns dict with:
        'psi':         final state (normalized).
        'prob_total':  product of per-segment post-selection probs.
        'fidelity':    |⟨ψ_exact | ψ_approx⟩|.
    """
    # Build the exact Hamiltonian matrix for comparison.
    d = psi.shape[0]
    H = np.zeros((d, d), dtype=np.complex128)
    for h, U in H_terms:
        H = H + h * U
    from scipy.linalg import expm
    psi_exact = expm(-1j * H * t) @ psi
    psi_exact /= np.linalg.norm(psi_exact)

    tau = t / n_segments
    state = psi.copy().astype(np.complex128)
    prob_total = 1.0
    for _ in range(n_segments):
        alphas, U_list = _taylor_lcu_terms(H_terms, tau, K)
        res = apply_lcu(alphas, U_list, state)
        prob_total *= max(res["prob"], 1e-12)
        if res["prob"] > 0:
            state = res["psi_norm"]
    fid = float(abs(state.conj() @ psi_exact))
    return {"psi": state, "prob_total": prob_total, "fidelity": fid,
            "psi_exact": psi_exact}


def _taylor_lcu_terms(H_terms: list[tuple[float, np.ndarray]],
                       tau: float, K: int) -> tuple[np.ndarray, list]:
    """Build (alphas, U_list) implementing Σ_{k=0..K} (-iτ H)^k / k!.

    Multi-index expansion: for each k, sum over all length-k tuples
    (j_1, ..., j_k). Magnitudes go to α, signs/phases to a wrapping
    diagonal global phase absorbed into U.
    """
    d = H_terms[0][1].shape[0]
    I_sys = np.eye(d, dtype=np.complex128)
    alphas = []
    U_list = []
    for k in range(K + 1):
        prefactor_mag = (tau ** k) / math.factorial(k)
        # (-i)^k provides a phase that depends on k.
        # We absorb (-1j)^k into U as a global phase.
        phase = (-1j) ** k
        if k == 0:
            alphas.append(prefactor_mag)
            U_list.append(I_sys * phase)
            continue
        # Iterate over all multi-indices (j_1, ..., j_k).
        from itertools import product
        for idx in product(range(len(H_terms)), repeat=k):
            # Magnitude = |h_j1 ... h_jk|; sign goes into phase.
            mag = 1.0; sign = 1.0
            U_prod = I_sys
            for j in idx:
                h_j, U_j = H_terms[j]
                mag *= abs(h_j)
                if h_j < 0:
                    sign *= -1
                U_prod = U_j @ U_prod
            alphas.append(prefactor_mag * mag)
            U_list.append(sign * phase * U_prod)
    return np.array(alphas, dtype=np.float64), U_list

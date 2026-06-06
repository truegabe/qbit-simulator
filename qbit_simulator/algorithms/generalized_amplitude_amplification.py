"""Generalized amplitude amplification (Brassard-Hoyer-Mosca-Tapp 2002).

Grover's algorithm is the special case of GAA where the initial state
is the uniform superposition and the oracle marks one basis state.
The general version handles ANY state-preparation unitary A and ANY
projector P onto a "good" subspace:

    |ψ⟩ = A |0⟩ = sin(θ) |good⟩ + cos(θ) |bad⟩

with sin²(θ) = ⟨ψ|P|ψ⟩ = initial success probability. The amplitude-
amplification operator

    Q = -A · R_0 · A† · R_P,           where R_0 = I - 2|0⟩⟨0|,  R_P = I - 2P,

is a rotation by 2θ in the 2D plane spanned by |good⟩ and |bad⟩. After
k applications, the success probability is sin²((2k + 1)θ). The optimum
is k* ≈ (π/4) / θ - 1/2 (round to nearest int), giving success ≥ 1 - sin²(θ).

This also enables OBLIVIOUS AMPLITUDE AMPLIFICATION (used in
LCU-based Hamiltonian simulation): when A is the LCU unitary and the
"good" subspace is the |0⟩-ancilla branch, oblivious AA boosts the
success probability deterministically from 1/s² to 1.

This module gives a numpy implementation operating on dense unitaries
and projectors:
  - `amplitude_amplification(A, projector, n_iters, ancilla_dim, ...)`
  - `optimal_iterations(initial_prob)` — k* given sin²(θ).
  - `oblivious_amplitude_amplification(W, n_anc, n_sys, n_iters)`:
    specialized to LCU-style block encodings.
"""

from __future__ import annotations

import numpy as np


def optimal_iterations(initial_prob: float) -> int:
    """k* ≈ (π/4) / θ - 1/2, where sin²(θ) = initial_prob."""
    initial_prob = max(min(initial_prob, 1.0), 1e-12)
    theta = np.arcsin(np.sqrt(initial_prob))
    k = round((np.pi / 4) / theta - 0.5)
    return max(0, int(k))


def success_probability(initial_prob: float, k: int) -> float:
    """Probability after k amplitude-amplification iterations."""
    theta = np.arcsin(np.sqrt(initial_prob))
    return float(np.sin((2 * k + 1) * theta) ** 2)


def amplitude_amplification(A: np.ndarray,
                              projector: np.ndarray,
                              n_iters: int | None = None) -> dict:
    """Apply n_iters rounds of GAA to |ψ⟩ = A|0⟩.

    Args:
        A:         state-preparation unitary of shape (d, d).
        projector: Hermitian projector onto the "good" subspace, shape (d, d).
        n_iters:   number of Q applications. If None, use optimal.

    Returns dict with 'psi_final', 'prob_initial', 'prob_final', 'n_iters'.
    """
    d = A.shape[0]
    psi0 = np.zeros(d, dtype=np.complex128); psi0[0] = 1.0
    psi = A @ psi0
    P_good = projector
    prob_init = float(np.real(psi.conj() @ P_good @ psi))
    if n_iters is None:
        n_iters = optimal_iterations(prob_init)
    R_0 = np.eye(d, dtype=np.complex128) - 2 * np.outer(psi0, psi0.conj())
    R_P = np.eye(d, dtype=np.complex128) - 2 * P_good
    Q = -A @ R_0 @ A.conj().T @ R_P
    for _ in range(n_iters):
        psi = Q @ psi
    prob_final = float(np.real(psi.conj() @ P_good @ psi))
    return {"psi_final": psi, "prob_initial": prob_init,
            "prob_final": prob_final, "n_iters": n_iters}


def oblivious_amplitude_amplification(W: np.ndarray, n_anc: int,
                                        n_iters: int = 1) -> dict:
    """OAA on an LCU-style block-encoded unitary W on (n_anc + n_sys) qubits.

    "Good" subspace = ancilla is |0⟩^{n_anc}. When the block encoding
    implements (1/s) · V for a TRUE UNITARY V (so |A|ψ⟩|² is *deterministic*
    = 1/s² for every ψ), then OAA at k = ⌊(π/4) · s - 1/2⌋ deterministically
    boosts the success probability to ≈ 1.

    Args:
        W:        the LCU unitary on system + ancilla.
        n_anc:    number of ancilla qubits.
        n_iters:  number of OAA iterations.

    Returns dict with the OAA unitary and an example of its action on
    |0⟩^anc ⊗ |0⟩^sys.
    """
    d_total = W.shape[0]
    d_anc = 2 ** n_anc
    d_sys = d_total // d_anc
    # Projector onto ancilla = |0⟩: kron(|0⟩⟨0|, I_sys).
    P0_anc = np.zeros((d_anc, d_anc), dtype=np.complex128); P0_anc[0, 0] = 1.0
    Pi = np.kron(P0_anc, np.eye(d_sys, dtype=np.complex128))
    # R_pi = I - 2 Pi.
    R_pi = np.eye(d_total, dtype=np.complex128) - 2 * Pi
    # R_0 acts on the FULL space as I - 2|0⟩⟨0|.
    psi0 = np.zeros(d_total, dtype=np.complex128); psi0[0] = 1.0
    R_0 = np.eye(d_total, dtype=np.complex128) - 2 * np.outer(psi0, psi0.conj())
    Q = -W @ R_0 @ W.conj().T @ R_pi
    U_oaa = np.eye(d_total, dtype=np.complex128)
    for _ in range(n_iters):
        U_oaa = Q @ U_oaa
    U_oaa = U_oaa @ W
    # Example application.
    psi_out = U_oaa @ psi0
    prob = float(np.real(psi_out.conj() @ Pi @ psi_out))
    return {"U_oaa": U_oaa, "psi_out": psi_out,
             "prob_good_anc": prob, "n_iters": n_iters}


# ----------------------------------------------------------------------------
# Convenience: amplitude amplification over an oracle on n qubits.
# ----------------------------------------------------------------------------

def amp_amp_with_oracle(state_prep: np.ndarray,
                          marked: list[int],
                          n_iters: int | None = None) -> dict:
    """Wrapper: state_prep is the unitary A, oracle marks indices.

    Equivalent to Grover when state_prep is the n-qubit Hadamard.
    """
    d = state_prep.shape[0]
    P = np.zeros((d, d), dtype=np.complex128)
    for m in marked:
        P[m, m] = 1.0
    return amplitude_amplification(state_prep, P, n_iters=n_iters)

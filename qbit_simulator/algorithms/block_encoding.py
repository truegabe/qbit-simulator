"""Block encodings and Linear Combination of Unitaries (LCU).

A **block encoding** of a (sub-)stochastic matrix A is a unitary U such
that A appears as the top-left block:

    U = [[ A,   * ],
         [ * ,   * ]]

i.e. ⟨0_a| U |0_a⟩ = A, where |0_a⟩ is the ancilla state |0...0⟩.
Block encodings are the fundamental input to QSVT — given a block-
encoded A, QSVT lets us apply any polynomial of A.

**Linear Combination of Unitaries (LCU)** is a standard way to BUILD
block encodings. For

    A = sum_k α_k U_k    with    α_k > 0,  sum_k α_k = ||A||₁

we can construct a block encoding of A/||A||₁ on n + log(L) qubits
using a "PREPARE-SELECT-PREPARE†" structure:

  - PREPARE  =  |s⟩ → sum_k √(α_k/||α||₁) |k⟩  (ancilla state prep)
  - SELECT   =  |k⟩|ψ⟩ → |k⟩ U_k |ψ⟩            (controlled-Us)
  - Output: U = PREPARE† · SELECT · PREPARE encodes A/||α||₁.

Provides:

  - `block_encode_lcu(coeffs, unitaries)`: build U from a list of
    (alpha, U_k) pairs.
  - `extract_block_encoded(U, n_ancilla)`: pull A out of U.
  - `apply_block_encoded(U, n_ancilla, psi)`: apply A to a state via
    block encoding (with post-selection on the ancilla being |0⟩).
  - `lcu_hamiltonian_simulation(H_paulis, t)`: build a LCU-based
    approximation to exp(-iHt) via the truncated Taylor series method.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# LCU block encoding
# ----------------------------------------------------------------------------

def block_encode_lcu(
    coeffs: list[float], unitaries: list[np.ndarray]
) -> tuple[np.ndarray, float]:
    """Build the LCU block encoding U for A = sum_k α_k U_k.

    Returns:
        (U, alpha) where U is a (d·L) × (d·L) unitary on n_ancilla + n
        qubits and alpha = sum_k |α_k| is the normalization, so that
        the top-left d×d block of U equals A / alpha.

    Requires:
      * All α_k > 0  (sign / phase folded into the U_k if needed).
      * All U_k unitary and of the same dimension.
      * L is rounded up to a power of 2 (we pad with identity blocks).
    """
    L_raw = len(coeffs)
    if L_raw == 0:
        raise ValueError("need at least one term")
    if any(c < 0 for c in coeffs):
        # Fold negative signs into the unitaries.
        new_coeffs = []
        new_us = []
        for c, U in zip(coeffs, unitaries):
            if c < 0:
                new_coeffs.append(-c)
                new_us.append(-U)
            else:
                new_coeffs.append(c)
                new_us.append(U)
        coeffs = new_coeffs
        unitaries = new_us
    alpha = float(sum(coeffs))

    # Round L up to a power of 2 by padding with identity (α_k=0 dummy).
    L_pad = 1
    while L_pad < L_raw:
        L_pad *= 2
    if L_pad > L_raw:
        # Add identity terms with α_k = 0 (they contribute nothing).
        d = unitaries[0].shape[0]
        for _ in range(L_pad - L_raw):
            coeffs.append(0.0)
            unitaries.append(np.eye(d, dtype=np.complex128))

    L = L_pad
    d = unitaries[0].shape[0]
    n_anc = int(np.log2(L))

    # PREPARE matrix: maps |0_a⟩ → sum_k √(α_k/alpha) |k⟩.
    # We define the (L × L) prep unitary by Gram-Schmidt extension of
    # the prepared column.
    prep_col = np.zeros(L, dtype=np.complex128)
    for k in range(L):
        prep_col[k] = np.sqrt(coeffs[k] / alpha) if alpha > 0 else 0.0
    # Build orthonormal basis containing prep_col as the first column.
    Q = np.eye(L, dtype=np.complex128)
    if alpha > 0:
        Q[:, 0] = prep_col
        # Gram-Schmidt rest of the columns against prep_col.
        for c in range(1, L):
            v = Q[:, c]
            v = v - np.vdot(Q[:, 0], v) * Q[:, 0]
            n = np.linalg.norm(v)
            if n > 1e-14:
                v = v / n
            Q[:, c] = v
        # Re-orthogonalize.
        Q, _ = np.linalg.qr(Q)
        # Ensure first column is exactly prep_col.
        Q[:, 0] = prep_col

    PREPARE = Q   # acts on ancilla register

    # SELECT: block-diagonal with U_k on the k-th block.
    SELECT = np.zeros((L * d, L * d), dtype=np.complex128)
    for k in range(L):
        SELECT[k * d:(k + 1) * d, k * d:(k + 1) * d] = unitaries[k]

    # Combine: U = (PREP† ⊗ I) · SELECT · (PREP ⊗ I).
    PREP_full = np.kron(PREPARE, np.eye(d, dtype=np.complex128))
    PREP_DAG_full = PREP_full.conj().T
    U = PREP_DAG_full @ SELECT @ PREP_full
    return U, alpha


def extract_block_encoded(U: np.ndarray, n_ancilla: int) -> np.ndarray:
    """Pull A out of a block encoding U: top-left d×d block."""
    L = 2 ** n_ancilla
    total = U.shape[0]
    d = total // L
    return U[:d, :d]


def apply_block_encoded(
    U_be: np.ndarray, n_ancilla: int, psi: np.ndarray
) -> tuple[np.ndarray, float]:
    """Apply the block-encoded operator A to |psi⟩ with post-selection.

    Procedure:
        1. Embed |psi⟩ as |0_a⟩ |psi⟩ in the ancilla+system register.
        2. Apply U_be.
        3. Post-select on ancilla = |0_a⟩; the resulting (unnormalized)
           state on the system is A |psi⟩.

    Returns:
        (A·psi normalized, p_success) where p_success = ||A·psi||².
    """
    L = 2 ** n_ancilla
    d = U_be.shape[0] // L
    if psi.shape != (d,):
        raise ValueError(f"psi must be length {d}, got {psi.shape}")
    big = np.zeros(L * d, dtype=np.complex128)
    big[:d] = psi   # ancilla in |0⟩
    out = U_be @ big
    a_psi = out[:d]
    p_success = float(np.real(np.vdot(a_psi, a_psi)))
    if p_success > 1e-12:
        a_psi = a_psi / np.sqrt(p_success)
    return a_psi, p_success


# ----------------------------------------------------------------------------
# LCU Hamiltonian simulation
# ----------------------------------------------------------------------------

def _pauli_string_matrix(s: str) -> np.ndarray:
    _I = np.eye(2, dtype=np.complex128)
    _X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    _Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    table = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, table[ch])
    return M


def truncated_taylor_simulation(
    H_paulis: list[tuple[float, str]],
    t: float,
    k_max: int = 6,
) -> np.ndarray:
    """Apply exp(-iHt) via the truncated Taylor series method
    (Berry-Childs-Cleve-Kothari-Somma 2015).

        exp(-iHt) = sum_{k=0}^∞ (-iHt)^k / k!
                  ≈ sum_{k=0}^{k_max} (-iHt)^k / k!

    Each H^k is a sum of Pauli strings; we expand and apply directly.
    For demonstration we just return the matrix sum.

    Args:
        H_paulis:  list of (coef, Pauli string).
        t:         evolution time.
        k_max:     truncation order.

    Returns:
        2^n × 2^n approximation to exp(-iHt).
    """
    if not H_paulis:
        raise ValueError("need at least one Pauli term")
    n = len(H_paulis[0][1])
    d = 2 ** n
    # Build the dense H matrix.
    H = np.zeros((d, d), dtype=np.complex128)
    for coef, s in H_paulis:
        H = H + coef * _pauli_string_matrix(s)
    # Build the Taylor sum.
    U = np.eye(d, dtype=np.complex128)
    term = np.eye(d, dtype=np.complex128)
    for k in range(1, k_max + 1):
        term = term @ (-1j * t * H) / k
        U = U + term
    return U

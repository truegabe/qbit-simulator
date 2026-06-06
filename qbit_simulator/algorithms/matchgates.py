"""Matchgate circuits and free-fermion simulation.

Matchgates (Knill 2001 / Terhal-DiVincenzo 2002) are 2-qubit gates that
satisfy a specific algebraic structure. Crucially, **circuits composed
entirely of nearest-neighbor matchgates are CLASSICALLY SIMULABLE in
polynomial time**, despite acting on an exponentially-large Hilbert
space — because they map under Jordan-Wigner to **free-fermion**
dynamics, which is solvable via a 2n × 2n covariance-matrix update.

A 2-qubit matchgate has the form

    M(A, B) = [[ a, 0, 0, b ],
                [ 0, p, q, 0 ],
                [ 0, r, s, 0 ],
                [ c, 0, 0, d ]]

with det(A) = det(B), where A = [[a,b],[c,d]] and B = [[p,q],[r,s]].

This module provides:

  - `matchgate(A, B)`: construct the 4x4 matrix (with validation).
  - `random_matchgate(rng)`: sample uniformly random matchgates.
  - `apply_matchgate_to_majorana(M, gamma_in)`: update the 2n×2n
    Majorana covariance representation under a single matchgate gate.
  - `simulate_free_fermion_circuit(n_qubits, gate_list)`: efficient
    classical simulation: O(n²) per gate, O(n³) for expectation values.
  - `majorana_correlation(Gamma, i, j)`: ⟨γ_i γ_j⟩ from the covariance.

These let us simulate matchgate circuits on HUNDREDS of qubits, which
would be impossible by direct state-vector simulation.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Matchgate construction
# ----------------------------------------------------------------------------

def matchgate(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Build the 4x4 matchgate from A, B ∈ U(2) with det(A) = det(B).

    M = [[ A[0,0],     0,        0,     A[0,1] ],
         [   0,    B[0,0],   B[0,1],     0     ],
         [   0,    B[1,0],   B[1,1],     0     ],
         [ A[1,0],     0,        0,     A[1,1] ]]
    """
    if A.shape != (2, 2) or B.shape != (2, 2):
        raise ValueError("A and B must be 2x2")
    if abs(np.linalg.det(A) - np.linalg.det(B)) > 1e-9:
        raise ValueError("matchgate requires det(A) = det(B)")
    M = np.zeros((4, 4), dtype=np.complex128)
    M[0, 0] = A[0, 0]; M[0, 3] = A[0, 1]
    M[3, 0] = A[1, 0]; M[3, 3] = A[1, 1]
    M[1, 1] = B[0, 0]; M[1, 2] = B[0, 1]
    M[2, 1] = B[1, 0]; M[2, 2] = B[1, 1]
    return M


def random_matchgate(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random matchgate (Haar on the U(2)×U(2)
    submanifold with det(A) = det(B))."""
    # Sample A from U(2).
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q_A, _ = np.linalg.qr(a)
    det_A = np.linalg.det(Q_A)
    # Sample B from U(2), then rescale so det(B) = det(A).
    b = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q_B, _ = np.linalg.qr(b)
    det_B = np.linalg.det(Q_B)
    # Rotate Q_B by phase so det becomes det_A.
    phase = (det_A / det_B) ** 0.5
    Q_B = Q_B * phase
    return matchgate(Q_A, Q_B)


def is_matchgate(M: np.ndarray, tol: float = 1e-9) -> bool:
    """Check whether a 4×4 matrix has the matchgate structure."""
    if M.shape != (4, 4):
        return False
    # Check zero positions.
    for i, j in [(0, 1), (0, 2), (1, 0), (1, 3),
                  (2, 0), (2, 3), (3, 1), (3, 2)]:
        if abs(M[i, j]) > tol:
            return False
    # Check det(A) = det(B).
    A = np.array([[M[0, 0], M[0, 3]], [M[3, 0], M[3, 3]]])
    B = np.array([[M[1, 1], M[1, 2]], [M[2, 1], M[2, 2]]])
    return abs(np.linalg.det(A) - np.linalg.det(B)) < tol


# ----------------------------------------------------------------------------
# Majorana correlation matrix
# ----------------------------------------------------------------------------

def initial_correlation_matrix(n_qubits: int, occupied: list[int] | None = None
                                  ) -> np.ndarray:
    """Initial 2n × 2n Majorana correlation matrix.

    For the vacuum |00...0⟩, the only nonzero correlations are
    ⟨γ_{2k} γ_{2k+1}⟩ = i (from c†_k c_k = 0, c_k c†_k = 1).

    Args:
        n_qubits:  number of qubits = number of fermion modes.
        occupied:  list of initially-occupied modes (default: none).

    Returns:
        anti-symmetric 2n × 2n real matrix Γ with Γ_{ij} = ⟨γ_i γ_j⟩ - δ_{ij}.
    """
    if occupied is None:
        occupied = []
    Gamma = np.zeros((2 * n_qubits, 2 * n_qubits))
    for k in range(n_qubits):
        sign = -1 if k in occupied else 1
        Gamma[2 * k, 2 * k + 1] = sign
        Gamma[2 * k + 1, 2 * k] = -sign
    return Gamma


# ----------------------------------------------------------------------------
# Matchgate → SO(2n) action on Majoranas
# ----------------------------------------------------------------------------

def matchgate_to_so2n_block(M: np.ndarray) -> np.ndarray:
    """Compute the 4×4 SO(4) action on Majorana operators γ_{2k},
    γ_{2k+1}, γ_{2(k+1)}, γ_{2(k+1)+1} induced by a matchgate
    acting on qubits k, k+1.

    Formula (standard JW correspondence): the matchgate M conjugates
    Majoranas as γ_i → R_ij γ_j with R = exp(h) where h is the
    Hermitian-quadratic Hamiltonian encoded by M.

    For practical implementation, we extract R numerically by computing
    M γ_i M† for i = 0, 1, 2, 3 (the four Majoranas spanning the 2-qubit
    block).
    """
    # Build the 4 Majorana operators for a 2-qubit block.
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    # γ_0 = X ⊗ I, γ_1 = Y ⊗ I, γ_2 = Z ⊗ X, γ_3 = Z ⊗ Y.
    gammas = [
        np.kron(X, I),
        np.kron(Y, I),
        np.kron(Z, X),
        np.kron(Z, Y),
    ]
    R = np.zeros((4, 4), dtype=np.float64)
    for j in range(4):
        # M γ_j M† should be a real linear combination of γ_0..γ_3.
        transformed = M @ gammas[j] @ M.conj().T
        # Extract components: γ_i has Tr(γ_i γ_j) = 4 δ_{ij}.
        for i in range(4):
            coef = np.trace(gammas[i] @ transformed) / 4
            R[i, j] = float(np.real(coef))
    return R


# ----------------------------------------------------------------------------
# Free-fermion circuit simulator
# ----------------------------------------------------------------------------

def simulate_free_fermion_circuit(
    n_qubits: int,
    gate_list: list[tuple[np.ndarray, int]],
    initial_occupied: list[int] | None = None,
) -> np.ndarray:
    """Simulate a matchgate circuit using the 2n × 2n covariance matrix.

    Args:
        n_qubits:         number of fermion modes (= qubits in JW).
        gate_list:        list of (matchgate, qubit_k) where the gate
                          acts on (k, k+1). Each matchgate must be 4x4.
        initial_occupied: list of initially-occupied modes.

    Returns:
        the final Majorana correlation matrix Γ.
    """
    Gamma = initial_correlation_matrix(n_qubits, initial_occupied)
    for M, k in gate_list:
        if M.shape != (4, 4):
            raise ValueError("each gate must be 4x4")
        if not (0 <= k <= n_qubits - 2):
            raise ValueError(f"qubit index k={k} out of range")
        R = matchgate_to_so2n_block(M)
        # Embed R into the full 2n × 2n rotation matrix.
        R_full = np.eye(2 * n_qubits)
        idx = [2 * k, 2 * k + 1, 2 * (k + 1), 2 * (k + 1) + 1]
        for i, ii in enumerate(idx):
            for j, jj in enumerate(idx):
                R_full[ii, jj] = R[i, j]
        # Γ → R Γ Rᵀ
        Gamma = R_full @ Gamma @ R_full.T
    return Gamma


def majorana_correlation(Gamma: np.ndarray, i: int, j: int) -> float:
    """Return ⟨γ_i γ_j⟩ - δ_{ij}, the (i, j) entry of Γ."""
    return float(Gamma[i, j])


def occupation_from_gamma(Gamma: np.ndarray, mode: int) -> float:
    """Return ⟨n_k⟩ for fermionic mode `mode`.

    Relation: n_k = c†_k c_k = (1 - i γ_{2k} γ_{2k+1}) / 2.
    Using Γ_{2k, 2k+1} = ⟨γ_{2k} γ_{2k+1}⟩ - 0 (off-diagonal) — but
    actually the Majorana-correlation convention gives
    Γ_{2k,2k+1} = i · (1 - 2 n_k). So:
        n_k  =  (1 + Γ_{2k+1, 2k}) / 2.
    """
    return float((1 + Gamma[2 * mode + 1, 2 * mode]) / 2)

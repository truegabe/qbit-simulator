"""KAK decomposition of arbitrary 2-qubit unitaries.

The Cartan / KAK decomposition factors any element U in U(4) as:

    U  =  phase · (K1_a ⊗ K1_b) · exp(i (a·XX + b·YY + c·ZZ)) · (K2_a ⊗ K2_b)

where K1_a, K1_b, K2_a, K2_b ∈ SU(2) are single-qubit unitaries and
(a, b, c) are the "interaction" parameters of the Weyl chamber.

This gives the optimal 2-qubit gate count:

  - (a, b, c) = (0, 0, 0):                0 CNOTs (separable)
  - Exactly one of {a, b, c} ≠ 0:         1 CNOT
  - Two are nonzero:                      2 CNOTs
  - All three nonzero (generic):          3 CNOTs   (Vatan-Williams 2004)

Algorithm: the "magic basis" approach (Kraus-Cirac 2001 / Tucci 1999).
Conjugating U by the magic basis matrix M maps SU(2)⊗SU(2) to SO(4).
For U ∈ SU(4), the matrix S = U_M U_M^T (in the magic basis) is a
complex symmetric unitary; its eigenvectors form a REAL orthogonal
matrix O_L, and the eigenvalues encode the Weyl-chamber coordinates.
"""

from __future__ import annotations

import numpy as np


# Pauli matrices.
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# ----------------------------------------------------------------------------
# Magic basis
# ----------------------------------------------------------------------------
#
# Magic basis vectors (Hill-Wootters):
#   |m_1⟩ = (|00⟩ + |11⟩) / √2
#   |m_2⟩ = i(|00⟩ − |11⟩) / √2
#   |m_3⟩ = i(|01⟩ + |10⟩) / √2
#   |m_4⟩ = (|01⟩ − |10⟩) / √2
#
# Property: M† (A ⊗ B) M is real-orthogonal for A, B ∈ SU(2), giving the
# canonical SO(4) ≅ (SU(2) × SU(2)) / Z_2 isomorphism.

M = (1.0 / np.sqrt(2.0)) * np.array([
    [1,  1j, 0,  0],
    [0,  0,  1j, 1],
    [0,  0,  1j, -1],
    [1, -1j, 0,  0],
], dtype=np.complex128)

M_DAG = M.conj().T


# ----------------------------------------------------------------------------
# Diagonalize a complex symmetric unitary by a REAL orthogonal matrix
# ----------------------------------------------------------------------------

def _real_diagonalize_unitary_symmetric(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For complex-symmetric unitary S, find real orthogonal O such that
    O^T S O = diag(λ_1, …, λ_n) with each |λ_k| = 1.

    Method: S = A + i B with A, B real symmetric. Because S is normal
    and complex symmetric, A and B commute, so they share a common
    real-orthogonal eigenbasis. We diagonalize A; if A is degenerate,
    we rotate inside each degenerate block using B.
    """
    A = S.real
    B = S.imag
    # Diagonalize A.
    eigvals_A, O = np.linalg.eigh(A)
    # Group columns of O by equal eigenvalues of A.
    n = S.shape[0]
    tol = 1e-7
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(eigvals_A[j] - eigvals_A[i]) < tol:
            j += 1
        if j > i + 1:
            # Degenerate block; diagonalize O^T B O restricted to it.
            block = O[:, i:j].T @ B @ O[:, i:j]
            block = 0.5 * (block + block.T)   # symmetrize
            _, V_sub = np.linalg.eigh(block)
            O[:, i:j] = O[:, i:j] @ V_sub
        i = j
    # Verify
    return O


# ----------------------------------------------------------------------------
# Decompose O ∈ O(4) as (A ⊗ B) in the computational basis
# ----------------------------------------------------------------------------

def _so4_to_su2_su2(O_so4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given O ∈ SO(4), return (A, B) ∈ SU(2) × SU(2) such that
    in the computational basis, M O M† = A ⊗ B."""
    U_local = M @ O_so4.astype(np.complex128) @ M_DAG
    # U_local is supposed to be A ⊗ B. Extract via the partial-trace
    # singular value method: the leading singular vector of the
    # "reshuffled" 4×4 matrix yields (vec(A), vec(B)).
    R = U_local.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    U_svd, s, Vh = np.linalg.svd(R)
    # For R = A_vec · B_vec^T (rank 1), SVD gives R = σ u v^H with
    #   u = A_vec / |A_vec|  (up to a phase)
    #   v^H ∝ B_vec^T  (so Vh[0,:] is B_vec as a row, up to a phase)
    a_vec = U_svd[:, 0] * np.sqrt(s[0])
    b_vec = Vh[0, :] * np.sqrt(s[0])
    A = a_vec.reshape(2, 2)
    B = b_vec.reshape(2, 2)
    # Force A and B to be in SU(2): rescale so det = 1.
    detA = np.linalg.det(A)
    detB = np.linalg.det(B)
    # A ⊗ B is invariant under (α A) ⊗ (B / α); we can rescale by a phase.
    # We want both unitary; project onto nearest unitary via SVD.
    Ua, _, Vta = np.linalg.svd(A)
    A_uni = Ua @ Vta
    Ub, _, Vtb = np.linalg.svd(B)
    B_uni = Ub @ Vtb
    # Force det(A) = det(B) = 1 (SU(2)) by absorbing the phase symmetrically.
    phase_A = np.linalg.det(A_uni) ** 0.5
    phase_B = np.linalg.det(B_uni) ** 0.5
    A_uni = A_uni / phase_A
    B_uni = B_uni / phase_B
    return A_uni, B_uni


# ----------------------------------------------------------------------------
# Main KAK
# ----------------------------------------------------------------------------

def kak_decompose(U: np.ndarray) -> dict:
    """Decompose U ∈ U(4) into KAK form.

    Returns dict with keys:
        K1_a, K1_b:  single-qubit unitaries applied AFTER the interaction
        K2_a, K2_b:  single-qubit unitaries applied BEFORE the interaction
        a, b, c:     interaction coordinates (real)
        phase:       complex global phase

    Such that:
        U = phase · (K1_a ⊗ K1_b) · exp(i(a·XX + b·YY + c·ZZ)) · (K2_a ⊗ K2_b)
    """
    if U.shape != (4, 4):
        raise ValueError(f"U must be 4x4, got {U.shape}")
    if not np.allclose(U @ U.conj().T, np.eye(4), atol=1e-8):
        raise ValueError("U must be unitary")

    # Strip global phase: U_su4 = U / det(U)^(1/4) ∈ SU(4).
    det_U = np.linalg.det(U)
    phase = det_U ** (1.0 / 4.0)
    U_su4 = U / phase

    # Move to magic basis.
    U_M = M_DAG @ U_su4 @ M

    # S = U_M U_M^T is complex symmetric and unitary.
    S = U_M @ U_M.T
    # Symmetrize numerically.
    S = 0.5 * (S + S.T)

    # Real-orthogonal eigendecomp.
    O_L = _real_diagonalize_unitary_symmetric(S)
    if np.linalg.det(O_L) < 0:
        O_L[:, 0] *= -1

    # Eigenvalues lambda_k = e^{2 i θ_k}. We need D = diag(e^{i θ_k}).
    # Branch choice: there's a ± sign per eigenvalue, and we must pick
    # signs so that O_R^T := D^{-1} O_L^T U_M comes out real.
    lambdas = np.diag(O_L.T @ S @ O_L)
    thetas = 0.5 * np.angle(lambdas)
    # Compute candidate U'' = D^{-1} O_L^T U_M.
    sqrt_lambdas = np.exp(1j * thetas)
    M_candidate = np.diag(1.0 / sqrt_lambdas) @ O_L.T @ U_M
    # Each row should be either purely real or purely imaginary.
    # Flip a sign in sqrt_lambdas[k] iff row k has more imaginary than real.
    for k in range(4):
        row = M_candidate[k, :]
        if np.linalg.norm(row.imag) > np.linalg.norm(row.real):
            sqrt_lambdas[k] *= -1
            thetas[k] += np.pi
            M_candidate[k, :] *= np.exp(-1j * np.pi)   # equivalently, *= -1
    D = np.diag(sqrt_lambdas)

    # O_R^T should now be real.
    O_R = M_candidate.real.T
    # Enforce orthogonality numerically (SVD-based projection).
    Uo, _, Vto = np.linalg.svd(O_R)
    O_R = Uo @ Vto
    if np.linalg.det(O_R) < 0:
        O_R[:, 0] *= -1
        thetas[0] += np.pi
        sqrt_lambdas[0] *= -1
        D = np.diag(sqrt_lambdas)

    # Recover K1, K2 from O_L, O_R via SO(4) ≅ SU(2)⊗SU(2).
    K1_a, K1_b = _so4_to_su2_su2(O_L)
    K2_a, K2_b = _so4_to_su2_su2(O_R.T)

    # Extract (a, b, c) from the middle matrix directly via Pauli inner
    # products. The middle in computational basis is:
    #   middle = M · diag(e^{i θ_k}) · M† = exp(i(a XX + b YY + c ZZ))
    # We compute H = -i log(middle), then a = tr(H·XX)/4, b = tr(H·YY)/4,
    # c = tr(H·ZZ)/4. This is independent of the eigenvector ordering.
    middle = M @ D @ M_DAG
    from scipy.linalg import logm
    H_log = -1j * logm(middle)
    # H_log should be a real-linear combination of XX, YY, ZZ.
    XX = np.kron(_X, _X)
    YY = np.kron(_Y, _Y)
    ZZ = np.kron(_Z, _Z)
    a = float(np.trace(H_log @ XX).real / 4)
    b = float(np.trace(H_log @ YY).real / 4)
    c = float(np.trace(H_log @ ZZ).real / 4)

    return {
        "K1_a":   K1_a,
        "K1_b":   K1_b,
        "K2_a":   K2_a,
        "K2_b":   K2_b,
        "a":      float(a),
        "b":      float(b),
        "c":      float(c),
        "phase":  complex(phase),
        "thetas": thetas,
    }


def _thetas_to_abc(thetas: np.ndarray) -> tuple[float, float, float]:
    """Solve the 4-equation linear system for (a, b, c)."""
    A_mat = np.array([
        [1,  1, -1],
        [1, -1,  1],
        [-1, 1,  1],
        [-1, -1, -1],
    ], dtype=float)
    # Unwrap thetas modulo π.
    thetas_unwrapped = np.array(thetas, dtype=float)
    abc, *_ = np.linalg.lstsq(A_mat, thetas_unwrapped, rcond=None)
    return float(abc[0]), float(abc[1]), float(abc[2])


# ----------------------------------------------------------------------------
# Reconstruction & analysis
# ----------------------------------------------------------------------------

def kak_to_unitary(kak: dict) -> np.ndarray:
    """Rebuild U from its KAK decomposition (for verification)."""
    K1 = np.kron(kak["K1_a"], kak["K1_b"])
    K2 = np.kron(kak["K2_a"], kak["K2_b"])
    a, b, c = kak["a"], kak["b"], kak["c"]
    XX = np.kron(_X, _X)
    YY = np.kron(_Y, _Y)
    ZZ = np.kron(_Z, _Z)
    middle = _expm_iH(a * XX + b * YY + c * ZZ)
    return kak["phase"] * (K1 @ middle @ K2)


def _expm_iH(H: np.ndarray) -> np.ndarray:
    """exp(i H) via eigendecomposition for Hermitian H."""
    eigs, V = np.linalg.eigh(H)
    return V @ np.diag(np.exp(1j * eigs)) @ V.conj().T


def cnot_count(U: np.ndarray, tol: float = 1e-6) -> int:
    """Minimum number of CNOTs needed to implement U.

    Determined by the KAK Weyl parameters. CNOT corresponds to
    (a, b, c) = (π/4, 0, 0); each nonzero parameter (mod π/2) adds a
    CNOT to the synthesis cost.
    """
    kak = kak_decompose(U)
    a, b, c = kak["a"], kak["b"], kak["c"]

    def is_zero(x):
        x = x % (np.pi / 2)
        return min(abs(x), abs(x - np.pi / 2)) < tol

    return sum(0 if is_zero(p) else 1 for p in (a, b, c))


def reconstruction_error(U: np.ndarray) -> float:
    """How close is kak_to_unitary(kak_decompose(U)) to U?

    Returns the operator distance |1 - |⟨U, U_reconstructed⟩| / 4|.
    """
    kak = kak_decompose(U)
    U_rec = kak_to_unitary(kak)
    overlap = np.trace(U.conj().T @ U_rec) / 4
    return float(abs(1 - abs(overlap)))

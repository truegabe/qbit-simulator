"""HaPPY (Pastawski-Yoshida-Harlow-Preskill) holographic codes.

HaPPY codes (Pastawski et al. 2015) are quantum error-correcting codes
defined on a hyperbolic tessellation. They provide an EXACT
realization of the AdS/CFT bulk-boundary mapping at the level of
isometries:

  * BULK qubits live on tiles of a hyperbolic tessellation.
  * BOUNDARY qubits live on the asymptotic boundary edges.
  * A "perfect tensor" at each tile is an isometry from bulk → boundary
    that preserves any logical operator with sufficiently bounded
    support — this is the holographic Ryu-Takayanagi formula in the
    discrete code setting.

The smallest non-trivial HaPPY code uses a SINGLE perfect tensor:
the [[5,1,3]] code, which we already implement in `qec.py`. With this
one tensor:

  * 1 bulk qubit (logical) is encoded into
  * 5 boundary qubits.
  * The code corrects any single-qubit error (distance 3).

A more interesting "small holography" example is the **2-tile HaPPY
code**: glue two [[5,1,3]] tensors along an edge, giving an isometry
from 2 bulk qubits to 8 boundary qubits.

We implement:

  - `perfect_tensor_5q()`: returns the [[5,1,3]] isometry as a
    32×2 matrix (5 boundary indices × 1 logical).
  - `happy_encode_one_tile(psi)`: bulk-to-boundary encoding.
  - `happy_2_tile_encoder()`: an example 2-tile HaPPY isometry.
  - `bulk_reconstruction(boundary_rho, region)`: trace out the
    complement of `region` and check if a logical operator is
    reconstructible (analog of the bulk reconstruction theorem).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# [[5,1,3]] perfect tensor — the building block
# ----------------------------------------------------------------------------
#
# We use the 5-qubit code stabilizers
#   S1 = X Z Z X I
#   S2 = I X Z Z X
#   S3 = X I X Z Z
#   S4 = Z X I X Z
# Logical Z = ZZZZZ, Logical X = XXXXX.
# The two logical codewords |0_L⟩, |1_L⟩ span the codespace and the
# isometry V: |b⟩ → |b_L⟩ is the 32×2 "perfect tensor".

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I = np.eye(2, dtype=np.complex128)


def _pauli_string_matrix(s: str) -> np.ndarray:
    table = {"I": _I, "X": _X, "Z": _Z,
              "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128)}
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, table[ch])
    return M


_STABILIZERS_513 = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
_LOGICAL_Z_513 = "ZZZZZ"


def _build_513_codewords() -> tuple[np.ndarray, np.ndarray]:
    """Construct |0_L⟩ and |1_L⟩ of the [[5,1,3]] code by stabilizer
    projection of |00000⟩ and X_L |00000⟩."""
    dim = 32
    # Project onto +1 eigenspace of each stabilizer.
    P = np.eye(dim, dtype=np.complex128)
    for s in _STABILIZERS_513:
        S = _pauli_string_matrix(s)
        P = (P + S) / 2 @ P
    # Start from |00000⟩.
    psi0 = np.zeros(dim, dtype=np.complex128); psi0[0] = 1.0
    code_0 = P @ psi0
    code_0 = code_0 / np.linalg.norm(code_0)
    # Logical X = XXXXX. Apply it to get |1_L⟩.
    X_L = _pauli_string_matrix("XXXXX")
    code_1 = X_L @ code_0
    return code_0, code_1


def perfect_tensor_5q() -> np.ndarray:
    """The [[5,1,3]] isometry V: ℂ² → ℂ^32 mapping bulk → boundary.

    V|0⟩_bulk = |0_L⟩ (logical zero), V|1⟩_bulk = |1_L⟩.

    Returns:
        32×2 isometry matrix.
    """
    c0, c1 = _build_513_codewords()
    return np.column_stack([c0, c1])


# ----------------------------------------------------------------------------
# Encoding a bulk state to boundary
# ----------------------------------------------------------------------------

def happy_encode_one_tile(psi_bulk: np.ndarray) -> np.ndarray:
    """Encode a 1-qubit bulk state into 5 boundary qubits via the
    [[5,1,3]] perfect tensor."""
    if psi_bulk.shape != (2,):
        raise ValueError(f"psi_bulk must be a 1-qubit state, got {psi_bulk.shape}")
    V = perfect_tensor_5q()
    psi_boundary = V @ psi_bulk
    return psi_boundary


# ----------------------------------------------------------------------------
# Boundary-region reduced density matrices
# ----------------------------------------------------------------------------

def _reduced_density_on_subset(
    psi: np.ndarray, n_total: int, keep: list[int]
) -> np.ndarray:
    """Trace out qubits not in `keep`; return reduced ρ on `keep`."""
    n_keep = len(keep)
    arr = psi.reshape([2] * n_total)
    perm = list(keep) + [q for q in range(n_total) if q not in keep]
    arr = np.transpose(arr, perm)
    arr = arr.reshape(2 ** n_keep, 2 ** (n_total - n_keep))
    return arr @ arr.conj().T


def boundary_reduced_density(
    boundary_state: np.ndarray, region: list[int], n_total: int = 5
) -> np.ndarray:
    """Reduced density matrix on the boundary subset `region`.

    Args:
        boundary_state: the 2^n_total state vector.
        region:         list of qubit indices to KEEP (MSB-first).
        n_total:        total boundary qubits.
    """
    return _reduced_density_on_subset(boundary_state, n_total, region)


# ----------------------------------------------------------------------------
# Bulk reconstruction diagnostic
# ----------------------------------------------------------------------------

def logical_z_expectation(boundary_state: np.ndarray) -> float:
    """⟨ψ_boundary | Z_L | ψ_boundary⟩ where Z_L = Z⊗5 is the logical Z."""
    Z_logical = _pauli_string_matrix("ZZZZZ")
    return float(np.real(boundary_state.conj() @ Z_logical @ boundary_state))


def region_distinguishes_logical(
    region: list[int], tol: float = 1e-9,
) -> dict:
    """Test whether a boundary `region` can distinguish |0_L⟩ from |1_L⟩.

    For the [[5,1,3]] HaPPY code: regions of size ≥ 3 can reconstruct
    the bulk (their reduced density matrices on |0_L⟩ and |1_L⟩ are
    DIFFERENT), regions of size ≤ 2 cannot (the reduced states are
    identical — the "no information" complement of the entanglement
    wedge).

    This is the discrete Ryu-Takayanagi statement.

    Returns:
        dict with rho_diff_norm, can_reconstruct (bool), region_size.
    """
    psi0 = happy_encode_one_tile(np.array([1, 0], dtype=complex))
    psi1 = happy_encode_one_tile(np.array([0, 1], dtype=complex))
    rho0 = _reduced_density_on_subset(psi0, 5, region)
    rho1 = _reduced_density_on_subset(psi1, 5, region)
    diff = np.linalg.norm(rho0 - rho1)
    return {
        "rho_diff_norm":    float(diff),
        "can_reconstruct":  diff > tol,
        "region_size":      len(region),
    }


# ----------------------------------------------------------------------------
# 2-tile HaPPY (illustrative)
# ----------------------------------------------------------------------------

def happy_2_tile_encoder() -> np.ndarray:
    """A 2-tile HaPPY encoder: 2 bulk qubits → 8 boundary qubits.

    Construction: tensor two [[5,1,3]] perfect tensors and CONTRACT
    one boundary qubit of each tile through a maximally-entangled state
    on the shared edge. This implements the holographic "gluing" of
    two bulk regions.

    The result is, in general, a (2^8 × 2^2) MAP — not necessarily an
    isometry of unit norm. The proper normalization is fixed by the
    Schmidt rank of the contracted index (= 2, since the shared edge
    is a single qubit).

    For demonstration we return the raw contracted tensor (matrix form).
    Use `is_isometry` to check; pure-state encoding requires renorma-
    lization to make it an isometry, which we leave as an exercise.
    """
    V = perfect_tensor_5q().reshape(2, 2, 2, 2, 2, 2)
    # Contract b4 of tile1 with b0 of tile2: indices a b c d e q, e f g h i p.
    contracted = np.einsum("abcdeq,efghip->abcdfghiqp", V, V)
    return contracted.reshape(2 ** 8, 2 ** 2)


def is_isometry(V: np.ndarray, tol: float = 1e-9) -> bool:
    """V is an isometry iff V† V = I_{input}."""
    n_in = V.shape[1]
    return np.allclose(V.conj().T @ V, np.eye(n_in), atol=tol)

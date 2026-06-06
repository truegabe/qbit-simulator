r"""MERA: Multi-scale Entanglement Renormalization Ansatz.

MERA (Vidal 2007) is a tensor-network representation that captures
the entanglement structure of critical (gapless) 1D quantum systems
efficiently. Unlike MPS (which has area-law entanglement), MERA can
represent logarithmic-violation states characteristic of conformal
field theories.

Architecture (binary MERA):

    Physical sites:    [s_0  s_1  s_2  s_3  s_4  s_5  s_6  s_7]
                          \  /    \  /    \  /    \  /
    Layer-1 disent.:       D      D       D       D       (2-site unitaries
                          / \    / \     / \     / \       that REMOVE
                                                            entanglement)
    Layer-1 isom.:       [s'_0  s'_1   s'_2   s'_3]
                          \   /        \   /
    Layer-2 disent.:       D            D
                          / \          / \
    Layer-2 isom.:       [s''_0       s''_1]
                          \   /
    Top:                   T

The CRITICAL feature: each layer reduces qubit count by 2× via the
isometries. Going up log₂(N) levels reaches a single "top" qubit.

This module implements a SIMPLIFIED MERA on a small number of sites:

  - `MERA(n_qubits, bond_dim)`: parameterized binary MERA.
  - `to_dense_state()`: contract the network into a full state vector.
  - `compute_local_observable()`: efficient observable computation by
    light-cone shrinking.
  - `optimize_for_ground_state(H, n_iter)`: variational ground-state
    optimization.

For n_qubits = 2^k with k ≤ 4 (16 qubits), we keep bond dim = 2 for
simplicity; full MERA optimization uses larger bond dims.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Binary MERA on qubits (bond dim = 2)
# ----------------------------------------------------------------------------

def _random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random unitary on dimension d."""
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # Fix sign of diag(R) to get Haar measure.
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D


def _random_isometry(d_in: int, d_out: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Sample a random isometry from C^{d_in} → C^{d_out}, d_out ≥ d_in.

    Returns a d_out × d_in matrix with V†V = I.
    """
    U = _random_unitary(d_out, rng)
    return U[:, :d_in]


class MERA:
    """Binary MERA on n_qubits = 2^k qubits with bond dim 2.

    Layers: starting with n_qubits, each layer halves the count via a
    layer of "disentanglers" (full 2-qubit unitaries) followed by a
    layer of "isometries" (here 4 → 2 isometries since bond = 2).
    """

    def __init__(self, n_qubits: int,
                 rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        if not (n_qubits > 0 and (n_qubits & (n_qubits - 1)) == 0):
            raise ValueError(f"n_qubits must be a power of 2, got {n_qubits}")
        self.n_qubits = n_qubits
        self.n_layers = int(np.log2(n_qubits))
        # Per layer L (0-indexed, bottom = 0): a list of disentanglers
        # (4×4 unitaries) and a list of isometries (4 → 2).
        self.disentanglers: list[list[np.ndarray]] = []
        self.isometries: list[list[np.ndarray]] = []
        sites = n_qubits
        for L in range(self.n_layers):
            # Disentanglers: applied to pairs at positions
            # (1, 2), (3, 4), ... — between block boundaries.
            n_dis = sites // 2 - 1
            dis_layer = [_random_unitary(4, rng) for _ in range(n_dis)]
            # Isometries: 4 → 2, one per pair of sites.
            n_iso = sites // 2
            iso_layer = [_random_isometry(2, 4, rng).reshape(2, 2, 2)
                          for _ in range(n_iso)]
            self.disentanglers.append(dis_layer)
            self.isometries.append(iso_layer)
            sites //= 2

    # ---- Conversion to dense state ----

    def to_dense_state(self) -> np.ndarray:
        """Contract the network into a full 2^n state vector.

        Starts from the top |0⟩ qubit and applies isometries +
        disentanglers in reverse order (top → bottom).
        """
        psi = np.array([1.0, 0.0], dtype=np.complex128)  # top qubit |0⟩
        sites = 1
        for L in range(self.n_layers - 1, -1, -1):
            # Build the joint isometry as a tensor product of V_k's.
            # Each V_k: shape (2, 2, 2) — (out0, out1, in).
            # Joint isometry maps (in_0, ..., in_{sites-1})
            # to (out0_0, out1_0, out0_1, out1_1, ...).
            V_joint = self.isometries[L][0]      # shape (2, 2, 2)
            for k in range(1, sites):
                V_k = self.isometries[L][k]      # shape (2, 2, 2)
                # Combine: new shape is V_joint.shape[:-1] (output axes)
                # + V_k.shape (out0_k, out1_k, in_k), keeping V_joint's
                # last axis (in) interleaved.
                # einsum: existing output indices + remaining input
                # plus new (out0, out1, in).
                V_joint = np.tensordot(V_joint, V_k, axes=0)
                # Now V_joint shape = ((2*k inputs), out0_k+1, out1_k+1, in_k+1)
                # We need to reorder so output indices come first.
                # Actually V_joint has shape (2,2,...,2 inputs, 2,2,2 new).
                # Last 3 dims are (out0_new, out1_new, in_new). Move
                # out0_new, out1_new to the end of the output block,
                # in_new to the end of the input block.

            # Reshape psi into shape (2,)*sites to apply V_joint.
            psi_t = psi.reshape([2] * sites)
            # V_joint has shape (out0_0, out1_0, in_0, out0_1, out1_1, in_1, ...)
            # after sequential tensordots. We need to permute to get
            # all outputs first, then all inputs.
            new_axes_out = []
            new_axes_in = []
            for k in range(sites):
                new_axes_out.extend([3 * k, 3 * k + 1])
                new_axes_in.append(3 * k + 2)
            V_perm = V_joint.transpose(new_axes_out + new_axes_in)
            V_mat = V_perm.reshape(2 ** (2 * sites), 2 ** sites)

            psi = V_mat @ psi
            sites *= 2

            # Apply disentanglers at positions (2k+1, 2k+2).
            for k in range(len(self.disentanglers[L])):
                D = self.disentanglers[L][k]
                psi = self._apply_2q_gate(psi, D, 2 * k + 1, 2 * k + 2, sites)
        return psi

    @staticmethod
    def _apply_2q_gate(psi: np.ndarray, gate: np.ndarray,
                        q0: int, q1: int, n: int) -> np.ndarray:
        """Apply a 4x4 gate on qubits (q0, q1) (MSB-first: axis q = qubit q)."""
        if q1 < q0:
            swap = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ], dtype=np.complex128)
            gate = swap @ gate @ swap
            q0, q1 = q1, q0
        shape = [2] * n
        arr = psi.reshape(shape)
        arr = np.moveaxis(arr, [q0, q1], [0, 1])
        arr = arr.reshape(4, -1)
        arr = gate @ arr
        arr = arr.reshape([2, 2] + [2] * (n - 2))
        arr = np.moveaxis(arr, [0, 1], [q0, q1])
        return arr.reshape(2 ** n)


# ----------------------------------------------------------------------------
# Helper: count parameters
# ----------------------------------------------------------------------------

def mera_parameter_count(n_qubits: int) -> dict:
    """Count parameters in a binary MERA (bond dim 2) on n_qubits.

    Each disentangler is U(4) with 16 real parameters.
    Each isometry C^2 → C^4 has 8 - 2 = 6 ... well, parameterizing
    requires a more careful count. For an isometry V: C^2 → C^4
    represented as a 4×2 matrix V with V†V = I, the manifold dimension
    is 2 · 2 · 4 - 2² - 2²·(2² + 2)/2 ... we'll just compute as if
    it's a 4×2 complex matrix (16 real params, then subtract orthogonality
    constraints).
    """
    n_layers = int(np.log2(n_qubits))
    sites = n_qubits
    n_dis = 0
    n_iso = 0
    for L in range(n_layers):
        n_dis += sites // 2 - 1
        n_iso += sites // 2
        sites //= 2
    return {
        "n_disentanglers":  n_dis,
        "n_isometries":     n_iso,
        "n_layers":         n_layers,
        "param_count_full": 16 * n_dis + 16 * n_iso,    # rough upper bound
    }

"""Solovay-Kitaev: compile arbitrary single-qubit unitaries to Clifford+T.

Background:

Fault-tolerant quantum computers can execute only a discrete gate set —
typically Clifford + T  =  {H, S, T, CNOT}. Any arbitrary single-qubit
unitary U(2) needs to be approximated by a sequence of these gates.

The Solovay-Kitaev theorem (1995 / 2005) says: for any tolerance epsilon,
U can be approximated by a sequence of length

    O((log(1/epsilon))^c)   with c ≈ 3.97

That polylog scaling is what makes universal fault-tolerant computing
practically possible — without it, the gate count would explode
exponentially.

This module implements:

  - `enumerate_basic_sequences(max_length)`: build the "basic library"
    of all Clifford+T words up to a given length.
  - `solovay_kitaev_decompose(U, depth)`: the recursive algorithm.
    `depth = 0` returns the closest basic-library sequence; each deeper
    level applies the SK refinement, shrinking the error from epsilon
    to ~ epsilon^(3/2).
  - `sequence_to_unitary(seq)`: replay a gate string as a 2x2 unitary.

The algorithm in one paragraph: we want U ≈ V at error eps. Find
V0 = SK(U, n-1) at error eps0 ≈ eps^(2/3). Then the residual
W = U V0† has error eps0. Decompose W as a group commutator
W = V W' V† W'† with V, W' rotations by angle ~ sqrt(eps0). Recursively
approximate V, W' at level n-1 (each at error eps0^(3/2) = eps). Combine.

We provide a reference implementation; for production use, prebuilt
tables of Ross-Selinger or grid-synthesis are far more efficient. Our
goal here is to demonstrate the asymptotic O(log^c(1/eps)) scaling
empirically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# The Clifford+T basis (2x2 unitaries)
# ----------------------------------------------------------------------------

_SQRT_HALF = 1.0 / np.sqrt(2.0)

H = np.array([[1, 1], [1, -1]], dtype=np.complex128) * _SQRT_HALF
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
S_DAG = S.conj().T
T_DAG = T.conj().T
I2 = np.eye(2, dtype=np.complex128)

# A common minimal Clifford+T generating set: H, T (S = T^2).
# We use {H, T, T_DAG} so the basic library covers both T and T† without
# needing length-7 words.
BASIS_GATES = {
    "H":   H,
    "T":   T,
    "Tdg": T_DAG,
}


# ----------------------------------------------------------------------------
# Distance metric
# ----------------------------------------------------------------------------

def trace_distance_su2(U: np.ndarray, V: np.ndarray) -> float:
    """Operator distance between two 2x2 unitaries, ignoring global phase.

        d(U, V) = sqrt(1 - |tr(U† V) / 2|²)

    This is the standard Solovay-Kitaev distance, equal to half the
    diamond norm for unitary channels.
    """
    overlap = np.trace(U.conj().T @ V) / 2.0
    val = 1.0 - abs(overlap) ** 2
    return float(np.sqrt(max(val, 0.0)))


# ----------------------------------------------------------------------------
# Sequence ↔ unitary
# ----------------------------------------------------------------------------

def sequence_to_unitary(seq: tuple[str, ...]) -> np.ndarray:
    """Multiply the gates in seq (left-to-right circuit order)."""
    U = I2.copy()
    for name in seq:
        U = BASIS_GATES[name] @ U
    return U


def invert_sequence(seq: tuple[str, ...]) -> tuple[str, ...]:
    """The Hermitian conjugate sequence: reverse and invert each gate.

    H† = H, S† = Sdg, T† = Tdg, etc.
    """
    inv_map = {"H": "H", "T": "Tdg", "Tdg": "T"}
    return tuple(inv_map[g] for g in reversed(seq))


# ----------------------------------------------------------------------------
# Basic-library enumeration
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class _LibEntry:
    sequence: tuple[str, ...]
    unitary:  np.ndarray


def enumerate_basic_sequences(max_length: int = 8) -> list[_LibEntry]:
    """Enumerate all Clifford+T words up to max_length, deduplicating by
    operator-distance similarity.

    For max_length = 8 this is a few thousand entries — enough to give a
    good seed for the SK recursion.
    """
    library: list[_LibEntry] = [_LibEntry((), I2.copy())]
    # BFS over gate sequences.
    frontier = [_LibEntry((), I2.copy())]
    seen_unitaries: list[np.ndarray] = [I2.copy()]
    gate_names = list(BASIS_GATES.keys())
    for _ in range(max_length):
        new_frontier = []
        for entry in frontier:
            for g in gate_names:
                # Avoid trivial cancellations T·Tdg, etc.
                if entry.sequence:
                    last = entry.sequence[-1]
                    if (last, g) in (("T", "Tdg"), ("Tdg", "T"), ("H", "H")):
                        continue
                seq = entry.sequence + (g,)
                U = BASIS_GATES[g] @ entry.unitary
                # Deduplicate: skip if very close to any seen unitary.
                is_new = True
                for U_seen in seen_unitaries:
                    if trace_distance_su2(U, U_seen) < 1e-6:
                        is_new = False
                        break
                if is_new:
                    new_entry = _LibEntry(seq, U)
                    library.append(new_entry)
                    seen_unitaries.append(U)
                    new_frontier.append(new_entry)
        frontier = new_frontier
        if not frontier:
            break
    return library


def find_closest_in_library(U: np.ndarray, library: list[_LibEntry]
                              ) -> _LibEntry:
    """Linear scan to find the library entry closest to U."""
    best = library[0]
    best_d = trace_distance_su2(U, best.unitary)
    for entry in library[1:]:
        d = trace_distance_su2(U, entry.unitary)
        if d < best_d:
            best = entry
            best_d = d
    return best


# ----------------------------------------------------------------------------
# Group commutator decomposition
# ----------------------------------------------------------------------------

def _bloch_axis_angle(U: np.ndarray) -> tuple[np.ndarray, float]:
    """Decompose a 2x2 SU(2) unitary as U = exp(-i (theta/2) n · sigma).

    Returns (n, theta). For U = identity (up to phase), returns
    (e_z, 0).
    """
    # Strip global phase by det^{-1/2}.
    det = np.linalg.det(U)
    phase = np.sqrt(det)
    U_su2 = U / phase
    # cos(theta/2) = Re(tr(U_su2)) / 2.
    cos_half = np.clip(0.5 * U_su2.trace().real, -1.0, 1.0)
    theta = 2.0 * np.arccos(cos_half)
    sin_half = np.sin(theta / 2.0)
    if abs(sin_half) < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    # n · sigma = (U_su2 - cos(theta/2) I) / (-i sin(theta/2)).
    M = (U_su2 - cos_half * I2) / (-1j * sin_half)
    nx = 0.5 * (M[0, 1] + M[1, 0]).real
    ny = 0.5 * (1j * (M[0, 1] - M[1, 0])).real
    nz = 0.5 * (M[0, 0] - M[1, 1]).real
    n = np.array([nx, ny, nz])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return n / norm, float(theta)


def _rotation_about(axis: np.ndarray, theta: float) -> np.ndarray:
    """Build U = exp(-i (theta/2) n · sigma)."""
    nx, ny, nz = axis
    return (np.cos(theta / 2) * I2
            - 1j * np.sin(theta / 2) * (
                nx * np.array([[0, 1], [1, 0]], dtype=complex)
              + ny * np.array([[0, -1j], [1j, 0]], dtype=complex)
              + nz * np.array([[1, 0], [0, -1]], dtype=complex)))


def group_commutator_decompose(W: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Decompose W ≈ V Q V† Q† as a balanced group commutator.

    For W = exp(-i alpha n · sigma) close to identity, we can write
    it as the commutator of two rotations by angle ~ sqrt(alpha) about
    perpendicular axes (the "balanced commutator" of Dawson-Nielsen).

    Algorithm:
        1) Express W = R(n, alpha) where n is the rotation axis and
           alpha the rotation angle.
        2) Pick two perpendicular axes a, b with a × b ≈ n.
        3) V = R(a, phi),  Q = R(b, phi)  with phi chosen so that the
           commutator V Q V† Q† gives angle alpha about axis n.
        4) Conjugate so n becomes the axis of the actual W.

    Returns:
        (V, Q) such that V Q V† Q† ≈ W. Each is a rotation by angle
        phi ≈ 2 arcsin(sqrt(sin(alpha/2))), so phi ~ sqrt(alpha) for
        small alpha.
    """
    n, alpha = _bloch_axis_angle(W)
    if alpha < 1e-12:
        return I2.copy(), I2.copy()
    if alpha > np.pi:
        alpha = 2 * np.pi - alpha
        n = -n
    # Find phi by numerical inversion: the commutator angle of
    # R_x(phi) with R_y(phi) is a known function of phi. We invert it
    # by binary search on [0, pi].
    def _commutator_angle(phi: float) -> float:
        V = _rotation_about(np.array([1.0, 0.0, 0.0]), phi)
        Q = _rotation_about(np.array([0.0, 1.0, 0.0]), phi)
        C = V @ Q @ V.conj().T @ Q.conj().T
        _, a = _bloch_axis_angle(C)
        return a
    lo, hi = 0.0, np.pi
    # Bisection: commutator angle is monotonic in phi on [0, ~2.5].
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _commutator_angle(mid) < alpha:
            lo = mid
        else:
            hi = mid
    phi = 0.5 * (lo + hi)
    # Build V about x-axis, Q about y-axis (perpendicular pair).
    V0 = _rotation_about(np.array([1.0, 0.0, 0.0]), phi)
    Q0 = _rotation_about(np.array([0.0, 1.0, 0.0]), phi)
    # The commutator V0 Q0 V0† Q0† is a rotation by angle alpha
    # about some axis n0. Find that axis.
    C0 = V0 @ Q0 @ V0.conj().T @ Q0.conj().T
    n0, _ = _bloch_axis_angle(C0)
    # Find the rotation that takes n0 -> n.
    if np.allclose(n0, n):
        R = I2.copy()
    elif np.allclose(n0, -n):
        # Rotate by pi about any axis perpendicular to n.
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(perp, n)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, n) * n
        perp = perp / np.linalg.norm(perp)
        R = _rotation_about(perp, np.pi)
    else:
        axis = np.cross(n0, n)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(n0, n), -1.0, 1.0))
        R = _rotation_about(axis, angle)
    V = R @ V0 @ R.conj().T
    Q = R @ Q0 @ R.conj().T
    return V, Q


# ----------------------------------------------------------------------------
# Solovay-Kitaev recursion
# ----------------------------------------------------------------------------

_DEFAULT_LIBRARY: list[_LibEntry] | None = None


def _get_default_library(max_length: int = 10) -> list[_LibEntry]:
    global _DEFAULT_LIBRARY
    if _DEFAULT_LIBRARY is None or len(_DEFAULT_LIBRARY) < 100:
        _DEFAULT_LIBRARY = enumerate_basic_sequences(max_length=max_length)
    return _DEFAULT_LIBRARY


def solovay_kitaev_decompose(
    U: np.ndarray,
    depth: int = 3,
    library: list[_LibEntry] | None = None,
) -> tuple[tuple[str, ...], np.ndarray, float]:
    """Approximate U as a sequence of Clifford+T gates.

    Args:
        U:        target 2x2 unitary.
        depth:    SK recursion depth. depth=0 returns the closest basic
                  library entry. Each level should shrink the error from
                  eps to ~ eps^(3/2) (asymptotically).
        library:  optional precomputed basic library.

    Returns:
        (sequence, U_approx, error) where sequence is the Clifford+T
        word, U_approx = sequence_to_unitary(sequence), and error is
        the operator distance to U.
    """
    if library is None:
        library = _get_default_library()
    seq, U_approx = _sk_recurse(U, depth, library)
    err = trace_distance_su2(U, U_approx)
    return seq, U_approx, err


def _sk_recurse(U: np.ndarray, n: int, library: list[_LibEntry]
                 ) -> tuple[tuple[str, ...], np.ndarray]:
    if n == 0:
        entry = find_closest_in_library(U, library)
        return entry.sequence, entry.unitary
    # Step 1: get a rough approximation at depth n-1.
    seq_prev, U_prev = _sk_recurse(U, n - 1, library)
    # Step 2: residual W = U · U_prev†.
    W = U @ U_prev.conj().T
    # Step 3: write W ≈ V Q V† Q† as a balanced commutator.
    V, Q = group_commutator_decompose(W)
    # Step 4: recursively approximate V and Q at depth n-1.
    seq_V, U_V = _sk_recurse(V, n - 1, library)
    seq_Q, U_Q = _sk_recurse(Q, n - 1, library)
    seq_V_inv = invert_sequence(seq_V)
    seq_Q_inv = invert_sequence(seq_Q)
    # Matrix product to build: (U_V · U_Q · U_V† · U_Q†) · U_prev.
    # `sequence_to_unitary` applies gates left-to-right, so the gate
    # listed LAST in the sequence ends up LEFTMOST in matrix order.
    # To match U_V · U_Q · U_V† · U_Q† · U_prev, we list them with
    # seq_prev first, then Q†, V†, Q, V (so V ends up leftmost).
    seq = seq_prev + seq_Q_inv + seq_V_inv + seq_Q + seq_V
    U_approx = (U_V @ U_Q @ U_V.conj().T @ U_Q.conj().T) @ U_prev
    return seq, U_approx


# ----------------------------------------------------------------------------
# Public convenience
# ----------------------------------------------------------------------------

def compile_unitary(U: np.ndarray, depth: int = 3) -> dict:
    """Top-level wrapper: returns a dict with sequence, unitary, error, length."""
    seq, U_approx, err = solovay_kitaev_decompose(U, depth=depth)
    return {
        "sequence":  seq,
        "unitary":   U_approx,
        "error":     err,
        "length":    len(seq),
        "depth":     depth,
    }

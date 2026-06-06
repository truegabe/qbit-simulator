"""Qudit (d-dimensional) quantum simulator.

A "qudit" generalises the qubit from d=2 to any integer d≥2.
For d=10 ("qudecimal") each register digit can be 0–9, so a single
particle carries log₂(10) ≈ 3.32 bits of classical alphabet, and n
qudits span a Hilbert space of dimension 10ⁿ.

Generalized Pauli / Heisenberg-Weyl gates (d arbitrary):
  X_d   shift gate:   |j⟩ → |(j+1) mod d⟩
  Z_d   phase gate:   |j⟩ → ω^j |j⟩   with ω = e^(2πi/d)
  H_d   QFT gate:     H[j,k] = ω^(jk) / √d
  CSUM  controlled-add: |j,k⟩ → |j, (k+j) mod d⟩

This module provides:
  - `QuditState`   — state-vector for n qudits of dimension d
  - Gate functions returning d×d (or d²×d²) matrices
  - `QuditCircuit` — thin wrapper with apply / measure helpers
  - `partial_trace_marginals` — Bob's reduced probabilities given Alice's side
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Gate matrices (single qudit, d-dimensional)
# ---------------------------------------------------------------------------

def omega(d: int) -> complex:
    """Primitive d-th root of unity."""
    return np.exp(2j * np.pi / d)


def X_gate(d: int) -> np.ndarray:
    """Generalized shift (Pauli-X) for dimension d.

    X|j⟩ = |(j+1) mod d⟩
    Matrix: X[i,j] = δ_{i, (j+1) mod d}
    """
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0
    return X


def Z_gate(d: int) -> np.ndarray:
    """Generalized phase (Pauli-Z) for dimension d.

    Z|j⟩ = ω^j |j⟩
    """
    w = omega(d)
    return np.diag([w ** j for j in range(d)]).astype(complex)


def H_gate(d: int) -> np.ndarray:
    """Quantum Fourier Transform gate (generalized Hadamard) for dimension d.

    H[j,k] = ω^(jk) / √d
    """
    w = omega(d)
    H = np.array([[w ** (j * k) for k in range(d)] for j in range(d)],
                 dtype=complex) / np.sqrt(d)
    return H


def CSUM_gate(d: int) -> np.ndarray:
    """Controlled-SUM gate for two qudits of dimension d.

    CSUM |j, k⟩ = |j, (k + j) mod d⟩
    Layout: computational basis ordered as |00⟩, |01⟩, ..., |0,d-1⟩, |10⟩, ...
    So index (j, k) → j*d + k.
    """
    dim = d * d
    U = np.zeros((dim, dim), dtype=complex)
    for j in range(d):
        for k in range(d):
            row = j * d + (k + j) % d
            col = j * d + k
            U[row, col] = 1.0
    return U


def X_power(d: int, t: int) -> np.ndarray:
    """X_d^t — shift by t positions."""
    Xt = np.eye(d, dtype=complex)
    X = X_gate(d)
    for _ in range(t % d):
        Xt = X @ Xt
    return Xt


# ---------------------------------------------------------------------------
# Qudit state vector
# ---------------------------------------------------------------------------

@dataclass
class QuditState:
    """State vector for n qudits of local dimension d.

    The basis is ordered as:
        |x_0, x_1, ..., x_{n-1}⟩  →  index = Σ x_i * d^(n-1-i)
    (big-endian: qudit 0 is the most significant).
    """
    n: int                      # number of qudits
    d: int = 10                 # local dimension (10 for "qudecimal")
    amplitudes: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        dim = self.d ** self.n
        if self.amplitudes is None:
            # Default: |00...0⟩
            self.amplitudes = np.zeros(dim, dtype=complex)
            self.amplitudes[0] = 1.0
        else:
            self.amplitudes = np.asarray(self.amplitudes, dtype=complex)
            assert len(self.amplitudes) == dim, \
                f"Expected {dim} amplitudes, got {len(self.amplitudes)}"

    @property
    def dim(self) -> int:
        return self.d ** self.n

    def probs(self) -> np.ndarray:
        """Born probabilities for each basis state."""
        return (np.abs(self.amplitudes) ** 2).real

    def index_to_digits(self, idx: int) -> list[int]:
        """Decompose a basis index into per-qudit values [x_0, ..., x_{n-1}]."""
        digits = []
        for _ in range(self.n):
            digits.append(idx % self.d)
            idx //= self.d
        return list(reversed(digits))

    def digits_to_index(self, digits: list[int]) -> int:
        idx = 0
        for x in digits:
            idx = idx * self.d + x
        return idx

    def marginal(self, qudit: int) -> np.ndarray:
        """Reduced probability distribution p(x_qudit = v) for v in 0..d-1.

        Vectorized: reshape prob vector to (d,d,...,d) then sum all axes
        except the target qudit axis.  O(d^n) time, no Python element loop.
        """
        probs = self.probs().reshape([self.d] * self.n)
        # Sum over every axis except `qudit`.
        axes_to_sum = tuple(i for i in range(self.n) if i != qudit)
        return probs.sum(axis=axes_to_sum)

    def measure(self, qudit: int,
                rng: np.random.Generator | None = None) -> int:
        """Projective measurement on `qudit`. Collapses state in-place.

        Returns the observed outcome (integer 0..d-1).
        """
        if rng is None:
            rng = np.random.default_rng()
        marg = self.marginal(qudit)
        marg = np.maximum(marg, 0)
        marg /= marg.sum()
        outcome = int(rng.choice(self.d, p=marg))
        # Project: zero out all amplitudes where qudit != outcome.
        # Vectorized: reshape, zero all slices except outcome, reshape back.
        psi = self.amplitudes.reshape([self.d] * self.n)
        mask = np.zeros([self.d] * self.n, dtype=complex)
        # Build index to select only the outcome slice along `qudit` axis.
        idx = [slice(None)] * self.n
        idx[qudit] = outcome
        mask[tuple(idx)] = psi[tuple(idx)]
        norm = np.linalg.norm(mask)
        self.amplitudes = (mask / norm).reshape(self.dim)
        return outcome

    def apply_single(self, gate: np.ndarray, qudit: int) -> None:
        """Apply a d×d unitary to a single qudit."""
        # Reshape amplitudes to (d, d, ..., d) — one axis per qudit.
        psi = self.amplitudes.reshape([self.d] * self.n)
        # Contract: new[..., v, ...] = Σ_u gate[v, u] * psi[..., u, ...]
        psi = np.tensordot(gate, psi, axes=([1], [qudit]))
        # tensordot puts the new axis first; move it back to position `qudit`.
        psi = np.moveaxis(psi, 0, qudit)
        self.amplitudes = psi.reshape(self.dim)

    def apply_two(self, gate: np.ndarray,
                  qudit_a: int, qudit_b: int) -> None:
        """Apply a d²×d² unitary to qudit pair (qudit_a, qudit_b).

        gate[v_a*d + v_b, u_a*d + u_b]
        """
        psi = self.amplitudes.reshape([self.d] * self.n)
        # Collect the two target axes, contract, then put them back.
        # Step 1: merge the two target axes to a single d²-dim axis.
        psi = np.moveaxis(psi, [qudit_a, qudit_b], [0, 1])
        original_shape = psi.shape  # (d, d, rest...)
        rest_shape = psi.shape[2:]
        psi = psi.reshape(self.d * self.d, -1)  # (d², rest_flat)
        gate = gate.reshape(self.d * self.d, self.d * self.d)
        psi = gate @ psi              # (d², rest_flat)
        psi = psi.reshape((self.d, self.d) + rest_shape)
        psi = np.moveaxis(psi, [0, 1], [qudit_a, qudit_b])
        self.amplitudes = psi.reshape(self.dim)

    def copy(self) -> "QuditState":
        return QuditState(self.n, self.d, self.amplitudes.copy())


# ---------------------------------------------------------------------------
# QuditCircuit — convenience wrapper
# ---------------------------------------------------------------------------

class QuditCircuit:
    """High-level qudit circuit builder.

    Example (d=10, 2 qudits — the entanglement experiment):

        qc = QuditCircuit(n=2, d=10)
        qc.H(0)                   # Fourier-superpose qudit 0
        qc.CSUM(0, 1)             # Entangle: |Φ⟩ = Σ_j (1/√10)|j,j⟩
        a = qc.measure(0, rng)    # Alice measures — gets 0..9
        b = qc.measure(1, rng)    # Bob measures — always == a (if same basis)
    """

    def __init__(self, n: int, d: int = 10,
                 init_state: np.ndarray | None = None) -> None:
        self.n = n
        self.d = d
        self.state = QuditState(n, d, init_state)

    # ---- single-qudit gates ----

    def X(self, qudit: int, t: int = 1) -> None:
        """Apply X_d^t (shift by t) to `qudit`."""
        self.state.apply_single(X_power(self.d, t), qudit)

    def Z(self, qudit: int) -> None:
        """Apply Z_d to `qudit`."""
        self.state.apply_single(Z_gate(self.d), qudit)

    def H(self, qudit: int) -> None:
        """Apply QFT gate H_d to `qudit`."""
        self.state.apply_single(H_gate(self.d), qudit)

    def Hdg(self, qudit: int) -> None:
        """Apply H_d† (inverse QFT) to `qudit`."""
        self.state.apply_single(H_gate(self.d).conj().T, qudit)

    def custom(self, gate: np.ndarray, qudit: int) -> None:
        """Apply an arbitrary d×d unitary to a single qudit."""
        self.state.apply_single(gate, qudit)

    # ---- two-qudit gates ----

    def CSUM(self, control: int, target: int) -> None:
        """Apply CSUM_d: |j,k⟩ → |j,(k+j) mod d⟩."""
        self.state.apply_two(CSUM_gate(self.d), control, target)

    def CSUMdg(self, control: int, target: int) -> None:
        """Apply CSUM_d†: |j,k⟩ → |j,(k-j) mod d⟩."""
        self.state.apply_two(CSUM_gate(self.d).conj().T, control, target)

    # ---- measurement ----

    def measure(self, qudit: int,
                rng: np.random.Generator | None = None) -> int:
        """Projective Z-basis measurement. Returns int 0..d-1."""
        return self.state.measure(qudit, rng)

    def marginal(self, qudit: int) -> np.ndarray:
        """Return marginal probability vector for `qudit` without collapsing."""
        return self.state.marginal(qudit)

    def probs(self) -> np.ndarray:
        return self.state.probs()

    def amplitudes(self) -> np.ndarray:
        return self.state.amplitudes.copy()

    def copy(self) -> "QuditCircuit":
        qc = QuditCircuit(self.n, self.d)
        qc.state = self.state.copy()
        return qc


# ---------------------------------------------------------------------------
# Partial trace helpers (for no-communication analysis)
# ---------------------------------------------------------------------------

def partial_trace_marginals(qc: QuditCircuit, bob_qudit: int) -> np.ndarray:
    """Return Bob's reduced density diagonal without Alice measuring.

    This gives P(Bob = v) by tracing out all other qudits.
    For a pure state this equals `qc.state.marginal(bob_qudit)`.
    """
    return qc.state.marginal(bob_qudit)


def bob_marginal_after_alice_measures(qc_fresh: QuditCircuit,
                                       alice_qudit: int,
                                       alice_basis_gate: np.ndarray | None,
                                       bob_qudit: int,
                                       rng: np.random.Generator,
                                       n_shots: int = 2000,
                                       ) -> dict:
    """Simulate `n_shots` trials.  Alice optionally rotates to a different
    basis before measuring.  Return Bob's empirical marginal.

    This is the key test: if Alice's choice of basis gate shifts Bob's
    marginal distribution, information is transmitted.  If Bob's
    marginal is always {0: 1/d, 1: 1/d, ..., d-1: 1/d} regardless of
    what Alice does, the no-communication theorem is confirmed.
    """
    bob_counts = np.zeros(qc_fresh.d, dtype=int)
    for _ in range(n_shots):
        qc = qc_fresh.copy()
        # Alice's optional basis rotation.
        if alice_basis_gate is not None:
            qc.state.apply_single(alice_basis_gate, alice_qudit)
        # Alice measures — she gets some outcome; we discard it.
        qc.measure(alice_qudit, rng)
        # Bob measures.
        b = qc.measure(bob_qudit, rng)
        bob_counts[b] += 1
    return {
        "counts": bob_counts,
        "freqs":  bob_counts / n_shots,
        "entropy_bits": float(_entropy_bits(bob_counts / n_shots)),
    }


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

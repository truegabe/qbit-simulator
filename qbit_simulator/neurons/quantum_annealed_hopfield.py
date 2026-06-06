"""Quantum-annealed Hopfield retrieval.

The classical Hopfield network's recall operation iterates

    s_i  <-  sign( Σ_j W_ij s_j )

This is greedy descent on the energy landscape

    E(s) = -1/2 Σ_{i,j} W_ij s_i s_j.

It can fall into local minima ("spurious" attractors) rather than the
true stored pattern, especially when:
  - The probe is very corrupted.
  - Stored patterns are correlated.
  - Network is loaded near capacity (~0.14 N).

Quantum annealing replaces this iterative descent with adiabatic
evolution from a UNIFORM superposition (ground state of a transverse
field) to the Hopfield Hamiltonian:

    H_0       =  -Γ Σ_i X_i              (transverse field, easy)
    H_target  =  -1/2 Σ_ij W_ij Z_i Z_j  (Hopfield, hard)
    H(s)      =  (1-s) H_0  +  s H_target,    s ∈ [0, 1]

When evolved slowly enough, the system stays in the instantaneous
ground state — and the final ground state of H_target is the stored
pattern that best matches the probe (encoded as initial-state bias).

For small N (≤ 8 qubits) this is *exactly* simulable in our state-
vector engine; we leverage the existing adiabatic_evolve infrastructure.

Provides:
  - `hopfield_to_ising(W)`: convert Hopfield weight matrix into a
    transverse-field Ising Hamiltonian.
  - `QuantumAnnealedHopfield`: a drop-in replacement for HopfieldNetwork
    that uses quantum annealing for retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Single-qubit Paulis.
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)


def _op_on(op: np.ndarray, idx: int, n: int) -> np.ndarray:
    ops = [_I] * n
    ops[idx] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def hopfield_hamiltonian(W: np.ndarray, h_bias: np.ndarray | None = None
                          ) -> np.ndarray:
    """Build the Hopfield target Hamiltonian H_target as a dense matrix.

        H = -1/2 Σ_{i<j} (W_ij + W_ji) Z_i Z_j  -  Σ_i h_i Z_i

    (Hopfield W is symmetric, so 1/2(W+W^T) == W; we keep the form general.)
    """
    n = W.shape[0]
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            J = 0.5 * (W[i, j] + W[j, i])
            if J != 0:
                H -= J * (_op_on(_Z, i, n) @ _op_on(_Z, j, n))
    if h_bias is not None:
        for i in range(n):
            if h_bias[i] != 0:
                H -= h_bias[i] * _op_on(_Z, i, n)
    return H


def transverse_field(n: int, gamma: float = 1.0) -> np.ndarray:
    """H_0 = -gamma · sum_i X_i. Ground state = uniform |+>^n."""
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n):
        H -= gamma * _op_on(_X, i, n)
    return H


def plus_state(n: int) -> np.ndarray:
    """|+>^{⊗n}: ground state of -sum X_i, uniform superposition."""
    d = 2 ** n
    return np.ones(d, dtype=complex) / np.sqrt(d)


def state_to_pattern(psi: np.ndarray, n: int) -> np.ndarray:
    """Decode a state vector: read the maximum-probability basis state and
    convert to a ±1 spin pattern.

    Convention: Z|0> = +|0>, Z|1> = -|1>, so bit=0 ↔ spin=+1, bit=1 ↔ spin=-1.
    """
    probs = np.abs(psi) ** 2
    idx = int(np.argmax(probs))
    bits = np.array([(idx >> (n - 1 - k)) & 1 for k in range(n)])
    return np.where(bits > 0, -1, 1)


@dataclass
class QuantumAnnealedHopfield:
    """Hopfield network with quantum-annealing-based retrieval.

    Stores patterns via the standard Hebbian rule. Recall iterates a
    slow adiabatic schedule from H_0 (transverse field) to H_target
    (Hopfield Hamiltonian).

    For N > 8 the dense state-vector evolution becomes expensive; this
    class is intended as a cognitive demonstration on small problems.
    """
    n: int
    W: np.ndarray = field(default=None, repr=False)
    patterns: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.W is None:
            self.W = np.zeros((self.n, self.n))

    # ---- storage ----
    def store(self, pattern: np.ndarray) -> None:
        """Add a ±1 pattern to memory via outer-product Hebbian rule."""
        p = np.where(pattern > 0, 1, -1).astype(np.float64)
        self.W += np.outer(p, p) / self.n
        np.fill_diagonal(self.W, 0)
        self.patterns.append(p)

    def store_patterns(self, patterns) -> None:
        for p in patterns:
            self.store(p)

    # ---- classical retrieval (for comparison) ----
    def classical_retrieve(self, probe: np.ndarray, n_iter: int = 50
                            ) -> np.ndarray:
        s = np.where(probe > 0, 1, -1).astype(np.float64)
        for _ in range(n_iter):
            s_new = np.sign(self.W @ s)
            s_new = np.where(s_new == 0, s, s_new)
            if np.array_equal(s_new, s):
                break
            s = s_new
        return s.astype(int)

    # ---- quantum retrieval ----
    def quantum_retrieve(self, probe: np.ndarray,
                          n_steps: int = 100,
                          total_time: float = 8.0,
                          bias_strength: float = 1.0) -> dict:
        """Quantum-annealed retrieval.

        Args:
            probe: ±1 array of length n. Encoded as a small longitudinal
                bias on the target Hamiltonian so the annealer is gently
                pulled toward matching patterns.
            n_steps: number of Trotter steps in the schedule.
            total_time: total annealing time (larger = more adiabatic).
            bias_strength: how strongly the probe biases the target H.

        Returns dict with:
            'pattern' (decoded ±1 spins),
            'overlaps' (cosine overlap with each stored pattern),
            'final_energy' (⟨ψ| H_target |ψ⟩).
        """
        n = self.n
        # Build Hamiltonians.
        h_bias = bias_strength * np.where(probe > 0, 1.0, -1.0)
        H_T = hopfield_hamiltonian(self.W, h_bias=h_bias)
        H_0 = transverse_field(n, gamma=1.0)
        # Initial state = |+>^n (ground of H_0).
        psi = plus_state(n)
        dt = total_time / n_steps
        # Trotterized adiabatic evolution.
        from scipy.linalg import expm
        for k in range(1, n_steps + 1):
            s = k / n_steps
            H_s = (1 - s) * H_0 + s * H_T
            U = expm(-1j * H_s * dt)
            psi = U @ psi
            psi /= np.linalg.norm(psi)
        # Decode.
        pattern = state_to_pattern(psi, n)
        # Overlaps with stored patterns.
        overlaps = []
        for p in self.patterns:
            overlaps.append(float(pattern @ p) / n)
        # Final energy.
        H_T_no_bias = hopfield_hamiltonian(self.W, h_bias=None)
        E = float(np.real(psi.conj() @ H_T_no_bias @ psi))
        return {
            "pattern":      pattern,
            "overlaps":     overlaps,
            "final_energy": E,
            "psi":          psi,
        }


def closest_stored_pattern(decoded: np.ndarray, patterns: list
                            ) -> tuple[int, float]:
    """Return (best_index, overlap) of the stored pattern closest to decoded."""
    best = -1; best_o = -np.inf
    for k, p in enumerate(patterns):
        o = float(decoded @ p) / len(decoded)
        if o > best_o:
            best_o = o; best = k
    return best, best_o

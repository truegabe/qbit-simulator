"""Quantum variational predictive-coding inference.

Predictive coding (Rao & Ballard 1999) infers latent z given sensory x
by minimizing prediction error:

    z* = argmin_z  ||x - g(z)||^2 + λ ||z||^2

Standard PC does this by gradient descent in z. This module replaces
that descent with a VARIATIONAL QUANTUM approach: a parameterized
quantum state |ψ(θ)⟩ represents a posterior q(z); training θ to
minimize ⟨ψ| H_PC |ψ⟩ finds the best q.

Crucially, |ψ(θ)⟩ can place amplitude on MULTIPLE plausible z values
simultaneously (genuine superposition / posterior uncertainty), where
the classical optimizer commits to one z at a time. This matches the
Bayesian-brain hypothesis that cortex represents posteriors, not just
point estimates.

We implement:
  - `pc_hamiltonian(g, x, n_qubits)`: build a diagonal H whose entry
    H[k, k] = error of generative model at z = decode_index(k).
  - `HardwareEfficientAnsatz`: a parameterized circuit
    (RY layers + CNOT entanglers) implemented in numpy.
  - `QuantumVariationalPC.fit(x)`: trains θ to minimize ⟨H⟩.
  - `.posterior_probs()`: read out p(z) from |ψ(θ*)|².
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2); s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _kron_apply(op: np.ndarray, target: int, n: int,
                 state: np.ndarray) -> np.ndarray:
    """Apply 1-qubit op to qubit `target` of an n-qubit state."""
    tensor = state.reshape((2,) * n)
    tensor = np.moveaxis(tensor, target, 0)
    shape = tensor.shape
    tensor = op @ tensor.reshape(2, -1)
    tensor = tensor.reshape(shape)
    tensor = np.moveaxis(tensor, 0, target)
    return tensor.reshape(2 ** n)


def _cnot_apply(control: int, target: int, n: int,
                 state: np.ndarray) -> np.ndarray:
    """Apply CNOT(control, target)."""
    tensor = state.reshape((2,) * n)
    tensor = np.moveaxis(tensor, [control, target], [0, 1])
    shape = tensor.shape
    flat = tensor.reshape(4, -1)
    CNOT = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype=complex)
    flat = CNOT @ flat
    tensor = flat.reshape(shape)
    tensor = np.moveaxis(tensor, [0, 1], [control, target])
    return tensor.reshape(2 ** n)


# ----------------------------------------------------------------------------
# Hardware-efficient ansatz
# ----------------------------------------------------------------------------

@dataclass
class HardwareEfficientAnsatz:
    """L layers of (RY on each qubit) + (chained CNOTs)."""
    n_qubits: int
    n_layers: int = 3

    @property
    def n_params(self) -> int:
        return self.n_qubits * (self.n_layers + 1)

    def state(self, params: np.ndarray) -> np.ndarray:
        """Build |ψ(params)⟩ starting from |0⟩^{⊗n}."""
        n = self.n_qubits
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1.0
        idx = 0
        # First layer of RYs.
        for q in range(n):
            psi = _kron_apply(_ry(params[idx]), q, n, psi)
            idx += 1
        # L entangling layers.
        for _ in range(self.n_layers):
            for q in range(n - 1):
                psi = _cnot_apply(q, q + 1, n, psi)
            for q in range(n):
                psi = _kron_apply(_ry(params[idx]), q, n, psi)
                idx += 1
        return psi


# ----------------------------------------------------------------------------
# PC Hamiltonian construction
# ----------------------------------------------------------------------------

def pc_hamiltonian(generator: Callable[[int], np.ndarray],
                    x: np.ndarray, n_qubits: int,
                    reg: float = 0.0) -> np.ndarray:
    """Build a diagonal Hamiltonian over the latent space {0, ..., 2^n - 1}.

    Args:
        generator: callable z (int) → predicted sensory vector.
        x: observed sensory vector.
        n_qubits: log2 of the latent space dimension.
        reg: optional L2 penalty on z (encoded as |k - 2^{n-1}|).

    Returns diagonal complex matrix of shape (2^n, 2^n).
    """
    d = 2 ** n_qubits
    diag = np.zeros(d)
    for k in range(d):
        pred = generator(k)
        diag[k] = 0.5 * float(np.dot(x - pred, x - pred))
        if reg > 0:
            diag[k] += reg * abs(k - d // 2) ** 2
    return np.diag(diag.astype(np.complex128))


# ----------------------------------------------------------------------------
# Quantum variational PC
# ----------------------------------------------------------------------------

@dataclass
class QuantumVariationalPC:
    """Variational quantum inference of PC posterior."""
    n_qubits: int
    n_layers: int = 3
    eta: float = 0.05
    n_iter: int = 200
    ansatz: HardwareEfficientAnsatz = field(default=None)
    theta: np.ndarray = field(default=None, repr=False)
    H: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.ansatz is None:
            self.ansatz = HardwareEfficientAnsatz(self.n_qubits, self.n_layers)
        if self.theta is None:
            self.theta = self.rng.uniform(-0.1, 0.1, size=self.ansatz.n_params)

    def energy(self, theta: np.ndarray) -> float:
        psi = self.ansatz.state(theta)
        return float(np.real(psi.conj() @ self.H @ psi))

    def fit(self, generator: Callable[[int], np.ndarray], x: np.ndarray,
             verbose: bool = False) -> list:
        """Build H from generator+x, train θ to minimize energy."""
        self.H = pc_hamiltonian(generator, x, self.n_qubits)
        # Finite-difference gradient (parameter-shift would be cleaner
        # but FD is fine on a numpy simulator).
        eps = 1e-3
        losses = []
        for it in range(self.n_iter):
            losses.append(self.energy(self.theta))
            grad = np.zeros_like(self.theta)
            for k in range(len(self.theta)):
                self.theta[k] += eps
                e_plus = self.energy(self.theta)
                self.theta[k] -= 2 * eps
                e_minus = self.energy(self.theta)
                self.theta[k] += eps
                grad[k] = (e_plus - e_minus) / (2 * eps)
            self.theta -= self.eta * grad
            if verbose and it % 20 == 0:
                print(f"iter {it}: F = {losses[-1]:.4f}")
        return losses

    def posterior_probs(self) -> np.ndarray:
        """Read out p(z) = |⟨z | ψ(θ)⟩|² over all 2^n latents."""
        psi = self.ansatz.state(self.theta)
        return np.abs(psi) ** 2

    def map_estimate(self) -> int:
        return int(np.argmax(self.posterior_probs()))

    def free_energy(self) -> float:
        return self.energy(self.theta)

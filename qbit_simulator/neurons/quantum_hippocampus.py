"""Quantum-encoded hippocampal priors.

Hippocampus stores episodic memories as sparse patterns; on retrieval,
a partial cue triggers pattern completion. Classically this is a
Hopfield/CA3 attractor network.

This module wraps stored patterns into a QUANTUM PRIOR — a density
matrix whose eigenvectors are the stored memories and whose
eigenvalues encode their relative strength / recency. Then retrieval
becomes full Bayesian inference:

    posterior  ∝  prior · likelihood
    ρ_post     ∝  exp(-β H_cue) · ρ_prior · exp(-β H_cue)

where H_cue is a Hermitian operator whose ground state matches the
cue. The retrieved memory is the eigenstate of ρ_post with the
largest eigenvalue.

Compared to classical Hopfield:
  - The prior is normalized (Tr ρ = 1) so storage capacity is
    explicit, not an emergent property.
  - Multiple plausible completions can be retrieved simultaneously
    (read the top-k eigenvectors of ρ_post).
  - Recency / familiarity is a tunable spectrum on ρ_prior.

We work with dense matrices for small N (≤ 8 qubits, ≤ 256 states).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _pattern_to_state(pattern: np.ndarray) -> np.ndarray:
    """Map ±1 spin pattern to a computational-basis state vector.

    Convention: spin=+1 ↔ |0>, spin=-1 ↔ |1>.
    """
    n = len(pattern)
    bits = np.where(pattern > 0, 0, 1)
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[idx] = 1.0
    return psi


def _state_to_pattern(psi: np.ndarray, n: int) -> np.ndarray:
    """Inverse: max-prob basis state → ±1 pattern."""
    idx = int(np.argmax(np.abs(psi) ** 2))
    bits = np.array([(idx >> (n - 1 - k)) & 1 for k in range(n)])
    return np.where(bits > 0, -1, 1)


@dataclass
class QuantumHippocampus:
    """Stores patterns as eigenvectors of a density matrix prior."""
    n: int
    decay: float = 0.95     # recency decay per new memory
    rho_prior: np.ndarray = field(default=None, repr=False)
    patterns: list = field(default_factory=list)
    weights:  list = field(default_factory=list)

    def __post_init__(self) -> None:
        d = 2 ** self.n
        if self.rho_prior is None:
            # Start with maximally mixed prior (uniform belief).
            self.rho_prior = np.eye(d, dtype=complex) / d

    # ---- storage ----
    def store(self, pattern: np.ndarray, weight: float = 1.0) -> None:
        """Add a memory. Older memories decay in weight."""
        # Decay existing weights.
        self.weights = [w * self.decay for w in self.weights]
        psi = _pattern_to_state(pattern)
        self.patterns.append(psi)
        self.weights.append(weight)
        self._rebuild_prior()

    def _rebuild_prior(self) -> None:
        d = 2 ** self.n
        rho = np.zeros((d, d), dtype=complex)
        for psi, w in zip(self.patterns, self.weights):
            rho += w * np.outer(psi, psi.conj())
        # Tiny mixed background so prior is full-rank (numerical only).
        rho += 1e-6 * np.eye(d, dtype=complex)
        rho /= np.real(np.trace(rho))
        self.rho_prior = rho

    # ---- retrieval ----
    def cue_hamiltonian(self, cue: np.ndarray,
                          bias: float = 1.0) -> np.ndarray:
        """Build H_cue diagonal in Z basis: low energy where bits agree
        with cue."""
        n = self.n
        d = 2 ** n
        H = np.zeros((d, d), dtype=complex)
        for k in range(d):
            bits = np.array([(k >> (n - 1 - i)) & 1 for i in range(n)])
            spins = np.where(bits > 0, -1, 1)
            # Energy = -bias * sum(cue_i * spin_i): low when aligned.
            H[k, k] = -bias * float(cue @ spins)
        return H

    def retrieve(self, cue: np.ndarray, beta: float = 4.0,
                  bias: float = 1.0) -> dict:
        """Bayesian retrieval.

        ρ_post ∝ exp(-β H_cue/2) · ρ_prior · exp(-β H_cue/2)
        (symmetric ordering to keep ρ Hermitian).

        Returns:
            'pattern':   the MAP completion (±1 array).
            'posterior_probs': p(pattern_k | cue) for each stored pattern.
            'top_eigenvectors': decoded eigenvectors of ρ_post by largest weight.
        """
        H_cue = self.cue_hamiltonian(cue, bias=bias)
        # exp(-βH/2). Since H is diagonal:
        diag = np.diag(H_cue)
        sqrt_factor = np.exp(-beta * np.real(diag) / 2)
        # Symmetric similarity transform on prior.
        M = np.diag(sqrt_factor).astype(complex)
        rho_post = M @ self.rho_prior @ M
        rho_post /= np.real(np.trace(rho_post))
        # Diagonalize ρ_post.
        evals, evecs = np.linalg.eigh(rho_post)
        # Sort descending.
        order = np.argsort(-evals.real)
        evals = evals[order]; evecs = evecs[:, order]
        # MAP completion: top eigenvector decoded.
        pattern = _state_to_pattern(evecs[:, 0], self.n)
        # Posteriors over stored memories.
        posterior_probs = []
        for psi_k in self.patterns:
            # Probability mass on pattern k = <psi_k| rho_post |psi_k>.
            p_k = float(np.real(psi_k.conj() @ rho_post @ psi_k))
            posterior_probs.append(p_k)
        return {
            "pattern":         pattern,
            "posterior_probs": posterior_probs,
            "rho_post":        rho_post,
            "eigenvalues":     evals.real,
            "top_eigenvectors": evecs,
        }

    def memory_purity(self) -> float:
        """Tr(ρ²): 1 if all weight on one memory, 1/d for uniform."""
        return float(np.real(np.trace(self.rho_prior @ self.rho_prior)))

    def memory_entropy(self) -> float:
        """von Neumann entropy of the prior."""
        evals = np.linalg.eigvalsh(self.rho_prior).real
        evals = evals[evals > 1e-12]
        return float(-(evals * np.log(evals)).sum())

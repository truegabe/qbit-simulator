"""Quantum-enhanced active inference.

Friston's active inference picks the action by minimizing Expected
Free Energy (EFE) over all candidate POLICIES π (sequences of actions):

    π* = argmin_π  G(π)

The classical `ActiveInferenceAgent` enumerates all (n_actions)^depth
policies exhaustively — O(A^D) work. For a 3-action agent at depth 5,
that's 243 policies; at depth 8, 6561.

This module replaces that brute-force search with **quantum amplitude
amplification**: prepare a uniform superposition over all policies,
mark the policy with the lowest EFE via a phase oracle, and use
Generalized AA to amplify it. Speedup: O(√(A^D)) oracle queries
instead of O(A^D).

Architecture
------------
1. Compute EFE for every policy classically (one pass, cheap for our
   sims since each policy evaluation is a few matrix-vector products).
2. Translate EFEs into a sign-flip oracle that marks the K lowest-EFE
   policies (default K=1).
3. Run GAA on the uniform superposition over (A^D) policies.
4. Return the most-amplified policy.

For our small toy problems the cost of step 1 dominates step 3 — so
this is a TOY DEMONSTRATION of the speedup, not a wall-clock
improvement. The classical EFE pre-computation is needed to BUILD the
oracle, since we don't have a real quantum oracle for the world
model. On a fault-tolerant quantum computer, this oracle would be
implemented directly from a quantum forward-model circuit, and the
full algorithm would achieve the √-speedup end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .active_inference import ActiveInferenceAgent
from ..algorithms.generalized_amplitude_amplification import (
    amp_amp_with_oracle, optimal_iterations,
)


@dataclass
class QuantumActiveInferenceAgent(ActiveInferenceAgent):
    """Active-inference agent that selects policies via amplitude amplification.

    Inherits the full classical machinery (belief update, A/B/C matrices,
    EFE computation) and adds a quantum policy-selection method.
    """

    def enumerate_policies(self) -> list[tuple]:
        """All (n_actions)^policy_depth candidate policies."""
        return list(product(range(self.n_actions), repeat=self.policy_depth))

    def policy_efes(self) -> tuple[list, np.ndarray]:
        """Return (list_of_policies, EFE_array)."""
        polies = self.enumerate_policies()
        # Classical EFE returns SCORE (higher = better); EFE = -score.
        scores = np.array([self.expected_free_energy(list(p)) for p in polies])
        return polies, -scores

    def best_action_quantum(self, n_marked: int = 1) -> dict:
        """Pick the first action of the best policy via amplitude amplification.

        Args:
            n_marked: how many of the lowest-EFE policies to mark as "good".

        Returns dict with:
          - 'action':         first action of the chosen policy.
          - 'policy':         the chosen policy tuple.
          - 'prob_marked':    probability mass on marked policies AFTER AA.
          - 'prob_initial':   uniform prob = n_marked / N.
          - 'n_policies':     A^D.
          - 'n_aa_iters':     applied iterations.
        """
        polies, efes = self.policy_efes()
        N = len(polies)
        # Pad to power of two if needed.
        n_qubits = int(np.ceil(np.log2(max(2, N))))
        d = 2 ** n_qubits
        # Lowest-EFE policy indices (within polies array).
        marked_idx = np.argsort(efes)[:n_marked]
        marked = [int(i) for i in marked_idx]
        # Uniform-superposition preparation = n-qubit Hadamard.
        H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H = H1
        for _ in range(n_qubits - 1):
            H = np.kron(H, H1)
        # Run GAA. Treat padded slots as un-marked.
        out = amp_amp_with_oracle(H, marked=marked)
        probs = np.abs(out["psi_final"]) ** 2
        # MAP outcome among the d-dim register.
        idx = int(np.argmax(probs))
        # Map back to valid policy index (mod N to handle padding).
        chosen = polies[idx % N]
        action = chosen[0]
        return {
            "action":       action,
            "policy":       chosen,
            "all_efes":     efes,
            "prob_marked":  out["prob_final"],
            "prob_initial": out["prob_initial"],
            "n_policies":   N,
            "n_aa_iters":   out["n_iters"],
            "n_qubits":     n_qubits,
        }

    def step_quantum(self, observation: int, n_marked: int = 1) -> int:
        """One full step using quantum action selection."""
        self.update_belief(observation)
        out = self.best_action_quantum(n_marked=n_marked)
        return out["action"]


def speedup_analysis(n_actions: int, policy_depth: int) -> dict:
    """Compare classical (O(N)) vs. quantum (O(√N)) policy search."""
    N = n_actions ** policy_depth
    classical_queries = N
    quantum_queries = max(1, optimal_iterations(1.0 / max(N, 2)))
    speedup = classical_queries / max(quantum_queries, 1)
    return {
        "n_policies":           N,
        "classical_queries":    classical_queries,
        "quantum_queries":      quantum_queries,
        "theoretical_speedup":  speedup,
    }

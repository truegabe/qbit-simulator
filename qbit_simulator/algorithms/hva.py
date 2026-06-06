"""Hamiltonian Variational Ansatz (HVA) — Wecker et al. 2015.

A physically motivated ansatz for VQE on molecular and lattice Hamiltonians:

    |ψ(θ)⟩ = ∏_l ∏_α exp(-i θ_{l,α} H_α) |ψ_ref⟩

where H = Σ_α H_α is a decomposition of the Hamiltonian into terms, and
|ψ_ref⟩ is some easy-to-prepare reference state (e.g. Hartree-Fock).

Each layer exponentiates each Pauli term once with a separate angle. The
ansatz is "naturally adapted" to the Hamiltonian: at θ = 0, |ψ(0)⟩ = |ψ_ref⟩,
and small θ probes states close to that reference along the directions
corresponding to each Hamiltonian term.

Built on top of `algorithms.trotter.apply_pauli_rotation` so each
exponential is decomposed into elementary gates (basis-change, CNOT
cascade, Rz, uncompute).

For our existing VQE pipeline:
    - Use `make_hva_ansatz(H, n_layers, reference_state)` to get a
      parameterized circuit-builder.
    - Pass it to `vqe(...)` or any optimizer.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..circuit import QuantumCircuit
from ..pauli import PauliOp
from .trotter import apply_pauli_rotation


def make_hva_ansatz(
    H: PauliOp,
    n_layers: int = 1,
    reference_state: np.ndarray | None = None,
) -> tuple[Callable[[np.ndarray], QuantumCircuit], int]:
    """Construct an HVA ansatz callable for `vqe(...)`-style optimization.

    Args:
        H:                Hamiltonian as a PauliOp.
        n_layers:         number of repetitions of the layer structure.
        reference_state:  initial state. Default: |0...0⟩.

    Returns:
        (ansatz_fn, n_params) where ansatz_fn(θ) returns a QuantumCircuit
        and n_params = n_layers · (number of non-identity terms in H).
    """
    # Filter out identity terms (they only contribute a global phase).
    nontrivial = [(c, s) for (c, s) in H.terms if any(ch != "I" for ch in s)]
    n_terms = len(nontrivial)
    n_qubits = len(H.terms[0][1])
    total_params = n_layers * n_terms

    def ansatz(theta: np.ndarray) -> QuantumCircuit:
        if len(theta) != total_params:
            raise ValueError(f"need {total_params} parameters, got {len(theta)}")
        qc = QuantumCircuit(n_qubits)
        if reference_state is not None:
            qc.state = np.asarray(reference_state, dtype=np.complex128).copy()
        idx = 0
        for _ in range(n_layers):
            for (c, s) in nontrivial:
                # Combine the parameter with the term's coefficient.
                apply_pauli_rotation(qc, s, float(c.real) * float(theta[idx]))
                idx += 1
        return qc

    return ansatz, total_params


def hva_vqe(
    H: PauliOp,
    n_layers: int = 2,
    reference_state: np.ndarray | None = None,
    seed: int = 0,
    max_iter: int = 300,
    init_scale: float = 0.1,
) -> dict:
    """Run VQE with an HVA ansatz on the given Hamiltonian.

    Returns:
        dict with energy_opt, params_opt, fidelity_with_ground.
    """
    from scipy.optimize import minimize

    ansatz_fn, n_params = make_hva_ansatz(H, n_layers, reference_state)
    H_matrix = H.matrix()

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-init_scale, init_scale, size=n_params)

    def cost(theta):
        qc = ansatz_fn(theta)
        psi = qc.state
        return float(np.real(psi.conj() @ H_matrix @ psi))

    result = minimize(cost, theta0, method="L-BFGS-B",
                       options={"maxiter": max_iter, "ftol": 1e-9})

    qc_opt = ansatz_fn(result.x)
    psi_opt = qc_opt.state
    e_exact, gs = H.ground_state()
    fidelity = abs(np.vdot(gs, psi_opt)) ** 2

    return {
        "energy_opt":   float(result.fun),
        "params_opt":   result.x,
        "fidelity":     float(fidelity),
        "ground_energy": float(e_exact),
        "n_params":     n_params,
    }

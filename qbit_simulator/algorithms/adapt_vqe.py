"""AdaPT-VQE: adaptive ansatz growth from an operator pool.

Standard UCCSD picks a fixed list of single + double excitations and
trotterizes them. AdaPT-VQE (Grimsley et al. 2019) instead **grows** the
ansatz one operator at a time:

  1. Start with the reference state |HF⟩ and an empty ansatz.
  2. At each iteration:
        a. Compute the energy gradient with respect to each operator in
           the pool:  g_k = ⟨ψ | [H, A_k] | ψ⟩.
        b. Pick the operator with the largest |g_k|.
        c. Append exp(θ_k · A_k) to the ansatz; re-optimize ALL θ's.
  3. Stop when the maximum gradient is below a threshold ε.

The resulting ansatz is much shorter than full UCCSD for the same
energy — and addresses the trotter-ordering issue that limits standard
UCCSD's expressivity (which we saw on 2-site Hubbard, where Trotterized
UCCSD couldn't reach the exact GS).

This module:

  - `operator_pool_singles_doubles(n_qubits, occupied)`: generate the
    standard UCCSD pool (singles and doubles, separated into individual
    "anti-Hermitian terms" rather than full T_ia - T_ia^†).
  - `gradient(psi, H, A, n_qubits)`: ⟨ψ| [H, A] |ψ⟩.
  - `adapt_vqe(H, n_qubits, occupied, ...)`: run the full AdaPT-VQE
    procedure and return the optimized energy + ansatz history.

We re-use the UCCSD primitives (excitation generators, state-vector
application) from `ucc.py`.
"""

from __future__ import annotations

import numpy as np

from .ucc import (
    single_excitation, double_excitation,
    singles_generators, doubles_generators,
    apply_excitation, hartree_fock_state,
    _generator_as_hermitian_matrix, _pauli_op_to_matrix,
)
from ..fermion import FermionOp
from ..pauli import PauliOp


# ----------------------------------------------------------------------------
# Operator pool
# ----------------------------------------------------------------------------

def operator_pool_singles_doubles(n_qubits: int, occupied: list[int]
                                    ) -> list[tuple[tuple, FermionOp]]:
    """The standard UCCSD-style pool: all occ→virt singles and doubles."""
    return (singles_generators(n_qubits, occupied)
            + doubles_generators(n_qubits, occupied))


# ----------------------------------------------------------------------------
# Gradient
# ----------------------------------------------------------------------------

def operator_gradient(psi: np.ndarray, H_matrix: np.ndarray,
                       A: FermionOp, n_qubits: int) -> float:
    """⟨ψ | [H, A] | ψ⟩ — the gradient of ⟨ψ| exp(-θ A) H exp(θ A) |ψ⟩
    at θ = 0.

    Since A is anti-Hermitian (after JW: i · Hermitian), we get a real
    gradient. We compute it as:

        ∂E/∂θ at θ=0 = ⟨ψ | [H, A] | ψ⟩

    Internally we use the Hermitian version H_A = i · JW(A) and:
        [H, A]  =  [H, -i · H_A]  =  -i [H, H_A]
        ⟨[H, A]⟩  =  -i ⟨[H, H_A]⟩
    which is purely imaginary; we return -i · (it) = real gradient.
    """
    H_A = _generator_as_hermitian_matrix(A, n_modes=n_qubits)
    # A is anti-Hermitian, with A = -i · H_A and H_A Hermitian.
    # [H, A] = -i · [H, H_A]; since H and H_A are both Hermitian, [H, H_A]
    # is anti-Hermitian, so ⟨[H, A]⟩ = -i · (purely imag) = purely real.
    commutator = H_matrix @ (-1j * H_A) - (-1j * H_A) @ H_matrix
    val = psi.conj() @ commutator @ psi
    return float(np.real(val))


# ----------------------------------------------------------------------------
# Ansatz application
# ----------------------------------------------------------------------------

def apply_ansatz(thetas: list[float], generators: list[FermionOp],
                  ref: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply prod_k exp(theta_k · G_k) to the reference state."""
    psi = ref.copy()
    for theta, G in zip(thetas, generators):
        psi = apply_excitation(psi, G, float(theta), n_qubits)
    return psi


def ansatz_energy(thetas: list[float], H_matrix: np.ndarray,
                   generators: list[FermionOp], ref: np.ndarray,
                   n_qubits: int) -> float:
    psi = apply_ansatz(thetas, generators, ref, n_qubits)
    return float(np.real(psi.conj() @ H_matrix @ psi))


# ----------------------------------------------------------------------------
# AdaPT-VQE main loop
# ----------------------------------------------------------------------------

def adapt_vqe(
    hamiltonian: PauliOp,
    n_qubits: int,
    occupied: list[int],
    pool: list[tuple[tuple, FermionOp]] | None = None,
    gradient_tol: float = 1e-3,
    max_iter: int = 20,
    optimizer: str = "BFGS",
    verbose: bool = False,
) -> dict:
    """Run AdaPT-VQE.

    Returns:
        dict with keys:
            "energy":      final energy
            "thetas":      optimized parameters
            "operators":   (indices, generators) selected, in order
            "gradients":   gradient norms at each iteration
            "energies":    energy at each iteration
            "converged":   bool
    """
    from scipy.optimize import minimize

    if pool is None:
        pool = operator_pool_singles_doubles(n_qubits, occupied)

    H_matrix = _pauli_op_to_matrix(hamiltonian, n_qubits)
    H_matrix = 0.5 * (H_matrix + H_matrix.conj().T)
    ref = hartree_fock_state(n_qubits, occupied)

    selected: list[tuple[tuple, FermionOp]] = []
    thetas: list[float] = []
    gradient_history: list[float] = []
    energy_history: list[float] = [float(np.real(ref.conj() @ H_matrix @ ref))]

    for it in range(max_iter):
        # Current state with current ansatz.
        psi_current = apply_ansatz(thetas, [g for _, g in selected], ref, n_qubits)

        # Compute gradient for each operator in the pool.
        grads = []
        for idx, A in pool:
            g = operator_gradient(psi_current, H_matrix, A, n_qubits)
            grads.append(abs(g))
        max_grad = max(grads)
        gradient_history.append(max_grad)
        if verbose:
            print(f"iter {it}: max |gradient| = {max_grad:.4e}, "
                  f"current E = {energy_history[-1]:.6f}")
        if max_grad < gradient_tol:
            return {
                "energy":     energy_history[-1],
                "thetas":     thetas,
                "operators":  selected,
                "gradients":  gradient_history,
                "energies":   energy_history,
                "converged":  True,
            }

        # Pick the operator with largest gradient.
        best_k = int(np.argmax(grads))
        selected.append(pool[best_k])
        thetas.append(0.0)

        # Re-optimize ALL thetas.
        def loss(x):
            return ansatz_energy(list(x), H_matrix,
                                  [g for _, g in selected], ref, n_qubits)

        res = minimize(loss, np.array(thetas), method=optimizer,
                        options={"maxiter": 200, "gtol": 1e-8})
        thetas = list(res.x)
        energy_history.append(float(res.fun))

    return {
        "energy":     energy_history[-1],
        "thetas":     thetas,
        "operators":  selected,
        "gradients":  gradient_history,
        "energies":   energy_history,
        "converged":  False,
    }

"""Quantum state and process tomography.

State tomography reconstructs an unknown density matrix ρ from measurement
data. Given enough measurements in tomographically-complete Pauli bases, ρ
is uniquely determined by

    ρ = (1 / 2^N) · Σ_P ⟨P⟩ · P

summed over all 4^N Pauli strings P, where ⟨P⟩ = Tr(ρP).

Process tomography characterizes a quantum channel ε by performing state
tomography on the output for a tomographically-complete set of input states.

This implementation provides:
    - state_tomography(state, shots): measure Pauli expectations via
      finite-sample simulation, returning estimated ⟨P⟩ for all P.
    - reconstruct_density_matrix(expectations): linear inversion ρ_est = Σ ⟨P⟩P/2^N.
    - state_fidelity(rho_a, rho_b): F = Tr(√(√ρ_a · ρ_b · √ρ_a))² for two
      density matrices.
    - process_tomography(channel_fn, n_qubits, shots): characterize ε.

For finite-sample tomography, the reconstructed density matrix may have
negative eigenvalues; we expose both the raw linear-inversion result and
an optional projection onto the nearest positive-semidefinite matrix.
"""

from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np
from scipy.linalg import sqrtm

from .gates import I2, X, Y, Z


PAULI_MATRICES = {"I": I2, "X": X, "Y": Y, "Z": Z}


def all_pauli_strings(n: int) -> list[str]:
    """Enumerate all 4^N Pauli strings on n qubits."""
    return ["".join(p) for p in product("IXYZ", repeat=n)]


def pauli_string_matrix(s: str) -> np.ndarray:
    """Build the dense 2^N × 2^N matrix for a Pauli string."""
    m = PAULI_MATRICES[s[0]]
    for ch in s[1:]:
        m = np.kron(m, PAULI_MATRICES[ch])
    return m


# ----------------------------------------------------------------------------
# State tomography
# ----------------------------------------------------------------------------

def exact_pauli_expectations(rho: np.ndarray) -> dict[str, float]:
    """Compute Tr(ρ P) exactly for every Pauli string P. Returns a dict."""
    n = int(np.log2(rho.shape[0]))
    out: dict[str, float] = {}
    for s in all_pauli_strings(n):
        P = pauli_string_matrix(s)
        out[s] = float(np.real(np.trace(rho @ P)))
    return out


def sampled_pauli_expectations(
    rho: np.ndarray,
    shots_per_basis: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Estimate ⟨P⟩ for each P from finite-shot measurements in each Pauli basis.

    For each non-identity Pauli string P, we sample `shots_per_basis` joint
    measurements (with simulated shot noise) and estimate ⟨P⟩ as the mean
    parity of the outcomes. The all-identity string is exactly 1.
    """
    n = int(np.log2(rho.shape[0]))
    out: dict[str, float] = {}
    for s in all_pauli_strings(n):
        if all(c == "I" for c in s):
            out[s] = 1.0
            continue
        # Diagonalize P: eigenvalues are ±1.
        P = pauli_string_matrix(s)
        eigvals, eigvecs = np.linalg.eigh(P)
        # Compute probability of each eigenvalue from ρ in the P-eigenbasis.
        rho_in_basis = eigvecs.conj().T @ rho @ eigvecs
        probs = np.real(np.diag(rho_in_basis))
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        # Sample `shots_per_basis` outcomes weighted by probs.
        samples = rng.choice(len(probs), size=shots_per_basis, p=probs)
        sample_eigs = eigvals[samples]
        out[s] = float(sample_eigs.mean())
    return out


def reconstruct_density_matrix(
    expectations: dict[str, float], n_qubits: int,
) -> np.ndarray:
    """Linear-inversion tomography:
        ρ_est = (1 / 2^N) · Σ_P ⟨P⟩ · P
    """
    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=np.complex128)
    norm = 1.0 / dim
    for s, e in expectations.items():
        rho += norm * e * pauli_string_matrix(s)
    return rho


def project_to_psd(rho: np.ndarray) -> np.ndarray:
    """Project a Hermitian matrix to the nearest positive-semidefinite matrix
    by clipping negative eigenvalues and renormalizing the trace to 1."""
    rho = (rho + rho.conj().T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0, None)
    if eigvals.sum() < 1e-14:
        # Degenerate; return uniform.
        n = rho.shape[0]
        return np.eye(n, dtype=np.complex128) / n
    eigvals /= eigvals.sum()
    return eigvecs @ np.diag(eigvals.astype(np.complex128)) @ eigvecs.conj().T


def state_tomography(
    state_or_rho: np.ndarray,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
    project: bool = True,
) -> dict:
    """Estimate a density matrix from measurement data.

    Args:
        state_or_rho: either a state vector (will be converted to ρ = |ψ⟩⟨ψ|)
                      or a density matrix.
        shots: if None, compute exact Pauli expectations (no shot noise).
               Otherwise, simulate `shots` per Pauli basis.
        rng:   numpy generator for shot simulation.
        project: project the linear-inversion result onto the PSD cone.

    Returns:
        dict with:
            rho_estimated: 2^N × 2^N density matrix
            expectations:  dict of Pauli expectation values
            fidelity:      |⟨ψ|ρ_est|ψ⟩| if pure-state input given
    """
    if state_or_rho.ndim == 1:
        rho_true = np.outer(state_or_rho, state_or_rho.conj())
        original_state = state_or_rho
    else:
        rho_true = state_or_rho
        original_state = None
    n = int(np.log2(rho_true.shape[0]))

    if shots is None:
        expectations = exact_pauli_expectations(rho_true)
    else:
        rng = rng or np.random.default_rng()
        expectations = sampled_pauli_expectations(rho_true, shots, rng)

    rho_est = reconstruct_density_matrix(expectations, n)
    if project:
        rho_est = project_to_psd(rho_est)

    result: dict = {
        "rho_estimated": rho_est,
        "expectations":  expectations,
        "n_qubits":      n,
    }
    if original_state is not None:
        result["fidelity"] = float(np.real(
            original_state.conj() @ rho_est @ original_state
        ))
    return result


# ----------------------------------------------------------------------------
# Process tomography (single-qubit channels)
# ----------------------------------------------------------------------------

# Tomographically complete set of single-qubit input states (4 pure states).
SINGLE_QUBIT_TOMO_INPUTS = {
    "0":      np.array([1, 0],          dtype=np.complex128),
    "1":      np.array([0, 1],          dtype=np.complex128),
    "+":      np.array([1, 1],          dtype=np.complex128) / np.sqrt(2),
    "+i":     np.array([1, 1j],         dtype=np.complex128) / np.sqrt(2),
}


def choi_matrix_from_kraus(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Convert a Kraus representation to the (un-normalized) Choi matrix.

    Choi matrix: J(ε) = (I ⊗ ε)|Φ+⟩⟨Φ+| where |Φ+⟩ = (|00⟩+|11⟩)/√2 (on
    a 1-qubit channel). For dimension d input, the Choi matrix is d²×d².
    """
    d = kraus_ops[0].shape[0]
    # Maximally entangled state |Φ+⟩ on (input ⊗ output).
    phi_plus = np.zeros(d * d, dtype=np.complex128)
    for i in range(d):
        phi_plus[i * d + i] = 1.0
    phi_plus /= np.sqrt(d)
    # Apply (I ⊗ K) to |Φ+⟩, sum |K_i Φ+⟩⟨K_i Φ+|.
    J = np.zeros((d * d, d * d), dtype=np.complex128)
    for K in kraus_ops:
        IK = np.kron(np.eye(d, dtype=np.complex128), K)
        v = IK @ phi_plus
        J += np.outer(v, v.conj())
    return J


def process_tomography_single_qubit(
    channel_fn: Callable[[np.ndarray], np.ndarray],
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Tomographically characterize a single-qubit channel ε.

    Args:
        channel_fn: callable that takes a 2-d state vector OR 2×2 density
                    matrix, and returns the output 2×2 density matrix.
        shots: shots per Pauli basis per input state (None = exact).
        rng: RNG for shot simulation.

    Returns:
        dict with:
            output_states: dict of {label: estimated ρ_out}
            choi_matrix:   estimated 4×4 Choi matrix
    """
    rng = rng or np.random.default_rng()
    outputs: dict[str, np.ndarray] = {}
    for label, psi_in in SINGLE_QUBIT_TOMO_INPUTS.items():
        rho_in = np.outer(psi_in, psi_in.conj())
        rho_out = channel_fn(rho_in)
        # Estimate via tomography on output state.
        tomo = state_tomography(rho_out, shots=shots, rng=rng, project=True)
        outputs[label] = tomo["rho_estimated"]

    # Reconstruct Choi matrix from output states using the standard formula
    # J = Σ_{ij} |i⟩⟨j| ⊗ ε(|i⟩⟨j|).
    rho_00 = outputs["0"]
    rho_11 = outputs["1"]
    rho_p  = outputs["+"]
    rho_pi = outputs["+i"]
    # ε(|0⟩⟨1|) = (1 - i)[ρ_+] − (1 + i)[ρ_+i] + (1/2 − i/2)(ρ_00 + ρ_11)?
    # Use the cleaner formula:
    #   ε(|0⟩⟨1|) = ρ_+ + i·ρ_+i − (1+i)/2 (ρ_00 + ρ_11)
    eps_01 = rho_p + 1j * rho_pi - (1 + 1j) / 2 * (rho_00 + rho_11)
    eps_10 = eps_01.conj().T
    # Assemble Choi: J = |0⟩⟨0|⊗ρ_00 + |0⟩⟨1|⊗eps_01 + |1⟩⟨0|⊗eps_10 + |1⟩⟨1|⊗ρ_11.
    J = np.zeros((4, 4), dtype=np.complex128)
    J[0:2, 0:2] = rho_00
    J[0:2, 2:4] = eps_01
    J[2:4, 0:2] = eps_10
    J[2:4, 2:4] = rho_11

    return {
        "output_states": outputs,
        "choi_matrix":   J,
    }


def state_fidelity(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Quantum-information state fidelity F(ρ_a, ρ_b) = Tr(√(√ρ_a · ρ_b · √ρ_a))²."""
    sqrtA = sqrtm(rho_a)
    inner = sqrtA @ rho_b @ sqrtA
    sqrtInner = sqrtm(inner)
    tr = np.trace(sqrtInner)
    return float(np.real(tr) ** 2)

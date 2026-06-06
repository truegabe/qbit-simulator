"""Sachdev-Ye-Kitaev (SYK) model — a tractable quantum-gravity toy model.

The SYK model is a system of M Majorana fermions {χ_i} with random q-body
interactions:

    H = (i^(q/2) / q!) · Σ_{i_1 < ... < i_q} J_{i_1,...,i_q} χ_{i_1} ... χ_{i_q}

where the J's are independent Gaussian random variables with variance

    ⟨J²⟩ = J₀² · (q-1)! / M^(q-1)

(the standard normalization that gives a well-defined large-M limit).

Why this matters for quantum gravity:
  - SYK is maximally chaotic (saturates the Maldacena-Shenker-Stanford bound
    on the Lyapunov exponent λ = 2π/β for the out-of-time-ordered correlator).
  - It has a holographic dual: at low energies, SYK reduces to the
    "Schwarzian theory," which is the boundary theory of a 2D anti-de-Sitter
    black hole.
  - It's exactly solvable in the large-M limit while remaining tractable on
    a quantum simulator at finite M — letting us check gravity predictions
    against quantum-mechanical numerics.

This module:
  - `syk_hamiltonian(M, q, seed, J)` builds a random SYK Hamiltonian as a
    PauliOp (M Majoranas → M/2 qubits via Jordan-Wigner).
  - `level_spacing_distribution(H)` computes the unfolded level spacings
    (Wigner-Dyson statistics = chaotic; Poisson = integrable).
  - `out_of_time_ordered_correlator(H, V, W, beta, times)` computes the
    OTOC ⟨[W(t), V]² ⟩_β, whose Lyapunov-like growth is a chaos signature.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

from ..pauli import PauliOp
from ..fermion import _pauli_string_mul, _pauli_terms_simplify


def majorana_pauli(k: int, N_qubits: int) -> tuple[str, complex]:
    """k-th Majorana operator as a Pauli string on `N_qubits` qubits.

    Jordan-Wigner mapping for 2N Majoranas on N qubits:
        χ_{2j+1} = Z^⊗j ⊗ X ⊗ I^⊗(N-j-1)
        χ_{2j+2} = Z^⊗j ⊗ Y ⊗ I^⊗(N-j-1)

    Args:
        k: Majorana index, 1-indexed (so k ∈ [1, 2N]).
        N_qubits: number of qubits = number of fermion modes.

    Returns:
        (pauli_string, phase) — the phase is always 1 here; included for
        compatibility with the Pauli-multiplication helpers.
    """
    if not (1 <= k <= 2 * N_qubits):
        raise ValueError(f"Majorana index {k} out of [1, {2*N_qubits}]")
    k0 = k - 1
    j = k0 // 2
    chars = ["Z"] * j + ["X" if k0 % 2 == 0 else "Y"] + ["I"] * (N_qubits - j - 1)
    return "".join(chars), 1.0 + 0j


def syk_hamiltonian(
    M_majoranas: int,
    q: int = 4,
    seed: int = 0,
    J: float = 1.0,
) -> PauliOp:
    """Build a random SYK Hamiltonian as a PauliOp.

    Args:
        M_majoranas: number of Majorana fermions (must be even).
        q: body order of interactions (typically 4).
        seed: RNG seed for the random couplings.
        J: overall coupling-strength scale.

    Returns:
        PauliOp on M_majoranas / 2 qubits.
    """
    if M_majoranas <= 0 or M_majoranas % 2 != 0:
        raise ValueError("M_majoranas must be a positive even integer")
    if q < 2 or q > M_majoranas:
        raise ValueError("q must satisfy 2 ≤ q ≤ M_majoranas")
    N_qubits = M_majoranas // 2

    rng = np.random.default_rng(seed)
    # Standard SYK normalization for the coupling variance.
    variance = (J ** 2) * factorial(q - 1) / (M_majoranas ** (q - 1))
    sigma = np.sqrt(variance)

    # Real prefactor that makes H Hermitian for even q: i^(q/2) / q!
    # For q = 4 → i^2 / 24 = -1/24, which is real. For odd q (rare in SYK)
    # the prefactor would be imaginary — we restrict to even q.
    if q % 2 != 0:
        raise ValueError("only even q is implemented (Hermitian H requires it)")
    prefactor = ((1j) ** (q // 2)).real / factorial(q)

    accumulated: list[tuple[complex, str]] = []
    identity_pauli = "I" * N_qubits
    for tuple_indices in combinations(range(1, M_majoranas + 1), q):
        # Sample one random Gaussian coupling.
        J_coupling = float(rng.normal(0.0, sigma))
        if J_coupling == 0.0:
            continue
        # Multiply the q Majoranas together as Pauli strings.
        coef = complex(prefactor * J_coupling)
        product_pauli = identity_pauli
        product_phase = 1 + 0j
        for k in tuple_indices:
            chi_pauli, chi_phase = majorana_pauli(k, N_qubits)
            phase, new_pauli = _pauli_string_mul(product_pauli, chi_pauli)
            product_phase *= phase * chi_phase
            product_pauli = new_pauli
        accumulated.append((coef * product_phase, product_pauli))

    # Combine like terms.
    terms = _pauli_terms_simplify(accumulated)
    if not terms:
        # Degenerate (all couplings sampled to zero, ~probability 0). Return a
        # tiny identity Hamiltonian to keep PauliOp happy.
        terms = [(0 + 0j, identity_pauli)]
    return PauliOp(terms)


def fermion_parity_operator(N_qubits: int) -> np.ndarray:
    """Fermion parity P for 2N Majoranas (= N qubits) after Jordan-Wigner.

    Direct calculation:
        P = i^N · χ_1 χ_2 ... χ_{2N}
          = i^N · ∏_j (iZ_j)
          = i^N · i^N · ∏_j Z_j
          = (-1)^N · Z^⊗N
    """
    sign = (-1) ** N_qubits
    P = np.array([[sign]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    for _ in range(N_qubits):
        P = np.kron(P, Z)
    return P


def project_to_parity_sector(
    H_matrix: np.ndarray, parity_sign: int = 1,
) -> np.ndarray:
    """Restrict H to one fermion-parity sector (±1 eigenspace of P).

    For SYK (and any number-conserving fermionic Hamiltonian) the
    spectrum splits into ±1 sectors of P. Each sector independently
    obeys random-matrix-theory statistics (GUE for SYK_q=4 with M mod 8 ∈ {2, 6}).
    """
    N = int(np.log2(H_matrix.shape[0]))
    P = fermion_parity_operator(N)
    eigvals_P, eigvecs_P = np.linalg.eigh(P)
    mask = np.abs(eigvals_P - parity_sign) < 1e-9
    V = eigvecs_P[:, mask]
    return V.conj().T @ H_matrix @ V


def level_spacing_distribution(H: PauliOp) -> dict:
    """Compute the unfolded level-spacing distribution of H's spectrum.

    For chaotic systems: spacings follow Wigner-Dyson distribution P(s) =
    (π/2) s exp(-π s²/4) (GOE) — show level repulsion (P(0) = 0).
    For integrable systems: spacings follow Poisson P(s) = exp(-s) — no
    level repulsion.

    Returns:
        dict with eigenvalues, spacings, mean_spacing,
        and r_statistic (the average ratio of consecutive spacings,
        a parameter-free chaos diagnostic; GOE: ~0.53, Poisson: ~0.39).
    """
    M = H.matrix()
    eigvals = np.sort(np.linalg.eigvalsh(M))
    spacings = np.diff(eigvals)
    # r-statistic: r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    if len(spacings) >= 2:
        r_values = []
        for i in range(len(spacings) - 1):
            a, b = spacings[i], spacings[i + 1]
            if max(a, b) > 1e-12:
                r_values.append(min(a, b) / max(a, b))
        r_stat = float(np.mean(r_values)) if r_values else 0.0
    else:
        r_stat = 0.0
    return {
        "eigenvalues":   eigvals,
        "spacings":      spacings,
        "mean_spacing":  float(np.mean(spacings)) if len(spacings) else 0.0,
        "r_statistic":   r_stat,
    }


def thermal_density_matrix(H_matrix: np.ndarray, beta: float) -> np.ndarray:
    """ρ_β = exp(-β H) / Z."""
    eigvals, eigvecs = np.linalg.eigh(H_matrix)
    eigvals -= eigvals[0]   # avoid overflow at large beta
    weights = np.exp(-beta * eigvals)
    Z = weights.sum()
    weights /= Z
    return eigvecs @ np.diag(weights.astype(np.complex128)) @ eigvecs.conj().T


def out_of_time_ordered_correlator(
    H: PauliOp,
    V: np.ndarray,
    W: np.ndarray,
    beta: float,
    times: np.ndarray,
) -> np.ndarray:
    """Thermal OTOC C(t) = -⟨ [W(t), V]² ⟩_β.

    W(t) = e^{iHt} W e^{-iHt}. For chaotic systems with a holographic dual,
    C(t) grows as ~ exp(λ_L t) where λ_L is the Lyapunov exponent, with
    λ_L ≤ 2π/β (Maldacena-Shenker-Stanford bound). SYK saturates this bound.

    Args:
        H: Hamiltonian PauliOp.
        V, W: observables (same dimension as H).
        beta: inverse temperature.
        times: 1-D array of evaluation times.

    Returns:
        Array of C(t) values (real).
    """
    H_mat = H.matrix()
    rho_beta = thermal_density_matrix(H_mat, beta)
    eigvals, eigvecs = np.linalg.eigh(H_mat)
    V_e = eigvecs.conj().T @ V @ eigvecs
    W_e = eigvecs.conj().T @ W @ eigvecs
    rho_e = eigvecs.conj().T @ rho_beta @ eigvecs

    out = np.zeros(len(times), dtype=np.float64)
    for ti, t in enumerate(times):
        phase = np.exp(-1j * eigvals * t)
        # W(t) in eigenbasis: (phase_i^*) · W_e[i,j] · phase_j
        Wt = (np.outer(phase.conj(), phase)) * W_e
        commutator = Wt @ V_e - V_e @ Wt
        otoc = np.trace(rho_e @ commutator @ commutator)
        out[ti] = float(-np.real(otoc))
    return out

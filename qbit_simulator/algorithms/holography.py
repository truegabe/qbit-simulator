"""Holographic primitives — thermofield double states, entanglement entropy,
spectral form factor, Renyi entropies, Page curve, and operator-size growth.

These are the standard tools researchers use to study AdS/CFT, black-hole
information, and the SYK / Schwarzian / 2D-gravity correspondence:

  - **Thermofield Double (TFD)**: |TFD⟩_β = (1/√Z) Σ_n e^{-βE_n/2} |n⟩_L |n⟩_R
    on two copies of the system. Holographically dual to the eternal AdS
    black hole geometry connecting two boundary regions through a wormhole.

  - **Bipartite entanglement entropy**: S(ρ_A) = -Σ λ_i log λ_i where λ_i
    are the Schmidt singular values squared. Holographically equals the
    area of a minimal surface in the bulk (Ryu-Takayanagi).

  - **Page curve**: for a Haar-random pure state on N qubits, the average
    entropy ⟨S_A⟩ of a k-qubit subsystem peaks at k = N/2 and is symmetric
    around that point. The original prediction by Don Page (1993) for
    how a black hole's entanglement entropy with its environment evolves
    during evaporation — the "Page curve" that recent islands/replica
    papers have explained from first principles.

This module provides:
    - `thermofield_double_state(H, beta)`
    - `bipartite_entropy(psi, n_qubits, A)`
    - `mutual_information(psi, n_qubits, A, B)`
    - `haar_random_state(n_qubits, rng)`
    - `page_curve(n_qubits, n_samples)`
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def thermofield_double_state(H: np.ndarray, beta: float) -> np.ndarray:
    """Build the thermofield double state |TFD⟩_β on two copies of the system.

    |TFD⟩_β = (1/√Z) Σ_n e^{-β E_n / 2} |n⟩_L ⊗ |n⟩_R

    where {|n⟩} is an eigenbasis of H with eigenvalues E_n. Returns a
    state vector of dimension d² (d = dim(H)).

    Properties:
      - The reduced density matrix on either side is ρ_thermal = e^{-βH}/Z.
      - At β → ∞: TFD reduces to |GS⟩|GS⟩ (product of ground states).
      - At β → 0: TFD becomes maximally entangled |Φ+⟩.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix")
    eigvals, eigvecs = np.linalg.eigh(H)
    # Shift for numerical safety (avoids overflow at large β).
    eigvals_shifted = eigvals - eigvals[0]
    weights = np.exp(-beta * eigvals_shifted / 2)
    norm = np.sqrt((weights ** 2).sum())
    weights /= norm
    d = H.shape[0]
    tfd = np.zeros(d * d, dtype=np.complex128)
    for n in range(d):
        # |n⟩_L ⊗ |n⟩_R, where |n⟩ in computational basis is eigvecs[:, n].
        tfd += weights[n] * np.kron(eigvecs[:, n], eigvecs[:, n])
    return tfd


def _reshape_state_for_partition(
    psi: np.ndarray, n_qubits: int, A: Sequence[int],
) -> np.ndarray:
    """Reshape ψ into a |A| × |B| matrix where (row = A-subsystem index)."""
    A = sorted(A)
    B = [q for q in range(n_qubits) if q not in A]
    # Treat psi as a rank-n tensor of shape (2,)*n_qubits.
    t = psi.reshape((2,) * n_qubits)
    # Permute axes so that A axes come first.
    perm = list(A) + list(B)
    t = np.transpose(t, perm)
    n_A = len(A)
    return t.reshape(2 ** n_A, 2 ** (n_qubits - n_A))


def schmidt_values(psi: np.ndarray, n_qubits: int, A: Sequence[int]) -> np.ndarray:
    """Schmidt singular values of ψ across the A | B partition."""
    M = _reshape_state_for_partition(psi, n_qubits, A)
    s = np.linalg.svd(M, compute_uv=False)
    return s


def bipartite_entropy(
    psi: np.ndarray, n_qubits: int, A: Sequence[int], base: float = np.e,
) -> float:
    """Von Neumann entropy S(ρ_A) of subsystem A.

    Computed from the Schmidt singular values: S = -Σ λ²ᵢ log(λ²ᵢ) where λ
    are the SVD singular values of ψ reshaped as a |A|×|B| matrix.

    Args:
        psi: state vector of dimension 2^n_qubits.
        n_qubits: total qubit count.
        A: list of qubit indices in subsystem A.
        base: log base (default e for "nats"; use 2 for "bits").
    """
    s = schmidt_values(psi, n_qubits, A)
    p = s * s
    p = p[p > 1e-14]
    if base == np.e:
        return float(-(p * np.log(p)).sum())
    return float(-(p * np.log(p) / np.log(base)).sum())


def mutual_information(
    psi: np.ndarray, n_qubits: int,
    A: Sequence[int], B: Sequence[int],
    base: float = np.e,
) -> float:
    """Mutual information I(A : B) = S(A) + S(B) − S(A ∪ B).

    For a pure state, S(A ∪ B) is the entropy of the complement of A ∪ B
    (so the formula is symmetric and well-defined).
    """
    A_set = set(A); B_set = set(B)
    if A_set & B_set:
        raise ValueError("A and B must be disjoint")
    AB = sorted(A_set | B_set)
    s_A   = bipartite_entropy(psi, n_qubits, list(A), base=base)
    s_B   = bipartite_entropy(psi, n_qubits, list(B), base=base)
    s_AB  = bipartite_entropy(psi, n_qubits, AB, base=base)
    return s_A + s_B - s_AB


# ---- Renyi entropies ----

def renyi_entropy(
    psi: np.ndarray, n_qubits: int, A: Sequence[int],
    alpha: float = 2.0, base: float = np.e,
) -> float:
    """Renyi entropy of order α:

        S_α = (1 / (1 - α)) · log(Σ p_i^α)

    Special cases:
        α = 1   → von Neumann entropy (computed via limit / use `bipartite_entropy`)
        α = 2   → −log(purity) = −log(Σ p_i²)
        α = ∞   → −log(max p_i)

    Uses the Schmidt squared singular values as p_i.
    """
    if alpha < 0:
        raise ValueError("Renyi order must be non-negative")
    if abs(alpha - 1.0) < 1e-9:
        return bipartite_entropy(psi, n_qubits, A, base=base)
    s = schmidt_values(psi, n_qubits, A)
    p = s * s
    p = p[p > 1e-14]
    if alpha == float("inf"):
        val = -np.log(p.max())
    else:
        val = (1.0 / (1.0 - alpha)) * np.log((p ** alpha).sum())
    if base != np.e:
        val /= np.log(base)
    return float(val)


# ---- Spectral form factor ----

def spectral_form_factor(
    H: np.ndarray, times: np.ndarray, beta: float = 0.0,
) -> np.ndarray:
    """Spectral form factor

        g(t) = |Σ_n e^{-β E_n - i E_n t}|² / Z²

    where Z = Σ_n e^{-β E_n}. A characteristic chaos diagnostic:

      - Chaotic systems show a **dip-ramp-plateau** structure: initial
        decay, linear ramp ∝ t, then a plateau at level ~ 1/D where D is
        the Hilbert-space dimension. The slope of the ramp is governed
        by random-matrix-theory statistics (universal across all chaotic
        systems, including SYK).
      - Integrable systems show smooth Poisson-like decay without a ramp.

    The SFF complements the OTOC: OTOC probes operator scrambling,
    SFF probes spectral correlations.

    Args:
        H:     Hermitian Hamiltonian (matrix).
        times: 1-D array of t values.
        beta:  inverse temperature (default 0 = infinite temperature).

    Returns:
        Array of g(t) values.
    """
    eigs = np.linalg.eigvalsh(H)
    if beta > 0:
        eigs_shifted = eigs - eigs[0]
        weights = np.exp(-beta * eigs_shifted)
    else:
        weights = np.ones_like(eigs)
    Z = weights.sum()
    sff = np.zeros(len(times), dtype=np.float64)
    for ti, t in enumerate(times):
        amp = (weights * np.exp(-1j * eigs * t)).sum()
        sff[ti] = float(abs(amp) ** 2 / (Z ** 2))
    return sff


# ---- Page curve ----

def haar_random_state(
    n_qubits: int, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a Haar-uniformly random pure state on n qubits.

    Generate complex Gaussian amplitudes and normalize. The result is
    uniformly distributed on the (2·2^n - 1)-sphere — the Haar measure
    on pure states.
    """
    rng = rng or np.random.default_rng()
    d = 2 ** n_qubits
    re = rng.normal(size=d)
    im = rng.normal(size=d)
    psi = re + 1j * im
    psi /= np.linalg.norm(psi)
    return psi


def operator_size_growth(
    H: np.ndarray, O_initial: np.ndarray, n_qubits: int,
    times: np.ndarray,
) -> np.ndarray:
    """Average Pauli weight of the Heisenberg-evolved operator O(t).

    Decompose O(t) = e^{iHt} O e^{-iHt} in the Pauli basis:
        O(t) = Σ_P c_P(t) · P
    The "size" of P is its weight (number of non-identity Pauli factors).
    Then ⟨size(t)⟩ = Σ_P |c_P(t)|² · size(P) / Σ_P |c_P(t)|².

    For chaotic systems, ⟨size(t)⟩ grows quickly (often exponentially) at
    intermediate times — the "operator-spreading" phenomenon. For
    integrable systems, growth is polynomial.

    Cost: O(4^N · 2^(2N)) per time step. Practical up to N ≈ 6.

    Args:
        H:         Hamiltonian matrix.
        O_initial: initial operator (matrix on n_qubits).
        n_qubits:  number of qubits.
        times:     array of times to evaluate.

    Returns:
        Array of ⟨size(t)⟩ values.
    """
    from ..tomography import all_pauli_strings, pauli_string_matrix

    paulis = all_pauli_strings(n_qubits)
    P_matrices = [pauli_string_matrix(p) for p in paulis]
    sizes = np.array([sum(1 for c in p if c != "I") for p in paulis],
                     dtype=np.float64)

    eigvals, eigvecs = np.linalg.eigh(H)
    O_e = eigvecs.conj().T @ O_initial @ eigvecs

    out = np.zeros(len(times), dtype=np.float64)
    for ti, t in enumerate(times):
        phase = np.exp(-1j * eigvals * t)
        # O(t) in eigenbasis: (phase_i*) · O_e[i,j] · phase_j
        Ot_e = (np.outer(phase.conj(), phase)) * O_e
        Ot = eigvecs @ Ot_e @ eigvecs.conj().T
        # Decompose in Pauli basis: c_P = Tr(P · O) / 2^N
        coefs = np.array([np.trace(P @ Ot) for P in P_matrices])
        weights = np.abs(coefs) ** 2
        total = weights.sum()
        if total < 1e-14:
            out[ti] = 0.0
        else:
            out[ti] = float((weights * sizes).sum() / total)
    return out


def wormhole_teleportation_fidelity(
    H: np.ndarray,
    beta: float,
    message_state: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Teleport a qubit through a wormhole — TFD-as-quantum-channel demo.

    Setup (simplified Maldacena-Stanford / Gao-Jafferis-Wall style):

        - Two systems L and R prepared in the thermofield-double |TFD⟩_β
          of Hamiltonian H. They share thermal entanglement; at β = 0 the
          state is maximally entangled (a perfect wormhole), at β = ∞ it
          factorizes into ground states (no wormhole).

        - Alice has an additional "message" qubit she wishes to send to Bob
          on the R side. She performs a Bell-basis measurement on (her
          message qubit, the first qubit of L), then transmits classical
          bits.

        - Bob applies Pauli corrections on the first qubit of R based on
          Alice's classical bits, and ends up with a qubit whose state has
          fidelity F to Alice's original message.

    The fidelity F probes the "wormhole geometry":
        F = 1 at β = 0 (maximally entangled wormhole, perfect teleportation)
        F → 1/d at β = ∞ (no entanglement, random output)

    For the simple case (H = 0, β doesn't matter, TFD is maximally
    entangled), this reduces to standard quantum teleportation.

    Args:
        H:              Hamiltonian on n qubits (each side of TFD).
        beta:           inverse temperature.
        message_state:  2-d unit complex vector. Default: random.
        rng:            numpy generator.

    Returns:
        dict with input_state, output_state, fidelity, beta.
    """
    rng = rng or np.random.default_rng()
    d = H.shape[0]
    n_sys = int(np.log2(d))
    if 2 ** n_sys != d:
        raise ValueError("H dimension must be a power of 2")

    # Build TFD on 2n qubits.
    tfd = thermofield_double_state(H, beta)

    # Default message: a random pure state.
    if message_state is None:
        psi_msg = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi_msg /= np.linalg.norm(psi_msg)
    else:
        psi_msg = np.asarray(message_state, dtype=np.complex128).reshape(2)
        psi_msg /= np.linalg.norm(psi_msg)

    # Combined state: (message qubit) ⊗ (TFD on 2n qubits) = 2n+1 qubits.
    full_state = np.kron(psi_msg, tfd)

    # Alice's Bell measurement on (msg qubit, first qubit of L = qubit index 1
    # in the combined state). We apply CNOT(0, 1), H(0), then measure both.
    # Build the operators as full matrices acting on the (2n+1)-qubit space.
    n_total = 1 + 2 * n_sys
    dim_total = 2 ** n_total

    def kron_chain(ops: list[np.ndarray]) -> np.ndarray:
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    from ..gates import H as H_gate, I2, X, Z
    # CNOT(control=0, target=1) on (n_total)-qubit space:
    # The state has 2^(n_total - 2) "other" qubits — CNOT acts on bits 0 and 1.
    # We'll build it as a permutation matrix.
    CNOT_01 = np.eye(dim_total, dtype=np.complex128)
    n_others = n_total - 2
    for c in range(2):
        for t in range(2):
            for other in range(2 ** n_others):
                # Input index: c bit at qubit 0, t bit at qubit 1, other bits.
                in_idx = (c << (n_total - 1)) | (t << (n_total - 2)) | other
                # Output: t' = t XOR c (only when c = 1).
                t_new = t ^ c
                out_idx = (c << (n_total - 1)) | (t_new << (n_total - 2)) | other
                if in_idx != out_idx:
                    CNOT_01[in_idx, in_idx] = 0
                    CNOT_01[out_idx, in_idx] = 1
    full_state = CNOT_01 @ full_state

    # H on qubit 0.
    H_on_0 = kron_chain([H_gate] + [I2] * (n_total - 1))
    full_state = H_on_0 @ full_state

    # Measure qubits 0 and 1, then apply Pauli corrections on qubit 2 (the
    # "first qubit of R" — index 1 + n_sys in our layout).
    # Actually the standard teleportation: corrections on the R side's qubit
    # that's "paired" with the L qubit Alice consumed. In our TFD, the
    # qubits are paired index-for-index: qubit 1+i_L on L pairs with
    # qubit 1+n_sys+i_R on R. Alice consumed qubit 1 (first L qubit), so
    # the pair is qubit (1 + n_sys).
    target_R = 1 + n_sys

    # Compute joint outcome probabilities.
    probs = np.abs(full_state) ** 2
    # Marginalize over qubit-0 and qubit-1 outcomes.
    marginal = np.zeros((2, 2), dtype=np.float64)
    for idx in range(dim_total):
        m0 = (idx >> (n_total - 1)) & 1
        m1 = (idx >> (n_total - 2)) & 1
        marginal[m0, m1] += probs[idx]
    marginal_flat = marginal.flatten()
    marginal_flat = np.clip(marginal_flat, 0, None)
    marginal_flat /= marginal_flat.sum()
    outcome = int(rng.choice(4, p=marginal_flat))
    m0, m1 = outcome >> 1, outcome & 1

    # Project the state onto this outcome.
    projected = np.zeros_like(full_state)
    for idx in range(dim_total):
        if (((idx >> (n_total - 1)) & 1) == m0
                and ((idx >> (n_total - 2)) & 1) == m1):
            projected[idx] = full_state[idx]
    nrm = np.linalg.norm(projected)
    if nrm > 1e-12:
        projected /= nrm

    # Pauli corrections on target_R: if m1=1, apply X; if m0=1, apply Z.
    if m1:
        X_op = [I2] * n_total
        X_op[target_R] = X
        projected = kron_chain(X_op) @ projected
    if m0:
        Z_op = [I2] * n_total
        Z_op[target_R] = Z
        projected = kron_chain(Z_op) @ projected

    # Now extract the reduced state of target_R.
    # Reshape projected and trace out everything except target_R.
    psi_tensor = projected.reshape((2,) * n_total)
    # Compute density matrix of target_R by partial trace over all other qubits.
    # We move target_R to position 0, then sum over remaining axes.
    psi_moved = np.moveaxis(psi_tensor, target_R, 0)
    flat = psi_moved.reshape(2, -1)
    rho_R = flat @ flat.conj().T
    nrm_rho = np.trace(rho_R)
    if abs(nrm_rho) > 1e-12:
        rho_R = rho_R / nrm_rho

    # Fidelity with the input message.
    fidelity = float(np.real(psi_msg.conj() @ rho_R @ psi_msg))

    return {
        "input_state":   psi_msg,
        "output_rho":    rho_R,
        "fidelity":      fidelity,
        "beta":          beta,
        "bell_outcome":  (m0, m1),
    }


def lyapunov_from_otoc(
    times: np.ndarray,
    otoc_values: np.ndarray,
    fit_range: tuple[float, float] | None = None,
) -> dict:
    """Extract the Lyapunov exponent λ_L from OTOC growth.

    For chaotic systems with a holographic dual, the OTOC grows as:

        C(t) ~ C_0 · exp(λ_L · t)

    over an intermediate "Lyapunov regime" before saturation. λ_L is the
    quantum-Lyapunov-exponent analog of classical chaos: it measures the
    rate at which a small perturbation spreads through the operator.

    The Maldacena-Shenker-Stanford bound states λ_L ≤ 2π / β (in natural
    units, with β the inverse temperature). Holographic systems — and
    SYK at low T — saturate this bound exactly. This is one of the
    cleanest predictions of the SYK/holography correspondence.

    Args:
        times:        1-D time array.
        otoc_values:  corresponding C(t) values.
        fit_range:    (t_min, t_max) range to fit the exponential to. If
                      None, uses the middle third of the data (avoiding
                      the linear initial and the saturated late times).

    Returns:
        dict with:
            lyapunov:     extracted λ_L
            fit_t_min:    t_min used
            fit_t_max:    t_max used
            log_intercept: log(C_0)
    """
    times = np.asarray(times)
    otoc = np.asarray(otoc_values)
    if fit_range is None:
        # Use the middle third by default.
        n = len(times)
        fit_range = (times[n // 3], times[2 * n // 3])
    t_min, t_max = fit_range
    mask = (times >= t_min) & (times <= t_max) & (otoc > 1e-12)
    if mask.sum() < 2:
        return {"lyapunov": 0.0, "fit_t_min": t_min, "fit_t_max": t_max,
                "log_intercept": 0.0}
    t_fit = times[mask]
    log_C = np.log(otoc[mask])
    # Linear fit log(C) ≈ log(C_0) + λ_L · t
    slope, intercept = np.polyfit(t_fit, log_C, 1)
    return {
        "lyapunov":      float(slope),
        "fit_t_min":     float(t_min),
        "fit_t_max":     float(t_max),
        "log_intercept": float(intercept),
    }


def krylov_complexity(
    b_coeffs: list[float], times: np.ndarray,
) -> np.ndarray:
    """Krylov complexity C_K(t) from precomputed Lanczos b_n coefficients.

    Krylov complexity is the average index ⟨n⟩(t) of the operator's
    projection in the Krylov basis:

        C_K(t) = Σ_n n · |φ_n(t)|²

    where φ_n is the n-th amplitude. The φ_n's evolve via a "Schrödinger"
    equation on the Krylov chain with hopping amplitudes b_n:

        i dφ_n/dt = b_n φ_{n-1} + b_{n+1} φ_n

    For chaotic systems with linearly growing b_n ~ α·n + β, C_K(t) grows
    exponentially: C_K ~ exp(2α·t). For integrable systems, C_K grows
    polynomially. This is the "operator growth hypothesis" of Parker et al.
    2019.

    Args:
        b_coeffs:  list of Lanczos coefficients (from `lanczos_coefficients`).
        times:     time array at which to evaluate C_K.

    Returns:
        Array of C_K(t) values.
    """
    if not b_coeffs:
        return np.zeros(len(times))
    n_basis = len(b_coeffs) + 1
    bs = np.asarray(b_coeffs, dtype=np.float64)
    # Build the Krylov-chain Hamiltonian H_K (tridiagonal with b_n off-diagonals).
    H_K = np.zeros((n_basis, n_basis), dtype=np.float64)
    for n, b in enumerate(bs):
        H_K[n, n + 1] = b
        H_K[n + 1, n] = b
    eigvals, eigvecs = np.linalg.eigh(H_K)
    # Initial state: |φ_0⟩ = (1, 0, 0, ...)
    phi0 = np.zeros(n_basis); phi0[0] = 1.0
    phi0_diag = eigvecs.conj().T @ phi0
    out = np.zeros(len(times))
    n_indices = np.arange(n_basis)
    for ti, t in enumerate(times):
        # Evolve in eigenbasis
        phi_t_diag = np.exp(-1j * eigvals * t) * phi0_diag
        phi_t = eigvecs @ phi_t_diag
        # Probabilities |φ_n(t)|²
        probs = np.abs(phi_t) ** 2
        # Renormalize (numerical drift on Krylov truncation).
        probs /= probs.sum()
        out[ti] = float((n_indices * probs).sum())
    return out


def hayden_preskill_protocol(
    n_bh: int,
    k_alice: int,
    rng: np.random.Generator | None = None,
    base: float = np.e,
) -> dict:
    """Hayden-Preskill 2007: how a black hole "decodes" information.

    Setup:
        - The black hole is a system of N_BH qubits, initially in a Haar-
          random pure state (or you can think of it as scrambled to look so).
        - Alice has k qubits of "information" she's about to throw in.
          She creates a maximally entangled pair (A, R) where R is her
          reference outside the black hole.
        - She throws A in. Now the combined "BH + Alice" system is
          (N_BH + k) qubits.
        - A scrambling unitary acts on those (N_BH + k) qubits (Haar-random).
        - Hawking radiation D emerges sequentially as subsystems of the
          combined system. We compute the mutual information I(R : D) as
          a function of |D|.

    The Hayden-Preskill claim: if before Alice threw A in, the black hole
    was already past the Page time (so half its qubits had been radiated),
    then Bob can recover Alice's information by collecting **just a bit
    more than k** additional Hawking quanta. I(R : D) approaches 2k once
    |D| > k + (small overhead from the pre-radiation state).

    This is one of the deepest results in modern holography — it shows
    information falls out of black holes far faster than naively expected.

    Args:
        n_bh:     pre-existing black-hole qubit count (before Alice).
        k_alice:  number of qubits in Alice's message.
        rng:      numpy generator.
        base:     log base (e for nats, 2 for bits).

    Returns:
        dict with subsystem_sizes (|D| from 0 to n_bh+k_alice),
        mutual_info_R_D values, and the system parameters.
    """
    rng = rng or np.random.default_rng()
    n_total = n_bh + 2 * k_alice           # BH + Alice + Reference R

    # Build initial state:
    #   - Reference R (last k qubits): maximally entangled with Alice's
    #     register (the first k qubits of the BH+Alice block).
    #   - The remaining (n_bh - k_alice + k_alice) = n_bh qubits = random
    #     scrambled BH state (we just take them as |0...0⟩ here; the
    #     scrambling unitary below randomizes).
    # State vector layout: [k_alice qubits of A][n_bh - k_alice qubits BH][k_alice qubits R]
    d = 2 ** n_total
    psi = np.zeros(d, dtype=np.complex128)
    # Maximally entangled (|00⟩+|11⟩+...)/√(2^k) on (A, R).
    # Indices: A bits are positions [0, k_alice), R bits are positions
    # [n_bh + k_alice, n_total). The middle BH qubits are 0.
    norm = 1.0 / np.sqrt(2 ** k_alice)
    for a_val in range(2 ** k_alice):
        # Place |a⟩_A |0⟩_BH |a⟩_R
        idx = (a_val << (n_total - k_alice)) | a_val
        psi[idx] = norm

    # Apply Haar-random scrambling unitary to (A + BH) = first (k_alice + n_bh) qubits.
    n_scramble = k_alice + n_bh
    from .boson_sampling import random_haar_unitary
    U = random_haar_unitary(2 ** n_scramble, rng)
    # Embed into the full Hilbert space: act on the first n_scramble qubits.
    # Reshape psi into (2^n_scramble, 2^k_alice), apply U on the first axis.
    psi_mat = psi.reshape(2 ** n_scramble, 2 ** k_alice)
    psi_mat = U @ psi_mat
    psi = psi_mat.reshape(-1)

    # Now compute mutual information I(R : D) for various radiation slice sizes
    # |D|. D = the first |D| qubits of the BH+Alice block.
    R_qubits = list(range(n_total - k_alice, n_total))
    sizes = list(range(n_scramble + 1))
    mi_values = []
    for d_size in sizes:
        D_qubits = list(range(d_size))
        if not D_qubits:
            mi_values.append(0.0)
            continue
        mi = mutual_information(psi, n_total, R_qubits, D_qubits, base=base)
        mi_values.append(mi)

    return {
        "subsystem_sizes":  np.array(sizes),
        "mutual_info_R_D":  np.array(mi_values),
        "n_bh":             n_bh,
        "k_alice":          k_alice,
        "n_total":          n_total,
    }


def lanczos_coefficients(
    H: np.ndarray, O: np.ndarray, n_steps: int = 30, tol: float = 1e-12,
) -> dict:
    """Compute Lanczos coefficients b_n of an operator O under the Liouvillian
    L = i[H, ·] via the Lanczos recursion in operator space.

    Procedure (operator-space inner product ⟨A|B⟩ = Tr(A†B)/dim):
        1. O_0 = O / ||O||
        2. |O_{n+1}⟩ = L|O_n⟩ - a_n |O_n⟩ - b_n |O_{n-1}⟩    (with a_n = ⟨O_n|L|O_n⟩)
        3. b_{n+1} = ||O_{n+1}||, then normalize.

    The growth of b_n is a fundamental chaos diagnostic (Parker, Cao,
    Avdoshkin, Scaffidi, Altman 2019, "A Universal Operator Growth
    Hypothesis"):

      - **Chaotic** systems: b_n grows linearly, b_n ~ α n + β.
      - **Integrable** systems: b_n saturates or grows sub-linearly.

    SYK's b_n grow linearly with the maximum-possible slope, consistent
    with maximal chaos.

    Args:
        H:        Hermitian Hamiltonian matrix.
        O:        Initial operator (same dim as H).
        n_steps:  number of Lanczos steps to take.
        tol:      stop early if b becomes smaller than this.

    Returns:
        dict with:
            b_coeffs:   list of Lanczos coefficients
            a_coeffs:   list of diagonal coefficients
            n_steps:    actual number of steps taken
    """
    d = H.shape[0]
    inv_d = 1.0 / d

    def inner(A: np.ndarray, B: np.ndarray) -> complex:
        return inv_d * np.trace(A.conj().T @ B)

    def L_apply(A: np.ndarray) -> np.ndarray:
        return 1j * (H @ A - A @ H)

    O = np.asarray(O, dtype=np.complex128)
    n0 = float(np.real(inner(O, O))) ** 0.5
    if n0 < tol:
        return {"b_coeffs": [], "a_coeffs": [], "n_steps": 0}
    O_curr = O / n0
    O_prev = np.zeros_like(O)
    b_prev = 0.0

    a_list: list[float] = []
    b_list: list[float] = []

    for _ in range(n_steps):
        LO = L_apply(O_curr)
        a_n = float(np.real(inner(O_curr, LO)))
        a_list.append(a_n)
        residual = LO - a_n * O_curr - b_prev * O_prev
        b_next = float(np.real(inner(residual, residual))) ** 0.5
        if b_next < tol:
            break
        b_list.append(b_next)
        O_prev = O_curr
        O_curr = residual / b_next
        b_prev = b_next

    return {
        "a_coeffs": a_list,
        "b_coeffs": b_list,
        "n_steps":  len(b_list),
    }


def page_curve(
    n_qubits: int, n_samples: int = 50,
    rng: np.random.Generator | None = None,
    base: float = 2.0,
) -> dict:
    """Compute the Page curve for Haar-random states on n qubits.

    For each subsystem size k ∈ [0, n], sample n_samples Haar-random states
    and average the bipartite entropy S_A across them.

    The famous prediction (Don Page, 1993):
        ⟨S_A⟩ ≈ k · log(2)              for k ≤ n/2
        ⟨S_A⟩ ≈ (n - k) · log(2)        for k ≥ n/2
    The peak is at k = n/2 (with the entropy of maximally entangled half).

    Returns dict with:
        subsystem_sizes:    array of k values
        mean_entropy:       array of ⟨S_A⟩ (in `base` units; default bits)
        std_entropy:        sample-to-sample standard deviation
    """
    if n_qubits < 2:
        raise ValueError("Page curve needs n ≥ 2 qubits")
    rng = rng or np.random.default_rng()
    sizes = np.arange(n_qubits + 1)
    mean = np.zeros(len(sizes))
    std = np.zeros(len(sizes))
    for ki, k in enumerate(sizes):
        if k == 0 or k == n_qubits:
            mean[ki] = 0.0
            std[ki] = 0.0
            continue
        entropies = []
        for _ in range(n_samples):
            psi = haar_random_state(n_qubits, rng)
            A = list(range(k))           # take the first k qubits
            entropies.append(bipartite_entropy(psi, n_qubits, A, base=base))
        mean[ki] = np.mean(entropies)
        std[ki]  = np.std(entropies)
    return {
        "subsystem_sizes": sizes,
        "mean_entropy":    mean,
        "std_entropy":     std,
    }

"""Noise models — apply Kraus channels to simulate hardware errors.

The full state-vector formalism is pure-state only; for incoherent noise we
need either density matrices (memory-expensive) or a Monte Carlo trajectory
approach. We use trajectories: each "shot" through the circuit picks a
random Kraus operator at each noisy step, then we accumulate measurement
statistics over many shots.
"""

from __future__ import annotations

import numpy as np

from .gates import I2, X, Y, Z


# ---- Kraus operators for common channels ----

def bit_flip_kraus(p: float) -> list[np.ndarray]:
    """Bit-flip channel: with probability p, apply X."""
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * X]


def phase_flip_kraus(p: float) -> list[np.ndarray]:
    """Phase-flip channel: with probability p, apply Z."""
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]


def depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Depolarizing channel: with probability p, replace state with maximally mixed.

    Equivalent Kraus form: I with prob (1-p), X/Y/Z each with prob p/3.
    """
    return [
        np.sqrt(1 - p) * I2,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]


def amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    """Amplitude damping channel.

    Models energy relaxation: |1⟩ decays to |0⟩ with probability γ.
    On a qubit with characteristic decay time T1 over a gate time t,
    γ = 1 - exp(-t/T1).
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    K0 = np.array([[1.0, 0.0],
                   [0.0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0.0, np.sqrt(gamma)],
                   [0.0, 0.0]],              dtype=np.complex128)
    return [K0, K1]


def phase_damping_kraus(lam: float) -> list[np.ndarray]:
    """Phase damping channel.

    Models pure dephasing: loss of phase coherence without energy loss.
    For dephasing time T_phi over gate time t, λ = 1 - exp(-t/T_phi).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    K0 = np.array([[1.0, 0.0],
                   [0.0, np.sqrt(1 - lam)]], dtype=np.complex128)
    K1 = np.array([[0.0, 0.0],
                   [0.0, np.sqrt(lam)]],     dtype=np.complex128)
    return [K0, K1]


def thermal_relaxation_kraus(t1: float, t2: float, gate_time: float
                              ) -> list[np.ndarray]:
    """Composite T1/T2 thermal relaxation channel.

    Args:
        t1:        amplitude relaxation time (T1)
        t2:        coherence/dephasing time (T2). Must satisfy T2 ≤ 2 T1.
        gate_time: physical gate duration over which the channel acts.

    Returns a Kraus decomposition combining amplitude damping at rate
    γ = 1 - exp(-t/T1) with phase damping at rate λ such that the total
    T2 matches the input.
    """
    if t1 <= 0 or t2 <= 0 or gate_time < 0:
        raise ValueError("t1, t2, gate_time must be positive")
    if t2 > 2 * t1:
        raise ValueError("T2 must be ≤ 2·T1 (physical constraint)")
    gamma = 1.0 - np.exp(-gate_time / t1)
    # Total dephasing rate from T2 needs to account for the T1 contribution.
    # Using the relation 1/T2 = 1/(2 T1) + 1/T_phi, solve for T_phi.
    inv_t_phi = 1.0 / t2 - 1.0 / (2.0 * t1)
    if inv_t_phi < 0:                       # numerically possible if T2 = 2 T1
        inv_t_phi = 0.0
    lam = 1.0 - np.exp(-gate_time * inv_t_phi)
    # Compose the two channels as Kraus operators.
    # K_total[i,j] = K_amp[i] · K_phase[j] for each pair.
    ad = amplitude_damping_kraus(gamma)
    pd = phase_damping_kraus(lam)
    out = []
    for K_a in ad:
        for K_p in pd:
            out.append(K_p @ K_a)
    return out


# ---- Two-qubit noise channels -----------------------------------------------

def two_qubit_depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Two-qubit depolarizing channel — 16-operator Kraus set.

    With probability (1−p) the pair is untouched; with probability p a
    uniformly random non-identity two-qubit Pauli P_i⊗P_j is applied.

    Kraus operators:
        K_{I,I} = √(1−p) · I⊗I
        K_{i,j} = √(p/15) · P_i⊗P_j   for (i,j) ≠ (I,I)   [15 terms]

    Completeness: Σ_k K_k† K_k = I  ✓  (since Σ_{i,j} P†P = 16·I and
    the coefficients sum to (1−p) + 15·(p/15) = 1).

    Typical use: apply after a two-qubit gate to model CNOT/CZ error.
    For a CNOT with gate error rate ε (from randomized benchmarking),
    pass p = ε * 16/15 to match the standard depolarizing convention.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    paulis = [I2, X, Y, Z]
    ops: list[np.ndarray] = []
    for i, Pi in enumerate(paulis):
        for j, Pj in enumerate(paulis):
            coeff = np.sqrt(1.0 - p) if (i == 0 and j == 0) else np.sqrt(p / 15.0)
            ops.append(coeff * np.kron(Pi, Pj))
    return ops  # 16 operators of shape (4, 4)


def crosstalk_kraus(p_zz: float) -> list[np.ndarray]:
    """ZZ-crosstalk channel between a pair of qubits.

    Models always-on ZZ coupling (common in superconducting devices):
        ρ → (1−p) ρ + p (Z⊗Z) ρ (Z⊗Z)

    Two Kraus operators: [√(1−p) I⊗I,  √p Z⊗Z].
    """
    if not 0.0 <= p_zz <= 1.0:
        raise ValueError("p_zz must be in [0, 1]")
    II = np.kron(I2, I2)
    ZZ = np.kron(Z, Z)
    return [np.sqrt(1.0 - p_zz) * II, np.sqrt(p_zz) * ZZ]


# ---- Trajectory application ----

def apply_channel_trajectory(
    state: np.ndarray,
    kraus_ops: list[np.ndarray],
    target: int,
    n_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a single-qubit Kraus channel to `target` via the trajectory method.

    Each Kraus operator K_i has associated probability p_i = ⟨ψ|K_i^† K_i|ψ⟩.
    Sample one i with this distribution, then update state to K_i|ψ⟩ / √p_i.
    """
    tensor = state.reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, target, 0).copy()
    shape = moved.shape
    flat = moved.reshape(2, -1)

    # Compute per-operator probabilities.
    probs = []
    candidates = []
    for K in kraus_ops:
        new_flat = K @ flat
        amp = float(np.real(np.vdot(new_flat, new_flat)))
        probs.append(amp)
        candidates.append(new_flat)
    probs = np.array(probs)
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()

    choice = rng.choice(len(kraus_ops), p=probs)
    new_flat = candidates[choice]
    norm = np.sqrt(probs[choice] * sum(probs))  # rescale to unit norm
    if norm > 0:
        new_flat = new_flat / np.linalg.norm(new_flat)
    moved = new_flat.reshape(shape)
    out = np.moveaxis(moved, 0, target).reshape(state.shape)
    return out


def apply_2q_channel_trajectory(
    state: np.ndarray,
    kraus_ops: list[np.ndarray],
    targets: tuple[int, int],
    n_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a two-qubit Kraus channel to `targets` via trajectory sampling.

    Works with any list of 4×4 Kraus operators, including
    `two_qubit_depolarizing_kraus` and `crosstalk_kraus`.

    Parameters
    ----------
    state     : 2^n complex state vector.
    kraus_ops : list of (4, 4) Kraus matrices acting on the two-qubit subspace.
    targets   : (control_qubit, target_qubit) — order matters for asymmetric ops.
    n_qubits  : total number of qubits in the system.
    rng       : NumPy random generator.
    """
    q0, q1 = targets
    if q0 == q1:
        raise ValueError("targets must be distinct qubits")

    # Reshape into rank-n tensor, move both target axes to the front
    tensor = state.reshape((2,) * n_qubits)
    moved  = np.moveaxis(tensor, [q0, q1], [0, 1]).copy()
    shape  = moved.shape
    flat   = moved.reshape(4, -1)   # 4 = 2*2 two-qubit subspace

    # Evaluate each Kraus branch: probability = ||K|ψ⟩||²
    probs:      list[float]      = []
    candidates: list[np.ndarray] = []
    for K in kraus_ops:
        branch = K @ flat
        probs.append(float(np.real(np.vdot(branch, branch))))
        candidates.append(branch)

    probs_arr = np.clip(np.array(probs, dtype=np.float64), 0.0, None)
    total = probs_arr.sum()
    if total > 0:
        probs_arr /= total

    # Sample one Kraus branch, renormalise to unit state
    idx     = int(rng.choice(len(kraus_ops), p=probs_arr))
    chosen  = candidates[idx]
    norm    = np.linalg.norm(chosen)
    if norm > 0:
        chosen /= norm

    # Restore original axis ordering
    out = np.moveaxis(chosen.reshape(shape), [0, 1], [q0, q1])
    return out.reshape(state.shape)


def noisy_run(
    build_circuit,
    n_qubits: int,
    noise_op,
    noise_qubit: int,
    shots: int,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    """Run `build_circuit(qc)` many times, applying `noise_op` (Kraus ops) to
    `noise_qubit` after the build, then measure all qubits. Returns counts.
    """
    from .circuit import QuantumCircuit
    rng = rng or np.random.default_rng()
    counts: dict[str, int] = {}
    for _ in range(shots):
        qc = QuantumCircuit(n_qubits)
        build_circuit(qc)
        qc.state = apply_channel_trajectory(
            qc.state, noise_op, noise_qubit, n_qubits, rng
        )
        outcome = int(qc.measure_all(shots=1, rng=rng)[0])
        key = format(outcome, f"0{n_qubits}b")
        counts[key] = counts.get(key, 0) + 1
    return counts

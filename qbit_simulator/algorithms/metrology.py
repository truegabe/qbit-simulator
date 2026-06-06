"""Quantum metrology: Heisenberg-limited phase estimation.

The fundamental setting:
    A small unknown phase φ is imprinted on each of N qubits via
    e^{iφ Z/2}. We want to estimate φ as precisely as possible.

Two strategies:

    1. **Classical / Standard Quantum Limit (SQL)**:
       Prepare N independent |+⟩ states. Each measurement after the phase
       evolution gives information about φ with variance ~ 1/N (one shot
       per qubit × N qubits = N independent measurements). The estimate
       has uncertainty σ_φ ~ 1 / √N — the shot-noise limit.

    2. **Heisenberg-limited (HL)**:
       Prepare a GHZ state (|0...0⟩ + |1...1⟩)/√2. After the phase
       evolution, the |1...1⟩ component picks up phase e^{iNφ} (an
       N-fold enhancement). A single measurement of the parity operator
       gives uncertainty σ_φ ~ 1 / N — a √N improvement over SQL.

The √N speedup is the basis for atomic-clock improvements, gravimetry
sensitivity gains, and entanglement-enhanced imaging. We demonstrate
both bounds empirically by simulating many shots and comparing the
estimator variance to the theoretical predictions.
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit


def sql_phase_estimate(
    phi: float,
    n_qubits: int,
    n_shots: int,
    rng: np.random.Generator | None = None,
) -> dict:
    """Estimate φ using the standard-quantum-limit (independent |+⟩) strategy.

    For each of `n_qubits` independent |+⟩ states, apply Rz(φ), then measure
    in the X basis. The probability of outcome 0 is cos²(φ/2). Averaging
    over shots gives an estimate of φ.

    Args:
        phi:       true phase (small).
        n_qubits:  number of probe qubits.
        n_shots:   measurement shots per qubit.

    Returns:
        dict with phi_estimate, variance, etc.
    """
    rng = rng or np.random.default_rng()
    # Probability of measuring |+⟩ (which is the |0⟩ in X basis) is cos²(φ/2).
    p_plus = np.cos(phi / 2) ** 2
    # Total measurements: n_qubits × n_shots (each qubit measured n_shots times,
    # but really one shot per probe is normal — here we let users average).
    n_total = n_qubits * n_shots
    outcomes = rng.binomial(1, 1 - p_plus, size=n_total)   # 1 = measured |−⟩
    # Estimate p_minus = sin²(φ/2) = mean(outcomes); recover φ.
    p_minus_est = float(np.mean(outcomes))
    # Clip for numerical safety.
    p_minus_est = np.clip(p_minus_est, 0.0, 1.0)
    phi_est = 2 * np.arcsin(np.sqrt(p_minus_est))
    # Variance estimate: from binomial sampling.
    var_p = p_minus_est * (1 - p_minus_est) / max(n_total, 1)
    # Propagate to φ: dφ/dp = 1/(sin(φ)/2) ≈ 2/sin(φ).
    if abs(np.sin(phi_est)) > 1e-9:
        var_phi = var_p * (2.0 / np.sin(phi_est)) ** 2
    else:
        var_phi = np.inf
    return {
        "phi_estimate":  float(phi_est),
        "true_phi":      float(phi),
        "variance":      float(var_phi),
        "uncertainty":   float(np.sqrt(var_phi)),
        "n_total_shots": int(n_total),
        "strategy":      "SQL",
    }


def heisenberg_phase_estimate(
    phi: float,
    n_qubits: int,
    n_shots: int,
    rng: np.random.Generator | None = None,
) -> dict:
    """Estimate φ using an N-qubit GHZ state (Heisenberg-limited).

    Strategy:
        1. Prepare |GHZ⟩ = (|0...0⟩ + |1...1⟩) / √2.
        2. Apply Rz(φ) on each qubit → state has phase e^{iNφ} on |1...1⟩.
        3. Apply H on each qubit, measure parity (XOR of bits).

    The probability of even parity is cos²(Nφ/2) — an N-fold faster
    oscillation than the SQL case. Each shot gives O(1/N²) variance on
    the parity estimate, so σ_φ ~ 1/N.

    Args:
        phi:      true phase.
        n_qubits: N qubits in the GHZ state.
        n_shots:  total measurement shots (each gives one bit of info).
    """
    rng = rng or np.random.default_rng()
    # Probability of even parity after the protocol:
    #   P(even) = cos²(N φ / 2)
    p_even = np.cos(n_qubits * phi / 2) ** 2
    p_even = np.clip(p_even, 0.0, 1.0)
    # Simulate n_shots binary outcomes.
    outcomes_even = rng.binomial(1, p_even, size=n_shots)   # 1 = even parity
    p_even_est = float(np.mean(outcomes_even))
    p_even_est = np.clip(p_even_est, 0.0, 1.0)
    # Recover φ: phi_est = (2 / N) · arccos(√P_even). Ambiguity from arccos —
    # we take the principal branch.
    if p_even_est > 1.0 - 1e-12:
        phi_est = 0.0
    elif p_even_est < 1e-12:
        phi_est = np.pi / n_qubits
    else:
        phi_est = 2.0 / n_qubits * np.arccos(np.sqrt(p_even_est))
    # Variance: binomial variance on P_even propagated.
    var_p = p_even_est * (1 - p_even_est) / max(n_shots, 1)
    # dφ/dp = -1 / (N · sin(Nφ/2) · cos(Nφ/2)) = -2/(N · sin(Nφ)).
    if abs(np.sin(n_qubits * phi_est)) > 1e-9:
        var_phi = var_p * (2.0 / (n_qubits * np.sin(n_qubits * phi_est))) ** 2
    else:
        var_phi = np.inf
    return {
        "phi_estimate":  float(phi_est),
        "true_phi":      float(phi),
        "variance":      float(var_phi),
        "uncertainty":   float(np.sqrt(var_phi)),
        "n_total_shots": int(n_shots),
        "strategy":      "Heisenberg",
    }


def build_ghz_state(n: int) -> np.ndarray:
    """Return the |GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2 state vector on n qubits.

    Useful for protocol simulations beyond the analytic SQL/Heisenberg
    estimators (e.g. with noise).
    """
    qc = QuantumCircuit(n).h(0)
    for q in range(n - 1):
        qc.cnot(q, q + 1)
    return qc.state


def metrology_comparison(
    phi: float,
    n_qubits: int,
    n_shots: int,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run both strategies side by side. Returns ratio of uncertainties."""
    rng = rng or np.random.default_rng()
    sql = sql_phase_estimate(phi, n_qubits, n_shots // n_qubits, rng=rng)
    hl  = heisenberg_phase_estimate(phi, n_qubits, n_shots, rng=rng)
    return {
        "sql":           sql,
        "heisenberg":    hl,
        "speedup_ratio": float(sql["uncertainty"] / hl["uncertainty"])
                          if hl["uncertainty"] > 0 else float("inf"),
        "theoretical_ratio": float(np.sqrt(n_qubits)),
    }

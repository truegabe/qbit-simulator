"""NISQ error mitigation techniques.

Implements:
    - Zero-noise extrapolation (ZNE): run a noisy circuit at multiple
      "stretched" noise levels and polynomially extrapolate the observable
      back to zero noise.
    - Pauli twirling: convert coherent noise to incoherent Pauli noise
      by randomly conjugating each gate with Pauli operators (canceled
      ideally, leaving Pauli error otherwise).

These are post-hoc techniques: they don't reduce noise, they extract more
information from noisy data. Standard tooling for current quantum hardware.

ZNE in particular is widely used in production: IBM's quantum systems
ship with built-in ZNE for many algorithms. The extrapolation only needs
to be polynomial in the noise level for the standard noise models.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def zero_noise_extrapolation(
    noisy_run_fn: Callable[[float], float],
    stretch_factors: list[float] | None = None,
    fit_order: int = 2,
) -> dict:
    """Estimate an observable at zero noise via polynomial extrapolation.

    Args:
        noisy_run_fn: callable that takes a stretch factor (≥ 1) and returns
                      the observable value at noise level scale·base_noise.
                      A stretch factor of 1.0 means the natural noise level;
                      higher values amplify noise (e.g. by inserting redundant
                      gate pairs).
        stretch_factors: list of stretch values to sample (default: [1, 2, 3, 4]).
        fit_order: polynomial degree for extrapolation (1=linear, 2=quadratic).

    Returns:
        dict with:
            extrapolated:   estimated observable at λ → 0
            samples:        list of (stretch, observable) pairs
            polynomial:     numpy polynomial fit coefficients (highest order first)
    """
    if stretch_factors is None:
        stretch_factors = [1.0, 2.0, 3.0, 4.0]
    samples = []
    for s in stretch_factors:
        if s < 1.0:
            raise ValueError("stretch factors must be ≥ 1.0")
        obs = noisy_run_fn(s)
        samples.append((float(s), float(obs)))
    xs = np.array([p[0] for p in samples])
    ys = np.array([p[1] for p in samples])
    if len(xs) <= fit_order:
        fit_order = max(1, len(xs) - 1)
    coeffs = np.polyfit(xs, ys, fit_order)
    # Evaluate the polynomial at x = 0.
    extrapolated = float(np.polyval(coeffs, 0.0))
    return {
        "extrapolated": extrapolated,
        "samples":      samples,
        "polynomial":   coeffs,
        "fit_order":    fit_order,
    }


def pauli_twirl_channel(
    state: np.ndarray,
    n_qubits: int,
    target_qubits: list[int],
    apply_gate_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a gate with Pauli twirling on `target_qubits`.

    For each target qubit, sample a random Pauli P_q from {I, X, Y, Z}
    before the gate and apply P_q' = (gate-conjugated P_q) after. In the
    noise-free limit, the P_q's cancel and the original gate is recovered.
    In the noisy case, the average over many twirl choices converts
    coherent gate errors into incoherent (Pauli) errors, which are often
    easier to handle.

    For simplicity this implementation does the random Pauli sampling but
    leaves the gate-conjugation to the caller (just inverts each Pauli
    after the gate). For Clifford `gate`, conjugation of Pauli stays in
    the Pauli group.
    """
    from ..gates import I2, X, Y, Z
    pauli_list = [I2, X, Y, Z]

    # Sample a Pauli for each target qubit.
    paulis_pre = [pauli_list[int(rng.integers(0, 4))] for _ in target_qubits]
    # Apply pre-twirl Paulis to the state.
    for q, P in zip(target_qubits, paulis_pre):
        state = _apply_single_qubit_op(state, P, q, n_qubits)
    # Apply the gate.
    state = apply_gate_fn(state)
    # Apply inverse Paulis (Pauli operators are self-inverse).
    for q, P in zip(target_qubits, paulis_pre):
        state = _apply_single_qubit_op(state, P, q, n_qubits)
    return state


# ----------------------------------------------------------------------------
# Measurement error mitigation
# ----------------------------------------------------------------------------

def build_readout_calibration_matrix(
    n_qubits: int,
    readout_kraus: Callable[[int], list[np.ndarray]] | None = None,
    p_flip: float = 0.05,
) -> np.ndarray:
    """Build the readout-error matrix M[i, j] = P(measure i | prepared j).

    For a simple symmetric bit-flip readout error, M is built from per-qubit
    flip probability `p_flip`. For each input basis state j, simulate
    preparation and the noisy measurement, count outcomes.

    Args:
        n_qubits: number of qubits.
        readout_kraus: optional callable returning Kraus operators per qubit
                       (advanced; default is symmetric p_flip on each qubit).
        p_flip: per-qubit bit-flip probability for the default model.

    Returns:
        2^n × 2^n stochastic matrix M.
    """
    dim = 2 ** n_qubits
    M = np.zeros((dim, dim), dtype=np.float64)
    # For symmetric bit-flip readout per qubit:
    # P(measure i | prepared j) = prod over qubits of P(bit_i^q | bit_j^q)
    # where P(b' | b) = (1 - p_flip) if b' == b, else p_flip.
    for j in range(dim):
        for i in range(dim):
            p = 1.0
            for q in range(n_qubits):
                bi = (i >> q) & 1
                bj = (j >> q) & 1
                p *= (1 - p_flip) if bi == bj else p_flip
            M[i, j] = p
    return M


def measurement_mitigation_invert(
    M: np.ndarray, noisy_counts: dict[str, int] | np.ndarray,
) -> np.ndarray:
    """Apply the calibration inverse to recover the underlying distribution.

    Args:
        M:            readout-error matrix from `build_readout_calibration_matrix`.
        noisy_counts: dict of bitstring → count, or a length-2^N array.

    Returns:
        Vector of estimated true probabilities (may have small negatives;
        users may want to clip + renormalize).
    """
    dim = M.shape[0]
    if isinstance(noisy_counts, dict):
        n_qubits = int(np.log2(dim))
        p_noisy = np.zeros(dim, dtype=np.float64)
        total = sum(noisy_counts.values())
        for bitstr, n in noisy_counts.items():
            idx = int(bitstr, 2)
            p_noisy[idx] = n / total
    else:
        p_noisy = np.asarray(noisy_counts, dtype=np.float64)
        if p_noisy.sum() > 1e-9:
            p_noisy = p_noisy / p_noisy.sum()
    try:
        p_true = np.linalg.solve(M, p_noisy)
    except np.linalg.LinAlgError:
        p_true = np.linalg.lstsq(M, p_noisy, rcond=None)[0]
    return p_true


def _apply_single_qubit_op(state: np.ndarray, U: np.ndarray,
                            target: int, n: int) -> np.ndarray:
    """Helper: apply a 2×2 unitary to qubit `target` of an n-qubit state."""
    tensor = state.reshape((2,) * n)
    tensor = np.moveaxis(tensor, target, 0)
    shape = tensor.shape
    flat = tensor.reshape(2, -1)
    flat = U @ flat
    tensor = flat.reshape(shape)
    tensor = np.moveaxis(tensor, 0, target)
    return tensor.reshape(-1)

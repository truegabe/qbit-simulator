"""Error mitigation tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.error_mitigation import (
    zero_noise_extrapolation, pauli_twirl_channel,
)


# ---- zero-noise extrapolation ----

def test_zne_recovers_zero_noise_value_linear():
    """If the true observable is E(λ) = E_0 + α·λ, ZNE should recover E_0."""
    E_0 = -1.137
    alpha = 0.05
    def noisy(stretch):
        return E_0 + alpha * stretch
    result = zero_noise_extrapolation(noisy, stretch_factors=[1, 2, 3, 4],
                                       fit_order=1)
    assert abs(result["extrapolated"] - E_0) < 1e-6


def test_zne_recovers_zero_noise_value_quadratic():
    """E(λ) = E_0 + α·λ + β·λ². Quadratic fit recovers E_0."""
    E_0 = -7.882
    alpha, beta = 0.02, 0.005
    def noisy(stretch):
        return E_0 + alpha * stretch + beta * stretch ** 2
    result = zero_noise_extrapolation(noisy, stretch_factors=[1, 2, 3, 4, 5],
                                       fit_order=2)
    assert abs(result["extrapolated"] - E_0) < 1e-6


def test_zne_rejects_stretch_below_one():
    def noisy(s):
        return 0.0
    with pytest.raises(ValueError):
        zero_noise_extrapolation(noisy, stretch_factors=[0.5, 1, 2])


def test_zne_with_default_factors():
    def noisy(stretch):
        return 1.0 + 0.1 * stretch
    result = zero_noise_extrapolation(noisy, fit_order=1)
    assert abs(result["extrapolated"] - 1.0) < 1e-9
    assert len(result["samples"]) == 4


# ---- Measurement error mitigation ----

def test_readout_matrix_is_stochastic():
    """Columns of M should sum to 1 (each is a probability distribution)."""
    from qbit_simulator.algorithms.error_mitigation import (
        build_readout_calibration_matrix,
    )
    M = build_readout_calibration_matrix(3, p_flip=0.1)
    assert np.allclose(M.sum(axis=0), 1.0, atol=1e-12)


def test_readout_matrix_p_flip_zero_is_identity():
    from qbit_simulator.algorithms.error_mitigation import (
        build_readout_calibration_matrix,
    )
    M = build_readout_calibration_matrix(2, p_flip=0.0)
    assert np.allclose(M, np.eye(4), atol=1e-12)


def test_measurement_mitigation_recovers_true_dist():
    """Apply known readout noise to a true distribution, then mitigate;
    the mitigated distribution should match the true one."""
    from qbit_simulator.algorithms.error_mitigation import (
        build_readout_calibration_matrix, measurement_mitigation_invert,
    )
    n_qubits = 2
    p_true = np.array([0.5, 0.3, 0.15, 0.05])
    M = build_readout_calibration_matrix(n_qubits, p_flip=0.05)
    p_noisy = M @ p_true
    p_recovered = measurement_mitigation_invert(M, p_noisy)
    assert np.allclose(p_recovered, p_true, atol=1e-8)


def test_measurement_mitigation_with_dict_input():
    """Accept dict of bitstring → count as input."""
    from qbit_simulator.algorithms.error_mitigation import (
        build_readout_calibration_matrix, measurement_mitigation_invert,
    )
    M = build_readout_calibration_matrix(2, p_flip=0.05)
    counts = {"00": 500, "11": 500}
    p_rec = measurement_mitigation_invert(M, counts)
    assert p_rec.shape == (4,)


# ---- Pauli twirling ----

def test_pauli_twirl_with_identity_gate():
    """Twirling around the identity gate leaves the state unchanged.
    (The pre-Pauli is cancelled by its inverse post-Pauli.)"""
    state = np.array([0.6, 0.8], dtype=np.complex128)
    rng = np.random.default_rng(0)
    for _ in range(10):
        out = pauli_twirl_channel(
            state, n_qubits=1, target_qubits=[0],
            apply_gate_fn=lambda s: s,   # identity gate
            rng=rng,
        )
        assert np.allclose(out, state, atol=1e-10)


def test_pauli_twirl_preserves_norm():
    rng = np.random.default_rng(0)
    state = np.zeros(4, dtype=np.complex128); state[0] = 1
    for _ in range(5):
        out = pauli_twirl_channel(
            state, n_qubits=2, target_qubits=[0, 1],
            apply_gate_fn=lambda s: s,
            rng=rng,
        )
        assert abs(np.linalg.norm(out) - 1.0) < 1e-10

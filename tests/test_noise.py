import numpy as np
import pytest

from qbit_simulator.noise import (
    bit_flip_kraus, phase_flip_kraus, depolarizing_kraus,
    apply_channel_trajectory, noisy_run,
)


def test_kraus_completeness_bit_flip():
    # Sum K_i^dag K_i should equal I.
    ops = bit_flip_kraus(0.3)
    s = sum(K.conj().T @ K for K in ops)
    assert np.allclose(s, np.eye(2))


def test_kraus_completeness_depolarizing():
    ops = depolarizing_kraus(0.5)
    s = sum(K.conj().T @ K for K in ops)
    assert np.allclose(s, np.eye(2))


def test_bit_flip_at_p1_flips_state():
    # With p=1, bit-flip channel should always flip |0> to |1> (up to global phase).
    rng = np.random.default_rng(0)
    state = np.array([1, 0], dtype=np.complex128)
    new_state = apply_channel_trajectory(state, bit_flip_kraus(1.0), 0, 1, rng)
    assert abs(new_state[1]) == pytest.approx(1.0)


def test_noisy_bell_pair_with_low_noise_mostly_correlated():
    # Build a Bell pair, then apply a small phase flip; correlation should mostly survive.
    rng = np.random.default_rng(0)
    counts = noisy_run(
        build_circuit=lambda qc: qc.h(0).cnot(0, 1),
        n_qubits=2,
        noise_op=phase_flip_kraus(0.05),
        noise_qubit=0,
        shots=500,
        rng=rng,
    )
    correlated = counts.get("00", 0) + counts.get("11", 0)
    assert correlated > 450  # phase flip doesn't affect Z-basis measurement

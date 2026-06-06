import numpy as np
import pytest

from qbit_simulator.density import DensityMatrix
from qbit_simulator.gates import H, X, Z, CNOT
from qbit_simulator.noise import bit_flip_kraus, depolarizing_kraus


def test_pure_state_has_purity_one():
    state = np.array([1, 0], dtype=np.complex128)
    rho = DensityMatrix.from_state(state)
    assert rho.purity() == pytest.approx(1.0)


def test_maximally_mixed_purity():
    rho = DensityMatrix.maximally_mixed(2)
    assert rho.purity() == pytest.approx(0.25)  # 1/4


def test_apply_unitary_h_on_zero():
    rho = DensityMatrix.from_state(np.array([1, 0], dtype=np.complex128))
    rho.apply_unitary(H, [0])
    probs = rho.probabilities()
    assert probs[0] == pytest.approx(0.5)
    assert probs[1] == pytest.approx(0.5)


def test_bit_flip_channel_full_flip():
    rho = DensityMatrix.from_state(np.array([1, 0], dtype=np.complex128))
    rho.apply_kraus(bit_flip_kraus(1.0), 0)
    assert rho.probabilities()[1] == pytest.approx(1.0)


def test_depolarizing_to_max_mixed():
    rho = DensityMatrix.from_state(np.array([1, 0], dtype=np.complex128))
    rho.apply_kraus(depolarizing_kraus(0.75), 0)  # full depolarization
    # Should be close to maximally mixed.
    assert np.allclose(rho.rho, np.eye(2) / 2, atol=1e-12)


def test_partial_trace_of_bell_pair_is_mixed():
    # Bell state.
    bell = np.zeros(4, dtype=np.complex128)
    bell[0] = bell[3] = 1 / np.sqrt(2)
    rho = DensityMatrix.from_state(bell)
    reduced = rho.partial_trace([1])
    # Reduced should be I/2 (maximally mixed) — characteristic signature of entanglement.
    assert np.allclose(reduced.rho, np.eye(2) / 2, atol=1e-12)
    assert reduced.von_neumann_entropy() == pytest.approx(1.0)

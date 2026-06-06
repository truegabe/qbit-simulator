import numpy as np
import pytest

from qbit_simulator import QuantumCircuit
from qbit_simulator.algorithms import qft, qft_matrix, apply_qft


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_matches_dft_on_zero_state(n):
    qc = qft(n)
    expected = qft_matrix(n) @ np.eye(2**n)[:, 0]
    assert np.allclose(qc.state, expected)


@pytest.mark.parametrize("n,k", [(2, 1), (3, 5), (4, 9)])
def test_qft_on_basis_states(n, k):
    qc = QuantumCircuit(n)
    # Prepare |k> by flipping bits.
    for bit in range(n):
        if (k >> (n - 1 - bit)) & 1:
            qc.x(bit)
    apply_qft(qc)
    expected = qft_matrix(n) @ np.eye(2**n)[:, k]
    assert np.allclose(qc.state, expected)


def test_qft_is_unitary_via_inverse():
    n = 3
    qc = qft(n)
    # Applying QFT to |0...0> should give the uniform superposition.
    expected = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    assert np.allclose(qc.state, expected)

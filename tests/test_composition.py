import os
import tempfile

import numpy as np
import pytest

from qbit_simulator import QuantumCircuit


def test_copy_is_independent():
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    qc2 = qc.copy()
    qc.x(0)
    assert not np.allclose(qc.state, qc2.state)


def test_save_load_roundtrip(tmp_path):
    qc = QuantumCircuit(3).h(0).cnot(0, 1).x(2)
    path = tmp_path / "circuit.npz"
    qc.save(path)
    qc2 = QuantumCircuit.load(path)
    assert qc2.n == qc.n
    assert np.allclose(qc2.state, qc.state)
    assert qc2.history == qc.history

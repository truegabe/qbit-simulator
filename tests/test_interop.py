"""Qiskit and Cirq object-level interop tests.

These tests use `pytest.importorskip` so they're cleanly skipped on systems
where Qiskit / Cirq aren't installed. On a system with both libraries,
the tests verify round-trip equivalence.
"""

import numpy as np
import pytest

from qbit_simulator import QuantumCircuit
from qbit_simulator.interop import (
    to_qiskit, from_qiskit, to_cirq, from_cirq,
)


# ---- Qiskit interop ----

def test_qiskit_to_qiskit_returns_qiskit_object():
    qiskit = pytest.importorskip("qiskit")
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    out = to_qiskit(qc)
    assert isinstance(out, qiskit.QuantumCircuit)
    assert out.num_qubits == 2


def test_qiskit_round_trip_bell_pair():
    """Bell pair: ours → qiskit → ours, state matches."""
    qiskit = pytest.importorskip("qiskit")
    qc1 = QuantumCircuit(2).h(0).cnot(0, 1)
    qiskit_qc = to_qiskit(qc1)
    qc2 = from_qiskit(qiskit_qc)
    assert np.allclose(qc1.state, qc2.state, atol=1e-9)


def test_qiskit_round_trip_parameterized():
    qiskit = pytest.importorskip("qiskit")
    qc1 = QuantumCircuit(3).rx(0.5, 0).cnot(0, 1).ry(1.2, 2).cz(1, 2)
    qiskit_qc = to_qiskit(qc1)
    qc2 = from_qiskit(qiskit_qc)
    assert np.allclose(qc1.state, qc2.state, atol=1e-9)


def test_qiskit_import_qiskit_native_circuit():
    """Build a circuit directly in Qiskit, import to ours."""
    qiskit = pytest.importorskip("qiskit")
    qc_qk = qiskit.QuantumCircuit(2)
    qc_qk.h(0)
    qc_qk.cx(0, 1)
    qc_ours = from_qiskit(qc_qk)
    # Should be a Bell pair.
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[3] = 1 / np.sqrt(2)
    assert np.allclose(qc_ours.state, expected, atol=1e-9)


def test_qiskit_missing_raises_clear_error(monkeypatch):
    """Without qiskit, to_qiskit / from_qiskit raise ImportError."""
    import qbit_simulator.interop as interop
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "qiskit":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install qiskit"):
        to_qiskit(QuantumCircuit(2).h(0))


# ---- Cirq interop ----

def test_cirq_to_cirq_returns_cirq_object():
    cirq = pytest.importorskip("cirq")
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    out = to_cirq(qc)
    assert isinstance(out, cirq.Circuit)


def test_cirq_round_trip_bell_pair():
    cirq = pytest.importorskip("cirq")
    qc1 = QuantumCircuit(2).h(0).cnot(0, 1)
    cirq_c = to_cirq(qc1)
    qc2 = from_cirq(cirq_c)
    assert np.allclose(qc1.state, qc2.state, atol=1e-9)


def test_cirq_round_trip_3qubit_with_rotations():
    cirq = pytest.importorskip("cirq")
    qc1 = QuantumCircuit(3).h(0).cnot(0, 1).rx(0.5, 2).cz(0, 2)
    cirq_c = to_cirq(qc1)
    qc2 = from_cirq(cirq_c)
    assert np.allclose(qc1.state, qc2.state, atol=1e-9)


def test_cirq_import_cirq_native_circuit():
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(2)
    cirq_qc = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1])])
    qc_ours = from_cirq(cirq_qc)
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[3] = 1 / np.sqrt(2)
    assert np.allclose(qc_ours.state, expected, atol=1e-9)


def test_cirq_missing_raises_clear_error(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "cirq":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install cirq"):
        to_cirq(QuantumCircuit(2).h(0))

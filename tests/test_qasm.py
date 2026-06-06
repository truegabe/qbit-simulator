"""OpenQASM 2.0 import/export tests."""

import numpy as np
import pytest

from qbit_simulator import QuantumCircuit
from qbit_simulator.qasm import to_qasm, from_qasm


# ---- export: QuantumCircuit -> QASM ----

def test_simple_circuit_to_qasm():
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    out = to_qasm(qc)
    assert "OPENQASM 2.0;" in out
    assert "qreg q[2];" in out
    assert "h q[0];" in out
    assert "cx q[0], q[1];" in out


def test_parameterized_gates_to_qasm():
    qc = QuantumCircuit(3).rx(0.5, 0).ry(1.2, 1).rz(np.pi / 4, 2)
    out = to_qasm(qc)
    assert "rx(0.5)" in out
    assert "ry(1.2)" in out
    assert "rz" in out


def test_cp_and_cz_export():
    qc = QuantumCircuit(2).cp(0.3, 0, 1).cz(0, 1)
    out = to_qasm(qc)
    assert "cu1(0.3)" in out
    assert "cz q[0], q[1];" in out


# ---- import: QASM -> QuantumCircuit ----

def test_minimal_qasm_import():
    src = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    h q[0];
    cx q[0], q[1];
    """
    qc = from_qasm(src)
    assert qc.n == 2
    assert len(qc.history) == 2
    # State should be the Bell pair.
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2); expected[3] = 1 / np.sqrt(2)
    assert np.allclose(qc.state, expected, atol=1e-10)


def test_qasm_import_with_parameters():
    src = """
    OPENQASM 2.0;
    qreg q[1];
    rx(pi/2) q[0];
    """
    qc = from_qasm(src)
    assert qc.n == 1
    # After Rx(π/2) on |0⟩, the state should be cos(π/4)|0⟩ - i sin(π/4)|1⟩
    a, b = qc.state
    assert abs(a - np.cos(np.pi / 4)) < 1e-9
    assert abs(b - (-1j * np.sin(np.pi / 4))) < 1e-9


def test_qasm_import_handles_comments_and_includes():
    src = """
    // top comment
    OPENQASM 2.0;
    include "qelib1.inc";
    /* block
       comment */
    qreg q[3];
    creg c[3];
    h q[0];
    // mid comment
    barrier q;
    x q[2];
    """
    qc = from_qasm(src)
    assert qc.n == 3
    assert len(qc.history) == 2  # h, x (barrier is ignored)


# ---- round trip: export then re-import ----

def test_round_trip_bell_pair():
    qc1 = QuantumCircuit(2).h(0).cnot(0, 1)
    out = to_qasm(qc1)
    qc2 = from_qasm(out)
    assert np.allclose(qc1.state, qc2.state, atol=1e-10)


def test_round_trip_parameterized():
    qc1 = QuantumCircuit(3).rx(0.7, 0).cnot(0, 1).ry(1.5, 2).cz(1, 2)
    out = to_qasm(qc1)
    qc2 = from_qasm(out)
    assert np.allclose(qc1.state, qc2.state, atol=1e-9)


def test_round_trip_grover_4_qubit():
    """A multi-gate circuit round-trips bit-exactly."""
    qc1 = QuantumCircuit(4)
    for q in range(4):
        qc1.h(q)
    qc1.cz(0, 3)
    for q in range(4):
        qc1.h(q)
        qc1.x(q)
    qc1.cz(0, 1)
    qc1.cnot(2, 3)
    out = to_qasm(qc1)
    qc2 = from_qasm(out)
    assert np.allclose(qc1.state, qc2.state, atol=1e-9)


# ---- error handling ----

def test_qasm_unknown_gate_raises():
    src = """
    OPENQASM 2.0;
    qreg q[1];
    bogus q[0];
    """
    with pytest.raises(ValueError):
        from_qasm(src)


def test_qasm_no_qreg_raises():
    src = """
    OPENQASM 2.0;
    h q[0];
    """
    with pytest.raises(ValueError):
        from_qasm(src)


# ---- OpenQASM 3.0 ----

def test_qasm3_export_header():
    """Exporting with version='3.0' produces stdgates.inc + qubit[N] q;"""
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    out = to_qasm(qc, version="3.0")
    assert "OPENQASM 3.0" in out
    assert 'include "stdgates.inc";' in out
    assert "qubit[2] q;" in out
    assert "bit[2] c;" in out


def test_qasm3_parses_qubit_declaration():
    """qubit[N] q; syntax should be accepted."""
    src = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    h q[0];
    cx q[0], q[1];
    """
    qc = from_qasm(src)
    assert qc.n == 3
    assert len(qc.history) == 2


def test_qasm3_parses_bit_declaration():
    src = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    """
    qc = from_qasm(src)
    assert qc.n == 2


def test_qasm3_measure_syntax():
    """c[i] = measure q[i] should be accepted as 3.0 measurement form."""
    src = """
    OPENQASM 3.0;
    qubit[2] q;
    bit[2] c;
    h q[0];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """
    qc = from_qasm(src)
    assert qc.n == 2


def test_qasm3_round_trip_bell_pair():
    """Export → re-import → state should match."""
    qc1 = QuantumCircuit(2).h(0).cnot(0, 1)
    src = to_qasm(qc1, version="3.0")
    qc2 = from_qasm(src)
    assert np.allclose(qc1.state, qc2.state, atol=1e-10)


def test_qasm3_supports_phase_keyword():
    """`phase(theta)` is the 3.0 name for the U1 / p gate."""
    src = """
    OPENQASM 3.0;
    qubit[1] q;
    phase(0.5) q[0];
    """
    qc = from_qasm(src)
    assert qc.n == 1
    # Same state as p(0.5) on |0⟩: just a global phase, so |0⟩ remains |0⟩.
    expected = np.array([1, 0], dtype=np.complex128)
    assert np.allclose(qc.state, expected, atol=1e-10)


def test_qasm3_tau_constant():
    """tau = 2π should be supported in parameter expressions."""
    src = """
    OPENQASM 3.0;
    qubit[1] q;
    rx(tau/4) q[0];
    """
    qc = from_qasm(src)
    # Rx(τ/4) = Rx(π/2) — known 1-qubit state.
    expected_phase = np.exp(-1j * np.pi / 4)
    # |0⟩ → cos(π/4)|0⟩ - i sin(π/4)|1⟩
    expected = np.array([np.cos(np.pi / 4), -1j * np.sin(np.pi / 4)],
                        dtype=np.complex128)
    assert np.allclose(qc.state, expected, atol=1e-9)


def test_qasm3_cphase_alias():
    """In OpenQASM 3.0 'cphase' is an alias for cp."""
    src = """
    OPENQASM 3.0;
    qubit[2] q;
    h q[0];
    cphase(pi/4) q[0], q[1];
    """
    qc = from_qasm(src)
    assert qc.n == 2
    assert len(qc.history) == 2


# ---- can ingest an externally-styled QASM ----

def test_ingest_qiskit_style_grover_oracle():
    """A QASM string in the style Qiskit might emit -- check we can parse it."""
    src = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    h q[0]; h q[1]; h q[2];
    cz q[0], q[2];
    h q[0]; h q[1]; h q[2];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    """
    qc = from_qasm(src)
    assert qc.n == 3

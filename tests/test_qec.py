"""Tests for the quantum error correcting codes."""

import numpy as np
import pytest

from qbit_simulator.stabilizer import StabilizerState
from qbit_simulator.qec import (
    three_qubit_repetition_code,
    five_qubit_perfect_code,
    steane_code,
    shor_nine_qubit_code,
    verify_codespace,
)


# ---- pauli_expectation primitive (needed by every QEC test) ----

def test_pauli_expectation_on_zero_state():
    st = StabilizerState(3)
    assert st.pauli_expectation("ZII") == 1
    assert st.pauli_expectation("IZI") == 1
    assert st.pauli_expectation("IIZ") == 1
    assert st.pauli_expectation("XII") == 0
    assert st.pauli_expectation("YII") == 0
    # Product of stabilizers is also +1.
    assert st.pauli_expectation("ZZZ") == 1


def test_pauli_expectation_on_bell():
    st = StabilizerState(2).h(0).cnot(0, 1)
    assert st.pauli_expectation("XX") == 1
    assert st.pauli_expectation("ZZ") == 1
    assert st.pauli_expectation("YY") == -1
    assert st.pauli_expectation("XI") == 0
    assert st.pauli_expectation("IX") == 0


def test_pauli_expectation_on_ghz_three():
    st = StabilizerState(3).h(0).cnot(0, 1).cnot(1, 2)
    assert st.pauli_expectation("XXX") == 1
    assert st.pauli_expectation("YYX") == -1
    assert st.pauli_expectation("YXY") == -1
    assert st.pauli_expectation("XYY") == -1


# ---- 3-qubit repetition code ----

def test_repetition_zero_state_satisfies_stabs():
    """|0⟩^⊗3 already satisfies both Z-stabilizers of the repetition code."""
    st = StabilizerState(3)
    expectations = verify_codespace(st, three_qubit_repetition_code, [0, 1, 2])
    assert all(v == 1 for v in expectations.values())


def test_repetition_detects_single_bit_flip():
    """Injecting X on one qubit must flip exactly the stabilizer that detects it."""
    for err_qubit in (0, 1, 2):
        st = StabilizerState(3)
        st.x(err_qubit)
        e = verify_codespace(st, three_qubit_repetition_code, [0, 1, 2])
        # Each Z stabilizer (ZZI or IZZ) anticommutes with X iff X is on one
        # of its support qubits.
        # ZZI detects X on qubit 0 or 1; IZZ detects X on qubit 1 or 2.
        flipped = [s for s, v in e.items() if v == -1]
        if err_qubit == 0:
            assert flipped == ["ZZI"]
        elif err_qubit == 1:
            assert set(flipped) == {"ZZI", "IZZ"}
        else:
            assert flipped == ["IZZ"]


# ---- 5-qubit perfect code [[5,1,3]] — structure only ----

def test_five_qubit_stabilizer_structure():
    """The four stabilizer generators of the 5-qubit perfect code."""
    assert five_qubit_perfect_code.n_data == 5
    assert five_qubit_perfect_code.distance == 3
    assert five_qubit_perfect_code.stabilizers == [
        "XZZXI", "IXZZX", "XIXZZ", "ZXIXZ",
    ]


def test_five_qubit_encoded_state_in_codespace():
    """After encoding |0_L⟩ via stabilizer projection, the state is a +1
    eigenstate of every stabilizer of the 5-qubit code."""
    from qbit_simulator.stabilizer import StabilizerState
    st = StabilizerState(6)              # 5 data + 1 ancilla
    five_qubit_perfect_code.encode(st, 0, 1, 2, 3, 4)
    for s in five_qubit_perfect_code.stabilizers:
        # Pad with I on the ancilla qubit.
        full = s + "I"
        assert st.pauli_expectation(full) == 1, (
            f"stabilizer {s} not +1 on encoded state"
        )


@pytest.mark.parametrize("err_qubit", range(5))
@pytest.mark.parametrize("err_op",   ["X", "Y", "Z"])
def test_five_qubit_detects_every_single_qubit_error(err_qubit, err_op):
    """The [[5,1,3]] code detects every single-qubit Pauli error: at least
    one stabilizer must report -1 after the error is injected."""
    from qbit_simulator.stabilizer import StabilizerState
    st = StabilizerState(6)
    five_qubit_perfect_code.encode(st, 0, 1, 2, 3, 4)
    {"X": st.x, "Y": st.y, "Z": st.z}[err_op](err_qubit)
    violations = [
        s for s in five_qubit_perfect_code.stabilizers
        if st.pauli_expectation(s + "I") == -1
    ]
    assert len(violations) >= 1, (
        f"single {err_op} on qubit {err_qubit} not detected by any stabilizer"
    )


# ---- Steane code ----

def test_steane_stabilizers_correct():
    """X- and Z-stabilizers should be 7-char strings of X/I and Z/I."""
    for s in steane_code.x_stabilizers:
        assert len(s) == 7
        assert set(s).issubset({"X", "I"})
    for s in steane_code.z_stabilizers:
        assert len(s) == 7
        assert set(s).issubset({"Z", "I"})


def test_steane_encoded_zero_in_codespace():
    """All 6 stabilizers must report +1 on the encoded |0_L⟩."""
    st = StabilizerState(7)
    steane_code.encode(st, *range(7))
    bad = []
    for s in steane_code.x_stabilizers + steane_code.z_stabilizers:
        v = st.pauli_expectation(s)
        if v != 1:
            bad.append((s, v))
    assert not bad, f"non-codespace state: {bad}"


@pytest.mark.parametrize("err_qubit", range(7))
def test_steane_detects_single_x_error(err_qubit):
    """Every single bit-flip is detected by at least one Z-stabilizer."""
    st = StabilizerState(7)
    steane_code.encode(st, *range(7))
    st.x(err_qubit)
    violations = [s for s in steane_code.z_stabilizers
                  if st.pauli_expectation(s) == -1]
    assert len(violations) >= 1


@pytest.mark.parametrize("err_qubit", range(7))
def test_steane_detects_single_z_error(err_qubit):
    """Every single phase-flip is detected by at least one X-stabilizer."""
    st = StabilizerState(7)
    steane_code.encode(st, *range(7))
    st.z(err_qubit)
    violations = [s for s in steane_code.x_stabilizers
                  if st.pauli_expectation(s) == -1]
    assert len(violations) >= 1


# ---- 9-qubit Shor code ----

def test_shor_nine_encoded_zero_in_codespace():
    st = StabilizerState(9)
    shor_nine_qubit_code.encode(st, *range(9))
    bad = []
    for s in (shor_nine_qubit_code.x_stabilizers
              + shor_nine_qubit_code.z_stabilizers):
        v = st.pauli_expectation(s)
        if v != 1:
            bad.append((s, v))
    assert not bad, f"non-codespace state: {bad}"


@pytest.mark.parametrize("err_qubit", range(9))
def test_shor_detects_single_x_error(err_qubit):
    """Bit flips are detected by inner-code Z-stabilizers."""
    st = StabilizerState(9)
    shor_nine_qubit_code.encode(st, *range(9))
    st.x(err_qubit)
    violations = [s for s in shor_nine_qubit_code.z_stabilizers
                  if st.pauli_expectation(s) == -1]
    assert len(violations) >= 1


@pytest.mark.parametrize("err_qubit", range(9))
def test_shor_detects_single_z_error(err_qubit):
    """Phase flips are detected by outer-code X-stabilizers."""
    st = StabilizerState(9)
    shor_nine_qubit_code.encode(st, *range(9))
    st.z(err_qubit)
    violations = [s for s in shor_nine_qubit_code.x_stabilizers
                  if st.pauli_expectation(s) == -1]
    assert len(violations) >= 1

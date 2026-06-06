"""End-to-end QEC tests: prepare codespace, inject error, decode, recover."""

import numpy as np
import pytest

from qbit_simulator.stabilizer import StabilizerState
from qbit_simulator.qec import (
    three_qubit_repetition_code, steane_code, shor_nine_qubit_code,
    five_qubit_perfect_code, surface_code_d3,
    prepare_codespace_state, measure_syndromes,
    build_decoder_table, decode_and_correct, verify_codespace,
)


# ---- decoder table sanity ----

def test_decoder_table_has_identity_for_zero_syndrome():
    table = build_decoder_table(steane_code)
    zero = tuple([0] * (len(steane_code.x_stabilizers) + len(steane_code.z_stabilizers)))
    assert table[zero] == "I" * 7


def test_decoder_table_covers_all_single_x_errors_for_steane():
    """Single X errors should produce distinct nonzero syndromes for Steane."""
    table = build_decoder_table(steane_code)
    seen_syndromes = []
    for q in range(7):
        err = ["I"] * 7
        err[q] = "X"
        # Manually compute the syndrome.
        syndrome = []
        for s in steane_code.z_stabilizers + steane_code.x_stabilizers:
            anti = 0
            for pe, ps in zip("".join(err), s):
                ex = pe in "XY"; ez = pe in "ZY"
                sx = ps in "XY"; sz = ps in "ZY"
                anti ^= (ex & sz) ^ (ez & sx)
            syndrome.append(anti)
        seen_syndromes.append(tuple(syndrome))
    # Each of the 7 syndromes should be distinct (and the decoder maps them
    # back to the correct correction).
    assert len(set(seen_syndromes)) == 7


# ---- end-to-end correction on Steane ----

@pytest.mark.parametrize("err_qubit", range(7))
def test_steane_end_to_end_x_correction(err_qubit):
    """Encode |0_L⟩, inject X on qubit `err_qubit`, decode, verify codespace."""
    rng = np.random.default_rng(0)
    st = StabilizerState(7 + 1)  # 7 data + 1 ancilla scratch
    data_qubits = list(range(7))
    # Encode using the canonical circuit (deterministic).
    steane_code.encode(st, *data_qubits)
    # Inject error.
    st.x(err_qubit)
    # Decode + correct.
    decoder = build_decoder_table(steane_code)
    decode_and_correct(st, steane_code, data_qubits, decoder)
    # Verify all stabilizers are +1 again.
    exps = verify_codespace(st, steane_code, data_qubits)
    assert all(v == 1 for v in exps.values())


@pytest.mark.parametrize("err_qubit", range(7))
def test_steane_end_to_end_z_correction(err_qubit):
    """Same with Z errors."""
    rng = np.random.default_rng(0)
    st = StabilizerState(7 + 1)
    data_qubits = list(range(7))
    steane_code.encode(st, *data_qubits)
    st.z(err_qubit)
    decoder = build_decoder_table(steane_code)
    decode_and_correct(st, steane_code, data_qubits, decoder)
    exps = verify_codespace(st, steane_code, data_qubits)
    assert all(v == 1 for v in exps.values())


# ---- end-to-end on Shor 9 ----

@pytest.mark.parametrize("err_qubit", range(9))
def test_shor_nine_end_to_end_x_correction(err_qubit):
    st = StabilizerState(9 + 1)
    data_qubits = list(range(9))
    shor_nine_qubit_code.encode(st, *data_qubits)
    st.x(err_qubit)
    decoder = build_decoder_table(shor_nine_qubit_code)
    decode_and_correct(st, shor_nine_qubit_code, data_qubits, decoder)
    exps = verify_codespace(st, shor_nine_qubit_code, data_qubits)
    assert all(v == 1 for v in exps.values())


@pytest.mark.parametrize("err_qubit", range(9))
def test_shor_nine_end_to_end_z_correction(err_qubit):
    st = StabilizerState(9 + 1)
    data_qubits = list(range(9))
    shor_nine_qubit_code.encode(st, *data_qubits)
    st.z(err_qubit)
    decoder = build_decoder_table(shor_nine_qubit_code)
    decode_and_correct(st, shor_nine_qubit_code, data_qubits, decoder)
    exps = verify_codespace(st, shor_nine_qubit_code, data_qubits)
    assert all(v == 1 for v in exps.values())


# ---- 5-qubit code end-to-end (now that encoding works) ----

@pytest.mark.parametrize("err_qubit", range(5))
@pytest.mark.parametrize("err_op",   ["X", "Y", "Z"])
def test_five_qubit_end_to_end_correction(err_qubit, err_op):
    """[[5,1,3]] code: encode |0_L⟩, inject any single-qubit Pauli error,
    decode via lookup, verify state is back in the codespace."""
    st = StabilizerState(5 + 1)        # 5 data + 1 ancilla
    data_qubits = list(range(5))
    five_qubit_perfect_code.encode(st, *data_qubits)
    # Inject error.
    {"X": st.x, "Y": st.y, "Z": st.z}[err_op](err_qubit)
    decoder = build_decoder_table(five_qubit_perfect_code)
    decode_and_correct(st, five_qubit_perfect_code, data_qubits, decoder)
    # Verify all stabilizers report +1.
    for s in five_qubit_perfect_code.stabilizers:
        full = s + "I"
        assert st.pauli_expectation(full) == 1, (
            f"after {err_op}({err_qubit}) + decode: stabilizer {s} = "
            f"{st.pauli_expectation(full)}"
        )


# ---- repetition code end-to-end ----

@pytest.mark.parametrize("err_qubit", range(3))
def test_repetition_end_to_end(err_qubit):
    """3-qubit repetition code: encode |0⟩, flip a qubit, decode, verify."""
    st = StabilizerState(3 + 1)
    data_qubits = list(range(3))
    three_qubit_repetition_code.encode(st, *data_qubits)
    st.x(err_qubit)
    decoder = build_decoder_table(three_qubit_repetition_code)
    decode_and_correct(st, three_qubit_repetition_code, data_qubits, decoder)
    exps = verify_codespace(st, three_qubit_repetition_code, data_qubits)
    assert all(v == 1 for v in exps.values())

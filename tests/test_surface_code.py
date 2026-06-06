"""Surface code patch d=3 tests (stabilizer structure only — encoding circuit
is intentionally not implemented; demos work via direct stabilizer measurements
on injected errors)."""

import pytest

from qbit_simulator.stabilizer import StabilizerState
from qbit_simulator.qec import surface_code_d3


def test_surface_code_d3_parameters():
    assert surface_code_d3.n_data == 9
    assert surface_code_d3.n_logical == 1
    assert surface_code_d3.distance == 3
    assert len(surface_code_d3.x_stabilizers) == 4
    assert len(surface_code_d3.z_stabilizers) == 4


def test_surface_code_d3_stabilizers_commute():
    """All 8 stabilizers must commute pairwise (already verified at import,
    but we re-check for paranoia)."""
    stabs = surface_code_d3.x_stabilizers + surface_code_d3.z_stabilizers
    for i, a in enumerate(stabs):
        for b in stabs[i + 1:]:
            anti = 0
            for pa, pb in zip(a, b):
                ax = pa in "XY"; az = pa in "ZY"
                bx = pb in "XY"; bz = pb in "ZY"
                anti ^= (ax & bz) ^ (az & bx)
            assert anti == 0


def test_surface_code_logical_operators_commute_with_stabilizers():
    """Logical X and Z must commute with every stabilizer."""
    for L in (surface_code_d3.logical_x, surface_code_d3.logical_z):
        for s in surface_code_d3.x_stabilizers + surface_code_d3.z_stabilizers:
            anti = 0
            for pl, ps in zip(L, s):
                lx = pl in "XY"; lz = pl in "ZY"
                sx = ps in "XY"; sz = ps in "ZY"
                anti ^= (lx & sz) ^ (lz & sx)
            assert anti == 0, f"{L} anticommutes with {s}"


def test_surface_code_logicals_anticommute():
    """Logical X and Z should anticommute (defining a single logical qubit)."""
    Lx = surface_code_d3.logical_x
    Lz = surface_code_d3.logical_z
    anti = 0
    for pa, pb in zip(Lx, Lz):
        ax = pa in "XY"; az = pa in "ZY"
        bx = pb in "XY"; bz = pb in "ZY"
        anti ^= (ax & bz) ^ (az & bx)
    assert anti == 1


def test_surface_code_encoding_not_implemented():
    with pytest.raises(NotImplementedError):
        surface_code_d3.encode()


def test_surface_code_single_error_detection():
    """On |0⟩^⊗9 (a +1 eigenstate of all Z stabilizers), injecting an X error
    on any data qubit should be detected by at least one Z stabilizer."""
    for err_qubit in range(9):
        st = StabilizerState(9)
        st.x(err_qubit)
        # Count Z-stabilizer violations.
        violations = sum(
            1 for s in surface_code_d3.z_stabilizers
            if st.pauli_expectation(s) == -1
        )
        assert violations >= 1, f"X({err_qubit}) not detected"

"""QRAM tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.qram import (
    qram_query, qram_state_load,
    prepare_state_via_qram,
    qram_grover_oracle, diffusion_operator, qram_grover_search,
)


# ---- Basic query ----

def test_qram_query_basis_state():
    """For |a, 0⟩ input, qram_query gives |a, x_a⟩."""
    database = [5, 3, 0, 7]
    n_a, n_d = 2, 3
    # Input: |a=2, 0⟩ → index = 2 * 8 = 16.
    psi = np.zeros(2 ** (n_a + n_d), dtype=complex)
    psi[2 * 8] = 1.0
    out = qram_query(psi, database, n_a, n_d)
    # Expected: |a=2, y=0 XOR 0⟩ = |a=2, 0⟩ → still at index 16.
    assert abs(out[2 * 8] - 1.0) < 1e-12


def test_qram_query_loads_data():
    """Address 0, data y=0 → loads database[0]."""
    database = [5, 3, 6, 7]
    n_a, n_d = 2, 3
    psi = np.zeros(2 ** (n_a + n_d), dtype=complex)
    psi[0] = 1.0    # |a=0, y=0⟩
    out = qram_query(psi, database, n_a, n_d)
    # Expected: |a=0, y=5⟩ → index 0*8 + 5 = 5.
    assert abs(out[5] - 1.0) < 1e-12


def test_qram_query_superposition():
    """A superposition over addresses should load each datum coherently."""
    database = [1, 2]
    n_a, n_d = 1, 2
    psi = np.zeros(2 ** (n_a + n_d), dtype=complex)
    psi[0] = 1 / np.sqrt(2)    # |0, 0⟩
    psi[4] = 1 / np.sqrt(2)    # |1, 0⟩
    out = qram_query(psi, database, n_a, n_d)
    # Expected: (|0,1⟩ + |1,2⟩) / √2.
    expected = np.zeros_like(out)
    expected[1] = 1 / np.sqrt(2)   # |0,1⟩
    expected[6] = 1 / np.sqrt(2)   # |1,2⟩
    assert np.allclose(out, expected, atol=1e-12)


def test_qram_query_validates_database_size():
    with pytest.raises(ValueError):
        qram_query(np.zeros(8, dtype=complex), [1, 2, 3], 2, 1)


def test_qram_query_validates_state_size():
    with pytest.raises(ValueError):
        qram_query(np.zeros(7, dtype=complex), [1, 2, 3, 4], 2, 1)


# ---- Bulk state load ----

def test_qram_state_load_creates_superposition():
    """Each address-data pair has equal amplitude 1/√N."""
    database = [5, 3, 0, 7]
    n_a, n_d = 2, 3
    psi = qram_state_load(database, n_a, n_d)
    for a in range(4):
        idx = (a << n_d) | database[a]
        assert abs(psi[idx] - 0.5) < 1e-12
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_qram_state_load_zero_data():
    database = [0, 0, 0, 0]
    n_a, n_d = 2, 2
    psi = qram_state_load(database, n_a, n_d)
    # All amplitudes in |a, 0⟩ states.
    for a in range(4):
        idx = a << n_d
        assert abs(psi[idx] - 0.5) < 1e-12


# ---- Prepare arbitrary state ----

def test_prepare_state_normalizes():
    amps = np.array([3.0, 4.0, 0.0, 0.0], dtype=complex)
    psi = prepare_state_via_qram(amps, n_address=2)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12
    assert abs(psi[0] - 0.6) < 1e-9
    assert abs(psi[1] - 0.8) < 1e-9


def test_prepare_state_rejects_zero():
    with pytest.raises(ValueError):
        prepare_state_via_qram(np.zeros(4, dtype=complex), n_address=2)


# ---- Grover oracle ----

def test_grover_oracle_marks_target_only():
    """Oracle: -1 on addresses where database[a] == target."""
    database = [1, 7, 3, 7]
    O = qram_grover_oracle(database, target=7, n_address=2)
    diag = np.diag(O).real
    assert diag[0] == 1.0     # database[0] = 1 ≠ 7
    assert diag[1] == -1.0    # database[1] = 7
    assert diag[2] == 1.0     # database[2] = 3
    assert diag[3] == -1.0    # database[3] = 7


def test_diffusion_operator_unitary():
    D = diffusion_operator(3)
    assert np.allclose(D @ D.conj().T, np.eye(8), atol=1e-12)


# ---- Grover search ----

def test_grover_search_unique_target():
    """A unique target in 8 entries should be found with high probability."""
    db = [1, 2, 3, 4, 5, 6, 7, 8]
    result = qram_grover_search(db, target=5)
    assert result["p_success"] > 0.9
    assert result["M"] == 1


def test_grover_search_multiple_matches():
    db = [1, 7, 3, 7, 2, 7, 5, 4]
    result = qram_grover_search(db, target=7)
    assert result["p_success"] > 0.7
    assert result["M"] == 3


def test_grover_search_missing_target():
    """If target is absent, p_success should be 0."""
    db = [1, 2, 3, 4]
    result = qram_grover_search(db, target=99)
    assert result["p_success"] == 0.0
    assert result["M"] == 0


def test_grover_search_non_power_of_two():
    with pytest.raises(ValueError):
        qram_grover_search([1, 2, 3], target=1)

"""Pauli grouping tests."""

import pytest

from qbit_simulator.algorithms.pauli_grouping import (
    qwc_compatible, qwc_group_compatible, greedy_qwc_grouping,
    group_basis, pauli_group_stats,
)


def test_qwc_compatible_basic():
    assert qwc_compatible("XII", "XII") is True
    assert qwc_compatible("XII", "IZI") is True   # disjoint supports
    # Conflict: X vs Z on the same qubit.
    assert qwc_compatible("XII", "ZII") is False
    # Conflict: X vs Y on the same qubit.
    assert qwc_compatible("XYZ", "YYZ") is False
    # XIZ vs XYZ: qubit 0 (X vs X) ok, qubit 1 (I vs Y) ok, qubit 2 (Z vs Z) ok — compatible.
    assert qwc_compatible("XIZ", "XYZ") is True


def test_qwc_compatible_disjoint_supports():
    """Two Paulis on disjoint supports are always QWC."""
    assert qwc_compatible("XIII", "IIIZ") is True
    assert qwc_compatible("IXII", "ZIZI") is True


def test_qwc_grouping_singletons():
    """A list of mutually incompatible Paulis should produce one group per Pauli."""
    paulis = ["X", "Y", "Z"]
    groups = greedy_qwc_grouping(paulis)
    assert len(groups) == 3


def test_qwc_grouping_all_compatible():
    """Paulis on disjoint supports should all fit in one group."""
    paulis = ["XII", "IYI", "IIZ"]
    groups = greedy_qwc_grouping(paulis)
    assert len(groups) == 1
    assert sorted(groups[0]) == sorted(paulis)


def test_qwc_grouping_h2_sto3g():
    """The H2 STO-3G Hamiltonian Pauli list typically groups into < n_paulis."""
    from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_hamiltonian
    H = h2_sto3g_hamiltonian(0.74)
    paulis = [s for _, s in H.terms]
    stats = pauli_group_stats(paulis)
    # At least some compression — typically H2 STO-3G groups by ~2-3×.
    assert stats["compression_ratio"] > 1.0


def test_group_basis_computes_correctly():
    """group_basis should give the union of all non-I characters."""
    group = ["XII", "IZI", "IIY"]
    assert group_basis(group) == "XZY"


def test_group_basis_rejects_inconsistent():
    """If two strings disagree on a non-I qubit, group_basis should raise."""
    group = ["XII", "ZII"]   # disagree at qubit 0
    with pytest.raises(ValueError):
        group_basis(group)


def test_pauli_group_stats_returns_expected_keys():
    stats = pauli_group_stats(["XI", "IZ", "XZ"])
    assert "n_paulis" in stats
    assert "n_groups" in stats
    assert "compression_ratio" in stats
    assert "group_bases" in stats

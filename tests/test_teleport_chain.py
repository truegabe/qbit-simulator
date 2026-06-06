"""Tests for entanglement-swapping teleportation chains."""

import numpy as np
import pytest

from qbit_simulator.algorithms.teleport_chain import teleport_chain


@pytest.mark.parametrize("n_links", [1, 2, 3, 5, 10])
def test_chain_produces_end_to_end_bell_pair(n_links):
    """After all swaps, qubits 0 and 2N-1 should form a Bell pair."""
    rng = np.random.default_rng(0)
    result = teleport_chain(n_links, rng=rng)
    assert result["is_bell_pair"], (
        f"chain length {n_links}: <XX>={result['xx_expectation']}, "
        f"<ZZ>={result['zz_expectation']}"
    )


def test_chain_qubit_count():
    result = teleport_chain(5)
    assert result["n_qubits"] == 10
    assert result["n_links"] == 5


def test_chain_correctness_at_scale():
    """A 50-link chain should still produce a Bell pair."""
    rng = np.random.default_rng(0)
    result = teleport_chain(50, rng=rng)
    assert result["is_bell_pair"]


def test_chain_outcomes_match_link_count():
    """N links → N-1 intermediate Bell-basis measurements."""
    n = 8
    result = teleport_chain(n)
    assert len(result["outcomes"]) == n - 1

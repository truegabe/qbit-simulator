"""Quantum nonlocality games tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.nonlocality_games import (
    ghz_game_single_round, ghz_game_full_simulation,
    verify_magic_square, magic_square_classical_max_win_rate,
    MAGIC_SQUARE_OBSERVABLES,
)


# ---- GHZ game ----

@pytest.mark.parametrize("inputs", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_ghz_game_quantum_wins_every_round(inputs):
    """With the optimal quantum strategy, every round of the GHZ game wins."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        r = ghz_game_single_round(inputs, rng)
        assert r["won"], f"GHZ game lost on inputs {inputs}: {r}"


def test_ghz_game_rejects_invalid_inputs():
    """Inputs that don't satisfy x ⊕ y ⊕ z = 0 must be rejected."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ghz_game_single_round((0, 0, 1), rng)


def test_ghz_game_full_simulation_above_classical_max():
    """Quantum win rate > 0.75 (classical maximum) — by a lot."""
    rng = np.random.default_rng(0)
    result = ghz_game_full_simulation(n_trials_per_input=100, rng=rng)
    assert result["overall_win_rate"] > 0.95
    assert result["classical_max"] == 0.75


def test_ghz_game_each_input_wins():
    """Every valid input gives 100% quantum win rate."""
    rng = np.random.default_rng(0)
    result = ghz_game_full_simulation(n_trials_per_input=50, rng=rng)
    for inputs, rate in result["per_input_win_rates"].items():
        assert rate == 1.0, f"input {inputs}: win rate {rate}"


# ---- Magic square ----

def test_magic_square_structure():
    """Verify the algebraic structure: each row/column product is ±I, and
    the overall row-product times column-product gives the impossibility."""
    result = verify_magic_square()
    # Every row and column product must be exactly ±I (not anywhere else).
    for s in result["row_signs"]:
        assert s in (-1, +1), f"row sign {s} not ±1"
    for s in result["col_signs"]:
        assert s in (-1, +1), f"column sign {s} not ±1"


def test_magic_square_parity_contradiction():
    """Mermin-Peres: the row-products and column-products have inconsistent
    parities. This is what makes classical ±1 assignment impossible."""
    result = verify_magic_square()
    row_parity = result["row_product_overall"]
    col_parity = result["col_product_overall"]
    # Both should be defined.
    assert row_parity is not None and col_parity is not None
    # The product over rows must NOT equal product over cols (the magic).
    # Equivalently: the (row × col) product evaluated as -I (well-known result).
    assert row_parity * col_parity == -1, (
        f"row_parity={row_parity}, col_parity={col_parity} — should multiply to -1"
    )


def test_magic_square_observables_2_by_2():
    """Each entry of the magic square should be a 2-character Pauli string."""
    for row in MAGIC_SQUARE_OBSERVABLES:
        for entry in row:
            assert len(entry) == 2
            for ch in entry:
                assert ch in "IXYZ"


def test_classical_max_win_rate():
    """The classical bound is 8/9 ≈ 0.889."""
    assert abs(magic_square_classical_max_win_rate() - 8 / 9) < 1e-12

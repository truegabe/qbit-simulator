import numpy as np
import pytest

from qbit_simulator.algorithms.chsh import (
    chsh_quantum_win_rate, chsh_classical_win_rate, tsirelson_bound,
)


def test_classical_strategy_wins_3_in_4():
    rng = np.random.default_rng(0)
    rate = chsh_classical_win_rate(n_rounds=4000, rng=rng)
    assert rate == pytest.approx(0.75, abs=0.02)


def test_tsirelson_bound_value():
    assert tsirelson_bound() == pytest.approx(np.cos(np.pi / 8) ** 2)
    assert tsirelson_bound() == pytest.approx(0.8536, abs=0.001)


def test_quantum_strategy_beats_classical_limit():
    """The quantum win rate must exceed the classical bound of 0.75
    AND approach the Tsirelson bound 0.8536."""
    rng = np.random.default_rng(7)
    rate = chsh_quantum_win_rate(n_rounds=4000, rng=rng)
    assert rate > 0.78   # comfortably above classical 0.75 with 4000 rounds
    assert rate < 0.88   # below Tsirelson + statistical wiggle

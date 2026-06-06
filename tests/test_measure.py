import numpy as np

from qbit_simulator import Qubit, H, measure, sample


def test_measure_zero_state_always_returns_zero():
    rng = np.random.default_rng(0)
    q = Qubit.zero()
    for _ in range(50):
        outcome, _ = measure(q.state, rng=rng)
        assert outcome == 0


def test_measure_one_state_always_returns_one():
    rng = np.random.default_rng(0)
    q = Qubit.one()
    for _ in range(50):
        outcome, _ = measure(q.state, rng=rng)
        assert outcome == 1


def test_superposition_is_roughly_balanced():
    rng = np.random.default_rng(42)
    q = Qubit.zero().apply(H)
    probs = np.abs(q.state) ** 2
    outcomes = sample(probs, shots=10_000, rng=rng)
    zeros = int(np.sum(outcomes == 0))
    assert 4800 <= zeros <= 5200


def test_collapsed_state_is_basis_state():
    rng = np.random.default_rng(1)
    q = Qubit.plus()
    outcome, collapsed = measure(q.state, rng=rng)
    assert np.isclose(np.linalg.norm(collapsed), 1.0)
    assert np.isclose(np.abs(collapsed[outcome]), 1.0)

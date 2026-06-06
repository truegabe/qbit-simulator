import numpy as np
import pytest

from qbit_simulator.algorithms.teleportation import teleport_state, fidelity


@pytest.mark.parametrize("alpha,beta", [
    (1.0, 0.0),                  # |0>
    (0.0, 1.0),                  # |1>
    (1 / np.sqrt(2), 1 / np.sqrt(2)),  # |+>
    (1 / np.sqrt(2), -1 / np.sqrt(2)),  # |->
    (0.6, 0.8),                  # arbitrary real
    (0.5, 0.5j * np.sqrt(3)),    # complex
])
def test_teleportation_fidelity_is_one(alpha, beta):
    rng = np.random.default_rng(0)
    _, ar, br, _ = teleport_state(alpha, beta, rng=rng)
    # The received amplitudes may differ by global phase from the original;
    # fidelity = |<orig|recv>|^2 should be 1.
    f = fidelity(complex(alpha), complex(beta), ar, br)
    assert f == pytest.approx(1.0, abs=1e-10)


def test_teleportation_destroys_source():
    """After teleportation, qubits 0 and 1 are in a known classical state,
    not the original |ψ⟩."""
    rng = np.random.default_rng(1)
    qc, _, _, (m0, m1) = teleport_state(0.6, 0.8, rng=rng)
    # Qubits 0 and 1 must each be in a definite computational basis state.
    p = qc.probabilities()
    # All probability should be on indices where bits q0=m0, q1=m1.
    for idx in range(8):
        b0 = (idx >> 2) & 1
        b1 = (idx >> 1) & 1
        if not (b0 == m0 and b1 == m1):
            assert p[idx] == pytest.approx(0.0, abs=1e-10)


def test_teleportation_all_four_outcomes_possible():
    """Over many trials each of the 4 (m0, m1) outcomes should occur."""
    rng = np.random.default_rng(42)
    seen = set()
    for _ in range(200):
        _, _, _, outcome = teleport_state(0.6, 0.8, rng=rng)
        seen.add(outcome)
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}

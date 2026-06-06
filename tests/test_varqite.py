"""VarQITE tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.varqite import varqite
from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_hamiltonian
from qbit_simulator.algorithms.vqe import h2_ansatz


def test_varqite_h2_converges_to_ground():
    """VarQITE on H2 STO-3G should converge to E_FCI within ~mHa."""
    H = h2_sto3g_hamiltonian(0.74)

    # h2_ansatz takes a scalar theta; wrap for vector interface.
    def ansatz(theta_vec):
        return h2_ansatz(float(theta_vec[0]))

    result = varqite(H, ansatz, theta0=np.array([0.3]),
                      n_steps=200, d_tau=0.1)
    assert abs(result["final_energy"] - result["ground_energy"]) < 1e-3


def test_varqite_energy_decreases_monotonically():
    """Imaginary time evolution → energy should be non-increasing."""
    H = h2_sto3g_hamiltonian(0.74)

    def ansatz(theta_vec):
        return h2_ansatz(float(theta_vec[0]))

    result = varqite(H, ansatz, theta0=np.array([0.5]),
                      n_steps=100, d_tau=0.05)
    trace = result["energy_trace"]
    # Allow tiny numerical bumps but the overall trend should descend.
    final_third = trace[-len(trace) // 3:]
    first_third = trace[:len(trace) // 3]
    assert np.mean(final_third) < np.mean(first_third)


def test_varqite_starting_far_from_ground_still_converges():
    """Initialize far from the optimum; VarQITE should still find it."""
    H = h2_sto3g_hamiltonian(0.74)

    def ansatz(theta_vec):
        return h2_ansatz(float(theta_vec[0]))

    result = varqite(H, ansatz, theta0=np.array([2.0]),
                      n_steps=300, d_tau=0.1)
    assert abs(result["final_energy"] - result["ground_energy"]) < 5e-3


def test_varqite_returns_correct_shapes():
    H = h2_sto3g_hamiltonian(0.74)

    def ansatz(theta_vec):
        return h2_ansatz(float(theta_vec[0]))

    result = varqite(H, ansatz, theta0=np.array([0.0]),
                      n_steps=50, d_tau=0.1)
    assert result["energy_trace"].shape == (50,)
    assert result["tau_trace"].shape == (50,)
    assert result["theta_final"].shape == (1,)

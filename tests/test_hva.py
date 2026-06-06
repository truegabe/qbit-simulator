"""Hamiltonian Variational Ansatz tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.hva import make_hva_ansatz, hva_vqe
from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_hamiltonian
from qbit_simulator.pauli import PauliOp


def test_hva_ansatz_zero_params_gives_reference():
    """At θ = 0, the ansatz should produce the reference state."""
    H = PauliOp([(1.0 + 0j, "ZZ"), (0.5 + 0j, "XI")])
    ref = np.array([0, 0, 1, 0], dtype=np.complex128)
    ansatz, n_params = make_hva_ansatz(H, n_layers=2, reference_state=ref)
    theta_zero = np.zeros(n_params)
    qc = ansatz(theta_zero)
    assert np.allclose(qc.state, ref, atol=1e-10)


def test_hva_n_params():
    """For 2 layers and 3 non-identity terms: n_params = 6."""
    H = PauliOp([
        (1.0 + 0j, "ZZ"), (0.5 + 0j, "XI"), (0.5 + 0j, "IX"),
        (0.1 + 0j, "II"),    # identity, should be filtered out
    ])
    ansatz, n_params = make_hva_ansatz(H, n_layers=2)
    assert n_params == 6


def test_hva_vqe_h2_sto3g():
    """HVA-VQE on H2 STO-3G should reach E_FCI within a few mHa."""
    H = h2_sto3g_hamiltonian(0.74)
    result = hva_vqe(H, n_layers=3, seed=0, max_iter=300)
    e_exact = result["ground_energy"]
    assert abs(result["energy_opt"] - e_exact) < 5e-3


def test_hva_variational_bound():
    """E_VQE ≥ E_ground (variational principle)."""
    H = h2_sto3g_hamiltonian(0.74)
    result = hva_vqe(H, n_layers=2, seed=0, max_iter=100)
    assert result["energy_opt"] >= result["ground_energy"] - 1e-9


def test_hva_more_layers_better_energy():
    """More layers → at least as good final energy."""
    H = h2_sto3g_hamiltonian(0.74)
    r1 = hva_vqe(H, n_layers=1, seed=0, max_iter=200)
    r3 = hva_vqe(H, n_layers=3, seed=0, max_iter=300)
    # More expressive ansatz should yield at least as good a result.
    assert r3["energy_opt"] <= r1["energy_opt"] + 1e-3

import pytest

from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_hamiltonian, h2_sto3g_energy


def test_h2_sto3g_equilibrium_matches_literature():
    """Textbook FCI/STO-3G value: E(R=0.74 Å) = -1.137 Hartree."""
    e = h2_sto3g_energy(0.74)
    assert e == pytest.approx(-1.137, abs=0.005)


def test_h2_sto3g_binding_energy():
    """Binding energy: E_eq - E_∞. Literature: ~0.205 Hartree."""
    e_eq = h2_sto3g_energy(0.74)
    e_far = h2_sto3g_energy(4.0)
    binding = abs(e_eq - e_far)
    assert binding == pytest.approx(0.205, abs=0.02)


def test_h2_sto3g_minimum_near_eq():
    """Binding minimum should be near R = 0.74 Å."""
    energies = {R: h2_sto3g_energy(R) for R in [0.5, 0.6, 0.7, 0.74, 0.8, 0.9, 1.0]}
    R_min = min(energies, key=energies.get)
    assert 0.65 <= R_min <= 0.78


def test_h2_sto3g_hamiltonian_is_hermitian():
    import numpy as np
    H = h2_sto3g_hamiltonian(0.74).matrix()
    assert np.allclose(H, H.conj().T)

import numpy as np
import pytest

from qbit_simulator.pauli import PauliOp
from qbit_simulator.algorithms import (
    h2_hamiltonian, h2_ansatz, vqe, golden_section, nelder_mead,
)


# ---- PauliOp ----

def test_pauli_matrix_ZZ_eigenvalues():
    H = PauliOp([(1.0, "ZZ")]).matrix()
    eigvals = np.linalg.eigvalsh(H)
    assert np.allclose(sorted(eigvals), [-1, -1, 1, 1])


def test_pauli_expectation_basis_states():
    H = PauliOp([(1.0, "ZI")])
    # |00>: <ZI> = +1
    s00 = np.array([1, 0, 0, 0], dtype=np.complex128)
    assert H.expectation(s00) == pytest.approx(1.0)
    # |10>: <ZI> = -1
    s10 = np.array([0, 0, 1, 0], dtype=np.complex128)
    assert H.expectation(s10) == pytest.approx(-1.0)


def test_pauli_ground_state_simple():
    # H = Z has ground state |1> with energy -1.
    H = PauliOp([(1.0, "Z")])
    e, v = H.ground_state()
    assert e == pytest.approx(-1.0)
    assert abs(v[1]) == pytest.approx(1.0)


# ---- 1D minimizer ----

def test_golden_section_finds_parabola_min():
    f = lambda x: (x - 1.5) ** 2 + 0.3
    x_opt, f_opt = golden_section(f, -5.0, 5.0)
    assert x_opt == pytest.approx(1.5, abs=1e-5)
    assert f_opt == pytest.approx(0.3, abs=1e-9)


# ---- Nelder-Mead ----

def test_nelder_mead_2d_quadratic():
    f = lambda x: (x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2
    x_opt, f_opt = nelder_mead(f, np.array([0.0, 0.0]))
    assert x_opt[0] == pytest.approx(1.0, abs=1e-4)
    assert x_opt[1] == pytest.approx(-2.0, abs=1e-4)


# ---- H₂ Hamiltonian + VQE ----

def test_h2_hamiltonian_is_hermitian():
    H = h2_hamiltonian(0.75).matrix()
    assert np.allclose(H, H.conj().T)


@pytest.mark.parametrize("R", [0.5, 0.75, 1.0, 1.5, 2.0])
def test_vqe_matches_exact_diagonalization(R):
    H = h2_hamiltonian(R)
    e_exact, _ = H.ground_state()
    theta_opt, e_vqe, _ = vqe(H, h2_ansatz, theta0=0.1)
    assert e_vqe == pytest.approx(e_exact, abs=1e-6)


def test_h2_equilibrium_near_known_value():
    # Ground-state energy at R ≈ 0.74 Å should be near -1.13 Hartree for our model.
    H = h2_hamiltonian(0.74)
    e_exact, _ = H.ground_state()
    assert -1.20 < e_exact < -1.10


def test_h2_dissociation_approaches_minus_one():
    # As R grows the two atoms separate and E → -1.0 Hartree.
    H = h2_hamiltonian(3.0)
    e_exact, _ = H.ground_state()
    assert -1.0 < e_exact < -0.8


def test_h2_curve_has_minimum_near_equilibrium():
    # The binding minimum should sit between 0.6 Å and 1.0 Å.
    Rs = np.linspace(0.5, 1.5, 21)
    energies = [h2_hamiltonian(R).ground_state()[0] for R in Rs]
    R_min = Rs[int(np.argmin(energies))]
    assert 0.6 < R_min < 1.0


def test_h2_ansatz_state_form():
    qc = h2_ansatz(0.0)
    assert qc.probabilities()[0b00] == pytest.approx(1.0)
    qc = h2_ansatz(np.pi)
    assert qc.probabilities()[0b01] == pytest.approx(1.0)

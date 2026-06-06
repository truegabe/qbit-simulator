"""SYK model tests — Majorana algebra, Hamiltonian construction, chaos signatures."""

import numpy as np
import pytest

from qbit_simulator.algorithms.syk import (
    majorana_pauli, syk_hamiltonian, level_spacing_distribution,
    out_of_time_ordered_correlator, thermal_density_matrix,
)
from qbit_simulator.fermion import _pauli_string_mul


# ---- Majorana algebra ----

def test_majorana_pauli_structure():
    """χ_1 should be X_0 on N qubits, χ_2 should be Y_0."""
    p1, _ = majorana_pauli(1, N_qubits=3)
    p2, _ = majorana_pauli(2, N_qubits=3)
    assert p1 == "XII"
    assert p2 == "YII"


def test_majorana_pauli_with_z_string():
    """χ_3 should be ZX, χ_4 should be ZY."""
    p3, _ = majorana_pauli(3, N_qubits=3)
    p4, _ = majorana_pauli(4, N_qubits=3)
    assert p3 == "ZXI"
    assert p4 == "ZYI"


def test_majorana_anticommutes_with_self_neighbor():
    """{χ_i, χ_j} = 2 δ_ij. So χ_i² = I and χ_i χ_j + χ_j χ_i = 0 for i ≠ j."""
    N = 3
    for i in range(1, 2 * N + 1):
        for j in range(1, 2 * N + 1):
            p_i, _ = majorana_pauli(i, N)
            p_j, _ = majorana_pauli(j, N)
            c1, prod_ij = _pauli_string_mul(p_i, p_j)
            c2, prod_ji = _pauli_string_mul(p_j, p_i)
            if i == j:
                # χ_i² = I, so product should be identity with coefficient +1.
                assert prod_ij == "I" * N
                assert abs(c1 - 1) < 1e-12
            else:
                # χ_i χ_j = -χ_j χ_i. So c1 = -c2 (and prod_ij = prod_ji).
                assert prod_ij == prod_ji
                assert abs(c1 + c2) < 1e-12


# ---- Hamiltonian construction ----

def test_syk_hamiltonian_is_hermitian():
    """SYK H must be Hermitian for even q."""
    H = syk_hamiltonian(M_majoranas=8, q=4, seed=0)
    M = H.matrix()
    assert np.allclose(M, M.conj().T, atol=1e-10)


def test_syk_hamiltonian_zero_J():
    """With J = 0, the Hamiltonian should be ~0."""
    H = syk_hamiltonian(M_majoranas=6, q=4, seed=0, J=0.0)
    M = H.matrix()
    assert np.allclose(M, 0, atol=1e-12)


@pytest.mark.parametrize("M,q", [(4, 4), (6, 4), (8, 4)])
def test_syk_spectrum_is_real(M, q):
    H = syk_hamiltonian(M, q=q, seed=0)
    eigs = np.linalg.eigvalsh(H.matrix())
    # All eigenvalues are real for a Hermitian matrix.
    assert eigs.dtype == np.float64
    assert all(np.isfinite(eigs))


def test_syk_different_seeds_give_different_spectra():
    """Different random seeds → different Hamiltonians."""
    H1 = syk_hamiltonian(8, seed=0)
    H2 = syk_hamiltonian(8, seed=1)
    eigs1 = np.linalg.eigvalsh(H1.matrix())
    eigs2 = np.linalg.eigvalsh(H2.matrix())
    assert not np.allclose(eigs1, eigs2)


# ---- chaos signatures ----

def test_level_spacing_basic():
    """Spectrum should produce nontrivial level spacings."""
    H = syk_hamiltonian(M_majoranas=10, seed=0)
    stats = level_spacing_distribution(H)
    assert len(stats["eigenvalues"]) == 2 ** (10 // 2)   # = 32
    assert len(stats["spacings"]) == 31
    assert stats["mean_spacing"] > 0


def test_fermion_parity_commutes_with_syk():
    """The fermion-parity operator P must commute with the SYK Hamiltonian
    (otherwise sector projection wouldn't be meaningful)."""
    from qbit_simulator.algorithms.syk import (
        fermion_parity_operator, syk_hamiltonian,
    )
    M = 8
    H = syk_hamiltonian(M_majoranas=M, q=4, seed=0).matrix()
    N = M // 2
    P = fermion_parity_operator(N)
    commutator = H @ P - P @ H
    assert np.allclose(commutator, 0, atol=1e-9)


def test_syk_r_statistic_within_parity_sector():
    """Within a single fermion-parity sector, SYK shows Wigner-Dyson level
    statistics (r-statistic ≈ 0.5–0.7 for GOE/GUE). Outside any sector
    projection, the naive r is corrupted by cross-sector intercalation.
    """
    from qbit_simulator.algorithms.syk import (
        syk_hamiltonian, project_to_parity_sector,
    )
    rs = []
    for seed in range(5):
        H_full = syk_hamiltonian(M_majoranas=10, q=4, seed=seed).matrix()
        H_sector = project_to_parity_sector(H_full, parity_sign=1)
        eigs = np.sort(np.linalg.eigvalsh(H_sector))
        spacings = np.diff(eigs)
        r_values = []
        for i in range(len(spacings) - 1):
            a, b = spacings[i], spacings[i + 1]
            if max(a, b) > 1e-9:
                r_values.append(min(a, b) / max(a, b))
        if r_values:
            rs.append(float(np.mean(r_values)))
    mean_r = float(np.mean(rs))
    # GOE: <r> ≈ 0.536, GUE: <r> ≈ 0.603, Poisson: <r> ≈ 0.386.
    # SYK_4 on M=10 majoranas: GUE-class. Allow generous interval for finite N.
    assert 0.40 < mean_r < 0.80, f"r-statistic {mean_r:.3f} outside chaotic range"


# ---- thermal density matrix ----

def test_thermal_density_matrix_trace_one():
    """Tr(ρ_β) = 1."""
    H = syk_hamiltonian(M_majoranas=8, seed=0).matrix()
    for beta in (0.1, 1.0, 5.0):
        rho = thermal_density_matrix(H, beta)
        assert abs(np.trace(rho) - 1.0) < 1e-9


def test_thermal_density_matrix_hermitian():
    H = syk_hamiltonian(M_majoranas=6, seed=0).matrix()
    rho = thermal_density_matrix(H, beta=2.0)
    assert np.allclose(rho, rho.conj().T, atol=1e-10)


# ---- OTOC ----

def test_otoc_returns_real_array():
    H = syk_hamiltonian(M_majoranas=8, seed=0)
    dim = 2 ** 4
    # V = X_0, W = X_1 in Pauli representation
    from qbit_simulator.gates import I2, X
    V = np.kron(np.kron(np.kron(X, I2), I2), I2)
    W = np.kron(np.kron(np.kron(I2, X), I2), I2)
    times = np.linspace(0, 2, 5)
    c = out_of_time_ordered_correlator(H, V, W, beta=1.0, times=times)
    assert c.shape == (5,)
    assert np.all(np.isfinite(c))


def test_otoc_at_t_zero_is_zero():
    """[W(0), V] = [W, V]; for [X_0, X_1] = 0 they commute, so OTOC at t=0 should be 0."""
    H = syk_hamiltonian(M_majoranas=8, seed=0)
    from qbit_simulator.gates import I2, X
    V = np.kron(np.kron(np.kron(X, I2), I2), I2)
    W = np.kron(np.kron(np.kron(I2, X), I2), I2)
    c = out_of_time_ordered_correlator(H, V, W, beta=1.0,
                                         times=np.array([0.0]))
    assert abs(c[0]) < 1e-9

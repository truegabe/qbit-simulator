"""DMRG ground-state solver tests.

For small N (up to ~10 qubits) we can compare DMRG against exact
diagonalization of the full 2^N × 2^N Hamiltonian. For larger N DMRG
should match known thermodynamic-limit ground energies.
"""

import numpy as np
import pytest

from qbit_simulator.mpo import tfim_mpo, heisenberg_mpo
from qbit_simulator.dmrg import dmrg


# ---- MPO sanity ----

def test_tfim_mpo_reconstructs_dense_h():
    """For small N, the contracted MPO must equal the explicit Hamiltonian."""
    from qbit_simulator.gates import X, Z, I2
    n = 4
    J, h = 1.0, 0.5
    mpo = tfim_mpo(n, J=J, h=h)
    H_mpo = mpo.to_dense()

    # Build H explicitly.
    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    H_ref = np.zeros((2**n, 2**n), dtype=np.complex128)
    for i in range(n - 1):
        chain = [I2] * n
        chain[i] = Z; chain[i + 1] = Z
        H_ref += -J * kron_chain(chain)
    for i in range(n):
        chain = [I2] * n
        chain[i] = X
        H_ref += -h * kron_chain(chain)
    assert np.allclose(H_mpo, H_ref, atol=1e-10)


def test_heisenberg_mpo_reconstructs_dense_h():
    from qbit_simulator.gates import X, Y, Z, I2
    n = 3
    J = 1.0
    mpo = heisenberg_mpo(n, J=J)
    H_mpo = mpo.to_dense()

    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    H_ref = np.zeros((2**n, 2**n), dtype=np.complex128)
    for i in range(n - 1):
        for op in (X, Y, Z):
            chain = [I2] * n
            chain[i] = op; chain[i + 1] = op
            H_ref += J * kron_chain(chain)
    assert np.allclose(H_mpo, H_ref, atol=1e-10)


# ---- DMRG correctness against exact diag ----

@pytest.mark.parametrize("n", [4, 6, 8])
def test_dmrg_tfim_matches_exact(n):
    """DMRG ground energy of TFIM matches exact diagonalization."""
    J, h = 1.0, 1.0    # critical point — hardest case
    mpo = tfim_mpo(n, J=J, h=h)
    H = mpo.to_dense()
    H_herm = (H + H.conj().T) / 2
    e_exact = float(np.linalg.eigvalsh(H_herm)[0])

    _, e_dmrg, _ = dmrg(mpo, max_chi=16, max_sweeps=15, tol=1e-9, seed=0)
    assert e_dmrg == pytest.approx(e_exact, abs=1e-6)


@pytest.mark.parametrize("n", [4, 6])
def test_dmrg_heisenberg_matches_exact(n):
    mpo = heisenberg_mpo(n)
    H = mpo.to_dense()
    H_herm = (H + H.conj().T) / 2
    e_exact = float(np.linalg.eigvalsh(H_herm)[0])

    _, e_dmrg, _ = dmrg(mpo, max_chi=16, max_sweeps=20, tol=1e-9, seed=1)
    assert e_dmrg == pytest.approx(e_exact, abs=1e-6)


# ---- thermodynamic-limit checks (no exact diag possible) ----

def test_dmrg_tfim_critical_long_chain_in_known_band():
    """TFIM at criticality (J=h=1): ground energy per site approaches -4/pi
    in the N→∞ limit (Pfeuty 1970). At N=30 the per-site energy should be
    very close to this value."""
    n = 30
    mpo = tfim_mpo(n, J=1.0, h=1.0)
    _, e_dmrg, _ = dmrg(mpo, max_chi=32, max_sweeps=10, tol=1e-9, seed=0)
    e_per_site = e_dmrg / n
    e_thermo = -4.0 / np.pi
    # Finite-size correction is O(1/N) → at N=30, within a few percent.
    assert abs(e_per_site - e_thermo) < 0.05


def test_dmrg_returns_normalized_mps():
    """The MPS DMRG returns should still be a normalized quantum state."""
    mpo = tfim_mpo(8, J=1.0, h=1.0)
    mps, _, _ = dmrg(mpo, max_chi=8, max_sweeps=5, tol=1e-8, seed=0)
    assert mps.norm() == pytest.approx(1.0, abs=1e-6)

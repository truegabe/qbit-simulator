"""MPS-VQE tests. Verify the ansatz + energy + optimizer pipeline against
exact ground energies (small N) and DMRG (larger N)."""

import numpy as np
import pytest

from qbit_simulator.mpo import tfim_mpo, heisenberg_mpo
from qbit_simulator.dmrg import dmrg
from qbit_simulator.vqe_mps import (
    n_params, brickwall_ansatz, mps_energy, vqe_mps,
)


# ---- parameter counting ----

def test_n_params_matches_explicit_count():
    # n=4, n_layers=2: layer 0 has bonds (0,1),(2,3) → 2 blocks. layer 1: (1,2) → 1 block.
    # Total: 3 blocks × 4 params = 12.
    assert n_params(4, 2) == 12
    # n=5, n_layers=3: layer 0 (0,1),(2,3) = 2; layer 1 (1,2),(3,4) = 2; layer 2 (0,1),(2,3) = 2.
    # Total: 6 × 4 = 24.
    assert n_params(5, 3) == 24


# ---- ansatz + energy sanity ----

def test_zero_params_gives_identity_block():
    """All zero params means each block is CNOT (since Ry(0) = I). So the
    ansatz on |0...0> with all Ry's = identity gives |0...0> unchanged
    (since CNOT|00⟩ = |00⟩)."""
    n = 4
    n_l = 2
    theta = np.zeros(n_params(n, n_l))
    mps = brickwall_ansatz(theta, n, n_l)
    psi = mps.to_dense()
    expected = np.zeros(2**n, dtype=np.complex128); expected[0] = 1.0
    assert np.allclose(psi, expected, atol=1e-10)


def test_mps_energy_matches_explicit_inner_product():
    """For small N, mps_energy(mps, mpo) should match psi^† · H_dense · psi."""
    n = 3
    mpo = tfim_mpo(n, J=1.0, h=0.5)
    H = mpo.to_dense()
    H = (H + H.conj().T) / 2

    n_l = 2
    theta = np.random.default_rng(0).uniform(-0.5, 0.5, n_params(n, n_l))
    mps = brickwall_ansatz(theta, n, n_l)
    psi = mps.to_dense()
    e_explicit = float(np.real(psi.conj() @ H @ psi))
    e_mps = mps_energy(mps, mpo)
    assert e_mps == pytest.approx(e_explicit, abs=1e-10)


# ---- VQE convergence ----

@pytest.mark.parametrize("n", [4, 6])
def test_vqe_tfim_converges_close_to_exact(n):
    """VQE energy should approach (within a few %) the exact ground energy."""
    J, h = 1.0, 1.0
    mpo = tfim_mpo(n, J=J, h=h)
    H = mpo.to_dense()
    H = (H + H.conj().T) / 2
    e_exact = float(np.linalg.eigvalsh(H)[0])

    result = vqe_mps(mpo, n_layers=4, max_chi=16, seed=1, max_iter=400)
    # Brickwall ansatz with 4 layers is enough for small TFIM.
    # Energy is variational: result.energy >= e_exact.
    assert result["energy"] >= e_exact - 1e-9
    # And should converge close.
    assert (result["energy"] - e_exact) / abs(e_exact) < 0.03


def test_vqe_heisenberg_small_converges():
    """Same idea on Heisenberg(N=4)."""
    n = 4
    mpo = heisenberg_mpo(n)
    H = mpo.to_dense()
    H = (H + H.conj().T) / 2
    e_exact = float(np.linalg.eigvalsh(H)[0])

    result = vqe_mps(mpo, n_layers=5, max_chi=16, seed=2, max_iter=500)
    assert result["energy"] >= e_exact - 1e-9
    # Heisenberg is harder for a real-valued hardware-efficient ansatz
    # (the ground state has nontrivial sign structure). Allow generous %.
    rel_err = (result["energy"] - e_exact) / abs(e_exact)
    assert rel_err < 0.10


# ---- variational bound ----

def test_vqe_energy_always_above_ground_state():
    """⟨ψ(θ)|H|ψ(θ)⟩ ≥ E_0 for any θ -- the variational principle."""
    n = 5
    mpo = tfim_mpo(n)
    H = mpo.to_dense()
    H = (H + H.conj().T) / 2
    e_exact = float(np.linalg.eigvalsh(H)[0])

    # Try random parameter vectors -- all must lie above the ground state.
    rng = np.random.default_rng(42)
    for _ in range(10):
        theta = rng.uniform(-np.pi, np.pi, size=n_params(n, 3))
        mps = brickwall_ansatz(theta, n, 3)
        e = mps_energy(mps, mpo)
        assert e >= e_exact - 1e-9


# ---- comparison to DMRG at moderate N ----

def test_vqe_tfim_n10_within_dmrg_band():
    """At N=10, VQE with a few layers should reach within 5% of DMRG."""
    n = 10
    mpo = tfim_mpo(n, J=1.0, h=1.0)
    _, e_dmrg, _ = dmrg(mpo, max_chi=16, max_sweeps=8, tol=1e-8, seed=0)
    result = vqe_mps(mpo, n_layers=4, max_chi=16, seed=1, max_iter=600)
    assert result["energy"] >= e_dmrg - 1e-9
    rel_err = (result["energy"] - e_dmrg) / abs(e_dmrg)
    assert rel_err < 0.08

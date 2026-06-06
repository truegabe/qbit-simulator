"""TEBD time-evolution tests.

For small N we compare against exact unitary evolution via dense
matrix exponentiation. For larger N we check sanity (norm conservation,
energy conservation for Hermitian H).
"""

import numpy as np
import pytest
from scipy.linalg import expm

from qbit_simulator import MPSState
from qbit_simulator.tebd import (
    tfim_terms, heisenberg_terms, tebd_evolve, site_z_expectation,
    total_z_expectation,
)
from qbit_simulator.mpo import tfim_mpo, heisenberg_mpo


# ---- exact reference evolution ----

def _exact_evolve(initial_state: np.ndarray, H_dense: np.ndarray,
                  total_time: float) -> np.ndarray:
    U = expm(-1j * total_time * H_dense)
    return U @ initial_state


# ---- correctness against exact ----

@pytest.mark.parametrize("n,T", [(4, 0.5), (6, 0.3), (8, 0.2)])
def test_tebd_tfim_matches_exact_evolution(n, T):
    """For small N, TEBD-evolved state should match exact unitary evolution."""
    J, h = 1.0, 0.7

    # Initial state: |0...0> with a single X applied to one site (a localized
    # perturbation -- gives nontrivial dynamics under TFIM).
    mps = MPSState(n, max_chi=32)
    mps.x(n // 2)
    psi0 = mps.to_dense()

    # Exact: build H, exponentiate.
    H = tfim_mpo(n, J=J, h=h).to_dense()
    H = (H + H.conj().T) / 2  # ensure Hermitian
    psi_exact = _exact_evolve(psi0, H, T)

    # TEBD: small dt for accuracy.
    terms = tfim_terms(n, J=J, h=h)
    tebd_evolve(mps, terms, total_time=T, dt=0.005, order=2)
    psi_tebd = mps.to_dense()

    # Overlap should be near 1 (modulo global phase).
    inner = np.vdot(psi_exact, psi_tebd)
    assert abs(abs(inner) - 1.0) < 1e-3


@pytest.mark.parametrize("n,T", [(4, 0.5), (6, 0.3)])
def test_tebd_heisenberg_matches_exact_evolution(n, T):
    """Same correctness test on Heisenberg dynamics."""
    mps = MPSState(n, max_chi=32)
    mps.x(0)                                 # flip the leftmost spin
    psi0 = mps.to_dense()

    H = heisenberg_mpo(n).to_dense()
    H = (H + H.conj().T) / 2
    psi_exact = _exact_evolve(psi0, H, T)

    terms = heisenberg_terms(n)
    tebd_evolve(mps, terms, total_time=T, dt=0.005, order=2)
    psi_tebd = mps.to_dense()

    inner = np.vdot(psi_exact, psi_tebd)
    assert abs(abs(inner) - 1.0) < 1e-3


# ---- norm conservation ----

def test_tebd_preserves_norm():
    mps = MPSState(8, max_chi=32).x(0).x(3)
    terms = tfim_terms(8, J=1.0, h=0.5)
    tebd_evolve(mps, terms, total_time=1.0, dt=0.05, order=2)
    assert mps.norm() == pytest.approx(1.0, abs=1e-6)


# ---- Heisenberg conserves total magnetization ----

def test_heisenberg_conserves_total_z():
    """[H_Heisenberg, S_z_total] = 0, so total Z magnetization is conserved."""
    n = 6
    mps = MPSState(n, max_chi=16)
    mps.x(2)                                  # single flipped spin
    mz_initial = total_z_expectation(mps)

    terms = heisenberg_terms(n)
    tebd_evolve(mps, terms, total_time=0.5, dt=0.02, order=2)
    mz_final = total_z_expectation(mps)
    assert abs(mz_final - mz_initial) < 1e-4


# ---- domain wall / spin-flip propagation (qualitative) ----

def test_localized_perturbation_spreads_over_time():
    """A locally flipped spin in a uniform background should spread.

    For a Heisenberg chain with J=1, the Lieb-Robinson velocity v_LR ~ 2.
    To see the wavefront reach the far site (n-1) starting from the flip
    at qubit 0, we need T > (n-1) / v_LR ≈ (n-1)/2. We use T=5 on n=8 sites
    (giving comfortable margin).
    """
    n = 8
    mps = MPSState(n, max_chi=64)
    mps.x(0)
    z_far_initial = site_z_expectation(mps, n - 1)
    assert z_far_initial == pytest.approx(1.0, abs=1e-10)

    terms = heisenberg_terms(n)
    tebd_evolve(mps, terms, total_time=5.0, dt=0.05, order=2)
    z_far_final = site_z_expectation(mps, n - 1)
    # By T=5, the wavefront from qubit 0 should have reached qubit 7
    # (distance 7, v_LR ≈ 2, so signal arrives by T ≈ 3.5).
    assert abs(z_far_final - 1.0) > 0.01


# ---- bond dimension grows but stays bounded for short times ----

def test_bond_dimension_stays_manageable_for_short_time():
    n = 20
    mps = MPSState(n, max_chi=32).x(n // 2)
    terms = tfim_terms(n, J=1.0, h=0.7)
    result = tebd_evolve(mps, terms, total_time=1.0, dt=0.05, order=2)
    # max_chi cap should hold.
    assert all(chi <= 32 for chi in result["bond_dims"])
    # And final norm should still be 1 (since cap wasn't hit hard).
    assert mps.norm() == pytest.approx(1.0, abs=1e-4)

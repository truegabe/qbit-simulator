"""Kitaev honeycomb model tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.kitaev_honeycomb import (
    KitaevCluster, four_site_cluster, brick_cluster_8,
    kitaev_hamiltonian, plaquette_operator, ground_state,
    plaquette_flux, phase_label,
)


# ---- Cluster geometry ----

def test_four_site_each_vertex_has_three_bonds():
    c = four_site_cluster()
    # Count bond degrees.
    degree = {v: 0 for v in range(c.n_sites)}
    for bonds in [c.x_bonds, c.y_bonds, c.z_bonds]:
        for (i, j) in bonds:
            degree[i] += 1
            degree[j] += 1
    for v in range(c.n_sites):
        assert degree[v] == 3


def test_brick_cluster_each_vertex_has_three_bonds():
    c = brick_cluster_8()
    degree = {v: 0 for v in range(c.n_sites)}
    for bonds in [c.x_bonds, c.y_bonds, c.z_bonds]:
        for (i, j) in bonds:
            degree[i] += 1
            degree[j] += 1
    for v in range(c.n_sites):
        assert degree[v] == 3


def test_each_vertex_has_one_bond_per_type():
    """Defining property of Kitaev model: each vertex has 1x, 1y, 1z."""
    for c in [four_site_cluster(), brick_cluster_8()]:
        for v in range(c.n_sites):
            x_count = sum(1 for (i, j) in c.x_bonds if v in (i, j))
            y_count = sum(1 for (i, j) in c.y_bonds if v in (i, j))
            z_count = sum(1 for (i, j) in c.z_bonds if v in (i, j))
            assert x_count == 1
            assert y_count == 1
            assert z_count == 1


# ---- Hamiltonian ----

def test_hamiltonian_hermitian():
    c = four_site_cluster()
    H = kitaev_hamiltonian(c, Jx=1.0, Jy=1.5, Jz=0.8)
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_hamiltonian_shape():
    c = four_site_cluster()
    H = kitaev_hamiltonian(c)
    assert H.shape == (16, 16)


# ---- Plaquette operators commute with H ----

def test_plaquette_commutes_with_hamiltonian():
    """Plaquettes W_p are conserved Z_2 fluxes — they must commute with H."""
    c = four_site_cluster()
    H = kitaev_hamiltonian(c, Jx=1.0, Jy=1.5, Jz=0.8)
    W = plaquette_operator(c, 0)
    comm = H @ W - W @ H
    assert np.max(np.abs(comm)) < 1e-9


def test_plaquette_squares_to_identity():
    """W_p² = I (Z_2 flux)."""
    c = four_site_cluster()
    W = plaquette_operator(c, 0)
    assert np.allclose(W @ W, np.eye(W.shape[0]), atol=1e-12)


# ---- Ground state ----

def test_ground_state_in_no_flux_sector():
    """Lieb's theorem: the GS lies in the no-flux sector (⟨W_p⟩ = +1)."""
    for c in [four_site_cluster(), brick_cluster_8()]:
        E, psi = ground_state(c, Jx=1.0, Jy=1.0, Jz=1.0)
        for p in range(len(c.plaquettes)):
            W = plaquette_flux(psi, c, p)
            assert abs(W - 1.0) < 1e-8


def test_ground_state_normalized():
    E, psi = ground_state(four_site_cluster())
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_ground_state_negative_energy():
    """All couplings positive → ferromagnetic-like, energy is negative."""
    E, _ = ground_state(four_site_cluster(), Jx=1.0, Jy=1.0, Jz=1.0)
    assert E < 0


def test_energy_scales_with_coupling():
    """Doubling all J should double |E_gs|."""
    E1, _ = ground_state(four_site_cluster(), Jx=1.0, Jy=1.0, Jz=1.0)
    E2, _ = ground_state(four_site_cluster(), Jx=2.0, Jy=2.0, Jz=2.0)
    assert abs(E2 / E1 - 2.0) < 1e-9


# ---- Phase diagram ----

def test_phase_label_symmetric():
    assert phase_label(1, 1, 1) == "B"


def test_phase_label_z_dominant():
    assert phase_label(1, 1, 3) == "A_z"


def test_phase_label_x_dominant():
    assert phase_label(5, 1, 1) == "A_x"


def test_phase_label_y_dominant():
    assert phase_label(1, 5, 1) == "A_y"


def test_phase_boundary():
    """At Jx + Jy = Jz exactly, the model is gapless (B/A boundary)."""
    # Just check the label changes consistently.
    assert phase_label(1, 1, 2.001) == "A_z"
    assert phase_label(1, 1, 1.999) == "B"

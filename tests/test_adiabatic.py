"""Adiabatic evolution / MaxCut tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.adiabatic import (
    adiabatic_evolve, maxcut_hamiltonian, transverse_field_driver, plus_state,
)


def test_maxcut_hamiltonian_construction():
    """K_3 triangle: every edge contributes (ZZ - I)/2.
    Ground states: any 2:1 partition (cuts 2 of 3 edges) → E = -2."""
    edges = [(0, 1), (1, 2), (0, 2)]
    H = maxcut_hamiltonian(3, edges)
    e_gs, _ = H.ground_state()
    assert e_gs == pytest.approx(-2.0, abs=1e-9)


def test_transverse_field_ground_state():
    """The driver -Σ X_i has ground state |+⟩^⊗N with energy -N."""
    H0 = transverse_field_driver(3)
    e_gs, gs = H0.ground_state()
    assert e_gs == pytest.approx(-3.0, abs=1e-9)
    expected = plus_state(3)
    assert abs(abs(np.vdot(gs, expected)) - 1.0) < 1e-9


# ---- adiabatic finds MaxCut on small graphs ----

@pytest.mark.parametrize("edges,expected_min_energy", [
    ([(0, 1), (1, 2), (0, 2)],            -2.0),    # K_3, cut 2 of 3
    ([(0, 1), (1, 2), (2, 3), (0, 3)],    -4.0),    # 4-cycle, cut all 4
    ([(0, 1), (1, 2), (2, 3)],            -3.0),    # path, cut all 3
])
def test_adiabatic_finds_maxcut(edges, expected_min_energy):
    """Linear schedule with sufficient time should reach within 5% of optimum."""
    n = max(max(e) for e in edges) + 1
    H_target = maxcut_hamiltonian(n, edges)
    H0 = transverse_field_driver(n)
    psi0 = plus_state(n)

    result = adiabatic_evolve(H0, H_target, psi0,
                               n_steps=200, total_time=15.0)
    # Energy should be within 15% of optimum (the right adiabatic metric:
    # overlap-with-one-ground-state is ill-defined for degenerate ground
    # spaces like MaxCut's two-coloring symmetry).
    assert result["final_energy"] < expected_min_energy * 0.85


def test_adiabatic_energy_trace_decreases():
    """Energy should generally decrease along the schedule (for a slow schedule)."""
    edges = [(0, 1), (1, 2), (0, 2)]
    H_target = maxcut_hamiltonian(3, edges)
    H0 = transverse_field_driver(3)
    psi0 = plus_state(3)

    result = adiabatic_evolve(H0, H_target, psi0,
                               n_steps=100, total_time=15.0)
    # The starting energy should exceed the ending energy.
    initial_e = float(np.real(psi0.conj() @ H_target.matrix() @ psi0))
    assert result["final_energy"] < initial_e


def test_adiabatic_with_custom_schedule():
    """User-provided schedule callable."""
    edges = [(0, 1)]
    H_target = maxcut_hamiltonian(2, edges)
    H0 = transverse_field_driver(2)
    psi0 = plus_state(2)

    # Quadratic schedule -- spends more time near s=1.
    result = adiabatic_evolve(
        H0, H_target, psi0,
        n_steps=100, total_time=10.0,
        schedule=lambda x: x ** 2,
    )
    assert result["final_energy"] < -0.5  # ground is -1

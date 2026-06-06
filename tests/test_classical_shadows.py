"""Classical-shadow tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.classical_shadows import (
    random_pauli_basis, apply_basis_measurement,
    single_shadow, collect_shadows,
    shadow_estimate, shadow_estimate_observable,
    shadow_estimate_median_of_means,
    _single_shadow_pauli_estimate, _rotate_to_basis,
)


# ---- Random basis ----

def test_random_pauli_basis_correct_length():
    rng = np.random.default_rng(0)
    s = random_pauli_basis(5, rng)
    assert len(s) == 5
    for ch in s:
        assert ch in ("X", "Y", "Z")


def test_random_pauli_basis_covers_all_three():
    """Over many samples, all 3 bases should appear at each qubit."""
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        b = random_pauli_basis(1, rng)
        seen.add(b)
    assert seen == {"X", "Y", "Z"}


# ---- Basis-rotation ----

def test_rotate_z_is_identity():
    psi = np.array([0.6, 0.8], dtype=complex)
    out = _rotate_to_basis(psi, "Z")
    assert np.allclose(out, psi, atol=1e-12)


def test_rotate_x_maps_x_eigenstate_to_z():
    """|+⟩ in X-basis is |0⟩ in Z; after rotation should be [1, 0]."""
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    out = _rotate_to_basis(psi, "X")
    assert abs(out[0] - 1.0) < 1e-9
    assert abs(out[1]) < 1e-9


# ---- Measurement ----

def test_apply_measurement_returns_bits():
    rng = np.random.default_rng(0)
    psi = np.array([1, 0, 0, 0], dtype=complex)
    bits = apply_basis_measurement(psi, "ZZ", rng)
    assert len(bits) == 2
    assert all(b in (0, 1) for b in bits)


def test_measurement_z_on_zero_state_gives_zero():
    """|00⟩ measured in ZZ always gives (0, 0)."""
    rng = np.random.default_rng(0)
    psi = np.array([1, 0, 0, 0], dtype=complex)
    for _ in range(20):
        bits = apply_basis_measurement(psi, "ZZ", rng)
        assert bits == [0, 0]


# ---- Shadow construction ----

def test_single_shadow_returns_tuple():
    rng = np.random.default_rng(0)
    psi = np.array([1, 0, 0, 0], dtype=complex)
    basis, outcome = single_shadow(psi, rng)
    assert len(basis) == 2
    assert len(outcome) == 2


def test_collect_shadows_count():
    rng = np.random.default_rng(0)
    psi = np.array([1, 0, 0, 0], dtype=complex)
    shadows = collect_shadows(psi, n_shots=50, rng=rng)
    assert len(shadows) == 50


# ---- Per-shadow estimator ----

def test_per_shadow_zero_for_mismatched_basis():
    """If the random basis doesn't match the Pauli at a non-I qubit,
    the per-shot estimator is 0."""
    val = _single_shadow_pauli_estimate("X", [0], "Z")
    assert val == 0.0


def test_per_shadow_factor_3_per_match():
    """When basis matches a single-qubit Pauli and outcome = 0, the
    estimator is +3."""
    val = _single_shadow_pauli_estimate("Z", [0], "Z")
    assert val == 3.0
    val = _single_shadow_pauli_estimate("Z", [1], "Z")
    assert val == -3.0


def test_per_shadow_product_for_two_qubits():
    """For weight-2 Pauli, both qubits must match: factor 3^2 = 9."""
    val = _single_shadow_pauli_estimate("ZX", [0, 0], "ZX")
    assert val == 9.0


def test_per_shadow_identity_factor_one():
    """All-identity Pauli always gives 1."""
    val = _single_shadow_pauli_estimate("X", [1], "I")
    assert val == 1.0


# ---- Estimator correctness ----

def test_shadow_estimate_zero_state():
    """|0⟩ has ⟨Z⟩ = +1. Shadow estimator should converge to +1."""
    rng = np.random.default_rng(0)
    psi = np.array([1, 0], dtype=complex)
    shadows = collect_shadows(psi, n_shots=2000, rng=rng)
    est = shadow_estimate(shadows, "Z")
    assert abs(est - 1.0) < 0.15


def test_shadow_estimate_plus_state_x_basis():
    """|+⟩ has ⟨X⟩ = +1."""
    rng = np.random.default_rng(0)
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    shadows = collect_shadows(psi, n_shots=2000, rng=rng)
    est = shadow_estimate(shadows, "X")
    assert abs(est - 1.0) < 0.15


def test_shadow_estimate_bell_state_correlations():
    """For Φ+ = (|00⟩ + |11⟩)/√2: ⟨ZZ⟩ = +1, ⟨XX⟩ = +1, ⟨YY⟩ = -1."""
    rng = np.random.default_rng(0)
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[3] = 1 / np.sqrt(2)
    shadows = collect_shadows(psi, n_shots=3000, rng=rng)
    assert abs(shadow_estimate(shadows, "ZZ") - 1.0) < 0.2
    assert abs(shadow_estimate(shadows, "XX") - 1.0) < 0.2
    assert abs(shadow_estimate(shadows, "YY") - (-1.0)) < 0.2


def test_shadow_estimate_observable_sum():
    """A sum of Pauli terms should match the linear combination of
    individual estimates."""
    rng = np.random.default_rng(0)
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[3] = 1 / np.sqrt(2)
    shadows = collect_shadows(psi, n_shots=3000, rng=rng)
    obs = [(0.5, "ZZ"), (0.5, "XX")]
    est = shadow_estimate_observable(shadows, obs)
    # Ideal: 0.5 * 1 + 0.5 * 1 = 1.
    assert abs(complex(est).real - 1.0) < 0.2


def test_shadow_estimate_zero_shots():
    assert shadow_estimate([], "Z") == 0.0


def test_median_of_means_estimates():
    rng = np.random.default_rng(0)
    psi = np.array([1, 0], dtype=complex)
    shadows = collect_shadows(psi, n_shots=2000, rng=rng)
    est = shadow_estimate_median_of_means(shadows, "Z", n_chunks=10)
    assert abs(est - 1.0) < 0.2

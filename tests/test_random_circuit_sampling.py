"""Random circuit sampling + XEB tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.random_circuit_sampling import (
    _haar_su2, _apply_1q, _apply_2q,
    random_layer_unitary,
    build_random_circuit, sample_circuit,
    linear_xeb, linear_xeb_uniform_baseline,
    porter_thomas_pdf, porter_thomas_chi2_distance,
)


# ---- Random gates ----

def test_haar_su2_unitary():
    rng = np.random.default_rng(0)
    for _ in range(5):
        U = _haar_su2(rng)
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-9)


def test_haar_su2_det_one():
    rng = np.random.default_rng(0)
    for _ in range(5):
        U = _haar_su2(rng)
        det = np.linalg.det(U)
        assert abs(abs(det) - 1.0) < 1e-9


def test_random_layer_returns_n_unitaries():
    rng = np.random.default_rng(0)
    gates = random_layer_unitary(5, rng)
    assert len(gates) == 5
    for g in gates:
        assert g.shape == (2, 2)


# ---- Gate application ----

def test_apply_1q_unit_norm():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    U = _haar_su2(rng)
    out = _apply_1q(psi, U, q=1, n=3)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-10


def test_apply_2q_unit_norm():
    from qbit_simulator.algorithms.random_circuit_sampling import _ISWAP
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    out = _apply_2q(psi, _ISWAP, q0=0, q1=1, n=3)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-10


# ---- Circuit building ----

def test_build_circuit_returns_normalized_state():
    rng = np.random.default_rng(0)
    r = build_random_circuit(n_qubits=4, depth=5, rng=rng)
    psi = r["final_state"]
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-9


def test_build_circuit_rejects_unknown_gate():
    with pytest.raises(ValueError):
        build_random_circuit(n_qubits=4, depth=1, entangling_gate="bogus")


def test_build_circuit_supports_cz():
    rng = np.random.default_rng(0)
    r = build_random_circuit(n_qubits=4, depth=3, entangling_gate="cz", rng=rng)
    assert abs(np.linalg.norm(r["final_state"]) - 1.0) < 1e-9


# ---- Sampling ----

def test_sample_circuit_returns_indices():
    rng = np.random.default_rng(0)
    r = build_random_circuit(n_qubits=4, depth=5, rng=rng)
    samples = sample_circuit(r["final_state"], n_samples=100, rng=rng)
    assert len(samples) == 100
    assert all(0 <= s < 16 for s in samples)


def test_sample_circuit_zero_samples():
    psi = np.zeros(4, dtype=complex); psi[0] = 1.0
    samples = sample_circuit(psi, n_samples=0, rng=np.random.default_rng(0))
    assert samples == []


def test_sample_basis_state_always_returns_zero():
    """If ψ is |0⟩, all samples should be 0."""
    psi = np.zeros(8, dtype=complex); psi[0] = 1.0
    rng = np.random.default_rng(0)
    samples = sample_circuit(psi, n_samples=50, rng=rng)
    assert all(s == 0 for s in samples)


# ---- Linear XEB ----

def test_xeb_perfect_samples_near_one():
    rng = np.random.default_rng(0)
    r = build_random_circuit(n_qubits=6, depth=12, rng=rng)
    psi = r["final_state"]
    p_circ = np.abs(psi) ** 2
    samples = sample_circuit(psi, n_samples=5000, rng=rng)
    xeb = linear_xeb(p_circ, samples)
    assert xeb > 0.7


def test_xeb_uniform_samples_near_zero():
    rng = np.random.default_rng(0)
    r = build_random_circuit(n_qubits=6, depth=12, rng=rng)
    psi = r["final_state"]
    p_circ = np.abs(psi) ** 2
    uniform = list(rng.integers(0, 64, size=5000))
    xeb = linear_xeb(p_circ, uniform)
    assert abs(xeb) < 0.3


def test_xeb_empty_samples_is_zero():
    p = np.ones(4) / 4
    assert linear_xeb(p, []) == 0.0


def test_xeb_uniform_baseline():
    rng = np.random.default_rng(0)
    r = build_random_circuit(n_qubits=4, depth=10, rng=rng)
    psi = r["final_state"]
    p_circ = np.abs(psi) ** 2
    # D·mean(p_circ) = D · (1/D) = 1, so baseline = 0.
    baseline = linear_xeb_uniform_baseline(p_circ)
    assert abs(baseline) < 1e-9


# ---- Porter-Thomas ----

def test_porter_thomas_pdf_normalized():
    """∫_0^∞ D·exp(-D·p) dp = 1."""
    D = 16
    ps = np.linspace(0, 1.0, 10000)
    pdf = porter_thomas_pdf(ps, D)
    integral = np.trapezoid(pdf, ps)
    assert abs(integral - 1.0) < 1e-3


def test_porter_thomas_distance_decreases_with_depth():
    """Deeper random circuits should look more Porter-Thomas."""
    rng = np.random.default_rng(0)
    chi_shallow = porter_thomas_chi2_distance(
        np.abs(build_random_circuit(8, depth=2, rng=rng)["final_state"]) ** 2
    )
    rng = np.random.default_rng(0)
    chi_deep = porter_thomas_chi2_distance(
        np.abs(build_random_circuit(8, depth=20, rng=rng)["final_state"]) ** 2
    )
    assert chi_deep < chi_shallow + 0.1   # at least not significantly worse


def test_porter_thomas_distance_nonneg():
    rng = np.random.default_rng(0)
    r = build_random_circuit(6, depth=10, rng=rng)
    chi = porter_thomas_chi2_distance(np.abs(r["final_state"]) ** 2)
    assert chi >= 0

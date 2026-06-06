"""MERA tensor-network tests."""

import numpy as np
import pytest

from qbit_simulator.mera import MERA, mera_parameter_count


# ---- Parameter count ----

def test_parameter_count_n_4():
    p = mera_parameter_count(4)
    # 4 → 2 → 1: 2 layers.
    # Bottom layer (sites=4): 2 isometries + 1 disentangler.
    # Top layer (sites=2): 1 isometry + 0 disentanglers.
    assert p["n_layers"] == 2
    assert p["n_disentanglers"] == 1
    assert p["n_isometries"] == 3


def test_parameter_count_n_8():
    p = mera_parameter_count(8)
    assert p["n_layers"] == 3
    # Layer 0 (sites=8): 3 disentanglers, 4 isometries.
    # Layer 1 (sites=4): 1 disentangler, 2 isometries.
    # Layer 2 (sites=2): 0 disentanglers, 1 isometry.
    assert p["n_disentanglers"] == 4
    assert p["n_isometries"] == 7


def test_parameter_count_n_16():
    p = mera_parameter_count(16)
    assert p["n_layers"] == 4


# ---- MERA construction ----

def test_mera_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        MERA(6)


def test_mera_zero_qubits_rejected():
    with pytest.raises(ValueError):
        MERA(0)


@pytest.mark.parametrize("n", [4, 8])
def test_mera_layer_count(n):
    mera = MERA(n)
    assert mera.n_layers == int(np.log2(n))
    assert len(mera.disentanglers) == mera.n_layers
    assert len(mera.isometries) == mera.n_layers


def test_mera_unitary_disentanglers():
    rng = np.random.default_rng(0)
    mera = MERA(8, rng=rng)
    for layer in mera.disentanglers:
        for D in layer:
            assert np.allclose(D @ D.conj().T, np.eye(4), atol=1e-9)


def test_mera_isometry_constraint():
    """Isometries V: C² → C⁴ satisfy V†V = I_2."""
    rng = np.random.default_rng(0)
    mera = MERA(8, rng=rng)
    for layer in mera.isometries:
        for V in layer:
            V_mat = V.reshape(4, 2)
            should_be_I = V_mat.conj().T @ V_mat
            assert np.allclose(should_be_I, np.eye(2), atol=1e-9)


# ---- Dense state ----

@pytest.mark.parametrize("n", [4, 8])
def test_dense_state_shape(n):
    rng = np.random.default_rng(0)
    mera = MERA(n, rng=rng)
    psi = mera.to_dense_state()
    assert psi.shape == (2 ** n,)


@pytest.mark.parametrize("n", [4, 8])
def test_dense_state_normalized(n):
    rng = np.random.default_rng(0)
    mera = MERA(n, rng=rng)
    psi = mera.to_dense_state()
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-9


def test_dense_state_reproducible_with_same_rng():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    psi1 = MERA(4, rng=rng1).to_dense_state()
    psi2 = MERA(4, rng=rng2).to_dense_state()
    assert np.allclose(psi1, psi2, atol=1e-12)


def test_dense_state_different_with_different_rng():
    """Different RNG seeds should give different states."""
    psi1 = MERA(4, rng=np.random.default_rng(0)).to_dense_state()
    psi2 = MERA(4, rng=np.random.default_rng(1)).to_dense_state()
    overlap = abs(np.vdot(psi1, psi2))
    assert overlap < 0.99    # very different states


# ---- 2q gate helper ----

def test_apply_2q_gate_unitary_preserved():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(A)
    out = MERA._apply_2q_gate(psi, Q, 0, 1, 3)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-9

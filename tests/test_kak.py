"""KAK decomposition tests."""

import numpy as np
import pytest

from qbit_simulator.kak import (
    kak_decompose, kak_to_unitary, reconstruction_error, cnot_count,
    M, M_DAG, _so4_to_su2_su2,
)


# ---- Magic basis ----

def test_magic_basis_unitary():
    assert np.allclose(M @ M_DAG, np.eye(4), atol=1e-12)


def test_magic_basis_maps_tensor_to_real_orthogonal():
    """M† (A ⊗ B) M should be real-orthogonal for A, B ∈ SU(2)."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        def haar_su2():
            X = rng.normal(size=(2,2)) + 1j*rng.normal(size=(2,2))
            Q, _ = np.linalg.qr(X)
            return Q / np.linalg.det(Q) ** 0.5
        A = haar_su2()
        B = haar_su2()
        O = M_DAG @ np.kron(A, B) @ M
        assert np.max(np.abs(O.imag)) < 1e-10
        assert np.allclose(O.real @ O.real.T, np.eye(4), atol=1e-10)


def test_so4_to_su2_su2_tensor_roundtrip():
    """Round-trip A ⊗ B → SO(4) → A', B' should reproduce the tensor."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        def haar_su2():
            X = rng.normal(size=(2,2)) + 1j*rng.normal(size=(2,2))
            Q, _ = np.linalg.qr(X)
            return Q / np.linalg.det(Q) ** 0.5
        A = haar_su2()
        B = haar_su2()
        U_tensor = np.kron(A, B)
        O = (M_DAG @ U_tensor @ M).real
        A_rec, B_rec = _so4_to_su2_su2(O)
        # Up to global ±1 sign (since (-A) ⊗ (-B) = A ⊗ B).
        err = min(np.linalg.norm(U_tensor - np.kron(A_rec, B_rec)),
                  np.linalg.norm(U_tensor - np.kron(-A_rec, B_rec)))
        assert err < 1e-9


# ---- Reconstruction ----

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_u4_reconstruction(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))
    Q, _ = np.linalg.qr(A)
    err = reconstruction_error(Q)
    assert err < 1e-9


def test_identity_reconstruction():
    err = reconstruction_error(np.eye(4, dtype=complex))
    assert err < 1e-12


def test_cnot_reconstruction():
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    CNOT = (np.kron(np.diag([1,0]).astype(complex), I)
            + np.kron(np.diag([0,1]).astype(complex), X))
    assert reconstruction_error(CNOT) < 1e-9


def test_swap_reconstruction():
    SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    assert reconstruction_error(SWAP) < 1e-9


# ---- CNOT counts ----

def test_cnot_count_identity():
    assert cnot_count(np.eye(4, dtype=complex)) == 0


def test_cnot_count_cnot():
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    CNOT = (np.kron(np.diag([1,0]).astype(complex), I)
            + np.kron(np.diag([0,1]).astype(complex), X))
    assert cnot_count(CNOT) == 1


def test_cnot_count_iswap():
    """iSWAP needs 2 CNOTs (Vatan-Williams)."""
    ISWAP = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
    assert cnot_count(ISWAP) == 2


def test_cnot_count_swap():
    SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    assert cnot_count(SWAP) == 3


def test_cnot_count_tensor_product():
    """Pure tensor products (no entangling) need 0 CNOTs."""
    H_mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    T_prod = np.kron(H_mat, H_mat).astype(complex)
    assert cnot_count(T_prod) == 0


# ---- Validation ----

def test_kak_rejects_non_unitary():
    M_bad = np.eye(4) * 2
    with pytest.raises(ValueError):
        kak_decompose(M_bad)


def test_kak_rejects_wrong_shape():
    with pytest.raises(ValueError):
        kak_decompose(np.eye(2, dtype=complex))


def test_kak_returns_su2_factors():
    """K1_a, K1_b, K2_a, K2_b should all be unitary."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))
    Q, _ = np.linalg.qr(A)
    kak = kak_decompose(Q)
    for key in ["K1_a", "K1_b", "K2_a", "K2_b"]:
        K = kak[key]
        assert np.allclose(K @ K.conj().T, np.eye(2), atol=1e-10)

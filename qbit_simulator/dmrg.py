"""Density Matrix Renormalization Group (DMRG) ground-state solver.

Given an MPO H and a number of sites N, find an MPS approximation to the
ground state of H by two-site optimization sweeps:

  1. Initialize MPS randomly in left-canonical form.
  2. Sweep right-to-left, then left-to-right, then right-to-left, ...
  3. At each step, take two adjacent sites (i, i+1), form the local
     "effective Hamiltonian" H_eff acting on the (2*2*chi*chi)-dim block,
     find its lowest eigenvector via Lanczos, SVD-split back into two
     site tensors with truncation to max_chi.
  4. Stop when the energy stops decreasing.

This is the standard workhorse of 1D condensed-matter physics. For
gapped 1D systems with local Hamiltonians, DMRG converges to the exact
ground state in O(N) work per sweep with bond dimension bounded by the
area law. For critical systems chi grows polynomially in N.

What works:
  - Transverse-field Ising at any field strength
  - Heisenberg XXX/XXZ chains
  - Any 1D nearest-neighbor Hamiltonian expressible as a small-bond MPO

Currently limited to:
  - Two-site update (standard)
  - Real eigenvalues (Hermitian H assumed)
  - No explicit symmetry exploitation (could speed up further)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .mps import MPSState
from .mpo import MPO


# ---- random initialization ----

def _random_mps(n: int, max_chi: int, seed: int = 0) -> MPSState:
    """Random product-state seed in MPS form, then left-canonical sweep."""
    rng = np.random.default_rng(seed)
    mps = MPSState(n, max_chi=max_chi)
    # Random local rotation on each qubit so we don't start at |0...0>
    # (eigenstate of Z, which is a fixed point for some Hamiltonians).
    for q in range(n):
        theta = rng.uniform(0.1, np.pi - 0.1)
        phi = rng.uniform(0, 2 * np.pi)
        # An arbitrary single-qubit rotation.
        U = np.array([
            [np.cos(theta / 2), -np.exp(1j * phi) * np.sin(theta / 2)],
            [np.exp(-1j * phi) * np.sin(theta / 2),  np.cos(theta / 2)],
        ], dtype=np.complex128)
        mps.apply_1q(U, q)
    return mps


# ---- canonical form management ----

def _left_canonicalize_site(mps: MPSState, q: int) -> None:
    """Make site `q` left-canonical, absorbing the residual into site q+1."""
    if q >= mps.n - 1:
        return
    A = mps.tensors[q]                    # (l, p, r)
    l, p, r = A.shape
    M = A.reshape(l * p, r)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    chi = min(len(S), mps.max_chi)
    U  = U[:, :chi]
    S  = S[:chi]
    Vh = Vh[:chi, :]
    mps.tensors[q] = U.reshape(l, p, chi)
    # Absorb S @ Vh into the next site.
    SV = (np.diag(S) @ Vh).astype(np.complex128)
    nxt = mps.tensors[q + 1]
    mps.tensors[q + 1] = np.einsum("ab,bpr->apr", SV, nxt)


def _right_canonicalize_site(mps: MPSState, q: int) -> None:
    """Make site `q` right-canonical, absorbing the residual into site q-1."""
    if q <= 0:
        return
    A = mps.tensors[q]
    l, p, r = A.shape
    M = A.reshape(l, p * r)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    chi = min(len(S), mps.max_chi)
    U  = U[:, :chi]
    S  = S[:chi]
    Vh = Vh[:chi, :]
    mps.tensors[q] = Vh.reshape(chi, p, r)
    US = (U @ np.diag(S)).astype(np.complex128)
    prv = mps.tensors[q - 1]
    mps.tensors[q - 1] = np.einsum("lpa,ab->lpb", prv, US)


def _make_right_canonical(mps: MPSState) -> None:
    """Sweep right→left, making every site (except site 0) right-canonical."""
    for q in range(mps.n - 1, 0, -1):
        _right_canonicalize_site(mps, q)


# ---- environment tensors ----
#
# Left environment L[i]: contraction of the first i sites, indexed
# (l_bra, l_mpo, l_ket). L[0] is a (1,1,1) tensor of all-ones.
# Right environment R[i]: contraction of sites i..N-1, indexed
# (r_bra, r_mpo, r_ket). R[N] is a (1,1,1) tensor of all-ones.

def _contract_right_env(W: np.ndarray, A: np.ndarray,
                        Ac: np.ndarray, R: np.ndarray) -> np.ndarray:
    """R[q] from R[q+1] for site q. Output shape: (l_bra, l_mpo, l_ket).

    Letters:
        B = l_bra,   L = l_mpo,   K = l_ket
        i = p_in,    o = p_out
        b = r_bra,   w = r_mpo,   r = r_ket
    """
    # A(K,i,r) · R(b,w,r)  →  (K,i,b,w)
    tmp = np.einsum("Kir,bwr->Kibw", A, R)
    # tmp(K,i,b,w) · W(L,o,i,w)  →  (K,o,L,b)
    tmp = np.einsum("Kibw,Loiw->KoLb", tmp, W)
    # tmp(K,o,L,b) · Ac(B,o,b)  →  (B,L,K)
    return np.einsum("KoLb,Bob->BLK", tmp, Ac)


def _contract_left_env(L: np.ndarray, W: np.ndarray,
                       A: np.ndarray, Ac: np.ndarray) -> np.ndarray:
    """L[q+1] from L[q] for site q. Output shape: (r_bra, r_mpo, r_ket).

    Letters as in `_contract_right_env`.
    """
    # L(B,L,K) · A(K,i,r)  →  (B,L,i,r)
    tmp = np.einsum("BLK,Kir->BLir", L, A)
    # tmp(B,L,i,r) · W(L,o,i,w)  →  (B,o,w,r)
    tmp = np.einsum("BLir,Loiw->Bowr", tmp, W)
    # tmp(B,o,w,r) · Ac(B,o,b)  →  (b,w,r)
    return np.einsum("Bowr,Bob->bwr", tmp, Ac)


def _build_right_envs(mps: MPSState, mpo: MPO) -> list[np.ndarray]:
    """Compute R[N], R[N-1], ..., R[1]. R[i] has shape (r_bra, r_mpo, r_ket)."""
    R: list[np.ndarray] = [None] * (mps.n + 1)              # type: ignore
    R[mps.n] = np.ones((1, 1, 1), dtype=np.complex128)
    for q in range(mps.n - 1, 0, -1):
        A = mps.tensors[q]
        W = mpo.tensors[q]
        R[q] = _contract_right_env(W, A, np.conj(A), R[q + 1])
    return R


# ---- two-site effective Hamiltonian ----

def _apply_H_eff(L: np.ndarray, W1: np.ndarray, W2: np.ndarray,
                 R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply the local two-site H_eff to a two-site tensor T.

    Shapes (letters: B=l_bra, L=l_mpo, K=l_ket; i,j=phys_in; o,p=phys_out;
                     m=mid_mpo; b=r_bra, w=r_mpo, r=r_ket):
        L:  (B, L, K)
        W1: (L, o, i, m)
        W2: (m, p, j, w)
        R:  (b, w, r)
        T:  (K, i, j, r)
    Returns: same shape as T but indexed (B, o, p, b).
    """
    # L(B,L,K) · T(K,i,j,r)  →  (B,L,i,j,r)
    tmp = np.einsum("BLK,Kijr->BLijr", L, T)
    # tmp(B,L,i,j,r) · W1(L,o,i,m)  →  (B,o,m,j,r)
    tmp = np.einsum("BLijr,Loim->Bomjr", tmp, W1)
    # tmp(B,o,m,j,r) · W2(m,p,j,w)  →  (B,o,p,w,r)
    tmp = np.einsum("Bomjr,mpjw->Bopwr", tmp, W2)
    # tmp(B,o,p,w,r) · R(b,w,r)  →  (B,o,p,b)
    out = np.einsum("Bopwr,bwr->Bopb", tmp, R)
    return out


def _two_site_eigensolve(L: np.ndarray, W1: np.ndarray, W2: np.ndarray,
                         R: np.ndarray, T0: np.ndarray,
                         krylov_tol: float = 1e-10,
                         max_krylov: int = 30,
                         ) -> tuple[float, np.ndarray]:
    """Find the lowest eigenvalue + eigenvector of H_eff on the two-site block.

    Uses scipy.sparse.linalg.eigsh with a LinearOperator that never forms
    the full (4·chi²)² matrix.
    """
    shape = T0.shape
    dim = int(np.prod(shape))

    def matvec(v):
        v_t = v.reshape(shape)
        out = _apply_H_eff(L, W1, W2, R, v_t)
        return out.reshape(-1)

    H_op = LinearOperator((dim, dim), matvec=matvec, dtype=np.complex128)
    # Hermitian eigenvalue solve for the smallest eigenvalue.
    try:
        evals, evecs = eigsh(H_op, k=1, which="SA", v0=T0.reshape(-1),
                             tol=krylov_tol, maxiter=max_krylov * dim)
    except Exception:
        # Fall back to dense if Lanczos struggles (rare, only for tiny problems).
        H_dense = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            e = np.zeros(dim, dtype=np.complex128); e[i] = 1.0
            H_dense[:, i] = matvec(e)
        evals_d, evecs_d = np.linalg.eigh((H_dense + H_dense.conj().T) / 2)
        return float(evals_d[0]), evecs_d[:, 0].reshape(shape)
    return float(np.real(evals[0])), evecs[:, 0].reshape(shape)


# ---- main DMRG loop ----

def dmrg(
    mpo: MPO,
    max_chi: int = 32,
    max_sweeps: int = 20,
    tol: float = 1e-8,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[MPSState, float, list[float]]:
    """Find the ground state of `mpo` via two-site DMRG.

    Returns (mps, ground_energy, energy_trace).
    """
    n = mpo.n
    # Initialize MPS, then bring to right-canonical form (so site 0 is the
    # "active" site and all sites to its right are right-canonical).
    mps = _random_mps(n, max_chi=max_chi, seed=seed)
    _make_right_canonical(mps)

    energy_trace: list[float] = []
    prev_energy: float = float("inf")
    R_envs = _build_right_envs(mps, mpo)
    L_env  = np.ones((1, 1, 1), dtype=np.complex128)
    Ls: list[np.ndarray] = [None] * (n + 1)        # type: ignore
    Ls[0] = L_env

    for sweep in range(max_sweeps):
        # --- Right-going sweep: optimize pairs (0,1), (1,2), ..., (n-2, n-1). ---
        for i in range(n - 1):
            L_i  = Ls[i]
            R_ip2 = R_envs[i + 2]
            W1, W2 = mpo.tensors[i], mpo.tensors[i + 1]
            A1, A2 = mps.tensors[i], mps.tensors[i + 1]
            T0 = np.einsum("lpm,mqr->lpqr", A1, A2)
            energy, T_new = _two_site_eigensolve(L_i, W1, W2, R_ip2, T0)
            # SVD-split T_new back into two site tensors.
            l, p1, p2, r = T_new.shape
            M = T_new.reshape(l * p1, p2 * r)
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            chi = min(len(S), max_chi)
            U, S, Vh = U[:, :chi], S[:chi], Vh[:chi, :]
            # Left site goes left-canonical; multiply S into the right site.
            mps.tensors[i]     = U.reshape(l, p1, chi).astype(np.complex128)
            mps.tensors[i + 1] = (np.diag(S) @ Vh).reshape(chi, p2, r).astype(np.complex128)
            # Update left environment.
            Ls[i + 1] = _contract_left_env(
                Ls[i], mpo.tensors[i],
                mps.tensors[i], np.conj(mps.tensors[i])
            )

        # --- Left-going sweep ---
        # Rebuild right environments after the right-going sweep changed every site.
        R_envs = _build_right_envs(mps, mpo)
        for i in range(n - 2, -1, -1):
            L_i  = Ls[i]
            R_ip2 = R_envs[i + 2]
            W1, W2 = mpo.tensors[i], mpo.tensors[i + 1]
            A1, A2 = mps.tensors[i], mps.tensors[i + 1]
            T0 = np.einsum("lpm,mqr->lpqr", A1, A2)
            energy, T_new = _two_site_eigensolve(L_i, W1, W2, R_ip2, T0)
            l, p1, p2, r = T_new.shape
            M = T_new.reshape(l * p1, p2 * r)
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            chi = min(len(S), max_chi)
            U, S, Vh = U[:, :chi], S[:chi], Vh[:chi, :]
            # Right site goes right-canonical; multiply S into the left site.
            mps.tensors[i + 1] = Vh.reshape(chi, p2, r).astype(np.complex128)
            mps.tensors[i]     = (U @ np.diag(S)).reshape(l, p1, chi).astype(np.complex128)
            # Update right environment.
            R_envs[i + 1] = _contract_right_env(
                mpo.tensors[i + 1],
                mps.tensors[i + 1], np.conj(mps.tensors[i + 1]),
                R_envs[i + 2],
            )

        energy_trace.append(energy)
        if verbose:
            print(f"  sweep {sweep:>3}: E = {energy:.10f}  "
                  f"chi_max = {max(mps.bond_dimensions() or [1])}")
        if abs(prev_energy - energy) < tol:
            break
        prev_energy = energy

        # Rebuild Ls (cleared after left-going sweep).
        Ls = [None] * (n + 1)                              # type: ignore
        Ls[0] = np.ones((1, 1, 1), dtype=np.complex128)

    return mps, energy, energy_trace

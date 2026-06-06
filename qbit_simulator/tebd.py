"""Time-Evolving Block Decimation (TEBD) — real-time evolution of an MPS.

Given a 1D Hamiltonian H = Σ_i h_{i, i+1} (sum of nearest-neighbor 2-site
terms), we want to compute |ψ(t)⟩ = exp(-i H t) |ψ(0)⟩ for some initial
MPS state. The trick:

  1. Split H into "even bonds" (i = 0, 2, 4, ...) and "odd bonds"
     (i = 1, 3, 5, ...). Within each set, terms commute trivially.
  2. Trotter decomposition (order 2, "Strang splitting"):
         exp(-i H dt) ≈ exp(-i H_odd dt/2) · exp(-i H_even dt) · exp(-i H_odd dt/2)
     Error per step: O(dt³).
  3. Each exp(-i h_{i,i+1} dt) is just a 4×4 unitary applied as a 2-qubit
     gate on adjacent MPS sites — exactly what our MPSState already does.

The bond dimension may grow during evolution as entanglement builds up.
If it exceeds max_chi, the SVD truncation in `apply_2q_adjacent` keeps the
top singular values. For low-entanglement physics (gapped ground states,
short times) this stays manageable; for long times and chaotic dynamics
it eventually fails — that's the fundamental limit of TEBD.

Built-in term builders for the two textbook 1D models:
  - tfim_terms(n, J, h): -J Z_i Z_{i+1} - h X_i
  - heisenberg_terms(n, J): J (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

The 1-site terms (like -h X_i in TFIM) are split half-and-half into the
adjacent bonds so everything is uniformly 2-site.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .gates import X, Y, Z, I2
from .mps import MPSState


# ---- term builders ----

def tfim_terms(n: int, J: float = 1.0, h: float = 1.0) -> list[np.ndarray]:
    """List of 4×4 2-site Hamiltonian terms for TFIM on n sites.

    H = -J Σ Z_i Z_{i+1} - h Σ X_i

    The -h X_i on-site terms are split: each interior site contributes
    -h/2 to its left-bond and -h/2 to its right-bond. End sites contribute
    the full -h to their only bond.
    """
    if n < 2:
        raise ValueError("need at least 2 sites")
    terms: list[np.ndarray] = []
    ZZ = np.kron(Z, Z)
    XI = np.kron(X, I2)
    IX = np.kron(I2, X)
    for i in range(n - 1):
        # X weight for left site of this bond:
        left_w  = 1.0 if i == 0 else 0.5
        # X weight for right site of this bond:
        right_w = 1.0 if i == n - 2 else 0.5
        h_local = -J * ZZ - h * left_w * XI - h * right_w * IX
        terms.append(h_local.astype(np.complex128))
    return terms


def heisenberg_terms(n: int, J: float = 1.0) -> list[np.ndarray]:
    """List of 4×4 2-site terms for Heisenberg XXX on n sites.

    H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    """
    if n < 2:
        raise ValueError("need at least 2 sites")
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    ZZ = np.kron(Z, Z)
    h_local = J * (XX + YY + ZZ)
    return [h_local.astype(np.complex128) for _ in range(n - 1)]


# ---- the TEBD sweep ----

def _bond_unitary(h_local: np.ndarray, dt: float) -> np.ndarray:
    """exp(-i · h_local · dt) -- a 4×4 unitary."""
    return expm(-1j * dt * h_local)


def tebd_step(mps: MPSState, terms: list[np.ndarray], dt: float,
              order: int = 2) -> None:
    """Apply one Trotter step exp(-i H dt) in place.

    order=1: Lie splitting   exp(-iH_odd dt) · exp(-iH_even dt)
    order=2: Strang splitting (default, O(dt³) error per step)
    """
    n_bonds = mps.n - 1
    even_bonds = list(range(0, n_bonds, 2))
    odd_bonds  = list(range(1, n_bonds, 2))

    if order == 1:
        # exp(-iH_even dt) then exp(-iH_odd dt) (order doesn't matter at O(dt²))
        for i in even_bonds:
            mps.apply_2q_adjacent(_bond_unitary(terms[i], dt), i)
        for i in odd_bonds:
            mps.apply_2q_adjacent(_bond_unitary(terms[i], dt), i)
    elif order == 2:
        # Strang: odd(dt/2) · even(dt) · odd(dt/2)
        for i in odd_bonds:
            mps.apply_2q_adjacent(_bond_unitary(terms[i], dt / 2), i)
        for i in even_bonds:
            mps.apply_2q_adjacent(_bond_unitary(terms[i], dt), i)
        for i in odd_bonds:
            mps.apply_2q_adjacent(_bond_unitary(terms[i], dt / 2), i)
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")


def tebd_evolve(
    mps: MPSState,
    terms: list[np.ndarray],
    total_time: float,
    dt: float = 0.05,
    order: int = 2,
    observables: list | None = None,
    record_every: int = 1,
) -> dict:
    """Evolve `mps` under H = Σ terms for time `total_time` in place.

    Args:
        mps: initial state (modified in place).
        terms: list of 4×4 2-site Hamiltonian terms.
        total_time: target evolution time.
        dt: Trotter step size.
        order: Trotter order (1 or 2).
        observables: optional list of callables taking the MPS and returning
            a scalar; values are recorded at every `record_every` steps.
        record_every: how often (in steps) to record observables.

    Returns:
        dict with keys:
            'times'       : array of recorded times
            'bond_dims'   : list of (chi_max) per recording
            'observables' : dict mapping each observable's str(callable) to
                            an array of recorded values
            'n_steps'     : how many Trotter steps were applied
    """
    if mps.n - 1 != len(terms):
        raise ValueError(
            f"got {len(terms)} terms for {mps.n} sites; need {mps.n - 1}"
        )
    n_steps = int(np.round(total_time / dt))
    times = [0.0]
    chi_history = [max(mps.bond_dimensions() or [1])]
    obs_records: dict[str, list[float]] = {}
    if observables:
        for fn in observables:
            obs_records[str(fn)] = [float(fn(mps))]

    for k in range(1, n_steps + 1):
        tebd_step(mps, terms, dt, order=order)
        if k % record_every == 0 or k == n_steps:
            times.append(k * dt)
            chi_history.append(max(mps.bond_dimensions() or [1]))
            if observables:
                for fn in observables:
                    obs_records[str(fn)].append(float(fn(mps)))

    return {
        "times": np.array(times),
        "bond_dims": np.array(chi_history),
        "observables": {k: np.array(v) for k, v in obs_records.items()},
        "n_steps": n_steps,
    }


# ---- observables ----

def site_z_expectation(mps: MPSState, site: int) -> float:
    """⟨Z_site⟩ — magnetization at a single site, computed natively in MPS.

    Cost: O(N · χ³). Avoids the 2^N dense conversion entirely, so it works
    for any chain length our MPS can hold.
    """
    return _site_op_expectation(mps, site, Z)


def _site_op_expectation(mps: MPSState, site: int,
                         op: np.ndarray) -> float:
    """⟨ψ|op_site|ψ⟩ for a 2×2 Hermitian op acting on a single site."""
    n = mps.n
    if not (0 <= site < n):
        raise IndexError(f"site {site} out of range")

    # Left environment: contract sites 0..site-1.
    left = np.ones((1, 1), dtype=np.complex128)       # (l_bra, l_ket)
    for k in range(site):
        A = mps.tensors[k]
        tmp = np.einsum("bk,kpr->bpr", left, A)       # (l_bra, p, r_ket)
        left = np.einsum("bpr,bps->sr", tmp, np.conj(A))   # (r_bra, r_ket)

    # Right environment: contract sites site+1..n-1.
    # right[r_bra, r_ket] starts as a (1,1) scalar.
    right = np.ones((1, 1), dtype=np.complex128)
    for k in range(n - 1, site, -1):
        A = mps.tensors[k]                            # (l_ket=K, p, r_ket=r)
        # tmp[K, p, b] = Σ_r A[K, p, r] · right[b, r]
        tmp = np.einsum("Kpr,br->Kpb", A, right)
        # new_right[B, K] = Σ_{p, b} tmp[K, p, b] · conj(A)[B, p, b]
        right = np.einsum("Kpb,Bpb->BK", tmp, np.conj(A))

    # Now combine with the site-q tensor and the operator.
    A = mps.tensors[site]                              # (l_ket, p_in, r_ket)
    A_op = np.einsum("ij,ljr->lir", op, A)             # apply op on physical leg
    tmp = np.einsum("bl,lpr->bpr", left, A_op)         # (l_bra, p_out, r_ket)
    tmp = np.einsum("bpr,bps->sr", tmp, np.conj(A))    # (r_bra, r_ket)
    val = np.einsum("br,br->", tmp, right)             # scalar
    return float(np.real(val))


def total_z_expectation(mps: MPSState) -> float:
    """⟨Σ_i Z_i⟩ — total magnetization."""
    return sum(site_z_expectation(mps, q) for q in range(mps.n))

"""General molecular electronic-structure Hamiltonians.

Given the one- and two-electron MO integrals h_pq and (pq|rs) of a
molecule, build the second-quantized fermionic Hamiltonian:

    H = sum_{pq,σ}   h_pq c†_{pσ} c_{qσ}
      + (1/2) sum_{pqrs,στ}  (pq|rs) c†_{pσ} c†_{rτ} c_{sτ} c_{qσ}
      + V_nn

Then Jordan-Wigner map to a Pauli operator on 2 · n_orbitals qubits
(one qubit per spin-orbital).

Convention: spin-orbital index k = 2*p + σ where p is the spatial
orbital and σ ∈ {0, 1} for α/β spins. So mode 0 = orbital 0 α, mode 1
= orbital 0 β, mode 2 = orbital 1 α, etc.

We validate this against the existing H₂ STO-3G implementation in
`h2_sto3g.py` — building the full 4-qubit fermionic Hamiltonian from
MO integrals and checking that exact diagonalization gives the same
ground-state energy as the 2-qubit reduced version.

For LiH or larger systems, the user supplies their own MO integrals
(computed e.g. with PySCF). We provide a `LiHIntegralsStub` placeholder
that documents the required interface.
"""

from __future__ import annotations

import numpy as np

from ..fermion import FermionOp
from ..pauli import PauliOp


# ----------------------------------------------------------------------------
# Generic molecular Hamiltonian
# ----------------------------------------------------------------------------

def molecular_hamiltonian(
    h_mo: np.ndarray,        # (n_orb, n_orb) one-electron integrals
    eri_mo: np.ndarray,      # (n_orb, n_orb, n_orb, n_orb) ERIs in chemist (pq|rs)
    V_nn: float = 0.0,       # nuclear repulsion
) -> FermionOp:
    """Build the full second-quantized molecular Hamiltonian as a FermionOp.

    Spin-orbital ordering: mode 2p+σ for spatial orbital p, spin σ.
    """
    n_orb = h_mo.shape[0]
    H = (V_nn + 0j) * FermionOp.identity()

    # One-electron part.
    for p in range(n_orb):
        for q in range(n_orb):
            if abs(h_mo[p, q]) < 1e-14:
                continue
            for sigma in (0, 1):
                H = H + (complex(h_mo[p, q])
                         * (FermionOp.cdag(2 * p + sigma)
                            * FermionOp.c(2 * q + sigma)))

    # Two-electron part: (1/2) sum_{pqrs,στ} (pq|rs) c†_pσ c†_rτ c_sτ c_qσ.
    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    coef = 0.5 * eri_mo[p, q, r, s]
                    if abs(coef) < 1e-14:
                        continue
                    for sigma in (0, 1):
                        for tau in (0, 1):
                            ps = 2 * p + sigma
                            qs = 2 * q + sigma
                            rs_ = 2 * r + tau
                            ss = 2 * s + tau
                            term = (FermionOp.cdag(ps) * FermionOp.cdag(rs_)
                                    * FermionOp.c(ss) * FermionOp.c(qs))
                            H = H + complex(coef) * term
    return H


def molecular_hamiltonian_pauli(
    h_mo: np.ndarray, eri_mo: np.ndarray, V_nn: float = 0.0,
) -> PauliOp:
    """Build the Hamiltonian as a PauliOp via JW (convenient for VQE)."""
    H_ferm = molecular_hamiltonian(h_mo, eri_mo, V_nn)
    n_modes = 2 * h_mo.shape[0]
    return H_ferm.to_pauli_op(n_modes)


# ----------------------------------------------------------------------------
# Particle-number-sector projection
# ----------------------------------------------------------------------------

def project_to_n_electron_sector(
    H_matrix: np.ndarray, n_qubits: int, n_electrons: int,
) -> tuple[np.ndarray, list[int]]:
    """Restrict a dense Hamiltonian on n_qubits spin-orbitals to the
    subspace with exactly n_electrons occupied.

    Returns:
        (H_sector, indices) — the projected matrix plus the list of
        basis-state indices in the sector.
    """
    indices = [k for k in range(2 ** n_qubits) if bin(k).count("1") == n_electrons]
    H_sector = H_matrix[np.ix_(indices, indices)]
    H_sector = 0.5 * (H_sector + H_sector.conj().T)
    return H_sector, indices


# ----------------------------------------------------------------------------
# H₂ STO-3G via the generic framework (for validation)
# ----------------------------------------------------------------------------

def h2_sto3g_mo_integrals(R: float) -> dict:
    """Extract H₂ STO-3G MO integrals using the existing h2_sto3g module.

    Returns dict with h_mo (2×2), eri_mo (2×2×2×2), V_nn.
    """
    from .h2_sto3g import _ao_integrals, _mo_transform
    ao = _ao_integrals(R)
    C, h_mo, eri_mo = _mo_transform(ao)
    return {
        "h_mo":   h_mo,
        "eri_mo": eri_mo,
        "V_nn":   ao["V_nn"],
    }


def h2_sto3g_full_hamiltonian(R: float) -> PauliOp:
    """Build the FULL 4-qubit fermionic H₂ STO-3G Hamiltonian (vs the
    2-qubit reduced version in h2_sto3g.py)."""
    ints = h2_sto3g_mo_integrals(R)
    return molecular_hamiltonian_pauli(
        ints["h_mo"], ints["eri_mo"], ints["V_nn"],
    )


def h2_full_fci_energy(R: float) -> float:
    """FCI energy of H₂ STO-3G from the 4-qubit fermionic Hamiltonian
    (restricted to the 2-electron sector)."""
    H_pauli = h2_sto3g_full_hamiltonian(R)
    # Build dense matrix.
    from .ucc import _pauli_op_to_matrix
    H_mat = _pauli_op_to_matrix(H_pauli, n=4)
    H_2e, _ = project_to_n_electron_sector(H_mat, n_qubits=4, n_electrons=2)
    return float(np.linalg.eigvalsh(H_2e)[0])


# ----------------------------------------------------------------------------
# LiH STO-3G interface stub
# ----------------------------------------------------------------------------

def lih_sto3g_integrals_stub() -> dict:
    """Placeholder: structure of LiH STO-3G MO integrals.

    LiH has 3 spatial orbitals in STO-3G (Li 1s, Li 2s, H 1s), giving
    6 spin-orbitals and a 4-electron problem. The real integrals must
    be computed (e.g. via PySCF) and dropped in as h_mo (3×3) and
    eri_mo (3×3×3×3) numpy arrays — then `molecular_hamiltonian_pauli`
    will build the 6-qubit Hamiltonian.

    This stub documents the expected shape and lets downstream code
    test the build pipeline without requiring real integrals.
    """
    # Random but Hermitian h, symmetric ERI — NOT real LiH values.
    rng = np.random.default_rng(0)
    n_orb = 3
    h_mo = 0.1 * rng.normal(size=(n_orb, n_orb))
    h_mo = 0.5 * (h_mo + h_mo.T)
    # Diagonal: rough order-of-magnitude.
    np.fill_diagonal(h_mo, [-2.4, -0.5, 0.1])
    # ERI: full 8-fold symmetric tensor with small random values.
    eri_unique = 0.05 * rng.normal(size=(n_orb, n_orb, n_orb, n_orb))
    # Symmetrize: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq).
    eri = np.zeros_like(eri_unique)
    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    eri[p, q, r, s] = (
                        eri_unique[p, q, r, s]
                        + eri_unique[q, p, r, s]
                        + eri_unique[p, q, s, r]
                        + eri_unique[r, s, p, q]
                    ) / 4
    np.fill_diagonal(eri.reshape(9, 9), np.abs(np.diag(eri.reshape(9, 9))))
    return {
        "h_mo":      h_mo,
        "eri_mo":    eri,
        "V_nn":      0.9922,    # ~1/R_bohr at R = 1.6 Å
        "stub":      True,
        "note":      "Random integrals for INTERFACE TESTING only — "
                     "drop in real PySCF values for LiH chemistry.",
    }

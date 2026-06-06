"""H₂-like Hamiltonian for VQE demonstration.

Two-site Hubbard model with H₂-like parameters. The Hamiltonian is built in
the 4-dimensional singlet+triplet space of two electrons on two sites and
decomposed into 2-qubit Pauli strings.

Physical shape:
  - Repulsive wall at small R (nuclear repulsion dominates)
  - Binding minimum near R ≈ 0.74 Å (~0.15 Hartree below dissociation)
  - Asymptote E → -1.0 Hartree as R → ∞ (two separated H atoms)

NOT a quantitatively exact STO-3G calculation — the literature coefficients
require precomputed Gaussian integrals from a quantum chemistry package.
This model captures the qualitative dissociation curve and is sufficient to
demonstrate that VQE finds the true ground state of the given Hamiltonian.
"""

from __future__ import annotations

import numpy as np

from ..gates import I2, X, Y, Z
from ..pauli import PauliOp

_BOHR_PER_ANGSTROM = 1.0 / 0.5291772109


def _hopping(R_bohr: float) -> float:
    """Hopping integral t(R). Goes like the square of a 1s Slater overlap."""
    return 0.70 * np.exp(-R_bohr) * (1 + R_bohr + R_bohr**2 / 3.0)


def _h2_ci_matrix(R: float) -> np.ndarray:
    """4x4 Hamiltonian in the basis {|cov⟩, |ionic_+⟩, |ionic_-⟩, |triplet⟩}.

    |cov⟩       = covalent singlet (one electron on each site)
    |ionic_±⟩   = symmetric/antisymmetric ionic singlets (both electrons on
                  the same site)
    |triplet⟩   = S_z=0 component of the triplet (decoupled by symmetry)
    """
    R_bohr = R * _BOHR_PER_ANGSTROM
    t = _hopping(R_bohr)
    U = 0.40           # on-site Coulomb repulsion
    eps = -0.5         # single-atom orbital energy (-> dissociation at -1.0)
    V_nn = 1.0 / R_bohr

    H = np.zeros((4, 4), dtype=np.complex128)
    H[0, 0] = 2 * eps + V_nn
    H[1, 1] = 2 * eps + U + V_nn
    H[2, 2] = 2 * eps + U + V_nn
    H[3, 3] = 2 * eps + V_nn
    H[0, 1] = -2 * t
    H[1, 0] = -2 * t
    return H


def _pauli_decompose_2q(H: np.ndarray, tol: float = 1e-10) -> list[tuple[complex, str]]:
    """Decompose a 4x4 Hermitian matrix into a sum of weighted Pauli strings."""
    paulis = {"I": I2, "X": X, "Y": Y, "Z": Z}
    terms: list[tuple[complex, str]] = []
    for a in "IXYZ":
        for b in "IXYZ":
            P = np.kron(paulis[a], paulis[b])
            coeff = np.trace(P.conj().T @ H) / 4
            if abs(coeff) > tol:
                terms.append((complex(coeff), a + b))
    return terms


def h2_hamiltonian(R: float) -> PauliOp:
    """Build the 2-qubit H₂-like Hamiltonian at bond length R (Å)."""
    H = _h2_ci_matrix(R)
    return PauliOp(_pauli_decompose_2q(H))


def h2_coefficients(R: float) -> dict[str, float]:
    return {s: float(np.real(c)) for c, s in h2_hamiltonian(R).terms}


def bond_length_range() -> tuple[float, float]:
    return 0.30, 3.00

"""Real H₂ Hamiltonian in the STO-3G basis, computed from scratch.

We compute one- and two-electron integrals analytically over contracted
Gaussian 1s functions, transform to the molecular-orbital (bonding/anti-
bonding) basis, build the 4-spin-orbital configuration-interaction
Hamiltonian, and decompose into 2-qubit Pauli strings after symmetry
reduction.

STO-3G parameters for a hydrogenic 1s (zeta=1.0): three primitive Gaussians
contracted with the standard STO-3G coefficients.

Reference matches at R = 0.7414 Å:
    E_HF  ≈ -1.117 Hartree
    E_FCI ≈ -1.137 Hartree

These are the canonical numbers in any quantum-chemistry textbook.
"""

from __future__ import annotations

from math import erf

import numpy as np

from ..gates import I2, X, Y, Z
from ..pauli import PauliOp


# ---- STO-3G primitives for H (zeta = 1.0) ----
# Standard exponents (alpha) and coefficients (d) for a single 1s STO
# represented as a sum of three primitive Gaussians.
_STO3G_ALPHA = np.array([3.42525091, 0.62391373, 0.16885540])
_STO3G_COEFF = np.array([0.15432897, 0.53532814, 0.44463454])


def _norm(alpha: float) -> float:
    """Normalization constant for a primitive s-type Gaussian."""
    return (2.0 * alpha / np.pi) ** 0.75


# ---- Boys function F_0 ----

def _boys_F0(t: float) -> float:
    if t < 1e-12:
        return 1.0 - t / 3.0
    return 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))


# ---- primitive integrals (s-s only, both 1s) ----

def _overlap_p(a: float, A: np.ndarray, b: float, B: np.ndarray) -> float:
    p = a + b
    mu = a * b / p
    return (np.pi / p) ** 1.5 * np.exp(-mu * np.dot(A - B, A - B))


def _kinetic_p(a: float, A: np.ndarray, b: float, B: np.ndarray) -> float:
    p = a + b
    mu = a * b / p
    r2 = np.dot(A - B, A - B)
    return mu * (3.0 - 2.0 * mu * r2) * _overlap_p(a, A, b, B)


def _nuclear_p(a: float, A: np.ndarray, b: float, B: np.ndarray, C: np.ndarray) -> float:
    p = a + b
    mu = a * b / p
    P = (a * A + b * B) / p
    PC2 = np.dot(P - C, P - C)
    return -2.0 * np.pi / p * np.exp(-mu * np.dot(A - B, A - B)) * _boys_F0(p * PC2)


def _eri_p(a: float, A: np.ndarray, b: float, B: np.ndarray,
           c: float, C: np.ndarray, d: float, D: np.ndarray) -> float:
    p = a + b
    q = c + d
    mu_p = a * b / p
    mu_q = c * d / q
    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q
    alpha = p * q / (p + q)
    PQ2 = np.dot(P - Q, P - Q)
    pref = 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
    return (pref
            * np.exp(-mu_p * np.dot(A - B, A - B))
            * np.exp(-mu_q * np.dot(C - D, C - D))
            * _boys_F0(alpha * PQ2))


# ---- contracted integrals ----

def _contract2(fn, A: np.ndarray, B: np.ndarray) -> float:
    """Contract a two-center primitive integral over the STO-3G exponents."""
    total = 0.0
    for i, ai in enumerate(_STO3G_ALPHA):
        for j, aj in enumerate(_STO3G_ALPHA):
            total += (_STO3G_COEFF[i] * _STO3G_COEFF[j]
                      * _norm(ai) * _norm(aj)
                      * fn(ai, A, aj, B))
    return total


def _contract2_nuclear(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    total = 0.0
    for i, ai in enumerate(_STO3G_ALPHA):
        for j, aj in enumerate(_STO3G_ALPHA):
            total += (_STO3G_COEFF[i] * _STO3G_COEFF[j]
                      * _norm(ai) * _norm(aj)
                      * _nuclear_p(ai, A, aj, B, C))
    return total


def _contract4(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> float:
    """Contract the two-electron integral (AB|CD)."""
    total = 0.0
    for i, ai in enumerate(_STO3G_ALPHA):
        for j, aj in enumerate(_STO3G_ALPHA):
            for k, ak in enumerate(_STO3G_ALPHA):
                for l, al in enumerate(_STO3G_ALPHA):
                    total += (_STO3G_COEFF[i] * _STO3G_COEFF[j]
                              * _STO3G_COEFF[k] * _STO3G_COEFF[l]
                              * _norm(ai) * _norm(aj) * _norm(ak) * _norm(al)
                              * _eri_p(ai, A, aj, B, ak, C, al, D))
    return total


# ---- AO -> MO transform and Hamiltonian construction ----

def _ao_integrals(R: float) -> dict:
    """Compute all AO integrals at bond length R (Å). Internally uses Bohr."""
    R_bohr = R / 0.5291772109
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R_bohr])

    S_AA = _contract2(_overlap_p, A, A)
    S_AB = _contract2(_overlap_p, A, B)
    S_BB = _contract2(_overlap_p, B, B)
    T_AA = _contract2(_kinetic_p, A, A)
    T_AB = _contract2(_kinetic_p, A, B)
    T_BB = _contract2(_kinetic_p, B, B)
    V_AA = _contract2_nuclear(A, A, A) + _contract2_nuclear(A, A, B)
    V_AB = _contract2_nuclear(A, B, A) + _contract2_nuclear(A, B, B)
    V_BB = _contract2_nuclear(B, B, A) + _contract2_nuclear(B, B, B)

    h_AA = T_AA + V_AA
    h_AB = T_AB + V_AB
    h_BB = T_BB + V_BB

    # Two-electron integrals (only the symmetry-unique ones).
    eri_AAAA = _contract4(A, A, A, A)
    eri_AABB = _contract4(A, A, B, B)
    eri_ABAB = _contract4(A, B, A, B)
    eri_AAAB = _contract4(A, A, A, B)
    eri_ABBB = _contract4(A, B, B, B)

    V_nn = 1.0 / R_bohr

    return {
        "S": np.array([[S_AA, S_AB], [S_AB, S_BB]]),
        "h": np.array([[h_AA, h_AB], [h_AB, h_BB]]),
        "eri": {
            "AAAA": eri_AAAA, "BBBB": eri_AAAA,  # symmetric H₂
            "AABB": eri_AABB, "BBAA": eri_AABB,
            "ABAB": eri_ABAB, "BABA": eri_ABAB,
            "AABA": eri_AAAB, "ABAA": eri_AAAB,
            "BBAB": eri_ABBB, "ABBB": eri_ABBB,
            "ABBA": eri_ABAB, "BAAB": eri_ABAB,
            "BABB": eri_ABBB, "BBBA": eri_ABBB,
            "AAAB": eri_AAAB, "BABB_alt": eri_ABBB,
        },
        "V_nn": V_nn,
    }


def _mo_transform(ao: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lowdin-orthogonalize AOs, then diagonalize core Hamiltonian to get
    bonding/antibonding MOs. Returns (C, h_mo, eri_mo) — but for symmetric
    H₂ the MOs are simply (φ_A ± φ_B) / √(2(1 ± S))."""
    S = ao["S"]
    h = ao["h"]
    s = S[0, 1]
    norm_g = 1.0 / np.sqrt(2 * (1 + s))
    norm_u = 1.0 / np.sqrt(2 * (1 - s))
    # C[:, i] is MO i in terms of AOs
    C = np.array([
        [norm_g,  norm_u],
        [norm_g, -norm_u],
    ])
    h_mo = C.T @ h @ C

    # Transform two-electron integrals into MO basis. Use eight-fold symmetry.
    # For our symmetric H₂ with AOs labeled 0=A, 1=B:
    # (pq|rs)_AO -> (PQ|RS)_MO = Σ C_{p,P} C_{q,Q} C_{r,R} C_{s,S} (pq|rs)
    eri_ao = np.zeros((2, 2, 2, 2))
    eri_AAAA = ao["eri"]["AAAA"]
    eri_AABB = ao["eri"]["AABB"]
    eri_ABAB = ao["eri"]["ABAB"]
    eri_AAAB = ao["eri"]["AAAB"]
    for p in range(2):
        for q in range(2):
            for r in range(2):
                for ss in range(2):
                    a, b, c, d = p, q, r, ss
                    # Look up by the multiset {(a,b),(c,d)} pattern.
                    # In H₂: equivalent under A↔B swap of any *pair*.
                    same_pair = (a == b) and (c == d)
                    cross = (a != b) and (c != d)
                    if same_pair and a == c:
                        eri_ao[p, q, r, ss] = eri_AAAA
                    elif same_pair and a != c:
                        eri_ao[p, q, r, ss] = eri_AABB
                    elif cross:
                        eri_ao[p, q, r, ss] = eri_ABAB
                    else:
                        eri_ao[p, q, r, ss] = eri_AAAB
    eri_mo = np.einsum("ip,jq,kr,ls,ijkl->pqrs", C, C, C, C, eri_ao)
    return C, h_mo, eri_mo


def _build_ci_4x4(h_mo: np.ndarray, eri_mo: np.ndarray, V_nn: float) -> np.ndarray:
    """Build the singlet+triplet 4x4 CI Hamiltonian for two electrons in two MOs.

    Basis: {|11⟩_singlet, |22⟩_singlet, |12⟩_singlet, |12⟩_triplet}.
    (Where 1 = bonding MO, 2 = antibonding MO; doubly-occupied states are spin
    singlets automatically.)
    """
    h11, h22, h12 = h_mo[0, 0], h_mo[1, 1], h_mo[0, 1]
    # Two-electron integrals in chemist's notation (pq|rs) = ⟨pr|qs⟩.
    J11 = eri_mo[0, 0, 0, 0]
    J22 = eri_mo[1, 1, 1, 1]
    J12 = eri_mo[0, 0, 1, 1]
    K12 = eri_mo[0, 1, 0, 1]

    H = np.zeros((4, 4), dtype=np.complex128)
    # |11⟩_s — both electrons in bonding MO
    H[0, 0] = 2 * h11 + J11 + V_nn
    # |22⟩_s — both in antibonding
    H[1, 1] = 2 * h22 + J22 + V_nn
    # |12⟩_singlet
    H[2, 2] = h11 + h22 + J12 + K12 + V_nn
    # |12⟩_triplet (S_z = 0 component)
    H[3, 3] = h11 + h22 + J12 - K12 + V_nn
    # Coupling between |11⟩ and |22⟩: 2 * ⟨11|H_2e|22⟩ = (12|12) ≡ K12.
    H[0, 1] = K12; H[1, 0] = K12
    return H


def _pauli_decompose_2q(H: np.ndarray, tol: float = 1e-12):
    paulis = {"I": I2, "X": X, "Y": Y, "Z": Z}
    terms = []
    for a in "IXYZ":
        for b in "IXYZ":
            P = np.kron(paulis[a], paulis[b])
            coeff = np.trace(P.conj().T @ H) / 4
            if abs(coeff) > tol:
                terms.append((complex(coeff), a + b))
    return terms


def h2_sto3g_hamiltonian(R: float) -> PauliOp:
    """Build the STO-3G H₂ Hamiltonian at bond length R (Å) as a 2-qubit PauliOp."""
    ao = _ao_integrals(R)
    C, h_mo, eri_mo = _mo_transform(ao)
    H4 = _build_ci_4x4(h_mo, eri_mo, ao["V_nn"])
    return PauliOp(_pauli_decompose_2q(H4))


def h2_sto3g_energy(R: float) -> float:
    """Ground state energy at R (Å) from exact diagonalization."""
    return float(h2_sto3g_hamiltonian(R).ground_state()[0])

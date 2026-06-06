"""HHL algorithm — Harrow, Hassidim, Lloyd 2009.

Solve a linear system A|x⟩ = |b⟩ for a Hermitian matrix A. The quantum
output |x⟩ is proportional to A⁻¹|b⟩.

Algorithm:
    1. Prepare |b⟩ in a "b register".
    2. Apply QPE with U = exp(iAt) onto a clock register. The clock now
       encodes the eigenvalues λ_j of A.
    3. Conditional rotation on an ancilla:  Ry(2·arcsin(C/λ_j)) for each
       clock value, where C ≤ λ_min ensures the argument is in [0, 1].
    4. Inverse QPE (uncomputes the clock register back to |0⟩).
    5. Post-select on the ancilla being |1⟩. The b-register is now |x⟩.

This is the textbook formulation. We implement it for small dense
Hermitian matrices (2×2, 4×4) where each step is a dense matrix and
post-selection is a project-and-renormalize.

Honest caveats:
    - The "quantum exponential speedup" of HHL is *asymptotic in dimension*.
      At small N our simulator gives the same answer classical inversion
      would, in roughly the same time. We're verifying the algorithm
      works, not claiming a speedup.
    - HHL prepares |x⟩ as a quantum amplitude; reading off the classical
      vector takes O(N) samples. The advantage is in cases where you only
      need to *measure expectations* on |x⟩.

Reference: Harrow, Hassidim, Lloyd, Phys. Rev. Lett. 103, 150502 (2009).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _validate_A(A: np.ndarray, tol: float = 1e-9) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    dim = A.shape[0]
    if dim & (dim - 1):
        raise ValueError(f"A's dimension {dim} must be a power of 2")
    if not np.allclose(A, A.conj().T, atol=tol):
        raise ValueError("A must be Hermitian")


def _choose_time(A: np.ndarray) -> float:
    """Pick the evolution time `t` so that all eigenvalues fit in [0, 2π)."""
    eigs = np.linalg.eigvalsh(A)
    lam_max = float(np.max(np.abs(eigs)))
    if lam_max < 1e-12:
        raise ValueError("A has no nonzero eigenvalue")
    # Leave some headroom so the maximum phase is below 2π.
    return 2 * np.pi / (lam_max * 4)


def _controlled_rotation_unitary(n_counting: int, t: float,
                                  C: float) -> np.ndarray:
    """Build the (clock + ancilla)-acting unitary that performs

        |c⟩ |0⟩ → |c⟩ ( √(1 - (C/λ_c)²) |0⟩ + (C/λ_c) |1⟩ )

    where λ_c = 2π · c / (t · 2^n_counting) is the eigenvalue encoded by
    clock value c. We treat c = 0 as "no eigenvalue" (no rotation).

    Returned matrix has shape (2^n_counting · 2,  2^n_counting · 2). Layout:
    clock register first, ancilla last (the usual qubit-0-is-MSB convention).
    """
    dim_c = 1 << n_counting
    U = np.eye(2 * dim_c, dtype=np.complex128)
    for c in range(dim_c):
        if c == 0:
            continue
        # Eigenvalue λ_c encoded by clock value c.
        lam = 2 * np.pi * c / (t * dim_c)
        # Standard HHL: choose the phase branch by symmetry — c > dim_c/2
        # corresponds to negative eigenvalues.
        if c > dim_c // 2:
            lam = -2 * np.pi * (dim_c - c) / (t * dim_c)
        ratio = C / lam
        # Numerical safety: clip to [-1, 1] (occasionally |ratio| > 1 from
        # discretization error on small eigenvalues).
        ratio = np.clip(ratio.real, -1.0, 1.0)
        # Ry(2·arcsin(ratio)) applied to ancilla.
        theta = 2.0 * np.arcsin(ratio)
        cos_h = np.cos(theta / 2)
        sin_h = np.sin(theta / 2)
        # 2×2 block on the ancilla for this clock value c.
        block = np.array([[cos_h, -sin_h], [sin_h,  cos_h]], dtype=np.complex128)
        base = 2 * c
        U[base:base + 2, base:base + 2] = block
    return U


def _qpe_unitary(A: np.ndarray, n_counting: int, t: float) -> np.ndarray:
    """Build the unitary that applies QPE for U = exp(iAt) onto a clock
    register and a b register. Layout: clock register (MSB) then b register.
    """
    n_b = int(np.log2(A.shape[0]))
    dim_c = 1 << n_counting
    dim_b = 1 << n_b
    total = dim_c * dim_b

    # Start: identity.
    QPE = np.eye(total, dtype=np.complex128)

    # Step 1: Hadamards on each clock qubit.
    H_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    I_b = np.eye(dim_b, dtype=np.complex128)
    # Apply H to each clock qubit one at a time.
    for q in range(n_counting):
        # Build the operator H on clock qubit q, I elsewhere.
        op = 1
        for i in range(n_counting):
            op = np.kron(op, H_gate if i == q else np.eye(2))
        op = np.kron(op, I_b)
        QPE = op @ QPE

    # Step 2: controlled-U^{2^p}, controlled on clock qubit (n_counting-1-p).
    U_t = expm(1j * A * t)
    U_power = U_t.copy()
    for p in range(n_counting):
        control = n_counting - 1 - p
        # Build the controlled-U_power gate over (control + all b qubits).
        # Use block-diagonal structure: if control=0, identity; if 1, U_power.
        # Then embed in the full Hilbert space.
        CU = np.eye(2 * dim_b, dtype=np.complex128)
        CU[dim_b:, dim_b:] = U_power
        # Embed CU into (clock + b) where the relevant clock qubit is `control`.
        # Use tensor-product reshape over the clock register.
        op = _embed_2q_block(CU, control, n_counting, n_b)
        QPE = op @ QPE
        U_power = U_power @ U_power

    # Step 3: inverse QFT on the clock register.
    iqft_c = _inverse_qft_matrix(n_counting)
    op = np.kron(iqft_c, I_b)
    QPE = op @ QPE
    return QPE


def _embed_2q_block(CU: np.ndarray, control_qubit: int,
                    n_counting: int, n_b: int) -> np.ndarray:
    """Embed a (2 · 2^n_b) × (2 · 2^n_b) "controlled-U_b" gate into the full
    Hilbert space, where the control is clock qubit `control_qubit` and the
    target spans all b qubits.

    The full Hilbert space has qubit order: clock_0, clock_1, ..., clock_{t-1},
    b_0, b_1, ..., b_{n_b-1}. So the relevant axes are 0..n_counting-1+n_b.
    """
    n_total = n_counting + n_b
    dim_total = 1 << n_total
    # We rearrange axes so the control qubit and b qubits are leading.
    targets = [control_qubit] + list(range(n_counting, n_total))
    # Build full matrix via tensor reshape.
    op = np.eye(dim_total, dtype=np.complex128)
    op = op.reshape([2] * n_total + [2] * n_total)
    # We'll build the new operator directly by matrix multiplication on the
    # reshaped state. Simpler: just construct from scratch via tensor moveaxis.

    # For correctness with minimal cleverness: directly construct the full
    # matrix by acting on every basis state.
    op = np.zeros((dim_total, dim_total), dtype=np.complex128)
    n_target = len(targets)
    dim_target = 1 << n_target
    untouched = [q for q in range(n_total) if q not in targets]
    n_unt = len(untouched)
    dim_unt = 1 << n_unt

    def _bit(idx, qubit):
        return (idx >> (n_total - 1 - qubit)) & 1

    def _build_idx(target_bits, untouched_bits):
        # Reconstruct the full N-bit index given target qubits' bit values
        # (in order of `targets`) and untouched qubits' bit values.
        bits = [0] * n_total
        for q, b in zip(targets, target_bits):
            bits[q] = b
        for q, b in zip(untouched, untouched_bits):
            bits[q] = b
        idx = 0
        for b in bits:
            idx = (idx << 1) | b
        return idx

    # For every untouched bit pattern, apply CU on the target sub-block.
    for u in range(dim_unt):
        ubits = [(u >> (n_unt - 1 - i)) & 1 for i in range(n_unt)]
        for ti in range(dim_target):
            for tj in range(dim_target):
                if CU[ti, tj] == 0:
                    continue
                tibits = [(ti >> (n_target - 1 - i)) & 1 for i in range(n_target)]
                tjbits = [(tj >> (n_target - 1 - i)) & 1 for i in range(n_target)]
                row = _build_idx(tibits, ubits)
                col = _build_idx(tjbits, ubits)
                op[row, col] = CU[ti, tj]
    return op


def _inverse_qft_matrix(n: int) -> np.ndarray:
    """Inverse QFT as a 2^n × 2^n matrix."""
    N = 1 << n
    F = np.zeros((N, N), dtype=np.complex128)
    for x in range(N):
        for y in range(N):
            F[x, y] = np.exp(-2j * np.pi * x * y / N)
    return F / np.sqrt(N)


def hhl(
    A: np.ndarray,
    b: np.ndarray,
    n_counting: int = 6,
    C: float | None = None,
) -> dict:
    """Run HHL on a small Hermitian system Ax = b.

    Args:
        A:           Hermitian matrix, shape (2^n_b, 2^n_b).
        b:           Right-hand-side vector, shape (2^n_b,). Need not be unit.
        n_counting:  Number of clock qubits (precision).
        C:           Constant such that C ≤ λ_min(|A|). Defaults to smallest
                     |eigenvalue| of A.

    Returns:
        dict with:
            x_quantum:           normalized estimate of A⁻¹b (from the quantum
                                 simulator after post-selection on ancilla=1)
            x_classical:         A⁻¹b / ||A⁻¹b|| (reference, for comparison)
            fidelity:            |⟨x_quantum | x_classical⟩|² ∈ [0, 1]
            success_probability: probability of measuring ancilla = 1
            n_qubits:            n_counting + n_b + 1
    """
    _validate_A(A)
    n_b = int(np.log2(A.shape[0]))
    dim_b = 1 << n_b
    dim_c = 1 << n_counting

    b = np.asarray(b, dtype=np.complex128).reshape(-1)
    if b.shape[0] != dim_b:
        raise ValueError(f"b must have length {dim_b}, got {b.shape[0]}")
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-14:
        raise ValueError("b must be nonzero")
    b_normalized = b / b_norm

    t = _choose_time(A)
    if C is None:
        eigs = np.linalg.eigvalsh(A)
        C = float(np.min(np.abs(eigs[np.abs(eigs) > 1e-12])))

    # Build the full circuit unitary acting on (clock, b, ancilla) order.
    # We'll compose three pieces:
    #   1. QPE(A, t, n_counting) on (clock, b)  ⊗ I on ancilla
    #   2. Controlled rotation on (clock, ancilla)  ⊗ I on b
    #   3. Inverse QPE on (clock, b)  ⊗ I on ancilla
    QPE = _qpe_unitary(A, n_counting, t)
    QPE_dag = QPE.conj().T
    CR_ca = _controlled_rotation_unitary(n_counting, t, C)

    # Lift each piece to the full (clock + b + ancilla)-qubit Hilbert space.
    I_anc = np.eye(2, dtype=np.complex128)
    I_b   = np.eye(dim_b, dtype=np.complex128)

    QPE_full     = np.kron(QPE,     I_anc)
    QPE_dag_full = np.kron(QPE_dag, I_anc)
    # CR_ca is (clock ⊗ ancilla); embed (clock ⊗ b ⊗ ancilla) via permutation.
    # CR_ca has shape (2·dim_c, 2·dim_c). We want a unitary on
    # (dim_c · dim_b · 2). We do this by placing CR_ca's clock part on the
    # clock register and ancilla part on the ancilla, with identity on b.
    CR_full = _embed_clock_ancilla(CR_ca, n_counting, n_b)

    full_U = QPE_dag_full @ CR_full @ QPE_full

    # Initial state: |0_clock⟩ ⊗ |b⟩ ⊗ |0_anc⟩
    initial = np.zeros(dim_c * dim_b * 2, dtype=np.complex128)
    # Index layout: c_msb ... c_lsb b_msb ... b_lsb a
    for i_b in range(dim_b):
        # Set state at clock=0, b=i_b, anc=0
        idx = (0 << (n_b + 1)) | (i_b << 1) | 0
        initial[idx] = b_normalized[i_b]

    final = full_U @ initial

    # Post-select on ancilla = 1: collect amplitudes with a=1, zero out a=0.
    post = np.zeros(dim_b, dtype=np.complex128)
    success_amp_sq = 0.0
    for i_b in range(dim_b):
        # Look at all clock values, ancilla = 1
        for c_idx in range(dim_c):
            idx = (c_idx << (n_b + 1)) | (i_b << 1) | 1
            amp = final[idx]
            if c_idx == 0:
                # Add to post; we want the part where clock has returned to |0⟩
                post[i_b] += amp
            success_amp_sq += abs(amp) ** 2

    post_norm = np.linalg.norm(post)
    if post_norm < 1e-12:
        # Couldn't recover x; return what we have.
        x_quantum = post
    else:
        x_quantum = post / post_norm

    # Classical reference.
    x_classical_unnorm = np.linalg.solve(A, b)
    x_classical = x_classical_unnorm / np.linalg.norm(x_classical_unnorm)

    fidelity = abs(np.vdot(x_classical, x_quantum)) ** 2

    return {
        "x_quantum":           x_quantum,
        "x_classical":         x_classical,
        "fidelity":            float(fidelity),
        "success_probability": float(abs(post_norm) ** 2),
        "n_qubits":            n_counting + n_b + 1,
        "n_counting":          n_counting,
        "evolution_time":      t,
        "C":                   C,
    }


def _embed_clock_ancilla(CR_ca: np.ndarray, n_counting: int,
                          n_b: int) -> np.ndarray:
    """Embed a (clock ⊗ ancilla)-acting unitary into the (clock ⊗ b ⊗ ancilla)
    Hilbert space (identity on b)."""
    dim_c = 1 << n_counting
    dim_b = 1 << n_b
    total = dim_c * dim_b * 2

    out = np.zeros((total, total), dtype=np.complex128)
    # CR_ca has shape (2·dim_c, 2·dim_c). Index format: c_bits then a.
    for c_in in range(dim_c):
        for a_in in range(2):
            for c_out in range(dim_c):
                for a_out in range(2):
                    val = CR_ca[2 * c_out + a_out, 2 * c_in + a_in]
                    if val == 0:
                        continue
                    for i_b in range(dim_b):
                        row = (c_out << (n_b + 1)) | (i_b << 1) | a_out
                        col = (c_in  << (n_b + 1)) | (i_b << 1) | a_in
                        out[row, col] = val
    return out

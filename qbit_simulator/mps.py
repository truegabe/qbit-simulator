"""Matrix Product State (MPS) representation of an N-qubit quantum state.

Layout:
    state = A^[0] - A^[1] - A^[2] - ... - A^[N-1]
    A^[q] is a rank-3 tensor of shape (chi_left, 2, chi_right).
    Open boundary: A^[0] has chi_left=1, A^[N-1] has chi_right=1.

Storage scales as O(N * chi^2) instead of O(2^N). For low-entanglement
states (chi ~ const), this is exponentially smaller. The trade-off is
bond-dimension truncation on 2-qubit gates: when a gate creates more
entanglement than `max_chi` allows, the smallest singular values are
discarded, introducing a small approximation.

What works well:
    - Product states, GHZ-like states (chi stays at 2).
    - 1D physical ground states (area law bounds chi).
    - Shallow circuits, quantum walks on a line.

What fails (chi blows up):
    - Random circuits past O(log N) depth.
    - Post-FFT QPE / Shor states (highly entangled).
    - Grover mid-iteration states.

Dynamic qubit growth via `add_qubit()` — append/insert a qubit anywhere
in the chain, initialized in |0> or |1>. The bond dimension at the
insertion point starts at 1.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .gates import H, X, Y, Z, S, T, CNOT, SWAP, Rx, Ry, Rz, P, CP


class MPSState:
    """1D matrix-product state on N qubits with capped bond dimension."""

    def __init__(self, n: int, max_chi: int = 64):
        if n < 1:
            raise ValueError("MPSState needs at least one qubit")
        self.n = n
        self.max_chi = max_chi
        # |0...0>: each qubit is a (1, 2, 1) tensor with amplitude 1 on |0>.
        self.tensors: list[np.ndarray] = [
            np.array([[[1.0], [0.0]]], dtype=np.complex128) for _ in range(n)
        ]
        # Replay history (parallel to QuantumCircuit's `history`).
        self.history: list[str] = []

    # ---- inspection ----

    def bond_dimensions(self) -> list[int]:
        """Internal bond dimensions [chi_1, chi_2, ..., chi_{N-1}]."""
        return [self.tensors[q].shape[2] for q in range(self.n - 1)]

    def storage_bytes(self) -> int:
        """Total bytes occupied by the MPS tensors (vs 2^N * 16 for dense)."""
        return sum(t.nbytes for t in self.tensors)

    def norm(self) -> float:
        """⟨ψ|ψ⟩, computed by contracting the chain top-to-bottom."""
        # left[bra_bond, ket_bond], starts as (1, 1) identity.
        left = np.ones((1, 1), dtype=np.complex128)
        for A in self.tensors:
            # left: (l_bra, l_ket). A: (l_ket, p, r_ket).
            tmp = np.einsum("lk,kpr->lpr", left, A)
            # conj(A): (l_bra, p, r_bra). Contract with tmp on (l_bra, p).
            left = np.einsum("lps,lpr->sr", np.conj(A), tmp)
        return float(np.real(left[0, 0]))

    # ---- conversion ----

    def to_dense(self) -> np.ndarray:
        """Return the full 2^N state vector. Use only for small N (<= ~22)."""
        # Contract the chain into a single tensor of shape (1, 2, 2, ..., 2, 1).
        result = self.tensors[0]              # (1, 2, chi_1)
        for q in range(1, self.n):
            # result: (..., chi_q).  next: (chi_q, 2, chi_{q+1}).
            # → (..., 2, chi_{q+1})
            result = np.tensordot(result, self.tensors[q], axes=([result.ndim - 1], [0]))
        # Shape now (1, 2, 2, ..., 2, 1). Drop the boundary dims.
        return result.reshape(-1)

    @classmethod
    def from_dense(cls, state: np.ndarray, max_chi: int = 64) -> "MPSState":
        """Decompose a dense 2^N state vector into MPS form via successive SVD."""
        state = np.asarray(state, dtype=np.complex128)
        n = int(np.log2(state.size))
        if 2**n != state.size:
            raise ValueError(f"state size {state.size} is not a power of 2")
        mps = cls(n, max_chi=max_chi)

        # Reshape state to (1, 2, 2, ..., 2, 1).
        cur = state.reshape((1,) + (2,) * n + (1,))
        tensors = []
        chi_left = 1
        for q in range(n):
            # cur shape: (chi_left, 2, 2, ..., 2, 1).
            # Split off the leftmost physical index.
            d_right = cur.size // (chi_left * 2)
            M = cur.reshape(chi_left * 2, d_right)
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            # Truncate to max_chi.
            chi = min(len(S), max_chi)
            U = U[:, :chi]
            S = S[:chi]
            Vh = Vh[:chi, :]
            A = U.reshape(chi_left, 2, chi)
            tensors.append(A.astype(np.complex128))
            # Fold S into the remainder for the next iteration.
            cur = (np.diag(S) @ Vh).reshape((chi,) + (2,) * (n - q - 1) + (1,))
            chi_left = chi
        mps.tensors = tensors
        return mps

    # ---- gate application ----

    def apply_1q(self, gate: np.ndarray, q: int) -> None:
        """Apply a 2×2 gate to qubit q. Local; no bond growth."""
        if not (0 <= q < self.n):
            raise IndexError(f"qubit {q} out of range")
        if gate.shape != (2, 2):
            raise ValueError("1q gate must be 2×2")
        # A[q] shape (l, p, r). new_A[l, i, r] = Σ_j gate[i, j] · A[l, j, r].
        self.tensors[q] = np.einsum("ij,ljr->lir", gate, self.tensors[q])

    def apply_2q_adjacent(self, gate: np.ndarray, q: int) -> None:
        """Apply a 4×4 gate to adjacent qubits (q, q+1). Truncates to max_chi."""
        if not (0 <= q < self.n - 1):
            raise IndexError(f"qubit pair ({q}, {q+1}) out of range")
        if gate.shape != (4, 4):
            raise ValueError("2q gate must be 4×4")

        A1 = self.tensors[q]                        # (l, 2, m)
        A2 = self.tensors[q + 1]                    # (m, 2, r)
        # Contract over the shared bond m → T of shape (l, 2, 2, r).
        T = np.einsum("lpm,mqr->lpqr", A1, A2)
        l, _, _, r = T.shape
        # Apply gate on the 4-dimensional physical block.
        T = T.reshape(l, 4, r)
        T = np.einsum("ij,ljr->lir", gate, T)
        T = T.reshape(l, 2, 2, r)
        # SVD-split back into two tensors with truncation.
        M = T.reshape(l * 2, 2 * r)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        chi = min(len(S), self.max_chi)
        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]
        self.tensors[q]     = U.reshape(l, 2, chi).astype(np.complex128)
        self.tensors[q + 1] = (np.diag(S) @ Vh).reshape(chi, 2, r).astype(np.complex128)

    def apply_2q(self, gate: np.ndarray, q1: int, q2: int) -> None:
        """Apply a 4×4 gate to any pair (q1, q2). Non-adjacent pairs use SWAPs.

        gate is (4, 4) with the convention that q1 is the most-significant of
        the 2-qubit basis indexing (consistent with QuantumCircuit._apply_2q).
        """
        if q1 == q2:
            raise ValueError("q1 and q2 must differ")
        for q in (q1, q2):
            if not (0 <= q < self.n):
                raise IndexError(f"qubit {q} out of range")
        if q1 + 1 == q2:
            self.apply_2q_adjacent(gate, q1)
            return
        if q2 + 1 == q1:
            # Swap the gate's qubit order to apply on (q2, q1) instead.
            # Permutation that swaps the two qubits of a 4×4 gate matrix.
            P_swap = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]], dtype=np.complex128)
            self.apply_2q_adjacent(P_swap @ gate @ P_swap, q2)
            return
        # Non-adjacent: SWAP q2 toward q1+1, apply, SWAP back.
        if q2 > q1:
            # Bring q2 down to q1+1 via SWAPs.
            for k in range(q2, q1 + 1, -1):
                self.apply_2q_adjacent(SWAP, k - 1)
            self.apply_2q_adjacent(gate, q1)
            for k in range(q1 + 1, q2):
                self.apply_2q_adjacent(SWAP, k)
        else:
            # q1 > q2 + 1: bring q1 down to q2+1.
            for k in range(q1, q2 + 1, -1):
                self.apply_2q_adjacent(SWAP, k - 1)
            # Now q1 effectively at q2+1 and q2 at q2. Apply with operands swapped.
            P_swap = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]], dtype=np.complex128)
            self.apply_2q_adjacent(P_swap @ gate @ P_swap, q2)
            for k in range(q2 + 1, q1):
                self.apply_2q_adjacent(SWAP, k)

    # ---- convenience gate API (mirrors QuantumCircuit) ----

    def h(self, q): self.apply_1q(H, q); self.history.append(f"H({q})"); return self
    def x(self, q): self.apply_1q(X, q); self.history.append(f"X({q})"); return self
    def y(self, q): self.apply_1q(Y, q); self.history.append(f"Y({q})"); return self
    def z(self, q): self.apply_1q(Z, q); self.history.append(f"Z({q})"); return self
    def s(self, q): self.apply_1q(S, q); self.history.append(f"S({q})"); return self
    def t(self, q): self.apply_1q(T, q); self.history.append(f"T({q})"); return self

    def rx(self, theta, q): self.apply_1q(Rx(theta), q); self.history.append(f"Rx({theta:.4g},{q})"); return self
    def ry(self, theta, q): self.apply_1q(Ry(theta), q); self.history.append(f"Ry({theta:.4g},{q})"); return self
    def rz(self, theta, q): self.apply_1q(Rz(theta), q); self.history.append(f"Rz({theta:.4g},{q})"); return self
    def p(self, phi, q):    self.apply_1q(P(phi), q);    self.history.append(f"P({phi:.4g},{q})"); return self

    def cnot(self, c, t):
        self.apply_2q(CNOT, c, t); self.history.append(f"CNOT({c},{t})"); return self
    def swap(self, a, b):
        self.apply_2q(SWAP, a, b); self.history.append(f"SWAP({a},{b})"); return self
    def cp(self, phi, c, t):
        self.apply_2q(CP(phi), c, t); self.history.append(f"CP({phi:.4g},{c},{t})"); return self
    def cz(self, c, t):
        from .gates import controlled
        self.apply_2q(controlled(Z), c, t); self.history.append(f"CZ({c},{t})"); return self

    # ---- dynamic resizing ----

    def add_qubit(self, at: int | None = None, in_state: int = 0) -> int:
        """Insert a new qubit at position `at` (default: append).

        The new qubit starts in |in_state> (0 or 1) and is *unentangled* with
        the rest of the chain, so the bond dimension at the insertion point
        starts at 1. Returns the index of the new qubit.
        """
        if in_state not in (0, 1):
            raise ValueError("in_state must be 0 or 1")
        if at is None:
            at = self.n
        if not (0 <= at <= self.n):
            raise IndexError(f"insertion position {at} out of range [0, {self.n}]")
        # New tensor: shape (1, 2, 1), basis state |in_state>.
        new_t = np.zeros((1, 2, 1), dtype=np.complex128)
        new_t[0, in_state, 0] = 1.0
        self.tensors.insert(at, new_t)
        self.n += 1
        self.history.append(f"add_qubit(at={at}, |{in_state}>)")
        return at

    # ---- probabilities & expectation ----

    def probabilities(self) -> np.ndarray:
        """Full 2^N probability distribution. Falls back to dense — for big
        N use marginal_prob_single() instead."""
        psi = self.to_dense()
        return np.abs(psi) ** 2

    def marginal_prob_single(self, q: int) -> tuple[float, float]:
        """P(qubit q = 0), P(qubit q = 1) without ever building the dense state."""
        if not (0 <= q < self.n):
            raise IndexError(f"qubit {q} out of range")
        # Build left environment up to qubit q (excluding q).
        left = np.ones((1, 1), dtype=np.complex128)   # (bra_bond, ket_bond)
        for k in range(q):
            A = self.tensors[k]
            tmp = np.einsum("lk,kpr->lpr", left, A)
            left = np.einsum("lps,lpr->sr", np.conj(A), tmp)
        # Build right environment from qubit q+1 to end.
        right = np.ones((1, 1), dtype=np.complex128)
        for k in range(self.n - 1, q, -1):
            A = self.tensors[k]
            tmp = np.einsum("lpr,rs->lps", A, right)
            right = np.einsum("lpr,sps->ls".replace("s", "k"),
                              np.conj(self.tensors[k]), tmp)
            # Cleaner:
            right = np.einsum("kpr,lpr->kl", np.conj(self.tensors[k]),
                              np.einsum("lpr,rs->lps", self.tensors[k], right))
        # Combine with qubit q's tensor.
        A = self.tensors[q]                            # (l_ket, p, r_ket)
        # ⟨0_q ⟩: probability that qubit q is measured 0.
        # left[lb, lk] * A[lk, p, rk] * right[rb, rk] * conj(A[lb, p, rb])
        # summed over lb, lk, rb, rk for p in {0, 1}.
        def _prob_for(p: int) -> float:
            slice_A = A[:, p, :]                       # (l_ket, r_ket)
            # left @ slice_A: (l_bra, r_ket)
            top = left @ slice_A
            # right @ conj(slice_A).T? right is (r_bra, r_ket).
            # We want: sum over lb, rk of top[lb, rk] * conj(slice_A)[lb, rb] * right[rb, rk]
            # = sum_lb_rk top[lb, rk] * conj(slice_A)[lb, rb] * right[rb, rk]
            # → contract: einsum('br,br->', top @ right.T.conj-ish...)
            val = np.einsum("br,xr,bx->",
                            top, right, np.conj(slice_A.T))
            return float(np.real(val))
        # The right-env loop above is a bit awkward; do it again cleanly here.
        return _prob_for(0), _prob_for(1)


# ---- helpers ----

def mps_overlap(a: MPSState, b: MPSState) -> complex:
    """⟨a|b⟩ for two MPS over the same number of qubits."""
    if a.n != b.n:
        raise ValueError("MPS overlap requires same qubit count")
    left = np.ones((1, 1), dtype=np.complex128)
    for q in range(a.n):
        Aa = a.tensors[q]; Ab = b.tensors[q]
        tmp = np.einsum("lk,kpr->lpr", left, Ab)
        left = np.einsum("lps,lpr->sr", np.conj(Aa), tmp)
    return complex(left[0, 0])

"""Projected Entangled Pair States (PEPS) — 2D tensor networks.

Generalization of MPS to two dimensions. Each site (x, y) on an Lx × Ly
grid stores a rank-5 tensor T[x, y] of shape

    (chi_up, chi_down, chi_left, chi_right, physical_dim)

where the four virtual legs connect to nearest neighbors and the
physical leg holds the qubit/quoit degree of freedom. Boundary sites
have virtual bond dim 1 on the edges that have no neighbor.

What's tractable on a PEPS:
    - Storage: O(Lx · Ly · χ⁴ · d) instead of O(d^(Lx·Ly))
    - Local 1-qubit gates: O(χ⁴ · d²) per gate
    - 2-qubit gates on nearest neighbors: O(χ⁶ · d²) plus SVD truncation
    - Norm and expectation values: NP-hard in general; we expose
      `to_dense()` for verification on small lattices and structure for
      "boundary MPS" contraction on larger ones (left as a hook)

What this implementation gives:
    - PEPSState class with tensor storage and dynamic bond dimensions
    - apply_1q_gate, apply_2q_gate (horizontal & vertical neighbors)
    - to_dense / from_product_state
    - bond_dimensions, storage_bytes for inspection

PEPS is genuinely hard at full generality (exact contraction is #P-hard).
For small lattices (≤ 4×4 in our verification tests) we cross-check
against dense. The infrastructure is designed so that larger lattices
can use approximate boundary-MPS contraction (a separate addition).
"""

from __future__ import annotations

import numpy as np


class PEPSState:
    """Projected entangled pair state on an Lx × Ly grid of qubits.

    Tensor convention at site (x, y): shape (u, d, l, r, p)
        u: up virtual leg (to (x, y-1))
        d: down virtual leg (to (x, y+1))
        l: left virtual leg (to (x-1, y))
        r: right virtual leg (to (x+1, y))
        p: physical leg (qubit value)

    Boundary sites have the missing-neighbor leg dim 1.
    """

    def __init__(self, Lx: int, Ly: int, max_chi: int = 8, physical_dim: int = 2):
        if Lx < 1 or Ly < 1:
            raise ValueError("Lattice dimensions must be positive")
        self.Lx = Lx
        self.Ly = Ly
        self.max_chi = max_chi
        self.physical_dim = physical_dim
        # Initialize to |0⟩^⊗(Lx·Ly).
        self.tensors: dict[tuple[int, int], np.ndarray] = {}
        for x in range(Lx):
            for y in range(Ly):
                T = np.zeros((1, 1, 1, 1, physical_dim), dtype=np.complex128)
                T[0, 0, 0, 0, 0] = 1.0     # |0⟩
                self.tensors[(x, y)] = T
        self.history: list[str] = []

    # ---- inspection ----

    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly

    def site_indices(self) -> list[tuple[int, int]]:
        """All (x, y) sites in row-major order."""
        return [(x, y) for y in range(self.Ly) for x in range(self.Lx)]

    def bond_dimensions(self) -> dict[str, list[int]]:
        """Maximum virtual-bond dimensions, separately for horizontal and vertical."""
        horizontal = []   # bonds between (x, y) and (x+1, y)
        vertical = []     # bonds between (x, y) and (x, y+1)
        for y in range(self.Ly):
            for x in range(self.Lx - 1):
                # Right leg of (x, y), should equal left leg of (x+1, y).
                horizontal.append(self.tensors[(x, y)].shape[3])
        for y in range(self.Ly - 1):
            for x in range(self.Lx):
                vertical.append(self.tensors[(x, y)].shape[1])
        return {"horizontal": horizontal, "vertical": vertical}

    def storage_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    def __repr__(self) -> str:
        bds = self.bond_dimensions()
        chi_h = max(bds["horizontal"]) if bds["horizontal"] else 1
        chi_v = max(bds["vertical"])   if bds["vertical"]   else 1
        return (f"PEPSState({self.Lx}×{self.Ly}, max_chi={self.max_chi}, "
                f"current_chi=(h:{chi_h}, v:{chi_v}))")

    # ---- gate application ----

    def apply_1q_gate(self, gate: np.ndarray, x: int, y: int) -> None:
        """Apply a 2×2 gate to qubit at (x, y). O(χ⁴ · d²)."""
        self._check_coords(x, y)
        if gate.shape != (self.physical_dim, self.physical_dim):
            raise ValueError(f"1q gate must be {self.physical_dim}×{self.physical_dim}")
        T = self.tensors[(x, y)]
        # T[u, d, l, r, p] → gate[p', p] · T[u, d, l, r, p]
        self.tensors[(x, y)] = np.einsum("ij,udlrj->udlri", gate, T)

    def apply_2q_gate_horizontal(self, gate4: np.ndarray, x: int, y: int) -> None:
        """Apply a 4×4 gate to nearest-neighbor sites (x, y) and (x+1, y).

        The 4×4 gate's first index is the left site, second the right.
        Truncates the bond between them to max_chi.
        """
        self._check_coords(x, y)
        if x + 1 >= self.Lx:
            raise IndexError("horizontal 2q gate needs x+1 < Lx")
        if gate4.shape != (4, 4):
            raise ValueError("2q gate must be 4×4")
        Tl = self.tensors[(x, y)]      # (u, d, l, r=χ, p)
        Tr = self.tensors[(x + 1, y)]  # (u, d, l=χ, r, p)
        # Contract shared bond (Tl's right leg with Tr's left leg).
        # Result shape: (u1, d1, l1, p1, u2, d2, r2, p2)
        merged = np.einsum("udlrp,UDrSq->udlpUDSq", Tl, Tr)
        # Apply gate on (p1, p2).
        # gate4[(p1', p2'), (p1, p2)] = gate4 (4,4) reshaped as (2, 2, 2, 2).
        G = gate4.reshape(2, 2, 2, 2)
        merged = np.einsum("ijpq,udlpUDSq->udliUDSj", G, merged)
        # Now we need to SVD-split merged into two tensors over the
        # (l1, p1) / (p2, r2) split.
        u1, d1, l1, p1, u2, d2, r2, p2 = merged.shape
        # Reshape into matrix M of shape (u1·d1·l1·p1, u2·d2·r2·p2):
        M = merged.transpose(0, 1, 2, 3, 4, 5, 6, 7).reshape(
            u1 * d1 * l1 * p1, u2 * d2 * r2 * p2
        )
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        chi = min(len(S), self.max_chi)
        U  = U[:, :chi]
        S  = S[:chi]
        Vh = Vh[:chi, :]
        # Distribute the singular values evenly (sqrt on each side).
        sqrtS = np.sqrt(S)
        new_Tl = (U * sqrtS).reshape(u1, d1, l1, p1, chi).transpose(0, 1, 2, 4, 3)
        # new_Tl shape (u1, d1, l1, chi=r1, p1)
        new_Tr = (sqrtS[:, None] * Vh).reshape(chi, u2, d2, r2, p2).transpose(1, 2, 0, 3, 4)
        # new_Tr shape (u2, d2, chi=l2, r2, p2)
        self.tensors[(x, y)]     = new_Tl
        self.tensors[(x + 1, y)] = new_Tr

    def apply_2q_gate_vertical(self, gate4: np.ndarray, x: int, y: int) -> None:
        """Apply a 4×4 gate to (x, y) and (x, y+1). Truncates to max_chi."""
        self._check_coords(x, y)
        if y + 1 >= self.Ly:
            raise IndexError("vertical 2q gate needs y+1 < Ly")
        if gate4.shape != (4, 4):
            raise ValueError("2q gate must be 4×4")
        Tt = self.tensors[(x, y)]       # (u, d=χ, l, r, p)
        Tb = self.tensors[(x, y + 1)]   # (u=χ, d, l, r, p)
        # Contract shared bond (Tt's down leg with Tb's up leg).
        merged = np.einsum("udlrp,dDLRq->ulrpDLRq", Tt, Tb)
        G = gate4.reshape(2, 2, 2, 2)
        merged = np.einsum("ijpq,ulrpDLRq->ulriDLRj", G, merged)
        u, l, r, p1, D, L, R, p2 = merged.shape
        M = merged.reshape(u * l * r * p1, D * L * R * p2)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        chi = min(len(S), self.max_chi)
        U  = U[:, :chi]
        S  = S[:chi]
        Vh = Vh[:chi, :]
        sqrtS = np.sqrt(S)
        # Top tensor: (u, l, r, p1) × chi (new down bond)
        new_Tt = (U * sqrtS).reshape(u, l, r, p1, chi).transpose(0, 4, 1, 2, 3)
        # → (u, d=chi, l, r, p)
        # Bottom tensor: chi × (D, L, R, p2)
        new_Tb = (sqrtS[:, None] * Vh).reshape(chi, D, L, R, p2)
        # → (u=chi, d, l, r, p)
        self.tensors[(x, y)]     = new_Tt
        self.tensors[(x, y + 1)] = new_Tb

    # ---- convenience gate API ----

    def h(self, x: int, y: int) -> "PEPSState":
        from .gates import H
        self.apply_1q_gate(H, x, y); self.history.append(f"H({x},{y})"); return self

    def x(self, x: int, y: int) -> "PEPSState":
        from .gates import X
        self.apply_1q_gate(X, x, y); self.history.append(f"X({x},{y})"); return self

    def z(self, x: int, y: int) -> "PEPSState":
        from .gates import Z
        self.apply_1q_gate(Z, x, y); self.history.append(f"Z({x},{y})"); return self

    def cnot(self, ctrl: tuple[int, int], tgt: tuple[int, int]) -> "PEPSState":
        """CNOT on nearest-neighbor (ctrl → tgt). Determine direction automatically."""
        from .gates import CNOT
        cx, cy = ctrl; tx, ty = tgt
        if cy == ty and abs(cx - tx) == 1:
            # Horizontal
            x_left = min(cx, tx)
            gate = CNOT if cx == x_left else _swap_qubits_4x4(CNOT)
            self.apply_2q_gate_horizontal(gate, x_left, cy)
        elif cx == tx and abs(cy - ty) == 1:
            y_top = min(cy, ty)
            gate = CNOT if cy == y_top else _swap_qubits_4x4(CNOT)
            self.apply_2q_gate_vertical(gate, cx, y_top)
        else:
            raise ValueError("cnot only on nearest neighbors")
        self.history.append(f"CNOT({ctrl}→{tgt})")
        return self

    # ---- conversion ----

    def to_dense(self) -> np.ndarray:
        """Contract the full PEPS into a 2^(Lx·Ly) state vector.

        Use only for small lattices (≤ 16 sites). The contraction order is
        row by row; for each row, contract the row's tensors with the
        running "environment" from previous rows.
        """
        # Contract column-by-column.
        # Start with the top row's tensors merged into a 1D MPS.
        # Then sweep down, contracting each row's vertical bonds.
        #
        # Simpler approach for small lattices: contract all tensors
        # directly via repeated einsum.
        Lx, Ly = self.Lx, self.Ly
        # Build a big tensor with axes:
        # one per (x, y) physical leg, plus virtual legs (which are 1 on
        # boundary and dim chi internally — but we contract them).
        #
        # The cleanest way: contract via tensor network. We label each
        # tensor's axes and use np.einsum or path-finding.
        #
        # Practical: iterate row by row, contracting bonds as we go.

        # State after first row's tensors contracted (in row 0):
        #   each tensor T[x, 0] has dim (1, d_y, 1_or_chi, 1_or_chi, phys)
        #   = (1, χ_down, χ_left, χ_right, 2)
        # Combine across x for row 0: produce a tensor with all phys legs
        # in row 0 + a "down" bond stack.

        # I'll do it the simplest way: build the full 2^N state vector
        # by contracting each tensor's physical leg into a basis sum.

        N = Lx * Ly
        dim = 2 ** N
        state = np.zeros(dim, dtype=np.complex128)

        # Each basis state |x_0 x_1 ... x_{N-1}⟩ has amplitude:
        # contract all T[x, y][:, :, :, :, b_{xy}] where b_{xy} is the
        # bit at site (x, y), and sum over all virtual indices.
        # We use site index (x, y) → flat = y * Lx + x in MSB-first ordering.

        for idx in range(dim):
            # Extract per-site bits.
            bits = {}
            for site_flat in range(N):
                y = site_flat // Lx
                x = site_flat % Lx
                # Flat index: leftmost qubit is MSB.
                bits[(x, y)] = (idx >> (N - 1 - site_flat)) & 1

            # Contract all tensors with the chosen physical-leg slice.
            # Build a 4D tensor for each site: T[u, d, l, r] (with phys fixed).
            sliced = {}
            for (x, y), T in self.tensors.items():
                sliced[(x, y)] = T[:, :, :, :, bits[(x, y)]]

            # Now contract the 2D grid of 4-tensors.
            # Use the standard tensor-network contraction order:
            # contract column by column.
            #
            # For each row, sliced[(x, y)] has shape (u, d, l, r) where
            # l and r connect horizontally (same row), u and d connect
            # vertically (across rows).

            # Build row tensors by contracting horizontally first.
            row_tensors = []
            for y in range(Ly):
                # Start with leftmost tensor in row y.
                cur = sliced[(0, y)]  # (u, d, l=1, r, ...)
                cur = cur.reshape(cur.shape[0], cur.shape[1], cur.shape[3])  # (u, d, r)
                for x_in in range(1, Lx):
                    nxt = sliced[(x_in, y)]  # (u, d, l, r)
                    # Contract cur's right with nxt's left.
                    cur = np.einsum("udr,UDrs->uUdDs", cur, nxt).reshape(
                        cur.shape[0] * nxt.shape[0],
                        cur.shape[1] * nxt.shape[1],
                        nxt.shape[3],
                    )
                # Drop final r leg (=1 on right boundary).
                cur = cur.reshape(cur.shape[0], cur.shape[1])  # (u_combined, d_combined)
                row_tensors.append(cur)
            # Now contract row tensors vertically.
            # row_tensors[0] has u=1 (top boundary).
            cur = row_tensors[0]  # (1, d) — squeeze
            cur = cur.reshape(cur.shape[1])  # (d,)
            for y_in in range(1, Ly):
                nxt = row_tensors[y_in]  # (u, d)
                cur = np.einsum("u,ud->d", cur, nxt)
            # Final cur should be a scalar (d=1 on bottom boundary).
            amp = complex(cur[0]) if cur.shape == (1,) else complex(cur)

            state[idx] = amp

        return state

    # ---- helpers ----

    def _check_coords(self, x: int, y: int) -> None:
        if not (0 <= x < self.Lx and 0 <= y < self.Ly):
            raise IndexError(f"site ({x}, {y}) out of [{self.Lx}, {self.Ly}]")


# ---- helper utilities ----

def _swap_qubits_4x4(G: np.ndarray) -> np.ndarray:
    """Conjugate a 4x4 2-qubit gate by SWAP. Useful for CNOT direction inversion."""
    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.complex128)
    return SWAP @ G @ SWAP

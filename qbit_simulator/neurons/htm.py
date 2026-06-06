"""Hierarchical Temporal Memory — simplified Numenta-style model.

Two key modules:

  - Spatial Pooler: maps a sparse binary input to a sparse binary
    representation of "active columns" using a fixed proximal-synapse
    map. Implements competitive selection (top-k columns by overlap).

  - Temporal Memory: given a sequence of active columns, learns
    sequences by tracking active CELLS within each column. Cells form
    distal segments that predict next-step activations.

This is a simplified educational version. The full Numenta HTM has
more biological detail (boosting, structural plasticity, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpatialPooler:
    """Maps a binary input to a sparse set of active columns."""
    n_input: int
    n_columns: int = 256
    sparsity: float = 0.04
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    W: np.ndarray = field(default=None, repr=False)
    boost: np.ndarray = field(default=None, repr=False)
    activity_avg: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.W is None:
            # Each column connects to ~25% of inputs.
            self.W = (self.rng.uniform(size=(self.n_columns, self.n_input))
                       < 0.25).astype(np.float64)
        if self.boost is None:
            self.boost = np.ones(self.n_columns)
        if self.activity_avg is None:
            self.activity_avg = np.zeros(self.n_columns)

    def compute(self, x: np.ndarray, learn: bool = True) -> np.ndarray:
        overlap = self.W @ x
        boosted = overlap * self.boost
        k = max(int(self.sparsity * self.n_columns), 1)
        # Top-k columns active.
        if k >= self.n_columns:
            active = np.ones(self.n_columns, dtype=bool)
        else:
            thresh = np.partition(boosted, -k)[-k]
            active = boosted >= thresh
        if learn:
            # Update activity average and boost.
            self.activity_avg = 0.99 * self.activity_avg + 0.01 * active
            mean_act = self.activity_avg.mean() + 1e-9
            self.boost = np.exp(-(self.activity_avg / mean_act - 1) * 0.1)
        return active.astype(np.float64)


@dataclass
class TemporalMemory:
    """Simplified temporal memory: per-column cells with sequence tracking."""
    n_columns: int
    cells_per_column: int = 8
    n_segments_per_cell: int = 4
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    predictive: np.ndarray = field(default=None, repr=False)
    active: np.ndarray = field(default=None, repr=False)
    # Segment connectivity: list of (col, cell, segment) -> connected presyn cells
    segments: list = field(default_factory=list)

    def __post_init__(self) -> None:
        N = self.n_columns * self.cells_per_column
        if self.predictive is None:
            self.predictive = np.zeros(N, dtype=bool)
        if self.active is None:
            self.active = np.zeros(N, dtype=bool)
        if not self.segments:
            self.segments = [[] for _ in range(N)]

    def _cell_idx(self, col: int, cell: int) -> int:
        return col * self.cells_per_column + cell

    def step(self, active_cols: np.ndarray) -> dict:
        """Process one time step of active columns. Returns dict of state."""
        prev_active = self.active.copy()
        new_active = np.zeros_like(self.active)
        for col in np.where(active_cols > 0)[0]:
            col_cells = [self._cell_idx(col, c) for c in range(self.cells_per_column)]
            predicted_in_col = [c for c in col_cells if self.predictive[c]]
            if predicted_in_col:
                # Activate only predicted cells.
                for c in predicted_in_col:
                    new_active[c] = True
            else:
                # Burst: activate all cells in column.
                for c in col_cells:
                    new_active[c] = True
        # Learn: any active cell records its presynaptic predecessor.
        for c in np.where(new_active)[0]:
            pred_input = list(np.where(prev_active)[0])
            if pred_input and len(self.segments[c]) < self.n_segments_per_cell:
                # Pick a random subset to remember.
                k = min(8, len(pred_input))
                sample = self.rng.choice(pred_input, size=k, replace=False).tolist()
                self.segments[c].append(set(sample))
        # Predict: any cell with a segment that matches active cells.
        new_pred = np.zeros_like(self.predictive)
        active_set = set(np.where(new_active)[0])
        for c, segs in enumerate(self.segments):
            for seg in segs:
                if len(seg & active_set) >= max(int(0.5 * len(seg)), 1):
                    new_pred[c] = True
                    break
        self.active = new_active
        self.predictive = new_pred
        return {"active": new_active, "predictive": new_pred,
                "n_active": int(new_active.sum()),
                "n_predicted": int(new_pred.sum())}


@dataclass
class HTM:
    """Spatial pooler + temporal memory pipeline."""
    n_input: int
    n_columns: int = 256
    cells_per_column: int = 8
    sparsity: float = 0.04
    sp: SpatialPooler = field(default=None)
    tm: TemporalMemory = field(default=None)

    def __post_init__(self) -> None:
        if self.sp is None:
            self.sp = SpatialPooler(n_input=self.n_input,
                                      n_columns=self.n_columns,
                                      sparsity=self.sparsity)
        if self.tm is None:
            self.tm = TemporalMemory(n_columns=self.n_columns,
                                      cells_per_column=self.cells_per_column)

    def step(self, x: np.ndarray) -> dict:
        cols = self.sp.compute(x)
        return self.tm.step(cols)

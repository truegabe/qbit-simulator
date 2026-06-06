"""Two-stage memory consolidation (McClelland, McNaughton, O'Reilly 1995).

Memories are first encoded rapidly in a HIPPOCAMPAL fast-learning system
(autoassociative, sparse) and gradually transferred to NEOCORTEX during
sleep through replay-driven, slow Hebbian learning.

Model:
  - Hippocampus = fast Hopfield network (high learning rate).
  - Cortex      = slow Hopfield-like network with small learning rate.
  - During wake: store new memory in hippocampus.
  - During sleep: replay hippocampal memories and slowly transfer
    them into cortex.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TwoStageMemory:
    n: int
    eta_hipp: float = 1.0
    eta_cort: float = 0.05
    W_hipp: np.ndarray = field(default=None, repr=False)
    W_cort: np.ndarray = field(default=None, repr=False)
    hipp_patterns: list = field(default_factory=list)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.W_hipp is None:
            self.W_hipp = np.zeros((self.n, self.n))
        if self.W_cort is None:
            self.W_cort = np.zeros((self.n, self.n))

    def encode(self, x: np.ndarray) -> None:
        """Wake-phase encoding: imprint pattern on hippocampus."""
        self.W_hipp += self.eta_hipp * np.outer(x, x) / self.n
        np.fill_diagonal(self.W_hipp, 0)
        self.hipp_patterns.append(x.copy())

    def consolidate(self, n_replays: int = 100) -> None:
        """Sleep-phase replay → slow cortical learning."""
        if not self.hipp_patterns:
            return
        for _ in range(n_replays):
            idx = self.rng.integers(len(self.hipp_patterns))
            p = self.hipp_patterns[idx]
            self.W_cort += self.eta_cort * np.outer(p, p) / self.n
            np.fill_diagonal(self.W_cort, 0)

    def hipp_recall(self, cue: np.ndarray, n_iter: int = 5) -> np.ndarray:
        s = cue.copy().astype(np.float64)
        for _ in range(n_iter):
            s = np.sign(self.W_hipp @ s)
        return s

    def cort_recall(self, cue: np.ndarray, n_iter: int = 5) -> np.ndarray:
        s = cue.copy().astype(np.float64)
        for _ in range(n_iter):
            s = np.sign(self.W_cort @ s)
        return s

    def forget_hippocampus(self, fraction: float = 0.5) -> None:
        """Mimic post-consolidation forgetting in hippocampus."""
        self.W_hipp *= 1 - fraction
        # Random removal of patterns to model decay.
        n_keep = max(int(len(self.hipp_patterns) * (1 - fraction)), 0)
        if n_keep < len(self.hipp_patterns):
            keep_idx = self.rng.choice(len(self.hipp_patterns),
                                          size=n_keep, replace=False)
            self.hipp_patterns = [self.hipp_patterns[i] for i in keep_idx]

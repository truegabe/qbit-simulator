"""Loopy belief propagation on factor graphs.

Sum-product algorithm: nodes exchange messages until convergence.

  variable → factor:  m_{v→f}(x_v) = prod_{f' ≠ f} m_{f'→v}(x_v)
  factor → variable:  m_{f→v}(x_v) = sum_{x_others} f(x) prod_{v'} m_{v'→f}(x_{v'})

Exact on trees; approximate but often good on loopy graphs.

We implement a small reference version on pairwise binary MRFs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PairwiseBinaryMRF:
    """Binary pairwise MRF with unary fields h_i and pairwise J_ij.

    p(x) ∝ exp(sum_i h_i x_i + sum_{ij ∈ E} J_ij x_i x_j),  x_i ∈ {-1, +1}.
    """
    n: int
    h: np.ndarray = field(default=None, repr=False)
    J: np.ndarray = field(default=None, repr=False)
    edges: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.h is None:
            self.h = np.zeros(self.n)
        if self.J is None:
            self.J = np.zeros((self.n, self.n))
        if not self.edges:
            self.edges = [(i, j) for i in range(self.n)
                          for j in range(i + 1, self.n) if self.J[i, j] != 0]

    def _neighbors(self) -> dict:
        """Build neighbor map: node i -> list of neighbors."""
        nbr = {i: [] for i in range(self.n)}
        for (a, b) in self.edges:
            nbr[a].append(b); nbr[b].append(a)
        return nbr

    def loopy_bp(self, n_iter: int = 50, damping: float = 0.5) -> np.ndarray:
        """Returns marginal probabilities p(x_i = +1).

        Messages are stored as log-ratios λ_{k→i} = log[m(+1)/m(-1)].
        """
        nbr = self._neighbors()
        # Initialize all directed messages to 0 (uniform).
        msg = {}
        for i in range(self.n):
            for k in nbr[i]:
                msg[(k, i)] = 0.0
        # Symmetric J accessor.
        def J_sym(a, b):
            return self.J[a, b] if self.J[a, b] != 0 else self.J[b, a]
        for _ in range(n_iter):
            new_msg = {}
            # Compute every directed message i -> j synchronously.
            for j in range(self.n):
                for i in nbr[j]:
                    # Field at i excluding the message coming from j.
                    f_i = self.h[i] + sum(msg[(k, i)] for k in nbr[i] if k != j)
                    Jij = J_sym(i, j)
                    # Sum over x_i ∈ {-1, +1}:
                    #   x_j=+1: exp(+Jij + f_i) + exp(-Jij - f_i)
                    #   x_j=-1: exp(-Jij + f_i) + exp(+Jij - f_i)
                    # log[m(+1)/m(-1)] = log cosh(J+f) - log cosh(J-f),
                    # which equals 2·atanh(tanh(J)·tanh(f)).
                    # Cavity-field convention stores HALF of that.
                    num = np.logaddexp(Jij + f_i, -Jij - f_i)
                    den = np.logaddexp(-Jij + f_i, Jij - f_i)
                    new_msg[(i, j)] = ((1 - damping) * 0.5 * (num - den)
                                        + damping * msg[(i, j)])
            msg = new_msg
        # Marginals: log-odds = h_i + sum of incoming messages.
        marg = np.zeros(self.n)
        for i in range(self.n):
            log_ratio = self.h[i] + sum(msg[(k, i)] for k in nbr[i])
            marg[i] = 1.0 / (1.0 + np.exp(-2 * log_ratio))
        return marg

"""Wake-sleep / memory consolidation.

The brain stores new experiences in fast HIPPOCAMPAL memory during
waking, then gradually transfers them to slower, more distributed
CORTICAL memory during sleep (especially slow-wave sleep). This is
the "two-stage" memory consolidation model (Marr 1971; Buzsáki 1989;
McClelland-McNaughton-O'Reilly 1995).

This module implements a SIMPLIFIED two-store consolidation framework:

  - **Hippocampal store** (`HippocampalMemory` from `episodic_memory.py`):
    fast write, high capacity for individual episodes.
  - **Cortical store** (a Hopfield-like associative memory): slow
    learning, generalizes over many similar episodes.

During "sleep":
  1. Sample stored episodes from hippocampal memory (replay).
  2. Each replay event triggers a small Hebbian update in cortex.
  3. Repeat for many cycles.

Result: cortex generalizes the patterns, smoothing over noise. After
consolidation, hippocampus can be "cleared" (simulating memory
forgetting) and cortex still recalls the gist.

Provides:
  - `WakeSleepAgent(context_dim, content_dim)`: encapsulates both stores.
  - `.wake_observe(context, content)`: store in hippocampus.
  - `.sleep_cycle(n_replays)`: transfer to cortex.
  - `.cortical_recall(context)`: read from the slow store.
  - `.hippocampal_recall(context)`: read from the fast store.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .episodic_memory import HippocampalMemory
from .hopfield import HopfieldNetwork


# ----------------------------------------------------------------------------
# Wake-sleep agent
# ----------------------------------------------------------------------------

@dataclass
class WakeSleepAgent:
    """Two-store memory: fast hippocampus + slow cortex.

    Hippocampus stores RAW episode (context + content concatenated).
    Cortex slowly learns the concatenated pattern in a Hopfield network.
    """
    context_dim: int = 32
    content_dim: int = 32
    n_hippocampal_addresses: int = 100
    k_nearest: int = 8
    seed: int = 0

    hippocampus: HippocampalMemory = field(default=None)
    cortex: HopfieldNetwork = field(default=None)
    cortical_patterns_stored: int = 0
    episodes_seen: list[tuple[np.ndarray, np.ndarray]] = field(
        default_factory=list,
    )
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.hippocampus is None:
            self.hippocampus = HippocampalMemory(
                context_dim=self.context_dim,
                content_dim=self.content_dim,
                n_addresses=self.n_hippocampal_addresses,
                k_nearest=self.k_nearest,
                seed=self.seed,
            )
        if self.cortex is None:
            self.cortex = HopfieldNetwork(
                n=self.context_dim + self.content_dim,
            )

    # ---- WAKE ----

    def wake_observe(self, context: np.ndarray, content: np.ndarray) -> None:
        """Fast hippocampal write."""
        self.hippocampus.store(context, content)
        self.episodes_seen.append((context.copy(), content.copy()))

    def hippocampal_recall(self, context: np.ndarray) -> np.ndarray:
        """Fast retrieval from hippocampus."""
        return self.hippocampus.recall(context)

    # ---- SLEEP ----

    def sleep_cycle(self, n_replays: int = 20,
                      rng: np.random.Generator | None = None,
                      decay_factor: float = 0.95) -> None:
        """Sleep phase: replay episodes from hippocampus and incrementally
        store them in cortex.

        Each replay does an `add_pattern` step on the Hopfield network
        with the FULL (context ⊕ content) vector.
        """
        rng = rng or np.random.default_rng()
        if not self.episodes_seen:
            return
        # Optional weight decay before replay (slow forgetting).
        if decay_factor < 1.0:
            self.cortex.weights *= decay_factor

        for _ in range(n_replays):
            ctx, cnt = self.episodes_seen[
                rng.integers(0, len(self.episodes_seen))
            ]
            pattern = np.concatenate([ctx, cnt])
            self.cortex.add_pattern(pattern)
            self.cortical_patterns_stored += 1

    # ---- CORTICAL RECALL ----

    def cortical_recall(self, context: np.ndarray,
                          max_iter: int = 50,
                          rng: np.random.Generator | None = None
                          ) -> np.ndarray:
        """Cortex retrieves content given context.

        Strategy: clamp the first `context_dim` neurons to `context`
        (treated as a noisy probe of the full pattern), run Hopfield
        dynamics, read out the last `content_dim` neurons.
        """
        rng = rng or np.random.default_rng()
        # Compose a probe: context ⊕ random ±1 placeholder for content.
        n = self.context_dim + self.content_dim
        probe = np.concatenate([
            context,
            rng.choice([-1.0, 1.0], size=self.content_dim),
        ])
        r = self.cortex.retrieve(probe, max_iter=max_iter, rng=rng)
        return r["retrieved_state"][self.context_dim:]

    # ---- forgetting ----

    def hippocampal_forget(self, fraction: float = 0.5,
                              rng: np.random.Generator | None = None) -> None:
        """Remove a fraction of stored episodes from the hippocampus
        (simulating "consolidated → no longer needed in hippocampus")."""
        rng = rng or np.random.default_rng()
        if not self.episodes_seen:
            return
        n_keep = int(len(self.episodes_seen) * (1 - fraction))
        if n_keep <= 0:
            self.episodes_seen.clear()
            self.hippocampus.sdm.clear()
            return
        keep_idx = rng.choice(len(self.episodes_seen), size=n_keep, replace=False)
        self.episodes_seen = [self.episodes_seen[i] for i in keep_idx]
        self.hippocampus.sdm.clear()
        for ctx, cnt in self.episodes_seen:
            self.hippocampus.store(ctx, cnt)


# ----------------------------------------------------------------------------
# Diagnostic: compare recall before vs after sleep
# ----------------------------------------------------------------------------

def consolidation_test(
    n_episodes: int = 5, context_dim: int = 32, content_dim: int = 32,
    n_sleep_replays: int = 50,
    rng: np.random.Generator | None = None,
) -> dict:
    """Store some episodes, recall from BOTH stores, sleep, then recall
    from cortex after forgetting hippocampus.
    """
    rng = rng or np.random.default_rng()
    agent = WakeSleepAgent(context_dim=context_dim, content_dim=content_dim)
    # Generate random episodes.
    contexts = [rng.choice([-1, 1], size=context_dim).astype(np.float64)
                for _ in range(n_episodes)]
    contents = [rng.choice([-1, 1], size=content_dim).astype(np.float64)
                for _ in range(n_episodes)]
    for c, k in zip(contexts, contents):
        agent.wake_observe(c, k)

    # Before sleep: cortical recall should be poor.
    before = []
    for c, k in zip(contexts, contents):
        recovered = agent.cortical_recall(c, rng=rng)
        overlap = float((recovered * k).mean())
        before.append(overlap)

    # Sleep
    agent.sleep_cycle(n_replays=n_sleep_replays, rng=rng)

    # After sleep
    after = []
    for c, k in zip(contexts, contents):
        recovered = agent.cortical_recall(c, rng=rng)
        overlap = float((recovered * k).mean())
        after.append(overlap)

    return {
        "overlap_before_sleep": before,
        "overlap_after_sleep":  after,
        "mean_before":          float(np.mean(before)),
        "mean_after":           float(np.mean(after)),
    }

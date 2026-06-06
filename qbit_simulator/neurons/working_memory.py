"""Working memory: bistable attractor networks for short-term storage.

In cortex, working memory is implemented by recurrently-connected
neural populations whose persistent firing keeps information "active"
for seconds (~100x longer than any single synaptic time constant).
This is the canonical "persistent activity" model.

The simplest implementation: a population of N excitatory neurons with
self-recurrent connectivity W_E > 0. Once kicked above a threshold by
a brief input cue, the population SUSTAINS its firing rate even after
the cue is removed — a stable "high-firing" attractor.

Combined with global inhibition (or fatigue), the network also has a
"silent" attractor. The two stable states encode 1 bit of information.

Multiple bistable populations (each storing 1 bit) → a working-memory
buffer with k items, like a brain's "magical number 7 ± 2" capacity.

This module provides:

  - `BistableAttractor`: one population that can be "loaded" with a cue
    and later "read out" or "cleared".
  - `WorkingMemoryBuffer(n_items, neurons_per_item)`: a bank of
    bistable populations.
  - `.load(item_idx)`, `.clear(item_idx)`, `.read(item_idx)`: API.
  - `.run_for(n_steps)`: free-running simulation between operations.

The dynamics use the LIF population from `lif.py` with a recurrent
weight matrix tuned for bistability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lif import LIFPopulation


# ----------------------------------------------------------------------------
# Single bistable attractor (one memory slot)
# ----------------------------------------------------------------------------

@dataclass
class BistableAttractor:
    """N excitatory LIF neurons with self-recurrent excitation forming
    a 2-state attractor (silent / high-firing).

    Parameters:
      - `n`:           number of neurons in the pool.
      - `w_recurrent`: per-pair recurrent weight (W_E / n).
      - `tonic`:       tonic input current (small, helps maintain firing).
      - `cue_strength`, `cue_duration`: parameters of the "load" input.
    """
    n:                int = 20
    w_recurrent:      float = 3.0     # per-pair (units of input current)
    tonic:            float = 0.3     # baseline excitation
    syn_tau:          float = 30.0    # synaptic-current decay time
    cue_strength:     float = 5.0     # strong cue to drive initial firing
    cue_duration:     int = 40
    inhibit_strength: float = -30.0  # strong enough to silence persistent state
    inhibit_duration: int = 60

    population: LIFPopulation = field(default=None)
    spike_history: list[np.ndarray] = field(default_factory=list)
    syn_current:   np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.population is None:
            self.population = LIFPopulation(n=self.n)
        if self.syn_current is None:
            self.syn_current = np.zeros(self.n)

    @property
    def W(self) -> np.ndarray:
        """Recurrent weight matrix (no self-connections)."""
        return self.w_recurrent * (np.ones((self.n, self.n))
                                    - np.eye(self.n)) / self.n

    def reset(self) -> None:
        self.population.reset()
        self.spike_history.clear()
        self.syn_current[:] = 0

    def step(self, external: float, t: int) -> np.ndarray:
        """One simulation step.

        Synaptic currents decay exponentially with `syn_tau`; each
        pre-spike adds an impulse to the post-synaptic neuron's
        synaptic-current pool. This gives recurrent input a memory
        longer than one step — enough for persistent activity.
        """
        # Decay current.
        self.syn_current *= np.exp(-1.0 / self.syn_tau)
        # On any prior spikes, add weighted impulses.
        if self.spike_history:
            last = self.spike_history[-1].astype(float)
            self.syn_current += self.W @ last
        ext = np.full(self.n, self.tonic + external) + self.syn_current
        spikes = self.population.step(ext, t=t)
        self.spike_history.append(spikes.copy())
        return spikes

    def run_for(self, n_steps: int, external: float = 0.0,
                  start_t: int = 0) -> np.ndarray:
        """Run with constant `external` for n_steps. Returns spike rates."""
        for k in range(n_steps):
            self.step(external, t=start_t + k)
        spikes = np.array(self.spike_history[-n_steps:])
        return spikes.mean(axis=0)

    def load(self, n_steps_cue: int | None = None,
              n_steps_relax: int = 50, start_t: int = 0) -> int:
        """Apply the cue input briefly, then run free.

        Returns the time index AFTER load + relax (for chaining).
        """
        n_steps_cue = n_steps_cue or self.cue_duration
        t = start_t
        for k in range(n_steps_cue):
            self.step(self.cue_strength, t=t)
            t += 1
        for k in range(n_steps_relax):
            self.step(0.0, t=t)
            t += 1
        return t

    def clear(self, n_steps_inhibit: int | None = None,
               n_steps_relax: int = 30, start_t: int = 0) -> int:
        """Apply strong inhibition to push the network to the silent
        attractor; also zero out lingering synaptic currents so the
        clear is decisive."""
        n_steps_inhibit = n_steps_inhibit or self.inhibit_duration
        t = start_t
        for k in range(n_steps_inhibit):
            self.step(self.inhibit_strength, t=t)
            t += 1
        # Wipe the accumulated synaptic current so the attractor can't
        # re-ignite from leftover excitation.
        self.syn_current[:] = 0
        for k in range(n_steps_relax):
            self.step(0.0, t=t)
            t += 1
        return t

    def read(self, window: int = 30) -> float:
        """Return the average population firing rate over the last `window`
        steps. > 0.05 → HIGH state, < 0.01 → LOW state (parameters depend
        on n and w_recurrent)."""
        if not self.spike_history:
            return 0.0
        recent = np.array(self.spike_history[-window:])
        return float(recent.mean())


# ----------------------------------------------------------------------------
# Multi-slot working memory buffer
# ----------------------------------------------------------------------------

@dataclass
class WorkingMemoryBuffer:
    """A bank of `n_items` independent bistable attractors, each storing
    one bit. Total neurons = n_items × neurons_per_item.

    Provides a high-level API: `set(idx)`, `clear(idx)`, `get(idx)`.
    """
    n_items:          int = 4
    neurons_per_item: int = 20
    slot_kwargs:      dict = field(default_factory=dict)

    slots:    list[BistableAttractor] = field(default_factory=list)
    t_global: int = 0    # global simulation time

    def __post_init__(self) -> None:
        if not self.slots:
            self.slots = [
                BistableAttractor(n=self.neurons_per_item, **self.slot_kwargs)
                for _ in range(self.n_items)
            ]

    def reset(self) -> None:
        for s in self.slots:
            s.reset()
        self.t_global = 0

    def set(self, idx: int, n_relax: int = 50) -> None:
        """Load slot `idx` to HIGH state."""
        if not (0 <= idx < self.n_items):
            raise IndexError(idx)
        self.t_global = self.slots[idx].load(
            n_steps_relax=n_relax, start_t=self.t_global,
        )

    def clear(self, idx: int, n_relax: int = 30) -> None:
        if not (0 <= idx < self.n_items):
            raise IndexError(idx)
        self.t_global = self.slots[idx].clear(
            n_steps_relax=n_relax, start_t=self.t_global,
        )

    def get(self, idx: int, window: int = 30) -> int:
        """Read slot `idx` as a 0/1 bit by thresholding firing rate."""
        if not (0 <= idx < self.n_items):
            raise IndexError(idx)
        rate = self.slots[idx].read(window=window)
        return 1 if rate > 0.02 else 0

    def get_rate(self, idx: int, window: int = 30) -> float:
        return self.slots[idx].read(window=window)

    def all_states(self, window: int = 30) -> list[int]:
        return [self.get(i, window=window) for i in range(self.n_items)]

    def all_rates(self, window: int = 30) -> list[float]:
        return [self.get_rate(i, window=window) for i in range(self.n_items)]

    def free_run(self, n_steps: int) -> None:
        """Run all slots for n_steps with no external input.

        Tests "persistence": HIGH slots should stay HIGH; silent stay silent.
        """
        for slot in self.slots:
            for k in range(n_steps):
                slot.step(0.0, t=self.t_global + k)
        self.t_global += n_steps


# ----------------------------------------------------------------------------
# Capacity diagnostic
# ----------------------------------------------------------------------------

def capacity_test(
    n_items: int = 4,
    pattern: list[int] | None = None,
    n_free_steps: int = 100,
    neurons_per_item: int = 20,
) -> dict:
    """Set the buffer to a given pattern, free-run, then read.

    A working memory is "successful" if the read-out matches the
    originally-set pattern after the free-run.
    """
    if pattern is None:
        pattern = [1, 0, 1, 0][:n_items]
    buf = WorkingMemoryBuffer(n_items=n_items, neurons_per_item=neurons_per_item)
    for i, bit in enumerate(pattern):
        if bit == 1:
            buf.set(i)
        # bit==0 → leave silent
    rates_after_load = buf.all_rates()
    buf.free_run(n_free_steps)
    rates_after_relax = buf.all_rates()
    read_pattern = buf.all_states()
    return {
        "set_pattern":         pattern,
        "rates_after_load":    rates_after_load,
        "rates_after_relax":   rates_after_relax,
        "read_pattern":        read_pattern,
        "match":               read_pattern == pattern,
    }

"""Leaky integrate-and-fire (LIF) neurons + small SNN runner.

The LIF neuron is the standard simple spiking-neuron model in
computational neuroscience. Each neuron has:

  - A membrane potential V(t).
  - A leak that pulls V back toward V_rest with time constant tau.
  - Input current I(t) from other neurons + external drive.
  - A spike threshold V_threshold; when V crosses it, the neuron emits
    a spike, V is reset to V_reset, and the neuron is in a refractory
    period for t_refrac time steps.

Continuous-time ODE:

    tau · dV/dt = -(V - V_rest) + R · I(t)

Discrete update (Euler, time step dt):

    V(t + dt)  =  V(t)  +  dt/tau · ( -(V - V_rest) + R · I(t) )

This module provides:

  - `LIFNeuron`: parameters + state for one neuron.
  - `LIFPopulation(n)`: vectorized population (NumPy, much faster than
    per-neuron loops).
  - `SNN(n_neurons, weights, ...)`: a recurrent SNN. Connectivity is a
    weight matrix W; pre-synaptic spikes inject current into
    post-synaptic neurons weighted by W.
  - `run(input_current, n_steps)`: simulate.
  - `spike_raster(snn)`: ASCII rendering of the spike train.

We work in dimensionless units: V_rest = 0, V_threshold = 1, V_reset = 0,
tau = 20·dt by default, R = 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ----------------------------------------------------------------------------
# Single LIF neuron (reference impl; vectorized version below is what we use)
# ----------------------------------------------------------------------------

@dataclass
class LIFNeuron:
    """A single LIF neuron with state."""
    V_rest:      float = 0.0
    V_reset:     float = 0.0
    V_threshold: float = 1.0
    tau:         float = 20.0     # in units of dt
    R:           float = 1.0
    t_refrac:    int = 2          # refractory period in time steps

    # Mutable state
    V:           float = 0.0
    refrac_until: int = -1

    def step(self, I: float, dt: float = 1.0, t: int = 0) -> bool:
        """One time step. Returns True if the neuron spiked this step.

        Convention: `t_refrac` = number of FULLY-BLOCKED steps after a
        spike (biological "absolute refractory period"). Minimum gap
        between two consecutive spikes is therefore t_refrac + 1.
        """
        if t <= self.refrac_until:
            self.V = self.V_reset
            return False
        # Euler update.
        self.V += dt / self.tau * (-(self.V - self.V_rest) + self.R * I)
        if self.V >= self.V_threshold:
            self.V = self.V_reset
            self.refrac_until = t + self.t_refrac
            return True
        return False


# ----------------------------------------------------------------------------
# Vectorized LIF population — the primitive most code uses
# ----------------------------------------------------------------------------

@dataclass
class LIFPopulation:
    """A vectorized population of N LIF neurons.

    Same hyperparameters for all neurons by default. Each neuron has
    its own state (V, refrac_until).
    """
    n: int
    V_rest:      float = 0.0
    V_reset:     float = 0.0
    V_threshold: float = 1.0
    tau:         float = 20.0
    R:           float = 1.0
    t_refrac:    int = 2

    V:            np.ndarray = field(default=None, repr=False)
    refrac_until: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.V is None:
            self.V = np.full(self.n, self.V_rest, dtype=np.float64)
        if self.refrac_until is None:
            self.refrac_until = np.full(self.n, -1, dtype=np.int64)

    def reset(self) -> None:
        """Clear membrane potentials and refractory counters."""
        self.V[:] = self.V_rest
        self.refrac_until[:] = -1

    def step(self, I: np.ndarray, dt: float = 1.0, t: int = 0
              ) -> np.ndarray:
        """One time step on the whole population.

        Args:
            I:  per-neuron input current, shape (n,).
            dt: time step.
            t:  current discrete time index.

        Returns:
            Boolean array of length n: which neurons spiked this step.
        """
        if I.shape != (self.n,):
            raise ValueError(f"I must have shape ({self.n},), got {I.shape}")
        # Refractory neurons stay clamped to V_reset and can't spike.
        # Convention: t_refrac is the number of FULLY-BLOCKED steps after
        # a spike (so min gap = t_refrac + 1).
        active = t > self.refrac_until
        # Euler update on active neurons only.
        dV = dt / self.tau * (-(self.V - self.V_rest) + self.R * I)
        self.V = np.where(active, self.V + dV, self.V_reset)
        # Spike emission: threshold crossing.
        spikes = active & (self.V >= self.V_threshold)
        self.V = np.where(spikes, self.V_reset, self.V)
        self.refrac_until = np.where(spikes, t + self.t_refrac, self.refrac_until)
        return spikes


# ----------------------------------------------------------------------------
# Recurrent SNN
# ----------------------------------------------------------------------------

@dataclass
class SNN:
    """A recurrent spiking-neural network.

    Connectivity is a weight matrix W of shape (n_post, n_pre). At each
    time step:
      1. External input + recurrent input via W is delivered.
      2. Each neuron does an LIF step.
      3. Spike train is recorded.

    Synaptic current model: an instantaneous pulse of magnitude W[i, j]
    is delivered to neuron i whenever neuron j spikes. (More realistic
    models use exponential / alpha kernels; we keep it simple.)
    """
    weights: np.ndarray
    population: LIFPopulation = field(default=None)
    spike_history: list[np.ndarray] = field(default_factory=list)
    voltage_history: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.weights.ndim != 2 or self.weights.shape[0] != self.weights.shape[1]:
            raise ValueError("weights must be a square (n × n) matrix")
        n = self.weights.shape[0]
        if self.population is None:
            self.population = LIFPopulation(n=n)
        elif self.population.n != n:
            raise ValueError(
                f"population size {self.population.n} != weights size {n}"
            )

    def reset(self) -> None:
        self.population.reset()
        self.spike_history.clear()
        self.voltage_history.clear()

    def step(self, external_current: np.ndarray, t: int) -> np.ndarray:
        """One simulation step. external_current has shape (n,).

        Returns:
            Boolean spike array for this step.
        """
        # Recurrent input = W · last_spikes (if available).
        if self.spike_history:
            last = self.spike_history[-1].astype(np.float64)
            recurrent = self.weights @ last
        else:
            recurrent = np.zeros(self.population.n)
        spikes = self.population.step(external_current + recurrent, t=t)
        self.spike_history.append(spikes.copy())
        self.voltage_history.append(self.population.V.copy())
        return spikes

    def run(self, external_current: np.ndarray | Callable[[int], np.ndarray],
             n_steps: int = 100) -> dict:
        """Run for n_steps. external_current is either a fixed (n,) array
        or a callable t → (n,)."""
        n = self.population.n
        for t in range(n_steps):
            if callable(external_current):
                I = np.asarray(external_current(t), dtype=np.float64)
            else:
                I = np.asarray(external_current, dtype=np.float64)
            self.step(I, t=t)
        spikes = np.array(self.spike_history)
        return {
            "spikes":     spikes,                # shape (n_steps, n)
            "voltages":   np.array(self.voltage_history),
            "rates":      spikes.mean(axis=0),   # per-neuron firing rate
            "total_spikes": int(spikes.sum()),
        }


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def firing_rates(spike_history: np.ndarray, window: int | None = None
                  ) -> np.ndarray:
    """Per-neuron firing rates (spikes per time step).

    If `window` is provided, returns a moving-window rate of shape
    (n_steps - window + 1, n_neurons).
    """
    if window is None:
        return spike_history.mean(axis=0)
    n_steps, n = spike_history.shape
    if window > n_steps:
        return spike_history.mean(axis=0, keepdims=True)
    out = np.zeros((n_steps - window + 1, n))
    for t in range(n_steps - window + 1):
        out[t] = spike_history[t:t + window].mean(axis=0)
    return out


def spike_raster(spike_history: np.ndarray, max_neurons: int = 20,
                  max_steps: int = 80) -> str:
    """ASCII raster plot. Rows = neurons (capped), columns = time steps.

    '|' = spike, '.' = no spike.
    """
    n_steps, n = spike_history.shape
    n_show = min(n, max_neurons)
    t_show = min(n_steps, max_steps)
    lines = []
    for i in range(n_show):
        row = ""
        for t in range(t_show):
            row += "|" if spike_history[t, i] else "."
        lines.append(f"n{i:>2}: {row}")
    if n > max_neurons:
        lines.append(f"... and {n - max_neurons} more neurons")
    return "\n".join(lines)


def inter_spike_intervals(spike_history: np.ndarray, neuron: int
                            ) -> np.ndarray:
    """Times between consecutive spikes for a given neuron."""
    spike_times = np.where(spike_history[:, neuron])[0]
    if len(spike_times) < 2:
        return np.array([])
    return np.diff(spike_times)


def coefficient_of_variation(spike_history: np.ndarray, neuron: int
                                ) -> float:
    """CV of inter-spike intervals. 0 = perfectly regular, 1 = Poisson."""
    isis = inter_spike_intervals(spike_history, neuron)
    if len(isis) < 2:
        return 0.0
    return float(np.std(isis) / np.mean(isis))


# ----------------------------------------------------------------------------
# Convenience: feedforward / pure-input SNN
# ----------------------------------------------------------------------------

def make_pure_input_snn(n: int, **kwargs) -> SNN:
    """An SNN with NO recurrent connections — neurons only respond to
    external input. Useful for testing single-neuron dynamics in a
    population."""
    W = np.zeros((n, n))
    return SNN(weights=W, population=LIFPopulation(n=n, **kwargs))


def poisson_input_current(n: int, rate: float,
                            rng: np.random.Generator) -> Callable[[int], np.ndarray]:
    """Construct a callable t → (n,) where each component is a Poisson
    spike train with the given `rate` (spikes per step), encoded as
    binary 0/1 currents."""
    def fn(t: int) -> np.ndarray:
        return (rng.uniform(size=n) < rate).astype(np.float64)
    return fn

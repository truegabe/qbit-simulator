"""Reflex arc — fastest sensorimotor loop.

A monosynaptic reflex (like the patellar / "knee-jerk" reflex) has:
  - sensory afferent (e.g. muscle spindle)
  - single synapse onto motor neuron in spinal cord
  - motor efferent (e.g. quadriceps)

Latency is ~30 ms (no cortex involved). Polysynaptic reflexes
(withdrawal, crossed-extensor) add one or more interneurons.

This module gives a minimal LIF-based reflex arc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lif import LIFPopulation


@dataclass
class ReflexArc:
    """Single sensory → motor reflex arc."""
    delay_steps: int = 3       # axonal conduction delay
    gain: float = 2.0
    sensory: LIFPopulation = field(default=None)
    motor:   LIFPopulation = field(default=None)
    _buffer: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sensory is None:
            self.sensory = LIFPopulation(n=1, tau=10.0, t_refrac=1)
        if self.motor is None:
            self.motor = LIFPopulation(n=1, tau=10.0, t_refrac=1)
        self._buffer = [np.zeros(1) for _ in range(self.delay_steps)]

    def step(self, stimulus: float, t: int) -> tuple[bool, bool]:
        """One step. Returns (sensory_spike, motor_spike)."""
        s_spike = self.sensory.step(np.array([stimulus]), t=t)
        # Buffer the sensory spike (axonal delay).
        delayed = self._buffer.pop(0)
        self._buffer.append(s_spike.astype(np.float64))
        m_spike = self.motor.step(self.gain * delayed, t=t)
        return bool(s_spike[0]), bool(m_spike[0])

    def run(self, stim_func, n_steps: int = 100) -> dict:
        s_spikes = np.zeros(n_steps, dtype=bool)
        m_spikes = np.zeros(n_steps, dtype=bool)
        for t in range(n_steps):
            stim = stim_func(t) if callable(stim_func) else stim_func
            s, m = self.step(stim, t=t)
            s_spikes[t] = s; m_spikes[t] = m
        # Latency from first sensory to first motor spike.
        latency = -1
        if s_spikes.any() and m_spikes.any():
            ts = int(np.argmax(s_spikes))
            ms = m_spikes[ts:]
            if ms.any():
                latency = int(np.argmax(ms))
        return {"sensory": s_spikes, "motor": m_spikes, "latency": latency}

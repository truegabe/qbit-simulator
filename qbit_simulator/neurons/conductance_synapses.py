"""Conductance-based synapses: AMPA, NMDA, GABA-A, GABA-B.

These are the four canonical fast/slow excitatory/inhibitory synapses
in cortex. Each is modelled as a kinetic conductance that opens on
pre-synaptic spikes and decays exponentially.

    dg/dt = -g / tau    plus spike-triggered increment g <- g + w

Synaptic current onto the post-synaptic neuron:
    I_syn = -g (V_post - E_rev)

Receptor    E_rev (mV)   tau (ms)   Notes
AMPA         0             5        Fast excitatory, primary
NMDA         0             80       Slow exc; voltage-dependent Mg block
GABA-A      -70            10       Fast inhibitory
GABA-B      -90            150      Slow inhibitory (K+ current)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExpSynapse:
    """Generic exponential-decay conductance synapse."""
    n_post: int
    tau: float = 5.0
    E_rev: float = 0.0
    g: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.g is None:
            self.g = np.zeros(self.n_post)

    def step_decay(self, dt: float = 1.0) -> None:
        self.g *= np.exp(-dt / self.tau)

    def receive(self, input_current: np.ndarray) -> None:
        """Add input (already weighted) into the conductance."""
        self.g += input_current

    def current(self, V_post: np.ndarray) -> np.ndarray:
        """Synaptic current onto post-synaptic neurons.

        Convention: positive = depolarizing (inward current).
        I_syn = -g * (V_post - E_rev)  has the right sign for both
        excitation (E_rev=0 > V_rest) and inhibition (E_rev<V_rest).
        """
        return -self.g * (V_post - self.E_rev)


@dataclass
class AMPASynapse(ExpSynapse):
    tau: float = 5.0
    E_rev: float = 0.0


@dataclass
class GABAASynapse(ExpSynapse):
    tau: float = 10.0
    E_rev: float = -70.0


@dataclass
class GABABSynapse(ExpSynapse):
    tau: float = 150.0
    E_rev: float = -90.0


@dataclass
class NMDASynapse(ExpSynapse):
    """NMDA with Mg2+ voltage-dependent block.

    Block factor B(V) = 1 / (1 + [Mg]/3.57 · exp(-0.062 V)).
    With default [Mg] = 1 mM and V in mV.
    """
    tau: float = 80.0
    E_rev: float = 0.0
    Mg: float = 1.0

    def block(self, V_post: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + self.Mg / 3.57 * np.exp(-0.062 * V_post))

    def current(self, V_post: np.ndarray) -> np.ndarray:
        return -self.g * self.block(V_post) * (V_post - self.E_rev)


@dataclass
class DoubleExpSynapse:
    """Difference-of-exponentials kernel — has finite rise time.

    g(t) = g_max · (exp(-t/tau_d) - exp(-t/tau_r))
    Implemented as two coupled state variables s_rise, s_decay.
    """
    n_post: int
    tau_r: float = 0.5
    tau_d: float = 5.0
    E_rev: float = 0.0
    s_r: np.ndarray = field(default=None, repr=False)
    s_d: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.s_r is None:
            self.s_r = np.zeros(self.n_post)
        if self.s_d is None:
            self.s_d = np.zeros(self.n_post)

    def step_decay(self, dt: float = 1.0) -> None:
        self.s_r *= np.exp(-dt / self.tau_r)
        self.s_d *= np.exp(-dt / self.tau_d)

    def receive(self, input_current: np.ndarray) -> None:
        self.s_r += input_current
        self.s_d += input_current

    def current(self, V_post: np.ndarray) -> np.ndarray:
        g = self.s_d - self.s_r
        return -g * (V_post - self.E_rev)


# ---- a small post-synaptic conductance-based LIF using these synapses ----

@dataclass
class CondBasedLIF:
    """Conductance-based LIF with AMPA + GABA-A inputs."""
    n: int
    C_m:  float = 200.0      # pF
    g_L:  float = 10.0       # nS
    E_L:  float = -70.0      # mV
    V_th: float = -50.0
    V_reset: float = -70.0
    t_refrac: int = 2

    V: np.ndarray = field(default=None, repr=False)
    refrac_until: np.ndarray = field(default=None, repr=False)
    ampa: AMPASynapse = field(default=None, repr=False)
    gaba: GABAASynapse = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.V is None:
            self.V = np.full(self.n, self.E_L)
        if self.refrac_until is None:
            self.refrac_until = np.full(self.n, -1, dtype=np.int64)
        if self.ampa is None:
            self.ampa = AMPASynapse(n_post=self.n)
        if self.gaba is None:
            self.gaba = GABAASynapse(n_post=self.n)

    def step(self, I_inj: np.ndarray, t: int, dt: float = 1.0) -> np.ndarray:
        self.ampa.step_decay(dt); self.gaba.step_decay(dt)
        I_syn = self.ampa.current(self.V) + self.gaba.current(self.V)
        I_leak = -self.g_L * (self.V - self.E_L)
        active = t > self.refrac_until
        dV = dt * (I_leak + I_syn + I_inj) / self.C_m
        self.V = np.where(active, self.V + dV, self.V_reset)
        spikes = active & (self.V >= self.V_th)
        self.V = np.where(spikes, self.V_reset, self.V)
        self.refrac_until = np.where(spikes, t + self.t_refrac,
                                      self.refrac_until)
        return spikes

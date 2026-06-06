"""Spike-based logic gates: AND, OR, NOT, XOR via tuned LIF networks.

Any Boolean function can be computed by a spiking network of LIF
neurons with carefully-chosen synaptic weights and thresholds. This
module provides:

  - `AND_gate(a, b)`:    fires when both inputs fire (sum > threshold).
  - `OR_gate(a, b)`:     fires when either input fires (each above threshold).
  - `NOT_gate(a)`:       fires when input is SILENT (inhibitory drive).
  - `XOR_gate(a, b)`:    a 3-neuron network (AND/OR + inhibitory connection).

The "rate-coding" convention used here:
  - Input bit "1" = neuron fires every step (driven above threshold).
  - Input bit "0" = neuron is silent.
  - Output bit decoded by checking the output neuron's firing rate.

For each gate we provide:

  * A `compute_<gate>(a, b)` convenience function that builds the
    network, runs it, and returns the output rate.
  * A `<gate>_gate(...)` low-level function returning the weight matrix
    + LIF params so users can compose gates.

This module is meant as a sanity check: the SNN substrate CAN do
classical logic — the difficulty in `examples/snn_xor.py` was purely the
LEARNING (STDP credit assignment), not the network expressivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lif import LIFPopulation


# ----------------------------------------------------------------------------
# Common building block: drive an LIF neuron with a per-step input
# ----------------------------------------------------------------------------

def _run_lif(input_current: np.ndarray, n_steps: int,
              tau: float = 20.0, V_threshold: float = 1.0,
              t_refrac: int = 2) -> np.ndarray:
    """Run a single LIF neuron under the given input-current trajectory.

    Returns the spike train (length n_steps, bool).
    """
    pop = LIFPopulation(n=1, tau=tau, V_threshold=V_threshold,
                          t_refrac=t_refrac)
    spikes = np.zeros(n_steps, dtype=bool)
    for t in range(n_steps):
        s = pop.step(np.array([input_current[t]]), t=t)
        spikes[t] = s[0]
    return spikes


# ----------------------------------------------------------------------------
# Single-neuron gates: AND, OR, NOT
# ----------------------------------------------------------------------------

def compute_AND(a: int, b: int, n_steps: int = 100) -> float:
    """Spiking AND: output fires only when BOTH inputs fire.

    Strategy: each input contributes 0.6 to the neuron's input current.
    With threshold 1.0, a single input is subthreshold but the sum (1.2)
    crosses threshold reliably.

    Args:
        a, b:    input bits ∈ {0, 1}.
        n_steps: time window for rate-coding.

    Returns:
        output firing rate (spikes / n_steps).
    """
    I = np.full(n_steps, 0.6 * a + 0.6 * b)
    spikes = _run_lif(I, n_steps)
    return float(spikes.mean())


def compute_OR(a: int, b: int, n_steps: int = 100) -> float:
    """Spiking OR: output fires when either input is high.

    Strategy: each input contributes 1.5 (above threshold by itself).
    """
    I = np.full(n_steps, 1.5 * a + 1.5 * b)
    spikes = _run_lif(I, n_steps)
    return float(spikes.mean())


def compute_NOT(a: int, n_steps: int = 100) -> float:
    """Spiking NOT: output fires when input is silent.

    Strategy: a constant tonic drive (1.5) keeps the neuron firing;
    the input acts as INHIBITION, subtracting more than the tonic drive.
    """
    I = np.full(n_steps, 1.5 - 1.8 * a)   # 1.5 if a=0, -0.3 if a=1
    spikes = _run_lif(I, n_steps)
    return float(spikes.mean())


# ----------------------------------------------------------------------------
# Multi-neuron XOR network
# ----------------------------------------------------------------------------

@dataclass
class XORNetwork:
    """A 3-neuron LIF network implementing XOR.

    Architecture:
        - Input A, Input B (external currents, not neurons).
        - Hidden OR  fires when either input fires.
        - Hidden AND fires only when both inputs fire.
        - Output     fires when OR fires AND AND doesn't (= XOR).

    Implementation: output neuron is driven by:
      * a small TONIC current that ALONE doesn't cause firing,
      * EXCITATORY pulses from OR (each pulse pushes V above threshold),
      * STRONG INHIBITORY pulses from AND (cancels the OR excitation).

    The gains are tuned for the LIF defaults (tau=20, V_th=1).
    """
    n_steps: int = 100
    w_excit: float = 1.0     # OR  -> Output (excitatory boost)
    w_inhib: float = -15.0   # AND -> Output (very strong inhibition)
    syn_tau: float = 80.0    # synaptic-current decay (much longer than membrane)
    output_tau: float = 40.0 # slow output membrane

    def run(self, a: int, b: int) -> dict:
        """Run the XOR network on inputs (a, b).

        The output neuron has a slower membrane and longer synaptic
        currents than the hidden OR/AND neurons. This lets it
        INTEGRATE the average rates of OR and AND, rather than reacting
        to each spike individually — so the inhibitory AND signal
        properly cancels the excitatory OR signal when both fire.
        """
        n = 3
        # Hidden neurons: default LIF params.
        hidden = LIFPopulation(n=2)
        # Output neuron: slower membrane.
        out = LIFPopulation(n=1, tau=self.output_tau, t_refrac=2)

        spikes = np.zeros((self.n_steps, n), dtype=bool)
        # Synaptic currents accumulate from incoming spikes and decay
        # over self.syn_tau steps.
        I_excit = 0.0
        I_inhib = 0.0
        for t in range(self.n_steps):
            # Decay synaptic currents.
            I_excit *= np.exp(-1.0 / self.syn_tau)
            I_inhib *= np.exp(-1.0 / self.syn_tau)
            # Hidden neurons get instantaneous tonic input.
            hidden_in = np.array([1.5 * a + 1.5 * b,
                                   0.6 * a + 0.6 * b])
            hidden_spikes = hidden.step(hidden_in, t=t)
            spikes[t, 0] = hidden_spikes[0]
            spikes[t, 1] = hidden_spikes[1]
            # Add spike-driven impulses to the synaptic currents.
            if hidden_spikes[0]:
                I_excit += self.w_excit
            if hidden_spikes[1]:
                I_inhib += self.w_inhib
            # Output integrates excitatory + inhibitory currents.
            out_spike = out.step(np.array([I_excit + I_inhib]), t=t)
            spikes[t, 2] = out_spike[0]
        return {
            "spikes":        spikes,
            "OR_rate":       float(spikes[:, 0].mean()),
            "AND_rate":      float(spikes[:, 1].mean()),
            "output_rate":   float(spikes[:, 2].mean()),
        }


def compute_XOR(a: int, b: int, n_steps: int = 100) -> float:
    """Spiking XOR via the 3-neuron OR-AND-inhibition network.

    Returns the output neuron's firing rate.
    """
    net = XORNetwork(n_steps=n_steps)
    return net.run(a, b)["output_rate"]


# ----------------------------------------------------------------------------
# Truth-table tester
# ----------------------------------------------------------------------------

def truth_table(gate_fn, arity: int = 2, n_steps: int = 100,
                  threshold_rate: float = 0.01) -> list[tuple]:
    """Build the truth table for `gate_fn` (a, [b, [c, ...]]) → rate.

    Args:
        gate_fn:        callable mapping bits → float rate.
        arity:          number of input bits (1 or 2).
        n_steps:        simulation window.
        threshold_rate: rate above which the output is considered "1".

    Returns:
        list of (inputs..., raw_rate, classified_bit).
    """
    rows = []
    if arity == 1:
        for a in (0, 1):
            r = gate_fn(a, n_steps=n_steps)
            rows.append((a, r, 1 if r > threshold_rate else 0))
    elif arity == 2:
        for a in (0, 1):
            for b in (0, 1):
                r = gate_fn(a, b, n_steps=n_steps)
                rows.append((a, b, r, 1 if r > threshold_rate else 0))
    else:
        raise ValueError("arity must be 1 or 2")
    return rows

"""Neural-substrate package: classical spiking-neuron + plasticity + memory.

This sits *below* the agent / brain layer and *parallel to* the quantum
simulator. It provides the building blocks for a virtual nervous system:

  - `lif`:               leaky integrate-and-fire neurons + small SNN runner.
  - `stdp`:              spike-timing-dependent plasticity learning rule.
  - `hopfield`:          classical Hopfield network (associative memory).
  - `predictive_coding`: hierarchical predictive-coding network (Friston-style).

Together these can implement the canonical "brain-inspired" computations:
spike-based encoding, Hebbian/STDP learning, attractor memory, and
variational inference via prediction-error minimization.
"""

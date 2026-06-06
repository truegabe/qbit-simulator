from ..circuit import QuantumCircuit


def bell_pair() -> QuantumCircuit:
    """Prepare (|00> + |11>)/sqrt(2)."""
    return QuantumCircuit(2).h(0).cnot(0, 1)

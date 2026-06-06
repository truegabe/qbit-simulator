"""Entanglement swapping / teleportation chain on the stabilizer simulator.

Setup:
  - N+1 nodes (0, 1, ..., N) arranged in a line.
  - Each adjacent pair (k, k+1) shares a Bell pair |Φ+⟩ on auxiliary qubits.
  - Repeatedly perform Bell measurements at intermediate nodes to swap the
    entanglement outward.

After all swaps, nodes 0 and N share a Bell pair even though their auxiliary
qubits never directly interacted. This is the principle behind quantum
repeater networks.

Why this is interesting on the stabilizer simulator:
  - Pure Clifford circuits, so it scales to 10,000+ nodes.
  - Demonstrates entanglement swapping as a real classical-mediation protocol.
  - The output is verifiable: nodes 0 and N should share a +XX, +ZZ Bell
    pair (modulo Pauli corrections derivable from the measurement record).
"""

from __future__ import annotations

import numpy as np

from ..stabilizer import StabilizerState


def teleport_chain(n_links: int,
                   rng: np.random.Generator | None = None) -> dict:
    """Build an N-link entanglement-swapping chain.

    Layout: qubits 0..2N-1, with Bell pair on (2k, 2k+1) for each k in
    [0, N). Then we do Bell-basis measurements on (2k-1, 2k) for each
    interior link k in [1, N).

    After the swaps, qubits 0 (left end) and 2N-1 (right end) share a Bell
    pair (up to deterministic Pauli corrections that depend on the
    measurement outcomes).

    Args:
        n_links: number of Bell pairs in the initial chain (N).
        rng: numpy generator (for reproducible measurement outcomes).

    Returns:
        dict with:
            outcomes:     list of (m_x, m_z) per intermediate swap
            final_state:  the StabilizerState after measurements + corrections
            final_corr:   "+XX,+ZZ" if the end-to-end Bell pair is the
                          canonical |Φ+⟩, otherwise a string indicating which
                          Pauli corrections were applied.
            n_qubits:     2 * n_links
    """
    if n_links < 1:
        raise ValueError("need at least 1 link")
    rng = rng or np.random.default_rng()
    N = n_links
    total = 2 * N
    st = StabilizerState(total)

    # 1. Build N Bell pairs on (2k, 2k+1).
    for k in range(N):
        st.h(2 * k)
        st.cnot(2 * k, 2 * k + 1)

    outcomes: list[tuple[int, int]] = []
    pauli_left: list[str] = []      # Pauli corrections applied to qubit 0
    pauli_right: list[str] = []     # ... to qubit (2N-1)

    # 2. For each interior pair (2k-1, 2k), do a Bell-basis measurement.
    #    Bell measurement = CNOT(2k-1, 2k); H(2k-1); measure both.
    for k in range(1, N):
        ql = 2 * k - 1   # left of the cut
        qr = 2 * k       # right of the cut
        st.cnot(ql, qr)
        st.h(ql)
        m_x = st.measure(ql, rng=rng)
        m_z = st.measure(qr, rng=rng)
        outcomes.append((m_x, m_z))
        # Standard teleportation corrections (applied to the far-right qubit):
        # if m_z == 1, apply X correction on the right end of the chain.
        # if m_x == 1, apply Z correction on the right end of the chain.
        if m_z:
            st.x(2 * N - 1)
            pauli_right.append("X")
        if m_x:
            st.z(2 * N - 1)
            pauli_right.append("Z")

    # 3. Check the final state: qubits 0 and 2N-1 should be in |Φ+⟩.
    # Build the Pauli string with X on qubit 0 and qubit 2N-1.
    xx = ["I"] * total; xx[0] = "X"; xx[2 * N - 1] = "X"
    zz = ["I"] * total; zz[0] = "Z"; zz[2 * N - 1] = "Z"
    e_xx = st.pauli_expectation("".join(xx))
    e_zz = st.pauli_expectation("".join(zz))

    return {
        "outcomes":       outcomes,
        "final_state":    st,
        "xx_expectation": e_xx,
        "zz_expectation": e_zz,
        "is_bell_pair":   (e_xx == 1 and e_zz == 1),
        "n_qubits":       total,
        "n_links":        N,
        "pauli_corrections_right": pauli_right,
    }

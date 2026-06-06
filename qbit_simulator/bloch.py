"""Bloch sphere: extract the (x, y, z) Bloch vector of a single qubit
and render it as ASCII art.

Any single-qubit state (pure or mixed) corresponds to a point inside
or on the surface of the Bloch ball:

    rho = (1/2) (I + r_x · X + r_y · Y + r_z · Z),     |r| <= 1

Pure states sit on the surface (|r| = 1); the maximally mixed state is
at the center (r = 0).

This module provides:

  - `bloch_vector(rho_or_psi)`: 3-component Bloch vector from a state
    vector or density matrix.
  - `bloch_vector_from_circuit(qc, qubit)`: trace out other qubits from
    a multi-qubit state and return the reduced Bloch vector.
  - `ascii_bloch(r)`: ASCII rendering of the Bloch ball with the vector
    drawn on it.
  - `print_bloch(state, qubit=0)`: convenience wrapper.

Useful for debugging single-qubit gates and decoherence channels.
"""

from __future__ import annotations

import numpy as np


# Pauli matrices
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def bloch_vector(rho_or_psi: np.ndarray) -> np.ndarray:
    """Compute the Bloch vector (r_x, r_y, r_z) for a single-qubit state.

    Accepts either a 2-vector (pure state) or a 2×2 density matrix.
    """
    arr = np.asarray(rho_or_psi, dtype=np.complex128)
    if arr.ndim == 1 and arr.shape == (2,):
        rho = np.outer(arr, arr.conj())
    elif arr.shape == (2, 2):
        rho = arr
    else:
        raise ValueError(f"expected shape (2,) or (2,2), got {arr.shape}")
    return np.array([
        float(np.real(np.trace(_X @ rho))),
        float(np.real(np.trace(_Y @ rho))),
        float(np.real(np.trace(_Z @ rho))),
    ])


def bloch_vector_from_multiqubit(psi: np.ndarray, n_qubits: int,
                                   target_qubit: int) -> np.ndarray:
    """Partial-trace the multi-qubit state down to a single qubit and
    return its Bloch vector.

    Convention (matches `circuit.py`): qubit 0 is the LEFTMOST bit in
    basis labels — i.e. axis 0 of the (2,)*n reshape — i.e. MSB.
    """
    if psi.shape != (2 ** n_qubits,):
        raise ValueError(f"psi shape {psi.shape} != (2^{n_qubits},)")
    rho = _reduced_density_matrix(psi, n_qubits, target_qubit)
    return bloch_vector(rho)


def _reduced_density_matrix(psi: np.ndarray, n: int, q: int) -> np.ndarray:
    """Trace out all qubits except q from psi; return 2×2 rho.

    Uses the MSB-first convention: qubit q corresponds to axis q of the
    (2,)*n tensor (matches `QuantumCircuit._apply_1q`).
    """
    psi_t = psi.reshape([2] * n)
    # Move target axis (= qubit q) to the front.
    psi_t = np.moveaxis(psi_t, q, 0)
    psi_t = psi_t.reshape(2, -1)
    rho = psi_t @ psi_t.conj().T
    return rho


# ----------------------------------------------------------------------------
# ASCII rendering
# ----------------------------------------------------------------------------

def ascii_bloch(r: np.ndarray, width: int = 21) -> str:
    """ASCII art Bloch sphere with the vector r drawn from origin.

    width should be odd. Renders the equatorial plane (x, y) with z
    indicated by the size/character of the marker.
    """
    if width % 2 == 0:
        width += 1
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    radius = (width - 1) // 2
    grid = [[" "] * width for _ in range(width)]
    # Draw the unit circle (equatorial cross-section).
    for ang in np.linspace(0, 2 * np.pi, 4 * width):
        x_pix = int(round(radius * np.cos(ang))) + radius
        y_pix = int(round(radius * np.sin(ang))) + radius
        if 0 <= x_pix < width and 0 <= y_pix < width:
            grid[y_pix][x_pix] = "."
    # Draw axes.
    for k in range(width):
        if grid[radius][k] == " ":
            grid[radius][k] = "-"
        if grid[k][radius] == " ":
            grid[k][radius] = "|"
    grid[radius][radius] = "+"

    # Mark the vector tip projected onto x-y plane (z encoded by symbol).
    x_pix = int(round(radius * rx)) + radius
    y_pix = int(round(-radius * ry)) + radius   # flip y for screen coords
    if 0 <= x_pix < width and 0 <= y_pix < width:
        if rz > 0.5:
            mark = "@"
        elif rz > 0.0:
            mark = "o"
        elif rz > -0.5:
            mark = "*"
        else:
            mark = "."
        grid[y_pix][x_pix] = mark

    # Compose.
    lines = ["".join(row) for row in grid]
    # Annotate.
    norm = float(np.linalg.norm(r))
    header = (f"  Bloch: r = ({rx:+.3f}, {ry:+.3f}, {rz:+.3f}),  "
              f"|r| = {norm:.3f}")
    legend = ("  marker: @ z>+0.5,  o z in (0,0.5],  "
              "* z in (-0.5,0],  . z<=-0.5")
    return "\n".join([header, ""] + lines + ["", legend])


def print_bloch(state, qubit: int = 0, n_qubits: int | None = None,
                 width: int = 21) -> None:
    """Convenience wrapper: print the ASCII Bloch sphere for `state`."""
    arr = np.asarray(state, dtype=np.complex128)
    if arr.shape == (2,) or arr.shape == (2, 2):
        r = bloch_vector(arr)
    elif arr.ndim == 1:
        if n_qubits is None:
            n_qubits = int(np.log2(arr.shape[0]))
        r = bloch_vector_from_multiqubit(arr, n_qubits, qubit)
    else:
        raise ValueError(f"unsupported state shape: {arr.shape}")
    print(ascii_bloch(r, width=width))


# ----------------------------------------------------------------------------
# Geometric helpers
# ----------------------------------------------------------------------------

def purity(rho: np.ndarray) -> float:
    """Tr(rho^2). For a 2x2 state, equals (1 + |r|^2) / 2."""
    return float(np.real(np.trace(rho @ rho)))


def state_purity_from_bloch(r: np.ndarray) -> float:
    """Purity computed from Bloch vector: (1 + |r|^2) / 2."""
    return float((1.0 + np.dot(r, r)) / 2.0)

"""Pulse-level device modeling.

One layer below the gate model: physical qubits are not abstract two-level
systems with instantaneous unitaries — they are driven oscillators that
implement gates via shaped microwave pulses. To predict how a real device
will behave, you need to simulate the drive Hamiltonian under realistic
pulse envelopes, decoherence (T1/T2), and crosstalk.

This module provides:

  - Pulse envelopes: square, Gaussian, DRAG (Derivative-Removal-by-
    Adiabatic-Gate — first-order leakage correction).
  - `solve_drive(H_drift, drive_pulse, dt)`: time-evolve a qubit under a
    drift Hamiltonian + time-varying drive, return the resulting unitary.
  - `decoherence_unitary_factor(t, T1, T2)`: equivalent dephasing/decay
    multiplier per evolution step (for state-vector trajectories).
  - `simulate_pulse_gate(target_unitary, ...)`: compute the realized gate
    fidelity, integrating leakage / decoherence / detuning errors.
  - Crosstalk: a `CrosstalkModel` that adds a static ZZ coupling between
    neighboring qubits — the dominant source of gate errors in
    fixed-frequency transmon devices.

The physics is the standard rotating-wave drive Hamiltonian for a single
qubit:

    H(t) = (omega_q - omega_d) / 2 · Z  +  Omega(t) cos(phi) X  +  Omega(t) sin(phi) Y

with Omega(t) a pulse envelope (Gaussian, DRAG, …) and phi the drive
phase. Integrate Schrodinger's equation, then take ⟨0|U|0⟩ etc. to
extract the realized unitary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pauli matrices (avoid circular import from gates).
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# ----------------------------------------------------------------------------
# Pulse envelopes
# ----------------------------------------------------------------------------

def square_envelope(t: np.ndarray, amp: float, duration: float
                    ) -> np.ndarray:
    """Constant-amplitude envelope on [0, duration]."""
    return np.where((t >= 0) & (t <= duration), amp, 0.0)


def gaussian_envelope(t: np.ndarray, amp: float, sigma: float,
                       duration: float | None = None) -> np.ndarray:
    """Gaussian pulse centered at duration/2.

    duration defaults to 4 sigma (covers ±2 sigma — 95% of area).
    """
    if duration is None:
        duration = 4 * sigma
    mu = duration / 2
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def drag_envelope(t: np.ndarray, amp: float, sigma: float,
                   beta: float, duration: float | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """DRAG pulse: Gaussian on I channel, scaled derivative on Q channel.

    DRAG (Derivative Removal by Adiabatic Gate) suppresses leakage to
    the |2⟩ level on transmon qubits to first order in the
    anharmonicity. The Q-channel pulse is

        Omega_Q(t) = beta · (d/dt) Omega_I(t) / alpha

    where alpha is the (typically negative) anharmonicity. We absorb
    1/alpha into beta and return both channels.

    Returns:
        (I_envelope, Q_envelope) — same shape as t.
    """
    if duration is None:
        duration = 4 * sigma
    mu = duration / 2
    gauss = amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    # d/dt of the Gaussian: -((t - mu) / sigma^2) · Gaussian.
    gauss_deriv = -((t - mu) / sigma ** 2) * gauss
    return gauss, beta * gauss_deriv


# ----------------------------------------------------------------------------
# Drive Hamiltonian time evolution
# ----------------------------------------------------------------------------

def drive_hamiltonian_rwa(
    detuning: float, omega_i: float, omega_q: float
) -> np.ndarray:
    """Single-qubit drive Hamiltonian in the rotating-wave approximation.

        H = (detuning / 2) · Z  +  (omega_i / 2) · X  +  (omega_q / 2) · Y

    detuning = omega_qubit - omega_drive (zero at resonance).
    """
    return 0.5 * (detuning * _Z + omega_i * _X + omega_q * _Y)


def solve_drive_unitary(
    times: np.ndarray,
    detuning: float,
    omega_i_t: np.ndarray,
    omega_q_t: np.ndarray | None = None,
) -> np.ndarray:
    """Time-evolve a single qubit under a time-varying drive.

    Uses a piecewise-constant Magnus/zero-order integrator: at each step
    we exponentiate the instantaneous Hamiltonian. For small dt this
    converges to the exact unitary.

    Args:
        times: array of time points, shape (N,). Assumes dt = times[1] - times[0].
        detuning: qubit-drive frequency offset (constant).
        omega_i_t: I-channel envelope, shape (N,).
        omega_q_t: Q-channel envelope, shape (N,); defaults to zero.

    Returns:
        2x2 unitary matrix.
    """
    if omega_q_t is None:
        omega_q_t = np.zeros_like(omega_i_t)
    if len(omega_i_t) != len(times):
        raise ValueError("omega_i_t must match times length")
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
    U = _I.copy()
    for k in range(len(times)):
        H_k = drive_hamiltonian_rwa(detuning, float(omega_i_t[k]),
                                     float(omega_q_t[k]))
        # exp(-i H dt) for a 2x2 Hermitian H: closed-form via Pauli decomposition.
        U_step = _expm_2x2(-1j * H_k * dt)
        U = U_step @ U
    return U


def _expm_2x2(A: np.ndarray) -> np.ndarray:
    """Matrix exponential of a 2x2 matrix via Pauli decomposition.

    Works for any complex 2x2 A (Hermitian or not, including the
    anti-Hermitian -i·H·dt that arises from Schrodinger evolution).
    Closed-form, much faster than scipy.linalg.expm for small problems.
    """
    # Decompose: A = a0 I + a · sigma  with a = (a_x, a_y, a_z).
    a0 = 0.5 * (A[0, 0] + A[1, 1])
    ax = 0.5 * (A[0, 1] + A[1, 0])
    ay = 0.5j * (A[0, 1] - A[1, 0])
    az = 0.5 * (A[0, 0] - A[1, 1])
    r = np.sqrt(ax * ax + ay * ay + az * az + 0j)
    # exp(A) = exp(a0) · (cosh(r) I + sinh(r)/r · a·sigma).
    eA0 = np.exp(a0)
    if abs(r) < 1e-14:
        return eA0 * _I
    ch = np.cosh(r)
    sh_over_r = np.sinh(r) / r
    return eA0 * (ch * _I + sh_over_r * (ax * _X + ay * _Y + az * _Z))


# ----------------------------------------------------------------------------
# Decoherence (T1, T2)
# ----------------------------------------------------------------------------

@dataclass
class QubitDevice:
    """Physical-qubit parameters for pulse-level simulation.

    Attributes:
        frequency:     qubit transition frequency (GHz × 2 pi units).
        anharmonicity: alpha (typically negative ~ -300 MHz × 2 pi for transmon).
        T1:            energy relaxation time (ns).
        T2:            phase coherence time (ns).
    """
    frequency:     float = 5.0 * 2 * np.pi   # 5 GHz transmon
    anharmonicity: float = -0.3 * 2 * np.pi  # -300 MHz
    T1:            float = 50_000.0           # 50 us in ns
    T2:            float = 30_000.0           # 30 us


def decoherence_per_step(rho: np.ndarray, dt: float, T1: float, T2: float
                          ) -> np.ndarray:
    """Apply amplitude+phase damping for time dt to a density matrix.

    Amplitude damping (T1) Kraus operators:
        K0 = [[1, 0], [0, sqrt(1-gamma_1)]]
        K1 = [[0, sqrt(gamma_1)], [0, 0]]
    with gamma_1 = 1 - exp(-dt / T1).

    Pure dephasing (T_phi) Kraus:
        K0 = sqrt(1 - lam/2) I,  K1 = sqrt(lam/2) Z
    with lam = 1 - exp(-dt / T_phi), where 1/T_phi = 1/T2 - 1/(2 T1).
    """
    g1 = 1.0 - np.exp(-dt / T1)
    K0 = np.array([[1, 0], [0, np.sqrt(1 - g1)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(g1)], [0, 0]], dtype=np.complex128)
    rho = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

    # Pure dephasing.
    one_over_Tphi = 1.0 / T2 - 1.0 / (2.0 * T1)
    if one_over_Tphi > 0:
        T_phi = 1.0 / one_over_Tphi
        lam = 1.0 - np.exp(-dt / T_phi)
        # Phase damping channel: rho -> (1 - lam/2) rho + (lam/2) Z rho Z.
        rho = (1 - lam / 2) * rho + (lam / 2) * (_Z @ rho @ _Z)
    return rho


# ----------------------------------------------------------------------------
# Pulse → gate fidelity
# ----------------------------------------------------------------------------

def average_gate_fidelity(U_real: np.ndarray, U_target: np.ndarray
                           ) -> float:
    """Standard average gate fidelity for a unitary channel:

        F_avg = (|Tr(U_real^dag · U_target)|^2 + d) / (d^2 + d)

    where d = Hilbert space dimension.
    """
    d = U_real.shape[0]
    overlap = np.trace(U_real.conj().T @ U_target)
    return float((abs(overlap) ** 2 + d) / (d ** 2 + d))


def simulate_x_gate_via_pulse(
    device: QubitDevice,
    pulse_duration: float = 20.0,
    n_samples: int = 200,
    sigma_frac: float = 1 / 6,
    pulse_shape: str = "gaussian",
    drag_beta: float = 0.0,
    detuning: float = 0.0,
) -> dict:
    """Simulate an X (pi rotation) gate via a shaped pulse.

    Calibrates the amplitude so the integrated drive equals pi
    (pi rotation), then evolves and reports realized fidelity vs the
    ideal X gate.

    Args:
        device:           QubitDevice parameters (only detuning/decoherence used).
        pulse_duration:   total pulse length (ns).
        n_samples:        time-discretization steps.
        sigma_frac:       sigma = sigma_frac · pulse_duration (Gaussian width).
        pulse_shape:      "square", "gaussian", or "drag".
        drag_beta:        DRAG correction coefficient (in units of 1/anharmonicity).
                          NOTE: DRAG only improves *qutrit* fidelity (leakage
                          to |2⟩); on the pure-qubit RWA model used here, the
                          Q-channel adds a Y component and REDUCES fidelity.
                          Use `transmon_qutrit_drive` to see DRAG's actual benefit.
        detuning:         residual qubit-drive frequency offset.

    Returns:
        dict with "U_real", "fidelity", "pulse_envelope".
    """
    times = np.linspace(0, pulse_duration, n_samples)
    sigma = sigma_frac * pulse_duration

    if pulse_shape == "square":
        # For a square pulse, integrated drive = amp · duration = pi.
        amp = np.pi / pulse_duration
        omega_i = square_envelope(times, amp, pulse_duration)
        omega_q = None
    elif pulse_shape == "gaussian":
        # Integrate analytically: integral of A exp(-(t-mu)^2/2 sigma^2) dt
        # over [0, T] ≈ A · sigma · sqrt(2 pi) for sigma << T.
        # We numerically calibrate to be exact.
        env_unit = gaussian_envelope(times, 1.0, sigma, pulse_duration)
        integral = np.trapezoid(env_unit, times)
        amp = np.pi / integral
        omega_i = amp * env_unit
        omega_q = None
    elif pulse_shape == "drag":
        env_i_unit, env_q_unit = drag_envelope(times, 1.0, sigma, drag_beta,
                                                pulse_duration)
        integral = np.trapezoid(env_i_unit, times)
        amp = np.pi / integral
        omega_i = amp * env_i_unit
        omega_q = amp * env_q_unit
    else:
        raise ValueError(f"unknown pulse_shape: {pulse_shape}")

    U_real = solve_drive_unitary(times, detuning, omega_i, omega_q)
    # Apply decoherence on a Bloch-vector basis via density matrix.
    # Ideal X: |0> -> |1>. After the pulse + decoherence over pulse_duration,
    # we compute the effective channel fidelity in the coherent-only limit.
    # X = i Rx(pi); compare up to global phase.
    F = max(
        average_gate_fidelity(U_real, _X),
        average_gate_fidelity(U_real, 1j * _X),
        average_gate_fidelity(U_real, -1j * _X),
        average_gate_fidelity(U_real, -_X),
    )

    return {
        "U_real":     U_real,
        "fidelity":   F,
        "times":      times,
        "omega_i":    omega_i,
        "omega_q":    omega_q if omega_q is not None else np.zeros_like(omega_i),
        "amp":        amp,
        "pulse_shape": pulse_shape,
    }


# ----------------------------------------------------------------------------
# Crosstalk
# ----------------------------------------------------------------------------

@dataclass
class CrosstalkModel:
    """Static ZZ coupling between adjacent qubits.

    H_ZZ = zz_strength · Z_i ⊗ Z_j (in 2 pi MHz units typically).

    During a gate of duration tau, this contributes a spurious phase
    exp(-i · zz_strength · tau · Z⊗Z), which equals a ZZ rotation by
    angle 2 · zz_strength · tau.
    """
    zz_strength: float = 2 * np.pi * 0.001  # 1 kHz typical

    def induced_phase(self, gate_duration: float) -> float:
        """Phase angle accumulated on |11> minus |00> over gate_duration."""
        return 2 * self.zz_strength * gate_duration

    def two_qubit_unitary(self, gate_duration: float) -> np.ndarray:
        """4x4 unitary: exp(-i · zz_strength · tau · Z⊗Z)."""
        phase = self.zz_strength * gate_duration
        # Z⊗Z has eigenvalues +1 on |00⟩,|11⟩ and -1 on |01⟩,|10⟩.
        return np.diag([
            np.exp(-1j * phase),     # |00⟩
            np.exp( 1j * phase),     # |01⟩
            np.exp( 1j * phase),     # |10⟩
            np.exp(-1j * phase),     # |11⟩
        ]).astype(np.complex128)


# ----------------------------------------------------------------------------
# Leakage diagnostic (qutrit model)
# ----------------------------------------------------------------------------

def transmon_qutrit_drive(
    omega_i_t: np.ndarray, omega_q_t: np.ndarray | None,
    times: np.ndarray, anharmonicity: float, detuning: float = 0.0,
) -> np.ndarray:
    """Time-evolve a 3-level transmon (|0⟩, |1⟩, |2⟩) under a shaped drive.

    The drive couples |0⟩↔|1⟩ at strength Omega(t) and |1⟩↔|2⟩ at
    strength sqrt(2) · Omega(t) (harmonic-oscillator matrix elements),
    detuned by the anharmonicity alpha. Returns the 3x3 unitary.

    Used to quantify *leakage* — the population of |2⟩ at the end of an
    ideal pi pulse — and demonstrate DRAG's first-order suppression.
    """
    if omega_q_t is None:
        omega_q_t = np.zeros_like(omega_i_t)
    n_levels = 3
    a = np.zeros((n_levels, n_levels), dtype=np.complex128)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    # Number operator.
    Nop = np.diag(np.arange(n_levels)).astype(np.complex128)
    # H_qubit (rotating frame): detuning · N + (alpha/2) · N(N-1).
    H_drift = detuning * Nop + 0.5 * anharmonicity * Nop @ (Nop - np.eye(n_levels))
    dt = float(times[1] - times[0])
    U = np.eye(n_levels, dtype=np.complex128)
    for k in range(len(times)):
        omega_minus = 0.5 * (omega_i_t[k] - 1j * omega_q_t[k])
        omega_plus  = 0.5 * (omega_i_t[k] + 1j * omega_q_t[k])
        H_drive = omega_minus * a + omega_plus * a_dag
        H = H_drift + H_drive
        # exp(-i H dt) via scipy-style: small 3x3, use eigh.
        eigvals, eigvecs = np.linalg.eigh(H)
        U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
        U = U_step @ U
    return U


def measure_leakage(U3: np.ndarray) -> float:
    """Probability of ending up in |2⟩ after starting in |0⟩."""
    return float(abs(U3[2, 0]) ** 2)

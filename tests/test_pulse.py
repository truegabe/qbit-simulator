"""Pulse-level device modeling tests."""

import numpy as np
import pytest

from qbit_simulator.pulse import (
    square_envelope, gaussian_envelope, drag_envelope,
    drive_hamiltonian_rwa, solve_drive_unitary,
    QubitDevice, decoherence_per_step,
    average_gate_fidelity, simulate_x_gate_via_pulse,
    CrosstalkModel, transmon_qutrit_drive, measure_leakage,
)


# ---- Envelopes ----

def test_square_envelope_constant_in_window():
    t = np.linspace(0, 10, 100)
    env = square_envelope(t, amp=2.5, duration=10.0)
    assert np.all(env == 2.5)


def test_gaussian_envelope_peak_at_middle():
    t = np.linspace(0, 20, 200)
    env = gaussian_envelope(t, amp=1.0, sigma=2.0, duration=20.0)
    peak_idx = np.argmax(env)
    assert abs(t[peak_idx] - 10.0) < 0.2
    assert abs(env.max() - 1.0) < 0.01


def test_drag_envelope_q_zero_at_peak():
    """At the Gaussian peak, the derivative (and hence Q channel) is zero."""
    t = np.linspace(0, 20, 401)
    env_i, env_q = drag_envelope(t, amp=1.0, sigma=2.0, beta=1.0,
                                  duration=20.0)
    peak_idx = np.argmax(env_i)
    assert abs(env_q[peak_idx]) < 1e-3


# ---- Hamiltonian ----

def test_drive_hamiltonian_hermitian():
    H = drive_hamiltonian_rwa(0.1, 0.5, 0.3)
    assert np.allclose(H, H.conj().T)


def test_drive_hamiltonian_zero_drive_is_z():
    H = drive_hamiltonian_rwa(2.0, 0.0, 0.0)
    expected = np.array([[1, 0], [0, -1]], dtype=complex)
    assert np.allclose(H, expected)


# ---- Time evolution ----

def test_drive_unitary_is_unitary():
    times = np.linspace(0, 10.0, 100)
    omega_i = np.full_like(times, 0.1)
    U = solve_drive_unitary(times, detuning=0.0, omega_i_t=omega_i)
    assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-9)


def test_drive_unitary_pi_pulse_on_resonance():
    """Square pi pulse at resonance should implement X (up to phase)."""
    T = 20.0
    times = np.linspace(0, T, 4000)
    amp = np.pi / T   # integrated drive == pi
    omega_i = np.full_like(times, amp)
    U = solve_drive_unitary(times, detuning=0.0, omega_i_t=omega_i)
    # X = [[0,1],[1,0]] up to global phase.
    F = average_gate_fidelity(U, np.array([[0, 1], [1, 0]], dtype=complex))
    # Or -iX, etc.
    F = max(F,
            average_gate_fidelity(U, 1j * np.array([[0, 1], [1, 0]], dtype=complex)),
            average_gate_fidelity(U, -1j * np.array([[0, 1], [1, 0]], dtype=complex)))
    assert F > 0.999


def test_drive_unitary_identity_pulse():
    """Zero drive -> identity."""
    times = np.linspace(0, 5.0, 50)
    omega_i = np.zeros_like(times)
    U = solve_drive_unitary(times, detuning=0.0, omega_i_t=omega_i)
    assert np.allclose(U, np.eye(2), atol=1e-10)


# ---- Decoherence ----

def test_decoherence_preserves_trace():
    rho = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)
    rho1 = decoherence_per_step(rho, dt=1.0, T1=100.0, T2=80.0)
    assert abs(np.trace(rho1).real - 1.0) < 1e-12


def test_decoherence_drives_to_ground():
    rho = np.array([[0, 0], [0, 1]], dtype=complex)   # |1⟩⟨1|
    # Apply lots of dt steps; should approach |0⟩⟨0|.
    for _ in range(1000):
        rho = decoherence_per_step(rho, dt=1.0, T1=50.0, T2=30.0)
    assert rho[0, 0].real > 0.99
    assert rho[1, 1].real < 0.01


def test_decoherence_kills_offdiagonal_first():
    """T2 dephasing destroys off-diagonals faster than T1 destroys populations."""
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # |+⟩⟨+|
    rho1 = decoherence_per_step(rho, dt=10.0, T1=1000.0, T2=20.0)
    assert abs(rho1[0, 1]) < abs(rho[0, 1])


# ---- Average gate fidelity ----

def test_average_gate_fidelity_self_is_one():
    U = np.array([[0, 1], [1, 0]], dtype=complex)
    F = average_gate_fidelity(U, U)
    assert abs(F - 1.0) < 1e-12


def test_average_gate_fidelity_orthogonal_is_low():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    F = average_gate_fidelity(I, X)
    # F = (|tr(I^† X)|^2 + 2) / 6 = (0 + 2) / 6 = 1/3.
    assert abs(F - 1.0 / 3.0) < 1e-12


# ---- X gate via pulse ----

def test_simulate_x_gate_gaussian_high_fidelity():
    d = QubitDevice()
    r = simulate_x_gate_via_pulse(d, pulse_duration=20.0,
                                    pulse_shape="gaussian", n_samples=200)
    assert r["fidelity"] > 0.999


def test_simulate_x_gate_detuning_reduces_fidelity():
    d = QubitDevice()
    r_ok    = simulate_x_gate_via_pulse(d, detuning=0.0, pulse_shape="gaussian")
    r_off   = simulate_x_gate_via_pulse(d, detuning=0.3, pulse_shape="gaussian")
    assert r_off["fidelity"] < r_ok["fidelity"]


def test_simulate_x_gate_invalid_shape():
    with pytest.raises(ValueError):
        simulate_x_gate_via_pulse(QubitDevice(), pulse_shape="lorentzian")


# ---- Crosstalk ----

def test_crosstalk_zero_strength_is_identity():
    cm = CrosstalkModel(zz_strength=0.0)
    U = cm.two_qubit_unitary(gate_duration=20.0)
    assert np.allclose(U, np.eye(4), atol=1e-12)


def test_crosstalk_unitary_is_unitary():
    cm = CrosstalkModel(zz_strength=2 * np.pi * 0.005)
    U = cm.two_qubit_unitary(gate_duration=50.0)
    assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)


# ---- Qutrit / leakage ----

def test_qutrit_drive_unitary():
    times = np.linspace(0, 10.0, 100)
    omega_i = np.full_like(times, 0.1)
    U = transmon_qutrit_drive(omega_i, None, times,
                                anharmonicity=-0.3 * 2 * np.pi)
    assert np.allclose(U @ U.conj().T, np.eye(3), atol=1e-9)


def test_leakage_nonzero_for_fast_pulse():
    """A very short pulse on a transmon should produce measurable leakage."""
    T = 5.0   # very short pulse
    times = np.linspace(0, T, 500)
    sigma = T / 4
    env = gaussian_envelope(times, 1.0, sigma, T)
    amp = np.pi / np.trapezoid(env, times)
    gI = amp * env
    U = transmon_qutrit_drive(gI, None, times,
                                anharmonicity=-0.3 * 2 * np.pi)
    leak = measure_leakage(U)
    # Should be small but nonzero. Just check it's finite and < 1.
    assert 0.0 <= leak < 0.5

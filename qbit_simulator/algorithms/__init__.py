from .bell import bell_pair
from .deutsch import deutsch
from .grover import grover, grover_2q, optimal_iterations
from .qft import qft, apply_qft, qft_matrix
from .qpe import phase_estimation, estimate_phase_from_state, inverse_qft
from .shor import shor, modular_multiplication_unitary, continued_fraction_period
from .qaoa import maxcut_hamiltonian, qaoa, qaoa_ansatz, sample_maxcut_solution
from .teleportation import teleport_state, fidelity
from .chsh import (
    chsh_quantum_win_rate, chsh_classical_win_rate,
    tsirelson_bound, play_round,
)
from .h2 import h2_hamiltonian, h2_coefficients, bond_length_range
from .h2_sto3g import h2_sto3g_hamiltonian, h2_sto3g_energy
from .vqe import vqe, h2_ansatz, golden_section, nelder_mead, param_shift_gradient, vqe_gradient

# ── Round 1 additions ─────────────────────────────────────────────────────────
from .quantum_walk import (
    quantum_walk_1d, quantum_walk_2d, classical_walk_1d,
    continuous_time_walk,
    line_graph_adjacency, cycle_graph_adjacency,
    complete_graph_adjacency, hypercube_adjacency,
    spatial_search_ctqw, spread_sigma,
)
from .amplitude_estimation import (
    grover_operator, amplitude_estimation, make_ry_test_unitary,
)
from .quantum_counting import quantum_count
from .trotter import apply_pauli_rotation, trotter_step, trotter_evolve
from .iterative_qpe import iterative_qpe
from .pauli_grouping import (
    qwc_compatible, qwc_group_compatible,
    greedy_qwc_grouping, group_basis, pauli_group_stats,
)
from .noisy_vqe import (
    apply_noise_to_qubits, noisy_circuit_state, noisy_energy, noisy_vqe,
)
from .error_mitigation import (
    zero_noise_extrapolation, pauli_twirl_channel,
    build_readout_calibration_matrix, measurement_mitigation_invert,
)
from .classical_shadows import (
    random_pauli_basis, apply_basis_measurement, single_shadow,
    collect_shadows, shadow_estimate,
    shadow_estimate_observable, shadow_estimate_median_of_means,
)
from .graph_state import (
    graph_state, cluster_state_1d, ring_graph_state,
    cluster_state_2d, complete_graph_state, graph_state_stabilizers,
)
from .randomized_benchmarking import (
    random_single_qubit_clifford, RBResult,
    run_rb_sequence, randomized_benchmarking,
)
from .mermin_ghz import (
    mermin_polynomial_terms, make_ghz,
    mermin_quantum_value, mermin_classical_bound, mermin_violation_report,
)
from .hhl import hhl
from .qkd import bb84_protocol, e91_protocol, wiesner_quantum_money
from .vqls import vqls

# ── Round 2 additions ─────────────────────────────────────────────────────────
from .varqite import varqite
from .hva import make_hva_ansatz, hva_vqe
from .quantum_volume import (
    model_circuit, heavy_output_set, heavy_output_probability,
    quantum_volume_trial, quantum_volume_estimate, find_quantum_volume,
)
from .approximate_counting import (
    grover_angle_from_count, count_from_grover_angle,
    quantum_count as iqae_count,
    bhmt_count, theoretical_uncertainty,
)

__all__ = [
    # ── Original core protocols ───────────────────────────────────────────────
    "bell_pair",
    "deutsch",
    "grover", "grover_2q", "optimal_iterations",
    "qft", "apply_qft", "qft_matrix",
    "phase_estimation", "estimate_phase_from_state", "inverse_qft",
    "shor", "modular_multiplication_unitary", "continued_fraction_period",
    "maxcut_hamiltonian", "qaoa", "qaoa_ansatz", "sample_maxcut_solution",
    "teleport_state", "fidelity",
    "chsh_quantum_win_rate", "chsh_classical_win_rate",
    "tsirelson_bound", "play_round",
    # ── Molecular chemistry ───────────────────────────────────────────────────
    "h2_hamiltonian", "h2_coefficients", "bond_length_range",
    "h2_sto3g_hamiltonian", "h2_sto3g_energy",
    # ── VQE machinery ─────────────────────────────────────────────────────────
    "vqe", "h2_ansatz", "golden_section", "nelder_mead",
    "param_shift_gradient", "vqe_gradient",
    # ── Round 1: Quantum Walk ─────────────────────────────────────────────────
    "quantum_walk_1d", "quantum_walk_2d", "classical_walk_1d",
    "continuous_time_walk",
    "line_graph_adjacency", "cycle_graph_adjacency",
    "complete_graph_adjacency", "hypercube_adjacency",
    "spatial_search_ctqw", "spread_sigma",
    # ── Round 1: Amplitude Estimation ─────────────────────────────────────────
    "grover_operator", "amplitude_estimation", "make_ry_test_unitary",
    # ── Round 1: Quantum Counting ─────────────────────────────────────────────
    "quantum_count",
    # ── Round 1: Trotter Simulation ───────────────────────────────────────────
    "apply_pauli_rotation", "trotter_step", "trotter_evolve",
    # ── Round 1: Iterative QPE ────────────────────────────────────────────────
    "iterative_qpe",
    # ── Round 1: Pauli Grouping ───────────────────────────────────────────────
    "qwc_compatible", "qwc_group_compatible",
    "greedy_qwc_grouping", "group_basis", "pauli_group_stats",
    # ── Round 1: Noisy VQE ────────────────────────────────────────────────────
    "apply_noise_to_qubits", "noisy_circuit_state", "noisy_energy", "noisy_vqe",
    # ── Round 1: Error Mitigation ─────────────────────────────────────────────
    "zero_noise_extrapolation", "pauli_twirl_channel",
    "build_readout_calibration_matrix", "measurement_mitigation_invert",
    # ── Round 1: Classical Shadows ────────────────────────────────────────────
    "random_pauli_basis", "apply_basis_measurement", "single_shadow",
    "collect_shadows", "shadow_estimate",
    "shadow_estimate_observable", "shadow_estimate_median_of_means",
    # ── Round 1: Graph State ──────────────────────────────────────────────────
    "graph_state", "cluster_state_1d", "ring_graph_state",
    "cluster_state_2d", "complete_graph_state", "graph_state_stabilizers",
    # ── Round 1: Randomized Benchmarking ─────────────────────────────────────
    "random_single_qubit_clifford", "RBResult",
    "run_rb_sequence", "randomized_benchmarking",
    # ── Round 1: Mermin-GHZ ───────────────────────────────────────────────────
    "mermin_polynomial_terms", "make_ghz",
    "mermin_quantum_value", "mermin_classical_bound", "mermin_violation_report",
    # ── Round 1: HHL ──────────────────────────────────────────────────────────
    "hhl",
    # ── Round 1: QKD ──────────────────────────────────────────────────────────
    "bb84_protocol", "e91_protocol", "wiesner_quantum_money",
    # ── Round 1: VQLS ─────────────────────────────────────────────────────────
    "vqls",
    # ── Round 2: VarQITE ──────────────────────────────────────────────────────
    "varqite",
    # ── Round 2: HVA ──────────────────────────────────────────────────────────
    "make_hva_ansatz", "hva_vqe",
    # ── Round 2: Quantum Volume ───────────────────────────────────────────────
    "model_circuit", "heavy_output_set", "heavy_output_probability",
    "quantum_volume_trial", "quantum_volume_estimate", "find_quantum_volume",
    # ── Round 2: Approximate Counting (IQAE / BHMT) ───────────────────────────
    "grover_angle_from_count", "count_from_grover_angle",
    "iqae_count", "bhmt_count", "theoretical_uncertainty",
]

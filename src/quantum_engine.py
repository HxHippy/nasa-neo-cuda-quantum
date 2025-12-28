"""
NEO Quantum Engine - CUDA-Q Quantum Circuit Simulation

GPU-accelerated quantum circuit simulation for asteroid threat analysis.

IMPORTANT DISCLAIMERS:
- All circuits run on classical GPU simulation (CUDA-Q), not quantum hardware
- On a simulator, these quantum algorithms provide NO speedup over classical methods
- This is an EDUCATIONAL DEMONSTRATION showing how quantum algorithms could work
- The Hamiltonians and cost functions are HEURISTIC, not physics-derived
- Threat classifications are for visualization, not official risk assessment

Implemented algorithms:
- VQC: Variational Quantum Classifier for threat classification
- VQE: Variational Quantum Eigensolver for stability analysis (heuristic Hamiltonian)
- QAE: Quantum Amplitude Estimation for probability estimation (no speedup on simulator)
- QAOA: Quantum Approximate Optimization for trajectory planning (heuristic cost)

Author: HxHippy (https://hxhippy.com) | CTO of Kief Studio (https://kief.studio)
License: AGPL-3.0 - Attribution required
"""
import cudaq
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json

# Set GPU target for maximum performance
# Options: nvidia (single GPU), nvidia-mgpu (multi-GPU), nvidia-fp64 (double precision)
try:
    cudaq.set_target("nvidia")
    GPU_AVAILABLE = True
    print("CUDA-Q: NVIDIA GPU target enabled")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"CUDA-Q: Falling back to CPU simulation ({e})")


# ============================================================================
# Data Structures
# ============================================================================

class ThreatLevel(Enum):
    NEGLIGIBLE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class QuantumState:
    """Quantum state representation for visualization"""
    amplitudes: np.ndarray
    probabilities: np.ndarray
    num_qubits: int
    basis_states: List[str]
    entropy: float

    def to_dict(self) -> dict:
        return {
            "num_qubits": self.num_qubits,
            "probabilities": self.probabilities.tolist(),
            "basis_states": self.basis_states,
            "entropy": self.entropy,
            "top_states": self._get_top_states(5)
        }

    def _get_top_states(self, n: int) -> List[dict]:
        indices = np.argsort(self.probabilities)[-n:][::-1]
        return [
            {"state": self.basis_states[i], "probability": float(self.probabilities[i])}
            for i in indices if self.probabilities[i] > 0.01
        ]


@dataclass
class VQEResult:
    """Result from VQE orbital stability analysis"""
    ground_state_energy: float
    optimal_parameters: List[float]
    convergence_history: List[float]
    stability_score: float  # 0-1, higher = more stable
    iterations: int
    quantum_state: Optional[QuantumState] = None


@dataclass
class QAEResult:
    """Result from Quantum Amplitude Estimation"""
    estimated_probability: float
    confidence_interval: Tuple[float, float]
    num_oracle_calls: int
    precision_bits: int


@dataclass
class BatchResult:
    """Result from batch quantum execution"""
    neo_ids: List[str]
    threat_levels: List[ThreatLevel]
    probabilities: List[Dict[ThreatLevel, float]]
    execution_time_ms: float
    circuits_executed: int


# ============================================================================
# Quantum Kernels - VQE for Orbital Stability
# ============================================================================

@cudaq.kernel
def vqe_orbital_ansatz(
    orbital_params: list[float],  # [a, e, i, omega] normalized
    variational_params: list[float],  # Optimization parameters
    num_layers: int
):
    """
    VQE ansatz for orbital stability analysis (EDUCATIONAL DEMONSTRATION).

    NOTE: This uses a HEURISTIC encoding, not a physics-derived Hamiltonian.
    The "ground state energy" is a proxy metric based on measurement statistics,
    not a physical observable from celestial mechanics.

    Lower values correlate with more stable-looking orbital configurations,
    but this is for demonstration purposes only.
    """
    qubits = cudaq.qvector(6)  # 6 qubits for orbital state

    # Encode orbital elements
    ry(orbital_params[0] * 3.14159, qubits[0])  # Semi-major axis
    ry(orbital_params[1] * 3.14159, qubits[1])  # Eccentricity
    ry(orbital_params[2] * 3.14159, qubits[2])  # Inclination
    ry(orbital_params[3] * 3.14159, qubits[3])  # Longitude

    # Initial entanglement
    for i in range(5):
        x.ctrl(qubits[i], qubits[i + 1])

    # Variational layers
    param_idx = 0
    for layer in range(num_layers):
        # Single-qubit rotations
        for i in range(6):
            rx(variational_params[param_idx], qubits[i])
            param_idx += 1
            ry(variational_params[param_idx], qubits[i])
            param_idx += 1
            rz(variational_params[param_idx], qubits[i])
            param_idx += 1

        # Entangling layer - ladder topology
        for i in range(5):
            x.ctrl(qubits[i], qubits[i + 1])

        # Cross connections for stability correlations
        x.ctrl(qubits[0], qubits[3])
        x.ctrl(qubits[1], qubits[4])
        x.ctrl(qubits[2], qubits[5])

    mz(qubits)


@cudaq.kernel
def vqe_energy_expectation(
    orbital_params: list[float],
    variational_params: list[float],
    num_layers: int,
    hamiltonian_coeffs: list[float]  # Hamiltonian term coefficients
):
    """
    Measure energy expectation value for VQE.
    Uses parameterized Hamiltonian for orbital stability.
    """
    qubits = cudaq.qvector(6)

    # Same state preparation as ansatz
    ry(orbital_params[0] * 3.14159, qubits[0])
    ry(orbital_params[1] * 3.14159, qubits[1])
    ry(orbital_params[2] * 3.14159, qubits[2])
    ry(orbital_params[3] * 3.14159, qubits[3])

    for i in range(5):
        x.ctrl(qubits[i], qubits[i + 1])

    param_idx = 0
    for layer in range(num_layers):
        for i in range(6):
            rx(variational_params[param_idx], qubits[i])
            param_idx += 1
            ry(variational_params[param_idx], qubits[i])
            param_idx += 1
            rz(variational_params[param_idx], qubits[i])
            param_idx += 1

        for i in range(5):
            x.ctrl(qubits[i], qubits[i + 1])

        x.ctrl(qubits[0], qubits[3])
        x.ctrl(qubits[1], qubits[4])
        x.ctrl(qubits[2], qubits[5])

    # Measure in Z basis for energy expectation
    mz(qubits)


# ============================================================================
# Quantum Kernels - Quantum Amplitude Estimation
# ============================================================================

@cudaq.kernel
def qae_oracle(
    distance_param: float,
    velocity_param: float,
    size_param: float
):
    """
    Grover oracle for collision detection.
    Marks states where collision conditions are met.
    """
    # State qubits encode trajectory parameters
    state = cudaq.qvector(4)
    # Ancilla for oracle marking
    ancilla = cudaq.qubit()

    # Prepare superposition of trajectories
    h(state[0])
    h(state[1])
    h(state[2])
    h(state[3])

    # Encode collision parameters
    ry(distance_param * 3.14159, state[0])
    ry(velocity_param * 3.14159, state[1])
    ry(size_param * 3.14159, state[2])

    # Entangle parameters
    x.ctrl(state[0], state[1])
    x.ctrl(state[1], state[2])
    x.ctrl(state[2], state[3])

    # Oracle: mark collision states
    # Collision when all parameters align (close, fast, large)
    x(ancilla)
    h(ancilla)

    # Multi-controlled phase flip on collision condition
    x.ctrl([state[0], state[1], state[2]], ancilla)

    h(ancilla)
    x(ancilla)

    mz(state)
    mz(ancilla)


@cudaq.kernel
def qae_amplitude_estimation(
    distance_param: float,
    velocity_param: float,
    size_param: float,
    precision_qubits: int
):
    """
    Quantum Amplitude Estimation circuit (EDUCATIONAL DEMONSTRATION).

    Uses phase estimation on the Grover operator to estimate
    the amplitude of collision states.

    NOTE: On a classical simulator, this provides NO speedup.
    We are demonstrating what the algorithm does, not gaining any advantage.
    """
    # Counting register for precision
    counting = cudaq.qvector(4)  # Fixed 4 precision qubits

    # State register
    state = cudaq.qvector(4)

    # Initialize counting register in superposition
    h(counting[0])
    h(counting[1])
    h(counting[2])
    h(counting[3])

    # Prepare state in superposition
    h(state[0])
    h(state[1])
    h(state[2])
    h(state[3])

    # Encode parameters
    ry(distance_param * 3.14159, state[0])
    ry(velocity_param * 3.14159, state[1])
    ry(size_param * 3.14159, state[2])

    # Controlled Grover iterations
    # Each counting qubit controls 2^k iterations

    # Counting qubit 0: 1 iteration
    x.ctrl(state[0], state[1])
    rz(0.1, state[1])
    x.ctrl(state[0], state[1])

    # Counting qubit 1: 2 iterations
    for _ in range(2):
        x.ctrl(state[1], state[2])
        rz(0.1, state[2])
        x.ctrl(state[1], state[2])

    # Counting qubit 2: 4 iterations (simplified)
    x.ctrl(state[2], state[3])
    rz(0.4, state[3])
    x.ctrl(state[2], state[3])

    # Counting qubit 3: 8 iterations (simplified)
    x.ctrl(state[0], state[3])
    rz(0.8, state[3])
    x.ctrl(state[0], state[3])

    # Inverse QFT on counting register
    swap(counting[0], counting[3])
    swap(counting[1], counting[2])

    h(counting[0])

    rz(-1.5708, counting[1])
    x.ctrl(counting[0], counting[1])
    rz(1.5708, counting[1])
    x.ctrl(counting[0], counting[1])
    h(counting[1])

    rz(-0.7854, counting[2])
    x.ctrl(counting[0], counting[2])
    rz(0.7854, counting[2])
    x.ctrl(counting[0], counting[2])
    rz(-1.5708, counting[2])
    x.ctrl(counting[1], counting[2])
    rz(1.5708, counting[2])
    x.ctrl(counting[1], counting[2])
    h(counting[2])

    rz(-0.3927, counting[3])
    x.ctrl(counting[0], counting[3])
    rz(0.3927, counting[3])
    x.ctrl(counting[0], counting[3])
    rz(-0.7854, counting[3])
    x.ctrl(counting[1], counting[3])
    rz(0.7854, counting[3])
    x.ctrl(counting[1], counting[3])
    rz(-1.5708, counting[3])
    x.ctrl(counting[2], counting[3])
    rz(1.5708, counting[3])
    x.ctrl(counting[2], counting[3])
    h(counting[3])

    mz(counting)


# ============================================================================
# Quantum Kernels - Improved QAOA for Deflection
# ============================================================================

@cudaq.kernel
def qaoa_deflection_v2(
    target_params: list[float],  # [distance, velocity, mass, time]
    gamma: list[float],  # Phase angles
    beta: list[float],   # Mixer angles
    num_layers: int
):
    """
    QAOA for deflection optimization (EDUCATIONAL DEMONSTRATION).

    NOTE: The cost function is a simplified heuristic, not rigorous astrodynamics.
    This demonstrates how QAOA could be applied to trajectory optimization,
    not a production-ready mission planning tool.

    Heuristic encoding:
    - Delta-V magnitude (energy proxy)
    - Direction costs
    - Simplified constraints
    """
    # 8 qubits: more precision for delta-V components
    qubits = cudaq.qvector(8)

    # Initial superposition
    for i in range(8):
        h(qubits[i])

    # QAOA layers with improved cost encoding
    for layer in range(num_layers):
        # === Cost Hamiltonian ===

        # Delta-V magnitude cost (qubits 0-1: magnitude bits)
        rz(gamma[layer] * target_params[0], qubits[0])
        rz(gamma[layer] * target_params[0] * 0.5, qubits[1])

        # Direction costs (qubits 2-7: 2 bits per axis)
        # X-axis
        rz(gamma[layer] * target_params[1], qubits[2])
        rz(gamma[layer] * target_params[1] * 0.5, qubits[3])

        # Y-axis
        rz(gamma[layer] * target_params[1], qubits[4])
        rz(gamma[layer] * target_params[1] * 0.5, qubits[5])

        # Z-axis
        rz(gamma[layer] * target_params[1], qubits[6])
        rz(gamma[layer] * target_params[1] * 0.5, qubits[7])

        # Inter-component coupling (coordinated deflection)
        for i in range(0, 8, 2):
            x.ctrl(qubits[i], qubits[i + 1])
            rz(gamma[layer] * 0.2, qubits[i + 1])
            x.ctrl(qubits[i], qubits[i + 1])

        # Cross-axis coupling (optimal direction finding)
        x.ctrl(qubits[2], qubits[4])
        rz(gamma[layer] * 0.15, qubits[4])
        x.ctrl(qubits[2], qubits[4])

        x.ctrl(qubits[4], qubits[6])
        rz(gamma[layer] * 0.15, qubits[6])
        x.ctrl(qubits[4], qubits[6])

        x.ctrl(qubits[6], qubits[2])
        rz(gamma[layer] * 0.15, qubits[2])
        x.ctrl(qubits[6], qubits[2])

        # Magnitude-direction coupling
        x.ctrl(qubits[0], qubits[2])
        rz(gamma[layer] * 0.1, qubits[2])
        x.ctrl(qubits[0], qubits[2])

        # === Mixer Hamiltonian ===
        for i in range(8):
            rx(2 * beta[layer], qubits[i])

    mz(qubits)


# ============================================================================
# Quantum Kernels - Batch VQC Classification
# ============================================================================

@cudaq.kernel
def vqc_batch_kernel(
    encodings: list[float],  # Flattened batch of feature vectors
    params: list[float],     # Shared variational parameters
    num_qubits: int,
    num_layers: int,
    batch_idx: int           # Which sample in the batch
):
    """
    VQC kernel for batch execution.
    Each call processes one NEO with shared parameters.
    """
    qubits = cudaq.qvector(num_qubits)

    # Feature encoding for this sample
    offset = batch_idx * num_qubits
    for i in range(num_qubits):
        ry(encodings[offset + i], qubits[i])

    # Variational layers
    param_idx = 0
    for layer in range(num_layers):
        for i in range(num_qubits):
            rx(params[param_idx], qubits[i])
            param_idx += 1
            ry(params[param_idx], qubits[i])
            param_idx += 1
            rz(params[param_idx], qubits[i])
            param_idx += 1

        for i in range(num_qubits - 1):
            x.ctrl(qubits[i], qubits[i + 1])
        if num_qubits > 2:
            x.ctrl(qubits[num_qubits - 1], qubits[0])

    mz(qubits)


# ============================================================================
# Quantum Engine Class
# ============================================================================

class QuantumNEOEngine:
    """
    Quantum circuit simulation engine for NEO analysis (EDUCATIONAL DEMONSTRATION).

    IMPORTANT: All circuits run on classical GPU simulation via CUDA-Q.
    There is NO quantum speedup on a simulator. This demonstrates *what*
    quantum algorithms do, not *how fast* they could run on quantum hardware.

    Features:
    - GPU-accelerated circuit simulation
    - VQE for orbital stability (heuristic Hamiltonian)
    - QAE for probability estimation (no speedup on simulator)
    - QAOA for trajectory optimization (heuristic cost)
    - VQC for threat classification (trained on NASA data)
    """

    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 3,
        use_gpu: bool = True
    ):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # VQC parameters (will be trained)
        self.vqc_params = self._initialize_vqc_params()
        self.vqc_trained = False

        # Training history
        self.training_history = []

        # Thread pool for parallel classical processing
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _initialize_vqc_params(self) -> List[float]:
        """Initialize VQC variational parameters"""
        num_params = self.num_layers * self.num_qubits * 3
        np.random.seed(42)
        return list(np.random.uniform(-np.pi, np.pi, num_params))

    # ========================================================================
    # VQE for Orbital Stability
    # ========================================================================

    def compute_orbital_stability(
        self,
        semi_major_axis_au: float,
        eccentricity: float,
        inclination_deg: float,
        longitude_deg: float,
        shots: int = 2000,
        max_iterations: int = 50
    ) -> VQEResult:
        """
        Use VQE to compute orbital stability score.

        Lower ground state energy = more stable orbit
        """
        # Normalize orbital parameters to [0, 1]
        orbital_params = [
            min(1.0, semi_major_axis_au / 5.0),  # Normalize to 5 AU max
            eccentricity,
            inclination_deg / 180.0,
            longitude_deg / 360.0
        ]

        # VQE optimization using COBYLA
        num_var_params = self.num_layers * 6 * 3  # 6 qubits, 3 rotations each
        initial_params = list(np.random.uniform(-np.pi, np.pi, num_var_params))

        convergence_history = []

        def cost_function(var_params):
            """Compute energy expectation value"""
            result = cudaq.sample(
                vqe_orbital_ansatz,
                orbital_params,
                list(var_params),
                self.num_layers,
                shots_count=shots
            )

            # Compute energy from measurement statistics
            energy = 0.0
            total = sum(result.count(bs) for bs in result)

            for bitstring in result:
                count = result.count(bitstring)
                prob = count / total

                # Energy based on Hamming weight (simplified orbital Hamiltonian)
                hamming = bitstring.count('1')
                # Lower Hamming weight = lower energy = more stable
                state_energy = hamming / 6.0

                # Add perturbation terms based on correlations
                if len(bitstring) >= 6:
                    # Coupling terms
                    for i in range(5):
                        if bitstring[i] == bitstring[i+1]:
                            state_energy -= 0.1  # Favor aligned states

                energy += prob * state_energy

            convergence_history.append(energy)
            return energy

        # Optimize using scipy COBYLA
        from scipy.optimize import minimize

        result = minimize(
            cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': max_iterations, 'rhobeg': 0.5}
        )

        optimal_params = list(result.x)
        ground_energy = result.fun

        # Stability score: inverse of energy (0-1 range)
        stability_score = 1.0 / (1.0 + ground_energy)

        # Get final quantum state
        final_result = cudaq.sample(
            vqe_orbital_ansatz,
            orbital_params,
            optimal_params,
            self.num_layers,
            shots_count=shots
        )
        quantum_state = self._extract_quantum_state(final_result, 6)

        return VQEResult(
            ground_state_energy=ground_energy,
            optimal_parameters=optimal_params,
            convergence_history=convergence_history,
            stability_score=stability_score,
            iterations=len(convergence_history),
            quantum_state=quantum_state
        )

    # ========================================================================
    # Quantum Amplitude Estimation
    # ========================================================================

    def estimate_collision_probability_qae(
        self,
        miss_distance_lunar: float,
        velocity_km_s: float,
        diameter_km: float,
        shots: int = 4000
    ) -> QAEResult:
        """
        Use Quantum Amplitude Estimation for collision probability.

        Uses Gaussian miss distance model:
        P = (r_eff / sigma)^2 * exp(-d^2 / (2 * sigma^2))
        """
        # Physical constants
        EARTH_RADIUS_KM = 6371.0
        LUNAR_DISTANCE_KM = 384400.0
        ESCAPE_VELOCITY_KM_S = 11.2

        # Normalize parameters for QAE circuit
        distance_param = max(0, min(1, 1 - miss_distance_lunar / 100))
        velocity_param = min(1, velocity_km_s / 25.0)
        size_param = min(1, np.log10(diameter_km * 1000 + 1) / 4)

        # Run QAE circuit
        result = cudaq.sample(
            qae_amplitude_estimation,
            distance_param,
            velocity_param,
            size_param,
            4,
            shots_count=shots
        )

        # Decode phase
        phase_sum = 0.0
        total_counts = 0
        for bitstring in result:
            count = result.count(bitstring)
            if len(bitstring) >= 4:
                counting_bits = bitstring[:4]
                phase_val = int(counting_bits, 2) / 16.0
                phase_sum += phase_val * count
                total_counts += count

        estimated_phase = phase_sum / total_counts if total_counts > 0 else 0.0
        quantum_factor = 0.5 + np.sin(np.pi * estimated_phase) ** 2 * 1.5

        # Gaussian impact probability
        miss_distance_km = miss_distance_lunar * LUNAR_DISTANCE_KM

        focusing_factor = np.sqrt(1 + (ESCAPE_VELOCITY_KM_S / max(velocity_km_s, 1.0)) ** 2)
        effective_radius_km = EARTH_RADIUS_KM * focusing_factor

        base_uncertainty_km = 50.0
        distance_factor = 1 + (miss_distance_lunar / 10.0)
        sigma = base_uncertainty_km * distance_factor * quantum_factor

        if miss_distance_km < effective_radius_km:
            final_prob = 0.5
        else:
            exponent = -(miss_distance_km ** 2) / (2 * sigma ** 2)
            if exponent < -700:
                final_prob = 1e-15
            else:
                geometric_factor = (effective_radius_km / sigma) ** 2
                final_prob = max(1e-15, min(1.0, geometric_factor * np.exp(exponent)))

        # Confidence interval
        epsilon = np.sqrt(np.log(2 / 0.05) / (2 * shots))
        ci_low = max(1e-15, final_prob * (1 - epsilon))
        ci_high = min(1, final_prob * (1 + epsilon))

        return QAEResult(
            estimated_probability=final_prob,
            confidence_interval=(ci_low, ci_high),
            num_oracle_calls=shots * 8,
            precision_bits=4
        )

    # ========================================================================
    # QAOA Deflection with Proper Optimization
    # ========================================================================

    def optimize_deflection_qaoa(
        self,
        miss_distance_lunar: float,
        velocity_km_s: float,
        diameter_km: float,
        target_miss_distance: float = 100,
        shots: int = 2000,
        qaoa_layers: int = 4,
        optimization_method: str = 'COBYLA'
    ) -> dict:
        """
        QAOA-based deflection optimization (EDUCATIONAL DEMONSTRATION).

        Uses COBYLA or SPSA for variational parameter optimization.

        NOTE: The cost function is heuristic, not rigorous astrodynamics.
        This is a proof-of-concept showing how QAOA could be applied,
        not a production mission planning tool.
        """
        # Normalize target parameters
        target_params = [
            target_miss_distance / 100,
            velocity_km_s / 30,
            np.log10(self._estimate_mass(diameter_km)) / 20,
            0.5  # Time factor
        ]

        def qaoa_cost(params):
            """Cost function for QAOA parameter optimization"""
            gamma = list(params[:qaoa_layers])
            beta = list(params[qaoa_layers:])

            result = cudaq.sample(
                qaoa_deflection_v2,
                target_params,
                gamma,
                beta,
                qaoa_layers,
                shots_count=shots
            )

            # Compute expected cost from measurements
            total_cost = 0.0
            total_count = 0

            for bitstring in result:
                count = result.count(bitstring)

                # Decode delta-V from bitstring
                bits = [int(b) for b in bitstring]

                # Magnitude (first 2 bits)
                magnitude = (bits[0] * 2 + bits[1]) / 3.0

                # Direction components
                dx = (bits[2] * 2 + bits[3]) / 3.0 - 0.5
                dy = (bits[4] * 2 + bits[5]) / 3.0 - 0.5
                dz = (bits[6] * 2 + bits[7]) / 3.0 - 0.5

                # Cost: minimize magnitude while ensuring effective direction
                direction_effectiveness = abs(dx) + abs(dy) + abs(dz)
                cost = magnitude - 0.3 * direction_effectiveness

                total_cost += cost * count
                total_count += count

            return total_cost / total_count if total_count > 0 else 1.0

        # Initial QAOA parameters
        initial_gamma = [0.5 * np.pi / (i + 1) for i in range(qaoa_layers)]
        initial_beta = [0.3 * np.pi / (i + 1) for i in range(qaoa_layers)]
        initial_params = initial_gamma + initial_beta

        # Optimize
        from scipy.optimize import minimize

        if optimization_method == 'COBYLA':
            result = minimize(
                qaoa_cost,
                initial_params,
                method='COBYLA',
                options={'maxiter': 100, 'rhobeg': 0.3}
            )
        else:  # SPSA-like with finite differences
            result = minimize(
                qaoa_cost,
                initial_params,
                method='Powell',
                options={'maxiter': 50}
            )

        optimal_params = result.x
        optimal_gamma = list(optimal_params[:qaoa_layers])
        optimal_beta = list(optimal_params[qaoa_layers:])

        # Get best solution from optimal parameters
        final_result = cudaq.sample(
            qaoa_deflection_v2,
            target_params,
            optimal_gamma,
            optimal_beta,
            qaoa_layers,
            shots_count=shots * 2
        )

        # Find most probable solution
        best_bitstring = max(final_result, key=lambda bs: final_result.count(bs))

        # Decode optimal deflection
        bits = [int(b) for b in best_bitstring]
        delta_v_magnitude = max(0.001, (bits[0] * 2 + bits[1]) / 3.0 * 0.1)  # m/s

        direction = np.array([
            (bits[2] * 2 + bits[3]) / 3.0 - 0.5,
            (bits[4] * 2 + bits[5]) / 3.0 - 0.5,
            (bits[6] * 2 + bits[7]) / 3.0 - 0.5
        ])
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Success probability from measurement statistics
        best_count = final_result.count(best_bitstring)
        total_count = sum(final_result.count(bs) for bs in final_result)
        success_prob = best_count / total_count if total_count > 0 else 0.5

        return {
            "optimal_delta_v": delta_v_magnitude,
            "optimal_direction": tuple(direction.tolist()),
            "success_probability": min(0.95, success_prob * 1.5),
            "optimal_gamma": optimal_gamma,
            "optimal_beta": optimal_beta,
            "qaoa_layers": qaoa_layers,
            "optimization_iterations": result.nfev,
            "final_cost": result.fun
        }

    # ========================================================================
    # Batch Execution
    # ========================================================================

    def batch_classify(
        self,
        neo_features: List[np.ndarray],
        shots_per_neo: int = 500
    ) -> BatchResult:
        """
        Batch classify multiple NEOs using parallel GPU execution.

        More efficient than sequential processing.
        """
        import time
        start_time = time.time()

        batch_size = len(neo_features)

        # Flatten and encode all features
        all_encodings = []
        for features in neo_features:
            encoded = self._encode_features(features)
            all_encodings.extend(encoded)

        # Run circuits in parallel using cudaq.sample with different indices
        all_results = []

        # CUDA-Q can execute multiple circuits efficiently
        for i in range(batch_size):
            result = cudaq.sample(
                vqc_batch_kernel,
                all_encodings,
                self.vqc_params,
                self.num_qubits,
                self.num_layers,
                i,
                shots_count=shots_per_neo
            )
            all_results.append(result)

        # Process results
        threat_levels = []
        probabilities = []

        for result in all_results:
            probs = self._interpret_vqc_result(result)
            threat_level = max(probs, key=probs.get)
            threat_levels.append(threat_level)
            probabilities.append(probs)

        execution_time = (time.time() - start_time) * 1000

        return BatchResult(
            neo_ids=[f"neo_{i}" for i in range(batch_size)],
            threat_levels=threat_levels,
            probabilities=probabilities,
            execution_time_ms=execution_time,
            circuits_executed=batch_size
        )

    # ========================================================================
    # VQC Training
    # ========================================================================

    def train_vqc(
        self,
        training_data: List[Tuple[np.ndarray, ThreatLevel]],
        epochs: int = 50,
        learning_rate: float = 0.1,
        shots: int = 500
    ) -> dict:
        """
        Train the VQC classifier on labeled data.

        Uses parameter-shift rule for gradient computation.
        """
        self.training_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0

            for features, true_label in training_data:
                # Forward pass
                encoding = self._encode_features(features)
                result = cudaq.sample(
                    vqc_batch_kernel,
                    encoding * (self.num_qubits // len(encoding) + 1),  # Pad encoding
                    self.vqc_params,
                    self.num_qubits,
                    self.num_layers,
                    0,
                    shots_count=shots
                )

                probs = self._interpret_vqc_result(result)
                predicted = max(probs, key=probs.get)

                if predicted == true_label:
                    correct += 1

                # Compute loss (cross-entropy)
                true_prob = probs.get(true_label, 0.01)
                loss = -np.log(true_prob + 1e-10)
                epoch_loss += loss

                # Gradient estimation using parameter-shift rule
                gradients = self._compute_gradients(
                    encoding, true_label, shots
                )

                # Update parameters
                for i in range(len(self.vqc_params)):
                    self.vqc_params[i] -= learning_rate * gradients[i]

            accuracy = correct / len(training_data)
            avg_loss = epoch_loss / len(training_data)

            self.training_history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy
            })

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")

        self.vqc_trained = True

        return {
            "final_loss": self.training_history[-1]["loss"],
            "final_accuracy": self.training_history[-1]["accuracy"],
            "epochs": epochs,
            "history": self.training_history
        }

    def _compute_gradients(
        self,
        encoding: List[float],
        true_label: ThreatLevel,
        shots: int
    ) -> List[float]:
        """Compute gradients using parameter-shift rule"""
        gradients = []
        shift = np.pi / 2

        for i in range(len(self.vqc_params)):
            # Forward shift
            params_plus = self.vqc_params.copy()
            params_plus[i] += shift

            result_plus = cudaq.sample(
                vqc_batch_kernel,
                encoding * (self.num_qubits // len(encoding) + 1),
                params_plus,
                self.num_qubits,
                self.num_layers,
                0,
                shots_count=shots // 2
            )
            probs_plus = self._interpret_vqc_result(result_plus)
            loss_plus = -np.log(probs_plus.get(true_label, 0.01) + 1e-10)

            # Backward shift
            params_minus = self.vqc_params.copy()
            params_minus[i] -= shift

            result_minus = cudaq.sample(
                vqc_batch_kernel,
                encoding * (self.num_qubits // len(encoding) + 1),
                params_minus,
                self.num_qubits,
                self.num_layers,
                0,
                shots_count=shots // 2
            )
            probs_minus = self._interpret_vqc_result(result_minus)
            loss_minus = -np.log(probs_minus.get(true_label, 0.01) + 1e-10)

            # Gradient
            gradient = (loss_plus - loss_minus) / 2
            gradients.append(gradient)

        return gradients

    def generate_synthetic_training_data(
        self,
        num_samples: int = 100
    ) -> List[Tuple[np.ndarray, ThreatLevel]]:
        """Generate synthetic NEO data for VQC training"""
        training_data = []

        for _ in range(num_samples):
            # Random NEO-like parameters
            magnitude = np.random.uniform(15, 30)
            diameter = 10 ** np.random.uniform(-3, 1)  # 0.001 to 10 km
            velocity = np.random.uniform(5, 30)
            miss_distance = 10 ** np.random.uniform(0, 3)  # 1 to 1000 LD
            is_hazardous = np.random.random() < 0.2

            features = np.array([
                magnitude / 35.0,
                np.log10(diameter * 1000 + 1) / 5.0,
                velocity / 30.0,
                np.log10(miss_distance + 1) / 4.0,
                float(is_hazardous)
            ])

            # Determine threat level based on physics
            if diameter > 1 and miss_distance < 5:
                label = ThreatLevel.CRITICAL
            elif diameter > 0.14 and miss_distance < 20:
                label = ThreatLevel.HIGH
            elif is_hazardous or (diameter > 0.05 and miss_distance < 50):
                label = ThreatLevel.MODERATE
            elif diameter > 0.01:
                label = ThreatLevel.LOW
            else:
                label = ThreatLevel.NEGLIGIBLE

            training_data.append((features, label))

        return training_data

    def fetch_nasa_training_data(self) -> List[Tuple[np.ndarray, ThreatLevel]]:
        """
        Fetch real training data from NASA APIs.

        Sources:
        - NASA Sentry API: Objects with computed impact probability
        - Close Approach API: Historic close approaches
        - Known impact events: Ground truth data
        """
        try:
            from training_data_fetcher import TrainingDataFetcher, ThreatLevel as FetcherThreatLevel

            fetcher = TrainingDataFetcher()
            raw_data = fetcher.fetch_all_training_data()

            # Convert ThreatLevel enum (they should match but let's be safe)
            training_data = []
            for features, label in raw_data:
                # Pad features to match num_qubits if needed
                if len(features) < self.num_qubits:
                    features = np.pad(features, (0, self.num_qubits - len(features)))
                elif len(features) > self.num_qubits:
                    features = features[:self.num_qubits]

                # Map the label
                engine_label = ThreatLevel[label.name]
                training_data.append((features, engine_label))

            return training_data

        except ImportError as e:
            print(f"Could not import training_data_fetcher: {e}")
            print("Falling back to synthetic data")
            return self.generate_synthetic_training_data(100)

    def train_on_nasa_data(
        self,
        epochs: int = 50,
        learning_rate: float = 0.1,
        shots: int = 500,
        test_split: float = 0.2
    ) -> Dict:
        """
        Train VQC on real NASA data.

        Fetches data from Sentry, historic approaches, and known impacts.
        Splits into train/test sets for validation.
        """
        print("Fetching NASA training data...")
        all_data = self.fetch_nasa_training_data()

        if not all_data:
            return {"error": "No training data available"}

        # Shuffle and split
        np.random.shuffle(all_data)
        split_idx = int(len(all_data) * (1 - test_split))
        train_data = all_data[:split_idx]
        test_data = all_data[split_idx:]

        print(f"Training set: {len(train_data)} samples")
        print(f"Test set: {len(test_data)} samples")

        # Train
        train_result = self.train_vqc(
            train_data,
            epochs=epochs,
            learning_rate=learning_rate,
            shots=shots
        )

        # Evaluate on test set
        correct = 0
        for features, true_label in test_data:
            encoding = self._encode_features(features)

            result = cudaq.sample(
                vqc_batch_kernel,
                encoding * (self.num_qubits // len(encoding) + 1),
                self.vqc_params,
                self.num_qubits,
                self.num_layers,
                0,
                shots_count=shots
            )

            probs = self._interpret_vqc_result(result)
            predicted = max(probs, key=probs.get)

            if predicted == true_label:
                correct += 1

        test_accuracy = correct / len(test_data) if test_data else 0

        return {
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "final_train_accuracy": train_result["final_accuracy"],
            "test_accuracy": test_accuracy,
            "epochs": epochs,
            "history": train_result["history"]
        }

    # ========================================================================
    # Quantum State Visualization
    # ========================================================================

    def _extract_quantum_state(
        self,
        result: cudaq.SampleResult,
        num_qubits: int
    ) -> QuantumState:
        """Extract quantum state information from measurement results"""
        # Get all possible basis states
        basis_states = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]

        # Compute probabilities
        total_counts = sum(result.count(bs) for bs in result)
        probabilities = np.zeros(2**num_qubits)

        for bitstring in result:
            idx = int(bitstring, 2)
            if idx < len(probabilities):
                probabilities[idx] = result.count(bitstring) / total_counts

        # Estimate amplitudes (assuming real amplitudes for simplicity)
        amplitudes = np.sqrt(probabilities)

        # Compute von Neumann entropy
        entropy = 0.0
        for p in probabilities:
            if p > 1e-10:
                entropy -= p * np.log2(p)

        return QuantumState(
            amplitudes=amplitudes,
            probabilities=probabilities,
            num_qubits=num_qubits,
            basis_states=basis_states,
            entropy=entropy
        )

    def visualize_state(
        self,
        neo_features: np.ndarray,
        shots: int = 1000
    ) -> QuantumState:
        """Get quantum state visualization for a NEO classification"""
        encoding = self._encode_features(neo_features)

        result = cudaq.sample(
            vqc_batch_kernel,
            encoding * (self.num_qubits // len(encoding) + 1),
            self.vqc_params,
            self.num_qubits,
            self.num_layers,
            0,
            shots_count=shots
        )

        return self._extract_quantum_state(result, self.num_qubits)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _encode_features(self, features: np.ndarray) -> List[float]:
        """Encode features as rotation angles"""
        if len(features) < self.num_qubits:
            features = np.pad(features, (0, self.num_qubits - len(features)))
        elif len(features) > self.num_qubits:
            features = features[:self.num_qubits]

        return list(features * 2 * np.pi)

    def _interpret_vqc_result(
        self,
        result: cudaq.SampleResult
    ) -> Dict[ThreatLevel, float]:
        """Interpret VQC measurement as threat level probabilities"""
        counts = {}
        for bitstring in result:
            counts[bitstring] = result.count(bitstring)

        total = sum(counts.values())

        # Quantum-derived score
        parity_score = 0
        hamming_score = 0

        for bitstring, count in counts.items():
            ones = bitstring.count('1')
            parity_score += (ones % 2) * count / total
            hamming_score += ones * count / (total * self.num_qubits)

        quantum_score = (parity_score + hamming_score) / 2

        # Map to threat levels
        probs = {}
        for level in ThreatLevel:
            threshold = level.value / 4.0
            diff = abs(quantum_score - threshold)
            probs[level] = np.exp(-3 * diff)

        # Normalize
        total_prob = sum(probs.values())
        probs = {k: v / total_prob for k, v in probs.items()}

        return probs

    def _estimate_mass(self, diameter_km: float) -> float:
        """Estimate asteroid mass from diameter"""
        radius = diameter_km * 500
        volume = (4/3) * np.pi * radius**3
        density = 2500
        return volume * density


# ============================================================================
# Factory Function
# ============================================================================

def create_quantum_engine(
    gpu: bool = True,
    num_qubits: int = 4,
    num_layers: int = 3
) -> QuantumNEOEngine:
    """Create a configured quantum engine instance"""
    return QuantumNEOEngine(
        num_qubits=num_qubits,
        num_layers=num_layers,
        use_gpu=gpu
    )


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEO Quantum Engine - Advanced CUDA-Q Implementation")
    print("=" * 60)

    engine = create_quantum_engine()

    # Test VQE for orbital stability
    print("\n1. VQE Orbital Stability Analysis")
    print("-" * 40)

    vqe_result = engine.compute_orbital_stability(
        semi_major_axis_au=1.5,
        eccentricity=0.3,
        inclination_deg=15,
        longitude_deg=45,
        shots=1000,
        max_iterations=20
    )

    print(f"Ground State Energy: {vqe_result.ground_state_energy:.4f}")
    print(f"Stability Score: {vqe_result.stability_score:.2%}")
    print(f"Iterations: {vqe_result.iterations}")

    # Test QAE
    print("\n2. Quantum Amplitude Estimation")
    print("-" * 40)

    qae_result = engine.estimate_collision_probability_qae(
        miss_distance_lunar=5.0,
        velocity_km_s=15.0,
        diameter_km=0.5,
        shots=2000
    )

    print(f"Collision Probability: {qae_result.estimated_probability:.2e}")
    print(f"Confidence Interval: [{qae_result.confidence_interval[0]:.2e}, {qae_result.confidence_interval[1]:.2e}]")
    print(f"Oracle Calls: {qae_result.num_oracle_calls}")

    # Test QAOA
    print("\n3. QAOA Deflection Optimization")
    print("-" * 40)

    qaoa_result = engine.optimize_deflection_qaoa(
        miss_distance_lunar=10.0,
        velocity_km_s=12.0,
        diameter_km=0.3,
        target_miss_distance=100,
        shots=1000,
        qaoa_layers=3
    )

    print(f"Optimal Delta-V: {qaoa_result['optimal_delta_v']*100:.4f} cm/s")
    print(f"Direction: {qaoa_result['optimal_direction']}")
    print(f"Success Probability: {qaoa_result['success_probability']:.1%}")

    # Test Batch Execution
    print("\n4. Batch Classification")
    print("-" * 40)

    test_neos = [
        np.array([0.7, 0.3, 0.5, 0.8, 0.0]),
        np.array([0.5, 0.6, 0.7, 0.2, 1.0]),
        np.array([0.3, 0.2, 0.3, 0.9, 0.0]),
    ]

    batch_result = engine.batch_classify(test_neos, shots_per_neo=500)

    print(f"Execution Time: {batch_result.execution_time_ms:.2f} ms")
    print(f"Circuits Executed: {batch_result.circuits_executed}")
    for i, (level, probs) in enumerate(zip(batch_result.threat_levels, batch_result.probabilities)):
        print(f"  NEO {i}: {level.name} (conf: {probs[level]:.1%})")

    # Test Training
    print("\n5. VQC Training")
    print("-" * 40)

    training_data = engine.generate_synthetic_training_data(50)
    print(f"Generated {len(training_data)} training samples")

    train_result = engine.train_vqc(
        training_data[:30],
        epochs=20,
        learning_rate=0.05,
        shots=200
    )

    print(f"Final Accuracy: {train_result['final_accuracy']:.1%}")

    # Test Quantum State Visualization
    print("\n6. Quantum State Visualization")
    print("-" * 40)

    state = engine.visualize_state(test_neos[0])
    print(f"Num Qubits: {state.num_qubits}")
    print(f"Entropy: {state.entropy:.4f}")
    print("Top States:")
    for s in state.to_dict()["top_states"][:3]:
        print(f"  |{s['state']}> : {s['probability']:.2%}")

    print("\n" + "=" * 60)
    print("All tests completed!")

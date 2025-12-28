"""
Quantum Orbital Dynamics Simulator

Uses CUDA-Q to simulate gravitational perturbations and orbital evolution
of Near Earth Objects through quantum Hamiltonian simulation.
"""
import cudaq
import numpy as np
from dataclasses import dataclass
from typing import Optional
from neo_fetcher import NEO


# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_SUN = 1.989e30  # Solar mass (kg)
M_EARTH = 5.972e24  # Earth mass (kg)
AU = 1.496e11  # Astronomical unit (m)


@dataclass
class OrbitalState:
    """Quantum-derived orbital state"""
    neo: NEO
    position_uncertainty: float  # km
    velocity_uncertainty: float  # km/s
    collision_probability: float
    evolution_states: list[np.ndarray]
    time_steps: list[float]


@cudaq.kernel
def gravitational_hamiltonian_evolution(
    initial_state: list[complex],
    perturbation_strength: float,
    num_steps: int
):
    """
    Quantum simulation of gravitational perturbation.

    Uses a simplified Hamiltonian to model how gravitational interactions
    perturb the orbital state. This is a proof-of-concept showing how
    quantum computers could simulate orbital mechanics.

    The Hamiltonian represents:
    H = H_0 + epsilon * V
    where H_0 is the unperturbed Keplerian orbit and V is the perturbation
    from nearby massive bodies.
    """
    # 2 qubits encode radial and angular momentum states
    qubits = cudaq.qvector(initial_state)

    # Apply time evolution through Trotterization
    dt = 0.1  # Time step in orbital periods
    for step in range(num_steps):
        # Unperturbed orbital dynamics (rotation in phase space)
        for i in range(qubits.size()):
            rz(dt * 2 * np.pi, qubits[i])

        # Gravitational perturbation (coupling between modes)
        for i in range(qubits.size() - 1):
            # XX interaction represents radial-angular coupling
            h(qubits[i])
            h(qubits[i + 1])
            x.ctrl(qubits[i], qubits[i + 1])
            rz(perturbation_strength * dt, qubits[i + 1])
            x.ctrl(qubits[i], qubits[i + 1])
            h(qubits[i])
            h(qubits[i + 1])

            # ZZ interaction represents energy exchange
            x.ctrl(qubits[i], qubits[i + 1])
            rz(perturbation_strength * dt * 0.5, qubits[i + 1])
            x.ctrl(qubits[i], qubits[i + 1])

    mz(qubits)


@cudaq.kernel
def collision_probability_circuit(
    earth_distance_param: float,
    velocity_param: float,
    size_param: float,
    num_qubits: int
):
    """
    Quantum circuit to estimate collision probability.

    Encodes orbital parameters and uses quantum interference
    to compute probability amplitude for collision.
    """
    qubits = cudaq.qvector(num_qubits)

    # Encode distance (closer = higher amplitude for |1>)
    # earth_distance_param should be normalized 0-1 where 0 = far, 1 = close
    ry(earth_distance_param * np.pi, qubits[0])

    # Encode velocity alignment (aligned = higher collision risk)
    ry(velocity_param * np.pi, qubits[1])

    # Encode size (larger = higher cross-section)
    ry(size_param * np.pi / 2, qubits[2])

    # Entangle parameters to compute joint probability
    x.ctrl(qubits[0], qubits[1])
    x.ctrl(qubits[1], qubits[2])

    # Apply phase for interference
    t(qubits[0])
    s(qubits[1])
    t(qubits[2])

    # Reverse entanglement for interference pattern
    x.ctrl(qubits[1], qubits[2])
    x.ctrl(qubits[0], qubits[1])

    # Hadamard for superposition readout
    h(qubits[0])

    mz(qubits)


class OrbitalSimulator:
    """
    Quantum orbital dynamics simulator for NEOs.

    Uses quantum Hamiltonian simulation to model orbital evolution
    and estimate collision probabilities.
    """

    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits

    def estimate_collision_probability(self, neo: NEO, shots: int = 2000) -> float:
        """
        Estimate collision probability using Gaussian miss distance model.

        For a NEO with nominal miss distance d and position uncertainty sigma,
        the probability of impact is approximately:

        P = (r_eff / sigma)^2 * exp(-d^2 / (2 * sigma^2))

        Where r_eff is Earth's effective capture radius with gravitational focusing.
        """
        # Physical constants
        EARTH_RADIUS_KM = 6371.0
        LUNAR_DISTANCE_KM = 384400.0
        ESCAPE_VELOCITY_KM_S = 11.2

        # Convert miss distance to km
        miss_distance_km = neo.miss_distance_lunar * LUNAR_DISTANCE_KM

        # Gravitational focusing - increases effective Earth cross-section
        focusing_factor = np.sqrt(1 + (ESCAPE_VELOCITY_KM_S / max(neo.velocity_km_s, 1.0)) ** 2)
        effective_radius_km = EARTH_RADIUS_KM * focusing_factor

        # Position uncertainty - scales with distance and time
        # Well-tracked NEOs: ~1-10 km uncertainty
        # Newly discovered: 100-1000 km uncertainty
        # Far objects: uncertainty grows with distance
        base_uncertainty_km = 50.0  # Moderate tracking uncertainty
        distance_factor = 1 + (neo.miss_distance_lunar / 10.0)  # Grows with distance
        position_uncertainty_km = base_uncertainty_km * distance_factor

        # Use quantum circuit to modulate uncertainty
        distance_param = max(0, min(1, 1 - neo.miss_distance_lunar / 100))
        velocity_param = min(1, neo.velocity_km_s / 25.0)
        size_param = min(1, np.log10(neo.avg_diameter_km * 1000 + 1) / 4)

        result = cudaq.sample(
            collision_probability_circuit,
            distance_param,
            velocity_param,
            size_param,
            3,
            shots_count=shots
        )

        # Quantum factor (0.5 to 2.0)
        collision_count = 0
        for bitstring in result:
            if bitstring.startswith('11'):
                collision_count += result.count(bitstring)
        quantum_factor = 0.5 + (collision_count / shots)

        sigma = position_uncertainty_km * quantum_factor

        # Gaussian impact probability
        # P = (r_eff/sigma)^2 * exp(-d^2/(2*sigma^2))
        if miss_distance_km < effective_radius_km:
            # Direct hit trajectory
            return 0.5

        # Calculate exponent carefully to avoid overflow
        exponent = -(miss_distance_km ** 2) / (2 * sigma ** 2)

        # For very negative exponents, probability is essentially zero
        if exponent < -700:
            return 1e-15

        geometric_factor = (effective_radius_km / sigma) ** 2
        probability = geometric_factor * np.exp(exponent)

        # Clamp to realistic range
        return max(1e-15, min(1.0, probability))

    def simulate_orbital_evolution(
        self,
        neo: NEO,
        num_steps: int = 10,
        shots: int = 1000
    ) -> OrbitalState:
        """
        Simulate orbital state evolution using quantum dynamics.

        Models how gravitational perturbations from Earth affect
        the asteroid's orbit over time.
        """
        # Initial quantum state (equal superposition of orbital modes)
        num_state_qubits = 2
        dim = 2 ** num_state_qubits
        initial_state = [complex(1/np.sqrt(dim))] * dim

        # Perturbation strength based on Earth proximity
        # Closer approaches = stronger perturbation
        perturbation = 0.1 * (1 - neo.miss_distance_lunar / 100)
        perturbation = max(0.01, min(0.5, perturbation))

        # Sample multiple trajectories
        evolution_results = []
        for step in range(num_steps):
            result = cudaq.sample(
                gravitational_hamiltonian_evolution,
                initial_state,
                perturbation,
                step + 1,
                shots_count=shots
            )
            evolution_results.append(result)

        # Compute uncertainties from measurement distributions
        position_uncertainties = []
        velocity_uncertainties = []

        for result in evolution_results:
            # Shannon entropy as measure of uncertainty
            probs = []
            for bitstring in result:
                probs.append(result.count(bitstring) / shots)

            entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)

            # Map entropy to physical uncertainty (heuristic scaling)
            pos_unc = entropy * neo.miss_distance_km / 1000  # Scale by distance
            vel_unc = entropy * neo.velocity_km_s / 10

            position_uncertainties.append(pos_unc)
            velocity_uncertainties.append(vel_unc)

        # Time steps (in days, assuming ~1 year orbital period)
        time_steps = [i * 36.5 for i in range(1, num_steps + 1)]

        # Collision probability from dedicated circuit
        collision_prob = self.estimate_collision_probability(neo)

        return OrbitalState(
            neo=neo,
            position_uncertainty=position_uncertainties[-1],
            velocity_uncertainty=velocity_uncertainties[-1],
            collision_probability=collision_prob,
            evolution_states=position_uncertainties,
            time_steps=time_steps
        )

    def batch_simulate(self, neos: list[NEO], num_steps: int = 5) -> list[OrbitalState]:
        """Simulate multiple NEOs"""
        return [self.simulate_orbital_evolution(neo, num_steps) for neo in neos]


def visualize_evolution(state: OrbitalState):
    """Print text-based visualization of orbital evolution"""
    print(f"\nOrbital Evolution: {state.neo.name}")
    print("=" * 50)
    print(f"Collision Probability: {state.collision_probability:.2e}")
    print(f"Final Position Uncertainty: {state.position_uncertainty:.1f} km")
    print(f"Final Velocity Uncertainty: {state.velocity_uncertainty:.3f} km/s")

    print("\nPosition Uncertainty Evolution:")
    max_unc = max(state.evolution_states)
    for i, (t, unc) in enumerate(zip(state.time_steps, state.evolution_states)):
        bar_len = int(40 * unc / (max_unc + 0.001))
        bar = "#" * bar_len
        print(f"  Day {t:5.0f}: {bar} {unc:.1f} km")


if __name__ == "__main__":
    import os
    from neo_fetcher import NEOFetcher

    print("Quantum Orbital Dynamics Simulator")
    print("=" * 60)

    # Fetch NEOs - use env var or DEMO_KEY
    fetcher = NEOFetcher(os.getenv("NASA_API_KEY", "DEMO_KEY"))
    neos = fetcher.fetch_upcoming(3)

    # Filter to closest approaches
    close_neos = sorted(neos, key=lambda x: x.miss_distance_km)[:5]

    print(f"\nSimulating {len(close_neos)} closest approaching NEOs...")

    simulator = OrbitalSimulator()

    for neo in close_neos:
        state = simulator.simulate_orbital_evolution(neo, num_steps=8)
        visualize_evolution(state)

    # Show collision probabilities for all
    print("\n" + "=" * 60)
    print("Collision Probability Summary")
    print("=" * 60)

    collision_probs = []
    for neo in neos[:15]:
        prob = simulator.estimate_collision_probability(neo)
        collision_probs.append((neo, prob))

    collision_probs.sort(key=lambda x: x[1], reverse=True)

    for neo, prob in collision_probs:
        indicator = "**" if prob > 1e-6 else "  "
        print(f"{indicator} {neo.name[:30]:<32} P(collision) = {prob:.2e}")

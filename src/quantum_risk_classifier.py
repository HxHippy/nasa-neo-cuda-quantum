"""
Quantum Risk Classifier for Near Earth Objects

Uses a Variational Quantum Classifier (VQC) with CUDA-Q to classify
asteroid threat levels based on physical properties.
"""
import cudaq
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from neo_fetcher import NEO


class ThreatLevel(Enum):
    NEGLIGIBLE = 0  # No practical threat
    LOW = 1         # Very small, will burn up in atmosphere
    MODERATE = 2    # Could cause local damage
    HIGH = 3        # Regional damage potential
    CRITICAL = 4    # Extinction-level threat


@dataclass
class RiskAssessment:
    """Result of quantum risk assessment"""
    neo: NEO
    threat_level: ThreatLevel
    probability_distribution: dict[ThreatLevel, float]
    quantum_state: Optional[np.ndarray] = None

    def __str__(self):
        return (
            f"Risk Assessment: {self.neo.name}\n"
            f"  Threat Level: {self.threat_level.name}\n"
            f"  Probabilities:\n" +
            "\n".join(f"    {k.name}: {v:.2%}" for k, v in self.probability_distribution.items())
        )


# Standalone quantum kernel for CUDA-Q
@cudaq.kernel
def vqc_circuit(encoding: list[float], params: list[float], num_qubits: int, num_layers: int):
    """
    Variational quantum circuit for classification.

    Architecture:
    1. Feature encoding via Ry rotations
    2. Variational layers with Rx, Ry, Rz rotations
    3. CNOT entangling gates between adjacent qubits
    """
    qubits = cudaq.qvector(num_qubits)

    # Feature encoding layer
    for i in range(num_qubits):
        ry(encoding[i], qubits[i])

    # Variational layers
    param_idx = 0
    for layer in range(num_layers):
        # Single qubit rotations
        for i in range(num_qubits):
            rx(params[param_idx], qubits[i])
            param_idx += 1
            ry(params[param_idx], qubits[i])
            param_idx += 1
            rz(params[param_idx], qubits[i])
            param_idx += 1

        # Entangling layer (ring topology)
        for i in range(num_qubits - 1):
            x.ctrl(qubits[i], qubits[i + 1])
        if num_qubits > 2:
            x.ctrl(qubits[num_qubits - 1], qubits[0])

    # Measure all qubits for classification
    mz(qubits)


class QuantumRiskClassifier:
    """
    Variational Quantum Classifier for NEO risk assessment.

    Uses amplitude encoding to embed NEO features into quantum state,
    then applies parameterized quantum circuit for classification.
    """

    def __init__(self, num_qubits: int = 4, num_layers: int = 3):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # Pre-trained parameters (in a real system, these would be trained on historical data)
        self.parameters = self._initialize_parameters()

    def _initialize_parameters(self) -> list[float]:
        """Initialize variational parameters"""
        # Number of parameters: 3 per qubit per layer (rx, ry, rz)
        num_params = self.num_layers * self.num_qubits * 3
        np.random.seed(42)  # For reproducibility
        return list(np.random.uniform(-np.pi, np.pi, num_params))

    def _encode_features(self, features: np.ndarray) -> list[float]:
        """
        Encode NEO features as rotation angles for amplitude encoding.
        Maps features to angles in [0, 2*pi]
        """
        # Pad or truncate features to match qubit count
        if len(features) < self.num_qubits:
            features = np.pad(features, (0, self.num_qubits - len(features)))
        elif len(features) > self.num_qubits:
            features = features[:self.num_qubits]

        # Map to rotation angles
        return list(features * 2 * np.pi)

    def classify(self, neo: NEO, shots: int = 1000) -> RiskAssessment:
        """
        Classify NEO threat level using quantum circuit.

        Args:
            neo: Near Earth Object to classify
            shots: Number of quantum circuit executions

        Returns:
            RiskAssessment with threat level and probabilities
        """
        # Extract and encode features
        features = neo.to_feature_vector()
        encoding = self._encode_features(features)

        # Run quantum circuit
        result = cudaq.sample(
            vqc_circuit,
            encoding,
            self.parameters,
            self.num_qubits,
            self.num_layers,
            shots_count=shots
        )

        # Interpret measurement results as threat level
        threat_probs = self._interpret_results(result, neo)
        threat_level = max(threat_probs, key=threat_probs.get)

        return RiskAssessment(
            neo=neo,
            threat_level=threat_level,
            probability_distribution=threat_probs,
        )

    def _interpret_results(self, result: cudaq.SampleResult, neo: NEO) -> dict[ThreatLevel, float]:
        """
        Interpret quantum measurement results as threat level probabilities.

        Uses bitstring parity and count to map to threat levels,
        combined with classical risk factors.
        """
        # Count measurement outcomes
        counts = {}
        for bitstring in result:
            counts[bitstring] = result.count(bitstring)

        total = sum(counts.values())

        # Calculate quantum-derived score based on measurement statistics
        parity_score = 0
        hamming_score = 0

        for bitstring, count in counts.items():
            # Parity contribution
            ones = bitstring.count('1')
            parity_score += (ones % 2) * count / total
            # Hamming weight contribution
            hamming_score += ones * count / (total * self.num_qubits)

        # Combine quantum score with classical risk factors
        quantum_score = (parity_score + hamming_score) / 2

        # Classical risk factors (0-1 scale)
        size_risk = min(neo.avg_diameter_km / 1.0, 1.0)  # 1km+ is max
        velocity_risk = min(neo.velocity_km_s / 20.0, 1.0)  # 20 km/s is max
        distance_risk = max(0, 1.0 - neo.miss_distance_lunar / 100.0)  # Closer = higher risk
        energy_risk = min(np.log10(neo.kinetic_energy_estimate + 1) / 6.0, 1.0)

        # Weighted combination
        combined_score = (
            0.3 * quantum_score +
            0.25 * size_risk +
            0.15 * velocity_risk +
            0.15 * distance_risk +
            0.15 * energy_risk
        )

        # Map to threat level distribution using softmax-like approach
        levels = list(ThreatLevel)
        probs = {}

        for level in levels:
            # Center score around each level's threshold
            threshold = level.value / 4.0
            diff = abs(combined_score - threshold)
            probs[level] = np.exp(-3 * diff)

        # Normalize probabilities
        total_prob = sum(probs.values())
        probs = {k: v / total_prob for k, v in probs.items()}

        # If actually hazardous per NASA, boost HIGH/CRITICAL
        if neo.is_hazardous:
            probs[ThreatLevel.HIGH] *= 1.5
            probs[ThreatLevel.CRITICAL] *= 1.3
            # Renormalize
            total_prob = sum(probs.values())
            probs = {k: v / total_prob for k, v in probs.items()}

        return probs

    def batch_classify(self, neos: list[NEO], shots: int = 500) -> list[RiskAssessment]:
        """Classify multiple NEOs"""
        return [self.classify(neo, shots) for neo in neos]


def visualize_circuit():
    """Visualize the quantum circuit structure"""
    # Create dummy encoding and parameters for visualization
    encoding = [0.5, 0.5, 0.5, 0.5]
    num_qubits = 4
    num_layers = 2
    num_params = num_layers * num_qubits * 3
    np.random.seed(42)
    params = list(np.random.uniform(-np.pi, np.pi, num_params))

    print("Quantum Risk Classifier Circuit:")
    print("=" * 50)
    print(cudaq.draw(vqc_circuit, encoding, params, num_qubits, num_layers))


if __name__ == "__main__":
    import os
    from neo_fetcher import NEOFetcher

    # Fetch recent NEOs - use env var or DEMO_KEY
    fetcher = NEOFetcher(os.getenv("NASA_API_KEY", "DEMO_KEY"))
    neos = fetcher.fetch_upcoming(3)

    print("Quantum NEO Risk Classifier")
    print("=" * 60)
    print(f"Processing {len(neos)} Near Earth Objects on NVIDIA GPU\n")

    # Visualize circuit
    visualize_circuit()

    # Create classifier and assess risks
    classifier = QuantumRiskClassifier(num_qubits=4, num_layers=3)

    print("\n" + "=" * 60)
    print("Risk Assessments (sorted by threat level)")
    print("=" * 60)

    assessments = classifier.batch_classify(neos)

    # Sort by threat level
    assessments.sort(key=lambda x: x.threat_level.value, reverse=True)

    # Show top threats
    for assessment in assessments[:10]:
        print(f"\n{assessment}")
        print(f"  Kinetic Energy: {assessment.neo.kinetic_energy_estimate:.2f} MT TNT")
        print(f"  Miss Distance: {assessment.neo.miss_distance_lunar:.1f} lunar distances")

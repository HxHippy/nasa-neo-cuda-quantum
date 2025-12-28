"""
Quantum Worker - Async wrapper for CUDA-Q analysis

Runs quantum analysis in a thread pool to avoid blocking the event loop,
and streams results via callbacks.

Enhanced with:
- Batch GPU execution
- VQE orbital stability
- Quantum Amplitude Estimation
- QAOA optimization
- VQC training
"""
import asyncio
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, List, Dict, Any
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import config
from neo_fetcher import NEOFetcher, NEO
from quantum_risk_classifier import QuantumRiskClassifier, ThreatLevel as QThreatLevel
from orbital_simulator import OrbitalSimulator
from quantum_engine import (
    QuantumNEOEngine, create_quantum_engine,
    ThreatLevel as EngineThreatLevel,
    VQEResult, QAEResult, BatchResult
)

from .schemas import (
    NEOData, RiskData, OrbitalData, ThreatLevel,
    AnalysisStatus, AnalysisConfig,
    VQEStabilityResult, QAECollisionResult, QuantumStateData
)


# Thread pool for CPU/GPU bound quantum operations
_executor = ThreadPoolExecutor(max_workers=4)


def neo_to_schema(neo: NEO) -> NEOData:
    """Convert NEO dataclass to Pydantic schema with 3D position"""
    # Generate pseudo-random but deterministic position based on NEO properties
    np.random.seed(hash(neo.id) % (2**32))

    # Position NEO at its miss distance, with random angle
    distance = neo.miss_distance_lunar
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(-np.pi/4, np.pi/4)  # Mostly in ecliptic plane

    return NEOData(
        id=neo.id,
        name=neo.name,
        absolute_magnitude=neo.absolute_magnitude,
        diameter_min_km=neo.diameter_min_km,
        diameter_max_km=neo.diameter_max_km,
        avg_diameter_km=neo.avg_diameter_km,
        is_hazardous=neo.is_hazardous,
        close_approach_date=neo.close_approach_date,
        velocity_km_s=neo.velocity_km_s,
        miss_distance_km=neo.miss_distance_km,
        miss_distance_lunar=neo.miss_distance_lunar,
        kinetic_energy_mt=neo.kinetic_energy_estimate,
        position_x=distance * np.cos(theta) * np.cos(phi),
        position_y=distance * np.sin(phi),
        position_z=distance * np.sin(theta) * np.cos(phi),
    )


def risk_to_schema(neo_id: str, risk) -> RiskData:
    """Convert RiskAssessment to Pydantic schema"""
    return RiskData(
        neo_id=neo_id,
        threat_level=ThreatLevel(risk.threat_level.name),
        probabilities={
            level.name: prob
            for level, prob in risk.probability_distribution.items()
        }
    )


def orbital_to_schema(neo_id: str, orbital) -> OrbitalData:
    """Convert OrbitalState to Pydantic schema"""
    return OrbitalData(
        neo_id=neo_id,
        collision_probability=orbital.collision_probability,
        position_uncertainty_km=orbital.position_uncertainty,
        velocity_uncertainty_km_s=orbital.velocity_uncertainty,
        evolution_uncertainties=list(orbital.evolution_states),
        time_steps_days=list(orbital.time_steps),
    )


def engine_threat_to_schema(level: EngineThreatLevel) -> ThreatLevel:
    """Convert engine ThreatLevel to schema ThreatLevel"""
    return ThreatLevel(level.name)


class QuantumWorker:
    """
    Manages quantum analysis with async streaming of results.

    Enhanced with:
    - Advanced quantum engine with GPU acceleration
    - Batch processing for efficiency
    - VQE, QAE, and QAOA capabilities
    - VQC training support
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.fetcher = NEOFetcher(api_key)

        # Legacy classifiers (for compatibility)
        self.classifier = QuantumRiskClassifier(
            num_qubits=config.QUANTUM_NUM_QUBITS,
            num_layers=config.QUANTUM_NUM_LAYERS
        )
        self.simulator = OrbitalSimulator()

        # Advanced quantum engine
        self.quantum_engine = create_quantum_engine(
            gpu=True,
            num_qubits=config.QUANTUM_NUM_QUBITS,
            num_layers=config.QUANTUM_NUM_LAYERS
        )

        self._cancel_flag = False
        self._training_in_progress = False

    def cancel(self):
        """Request cancellation of ongoing analysis"""
        self._cancel_flag = True

    async def run_analysis(
        self,
        config: AnalysisConfig,
        on_neo_list: Callable[[list[NEOData]], None],
        on_risk: Callable[[RiskData], None],
        on_orbital: Optional[Callable[[OrbitalData], None]],
        on_progress: Callable[[AnalysisStatus], None],
    ):
        """
        Run full quantum analysis pipeline with streaming callbacks.

        Uses batch GPU execution for improved performance.
        """
        self._cancel_flag = False
        loop = asyncio.get_event_loop()

        # Phase 1: Fetch NEO data
        on_progress(AnalysisStatus(
            current=0, total=0,
            phase="fetching",
            current_neo=None
        ))

        neos = await loop.run_in_executor(
            _executor,
            lambda: self.fetcher.fetch_upcoming(config.days_ahead)
        )

        if self._cancel_flag:
            return

        # Convert and send NEO list
        neo_data = [neo_to_schema(neo) for neo in neos]
        on_neo_list(neo_data)

        total = len(neos)
        if config.include_orbital_sim:
            total *= 2  # Risk + Orbital for each

        # Phase 2: Batch quantum risk classification (GPU accelerated)
        on_progress(AnalysisStatus(
            current=0,
            total=total,
            phase="classifying",
            current_neo="Batch GPU processing..."
        ))

        if config.use_batch_processing and len(neos) > 1:
            # Use batch execution for efficiency
            batch_risks = await self._batch_classify_neos(neos, config.shots_per_neo)

            for i, (neo, risk_data) in enumerate(zip(neos, batch_risks)):
                if self._cancel_flag:
                    return

                on_progress(AnalysisStatus(
                    current=i + 1,
                    total=total,
                    phase="classifying",
                    current_neo=neo.name
                ))

                on_risk(risk_data)

        else:
            # Sequential processing (fallback)
            for i, neo in enumerate(neos):
                if self._cancel_flag:
                    return

                on_progress(AnalysisStatus(
                    current=i + 1,
                    total=total,
                    phase="classifying",
                    current_neo=neo.name
                ))

                risk = await loop.run_in_executor(
                    _executor,
                    lambda n=neo: self.classifier.classify(n, shots=config.shots_per_neo)
                )

                on_risk(risk_to_schema(neo.id, risk))

        # Phase 3: Orbital simulation with QAE (if enabled)
        if config.include_orbital_sim and on_orbital:
            for i, neo in enumerate(neos):
                if self._cancel_flag:
                    return

                on_progress(AnalysisStatus(
                    current=len(neos) + i + 1,
                    total=total,
                    phase="simulating",
                    current_neo=neo.name
                ))

                # Use QAE-enhanced collision probability if enabled
                if config.use_qae:
                    orbital = await self._simulate_with_qae(neo, config.shots_per_neo)
                else:
                    orbital = await loop.run_in_executor(
                        _executor,
                        lambda n=neo: self.simulator.simulate_orbital_evolution(
                            n, num_steps=8, shots=config.shots_per_neo
                        )
                    )
                    orbital = orbital_to_schema(neo.id, orbital)

                on_orbital(orbital)

        # Complete
        on_progress(AnalysisStatus(
            current=total,
            total=total,
            phase="complete",
            current_neo=None
        ))

    async def _batch_classify_neos(
        self,
        neos: List[NEO],
        shots: int
    ) -> List[RiskData]:
        """Batch classify NEOs using GPU-accelerated quantum engine"""
        loop = asyncio.get_event_loop()

        # Prepare feature vectors
        features = [neo.to_feature_vector() for neo in neos]

        # Run batch classification
        batch_result = await loop.run_in_executor(
            _executor,
            lambda: self.quantum_engine.batch_classify(features, shots)
        )

        # Convert to RiskData
        risk_data = []
        for i, (neo, threat_level, probs) in enumerate(zip(
            neos,
            batch_result.threat_levels,
            batch_result.probabilities
        )):
            risk_data.append(RiskData(
                neo_id=neo.id,
                threat_level=ThreatLevel(threat_level.name),
                probabilities={level.name: prob for level, prob in probs.items()}
            ))

        return risk_data

    async def _simulate_with_qae(
        self,
        neo: NEO,
        shots: int
    ) -> OrbitalData:
        """Simulate orbit with QAE-enhanced collision probability"""
        loop = asyncio.get_event_loop()

        # Get QAE collision probability
        qae_result = await loop.run_in_executor(
            _executor,
            lambda: self.quantum_engine.estimate_collision_probability_qae(
                neo.miss_distance_lunar,
                neo.velocity_km_s,
                neo.avg_diameter_km,
                shots
            )
        )

        # Get orbital evolution from simulator
        orbital = await loop.run_in_executor(
            _executor,
            lambda: self.simulator.simulate_orbital_evolution(
                neo, num_steps=8, shots=shots
            )
        )

        # Combine results - use QAE probability as it's more accurate
        return OrbitalData(
            neo_id=neo.id,
            collision_probability=qae_result.estimated_probability,
            collision_probability_ci_low=qae_result.confidence_interval[0],
            collision_probability_ci_high=qae_result.confidence_interval[1],
            position_uncertainty_km=orbital.position_uncertainty,
            velocity_uncertainty_km_s=orbital.velocity_uncertainty,
            evolution_uncertainties=list(orbital.evolution_states),
            time_steps_days=list(orbital.time_steps),
            qae_precision_bits=qae_result.precision_bits,
            qae_oracle_calls=qae_result.num_oracle_calls
        )

    async def analyze_single_neo(
        self,
        neo_id: str,
        neos: list[NEO],
        high_precision: bool = False
    ) -> tuple[Optional[RiskData], Optional[OrbitalData]]:
        """Run detailed analysis on a single NEO."""
        loop = asyncio.get_event_loop()

        # Find the NEO
        neo = next((n for n in neos if n.id == neo_id), None)
        if not neo:
            return None, None

        shots = config.QUANTUM_HIGH_PRECISION_SHOTS if high_precision else config.QUANTUM_BATCH_SHOTS

        # Run classification
        risk = await loop.run_in_executor(
            _executor,
            lambda: self.classifier.classify(neo, shots=shots)
        )

        # Run QAE-enhanced orbital simulation
        orbital = await self._simulate_with_qae(neo, shots)

        return risk_to_schema(neo.id, risk), orbital

    # ========================================================================
    # Advanced Quantum Features
    # ========================================================================

    async def compute_orbital_stability(
        self,
        semi_major_axis_au: float,
        eccentricity: float,
        inclination_deg: float,
        longitude_deg: float,
        shots: int = 2000
    ) -> VQEStabilityResult:
        """Use VQE to compute orbital stability score."""
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            _executor,
            lambda: self.quantum_engine.compute_orbital_stability(
                semi_major_axis_au,
                eccentricity,
                inclination_deg,
                longitude_deg,
                shots=shots,
                max_iterations=30
            )
        )

        return VQEStabilityResult(
            ground_state_energy=result.ground_state_energy,
            stability_score=result.stability_score,
            convergence_history=result.convergence_history,
            iterations=result.iterations,
            quantum_state=self._quantum_state_to_schema(result.quantum_state)
        )

    async def estimate_collision_qae(
        self,
        neo: NEO,
        shots: int = 4000
    ) -> QAECollisionResult:
        """Use Quantum Amplitude Estimation for collision probability."""
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            _executor,
            lambda: self.quantum_engine.estimate_collision_probability_qae(
                neo.miss_distance_lunar,
                neo.velocity_km_s,
                neo.avg_diameter_km,
                shots
            )
        )

        return QAECollisionResult(
            neo_id=neo.id,
            estimated_probability=result.estimated_probability,
            confidence_interval_low=result.confidence_interval[0],
            confidence_interval_high=result.confidence_interval[1],
            precision_bits=result.precision_bits,
            oracle_calls=result.num_oracle_calls
        )

    async def optimize_deflection_qaoa(
        self,
        neo: NEO,
        target_miss_distance: float = 100,
        qaoa_layers: int = 4,
        shots: int = 2000
    ) -> dict:
        """QAOA-optimized deflection strategy."""
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            _executor,
            lambda: self.quantum_engine.optimize_deflection_qaoa(
                neo.miss_distance_lunar,
                neo.velocity_km_s,
                neo.avg_diameter_km,
                target_miss_distance,
                shots,
                qaoa_layers
            )
        )

        return {
            "neo_id": neo.id,
            "optimal_delta_v": result["optimal_delta_v"],
            "optimal_direction": result["optimal_direction"],
            "success_probability": result["success_probability"],
            "qaoa_layers": result["qaoa_layers"],
            "optimization_iterations": result["optimization_iterations"]
        }

    async def get_quantum_state(
        self,
        neo: NEO,
        shots: int = 1000
    ) -> QuantumStateData:
        """Get quantum state visualization for NEO classification."""
        loop = asyncio.get_event_loop()

        features = neo.to_feature_vector()

        state = await loop.run_in_executor(
            _executor,
            lambda: self.quantum_engine.visualize_state(features, shots)
        )

        return self._quantum_state_to_schema(state)

    async def train_classifier(
        self,
        num_samples: int = 100,
        epochs: int = 30,
        learning_rate: float = 0.1,
        use_nasa_data: bool = False
    ) -> dict:
        """
        Train the VQC classifier.

        Args:
            num_samples: Number of synthetic samples (if not using NASA data)
            epochs: Training epochs
            learning_rate: Learning rate for parameter updates
            use_nasa_data: If True, fetch real data from NASA APIs
        """
        if self._training_in_progress:
            return {"error": "Training already in progress"}

        self._training_in_progress = True
        loop = asyncio.get_event_loop()

        try:
            if use_nasa_data:
                # Train on real NASA data (Sentry, historic approaches, impacts)
                result = await loop.run_in_executor(
                    _executor,
                    lambda: self.quantum_engine.train_on_nasa_data(
                        epochs=epochs,
                        learning_rate=learning_rate,
                        shots=config.TRAINING_DEFAULT_SHOTS,
                        test_split=config.TRAINING_TEST_SPLIT
                    )
                )

                return {
                    "success": True,
                    "data_source": "nasa",
                    "train_samples": result.get("train_samples", 0),
                    "test_samples": result.get("test_samples", 0),
                    "final_train_accuracy": result.get("final_train_accuracy", 0),
                    "test_accuracy": result.get("test_accuracy", 0),
                    "epochs": epochs
                }
            else:
                # Generate synthetic training data
                training_data = await loop.run_in_executor(
                    _executor,
                    lambda: self.quantum_engine.generate_synthetic_training_data(num_samples)
                )

                # Train
                result = await loop.run_in_executor(
                    _executor,
                    lambda: self.quantum_engine.train_vqc(
                        training_data,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        shots=config.TRAINING_DEFAULT_SHOTS
                    )
                )

                return {
                    "success": True,
                    "data_source": "synthetic",
                    "final_accuracy": result["final_accuracy"],
                    "final_loss": result["final_loss"],
                    "epochs": epochs,
                    "samples": num_samples
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

        finally:
            self._training_in_progress = False

    def _quantum_state_to_schema(self, state) -> Optional[QuantumStateData]:
        """Convert quantum state to schema"""
        if state is None:
            return None

        state_dict = state.to_dict()
        return QuantumStateData(
            num_qubits=state_dict["num_qubits"],
            entropy=state_dict["entropy"],
            top_states=state_dict["top_states"],
            probabilities=state_dict["probabilities"]
        )

    # ========================================================================
    # Engine Status
    # ========================================================================

    def get_engine_status(self) -> dict:
        """Get quantum engine status"""
        return {
            "gpu_available": self.quantum_engine.use_gpu,
            "num_qubits": self.quantum_engine.num_qubits,
            "num_layers": self.quantum_engine.num_layers,
            "vqc_trained": self.quantum_engine.vqc_trained,
            "training_in_progress": self._training_in_progress
        }

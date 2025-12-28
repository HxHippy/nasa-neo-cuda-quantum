"""
Pydantic schemas for NEO Quantum API

Defines data models for WebSocket messages and REST endpoints.

Enhanced with:
- VQE orbital stability results
- QAE collision probability results
- Quantum state visualization data
- Batch processing configuration
- Training configuration
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List
from datetime import datetime


class ThreatLevel(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class NEOData(BaseModel):
    """Near Earth Object data for frontend"""
    id: str
    name: str
    absolute_magnitude: float
    diameter_min_km: float
    diameter_max_km: float
    avg_diameter_km: float
    is_hazardous: bool
    close_approach_date: str
    velocity_km_s: float
    miss_distance_km: float
    miss_distance_lunar: float
    kinetic_energy_mt: float

    # Position for 3D visualization (in lunar distances from Earth)
    # These are computed based on miss distance and approach angle
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0


class RiskData(BaseModel):
    """Quantum risk classification result"""
    neo_id: str
    threat_level: ThreatLevel
    probabilities: dict[str, float]  # ThreatLevel -> probability


class OrbitalData(BaseModel):
    """Orbital simulation result with optional QAE enhancement"""
    neo_id: str
    collision_probability: float
    position_uncertainty_km: float
    velocity_uncertainty_km_s: float
    evolution_uncertainties: list[float]  # Position uncertainty over time
    time_steps_days: list[float]

    # QAE-enhanced fields (optional)
    collision_probability_ci_low: Optional[float] = None
    collision_probability_ci_high: Optional[float] = None
    qae_precision_bits: Optional[int] = None
    qae_oracle_calls: Optional[int] = None


class AnalysisStatus(BaseModel):
    """Analysis progress update"""
    current: int
    total: int
    current_neo: Optional[str] = None
    phase: str  # "fetching", "classifying", "simulating", "complete"


# WebSocket message types
class WSMessageType(str, Enum):
    # Server -> Client
    NEO_LIST = "neo_list"
    RISK_UPDATE = "risk_update"
    ORBITAL_UPDATE = "orbital_update"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETE = "analysis_complete"
    AI_INSIGHT = "ai_insight"
    AI_FLEET_ANALYSIS = "ai_fleet_analysis"
    WHAT_IF_RESULT = "what_if_result"
    DEFLECTION_RESULT = "deflection_result"
    INTERCEPT_MISSION = "intercept_mission"
    ERROR = "error"

    # Advanced Quantum Results
    VQE_STABILITY_RESULT = "vqe_stability_result"
    QAE_COLLISION_RESULT = "qae_collision_result"
    QAOA_DEFLECTION_RESULT = "qaoa_deflection_result"
    QUANTUM_STATE = "quantum_state"
    TRAINING_RESULT = "training_result"
    ENGINE_STATUS = "engine_status"

    # Client -> Server
    START_ANALYSIS = "start_analysis"
    REQUEST_NEO_DETAILS = "request_neo_details"
    REQUEST_AI_INSIGHT = "request_ai_insight"
    REQUEST_FLEET_ANALYSIS = "request_fleet_analysis"
    REQUEST_WHAT_IF = "request_what_if"
    REQUEST_DEFLECTION = "request_deflection"
    REQUEST_INTERCEPT = "request_intercept"

    # Advanced Quantum Requests
    REQUEST_VQE_STABILITY = "request_vqe_stability"
    REQUEST_QAE_COLLISION = "request_qae_collision"
    REQUEST_QAOA_DEFLECTION = "request_qaoa_deflection"
    REQUEST_QUANTUM_STATE = "request_quantum_state"
    REQUEST_TRAIN_VQC = "request_train_vqc"
    REQUEST_ENGINE_STATUS = "request_engine_status"


class WSMessage(BaseModel):
    """WebSocket message wrapper"""
    type: WSMessageType
    data: Optional[dict] = None
    timestamp: datetime = None

    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()
        super().__init__(**data)


class AnalysisConfig(BaseModel):
    """Configuration for quantum analysis"""
    days_ahead: int = 7
    shots_per_neo: int = 500
    include_orbital_sim: bool = True

    # Advanced quantum options
    use_batch_processing: bool = True   # GPU batch execution
    use_qae: bool = True                # Quantum Amplitude Estimation
    use_vqe: bool = False               # VQE for orbital stability (expensive)
    qaoa_layers: int = 3                # QAOA circuit depth


class NEODetailRequest(BaseModel):
    """Request for detailed NEO analysis"""
    neo_id: str
    high_precision: bool = False  # Use more shots for higher precision


class AIInsight(BaseModel):
    """AI-generated insight about a NEO"""
    neo_id: str
    summary: str
    threat_assessment: str
    historical_context: str
    recommendations: list[str]
    fun_facts: list[str]
    confidence: float


class FleetAnalysis(BaseModel):
    """AI analysis of multiple NEOs"""
    total_analyzed: int
    executive_summary: str
    highest_priority_threats: list[str]
    patterns_detected: list[str]
    weekly_outlook: str
    recommendations: list[str]


# Simulation schemas

class ScenarioType(str, Enum):
    NOMINAL = "nominal"
    CLOSE_APPROACH = "close_approach"
    COLLISION_COURSE = "collision_course"
    DEFLECTED = "deflected"
    KEYHOLE = "keyhole"


class SimulationResult(BaseModel):
    """Result from what-if scenario simulation"""
    scenario: ScenarioType
    min_earth_distance: float  # lunar distances
    min_distance_time: float  # days from now
    collision_probability: float
    impact_energy_mt: Optional[float] = None
    quantum_uncertainty: float = 0.0
    confidence: float = 0.95


class DeflectionResult(BaseModel):
    """Result from deflection optimization"""
    optimal_delta_v: float  # m/s
    optimal_direction: tuple  # (x, y, z) unit vector
    deflection_time: float  # days before close approach
    miss_distance_achieved: float  # lunar distances
    energy_required_mt: float  # megatons TNT for kinetic impactor
    success_probability: float
    quantum_iterations: int


class LaunchWindow(BaseModel):
    """Launch window for intercept mission"""
    window_open: float  # days from now
    window_close: float  # days from now
    optimal_launch: float  # days from now
    c3_energy: float  # km^2/s^2
    flight_time: float  # days
    arrival_velocity: float  # km/s


class MissionPhase(BaseModel):
    """Mission phase details"""
    name: str
    duration: float  # days
    delta_v: float  # km/s
    description: str


class InterceptMission(BaseModel):
    """Complete intercept mission plan"""
    mission_type: str  # 'kinetic_impactor', 'gravity_tractor', 'nuclear_standoff'
    launch_windows: list[LaunchWindow]
    recommended_window: LaunchWindow
    mission_phases: list[MissionPhase]
    total_delta_v: float  # km/s
    mission_duration: float  # days
    intercept_date: float  # days from now
    success_probability: float


class WhatIfAnalysis(BaseModel):
    """Complete what-if analysis for a NEO"""
    neo_id: str
    scenarios: dict[str, SimulationResult]  # ScenarioType -> result


# ============================================================================
# Advanced Quantum Feature Schemas
# ============================================================================

class QuantumStateData(BaseModel):
    """Quantum state visualization data"""
    num_qubits: int
    entropy: float
    top_states: List[dict]  # [{"state": "0101", "probability": 0.25}, ...]
    probabilities: List[float]  # Full probability distribution


class VQEStabilityResult(BaseModel):
    """VQE orbital stability analysis result"""
    ground_state_energy: float
    stability_score: float  # 0-1, higher = more stable
    convergence_history: List[float]
    iterations: int
    quantum_state: Optional[QuantumStateData] = None


class QAECollisionResult(BaseModel):
    """Quantum Amplitude Estimation collision probability result"""
    neo_id: str
    estimated_probability: float
    confidence_interval_low: float
    confidence_interval_high: float
    precision_bits: int
    oracle_calls: int


class QAOADeflectionResult(BaseModel):
    """QAOA-optimized deflection result"""
    neo_id: str
    optimal_delta_v: float  # m/s
    optimal_direction: tuple  # (x, y, z) unit vector
    success_probability: float
    qaoa_layers: int
    optimization_iterations: int
    final_cost: Optional[float] = None


class TrainingConfig(BaseModel):
    """VQC training configuration"""
    num_samples: int = 100
    epochs: int = 30
    learning_rate: float = 0.1


class TrainingResult(BaseModel):
    """VQC training result"""
    success: bool
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    epochs: Optional[int] = None
    samples: Optional[int] = None
    error: Optional[str] = None


class EngineStatus(BaseModel):
    """Quantum engine status"""
    gpu_available: bool
    num_qubits: int
    num_layers: int
    vqc_trained: bool
    training_in_progress: bool

"""
NEO Quantum Configuration

Centralized configuration management. All configurable values are loaded
from environment variables with sensible defaults.

Usage:
    from config import config

    api_key = config.NASA_API_KEY
    shots = config.QUANTUM_DEFAULT_SHOTS
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent
_env_path = _project_root / ".env"
load_dotenv(_env_path)


class Config:
    """Application configuration loaded from environment variables."""

    # ========================================================================
    # NASA API Configuration
    # ========================================================================
    NASA_API_KEY: str = os.getenv("NASA_API_KEY", "DEMO_KEY")
    NASA_NEO_BASE_URL: str = os.getenv(
        "NASA_NEO_BASE_URL",
        "https://api.nasa.gov/neo/rest/v1"
    )

    # JPL APIs (public, no key required)
    JPL_SENTRY_API_URL: str = os.getenv(
        "JPL_SENTRY_API_URL",
        "https://ssd-api.jpl.nasa.gov/sentry.api"
    )
    JPL_CAD_API_URL: str = os.getenv(
        "JPL_CAD_API_URL",
        "https://ssd-api.jpl.nasa.gov/cad.api"
    )
    JPL_SBDB_API_URL: str = os.getenv(
        "JPL_SBDB_API_URL",
        "https://ssd-api.jpl.nasa.gov/sbdb.api"
    )

    # ========================================================================
    # OpenRouter AI Configuration
    # ========================================================================
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1/chat/completions"
    )
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL",
        "nvidia/nemotron-3-nano-30b-a3b:free"
    )
    OPENROUTER_HTTP_REFERER: str = os.getenv(
        "OPENROUTER_HTTP_REFERER",
        "https://neo-quantum.local"
    )

    # ========================================================================
    # Server Configuration
    # ========================================================================
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    WEBSOCKET_PATH: str = os.getenv("WEBSOCKET_PATH", "/ws/neo-stream")

    # Full WebSocket URL for frontend (set this if behind a proxy)
    WEBSOCKET_URL: str = os.getenv(
        "WEBSOCKET_URL",
        f"ws://localhost:{os.getenv('API_PORT', '8000')}/ws/neo-stream"
    )

    # ========================================================================
    # Quantum Circuit Defaults
    # ========================================================================
    QUANTUM_NUM_QUBITS: int = int(os.getenv("QUANTUM_NUM_QUBITS", "4"))
    QUANTUM_NUM_LAYERS: int = int(os.getenv("QUANTUM_NUM_LAYERS", "3"))
    QUANTUM_DEFAULT_SHOTS: int = int(os.getenv("QUANTUM_DEFAULT_SHOTS", "1000"))
    QUANTUM_BATCH_SHOTS: int = int(os.getenv("QUANTUM_BATCH_SHOTS", "500"))
    QUANTUM_HIGH_PRECISION_SHOTS: int = int(os.getenv("QUANTUM_HIGH_PRECISION_SHOTS", "2000"))

    # VQE Configuration
    VQE_NUM_QUBITS: int = int(os.getenv("VQE_NUM_QUBITS", "6"))
    VQE_DEFAULT_SHOTS: int = int(os.getenv("VQE_DEFAULT_SHOTS", "2000"))
    VQE_MAX_ITERATIONS: int = int(os.getenv("VQE_MAX_ITERATIONS", "50"))

    # QAE Configuration
    QAE_NUM_QUBITS: int = int(os.getenv("QAE_NUM_QUBITS", "8"))
    QAE_PRECISION_QUBITS: int = int(os.getenv("QAE_PRECISION_QUBITS", "4"))
    QAE_DEFAULT_SHOTS: int = int(os.getenv("QAE_DEFAULT_SHOTS", "4000"))

    # QAOA Configuration
    QAOA_NUM_QUBITS: int = int(os.getenv("QAOA_NUM_QUBITS", "8"))
    QAOA_DEFAULT_LAYERS: int = int(os.getenv("QAOA_DEFAULT_LAYERS", "4"))
    QAOA_DEFAULT_SHOTS: int = int(os.getenv("QAOA_DEFAULT_SHOTS", "2000"))

    # ========================================================================
    # Training Configuration
    # ========================================================================
    TRAINING_DEFAULT_EPOCHS: int = int(os.getenv("TRAINING_DEFAULT_EPOCHS", "30"))
    TRAINING_DEFAULT_LEARNING_RATE: float = float(os.getenv("TRAINING_DEFAULT_LEARNING_RATE", "0.1"))
    TRAINING_DEFAULT_SHOTS: int = int(os.getenv("TRAINING_DEFAULT_SHOTS", "500"))
    TRAINING_TEST_SPLIT: float = float(os.getenv("TRAINING_TEST_SPLIT", "0.2"))

    # Data fetching limits
    SENTRY_FETCH_LIMIT: int = int(os.getenv("SENTRY_FETCH_LIMIT", "100"))
    CAD_FETCH_LIMIT: int = int(os.getenv("CAD_FETCH_LIMIT", "500"))
    PHA_FETCH_LIMIT: int = int(os.getenv("PHA_FETCH_LIMIT", "200"))

    # ========================================================================
    # Cache Configuration
    # ========================================================================
    CACHE_DIR: str = os.getenv(
        "CACHE_DIR",
        str(_project_root / "src" / "training_cache")
    )
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"

    # ========================================================================
    # Debug/Development
    # ========================================================================
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if cls.NASA_API_KEY == "DEMO_KEY":
            warnings.append("NASA_API_KEY is using DEMO_KEY - rate limited to 30 requests/hour")

        if not cls.OPENROUTER_API_KEY:
            warnings.append("OPENROUTER_API_KEY not set - AI analysis disabled")

        return warnings

    @classmethod
    def to_dict(cls) -> dict:
        """Export configuration as dictionary (excludes secrets)."""
        return {
            "NASA_NEO_BASE_URL": cls.NASA_NEO_BASE_URL,
            "JPL_SENTRY_API_URL": cls.JPL_SENTRY_API_URL,
            "API_HOST": cls.API_HOST,
            "API_PORT": cls.API_PORT,
            "QUANTUM_NUM_QUBITS": cls.QUANTUM_NUM_QUBITS,
            "QUANTUM_NUM_LAYERS": cls.QUANTUM_NUM_LAYERS,
            "QUANTUM_DEFAULT_SHOTS": cls.QUANTUM_DEFAULT_SHOTS,
            "DEBUG": cls.DEBUG,
        }


# Singleton instance
config = Config()

# Print warnings on import (in debug mode)
if config.DEBUG:
    for warning in config.validate():
        print(f"[CONFIG WARNING] {warning}")

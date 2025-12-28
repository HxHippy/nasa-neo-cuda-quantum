# NEO Quantum Configuration Guide

This document describes all configuration options available in NEO Quantum and how to customize them for your environment.

## Quick Start

```bash
# Copy the example configuration
cp .env.example .env

# Edit with your API keys
nano .env  # or your preferred editor

# For the frontend (optional)
cp ui/.env.example ui/.env
```

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Backend configuration (Python/FastAPI) |
| `ui/.env` | Frontend configuration (Vite/TypeScript) |
| `src/config.py` | Configuration loader module |

## Environment Variables Reference

### NASA API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NASA_API_KEY` | `DEMO_KEY` | Your NASA API key. Get one free at [api.nasa.gov](https://api.nasa.gov/) |
| `NASA_NEO_BASE_URL` | `https://api.nasa.gov/neo/rest/v1` | NASA NEO Web Service base URL |

**Example:**
```bash
# Using the demo key (rate limited to 30 requests/hour)
NASA_API_KEY=DEMO_KEY

# Using your own key (1000 requests/hour)
NASA_API_KEY=your_40_character_api_key_here
```

### JPL API Configuration

These are public APIs that don't require authentication.

| Variable | Default | Description |
|----------|---------|-------------|
| `JPL_SENTRY_API_URL` | `https://ssd-api.jpl.nasa.gov/sentry.api` | Impact risk monitoring data |
| `JPL_CAD_API_URL` | `https://ssd-api.jpl.nasa.gov/cad.api` | Close approach data |
| `JPL_SBDB_API_URL` | `https://ssd-api.jpl.nasa.gov/sbdb.api` | Small-body database |

### OpenRouter AI Configuration

Optional - enables AI-powered fleet analysis reports.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (empty) | Your OpenRouter API key. Get one at [openrouter.ai](https://openrouter.ai/) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1/chat/completions` | API endpoint |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-nano-30b-a3b:free` | Model to use for analysis |
| `OPENROUTER_HTTP_REFERER` | `https://neo-quantum.local` | HTTP referer header |

**Example - Using a different model:**
```bash
# Free tier (default)
OPENROUTER_MODEL=nvidia/nemotron-3-nano-30b-a3b:free

# Better quality (costs credits)
OPENROUTER_MODEL=anthropic/claude-3-haiku

# Most capable (higher cost)
OPENROUTER_MODEL=anthropic/claude-3-opus
```

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Host to bind the server to |
| `API_PORT` | `8000` | Port for the API server |
| `WEBSOCKET_PATH` | `/ws/neo-stream` | WebSocket endpoint path |
| `WEBSOCKET_URL` | `ws://localhost:8000/ws/neo-stream` | Full WebSocket URL for frontend |

**Example - Production deployment:**
```bash
API_HOST=0.0.0.0
API_PORT=8080
WEBSOCKET_URL=wss://neo-quantum.yourdomain.com/ws/neo-stream
```

### Quantum Circuit Configuration

These control the default parameters for quantum circuit simulation.

#### VQC (Variational Quantum Classifier)

| Variable | Default | Description |
|----------|---------|-------------|
| `QUANTUM_NUM_QUBITS` | `4` | Number of qubits in VQC circuit |
| `QUANTUM_NUM_LAYERS` | `3` | Number of variational layers |
| `QUANTUM_DEFAULT_SHOTS` | `1000` | Default measurement shots |
| `QUANTUM_BATCH_SHOTS` | `500` | Shots for batch classification |
| `QUANTUM_HIGH_PRECISION_SHOTS` | `2000` | Shots for detailed analysis |

**Example - Higher precision (slower):**
```bash
QUANTUM_NUM_QUBITS=6
QUANTUM_NUM_LAYERS=4
QUANTUM_DEFAULT_SHOTS=2000
QUANTUM_HIGH_PRECISION_SHOTS=5000
```

**Example - Faster execution (lower precision):**
```bash
QUANTUM_NUM_QUBITS=4
QUANTUM_NUM_LAYERS=2
QUANTUM_DEFAULT_SHOTS=500
QUANTUM_BATCH_SHOTS=200
```

#### VQE (Variational Quantum Eigensolver)

| Variable | Default | Description |
|----------|---------|-------------|
| `VQE_NUM_QUBITS` | `6` | Qubits for orbital stability analysis |
| `VQE_DEFAULT_SHOTS` | `2000` | Measurement shots for VQE |
| `VQE_MAX_ITERATIONS` | `50` | Maximum optimization iterations |

#### QAE (Quantum Amplitude Estimation)

| Variable | Default | Description |
|----------|---------|-------------|
| `QAE_NUM_QUBITS` | `8` | Total qubits (state + counting) |
| `QAE_PRECISION_QUBITS` | `4` | Qubits for phase estimation |
| `QAE_DEFAULT_SHOTS` | `4000` | Shots for probability estimation |

#### QAOA (Quantum Approximate Optimization)

| Variable | Default | Description |
|----------|---------|-------------|
| `QAOA_NUM_QUBITS` | `8` | Qubits for deflection optimization |
| `QAOA_DEFAULT_LAYERS` | `4` | QAOA circuit depth |
| `QAOA_DEFAULT_SHOTS` | `2000` | Measurement shots |

### Training Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_DEFAULT_EPOCHS` | `30` | Training epochs |
| `TRAINING_DEFAULT_LEARNING_RATE` | `0.1` | Learning rate for parameter updates |
| `TRAINING_DEFAULT_SHOTS` | `500` | Shots per training sample |
| `TRAINING_TEST_SPLIT` | `0.2` | Fraction of data for testing |

**Example - Quick training run:**
```bash
TRAINING_DEFAULT_EPOCHS=10
TRAINING_DEFAULT_SHOTS=200
```

**Example - Thorough training:**
```bash
TRAINING_DEFAULT_EPOCHS=100
TRAINING_DEFAULT_LEARNING_RATE=0.05
TRAINING_DEFAULT_SHOTS=1000
```

### Data Fetching Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTRY_FETCH_LIMIT` | `100` | Max objects from Sentry API |
| `CAD_FETCH_LIMIT` | `500` | Max close approaches to fetch |
| `PHA_FETCH_LIMIT` | `200` | Max PHAs for training data |

### Cache Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_DIR` | `src/training_cache` | Directory for cached data |
| `ENABLE_CACHE` | `true` | Enable/disable caching |

### Debug Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Frontend Configuration

The frontend uses Vite environment variables (prefixed with `VITE_`).

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_WEBSOCKET_URL` | `ws://localhost:8000/ws/neo-stream` | WebSocket server URL |

**Example - ui/.env for production:**
```bash
VITE_WEBSOCKET_URL=wss://api.yourdomain.com/ws/neo-stream
```

## Configuration Profiles

### Development (Default)

```bash
# .env
NASA_API_KEY=DEMO_KEY
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
LOG_LEVEL=DEBUG

# Faster iteration
QUANTUM_DEFAULT_SHOTS=500
TRAINING_DEFAULT_EPOCHS=10
```

### Production

```bash
# .env
NASA_API_KEY=your_production_key
OPENROUTER_API_KEY=your_openrouter_key
API_HOST=0.0.0.0
API_PORT=8080
DEBUG=false
LOG_LEVEL=WARNING

# Higher quality
QUANTUM_DEFAULT_SHOTS=2000
QUANTUM_HIGH_PRECISION_SHOTS=5000
TRAINING_DEFAULT_EPOCHS=50
```

### High-Performance GPU Server

```bash
# .env - For systems with powerful NVIDIA GPUs
QUANTUM_NUM_QUBITS=8
QUANTUM_NUM_LAYERS=5
QUANTUM_DEFAULT_SHOTS=5000
VQE_MAX_ITERATIONS=100
QAE_DEFAULT_SHOTS=10000
TRAINING_DEFAULT_EPOCHS=100
TRAINING_DEFAULT_SHOTS=1000
```

### Minimal Resources

```bash
# .env - For systems with limited GPU memory
QUANTUM_NUM_QUBITS=4
QUANTUM_NUM_LAYERS=2
QUANTUM_DEFAULT_SHOTS=200
QUANTUM_BATCH_SHOTS=100
VQE_NUM_QUBITS=4
QAE_NUM_QUBITS=6
QAOA_NUM_QUBITS=6
```

## Using Configuration in Code

### Python Backend

```python
from config import config

# Access any configuration value
api_key = config.NASA_API_KEY
num_qubits = config.QUANTUM_NUM_QUBITS

# Check if optional services are configured
if config.OPENROUTER_API_KEY:
    # AI analysis is available
    pass

# Validate configuration on startup
warnings = config.validate()
for warning in warnings:
    print(f"Warning: {warning}")

# Export safe config (no secrets) for debugging
print(config.to_dict())
```

### TypeScript Frontend

```typescript
// Access via import.meta.env
const wsUrl = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000/ws/neo-stream';

// Connect to WebSocket
const dataStream = new NEODataStream(wsUrl);
```

## Troubleshooting

### "NASA API Key: NOT SET"

```bash
# Check your .env file exists
cat .env | grep NASA_API_KEY

# Make sure it's not empty
NASA_API_KEY=your_key_here  # Not NASA_API_KEY=
```

### "Rate limit exceeded"

```bash
# Using DEMO_KEY limits you to 30 requests/hour
# Get a free key at https://api.nasa.gov/
NASA_API_KEY=your_personal_key_here
```

### WebSocket connection failed

```bash
# Check the backend is running
curl http://localhost:8000/

# Verify WEBSOCKET_URL matches your server
WEBSOCKET_URL=ws://localhost:8000/ws/neo-stream

# For HTTPS sites, use wss://
VITE_WEBSOCKET_URL=wss://your-domain.com/ws/neo-stream
```

### GPU not detected

```bash
# CUDA-Q will fall back to CPU simulation
# Check NVIDIA drivers are installed
nvidia-smi

# Check CUDA toolkit
nvcc --version
```

## Security Notes

1. **Never commit `.env` files** - They're in `.gitignore` for a reason
2. **Use `.env.example`** - Commit this with placeholder values
3. **Rotate keys regularly** - Especially for production deployments
4. **Use environment variables in production** - Don't rely on `.env` files in containers

```bash
# Docker/Kubernetes - pass as environment variables
docker run -e NASA_API_KEY=xxx -e OPENROUTER_API_KEY=yyy neo-quantum
```

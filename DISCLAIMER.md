# Scientific and Technical Disclaimer

## This is an Educational Demonstration

NEO Quantum is an **educational and research demonstration project**. It is NOT:
- An operational planetary defense system
- An official NASA or ESA tool
- A source of authoritative asteroid risk assessments
- A replacement for professional impact hazard analysis

## On Quantum Computing Claims

### What This Project Uses

This project uses **NVIDIA CUDA-Q** to execute quantum circuits. CUDA-Q simulates quantum computation on classical GPU hardware.

### Important Clarifications

1. **No Actual Quantum Hardware**
   - All quantum circuits run on GPU-simulated qubits
   - No quantum processors are involved
   - This is classical simulation of quantum algorithms

2. **No Quantum Speedup**
   - On a classical simulator, quantum algorithms provide **zero computational advantage** over classical algorithms
   - We are demonstrating *what* quantum algorithms do, not gaining any performance benefit
   - The same results could be achieved (often faster) with purely classical methods

3. **Educational Purpose Only**
   - The quantum implementations (VQC, VQE, QAE, QAOA) show *how* these algorithms work
   - They demonstrate *how* quantum computing *could* be applied to this domain
   - They do not represent production-ready solutions

### Specific Algorithm Limitations

| Algorithm | Limitation |
|-----------|------------|
| **VQC (Classifier)** | Trained on heuristic labels, not physics-derived threat levels |
| **VQE (Stability)** | Uses simplified heuristic Hamiltonian, not rigorous celestial mechanics |
| **QAE (Probability)** | Oracle is simplified; provides no actual speedup on simulator |
| **QAOA (Deflection)** | Cost function is heuristic, not precise astrodynamics |

## On Risk Assessment Claims

### What the Threat Levels Mean

The threat levels displayed (NEGLIGIBLE, LOW, MODERATE, HIGH, CRITICAL) are:
- **Illustrative demonstrations** for educational purposes
- Based on simplified heuristic criteria
- **NOT official NASA or ESA risk assessments**
- **NOT derived from official Sentry impact probability calculations**

### For Actual Asteroid Risk Information

Always refer to authoritative sources:

| Source | URL | Description |
|--------|-----|-------------|
| NASA CNEOS | https://cneos.jpl.nasa.gov/ | Official NEO data and risk tables |
| NASA Sentry | https://cneos.jpl.nasa.gov/sentry/ | Official impact probability system |
| ESA NEOCC | https://neo.ssa.esa.int/ | European NEO Coordination Centre |
| IAU MPC | https://minorplanetcenter.net/ | Official observation clearing house |

### Collision Probabilities

Any collision probabilities shown in this software:
- Are computed using simplified educational models
- Should NOT be used for risk assessment
- Do NOT represent official impact probability calculations
- Are for visualization and demonstration purposes only

## On Orbital Mechanics

The orbital simulations in this project:
- Use simplified two-body or restricted N-body models
- Do not include all gravitational perturbations
- Are not suitable for mission planning
- Are intended for visualization, not precision ephemeris

## On AI-Generated Content

The AI Fleet Report feature:
- Uses a large language model to summarize data
- May produce inaccurate or misleading statements
- Should be treated as entertainment, not expert analysis
- Is not reviewed by planetary defense professionals

## Legal Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. THE AUTHORS AND COPYRIGHT HOLDERS SHALL NOT BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE.

**Do not use this software for:**
- Actual impact hazard assessment
- Space mission planning
- Public safety decisions
- Any purpose requiring accurate predictions

## Summary

NEO Quantum is a fun, educational project that:
1. Visualizes real NASA asteroid data
2. Demonstrates quantum computing algorithms on classical hardware
3. Shows how these technologies *could* work together

It is NOT a tool for actual planetary defense or risk assessment.

---

For questions about actual asteroid threats, contact:
- NASA Planetary Defense Coordination Office: https://www.nasa.gov/planetarydefense
- ESA Planetary Defence Office: https://www.esa.int/Space_Safety/Planetary_Defence

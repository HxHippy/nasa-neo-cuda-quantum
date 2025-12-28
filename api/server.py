"""
NEO Quantum API Server

FastAPI application with WebSocket support for real-time
quantum analysis streaming.
"""
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add paths for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import config (handles .env loading)
from config import config

from .schemas import (
    WSMessage, WSMessageType, NEOData, RiskData, OrbitalData,
    AnalysisConfig, AnalysisStatus, NEODetailRequest,
    SimulationResult, DeflectionResult, InterceptMission,
    LaunchWindow, MissionPhase, WhatIfAnalysis, ScenarioType,
    VQEStabilityResult, QAECollisionResult, QAOADeflectionResult,
    QuantumStateData, TrainingConfig, TrainingResult, EngineStatus
)
from .quantum_worker import QuantumWorker, neo_to_schema
from .ai_analyst import AIAnalyst

# Import quantum orbital mechanics simulator
from quantum_orbital_mechanics import QuantumOrbitalMechanics


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: WSMessage):
        """Send message to all connected clients"""
        data = message.model_dump_json()
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_personal(self, websocket: WebSocket, message: WSMessage):
        """Send message to specific client"""
        await websocket.send_text(message.model_dump_json())


# Global state
manager = ConnectionManager()
worker: Optional[QuantumWorker] = None
ai_analyst: Optional[AIAnalyst] = None
orbital_sim: Optional[QuantumOrbitalMechanics] = None
cached_neos: list = []  # Cache of raw NEO objects for detailed queries
cached_neo_data: dict[str, NEOData] = {}  # Cache of NEOData by ID
cached_risks: dict[str, RiskData] = {}  # Cache of risk data by NEO ID
analysis_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global worker, ai_analyst, orbital_sim
    worker = QuantumWorker(config.NASA_API_KEY)
    ai_analyst = AIAnalyst(config.OPENROUTER_API_KEY, config.OPENROUTER_MODEL)
    orbital_sim = QuantumOrbitalMechanics()
    print("NEO Quantum API Server started")
    print(f"NASA API Key: {config.NASA_API_KEY[:10]}..." if config.NASA_API_KEY else "NASA API Key: NOT SET")
    print(f"OpenRouter: {config.OPENROUTER_MODEL}")
    print("Quantum orbital simulator initialized")
    yield
    if ai_analyst:
        await ai_analyst.close()
    print("NEO Quantum API Server shutting down")


def _neo_to_dict(neo: NEOData) -> dict:
    """Convert NEOData to dict for orbital simulator"""
    return {
        'id': neo.id,
        'name': neo.name,
        'avg_diameter_km': neo.avg_diameter_km,
        'velocity_km_s': neo.velocity_km_s,
        'miss_distance_km': neo.miss_distance_km,
        'miss_distance_lunar': neo.miss_distance_lunar,
        'kinetic_energy_mt': neo.kinetic_energy_mt,
    }


app = FastAPI(
    title="NEO Quantum API",
    description="Real-time Near Earth Object analysis with quantum computing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REST Endpoints

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "NEO Quantum API",
        "version": "1.0.0",
        "status": "running",
        "websocket": "/ws/neo-stream"
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/neos", response_model=list[NEOData])
async def get_neos(days: int = 7):
    """Fetch NEOs without quantum analysis (quick endpoint)"""
    from neo_fetcher import NEOFetcher
    fetcher = NEOFetcher(config.NASA_API_KEY)
    neos = fetcher.fetch_upcoming(days)
    return [neo_to_schema(neo) for neo in neos]


# WebSocket Endpoint

@app.websocket("/ws/neo-stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time NEO analysis streaming.

    Client can send:
    - {"type": "start_analysis", "data": {"days_ahead": 7, ...}}
    - {"type": "request_neo_details", "data": {"neo_id": "123"}}

    Server sends:
    - {"type": "neo_list", "data": [...]}
    - {"type": "risk_update", "data": {...}}
    - {"type": "orbital_update", "data": {...}}
    - {"type": "analysis_progress", "data": {...}}
    - {"type": "analysis_complete"}
    """
    global analysis_task, cached_neos

    await manager.connect(websocket)
    print(f"Client connected. Total connections: {len(manager.active_connections)}")

    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type")
                msg_data = message.get("data", {})

                if msg_type == "start_analysis":
                    # Cancel any existing analysis
                    if analysis_task and not analysis_task.done():
                        worker.cancel()
                        analysis_task.cancel()

                    # Parse config
                    config = AnalysisConfig(**msg_data) if msg_data else AnalysisConfig()

                    # Start analysis
                    analysis_task = asyncio.create_task(
                        run_analysis_stream(websocket, config)
                    )

                elif msg_type == "request_neo_details":
                    neo_id = msg_data.get("neo_id")
                    high_precision = msg_data.get("high_precision", False)

                    if neo_id and cached_neos:
                        risk, orbital = await worker.analyze_single_neo(
                            neo_id, cached_neos, high_precision
                        )
                        if risk:
                            await manager.send_personal(websocket, WSMessage(
                                type=WSMessageType.RISK_UPDATE,
                                data=risk.model_dump()
                            ))
                        if orbital:
                            await manager.send_personal(websocket, WSMessage(
                                type=WSMessageType.ORBITAL_UPDATE,
                                data=orbital.model_dump()
                            ))

                elif msg_type == "cancel_analysis":
                    if analysis_task and not analysis_task.done():
                        worker.cancel()

                elif msg_type == "request_ai_insight":
                    neo_id = msg_data.get("neo_id")
                    if neo_id and neo_id in cached_neo_data:
                        neo_data = cached_neo_data[neo_id]
                        risk_data = cached_risks.get(neo_id)
                        insight = await ai_analyst.analyze_neo(neo_data, risk_data)
                        await manager.send_personal(websocket, WSMessage(
                            type=WSMessageType.AI_INSIGHT,
                            data=insight.model_dump()
                        ))

                elif msg_type == "request_fleet_analysis":
                    if cached_neo_data:
                        neos_list = list(cached_neo_data.values())
                        fleet = await ai_analyst.analyze_fleet(neos_list, cached_risks)
                        await manager.send_personal(websocket, WSMessage(
                            type=WSMessageType.AI_FLEET_ANALYSIS,
                            data=fleet.model_dump()
                        ))

                # Simulation requests
                elif msg_type == "request_what_if":
                    neo_id = msg_data.get("neo_id")
                    if neo_id and neo_id in cached_neo_data:
                        neo = cached_neo_data[neo_id]
                        neo_dict = _neo_to_dict(neo)
                        results = await asyncio.to_thread(
                            orbital_sim.run_what_if_analysis, neo_dict, 500
                        )
                        # Convert to serializable format
                        scenarios = {}
                        for scenario_type, result in results.items():
                            scenarios[scenario_type.value] = {
                                "scenario": scenario_type.value,
                                "min_earth_distance": result.min_earth_distance,
                                "min_distance_time": result.min_distance_time,
                                "collision_probability": result.collision_probability,
                                "impact_energy_mt": result.impact_energy_mt,
                                "quantum_uncertainty": result.quantum_uncertainty,
                                "confidence": result.confidence
                            }
                        await manager.send_personal(websocket, WSMessage(
                            type=WSMessageType.WHAT_IF_RESULT,
                            data={"neo_id": neo_id, "scenarios": scenarios}
                        ))

                elif msg_type == "request_deflection":
                    neo_id = msg_data.get("neo_id")
                    target_distance = msg_data.get("target_distance", 100)
                    if neo_id and neo_id in cached_neo_data:
                        neo = cached_neo_data[neo_id]
                        neo_dict = _neo_to_dict(neo)
                        result = await asyncio.to_thread(
                            orbital_sim.optimize_deflection, neo_dict, target_distance
                        )
                        await manager.send_personal(websocket, WSMessage(
                            type=WSMessageType.DEFLECTION_RESULT,
                            data={
                                "neo_id": neo_id,
                                "optimal_delta_v": result.optimal_delta_v,
                                "optimal_direction": result.optimal_direction,
                                "deflection_time": result.deflection_time,
                                "miss_distance_achieved": result.miss_distance_achieved,
                                "energy_required_mt": result.energy_required_mt,
                                "success_probability": result.success_probability,
                                "quantum_iterations": result.quantum_iterations
                            }
                        ))

                elif msg_type == "request_intercept":
                    neo_id = msg_data.get("neo_id")
                    mission_type = msg_data.get("mission_type", "kinetic_impactor")
                    if neo_id and neo_id in cached_neo_data:
                        neo = cached_neo_data[neo_id]
                        neo_dict = _neo_to_dict(neo)
                        result = await asyncio.to_thread(
                            orbital_sim.plan_intercept_mission, neo_dict, mission_type
                        )
                        # Convert to serializable format
                        windows = [
                            {
                                "window_open": w.window_open,
                                "window_close": w.window_close,
                                "optimal_launch": w.optimal_launch,
                                "c3_energy": w.c3_energy,
                                "flight_time": w.flight_time,
                                "arrival_velocity": w.arrival_velocity
                            }
                            for w in result.launch_windows
                        ]
                        phases = [
                            {
                                "name": p["name"],
                                "duration": p["duration"],
                                "delta_v": p["delta_v"],
                                "description": p["description"]
                            }
                            for p in result.mission_phases
                        ]
                        await manager.send_personal(websocket, WSMessage(
                            type=WSMessageType.INTERCEPT_MISSION,
                            data={
                                "neo_id": neo_id,
                                "mission_type": result.mission_type,
                                "launch_windows": windows,
                                "recommended_window": {
                                    "window_open": result.recommended_window.window_open,
                                    "window_close": result.recommended_window.window_close,
                                    "optimal_launch": result.recommended_window.optimal_launch,
                                    "c3_energy": result.recommended_window.c3_energy,
                                    "flight_time": result.recommended_window.flight_time,
                                    "arrival_velocity": result.recommended_window.arrival_velocity
                                },
                                "mission_phases": phases,
                                "total_delta_v": result.total_delta_v,
                                "mission_duration": result.mission_duration,
                                "intercept_date": result.intercept_date,
                                "success_probability": result.success_probability
                            }
                        ))

                # ============================================================
                # Advanced Quantum Feature Handlers
                # ============================================================

                elif msg_type == "request_vqe_stability":
                    neo_id = msg_data.get("neo_id")
                    if neo_id and neo_id in cached_neo_data:
                        neo = cached_neo_data[neo_id]
                        # Use approximate orbital elements
                        result = await worker.compute_orbital_stability(
                            semi_major_axis_au=1.5,  # Approximate from velocity
                            eccentricity=0.3,
                            inclination_deg=15,
                            longitude_deg=45,
                            shots=msg_data.get("shots", 2000)
                        )
                        await manager.send_personal(websocket, WSMessage(
                            type=WSMessageType.VQE_STABILITY_RESULT,
                            data={
                                "neo_id": neo_id,
                                "ground_state_energy": result.ground_state_energy,
                                "stability_score": result.stability_score,
                                "convergence_history": result.convergence_history,
                                "iterations": result.iterations,
                                "quantum_state": result.quantum_state.model_dump() if result.quantum_state else None
                            }
                        ))

                elif msg_type == "request_qae_collision":
                    neo_id = msg_data.get("neo_id")
                    if neo_id and cached_neos:
                        neo = next((n for n in cached_neos if n.id == neo_id), None)
                        if neo:
                            result = await worker.estimate_collision_qae(
                                neo,
                                shots=msg_data.get("shots", 4000)
                            )
                            await manager.send_personal(websocket, WSMessage(
                                type=WSMessageType.QAE_COLLISION_RESULT,
                                data={
                                    "neo_id": neo_id,
                                    "estimated_probability": result.estimated_probability,
                                    "confidence_interval_low": result.confidence_interval_low,
                                    "confidence_interval_high": result.confidence_interval_high,
                                    "precision_bits": result.precision_bits,
                                    "oracle_calls": result.oracle_calls
                                }
                            ))

                elif msg_type == "request_qaoa_deflection":
                    neo_id = msg_data.get("neo_id")
                    if neo_id and cached_neos:
                        neo = next((n for n in cached_neos if n.id == neo_id), None)
                        if neo:
                            result = await worker.optimize_deflection_qaoa(
                                neo,
                                target_miss_distance=msg_data.get("target_distance", 100),
                                qaoa_layers=msg_data.get("qaoa_layers", 4),
                                shots=msg_data.get("shots", 2000)
                            )
                            await manager.send_personal(websocket, WSMessage(
                                type=WSMessageType.QAOA_DEFLECTION_RESULT,
                                data=result
                            ))

                elif msg_type == "request_quantum_state":
                    neo_id = msg_data.get("neo_id")
                    if neo_id and cached_neos:
                        neo = next((n for n in cached_neos if n.id == neo_id), None)
                        if neo:
                            state = await worker.get_quantum_state(
                                neo,
                                shots=msg_data.get("shots", 1000)
                            )
                            await manager.send_personal(websocket, WSMessage(
                                type=WSMessageType.QUANTUM_STATE,
                                data={
                                    "neo_id": neo_id,
                                    "num_qubits": state.num_qubits,
                                    "entropy": state.entropy,
                                    "top_states": state.top_states,
                                    "probabilities": state.probabilities
                                }
                            ))

                elif msg_type == "request_train_vqc":
                    result = await worker.train_classifier(
                        num_samples=msg_data.get("num_samples", 100),
                        epochs=msg_data.get("epochs", 30),
                        learning_rate=msg_data.get("learning_rate", 0.1),
                        use_nasa_data=msg_data.get("use_nasa_data", False)
                    )
                    await manager.send_personal(websocket, WSMessage(
                        type=WSMessageType.TRAINING_RESULT,
                        data=result
                    ))

                elif msg_type == "request_engine_status":
                    status = worker.get_engine_status()
                    await manager.send_personal(websocket, WSMessage(
                        type=WSMessageType.ENGINE_STATUS,
                        data=status
                    ))

            except json.JSONDecodeError:
                await manager.send_personal(websocket, WSMessage(
                    type=WSMessageType.ERROR,
                    data={"message": "Invalid JSON"}
                ))
            except Exception as e:
                await manager.send_personal(websocket, WSMessage(
                    type=WSMessageType.ERROR,
                    data={"message": str(e)}
                ))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client disconnected. Total connections: {len(manager.active_connections)}")


async def run_analysis_stream(websocket: WebSocket, analysis_config: AnalysisConfig):
    """Run quantum analysis and stream results to client"""
    global cached_neos, cached_neo_data, cached_risks

    from neo_fetcher import NEOFetcher
    fetcher = NEOFetcher(config.NASA_API_KEY)  # Use global config for API key

    async def on_neo_list(neos: list[NEOData]):
        # Cache raw NEOs for later detailed queries
        cached_neos.clear()
        cached_neo_data.clear()
        cached_risks.clear()
        cached_neos.extend(fetcher.fetch_upcoming(analysis_config.days_ahead))

        # Cache NEOData by ID for AI analysis
        for neo in neos:
            cached_neo_data[neo.id] = neo

        await manager.send_personal(websocket, WSMessage(
            type=WSMessageType.NEO_LIST,
            data={"neos": [n.model_dump() for n in neos]}
        ))

    async def on_risk(risk: RiskData):
        # Cache risk data for AI analysis
        cached_risks[risk.neo_id] = risk
        await manager.send_personal(websocket, WSMessage(
            type=WSMessageType.RISK_UPDATE,
            data=risk.model_dump()
        ))

    async def on_orbital(orbital: OrbitalData):
        await manager.send_personal(websocket, WSMessage(
            type=WSMessageType.ORBITAL_UPDATE,
            data=orbital.model_dump()
        ))

    async def on_progress(status: AnalysisStatus):
        await manager.send_personal(websocket, WSMessage(
            type=WSMessageType.ANALYSIS_PROGRESS,
            data=status.model_dump()
        ))

    try:
        await worker.run_analysis(
            analysis_config,
            on_neo_list=lambda neos: asyncio.create_task(on_neo_list(neos)),
            on_risk=lambda risk: asyncio.create_task(on_risk(risk)),
            on_orbital=lambda orbital: asyncio.create_task(on_orbital(orbital)),
            on_progress=lambda status: asyncio.create_task(on_progress(status)),
        )

        await manager.send_personal(websocket, WSMessage(
            type=WSMessageType.ANALYSIS_COMPLETE,
            data={}
        ))

    except asyncio.CancelledError:
        print("Analysis cancelled")
    except Exception as e:
        await manager.send_personal(websocket, WSMessage(
            type=WSMessageType.ERROR,
            data={"message": f"Analysis error: {str(e)}"}
        ))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

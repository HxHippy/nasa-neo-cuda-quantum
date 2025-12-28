"""
AI Analyst - Uses OpenRouter/Nemotron for intelligent NEO threat analysis

Provides natural language insights about asteroid threats, recommendations,
and contextual analysis beyond what quantum classification provides.
"""
import sys
import json
import httpx
from typing import Optional
from pydantic import BaseModel
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import config

from .schemas import NEOData, RiskData, OrbitalData, ThreatLevel


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


class AIAnalyst:
    """
    AI-powered asteroid threat analyst using OpenRouter API.

    Uses NVIDIA Nemotron model for generating intelligent insights
    about NEO data and quantum analysis results.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.model = model or config.OPENROUTER_MODEL
        self.base_url = config.OPENROUTER_BASE_URL

        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": config.OPENROUTER_HTTP_REFERER,
                "X-Title": "NEO Quantum Analyzer"
            },
            timeout=60.0
        )

    async def _chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send chat completion request to OpenRouter"""
        try:
            response = await self.client.post(
                self.base_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000,
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"AI API error: {e}")
            return f"AI analysis unavailable: {str(e)}"

    async def analyze_neo(
        self,
        neo: NEOData,
        risk: Optional[RiskData] = None,
        orbital: Optional[OrbitalData] = None
    ) -> AIInsight:
        """
        Generate AI insights for a single NEO.
        """
        system_prompt = """You are an expert asteroid threat analyst working with NASA data.
Provide concise, scientifically accurate analysis of Near Earth Objects.
Be informative but accessible. Include relevant historical context when appropriate.
Format your response as JSON with these fields:
- summary: 1-2 sentence overview
- threat_assessment: detailed threat evaluation
- historical_context: relevant comparisons to past events
- recommendations: list of 2-3 actionable items
- fun_facts: list of 2-3 interesting facts
- confidence: 0-1 confidence in assessment"""

        neo_info = f"""
NEO: {neo.name}
- Diameter: {neo.avg_diameter_km * 1000:.0f} meters ({neo.diameter_min_km * 1000:.0f}-{neo.diameter_max_km * 1000:.0f}m range)
- Velocity: {neo.velocity_km_s:.2f} km/s relative to Earth
- Miss Distance: {neo.miss_distance_lunar:.1f} lunar distances ({neo.miss_distance_km:,.0f} km)
- Close Approach Date: {neo.close_approach_date}
- Kinetic Energy: {neo.kinetic_energy_mt:.2f} megatons TNT equivalent
- NASA Hazard Classification: {"Potentially Hazardous Asteroid (PHA)" if neo.is_hazardous else "Not classified as hazardous"}
- Absolute Magnitude: {neo.absolute_magnitude}
"""

        if risk:
            neo_info += f"""
Quantum Risk Analysis:
- Threat Level: {risk.threat_level}
- Probability Distribution: {json.dumps(risk.probabilities)}
"""

        if orbital:
            neo_info += f"""
Orbital Simulation:
- Collision Probability: {orbital.collision_probability:.2e}
- Position Uncertainty: {orbital.position_uncertainty_km:,.0f} km
- Velocity Uncertainty: {orbital.velocity_uncertainty_km_s:.3f} km/s
"""

        user_prompt = f"Analyze this Near Earth Object and provide your assessment:\n{neo_info}"

        response = await self._chat(system_prompt, user_prompt)

        # Parse JSON response
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                raise ValueError("No JSON found")

            data = json.loads(json_str)
            return AIInsight(
                neo_id=neo.id,
                summary=data.get("summary", "Analysis completed"),
                threat_assessment=data.get("threat_assessment", response),
                historical_context=data.get("historical_context", ""),
                recommendations=data.get("recommendations", []),
                fun_facts=data.get("fun_facts", []),
                confidence=data.get("confidence", 0.7)
            )
        except (json.JSONDecodeError, ValueError):
            # Fallback if JSON parsing fails
            return AIInsight(
                neo_id=neo.id,
                summary="AI analysis completed",
                threat_assessment=response,
                historical_context="",
                recommendations=["Continue monitoring this object"],
                fun_facts=[],
                confidence=0.5
            )

    async def analyze_fleet(
        self,
        neos: list[NEOData],
        risks: dict[str, RiskData]
    ) -> FleetAnalysis:
        """
        Generate AI analysis of multiple NEOs as a fleet.
        """
        system_prompt = """You are an expert asteroid threat analyst providing a fleet-wide assessment.
Analyze the collection of Near Earth Objects and provide strategic insights.
Format your response as JSON with these fields:
- executive_summary: 2-3 sentence overview of the threat landscape
- highest_priority_threats: list of NEO names requiring attention
- patterns_detected: list of notable patterns in the data
- weekly_outlook: assessment for the coming week
- recommendations: list of 3-5 strategic recommendations"""

        # Summarize fleet data
        hazardous_count = sum(1 for n in neos if n.is_hazardous)
        threat_counts = {"NEGLIGIBLE": 0, "LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
        for risk in risks.values():
            threat_counts[risk.threat_level] = threat_counts.get(risk.threat_level, 0) + 1

        closest = min(neos, key=lambda n: n.miss_distance_lunar)
        largest = max(neos, key=lambda n: n.avg_diameter_km)
        fastest = max(neos, key=lambda n: n.velocity_km_s)
        most_energetic = max(neos, key=lambda n: n.kinetic_energy_mt)

        fleet_info = f"""
Fleet Summary: {len(neos)} Near Earth Objects

Threat Distribution:
- Critical: {threat_counts.get('CRITICAL', 0)}
- High: {threat_counts.get('HIGH', 0)}
- Moderate: {threat_counts.get('MODERATE', 0)}
- Low: {threat_counts.get('LOW', 0)}
- Negligible: {threat_counts.get('NEGLIGIBLE', 0)}

NASA Potentially Hazardous Asteroids: {hazardous_count}

Notable Objects:
- Closest Approach: {closest.name} at {closest.miss_distance_lunar:.1f} lunar distances
- Largest: {largest.name} at {largest.avg_diameter_km * 1000:.0f} meters
- Fastest: {fastest.name} at {fastest.velocity_km_s:.2f} km/s
- Most Energetic: {most_energetic.name} at {most_energetic.kinetic_energy_mt:.2f} MT
"""

        user_prompt = f"Provide a strategic assessment of this asteroid fleet:\n{fleet_info}"

        response = await self._chat(system_prompt, user_prompt)

        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                raise ValueError("No JSON found")

            data = json.loads(json_str)
            return FleetAnalysis(
                total_analyzed=len(neos),
                executive_summary=data.get("executive_summary", "Analysis completed"),
                highest_priority_threats=data.get("highest_priority_threats", []),
                patterns_detected=data.get("patterns_detected", []),
                weekly_outlook=data.get("weekly_outlook", ""),
                recommendations=data.get("recommendations", [])
            )
        except (json.JSONDecodeError, ValueError):
            return FleetAnalysis(
                total_analyzed=len(neos),
                executive_summary=response[:500],
                highest_priority_threats=[],
                patterns_detected=[],
                weekly_outlook="",
                recommendations=["Continue monitoring all tracked objects"]
            )

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

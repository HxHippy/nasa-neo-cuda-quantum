"""
NEO Data Fetcher - Retrieves Near Earth Object data from NASA API
"""
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

from config import config


@dataclass
class NEO:
    """Near Earth Object data structure"""
    id: str
    name: str
    absolute_magnitude: float
    diameter_min_km: float
    diameter_max_km: float
    is_hazardous: bool
    close_approach_date: str
    velocity_km_s: float
    miss_distance_km: float
    miss_distance_lunar: float

    @property
    def avg_diameter_km(self) -> float:
        return (self.diameter_min_km + self.diameter_max_km) / 2

    @property
    def kinetic_energy_estimate(self) -> float:
        """Estimate kinetic energy in megatons TNT equivalent"""
        # Assume density ~2500 kg/m^3 (typical S-type asteroid)
        radius_m = self.avg_diameter_km * 500  # diameter/2 * 1000
        volume = (4/3) * np.pi * radius_m**3
        mass = volume * 2500  # kg
        velocity_m_s = self.velocity_km_s * 1000
        energy_joules = 0.5 * mass * velocity_m_s**2
        # 1 megaton TNT = 4.184e15 joules
        return energy_joules / 4.184e15

    def to_feature_vector(self) -> np.ndarray:
        """Convert NEO to normalized feature vector for quantum processing"""
        return np.array([
            self.absolute_magnitude / 35.0,  # Normalize to ~[0,1]
            np.log10(self.avg_diameter_km * 1000 + 1) / 5.0,  # Log scale diameter
            self.velocity_km_s / 30.0,  # Normalize velocity
            np.log10(self.miss_distance_km + 1) / 10.0,  # Log scale distance
            float(self.is_hazardous),
        ])


class NEOFetcher:
    """Fetches NEO data from NASA NeoWs API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.NASA_API_KEY
        self.base_url = config.NASA_NEO_BASE_URL

    def fetch_feed(self, start_date: str, end_date: str) -> list[NEO]:
        """
        Fetch NEOs within a date range (max 7 days)

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of NEO objects
        """
        url = f"{self.base_url}/feed"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "api_key": self.api_key
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        neos = []
        for date, objects in data.get("near_earth_objects", {}).items():
            for obj in objects:
                neo = self._parse_neo(obj)
                if neo:
                    neos.append(neo)

        return neos

    def fetch_recent(self, days: int = 7) -> list[NEO]:
        """Fetch NEOs from the past N days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.fetch_feed(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

    def fetch_upcoming(self, days: int = 7) -> list[NEO]:
        """Fetch upcoming NEOs for the next N days"""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days)
        return self.fetch_feed(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

    def _parse_neo(self, obj: dict) -> Optional[NEO]:
        """Parse NASA API response into NEO object"""
        try:
            diameter = obj.get("estimated_diameter", {}).get("kilometers", {})
            approach = obj.get("close_approach_data", [{}])[0]

            return NEO(
                id=obj["id"],
                name=obj["name"],
                absolute_magnitude=obj.get("absolute_magnitude_h", 0),
                diameter_min_km=diameter.get("estimated_diameter_min", 0),
                diameter_max_km=diameter.get("estimated_diameter_max", 0),
                is_hazardous=obj.get("is_potentially_hazardous_asteroid", False),
                close_approach_date=approach.get("close_approach_date", ""),
                velocity_km_s=float(approach.get("relative_velocity", {}).get("kilometers_per_second", 0)),
                miss_distance_km=float(approach.get("miss_distance", {}).get("kilometers", 0)),
                miss_distance_lunar=float(approach.get("miss_distance", {}).get("lunar", 0)),
            )
        except (KeyError, ValueError, IndexError) as e:
            print(f"Warning: Could not parse NEO: {e}")
            return None


if __name__ == "__main__":
    import os
    # Test the fetcher - use env var or DEMO_KEY
    fetcher = NEOFetcher(os.getenv("NASA_API_KEY", "DEMO_KEY"))
    neos = fetcher.fetch_upcoming(3)

    print(f"\nFetched {len(neos)} NEOs")
    print("\nHazardous asteroids:")
    for neo in sorted(neos, key=lambda x: x.miss_distance_km):
        if neo.is_hazardous:
            print(f"  {neo.name}")
            print(f"    Diameter: {neo.avg_diameter_km:.2f} km")
            print(f"    Velocity: {neo.velocity_km_s:.2f} km/s")
            print(f"    Miss distance: {neo.miss_distance_lunar:.1f} lunar distances")
            print(f"    Est. kinetic energy: {neo.kinetic_energy_estimate:.2f} MT")

"""
Training Data Fetcher for VQC Classifier

Fetches real NASA data for training the quantum classifier:
- NASA Sentry API: Objects with non-zero impact probability
- Close Approach API: Historic and future close approaches
- SBDB API: Detailed orbital elements and physical properties
- Known impact events for ground truth labeling

Author: HxHippy (https://hxhippy.com) | CTO of Kief Studio
"""
import json
import requests
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
from enum import Enum
from pathlib import Path
import time

from config import config


class ThreatLevel(Enum):
    NEGLIGIBLE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TrainingNEO:
    """NEO with training label"""
    id: str
    name: str

    # Physical properties
    diameter_km: float
    absolute_magnitude: float
    albedo: Optional[float] = None

    # Orbital elements
    semi_major_axis_au: float = 1.0
    eccentricity: float = 0.0
    inclination_deg: float = 0.0

    # Close approach data
    min_distance_au: float = 1.0
    velocity_km_s: float = 10.0
    approach_date: str = ""

    # Risk data (from Sentry or computed)
    impact_probability: float = 0.0
    palermo_scale: float = -10.0  # Negative = low risk
    torino_scale: int = 0  # 0-10

    # Computed
    kinetic_energy_mt: float = 0.0
    is_pha: bool = False

    # Training label
    threat_level: ThreatLevel = ThreatLevel.NEGLIGIBLE
    label_source: str = "computed"  # "sentry", "historic", "impact", "computed"

    def to_feature_vector(self) -> np.ndarray:
        """Convert to normalized feature vector for training"""
        # Convert distance to lunar distances
        miss_distance_lunar = self.min_distance_au * 389.17  # AU to LD

        return np.array([
            self.absolute_magnitude / 35.0,
            np.log10(self.diameter_km * 1000 + 1) / 5.0,
            self.velocity_km_s / 30.0,
            np.log10(miss_distance_lunar + 1) / 4.0,
            float(self.is_pha),
            self.eccentricity,
            min(1.0, self.inclination_deg / 90.0),
            min(1.0, -self.palermo_scale / 10.0) if self.palermo_scale < 0 else 1.0,
        ])


@dataclass
class KnownImpactEvent:
    """Known asteroid impact for ground truth"""
    name: str
    date: str
    diameter_m: float
    velocity_km_s: float
    energy_mt: float
    casualties: int
    damage_description: str
    threat_level: ThreatLevel


# Known impact events for ground truth training
KNOWN_IMPACTS = [
    KnownImpactEvent(
        name="Chelyabinsk",
        date="2013-02-15",
        diameter_m=20,
        velocity_km_s=19.16,
        energy_mt=0.44,
        casualties=1500,  # Injuries
        damage_description="Airburst, 7,200 buildings damaged",
        threat_level=ThreatLevel.MODERATE
    ),
    KnownImpactEvent(
        name="Tunguska",
        date="1908-06-30",
        diameter_m=50,
        velocity_km_s=15,
        energy_mt=10,
        casualties=0,
        damage_description="2,150 km2 forest flattened",
        threat_level=ThreatLevel.HIGH
    ),
    KnownImpactEvent(
        name="Chicxulub",
        date="-65000000",  # 65 million years ago
        diameter_m=10000,
        velocity_km_s=20,
        energy_mt=100000000,  # 100 million MT
        casualties=0,  # Dinosaurs
        damage_description="Mass extinction event",
        threat_level=ThreatLevel.CRITICAL
    ),
    KnownImpactEvent(
        name="Barringer Crater",
        date="-50000",  # 50,000 years ago
        diameter_m=50,
        velocity_km_s=12.8,
        energy_mt=10,
        casualties=0,
        damage_description="1.2 km crater in Arizona",
        threat_level=ThreatLevel.HIGH
    ),
    KnownImpactEvent(
        name="2008 TC3",
        date="2008-10-07",
        diameter_m=4,
        velocity_km_s=12.4,
        energy_mt=0.001,
        casualties=0,
        damage_description="First predicted impact, Sudan desert",
        threat_level=ThreatLevel.NEGLIGIBLE
    ),
    KnownImpactEvent(
        name="2014 AA",
        date="2014-01-02",
        diameter_m=3,
        velocity_km_s=12,
        energy_mt=0.0005,
        casualties=0,
        damage_description="Atlantic Ocean airburst",
        threat_level=ThreatLevel.NEGLIGIBLE
    ),
    KnownImpactEvent(
        name="2018 LA",
        date="2018-06-02",
        diameter_m=2,
        velocity_km_s=17,
        energy_mt=0.0003,
        casualties=0,
        damage_description="Botswana airburst",
        threat_level=ThreatLevel.NEGLIGIBLE
    ),
    KnownImpactEvent(
        name="2022 EB5",
        date="2022-03-11",
        diameter_m=2,
        velocity_km_s=18.5,
        energy_mt=0.0002,
        casualties=0,
        damage_description="Norwegian Sea airburst",
        threat_level=ThreatLevel.NEGLIGIBLE
    ),
    KnownImpactEvent(
        name="2023 CX1",
        date="2023-02-13",
        diameter_m=1,
        velocity_km_s=14,
        energy_mt=0.0001,
        casualties=0,
        damage_description="English Channel airburst",
        threat_level=ThreatLevel.NEGLIGIBLE
    ),
    KnownImpactEvent(
        name="2024 BX1",
        date="2024-01-21",
        diameter_m=1,
        velocity_km_s=15,
        energy_mt=0.0001,
        casualties=0,
        damage_description="Berlin, Germany airburst",
        threat_level=ThreatLevel.NEGLIGIBLE
    ),
]


class TrainingDataFetcher:
    """
    Fetches and prepares training data from NASA APIs.

    Data sources:
    1. Sentry API - Objects with computed impact probability
    2. Close Approach API - Historic close approaches
    3. SBDB API - Physical and orbital properties
    4. Known impacts - Ground truth data
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(config.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

        # API URLs from config
        self.sentry_api = config.JPL_SENTRY_API_URL
        self.cad_api = config.JPL_CAD_API_URL
        self.sbdb_api = config.JPL_SBDB_API_URL
        self._request_count = 0
        self._last_request_time = 0

    def _rate_limit(self):
        """Respect NASA API rate limits"""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:  # Max 2 requests/second
            time.sleep(0.5 - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1

    def _get_cached_or_fetch(self, cache_key: str, url: str, params: dict = None) -> dict:
        """Get from cache or fetch from API"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            # Check if cache is less than 24 hours old
            age = time.time() - cache_file.stat().st_mtime
            if age < 86400:  # 24 hours
                with open(cache_file) as f:
                    return json.load(f)

        # Fetch from API
        self._rate_limit()
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)

            return data
        except Exception as e:
            print(f"API error for {cache_key}: {e}")
            return {}

    def fetch_sentry_objects(self, limit: int = 100) -> List[TrainingNEO]:
        """
        Fetch objects from NASA Sentry risk table.
        These are objects with non-zero computed impact probability.

        Args:
            limit: Max number of objects to fetch detailed data for (SBDB is rate-limited)
        """
        print("Fetching Sentry risk table...")

        data = self._get_cached_or_fetch("sentry_all", self.sentry_api, {"all": "1"})

        if "data" not in data:
            print("No Sentry data available")
            return []

        training_neos = []
        count = 0

        for obj in data["data"]:
            if count >= limit:
                break
            try:
                # Sentry API returns dicts with: des, fullname, ip, ps, energy, date, etc.
                des = obj.get("des", "")
                name = obj.get("fullname", des) or des
                ip = float(obj.get("ip", 0))  # Impact probability
                ps = float(obj.get("ps", -10))  # Palermo Scale
                energy = float(obj.get("energy", 0))  # Impact energy in MT

                # Determine threat level from Palermo scale
                if ps > 0:
                    threat = ThreatLevel.CRITICAL
                elif ps > -2:
                    threat = ThreatLevel.HIGH
                elif ps > -4:
                    threat = ThreatLevel.MODERATE
                elif ps > -6:
                    threat = ThreatLevel.LOW
                else:
                    threat = ThreatLevel.NEGLIGIBLE

                # Fetch detailed data from SBDB
                details = self._fetch_sbdb_details(des)

                neo = TrainingNEO(
                    id=des,
                    name=name,
                    diameter_km=details.get("diameter_km", 0.1),
                    absolute_magnitude=details.get("H", 25),
                    semi_major_axis_au=details.get("a", 1.5),
                    eccentricity=details.get("e", 0.3),
                    inclination_deg=details.get("i", 10),
                    min_distance_au=details.get("moid_au", 0.05),
                    velocity_km_s=details.get("v_inf", 15),
                    impact_probability=ip,
                    palermo_scale=ps,
                    kinetic_energy_mt=energy,
                    is_pha=details.get("pha", False),
                    threat_level=threat,
                    label_source="sentry"
                )

                training_neos.append(neo)
                count += 1

            except Exception as e:
                print(f"Error processing Sentry object {obj.get('des', 'unknown')}: {e}")
                continue

        print(f"Fetched {len(training_neos)} Sentry objects (limit: {limit})")
        return training_neos

    def _fetch_sbdb_details(self, designation: str) -> dict:
        """Fetch detailed object data from SBDB"""
        cache_key = f"sbdb_{designation.replace(' ', '_').replace('/', '_')}"

        # Use simple query - phys-par and orbit-par cause 400 errors
        data = self._get_cached_or_fetch(
            cache_key,
            self.sbdb_api,
            {"sstr": designation}
        )

        result = {
            "diameter_km": 0.1,
            "H": 25,
            "a": 1.5,
            "e": 0.3,
            "i": 10,
            "moid_au": 0.05,
            "v_inf": 15,
            "pha": False
        }

        if "object" not in data:
            return result

        obj = data["object"]

        # PHA flag
        if "pha" in obj:
            result["pha"] = obj["pha"] == True or obj["pha"] == "Y"

        # Orbital elements - they come as list of dicts with "name" and "value"
        if "orbit" in data:
            orbit = data["orbit"]

            # MOID
            if "moid" in orbit:
                try:
                    result["moid_au"] = float(orbit["moid"])
                except (ValueError, TypeError):
                    pass

            # Parse elements array
            if "elements" in orbit:
                for elem in orbit["elements"]:
                    name = elem.get("name")
                    value = elem.get("value")
                    if not name or not value:
                        continue
                    try:
                        if name == "a":
                            result["a"] = float(value)
                        elif name == "e":
                            result["e"] = float(value)
                        elif name == "i":
                            result["i"] = float(value)
                    except (ValueError, TypeError):
                        pass

        # Estimate diameter from H magnitude if not available
        # Using: D = 1329 * 10^(-H/5) / sqrt(albedo)
        # Assuming albedo = 0.14 (typical for S-type)
        if result["diameter_km"] == 0.1 and result["H"] != 25:
            try:
                albedo = 0.14
                result["diameter_km"] = 1.329 * (10 ** (-result["H"] / 5)) / np.sqrt(albedo)
            except:
                pass

        return result

    def fetch_historic_approaches(
        self,
        start_date: str = "1900-01-01",
        end_date: str = None,
        max_dist_au: float = 0.05
    ) -> List[TrainingNEO]:
        """
        Fetch historic close approaches.
        Objects that passed close but didn't impact -> NEGLIGIBLE/LOW
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Fetching historic approaches from {start_date} to {end_date}...")

        cache_key = f"cad_{start_date}_{end_date}_{max_dist_au}"

        data = self._get_cached_or_fetch(
            cache_key,
            self.cad_api,
            {
                "date-min": start_date,
                "date-max": end_date,
                "dist-max": str(max_dist_au),
                "body": "Earth",
                "sort": "dist"
            }
        )

        if "data" not in data:
            print("No close approach data available")
            return []

        training_neos = []
        fields = data.get("fields", [])

        # Map field names to indices
        field_map = {name: i for i, name in enumerate(fields)}

        for approach in data["data"][:500]:  # Limit to 500 for training
            try:
                des = approach[field_map.get("des", 0)]

                dist_au = float(approach[field_map.get("dist", 3)])
                v_rel = float(approach[field_map.get("v_rel", 4)])

                # These are historic non-impacts, so low threat
                # But larger/closer ones get higher baseline
                dist_ld = dist_au * 389.17

                if dist_ld < 1:
                    threat = ThreatLevel.LOW
                elif dist_ld < 5:
                    threat = ThreatLevel.NEGLIGIBLE
                else:
                    threat = ThreatLevel.NEGLIGIBLE

                # Get more details
                details = self._fetch_sbdb_details(des)

                neo = TrainingNEO(
                    id=des,
                    name=des,
                    diameter_km=details.get("diameter_km", 0.01),
                    absolute_magnitude=details.get("H", 28),
                    semi_major_axis_au=details.get("a", 1.5),
                    eccentricity=details.get("e", 0.3),
                    inclination_deg=details.get("i", 10),
                    min_distance_au=dist_au,
                    velocity_km_s=v_rel,
                    approach_date=approach[field_map.get("cd", 1)],
                    is_pha=details.get("pha", False),
                    threat_level=threat,
                    label_source="historic"
                )

                # Upgrade threat if it's a PHA
                if neo.is_pha and neo.threat_level == ThreatLevel.NEGLIGIBLE:
                    neo.threat_level = ThreatLevel.LOW

                training_neos.append(neo)

            except Exception as e:
                print(f"Error processing approach: {e}")
                continue

        print(f"Fetched {len(training_neos)} historic approaches")
        return training_neos

    def fetch_known_phas(self, limit: int = 200) -> List[TrainingNEO]:
        """
        Fetch known Potentially Hazardous Asteroids.
        These have known orbits that could pose future risk.
        """
        print(f"Fetching known PHAs (limit {limit})...")

        # Use CAD for future close approaches of PHAs
        end_date = (datetime.now() + timedelta(days=365*100)).strftime("%Y-%m-%d")

        data = self._get_cached_or_fetch(
            "future_phas",
            self.cad_api,
            {
                "date-min": datetime.now().strftime("%Y-%m-%d"),
                "date-max": end_date,
                "dist-max": "0.05",  # 0.05 AU = ~20 LD
                "body": "Earth",
                "sort": "dist",
                "limit": str(limit)
            }
        )

        if "data" not in data:
            return []

        training_neos = []
        fields = data.get("fields", [])
        field_map = {name: i for i, name in enumerate(fields)}

        for approach in data["data"]:
            try:
                des = approach[field_map.get("des", 0)]
                dist_au = float(approach[field_map.get("dist", 3)])
                v_rel = float(approach[field_map.get("v_rel", 4)])

                details = self._fetch_sbdb_details(des)
                diameter = details.get("diameter_km", 0.1)

                # Threat level based on size and distance
                dist_ld = dist_au * 389.17

                if diameter > 1 and dist_ld < 10:
                    threat = ThreatLevel.HIGH
                elif diameter > 0.14 and dist_ld < 20:
                    threat = ThreatLevel.MODERATE
                elif details.get("pha", False):
                    threat = ThreatLevel.LOW
                else:
                    threat = ThreatLevel.NEGLIGIBLE

                neo = TrainingNEO(
                    id=des,
                    name=des,
                    diameter_km=diameter,
                    absolute_magnitude=details.get("H", 22),
                    semi_major_axis_au=details.get("a", 1.5),
                    eccentricity=details.get("e", 0.3),
                    inclination_deg=details.get("i", 10),
                    min_distance_au=dist_au,
                    velocity_km_s=v_rel,
                    approach_date=approach[field_map.get("cd", 1)],
                    is_pha=details.get("pha", False),
                    threat_level=threat,
                    label_source="pha"
                )

                training_neos.append(neo)

            except Exception as e:
                continue

        print(f"Fetched {len(training_neos)} PHAs")
        return training_neos

    def generate_impact_training_data(self) -> List[TrainingNEO]:
        """
        Generate training data from known impact events.
        These are the ground truth for HIGH/CRITICAL cases.
        """
        print("Generating training data from known impacts...")

        training_neos = []

        for impact in KNOWN_IMPACTS:
            # Create synthetic pre-impact conditions
            neo = TrainingNEO(
                id=f"impact_{impact.name.lower()}",
                name=impact.name,
                diameter_km=impact.diameter_m / 1000,
                absolute_magnitude=self._diameter_to_H(impact.diameter_m / 1000),
                min_distance_au=0.0,  # It hit!
                velocity_km_s=impact.velocity_km_s,
                approach_date=impact.date,
                kinetic_energy_mt=impact.energy_mt,
                impact_probability=1.0,
                palermo_scale=0,  # Actual impact
                torino_scale=10 if impact.threat_level == ThreatLevel.CRITICAL else 8,
                threat_level=impact.threat_level,
                label_source="impact"
            )

            training_neos.append(neo)

            # Also create "near miss" variants for training diversity
            for miss_factor in [0.5, 1, 2, 5, 10]:
                miss_neo = TrainingNEO(
                    id=f"nearmiss_{impact.name.lower()}_{miss_factor}",
                    name=f"{impact.name} (miss variant)",
                    diameter_km=impact.diameter_m / 1000,
                    absolute_magnitude=self._diameter_to_H(impact.diameter_m / 1000),
                    min_distance_au=miss_factor * 0.00257 / 389.17,  # miss_factor lunar distances
                    velocity_km_s=impact.velocity_km_s,
                    kinetic_energy_mt=impact.energy_mt,
                    impact_probability=0.0,
                    threat_level=self._compute_near_miss_threat(impact, miss_factor),
                    label_source="impact_variant"
                )
                training_neos.append(neo)

        print(f"Generated {len(training_neos)} impact-based training samples")
        return training_neos

    def _diameter_to_H(self, diameter_km: float, albedo: float = 0.14) -> float:
        """Convert diameter to absolute magnitude"""
        # H = 15.618 - 5*log10(D) - 2.5*log10(albedo)
        return 15.618 - 5 * np.log10(diameter_km) - 2.5 * np.log10(albedo)

    def _compute_near_miss_threat(self, impact: KnownImpactEvent, miss_factor: float) -> ThreatLevel:
        """Compute threat level for near-miss variant"""
        if miss_factor <= 1 and impact.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            return ThreatLevel.HIGH
        elif miss_factor <= 5 and impact.threat_level == ThreatLevel.CRITICAL:
            return ThreatLevel.MODERATE
        elif impact.threat_level == ThreatLevel.CRITICAL:
            return ThreatLevel.LOW
        elif miss_factor <= 2:
            return impact.threat_level
        else:
            return ThreatLevel(max(0, impact.threat_level.value - 1))

    def fetch_all_training_data(self) -> List[Tuple[np.ndarray, ThreatLevel]]:
        """
        Fetch all training data and convert to feature vectors.

        Returns list of (feature_vector, label) tuples.
        """
        all_neos = []

        # 1. Sentry objects (highest priority - real risk data)
        all_neos.extend(self.fetch_sentry_objects())

        # 2. Known impacts (ground truth)
        all_neos.extend(self.generate_impact_training_data())

        # 3. Historic approaches (negative examples)
        all_neos.extend(self.fetch_historic_approaches())

        # 4. Known PHAs (moderate risk examples)
        all_neos.extend(self.fetch_known_phas())

        print(f"\nTotal training samples: {len(all_neos)}")

        # Convert to training format
        training_data = []
        for neo in all_neos:
            try:
                features = neo.to_feature_vector()
                training_data.append((features, neo.threat_level))
            except Exception as e:
                print(f"Error converting {neo.name}: {e}")
                continue

        # Print distribution
        distribution = {}
        for _, label in training_data:
            distribution[label.name] = distribution.get(label.name, 0) + 1

        print("\nLabel distribution:")
        for level, count in sorted(distribution.items()):
            print(f"  {level}: {count}")

        return training_data

    def save_training_data(self, filename: str = "neo_training_data.json"):
        """Save training data to JSON for reuse"""
        training_data = self.fetch_all_training_data()

        # Convert to serializable format
        data = [
            {
                "features": features.tolist(),
                "label": label.value,
                "label_name": label.name
            }
            for features, label in training_data
        ]

        filepath = self.cache_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved {len(data)} training samples to {filepath}")
        return filepath

    def load_training_data(self, filename: str = "neo_training_data.json") -> List[Tuple[np.ndarray, ThreatLevel]]:
        """Load training data from JSON"""
        filepath = self.cache_dir / filename

        if not filepath.exists():
            print(f"Training data not found at {filepath}, fetching...")
            self.save_training_data(filename)

        with open(filepath) as f:
            data = json.load(f)

        return [
            (np.array(item["features"]), ThreatLevel(item["label"]))
            for item in data
        ]


def fetch_and_prepare_training_data(cache_dir: str = None) -> List[Tuple[np.ndarray, ThreatLevel]]:
    """Convenience function to fetch all training data"""
    fetcher = TrainingDataFetcher(cache_dir)
    return fetcher.fetch_all_training_data()


if __name__ == "__main__":
    print("=" * 60)
    print("NEO Training Data Fetcher")
    print("=" * 60)

    fetcher = TrainingDataFetcher()

    # Fetch and save all training data
    filepath = fetcher.save_training_data()

    # Load and verify
    training_data = fetcher.load_training_data()
    print(f"\nLoaded {len(training_data)} training samples")

    # Show some examples
    print("\nSample training data:")
    for i, (features, label) in enumerate(training_data[:5]):
        print(f"  {i+1}. Label: {label.name}, Features: {features[:4]}...")

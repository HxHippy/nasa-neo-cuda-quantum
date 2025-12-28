#!/usr/bin/env python3
"""
NEO Quantum - Near Earth Object Analysis with Quantum Computing

A CUDA-Q powered application for analyzing asteroid threats using
quantum risk classification and orbital dynamics simulation.

Features:
- Real-time NEO data from NASA NeoWs API
- Variational Quantum Classifier for threat assessment
- Quantum Hamiltonian simulation for orbital evolution
- Collision probability estimation

Requires: NVIDIA GPU with CUDA support
"""
import os
import cudaq
import numpy as np
import argparse
import sys
from datetime import datetime

from neo_fetcher import NEOFetcher, NEO
from quantum_risk_classifier import QuantumRiskClassifier, ThreatLevel
from orbital_simulator import OrbitalSimulator


# ANSI colors for terminal output
class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def threat_color(level: ThreatLevel) -> str:
    """Get color for threat level"""
    colors = {
        ThreatLevel.NEGLIGIBLE: Colors.GREEN,
        ThreatLevel.LOW: Colors.CYAN,
        ThreatLevel.MODERATE: Colors.YELLOW,
        ThreatLevel.HIGH: Colors.RED,
        ThreatLevel.CRITICAL: Colors.RED + Colors.BOLD,
    }
    return colors.get(level, '')


def print_header():
    """Print application header"""
    print(f"""
{Colors.CYAN}{'=' * 70}
{Colors.BOLD}    NEO QUANTUM - Near Earth Object Analysis System
{Colors.END}{Colors.CYAN}    Powered by NVIDIA CUDA-Q Quantum Computing
{'=' * 70}{Colors.END}
""")


def print_quantum_info():
    """Print quantum backend information"""
    target = cudaq.get_target()
    print(f"{Colors.BLUE}Quantum Backend:{Colors.END} {target.name}")
    print(f"{Colors.BLUE}Date:{Colors.END} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def analyze_neos(neos: list[NEO], verbose: bool = False):
    """Perform full quantum analysis on NEOs"""
    print(f"\n{Colors.BOLD}Processing {len(neos)} Near Earth Objects...{Colors.END}\n")

    # Initialize quantum components
    classifier = QuantumRiskClassifier(num_qubits=4, num_layers=3)
    simulator = OrbitalSimulator()

    results = []

    for i, neo in enumerate(neos):
        if verbose:
            print(f"  [{i+1}/{len(neos)}] Analyzing {neo.name[:40]}...")

        # Quantum risk classification
        risk = classifier.classify(neo, shots=500)

        # Collision probability from quantum simulation
        collision_prob = simulator.estimate_collision_probability(neo, shots=1000)

        results.append({
            'neo': neo,
            'risk': risk,
            'collision_prob': collision_prob,
        })

    return results


def display_results(results: list[dict], top_n: int = 15):
    """Display analysis results"""
    # Sort by threat level, then collision probability
    results.sort(key=lambda x: (x['risk'].threat_level.value, x['collision_prob']), reverse=True)

    print(f"\n{Colors.BOLD}{'=' * 70}")
    print(f" THREAT ASSESSMENT RESULTS (Top {min(top_n, len(results))})")
    print(f"{'=' * 70}{Colors.END}\n")

    for i, r in enumerate(results[:top_n]):
        neo = r['neo']
        risk = r['risk']
        col_prob = r['collision_prob']

        color = threat_color(risk.threat_level)
        hazard_flag = " [PHA]" if neo.is_hazardous else ""

        print(f"{color}{Colors.BOLD}{i+1}. {neo.name}{hazard_flag}{Colors.END}")
        print(f"   Threat Level: {color}{risk.threat_level.name}{Colors.END}")
        print(f"   Diameter: {neo.avg_diameter_km*1000:.0f} m ({neo.diameter_min_km*1000:.0f}-{neo.diameter_max_km*1000:.0f} m)")
        print(f"   Velocity: {neo.velocity_km_s:.2f} km/s")
        print(f"   Miss Distance: {neo.miss_distance_lunar:.1f} lunar distances ({neo.miss_distance_km:,.0f} km)")
        print(f"   Kinetic Energy: {neo.kinetic_energy_estimate:.2f} MT TNT")
        print(f"   Collision Probability: {col_prob:.2e}")

        # Probability distribution bar
        print(f"   Risk Distribution: ", end="")
        for level in ThreatLevel:
            prob = risk.probability_distribution[level]
            if prob > 0.05:
                c = threat_color(level)
                bars = int(prob * 20)
                print(f"{c}{'|' * bars}{Colors.END}", end="")
        print()

        print()


def display_summary(results: list[dict]):
    """Display summary statistics"""
    print(f"\n{Colors.BOLD}{'=' * 70}")
    print(f" SUMMARY")
    print(f"{'=' * 70}{Colors.END}\n")

    # Count by threat level
    threat_counts = {level: 0 for level in ThreatLevel}
    for r in results:
        threat_counts[r['risk'].threat_level] += 1

    print("Threat Level Distribution:")
    for level in ThreatLevel:
        count = threat_counts[level]
        color = threat_color(level)
        bar = '#' * count
        print(f"  {color}{level.name:12}{Colors.END} {bar} ({count})")

    # PHAs
    pha_count = sum(1 for r in results if r['neo'].is_hazardous)
    print(f"\n{Colors.RED}Potentially Hazardous Asteroids (PHA): {pha_count}{Colors.END}")

    # Highest energy
    max_energy = max(results, key=lambda x: x['neo'].kinetic_energy_estimate)
    print(f"\nHighest Kinetic Energy: {max_energy['neo'].name}")
    print(f"  {max_energy['neo'].kinetic_energy_estimate:,.0f} MT TNT equivalent")

    # Closest approach
    closest = min(results, key=lambda x: x['neo'].miss_distance_km)
    print(f"\nClosest Approach: {closest['neo'].name}")
    print(f"  {closest['neo'].miss_distance_lunar:.1f} lunar distances on {closest['neo'].close_approach_date}")


def detailed_analysis(neo: NEO):
    """Perform detailed analysis on a single NEO"""
    print(f"\n{Colors.BOLD}{'=' * 70}")
    print(f" DETAILED ANALYSIS: {neo.name}")
    print(f"{'=' * 70}{Colors.END}\n")

    classifier = QuantumRiskClassifier(num_qubits=4, num_layers=3)
    simulator = OrbitalSimulator()

    # Risk classification with more shots
    print("Running quantum risk classification (2000 shots)...")
    risk = classifier.classify(neo, shots=2000)

    # Orbital evolution
    print("Running quantum orbital evolution simulation...")
    orbital_state = simulator.simulate_orbital_evolution(neo, num_steps=10, shots=1000)

    print(f"\n{Colors.CYAN}Physical Properties:{Colors.END}")
    print(f"  Absolute Magnitude: {neo.absolute_magnitude}")
    print(f"  Estimated Diameter: {neo.avg_diameter_km*1000:.0f} m")
    print(f"  Relative Velocity: {neo.velocity_km_s:.2f} km/s")
    print(f"  Close Approach: {neo.close_approach_date}")
    print(f"  Miss Distance: {neo.miss_distance_km:,.0f} km ({neo.miss_distance_lunar:.1f} LD)")

    print(f"\n{Colors.CYAN}Threat Assessment:{Colors.END}")
    color = threat_color(risk.threat_level)
    print(f"  {Colors.BOLD}Threat Level: {color}{risk.threat_level.name}{Colors.END}")
    print(f"  Probability Distribution:")
    for level, prob in risk.probability_distribution.items():
        c = threat_color(level)
        bars = int(prob * 30)
        print(f"    {c}{level.name:12}{Colors.END} {'#' * bars} {prob:.1%}")

    print(f"\n{Colors.CYAN}Impact Analysis:{Colors.END}")
    print(f"  Kinetic Energy: {neo.kinetic_energy_estimate:,.2f} MT TNT")

    # Impact comparison
    if neo.kinetic_energy_estimate < 0.01:
        print(f"  Equivalent to: Small meteor (burns up in atmosphere)")
    elif neo.kinetic_energy_estimate < 1:
        print(f"  Equivalent to: Large fireball/meteorite")
    elif neo.kinetic_energy_estimate < 100:
        print(f"  Equivalent to: Chelyabinsk-class event ({neo.kinetic_energy_estimate:.0f}x)")
    elif neo.kinetic_energy_estimate < 10000:
        print(f"  Equivalent to: Tunguska-class event ({neo.kinetic_energy_estimate/10:.0f}x)")
    else:
        print(f"  Equivalent to: {neo.kinetic_energy_estimate/65000:.1f}x Chicxulub (dinosaur extinction)")

    print(f"\n{Colors.CYAN}Orbital Simulation Results:{Colors.END}")
    print(f"  Collision Probability: {orbital_state.collision_probability:.2e}")
    print(f"  Position Uncertainty: {orbital_state.position_uncertainty:,.0f} km")
    print(f"  Velocity Uncertainty: {orbital_state.velocity_uncertainty:.3f} km/s")

    print(f"\n  Evolution over ~1 year:")
    for t, unc in zip(orbital_state.time_steps[:5], orbital_state.evolution_states[:5]):
        print(f"    Day {t:5.0f}: Position uncertainty = {unc:,.0f} km")


def main():
    parser = argparse.ArgumentParser(
        description="NEO Quantum - Asteroid threat analysis with quantum computing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Analyze upcoming NEOs (next 7 days)
  python main.py --days 14           # Analyze next 14 days
  python main.py --detail "164400"   # Detailed analysis of specific NEO
  python main.py --verbose           # Show progress during analysis
        """
    )

    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to look ahead (default: 7)')
    parser.add_argument('--top', type=int, default=15,
                       help='Number of top threats to display (default: 15)')
    parser.add_argument('--detail', type=str,
                       help='Show detailed analysis for NEO ID or name')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--api-key', type=str,
                       default=os.getenv("NASA_API_KEY", "DEMO_KEY"),
                       help='NASA API key (default: from NASA_API_KEY env var or DEMO_KEY)')

    args = parser.parse_args()

    print_header()
    print_quantum_info()

    # Fetch NEO data
    print(f"\n{Colors.CYAN}Fetching NEO data from NASA...{Colors.END}")
    fetcher = NEOFetcher(args.api_key)

    try:
        neos = fetcher.fetch_upcoming(args.days)
    except Exception as e:
        print(f"{Colors.RED}Error fetching NEO data: {e}{Colors.END}")
        sys.exit(1)

    print(f"Found {len(neos)} near-Earth objects in the next {args.days} days")

    if args.detail:
        # Find specific NEO
        target_neo = None
        for neo in neos:
            if args.detail.lower() in neo.id.lower() or args.detail.lower() in neo.name.lower():
                target_neo = neo
                break

        if target_neo:
            detailed_analysis(target_neo)
        else:
            print(f"{Colors.RED}NEO '{args.detail}' not found in upcoming objects{Colors.END}")
            print("Available NEOs:")
            for neo in neos[:10]:
                print(f"  {neo.id}: {neo.name}")
    else:
        # Full analysis
        results = analyze_neos(neos, verbose=args.verbose)
        display_results(results, top_n=args.top)
        display_summary(results)

    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}Analysis complete.{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}\n")


if __name__ == "__main__":
    main()

/**
 * TypeScript types for NEO Quantum
 */

export enum ThreatLevel {
  NEGLIGIBLE = 'NEGLIGIBLE',
  LOW = 'LOW',
  MODERATE = 'MODERATE',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL',
}

export interface NEOData {
  id: string;
  name: string;
  absolute_magnitude: number;
  diameter_min_km: number;
  diameter_max_km: number;
  avg_diameter_km: number;
  is_hazardous: boolean;
  close_approach_date: string;
  velocity_km_s: number;
  miss_distance_km: number;
  miss_distance_lunar: number;
  kinetic_energy_mt: number;
  position_x: number;
  position_y: number;
  position_z: number;
}

export interface RiskData {
  neo_id: string;
  threat_level: ThreatLevel;
  probabilities: Record<string, number>;
}

export interface OrbitalData {
  neo_id: string;
  collision_probability: number;
  position_uncertainty_km: number;
  velocity_uncertainty_km_s: number;
  evolution_uncertainties: number[];
  time_steps_days: number[];
}

export interface AnalysisStatus {
  current: number;
  total: number;
  current_neo: string | null;
  phase: 'fetching' | 'classifying' | 'simulating' | 'complete';
}

export type WSMessageType =
  | 'neo_list'
  | 'risk_update'
  | 'orbital_update'
  | 'analysis_progress'
  | 'analysis_complete'
  | 'ai_insight'
  | 'ai_fleet_analysis'
  | 'what_if_result'
  | 'deflection_result'
  | 'intercept_mission'
  | 'error'
  | 'start_analysis'
  | 'request_neo_details'
  | 'request_ai_insight'
  | 'request_fleet_analysis'
  | 'request_what_if'
  | 'request_deflection'
  | 'request_intercept'
  | 'cancel_analysis';

export interface AIInsight {
  neo_id: string;
  summary: string;
  threat_assessment: string;
  historical_context: string;
  recommendations: string[];
  fun_facts: string[];
  confidence: number;
}

export interface FleetAnalysis {
  total_analyzed: number;
  executive_summary: string;
  highest_priority_threats: string[];
  patterns_detected: string[];
  weekly_outlook: string;
  recommendations: string[];
}

export interface WSMessage {
  type: WSMessageType;
  data?: unknown;
  timestamp?: string;
}

// Combined NEO data with all analysis results
export interface NEOWithAnalysis extends NEOData {
  risk?: RiskData;
  orbital?: OrbitalData;
}

// Threat level colors for Three.js
export const THREAT_COLORS: Record<ThreatLevel, number> = {
  [ThreatLevel.NEGLIGIBLE]: 0x00ff00,
  [ThreatLevel.LOW]: 0x00ffff,
  [ThreatLevel.MODERATE]: 0xffff00,
  [ThreatLevel.HIGH]: 0xff6600,
  [ThreatLevel.CRITICAL]: 0xff0000,
};

// Get color for threat level
export function getThreatColor(level: ThreatLevel): number {
  return THREAT_COLORS[level] ?? 0xffffff;
}

// Simulation types
export type ScenarioType = 'nominal' | 'close_approach' | 'collision_course' | 'deflected' | 'keyhole';

export interface SimulationResult {
  scenario: ScenarioType;
  min_earth_distance: number;  // lunar distances
  min_distance_time: number;  // days
  collision_probability: number;
  impact_energy_mt: number | null;
  quantum_uncertainty: number;
  confidence: number;
}

export interface WhatIfResult {
  neo_id: string;
  scenarios: Record<ScenarioType, SimulationResult>;
}

export interface DeflectionResult {
  neo_id: string;
  optimal_delta_v: number;  // m/s
  optimal_direction: [number, number, number];
  deflection_time: number;  // days before approach
  miss_distance_achieved: number;  // lunar distances
  energy_required_mt: number;
  success_probability: number;
  quantum_iterations: number;
}

export interface LaunchWindow {
  window_open: number;  // days from now
  window_close: number;
  optimal_launch: number;
  c3_energy: number;  // km^2/s^2
  flight_time: number;  // days
  arrival_velocity: number;  // km/s
}

export interface MissionPhase {
  name: string;
  duration: number;  // days
  delta_v: number;  // km/s
  description: string;
}

export interface InterceptMission {
  neo_id: string;
  mission_type: 'kinetic_impactor' | 'gravity_tractor' | 'nuclear_standoff';
  launch_windows: LaunchWindow[];
  recommended_window: LaunchWindow;
  mission_phases: MissionPhase[];
  total_delta_v: number;  // km/s
  mission_duration: number;  // days
  intercept_date: number;  // days from now
  success_probability: number;
}

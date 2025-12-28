/**
 * NEO Quantum - Main Application Entry Point
 *
 * Three.js visualization of Near Earth Objects with real-time
 * quantum analysis via WebSocket.
 */

import * as THREE from 'three';
import { SceneManager, ViewMode } from './scene/SceneManager';
import { Earth } from './scene/Earth';
import { Moon } from './scene/Moon';
import { Sun } from './scene/Sun';
import { Planets } from './scene/Planets';
import { Starfield } from './scene/Starfield';
import { NEOObject } from './scene/NEOObject';
import { DistanceRings } from './scene/DistanceRings';
import { NEODataStream } from './websocket/NEODataStream';
import type {
  NEOData, RiskData, OrbitalData, NEOWithAnalysis, ThreatLevel, AIInsight, FleetAnalysis,
  WhatIfResult, DeflectionResult, InterceptMission
} from './types/neo';
import {
  DisplayMode,
  formatDistance,
  formatVelocity,
  formatDiameter,
  formatEnergy,
  formatProbability,
  formatThreatLevel,
  formatDate,
  getLegendItems
} from './ui/DisplayMode';

class NEOQuantumApp {
  private sceneManager: SceneManager;
  private dataStream: NEODataStream;

  // Scene objects
  private earth: Earth;
  private moon: Moon;
  private sun: Sun;
  private planets: Planets;
  private starfield: Starfield;
  private distanceRings: DistanceRings;


  // NEO objects
  private neoObjects: Map<string, NEOObject> = new Map();
  private neoData: Map<string, NEOWithAnalysis> = new Map();
  private neoOriginalPositions: Map<string, THREE.Vector3> = new Map(); // Earth-centric positions

  // UI elements
  private statusEl: HTMLElement;
  private infoEl: HTMLElement;
  private progressEl: HTMLElement;
  private progressTextEl: HTMLElement;
  private progressFillEl: HTMLElement;
  private aiPanelEl: HTMLElement;
  private aiContentEl: HTMLElement;
  private aboutPanelEl: HTMLElement;
  private aboutContentEl: HTMLElement;
  private legendEl: HTMLElement;

  // State
  private selectedNEO: NEOObject | null = null;
  private currentView: ViewMode = 'earth';
  private displayMode: DisplayMode = 'normal';

  // Playback controls
  private isPaused: boolean = false;
  private timeScale: number = 1.0;
  private hoverPauseEnabled: boolean = true;
  private isHovering: boolean = false;
  private readonly TIME_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

  // Visual options
  private backdropEnabled: boolean = false;
  private starfieldEnabled: boolean = true;

  // Risk level filters
  private riskFilters: Record<string, boolean> = {
    CRITICAL: true,
    HIGH: true,
    MODERATE: true,
    LOW: true,
    NEGLIGIBLE: true,
  };

  constructor() {
    // Get container
    const container = document.getElementById('canvas-container');
    if (!container) throw new Error('Canvas container not found');

    // Initialize scene
    this.sceneManager = new SceneManager(container);

    // Initialize WebSocket (URL from VITE_WEBSOCKET_URL env var or default)
    this.dataStream = new NEODataStream();

    // Get UI elements
    this.statusEl = document.getElementById('status')!;
    this.infoEl = document.getElementById('neo-info')!;
    this.progressEl = document.getElementById('progress')!;
    this.progressTextEl = document.getElementById('progress-text')!;
    this.progressFillEl = document.getElementById('progress-fill')!;
    this.aiPanelEl = document.getElementById('ai-panel')!;
    this.aiContentEl = document.getElementById('ai-content')!;
    this.aboutPanelEl = document.getElementById('about-panel')!;
    this.aboutContentEl = document.getElementById('about-content')!;
    this.legendEl = document.getElementById('legend')!;

    // Initialize legend
    this.updateLegend();

    // Create scene objects
    this.earth = new Earth();
    this.moon = new Moon();
    this.sun = new Sun();
    this.planets = new Planets(this.sun.position);
    this.starfield = new Starfield(5000);
    this.distanceRings = new DistanceRings();

    // Add to scene
    this.sceneManager.scene.add(this.earth);
    this.sceneManager.scene.add(this.moon);
    this.sceneManager.scene.add(this.sun);
    this.sceneManager.scene.add(this.planets);
    this.sceneManager.scene.add(this.starfield);
    this.sceneManager.scene.add(this.distanceRings);

    // Link Moon to Planets for orbital following (Earth in Planets orbits in solar view)
    this.moon.setPlanetsRef(this.planets);

    // Setup callbacks
    this.setupSceneCallbacks();
    this.setupUICallbacks();
    this.setupWebSocket();

    // Hide loading screen
    document.getElementById('loading')?.classList.add('hidden');

    // Start rendering
    this.sceneManager.start();
  }

  private setupSceneCallbacks(): void {
    // Animation update
    this.sceneManager.onRender((delta) => {
      // Apply pause and time scale
      const effectivePaused = this.isPaused || (this.hoverPauseEnabled && this.isHovering);
      const scaledDelta = effectivePaused ? 0 : delta * this.timeScale;

      // Pass current view mode directly to Earth
      const isSolar = this.currentView === 'solar';
      this.earth.update(scaledDelta, isSolar);
      this.moon.update(scaledDelta);
      this.sun.update(scaledDelta);
      this.planets.update(scaledDelta);

      for (const neo of this.neoObjects.values()) {
        neo.update(scaledDelta, this.sceneManager.camera);
      }

      // Update NEO positions based on view mode
      if (isSolar) {
        // In solar view, textured Earth and NEOs follow the Planets group's Earth orbit
        const earthPos = this.planets.getEarthPosition();

        // Sync textured Earth to orbiting position
        this.earth.position.copy(earthPos);

        // NEOs follow Earth's orbital position
        for (const [neoId, neo] of this.neoObjects) {
          const originalPos = this.neoOriginalPositions.get(neoId);
          if (originalPos) {
            // Add Earth's orbital position to the NEO's Earth-centric position
            neo.position.set(
              earthPos.x + originalPos.x,
              earthPos.y + originalPos.y,
              earthPos.z + originalPos.z
            );
          }
        }
      } else {
        // In earth view, Earth at origin, NEOs at their original Earth-centric positions
        this.earth.position.set(0, 0, 0);

        for (const [neoId, neo] of this.neoObjects) {
          const originalPos = this.neoOriginalPositions.get(neoId);
          if (originalPos) {
            neo.position.copy(originalPos);
          }
        }
      }

    });

    // Object click - select and focus
    this.sceneManager.onObjectClick((object) => {
      if (object && object.userData.neoId) {
        this.selectNEO(object.userData.neoId);
        // Focus camera on the selected NEO
        const neo = this.neoObjects.get(object.userData.neoId);
        if (neo) {
          const worldPos = new THREE.Vector3();
          neo.getWorldPosition(worldPos);
          this.focusOnObject(worldPos, 5);
        }
      } else {
        this.deselectNEO();
      }
    });
  }

  private setupUICallbacks(): void {
    // View toggle buttons
    document.getElementById('btn-earth-view')?.addEventListener('click', () => {
      this.setView('earth');
    });

    document.getElementById('btn-solar-view')?.addEventListener('click', () => {
      this.setView('solar');
    });

    // Analyze button
    document.getElementById('btn-analyze')?.addEventListener('click', () => {
      this.startAnalysis();
    });

    // AI Report button
    document.getElementById('btn-ai-report')?.addEventListener('click', () => {
      this.requestFleetAnalysis();
    });

    // AI Panel close button
    document.getElementById('ai-panel-close')?.addEventListener('click', () => {
      this.hideAIPanel();
    });

    // Display mode toggle
    document.getElementById('btn-normal-mode')?.addEventListener('click', () => {
      this.setDisplayMode('normal');
    });

    document.getElementById('btn-scientific-mode')?.addEventListener('click', () => {
      this.setDisplayMode('scientific');
    });

    // About panel
    document.getElementById('btn-about')?.addEventListener('click', () => {
      this.showAboutPanel();
    });

    document.getElementById('about-panel-close')?.addEventListener('click', () => {
      this.hideAboutPanel();
    });

    // Playback controls
    document.getElementById('btn-play-pause')?.addEventListener('click', () => {
      this.togglePlayPause();
    });

    document.getElementById('btn-speed-down')?.addEventListener('click', () => {
      this.decreaseSpeed();
    });

    document.getElementById('btn-speed-up')?.addEventListener('click', () => {
      this.increaseSpeed();
    });

    document.getElementById('btn-hover-pause')?.addEventListener('click', () => {
      this.toggleHoverPause();
    });

    document.getElementById('btn-backdrop')?.addEventListener('click', () => {
      this.toggleBackdrop();
    });

    document.getElementById('btn-starfield')?.addEventListener('click', () => {
      this.toggleStarfield();
    });

    document.getElementById('btn-reset-view')?.addEventListener('click', () => {
      this.resetView();
    });

    // Track mouse hover on canvas for hover-pause
    const canvas = this.sceneManager.renderer.domElement;
    canvas.addEventListener('mouseenter', () => {
      this.isHovering = true;
    });
    canvas.addEventListener('mouseleave', () => {
      this.isHovering = false;
    });

    // Risk level filters
    document.getElementById('filter-critical')?.addEventListener('change', (e) => {
      this.riskFilters.CRITICAL = (e.target as HTMLInputElement).checked;
      this.applyFilters();
    });
    document.getElementById('filter-high')?.addEventListener('change', (e) => {
      this.riskFilters.HIGH = (e.target as HTMLInputElement).checked;
      this.applyFilters();
    });
    document.getElementById('filter-moderate')?.addEventListener('change', (e) => {
      this.riskFilters.MODERATE = (e.target as HTMLInputElement).checked;
      this.applyFilters();
    });
    document.getElementById('filter-low')?.addEventListener('change', (e) => {
      this.riskFilters.LOW = (e.target as HTMLInputElement).checked;
      this.applyFilters();
    });
    document.getElementById('filter-negligible')?.addEventListener('change', (e) => {
      this.riskFilters.NEGLIGIBLE = (e.target as HTMLInputElement).checked;
      this.applyFilters();
    });

    // NEO dropdown selector
    document.getElementById('neo-dropdown')?.addEventListener('change', (e) => {
      const neoId = (e.target as HTMLSelectElement).value;
      if (neoId) {
        this.focusOnNEO(neoId);
      }
    });
  }

  private setDisplayMode(mode: DisplayMode): void {
    this.displayMode = mode;

    // Update button states
    document.getElementById('btn-normal-mode')?.classList.toggle('active', mode === 'normal');
    document.getElementById('btn-scientific-mode')?.classList.toggle('active', mode === 'scientific');

    // Update legend
    this.updateLegend();

    // Refresh info panel if NEO is selected
    if (this.selectedNEO) {
      this.updateInfoPanel();
    }
  }

  // Playback control methods
  private togglePlayPause(): void {
    this.isPaused = !this.isPaused;
    this.updatePlaybackUI();
  }

  private increaseSpeed(): void {
    const currentIndex = this.TIME_SCALES.indexOf(this.timeScale);
    if (currentIndex < this.TIME_SCALES.length - 1) {
      this.timeScale = this.TIME_SCALES[currentIndex + 1];
      this.updatePlaybackUI();
    }
  }

  private decreaseSpeed(): void {
    const currentIndex = this.TIME_SCALES.indexOf(this.timeScale);
    if (currentIndex > 0) {
      this.timeScale = this.TIME_SCALES[currentIndex - 1];
      this.updatePlaybackUI();
    }
  }

  private toggleHoverPause(): void {
    this.hoverPauseEnabled = !this.hoverPauseEnabled;
    this.updatePlaybackUI();
  }

  private updatePlaybackUI(): void {
    const playPauseBtn = document.getElementById('btn-play-pause');
    const speedDisplay = document.getElementById('speed-display');
    const hoverPauseBtn = document.getElementById('btn-hover-pause');

    if (playPauseBtn) {
      playPauseBtn.textContent = this.isPaused ? 'Play' : 'Pause';
      playPauseBtn.classList.toggle('paused', this.isPaused);
    }

    if (speedDisplay) {
      speedDisplay.textContent = `${this.timeScale}x`;
    }

    if (hoverPauseBtn) {
      hoverPauseBtn.classList.toggle('active', this.hoverPauseEnabled);
    }
  }

  private toggleBackdrop(): void {
    this.backdropEnabled = !this.backdropEnabled;
    for (const neo of this.neoObjects.values()) {
      neo.setBackdropVisible(this.backdropEnabled);
    }
    const btn = document.getElementById('btn-backdrop');
    if (btn) {
      btn.classList.toggle('active', this.backdropEnabled);
    }
  }

  private toggleStarfield(): void {
    this.starfieldEnabled = !this.starfieldEnabled;
    this.starfield.visible = this.starfieldEnabled;
    const btn = document.getElementById('btn-starfield');
    if (btn) {
      btn.classList.toggle('active', this.starfieldEnabled);
    }
  }

  private focusOnObject(position: THREE.Vector3, distance: number = 20): void {
    // Animate camera to focus on a specific position
    const targetPos = new THREE.Vector3(
      position.x,
      position.y + distance * 0.3,
      position.z + distance
    );
    this.sceneManager.animateCameraToTarget(targetPos, position);
  }

  private resetView(): void {
    this.setView(this.currentView);
  }

  private applyFilters(): void {
    // Show/hide NEOs based on risk level filters
    for (const [id, neo] of this.neoObjects) {
      const data = this.neoData.get(id);
      const threatLevel = data?.risk?.threat_level || 'NEGLIGIBLE';
      const shouldShow = this.riskFilters[threatLevel] ?? true;
      neo.visible = shouldShow;
    }
    // Update dropdown to only show visible NEOs
    this.updateNEODropdown();
  }

  private focusOnNEO(neoId: string): void {
    const neo = this.neoObjects.get(neoId);
    if (neo) {
      // Select the NEO
      this.selectNEO(neo);
      // Focus camera on it
      this.focusOnObject(neo.position, 15);
    }
  }

  private updateNEODropdown(): void {
    const dropdown = document.getElementById('neo-dropdown') as HTMLSelectElement;
    if (!dropdown) return;

    // Group NEOs by threat level
    const grouped: Record<string, Array<{ id: string; name: string }>> = {
      CRITICAL: [],
      HIGH: [],
      MODERATE: [],
      LOW: [],
      NEGLIGIBLE: [],
    };

    for (const [id, data] of this.neoData) {
      const threatLevel = data.risk?.threat_level || 'NEGLIGIBLE';
      // Only include if filter is enabled
      if (this.riskFilters[threatLevel]) {
        grouped[threatLevel].push({ id, name: data.name });
      }
    }

    // Build dropdown options with optgroups
    let html = '<option value="">Select an asteroid...</option>';

    const levelLabels: Record<string, string> = {
      CRITICAL: 'Critical',
      HIGH: 'High',
      MODERATE: 'Moderate',
      LOW: 'Low',
      NEGLIGIBLE: 'Negligible',
    };

    for (const level of ['CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'NEGLIGIBLE']) {
      const neos = grouped[level];
      if (neos.length > 0) {
        html += `<optgroup label="${levelLabels[level]} (${neos.length})">`;
        for (const neo of neos) {
          html += `<option value="${neo.id}">${neo.name}</option>`;
        }
        html += '</optgroup>';
      }
    }

    dropdown.innerHTML = html;
  }

  private updateLegend(): void {
    const items = getLegendItems(this.displayMode);
    this.legendEl.innerHTML = items.map(item => `
      <div class="legend-item" title="${item.description}">
        <div class="legend-dot" style="background:${item.color}"></div>
        <span>${item.label}</span>
      </div>
    `).join('');
  }

  private setupWebSocket(): void {
    this.dataStream.connect({
      onConnect: () => {
        this.setStatus('connected', 'Connected');
        // Auto-start analysis on connect
        this.startAnalysis();
      },

      onDisconnect: () => {
        this.setStatus('error', 'Disconnected');
      },

      onNEOList: (neos) => {
        this.handleNEOList(neos);
      },

      onRiskUpdate: (risk) => {
        this.handleRiskUpdate(risk);
      },

      onOrbitalUpdate: (orbital) => {
        this.handleOrbitalUpdate(orbital);
      },

      onProgress: (status) => {
        this.handleProgress(status);
      },

      onComplete: () => {
        this.handleComplete();
      },

      onAIInsight: (insight) => {
        this.handleAIInsight(insight);
      },

      onFleetAnalysis: (analysis) => {
        this.handleFleetAnalysis(analysis);
      },

      onWhatIfResult: (result) => {
        this.handleWhatIfResult(result);
      },

      onDeflectionResult: (result) => {
        this.handleDeflectionResult(result);
      },

      onInterceptMission: (mission) => {
        this.handleInterceptMission(mission);
      },

      onError: (message) => {
        this.setStatus('error', `Error: ${message}`);
      },
    });
  }

  private setStatus(className: string, text: string): void {
    this.statusEl.className = className;
    this.statusEl.textContent = text;
  }

  private setView(mode: ViewMode): void {
    this.currentView = mode;
    this.sceneManager.setViewMode(mode);

    const isSolar = mode === 'solar';

    // Update button states
    document.getElementById('btn-earth-view')?.classList.toggle('active', mode === 'earth');
    document.getElementById('btn-solar-view')?.classList.toggle('active', mode === 'solar');

    // Distance rings only in Earth view
    this.distanceRings.setVisible(!isSolar);

    // Planets only in Solar view
    this.planets.setVisible(isSolar);

    // Earth and Moon orbital mode
    console.log('Setting solar view:', isSolar);
    this.earth.setSolarView(isSolar);
    this.moon.setSolarView(isSolar);

    // Update Sun position for view mode
    if (isSolar) {
      this.sun.setSolarSystemPosition();
    } else {
      this.sun.setEarthCentricPosition();
    }
  }

  private startAnalysis(): void {
    this.setStatus('analyzing', 'Analyzing...');
    this.progressEl.classList.add('active');
    this.progressFillEl.style.width = '0%';
    this.dataStream.startAnalysis(7, 500);
  }

  private handleNEOList(neos: NEOData[]): void {
    // Clear existing NEOs
    for (const neo of this.neoObjects.values()) {
      this.sceneManager.scene.remove(neo);
      neo.dispose();
    }
    this.neoObjects.clear();
    this.neoData.clear();
    this.neoOriginalPositions.clear();

    // Create new NEO objects
    for (const data of neos) {
      const neoWithAnalysis: NEOWithAnalysis = { ...data };
      this.neoData.set(data.id, neoWithAnalysis);

      const neoObject = new NEOObject(neoWithAnalysis);
      this.neoObjects.set(data.id, neoObject);
      this.sceneManager.scene.add(neoObject);

      // Store original Earth-centric position for solar view transforms
      this.neoOriginalPositions.set(data.id, neoObject.position.clone());
    }

    this.setStatus('analyzing', `Loaded ${neos.length} NEOs`);

    // Populate dropdown with initial NEO list
    this.updateNEODropdown();
  }

  private handleRiskUpdate(risk: RiskData): void {
    const data = this.neoData.get(risk.neo_id);
    if (data) {
      data.risk = risk;
    }

    const neoObject = this.neoObjects.get(risk.neo_id);
    if (neoObject) {
      neoObject.updateRisk(risk.threat_level as ThreatLevel);
      neoObject.neoData = data!;
    }

    // Update info panel if this NEO is selected
    if (this.selectedNEO?.neoData.id === risk.neo_id) {
      this.updateInfoPanel();
    }
  }

  private handleOrbitalUpdate(orbital: OrbitalData): void {
    const data = this.neoData.get(orbital.neo_id);
    if (data) {
      data.orbital = orbital;
    }

    // Update info panel if this NEO is selected
    if (this.selectedNEO?.neoData.id === orbital.neo_id) {
      this.updateInfoPanel();
    }
  }

  private handleProgress(status: { current: number; total: number; phase: string; current_neo: string | null }): void {
    const percent = status.total > 0 ? (status.current / status.total) * 100 : 0;
    this.progressFillEl.style.width = `${percent}%`;

    let text = '';
    switch (status.phase) {
      case 'fetching':
        text = 'Fetching NEO data from NASA...';
        break;
      case 'classifying':
        text = `Quantum risk analysis: ${status.current_neo || ''}`;
        break;
      case 'simulating':
        text = `Orbital simulation: ${status.current_neo || ''}`;
        break;
      case 'complete':
        text = 'Analysis complete!';
        break;
    }

    this.progressTextEl.textContent = text;
    this.setStatus('analyzing', `${status.phase} (${status.current}/${status.total})`);
  }

  private handleComplete(): void {
    this.progressEl.classList.remove('active');
    this.setStatus('connected', 'Analysis complete');

    // Update dropdown with proper risk groupings
    this.updateNEODropdown();
  }

  private requestFleetAnalysis(): void {
    this.showAIPanel();
    this.aiContentEl.innerHTML = `
      <div class="ai-loading">
        <div class="spinner"></div>
        <p>Generating AI fleet analysis...</p>
        <p style="font-size:12px;margin-top:10px">Powered by NVIDIA Nemotron</p>
      </div>
    `;
    this.dataStream.requestFleetAnalysis();
  }

  private requestAIInsight(neoId: string): void {
    this.showAIPanel();
    this.aiContentEl.innerHTML = `
      <div class="ai-loading">
        <div class="spinner"></div>
        <p>Analyzing asteroid threat...</p>
      </div>
    `;
    this.dataStream.requestAIInsight(neoId);
  }

  private handleAIInsight(insight: AIInsight): void {
    const neo = this.neoData.get(insight.neo_id);
    const name = neo?.name || insight.neo_id;

    let html = `
      <div class="ai-summary">
        <strong>${name}</strong>
        <p>${insight.summary}</p>
      </div>

      <h3>Threat Assessment</h3>
      <p>${insight.threat_assessment}</p>
    `;

    if (insight.historical_context) {
      html += `
        <h3>Historical Context</h3>
        <p>${insight.historical_context}</p>
      `;
    }

    if (insight.recommendations.length > 0) {
      html += `
        <h3>Recommendations</h3>
        <ul>
          ${insight.recommendations.map(r => `<li>${r}</li>`).join('')}
        </ul>
      `;
    }

    if (insight.fun_facts.length > 0) {
      html += `
        <h3>Interesting Facts</h3>
        <ul>
          ${insight.fun_facts.map(f => `<li>${f}</li>`).join('')}
        </ul>
      `;
    }

    html += `
      <p style="margin-top:20px;font-size:12px;color:#666">
        AI Confidence: ${(insight.confidence * 100).toFixed(0)}%
      </p>
    `;

    this.aiContentEl.innerHTML = html;
  }

  private handleFleetAnalysis(analysis: FleetAnalysis): void {
    let html = `
      <div class="ai-summary">
        <strong>Fleet Analysis: ${analysis.total_analyzed} Objects</strong>
        <p>${analysis.executive_summary}</p>
      </div>
    `;

    if (analysis.highest_priority_threats.length > 0) {
      html += `
        <h3>Priority Threats</h3>
        <ul>
          ${analysis.highest_priority_threats.map(t => `<li class="ai-priority">${t}</li>`).join('')}
        </ul>
      `;
    }

    if (analysis.patterns_detected.length > 0) {
      html += `
        <h3>Patterns Detected</h3>
        <ul>
          ${analysis.patterns_detected.map(p => `<li>${p}</li>`).join('')}
        </ul>
      `;
    }

    if (analysis.weekly_outlook) {
      html += `
        <h3>Weekly Outlook</h3>
        <p>${analysis.weekly_outlook}</p>
      `;
    }

    if (analysis.recommendations.length > 0) {
      html += `
        <h3>Strategic Recommendations</h3>
        <ul>
          ${analysis.recommendations.map(r => `<li>${r}</li>`).join('')}
        </ul>
      `;
    }

    this.aiContentEl.innerHTML = html;
  }

  private handleWhatIfResult(result: WhatIfResult): void {
    const neo = this.neoData.get(result.neo_id);
    const name = neo?.name || result.neo_id;

    let html = `
      <div class="ai-summary">
        <strong>What-If Analysis: ${name}</strong>
        <p>Quantum orbital simulation across multiple scenarios</p>
      </div>
    `;

    for (const [scenario, data] of Object.entries(result.scenarios)) {
      const scenarioName = scenario.replace(/_/g, ' ').toUpperCase();
      html += `
        <div style="margin:15px 0;padding:10px;background:rgba(255,255,255,0.05);border-radius:4px">
          <h3 style="margin:0 0 10px">${scenarioName}</h3>
          <div class="stat"><span class="stat-label">Min Distance</span><span class="stat-value">${data.min_earth_distance.toFixed(1)} LD</span></div>
          <div class="stat"><span class="stat-label">Time to Closest</span><span class="stat-value">${data.min_distance_time.toFixed(0)} days</span></div>
          <div class="stat"><span class="stat-label">Collision Prob</span><span class="stat-value">${data.collision_probability.toExponential(2)}</span></div>
          ${data.impact_energy_mt ? `<div class="stat"><span class="stat-label">Impact Energy</span><span class="stat-value">${data.impact_energy_mt.toFixed(0)} MT</span></div>` : ''}
        </div>
      `;
    }

    this.aiContentEl.innerHTML = html;
  }

  private handleDeflectionResult(result: DeflectionResult): void {
    const neo = this.neoData.get(result.neo_id);
    const name = neo?.name || result.neo_id;

    const html = `
      <div class="ai-summary">
        <strong>QAOA Deflection Optimization: ${name}</strong>
        <p>Quantum-optimized asteroid deflection strategy</p>
      </div>

      <h3>Optimal Strategy</h3>
      <div class="stat"><span class="stat-label">Delta-V Required</span><span class="stat-value">${(result.optimal_delta_v * 100).toFixed(2)} cm/s</span></div>
      <div class="stat"><span class="stat-label">Direction</span><span class="stat-value">(${result.optimal_direction.map(d => d.toFixed(2)).join(', ')})</span></div>
      <div class="stat"><span class="stat-label">Execute At</span><span class="stat-value">${result.deflection_time.toFixed(0)} days before approach</span></div>
      <div class="stat"><span class="stat-label">Target Distance</span><span class="stat-value">${result.miss_distance_achieved.toFixed(0)} LD</span></div>

      <h3>Energy Requirements</h3>
      <div class="stat"><span class="stat-label">Kinetic Impactor</span><span class="stat-value">${result.energy_required_mt.toFixed(2)} MT</span></div>

      <h3>Analysis</h3>
      <div class="stat"><span class="stat-label">Success Probability</span><span class="stat-value">${(result.success_probability * 100).toFixed(1)}%</span></div>
      <div class="stat"><span class="stat-label">Quantum Iterations</span><span class="stat-value">${result.quantum_iterations.toLocaleString()}</span></div>
    `;

    this.aiContentEl.innerHTML = html;
  }

  private handleInterceptMission(mission: InterceptMission): void {
    const neo = this.neoData.get(mission.neo_id);
    const name = neo?.name || mission.neo_id;
    const missionTypeName = mission.mission_type.replace(/_/g, ' ').toUpperCase();

    let html = `
      <div class="ai-summary">
        <strong>Intercept Mission: ${name}</strong>
        <p>${missionTypeName}</p>
      </div>

      <h3>Mission Summary</h3>
      <div class="stat"><span class="stat-label">Duration</span><span class="stat-value">${mission.mission_duration.toFixed(0)} days</span></div>
      <div class="stat"><span class="stat-label">Total Delta-V</span><span class="stat-value">${mission.total_delta_v.toFixed(1)} km/s</span></div>
      <div class="stat"><span class="stat-label">Intercept Date</span><span class="stat-value">Day ${mission.intercept_date.toFixed(0)}</span></div>
      <div class="stat"><span class="stat-label">Success Probability</span><span class="stat-value">${(mission.success_probability * 100).toFixed(0)}%</span></div>

      <h3>Recommended Launch Window</h3>
      <div class="stat"><span class="stat-label">Window</span><span class="stat-value">Day ${mission.recommended_window.window_open.toFixed(0)}-${mission.recommended_window.window_close.toFixed(0)}</span></div>
      <div class="stat"><span class="stat-label">C3 Energy</span><span class="stat-value">${mission.recommended_window.c3_energy.toFixed(1)} km^2/s^2</span></div>
      <div class="stat"><span class="stat-label">Flight Time</span><span class="stat-value">${mission.recommended_window.flight_time.toFixed(0)} days</span></div>
      <div class="stat"><span class="stat-label">Arrival Velocity</span><span class="stat-value">${mission.recommended_window.arrival_velocity.toFixed(1)} km/s</span></div>

      <h3>Mission Phases</h3>
    `;

    for (const phase of mission.mission_phases) {
      html += `
        <div style="margin:5px 0;padding:8px;background:rgba(0,255,255,0.1);border-radius:4px">
          <strong>${phase.name}</strong>
          <p style="margin:5px 0;color:#aaa">${phase.description}</p>
          ${phase.duration > 0 ? `<span style="font-size:12px;color:#888">Duration: ${phase.duration.toFixed(0)} days</span>` : ''}
        </div>
      `;
    }

    this.aiContentEl.innerHTML = html;
  }

  private showAIPanel(): void {
    this.aiPanelEl.classList.remove('hidden');
  }

  private hideAIPanel(): void {
    this.aiPanelEl.classList.add('hidden');
  }

  private showAboutPanel(): void {
    this.aboutContentEl.innerHTML = this.generateAboutContent();
    this.aboutPanelEl.classList.remove('hidden');
  }

  private hideAboutPanel(): void {
    this.aboutPanelEl.classList.add('hidden');
  }

  private generateAboutContent(): string {
    const mode = this.displayMode;

    if (mode === 'normal') {
      return `
        <div class="ai-summary">
          <strong>What is NEO Quantum?</strong>
          <p>An educational demonstration that combines real NASA asteroid data with
          GPU-simulated quantum computing algorithms for threat visualization.</p>
        </div>

        <h3>Important Disclaimer</h3>
        <p style="color:#f90;font-size:12px;background:rgba(255,150,0,0.1);padding:10px;border-radius:4px;">
          This is an <strong>educational project</strong>, not an operational planetary defense system.
          The threat levels shown are illustrative demonstrations, not official NASA risk assessments.
          For actual asteroid impact risk, see <a href="https://cneos.jpl.nasa.gov/" style="color:#0ff" target="_blank">NASA CNEOS</a>.
        </p>

        <h3>What Am I Looking At?</h3>
        <p>The glowing dots are <strong>Near Earth Objects (NEOs)</strong> - asteroids and comets
        whose orbits bring them close to Earth. Their colors show illustrative threat levels:</p>
        <ul>
          <li><span style="color:#0f0">Green</span> - Safe, no concern</li>
          <li><span style="color:#0ff">Cyan</span> - Very low risk</li>
          <li><span style="color:#ff0">Yellow</span> - Worth watching</li>
          <li><span style="color:#f60">Orange</span> - Needs attention</li>
          <li><span style="color:#f00">Red</span> - High priority to monitor</li>
        </ul>

        <h3>Where Does The Data Come From?</h3>
        <p>All asteroid data comes directly from <strong>NASA's Near Earth Object database</strong>,
        updated in real-time. This is real data from space agencies worldwide.</p>

        <h3>About the Quantum Simulations</h3>
        <p>This project uses <strong>NVIDIA CUDA-Q</strong> to simulate quantum computing algorithms
        on classical GPU hardware. Key clarifications:</p>
        <ul>
          <li><strong>No quantum hardware:</strong> All circuits run on GPU-simulated qubits</li>
          <li><strong>No quantum speedup:</strong> On a simulator, quantum algorithms provide no computational advantage</li>
          <li><strong>Educational purpose:</strong> We demonstrate <em>what</em> quantum algorithms do, not gain speedup from them</li>
        </ul>

        <h3>How To Use It</h3>
        <ul>
          <li><strong>Click any dot</strong> to see details about that asteroid</li>
          <li><strong>Drag</strong> to rotate the view</li>
          <li><strong>Scroll</strong> to zoom in and out</li>
          <li><strong>Earth View</strong> shows asteroids relative to Earth</li>
          <li><strong>Solar System</strong> shows the bigger picture with planets</li>
          <li><strong>AI Fleet Report</strong> gets an AI summary of current threats</li>
        </ul>

        <h3>Should I Be Worried?</h3>
        <p>No! Space agencies continuously monitor all known NEOs. Most pass millions of kilometers
        away. For real asteroid impact risk, always refer to official sources like
        <a href="https://cneos.jpl.nasa.gov/" style="color:#0ff" target="_blank">NASA CNEOS</a> and
        <a href="https://neo.ssa.esa.int/" style="color:#0ff" target="_blank">ESA NEOCC</a>.</p>

        <h3>Who Made This?</h3>
        <p>Built by <strong>HxHippy</strong> (CTO of <a href="https://kief.studio" style="color:#0ff" target="_blank">Kief Studio</a>)
        as an open-source educational project to make space science and quantum computing accessible.
        Uses Three.js for 3D graphics, NVIDIA CUDA-Q for quantum circuit simulation, and NASA's public APIs.</p>
      `;
    } else {
      // Scientific mode
      return `
        <div class="ai-summary">
          <strong>NEO Quantum Technical Overview</strong>
          <p>Real-time Near Earth Object visualization with GPU-simulated quantum algorithms
          using NVIDIA CUDA-Q. This is an educational demonstration, not operational software.</p>
        </div>

        <h3>Important Disclaimer</h3>
        <p style="color:#f90;font-size:12px;background:rgba(255,150,0,0.1);padding:10px;border-radius:4px;">
          All quantum circuits run on <strong>classical GPU simulation</strong>, not quantum hardware.
          The threat classifications and collision probabilities are illustrative demonstrations
          using heuristic models, not official NASA/ESA risk assessments.
        </p>

        <h3>Data Sources</h3>
        <ul>
          <li><strong>NASA NeoWs API</strong> - Primary NEO data feed</li>
          <li><strong>NASA Sentry API</strong> - Objects with computed impact probabilities</li>
          <li><strong>JPL SBDB/CAD</strong> - Orbital elements and close approaches</li>
        </ul>

        <h3>Quantum Algorithms (GPU-Simulated)</h3>
        <ul>
          <li><strong>VQC (Classification):</strong> 4-qubit variational classifier trained on NASA data</li>
          <li><strong>VQE (Stability):</strong> 6-qubit ansatz for orbital stability analysis (heuristic Hamiltonian)</li>
          <li><strong>QAE (Probability):</strong> 8-qubit amplitude estimation for collision probability</li>
          <li><strong>QAOA (Optimization):</strong> 8-qubit optimizer for deflection trajectory planning</li>
        </ul>
        <p style="font-size:11px;color:#888;">Note: On a GPU simulator, these provide no speedup over classical algorithms.</p>

        <h3>VQC Circuit Structure</h3>
        <ul>
          <li><strong>Feature Vector:</strong> [diameter, velocity, miss_distance, kinetic_energy, ...]</li>
          <li><strong>Encoding:</strong> RY rotations with angle-encoded features</li>
          <li><strong>Ansatz:</strong> 3 variational layers, RX/RY/RZ + CNOT ring</li>
          <li><strong>Training:</strong> Parameter-shift gradients, NASA Sentry data</li>
          <li><strong>Output:</strong> Probability distribution mapped to 5 threat levels</li>
        </ul>

        <h3>Threat Level Criteria (Heuristic)</h3>
        <ul>
          <li><strong>NEGLIGIBLE:</strong> d &lt; 10m OR miss &gt; 20 LD</li>
          <li><strong>LOW:</strong> d &lt; 50m AND miss &gt; 5 LD</li>
          <li><strong>MODERATE:</strong> d &lt; 140m OR is_PHA</li>
          <li><strong>HIGH:</strong> E &gt; 10 MT AND miss &lt; 5 LD</li>
          <li><strong>CRITICAL:</strong> E &gt; 100 MT AND miss &lt; 1 LD</li>
        </ul>
        <p style="font-size:11px;color:#888;">These are simplified heuristics for educational purposes.</p>

        <h3>Visualization Parameters</h3>
        <ul>
          <li><strong>Distance Scale:</strong> 1 unit = 1 Lunar Distance (384,400 km)</li>
          <li><strong>Size Scale:</strong> NEOs exaggerated ~1000x for visibility</li>
          <li><strong>Coordinate Frame:</strong> Earth-centered J2000 equatorial</li>
        </ul>

        <h3>Technology Stack</h3>
        <ul>
          <li><strong>Frontend:</strong> Three.js + TypeScript + Vite</li>
          <li><strong>Backend:</strong> Python FastAPI + WebSocket streaming</li>
          <li><strong>Quantum Simulation:</strong> NVIDIA CUDA-Q (GPU-accelerated)</li>
          <li><strong>AI:</strong> NVIDIA Nemotron via OpenRouter</li>
        </ul>

        <h3>References</h3>
        <ul>
          <li><a href="https://cneos.jpl.nasa.gov/" style="color:#0ff" target="_blank">NASA CNEOS</a> - Official NEO data</li>
          <li><a href="https://developer.nvidia.com/cuda-quantum" style="color:#0ff" target="_blank">NVIDIA CUDA-Q</a> - Quantum SDK</li>
          <li><a href="https://neo.ssa.esa.int/" style="color:#0ff" target="_blank">ESA NEOCC</a> - European coordination</li>
        </ul>
      `;
    }
  }

  private selectNEO(neoId: string): void {
    // Deselect previous
    if (this.selectedNEO) {
      this.selectedNEO.setSelected(false);
    }

    // Select new
    const neoObject = this.neoObjects.get(neoId);
    if (neoObject) {
      this.selectedNEO = neoObject;
      neoObject.setSelected(true);
      this.updateInfoPanel();

      // Request high-precision analysis
      this.dataStream.requestNEODetails(neoId, true);
    }
  }

  private deselectNEO(): void {
    if (this.selectedNEO) {
      this.selectedNEO.setSelected(false);
      this.selectedNEO = null;
    }

    this.infoEl.innerHTML = '<p style="color:#888">Click on an asteroid to view details</p>';
  }

  private updateInfoPanel(): void {
    if (!this.selectedNEO) return;

    const neo = this.selectedNEO.neoData;
    const risk = neo.risk;
    const orbital = neo.orbital;
    const mode = this.displayMode;

    // Get threat level formatting
    const threatInfo = risk ? formatThreatLevel(risk.threat_level, mode) : null;

    // PHA badge text based on mode
    const phaLabel = mode === 'normal' ? 'Potentially Hazardous' : 'PHA';

    let html = `
      <div class="neo-name">${neo.name}</div>
      ${neo.is_hazardous ? `<span class="threat-badge threat-HIGH">${phaLabel}</span>` : ''}
    `;

    if (risk && threatInfo) {
      html += `
        <div class="stat">
          <span class="stat-label">${mode === 'normal' ? 'Status' : 'Threat Level'}</span>
          <span class="threat-badge threat-${risk.threat_level}">${threatInfo.label}</span>
        </div>
        <div style="font-size:11px;color:#888;margin-bottom:10px">${threatInfo.description}</div>
      `;
    }

    // Size
    html += `
      <div class="stat">
        <span class="stat-label">${mode === 'normal' ? 'Size' : 'Diameter'}</span>
        <span class="stat-value">${formatDiameter(neo.avg_diameter_km, mode)}</span>
      </div>
    `;

    // Velocity
    html += `
      <div class="stat">
        <span class="stat-label">${mode === 'normal' ? 'Speed' : 'Velocity'}</span>
        <span class="stat-value">${formatVelocity(neo.velocity_km_s, mode)}</span>
      </div>
    `;

    // Distance
    html += `
      <div class="stat">
        <span class="stat-label">${mode === 'normal' ? 'Closest Distance' : 'Miss Distance'}</span>
        <span class="stat-value">${formatDistance(neo.miss_distance_lunar, mode)}</span>
      </div>
    `;

    // Date
    html += `
      <div class="stat">
        <span class="stat-label">${mode === 'normal' ? 'When' : 'Close Approach'}</span>
        <span class="stat-value">${formatDate(neo.close_approach_date, mode)}</span>
      </div>
    `;

    // Energy
    html += `
      <div class="stat">
        <span class="stat-label">${mode === 'normal' ? 'Impact Power' : 'Kinetic Energy'}</span>
        <span class="stat-value">${formatEnergy(neo.kinetic_energy_mt, mode)}</span>
      </div>
    `;

    if (orbital) {
      html += `
        <div class="stat">
          <span class="stat-label">${mode === 'normal' ? 'Hit Chance' : 'Collision Prob.'}</span>
          <span class="stat-value">${formatProbability(orbital.collision_probability, mode)}</span>
        </div>
      `;

      // Scientific mode shows additional data
      if (mode === 'scientific') {
        html += `
          <div class="stat">
            <span class="stat-label">Pos. Uncertainty</span>
            <span class="stat-value">${orbital.position_uncertainty_km.toFixed(0)} km</span>
          </div>
        `;
      }
    }

    // Threat probability distribution (scientific mode only)
    if (risk && mode === 'scientific') {
      html += '<div style="margin-top:15px"><strong>Quantum Risk Distribution</strong></div>';
      html += '<div style="display:flex;height:20px;margin-top:5px;border-radius:4px;overflow:hidden">';

      const colors: Record<string, string> = {
        NEGLIGIBLE: '#0a0',
        LOW: '#0aa',
        MODERATE: '#aa0',
        HIGH: '#f60',
        CRITICAL: '#f00',
      };

      for (const [level, prob] of Object.entries(risk.probabilities)) {
        const width = prob * 100;
        if (width > 2) {
          html += `<div style="width:${width}%;background:${colors[level]}" title="${level}: ${(prob * 100).toFixed(1)}%"></div>`;
        }
      }

      html += '</div>';
    }

    // Simulation buttons - labels based on mode
    const btnLabels = mode === 'normal'
      ? { whatif: 'Scenarios', deflect: 'Stop It', intercept: 'Mission', ai: 'Explain' }
      : { whatif: 'What-If', deflect: 'Deflect', intercept: 'Intercept', ai: 'AI Analysis' };

    html += `
      <div style="margin-top:15px;display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <button id="btn-whatif" class="control-btn" style="font-size:12px;padding:8px">${btnLabels.whatif}</button>
        <button id="btn-deflect" class="control-btn" style="font-size:12px;padding:8px">${btnLabels.deflect}</button>
        <button id="btn-intercept" class="control-btn" style="font-size:12px;padding:8px">${btnLabels.intercept}</button>
        <button id="btn-ai-neo" class="control-btn" style="font-size:12px;padding:8px">${btnLabels.ai}</button>
      </div>
    `;

    this.infoEl.innerHTML = html;

    // Attach button handlers
    document.getElementById('btn-ai-neo')?.addEventListener('click', () => {
      this.requestAIInsight(neo.id);
    });

    document.getElementById('btn-whatif')?.addEventListener('click', () => {
      this.showAIPanel();
      this.aiContentEl.innerHTML = `
        <div class="ai-loading">
          <div class="spinner"></div>
          <p>Running quantum what-if scenarios...</p>
        </div>
      `;
      this.dataStream.requestWhatIfAnalysis(neo.id);
    });

    document.getElementById('btn-deflect')?.addEventListener('click', () => {
      this.showAIPanel();
      this.aiContentEl.innerHTML = `
        <div class="ai-loading">
          <div class="spinner"></div>
          <p>QAOA deflection optimization...</p>
        </div>
      `;
      this.dataStream.requestDeflection(neo.id);
    });

    document.getElementById('btn-intercept')?.addEventListener('click', () => {
      this.showAIPanel();
      this.aiContentEl.innerHTML = `
        <div class="ai-loading">
          <div class="spinner"></div>
          <p>Planning intercept mission...</p>
        </div>
      `;
      this.dataStream.requestInterceptMission(neo.id, 'kinetic_impactor');
    });
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new NEOQuantumApp();
});

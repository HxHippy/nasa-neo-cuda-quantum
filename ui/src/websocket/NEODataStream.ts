/**
 * WebSocket connection manager for NEO data streaming
 */

import type {
  WSMessage,
  NEOData,
  RiskData,
  OrbitalData,
  AnalysisStatus,
  AIInsight,
  FleetAnalysis,
  WhatIfResult,
  DeflectionResult,
  InterceptMission,
} from '../types/neo';

export type NEOEventHandlers = {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onNEOList?: (neos: NEOData[]) => void;
  onRiskUpdate?: (risk: RiskData) => void;
  onOrbitalUpdate?: (orbital: OrbitalData) => void;
  onProgress?: (status: AnalysisStatus) => void;
  onComplete?: () => void;
  onAIInsight?: (insight: AIInsight) => void;
  onFleetAnalysis?: (analysis: FleetAnalysis) => void;
  onWhatIfResult?: (result: WhatIfResult) => void;
  onDeflectionResult?: (result: DeflectionResult) => void;
  onInterceptMission?: (mission: InterceptMission) => void;
  onError?: (message: string) => void;
};

// Default WebSocket URL - can be overridden via VITE_WEBSOCKET_URL env var
const DEFAULT_WS_URL = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000/ws/neo-stream';

export class NEODataStream {
  private ws: WebSocket | null = null;
  private handlers: NEOEventHandlers = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000;
  private url: string;

  constructor(url: string = DEFAULT_WS_URL) {
    this.url = url;
  }

  connect(handlers: NEOEventHandlers): void {
    this.handlers = handlers;
    this.doConnect();
  }

  private doConnect(): void {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.handlers.onConnect?.();
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.handlers.onDisconnect?.();
        this.attemptReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.handlers.onError?.('Connection error');
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    } catch (error) {
      console.error('Failed to connect:', error);
      this.attemptReconnect();
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    console.log(`Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.doConnect();
    }, this.reconnectDelay);
  }

  private handleMessage(data: string): void {
    try {
      const message: WSMessage = JSON.parse(data);

      switch (message.type) {
        case 'neo_list':
          if (message.data && typeof message.data === 'object' && 'neos' in message.data) {
            this.handlers.onNEOList?.(message.data.neos as NEOData[]);
          }
          break;

        case 'risk_update':
          this.handlers.onRiskUpdate?.(message.data as RiskData);
          break;

        case 'orbital_update':
          this.handlers.onOrbitalUpdate?.(message.data as OrbitalData);
          break;

        case 'analysis_progress':
          this.handlers.onProgress?.(message.data as AnalysisStatus);
          break;

        case 'analysis_complete':
          this.handlers.onComplete?.();
          break;

        case 'ai_insight':
          this.handlers.onAIInsight?.(message.data as AIInsight);
          break;

        case 'ai_fleet_analysis':
          this.handlers.onFleetAnalysis?.(message.data as FleetAnalysis);
          break;

        case 'what_if_result':
          this.handlers.onWhatIfResult?.(message.data as WhatIfResult);
          break;

        case 'deflection_result':
          this.handlers.onDeflectionResult?.(message.data as DeflectionResult);
          break;

        case 'intercept_mission':
          this.handlers.onInterceptMission?.(message.data as InterceptMission);
          break;

        case 'error':
          if (message.data && typeof message.data === 'object' && 'message' in message.data) {
            this.handlers.onError?.(message.data.message as string);
          }
          break;
      }
    } catch (error) {
      console.error('Failed to parse message:', error);
    }
  }

  startAnalysis(daysAhead: number = 7, shotsPerNeo: number = 500): void {
    this.send({
      type: 'start_analysis',
      data: {
        days_ahead: daysAhead,
        shots_per_neo: shotsPerNeo,
        include_orbital_sim: true,
      },
    });
  }

  requestNEODetails(neoId: string, highPrecision: boolean = false): void {
    this.send({
      type: 'request_neo_details',
      data: {
        neo_id: neoId,
        high_precision: highPrecision,
      },
    });
  }

  cancelAnalysis(): void {
    this.send({ type: 'cancel_analysis' });
  }

  requestAIInsight(neoId: string): void {
    this.send({
      type: 'request_ai_insight',
      data: { neo_id: neoId },
    });
  }

  requestFleetAnalysis(): void {
    this.send({ type: 'request_fleet_analysis' });
  }

  requestWhatIfAnalysis(neoId: string): void {
    this.send({
      type: 'request_what_if',
      data: { neo_id: neoId },
    });
  }

  requestDeflection(neoId: string, targetDistance: number = 100): void {
    this.send({
      type: 'request_deflection',
      data: {
        neo_id: neoId,
        target_distance: targetDistance,
      },
    });
  }

  requestInterceptMission(neoId: string, missionType: string = 'kinetic_impactor'): void {
    this.send({
      type: 'request_intercept',
      data: {
        neo_id: neoId,
        mission_type: missionType,
      },
    });
  }

  private send(message: WSMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected');
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

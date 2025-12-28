/**
 * Display Mode Types and Utilities
 *
 * Handles switching between Normal (public-friendly) and Scientific (expert) modes
 */

export type DisplayMode = 'normal' | 'scientific';

/**
 * Format distance for display based on mode
 */
export function formatDistance(lunarDistances: number, mode: DisplayMode): string {
  if (mode === 'normal') {
    if (lunarDistances < 1) {
      return `${(lunarDistances * 100).toFixed(0)}% of Moon's distance`;
    } else if (lunarDistances < 10) {
      return `${lunarDistances.toFixed(1)}x Moon's distance`;
    } else {
      return `${Math.round(lunarDistances)}x Moon's distance`;
    }
  } else {
    // Scientific mode
    const km = lunarDistances * 384400;
    const au = km / 149597870.7;
    if (au > 0.01) {
      return `${au.toFixed(4)} AU (${lunarDistances.toFixed(2)} LD)`;
    }
    return `${km.toLocaleString()} km (${lunarDistances.toFixed(2)} LD)`;
  }
}

/**
 * Format velocity for display based on mode
 */
export function formatVelocity(kmPerSec: number, mode: DisplayMode): string {
  if (mode === 'normal') {
    const kmPerHour = kmPerSec * 3600;
    const bulletSpeed = 1.7; // km/s for rifle bullet
    const multiplier = kmPerSec / bulletSpeed;
    if (multiplier > 10) {
      return `${Math.round(multiplier)}x faster than a bullet`;
    }
    return `${Math.round(kmPerHour).toLocaleString()} km/h`;
  } else {
    return `${kmPerSec.toFixed(2)} km/s`;
  }
}

/**
 * Format diameter for display based on mode
 */
export function formatDiameter(km: number, mode: DisplayMode): string {
  const meters = km * 1000;

  if (mode === 'normal') {
    // Use familiar comparisons
    if (meters < 50) {
      return `About the size of a house (${Math.round(meters)}m)`;
    } else if (meters < 100) {
      return `Size of a football field (${Math.round(meters)}m)`;
    } else if (meters < 300) {
      return `Size of a stadium (${Math.round(meters)}m)`;
    } else if (meters < 1000) {
      return `Size of a small town (${Math.round(meters)}m)`;
    } else {
      return `${(km).toFixed(1)} km across - city-sized`;
    }
  } else {
    if (meters < 1000) {
      return `${meters.toFixed(0)} m`;
    }
    return `${km.toFixed(2)} km`;
  }
}

/**
 * Format kinetic energy for display based on mode
 */
export function formatEnergy(megatons: number, mode: DisplayMode): string {
  if (mode === 'normal') {
    const hiroshima = 0.015; // MT
    const multiplier = megatons / hiroshima;
    if (multiplier < 1) {
      return 'Less than Hiroshima bomb';
    } else if (multiplier < 100) {
      return `${Math.round(multiplier)}x Hiroshima bomb`;
    } else if (multiplier < 10000) {
      return `${(multiplier / 1000).toFixed(1)}k Hiroshima bombs`;
    } else {
      return `${(multiplier / 1000000).toFixed(1)}M Hiroshima bombs`;
    }
  } else {
    if (megatons < 1) {
      return `${(megatons * 1000).toFixed(1)} kT TNT`;
    }
    return `${megatons.toFixed(2)} MT TNT`;
  }
}

/**
 * Format collision probability for display based on mode
 */
export function formatProbability(prob: number, mode: DisplayMode): string {
  if (mode === 'normal') {
    if (prob < 1e-9) {
      return 'Virtually zero';
    } else if (prob < 1e-6) {
      return 'Extremely unlikely';
    } else if (prob < 1e-4) {
      return 'Very unlikely';
    } else if (prob < 0.01) {
      return 'Unlikely';
    } else if (prob < 0.1) {
      return 'Possible';
    } else {
      return 'Significant chance';
    }
  } else {
    return prob.toExponential(2);
  }
}

/**
 * Get threat level description based on mode
 */
export function formatThreatLevel(level: string, mode: DisplayMode): { label: string; description: string } {
  if (mode === 'normal') {
    switch (level) {
      case 'NEGLIGIBLE':
        return { label: 'Safe', description: 'No concern - passes at safe distance' };
      case 'LOW':
        return { label: 'Monitor', description: 'Being tracked - no immediate concern' };
      case 'MODERATE':
        return { label: 'Watch', description: 'Closer approach - under observation' };
      case 'HIGH':
        return { label: 'Alert', description: 'Close approach - active monitoring' };
      case 'CRITICAL':
        return { label: 'Danger', description: 'Very close - requires attention' };
      default:
        return { label: 'Unknown', description: 'Analyzing...' };
    }
  } else {
    switch (level) {
      case 'NEGLIGIBLE':
        return { label: 'NEGLIGIBLE', description: 'Torino Scale 0 - No hazard' };
      case 'LOW':
        return { label: 'LOW', description: 'Torino Scale 1 - Normal discovery' };
      case 'MODERATE':
        return { label: 'MODERATE', description: 'Torino Scale 2-4 - Merits attention' };
      case 'HIGH':
        return { label: 'HIGH', description: 'Torino Scale 5-7 - Threatening' };
      case 'CRITICAL':
        return { label: 'CRITICAL', description: 'Torino Scale 8-10 - Certain collision' };
      default:
        return { label: 'UNCLASSIFIED', description: 'Pending quantum analysis' };
    }
  }
}

/**
 * Format date for display based on mode
 */
export function formatDate(dateStr: string, mode: DisplayMode): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = date.getTime() - now.getTime();
  const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));

  if (mode === 'normal') {
    if (diffDays < 0) {
      return `${Math.abs(diffDays)} days ago`;
    } else if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Tomorrow';
    } else if (diffDays < 7) {
      return `In ${diffDays} days`;
    } else if (diffDays < 30) {
      return `In ${Math.ceil(diffDays / 7)} weeks`;
    } else if (diffDays < 365) {
      return `In ${Math.ceil(diffDays / 30)} months`;
    } else {
      return `In ${(diffDays / 365).toFixed(1)} years`;
    }
  } else {
    return `${dateStr} (T${diffDays >= 0 ? '+' : ''}${diffDays}d)`;
  }
}

/**
 * Get legend items for the current mode
 */
export function getLegendItems(mode: DisplayMode): Array<{ color: string; label: string; description: string }> {
  if (mode === 'normal') {
    return [
      { color: '#0f0', label: 'Safe', description: 'No danger' },
      { color: '#0ff', label: 'Monitor', description: 'Being watched' },
      { color: '#ff0', label: 'Watch', description: 'Pay attention' },
      { color: '#f60', label: 'Alert', description: 'Close approach' },
      { color: '#f00', label: 'Danger', description: 'Very close' },
    ];
  } else {
    return [
      { color: '#0f0', label: 'Negligible', description: 'Torino 0' },
      { color: '#0ff', label: 'Low', description: 'Torino 1' },
      { color: '#ff0', label: 'Moderate', description: 'Torino 2-4' },
      { color: '#f60', label: 'High', description: 'Torino 5-7' },
      { color: '#f00', label: 'Critical', description: 'Torino 8-10' },
    ];
  }
}

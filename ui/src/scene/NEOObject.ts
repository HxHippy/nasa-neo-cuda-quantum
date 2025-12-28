/**
 * NEO (Near Earth Object) visualization
 *
 * Renders asteroids with realistic rocky appearance,
 * threat-level coloring, and size scaling
 */

import * as THREE from 'three';
import { NEOWithAnalysis, ThreatLevel, getThreatColor } from '../types/neo';

// Shared asteroid textures (loaded once)
let asteroidTextures: THREE.Texture[] = [];
let texturesLoaded = false;

// Array of asteroid texture paths for variety - using rocky surfaces
const ASTEROID_TEXTURE_PATHS = [
  '/textures/moon.jpg',      // Rocky, cratered surface
  '/textures/mercury.jpg',   // Rocky, cratered surface
  '/textures/ceres.jpg',     // Asteroid belt dwarf planet
];

function loadAsteroidTextures(): void {
  if (texturesLoaded) return;

  const textureLoader = new THREE.TextureLoader();

  for (const path of ASTEROID_TEXTURE_PATHS) {
    const texture = textureLoader.load(path);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    asteroidTextures.push(texture);
  }

  texturesLoaded = true;
}

function getAsteroidTexture(seed: number): THREE.Texture {
  loadAsteroidTextures();
  // Use seed to pick a consistent texture for each asteroid
  const index = Math.abs(seed) % asteroidTextures.length;
  return asteroidTextures[index];
}

export class NEOObject extends THREE.Group {
  public neoData: NEOWithAnalysis;

  private mesh: THREE.Mesh;
  private glow: THREE.Mesh;
  private innerGlow: THREE.Mesh;
  private backdrop: THREE.Mesh;
  private ring: THREE.Mesh | null = null;
  private trail: THREE.Line | null = null;

  private baseColor: number = 0x888888;
  private pulsePhase: number = 0;
  private isHovered: boolean = false;
  private isSelected: boolean = false;
  private rotationSpeed: THREE.Vector3;

  constructor(neoData: NEOWithAnalysis) {
    super();

    this.neoData = neoData;
    this.userData.clickable = true;
    this.userData.neoId = neoData.id;

    // Random rotation speeds for each axis
    this.rotationSpeed = new THREE.Vector3(
      (Math.random() - 0.5) * 1.5,
      (Math.random() - 0.5) * 1.5,
      (Math.random() - 0.5) * 0.5
    );

    // Size based on diameter - scaled for visibility but more proportional
    // Earth radius in scene = 0.15 units (real Earth = 12,742 km diameter)
    // So 1 scene unit ≈ 85,000 km
    //
    // Realistic but exaggerated for visibility:
    //   18m asteroid  -> 0.02 (tiny but visible with glow)
    //   100m asteroid -> 0.03
    //   500m asteroid -> 0.05
    //   1km asteroid  -> 0.07
    //   5km asteroid  -> 0.12
    //   10km asteroid -> 0.15 (roughly Earth-sized in scene)
    const diameterKm = neoData.avg_diameter_km;
    const size = Math.max(0.02, Math.min(0.2, 0.015 + diameterKm * 0.015));

    // Get texture based on asteroid ID for consistent appearance
    const seed = neoData.id.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
    const texture = getAsteroidTexture(seed);

    // Create asteroid mesh (irregular shape)
    const geometry = this.createAsteroidGeometry(size);

    // Determine base color from composition type (or use gray)
    const rockColor = this.getAsteroidBaseColor();

    const material = new THREE.MeshStandardMaterial({
      map: texture,
      color: 0xaaaaaa,  // Light gray tint to not overpower texture
      roughness: 0.85,
      metalness: 0.05,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.add(this.mesh);

    // Edge rim light - thin shell for contrast against black space
    const rimGeometry = new THREE.SphereGeometry(size * 1.05, 24, 24);
    const rimMaterial = new THREE.MeshBasicMaterial({
      color: 0x666666,
      transparent: true,
      opacity: 0.6,
      side: THREE.BackSide,
    });
    const rim = new THREE.Mesh(rimGeometry, rimMaterial);
    this.add(rim);

    // Inner glow (close to asteroid)
    const innerGlowGeometry = new THREE.SphereGeometry(size * 1.2, 16, 16);
    const innerGlowMaterial = new THREE.MeshBasicMaterial({
      color: this.baseColor,
      transparent: true,
      opacity: 0.2,
      side: THREE.BackSide,
    });
    this.innerGlow = new THREE.Mesh(innerGlowGeometry, innerGlowMaterial);
    this.add(this.innerGlow);

    // Outer glow for visibility at distance
    const glowGeometry = new THREE.SphereGeometry(size * 1.8, 16, 16);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: this.baseColor,
      transparent: true,
      opacity: 0.12,
      side: THREE.BackSide,
    });
    this.glow = new THREE.Mesh(glowGeometry, glowMaterial);
    this.add(this.glow);

    // Backdrop disc for enhanced visibility (hidden by default)
    const backdropGeometry = new THREE.CircleGeometry(size * 2.5, 32);
    const backdropMaterial = new THREE.MeshBasicMaterial({
      color: 0x1a1a2e,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    });
    this.backdrop = new THREE.Mesh(backdropGeometry, backdropMaterial);
    this.backdrop.visible = false;
    this.add(this.backdrop);

    // Position based on calculated coordinates
    this.position.set(
      neoData.position_x,
      neoData.position_y,
      neoData.position_z
    );

    // Add hazard indicator for PHAs
    if (neoData.is_hazardous) {
      this.addHazardRing(size);
    }

    // Initial random rotation
    this.mesh.rotation.set(
      Math.random() * Math.PI * 2,
      Math.random() * Math.PI * 2,
      Math.random() * Math.PI * 2
    );
  }

  private getAsteroidBaseColor(): number {
    // Asteroid spectral types (simulated)
    // C-type: dark, carbonaceous (gray-black)
    // S-type: stony, silicate (gray-brown)
    // M-type: metallic (gray with slight shine)
    const types = [
      0x555555, // C-type dark
      0x666666, // C-type medium
      0x6b5b4f, // S-type brownish
      0x7a6b5c, // S-type lighter
      0x707070, // M-type metallic
    ];
    return types[Math.floor(Math.random() * types.length)];
  }

  private createAsteroidGeometry(size: number): THREE.BufferGeometry {
    // Create sphere with proper UVs, then deform vertices
    // UVs stay mapped to original positions = texture works on deformed shape
    const geometry = new THREE.SphereGeometry(size, 24, 24);
    const positions = geometry.attributes.position;

    // Seeded random for consistent shape per asteroid
    const seed = this.neoData.id.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
    const seededRandom = (i: number) => {
      const x = Math.sin(seed + i * 12.9898) * 43758.5453;
      return x - Math.floor(x);
    };

    // Random elliptical scaling for blob shape
    const scaleX = 0.7 + seededRandom(1) * 0.6;
    const scaleY = 0.7 + seededRandom(2) * 0.6;
    const scaleZ = 0.7 + seededRandom(3) * 0.6;

    // Deform vertices for blobby asteroid look
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const z = positions.getZ(i);

      // Apply elliptical scaling
      const newX = x * scaleX;
      const newY = y * scaleY;
      const newZ = z * scaleZ;

      // Add subtle bumps based on vertex position
      const bumpNoise = (seededRandom(i * 13) - 0.5) * size * 0.2;
      const len = Math.sqrt(newX * newX + newY * newY + newZ * newZ);
      const nx = newX / len;
      const ny = newY / len;
      const nz = newZ / len;

      positions.setXYZ(
        i,
        newX + nx * bumpNoise,
        newY + ny * bumpNoise,
        newZ + nz * bumpNoise
      );
    }

    // Recompute normals for proper lighting on deformed shape
    geometry.computeVertexNormals();
    return geometry;
  }

  private addHazardRing(size: number): void {
    const ringGeometry = new THREE.TorusGeometry(size * 2, size * 0.15, 8, 32);
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: 0xff3300,
      transparent: true,
      opacity: 0.6,
    });

    this.ring = new THREE.Mesh(ringGeometry, ringMaterial);
    this.ring.rotation.x = Math.PI / 2;
    this.add(this.ring);
  }

  updateRisk(threatLevel: ThreatLevel): void {
    const color = getThreatColor(threatLevel);
    this.baseColor = color;

    // Update glow materials
    const innerGlowMat = this.innerGlow.material as THREE.MeshBasicMaterial;
    innerGlowMat.color.setHex(color);

    const glowMat = this.glow.material as THREE.MeshBasicMaterial;
    glowMat.color.setHex(color);

    // Update ring color if hazardous
    if (this.ring) {
      const ringMat = this.ring.material as THREE.MeshBasicMaterial;
      ringMat.color.setHex(color);
    }

    // Critical threats pulse
    if (threatLevel === ThreatLevel.CRITICAL) {
      this.pulsePhase = Math.random() * Math.PI * 2;
    }
  }

  setHovered(hovered: boolean): void {
    this.isHovered = hovered;

    const innerGlowMat = this.innerGlow.material as THREE.MeshBasicMaterial;
    const glowMat = this.glow.material as THREE.MeshBasicMaterial;

    innerGlowMat.opacity = hovered ? 0.35 : 0.15;
    glowMat.opacity = hovered ? 0.2 : 0.08;

    // Scale up slightly on hover
    const scale = hovered ? 1.3 : 1.0;
    this.mesh.scale.setScalar(scale);
    this.innerGlow.scale.setScalar(scale);
    this.glow.scale.setScalar(scale);
  }

  setSelected(selected: boolean): void {
    this.isSelected = selected;

    if (selected && !this.trail) {
      this.addTrajectoryTrail();
    } else if (!selected && this.trail) {
      this.parent?.remove(this.trail);
      this.trail.geometry.dispose();
      (this.trail.material as THREE.Material).dispose();
      this.trail = null;
    }
  }

  private addTrajectoryTrail(): void {
    // Create trajectory line pointing toward Earth
    const points: THREE.Vector3[] = [];
    const direction = new THREE.Vector3(0, 0, 0).sub(this.position).normalize();

    for (let i = 0; i < 100; i++) {
      const t = i / 100;
      const pos = this.position.clone().add(
        direction.clone().multiplyScalar(-t * 50)
      );
      points.push(pos);
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);

    // Gradient opacity trail
    const colors: number[] = [];
    for (let i = 0; i < 100; i++) {
      const alpha = 1 - (i / 100);
      colors.push(
        ((this.baseColor >> 16) & 255) / 255 * alpha,
        ((this.baseColor >> 8) & 255) / 255 * alpha,
        (this.baseColor & 255) / 255 * alpha
      );
    }
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
    });

    this.trail = new THREE.Line(geometry, material);
    this.parent?.add(this.trail);
  }

  update(delta: number, camera?: THREE.Camera): void {
    // Rotate asteroid with unique rotation
    this.mesh.rotation.x += delta * this.rotationSpeed.x;
    this.mesh.rotation.y += delta * this.rotationSpeed.y;
    this.mesh.rotation.z += delta * this.rotationSpeed.z;

    // Pulse effect for critical threats
    if (this.neoData.risk?.threat_level === ThreatLevel.CRITICAL) {
      this.pulsePhase += delta * 4;
      const pulse = 0.15 + Math.sin(this.pulsePhase) * 0.1;
      const glowPulse = 0.08 + Math.sin(this.pulsePhase) * 0.05;

      const innerGlowMat = this.innerGlow.material as THREE.MeshBasicMaterial;
      innerGlowMat.opacity = pulse;

      const glowMat = this.glow.material as THREE.MeshBasicMaterial;
      glowMat.opacity = glowPulse;
    }

    // Hazard ring rotation
    if (this.ring) {
      this.ring.rotation.z += delta * 0.8;
    }

    // Billboard backdrop to face camera
    if (this.backdrop.visible && camera) {
      this.backdrop.lookAt(camera.position);
    }
  }

  setBackdropVisible(visible: boolean): void {
    this.backdrop.visible = visible;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.innerGlow.geometry.dispose();
    (this.innerGlow.material as THREE.Material).dispose();
    this.glow.geometry.dispose();
    (this.glow.material as THREE.Material).dispose();

    if (this.ring) {
      this.ring.geometry.dispose();
      (this.ring.material as THREE.Material).dispose();
    }

    if (this.trail) {
      this.trail.geometry.dispose();
      (this.trail.material as THREE.Material).dispose();
    }
  }
}

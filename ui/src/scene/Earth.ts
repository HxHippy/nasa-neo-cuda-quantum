/**
 * Earth visualization with NASA-based textures
 * Textures from Solar System Scope (based on NASA imagery)
 */

import * as THREE from 'three';

// Earth radius in scene units (1 LD = 1 unit, Earth ~1/60 LD)
const EARTH_RADIUS = 0.15;
const SUN_X = 200;
const EARTH_ORBIT_RADIUS = 200; // 1 AU

export class Earth extends THREE.Group {
  private sphere: THREE.Mesh;
  private clouds: THREE.Mesh;
  private atmosphere: THREE.Mesh;
  private nightLights: THREE.Mesh;

  // Orbital properties for solar system view
  private orbitAngle: number = 0;
  private orbitSpeed: number = 0.25; // Earth's orbital speed (exaggerated for visualization)
  public isSolarView: boolean = false; // Made public for debugging

  constructor() {
    super();

    const textureLoader = new THREE.TextureLoader();

    // Load textures with proper color space
    const dayTexture = textureLoader.load('/textures/earth_day.jpg');
    dayTexture.colorSpace = THREE.SRGBColorSpace;

    const nightTexture = textureLoader.load('/textures/earth_night.jpg');
    nightTexture.colorSpace = THREE.SRGBColorSpace;

    const cloudsTexture = textureLoader.load('/textures/earth_clouds.jpg');
    const specularTexture = textureLoader.load('/textures/earth_specular.jpg');

    // Earth geometry
    const geometry = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);

    // Day side material with specular highlights for water
    const earthMaterial = new THREE.MeshPhongMaterial({
      map: dayTexture,
      specularMap: specularTexture,
      specular: new THREE.Color(0x333333),
      shininess: 25,
    });

    this.sphere = new THREE.Mesh(geometry, earthMaterial);
    this.add(this.sphere);

    // Night lights (city lights on dark side)
    const nightGeometry = new THREE.SphereGeometry(EARTH_RADIUS * 1.001, 64, 64);
    const nightMaterial = new THREE.MeshBasicMaterial({
      map: nightTexture,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });

    this.nightLights = new THREE.Mesh(nightGeometry, nightMaterial);
    this.add(this.nightLights);

    // Cloud layer
    const cloudGeometry = new THREE.SphereGeometry(EARTH_RADIUS * 1.02, 64, 64);
    const cloudMaterial = new THREE.MeshPhongMaterial({
      map: cloudsTexture,
      transparent: true,
      opacity: 0.4,
      depthWrite: false,
    });

    this.clouds = new THREE.Mesh(cloudGeometry, cloudMaterial);
    this.add(this.clouds);

    // Atmospheric glow - using simple material instead of shader
    const atmosphereGeometry = new THREE.SphereGeometry(EARTH_RADIUS * 1.15, 32, 32);
    const atmosphereMaterial = new THREE.MeshBasicMaterial({
      color: 0x4488ff,
      transparent: true,
      opacity: 0.2,
      side: THREE.BackSide,
    });

    this.atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    this.add(this.atmosphere);
  }

  update(delta: number, forceSolarView?: boolean): void {
    // Earth rotation (24-hour day, but sped up for visualization)
    const rotationSpeed = 0.1;
    this.sphere.rotation.y += delta * rotationSpeed;
    this.nightLights.rotation.y += delta * rotationSpeed;

    // Clouds rotate slightly faster
    this.clouds.rotation.y += delta * rotationSpeed * 1.1;

    // Orbital motion in solar system view
    const shouldOrbit = forceSolarView !== undefined ? forceSolarView : this.isSolarView;

    // Debug: log what we're receiving
    if (Math.random() < 0.01) {
      console.log('Earth update - forceSolarView:', forceSolarView, 'shouldOrbit:', shouldOrbit);
    }

    // Orbital movement now handled by main.ts
  }

  private updateOrbitalPosition(): void {
    this.position.x = SUN_X - Math.cos(this.orbitAngle) * EARTH_ORBIT_RADIUS;
    this.position.z = Math.sin(this.orbitAngle) * EARTH_ORBIT_RADIUS;
  }

  setSolarView(enabled: boolean): void {
    this.isSolarView = enabled;
    console.log('Earth setSolarView:', enabled, 'orbitAngle:', this.orbitAngle);
    if (enabled) {
      // Set initial position when entering solar view
      this.updateOrbitalPosition();
    } else {
      // Reset to origin for Earth-centric view
      this.position.set(0, 0, 0);
    }
  }

  getOrbitAngle(): number {
    return this.orbitAngle;
  }
}

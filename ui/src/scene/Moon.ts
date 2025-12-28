/**
 * Moon visualization with NASA-based textures
 * Textures from Solar System Scope (based on NASA imagery)
 */

import * as THREE from 'three';
import type { Planets } from './Planets';

export class Moon extends THREE.Group {
  private sphere: THREE.Mesh;
  private orbitAngle: number = 0;

  // 1 lunar distance = 1 unit in scene
  private orbitRadius: number = 1;

  // Reference to Planets for getting Earth position in solar system view
  private planetsRef: Planets | null = null;
  private isSolarView: boolean = false;

  constructor() {
    super();

    // Moon is about 1/4 Earth's diameter (Earth radius = 0.15 in scene)
    const MOON_RADIUS = 0.04;
    const geometry = new THREE.SphereGeometry(MOON_RADIUS, 64, 64);

    const textureLoader = new THREE.TextureLoader();

    // Load moon texture
    const moonTexture = textureLoader.load('/textures/moon.jpg');
    moonTexture.colorSpace = THREE.SRGBColorSpace;

    // Moon material with realistic lighting
    const material = new THREE.MeshPhongMaterial({
      map: moonTexture,
      bumpMap: moonTexture,
      bumpScale: 0.002,
      shininess: 5,
    });

    this.sphere = new THREE.Mesh(geometry, material);
    this.add(this.sphere);

    // Initial position
    this.updatePosition();
  }

  private updatePosition(): void {
    // Local orbit position relative to Earth
    const localX = Math.cos(this.orbitAngle) * this.orbitRadius;
    const localZ = Math.sin(this.orbitAngle) * this.orbitRadius;
    const localY = Math.sin(this.orbitAngle) * 0.089; // Moon's orbital inclination

    if (this.isSolarView && this.planetsRef) {
      // Follow Earth's position in solar system view
      const earthPos = this.planetsRef.getEarthPosition();
      this.position.x = earthPos.x + localX;
      this.position.z = earthPos.z + localZ;
      this.position.y = earthPos.y + localY;
    } else {
      // Earth-centric view - Earth at origin
      this.position.x = localX;
      this.position.z = localZ;
      this.position.y = localY;
    }
  }

  update(delta: number): void {
    // Moon orbits Earth (scaled for visualization)
    this.orbitAngle += delta * 0.05;
    this.updatePosition();

    // Tidally locked rotation
    this.sphere.rotation.y = -this.orbitAngle;
  }

  setPlanetsRef(planets: Planets): void {
    this.planetsRef = planets;
  }

  setSolarView(enabled: boolean): void {
    this.isSolarView = enabled;
  }
}

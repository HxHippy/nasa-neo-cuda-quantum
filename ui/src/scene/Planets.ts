/**
 * Solar System Planets visualization
 * Shows Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune
 * Positioned at scaled orbital distances from Sun
 */

import * as THREE from 'three';

interface PlanetData {
  name: string;
  color: number;
  size: number;           // Visual size in scene units
  distanceFromSun: number; // Scene units (Sun at 200, Earth at 0, so Earth = 200 units from Sun)
  orbitSpeed: number;     // Radians per second
  texture?: string;       // Texture file path
  hasRings?: boolean;
  ringColor?: number;
  ringTexture?: string;   // Ring texture file path
}

// Sun is at x=200, Earth is at x=0
// So 1 AU (Earth's distance) = 200 scene units
// Real AU distances: Mercury 0.39, Venus 0.72, Earth 1.0, Mars 1.52, Jupiter 5.2, Saturn 9.5, Uranus 19.2, Neptune 30
// Outer planets compressed for visibility

const SUN_X = 200;

const PLANETS: PlanetData[] = [
  // Inner planets - roughly to scale (speeds exaggerated for visualization)
  { name: 'Mercury', color: 0x8c7853, size: 0.3, distanceFromSun: 78, orbitSpeed: 0.5, texture: '/textures/mercury.jpg' },
  { name: 'Venus', color: 0xffc649, size: 0.7, distanceFromSun: 144, orbitSpeed: 0.35, texture: '/textures/venus_atmosphere.jpg' },
  { name: 'Earth', color: 0x4488ff, size: 1.0, distanceFromSun: 200, orbitSpeed: 0.25, texture: '/textures/earth_day.jpg' },
  { name: 'Mars', color: 0xc1440e, size: 0.4, distanceFromSun: 304, orbitSpeed: 0.15, texture: '/textures/mars.jpg' },
  // Outer planets - compressed distances for visibility
  { name: 'Jupiter', color: 0xd8ca9d, size: 2.5, distanceFromSun: 450, orbitSpeed: 0.08, texture: '/textures/jupiter.jpg' },
  { name: 'Saturn', color: 0xead6b8, size: 2.0, distanceFromSun: 550, orbitSpeed: 0.05, texture: '/textures/saturn.jpg', hasRings: true, ringColor: 0xc9b896, ringTexture: '/textures/saturn_ring.png' },
  { name: 'Uranus', color: 0xd1e7e7, size: 1.2, distanceFromSun: 650, orbitSpeed: 0.03, texture: '/textures/uranus.jpg' },
  { name: 'Neptune', color: 0x5b5ddf, size: 1.1, distanceFromSun: 750, orbitSpeed: 0.02, texture: '/textures/neptune.jpg' },
];

class Planet extends THREE.Group {
  private sphere: THREE.Mesh;
  private label: THREE.Sprite;
  private orbitAngle: number;
  private distanceFromSun: number;
  private orbitSpeed: number;
  public planetName: string;

  constructor(data: PlanetData) {
    super();

    this.planetName = data.name;
    this.orbitAngle = Math.random() * Math.PI * 2; // Random starting position
    this.distanceFromSun = data.distanceFromSun;
    this.orbitSpeed = data.orbitSpeed;

    const textureLoader = new THREE.TextureLoader();

    // Planet sphere
    const geometry = new THREE.SphereGeometry(data.size, 32, 32);

    // Load texture if available, otherwise fall back to color
    let material: THREE.MeshPhongMaterial;
    if (data.texture) {
      const texture = textureLoader.load(data.texture);
      texture.colorSpace = THREE.SRGBColorSpace;
      material = new THREE.MeshPhongMaterial({
        map: texture,
        shininess: 10,
      });
    } else {
      material = new THREE.MeshPhongMaterial({
        color: data.color,
        shininess: 10,
      });
    }

    this.sphere = new THREE.Mesh(geometry, material);
    this.add(this.sphere);

    // Add rings for Saturn
    if (data.hasRings) {
      const ringGeometry = new THREE.RingGeometry(
        data.size * 1.4,
        data.size * 2.2,
        64
      );

      let ringMaterial: THREE.MeshBasicMaterial;
      if (data.ringTexture) {
        const ringTexture = textureLoader.load(data.ringTexture);
        ringTexture.colorSpace = THREE.SRGBColorSpace;
        // Rotate UV coordinates for ring (the texture is horizontal)
        const pos = ringGeometry.attributes.position;
        const uv = ringGeometry.attributes.uv;
        for (let i = 0; i < uv.count; i++) {
          const x = pos.getX(i);
          const y = pos.getY(i);
          const distance = Math.sqrt(x * x + y * y);
          // Map distance to U coordinate (0 = inner, 1 = outer)
          const u = (distance - data.size * 1.4) / (data.size * 0.8);
          uv.setXY(i, u, 0.5);
        }
        ringMaterial = new THREE.MeshBasicMaterial({
          map: ringTexture,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.9,
        });
      } else {
        ringMaterial = new THREE.MeshBasicMaterial({
          color: data.ringColor,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.7,
        });
      }
      const rings = new THREE.Mesh(ringGeometry, ringMaterial);
      rings.rotation.x = Math.PI / 2.5;
      this.add(rings);
    }

    // Create label
    this.label = this.createLabel(data.name);
    this.label.position.y = data.size + 1.5;
    this.add(this.label);

    // Initial position
    this.updatePosition();
  }

  private createLabel(text: string): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 32;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, 128, 32);

    ctx.font = '16px Arial';
    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.fillText(text, 64, 22);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(4, 1, 1);
    return sprite;
  }

  private updatePosition(): void {
    // Orbit around Sun at (SUN_X, 0, 0)
    // x position: SUN_X - distance * cos(angle)
    // When angle=0, planet is directly between Sun and Earth (for inner planets)
    // When angle=PI, planet is on opposite side of Sun from Earth
    this.position.x = SUN_X - Math.cos(this.orbitAngle) * this.distanceFromSun;
    this.position.z = Math.sin(this.orbitAngle) * this.distanceFromSun;
    this.position.y = Math.sin(this.orbitAngle * 2) * 3; // Slight orbital inclination
  }

  update(delta: number): void {
    this.orbitAngle += delta * this.orbitSpeed;
    this.updatePosition();

    // Planet rotation
    this.sphere.rotation.y += delta * 0.5;
  }
}

export class Planets extends THREE.Group {
  private planets: Planet[] = [];
  private orbitLines: THREE.Line[] = [];
  private earthPlanet: Planet | null = null;

  constructor(sunPosition: THREE.Vector3 = new THREE.Vector3(SUN_X, 0, 0)) {
    super();

    // Create planets
    for (const data of PLANETS) {
      const planet = new Planet(data);
      this.planets.push(planet);
      this.add(planet);

      // Keep reference to Earth for position tracking
      if (data.name === 'Earth') {
        this.earthPlanet = planet;
      }

      // Create orbit line
      const orbitLine = this.createOrbitLine(data.distanceFromSun);
      this.orbitLines.push(orbitLine);
      this.add(orbitLine);
    }

    // Hidden by default (shown in solar view)
    this.visible = false;
  }

  private createOrbitLine(distanceFromSun: number): THREE.Line {
    const points: THREE.Vector3[] = [];
    const segments = 128;

    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      points.push(new THREE.Vector3(
        SUN_X - Math.cos(angle) * distanceFromSun,
        0,
        Math.sin(angle) * distanceFromSun
      ));
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0x333355,
      transparent: true,
      opacity: 0.3,
    });

    return new THREE.Line(geometry, material);
  }

  private createEarthLabel(): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 32;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'rgba(0, 50, 100, 0.7)';
    ctx.fillRect(0, 0, 128, 32);

    ctx.font = 'bold 16px Arial';
    ctx.fillStyle = '#4488ff';
    ctx.textAlign = 'center';
    ctx.fillText('Earth', 64, 22);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(4, 1, 1);
    sprite.position.set(0, 2, 0); // Above Earth at origin
    return sprite;
  }

  update(delta: number): void {
    for (const planet of this.planets) {
      planet.update(delta);
    }
  }

  setVisible(visible: boolean): void {
    this.visible = visible;
  }

  /**
   * Get Earth's current orbital position in the solar system view.
   * Returns position relative to scene origin (not Sun).
   */
  getEarthPosition(): THREE.Vector3 {
    if (this.earthPlanet) {
      return this.earthPlanet.position.clone();
    }
    // Fallback to origin if Earth not found
    return new THREE.Vector3(0, 0, 0);
  }
}

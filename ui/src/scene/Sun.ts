/**
 * Sun visualization with NASA-based textures
 * Textures from Solar System Scope (based on NASA imagery)
 */

import * as THREE from 'three';

export class Sun extends THREE.Group {
  private sphere: THREE.Mesh;
  private glowSprite: THREE.Sprite;
  private coronaSprite: THREE.Sprite;
  private time: number = 0;

  constructor() {
    super();

    const textureLoader = new THREE.TextureLoader();

    // Load sun texture
    const sunTexture = textureLoader.load('/textures/sun.jpg');
    sunTexture.colorSpace = THREE.SRGBColorSpace;

    // Sun sphere
    const geometry = new THREE.SphereGeometry(5, 64, 64);
    const material = new THREE.MeshBasicMaterial({
      map: sunTexture,
    });

    this.sphere = new THREE.Mesh(geometry, material);
    this.add(this.sphere);

    // Inner glow sprite
    const glowTexture = this.createGlowTexture();
    const glowMaterial = new THREE.SpriteMaterial({
      map: glowTexture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.glowSprite = new THREE.Sprite(glowMaterial);
    this.glowSprite.scale.set(25, 25, 1);
    this.add(this.glowSprite);

    // Outer corona sprite
    const coronaTexture = this.createCoronaTexture();
    const coronaMaterial = new THREE.SpriteMaterial({
      map: coronaTexture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.coronaSprite = new THREE.Sprite(coronaMaterial);
    this.coronaSprite.scale.set(50, 50, 1);
    this.add(this.coronaSprite);

    // Position sun far away
    this.position.set(200, 0, 0);
  }

  private createGlowTexture(): THREE.Texture {
    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;

    const gradient = ctx.createRadialGradient(
      size / 2, size / 2, 0,
      size / 2, size / 2, size / 2
    );
    gradient.addColorStop(0, 'rgba(255, 255, 220, 1)');
    gradient.addColorStop(0.1, 'rgba(255, 240, 180, 0.9)');
    gradient.addColorStop(0.3, 'rgba(255, 200, 100, 0.6)');
    gradient.addColorStop(0.5, 'rgba(255, 150, 50, 0.3)');
    gradient.addColorStop(0.7, 'rgba(255, 100, 20, 0.1)');
    gradient.addColorStop(1, 'rgba(255, 50, 0, 0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
  }

  private createCoronaTexture(): THREE.Texture {
    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;

    const centerX = size / 2;
    const centerY = size / 2;

    // Draw corona rays
    const numRays = 24;
    for (let i = 0; i < numRays; i++) {
      const angle = (i / numRays) * Math.PI * 2;
      const rayLength = size / 2 * (0.6 + Math.random() * 0.4);
      const rayWidth = 0.1 + Math.random() * 0.15;

      const gradient = ctx.createRadialGradient(
        centerX, centerY, size * 0.1,
        centerX, centerY, rayLength
      );
      gradient.addColorStop(0, 'rgba(255, 200, 100, 0.3)');
      gradient.addColorStop(0.3, 'rgba(255, 150, 50, 0.15)');
      gradient.addColorStop(0.6, 'rgba(255, 100, 30, 0.05)');
      gradient.addColorStop(1, 'rgba(255, 50, 0, 0)');

      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(angle);
      ctx.scale(1, rayWidth);
      ctx.translate(-centerX, -centerY);

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, rayLength, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Add central gradient
    const centralGradient = ctx.createRadialGradient(
      centerX, centerY, 0,
      centerX, centerY, size / 3
    );
    centralGradient.addColorStop(0, 'rgba(255, 220, 150, 0.4)');
    centralGradient.addColorStop(0.5, 'rgba(255, 150, 80, 0.15)');
    centralGradient.addColorStop(1, 'rgba(255, 100, 50, 0)');

    ctx.fillStyle = centralGradient;
    ctx.fillRect(0, 0, size, size);

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
  }

  update(delta: number): void {
    this.time += delta;

    // Slow sun rotation
    this.sphere.rotation.y += delta * 0.02;

    // Subtle corona rotation
    this.coronaSprite.material.rotation += delta * 0.01;
  }

  // Set position for different views
  setSolarSystemPosition(): void {
    this.position.set(200, 0, 0);
    this.sphere.scale.setScalar(1);
    this.glowSprite.scale.set(25, 25, 1);
    this.coronaSprite.scale.set(50, 50, 1);
  }

  setEarthCentricPosition(): void {
    this.position.set(150, 20, 80);
    this.sphere.scale.setScalar(0.6);
    this.glowSprite.scale.set(15, 15, 1);
    this.coronaSprite.scale.set(30, 30, 1);
  }
}

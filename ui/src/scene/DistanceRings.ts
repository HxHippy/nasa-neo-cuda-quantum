/**
 * Distance rings around Earth showing lunar distances
 */

import * as THREE from 'three';

export class DistanceRings extends THREE.Group {
  private rings: THREE.Line[] = [];
  private labels: THREE.Sprite[] = [];

  constructor() {
    super();

    // Create rings at various lunar distances
    const distances = [1, 5, 10, 25, 50, 100];

    for (const ld of distances) {
      this.createRing(ld);
    }
  }

  private createRing(lunarDistances: number): void {
    // 1 LD = 10 units in scene
    const radius = lunarDistances;

    const segments = 128;
    const points = [];

    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      points.push(new THREE.Vector3(
        Math.cos(theta) * radius,
        0,
        Math.sin(theta) * radius
      ));
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);

    // Color based on distance (closer = more red)
    let color: number;
    if (lunarDistances <= 1) {
      color = 0xff0000;
    } else if (lunarDistances <= 5) {
      color = 0xff6600;
    } else if (lunarDistances <= 10) {
      color = 0xffff00;
    } else {
      color = 0x444466;
    }

    const material = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: lunarDistances <= 10 ? 0.5 : 0.2,
    });

    const ring = new THREE.Line(geometry, material);
    this.rings.push(ring);
    this.add(ring);

    // Add label
    const label = this.createLabel(`${lunarDistances} LD`);
    label.position.set(radius + 1, 0, 0);
    this.labels.push(label);
    this.add(label);
  }

  private createLabel(text: string): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 32;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 16);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(4, 1, 1);

    return sprite;
  }

  update(_delta: number): void {
    // Rings are static
  }

  setVisible(visible: boolean): void {
    this.visible = visible;
  }
}

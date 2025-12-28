/**
 * Three.js Scene Manager
 *
 * Handles renderer, scene, cameras, and animation loop
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export type ViewMode = 'earth' | 'solar';

export class SceneManager {
  public scene: THREE.Scene;
  public camera: THREE.PerspectiveCamera;
  public renderer: THREE.WebGLRenderer;
  public controls: OrbitControls;
  public raycaster: THREE.Raycaster;
  public mouse: THREE.Vector2;

  private container: HTMLElement;
  private animationId: number | null = null;
  private clock: THREE.Clock;
  private viewMode: ViewMode = 'earth';

  // Camera positions for different views
  // Earth view: closer, looking at Earth with NEOs visible
  private earthViewPosition = new THREE.Vector3(0, 15, 40);
  private earthViewTarget = new THREE.Vector3(0, 0, 0);
  // Solar view: far out to see full solar system centered on Sun (at x=200)
  private solarViewPosition = new THREE.Vector3(200, 600, 800);
  private solarViewTarget = new THREE.Vector3(200, 0, 0); // Sun position

  // Callbacks
  private onRenderCallbacks: Array<(delta: number) => void> = [];
  private onClickCallback: ((object: THREE.Object3D | null) => void) | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.clock = new THREE.Clock();
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000008);

    // Create camera
    const aspect = container.clientWidth / container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 10000);
    this.camera.position.copy(this.earthViewPosition);

    // Create renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    container.appendChild(this.renderer.domElement);

    // Create controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 2000;

    // Add lights
    this.setupLights();

    // Event listeners
    window.addEventListener('resize', this.onResize.bind(this));
    this.renderer.domElement.addEventListener('click', this.onClick.bind(this));
    this.renderer.domElement.addEventListener('mousemove', this.onMouseMove.bind(this));
  }

  private setupLights(): void {
    // Ambient light for base visibility
    const ambient = new THREE.AmbientLight(0x303040, 0.4);
    this.scene.add(ambient);

    // Directional light (from Sun direction)
    const sunLight = new THREE.DirectionalLight(0xfff8e8, 2.5);
    sunLight.position.set(200, 10, 0); // Sun is at x=200
    this.scene.add(sunLight);

    // Subtle fill light from opposite side
    const fillLight = new THREE.DirectionalLight(0x6080a0, 0.3);
    fillLight.position.set(-100, 0, 50);
    this.scene.add(fillLight);

    // Hemisphere light for ambient space lighting
    const hemi = new THREE.HemisphereLight(0x404060, 0x101020, 0.2);
    this.scene.add(hemi);
  }

  private onResize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  private onClick(event: MouseEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);

    // Find clickable objects (NEOs)
    const clickable = this.scene.children.filter(
      (obj) => obj.userData.clickable === true
    );
    const intersects = this.raycaster.intersectObjects(clickable, true);

    if (intersects.length > 0) {
      let target = intersects[0].object;
      // Walk up to find parent with neo data
      while (target.parent && !target.userData.neoId) {
        target = target.parent;
      }
      this.onClickCallback?.(target);
    } else {
      this.onClickCallback?.(null);
    }
  }

  private onMouseMove(event: MouseEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  setViewMode(mode: ViewMode): void {
    this.viewMode = mode;

    const targetPosition = mode === 'earth'
      ? this.earthViewPosition
      : this.solarViewPosition;

    const lookAtTarget = mode === 'earth'
      ? this.earthViewTarget
      : this.solarViewTarget;

    // Animate camera transition
    this.animateCameraTo(targetPosition, lookAtTarget);
  }

  private animateCameraTo(targetPos: THREE.Vector3, targetLookAt: THREE.Vector3): void {
    const startPos = this.camera.position.clone();
    const startLookAt = this.controls.target.clone();
    const duration = 1000;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);

      this.camera.position.lerpVectors(startPos, targetPos, eased);
      this.controls.target.lerpVectors(startLookAt, targetLookAt, eased);
      this.controls.update();

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }

  // Public method to animate camera to any target
  animateCameraToTarget(cameraPos: THREE.Vector3, lookAtPos: THREE.Vector3): void {
    this.animateCameraTo(cameraPos, lookAtPos);
  }

  onRender(callback: (delta: number) => void): void {
    this.onRenderCallbacks.push(callback);
  }

  onObjectClick(callback: (object: THREE.Object3D | null) => void): void {
    this.onClickCallback = callback;
  }

  start(): void {
    if (this.animationId !== null) return;

    const animate = () => {
      this.animationId = requestAnimationFrame(animate);

      const delta = this.clock.getDelta();

      // Update controls
      this.controls.update();

      // Call render callbacks
      for (const callback of this.onRenderCallbacks) {
        callback(delta);
      }

      // Render
      this.renderer.render(this.scene, this.camera);
    };

    animate();
  }

  stop(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  dispose(): void {
    this.stop();
    window.removeEventListener('resize', this.onResize.bind(this));
    this.renderer.dispose();
    this.controls.dispose();
  }
}

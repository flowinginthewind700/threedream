/**
 * three.js rendering bridge.
 *
 * Keeps the renderer as a *consumer* of simulation state, never a source of
 * truth. The physics world owns positions; this layer mirrors them into
 * `THREE.Object3D`s each frame. That separation is what lets the same scene run
 * headless (no WebGL) for training and CI, and rendered in the browser for play.
 */

import * as THREE from 'three';
import type { PhysicsBackend } from '../physics/types.js';

export interface RendererOptions {
  container: HTMLElement;
  backend: PhysicsBackend;
  /** physics body handle -> mesh */
  visualFor?: Map<number, THREE.Object3D>;
  antialias?: boolean;
  /** Cap the pixel ratio; high-DPI phones otherwise cost 4x fragments. */
  maxPixelRatio?: number;
  background?: number | THREE.Color;
  shadows?: boolean;
}

export interface VisualBinding {
  handle: number;
  object: THREE.Object3D;
  /** Render scale applied to physics-space positions. */
  scale?: number;
}

export class ThreeRenderer {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;

  private readonly backend: PhysicsBackend;
  private readonly bindings: VisualBinding[] = [];
  private readonly visualFor: Map<number, THREE.Object3D>;
  private frameId = 0;
  private readonly resizeObserver?: ResizeObserver;

  constructor(options: RendererOptions) {
    const container = options.container;
    this.backend = options.backend;
    this.visualFor = options.visualFor ?? new Map();

    this.scene = new THREE.Scene();
    this.scene.background =
      options.background instanceof THREE.Color
        ? options.background
        : new THREE.Color(options.background ?? 0x0b1020);
    this.scene.fog = new THREE.Fog(0x0b1020, 12, 34);

    this.camera = new THREE.PerspectiveCamera(
      55,
      Math.max(1, container.clientWidth) / Math.max(1, container.clientHeight),
      0.1,
      200,
    );
    this.camera.position.set(0, 3.2, 5.4);
    this.camera.lookAt(0, 0.4, 0);

    this.renderer = new THREE.WebGLRenderer({
      antialias: options.antialias ?? true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(
      Math.min(window.devicePixelRatio || 1, options.maxPixelRatio ?? 2),
    );
    this.renderer.setSize(
      Math.max(1, container.clientWidth),
      Math.max(1, container.clientHeight),
      false,
    );
    this.renderer.shadowMap.enabled = options.shadows ?? true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.renderer.domElement.style.display = 'block';
    container.appendChild(this.renderer.domElement);

    this.addDefaultLights();

    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.resize(container));
      this.resizeObserver.observe(container);
    } else {
      window.addEventListener('resize', () => this.resize(container));
    }
  }

  private addDefaultLights(): void {
    const hemi = new THREE.HemisphereLight(0xbcd4ff, 0x223047, 0.7);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(4, 7, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 30;
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 0.5);
    rim.position.set(-5, 3, -4);
    this.scene.add(rim);
  }

  /** Bind a physics body to a scene object so it is mirrored every frame. */
  bind(binding: VisualBinding): void {
    this.bindings.push(binding);
    this.visualFor.set(binding.handle, binding.object);
    this.scene.add(binding.object);
  }

  add(object: THREE.Object3D): void {
    this.scene.add(object);
  }

  /** Copy physics transforms into the scene graph. */
  syncFromPhysics(): void {
    for (const binding of this.bindings) {
      const state = this.backend.getBodyState(binding.handle);
      if (!state) continue;
      const scale = binding.scale ?? 1;
      const [x, y, z] = state.position;
      binding.object.position.set(x * scale, y * scale, z * scale);
      const [rx, ry, rz] = state.rotation;
      binding.object.rotation.set(rx, ry, rz);
    }
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  /** Render one frame with physics sync. Call from your rAF loop. */
  frame(): void {
    this.syncFromPhysics();
    this.render();
  }

  start(onFrame?: () => void): void {
    const loop = () => {
      this.frameId = requestAnimationFrame(loop);
      onFrame?.();
      this.frame();
    };
    this.frameId = requestAnimationFrame(loop);
  }

  stop(): void {
    if (this.frameId) cancelAnimationFrame(this.frameId);
    this.frameId = 0;
  }

  private resize(container: HTMLElement): void {
    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  dispose(): void {
    this.stop();
    this.resizeObserver?.disconnect();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    for (const binding of this.bindings) this.scene.remove(binding.object);
    this.bindings.length = 0;
  }
}

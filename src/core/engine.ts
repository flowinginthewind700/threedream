/**
 * Engine facade.
 *
 * One object that owns the fixed-timestep clock, the ECS world, the event bus
 * and the physics backend, and runs them in the correct order. Callers do
 * `engine.frame(realDt)` from a render loop, or `engine.step(n)` from headless
 * code; both produce identical simulation results for identical inputs.
 */

import { FixedClock, type FixedClockOptions } from './clock.js';
import { EventBus } from './events.js';
import { World } from './ecs.js';
import { BuiltinPhysics } from '../physics/builtin.js';
import type { PhysicsBackend, PhysicsWorldOptions, Vec3 } from '../physics/types.js';
import { vec3 } from '../physics/types.js';
import { PhysicsSystem, type PhysicsEvents } from '../physics/components.js';

export type EngineEvents = PhysicsEvents & {
  'engine:started': { fixedDt: number };
  'engine:paused': undefined;
  'engine:resumed': undefined;
};

export interface EngineOptions {
  physics: PhysicsBackend;
  clock?: FixedClockOptions;
  /** Attach the physics system automatically. Default true. */
  withPhysicsSystem?: boolean;
  bus?: EventBus<EngineEvents>;
}

export interface EngineStats {
  fixedDt: number;
  steps: number;
  simulationTime: number;
  entities: number;
  bodies: number;
  systems: string[];
}

export class Engine {
  readonly world = new World();
  readonly clock: FixedClock;
  readonly bus: EventBus<EngineEvents>;
  readonly physics: PhysicsBackend;
  readonly physicsSystem?: PhysicsSystem;

  private running = false;
  private lastFrameTime = 0;
  private rafId = 0;

  constructor(options: EngineOptions) {
    this.physics = options.physics;
    this.bus = options.bus ?? new EventBus<EngineEvents>();
    this.clock = new FixedClock({
      fixedDt: options.physics.fixedDt,
      ...options.clock,
    });
    if (options.withPhysicsSystem !== false) {
      this.physicsSystem = new PhysicsSystem({
        backend: options.physics,
        emitter: this.bus.asEmitter(),
      });
      this.world.addSystem(this.physicsSystem);
    }
  }

  get fixedDt(): number {
    return this.clock.fixedDt;
  }

  get isRunning(): boolean {
    return this.running;
  }

  /** Run exactly `count` fixed steps. Deterministic; used by headless training. */
  step(count = 1): void {
    for (let i = 0; i < count; i++) {
      this.world.step(this.clock.fixedDt);
      this.clock.advance(1);
    }
  }

  /**
   * Consume real elapsed time and run however many fixed steps that implies.
   * Returns the number of steps taken and the interpolation alpha.
   */
  frame(realDt: number): { steps: number; alpha: number } {
    const steps = this.clock.update(realDt);
    for (let i = 0; i < steps; i++) this.world.step(this.clock.fixedDt);
    return { steps, alpha: this.clock.alpha };
  }

  /** Drive the engine from `requestAnimationFrame`. No-op in non-DOM hosts. */
  start(): void {
    if (this.running || typeof requestAnimationFrame === 'undefined') return;
    this.running = true;
    this.lastFrameTime = performance.now();
    this.bus.emit('engine:started', { fixedDt: this.fixedDt });
    const loop = (now: number) => {
      if (!this.running) return;
      const dtSeconds = Math.max(0, (now - this.lastFrameTime) / 1000);
      this.lastFrameTime = now;
      this.frame(dtSeconds);
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  pause(): void {
    if (!this.running) return;
    this.running = false;
    if (this.rafId && typeof cancelAnimationFrame !== 'undefined') {
      cancelAnimationFrame(this.rafId);
    }
    this.rafId = 0;
    this.bus.emit('engine:paused', undefined);
  }

  resume(): void {
    if (this.running) return;
    this.lastFrameTime =
      typeof performance !== 'undefined' ? performance.now() : 0;
    this.bus.emit('engine:resumed', undefined);
    this.start();
  }

  /** Convenience: build a static box body and return its handle. */
  addStaticBox(position: Vec3, halfExtents: Vec3, label?: string): number {
    return this.physics.createBody({
      shape: { kind: 'box', halfExtents },
      position,
      kind: 'static',
      friction: 0.8,
      label: label ?? 'static-box',
    });
  }

  addGround(size = 20, thickness = 0.5, y = 0): number {
    return this.addStaticBox(
      vec3(0, y - thickness / 2, 0),
      vec3(size / 2, thickness / 2, size / 2),
      'ground',
    );
  }

  stats(): EngineStats {
    return {
      fixedDt: this.fixedDt,
      steps: this.clock.steps,
      simulationTime: this.clock.time,
      entities: this.world.entityCount,
      bodies: this.physics.bodyCount,
      systems: this.world.systemNames,
    };
  }

  dispose(options: { disposePhysics?: boolean } = {}): void {
    this.pause();
    this.world.clear();
    if (options.disposePhysics !== false) this.physics.dispose();
  }
}

export interface CreateEngineOptions {
  gravity?: Vec3;
  fixedDt?: number;
  physics?: PhysicsBackend;
  physicsOptions?: PhysicsWorldOptions;
}

/**
 * Build an engine on the built-in deterministic solver.
 *
 * For the Rapier backend use `createRapierPhysics()` (async, loads WASM) and
 * pass the result as `physics`. Keeping this factory synchronous means headless
 * scripts and tests never need to await WASM initialisation.
 */
export function createEngine(options: CreateEngineOptions = {}): Engine {
  const fixedDt = options.fixedDt ?? 1 / 60;
  const physics =
    options.physics ??
    new BuiltinPhysics({
      gravity: options.gravity ?? vec3(0, -9.81, 0),
      fixedDt,
      ...options.physicsOptions,
    });
  return new Engine({ physics });
}

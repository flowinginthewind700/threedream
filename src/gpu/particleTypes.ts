/**
 * The contract every particle backend satisfies, and the knobs they share.
 *
 * Type-only on purpose: `tests/tdd.test.ts` pins that a `types.ts` module carries
 * no runtime code, because a contract that grows behaviour needs a spec of its
 * own. Keeping the interface here rather than in `particles.ts` is what lets
 * `particleCpu.ts` and `particleGpu.ts` both declare `implements ParticleSystem`
 * without importing the factory that imports them.
 */

import type { Bounds, ParticleField, Vec3Tuple } from './particleField.js';

/** How a particle that reaches the box is handled. */
export type BoundsMode = 'reflect' | 'wrap' | 'none';

export interface ParticleSimOptions {
  /** Uniform acceleration, units/s^2. `[0, -9.81, 0]` by default. */
  gravity?: Vec3Tuple;
  /** Linear damping factor per second, applied as `v *= 1 - damping*dt`. */
  damping?: number;
  /** Coefficient of restitution for bounds and particle contacts. */
  restitution?: number;
  /** Speed clamp. Keeps a bad frame from turning into an escaped particle. */
  maxSpeed?: number;
  /** Sphere-sphere contacts through the spatial hash. On by default. */
  collisions?: boolean;
  /** All-pairs gravity. Off by default: it is O(n^2) and the demo says so. */
  nbody?: boolean;
  /** Gravitational constant for the n-body pass. */
  nbodyStrength?: number;
  /** Plummer softening, so a close pair cannot produce an infinite force. */
  softening?: number;
  /** Interaction cutoff radius. `0` means all pairs. */
  cutoff?: number;
  boundsMode?: BoundsMode;
  /** Broadphase cell edge. Defaults to twice the largest radius. */
  cellSize?: number;
  /** Slots per hash bucket. Overflow is counted, never silent. */
  bucketCapacity?: number;
  /** Simulation step. The renderer is decoupled from it; see `particles.ts`. */
  fixedDt?: number;
}

/** Options after defaults and validation, so kernels never re-check them. */
export interface ResolvedParticleOptions {
  readonly gravity: Vec3Tuple;
  readonly damping: number;
  readonly restitution: number;
  readonly maxSpeed: number;
  readonly collisions: boolean;
  readonly nbody: boolean;
  readonly nbodyStrength: number;
  readonly softening: number;
  readonly cutoff: number;
  readonly boundsMode: BoundsMode;
  readonly cellSize: number;
  readonly bucketCapacity: number;
  readonly fixedDt: number;
}

/** Per-step instrumentation. The demo HUD and the specs read the same struct. */
export interface ParticleStepStats {
  /** Contact pairs resolved this step. */
  readonly contacts: number;
  /** Particles whose centre left the box. Must stay 0 under `reflect`. */
  readonly escaped: number;
  /** Hash insertions that found a full bucket. */
  readonly hashOverflow: number;
  readonly maxSpeed: number;
  readonly kineticEnergy: number;
}

/**
 * A particle simulation the rest of the engine can drive without knowing which
 * device, if any, is running it.
 *
 * `deterministic` is the honest flag from the plan's layering rule: the CPU
 * backend is bit-reproducible and is what replay and training use; the GPU
 * backend is not, because its broadphase fills buckets with atomics and its
 * floating-point reduction order belongs to the driver.
 */
export interface ParticleSystem {
  readonly name: string;
  readonly deterministic: boolean;
  readonly fixedDt: number;
  readonly field: ParticleField;
  readonly options: ResolvedParticleOptions;
  readonly count: number;
  readonly bounds: Bounds;
  /** Simulation steps taken. Independent of frames rendered. */
  readonly steps: number;
  /** Simulated seconds, i.e. `steps * fixedDt`. */
  readonly time: number;

  /** Advance one fixed step. `dt` defaults to `fixedDt`. */
  step(dt?: number): void;
  /** Advance exactly `n` steps, ignoring wall-clock time. Headless path. */
  advance(n: number): void;
  stats(): ParticleStepStats;
  /** `hex:count` over the raw f32 state bytes. */
  digest(): string;
  /** Release device or heap resources. Idempotent. */
  dispose(): void;
}

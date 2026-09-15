/**
 * The CPU particle backend: the reference every GPU kernel is measured against.
 *
 * # Why a CPU backend exists at all
 *
 * The plan's M3 acceptance criteria say two things that pull in opposite
 * directions: "stable 50k-100k particles on the target hardware" and "replay and
 * training must not depend on the GPU layer". The way out is that the CPU path is
 * not a slow fallback that happens to exist -- it is the *specification*. The
 * WebGPU kernels and this file compute the same function in the same order, so a
 * parity test can compare digests instead of eyeballing a rendered frame, and a
 * machine with no usable GPU still runs the same simulation rather than a
 * reduced one.
 *
 * # Why f32 arithmetic
 *
 * WGSL has no f64. If this file computed in f64 and stored to a `Float32Array`,
 * every intermediate would differ from the shader's by up to one rounding, and
 * after a few hundred steps of a chaotic system the two runs would share nothing.
 * So every arithmetic step is `Math.fround`-ed, in the same association the WGSL
 * uses. That is verbose on purpose: the verbosity *is* the contract, and
 * `tests/particle_cpu.test.ts` pins the digest.
 *
 * # Why the pass structure mirrors the dispatch chain
 *
 * One `step()` here is exactly the GPU's `[nbody] -> [hash, collide] -> integrate`
 * chain, with the same intermediate buffers (`accel`, `dv`, `dx`). A bug found on
 * one side can be reproduced on the other by running the same pass, which is the
 * whole point of keeping them structurally identical rather than writing the CPU
 * version in whatever shape is most natural for scalar code.
 */

import {
  PARTICLE_OFFSET,
  PARTICLE_STRIDE,
  type Bounds,
  type ParticleField,
} from './particleField.js';
import { SpatialHash, type SpatialHashStats } from './particleHash.js';
import {
  assertFieldFits,
  effectiveCellSize,
  resolveParticleOptions,
} from './particleOptions.js';
import type {
  ParticleSimOptions,
  ParticleStepStats,
  ParticleSystem,
  ResolvedParticleOptions,
} from './particleTypes.js';

/** Shorthand, so the op order in the hot loops stays readable. */
const f = Math.fround;

/**
 * Fraction of a contact's overlap removed per step.
 *
 * 1.0 separates two spheres in a single step and then overshoots on the next one
 * if both are also being pushed by gravity, which reads as jitter in a settled
 * pile. 0.5 converges in a few steps and is stable under the default gravity.
 * The GPU kernel uses the same constant; changing it here without changing it
 * there breaks the parity test rather than the picture.
 */
export const POSITION_CORRECTION = 0.5;

export interface CpuParticleSystemOptions {
  field: ParticleField;
  options?: ParticleSimOptions;
}

export function createCpuParticleSystem(options: CpuParticleSystemOptions): CpuParticleSystem {
  return new CpuParticleSystem(options);
}

export class CpuParticleSystem implements ParticleSystem {
  readonly name = 'cpu';
  /** Bit-reproducible: same field, same options, same bytes. */
  readonly deterministic = true;
  readonly field: ParticleField;
  readonly options: ResolvedParticleOptions;
  /** Broadphase cell edge actually in use, i.e. what the uniform carries too. */
  readonly cellSize: number;
  readonly hash: SpatialHash;

  /** Per-particle acceleration from the n-body pass. Empty when disabled. */
  private readonly accel: Float32Array;
  /** Per-particle velocity impulse from contacts. Empty when disabled. */
  private readonly dv: Float32Array;
  /** Per-particle positional correction from contacts. Empty when disabled. */
  private readonly dx: Float32Array;

  private stepsTaken = 0;
  private lastContacts = 0;
  private lastEscaped = 0;
  private lastHashOverflow = 0;
  private disposed = false;

  /** Index the collision visitor is currently resolving candidates for. */
  private visitorI = 0;
  private readonly visitor: (j: number) => void;

  constructor({ field, options }: CpuParticleSystemOptions) {
    if (!field || !Number.isInteger(field.count) || !(field.count > 0)) {
      throw new TypeError('CpuParticleSystem needs a ParticleField with a positive count');
    }
    this.field = field;
    this.options = resolveParticleOptions(options);
    assertFieldFits(field);
    this.cellSize = effectiveCellSize(field, this.options);
    this.hash = SpatialHash.forField(field, {
      cellSize: this.cellSize,
      bucketCapacity: this.options.bucketCapacity,
    });
    const n3 = field.count * 3;
    this.accel = this.options.nbody ? new Float32Array(n3) : new Float32Array(0);
    this.dv = this.options.collisions ? new Float32Array(n3) : new Float32Array(0);
    this.dx = this.options.collisions ? new Float32Array(n3) : new Float32Array(0);
    // Bound once: `forEachCandidate` is called per particle per step, and a
    // fresh closure each time is 100k allocations per step at the target scale.
    this.visitor = (j: number): void => this.resolveContact(this.visitorI, j);
  }

  get fixedDt(): number {
    return this.options.fixedDt;
  }

  get count(): number {
    return this.field.count;
  }

  get bounds(): Bounds {
    return this.field.bounds;
  }

  get steps(): number {
    return this.stepsTaken;
  }

  get time(): number {
    return this.stepsTaken * this.options.fixedDt;
  }

  step(dt: number = this.options.fixedDt): void {
    this.assertLive();
    if (!Number.isFinite(dt) || !(dt > 0)) {
      throw new RangeError(`dt must be finite and positive, got ${dt}`);
    }
    if (this.options.nbody) this.nbodyPass();
    if (this.options.collisions) this.collidePass();
    this.lastEscaped = this.integratePass(dt);
    this.stepsTaken++;
  }

  advance(n: number): void {
    if (!Number.isInteger(n) || n < 0) {
      throw new RangeError(`advance needs a non-negative integer, got ${n}`);
    }
    for (let k = 0; k < n; k++) this.step();
  }

  stats(): ParticleStepStats {
    return {
      contacts: this.lastContacts,
      escaped: this.lastEscaped,
      hashOverflow: this.lastHashOverflow,
      maxSpeed: this.field.maxSpeed(),
      kineticEnergy: this.field.kineticEnergy(),
    };
  }

  hashStats(): SpatialHashStats {
    return this.hash.stats();
  }

  digest(): string {
    return this.field.digest();
  }

  dispose(): void {
    // Nothing to free: the scratch arrays are owned by this object and the field
    // belongs to the caller. Idempotent, and it makes `step()` after `dispose()`
    // a loud error instead of a silent no-op on a half-torn-down sim.
    this.disposed = true;
  }

  private assertLive(): void {
    if (this.disposed) throw new Error('CpuParticleSystem has been disposed');
  }

  /**
   * All-pairs gravity, O(n^2).
   *
   * Off by default and only ever enabled in the demo for a few thousand
   * particles: this is the pass that motivates a real GPU, and pretending
   * otherwise at 100k would make the benchmark meaningless. Written as a
   * straight double loop in particle order so the accumulation order matches the
   * WGSL kernel's inner loop.
   */
  private nbodyPass(): void {
    const { data, count } = this.field;
    const opts = this.options;
    const strength = f(opts.nbodyStrength);
    const soft2 = f(f(opts.softening) * f(opts.softening));
    const cutoff2 = opts.cutoff > 0 ? f(f(opts.cutoff) * f(opts.cutoff)) : 0;
    const accel = this.accel;
    for (let i = 0; i < count; i++) {
      const oi = i * PARTICLE_STRIDE;
      const xi = data[oi]!;
      const yi = data[oi + 1]!;
      const zi = data[oi + 2]!;
      let ax = 0;
      let ay = 0;
      let az = 0;
      for (let j = 0; j < count; j++) {
        if (j === i) continue;
        const oj = j * PARTICLE_STRIDE;
        const dxc = f(data[oj]! - xi);
        const dyc = f(data[oj + 1]! - yi);
        const dzc = f(data[oj + 2]! - zi);
        const d2 = f(f(f(dxc * dxc) + f(dyc * dyc)) + f(dzc * dzc));
        if (cutoff2 > 0 && d2 > cutoff2) continue;
        const mj = data[oj + PARTICLE_OFFSET.mass]!;
        // Plummer softening: the 3/2 power is what turns `d / |d|^3` into a form
        // that stays finite at d = 0, and `1 / (d2 + soft2)^1.5` needs no sqrt
        // of a separately computed length.
        const inv = f(f(strength * mj) / f(Math.pow(f(d2 + soft2), 1.5)));
        ax = f(ax + f(dxc * inv));
        ay = f(ay + f(dyc * inv));
        az = f(az + f(dzc * inv));
      }
      const o3 = i * 3;
      accel[o3] = ax;
      accel[o3 + 1] = ay;
      accel[o3 + 2] = az;
    }
  }

  /**
   * Broadphase rebuild plus the narrow-phase impulse accumulation.
   *
   * Each particle accumulates only its *own* share of every pair it finds. That
   * is what makes the pass order-independent: with symmetric writes into both
   * particles' accumulators, the GPU would need atomics on another invocation's
   * memory, and with a fixed order the CPU would disagree with the GPU about
   * which bucket was visited first.
   */
  private collidePass(): void {
    const { count } = this.field;
    this.hash.build(this.field);
    this.lastHashOverflow = this.hash.overflow;
    this.dv.fill(0);
    this.dx.fill(0);
    this.lastContacts = 0;
    for (let i = 0; i < count; i++) {
      this.visitorI = i;
      this.hash.forEachCandidate(this.field, i, this.visitor);
    }
  }

  /** Narrow phase for one candidate pair. Called from `collidePass` only. */
  private resolveContact(i: number, j: number): void {
    const { data } = this.field;
    const oi = i * PARTICLE_STRIDE;
    const oj = j * PARTICLE_STRIDE;
    const dx = f(data[oj]! - data[oi]!);
    const dy = f(data[oj + 1]! - data[oi + 1]!);
    const dz = f(data[oj + 2]! - data[oi + 2]!);
    const d2 = f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz));
    const ri = data[oi + PARTICLE_OFFSET.radius]!;
    const rj = data[oj + PARTICLE_OFFSET.radius]!;
    const rsum = f(ri + rj);
    const rsum2 = f(rsum * rsum);
    if (!(d2 < rsum2)) return;
    if (i < j) this.lastContacts++;
    if (!(d2 > 0)) return; // Exactly coincident centres have no normal to push along.
    const dist = f(Math.sqrt(d2));
    const inv = f(1 / dist);
    const nx = f(dx * inv);
    const ny = f(dy * inv);
    const nz = f(dz * inv);
    const mi = data[oi + PARTICLE_OFFSET.mass]!;
    const mj = data[oj + PARTICLE_OFFSET.mass]!;
    // `mass === 0` means immovable, so its inverse mass is 0 and it simply does
    // not move; a pair of immovables has nothing to solve.
    const invi = mi > 0 ? f(1 / mi) : 0;
    const invj = mj > 0 ? f(1 / mj) : 0;
    const invSum = f(invi + invj);
    if (!(invSum > 0)) return;
    const o3 = i * 3;
    // Relative velocity comes from the *source* state on both sides, never from
    // the accumulated `dv`. Reading the accumulator would make the impulse depend
    // on which candidate was visited first, which is exactly the thing the GPU
    // cannot promise -- its buckets are filled by atomics, so its visit order is
    // a race. Source-only reads make the total order-independent up to
    // floating-point association on both backends.
    const rvx = f(data[oj + 4]! - data[oi + 4]!);
    const rvy = f(data[oj + 5]! - data[oi + 5]!);
    const rvz = f(data[oj + 6]! - data[oi + 6]!);
    const vn = f(f(f(rvx * nx) + f(rvy * ny)) + f(rvz * nz));
    if (vn < 0) {
      // Approaching. `n` points from i to j, so i is pushed along -n.
      const e = f(this.options.restitution);
      const imp = f(f(-f(1 + e) * vn) / invSum);
      const si = f(imp * invi);
      this.dv[o3] = f(this.dv[o3]! - f(si * nx));
      this.dv[o3 + 1] = f(this.dv[o3 + 1]! - f(si * ny));
      this.dv[o3 + 2] = f(this.dv[o3 + 2]! - f(si * nz));
    }
    // Positional correction is applied to i only, unconditionally: two spheres
    // that overlap but are separating still need to be pushed apart, or a pile
    // under gravity sinks into itself.
    const overlap = f(rsum - dist);
    const corr = f(overlap * f(f(invi / invSum) * POSITION_CORRECTION));
    this.dx[o3] = f(this.dx[o3]! - f(corr * nx));
    this.dx[o3 + 1] = f(this.dx[o3 + 1]! - f(corr * ny));
    this.dx[o3 + 2] = f(this.dx[o3 + 2]! - f(corr * nz));
  }

  /**
   * Semi-implicit Euler plus bounds, reading the source state and writing it
   * back in place.
   *
   * Returns the number of particles whose centre ended the step outside the box.
   * Under `reflect` and `wrap` that is structurally impossible -- both clamp or
   * fold the coordinate -- so a non-zero count means `boundsMode: 'none'`, which
   * is the mode the specs use to prove the counter works.
   */
  private integratePass(dt: number): number {
    const { data, count, bounds } = this.field;
    const opts = this.options;
    const fdt = f(dt);
    const gx = f(opts.gravity[0]);
    const gy = f(opts.gravity[1]);
    const gz = f(opts.gravity[2]);
    const damp = f(1 - f(f(opts.damping) * fdt));
    const maxSpeed = f(opts.maxSpeed);
    const maxSpeed2 = f(maxSpeed * maxSpeed);
    const rest = f(opts.restitution);
    const useAccel = opts.nbody;
    const useCollide = opts.collisions;
    const mode = opts.boundsMode;
    const [minX, minY, minZ] = bounds.min;
    const [maxX, maxY, maxZ] = bounds.max;
    const sizeX = f(maxX - minX);
    const sizeY = f(maxY - minY);
    const sizeZ = f(maxZ - minZ);
    let escaped = 0;

    for (let i = 0; i < count; i++) {
      const o = i * PARTICLE_STRIDE;
      const o3 = i * 3;
      const r = data[o + PARTICLE_OFFSET.radius]!;
      let vx = data[o + 4]!;
      let vy = data[o + 5]!;
      let vz = data[o + 6]!;
      let ax = gx;
      let ay = gy;
      let az = gz;
      if (useAccel) {
        ax = f(ax + this.accel[o3]!);
        ay = f(ay + this.accel[o3 + 1]!);
        az = f(az + this.accel[o3 + 2]!);
      }
      vx = f(f(vx + f(ax * fdt)) * damp);
      vy = f(f(vy + f(ay * fdt)) * damp);
      vz = f(f(vz + f(az * fdt)) * damp);
      const s2 = f(f(f(vx * vx) + f(vy * vy)) + f(vz * vz));
      if (s2 > maxSpeed2) {
        const k = f(maxSpeed / f(Math.sqrt(s2)));
        vx = f(vx * k);
        vy = f(vy * k);
        vz = f(vz * k);
      }
      if (useCollide) {
        vx = f(vx + this.dv[o3]!);
        vy = f(vy + this.dv[o3 + 1]!);
        vz = f(vz + this.dv[o3 + 2]!);
      }
      let px = f(data[o]! + f(vx * fdt));
      let py = f(data[o + 1]! + f(vy * fdt));
      let pz = f(data[o + 2]! + f(vz * fdt));
      if (useCollide) {
        px = f(px + this.dx[o3]!);
        py = f(py + this.dx[o3 + 1]!);
        pz = f(pz + this.dx[o3 + 2]!);
      }

      if (mode === 'reflect') {
        const loX = f(minX + r);
        const hiX = f(maxX - r);
        const loY = f(minY + r);
        const hiY = f(maxY - r);
        const loZ = f(minZ + r);
        const hiZ = f(maxZ - r);
        if (px < loX) {
          px = f(loX + f(f(loX - px) * rest));
          vx = f(-vx * rest);
        } else if (px > hiX) {
          px = f(hiX - f(f(px - hiX) * rest));
          vx = f(-vx * rest);
        }
        if (py < loY) {
          py = f(loY + f(f(loY - py) * rest));
          vy = f(-vy * rest);
        } else if (py > hiY) {
          py = f(hiY - f(f(py - hiY) * rest));
          vy = f(-vy * rest);
        }
        if (pz < loZ) {
          pz = f(loZ + f(f(loZ - pz) * rest));
          vz = f(-vz * rest);
        } else if (pz > hiZ) {
          pz = f(hiZ - f(f(pz - hiZ) * rest));
          vz = f(-vz * rest);
        }
        // The reflection above can overshoot to the far wall when one step's
        // penetration exceeds the box, which only a clamped dt cannot do. The
        // clamp keeps "escaped === 0" a structural invariant rather than a
        // property of the numbers that happened to arrive.
        px = clamp(px, loX, hiX);
        py = clamp(py, loY, hiY);
        pz = clamp(pz, loZ, hiZ);
      } else if (mode === 'wrap') {
        px = wrapAxis(px, minX, sizeX);
        py = wrapAxis(py, minY, sizeY);
        pz = wrapAxis(pz, minZ, sizeZ);
      } else {
        if (
          !(
            px >= minX &&
            px <= maxX &&
            py >= minY &&
            py <= maxY &&
            pz >= minZ &&
            pz <= maxZ
          )
        ) {
          escaped++;
        }
      }

      data[o] = px;
      data[o + 1] = py;
      data[o + 2] = pz;
      data[o + 4] = vx;
      data[o + 5] = vy;
      data[o + 6] = vz;
    }
    return escaped;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Fold one axis into `[min, min + size)`.
 *
 * Written with `floor` rather than `%` because WGSL has no floored float modulo,
 * and this form is bit-identical on both sides for negative offsets.
 */
function wrapAxis(p: number, min: number, size: number): number {
  if (!(size > 0)) return p;
  const k = Math.floor(f((p - min) / size));
  return f(p - f(k * size));
}

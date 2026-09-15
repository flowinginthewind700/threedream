/**
 * The particle state container: one interleaved `Float32Array` that the CPU
 * reference simulation, the WebGPU kernels and the renderer all agree on.
 *
 * # Why interleaved, and why 32 bytes
 *
 * `pos.xyz | radius | vel.xyz | mass` is 8 f32, exactly one `vec4<f32>` pair in
 * WGSL, so a particle is two aligned `vec4` loads and no padding. A
 * structure-of-arrays layout would beat it for a kernel that touches only
 * positions, but every kernel here touches both position and velocity in the
 * same invocation, and SoA would then pay two cache lines per particle instead
 * of one. 32 bytes also divides the 16 MiB `LIMIT_FLOOR` binding evenly:
 * 524288 particles fit in one storage buffer, which is 5x the M3 target and
 * means the whole field is always a single binding rather than a
 * split-across-buffers special case.
 *
 * # Why f32 and not f64
 *
 * WGSL has no f64. The rigid-body kernel keeps f64 because it is the
 * deterministic reference path and bit-exactness across Node, browser and replay
 * is its contract. Particles are the opposite case: they are the *scale* layer,
 * the GPU backend declares `deterministic: false`, and a CPU implementation that
 * used f64 while the GPU used f32 could never be compared to it. So the CPU
 * particle path is f32 too -- which also makes it deterministic in the sense
 * that matters here: same seed, same step order, same bytes, every run.
 *
 * Nothing in this file imports `three` or WebGPU types, so the container, the
 * scene presets and the statistics all run in bare Node under vitest.
 */

import { Rng } from '../core/rng.js';
import { digestWithCount } from '../core/digest.js';

/** Floats per particle: `pos.xyz, radius, vel.xyz, mass`. */
export const PARTICLE_STRIDE = 8;

/** Bytes per particle. The number `maxElements` needs to size a binding. */
export const PARTICLE_BYTES = PARTICLE_STRIDE * 4;

/** Offsets into one particle record, in floats. */
export const PARTICLE_OFFSET = {
  position: 0,
  radius: 3,
  velocity: 4,
  mass: 7,
} as const;

export type Vec3Tuple = readonly [number, number, number];

/** An axis-aligned box the particles are confined to. */
export interface Bounds {
  readonly min: Vec3Tuple;
  readonly max: Vec3Tuple;
}

/**
 * The default volume: 16 x 16 x 16 with the floor lower than the ceiling, so a
 * gravity scene settles instead of hanging. Sized so 100k particles at the
 * default radius fill about half a percent of it -- dense enough that the
 * broadphase has real work, sparse enough that the demo reads as a gas rather
 * than a solid.
 */
export const DEFAULT_BOUNDS: Bounds = { min: [-8, -8, -8], max: [8, 8, 8] };

/** Initial distributions. All are deterministic for a given seed. */
export type ParticleScene = 'box' | 'sphere' | 'shell' | 'grid' | 'slab';

export const PARTICLE_SCENES: readonly ParticleScene[] = [
  'box',
  'sphere',
  'shell',
  'grid',
  'slab',
];

export function boundsSize(bounds: Bounds): Vec3Tuple {
  return [
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  ];
}

export function boundsCenter(bounds: Bounds): Vec3Tuple {
  return [
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[1] + bounds.max[1]) / 2,
    (bounds.min[2] + bounds.max[2]) / 2,
  ];
}

/** The largest sphere that fits inside `bounds`, as a radius. */
export function boundsInradius(bounds: Bounds): number {
  const [sx, sy, sz] = boundsSize(bounds);
  return Math.min(sx, sy, sz) / 2;
}

export interface ParticleFieldOptions {
  /** Particle count. Must be a positive integer. */
  count: number;
  /** Seed for the initial distribution. Same seed, same bytes. */
  seed?: number;
  /** Initial distribution. Defaults to `sphere`. */
  scene?: ParticleScene;
  bounds?: Bounds;
  /** Inclusive `[min, max]` radius range, sampled per particle. */
  radius?: readonly [number, number];
  /** Mass for every particle. `0` is allowed and means "immovable". */
  mass?: number;
  /** Scale of the initial velocity, in units/second. */
  speed?: number;
}

export const DEFAULT_RADIUS: readonly [number, number] = [0.03, 0.06];

/**
 * The particles, as one flat f32 buffer plus the box they live in.
 *
 * Deliberately a plain container rather than an entity in the ECS: at 100k
 * particles the per-entity overhead of a component map is the dominant cost, and
 * the GPU needs a contiguous buffer anyway. The ECS still owns anything that
 * reasons about individual bodies; this owns the bulk.
 */
export class ParticleField {
  readonly count: number;
  readonly bounds: Bounds;
  /** `count * PARTICLE_STRIDE` f32, laid out as documented at the top. */
  readonly data: Float32Array;

  constructor(options: ParticleFieldOptions);
  constructor(data: Float32Array, count: number, bounds?: Bounds);
  constructor(
    first: ParticleFieldOptions | Float32Array,
    second?: number,
    third?: Bounds,
  ) {
    if (first instanceof Float32Array) {
      const count = second ?? 0;
      if (!Number.isInteger(count) || count <= 0) {
        throw new RangeError(`count must be a positive integer, got ${count}`);
      }
      if (first.length !== count * PARTICLE_STRIDE) {
        throw new RangeError(
          `data holds ${first.length} floats, but ${count} particles need ${count * PARTICLE_STRIDE}`,
        );
      }
      this.data = first;
      this.count = count;
      this.bounds = third ?? DEFAULT_BOUNDS;
      assertBounds(this.bounds);
      return;
    }
    const options = first;
    if (!Number.isInteger(options.count) || options.count <= 0) {
      throw new RangeError(`count must be a positive integer, got ${options.count}`);
    }
    this.count = options.count;
    this.bounds = options.bounds ?? DEFAULT_BOUNDS;
    assertBounds(this.bounds);
    this.data = new Float32Array(options.count * PARTICLE_STRIDE);
    const radius = options.radius ?? DEFAULT_RADIUS;
    if (!(radius[0] > 0) || !(radius[1] >= radius[0])) {
      throw new RangeError(`radius range must be positive and ordered, got [${radius}]`);
    }
    const mass = options.mass ?? 1;
    if (!(mass >= 0) || !Number.isFinite(mass)) {
      throw new RangeError(`mass must be finite and >= 0, got ${mass}`);
    }
    const speed = options.speed ?? 0;
    if (!(speed >= 0) || !Number.isFinite(speed)) {
      throw new RangeError(`speed must be finite and >= 0, got ${speed}`);
    }
    fillScene(this, options.scene ?? 'sphere', new Rng(options.seed), radius, mass, speed);
  }

  /** Bytes the GPU needs for this many particles. */
  static bytesFor(count: number): number {
    return count * PARTICLE_BYTES;
  }

  /** An all-zero field of the same shape. Used by tests and by resize. */
  static empty(count: number, bounds: Bounds = DEFAULT_BOUNDS): ParticleField {
    return new ParticleField(new Float32Array(count * PARTICLE_STRIDE), count, bounds);
  }

  clone(): ParticleField {
    return new ParticleField(this.data.slice(), this.count, this.bounds);
  }

  copyFrom(other: ParticleField): void {
    if (other.count !== this.count) {
      throw new RangeError(`cannot copy ${other.count} particles into ${this.count}`);
    }
    this.data.set(other.data);
  }

  private at(i: number): number {
    if (!Number.isInteger(i) || i < 0 || i >= this.count) {
      throw new RangeError(`particle index ${i} out of range [0, ${this.count})`);
    }
    return i * PARTICLE_STRIDE;
  }

  position(i: number): Vec3Tuple {
    const o = this.at(i);
    return [this.data[o]!, this.data[o + 1]!, this.data[o + 2]!];
  }

  velocity(i: number): Vec3Tuple {
    const o = this.at(i) + PARTICLE_OFFSET.velocity;
    return [this.data[o]!, this.data[o + 1]!, this.data[o + 2]!];
  }

  radius(i: number): number {
    return this.data[this.at(i) + PARTICLE_OFFSET.radius]!;
  }

  mass(i: number): number {
    return this.data[this.at(i) + PARTICLE_OFFSET.mass]!;
  }

  setPosition(i: number, p: Vec3Tuple): void {
    const o = this.at(i);
    this.data[o] = p[0];
    this.data[o + 1] = p[1];
    this.data[o + 2] = p[2];
  }

  setVelocity(i: number, v: Vec3Tuple): void {
    const o = this.at(i) + PARTICLE_OFFSET.velocity;
    this.data[o] = v[0];
    this.data[o + 1] = v[1];
    this.data[o + 2] = v[2];
  }

  setRadius(i: number, r: number): void {
    if (!(r > 0)) throw new RangeError(`radius must be positive, got ${r}`);
    this.data[this.at(i) + PARTICLE_OFFSET.radius] = r;
  }

  setMass(i: number, m: number): void {
    if (!(m >= 0)) throw new RangeError(`mass must be >= 0, got ${m}`);
    this.data[this.at(i) + PARTICLE_OFFSET.mass] = m;
  }

  /** The largest radius present, which is what the broadphase cell size keys on. */
  maxRadius(): number {
    let max = 0;
    for (let i = 0; i < this.count; i++) {
      const r = this.data[i * PARTICLE_STRIDE + PARTICLE_OFFSET.radius]!;
      if (r > max) max = r;
    }
    return max;
  }

  /** Sum of 0.5*m*v^2. Conserved-ish, so a spike means the solver is broken. */
  kineticEnergy(): number {
    let total = 0;
    for (let i = 0; i < this.count; i++) {
      const o = i * PARTICLE_STRIDE;
      const vx = this.data[o + 4]!;
      const vy = this.data[o + 5]!;
      const vz = this.data[o + 6]!;
      total += 0.5 * this.data[o + 7]! * (vx * vx + vy * vy + vz * vz);
    }
    return total;
  }

  maxSpeed(): number {
    let max = 0;
    for (let i = 0; i < this.count; i++) {
      const o = i * PARTICLE_STRIDE;
      const vx = this.data[o + 4]!;
      const vy = this.data[o + 5]!;
      const vz = this.data[o + 6]!;
      const s2 = vx * vx + vy * vy + vz * vz;
      if (s2 > max) max = s2;
    }
    return Math.sqrt(max);
  }

  /**
   * Particles whose centre has left the box.
   *
   * A boundary kernel that silently clamps instead of reflecting still "works"
   * for a few hundred frames, so the demo and the browser specs count escapes
   * rather than trusting the picture.
   */
  outOfBounds(): number {
    const { min, max } = this.bounds;
    let escaped = 0;
    for (let i = 0; i < this.count; i++) {
      const o = i * PARTICLE_STRIDE;
      const x = this.data[o]!;
      const y = this.data[o + 1]!;
      const z = this.data[o + 2]!;
      if (!(x >= min[0] && x <= max[0] && y >= min[1] && y <= max[1] && z >= min[2] && z <= max[2])) {
        escaped++;
      }
    }
    return escaped;
  }

  /**
   * `hex:count` over the raw f32 bytes of the whole field.
   *
   * This is the comparison the CPU/GPU parity check uses: not "the pictures look
   * alike" but "these two buffers hash to the same string", which is only
   * possible because both sides are f32 and both integrate in the same order.
   */
  digest(): string {
    return digestWithCount(this.data);
  }
}

function assertBounds(bounds: Bounds): void {
  for (let axis = 0; axis < 3; axis++) {
    const lo = bounds.min[axis]!;
    const hi = bounds.max[axis]!;
    if (!(Number.isFinite(lo) && Number.isFinite(hi)) || !(hi > lo)) {
      throw new RangeError(
        `bounds axis ${axis} must be finite with max > min, got [${lo}, ${hi}]`,
      );
    }
  }
}

/**
 * Fill `field` from a scene preset.
 *
 * Every branch consumes the RNG in a fixed order and writes through the
 * `Float32Array`, so the result is bit-identical for a given seed on every
 * platform. `Math.fround` is applied to the generated values before they are
 * written only where the arithmetic itself was done in f64 -- the store rounds
 * anyway, so the explicit call would be noise.
 */
function fillScene(
  field: ParticleField,
  scene: ParticleScene,
  rng: Rng,
  radius: readonly [number, number],
  mass: number,
  speed: number,
): void {
  if (!PARTICLE_SCENES.includes(scene)) {
    throw new RangeError(`unknown scene "${scene}", expected one of ${PARTICLE_SCENES.join(', ')}`);
  }
  const { count, bounds, data } = field;
  const [cx, cy, cz] = boundsCenter(bounds);
  const inset = radius[1];
  // A lattice needs a per-axis count; `grid` uses the cube root rounded up so
  // the last rows are simply left empty rather than distorting the spacing.
  const perAxis = Math.max(1, Math.ceil(Math.cbrt(count)));
  const inradius = Math.max(1e-4, boundsInradius(bounds) - inset);

  for (let i = 0; i < count; i++) {
    const o = i * PARTICLE_STRIDE;
    const r = radius[0] === radius[1] ? radius[0] : rng.range(radius[0], radius[1]);
    let x = 0;
    let y = 0;
    let z = 0;
    switch (scene) {
      case 'box': {
        x = rng.range(bounds.min[0] + inset, bounds.max[0] - inset);
        y = rng.range(bounds.min[1] + inset, bounds.max[1] - inset);
        z = rng.range(bounds.min[2] + inset, bounds.max[2] - inset);
        break;
      }
      case 'sphere': {
        // Rejection-sample the unit ball: cheaper than a cube-root transform and
        // it keeps the RNG consumption uniform across platforms.
        let ux = 0;
        let uy = 0;
        let uz = 0;
        let s2 = 2;
        while (s2 >= 1) {
          ux = rng.next() * 2 - 1;
          uy = rng.next() * 2 - 1;
          uz = rng.next() * 2 - 1;
          s2 = ux * ux + uy * uy + uz * uz;
        }
        const scale = inradius * Math.cbrt(rng.next());
        const len = Math.sqrt(s2) || 1;
        x = cx + (ux / len) * scale;
        y = cy + (uy / len) * scale;
        z = cz + (uz / len) * scale;
        break;
      }
      case 'shell': {
        let ux = 0;
        let uy = 0;
        let uz = 0;
        let s2 = 2;
        while (s2 >= 1) {
          ux = rng.next() * 2 - 1;
          uy = rng.next() * 2 - 1;
          uz = rng.next() * 2 - 1;
          s2 = ux * ux + uy * uy + uz * uz;
        }
        const len = Math.sqrt(s2) || 1;
        x = cx + (ux / len) * inradius;
        y = cy + (uy / len) * inradius;
        z = cz + (uz / len) * inradius;
        break;
      }
      case 'grid': {
        const [sx, sy, sz] = boundsSize(bounds);
        const gx = i % perAxis;
        const gy = Math.floor(i / perAxis) % perAxis;
        const gz = Math.floor(i / (perAxis * perAxis));
        const step = (n: number): number => (perAxis <= 1 ? 0.5 : n / (perAxis - 1));
        x = bounds.min[0] + inset + step(gx) * (sx - 2 * inset);
        y = bounds.min[1] + inset + step(gy) * (sy - 2 * inset);
        z = bounds.min[2] + inset + step(gz) * (sz - 2 * inset);
        break;
      }
      case 'slab': {
        // The top 20% of the box, so a gravity run has somewhere to fall and the
        // first frames of the demo actually move.
        const [sx, slabHeight, sz] = boundsSize(bounds);
        x = cx + (rng.next() * 2 - 1) * (sx / 2 - inset);
        y = bounds.max[1] - inset - rng.next() * (slabHeight * 0.2);
        z = cz + (rng.next() * 2 - 1) * (sz / 2 - inset);
        break;
      }
    }
    data[o] = x;
    data[o + 1] = y;
    data[o + 2] = z;
    data[o + PARTICLE_OFFSET.radius] = r;
    data[o + 4] = speed === 0 ? 0 : rng.range(-speed, speed);
    data[o + 5] = speed === 0 ? 0 : rng.range(-speed, speed);
    data[o + 6] = speed === 0 ? 0 : rng.range(-speed, speed);
    data[o + PARTICLE_OFFSET.mass] = mass;
  }
}

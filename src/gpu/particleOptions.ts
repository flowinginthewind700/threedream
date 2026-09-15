/**
 * Defaults, validation, and the uniform-buffer layout the particle kernels read.
 *
 * The CPU and GPU simulations have to compute the *same* function of the same
 * options, and the cheapest way to guarantee that is to have exactly one place
 * resolve them and exactly one place pack them. So this module owns both: the
 * CPU sim reads the resolved struct field by field, and the GPU sim uploads the
 * same struct as a 64-byte uniform that `particleWgsl.ts` declares as `Params`.
 * A test pins the two against each other, which is what stops a WGSL struct from
 * drifting three fields out of alignment -- the failure mode where the shader
 * compiles, runs, and quietly uses `dt` as the restitution.
 */

import {
  PARTICLE_STRIDE,
  boundsSize,
  type ParticleField,
  type Vec3Tuple,
} from './particleField.js';
import type { BoundsMode, ParticleSimOptions, ResolvedParticleOptions } from './particleTypes.js';

export const DEFAULT_GRAVITY: Vec3Tuple = [0, -9.81, 0];

export const DEFAULT_PARTICLE_OPTIONS: ResolvedParticleOptions = {
  gravity: DEFAULT_GRAVITY,
  damping: 0,
  restitution: 0.6,
  maxSpeed: 50,
  collisions: true,
  nbody: false,
  nbodyStrength: 1,
  softening: 0.01,
  cutoff: 0,
  boundsMode: 'reflect',
  cellSize: 0,
  bucketCapacity: 8,
  fixedDt: 1 / 60,
};

/** `flags` bits in the uniform. Order is pinned by `tests/particle_options.test.ts`. */
export const PARTICLE_FLAG = {
  collisions: 1,
  nbody: 2,
  /** Bounds mode occupies bits 2-3 so `reflect` stays the zero value. */
  boundsShift: 2,
} as const;

export const BOUNDS_MODE_BITS: Readonly<Record<BoundsMode, number>> = {
  reflect: 0,
  wrap: 1,
  none: 2,
};

/** Uniform size in floats. 16 f32 = 64 bytes, which is `sizeOf(Params)` in WGSL. */
export const PARAMS_FLOATS = 16;
export const PARAMS_BYTES = PARAMS_FLOATS * 4;

/**
 * Offsets into the uniform, in 4-byte words.
 *
 * Words 0-11 are f32, words 12-15 are u32: a WGSL struct mixes them freely, and
 * keeping the integers in the tail means the float half is one contiguous run
 * that `writeParams` can fill without a DataView.
 */
export const PARAM_WORD = {
  gravityX: 0,
  gravityY: 1,
  gravityZ: 2,
  damping: 3,
  restitution: 4,
  maxSpeed: 5,
  dt: 6,
  nbodyStrength: 7,
  softening: 8,
  cutoffSquared: 9,
  cellSize: 10,
  invCell: 11,
  count: 12,
  tableMask: 13,
  bucketCapacity: 14,
  flags: 15,
} as const;

function finite(name: string, value: number, min: number, max: number): number {
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new RangeError(`${name} must be finite and within [${min}, ${max}], got ${value}`);
  }
  return value;
}

/**
 * Fill in every default and reject anything a kernel could not honour.
 *
 * `cellSize` resolves to `0` here rather than to a number, because the right
 * cell edge depends on the largest radius in the field, which is a property of
 * the data and not of the options. Backends ask `effectiveCellSize` once the
 * field exists. Validation that needs the field is in `assertFieldFits`.
 */
export function resolveParticleOptions(
  options: ParticleSimOptions = {},
): ResolvedParticleOptions {
  const gravity = options.gravity ?? DEFAULT_PARTICLE_OPTIONS.gravity;
  if (gravity.length !== 3 || gravity.some((v) => !Number.isFinite(v))) {
    throw new RangeError(`gravity must be three finite numbers, got [${gravity}]`);
  }
  const boundsMode = options.boundsMode ?? DEFAULT_PARTICLE_OPTIONS.boundsMode;
  if (!(boundsMode in BOUNDS_MODE_BITS)) {
    throw new RangeError(`boundsMode must be one of reflect|wrap|none, got "${boundsMode}"`);
  }
  return {
    gravity: [gravity[0], gravity[1], gravity[2]],
    damping: finite('damping', options.damping ?? 0, 0, 60),
    restitution: finite('restitution', options.restitution ?? 0.6, 0, 1),
    maxSpeed: finite('maxSpeed', options.maxSpeed ?? 50, 1e-6, 1e6),
    collisions: options.collisions ?? true,
    nbody: options.nbody ?? false,
    nbodyStrength: finite('nbodyStrength', options.nbodyStrength ?? 1, 0, 1e6),
    softening: finite('softening', options.softening ?? 0.01, 1e-6, 1e3),
    cutoff: finite('cutoff', options.cutoff ?? 0, 0, 1e6),
    boundsMode,
    cellSize: finite('cellSize', options.cellSize ?? 0, 0, 1e6),
    bucketCapacity: Math.trunc(
      finite('bucketCapacity', options.bucketCapacity ?? 8, 1, 4096),
    ),
    fixedDt: finite('fixedDt', options.fixedDt ?? 1 / 60, 1e-5, 1),
  };
}

/**
 * The cell edge a broadphase should use for this field.
 *
 * Twice the largest radius is the smallest edge that makes a 27-cell
 * neighbourhood complete: two spheres overlap only when their centres are within
 * `r_i + r_j <= 2 * r_max`, so any overlapping pair is either in the same cell or
 * in adjacent ones. A larger cell is legal and just wastes distance tests; a
 * smaller one silently misses contacts, which is why this is a function rather
 * than a constant each call site picks.
 */
export function effectiveCellSize(field: ParticleField, resolved: ResolvedParticleOptions): number {
  if (resolved.cellSize > 0) return resolved.cellSize;
  return Math.max(1e-4, field.maxRadius() * 2);
}

/**
 * Reject a field the bounds handling could not honour.
 *
 * A box thinner than one diameter turns `reflect` into a fight between the two
 * walls: the particle is clamped to `min + r`, which is past `max - r`, and every
 * step teleports it. That is not a simulation bug and no kernel can fix it, so it
 * is an argument error.
 */
export function assertFieldFits(field: ParticleField): void {
  const diameter = field.maxRadius() * 2;
  const size = boundsSize(field.bounds);
  for (let axis = 0; axis < 3; axis++) {
    if (!(size[axis] > diameter)) {
      throw new RangeError(
        `bounds axis ${axis} is ${size[axis]} wide, which cannot contain a particle of diameter ${diameter}`,
      );
    }
  }
  // A field whose buffer is not the size the stride promises would make every
  // GPU binding lie about its length, so it is checked here too rather than at
  // each dispatch site.
  if (field.data.length !== field.count * PARTICLE_STRIDE) {
    throw new RangeError('field data does not match its count and stride');
  }
}

/** Everything `writeParams` needs besides the resolved options. */
export interface ParamsFrame {
  /** Step size for this dispatch. */
  dt: number;
  /** Broadphase cell edge actually in use. */
  cellSize: number;
  /** Hash table size, a power of two. `tableMask` is written as `size - 1`. */
  tableSize: number;
  bucketCapacity: number;
  count: number;
}

/**
 * Pack one frame's uniform into `out`.
 *
 * `out` must be at least `PARAMS_BYTES`; the same call fills the CPU-side
 * scratch a test reads back, so what the shader sees is what the spec asserts.
 */
export function writeParams(
  out: ArrayBuffer,
  resolved: ResolvedParticleOptions,
  frame: ParamsFrame,
): void {
  if (out.byteLength < PARAMS_BYTES) {
    throw new RangeError(`params buffer is ${out.byteLength} bytes, needs ${PARAMS_BYTES}`);
  }
  if ((frame.tableSize & (frame.tableSize - 1)) !== 0 || frame.tableSize <= 0) {
    throw new RangeError(`tableSize must be a power of two, got ${frame.tableSize}`);
  }
  const floats = new Float32Array(out, 0, 12);
  const ints = new Uint32Array(out, 12 * 4, 4);
  floats[PARAM_WORD.gravityX] = resolved.gravity[0];
  floats[PARAM_WORD.gravityY] = resolved.gravity[1];
  floats[PARAM_WORD.gravityZ] = resolved.gravity[2];
  floats[PARAM_WORD.damping] = resolved.damping;
  floats[PARAM_WORD.restitution] = resolved.restitution;
  floats[PARAM_WORD.maxSpeed] = resolved.maxSpeed;
  floats[PARAM_WORD.dt] = frame.dt;
  floats[PARAM_WORD.nbodyStrength] = resolved.nbodyStrength;
  floats[PARAM_WORD.softening] = resolved.softening;
  // Squared, because the shader compares against `dot(d, d)` and a sqrt per pair
  // just to apply a cutoff would cost more than the pairs it rejects.
  floats[PARAM_WORD.cutoffSquared] = resolved.cutoff * resolved.cutoff;
  floats[PARAM_WORD.cellSize] = frame.cellSize;
  floats[PARAM_WORD.invCell] = frame.cellSize > 0 ? 1 / frame.cellSize : 0;
  let flags = 0;
  if (resolved.collisions) flags |= PARTICLE_FLAG.collisions;
  if (resolved.nbody) flags |= PARTICLE_FLAG.nbody;
  flags |= BOUNDS_MODE_BITS[resolved.boundsMode] << PARTICLE_FLAG.boundsShift;
  ints[0] = frame.count >>> 0;
  ints[1] = (frame.tableSize - 1) >>> 0;
  ints[2] = frame.bucketCapacity >>> 0;
  ints[3] = flags >>> 0;
}

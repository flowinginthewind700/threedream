/**
 * `gpu/particleOptions.ts` -- defaults, validation, and the uniform layout.
 *
 * The uniform packing gets the most scrutiny because it is the one place where a
 * mistake compiles, runs, and produces a plausible-looking simulation with the
 * wrong constants: WGSL struct offsets are checked against `Params` in
 * `particleWgsl.ts`, and `writeParams` is checked against `PARAM_WORD` here.
 */

import { describe, expect, it } from 'vitest';
import {
  BOUNDS_MODE_BITS,
  DEFAULT_GRAVITY,
  DEFAULT_PARTICLE_OPTIONS,
  PARAMS_BYTES,
  PARAMS_FLOATS,
  PARAM_WORD,
  PARTICLE_FLAG,
  assertFieldFits,
  effectiveCellSize,
  resolveParticleOptions,
  writeParams,
  type ParamsFrame,
} from '../src/gpu/particleOptions.js';
import { DEFAULT_BOUNDS, PARTICLE_STRIDE, ParticleField } from '../src/gpu/particleField.js';
import type { BoundsMode } from '../src/gpu/particleTypes.js';

const field = (count = 64, radius: readonly [number, number] = [0.05, 0.1]): ParticleField =>
  new ParticleField({ count, scene: 'grid', radius });

/** A frame whose bounds match the field the tests build, so the two agree. */
const BOUNDS = { min: [-8, -8, -8] as const, max: [8, 8, 8] as const };

describe('resolveParticleOptions', () => {
  it('fills every default', () => {
    expect(resolveParticleOptions()).toEqual(DEFAULT_PARTICLE_OPTIONS);
    expect(resolveParticleOptions({})).toEqual(DEFAULT_PARTICLE_OPTIONS);
    expect(DEFAULT_GRAVITY).toEqual([0, -9.81, 0]);
    expect(DEFAULT_PARTICLE_OPTIONS.fixedDt).toBeCloseTo(1 / 60, 12);
  });

  it('copies gravity rather than aliasing the caller array', () => {
    const gravity: [number, number, number] = [1, 2, 3];
    const resolved = resolveParticleOptions({ gravity });
    gravity[0] = 99;
    expect(resolved.gravity).toEqual([1, 2, 3]);
    expect(resolved.gravity).not.toBe(gravity);
  });

  it('rejects a gravity vector that is not three finite numbers', () => {
    expect(() => resolveParticleOptions({ gravity: [1, 2] as unknown as [number, number, number] })).toThrow(
      /gravity must be three finite numbers/,
    );
    expect(() =>
      resolveParticleOptions({ gravity: [1, Number.NaN, 3] as [number, number, number] }),
    ).toThrow(/gravity must be three finite numbers/);
  });

  it('rejects an unknown bounds mode', () => {
    expect(() => resolveParticleOptions({ boundsMode: 'bounce' as BoundsMode })).toThrow(
      /boundsMode must be one of reflect\|wrap\|none/,
    );
  });

  it.each([
    ['damping', { damping: -0.1 }],
    ['damping', { damping: 61 }],
    ['restitution', { restitution: 1.5 }],
    ['restitution', { restitution: -0.5 }],
    ['maxSpeed', { maxSpeed: 0 }],
    ['maxSpeed', { maxSpeed: Number.NaN }],
    ['nbodyStrength', { nbodyStrength: -1 }],
    ['softening', { softening: 0 }],
    ['cutoff', { cutoff: -1 }],
    ['cellSize', { cellSize: -1 }],
    ['bucketCapacity', { bucketCapacity: 0 }],
    ['bucketCapacity', { bucketCapacity: 5000 }],
    ['fixedDt', { fixedDt: 0 }],
    ['fixedDt', { fixedDt: 2 }],
  ] as const)('rejects %s out of range', (_name, options) => {
    expect(() => resolveParticleOptions(options)).toThrow(/must be finite and within/);
  });

  it('accepts the boundary values it allows', () => {
    expect(resolveParticleOptions({ damping: 60 }).damping).toBe(60);
    expect(resolveParticleOptions({ restitution: 1 }).restitution).toBe(1);
    expect(resolveParticleOptions({ restitution: 0 }).restitution).toBe(0);
    expect(resolveParticleOptions({ cutoff: 0 }).cutoff).toBe(0);
    expect(resolveParticleOptions({ bucketCapacity: 4096 }).bucketCapacity).toBe(4096);
  });

  it('truncates a fractional bucket capacity, because slots are integers', () => {
    expect(resolveParticleOptions({ bucketCapacity: 4.9 }).bucketCapacity).toBe(4);
  });

  it('leaves cellSize at 0, meaning "derive it from the field"', () => {
    expect(resolveParticleOptions({ cellSize: 0 }).cellSize).toBe(0);
    expect(resolveParticleOptions({ cellSize: 2.5 }).cellSize).toBe(2.5);
  });

  it('keeps reflect as the zero value of the bounds encoding', () => {
    expect(BOUNDS_MODE_BITS.reflect).toBe(0);
    expect(BOUNDS_MODE_BITS.wrap).toBe(1);
    expect(BOUNDS_MODE_BITS.none).toBe(2);
  });
});

describe('effectiveCellSize', () => {
  it('defaults to twice the largest radius, which makes 27 cells complete', () => {
    const f = field(64, [0.05, 0.25]);
    f.setRadius(3, 0.4);
    expect(effectiveCellSize(f, resolveParticleOptions())).toBeCloseTo(0.8, 6);
  });

  it('never returns a non-positive edge', () => {
    // An all-zero field has maxRadius 0; a cell size of 0 would divide by zero
    // in both the CPU and the shader.
    const empty = ParticleField.empty(4);
    expect(effectiveCellSize(empty, resolveParticleOptions())).toBe(1e-4);
  });

  it('an explicit cell size wins', () => {
    const f = field(64, [0.05, 0.25]);
    expect(effectiveCellSize(f, resolveParticleOptions({ cellSize: 3 }))).toBe(3);
  });
});

describe('assertFieldFits', () => {
  it('accepts a field its bounds can contain', () => {
    expect(() => assertFieldFits(field(32, [0.05, 0.1]))).not.toThrow();
  });

  it.each([0, 1, 2])('rejects an axis too thin for the largest diameter (axis %i)', (axis) => {
    const min: [number, number, number] = [-8, -8, -8];
    const max: [number, number, number] = [8, 8, 8];
    max[axis] = min[axis] + 0.1; // thinner than a 0.2 diameter
    const bounds = { min, max };
    const data = new Float32Array(2 * PARTICLE_STRIDE);
    data[3] = 0.1;
    data[PARTICLE_STRIDE + 3] = 0.1;
    const f = new ParticleField(data, 2, bounds);
    // The message carries the measured width, which is `0.1` only up to f64
    // subtraction, so assert on the axis and the reason rather than the digits.
    expect(() => assertFieldFits(f)).toThrow(new RegExp(`bounds axis ${axis} is 0\\.09`));
    expect(() => assertFieldFits(f)).toThrow(/cannot contain a particle of diameter/);
  });

  it('rejects a field whose buffer does not match its count and stride', () => {
    const lying = {
      count: 4,
      bounds: DEFAULT_BOUNDS,
      data: new Float32Array(7),
      maxRadius: () => 0.1,
    } as unknown as ParticleField;
    expect(() => assertFieldFits(lying)).toThrow(/field data does not match its count and stride/);
  });
});

describe('the uniform layout', () => {
  it('is 24 floats / 96 bytes, and the word offsets are all distinct', () => {
    expect(PARAMS_FLOATS).toBe(24);
    expect(PARAMS_BYTES).toBe(96);
    // WGSL rounds a struct's size up to its alignment, which is 16 for vec3.
    expect(PARAMS_BYTES % 16).toBe(0);
    const words = Object.values(PARAM_WORD);
    expect(new Set(words).size).toBe(words.length);
    expect(Math.max(...words)).toBe(PARAMS_FLOATS - 1);
    expect(Math.min(...words)).toBe(0);
  });

  it('puts the twenty f32 words first and the four u32 words last', () => {
    const intWords = ['count', 'tableMask', 'bucketCapacity', 'flags'] as const;
    for (const name of intWords) expect(PARAM_WORD[name]).toBeGreaterThanOrEqual(20);
    for (const [name, word] of Object.entries(PARAM_WORD)) {
      if (!(intWords as readonly string[]).includes(name)) expect(word).toBeLessThan(20);
    }
  });

  it('starts every vec3 on a 16-byte boundary, as WGSL alignment requires', () => {
    for (const vector of ['gravity', 'boundsMin', 'boundsMax'] as const) {
      const x = PARAM_WORD[`${vector}X`] as number;
      expect(x % 4, `${vector} must be 16-byte aligned`).toBe(0);
      expect(PARAM_WORD[`${vector}Y`]).toBe(x + 1);
      expect(PARAM_WORD[`${vector}Z`]).toBe(x + 2);
    }
  });

  it('keeps the two pad words, so the struct has no implicit gaps', () => {
    expect(PARAM_WORD.padA).toBe(18);
    expect(PARAM_WORD.padB).toBe(19);
  });

  const frame: ParamsFrame = {
    dt: 1 / 120,
    cellSize: 0.25,
    tableSize: 2048,
    bucketCapacity: 8,
    count: 1000,
    bounds: BOUNDS,
  };

  const packed = (
    resolved = resolveParticleOptions(),
    f = frame,
  ): { floats: Float32Array; ints: Uint32Array } => {
    const buffer = new ArrayBuffer(PARAMS_BYTES);
    writeParams(buffer, resolved, f);
    return { floats: new Float32Array(buffer, 0, 20), ints: new Uint32Array(buffer, 80, 4) };
  };

  it('writes every documented word', () => {
    const resolved = resolveParticleOptions({
      gravity: [1, -2, 3],
      damping: 0.25,
      restitution: 0.75,
      maxSpeed: 12,
      nbodyStrength: 2,
      softening: 0.5,
      cutoff: 4,
    });
    const { floats, ints } = packed(resolved);
    expect(floats[PARAM_WORD.gravityX]).toBe(1);
    expect(floats[PARAM_WORD.gravityY]).toBe(-2);
    expect(floats[PARAM_WORD.gravityZ]).toBe(3);
    expect(floats[PARAM_WORD.damping]).toBeCloseTo(0.25, 6);
    expect(floats[PARAM_WORD.restitution]).toBeCloseTo(0.75, 6);
    expect(floats[PARAM_WORD.maxSpeed]).toBe(12);
    expect(floats[PARAM_WORD.dt]).toBeCloseTo(1 / 120, 6);
    expect(floats[PARAM_WORD.nbodyStrength]).toBe(2);
    expect(floats[PARAM_WORD.softening]).toBe(0.5);
    expect(floats[PARAM_WORD.cutoffSquared]).toBe(16); // squared on the CPU side
    expect(floats[PARAM_WORD.cellSize]).toBe(0.25);
    expect(floats[PARAM_WORD.invCell]).toBe(4);
    expect(ints[0]).toBe(1000);
    expect(ints[1]).toBe(2047);
    expect(ints[2]).toBe(8);
  });

  it('carries the box, which is also the broadphase origin', () => {
    const bounds = { min: [-1, -2, -3] as const, max: [4, 5, 6] as const };
    const { floats } = packed(resolveParticleOptions(), { ...frame, bounds });
    expect(floats[PARAM_WORD.boundsMinX]).toBe(-1);
    expect(floats[PARAM_WORD.boundsMinY]).toBe(-2);
    expect(floats[PARAM_WORD.boundsMinZ]).toBe(-3);
    expect(floats[PARAM_WORD.boundsMaxX]).toBe(4);
    expect(floats[PARAM_WORD.boundsMaxY]).toBe(5);
    expect(floats[PARAM_WORD.boundsMaxZ]).toBe(6);
  });

  it('leaves the pad words zeroed', () => {
    const buffer = new ArrayBuffer(PARAMS_BYTES);
    new Uint8Array(buffer).fill(0xff);
    writeParams(buffer, resolveParticleOptions(), frame);
    const floats = new Float32Array(buffer);
    expect(floats[PARAM_WORD.padA]).toBe(0);
    expect(floats[PARAM_WORD.padB]).toBe(0);
  });

  it('stores invCell as 0 when the cell size is 0, rather than Infinity', () => {
    const { floats } = packed(resolveParticleOptions(), { ...frame, cellSize: 0 });
    expect(floats[PARAM_WORD.invCell]).toBe(0);
    expect(Number.isFinite(floats[PARAM_WORD.invCell]!)).toBe(true);
  });

  it('encodes the flags word: collisions, nbody, and bounds mode in bits 2-3', () => {
    expect(PARTICLE_FLAG.collisions).toBe(1);
    expect(PARTICLE_FLAG.nbody).toBe(2);
    expect(PARTICLE_FLAG.boundsShift).toBe(2);
    const flagsOf = (opts: Parameters<typeof resolveParticleOptions>[0]): number =>
      packed(resolveParticleOptions(opts), frame).ints[3]!;
    // reflect is 0, so the default frame carries only the collisions bit.
    expect(flagsOf({})).toBe(PARTICLE_FLAG.collisions);
    expect(flagsOf({ collisions: false })).toBe(0);
    expect(flagsOf({ collisions: false, nbody: true })).toBe(PARTICLE_FLAG.nbody);
    expect(flagsOf({ boundsMode: 'wrap' })).toBe(1 | (1 << 2));
    expect(flagsOf({ boundsMode: 'none', collisions: false })).toBe(2 << 2);
    expect(flagsOf({ boundsMode: 'reflect', collisions: true, nbody: true })).toBe(3);
  });

  it('is stable: packing twice into different buffers gives identical bytes', () => {
    const a = new ArrayBuffer(PARAMS_BYTES);
    const b = new ArrayBuffer(PARAMS_BYTES);
    const resolved = resolveParticleOptions({ damping: 0.1, boundsMode: 'wrap' });
    writeParams(a, resolved, frame);
    writeParams(b, resolved, frame);
    expect(new Uint8Array(a)).toEqual(new Uint8Array(b));
  });

  it('rejects a buffer too small to hold the uniform', () => {
    expect(() => writeParams(new ArrayBuffer(64), resolveParticleOptions(), frame)).toThrow(
      /params buffer is 64 bytes, needs 96/,
    );
  });

  it('accepts a larger buffer and writes only the first 64 bytes', () => {
    const buffer = new ArrayBuffer(256);
    new Uint8Array(buffer, 96).fill(0xab);
    writeParams(buffer, resolveParticleOptions(), frame);
    expect(new Uint8Array(buffer, 96).every((byte) => byte === 0xab)).toBe(true);
  });

  it.each([0, 3, -4, 1000])('rejects a table size that is not a power of two (%i)', (tableSize) => {
    expect(() => writeParams(new ArrayBuffer(PARAMS_BYTES), resolveParticleOptions(), { ...frame, tableSize })).toThrow(
      /tableSize must be a power of two/,
    );
  });

  it('round-trips a table size of 1, the smallest legal mask', () => {
    const { ints } = packed(resolveParticleOptions(), { ...frame, tableSize: 1 });
    expect(ints[1]).toBe(0);
  });

  it('rounds f32 words the way the shader will read them', () => {
    const { floats } = packed(resolveParticleOptions({ damping: 0.1 }));
    expect(floats[PARAM_WORD.damping]).toBe(Math.fround(0.1));
    expect(floats[PARAM_WORD.damping]).not.toBe(0.1);
  });
});

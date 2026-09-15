/**
 * `gpu/softOptions.ts` -- defaults, validation, the shared layout, and the uniform.
 *
 * Three things get pinned here that a wrong value would not fail loudly for. The
 * uniform packing, because a misaligned WGSL struct compiles, runs, and quietly uses
 * `dt` as the restitution. The cross-field refusals (`damping * fixedDt`, the absent
 * `wrap` mode), because both produce a simulation that looks broken rather than one
 * that throws. And `buildSoftLayout`, because it is the single call both tiers share:
 * if it ever stops being the single call, the CPU/GPU parity assertion degrades from
 * a bitwise claim into a tolerance nobody notices widening.
 */

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_SOFT_GRAVITY,
  DEFAULT_SOFT_OPTIONS,
  MAX_SOFT_ITERATIONS,
  SOFT_BOUNDS_MODE_BITS,
  SOFT_FIXED_DISPATCHES,
  SOFT_FLAG,
  SOFT_PARAMS_BYTES,
  SOFT_PARAMS_FLOATS,
  SOFT_PARAM_WORD,
  assertMeshFits,
  buildSoftLayout,
  resolveSoftOptions,
  writeSoftParams,
  type SoftParamsFrame,
} from '../src/gpu/softOptions.js';
import { SOFT_STRIDE, SoftMesh } from '../src/gpu/softMesh.js';
import { DEFAULT_BOUNDS } from '../src/gpu/particleField.js';
import { colorConstraints } from '../src/gpu/softColoring.js';
import { SOFT_WORKGROUP_SIZE, groupIslands } from '../src/gpu/softIslands.js';
import type { ResolvedSoftOptions, SoftBoundsMode, SoftSimOptions } from '../src/gpu/softTypes.js';

/** The bounds a built mesh uses, so the frame and the mesh agree. */
const BOUNDS = DEFAULT_BOUNDS;

const mesh = (count = 100, seed = 7): SoftMesh => new SoftMesh({ count, seed });

const frameFor = (m: SoftMesh, resolved: ResolvedSoftOptions, dt = resolved.fixedDt): SoftParamsFrame => {
  const layout = buildSoftLayout(m, resolved);
  return {
    dt,
    count: m.count,
    islandCount: layout.islands.islands,
    paddedNodes: layout.islands.paddedNodes,
    constraintCount: m.constraints.count,
    sleepAfter: resolved.sleepAfter,
    bounds: m.bounds,
  };
};

describe('resolveSoftOptions', () => {
  it('fills every default', () => {
    expect(resolveSoftOptions()).toEqual(DEFAULT_SOFT_OPTIONS);
    expect(resolveSoftOptions({})).toEqual(DEFAULT_SOFT_OPTIONS);
    expect(DEFAULT_SOFT_GRAVITY).toEqual([0, -9.81, 0]);
    expect(DEFAULT_SOFT_OPTIONS.fixedDt).toBeCloseTo(1 / 60, 12);
    expect(DEFAULT_SOFT_OPTIONS.iterations).toBe(3);
    expect(DEFAULT_SOFT_OPTIONS.sleep).toBe(false);
    expect(MAX_SOFT_ITERATIONS).toBe(64);
  });

  it('returns a resolved value for every option the interface declares', () => {
    const resolved = resolveSoftOptions();
    const declared = Object.keys({
      gravity: 0,
      damping: 0,
      restitution: 0,
      maxSpeed: 0,
      stiffness: 0,
      iterations: 0,
      boundsMode: 0,
      fixedDt: 0,
      sleep: 0,
      sleepThreshold: 0,
      sleepAfter: 0,
    } satisfies Record<keyof SoftSimOptions, 0>);
    expect(Object.keys(resolved).sort()).toEqual(declared.sort());
  });

  it('copies gravity rather than aliasing the caller array', () => {
    const gravity: [number, number, number] = [1, 2, 3];
    const resolved = resolveSoftOptions({ gravity });
    gravity[0] = 99;
    expect(resolved.gravity).toEqual([1, 2, 3]);
    expect(resolved.gravity).not.toBe(gravity);
  });

  it('rejects a gravity vector that is not three finite numbers', () => {
    expect(() =>
      resolveSoftOptions({ gravity: [1, 2] as unknown as [number, number, number] }),
    ).toThrow(/gravity must be three finite numbers/);
    expect(() =>
      resolveSoftOptions({ gravity: [1, Number.NaN, 3] as [number, number, number] }),
    ).toThrow(/gravity must be three finite numbers/);
  });

  it('refuses wrap, and says why rather than substituting reflect', () => {
    expect(() => resolveSoftOptions({ boundsMode: 'wrap' as SoftBoundsMode })).toThrow(
      /boundsMode must be one of reflect\|none, got "wrap"/,
    );
    expect(() => resolveSoftOptions({ boundsMode: 'wrap' as SoftBoundsMode })).toThrow(
      /wrapping a node tears every edge attached to it/,
    );
    expect(() => resolveSoftOptions({ boundsMode: 'bounce' as SoftBoundsMode })).toThrow(
      /boundsMode must be one of reflect\|none/,
    );
  });

  it('accepts both bounds modes it offers', () => {
    expect(resolveSoftOptions({ boundsMode: 'reflect' }).boundsMode).toBe('reflect');
    expect(resolveSoftOptions({ boundsMode: 'none' }).boundsMode).toBe('none');
    expect(SOFT_BOUNDS_MODE_BITS).toEqual({ reflect: 0, none: 1 });
  });

  it('refuses a damping that reverses velocity at this step size', () => {
    // The predict step multiplies by 1 - damping*dt, which is negative past 1 and
    // pumps energy in every step instead of removing it.
    expect(() => resolveSoftOptions({ damping: 60, fixedDt: 1 })).toThrow(
      /damping 60 with fixedDt 1 reverses velocity every step/,
    );
    expect(() => resolveSoftOptions({ damping: 2, fixedDt: 0.5 })).toThrow(
      /damping \* fixedDt must be < 1/,
    );
    // Just under the line is fine, and is what a heavily damped cloth asks for.
    expect(resolveSoftOptions({ damping: 1.9, fixedDt: 0.5 }).damping).toBe(1.9);
  });

  it.each([
    ['damping', { damping: -0.1 }],
    ['damping', { damping: 61 }],
    ['restitution', { restitution: 1.5 }],
    ['restitution', { restitution: -0.5 }],
    ['maxSpeed', { maxSpeed: 0 }],
    ['maxSpeed', { maxSpeed: Number.NaN }],
    ['stiffness', { stiffness: 1.5 }],
    ['stiffness', { stiffness: -0.1 }],
    ['iterations', { iterations: 0 }],
    ['iterations', { iterations: 65 }],
    ['fixedDt', { fixedDt: 0 }],
    ['fixedDt', { fixedDt: 2 }],
    ['sleepThreshold', { sleepThreshold: -1 }],
    ['sleepThreshold', { sleepThreshold: 1e4 }],
    ['sleepAfter', { sleepAfter: 0 }],
    ['sleepAfter', { sleepAfter: 1e7 }],
  ])('rejects an out-of-range %s', (_name, options) => {
    expect(() => resolveSoftOptions(options)).toThrow(RangeError);
  });

  it('truncates the integer options rather than rounding them', () => {
    const resolved = resolveSoftOptions({ iterations: 2.7, sleepAfter: 30.9 });
    expect(resolved.iterations).toBe(2);
    expect(resolved.sleepAfter).toBe(30);
    expect(Number.isInteger(resolved.iterations)).toBe(true);
  });

  it('keeps a stiffness of exactly 0 and exactly 1', () => {
    expect(resolveSoftOptions({ stiffness: 0 }).stiffness).toBe(0);
    expect(resolveSoftOptions({ stiffness: 1 }).stiffness).toBe(1);
  });
});

describe('assertMeshFits', () => {
  it('accepts every scene the builder produces', () => {
    for (const scene of ['cloth', 'sheets', 'cube', 'rope'] as const) {
      for (const count of [7, 100, 1000]) {
        expect(() => assertMeshFits(new SoftMesh({ count, scene, seed: 5, groups: 4 }))).not.toThrow();
      }
    }
  });

  it.each([0, 1, 2])('rejects an axis too thin for the largest diameter (axis %i)', (axis) => {
    const min: [number, number, number] = [-8, -8, -8];
    const max: [number, number, number] = [8, 8, 8];
    max[axis] = min[axis] + 0.1; // thinner than a 0.2 diameter
    const data = new Float32Array(2 * SOFT_STRIDE);
    data[7] = 0.1;
    data[SOFT_STRIDE + 7] = 0.1;
    const m = new SoftMesh(data, { count: 2, bounds: { min, max } });
    expect(() => assertMeshFits(m)).toThrow(new RegExp(`bounds axis ${axis} is 0\\.09`));
    expect(() => assertMeshFits(m)).toThrow(/cannot contain a node of diameter/);
  });

  it('rejects a mesh whose buffer does not match its count and stride', () => {
    const lying = {
      count: 4,
      bounds: BOUNDS,
      data: new Float32Array(7),
      maxRadius: () => 0.1,
    } as unknown as SoftMesh;
    expect(() => assertMeshFits(lying)).toThrow(/mesh data does not match its count and stride/);
  });
});

describe('buildSoftLayout', () => {
  it('reports the plan the two passes actually produced', () => {
    const resolved = resolveSoftOptions();
    const m = mesh(100, 7);
    const layout = buildSoftLayout(m, resolved);
    expect(layout.plan).toEqual({
      nodes: 100,
      constraints: 342,
      islands: 1,
      colors: 8,
      iterations: 3,
      nodeWorkgroups: 2,
      dispatchesPerStep: 29,
      islandSizes: [100],
      batchSizes: [48, 48, 48, 47, 44, 36, 35, 36],
    });
    // A second build of the same mesh agrees exactly, which is the property the
    // CPU/GPU plan comparison leans on.
    expect(buildSoftLayout(m, resolved).plan).toEqual(layout.plan);
  });

  it('hands back the same islands and coloring the passes produce on their own', () => {
    const m = new SoftMesh({ count: 1000, scene: 'cube', seed: 3 });
    const layout = buildSoftLayout(m, resolveSoftOptions());
    const islands = groupIslands(m);
    const coloring = colorConstraints(m);
    expect(Array.from(layout.islands.nodeOrder)).toEqual(Array.from(islands.nodeOrder));
    expect(Array.from(layout.coloring.order)).toEqual(Array.from(coloring.order));
    expect(layout.plan.islands).toBe(islands.islands);
    expect(layout.plan.colors).toBe(coloring.colors);
    expect(layout.plan.batchSizes).toEqual(coloring.batches.map((b) => b.count));
    expect(layout.plan.islandSizes).toEqual(Array.from(islands.islandSizes));
  });

  it('counts the padded node order in workgroups of 64', () => {
    expect(SOFT_WORKGROUP_SIZE).toBe(64);
    for (const count of [7, 100, 1000, 10000]) {
      const m = new SoftMesh({ count, seed: 1 });
      const layout = buildSoftLayout(m, resolveSoftOptions());
      expect(layout.plan.nodeWorkgroups).toBe(layout.islands.paddedNodes / SOFT_WORKGROUP_SIZE);
      expect(layout.islands.paddedNodes % SOFT_WORKGROUP_SIZE).toBe(0);
    }
  });

  it('moves dispatchesPerStep with the iteration count and nothing else', () => {
    const m = mesh(100, 7);
    const byIterations = [1, 3, 8, 64].map(
      (iterations) => buildSoftLayout(m, resolveSoftOptions({ iterations })).plan.dispatchesPerStep,
    );
    // Five fixed kernels plus one solve dispatch per color per iteration.
    expect(SOFT_FIXED_DISPATCHES).toBe(5);
    expect(byIterations).toEqual([13, 29, 69, 517]);
    for (const [i, iterations] of [1, 3, 8, 64].entries()) {
      expect(byIterations[i]).toBe(SOFT_FIXED_DISPATCHES + iterations * 8);
    }
  });

  it('sees every island of a disconnected mesh', () => {
    const m = new SoftMesh({ count: 100, scene: 'sheets', seed: 11, groups: 4 });
    const layout = buildSoftLayout(m, resolveSoftOptions());
    expect(layout.plan.islands).toBe(4);
    expect(layout.plan.islandSizes).toHaveLength(4);
    expect(layout.plan.islandSizes.reduce((a, b) => a + b, 0)).toBe(100);
    expect(layout.plan.nodeWorkgroups).toBeGreaterThanOrEqual(4);
  });

  it('handles a mesh with no constraints at all', () => {
    const data = new Float32Array(5 * SOFT_STRIDE);
    for (let i = 0; i < 5; i++) data[i * SOFT_STRIDE + 7] = 0.1;
    const m = new SoftMesh(data, { count: 5, bounds: BOUNDS });
    const layout = buildSoftLayout(m, resolveSoftOptions());
    expect(layout.plan).toMatchObject({ constraints: 0, colors: 0, islands: 5, batchSizes: [] });
    expect(layout.plan.dispatchesPerStep).toBe(SOFT_FIXED_DISPATCHES);
  });
});

describe('the uniform layout', () => {
  it('is 24 floats / 96 bytes, and the word offsets are all distinct', () => {
    expect(SOFT_PARAMS_FLOATS).toBe(24);
    expect(SOFT_PARAMS_BYTES).toBe(96);
    // WGSL rounds a struct's size up to its alignment, which is 16 for vec3.
    expect(SOFT_PARAMS_BYTES % 16).toBe(0);
    const words = Object.values(SOFT_PARAM_WORD);
    expect(new Set(words).size).toBe(words.length);
    expect(Math.max(...words)).toBe(SOFT_PARAMS_FLOATS - 1);
    expect(Math.min(...words)).toBe(0);
  });

  it('puts the sixteen f32 words first and the eight u32 words last', () => {
    const intWords = [
      'count',
      'islandCount',
      'paddedNodes',
      'sleepAfter',
      'flags',
      'constraintCount',
      'padA',
      'padB',
    ] as const;
    for (const name of intWords) expect(SOFT_PARAM_WORD[name]).toBeGreaterThanOrEqual(16);
    for (const [name, word] of Object.entries(SOFT_PARAM_WORD)) {
      if (!(intWords as readonly string[]).includes(name)) expect(word).toBeLessThan(16);
    }
    expect(intWords).toHaveLength(8);
  });

  it('starts every vec3 on a 16-byte boundary, as WGSL alignment requires', () => {
    for (const vector of ['gravity', 'boundsMin', 'boundsMax'] as const) {
      const x = SOFT_PARAM_WORD[`${vector}X`] as number;
      expect(x % 4, `${vector} must be 16-byte aligned`).toBe(0);
      expect(SOFT_PARAM_WORD[`${vector}Y`]).toBe(x + 1);
      expect(SOFT_PARAM_WORD[`${vector}Z`]).toBe(x + 2);
    }
  });

  it('keeps the two pad words, so the struct has no implicit gaps', () => {
    expect(SOFT_PARAM_WORD.padA).toBe(22);
    expect(SOFT_PARAM_WORD.padB).toBe(23);
  });

  it('packs the flags the way the kernels unpack them', () => {
    expect(SOFT_FLAG.sleep).toBe(1);
    expect(SOFT_FLAG.boundsShift).toBe(1);
    const flags = (options: SoftSimOptions): number => {
      const buffer = new ArrayBuffer(SOFT_PARAMS_BYTES);
      writeSoftParams(buffer, resolveSoftOptions(options), frameFor(mesh(8, 1), resolveSoftOptions(options)));
      return new Uint32Array(buffer)[SOFT_PARAM_WORD.flags];
    };
    expect(flags({})).toBe(0);
    expect(flags({ sleep: true })).toBe(1);
    expect(flags({ boundsMode: 'none' })).toBe(2);
    expect(flags({ sleep: true, boundsMode: 'none' })).toBe(3);
  });

  const packed = (resolved = resolveSoftOptions(), dt = 1 / 120) => {
    const m = mesh(100, 7);
    const buffer = new ArrayBuffer(SOFT_PARAMS_BYTES);
    writeSoftParams(buffer, resolved, frameFor(m, resolved, dt));
    return { floats: new Float32Array(buffer, 0, 16), ints: new Uint32Array(buffer, 64, 8) };
  };

  it('writes every documented word', () => {
    const resolved = resolveSoftOptions({
      gravity: [1, -2, 3],
      damping: 0.25,
      restitution: 0.75,
      maxSpeed: 12,
      stiffness: 0.5,
      sleepThreshold: 0.2,
      sleepAfter: 7,
      sleep: true,
    });
    const { floats, ints } = packed(resolved, 1 / 120);
    expect(floats[SOFT_PARAM_WORD.gravityX]).toBe(Math.fround(1));
    expect(floats[SOFT_PARAM_WORD.gravityY]).toBe(Math.fround(-2));
    expect(floats[SOFT_PARAM_WORD.gravityZ]).toBe(Math.fround(3));
    expect(floats[SOFT_PARAM_WORD.damping]).toBe(Math.fround(0.25));
    expect(floats[SOFT_PARAM_WORD.boundsMinX]).toBe(-8);
    expect(floats[SOFT_PARAM_WORD.boundsMaxY]).toBe(8);
    expect(floats[SOFT_PARAM_WORD.invDt]).toBe(Math.fround(120));
    expect(floats[SOFT_PARAM_WORD.maxSpeed]).toBe(Math.fround(12));
    expect(floats[SOFT_PARAM_WORD.dt]).toBe(Math.fround(1 / 120));
    expect(floats[SOFT_PARAM_WORD.stiffness]).toBe(Math.fround(0.5));
    expect(floats[SOFT_PARAM_WORD.restitution]).toBe(Math.fround(0.75));
    // Squared and rounded to f32, because the sleep test compares bit patterns.
    expect(floats[SOFT_PARAM_WORD.sleepThresholdSq]).toBe(Math.fround(0.2 * 0.2));
    expect(ints[SOFT_PARAM_WORD.count - 16]).toBe(100);
    expect(ints[SOFT_PARAM_WORD.islandCount - 16]).toBe(1);
    expect(ints[SOFT_PARAM_WORD.paddedNodes - 16]).toBe(128);
    expect(ints[SOFT_PARAM_WORD.sleepAfter - 16]).toBe(7);
    expect(ints[SOFT_PARAM_WORD.constraintCount - 16]).toBe(342);
    expect(ints[SOFT_PARAM_WORD.flags - 16] & SOFT_FLAG.sleep).toBe(SOFT_FLAG.sleep);
    expect(ints[SOFT_PARAM_WORD.padA - 16]).toBe(0);
    expect(ints[SOFT_PARAM_WORD.padB - 16]).toBe(0);
  });

  it('writes the default frame word for word', () => {
    const { floats, ints } = packed();
    expect(Array.from(floats)).toEqual([
      0,
      Math.fround(-9.81),
      0,
      Math.fround(0.4),
      -8,
      -8,
      -8,
      Math.fround(120),
      8,
      8,
      8,
      50,
      Math.fround(1 / 120),
      1,
      Math.fround(0.3),
      Math.fround(0.05 * 0.05),
    ]);
    expect(Array.from(ints)).toEqual([100, 1, 128, 60, 0, 342, 0, 0]);
  });

  it('follows the dt of the frame, not the resolved default', () => {
    const atDefault = packed(resolveSoftOptions(), 1 / 60);
    expect(atDefault.floats[SOFT_PARAM_WORD.dt]).toBe(Math.fround(1 / 60));
    expect(atDefault.floats[SOFT_PARAM_WORD.invDt]).toBe(Math.fround(60));
    const substepped = packed(resolveSoftOptions(), 1 / 240);
    expect(substepped.floats[SOFT_PARAM_WORD.invDt]).toBe(Math.fround(240));
  });

  it('is a pure function of its arguments', () => {
    const a = new ArrayBuffer(SOFT_PARAMS_BYTES);
    const b = new ArrayBuffer(SOFT_PARAMS_BYTES);
    const resolved = resolveSoftOptions({ stiffness: 0.75 });
    const m = mesh(64, 3);
    writeSoftParams(a, resolved, frameFor(m, resolved));
    writeSoftParams(b, resolved, frameFor(m, resolved));
    expect(new Uint8Array(a)).toEqual(new Uint8Array(b));
  });

  it('clears the pad words instead of leaving whatever was in the buffer', () => {
    const buffer = new ArrayBuffer(SOFT_PARAMS_BYTES);
    new Uint32Array(buffer).fill(0xdeadbeef);
    const resolved = resolveSoftOptions();
    writeSoftParams(buffer, resolved, frameFor(mesh(8, 1), resolved));
    const ints = new Uint32Array(buffer, 64, 8);
    expect(ints[SOFT_PARAM_WORD.padA - 16]).toBe(0);
    expect(ints[SOFT_PARAM_WORD.padB - 16]).toBe(0);
  });

  it('refuses a buffer too small to hold the struct', () => {
    const resolved = resolveSoftOptions();
    expect(() =>
      writeSoftParams(new ArrayBuffer(64), resolved, frameFor(mesh(8, 1), resolved)),
    ).toThrow(/params buffer is 64 bytes, needs 96/);
  });

  it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])('refuses a dt of %s', (dt) => {
    const resolved = resolveSoftOptions();
    expect(() => writeSoftParams(new ArrayBuffer(SOFT_PARAMS_BYTES), resolved, frameFor(mesh(8, 1), resolved, dt))).toThrow(
      /dt must be finite and > 0/,
    );
  });

  it('accepts a buffer larger than the struct', () => {
    const resolved = resolveSoftOptions();
    const buffer = new ArrayBuffer(SOFT_PARAMS_BYTES + 64);
    expect(() => writeSoftParams(buffer, resolved, frameFor(mesh(8, 1), resolved))).not.toThrow();
  });
});

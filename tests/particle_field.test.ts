/**
 * `gpu/particleField.ts` -- the interleaved f32 state container.
 *
 * Two things are load-bearing here and get the most attention: the *layout*
 * (every kernel and the renderer index into it by hardcoded offsets, so a drift
 * is silent corruption) and *determinism by seed* (the replay contract says a
 * saved run reproduces its bytes, and the initial distribution is part of those
 * bytes).
 */

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_BOUNDS,
  DEFAULT_RADIUS,
  PARTICLE_BYTES,
  PARTICLE_OFFSET,
  PARTICLE_SCENES,
  PARTICLE_STRIDE,
  ParticleField,
  boundsCenter,
  boundsInradius,
  boundsSize,
  type ParticleScene,
} from '../src/gpu/particleField.js';

describe('the record layout', () => {
  it('is 8 floats / 32 bytes, with the documented offsets', () => {
    expect(PARTICLE_STRIDE).toBe(8);
    expect(PARTICLE_BYTES).toBe(32);
    expect(PARTICLE_OFFSET).toEqual({ position: 0, radius: 3, velocity: 4, mass: 7 });
  });

  it('bytesFor agrees with the stride', () => {
    expect(ParticleField.bytesFor(1)).toBe(PARTICLE_BYTES);
    expect(ParticleField.bytesFor(100_000)).toBe(100_000 * PARTICLE_BYTES);
  });

  it('sized 100k particles fits one 16 MiB binding', () => {
    // The claim from the module header: the M3 target is not a special case that
    // needs the field split across buffers.
    expect(ParticleField.bytesFor(100_000)).toBeLessThanOrEqual(16 * 1024 * 1024);
  });

  it('allocates exactly count * stride floats', () => {
    const field = new ParticleField({ count: 7, scene: 'box', seed: 1 });
    expect(field.data.length).toBe(7 * PARTICLE_STRIDE);
    expect(field.count).toBe(7);
  });
});

describe('bounds helpers', () => {
  it('describe the default 16^3 box', () => {
    expect(boundsSize(DEFAULT_BOUNDS)).toEqual([16, 16, 16]);
    expect(boundsCenter(DEFAULT_BOUNDS)).toEqual([0, 0, 0]);
    expect(boundsInradius(DEFAULT_BOUNDS)).toBe(8);
  });

  it('track an off-centre box', () => {
    const bounds = { min: [0, -1, 2] as const, max: [4, 5, 8] as const };
    expect(boundsSize(bounds)).toEqual([4, 6, 6]);
    expect(boundsCenter(bounds)).toEqual([2, 2, 5]);
    expect(boundsInradius(bounds)).toBe(2);
  });

  it('reject a degenerate or non-finite box', () => {
    expect(() => new ParticleField({ count: 1, bounds: { min: [0, 0, 0], max: [0, 1, 1] } })).toThrow(
      /bounds axis 0/,
    );
    expect(
      () => new ParticleField({ count: 1, bounds: { min: [0, 0, 0], max: [1, Number.NaN, 1] } }),
    ).toThrow(/bounds axis 1/);
    expect(() => new ParticleField({ count: 1, bounds: { min: [0, 0, 0], max: [1, 1, -1] } })).toThrow(
      /bounds axis 2/,
    );
  });
});

describe('construction validation', () => {
  it.each([0, -1, 1.5, Number.NaN])('rejects count %s', (count) => {
    expect(() => new ParticleField({ count })).toThrow(/count must be a positive integer/);
  });

  it('rejects an unusable radius range', () => {
    expect(() => new ParticleField({ count: 2, radius: [0, 1] })).toThrow(/radius range/);
    expect(() => new ParticleField({ count: 2, radius: [1, 0.5] })).toThrow(/radius range/);
  });

  it('rejects a negative or non-finite mass, and a negative speed', () => {
    expect(() => new ParticleField({ count: 2, mass: -1 })).toThrow(/mass must be finite/);
    expect(() => new ParticleField({ count: 2, mass: Number.POSITIVE_INFINITY })).toThrow(/mass/);
    expect(() => new ParticleField({ count: 2, speed: -1 })).toThrow(/speed must be finite/);
  });

  it('rejects an unknown scene', () => {
    expect(() => new ParticleField({ count: 2, scene: 'torus' as ParticleScene })).toThrow(
      /unknown scene "torus"/,
    );
  });

  it('accepts mass 0, which means immovable', () => {
    const field = new ParticleField({ count: 3, mass: 0, scene: 'grid' });
    expect(field.mass(0)).toBe(0);
  });

  it('takes ownership of an existing buffer, and checks its length', () => {
    const data = new Float32Array(2 * PARTICLE_STRIDE).fill(0.5);
    const field = new ParticleField(data, 2);
    expect(field.data).toBe(data);
    expect(field.bounds).toEqual(DEFAULT_BOUNDS);
    expect(() => new ParticleField(new Float32Array(3), 2)).toThrow(/need 16/);
    expect(() => new ParticleField(new Float32Array(16), 0)).toThrow(/count must be a positive/);
  });
});

describe('scene presets', () => {
  it('lists five of them', () => {
    expect(PARTICLE_SCENES).toEqual(['box', 'sphere', 'shell', 'grid', 'slab']);
  });

  it.each(PARTICLE_SCENES.map((s) => [s] as const))('%s stays inside the box', (scene) => {
    const field = new ParticleField({ count: 400, scene, seed: 11, speed: 2 });
    expect(field.outOfBounds()).toBe(0);
    for (let i = 0; i < field.count; i++) {
      const [x, y, z] = field.position(i);
      expect(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)).toBe(true);
      expect(field.radius(i)).toBeGreaterThanOrEqual(DEFAULT_RADIUS[0] - 1e-6);
      expect(field.radius(i)).toBeLessThanOrEqual(DEFAULT_RADIUS[1] + 1e-6);
      expect(field.mass(i)).toBe(1);
    }
  });

  it('grid lays out a lattice whose spacing is uniform', () => {
    const field = new ParticleField({ count: 27, scene: 'grid', radius: [0.1, 0.1] });
    const xs = new Set<number>();
    for (let i = 0; i < field.count; i++) xs.add(field.position(i)[0]);
    // cbrt(27) = 3 per axis, so three distinct x planes and no two particles
    // share a position.
    expect(xs.size).toBe(3);
    const seen = new Set<string>();
    for (let i = 0; i < field.count; i++) seen.add(field.position(i).join(','));
    expect(seen.size).toBe(27);
  });

  it('slab starts in the top fifth, so a gravity run has somewhere to fall', () => {
    const field = new ParticleField({ count: 500, scene: 'slab', seed: 3 });
    // The scene insets by the largest radius before taking its 20% slice.
    const floorOfSlab = DEFAULT_BOUNDS.max[1] - DEFAULT_RADIUS[1] - boundsSize(DEFAULT_BOUNDS)[1] * 0.2;
    for (let i = 0; i < field.count; i++) {
      expect(field.position(i)[1]).toBeGreaterThanOrEqual(floorOfSlab - 1e-5);
    }
  });

  it('sphere fills the volume, shell sits on the surface', () => {
    const sphere = new ParticleField({ count: 800, scene: 'sphere', seed: 5 });
    const shell = new ParticleField({ count: 800, scene: 'shell', seed: 5 });
    const centre = boundsCenter(DEFAULT_BOUNDS);
    const dist = (f: ParticleField, i: number): number => {
      const [x, y, z] = f.position(i);
      return Math.hypot(x - centre[0], y - centre[1], z - centre[2]);
    };
    const inradius = boundsInradius(DEFAULT_BOUNDS) - DEFAULT_RADIUS[1];
    let minSphere = Infinity;
    for (let i = 0; i < sphere.count; i++) minSphere = Math.min(minSphere, dist(sphere, i));
    expect(minSphere).toBeLessThan(inradius * 0.25); // the ball is filled, not hollow
    for (let i = 0; i < shell.count; i++) {
      expect(dist(shell, i)).toBeCloseTo(inradius, 5);
    }
  });

  it('speed 0 leaves every velocity exactly zero, and the RNG unconsumed', () => {
    const field = new ParticleField({ count: 64, scene: 'box', seed: 9, speed: 0 });
    for (let i = 0; i < field.count; i++) expect(field.velocity(i)).toEqual([0, 0, 0]);
    expect(field.maxSpeed()).toBe(0);
    expect(field.kineticEnergy()).toBe(0);
  });

  it('a non-zero speed stays within +/- speed per axis', () => {
    const field = new ParticleField({ count: 500, scene: 'box', seed: 9, speed: 3 });
    for (let i = 0; i < field.count; i++) {
      for (const v of field.velocity(i)) expect(Math.abs(v)).toBeLessThanOrEqual(3);
    }
    expect(field.maxSpeed()).toBeGreaterThan(0);
  });
});

describe('determinism', () => {
  it('the same seed produces identical bytes for every scene', () => {
    for (const scene of PARTICLE_SCENES) {
      const a = new ParticleField({ count: 256, scene, seed: 42, speed: 1.5 });
      const b = new ParticleField({ count: 256, scene, seed: 42, speed: 1.5 });
      expect(a.digest(), scene).toBe(b.digest());
      expect(Array.from(a.data)).toEqual(Array.from(b.data));
    }
  });

  it('a different seed produces different bytes', () => {
    const a = new ParticleField({ count: 256, scene: 'sphere', seed: 1 });
    const b = new ParticleField({ count: 256, scene: 'sphere', seed: 2 });
    expect(a.digest()).not.toBe(b.digest());
  });

  it('a fixed radius range skips the per-particle radius draw and still fills', () => {
    const a = new ParticleField({ count: 8, scene: 'box', seed: 1, radius: [0.25, 0.25] });
    for (let i = 0; i < a.count; i++) expect(a.radius(i)).toBe(Math.fround(0.25));
  });

  it('the digest carries the element count', () => {
    const field = new ParticleField({ count: 4, scene: 'grid' });
    expect(field.digest()).toMatch(/^[0-9a-f]{16}:32$/);
  });
});

describe('accessors and mutators', () => {
  const field = (): ParticleField => new ParticleField({ count: 5, scene: 'grid', radius: [0.1, 0.2] });

  it('round-trip position and velocity through the buffer', () => {
    const f = field();
    f.setPosition(2, [1, -2, 3]);
    f.setVelocity(2, [4, 5, -6]);
    expect(f.position(2)).toEqual([1, -2, 3]);
    expect(f.velocity(2)).toEqual([4, 5, -6]);
    // The raw layout is the contract, not just the accessors.
    const radius = f.data[2 * PARTICLE_STRIDE + PARTICLE_OFFSET.radius]!;
    expect(radius).toBeGreaterThanOrEqual(0.1 - 1e-6);
    expect(radius).toBeLessThanOrEqual(0.2 + 1e-6);
    expect(radius).toBe(f.radius(2));
    expect(f.data[2 * PARTICLE_STRIDE + PARTICLE_OFFSET.mass]).toBe(1);
  });

  it.each([-1, 0.5, 5, Number.NaN])('reject an out-of-range index %s', (i) => {
    expect(() => field().position(i)).toThrow(/out of range \[0, 5\)/);
  });

  it('reject an invalid radius or mass on write', () => {
    expect(() => field().setRadius(0, 0)).toThrow(/radius must be positive/);
    expect(() => field().setMass(0, -1)).toThrow(/mass must be >= 0/);
  });

  it('maxRadius tracks the largest sphere present', () => {
    const f = field();
    f.setRadius(3, 0.75);
    expect(f.maxRadius()).toBeCloseTo(0.75, 6);
  });

  it('kineticEnergy is sum 0.5 m v^2', () => {
    const f = field();
    f.setMass(0, 2);
    f.setVelocity(0, [3, 0, 0]);
    f.setVelocity(1, [0, 1, 0]);
    expect(f.kineticEnergy()).toBeCloseTo(0.5 * 2 * 9 + 0.5 * 1 * 1, 5);
  });

  it('maxSpeed is the largest magnitude, not the largest component', () => {
    const f = field();
    f.setVelocity(0, [3, 4, 0]);
    f.setVelocity(1, [-4.9, 0, 0]);
    expect(f.maxSpeed()).toBeCloseTo(5, 5);
  });

  it('outOfBounds counts centres only, on every axis', () => {
    const f = new ParticleField({ count: 4, scene: 'grid' });
    expect(f.outOfBounds()).toBe(0);
    f.setPosition(0, [8.001, 0, 0]);
    f.setPosition(1, [0, -8.5, 0]);
    f.setPosition(2, [0, 0, 9]);
    expect(f.outOfBounds()).toBe(3);
  });
});

describe('empty / clone / copyFrom', () => {
  it('empty is all zeros and still a valid field', () => {
    const f = ParticleField.empty(3);
    expect(f.data.every((v) => v === 0)).toBe(true);
    expect(f.count).toBe(3);
    expect(f.maxRadius()).toBe(0);
  });

  it('clone copies the bytes but not the buffer', () => {
    const a = new ParticleField({ count: 16, scene: 'sphere', seed: 7 });
    const b = a.clone();
    expect(b.data).not.toBe(a.data);
    expect(b.digest()).toBe(a.digest());
    b.setPosition(0, [0, 0, 0]);
    expect(b.digest()).not.toBe(a.digest());
  });

  it('copyFrom overwrites in place, and refuses a size mismatch', () => {
    const a = new ParticleField({ count: 16, scene: 'sphere', seed: 7 });
    const b = ParticleField.empty(16);
    b.copyFrom(a);
    expect(b.digest()).toBe(a.digest());
    expect(() => ParticleField.empty(8).copyFrom(a)).toThrow(/cannot copy 16 particles into 8/);
  });

  it('keeps the bounds it was given', () => {
    const bounds = { min: [-1, -1, -1] as const, max: [1, 1, 1] as const };
    const f = ParticleField.empty(2, bounds);
    expect(f.bounds).toBe(bounds);
    expect(f.clone().bounds).toEqual(bounds);
  });
});

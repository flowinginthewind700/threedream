/**
 * `gpu/particleCpu.ts` -- the CPU particle backend.
 *
 * These specs double as the specification for the WGSL kernels: every behaviour
 * asserted here (free-fall value, reflect invariant, impulse symmetry, cutoff)
 * has a matching kernel, and `e2e/particles_gpu.spec.ts` compares the two
 * backends' state directly. Where a number is asserted it is the *discrete*
 * semi-implicit Euler value, not the continuous one, because that is what both
 * backends compute.
 */

import { describe, expect, it } from 'vitest';
import {
  CpuParticleSystem,
  POSITION_CORRECTION,
  createCpuParticleSystem,
} from '../src/gpu/particleCpu.js';
import {
  PARTICLE_STRIDE,
  ParticleField,
  type Bounds,
  type Vec3Tuple,
} from '../src/gpu/particleField.js';
import { resolveParticleOptions } from '../src/gpu/particleOptions.js';
import type { ParticleSimOptions } from '../src/gpu/particleTypes.js';

/** A field with hand-placed particles; velocities and masses optional. */
function placed(
  particles: readonly { p: Vec3Tuple; v?: Vec3Tuple; r?: number; m?: number }[],
  bounds: Bounds = { min: [-8, -8, -8], max: [8, 8, 8] },
): ParticleField {
  const data = new Float32Array(particles.length * PARTICLE_STRIDE);
  particles.forEach(({ p, v, r, m }, i) => {
    const o = i * PARTICLE_STRIDE;
    data[o] = p[0];
    data[o + 1] = p[1];
    data[o + 2] = p[2];
    data[o + 3] = r ?? 0.1;
    data[o + 4] = v?.[0] ?? 0;
    data[o + 5] = v?.[1] ?? 0;
    data[o + 6] = v?.[2] ?? 0;
    data[o + 7] = m ?? 1;
  });
  return new ParticleField(data, particles.length, bounds);
}

const ONE = (): ParticleField =>
  new ParticleField({ count: 1, scene: 'grid', radius: [0.1, 0.1] });

function sim(
  f: ParticleField,
  options: ParticleSimOptions = {},
): CpuParticleSystem {
  return new CpuParticleSystem({ field: f, options });
}

describe('the contract', () => {
  it('is named, deterministic, and exposes its resolved options', () => {
    const system = sim(ONE(), { damping: 0.5 });
    expect(system.name).toBe('cpu');
    expect(system.deterministic).toBe(true);
    expect(system.options).toEqual(resolveParticleOptions({ damping: 0.5 }));
    expect(system.fixedDt).toBeCloseTo(1 / 60, 12);
    expect(system.count).toBe(1);
    expect(system.bounds).toBe(system.field.bounds);
    expect(system.steps).toBe(0);
    expect(system.time).toBe(0);
  });

  it('the factory builds the same thing as the constructor', () => {
    const system = createCpuParticleSystem({ field: ONE() });
    expect(system).toBeInstanceOf(CpuParticleSystem);
    expect(system.name).toBe('cpu');
  });

  it('counts steps and simulated seconds, independently of frames', () => {
    const system = sim(ONE(), { fixedDt: 0.01 });
    system.advance(5);
    expect(system.steps).toBe(5);
    expect(system.time).toBeCloseTo(0.05, 10);
    system.step(0.5); // an explicit dt does not change the clock's own rate
    expect(system.steps).toBe(6);
    expect(system.time).toBeCloseTo(0.06, 10);
  });

  it('requires a field', () => {
    expect(
      () => new CpuParticleSystem({ field: undefined as unknown as ParticleField }),
    ).toThrow(/needs a ParticleField/);
  });

  it('rejects a field its bounds cannot contain', () => {
    const f = placed([{ p: [0, 0, 0], r: 5 }], { min: [-1, -1, -1], max: [1, 1, 1] });
    expect(() => sim(f)).toThrow(/cannot contain a particle of diameter 10/);
  });

  it('rejects a non-positive or non-finite dt', () => {
    const system = sim(ONE());
    expect(() => system.step(0)).toThrow(/dt must be finite and positive/);
    expect(() => system.step(-1)).toThrow(/dt must be finite and positive/);
    expect(() => system.step(Number.NaN)).toThrow(/dt must be finite and positive/);
  });

  it('rejects a negative advance', () => {
    const system = sim(ONE());
    expect(() => system.advance(-1)).toThrow(/non-negative integer/);
    expect(() => system.advance(1.5)).toThrow(/non-negative integer/);
    expect(() => system.advance(0)).not.toThrow();
  });

  it('refuses to step after dispose, and dispose is idempotent', () => {
    const system = sim(ONE());
    system.dispose();
    system.dispose();
    expect(() => system.step()).toThrow(/has been disposed/);
  });
});

describe('integration', () => {
  it('free fall follows semi-implicit Euler exactly', () => {
    const dt = 1 / 60;
    const g = -9.81;
    const system = sim(ONE(), {
      gravity: [0, g, 0],
      collisions: false,
      boundsMode: 'none',
      fixedDt: dt,
    });
    expect(system.field.position(0)).toEqual([0, 0, 0]);
    system.advance(10);
    // v_n = g*n*dt and y_n = g*dt^2 * n(n+1)/2, which is the discrete integral
    // of that velocity -- deliberately not 0.5*g*t^2.
    expect(system.field.velocity(0)[1]).toBeCloseTo(g * 10 * dt, 5);
    expect(system.field.position(0)[1]).toBeCloseTo(g * dt * dt * 55, 5);
    expect(system.field.position(0)[0]).toBe(0);
  });

  it('zero gravity and zero velocity leave the field byte-identical', () => {
    const f = ONE();
    const system = sim(f, { gravity: [0, 0, 0], collisions: false });
    const before = f.digest();
    system.advance(30);
    expect(f.digest()).toBe(before);
    expect(system.stats().maxSpeed).toBe(0);
    expect(system.stats().kineticEnergy).toBe(0);
  });

  it('damping multiplies velocity by 1 - damping*dt each step', () => {
    const dt = 0.1;
    const damping = 2;
    const f = placed([{ p: [0, 0, 0], v: [10, 0, 0] }]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      damping,
      collisions: false,
      fixedDt: dt,
    });
    system.advance(3);
    expect(f.velocity(0)[0]).toBeCloseTo(10 * Math.pow(1 - damping * dt, 3), 5);
  });

  it('clamps speed at maxSpeed, on every axis together', () => {
    const f = placed([{ p: [0, 0, 0], v: [3, 4, 0] }]); // |v| = 5
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      maxSpeed: 1,
      boundsMode: 'none',
    });
    system.step();
    expect(system.field.maxSpeed()).toBeCloseTo(1, 5);
    // Direction is preserved by the clamp, not just magnitude.
    const [vx, vy] = f.velocity(0);
    expect(vx / vy).toBeCloseTo(3 / 4, 5);
  });

  it('a runaway gravity field cannot exceed maxSpeed', () => {
    const system = sim(ONE(), {
      gravity: [0, -1e5, 0],
      collisions: false,
      boundsMode: 'none',
      maxSpeed: 20,
    });
    system.advance(60);
    expect(system.stats().maxSpeed).toBeLessThanOrEqual(20 + 1e-4);
  });

  it('an explicit dt overrides fixedDt for that step only', () => {
    const a = sim(placed([{ p: [0, 0, 0], v: [1, 0, 0] }]), {
      gravity: [0, 0, 0],
      collisions: false,
      boundsMode: 'none',
      fixedDt: 0.1,
    });
    a.step(0.5);
    expect(a.field.position(0)[0]).toBeCloseTo(0.5, 6);
  });
});

describe('bounds', () => {
  it('reflect keeps every particle inside, for a long fall', () => {
    const f = new ParticleField({
      count: 200,
      scene: 'slab',
      seed: 2,
      speed: 4,
      radius: [0.05, 0.1],
    });
    const system = sim(f, { collisions: false, restitution: 0.7 });
    for (let i = 0; i < 300; i++) {
      system.step();
      expect(system.stats().escaped, `step ${i}`).toBe(0);
    }
    expect(f.outOfBounds()).toBe(0);
    for (let i = 0; i < f.count; i++) {
      // The wall a particle reflects off is inset by *its own* radius, so the
      // bound is per particle rather than one box for the whole field.
      const r = f.radius(i);
      const [x, y, z] = f.position(i);
      expect(x, `particle ${i}`).toBeGreaterThanOrEqual(f.bounds.min[0] + r - 1e-4);
      expect(x, `particle ${i}`).toBeLessThanOrEqual(f.bounds.max[0] - r + 1e-4);
      expect(y, `particle ${i}`).toBeGreaterThanOrEqual(f.bounds.min[1] + r - 1e-4);
      expect(y, `particle ${i}`).toBeLessThanOrEqual(f.bounds.max[1] - r + 1e-4);
      expect(z, `particle ${i}`).toBeGreaterThanOrEqual(f.bounds.min[2] + r - 1e-4);
      expect(z, `particle ${i}`).toBeLessThanOrEqual(f.bounds.max[2] - r + 1e-4);
    }
  });

  it('reflect reverses and scales the velocity of a wall hit', () => {
    const f = placed([{ p: [7.9, 0, 0], v: [10, 0, 0], r: 0.1 }]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      restitution: 0.5,
      fixedDt: 0.1,
    });
    system.step(); // would reach x = 8.9, past the wall at 7.9
    expect(f.velocity(0)[0]).toBeCloseTo(-5, 5);
    expect(f.position(0)[0]).toBeLessThanOrEqual(7.9);
    expect(f.position(0)[0]).toBeGreaterThanOrEqual(-7.9);
    expect(system.stats().escaped).toBe(0);
  });

  it('reflect handles the low wall too, on all three axes', () => {
    for (const axis of [0, 1, 2]) {
      const p: [number, number, number] = [0, 0, 0];
      const v: [number, number, number] = [0, 0, 0];
      p[axis] = -7.9;
      v[axis] = -10;
      const f = placed([{ p, v, r: 0.1 }]);
      const system = sim(f, {
        gravity: [0, 0, 0],
        collisions: false,
        restitution: 0.5,
        fixedDt: 0.1,
      });
      system.step();
      expect(f.velocity(0)[axis], `axis ${axis}`).toBeCloseTo(5, 5);
      expect(system.stats().escaped, `axis ${axis}`).toBe(0);
    }
  });

  it('wrap folds a particle back into the box without touching velocity', () => {
    const f = placed([{ p: [7.9, 0, 0], v: [10, 0, 0], r: 0.1 }]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      boundsMode: 'wrap',
      fixedDt: 0.1,
    });
    system.step(); // x would be 8.9, i.e. 0.9 past the max
    expect(f.position(0)[0]).toBeCloseTo(-8 + 0.9, 5);
    expect(f.velocity(0)[0]).toBeCloseTo(10, 6); // unchanged, unlike reflect
    expect(system.stats().escaped).toBe(0);
    system.advance(40);
    expect(f.outOfBounds()).toBe(0);
  });

  it('wrap also folds from below the minimum', () => {
    const f = placed([{ p: [-7.9, 0, 0], v: [-10, 0, 0], r: 0.1 }]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      boundsMode: 'wrap',
      fixedDt: 0.1,
    });
    system.step(); // x would be -8.9, i.e. 0.9 below the min
    expect(f.position(0)[0]).toBeCloseTo(8 - 0.9, 5);
  });

  it('none lets particles leave, and the escape counter says so', () => {
    const f = placed([
      { p: [0, 0, 0], v: [0, 0, 0] },
      { p: [7, 7, 7], v: [10, 10, 10] },
    ]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      boundsMode: 'none',
      fixedDt: 0.5,
    });
    system.step();
    expect(system.stats().escaped).toBe(1);
    expect(f.outOfBounds()).toBe(1);
  });

  it('a particle exactly on the wall is not an escape', () => {
    const f = placed([{ p: [8, 8, 8], r: 0.1 }]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      boundsMode: 'none',
    });
    system.step();
    expect(system.stats().escaped).toBe(0);
  });
});

describe('collisions', () => {
  it('POSITION_CORRECTION is the relaxation both backends use', () => {
    expect(POSITION_CORRECTION).toBe(0.5);
  });

  it('pushes two overlapping spheres apart, converging on the sum of radii', () => {
    const f = placed([
      { p: [0, 0, 0], r: 0.2 },
      { p: [0.1, 0, 0], r: 0.2 },
    ]);
    const system = sim(f, { gravity: [0, 0, 0], collisions: true, boundsMode: 'none' });
    const gap = (): number => Math.abs(f.position(1)[0] - f.position(0)[0]);
    expect(gap()).toBeCloseTo(0.1, 6);
    let previous = gap();
    for (let i = 0; i < 12; i++) {
      system.step();
      expect(gap(), `step ${i}`).toBeGreaterThan(previous);
      previous = gap();
    }
    expect(gap()).toBeGreaterThan(0.39);
    expect(gap()).toBeLessThanOrEqual(0.4 + 1e-5);
    expect(system.stats().escaped).toBe(0);
  });

  it('reports one contact per pair, not per particle', () => {
    const f = placed([
      { p: [0, 0, 0], r: 0.2 },
      { p: [0.1, 0, 0], r: 0.2 },
    ]);
    const system = sim(f, { gravity: [0, 0, 0], boundsMode: 'none' });
    system.step();
    expect(system.stats().contacts).toBe(1);
  });

  it('an elastic head-on pair swaps velocities and conserves momentum', () => {
    const f = placed([
      { p: [-0.15, 0, 0], v: [2, 0, 0], r: 0.2 },
      { p: [0.15, 0, 0], v: [-2, 0, 0], r: 0.2 },
    ]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      restitution: 1,
      boundsMode: 'none',
      collisions: true,
    });
    const before = f.kineticEnergy();
    system.step();
    expect(f.velocity(0)[0]).toBeCloseTo(-2, 5);
    expect(f.velocity(1)[0]).toBeCloseTo(2, 5);
    expect(f.kineticEnergy()).toBeCloseTo(before, 4);
  });

  it('an inelastic pair loses exactly the restitution fraction of approach speed', () => {
    const f = placed([
      { p: [-0.15, 0, 0], v: [2, 0, 0], r: 0.2 },
      { p: [0.15, 0, 0], v: [-2, 0, 0], r: 0.2 },
    ]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      restitution: 0.5,
      boundsMode: 'none',
    });
    system.step();
    // Equal masses: the approach speed of 4 becomes a separation speed of 2.
    expect(f.velocity(0)[0]).toBeCloseTo(-1, 5);
    expect(f.velocity(1)[0]).toBeCloseTo(1, 5);
    expect(f.kineticEnergy()).toBeLessThan(4);
  });

  it('mass 0 is immovable, and reflects the moving particle', () => {
    const f = placed([
      { p: [-0.15, 0, 0], v: [2, 0, 0], r: 0.2, m: 1 },
      { p: [0.15, 0, 0], v: [0, 0, 0], r: 0.2, m: 0 },
    ]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      restitution: 0.6,
      boundsMode: 'none',
    });
    system.step();
    expect(f.velocity(0)[0]).toBeCloseTo(-2 * 0.6, 5);
    expect(f.velocity(1)).toEqual([0, 0, 0]);
    expect(f.position(1)[0]).toBeCloseTo(0.15, 6); // never corrected
  });

  it('a pair of immovables overlaps forever and does not produce NaN', () => {
    const f = placed([
      { p: [0, 0, 0], r: 0.2, m: 0 },
      { p: [0.1, 0, 0], r: 0.2, m: 0 },
    ]);
    const system = sim(f, { gravity: [0, 0, 0], boundsMode: 'none' });
    system.advance(5);
    expect(f.data.every((v) => Number.isFinite(v))).toBe(true);
    expect(Math.abs(f.position(1)[0] - f.position(0)[0])).toBeCloseTo(0.1, 6);
  });

  it('separating spheres are pushed apart but get no impulse', () => {
    const f = placed([
      { p: [-0.15, 0, 0], v: [-2, 0, 0], r: 0.2 },
      { p: [0.15, 0, 0], v: [2, 0, 0], r: 0.2 },
    ]);
    const system = sim(f, { gravity: [0, 0, 0], boundsMode: 'none', restitution: 1 });
    system.step();
    expect(f.velocity(0)[0]).toBeCloseTo(-2, 5);
    expect(f.velocity(1)[0]).toBeCloseTo(2, 5);
    expect(system.stats().contacts).toBe(1); // still an overlap, still counted
  });

  it('non-overlapping spheres do not interact', () => {
    const f = placed([
      { p: [-1, 0, 0], r: 0.2 },
      { p: [1, 0, 0], r: 0.2 },
    ]);
    const system = sim(f, { gravity: [0, 0, 0], boundsMode: 'none' });
    system.step();
    expect(system.stats().contacts).toBe(0);
    expect(f.velocity(0)).toEqual([0, 0, 0]);
  });

  it('exactly coincident centres have no normal, and are skipped not NaN-ed', () => {
    const f = placed([
      { p: [0, 0, 0], r: 0.2 },
      { p: [0, 0, 0], r: 0.2 },
    ]);
    const system = sim(f, { gravity: [0, 0, 0], boundsMode: 'none' });
    system.step();
    expect(f.data.every((v) => Number.isFinite(v))).toBe(true);
    expect(system.stats().contacts).toBe(1);
  });

  it('reports hash overflow instead of silently dropping contacts', () => {
    const particles: { p: Vec3Tuple; r: number }[] = [];
    for (let i = 0; i < 12; i++) particles.push({ p: [i * 0.001, 0, 0] as Vec3Tuple, r: 0.2 });
    const f = placed(particles);
    const system = sim(f, { gravity: [0, 0, 0], boundsMode: 'none', bucketCapacity: 2 });
    system.step();
    expect(system.stats().hashOverflow).toBeGreaterThan(0);
    expect(system.hashStats().overflow).toBe(system.stats().hashOverflow);
    expect(system.hashStats().inserted).toBeLessThan(12);
  });

  it('a settling pile stays inside the box and stops moving', () => {
    const f = new ParticleField({
      count: 220,
      scene: 'slab',
      seed: 21,
      speed: 0,
      radius: [0.15, 0.15],
      bounds: { min: [-2, -2, -2], max: [2, 2, 2] },
    });
    const system = sim(f, { restitution: 0.3, damping: 0.4, maxSpeed: 20 });
    expect(system.cellSize).toBeCloseTo(0.3, 6);
    for (let i = 0; i < 400; i++) {
      system.step();
      expect(system.stats().escaped, `step ${i}`).toBe(0);
    }
    expect(f.outOfBounds()).toBe(0);
    const settled = system.stats().kineticEnergy;
    system.advance(200);
    expect(system.stats().kineticEnergy).toBeLessThan(settled + 1e-3);
    expect(system.field.maxSpeed()).toBeLessThan(5);
    expect(system.stats().hashOverflow).toBe(0);
  });

  it('disabling collisions turns the whole pass off', () => {
    const f = placed([
      { p: [0, 0, 0], r: 0.2 },
      { p: [0.1, 0, 0], r: 0.2 },
    ]);
    const system = sim(f, { gravity: [0, 0, 0], boundsMode: 'none', collisions: false });
    system.step();
    expect(system.stats().contacts).toBe(0);
    expect(Math.abs(f.position(1)[0] - f.position(0)[0])).toBeCloseTo(0.1, 6);
  });
});

describe('n-body', () => {
  const pair = (strength = 1, cutoff = 0): ParticleField =>
    placed([
      { p: [-0.5, 0, 0], r: 0.05 },
      { p: [0.5, 0, 0], r: 0.05 },
    ]);

  it('attracts two masses toward each other', () => {
    const f = pair();
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      nbody: true,
      boundsMode: 'none',
      fixedDt: 0.01,
    });
    system.step();
    expect(f.velocity(0)[0]).toBeGreaterThan(0);
    expect(f.velocity(1)[0]).toBeLessThan(0);
    // a = G*m / (d^2 + soft^2)^1.5 with d = 1, m = 1, soft = 0.01
    const expected = 1 / Math.pow(1 + 1e-4, 1.5);
    expect(f.velocity(0)[0]).toBeCloseTo(expected * 0.01, 6);
  });

  it('scales with nbodyStrength and mass', () => {
    const f = placed([
      { p: [-0.5, 0, 0], r: 0.05, m: 1 },
      { p: [0.5, 0, 0], r: 0.05, m: 4 },
    ]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      nbody: true,
      nbodyStrength: 2,
      boundsMode: 'none',
      fixedDt: 0.01,
    });
    system.step();
    const expected = (2 * 4) / Math.pow(1 + 1e-4, 1.5);
    expect(f.velocity(0)[0]).toBeCloseTo(expected * 0.01, 5);
  });

  it('a cutoff skips pairs beyond it', () => {
    const f = pair();
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      nbody: true,
      cutoff: 0.5,
      boundsMode: 'none',
    });
    system.step();
    expect(f.velocity(0)).toEqual([0, 0, 0]);
    expect(f.velocity(1)).toEqual([0, 0, 0]);
  });

  it('softening keeps a coincident pair finite', () => {
    const f = placed([
      { p: [0, 0, 0], r: 0.05 },
      { p: [0, 0, 0], r: 0.05 },
    ]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      nbody: true,
      softening: 0.5,
      boundsMode: 'none',
    });
    system.step();
    expect(f.data.every((v) => Number.isFinite(v))).toBe(true);
    expect(system.stats().maxSpeed).toBeLessThan(system.options.maxSpeed);
  });

  it('a particle does not attract itself', () => {
    const f = placed([{ p: [0, 0, 0], r: 0.05 }]);
    const system = sim(f, {
      gravity: [0, 0, 0],
      collisions: false,
      nbody: true,
      boundsMode: 'none',
    });
    system.step();
    expect(f.velocity(0)).toEqual([0, 0, 0]);
  });

  it('gravity and n-body add, they do not replace each other', () => {
    const f = placed([{ p: [0, 0, 0], r: 0.05 }]);
    const system = sim(f, {
      gravity: [0, -1, 0],
      collisions: false,
      nbody: true,
      boundsMode: 'none',
      fixedDt: 0.1,
    });
    system.step();
    expect(f.velocity(0)[1]).toBeCloseTo(-0.1, 6);
  });

  it('is off unless asked for, and allocates no scratch when off', () => {
    const system = sim(placed([{ p: [0, 0, 0] }]), {
      gravity: [0, 0, 0],
      collisions: false,
    });
    expect(system.options.nbody).toBe(false);
    system.step();
    expect(system.field.velocity(0)).toEqual([0, 0, 0]);
  });
});

describe('determinism', () => {
  const build = (): CpuParticleSystem =>
    sim(
      new ParticleField({ count: 128, scene: 'sphere', seed: 8, speed: 2 }),
      { restitution: 0.5, damping: 0.05 },
    );

  it('two runs from the same seed produce identical bytes', () => {
    const a = build();
    const b = build();
    a.advance(60);
    b.advance(60);
    expect(a.digest()).toBe(b.digest());
    expect(a.digest()).toMatch(/^[0-9a-f]{16}:1024$/);
  });

  it('advance(n) equals n step() calls', () => {
    const a = build();
    const b = build();
    a.advance(30);
    for (let i = 0; i < 30; i++) b.step();
    expect(a.digest()).toBe(b.digest());
    expect(a.steps).toBe(b.steps);
  });

  it('interleaving stats() calls does not perturb the run', () => {
    const a = build();
    const b = build();
    for (let i = 0; i < 20; i++) {
      a.step();
      b.step();
      b.stats();
      b.digest();
      b.hashStats();
    }
    expect(a.digest()).toBe(b.digest());
  });

  it('a different seed diverges', () => {
    const a = build();
    const b = sim(new ParticleField({ count: 128, scene: 'sphere', seed: 9, speed: 2 }), {
      restitution: 0.5,
      damping: 0.05,
    });
    a.advance(30);
    b.advance(30);
    expect(a.digest()).not.toBe(b.digest());
  });

  it('the field is mutated in place, so a renderer sees the same buffer', () => {
    const f = new ParticleField({ count: 8, scene: 'grid', seed: 1 });
    const system = sim(f, {});
    expect(system.field).toBe(f);
    const data = f.data;
    system.advance(10);
    expect(f.data).toBe(data);
    expect(f.digest()).toBe(system.digest());
  });
});

describe('stats', () => {
  it('reports the five documented counters', () => {
    const system = sim(new ParticleField({ count: 32, scene: 'sphere', seed: 6, speed: 1 }), {});
    system.advance(5);
    const stats = system.stats();
    expect(Object.keys(stats).sort()).toEqual(
      ['contacts', 'escaped', 'hashOverflow', 'kineticEnergy', 'maxSpeed'].sort(),
    );
    for (const value of Object.values(stats)) expect(Number.isFinite(value)).toBe(true);
    expect(stats.maxSpeed).toBe(system.field.maxSpeed());
    expect(stats.kineticEnergy).toBe(system.field.kineticEnergy());
    expect(stats.escaped).toBe(0);
  });

  it('kinetic energy falls as a bouncing field settles', () => {
    const system = sim(
      new ParticleField({
        count: 120,
        scene: 'slab',
        seed: 12,
        speed: 0,
        radius: [0.12, 0.12],
        bounds: { min: [-2, -2, -2], max: [2, 2, 2] },
      }),
      { restitution: 0.2, damping: 0.5 },
    );
    system.advance(20);
    const falling = system.stats().kineticEnergy;
    expect(falling).toBeGreaterThan(0);
    system.advance(600);
    expect(system.stats().kineticEnergy).toBeLessThan(falling);
    expect(system.stats().escaped).toBe(0);
  });
});

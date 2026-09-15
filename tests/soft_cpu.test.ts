/**
 * `gpu/softCpu.ts` -- the deterministic reference every soft-body kernel is
 * measured against.
 *
 * Three kinds of claim are pinned here and they fail differently. Hand-computed
 * goldens (a two-node spring, a four-node chain, one node in free fall) catch a
 * wrong formula, because the expected number comes from the algebra rather than
 * from a previous run. Digests catch a *changed* formula, which is the point:
 * this module is the specification the WGSL parity spec compares against, so an
 * arithmetic tweak has to break something here before it can quietly move the
 * target the device tier is aimed at. The structural claims -- colored order
 * rather than index order, sleep per island, islands that cannot touch each other
 * -- are the ones the GPU backend inherits by construction and cannot re-test
 * without a device.
 *
 * The f32 discipline is asserted rather than trusted. Several goldens below are
 * exact single-precision values (`-4.986729145050049`, not the `-4.98675` the
 * same parabola gives in f64), which only come out right if every intermediate
 * rounds where WGSL rounds it.
 */

import { describe, expect, it } from 'vitest';
import { DEFAULT_BOUNDS } from '../src/gpu/particleField.js';
import { colorConstraints } from '../src/gpu/softColoring.js';
import { CpuSoftSystem, createCpuSoftSystem } from '../src/gpu/softCpu.js';
import {
  SOFT_OFFSET,
  SOFT_STRIDE,
  SoftMesh,
  emptyConstraints,
  type SoftConstraints,
} from '../src/gpu/softMesh.js';
import { buildSoftLayout, resolveSoftOptions } from '../src/gpu/softOptions.js';
import type { SoftSimOptions } from '../src/gpu/softTypes.js';

const f = Math.fround;

/** No gravity, no damping: the only settings a hand-computed golden survives. */
const STILL: SoftSimOptions = { gravity: [0, 0, 0], damping: 0 };

/** The x coordinate of every node, which is all a chain test needs to see. */
function xs(mesh: SoftMesh): number[] {
  return Array.from({ length: mesh.count }, (_, i) => mesh.position(i)[0]);
}

interface ChainSpec {
  spacing?: number;
  rest?: number;
  stiffness?: number;
  pinned?: readonly number[];
}

/**
 * A straight chain along +x with node `i` at `i * spacing`.
 *
 * Hand-built rather than scene-built so every input to a golden is a literal in
 * this file: `chain(2, { spacing: 2, rest: 1 })` is a spring stretched to twice
 * its rest length, and where it lands after one iteration is arithmetic a reader
 * can check without running anything.
 */
function chain(nodes: number, spec: ChainSpec = {}): SoftMesh {
  const spacing = spec.spacing ?? 1.5;
  const rest = spec.rest ?? 1;
  const stiffness = spec.stiffness ?? 1;
  const pinned = spec.pinned ?? [];
  const data = new Float32Array(nodes * SOFT_STRIDE);
  for (let i = 0; i < nodes; i++) {
    const o = i * SOFT_STRIDE;
    data[o] = i * spacing;
    data[o + SOFT_OFFSET.invMass] = pinned.includes(i) ? 0 : 1;
    data[o + SOFT_OFFSET.radius] = 0.1;
  }
  const count = nodes - 1;
  const ends = new Uint32Array(count * 2);
  for (let k = 0; k < count; k++) {
    ends[k * 2] = k;
    ends[k * 2 + 1] = k + 1;
  }
  const constraints: SoftConstraints = {
    count,
    ends,
    rest: new Float32Array(count).fill(rest),
    stiffness: new Float32Array(count).fill(stiffness),
  };
  return new SoftMesh(data, { count: nodes, constraints });
}

/** `count` unconnected nodes, so every one is its own island and none is constrained. */
function cloud(count: number, vy = 0, radius = 0.1): SoftMesh {
  const data = new Float32Array(count * SOFT_STRIDE);
  for (let i = 0; i < count; i++) {
    const o = i * SOFT_STRIDE;
    data[o] = i - (count - 1) / 2;
    data[o + SOFT_OFFSET.invMass] = 1;
    data[o + SOFT_OFFSET.velocity + 1] = vy;
    data[o + SOFT_OFFSET.radius] = radius;
  }
  return new SoftMesh(data, { count, constraints: emptyConstraints() });
}

/** One node at `y` moving at `vy`, for the wall and clamp cases. */
function lone(y: number, vy: number, radius = 0.5): SoftMesh {
  const data = new Float32Array(SOFT_STRIDE);
  data[1] = y;
  data[SOFT_OFFSET.invMass] = 1;
  data[SOFT_OFFSET.velocity + 1] = vy;
  data[SOFT_OFFSET.radius] = radius;
  return new SoftMesh(data, { count: 1, constraints: emptyConstraints() });
}

const cloth100 = (): SoftMesh => new SoftMesh({ count: 100, seed: 7 });

describe('the SoftSystem contract', () => {
  it('declares the tier, its determinism and its race-freedom', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100() });
    expect(sys).toBeInstanceOf(CpuSoftSystem);
    expect(sys.name).toBe('cpu');
    // The reference is the replayable tier; `softGpu.ts` declares the opposite.
    expect(sys.deterministic).toBe(true);
    // True here by inspection (the batches are colored) and on the device by
    // construction. Both tiers claim it, which is what makes it a contract.
    expect(sys.raceFree).toBe(true);
    expect(sys.count).toBe(100);
    expect(sys.fixedDt).toBeCloseTo(1 / 60, 12);
    expect(sys.bounds).toBe(sys.mesh.bounds);
    expect(sys.steps).toBe(0);
    expect(sys.time).toBe(0);
  });

  it('reports the options resolved by the one shared resolver', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100(), options: { iterations: 5 } });
    expect(sys.options).toEqual(resolveSoftOptions({ iterations: 5 }));
    expect(sys.options.iterations).toBe(5);
  });

  it('exposes exactly the plan buildSoftLayout produces', () => {
    const mesh = cloth100();
    const resolved = resolveSoftOptions();
    const sys = createCpuSoftSystem({ mesh });
    // Not a deep copy: the same object, so a spec comparing the two tiers' plans
    // is comparing the artefact both actually dispatched against.
    expect(sys.plan).toBe(sys.layout.plan);
    expect(sys.plan).toEqual(buildSoftLayout(mesh, resolved).plan);
    expect(sys.layout.islands.islands).toBe(sys.plan.islands);
    expect(sys.plan).toMatchObject({
      nodes: 100,
      constraints: 342,
      islands: 1,
      colors: 8,
      iterations: 3,
      nodeWorkgroups: 2,
      dispatchesPerStep: 29,
    });
    expect(Array.from(sys.plan.islandSizes)).toEqual([100]);
    expect(Array.from(sys.plan.batchSizes)).toEqual([48, 48, 48, 47, 44, 36, 35, 36]);
  });

  it('counts dispatches as five fixed passes plus iterations * colors', () => {
    for (const iterations of [1, 3, 8]) {
      const sys = createCpuSoftSystem({ mesh: cloth100(), options: { iterations } });
      expect(sys.plan.dispatchesPerStep).toBe(5 + iterations * sys.plan.colors);
    }
  });
});

describe('determinism', () => {
  it('reproduces a digest bit for bit across instances', () => {
    const a = createCpuSoftSystem({ mesh: new SoftMesh({ count: 100, seed: 7 }) });
    const b = createCpuSoftSystem({ mesh: new SoftMesh({ count: 100, seed: 7 }) });
    a.advance(60);
    b.advance(60);
    expect(a.digest()).toBe(b.digest());
    expect(a.digest()).toMatch(/^[0-9a-f]{16}:800$/);
    // Two instances must not share scratch: stepping one leaves the other alone.
    a.step();
    expect(a.digest()).not.toBe(b.digest());
  });

  it('digests the raw node bytes, so the count is count * stride', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100() });
    expect(sys.digest()).toBe(sys.mesh.digest());
    expect(Number(sys.digest().split(':')[1])).toBe(100 * SOFT_STRIDE);
  });

  it('treats advance(n) as exactly n steps', () => {
    const a = createCpuSoftSystem({ mesh: new SoftMesh({ count: 100, seed: 7 }) });
    const b = createCpuSoftSystem({ mesh: new SoftMesh({ count: 100, seed: 7 }) });
    a.advance(30);
    for (let i = 0; i < 30; i++) b.step();
    expect(a.digest()).toBe(b.digest());
    expect(a.steps).toBe(30);
    expect(a.time).toBeCloseTo(30 / 60, 12);
  });

  it('honours an explicit dt, and keeps `time` on the fixed clock', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100() });
    const half = createCpuSoftSystem({ mesh: cloth100() });
    const whole = createCpuSoftSystem({ mesh: cloth100() });
    half.step(1 / 30);
    whole.step(1 / 30);
    whole.step(1 / 30);
    sys.step();
    // A different dt is a different trajectory, and two of them are not one.
    expect(half.digest()).not.toBe(sys.digest());
    expect(whole.digest()).not.toBe(half.digest());
    // `time` is simulated fixed steps, not the sum of the dts handed to step():
    // the HUD reads a clock that does not drift with a caller's choice.
    expect(half.time).toBeCloseTo(1 / 60, 12);
    expect(whole.time).toBeCloseTo(2 / 60, 12);
  });
});

describe('scene goldens', () => {
  it('reproduces cloth100 after 60 steps', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100() });
    sys.advance(60);
    expect(sys.digest()).toBe('8c2955ea182facc6:800');
    expect(sys.stats()).toEqual({
      escaped: 0,
      maxSpeed: 0.10659191552722179,
      maxConstraintError: 0.018157457932829857,
      awakeIslands: 1,
      sleepingIslands: 0,
      kineticEnergy: 0.07636603215963332,
    });
    // One more step is a different state, so the digest above is not an accident
    // of a solver that stopped moving.
    sys.step();
    expect(sys.digest()).toBe('094315249e5f10b2:800');
  });

  it('pins the top row and leaves it pinned', () => {
    const mesh = cloth100();
    const pinned = Array.from({ length: mesh.count }, (_, i) => i).filter((i) => mesh.isPinned(i));
    expect(pinned).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const before = pinned.map((i) => mesh.position(i));
    const sys = createCpuSoftSystem({ mesh });
    sys.advance(60);
    pinned.forEach((i, k) => {
      expect(mesh.position(i)).toEqual(before[k]);
      expect(mesh.velocity(i)).toEqual([0, 0, 0]);
    });
  });

  it('agrees with the mesh-side f64 helpers to within f32 noise', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100() });
    sys.advance(60);
    const stats = sys.stats();
    // `SoftMesh` computes the same two quantities in f64 for inspection. They are
    // the same number up to rounding, which is the tolerance a parity spec uses.
    expect(stats.maxConstraintError).toBeCloseTo(sys.mesh.maxConstraintError(), 6);
    expect(stats.maxSpeed).toBeCloseTo(sys.mesh.maxSpeed(), 6);
  });

  it('converges monotonically as iterations rise', () => {
    const errors: number[] = [];
    for (const iterations of [1, 2, 3, 8, 32]) {
      const sys = createCpuSoftSystem({
        mesh: new SoftMesh({ count: 400, seed: 7 }),
        options: { iterations },
      });
      sys.advance(120);
      errors.push(sys.stats().maxConstraintError);
    }
    expect(errors.map((e) => Number(e.toFixed(5)))).toEqual([
      0.2737, 0.17896, 0.09314, 0.03577, 0.00814,
    ]);
    for (let k = 1; k < errors.length; k++) expect(errors[k]!).toBeLessThan(errors[k - 1]!);
    // 32 iterations buys a factor of 30 over one: the knob the demo exposes does
    // something, and this is the size of the something.
    expect(errors[0]! / errors[4]!).toBeGreaterThan(30);
  });

  it('holds a 1000-node lattice together inside the box', () => {
    const sys = createCpuSoftSystem({
      mesh: new SoftMesh({ count: 1000, scene: 'cube', seed: 3 }),
      options: { iterations: 4 },
    });
    expect(sys.plan.islands).toBe(1);
    expect(sys.plan.colors).toBe(6);
    sys.advance(300);
    const stats = sys.stats();
    expect(stats.escaped).toBe(0);
    expect(sys.mesh.outOfBounds()).toBe(0);
    expect(stats.maxSpeed).toBeLessThan(2);
    expect(stats.maxConstraintError).toBeLessThan(0.2);
    expect(sys.digest()).toBe('f7239c8c4d5a9aa0:8000');
  });

  it('drains kinetic energy under damping with no gravity', () => {
    const sys = createCpuSoftSystem({
      mesh: new SoftMesh({ count: 100, seed: 7, speed: 2 }),
      options: { gravity: [0, 0, 0], damping: 1.5, boundsMode: 'none' },
    });
    const ke: number[] = [];
    for (let k = 0; k < 5; k++) {
      ke.push(Number(sys.stats().kineticEnergy.toFixed(3)));
      sys.advance(10);
    }
    expect(ke).toEqual([177.759, 12.475, 4.553, 2.585, 2.141]);
    expect(sys.stats().kineticEnergy).toBeLessThan(ke[4]!);
  });
});

describe('the solve, computed by hand', () => {
  it('integrates one free node onto the semi-implicit parabola', () => {
    const sys = createCpuSoftSystem({
      mesh: lone(0, 0, 0.1),
      options: { boundsMode: 'none', damping: 0, maxSpeed: 1e6 },
    });
    sys.advance(60);
    const dt = 1 / 60;
    // v is updated before p each step, so the sum is dt^2 * g * n(n+1)/2, not the
    // n^2/2 of the explicit form. The exact f32 value is the assertion; the
    // analytic one is the check that it is the right parabola.
    expect(sys.mesh.position(0)[1]).toBe(-4.986729145050049);
    expect(sys.mesh.position(0)[1]).toBeCloseTo((-9.81 * dt * dt * 60 * 61) / 2, 4);
    expect(sys.mesh.velocity(0)[1]).toBe(-9.80996036529541);
    // Derived from the position change, so it carries the f32 rounding of 60
    // accumulated displacements: -9.80996 rather than the exact -9.81.
    expect(sys.mesh.velocity(0)[1]).toBeCloseTo(-9.81 * dt * 60, 4);
  });

  it('relaxes a stretched spring to its rest length in one iteration', () => {
    const sys = createCpuSoftSystem({
      mesh: chain(2, { spacing: 2, rest: 1 }),
      options: { ...STILL, iterations: 1 },
    });
    sys.step();
    // dist 2, rest 1, equal masses: each node takes half the 1-unit overshoot.
    expect(xs(sys.mesh)).toEqual([0.5, 1.5]);
    expect(sys.stats().maxConstraintError).toBe(0);
    // Velocity is derived from the position change, so the pair leaves the step
    // with equal and opposite speeds: momentum conserved without a mass matrix.
    expect(sys.mesh.velocity(0)[0]).toBeCloseTo(30, 5);
    expect(sys.mesh.velocity(1)[0]).toBeCloseTo(-30, 5);
  });

  it('scales a correction by per-edge and by global stiffness identically', () => {
    const perEdge = createCpuSoftSystem({
      mesh: chain(2, { spacing: 2, rest: 1, stiffness: 0.5 }),
      options: { ...STILL, iterations: 1 },
    });
    const global = createCpuSoftSystem({
      mesh: chain(2, { spacing: 2, rest: 1 }),
      options: { ...STILL, iterations: 1, stiffness: 0.5 },
    });
    perEdge.step();
    global.step();
    expect(xs(perEdge.mesh)).toEqual([0.25, 1.75]);
    expect(xs(global.mesh)).toEqual(xs(perEdge.mesh));
    expect(global.digest()).toBe(perEdge.digest());
  });

  it('does nothing at all when stiffness is zero', () => {
    const sys = createCpuSoftSystem({
      mesh: chain(2, { spacing: 2, rest: 1, stiffness: 0 }),
      options: { ...STILL, iterations: 4 },
    });
    sys.advance(10);
    expect(xs(sys.mesh)).toEqual([0, 2]);
    expect(sys.stats().maxConstraintError).toBe(1);
  });

  it('leaves a fully pinned edge alone and stays finite', () => {
    const sys = createCpuSoftSystem({ mesh: chain(2, { spacing: 2, rest: 1, pinned: [0, 1] }) });
    sys.advance(60);
    expect(xs(sys.mesh)).toEqual([0, 2]);
    expect(sys.digest()).toBe('d9521a19de50454b:16');
    // w = 0 is the one divide that could produce a NaN, and it is guarded.
    expect(sys.stats().maxConstraintError).toBe(1);
    expect(sys.stats().kineticEnergy).toBe(0);
    expect(sys.stats().maxSpeed).toBe(0);
    expect(Array.from(sys.mesh.data).some(Number.isNaN)).toBe(false);
  });

  it('skips a zero-length edge instead of normalising by zero', () => {
    const sys = createCpuSoftSystem({ mesh: chain(2, { spacing: 0, rest: 1 }), options: STILL });
    sys.advance(5);
    expect(xs(sys.mesh)).toEqual([0, 0]);
    expect(sys.digest()).toBe('29034019524a6bf3:16');
    expect(Array.from(sys.mesh.data).some(Number.isNaN)).toBe(false);
    // No direction to push along, so the measure pass skips it too: the error is
    // 0 rather than the Infinity a `|0 - rest| / rest` over a zero distance gives.
    expect(sys.stats().maxConstraintError).toBe(0);
  });

  it('pulls a free node to the rest length of its pinned anchor', () => {
    const sys = createCpuSoftSystem({
      mesh: chain(2, { spacing: 3, rest: 1, pinned: [0] }),
      options: { damping: 0.5, iterations: 8 },
    });
    sys.advance(240);
    const anchor = sys.mesh.position(0);
    const [bx, by] = sys.mesh.position(1);
    expect(anchor).toEqual([0, 0, 0]);
    expect(sys.mesh.velocity(0)).toEqual([0, 0, 0]);
    // It hangs off the anchor at exactly one rest length, below and inward.
    expect(Math.hypot(bx - anchor[0], by - anchor[1])).toBeCloseTo(1, 6);
    expect(by).toBeLessThan(0);
    expect(bx).toBeLessThan(3);
    expect(sys.stats().maxConstraintError).toBeLessThan(1e-6);
  });
});

describe('the colored order', () => {
  /**
   * The same PBD correction, written out longhand in f32, over a caller-chosen
   * edge order. Independent of `softCpu.ts` on purpose: if both sides shared a
   * helper the assertion would only prove the helper agrees with itself.
   */
  function referenceSolve(mesh: SoftMesh, order: readonly number[]): number[] {
    const p = new Float32Array(mesh.count * 3);
    for (let i = 0; i < mesh.count; i++) p[i * 3] = mesh.position(i)[0]!;
    const { ends, rest } = mesh.constraints;
    for (const e of order) {
      const ia = ends[e * 2]!;
      const ib = ends[e * 2 + 1]!;
      const dx = f(p[ib * 3]! - p[ia * 3]!);
      const dy = f(p[ib * 3 + 1]! - p[ia * 3 + 1]!);
      const dz = f(p[ib * 3 + 2]! - p[ia * 3 + 2]!);
      const d2 = f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz));
      if (!(d2 > 0)) continue;
      const dist = f(Math.sqrt(d2));
      const invDist = f(1 / dist);
      const invW = f(1 / 2);
      const s = f(f(f(1) * f(f(dist - rest[e]!) * invDist)) * invW);
      const sa = f(s * 1);
      const sb = f(s * 1);
      p[ia * 3] = f(p[ia * 3]! + f(dx * sa));
      p[ib * 3] = f(p[ib * 3]! - f(dx * sb));
    }
    return Array.from({ length: mesh.count }, (_, i) => p[i * 3]!);
  }

  it('solves in colored batch order, which is not index order', () => {
    // Two meshes with identical bytes, because a system steps the one it holds:
    // the reference has to start from the *initial* positions, not the solved ones.
    const untouched = chain(4, { spacing: 1.5, rest: 1 });
    const coloring = colorConstraints(untouched);
    // A path of four nodes two-colors: edges 0 and 2 are disjoint, edge 1 shares
    // a node with both and has to wait for the second batch.
    expect(coloring.colors).toBe(2);
    expect(Array.from(coloring.order)).toEqual([0, 2, 1]);

    const sys = createCpuSoftSystem({
      mesh: chain(4, { spacing: 1.5, rest: 1 }),
      options: { ...STILL, iterations: 1 },
    });
    expect(Array.from(sys.plan.batchSizes)).toEqual([2, 1]);
    sys.step();

    const landed = xs(sys.mesh);
    expect(landed).toEqual(referenceSolve(untouched, Array.from(coloring.order)));
    // The whole point: an index-order walk of the same edges, with the same
    // arithmetic, lands somewhere else. A solver that quietly iterated edges in
    // index order would still look like a cloth and would not match the GPU.
    expect(referenceSolve(untouched, [0, 1, 2])).toEqual([0.25, 1.625, 3.0625, 4.0625]);
    expect(landed).toEqual([0.25, 1.75, 2.75, 4.25]);
    expect(sys.digest()).toBe('d1501382009ef856:32');
  });

  it('walks every batch of every iteration', () => {
    const one = createCpuSoftSystem({
      mesh: chain(4, { spacing: 1.5, rest: 1 }),
      options: { ...STILL, iterations: 1 },
    });
    const three = createCpuSoftSystem({
      mesh: chain(4, { spacing: 1.5, rest: 1 }),
      options: { ...STILL, iterations: 3 },
    });
    // One step each, so the only difference between the two runs is the number of
    // times the whole colored order was walked.
    one.step();
    three.step();
    expect(three.plan.dispatchesPerStep).toBe(5 + 3 * 2);
    // More iterations is a different state and a tighter mesh, so the extra passes
    // really ran rather than being folded away.
    expect(three.digest()).not.toBe(one.digest());
    expect(three.stats().maxConstraintError).toBeLessThan(one.stats().maxConstraintError);
  });
});

describe('bounds and clamps', () => {
  it('reflects a node off the floor and scales the bounce by restitution', () => {
    const reflected = createCpuSoftSystem({
      mesh: lone(-7, -30, 0.5),
      options: { damping: 0, boundsMode: 'reflect', restitution: 0.5 },
    });
    reflected.step();
    // The floor for a node of radius 0.5 is y = -7.5. One step takes it to
    // -7.5027, i.e. 0.0027 through the wall, and the reflection puts it back out
    // by half that penetration with half the speed.
    expect(reflected.mesh.position(0)[1]).toBe(-7.4986371994018555);
    expect(reflected.mesh.velocity(0)[1]).toBe(15.08175277709961);
    expect(reflected.stats().escaped).toBe(0);

    const passed = createCpuSoftSystem({
      mesh: lone(-7, -30, 0.5),
      options: { damping: 0, boundsMode: 'none', maxSpeed: 1e6 },
    });
    passed.step();
    expect(passed.mesh.position(0)[1]).toBe(-7.502725124359131);
    expect(passed.mesh.velocity(0)[1]).toBe(-30.16350555419922);
  });

  it('keeps escaped at zero under reflect and counts every escape under none', () => {
    const held = createCpuSoftSystem({
      mesh: cloud(4, -40),
      options: { boundsMode: 'reflect', damping: 0 },
    });
    held.advance(600);
    expect(held.stats().escaped).toBe(0);
    expect(held.mesh.outOfBounds()).toBe(0);
    // Four nodes dropped at 40 u/s all come to rest on the same floor, just above
    // `min + radius`: the clamp is structural, not a property of these numbers.
    const floor = DEFAULT_BOUNDS.min[1] + 0.1;
    for (let i = 0; i < 4; i++) {
      expect(held.mesh.position(i)[1]).toBeCloseTo(floor, 3);
    }

    const lost = createCpuSoftSystem({
      mesh: cloud(4, -40),
      options: { boundsMode: 'none', damping: 0, maxSpeed: 1e6 },
    });
    lost.advance(400);
    expect(lost.stats().escaped).toBe(4);
    expect(lost.mesh.outOfBounds()).toBe(4);
  });

  it('clamps speed, not position', () => {
    const sys = createCpuSoftSystem({
      mesh: cloud(2),
      options: { boundsMode: 'none', damping: 0, gravity: [0, -1000, 0], maxSpeed: 1 },
    });
    sys.advance(30);
    expect(sys.stats().maxSpeed).toBe(1);
    expect(sys.mesh.velocity(0)[1]).toBe(-1);
    // A clamp on velocity still lets the node travel, and `none` puts nothing back.
    expect(sys.mesh.position(0)[1]).toBeLessThan(DEFAULT_BOUNDS.min[1]);
  });

  it('leaves a rope in free fall at its rest lengths', () => {
    const sys = createCpuSoftSystem({
      mesh: new SoftMesh({ count: 7, scene: 'rope', seed: 5 }),
      options: { boundsMode: 'none', damping: 0 },
    });
    sys.step();
    // Every node accelerates identically, so the shape survives and the only error
    // is f32 rounding of a displacement every node shares.
    expect(sys.stats().maxConstraintError).toBeLessThan(1e-5);
    expect(sys.stats().escaped).toBe(0);
    sys.advance(59);
    expect(sys.stats().maxConstraintError).toBeLessThan(0.05);
  });
});

describe('islands', () => {
  it('cannot let one island affect another by a single bit', () => {
    const options: SoftSimOptions = { gravity: [0, 0, 0], damping: 0, iterations: 3 };
    const plain = createCpuSoftSystem({
      mesh: new SoftMesh({ count: 400, scene: 'sheets', seed: 5, groups: 4, speed: 2 }),
      options,
    });
    const pinned = createCpuSoftSystem({
      mesh: new SoftMesh({ count: 400, scene: 'sheets', seed: 5, groups: 4, speed: 2 }),
      options,
    });
    expect(pinned.plan.islands).toBe(4);
    expect(Array.from(pinned.plan.islandSizes)).toEqual([100, 100, 100, 100]);
    // Pin island 0 and move one of its nodes somewhere else entirely. Islands 1-3
    // share no edge with it, so their trajectories must be bit-identical to the run
    // that never touched it -- which is what "island parallel" has to mean.
    for (let i = 0; i < 100; i++) pinned.mesh.pin(i);
    pinned.mesh.setPosition(0, [6, 6, 6]);
    plain.advance(20);
    pinned.advance(20);

    const islandOf = pinned.layout.islands.islandOfNode;
    let compared = 0;
    for (let i = 0; i < 400; i++) {
      if (islandOf[i] === 0) continue;
      expect(pinned.mesh.position(i)).toEqual(plain.mesh.position(i));
      compared++;
    }
    expect(compared).toBe(300);
    expect(pinned.mesh.position(0)).toEqual([6, 6, 6]);
    expect(pinned.mesh.velocity(0)).toEqual([0, 0, 0]);
  });

  it('accounts for every island as either awake or asleep', () => {
    const sys = createCpuSoftSystem({
      mesh: new SoftMesh({ count: 100, scene: 'sheets', seed: 11, groups: 4 }),
      options: {
        sleep: true,
        sleepAfter: 5,
        sleepThreshold: 0.5,
        damping: 2,
        gravity: [0, 0, 0],
      },
    });
    for (let i = 0; i < 12; i++) {
      sys.step();
      const stats = sys.stats();
      expect(stats.awakeIslands + stats.sleepingIslands).toBe(sys.plan.islands);
    }
  });
});

describe('sleep', () => {
  /** Four sheets of 25 nodes, no gravity: quiet from the first step on. */
  const sleepy = (over: SoftSimOptions = {}): CpuSoftSystem =>
    createCpuSoftSystem({
      mesh: new SoftMesh({ count: 100, scene: 'sheets', seed: 11, groups: 4 }),
      options: {
        sleep: true,
        sleepAfter: 5,
        sleepThreshold: 0.5,
        damping: 2,
        gravity: [0, 0, 0],
        ...over,
      },
    });

  it('sleeps an island once it has been quiet for sleepAfter steps', () => {
    const sys = sleepy();
    const sleeping: number[] = [];
    for (let i = 0; i < 8; i++) {
      sys.step();
      sleeping.push(sys.stats().sleepingIslands);
    }
    // Not 4,3,2,1: all four sheets are quiet at once, and all four cross the
    // counter on the same step.
    expect(sleeping).toEqual([0, 0, 0, 0, 4, 4, 4, 4]);
    expect(sys.stats().awakeIslands).toBe(0);
    expect(sys.stats().maxSpeed).toBe(0);
  });

  it('freezes the state and the stats of a sleeping island', () => {
    const sys = sleepy();
    sys.advance(10);
    const digest = sys.digest();
    const stats = sys.stats();
    sys.advance(30);
    expect(sys.digest()).toBe(digest);
    expect(sys.stats()).toEqual(stats);
    // The clock keeps running: a sleeping island is skipped, not a paused sim.
    expect(sys.steps).toBe(40);
    expect(sys.time).toBeCloseTo(40 / 60, 12);
  });

  it('skips every pass for a sleeping island, publish included', () => {
    const sys = sleepy();
    sys.advance(10);
    const at = sys.mesh.position(0);
    sys.mesh.setPosition(0, [at[0]! + 0.3, at[1]!, at[2]!]);
    const nudged = sys.digest();
    sys.advance(5);
    // The nudge survives: nothing predicted, solved or finalised it away.
    expect(sys.digest()).toBe(nudged);
    expect(sys.stats().maxConstraintError).toBeGreaterThan(0);
    // But publish still ran over it, so a renderer is never left drawing a stale
    // copy of a body somebody moved by hand.
    const published = new Float32Array(sys.count * 3);
    sys.copyPublishedTo(published);
    expect(published[0]).toBe(sys.mesh.data[0]);

    sys.wake();
    sys.step();
    // Awake, the same nudge is a constraint error and the solve works it down.
    expect(sys.digest()).not.toBe(nudged);
    expect(Math.abs(sys.mesh.position(0)[0]! - at[0]!)).toBeLessThan(0.3);
  });

  it('restarts the quiet counter on wake', () => {
    const sys = sleepy();
    sys.advance(10);
    const asleep = sys.digest();
    sys.wake();
    expect(sys.stats().sleepingIslands).toBe(0);
    expect(sys.stats().awakeIslands).toBe(4);
    const after: number[] = [];
    for (let i = 0; i < 8; i++) {
      sys.step();
      after.push(sys.stats().sleepingIslands);
    }
    expect(after).toEqual([0, 0, 0, 0, 4, 4, 4, 4]);
    expect(sys.digest()).not.toBe(asleep);
  });

  it('never sleeps when sleep is off, however long it runs', () => {
    const sys = sleepy({ sleep: false });
    sys.advance(600);
    expect(sys.stats().sleepingIslands).toBe(0);
    expect(sys.stats().awakeIslands).toBe(4);
  });

  it('never sleeps a body that is still moving', () => {
    const sys = createCpuSoftSystem({
      mesh: new SoftMesh({ count: 100, scene: 'sheets', seed: 11, groups: 4 }),
      options: { sleep: true, sleepAfter: 1, sleepThreshold: 1e-9 },
    });
    sys.advance(300);
    expect(sys.stats().sleepingIslands).toBe(0);
  });

  it('lets a hanging cloth settle and sleep on its own', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100(), options: { sleep: true, damping: 3 } });
    let firstAsleep = -1;
    for (let i = 0; i < 400; i++) {
      sys.step();
      if (sys.stats().sleepingIslands === 1 && firstAsleep < 0) firstAsleep = i + 1;
    }
    // Default threshold 0.05 and sleepAfter 60: a second and a half of drape, then
    // a second of quiet before the island comes out of the dispatch.
    expect(firstAsleep).toBe(153);
    const settled = sys.digest();
    expect(sys.stats().maxConstraintError).toBeCloseTo(0.01738, 5);
    sys.advance(50);
    expect(sys.digest()).toBe(settled);
    expect(sys.steps).toBe(450);
  });
});

describe('publish', () => {
  it('is publishable on frame zero, before any step', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100() });
    const target = new Float32Array(300);
    expect(sys.copyPublishedTo(target)).toBe(300 * 4);
    expect(target[0]).toBe(sys.mesh.data[0]);
    expect(target[1]).toBe(sys.mesh.data[1]);
    expect(target[299]).toBe(sys.mesh.data[99 * SOFT_STRIDE + 2]);
    expect(sys.steps).toBe(0);
  });

  it('writes tight xyz, three floats per node', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100() });
    sys.advance(3);
    const target = new Float32Array(sys.count * 3);
    sys.copyPublishedTo(target);
    for (const i of [0, 1, 37, 99]) {
      expect(Array.from(target.subarray(i * 3, i * 3 + 3))).toEqual(sys.mesh.position(i));
    }
  });

  it('truncates to a smaller target and reports the bytes it wrote', () => {
    const sys = createCpuSoftSystem({ mesh: cloth100() });
    const target = new Float32Array(6);
    expect(sys.copyPublishedTo(target)).toBe(24);
    expect(Array.from(target)).toEqual([
      sys.mesh.data[0],
      sys.mesh.data[1],
      sys.mesh.data[2],
      sys.mesh.data[SOFT_STRIDE],
      sys.mesh.data[SOFT_STRIDE + 1],
      sys.mesh.data[SOFT_STRIDE + 2],
    ]);
  });
});

/**
 * The scale test is the expensive one, and it is expensive for the same reason
 * the trainer's convergence tests are: the property under test is a wall clock
 * over the solver's inner loop, which is exactly what v8 coverage instrumentation
 * counts. Measured on this machine, 60 steps of a 10k-node cloth cost 767ms
 * un-instrumented and 1891ms instrumented in isolation -- and the instrumented
 * number is not a ceiling on its own, because the run shares the machine with 39
 * other files. Under a full `test:coverage` run it reaches ~3170ms, over the
 * 3000ms ceiling below, with no change to the code being measured.
 *
 * It also contributes nothing to coverage: the passes it walks are the same ones
 * the goldens above walk at counts in the tens, so what it adds is a scale
 * assertion rather than a line. Same trade as `tests/trainer.test.ts`, and safe
 * for the same reason -- the skip is confined to the coverage script, so
 * `npm test` still makes the assertion, and CI runs both.
 */
const underCoverage = process.env.COVERAGE === '1';

describe.skipIf(underCoverage)('scale', () => {
  it('runs a 10k-node cloth on the reference tier inside a ceiling', () => {
    const sys = createCpuSoftSystem({ mesh: new SoftMesh({ count: 10_000, seed: 1 }) });
    expect(sys.plan).toMatchObject({
      nodes: 10_000,
      constraints: 39_402,
      islands: 1,
      colors: 8,
      nodeWorkgroups: 157,
    });
    const started = performance.now();
    sys.advance(60);
    const elapsed = performance.now() - started;
    expect(sys.digest()).toBe('de1469331dfee403:80000');
    expect(sys.stats().escaped).toBe(0);
    expect(sys.mesh.outOfBounds()).toBe(0);
    // A ceiling, not a benchmark. The plan's per-step budget for the reference
    // tier is measured by the soft-body bench, where a slow machine fails on a
    // number rather than on a wall clock shared with 39 other test files.
    expect(elapsed).toBeLessThan(3000);
  });
});

describe('errors', () => {
  it('refuses a dt it cannot integrate', () => {
    const sys = createCpuSoftSystem({ mesh: cloud(2), options: { boundsMode: 'none' } });
    for (const dt of [0, -1, NaN, Infinity]) {
      expect(() => sys.step(dt)).toThrow(`dt must be finite and positive, got ${dt}`);
    }
    expect(sys.steps).toBe(0);
  });

  it('refuses an advance it cannot count', () => {
    const sys = createCpuSoftSystem({ mesh: cloud(2), options: { boundsMode: 'none' } });
    expect(() => sys.advance(-1)).toThrow('advance needs a non-negative integer, got -1');
    expect(() => sys.advance(1.5)).toThrow('advance needs a non-negative integer, got 1.5');
    expect(() => sys.advance(NaN)).toThrow('advance needs a non-negative integer, got NaN');
    sys.advance(0);
    expect(sys.steps).toBe(0);
  });

  it('refuses a mesh or a bounds mode it cannot simulate', () => {
    expect(() => createCpuSoftSystem({ mesh: undefined as never })).toThrow(TypeError);
    expect(() => createCpuSoftSystem({ mesh: undefined as never })).toThrow(
      'CpuSoftSystem needs a SoftMesh with a positive count',
    );
    // `'wrap'` is refused by the shared resolver rather than mapped to reflect.
    expect(() =>
      createCpuSoftSystem({ mesh: cloud(2), options: { boundsMode: 'wrap' as never } }),
    ).toThrow('wrapping a node tears every edge attached to it');
  });

  it('makes dispose idempotent and every later call loud', () => {
    const sys = createCpuSoftSystem({ mesh: cloud(2), options: { boundsMode: 'none' } });
    sys.dispose();
    sys.dispose();
    expect(() => sys.step()).toThrow('CpuSoftSystem has been disposed');
    expect(() => sys.wake()).toThrow('CpuSoftSystem has been disposed');
    expect(() => sys.copyPublishedTo(new Float32Array(6))).toThrow(
      'CpuSoftSystem has been disposed',
    );
    // Reading is still safe: a HUD that renders one frame late must not throw.
    expect(sys.stats().escaped).toBe(0);
    expect(sys.digest()).toMatch(/:16$/);
  });
});

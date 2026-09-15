/**
 * `gpu/softMesh.ts` -- the soft-body state container and its four scene builders.
 *
 * Three properties carry the layer and get the most attention:
 *
 * 1. *The record layout.* Every kernel, both backends and the renderer index
 *    into `data` by hard-coded offsets, so a drift here is silent corruption at
 *    10k nodes rather than a thrown error. The offsets are also the reason the
 *    GPU upload is one `writeBuffer`: the bytes are already the buffer.
 *
 * 2. *A freshly built scene is inside its own bounds, at every count.* This is
 *    what makes `outOfBounds()` mean something on frame zero. The default node
 *    radius is derived from the lattice spacing, which for a three-node rope is
 *    enormous, so the box cap exists purely to keep this property true -- and a
 *    scene/count/bounds matrix is the only way to pin it.
 *
 * 3. *Determinism by seed.* The replay contract says a saved run reproduces its
 *    bytes, and the initial mesh is the first of those bytes. Golden digests are
 *    recorded here so a change to the RNG consumption order -- for example
 *    drawing velocities before jitter -- turns a test red instead of quietly
 *    changing every saved run.
 */

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_JITTER,
  RADIUS_BOX_FRACTION,
  RADIUS_FRACTION,
  SCENE_EXTENT_FRACTION,
  SOFT_BYTES,
  SOFT_OFFSET,
  SOFT_SCENES,
  SOFT_STIFFNESS,
  SOFT_STRIDE,
  SoftMesh,
  assertConstraints,
  assertTriangles,
  defaultExtent,
  emptyConstraints,
  sizeForScene,
  type SoftScene,
} from '../src/gpu/softMesh.js';
import { DEFAULT_BOUNDS, boundsSize, type Bounds } from '../src/gpu/particleField.js';

/** Node degrees, so the graph-shape claims are checked rather than assumed. */
function degrees(mesh: SoftMesh): number[] {
  const out = new Array<number>(mesh.count).fill(0);
  const { ends, count } = mesh.constraints;
  for (let k = 0; k < count; k++) {
    out[ends[k * 2]]++;
    out[ends[k * 2 + 1]]++;
  }
  return out;
}

/** Rest-length spread, as `[min, max]`. */
function restRange(mesh: SoftMesh): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = 0;
  for (const r of mesh.constraints.rest) {
    if (r < min) min = r;
    if (r > max) max = r;
  }
  return [min, max];
}

/** The three boxes every scene is built and checked against. */
const BOXES: readonly [string, Bounds][] = [
  ['the default 16^3 box', DEFAULT_BOUNDS],
  ['a thin off-centre box', { min: [0, 0, 0], max: [2, 4, 6] }],
  ['a negative-corner box', { min: [-1, -2, -3], max: [9, 8, 7] }],
];

describe('the record layout', () => {
  it('is 8 floats / 32 bytes, with the documented offsets', () => {
    // Byte-identical to a particle record: `pos.xyz | invMass | vel.xyz | radius`.
    // The one field whose meaning changed is slot 3, mass -> invMass, because a
    // pinned node is `invMass === 0` and the solver can test that without a divide.
    expect(SOFT_STRIDE).toBe(8);
    expect(SOFT_BYTES).toBe(32);
    expect(SOFT_OFFSET).toEqual({ position: 0, invMass: 3, velocity: 4, radius: 7 });
  });

  it('bytesFor agrees with the stride, and 10k nodes fit one binding', () => {
    expect(SoftMesh.bytesFor(1)).toBe(SOFT_BYTES);
    expect(SoftMesh.bytesFor(10)).toBe(320);
    expect(SoftMesh.bytesFor(10_000)).toBe(320_000);
    expect(SoftMesh.bytesFor(10_000)).toBeLessThanOrEqual(16 * 1024 * 1024);
  });

  it('allocates exactly count * stride floats', () => {
    const mesh = new SoftMesh({ count: 7, seed: 1 });
    expect(mesh.count).toBe(7);
    expect(mesh.data.length).toBe(7 * SOFT_STRIDE);
    expect(mesh.data.byteLength).toBe(7 * SOFT_BYTES);
  });

  it('reads and writes through the same offsets the shaders will use', () => {
    const mesh = new SoftMesh({ count: 4, seed: 1, jitter: 0 });
    mesh.setPosition(2, [1, 2, 3]);
    mesh.setVelocity(2, [4, 5, 6]);
    mesh.setRadius(2, 0.25);
    const o = 2 * SOFT_STRIDE;
    expect(Array.from(mesh.data.subarray(o, o + 3))).toEqual([1, 2, 3]);
    expect(mesh.data[o + SOFT_OFFSET.invMass]).toBe(1);
    expect(Array.from(mesh.data.subarray(o + SOFT_OFFSET.velocity, o + 7))).toEqual([4, 5, 6]);
    expect(mesh.data[o + SOFT_OFFSET.radius]).toBe(0.25);
    expect(mesh.position(2)).toEqual([1, 2, 3]);
    expect(mesh.velocity(2)).toEqual([4, 5, 6]);
  });
});

describe('scene extent and default radius', () => {
  it('fill 0.4 of the smallest box axis', () => {
    expect(SCENE_EXTENT_FRACTION).toBe(0.4);
    expect(defaultExtent(DEFAULT_BOUNDS)).toBeCloseTo(6.4, 6);
    expect(defaultExtent({ min: [0, 0, 0], max: [2, 4, 6] })).toBeCloseTo(0.8, 6);
    expect(boundsSize(DEFAULT_BOUNDS)).toEqual([16, 16, 16]);
  });

  it('derive the radius from the spacing, then cap it by the box', () => {
    expect(RADIUS_FRACTION).toBe(0.35);
    expect(RADIUS_BOX_FRACTION).toBe(0.05);
    // 100 nodes -> a 10x10 lattice, spacing 6.4/9, radius 0.35 * spacing.
    const dense = new SoftMesh({ count: 100, seed: 7 });
    expect(dense.radius(0)).toBeCloseTo((6.4 / 9) * RADIUS_FRACTION, 6);
    expect(dense.maxRadius()).toBe(dense.radius(0));
  });

  it('cap a sparse scene by the box, or its ends start outside it', () => {
    // A three-node rope across a 16-unit box has spacing 5.6, and 0.35 * 5.6 = 1.96
    // would put the endpoints past the wall they were built inside. The cap is
    // 0.05 * 16 = 0.8, which is what keeps outOfBounds() zero on frame one.
    const sparse = new SoftMesh({ count: 3, seed: 2, scene: 'rope' });
    expect(sparse.radius(0)).toBeCloseTo(0.8, 6);
    expect(sparse.outOfBounds()).toBe(0);
  });

  it('honour an explicit radius, and reject a non-positive one', () => {
    expect(new SoftMesh({ count: 16, seed: 1, radius: 0.1 }).maxRadius()).toBeCloseTo(0.1, 6);
    expect(() => new SoftMesh({ count: 16, radius: 0 })).toThrow(/radius must be positive/);
    expect(() => new SoftMesh({ count: 16, radius: -1 })).toThrow(/radius must be positive/);
  });

  it('never exceed half the smallest box axis, at any count', () => {
    // The invariant the bounds kernel relies on: a node plus its radius still
    // leaves room to reflect, so the reflection cannot land outside the box.
    for (const scene of SOFT_SCENES) {
      for (const count of [1, 3, 17, 250, 1000]) {
        const mesh = new SoftMesh({ count, scene, seed: 5 });
        expect(mesh.maxRadius(), `${scene} @ ${count}`).toBeLessThan(8);
      }
    }
  });
});

describe('construction validation', () => {
  it.each([0, -1, 1.5, Number.NaN])('rejects count %s', (count) => {
    expect(() => new SoftMesh({ count })).toThrow(/count must be a positive integer/);
  });

  it('rejects an unknown scene, naming the ones it knows', () => {
    expect(() => new SoftMesh({ count: 4, scene: 'nope' as SoftScene })).toThrow(
      /unknown scene "nope", expected one of cloth, sheets, cube, rope/,
    );
    expect(SOFT_SCENES).toEqual(['cloth', 'sheets', 'cube', 'rope']);
  });

  it('rejects a degenerate or non-finite box, per axis', () => {
    expect(() => new SoftMesh({ count: 4, bounds: { min: [0, 0, 0], max: [0, 1, 1] } })).toThrow(
      /bounds axis 0 must be finite with max > min/,
    );
    expect(() => new SoftMesh({ count: 4, bounds: { min: [0, 0, 0], max: [1, 0, 1] } })).toThrow(
      /bounds axis 1 must be finite with max > min, got \[0, 0\]/,
    );
    expect(
      () => new SoftMesh({ count: 4, bounds: { min: [0, 0, 0], max: [1, 1, Number.NaN] } }),
    ).toThrow(/bounds axis 2/);
  });

  it('rejects a zero or negative invMass, because pinning is pin()', () => {
    // `invMass: 0` would pin every node with no way to say which ones, and a
    // negative one would make the solver push two nodes apart by pulling.
    expect(() => new SoftMesh({ count: 4, invMass: 0 })).toThrow(/invMass must be finite and > 0/);
    expect(() => new SoftMesh({ count: 4, invMass: -1 })).toThrow(/invMass must be finite and > 0/);
    // Node 2, not node 1: a 4-node cloth is 2x2 and row 0 is pinned.
    expect(new SoftMesh({ count: 4, invMass: 0.5, seed: 1 }).invMass(2)).toBe(0.5);
  });

  it('rejects a negative speed', () => {
    expect(() => new SoftMesh({ count: 4, speed: -1 })).toThrow(/speed must be finite and >= 0/);
    expect(() => new SoftMesh({ count: 4, speed: Number.NaN })).toThrow(/speed must be finite/);
  });

  it.each([0.5, 1, -0.1, Number.NaN])('rejects jitter %s outside [0, 0.5)', (jitter) => {
    // At 0.5 a node can land exactly on its neighbour, which makes that edge's
    // rest length zero and divides by it in the relative-error measure.
    expect(() => new SoftMesh({ count: 9, jitter })).toThrow(
      /jitter must be finite and within \[0, 0.5\)/,
    );
  });

  it('rejects a data buffer that does not hold count records', () => {
    expect(() => new SoftMesh(new Float32Array(7), { count: 1 })).toThrow(
      /data holds 7 floats, but 1 nodes need 8/,
    );
  });

  it('per-scene default jitter is small, and zero for a rope', () => {
    expect(DEFAULT_JITTER).toEqual({ cloth: 0.02, sheets: 0.02, cube: 0.02, rope: 0 });
    // A rope is a visual: jittering it makes the chain look broken rather than loose.
    const rope = new SoftMesh({ count: 8, seed: 2, scene: 'rope' });
    const bare = new SoftMesh({ count: 8, seed: 2, scene: 'rope', jitter: 0 });
    expect(rope.digest()).toBe(bare.digest());
  });
});

describe('graph validation', () => {
  const data = new Float32Array(4 * SOFT_STRIDE);
  const one = (ends: number[], rest: number[], stiffness: number[]) => ({
    count: ends.length / 2,
    ends: new Uint32Array(ends),
    rest: new Float32Array(rest),
    stiffness: new Float32Array(stiffness),
  });

  it('accepts an empty graph: a cloud of free nodes still simulates', () => {
    const empty = emptyConstraints();
    expect(empty.count).toBe(0);
    expect(() => assertConstraints(empty, 4)).not.toThrow();
    expect(() => new SoftMesh(new Float32Array(32), { count: 4 })).not.toThrow();
  });

  it('rejects a negative or non-integer constraint count', () => {
    expect(() =>
      assertConstraints(
        { count: -1, ends: new Uint32Array(0), rest: new Float32Array(0), stiffness: new Float32Array(0) },
        4,
      ),
    ).toThrow(/constraint count must be a non-negative integer/);
  });

  it('rejects mismatched parallel arrays', () => {
    // Count is declared, not derived, so each array can disagree with it on its own.
    const declared = (count: number, ends: number[], rest: number[], stiffness: number[]) =>
      new SoftMesh(data.slice(), {
        count: 4,
        constraints: {
          count,
          ends: new Uint32Array(ends),
          rest: new Float32Array(rest),
          stiffness: new Float32Array(stiffness),
        },
      });
    expect(() => declared(2, [0, 1], [1, 1], [1, 1])).toThrow(
      /ends holds 2 indices, but 2 constraints need 4/,
    );
    expect(() => declared(1, [0, 1], [], [])).toThrow(
      /rest \(0\) and stiffness \(0\) must each hold 1 entries/,
    );
    expect(() => assertConstraints(one([0, 1], [1], [1]), 4)).not.toThrow();
  });

  it('rejects an endpoint the mesh does not have, and a self-loop', () => {
    expect(() => new SoftMesh(data.slice(), { count: 4, constraints: one([0, 5], [1], [1]) })).toThrow(
      /constraint 0 references node 5, but the mesh has 4/,
    );
    expect(() => new SoftMesh(data.slice(), { count: 4, constraints: one([1, 1], [1], [1]) })).toThrow(
      /constraint 0 connects node 1 to itself/,
    );
  });

  it('rejects a zero rest length and a stiffness outside [0, 1]', () => {
    // A zero rest length divides by zero in the relative-error measure; a
    // stiffness above 1 overshoots every iteration and never settles, which
    // looks like a solver bug and is an argument bug.
    expect(() => new SoftMesh(data.slice(), { count: 4, constraints: one([0, 1], [0], [1]) })).toThrow(
      /constraint 0 has rest length 0, which must be > 0/,
    );
    expect(() => new SoftMesh(data.slice(), { count: 4, constraints: one([0, 1], [1], [1.5]) })).toThrow(
      /constraint 0 has stiffness 1\.5, outside \[0, 1\]/,
    );
    expect(() => new SoftMesh(data.slice(), { count: 4, constraints: one([0, 1], [1], [-0.5]) })).toThrow(
      /outside \[0, 1\]/,
    );
    expect(() => assertConstraints(one([0, 1], [1], [0]), 4)).not.toThrow();
  });

  it('rejects a triangle list that is not a multiple of three, or points outside', () => {
    expect(() => assertTriangles(new Uint32Array([0, 1, 2, 3]), 4)).toThrow(
      /triangles holds 4 indices, which is not a multiple of 3/,
    );
    expect(() => assertTriangles(new Uint32Array([0, 1, 9]), 8)).toThrow(
      /triangle index 2 references node 9, but the mesh has 8/,
    );
    expect(() => assertTriangles(new Uint32Array(0), 4)).not.toThrow();
  });
});

describe('the cloth scene', () => {
  it('builds a 10x10 lattice with shear diagonals at 100 nodes', () => {
    const mesh = new SoftMesh({ count: 100, seed: 7 });
    // 180 structural (90 right + 90 down) and 162 shear (2 per 9x9 cell).
    expect(mesh.constraints.count).toBe(342);
    expect(mesh.triangles.length / 3).toBe(162);
    const rates = mesh.constraints.stiffness;
    let structural = 0;
    let shear = 0;
    for (const r of rates) {
      // Every rate is one of exactly two f32 values, and neither is the f64
      // literal: the shader compares against these bytes.
      expect(r === Math.fround(1) || r === Math.fround(0.7)).toBe(true);
      if (r === SOFT_STIFFNESS.structural) structural++;
      else shear++;
    }
    expect(structural).toBe(180);
    expect(shear).toBe(162);
    expect(SOFT_STIFFNESS).toEqual({ structural: 1, shear: 0.7, bend: 0.3 });
    expect(rates.length).toBe(342);
    // The first cell's edges come out as right, down, then its two diagonals.
    expect(rates[0]).toBe(Math.fround(1));
    expect(rates[2]).toBe(Math.fround(0.7));
    expect(rates[3]).toBe(Math.fround(0.7));
  });

  it('pins the top row, so the cloth hangs instead of folding over itself', () => {
    const mesh = new SoftMesh({ count: 100, seed: 7 });
    expect(mesh.pinnedCount()).toBe(10);
    for (let i = 0; i < 100; i++) expect(mesh.isPinned(i)).toBe(i < 10);
  });

  it('winds two triangles per cell, consistently', () => {
    const mesh = new SoftMesh({ count: 100, seed: 7 });
    expect(Array.from(mesh.triangles.subarray(0, 6))).toEqual([0, 10, 1, 1, 10, 11]);
    // Every triangle references three distinct nodes and stays in range.
    for (let t = 0; t < mesh.triangles.length; t += 3) {
      const [a, b, c] = [mesh.triangles[t], mesh.triangles[t + 1], mesh.triangles[t + 2]];
      expect(new Set([a, b, c]).size).toBe(3);
      expect(Math.max(a, b, c)).toBeLessThan(100);
    }
  });

  it('measures rest lengths off the jittered nodes, so nothing starts pre-stressed', () => {
    const mesh = new SoftMesh({ count: 100, seed: 7 });
    const [min, max] = restRange(mesh);
    // spacing 6.4/9 = 0.7111, diagonal 1.0057, each moved by up to jitter*spacing.
    expect(min).toBeGreaterThan(0.68);
    expect(min).toBeLessThan(0.69);
    expect(max).toBeGreaterThan(1.03);
    expect(max).toBeLessThan(1.04);
    // The point of measuring rather than declaring: the built mesh already
    // satisfies its own graph to within f32 rounding.
    expect(mesh.maxConstraintError()).toBeLessThan(1e-6);
  });

  it('keeps node degree between 3 and 8', () => {
    const d = degrees(new SoftMesh({ count: 100, seed: 7 }));
    expect(Math.max(...d)).toBe(8);
    expect(Math.min(...d)).toBe(3);
  });

  it('adds bend edges only when asked, because they cost 2n for visible drape', () => {
    expect(new SoftMesh({ count: 100, seed: 7 }).constraints.count).toBe(342);
    expect(new SoftMesh({ count: 100, seed: 7, bend: true }).constraints.count).toBe(502);
    const bend = new SoftMesh({ count: 100, seed: 7, bend: true, shear: false });
    // 180 structural + 160 bend (80 horizontal + 80 vertical two-cell skips).
    expect(bend.constraints.count).toBe(340);
  });

  it('drops the diagonals and the surface with shear: false', () => {
    const mesh = new SoftMesh({ count: 100, seed: 7, shear: false });
    expect(mesh.constraints.count).toBe(180);
    expect(mesh.triangles.length).toBe(0);
  });

  it('handles counts that are not a perfect square, without building extra nodes', () => {
    expect(new SoftMesh({ count: 11, seed: 7 }).constraints.count).toBe(25);
    expect(new SoftMesh({ count: 11, seed: 7 }).triangles.length / 3).toBe(10);
    expect(new SoftMesh({ count: 11, seed: 7 }).pinnedCount()).toBe(4);
    expect(new SoftMesh({ count: 3, seed: 7 }).constraints.count).toBe(2);
    expect(Array.from(new SoftMesh({ count: 3, seed: 7 }).constraints.ends)).toEqual([0, 1, 0, 2]);
    expect(new SoftMesh({ count: 3, seed: 7 }).triangles.length).toBe(0);
    // One node: no edges, no surface, and the single node is the pinned row.
    const one = new SoftMesh({ count: 1, seed: 7 });
    expect(one.count).toBe(1);
    expect(one.constraints.count).toBe(0);
    expect(one.triangles.length).toBe(0);
    expect(one.pinnedCount()).toBe(1);
  });
});

describe('the sheets scene', () => {
  it('splits nodes into disconnected cloths spread along z', () => {
    // The multi-island scene. A single connected cloth is one island, and an
    // island grouper that only ever saw one island would prove nothing.
    const mesh = new SoftMesh({ count: 100, seed: 1, scene: 'sheets', groups: 3, jitter: 0 });
    const zs = new Map<number, number>();
    for (let i = 0; i < 100; i++) {
      const z = mesh.position(i)[2];
      zs.set(z, (zs.get(z) ?? 0) + 1);
    }
    // Uneven on purpose: 34/33/33 means the last sheet is a different size,
    // which is the case a padded workgroup mapping has to get right.
    expect(zs.size).toBe(3);
    expect([...zs.values()].sort((a, b) => b - a)).toEqual([34, 33, 33]);
    expect([...zs.keys()].sort((a, b) => a - b).map((z) => Number(z.toFixed(3)))).toEqual([
      -3.2, 0, 3.2,
    ]);
    expect(mesh.constraints.count).toBe(298);
    expect(mesh.triangles.length / 3).toBe(134);
    expect(mesh.pinnedCount()).toBe(18);
    // No edge may cross sheets, or the islands are not islands.
    const zOf = (i: number) => mesh.position(i)[2];
    for (let k = 0; k < mesh.constraints.count; k++) {
      expect(zOf(mesh.constraints.ends[k * 2])).toBe(zOf(mesh.constraints.ends[k * 2 + 1]));
    }
  });

  it('with one group is byte-identical to cloth', () => {
    const sheets = new SoftMesh({ count: 100, seed: 1, scene: 'sheets', groups: 1 });
    const cloth = new SoftMesh({ count: 100, seed: 1, scene: 'cloth' });
    expect(sheets.digest()).toBe(cloth.digest());
    expect(sheets.digest()).toBe('53cfbeca18b8ff22:800');
  });

  it('changes the graph with the group count', () => {
    const seven = new SoftMesh({ count: 100, seed: 1, scene: 'sheets', groups: 7 });
    expect(seven.constraints.count).toBe(246);
    expect(seven.triangles.length / 3).toBe(102);
    expect(seven.pinnedCount()).toBe(28);
  });

  it('degrades to single-node islands without inventing edges', () => {
    // 10 nodes in 4 sheets is 3/3/2/2; 7 nodes in 7 sheets is seven singletons,
    // which is the island pass's other degenerate case: every island size 1.
    const ten = new SoftMesh({ count: 10, seed: 1, scene: 'sheets', groups: 4 });
    expect(ten.constraints.count).toBe(6);
    expect(ten.triangles.length).toBe(0);
    expect(ten.pinnedCount()).toBe(8);
    const seven = new SoftMesh({ count: 7, seed: 1, scene: 'sheets', groups: 7 });
    expect(seven.constraints.count).toBe(0);
    expect(seven.pinnedCount()).toBe(7);
  });

  it('clamps the group count to the node count, and rejects a non-positive one', () => {
    const clamped = new SoftMesh({ count: 5, seed: 1, scene: 'sheets', groups: 50 });
    expect(clamped.constraints.count).toBe(0);
    expect(clamped.pinnedCount()).toBe(5);
    expect(() => new SoftMesh({ count: 8, scene: 'sheets', groups: 0 })).toThrow(
      /groups must be a positive integer, got 0/,
    );
    expect(() => new SoftMesh({ count: 8, scene: 'sheets', groups: 2.5 })).toThrow(
      /groups must be a positive integer/,
    );
  });
});

describe('the cube scene', () => {
  it('connects only along the axes, which keeps the degree low', () => {
    const mesh = new SoftMesh({ count: 1000, seed: 3, scene: 'cube' });
    // A 10x10x10 lattice: 3 * 900 axis edges.
    expect(mesh.constraints.count).toBe(2700);
    expect(mesh.triangles.length).toBe(0);
    // Maximum degree 6, so a greedy first-fit needs at most 7 colours. This is
    // what makes the cube the colouring pass's headline case.
    const d = degrees(mesh);
    expect(Math.max(...d)).toBe(6);
    expect(Math.min(...d)).toBe(3);
    const deltas = new Set<number>();
    for (let k = 0; k < mesh.constraints.count; k++) {
      deltas.add(mesh.constraints.ends[k * 2 + 1] - mesh.constraints.ends[k * 2]);
    }
    expect([...deltas].sort((a, b) => a - b)).toEqual([1, 10, 100]);
    expect(new Set(mesh.constraints.stiffness)).toEqual(new Set([SOFT_STIFFNESS.structural]));
  });

  it('pins the top y-layer, so it hangs and wobbles instead of falling', () => {
    const mesh = new SoftMesh({ count: 1000, seed: 3, scene: 'cube' });
    expect(mesh.pinnedCount()).toBe(100);
    for (let i = 0; i < 1000; i++) {
      expect(mesh.isPinned(i), `node ${i}`).toBe(Math.floor(i / 10) % 10 === 9);
    }
  });

  it('never connects across a partial last z-layer', () => {
    // 950 is not a full 10x10x10, so the last layer is short. Guarding only on
    // `i + s < nodes` would join the top of one layer to the bottom of the next.
    const mesh = new SoftMesh({ count: 950, seed: 3, scene: 'cube' });
    expect(mesh.count).toBe(950);
    expect(mesh.constraints.count).toBe(2555);
    expect(mesh.pinnedCount()).toBe(90);
    expect(mesh.outOfBounds()).toBe(0);
    for (let k = 0; k < mesh.constraints.count; k++) {
      expect(mesh.constraints.ends[k * 2 + 1]).toBeLessThan(950);
    }
  });
});

describe('the rope scene', () => {
  it('builds a chain pinned at both ends', () => {
    const mesh = new SoftMesh({ count: 50, seed: 2, scene: 'rope' });
    expect(mesh.constraints.count).toBe(49);
    expect(mesh.triangles.length).toBe(0);
    expect(mesh.pinnedCount()).toBe(2);
    expect(mesh.isPinned(0)).toBe(true);
    expect(mesh.isPinned(49)).toBe(true);
    expect(mesh.isPinned(25)).toBe(false);
    for (let k = 0; k < 49; k++) {
      expect(mesh.constraints.ends[k * 2]).toBe(k);
      expect(mesh.constraints.ends[k * 2 + 1]).toBe(k + 1);
    }
  });

  it('runs along x above centre, so there is somewhere to sag into', () => {
    // Both ends rather than one: a rope pinned only at node 0 hangs straight
    // down and every edge ends at the same rest length, so a solver that got the
    // mass weighting wrong would still look right. Two ends produce a catenary,
    // whose shape *is* the weighting.
    const mesh = new SoftMesh({ count: 5, seed: 2, scene: 'rope', jitter: 0 });
    expect(mesh.position(0)[1]).toBeCloseTo(3.2, 5);
    expect(mesh.position(0)[2]).toBeCloseTo(0, 5);
    for (let i = 0; i < 5; i++) {
      expect(mesh.position(i)[0]).toBeCloseTo(-5.6 + i * 2.8, 5);
      expect(mesh.position(i)[1]).toBeCloseTo(3.2, 5);
    }
    const longer = new SoftMesh({ count: 50, seed: 2, scene: 'rope' });
    for (let i = 1; i < 50; i++) {
      expect(longer.position(i)[0]).toBeGreaterThan(longer.position(i - 1)[0]);
      expect(longer.position(i)[1]).toBe(longer.position(0)[1]);
    }
  });
});

describe('determinism and the RNG consumption order', () => {
  it('reproduces its bytes from the seed alone', () => {
    expect(new SoftMesh({ count: 64, seed: 1 }).digest()).toBe(
      new SoftMesh({ count: 64, seed: 1 }).digest(),
    );
    expect(new SoftMesh({ count: 64, seed: 1 }).digest()).not.toBe(
      new SoftMesh({ count: 64, seed: 2 }).digest(),
    );
  });

  it('defaults the seed, so omitting it is still reproducible', () => {
    // Rng's default is 0x2f6e2b1; a mesh built without a seed has to match the
    // one built with it, or "same options, same bytes" has an exception in it.
    expect(new SoftMesh({ count: 64 }).digest()).toBe(new SoftMesh({ count: 64 }).digest());
    expect(new SoftMesh({ count: 64 }).digest()).toBe(new SoftMesh({ count: 64, seed: 0x2f6e2b1 }).digest());
  });

  it('pins the golden digests of the two headline scenes', () => {
    // Recorded rather than computed: these are the bytes a replay restores and
    // the values the GPU parity specs compare against.
    expect(new SoftMesh({ count: 100, seed: 7 }).digest()).toBe('90e23f10e3d89324:800');
    expect(new SoftMesh({ count: 1000, seed: 7 }).digest()).toBe('e7e196de32d8c112:8000');
    expect(new SoftMesh({ count: 1000, seed: 7, scene: 'cube' }).digest()).toMatch(
      /^[0-9a-f]{16}:8000$/,
    );
  });

  it('counts the digest suffix in floats, not bytes', () => {
    expect(new SoftMesh({ count: 4, seed: 1 }).digest()).toMatch(/:32$/);
    expect(new SoftMesh({ count: 100, seed: 1 }).digest()).toMatch(/:800$/);
  });

  it('draws jitter before velocity, per node ascending', () => {
    // The documented order. With jitter 0 the RNG is only consumed for velocity,
    // so positions must be identical across two speeds -- and they must differ
    // once jitter is on, because then the velocity draws land elsewhere.
    const still = new SoftMesh({ count: 16, seed: 5, jitter: 0, speed: 0 });
    const moving = new SoftMesh({ count: 16, seed: 5, jitter: 0, speed: 4 });
    for (let i = 0; i < 16; i++) {
      expect(still.position(i)).toEqual(moving.position(i));
    }
    expect(still.maxSpeed()).toBe(0);
    expect(moving.maxSpeed()).toBeCloseTo(6.3516819, 5);
    expect(moving.maxSpeed()).toBeLessThanOrEqual(4 * Math.sqrt(3));
    const jittered = new SoftMesh({ count: 16, seed: 5, speed: 4 });
    expect(jittered.digest()).not.toBe(moving.digest());
  });

  it('bounds the jitter by jitter * spacing', () => {
    const noisy = new SoftMesh({ count: 25, seed: 4, jitter: 0.1 });
    const lattice = new SoftMesh({ count: 25, seed: 4, jitter: 0 });
    const spacing = defaultExtent(DEFAULT_BOUNDS) / (Math.ceil(Math.sqrt(25)) - 1);
    let worst = 0;
    for (let i = 0; i < 25; i++) {
      const p = noisy.position(i);
      const q = lattice.position(i);
      for (let axis = 0; axis < 3; axis++) worst = Math.max(worst, Math.abs(p[axis] - q[axis]));
    }
    expect(worst).toBeGreaterThan(0);
    expect(worst).toBeLessThanOrEqual(0.1 * spacing);
    // A perfect lattice already satisfies its own graph; the jitter is then
    // measured into the rest lengths rather than left as pre-stress.
    expect(lattice.maxConstraintError()).toBeLessThan(1e-6);
    expect(noisy.maxConstraintError()).toBeLessThan(1e-6);
  });

  it('builds the same scene for the same seed on every tier, twice over', () => {
    // What a replay actually does: construct, save the digest, construct again.
    for (const scene of SOFT_SCENES) {
      const a = new SoftMesh({ count: 128, seed: 11, scene, speed: 2 });
      const b = new SoftMesh({ count: 128, seed: 11, scene, speed: 2 });
      expect(a.digest(), scene).toBe(b.digest());
      expect(a.constraints.count, scene).toBe(b.constraints.count);
      expect(Array.from(a.triangles), scene).toEqual(Array.from(b.triangles));
    }
  });
});

describe('a fresh scene is inside its own bounds', () => {
  it.each(BOXES)('holds for every scene and count in %s', (_name, bounds) => {
    for (const scene of SOFT_SCENES) {
      for (const count of [1, 2, 3, 7, 17, 64, 101, 1000]) {
        const mesh = new SoftMesh({ count, scene, seed: 3, bounds, speed: 3 });
        // The property that makes outOfBounds() meaningful on frame zero: if a
        // builder ever starts a node outside the box, the reflect kernel "fixes"
        // it on step one and the bug never shows up in the picture.
        expect(mesh.outOfBounds(), `${scene} @ ${count}`).toBe(0);
        expect(mesh.maxConstraintError(), `${scene} @ ${count}`).toBeLessThan(1e-5);
        expect(mesh.maxSpeed()).toBeLessThanOrEqual(3 * Math.sqrt(3));
      }
    }
  });

  it('counts an escape rather than trusting the picture', () => {
    const mesh = new SoftMesh({ count: 4, seed: 1, jitter: 0 });
    expect(mesh.outOfBounds()).toBe(0);
    mesh.setPosition(0, [100, 0, 0]);
    mesh.setPosition(1, [0, -100, 0]);
    expect(mesh.outOfBounds()).toBe(2);
    mesh.setPosition(0, [8, 8, 8]);
    expect(mesh.outOfBounds()).toBe(1);
  });
});

describe('accessors, pin and unpin', () => {
  it('range-checks the node index', () => {
    const mesh = new SoftMesh({ count: 9, seed: 1 });
    expect(() => mesh.position(9)).toThrow(/node index 9 out of range \[0, 9\)/);
    expect(() => mesh.position(-1)).toThrow(/out of range/);
    expect(() => mesh.velocity(1.5)).toThrow(/out of range/);
    expect(() => mesh.pin(9)).toThrow(/out of range/);
    expect(() => mesh.position(8)).not.toThrow();
  });

  it('pins by setting invMass to zero, which needs no branch in the solver', () => {
    const mesh = new SoftMesh({ count: 9, seed: 1 });
    expect(mesh.pinnedCount()).toBe(3);
    expect(mesh.invMass(0)).toBe(0);
    expect(mesh.isPinned(3)).toBe(false);
    mesh.unpin(0, 2);
    expect(mesh.invMass(0)).toBe(2);
    expect(mesh.pinnedCount()).toBe(2);
    mesh.pin(3);
    expect(mesh.invMass(3)).toBe(0);
    expect(mesh.pinnedCount()).toBe(3);
    // The mass is not recoverable from `invMass === 0`, so unpin takes one and
    // defaults to what every scene builder writes.
    mesh.unpin(3);
    expect(mesh.invMass(3)).toBe(1);
    expect(() => mesh.unpin(3, 0)).toThrow(/unpin needs invMass > 0, got 0/);
  });

  it('validates the setters', () => {
    const mesh = new SoftMesh({ count: 4, seed: 1 });
    expect(() => mesh.setInvMass(1, -1)).toThrow(/invMass must be finite and >= 0, got -1/);
    expect(() => mesh.setInvMass(1, Number.NaN)).toThrow(/invMass must be finite/);
    expect(() => mesh.setRadius(1, 0)).toThrow(/radius must be positive, got 0/);
    mesh.setInvMass(1, 0);
    expect(mesh.isPinned(1)).toBe(true);
    // Radii differ per node so maxRadius() has something to choose between; a
    // scene build writes one value everywhere and would not exercise the scan.
    for (let i = 0; i < mesh.count; i++) mesh.setRadius(i, 0.1 * (i + 1));
    expect(mesh.radius(1)).toBeCloseTo(0.2, 6);
    expect(mesh.maxRadius()).toBeCloseTo(0.4, 6);
  });
});

describe('the statistics', () => {
  /**
   * A four-node mesh with hand-picked velocities, so every statistic below has
   * an arithmetic answer that does not depend on a scene builder.
   */
  function built(): SoftMesh {
    const data = new Float32Array(4 * SOFT_STRIDE);
    const invMass = [1, 1, 1, 0];
    const vel = [[1, 2, 2], [3, 0, 4], [0, 0, 0], [9, 9, 9]];
    for (let i = 0; i < 4; i++) {
      data[i * SOFT_STRIDE] = i;
      data[i * SOFT_STRIDE + SOFT_OFFSET.invMass] = invMass[i];
      for (let axis = 0; axis < 3; axis++) {
        data[i * SOFT_STRIDE + SOFT_OFFSET.velocity + axis] = vel[i][axis];
      }
      data[i * SOFT_STRIDE + SOFT_OFFSET.radius] = 0.5;
    }
    return new SoftMesh(data, {
      count: 4,
      constraints: {
        count: 2,
        ends: new Uint32Array([0, 1, 1, 2]),
        rest: new Float32Array([1, 1]),
        stiffness: new Float32Array([1, 0.7]),
      },
    });
  }

  it('sum kinetic energy over unpinned nodes only', () => {
    const mesh = built();
    // 0.5*(1/1)*(1+4+4) + 0.5*(1/1)*(9+0+16) + 0 = 4.5 + 12.5. The pinned node
    // carries the fastest velocity in the mesh and contributes nothing, which is
    // the whole point of testing it against maxSpeed below.
    expect(mesh.kineticEnergy()).toBeCloseTo(17, 6);
    expect(mesh.maxSpeed()).toBeCloseTo(Math.sqrt(243), 5);
    expect(mesh.pinnedCount()).toBe(1);
  });

  it('report maxSpeed as the largest magnitude, pinned or not', () => {
    const mesh = built();
    mesh.setVelocity(3, [0, 0, 0]);
    expect(mesh.maxSpeed()).toBeCloseTo(5, 6);
    mesh.setVelocity(0, [0, 0, 0]);
    mesh.setVelocity(1, [0, 0, 0]);
    expect(mesh.maxSpeed()).toBe(0);
  });

  it('measure constraint error as a relative length deviation', () => {
    const mesh = built();
    // Nodes sit at x = 0, 1, 2, 3 on a line, so both edges are exactly 1 long
    // against a rest length of 1: the error is zero before anything is solved.
    expect(mesh.maxConstraintError()).toBe(0);
    mesh.setPosition(1, [1.5, 0, 0]);
    // Edge 0 is now 1.5 long against rest 1 -> 0.5; edge 1 is 0.5 -> 0.5.
    expect(mesh.maxConstraintError()).toBeCloseTo(0.5, 6);
    mesh.setPosition(3, [3, 0, 4]);
    expect(mesh.maxConstraintError()).toBeCloseTo(0.5, 6);
  });

  it('skip a zero rest length instead of dividing by it', () => {
    const mesh = built();
    mesh.constraints.rest[0] = 0;
    expect(() => mesh.maxConstraintError()).not.toThrow();
    expect(Number.isFinite(mesh.maxConstraintError())).toBe(true);
  });

  it('are zero on an all-zero mesh, where every node counts as pinned', () => {
    const mesh = SoftMesh.empty(4);
    expect(mesh.data.every((x) => x === 0)).toBe(true);
    expect(mesh.constraints.count).toBe(0);
    expect(mesh.triangles.length).toBe(0);
    expect(mesh.maxRadius()).toBe(0);
    expect(mesh.maxSpeed()).toBe(0);
    expect(mesh.kineticEnergy()).toBe(0);
    expect(mesh.maxConstraintError()).toBe(0);
    // All-zero means invMass 0 everywhere, i.e. a mesh nobody can move. Pinned
    // count is therefore 4, not 0 -- a fact a sleeping-island test keys on.
    expect(mesh.pinnedCount()).toBe(4);
    expect(mesh.outOfBounds()).toBe(0);
    expect(mesh.digest()).toMatch(/^[0-9a-f]{16}:32$/);
    expect(mesh.bounds).toEqual(DEFAULT_BOUNDS);
  });
});

describe('clone and copyFrom', () => {
  it('clone copies the bytes and shares the graph', () => {
    const mesh = new SoftMesh({ count: 64, seed: 9 });
    const copy = mesh.clone();
    expect(copy.digest()).toBe(mesh.digest());
    expect(copy.count).toBe(mesh.count);
    expect(copy.data).not.toBe(mesh.data);
    // The graph is immutable and large, so sharing it is safe and the alternative
    // is a 40k-edge copy per clone at 10k nodes.
    expect(copy.constraints).toBe(mesh.constraints);
    expect(copy.triangles).toBe(mesh.triangles);
    expect(copy.bounds).toBe(mesh.bounds);
    copy.setPosition(0, [7, 7, 7]);
    expect(mesh.position(0)).not.toEqual([7, 7, 7]);
  });

  it('copyFrom restores node bytes without touching the graph', () => {
    const mesh = new SoftMesh({ count: 4, seed: 1 });
    const saved = mesh.clone();
    mesh.setPosition(0, [5, 5, 5]);
    expect(mesh.digest()).not.toBe(saved.digest());
    mesh.copyFrom(saved);
    expect(mesh.digest()).toBe(saved.digest());
  });

  it('refuses to copy a different node count', () => {
    // Two meshes with the same count can still have different edges, and a solver
    // holding a plan built from this graph would then run the wrong one. Restoring
    // state is a node-bytes operation, so the count is the only thing to check.
    const small = new SoftMesh({ count: 4, seed: 1 });
    const big = SoftMesh.empty(64);
    expect(() => big.copyFrom(small)).toThrow(/cannot copy 4 nodes into 64/);
  });

  it('empty takes a bounds argument', () => {
    const bounds: Bounds = { min: [0, 0, 0], max: [2, 4, 6] };
    const mesh = SoftMesh.empty(8, bounds);
    expect(mesh.count).toBe(8);
    expect(mesh.bounds).toEqual(bounds);
    expect(mesh.data.length).toBe(8 * SOFT_STRIDE);
  });
});

describe('sizeForScene, and the 10k-node scale the plan asks for', () => {
  it('predicts a scene shape without keeping the mesh', () => {
    // A demo with a 10k-node picker has to size its buffers before it builds, and
    // a budget that has to build a throwaway cloth to learn its edge count is a
    // budget that runs twice.
    expect(sizeForScene('cloth', 10_000, { seed: 1 })).toEqual({
      nodes: 10_000,
      constraints: 39_402,
      triangles: 19_602,
    });
    expect(sizeForScene('sheets', 10_000, { seed: 1, groups: 8 })).toEqual({
      nodes: 10_000,
      constraints: 38_312,
      triangles: 18_880,
    });
    expect(sizeForScene('cube', 10_000, { seed: 1 })).toEqual({
      nodes: 10_000,
      constraints: 28_599,
      triangles: 0,
    });
    expect(sizeForScene('rope', 10_000, { seed: 1 })).toEqual({
      nodes: 10_000,
      constraints: 9_999,
      triangles: 0,
    });
  });

  it('agrees with the mesh it predicts, for every scene', () => {
    for (const scene of SOFT_SCENES) {
      const predicted = sizeForScene(scene, 512, { seed: 6, groups: 4 });
      const mesh = new SoftMesh({ count: 512, seed: 6, scene, groups: 4 });
      expect(predicted.nodes, scene).toBe(mesh.count);
      expect(predicted.constraints, scene).toBe(mesh.constraints.count);
      expect(predicted.triangles, scene).toBe(mesh.triangles.length / 3);
    }
  });

  it('builds a 10k-node cloth that is whole, in bounds and unstressed', () => {
    const mesh = new SoftMesh({ count: 10_000, seed: 1 });
    expect(mesh.count).toBe(10_000);
    expect(mesh.data.length).toBe(80_000);
    expect(mesh.data.byteLength).toBe(320_000);
    expect(mesh.constraints.count).toBe(39_402);
    expect(mesh.triangles.length / 3).toBe(19_602);
    expect(mesh.pinnedCount()).toBe(100);
    expect(mesh.outOfBounds()).toBe(0);
    expect(mesh.maxConstraintError()).toBeLessThan(1e-5);
    const d = degrees(mesh);
    expect(Math.max(...d)).toBe(8);
    expect(Math.min(...d)).toBe(3);
    // 10k nodes, ~4 edges each: the plan's floor for this milestone, and the size
    // the GPU budget in `softBudget.ts` has to fit inside one binding.
    expect(mesh.constraints.count).toBeGreaterThan(mesh.count * 3);
  });
});

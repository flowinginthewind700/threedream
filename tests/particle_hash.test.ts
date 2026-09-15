/**
 * `gpu/particleHash.ts` -- the uniform grid broadphase.
 *
 * The property that matters most is not "finds neighbours" but "computes the
 * same cell index the WGSL kernel computes", because a disagreement shows up as
 * particles passing through each other at random and nothing else. So the
 * hashing arithmetic is asserted against an independent restatement of the
 * Teschner formula rather than against itself.
 */

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_BUCKET_CAPACITY,
  HASH_PRIMES,
  SpatialHash,
  hashCellCoords,
  nextPow2,
} from '../src/gpu/particleHash.js';
import { PARTICLE_STRIDE, ParticleField, type Vec3Tuple } from '../src/gpu/particleField.js';

/** A field with hand-placed centres, so neighbour expectations are exact. */
function fieldAt(positions: readonly Vec3Tuple[], radius = 0.2): ParticleField {
  const data = new Float32Array(positions.length * PARTICLE_STRIDE);
  positions.forEach(([x, y, z], i) => {
    const o = i * PARTICLE_STRIDE;
    data[o] = x;
    data[o + 1] = y;
    data[o + 2] = z;
    data[o + 3] = radius;
    data[o + 7] = 1;
  });
  return new ParticleField(data, positions.length);
}

describe('nextPow2', () => {
  it.each([
    [1, 1],
    [2, 2],
    [3, 4],
    [5, 8],
    [1024, 1024],
    [1025, 2048],
    [100_000, 131_072],
  ])('nextPow2(%i) === %i', (input, expected) => {
    expect(nextPow2(input)).toBe(expected);
  });

  it('degrades to 1 rather than looping forever on unusable input', () => {
    expect(nextPow2(0)).toBe(1);
    expect(nextPow2(-8)).toBe(1);
    expect(nextPow2(Number.NaN)).toBe(1);
    expect(nextPow2(Number.POSITIVE_INFINITY)).toBe(1);
  });
});

describe('hashCellCoords', () => {
  it('is the Teschner xor of three imul products, masked', () => {
    const mask = 0xff;
    for (const [ix, iy, iz] of [
      [0, 0, 0],
      [1, 2, 3],
      [-4, 7, -9],
      [12345, -67890, 11111111],
    ] as readonly (readonly [number, number, number])[]) {
      const expected =
        (Math.imul(ix, HASH_PRIMES.x) ^ Math.imul(iy, HASH_PRIMES.y) ^ Math.imul(iz, HASH_PRIMES.z)) &
        mask;
      expect(hashCellCoords(ix, iy, iz, mask)).toBe(expected);
      expect(hashCellCoords(ix, iy, iz, mask)).toBeGreaterThanOrEqual(0);
      expect(hashCellCoords(ix, iy, iz, mask)).toBeLessThanOrEqual(mask);
    }
  });

  it('uses primes that fit in a signed 32-bit int, as WGSL i32 needs', () => {
    for (const p of Object.values(HASH_PRIMES)) {
      expect(Number.isInteger(p)).toBe(true);
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(2 ** 31);
      expect(p % 2).toBe(1); // odd, so the low bits are not thrown away
    }
  });

  it('spreads neighbouring cells apart rather than into one bucket', () => {
    const mask = 4095;
    const seen = new Set<number>();
    for (let i = 0; i < 64; i++) seen.add(hashCellCoords(i, 0, 0, mask));
    expect(seen.size).toBeGreaterThan(60);
  });
});

describe('SpatialHash construction', () => {
  it('sizes the table to a power of two and derives the mask', () => {
    const hash = new SpatialHash(1000, [0, 0, 0], 0.5);
    expect(hash.tableSize).toBe(1024);
    expect(hash.mask).toBe(1023);
    expect(hash.counts.length).toBe(1024);
    expect(hash.slots.length).toBe(1024 * DEFAULT_BUCKET_CAPACITY);
    expect(hash.bucketCapacity).toBe(DEFAULT_BUCKET_CAPACITY);
    expect(DEFAULT_BUCKET_CAPACITY).toBe(8);
  });

  it('honours an explicit table size and bucket capacity', () => {
    const hash = new SpatialHash(10, [0, 0, 0], 1, { tableSize: 300, bucketCapacity: 3 });
    expect(hash.tableSize).toBe(512);
    expect(hash.bucketCapacity).toBe(3);
    expect(hash.slots.length).toBe(512 * 3);
  });

  it('rejects unusable arguments', () => {
    expect(() => new SpatialHash(0, [0, 0, 0], 1)).toThrow(/count must be a positive integer/);
    expect(() => new SpatialHash(1.5, [0, 0, 0], 1)).toThrow(/count must be a positive integer/);
    expect(() => new SpatialHash(4, [0, 0, 0], 0)).toThrow(/cellSize must be finite and positive/);
    expect(() => new SpatialHash(4, [0, 0, 0], Number.NaN)).toThrow(/cellSize/);
    expect(() => new SpatialHash(4, [0, 0, 0], 1, { bucketCapacity: 0 })).toThrow(
      /bucketCapacity must be a positive integer/,
    );
    expect(() => new SpatialHash(4, [0, 0, 0], 1, { bucketCapacity: 2.5 })).toThrow(/bucketCapacity/);
  });

  it('forField takes the cell edge from twice the largest radius', () => {
    const field = fieldAt(
      [
        [0, 0, 0],
        [1, 0, 0],
      ],
      0.25,
    );
    field.setRadius(1, 0.4);
    const hash = SpatialHash.forField(field);
    expect(hash.cellSize).toBeCloseTo(0.8, 6);
    expect(hash.tableSize).toBe(2);
    const explicit = SpatialHash.forField(field, { cellSize: 2 });
    expect(explicit.cellSize).toBe(2);
  });
});

describe('cell coordinates', () => {
  it('floor rather than round, and measured from the origin', () => {
    const hash = new SpatialHash(8, [-8, -8, -8], 1);
    expect(hash.cellCoords(-8, -8, -8)).toEqual([0, 0, 0]);
    expect(hash.cellCoords(-7.5, -8, -8)).toEqual([0, 0, 0]);
    expect(hash.cellCoords(-7, -8, -8)).toEqual([1, 0, 0]);
    expect(hash.cellCoords(0, 0, 0)).toEqual([8, 8, 8]);
  });

  it('works with a non-zero, non-integer origin', () => {
    const hash = new SpatialHash(8, [0.5, 0.5, 0.5], 2);
    expect(hash.cellCoords(0.5, 0.5, 0.5)).toEqual([0, 0, 0]);
    expect(hash.cellCoords(2.4, 0.5, 0.5)).toEqual([0, 0, 0]);
    expect(hash.cellCoords(2.6, 0.5, 0.5)).toEqual([1, 0, 0]);
  });

  it('cellIndex is hashCellCoords of cellCoords, and always in range', () => {
    const hash = new SpatialHash(8, [-8, -8, -8], 1, { tableSize: 256 });
    for (const p of [
      [-8, -8, -8],
      [0, 0, 0],
      [7.9, -3.2, 5.5],
    ] as Vec3Tuple[]) {
      const [ix, iy, iz] = hash.cellCoords(p[0], p[1], p[2]);
      expect(hash.cellIndex(p[0], p[1], p[2])).toBe(hashCellCoords(ix, iy, iz, hash.mask));
      expect(hash.cellIndex(p[0], p[1], p[2])).toBeLessThan(hash.tableSize);
    }
  });
});

describe('build', () => {
  it('inserts every particle exactly once when buckets are roomy', () => {
    const field = fieldAt([
      [0, 0, 0],
      [3, 0, 0],
      [0, 3, 0],
      [0, 0, 3],
    ]);
    const hash = SpatialHash.forField(field, { tableSize: 1024 });
    hash.build(field);
    const stats = hash.stats();
    expect(stats.inserted).toBe(4);
    expect(stats.overflow).toBe(0);
    expect(stats.cells).toBe(1024);
    expect(stats.usedCells).toBe(4);
    expect(stats.maxBucket).toBe(1);
    expect(stats.loadFactor).toBe(1);
    expect(hash.counts.reduce((a, b) => a + b, 0)).toBe(4);
  });

  it('puts particles in the same cell into the same bucket, in index order', () => {
    const field = fieldAt([
      [0, 0, 0],
      [0.1, 0, 0],
      [0.2, 0, 0],
    ]);
    const hash = SpatialHash.forField(field, { cellSize: 1, tableSize: 1024 });
    hash.build(field);
    const cell = hash.cellIndex(0, 0, 0);
    expect(hash.counts[cell]).toBe(3);
    expect(Array.from(hash.slots.slice(cell * hash.bucketCapacity, cell * hash.bucketCapacity + 3))).toEqual(
      [0, 1, 2],
    );
  });

  it('counts overflow instead of corrupting a neighbour bucket', () => {
    const positions: Vec3Tuple[] = [];
    for (let i = 0; i < 6; i++) positions.push([i * 0.01, 0, 0]);
    const field = fieldAt(positions);
    const hash = SpatialHash.forField(field, { cellSize: 1, tableSize: 1024, bucketCapacity: 2 });
    hash.build(field);
    expect(hash.overflow).toBe(4);
    expect(hash.stats().overflow).toBe(4);
    expect(hash.stats().inserted).toBe(2);
    const cell = hash.cellIndex(0, 0, 0);
    expect(hash.counts[cell]).toBe(2); // clamped, not wrapped
    // Nothing landed past the end of this bucket.
    expect(hash.counts[(cell + 1) & hash.mask]).toBe(0);
  });

  it('clears the previous step rather than accumulating into it', () => {
    const field = fieldAt([
      [0, 0, 0],
      [3, 0, 0],
    ]);
    const hash = SpatialHash.forField(field, { tableSize: 256 });
    hash.build(field);
    hash.build(field);
    expect(hash.counts.reduce((a, b) => a + b, 0)).toBe(2);
    expect(hash.stats().inserted).toBe(2);
  });

  it('is deterministic: the same field builds the same table', () => {
    const field = new ParticleField({ count: 512, scene: 'sphere', seed: 4, speed: 1 });
    const a = SpatialHash.forField(field, { tableSize: 2048 });
    const b = SpatialHash.forField(field, { tableSize: 2048 });
    a.build(field);
    b.build(field);
    expect(Array.from(a.counts)).toEqual(Array.from(b.counts));
    expect(Array.from(a.slots)).toEqual(Array.from(b.slots));
    expect(a.stats()).toEqual(b.stats());
  });

  it('reports zero load factor on an empty table', () => {
    const field = fieldAt([[0, 0, 0]]);
    const hash = new SpatialHash(1, [0, 0, 0], 1, { tableSize: 16 });
    // `stats()` before the first build is the initial zeroed struct.
    expect(hash.stats()).toEqual({
      cells: 0,
      usedCells: 0,
      inserted: 0,
      overflow: 0,
      maxBucket: 0,
      loadFactor: 0,
    });
    hash.build(field);
    expect(hash.stats().usedCells).toBe(1);
  });
});

describe('forEachCandidate', () => {
  it('visits the 27-cell neighbourhood and nothing farther', () => {
    // Origin is `bounds.min`, so particle 0 at [0,0,0] sits exactly on a cell
    // corner: any negative offset leaves its cell immediately. The offsets below
    // are chosen with that in mind.
    const field = fieldAt([
      [0, 0, 0], // 0: self
      [0.3, 0, 0], // 1: same cell
      [1.2, 0, 0], // 2: next cell over, +x
      [0, -0.5, 0], // 3: next cell over, -y
      [0, 0, 1.2], // 4: next cell over, +z
      [-0.5, 1.2, 1.2], // 5: diagonal neighbour
      [6, 0, 0], // 6: far away
      [0, 6, 6], // 7: far away
    ]);
    const hash = SpatialHash.forField(field, { cellSize: 1, tableSize: 65_536 });
    hash.build(field);
    expect(hash.overflow).toBe(0);
    const visited: number[] = [];
    hash.forEachCandidate(field, 0, (j) => visited.push(j));
    expect(visited).not.toContain(0); // self is skipped
    expect([...visited].sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5]);
  });

  it('never misses an overlapping pair when cellSize >= r_i + r_j', () => {
    // 150 random spheres at a density where contacts exist, checked against a
    // brute-force O(n^2) pass. This is the assertion that would catch a
    // neighbourhood that is one cell too small.
    const field = new ParticleField({
      count: 150,
      scene: 'sphere',
      seed: 17,
      radius: [0.3, 0.3],
      bounds: { min: [-2, -2, -2], max: [2, 2, 2] },
    });
    const hash = SpatialHash.forField(field, { tableSize: 8192 });
    hash.build(field);
    expect(hash.overflow).toBe(0);
    const diameter = 0.6;
    let brute = 0;
    let found = 0;
    for (let i = 0; i < field.count; i++) {
      const [ax, ay, az] = field.position(i);
      hash.forEachCandidate(field, i, (j) => {
        const [bx, by, bz] = field.position(j);
        if (Math.hypot(ax - bx, ay - by, az - bz) < diameter) found++;
      });
      for (let j = 0; j < field.count; j++) {
        if (i === j) continue;
        const [bx, by, bz] = field.position(j);
        if (Math.hypot(ax - bx, ay - by, az - bz) < diameter) brute++;
      }
    }
    expect(brute).toBeGreaterThan(0); // the scene actually has contacts
    expect(found).toBe(brute);
  });

  it('allocates no closure per particle when the visitor is reused', () => {
    // The CPU sim binds one visitor for the whole run; this pins that
    // `forEachCandidate` does not depend on closure identity.
    const field = fieldAt([
      [0, 0, 0],
      [0.1, 0, 0],
    ]);
    const hash = SpatialHash.forField(field, { cellSize: 1, tableSize: 64 });
    hash.build(field);
    let current = 0;
    const seen: string[] = [];
    const visit = (j: number): void => seen.push(`${current}:${j}`);
    for (const i of [0, 1]) {
      current = i;
      hash.forEachCandidate(field, i, visit);
    }
    expect(seen).toEqual(['0:1', '1:0']);
  });

  it('visits an aliased bucket once, not once per offset that lands in it', () => {
    // A table of two buckets cannot keep 27 distinct cells apart, so most of the
    // neighbourhood collapses onto the same bucket. Without dedupe a single
    // contact is resolved up to 27 times and the impulse is 27x too large --
    // which shows up as a pile of particles exploding rather than as a wrong
    // answer you can read off one frame.
    const field = fieldAt([
      [0, 0, 0],
      [0.1, 0, 0],
    ]);
    const hash = SpatialHash.forField(field, { cellSize: 1, tableSize: 2 });
    hash.build(field);
    expect(hash.tableSize).toBe(2);
    const visits: number[] = [];
    hash.forEachCandidate(field, 0, (j) => visits.push(j));
    expect(visits).toEqual([1]);
  });

  it('dedupe does not hide a neighbour that lives in an aliasing cell', () => {
    // Both particles sit in cells that hash into the same two buckets; visiting
    // each bucket once must still surface the other particle exactly once.
    const field = fieldAt([
      [0, 0, 0],
      [3.25, 0, 0],
    ]);
    const hash = SpatialHash.forField(field, { cellSize: 1, tableSize: 2 });
    hash.build(field);
    const visits: number[] = [];
    hash.forEachCandidate(field, 0, (j) => visits.push(j));
    expect(visits).toContain(1);
    expect(new Set(visits).size).toBe(visits.length);
  });
});

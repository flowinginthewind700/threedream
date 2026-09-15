/**
 * Uniform-grid spatial hash, CPU side.
 *
 * This is the reference implementation the WGSL broadphase in `particleWgsl.ts`
 * mirrors, and "mirrors" has to mean *computes the same cell index for the same
 * position*, not "finds roughly the same neighbours". So the hashing arithmetic
 * here is written the way WGSL can write it: `Math.imul` for the multiply
 * (u32/i32 wrap identically), integer `floor` for the cell coordinate, and a
 * power-of-two table so the modulo is a mask on both sides.
 *
 * # Why fixed-capacity buckets rather than a counting sort
 *
 * The textbook GPU broadphase is count -> prefix sum -> scatter, which needs a
 * scan kernel over the whole cell table. A scan over 2^17 cells is either a
 * single slow workgroup or a three-pass block scan, and it is the one part of the
 * pipeline that cannot be written without either subgroups or a workgroup-shared
 * memory reduction. The feasibility study measured an adapter with an *empty*
 * feature set, so neither is portable.
 *
 * Fixed buckets skip the scan entirely: each cell owns `bucketCapacity` slots,
 * insertion is one `atomicAdd` on the cell counter, and a cell that overflows
 * drops the extra particles and bumps an overflow counter instead of corrupting
 * a neighbour list. Dropping is safe because a dropped particle simply is not
 * tested against that cell this step -- it is missed contact, not wrong contact,
 * and the counter makes it visible rather than silent. The cost is
 * `tableSize * bucketCapacity` u32 of memory, which at the defaults is the same
 * size as the particle state itself.
 *
 * # Determinism
 *
 * The CPU build is deterministic: cells are visited in a fixed order and slots
 * are filled in particle-index order because there is one thread. The GPU build
 * fills slots with `atomicAdd`, so the *order within a cell* is a race. That is
 * exactly why the GPU backend reports `deterministic: false` and why the
 * collision pass is written so each particle integrates only its own share of
 * every pair -- the sum is then order-independent up to floating-point
 * association, instead of depending on who won the race.
 */

import {
  PARTICLE_OFFSET,
  PARTICLE_STRIDE,
  type ParticleField,
} from './particleField.js';

/** Slots per cell. 8 covers a uniform gas; the overflow counter says otherwise. */
export const DEFAULT_BUCKET_CAPACITY = 8;

/**
 * The three primes from the Teschner et al. spatial hash, as signed 32-bit
 * constants. They are arbitrary as far as correctness goes; what matters is that
 * they are large, odd, and the same on both sides of the TS/WGSL boundary.
 */
export const HASH_PRIMES = { x: 73856093, y: 19349663, z: 83492791 } as const;

export function nextPow2(n: number): number {
  if (!(n > 0) || !Number.isFinite(n)) return 1;
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

/**
 * Cell index for integer cell coordinates.
 *
 * Exported because the parity test needs to call it directly: a broadphase that
 * disagrees with the shader about which cell a particle is in produces missed
 * contacts, which look like "particles sometimes pass through each other" and
 * are miserable to diagnose from a rendered frame.
 */
export function hashCellCoords(ix: number, iy: number, iz: number, mask: number): number {
  return (Math.imul(ix, HASH_PRIMES.x) ^ Math.imul(iy, HASH_PRIMES.y) ^ Math.imul(iz, HASH_PRIMES.z)) & mask;
}

export interface SpatialHashOptions {
  /** Edge length of one cell. Defaults to twice the field's largest radius. */
  cellSize?: number;
  /** Bucket count, rounded up to a power of two. Defaults to `nextPow2(count)`. */
  tableSize?: number;
  bucketCapacity?: number;
}

export interface SpatialHashStats {
  readonly cells: number;
  readonly usedCells: number;
  readonly inserted: number;
  readonly overflow: number;
  readonly maxBucket: number;
  readonly loadFactor: number;
}

export class SpatialHash {
  readonly tableSize: number;
  readonly mask: number;
  readonly bucketCapacity: number;
  readonly cellSize: number;
  /** Particles per cell. `counts[c]` is clamped at `bucketCapacity`. */
  readonly counts: Int32Array;
  /** `tableSize * bucketCapacity` particle indices. */
  readonly slots: Int32Array;
  /** Insertions that found a full bucket. Non-zero means the cell size is wrong. */
  overflow = 0;

  private readonly invCell: number;
  private readonly origin: readonly [number, number, number];
  /**
   * Scratch for `forEachCandidate`: the buckets already visited this call.
   *
   * Aliasing is not a performance detail here, it is a correctness one. Two of
   * the 27 offsets can hash to the same bucket whenever the table is smaller
   * than the neighbourhood, and without this the same neighbour is resolved once
   * per alias -- fourteen impulses for one contact at a table size of two. The
   * bucket is the right thing to dedupe on (rather than the particle index)
   * because a bucket holds the union of every cell that aliased into it, so
   * visiting it once still sees all of them.
   */
  private readonly visitedCells: Int32Array = new Int32Array(27);
  private lastStats: SpatialHashStats = {
    cells: 0,
    usedCells: 0,
    inserted: 0,
    overflow: 0,
    maxBucket: 0,
    loadFactor: 0,
  };

  constructor(
    count: number,
    origin: readonly [number, number, number],
    cellSize: number,
    options: SpatialHashOptions = {},
  ) {
    if (!Number.isInteger(count) || count <= 0) {
      throw new RangeError(`count must be a positive integer, got ${count}`);
    }
    if (!(cellSize > 0) || !Number.isFinite(cellSize)) {
      throw new RangeError(`cellSize must be finite and positive, got ${cellSize}`);
    }
    const capacity = options.bucketCapacity ?? DEFAULT_BUCKET_CAPACITY;
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new RangeError(`bucketCapacity must be a positive integer, got ${capacity}`);
    }
    this.tableSize = nextPow2(options.tableSize ?? count);
    this.mask = this.tableSize - 1;
    this.bucketCapacity = capacity;
    this.cellSize = cellSize;
    this.invCell = 1 / cellSize;
    this.origin = origin;
    this.counts = new Int32Array(this.tableSize);
    this.slots = new Int32Array(this.tableSize * capacity);
  }

  /**
   * A hash sized for `field`: cell size from the largest radius so the
   * 27-neighbourhood is guaranteed to contain every overlapping pair, table size
   * from the particle count so the average bucket stays short.
   */
  static forField(field: ParticleField, options: SpatialHashOptions = {}): SpatialHash {
    const maxRadius = field.maxRadius();
    const cellSize = options.cellSize ?? Math.max(1e-4, maxRadius * 2);
    return new SpatialHash(field.count, field.bounds.min, cellSize, options);
  }

  /** Integer cell coordinates of a world position. Matches the WGSL exactly. */
  cellCoords(x: number, y: number, z: number): readonly [number, number, number] {
    return [
      Math.floor((x - this.origin[0]) * this.invCell),
      Math.floor((y - this.origin[1]) * this.invCell),
      Math.floor((z - this.origin[2]) * this.invCell),
    ];
  }

  cellIndex(x: number, y: number, z: number): number {
    const [ix, iy, iz] = this.cellCoords(x, y, z);
    return hashCellCoords(ix, iy, iz, this.mask);
  }

  /** Rebuild the whole table from `field`. Clears first; callers never do. */
  build(field: ParticleField): void {
    this.counts.fill(0);
    this.overflow = 0;
    const { data, count } = field;
    let inserted = 0;
    for (let i = 0; i < count; i++) {
      const o = i * PARTICLE_STRIDE + PARTICLE_OFFSET.position;
      const cell = this.cellIndex(data[o]!, data[o + 1]!, data[o + 2]!);
      const n = this.counts[cell]!;
      if (n < this.bucketCapacity) {
        this.slots[cell * this.bucketCapacity + n] = i;
        this.counts[cell] = n + 1;
        inserted++;
      } else {
        this.overflow++;
      }
    }
    let used = 0;
    let maxBucket = 0;
    for (let c = 0; c < this.tableSize; c++) {
      const n = this.counts[c]!;
      if (n > 0) used++;
      if (n > maxBucket) maxBucket = n;
    }
    this.lastStats = {
      cells: this.tableSize,
      usedCells: used,
      inserted,
      overflow: this.overflow,
      maxBucket,
      loadFactor: used > 0 ? inserted / used : 0,
    };
  }

  stats(): SpatialHashStats {
    return this.lastStats;
  }

  /**
   * Visit every particle in the 27 cells around `i`'s own cell.
   *
   * Hash collisions mean a bucket can hold particles that are nowhere near `i`,
   * so this is a *candidate* set: callers must distance-test. What it guarantees
   * is the other direction -- no overlapping particle is missed, provided
   * `cellSize >= r_i + r_j` for every interacting pair, which is why the default
   * is twice the largest radius.
   *
   * Each bucket is visited at most once per call, so a neighbour is never
   * resolved twice no matter how much the table aliases.
   *
   * `i` itself is included in the visit (it lives in its own cell); the callback
   * skips it, keeping the loop shape identical to the WGSL kernel's.
   */
  forEachCandidate(
    field: ParticleField,
    i: number,
    visit: (j: number) => void,
  ): void {
    const o = i * PARTICLE_STRIDE + PARTICLE_OFFSET.position;
    const [cx, cy, cz] = this.cellCoords(field.data[o]!, field.data[o + 1]!, field.data[o + 2]!);
    const capacity = this.bucketCapacity;
    const seen = this.visitedCells;
    let seenCount = 0;
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const cell = hashCellCoords(cx + dx, cy + dy, cz + dz, this.mask);
          let duplicate = false;
          for (let k = 0; k < seenCount; k++) {
            if (seen[k] === cell) {
              duplicate = true;
              break;
            }
          }
          if (duplicate) continue;
          seen[seenCount++] = cell;
          const n = this.counts[cell]!;
          const base = cell * capacity;
          for (let s = 0; s < n; s++) {
            const j = this.slots[base + s]!;
            if (j !== i) visit(j);
          }
        }
      }
    }
  }
}

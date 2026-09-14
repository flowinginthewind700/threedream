/**
 * Seeded PRNG utilities.
 *
 * Everything stochastic in ThreeDream (weight init, action sampling,
 * environment reset noise, reward shaping jitter) goes through a `Rng` so a
 * run is reproducible bit-for-bit from its seed. Physics determinism alone is
 * not enough: a trained policy that cannot be re-derived from a seed cannot be
 * audited or A/B tested.
 */

export type RngState = { s: number };

/** xorshift32. Cheap, good enough for sampling; not cryptographic. */
export class Rng {
  private s: number;
  private spare: number | null = null;

  constructor(seed = 0x2f6e2b1) {
    // Avoid the all-zero fixed point of xorshift.
    this.s = (seed | 0) === 0 ? 0x9e3779b9 : seed | 0;
  }

  get seed(): number {
    return this.s;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    let x = this.s;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.s = x;
    return ((x >>> 0) % 16777216) / 16777216;
  }

  /** Uniform float in [min, max). */
  range(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  /** Uniform integer in [min, max] inclusive. */
  int(min: number, max: number): number {
    return Math.floor(this.range(min, max + 1));
  }

  /**
   * Standard normal via Box-Muller. Caches the second sample, so calls come in
   * pairs internally; do not interleave with a manual `next()` and expect the
   * same stream.
   */
  gaussian(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = this.next() * 2 - 1;
      v = this.next() * 2 - 1;
      s = u * u + v * v;
    } while (s === 0 || s >= 1);
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * mul;
    return u * mul;
  }

  /** Fisher-Yates shuffle in place. */
  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.int(0, i);
      const tmp = items[i]!;
      items[i] = items[j]!;
      items[j] = tmp;
    }
    return items;
  }

  /** Fill with uniform floats in [-limit, limit] (Glorot-style bound passed in). */
  uniformInto(out: Float32Array, limit: number): Float32Array {
    for (let i = 0; i < out.length; i++) out[i] = this.range(-limit, limit);
    return out;
  }

  fork(salt = 1): Rng {
    let h = this.s ^ Math.imul(salt, 0x85ebca6b);
    h = Math.imul(h ^ (h >>> 15), 0x2545f491);
    return new Rng(h ^ (h >>> 13));
  }
}

export function createRng(seed?: number): Rng {
  return new Rng(seed);
}

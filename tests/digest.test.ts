/**
 * `core/digest.ts` -- the byte-level hash every "these two runs agree" claim
 * in the repo is built on.
 *
 * The interesting failure mode here is not a wrong hash but a *vacuous* one: a
 * digest that ignores length, or ignores position, matches when it should not,
 * and then a truncated loop or a reordered buffer passes CI. Most of this file
 * is therefore about collisions that must not happen.
 */

import { describe, expect, it } from 'vitest';
import { digestHex, digestWithCount, hashBytes } from '../src/core/digest.js';
import { digestValuesHex } from '../src/physics/reference.js';

const bytes = (...xs: number[]): Uint8Array => new Uint8Array(xs);

describe('hashBytes', () => {
  it('is 16 lowercase hex characters for any input', () => {
    for (const input of [bytes(), bytes(0), bytes(1, 2, 3), bytes(255, 0, 128)]) {
      expect(hashBytes(input)).toMatch(/^[0-9a-f]{16}$/);
    }
  });

  it('is stable, and starts from the FNV offset basis on an empty buffer', () => {
    expect(hashBytes(bytes())).toBe(hashBytes(bytes()));
    // h1 is the untouched FNV-1a basis; h2 is the second accumulator's basis.
    expect(hashBytes(bytes())).toBe('811c9dc501000193');
  });

  it('distinguishes a reordering of the same bytes', () => {
    expect(hashBytes(bytes(1, 2, 3))).not.toBe(hashBytes(bytes(3, 2, 1)));
  });

  it('distinguishes a buffer from itself plus trailing zeros', () => {
    // The single-accumulator failure this guards: FNV-1a over `[7]` and over
    // `[7, 0, 0]` differ, but a hash that folded in no position information
    // would be one refactor away from not differing.
    expect(hashBytes(bytes(7))).not.toBe(hashBytes(bytes(7, 0, 0)));
  });

  it('distinguishes equal-length buffers that differ in one byte', () => {
    const a = bytes(1, 2, 3, 4);
    const b = bytes(1, 2, 3, 5);
    expect(hashBytes(a)).not.toBe(hashBytes(b));
  });

  it('mixes every byte: flipping a high bit anywhere changes the digest', () => {
    const base = new Uint8Array(64);
    for (let i = 0; i < base.length; i++) base[i] = i;
    const reference = hashBytes(base);
    for (let i = 0; i < base.length; i++) {
      const copy = base.slice();
      copy[i] ^= 0x80;
      expect(hashBytes(copy), `byte ${i} did not affect the digest`).not.toBe(reference);
    }
  });
});

describe('digestHex', () => {
  it('hashes exactly the bytes a view owns, including its offset', () => {
    const buffer = new ArrayBuffer(16);
    const all = new Uint8Array(buffer);
    all.set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const tail = new Uint8Array(buffer, 8, 8);
    expect(digestHex(tail)).toBe(hashBytes(bytes(9, 10, 11, 12, 13, 14, 15, 16)));
    expect(digestHex(tail)).not.toBe(digestHex(all));
  });

  it('agrees across typed-array types over the same bytes', () => {
    const f32 = new Float32Array([1.5, -2.25, 0]);
    const u32 = new Uint32Array(f32.buffer);
    expect(digestHex(f32)).toBe(digestHex(u32));
    expect(digestHex(f32)).not.toBe(digestHex(new Float32Array([1.5, -2.25])));
  });

  it('separates two floats that are close but not equal', () => {
    const a = new Float64Array([1 / 3]);
    const b = new Float64Array([1 / 3 + Number.EPSILON]);
    expect(digestHex(a)).not.toBe(digestHex(b));
  });
});

describe('digestWithCount', () => {
  it('reports element count, not byte count', () => {
    const values = new Float32Array(6);
    expect(digestWithCount(values)).toBe(`${hashBytes(new Uint8Array(24))}:6`);
  });

  it('is what a golden digest string looks like', () => {
    expect(digestWithCount(new Float64Array([1, 2, 3]))).toMatch(/^[0-9a-f]{16}:3$/);
  });

  it('does not collide with a shorter buffer that hashes the same prefix', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([1, 2, 3, 0]);
    expect(digestWithCount(a)).not.toBe(digestWithCount(b));
  });
});

describe('the physics reference digest delegates here', () => {
  it('digestValuesHex is digestHex, so the golden run digest is unchanged', () => {
    const values = new Float64Array([0.1, -0.2, 3.5, Number.MAX_SAFE_INTEGER]);
    expect(digestValuesHex(values)).toBe(digestHex(values));
  });
});

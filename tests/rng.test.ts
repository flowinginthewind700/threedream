import { describe, expect, it } from 'vitest';

import { Rng } from '../src/core/rng.js';

describe('Rng', () => {
  it('is reproducible from a seed', () => {
    const a = new Rng(42);
    const b = new Rng(42);
    for (let i = 0; i < 1000; i++) expect(a.next()).toBe(b.next());
  });

  it('avoids the xorshift all-zero fixed point', () => {
    const rng = new Rng(0);
    expect(rng.next()).not.toBe(0);
    expect(new Rng(0).next()).not.toBe(0);
  });

  it('keeps uniform draws inside [0, 1)', () => {
    const rng = new Rng(7);
    for (let i = 0; i < 10000; i++) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('honours range bounds', () => {
    const rng = new Rng(11);
    for (let i = 0; i < 5000; i++) {
      const v = rng.range(-3, 5);
      expect(v).toBeGreaterThanOrEqual(-3);
      expect(v).toBeLessThan(5);
    }
  });

  it('honours int bounds inclusively and hits both ends', () => {
    const rng = new Rng(3);
    const seen = new Set<number>();
    for (let i = 0; i < 5000; i++) {
      const v = rng.int(1, 3);
      expect(Number.isInteger(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(1);
      expect(v).toBeLessThanOrEqual(3);
      seen.add(v);
    }
    expect(seen).toEqual(new Set([1, 2, 3]));
  });

  it('produces an approximately standard gaussian', () => {
    const rng = new Rng(99);
    let sum = 0;
    let sumSq = 0;
    const n = 200000;
    for (let i = 0; i < n; i++) {
      const v = rng.gaussian();
      sum += v;
      sumSq += v * v;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    // Box-Muller with xorshift32 is not high-precision, so allow wide bounds.
    expect(Math.abs(mean)).toBeLessThan(0.02);
    expect(Math.abs(variance - 1)).toBeLessThan(0.05);
  });

  it('shuffles without adding or losing elements', () => {
    const rng = new Rng(5);
    const items = Array.from({ length: 50 }, (_, i) => i);
    const original = [...items];
    rng.shuffle(items);
    expect([...items].sort((a, b) => a - b)).toEqual(original);
  });

  it('is a permutation shuffle (deterministic per seed)', () => {
    const a = new Rng(13).shuffle([1, 2, 3, 4, 5, 6, 7, 8]);
    const b = new Rng(13).shuffle([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(a).toEqual(b);
  });

  it('forks into an independent but reproducible stream', () => {
    const parent = new Rng(21);
    const f1 = parent.fork(4);
    const f2 = new Rng(21).fork(4);
    expect(f1.next()).toBe(f2.next());
    // Different salt -> different stream.
    const f3 = new Rng(21).fork(5);
    expect(f1.seed).not.toBe(f3.seed);
  });
});

import { describe, expect, it } from 'vitest';

import { DEFAULT_FIXED_DT, FixedClock } from '../src/core/clock.js';

describe('FixedClock', () => {
  it('defaults to 60 Hz', () => {
    const clock = new FixedClock();
    expect(clock.fixedDt).toBeCloseTo(DEFAULT_FIXED_DT, 12);
    expect(clock.fixedDt).toBeCloseTo(1 / 60, 12);
  });

  it('rejects a non-positive fixed step', () => {
    expect(() => new FixedClock({ fixedDt: 0 })).toThrow(RangeError);
    expect(() => new FixedClock({ fixedDt: -1 })).toThrow(RangeError);
  });

  it('emits a step only once enough time has accumulated', () => {
    // Binary-exact values so the accumulator is not at the mercy of rounding.
    const clock = new FixedClock({ fixedDt: 0.5 });
    expect(clock.update(0.25)).toBe(0);
    expect(clock.update(0.25)).toBe(1);
    expect(clock.steps).toBe(1);
    expect(clock.time).toBeCloseTo(0.5, 10);
    expect(clock.alpha).toBeCloseTo(0, 10);
  });

  it('catches up after a long frame, in whole steps', () => {
    const clock = new FixedClock({ fixedDt: 1 / 60 });
    const steps = clock.update(0.1); // ~6 ticks
    expect(steps).toBe(6);
    expect(clock.steps).toBe(6);
    // The remainder stays in the accumulator and shows up as alpha.
    expect(clock.alpha).toBeGreaterThanOrEqual(0);
    expect(clock.alpha).toBeLessThan(1);
  });

  it('is deterministic: identical frame sequences give identical time', () => {
    const frames = [0.016, 0.033, 0.008, 0.05, 0.0167];
    const a = new FixedClock({ fixedDt: 1 / 60 });
    const b = new FixedClock({ fixedDt: 1 / 60 });
    const seenA = frames.map((dt) => a.update(dt));
    const seenB = frames.map((dt) => b.update(dt));
    expect(seenA).toEqual(seenB);
    expect(a.steps).toBe(b.steps);
    expect(a.time).toBe(b.time);
  });

  it('caps steps per frame to avoid the spiral of death', () => {
    const clock = new FixedClock({ fixedDt: 1 / 60, maxStepsPerFrame: 3 });
    expect(clock.update(10)).toBe(3);
    // Backlog is dropped, not carried forever.
    expect(clock.update(10)).toBe(3);
  });

  it('clamps the frame delta to maxFrameDt', () => {
    const clock = new FixedClock({ fixedDt: 1 / 60, maxFrameDt: 0.05, maxStepsPerFrame: 100 });
    expect(clock.update(5)).toBe(3); // 0.05 / (1/60) = 3
  });

  it('ignores negative frame deltas', () => {
    const clock = new FixedClock();
    expect(clock.update(-1)).toBe(0);
    expect(clock.steps).toBe(0);
  });

  it('advance() runs exact steps regardless of wall clock', () => {
    const clock = new FixedClock({ fixedDt: 1 / 120 });
    clock.advance(10);
    expect(clock.steps).toBe(10);
    expect(clock.time).toBeCloseTo(10 / 120, 12);
  });

  it('reset() clears accumulator, time and steps', () => {
    const clock = new FixedClock();
    clock.update(0.1);
    clock.reset();
    expect(clock.steps).toBe(0);
    expect(clock.time).toBe(0);
    expect(clock.alpha).toBe(0);
  });
});

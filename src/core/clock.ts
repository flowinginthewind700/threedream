/**
 * Fixed-timestep simulation clock.
 *
 * Rendering runs at whatever rate the display allows; simulation runs at a
 * constant rate so results are reproducible and physics stays stable. The
 * clock accumulates real time and hands it out in fixed slices, with an
 * interpolation alpha so rendering between two simulation ticks does not
 * stutter.
 */

export interface FixedClockOptions {
  /** Simulation step in seconds. 1/60 is the game default; 1/120 for precision work. */
  fixedDt?: number;
  /** Cap on real time consumed per frame, guarding against the spiral of death. */
  maxFrameDt?: number;
  /** Hard cap on simulation steps per frame. */
  maxStepsPerFrame?: number;
}

export const DEFAULT_FIXED_DT = 1 / 60;

export class FixedClock {
  readonly fixedDt: number;
  readonly maxFrameDt: number;
  readonly maxStepsPerFrame: number;

  private accumulator = 0;
  private simulationTime = 0;
  private stepCount = 0;

  constructor(options: FixedClockOptions = {}) {
    this.fixedDt = options.fixedDt ?? DEFAULT_FIXED_DT;
    this.maxFrameDt = options.maxFrameDt ?? 0.25;
    this.maxStepsPerFrame = options.maxStepsPerFrame ?? 8;
    if (!(this.fixedDt > 0)) throw new RangeError('fixedDt must be positive');
  }

  /** Seconds of simulated time elapsed (ticks * fixedDt). */
  get time(): number {
    return this.simulationTime;
  }

  get steps(): number {
    return this.stepCount;
  }

  /** Blend factor in [0, 1) from the last tick toward the next one. */
  get alpha(): number {
    return this.accumulator / this.fixedDt;
  }

  /**
   * Consume `frameDt` seconds of wall-clock time and report how many fixed
   * steps should run. Deterministic for a given sequence of inputs.
   */
  update(frameDt: number): number {
    const clamped = Math.min(Math.max(frameDt, 0), this.maxFrameDt);
    this.accumulator += clamped;
    let steps = 0;
    while (this.accumulator >= this.fixedDt && steps < this.maxStepsPerFrame) {
      this.accumulator -= this.fixedDt;
      this.simulationTime += this.fixedDt;
      this.stepCount++;
      steps++;
    }
    // Still saturated after the cap: drop the backlog instead of growing forever.
    if (steps >= this.maxStepsPerFrame) {
      this.accumulator = Math.min(this.accumulator, this.fixedDt);
    }
    return steps;
  }

  /** Run exactly `count` steps, ignoring wall-clock time. Used by headless training. */
  advance(count: number): void {
    for (let i = 0; i < count; i++) {
      this.simulationTime += this.fixedDt;
      this.stepCount++;
    }
  }

  reset(): void {
    this.accumulator = 0;
    this.simulationTime = 0;
    this.stepCount = 0;
  }
}

/**
 * Learned value baseline (critic).
 *
 * Why this exists: returns-to-go `R_t` in an episode with dense, mostly
 * negative rewards have a strong monotonic trend in `t` — later steps simply
 * have fewer terms left to accumulate. Standardising advantages *within* an
 * episode therefore makes the advantage correlate with the timestep index
 * rather than with action quality, and the policy learns a time artifact.
 * Subtracting a state-dependent estimate `V(s_t)` removes that trend and leaves
 * the part of the return the action actually caused.
 *
 * It is a plain MLP trained by MSE regression on observed returns-to-go, which
 * keeps REINFORCE honest without the machinery of a full actor-critic method.
 */

import { Rng } from '../core/rng.js';
import { Mlp } from './mlp.js';

export interface ValueBaselineOptions {
  observationSize: number;
  hiddenSizes?: number[];
  seed?: number;
  learningRate?: number;
  gradClip?: number;
}

export interface ValueSample {
  observation: ArrayLike<number>;
  /** Observed discounted return-to-go from this state. */
  target: number;
}

export class ValueBaseline {
  readonly mlp: Mlp;
  readonly learningRate: number;
  readonly gradClip: number;

  private readonly scratch: Float32Array;
  private readonly scratchGrad: Float32Array;
  /** Running stats of targets, used to keep the regression scale stable. */
  private targetMean = 0;
  private targetVar = 1;
  private targetCount = 0;

  constructor(options: ValueBaselineOptions) {
    this.mlp = new Mlp(
      {
        sizes: [
          options.observationSize,
          ...(options.hiddenSizes ?? [64, 64]),
          1,
        ],
      },
      new Rng(options.seed ?? 991),
    );
    this.learningRate = options.learningRate ?? 0.05;
    this.gradClip = options.gradClip ?? 10;
    this.scratch = new Float32Array(1);
    this.scratchGrad = new Float32Array(1);
  }

  /** State-value estimate. */
  predict(observation: ArrayLike<number>): number {
    const out = this.mlp.forward(observation, this.scratch);
    const value = out[0]!;
    return Number.isFinite(value) ? value : 0;
  }

  /**
   * Regression update over a batch of (state, return) samples.
   * Loss = mean (V(s) - R)^2, so dLoss/dV = 2 (V - R) / n and the MLP's
   * descent step minimises it directly.
   */
  train(samples: ValueSample[]): number {
    if (samples.length === 0) return 0;

    // Track target statistics; predictions are compared in target space, so a
    // drifting reward scale does not destabilise the step size.
    this.updateTargetStats(samples);

    this.mlp.zeroGradients();
    let loss = 0;
    const invN = 1 / samples.length;
    for (const sample of samples) {
      const value = this.predict(sample.observation);
      const error = value - sample.target;
      loss += error * error;
      // dLoss/dV averaged over the batch.
      this.scratchGrad[0] = 2 * error * invN;
      this.mlp.backward(this.scratchGrad);
    }
    this.mlp.applyGradients(this.learningRate, this.gradClip);
    return loss * invN;
  }

  private updateTargetStats(samples: ValueSample[]): void {
    for (const sample of samples) {
      const t = sample.target;
      if (!Number.isFinite(t)) continue;
      this.targetCount++;
      const delta = t - this.targetMean;
      this.targetMean += delta / this.targetCount;
      this.targetVar += (delta * (t - this.targetMean) - this.targetVar) / this.targetCount;
    }
    if (this.targetVar < 1e-6) this.targetVar = 1e-6;
  }

  /** Normalise an advantage by the observed return spread. */
  normalizeAdvantage(advantage: number): number {
    return advantage / Math.sqrt(this.targetVar);
  }

  get targetStd(): number {
    return Math.sqrt(this.targetVar);
  }

  get sampleCount(): number {
    return this.targetCount;
  }
}

/**
 * Diagonal-Gaussian policy over an MLP trunk.
 *
 * Continuous control needs a stochastic policy: the network emits per-dimension
 * means, a state-independent log-sigma vector is learned alongside it, and the
 * action is sampled from the resulting Gaussian. `logProb` and `entropy` give
 * the policy-gradient and exploration terms without a second framework.
 */

/**
 * Diagonal-Gaussian policy over an MLP trunk.
 *
 * Continuous control needs a stochastic policy: the network emits per-dimension
 * means, a state-independent log-sigma vector is learned alongside it, and the
 * action is sampled from the resulting Gaussian.
 *
 * Actions are returned *unbounded*. Bounding is the environment's job (it
 * clamps to its own action limits when applying a force), and the log density
 * must be evaluated on the same unbounded sample that was drawn. Clamping
 * inside the policy would break that: once the mean drifts past the bound, the
 * sample saturates while `mean` keeps moving, `raw - mean` becomes a large
 * one-sided constant, and the `logSigma` gradient turns systematically
 * negative, collapsing exploration to the sigma floor.
 */

import { Rng } from '../core/rng.js';
import { Mlp, type MlpSnapshot } from './mlp.js';

export interface GaussianPolicyOptions {
  observationSize: number;
  actionSize: number;
  hiddenSizes?: number[];
  seed?: number;
  /** Initial log standard deviation; exp(-0.5) ~= 0.61. */
  initialLogSigma?: number;
  /** Hard bound on sampled actions, keeps impulses physically sane. */
  actionScale?: number;
  minLogSigma?: number;
  maxLogSigma?: number;
}

export interface PolicySnapshot {
  kind: 'gaussian-mlp';
  version: 1;
  actionScale: number;
  logSigmas: number[];
  mlp: MlpSnapshot;
}

/** Reusable output buffer for {@link GaussianPolicy.policyGradients}. */
export interface PolicyGradientOut {
  meanGrad: Float32Array;
  logSigmaGrad: Float32Array;
  /** log pi(action | observation), filled in by the same forward pass. */
  logProb: number;
}

const SQRT_2PI = Math.sqrt(2 * Math.PI);

export class GaussianPolicy {
  readonly observationSize: number;
  readonly actionSize: number;
  readonly actionScale: number;
  readonly logSigmas: Float32Array;
  readonly mlp: Mlp;
  private readonly rng: Rng;
  private readonly minLogSigma: number;
  private readonly maxLogSigma: number;
  private readonly scratchMean: Float32Array;

  constructor(options: GaussianPolicyOptions) {
    this.observationSize = options.observationSize;
    this.actionSize = options.actionSize;
    this.actionScale = options.actionScale ?? 1;
    this.minLogSigma = options.minLogSigma ?? -3;
    this.maxLogSigma = options.maxLogSigma ?? 1.5;
    const hidden = options.hiddenSizes ?? [64, 64];
    this.mlp = new Mlp(
      { sizes: [this.observationSize, ...hidden, this.actionSize] },
      new Rng(options.seed ?? 1234),
    );
    this.rng = new Rng((options.seed ?? 1234) ^ 0x5eed);
    this.logSigmas = new Float32Array(this.actionSize);
    this.logSigmas.fill(options.initialLogSigma ?? -0.5);
    this.scratchMean = new Float32Array(this.actionSize);
  }

  get parameterCount(): number {
    return this.mlp.parameterCount + this.logSigmas.length;
  }

  /**
   * Deterministic action: the mean. Use for evaluation and deployment.
   * The environment clamps to its own action bounds.
   */
  actGreedy(observation: ArrayLike<number>, out?: Float32Array): Float32Array {
    const mean = this.mlp.forward(observation, this.scratchMean);
    const result = out ?? new Float32Array(this.actionSize);
    for (let i = 0; i < this.actionSize; i++) {
      result[i] = mean[i]! * this.actionScale;
    }
    return result;
  }

  /** Stochastic sample used during training. Returns the unbounded draw. */
  sample(observation: ArrayLike<number>, out?: Float32Array): Float32Array {
    const mean = this.mlp.forward(observation, this.scratchMean);
    const result = out ?? new Float32Array(this.actionSize);
    for (let i = 0; i < this.actionSize; i++) {
      const sigma = Math.exp(this.logSigmas[i]!);
      result[i] = (mean[i]! + sigma * this.rng.gaussian()) * this.actionScale;
    }
    return result;
  }

  /**
   * log pi(action | observation) for the *unbounded* action as sampled.
   * Re-runs the forward pass, which resets the MLP activation cache.
   */
  logProb(observation: ArrayLike<number>, action: ArrayLike<number>): number {
    const mean = this.mlp.forward(observation, this.scratchMean);
    let total = 0;
    for (let i = 0; i < this.actionSize; i++) {
      const logSigma = this.logSigmas[i]!;
      const sigma = Math.exp(logSigma);
      // Undo actionScale so the density is over the Gaussian variable itself.
      const raw = action[i]! / this.actionScale;
      const diff = raw - mean[i]!;
      total +=
        -0.5 * (diff * diff) / (sigma * sigma) - logSigma - Math.log(SQRT_2PI);
    }
    return total;
  }

  /**
   * Policy-gradient terms for one (observation, action) pair.
   *
   * Returns d logProb / d mean and d logProb / d logSigma, each multiplied by
   * `scale` (the advantage weight), plus the log density itself. Exactly one
   * forward pass happens, so the MLP's activation cache still matches this
   * observation when the caller invokes `mlp.backward(meanGrad)` — calling
   * `logProb` separately would clobber it and cost a second forward pass.
   *
   * Gradient of the Gaussian log density, with `raw = action / actionScale`:
   *   logp = -0.5 (raw - mean)^2 / sigma^2 - logSigma - const
   *   d logp / d mean     =  (raw - mean) / sigma^2
   *   d logp / d logSigma =  (raw - mean)^2 / sigma^2 - 1
   */
  policyGradients(
    observation: ArrayLike<number>,
    action: ArrayLike<number>,
    scale: number,
    out?: PolicyGradientOut,
  ): PolicyGradientOut {
    const mean = this.mlp.forward(observation, this.scratchMean);
    const result =
      out ?? {
        meanGrad: new Float32Array(this.actionSize),
        logSigmaGrad: new Float32Array(this.actionSize),
        logProb: 0,
      };
    const { meanGrad, logSigmaGrad } = result;
    let logProb = 0;
    for (let i = 0; i < this.actionSize; i++) {
      const logSigma = this.logSigmas[i]!;
      const sigma = Math.exp(logSigma);
      const raw = action[i]! / this.actionScale;
      const diff = raw - mean[i]!;
      const invVar = 1 / (sigma * sigma);
      meanGrad[i] = diff * invVar * scale;
      logSigmaGrad[i] = (diff * diff * invVar - 1) * scale;
      logProb += -0.5 * diff * diff * invVar - logSigma - Math.log(SQRT_2PI);
    }
    result.logProb = logProb;
    return result;
  }

  applyLogSigmaGradient(grad: Float32Array, learningRate: number): void {
    for (let i = 0; i < this.logSigmas.length; i++) {
      const next = this.logSigmas[i]! + learningRate * clamp(grad[i]!, -5, 5);
      this.logSigmas[i] = clamp(next, this.minLogSigma, this.maxLogSigma);
    }
  }

  /** Differential entropy, summed over action dimensions. */
  entropy(): number {
    let total = 0;
    for (let i = 0; i < this.actionSize; i++) {
      const sigma = Math.exp(this.logSigmas[i]!);
      total += 0.5 * Math.log(2 * Math.PI * Math.E * sigma * sigma);
    }
    return total;
  }

  toJSON(): PolicySnapshot {
    return {
      kind: 'gaussian-mlp',
      version: 1,
      actionScale: this.actionScale,
      logSigmas: Array.from(this.logSigmas),
      mlp: this.mlp.toJSON(),
    };
  }

  static fromJSON(snapshot: PolicySnapshot, seed?: number): GaussianPolicy {
    const policy = new GaussianPolicy({
      observationSize: snapshot.mlp.sizes[0]!,
      actionSize: snapshot.mlp.sizes[snapshot.mlp.sizes.length - 1]!,
      hiddenSizes: snapshot.mlp.sizes.slice(1, -1),
      actionScale: snapshot.actionScale,
      seed,
    });
    // Replace the trunk wholesale (fromJSON re-inits, so copy parameters).
    const restored = Mlp.fromJSON(snapshot.mlp);
    policy.mlp.setParameters(restored.getParameters());
    for (let i = 0; i < policy.logSigmas.length; i++) {
      policy.logSigmas[i] = snapshot.logSigmas[i] ?? policy.logSigmas[i]!;
    }
    return policy;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

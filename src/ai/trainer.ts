/**
 * Policy-gradient trainer: batched rollouts, GAE advantages, learned critic.
 *
 * Deliberately not PPO. There are no importance ratios and no replay buffer, so
 * the whole update is readable end to end while still being a real
 * policy-gradient method: roll out, estimate advantages, take one descent step,
 * throw the data away.
 *
 * Three design decisions here were each forced by a measured failure, and they
 * are worth recording because the failures are silent — entropy stays healthy,
 * training returns look plausible, and greedy evaluation simply never improves.
 *
 * 1. *Batch size.* One update per episode does not learn this task at all. A
 *    200-step trajectory yields a GAE advantage whose std is ~0.02 while the
 *    per-episode noise (where the puck spawned, which way the first sample
 *    kicked it) is an order of magnitude larger. Normalising the advantage to
 *    unit scale — which the learning-rate-independence argument requires — then
 *    amplifies that noise back up and the policy performs a random walk.
 *    `episodesPerUpdate` (default 16) averages the gradient over ~3200
 *    transitions per step, which is what actually makes the signal visible.
 *
 * 2. *GAE instead of a Monte-Carlo advantage.* With a scalar or per-episode
 *    baseline, dense rewards leave the advantage correlated with the timestep
 *    index: later steps have fewer reward terms left to accumulate, so the
 *    policy learns a time artifact. Fitting a state critic and using the raw
 *    residual `R_t - V(s_t)` is better but still wrong, because once the critic
 *    fits the return well the residual is *dominated by sampling noise* (value
 *    loss ~0.01 against a return spread of ~1, i.e. R^2 near 0.99). GAE uses
 *    the one-step TD residual `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`,
 *    which only has to be right on average, smoothed by a `(gamma * lambda)`
 *    window. `lambda = 1` recovers the Monte Carlo case, `lambda = 0` is
 *    one-step TD.
 *
 * 3. *Critic target.* The critic is fitted on the lambda-return `A_t + V(s_t)`,
 *    the exact quantity the advantage is measured against, rather than on raw
 *    returns-to-go. Values for the current policy are read *before* that fit, or
 *    the baseline absorbs the return it is supposed to explain and the advantage
 *    collapses toward zero.
 *
 * Time limits are handled by `terminated` vs `done` on `StepResult`. Hitting the
 * step budget is an artifact of the harness, not a property of the task, so a
 * truncated episode still bootstraps `V(s_{t+1})`; treating it as absorbing
 * teaches the critic that value decays to zero as the clock runs out.
 *
 * `train()` is synchronous and framework-free, so the same call drives the
 * browser demo and headless CI training (`npm run train`).
 */

import { Rng } from '../core/rng.js';
import { GaussianPolicy, type PolicyGradientOut } from './policy.js';
import { ValueBaseline, type ValueSample } from './baseline.js';
import type { LearningEnvironment } from '../envs/types.js';

export interface TrainerOptions {
  learningRate?: number;
  /** Entropy bonus weight; keeps exploration alive early in training. */
  entropyCoefficient?: number;
  /** Learning rate for the value critic. */
  criticLr?: number;
  /** Clip per-parameter gradient magnitude. */
  gradClip?: number;
  /** GAE lambda in [0, 1]; 1 is Monte Carlo, 0 is one-step TD. */
  gaeLambda?: number;
  /** Episodes rolled out per policy update. See class docs, item 1. */
  episodesPerUpdate?: number;
  /** Critic passes over the collected batch per update. */
  criticEpochs?: number;
  /** Minibatch size for the critic's SGD steps. */
  criticMinibatchSize?: number;
  seed?: number;
}

export interface EpisodeResult {
  /** Index of the episode within `train()`. */
  episode: number;
  steps: number;
  return: number;
  entropy: number;
  meanLogProb: number;
  /** Mean squared critic error for the update this episode belonged to. */
  valueLoss: number;
  /** Std-dev of the raw (un-normalised) advantages in that update. */
  advantageStd: number;
  /** Mean episode return over that update's batch. */
  batchMeanReturn: number;
}

export interface TrainResult {
  policy: GaussianPolicy;
  history: EpisodeResult[];
  bestReturn: number;
  episodes: number;
  steps: number;
  /** Number of policy updates performed. */
  updates: number;
}

/** Per-episode trace geometry; strides are fixed for the whole run. */
interface TraceLayout {
  readonly episodes: number;
  readonly maxSteps: number;
  readonly observationSize: number;
  readonly actionSize: number;
  /** maxSteps + 1: GAE bootstraps from V(s_{t+1}). */
  readonly obsStride: number;
  readonly stepStride: number;
}

export class Trainer {
  readonly learningRate: number;
  readonly entropyCoefficient: number;
  readonly gradClip: number;
  readonly gaeLambda: number;
  readonly episodesPerUpdate: number;
  readonly criticEpochs: number;
  readonly criticMinibatchSize: number;

  private readonly rng: Rng;
  private readonly criticRng: Rng;
  /**
   * Optional state-value critic. `train()` builds one automatically from the
   * environment's observation size when it is absent; set it explicitly via
   * `withCritic()` to control capacity or reuse across tasks.
   */
  private critic?: ValueBaseline;
  private criticLr: number;

  // Reused across updates: allocating per step inside a training loop
  // dominates cost once episodes run in the thousands.
  private obsTrace = new Float32Array(0);
  private actTrace = new Float32Array(0);
  private rewardTrace = new Float64Array(0);
  private terminalTrace = new Uint8Array(0);
  private valueTrace = new Float64Array(0);
  private returnTrace = new Float64Array(0);
  private advantageTrace = new Float64Array(0);
  private lengths = new Int32Array(0);
  private sampleIndex = new Int32Array(0);
  private readonly criticSamples: ValueSample[] = [];
  private layout?: TraceLayout;

  constructor(options: TrainerOptions = {}) {
    this.learningRate = options.learningRate ?? 0.01;
    this.entropyCoefficient = options.entropyCoefficient ?? 0.001;
    this.criticLr = options.criticLr ?? options.learningRate ?? 0.01;
    this.gradClip = options.gradClip ?? 5;
    this.gaeLambda = options.gaeLambda ?? 0.95;
    this.episodesPerUpdate = Math.max(1, options.episodesPerUpdate ?? 16);
    this.criticEpochs = Math.max(1, options.criticEpochs ?? 4);
    this.criticMinibatchSize = Math.max(1, options.criticMinibatchSize ?? 64);
    for (const [name, value] of Object.entries({
      gaeLambda: this.gaeLambda,
    })) {
      if (!(value >= 0 && value <= 1)) {
        throw new RangeError(`${name} must be in [0, 1]`);
      }
    }
    this.rng = new Rng(options.seed ?? 0xC0FFEE);
    this.criticRng = this.rng.fork(3);
  }

  /** Attach an explicit value critic. */
  withCritic(observationSize: number, hiddenSizes: number[] = [64, 64]): this {
    this.critic = new ValueBaseline({
      observationSize,
      hiddenSizes,
      learningRate: this.criticLr,
      seed: this.criticRng.seed,
    });
    return this;
  }

  /** The value critic in use, or `undefined` before the first update. */
  get valueBaseline(): ValueBaseline | undefined {
    return this.critic;
  }

  private ensureCritic(env: LearningEnvironment): ValueBaseline {
    if (!this.critic) this.withCritic(env.observationSize);
    return this.critic!;
  }

  /**
   * Run `episodes` episodes, updating the policy every `episodesPerUpdate` of
   * them (and once more for the trailing partial batch). `onEpisode` is called
   * after each episode for progress reporting.
   */
  train(
    policy: GaussianPolicy,
    env: LearningEnvironment,
    episodes: number,
    onEpisode?: (result: EpisodeResult) => void,
  ): TrainResult {
    const history: EpisodeResult[] = [];
    let bestReturn = Number.NEGATIVE_INFINITY;
    let totalSteps = 0;
    let updates = 0;
    let episode = 0;

    while (episode < episodes) {
      const count = Math.min(this.episodesPerUpdate, episodes - episode);
      const batch = this.runBatch(policy, env, count);
      updates++;
      for (let i = 0; i < count; i++) {
        const result: EpisodeResult = {
          episode: episode + i,
          steps: batch.lengths[i]!,
          return: batch.returns[i]!,
          entropy: batch.entropy,
          meanLogProb: batch.logProbs[i]! / Math.max(1, batch.lengths[i]!),
          valueLoss: batch.valueLoss,
          advantageStd: batch.advantageStd,
          batchMeanReturn: batch.meanReturn,
        };
        history.push(result);
        totalSteps += result.steps;
        if (result.return > bestReturn) bestReturn = result.return;
        onEpisode?.(result);
      }
      episode += count;
    }

    return { policy, history, bestReturn, episodes, steps: totalSteps, updates };
  }

  /** Roll out `count` episodes, then take one policy step and fit the critic. */
  private runBatch(
    policy: GaussianPolicy,
    env: LearningEnvironment,
    count: number,
  ): BatchStats {
    const critic = this.ensureCritic(env);
    const layout = this.ensureLayout(count, env, policy);
    const { observationSize, actionSize, obsStride, stepStride } = layout;

    const observationBuffer = new Float32Array(observationSize);
    const actionBuffer = new Float32Array(actionSize);
    const lengths = this.lengths;
    const rewards = new Float64Array(count);
    const logProbs = new Float64Array(count);

    for (let b = 0; b < count; b++) {
      const obsBase = b * obsStride * observationSize;
      const actBase = b * stepStride * actionSize;
      const stepBase = b * stepStride;
      let observation = env.reset(this.rng);
      this.obsTrace.set(observation, obsBase);

      let steps = 0;
      let done = false;
      let episodeReturn = 0;
      let logProbSum = 0;
      while (!done && steps < env.maxStepsPerEpisode) {
        const action = policy.sample(observation, actionBuffer);
        logProbSum += policy.logProb(observation, action);
        const next = env.step(action);
        this.actTrace.set(action, actBase + steps * actionSize);
        this.rewardTrace[stepBase + steps] = next.reward;
        // `terminated` distinguishes "the task ended" from "the clock ran out".
        this.terminalTrace[stepBase + steps] = (next.terminated ?? next.done)
          ? 1
          : 0;
        episodeReturn += next.reward;
        observation = env.observe(observationBuffer);
        this.obsTrace.set(observation, obsBase + (steps + 1) * observationSize);
        done = next.done;
        steps++;
      }
      lengths[b] = steps;
      rewards[b] = episodeReturn;
      logProbs[b] = logProbSum;
    }

    // State values under the *current* critic, read before it is refitted on
    // this batch: otherwise the baseline absorbs the return it must explain.
    for (let b = 0; b < count; b++) {
      const T = lengths[b]!;
      const obsBase = b * obsStride * observationSize;
      const valueBase = b * obsStride;
      for (let t = 0; t <= T; t++) {
        const at = obsBase + t * observationSize;
        this.valueTrace[valueBase + t] = critic.predict(
          this.obsTrace.subarray(at, at + observationSize),
        );
      }
    }

    // GAE, backwards per episode.
    const gamma = env.discount;
    const gammaLambda = gamma * this.gaeLambda;
    let totalSamples = 0;
    let advSum = 0;
    let advSqSum = 0;
    for (let b = 0; b < count; b++) {
      const T = lengths[b]!;
      const stepBase = b * stepStride;
      const valueBase = b * obsStride;
      let running = 0;
      for (let t = T - 1; t >= 0; t--) {
        const alive = this.terminalTrace[stepBase + t] === 1 ? 0 : 1;
        const delta =
          this.rewardTrace[stepBase + t]! +
          gamma * this.valueTrace[valueBase + t + 1]! * alive -
          this.valueTrace[valueBase + t]!;
        running = delta + gammaLambda * alive * running;
        this.advantageTrace[stepBase + t] = running;
        // Lambda-return: the critic's regression target.
        this.returnTrace[stepBase + t] = running + this.valueTrace[valueBase + t]!;
        advSum += running;
        advSqSum += running * running;
        totalSamples++;
      }
    }

    const advMean = totalSamples > 0 ? advSum / totalSamples : 0;
    const advVar =
      totalSamples > 0 ? Math.max(0, advSqSum / totalSamples - advMean * advMean) : 0;
    const rawAdvStd = Math.sqrt(advVar);
    const advStd = rawAdvStd + 1e-8;

    // Fit the critic on lambda-returns first: the policy gradient is already
    // computed against the pre-fit values, and a better critic helps the next
    // batch immediately.
    const valueLoss = this.fitCritic(critic, count, layout);

    this.updatePolicy(policy, count, layout, advMean, advStd);

    let meanReturn = 0;
    for (let b = 0; b < count; b++) meanReturn += rewards[b]!;
    meanReturn /= Math.max(1, count);

    return {
      lengths: Int32Array.from(lengths.subarray(0, count)),
      returns: rewards,
      logProbs,
      entropy: policy.entropy(),
      valueLoss,
      advantageStd: rawAdvStd,
      meanReturn,
    };
  }

  /** Minibatch SGD on the critic over the collected lambda-returns. */
  private fitCritic(
    critic: ValueBaseline,
    count: number,
    layout: TraceLayout,
  ): number {
    const { observationSize, stepStride, maxSteps } = layout;
    // Views into the trace buffers, safe because the critic consumes them
    // before the next batch overwrites the trace.
    const samples = this.criticSamples;
    samples.length = 0;
    for (let b = 0; b < count; b++) {
      const T = this.lengths[b]!;
      for (let t = 0; t < T; t++) {
        const index = b * stepStride + t;
        const obsAt = (b * (maxSteps + 1) + t) * observationSize;
        samples.push({
          observation: this.obsTrace.subarray(obsAt, obsAt + observationSize),
          target: this.returnTrace[index]!,
        });
      }
    }

    const minibatch: ValueSample[] = [];
    let loss = 0;
    let batches = 0;
    const size = Math.min(this.criticMinibatchSize, Math.max(1, samples.length));
    for (let epoch = 0; epoch < this.criticEpochs; epoch++) {
      this.shuffleSampleIndex(samples.length);
      for (let start = 0; start < samples.length; start += size) {
        const end = Math.min(samples.length, start + size);
        minibatch.length = 0;
        for (let i = start; i < end; i++) minibatch.push(samples[this.sampleIndex[i]!]!);
        loss += critic.train(minibatch);
        batches++;
      }
    }
    return batches > 0 ? loss / batches : 0;
  }

  /** One descent step on the policy over every transition in the batch. */
  private updatePolicy(
    policy: GaussianPolicy,
    count: number,
    layout: TraceLayout,
    advMean: number,
    advStd: number,
  ): void {
    const { observationSize, actionSize, stepStride, maxSteps } = layout;
    const scratch: PolicyGradientOut = {
      meanGrad: new Float32Array(actionSize),
      logSigmaGrad: new Float32Array(actionSize),
      logProb: 0,
    };
    const logSigmaAccum = new Float32Array(actionSize);
    policy.mlp.zeroGradients();

    let total = 0;
    for (let b = 0; b < count; b++) total += this.lengths[b]!;
    if (total === 0) return;
    const invTotal = 1 / total;

    for (let b = 0; b < count; b++) {
      const T = this.lengths[b]!;
      const obsBase = b * (maxSteps + 1) * observationSize;
      const actBase = b * stepStride * actionSize;
      const stepBase = b * stepStride;
      for (let t = 0; t < T; t++) {
        const obsAt = obsBase + t * observationSize;
        const actAt = actBase + t * actionSize;
        const observation = this.obsTrace.subarray(obsAt, obsAt + observationSize);
        const action = this.actTrace.subarray(actAt, actAt + actionSize);
        // Standardised advantage: keeps the learning rate independent of the
        // task's reward scale and centres the update so roughly half the
        // transitions push up and half push down.
        const advantage =
          (this.advantageTrace[stepBase + t]! - advMean) / advStd;
        const weight = advantage * invTotal;
        // Sign conventions are easy to get backwards:
        //   * `Mlp.backward` + `applyGradients` perform gradient DESCENT on the
        //     supplied output-gradient, so the trunk receives `-A * dlogpi/dmean`.
        //   * `applyLogSigmaGradient` performs ASCENT, so log-sigma accumulates
        //     `-grads.logSigmaGrad` (which already carries the negated scale)
        //     and the entropy bonus is added on top.
        const grads = policy.policyGradients(observation, action, -weight, scratch);
        // policyGradients ran the forward pass, so the activation cache matches
        // this sample and backward is consistent here.
        policy.mlp.backward(grads.meanGrad);
        for (let i = 0; i < actionSize; i++) {
          logSigmaAccum[i] -= grads.logSigmaGrad[i]!;
        }
      }
    }

    policy.mlp.applyGradients(this.learningRate, this.gradClip);
    // dH/dlogSigma = 1 per action dimension; one bonus per update, since the
    // weights above already average over the batch.
    for (let i = 0; i < logSigmaAccum.length; i++) {
      logSigmaAccum[i] += this.entropyCoefficient;
    }
    policy.applyLogSigmaGradient(logSigmaAccum, this.learningRate);
  }

  private shuffleSampleIndex(n: number): void {
    if (this.sampleIndex.length !== n) this.sampleIndex = new Int32Array(n);
    for (let i = 0; i < n; i++) this.sampleIndex[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = this.criticRng.int(0, i);
      const tmp = this.sampleIndex[i]!;
      this.sampleIndex[i] = this.sampleIndex[j]!;
      this.sampleIndex[j] = tmp;
    }
  }

  /**
   * Allocate (once per batch shape) the flat trace buffers GAE needs. Reused
   * across updates as long as the task geometry does not change.
   */
  private ensureLayout(
    episodes: number,
    env: LearningEnvironment,
    policy: GaussianPolicy,
  ): TraceLayout {
    const observationSize = policy.observationSize;
    const actionSize = policy.actionSize;
    const maxSteps = Math.max(1, env.maxStepsPerEpisode);
    const existing = this.layout;
    if (
      existing &&
      existing.episodes === episodes &&
      existing.maxSteps === maxSteps &&
      existing.observationSize === observationSize &&
      existing.actionSize === actionSize
    ) {
      return existing;
    }
    const obsStride = maxSteps + 1;
    const stepStride = maxSteps;
    this.layout = {
      episodes,
      maxSteps,
      observationSize,
      actionSize,
      obsStride,
      stepStride,
    };
    this.obsTrace = new Float32Array(episodes * obsStride * observationSize);
    this.actTrace = new Float32Array(episodes * stepStride * actionSize);
    this.rewardTrace = new Float64Array(episodes * stepStride);
    this.terminalTrace = new Uint8Array(episodes * stepStride);
    this.valueTrace = new Float64Array(episodes * obsStride);
    this.returnTrace = new Float64Array(episodes * stepStride);
    this.advantageTrace = new Float64Array(episodes * stepStride);
    this.lengths = new Int32Array(episodes);
    return this.layout;
  }
}

interface BatchStats {
  lengths: Int32Array;
  returns: Float64Array;
  logProbs: Float64Array;
  entropy: number;
  valueLoss: number;
  advantageStd: number;
  meanReturn: number;
}

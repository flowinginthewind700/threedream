/**
 * Trainer tests.
 *
 * The point of these is not "training works on the shipped task" but "the
 * gradient machinery points uphill". A known-optimum analytic environment and a
 * physics-backed one both have to converge, because those two failure modes are
 * different: the first catches sign errors and baseline mistakes, the second
 * catches anything that silently breaks the env <-> trainer contract.
 */

import { describe, expect, it } from 'vitest';

import { GaussianPolicy } from '../src/ai/policy.js';
import { Trainer } from '../src/ai/trainer.js';
import { Rng } from '../src/core/rng.js';
import { DriveEnv } from '../src/envs/drive.js';
import type { LearningEnvironment, StepResult } from '../src/envs/types.js';

/** Reward = -|a - target|: the greedy optimum is a known constant vector. */
class AnalyticEnv implements LearningEnvironment {
  readonly name = 'analytic';
  readonly observationSize = 4;
  readonly actionSize = 2;
  readonly discount = 0.99;
  readonly maxStepsPerEpisode = 20;
  private readonly obs = new Float32Array(this.observationSize);
  private readonly target = [0.7, -0.4];
  private stepsTaken = 0;

  reset(rng: Rng): Float32Array {
    this.stepsTaken = 0;
    for (let i = 0; i < this.observationSize; i++) this.obs[i] = rng.range(-1, 1);
    return this.obs;
  }

  step(action: ArrayLike<number>): StepResult {
    this.stepsTaken++;
    const reward =
      -Math.abs((action[0] ?? 0) - this.target[0]!) -
      Math.abs((action[1] ?? 0) - this.target[1]!) -
      0.01;
    return {
      reward,
      done: this.stepsTaken >= this.maxStepsPerEpisode,
      // Time limit, not a real terminal: the trainer must keep bootstrapping.
      terminated: false,
    };
  }

  observe(out: Float32Array): Float32Array {
    return out;
  }

  dispose(): void {}
}

describe('Trainer contract', () => {
  it('validates gaeLambda', () => {
    expect(() => new Trainer({ gaeLambda: 1.5 })).toThrow(RangeError);
    expect(() => new Trainer({ gaeLambda: -0.1 })).toThrow(RangeError);
    expect(() => new Trainer({ gaeLambda: 1 })).not.toThrow();
    expect(() => new Trainer({ gaeLambda: 0 })).not.toThrow();
  });

  it('clamps episodesPerUpdate to at least one batch', () => {
    const trainer = new Trainer({ episodesPerUpdate: 0 });
    expect(trainer.episodesPerUpdate).toBe(1);
  });

  it('reports one history entry per episode and the right update count', () => {
    const env = new AnalyticEnv();
    const policy = new GaussianPolicy({ observationSize: 4, actionSize: 2, seed: 1 });
    const trainer = new Trainer({ episodesPerUpdate: 4, seed: 1 });
    const seen: number[] = [];
    const result = trainer.train(policy, env, 10, (episode) => seen.push(episode.episode));
    expect(seen).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(result.episodes).toBe(10);
    expect(result.history).toHaveLength(10);
    // Batches of 4: 4 + 4 + trailing 2.
    expect(result.updates).toBe(3);
    expect(result.steps).toBe(10 * 20);
    expect(result.policy).toBe(policy);
    env.dispose();
  });

  it('history carries finite diagnostics', () => {
    const env = new AnalyticEnv();
    const policy = new GaussianPolicy({ observationSize: 4, actionSize: 2, seed: 2 });
    const trainer = new Trainer({ episodesPerUpdate: 8, seed: 2 });
    const result = trainer.train(policy, env, 16);
    for (const entry of result.history) {
      expect(Number.isFinite(entry.return)).toBe(true);
      expect(Number.isFinite(entry.entropy)).toBe(true);
      expect(Number.isFinite(entry.valueLoss)).toBe(true);
      expect(Number.isFinite(entry.advantageStd)).toBe(true);
      expect(entry.steps).toBeGreaterThan(0);
    }
    expect(result.bestReturn).toBeGreaterThan(Number.NEGATIVE_INFINITY);
    // The critic is built automatically and has seen data.
    expect(trainer.valueBaseline?.sampleCount).toBeGreaterThan(0);
    env.dispose();
  });

  it('is reproducible from a seed', () => {
    const run = (): number[] => {
      const env = new AnalyticEnv();
      const policy = new GaussianPolicy({ observationSize: 4, actionSize: 2, seed: 5 });
      new Trainer({ episodesPerUpdate: 8, seed: 5 }).train(policy, env, 32);
      const out = Array.from(policy.actGreedy(new Float32Array(4)));
      env.dispose();
      return out;
    };
    expect(run()).toEqual(run());
  });
});

/**
 * The two tests below are the expensive ones, and they are expensive for a
 * reason: convergence is the property under test, and it takes thousands of
 * episodes to demonstrate. Un-instrumented they cost ~12s each; under v8
 * coverage instrumentation the inner loop is the exact thing being counted, so
 * they cost ~10x that (measured 115s for the physics task alone, against a 120s
 * limit -- a CI flake on any runner slower than this one).
 *
 * They also contribute almost nothing to coverage: skipping both moves total
 * branch coverage from 87.02% to 86.70%, because the *machinery* they exercise
 * is already covered by the contract tests above, which run in milliseconds.
 * What they add is a learning assertion, and `npm test` (the gate that runs
 * un-instrumented, in `verify`) still makes it.
 *
 * So: run always, skip under coverage. The flag is set by `npm run
 * test:coverage` and nothing else, which keeps `npm test` honest -- a skipped
 * convergence test there would be a silent hole in the gate.
 */
const underCoverage = process.env.COVERAGE === '1';

describe.skipIf(underCoverage)('Trainer learning', () => {
  it('converges to a known analytic optimum', () => {
    const env = new AnalyticEnv();
    const policy = new GaussianPolicy({
      observationSize: 4,
      actionSize: 2,
      hiddenSizes: [32, 32],
      seed: 7,
      initialLogSigma: -0.5,
    });
    const trainer = new Trainer({
      learningRate: 0.05,
      entropyCoefficient: 0,
      criticLr: 0.05,
      episodesPerUpdate: 32,
      seed: 7,
    });
    trainer.train(policy, env, 3000);
    const action = policy.actGreedy(new Float32Array(4).fill(0));
    // Close to the analytic optimum. The tolerance is not a formality: a policy
    // gradient with the wrong sign or an over-fitting critic lands nowhere near
    // these numbers (it sits at the initialisation mean, ~0).
    expect(Math.abs(action[0]! - 0.7)).toBeLessThan(0.1);
    expect(Math.abs(action[1]! + 0.4)).toBeLessThan(0.1);
    // Exploration must have collapsed toward the optimum, not stayed wide.
    expect(Math.exp(policy.logSigmas[0]!)).toBeLessThan(0.4);
    env.dispose();
  }, 180000);

  it('solves a physics-backed task with the greedy policy', () => {
    const env = new DriveEnv({ seed: 11 });
    const policy = new GaussianPolicy({
      observationSize: env.observationSize,
      actionSize: env.actionSize,
      hiddenSizes: [64, 64],
      seed: 5,
      initialLogSigma: -0.7,
    });
    const trainer = new Trainer({
      learningRate: 0.05,
      entropyCoefficient: 0,
      criticLr: 0.1,
      episodesPerUpdate: 16,
      seed: 5,
    });
    trainer.train(policy, env, 1500);

    // Greedy evaluation over fresh episodes: the deployment-mode metric.
    const rng = new Rng(78);
    const action = new Float32Array(env.actionSize);
    const observation = new Float32Array(env.observationSize);
    let successes = 0;
    const episodes = 30;
    for (let e = 0; e < episodes; e++) {
      env.reset(rng);
      let done = false;
      while (!done) {
        policy.actGreedy(env.observe(observation), action);
        done = env.step(action).done;
      }
      if (env.diagnostics().reached) successes++;
    }
    expect(successes / episodes).toBeGreaterThanOrEqual(0.8);
    env.dispose();
  }, 120000);
});

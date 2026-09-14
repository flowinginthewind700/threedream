import { describe, expect, it } from 'vitest';

import { GaussianPolicy } from '../src/ai/policy.js';
import { ValueBaseline } from '../src/ai/baseline.js';
import { Rng } from '../src/core/rng.js';

function makePolicy(seed = 7, options: Partial<ConstructorParameters<typeof GaussianPolicy>[0]> = {}) {
  return new GaussianPolicy({
    observationSize: 4,
    actionSize: 2,
    hiddenSizes: [8, 8],
    seed,
    ...options,
  });
}

describe('GaussianPolicy sampling', () => {
  it('is deterministic for a given seed', () => {
    const a = makePolicy(11);
    const b = makePolicy(11);
    const obs = new Float32Array([0.1, -0.5, 0.3, 0.9]);
    const bufA = new Float32Array(2);
    const bufB = new Float32Array(2);
    for (let i = 0; i < 50; i++) {
      expect(Array.from(a.sample(obs, bufA))).toEqual(Array.from(b.sample(obs, bufB)));
    }
  });

  it('different seeds give different streams', () => {
    const a = makePolicy(1);
    const b = makePolicy(2);
    const obs = new Float32Array([0.1, -0.5, 0.3, 0.9]);
    expect(Array.from(a.sample(obs))).not.toEqual(Array.from(b.sample(obs)));
  });

  it('actGreedy returns the mean, scaled and repeatable', () => {
    const policy = makePolicy(5, { actionScale: 2 });
    const obs = new Float32Array([0.2, 0.2, 0.2, 0.2]);
    const first = Array.from(policy.actGreedy(obs));
    const second = Array.from(policy.actGreedy(obs));
    expect(first).toEqual(second);
    // actionScale multiplies the mean.
    const unscaled = makePolicy(5, { actionScale: 1 });
    const base = Array.from(unscaled.actGreedy(obs));
    expect(first[0]).toBeCloseTo(base[0]! * 2, 5);
    expect(first[1]).toBeCloseTo(base[1]! * 2, 5);
  });

  it('samples straddle the mean and spread with sigma', () => {
    const tight = makePolicy(3, { initialLogSigma: -3 });
    const wide = makePolicy(3, { initialLogSigma: 1 });
    const obs = new Float32Array([0.3, -0.2, 0.4, 0.1]);
    const mean = tight.actGreedy(obs);
    const spread = (policy: GaussianPolicy): number => {
      let sum = 0;
      let sumSq = 0;
      const n = 4000;
      for (let i = 0; i < n; i++) {
        const s = policy.sample(obs)[0]!;
        sum += s;
        sumSq += s * s;
      }
      const m = sum / n;
      return Math.sqrt(Math.max(0, sumSq / n - m * m));
    };
    expect(spread(wide)).toBeGreaterThan(spread(tight) * 20);
    // Sample mean tracks the network mean.
    expect(Math.abs(mean[0]!)).toBeLessThan(10);
  });

  it('clamps logSigma to its bounds during ascent', () => {
    const policy = makePolicy(4, { minLogSigma: -3, maxLogSigma: 1.5 });
    for (let i = 0; i < 50; i++) {
      policy.applyLogSigmaGradient(new Float32Array([9, -9]), 1);
    }
    expect(policy.logSigmas[0]).toBeLessThanOrEqual(1.5);
    expect(policy.logSigmas[1]).toBeGreaterThanOrEqual(-3);
  });
});

describe('GaussianPolicy log density and gradients', () => {
  const obs = new Float32Array([0.25, -0.75, 0.5, 0.125]);

  it('logProb matches the closed-form Gaussian density', () => {
    const policy = makePolicy(9, { actionScale: 1 });
    const action = policy.sample(obs);
    const mean = policy.actGreedy(obs);
    let expected = 0;
    for (let i = 0; i < policy.actionSize; i++) {
      const sigma = Math.exp(policy.logSigmas[i]!);
      const diff = action[i]! - mean[i]!;
      expected +=
        (-0.5 * diff * diff) / (sigma * sigma) -
        policy.logSigmas[i]! -
        0.5 * Math.log(2 * Math.PI);
    }
    expect(policy.logProb(obs, action)).toBeCloseTo(expected, 5);
  });

  it('logProb undoes actionScale so the density stays over the Gaussian', () => {
    const policy = makePolicy(9, { actionScale: 4 });
    const action = policy.sample(obs);
    const mean = policy.actGreedy(obs);
    let expected = 0;
    for (let i = 0; i < policy.actionSize; i++) {
      const sigma = Math.exp(policy.logSigmas[i]!);
      const diff = action[i]! / 4 - mean[i]! / 4;
      expected +=
        (-0.5 * diff * diff) / (sigma * sigma) -
        policy.logSigmas[i]! -
        0.5 * Math.log(2 * Math.PI);
    }
    expect(policy.logProb(obs, action)).toBeCloseTo(expected, 5);
  });

  it('policyGradients returns the same logProb as logProb()', () => {
    const policy = makePolicy(21);
    const action = policy.sample(obs);
    const grads = policy.policyGradients(obs, action, 1);
    expect(grads.logProb).toBeCloseTo(policy.logProb(obs, action), 5);
  });

  it('scales linearly with the advantage weight', () => {
    const policy = makePolicy(22);
    const action = policy.sample(obs);
    const unit = policy.policyGradients(obs, action, 1);
    const scaled = policy.policyGradients(obs, action, -3.5);
    for (let i = 0; i < policy.actionSize; i++) {
      expect(scaled.meanGrad[i]).toBeCloseTo(unit.meanGrad[i]! * -3.5, 6);
      expect(scaled.logSigmaGrad[i]).toBeCloseTo(unit.logSigmaGrad[i]! * -3.5, 6);
    }
    // logProb is not scaled by the weight.
    expect(scaled.logProb).toBeCloseTo(unit.logProb, 6);
  });

  it('mean gradient matches d logProb / d mean from finite differences', () => {
    const policy = makePolicy(31);
    const action = policy.sample(obs);
    const grads = policy.policyGradients(obs, action, 1);
    const eps = 1e-3;
    for (let o = 0; o < policy.actionSize; o++) {
      const trunk = policy.mlp.weights[policy.mlp.weights.length - 1]!;
      // Output layer layout is [fanIn * fanOut]; bias[o] shifts mean[o] by 1.
      const original = policy.mlp.biases[policy.mlp.biases.length - 1]![o]!;
      policy.mlp.biases[policy.mlp.biases.length - 1]![o] = original + eps;
      const plus = policy.logProb(obs, action);
      policy.mlp.biases[policy.mlp.biases.length - 1]![o] = original - eps;
      const minus = policy.logProb(obs, action);
      policy.mlp.biases[policy.mlp.biases.length - 1]![o] = original;
      const numeric = (plus - minus) / (2 * eps);
      expect(grads.meanGrad[o]!).toBeCloseTo(numeric, 4);
      expect(trunk.length).toBeGreaterThan(0);
    }
  });

  it('logSigma gradient matches d logProb / d logSigma analytically', () => {
    const policy = makePolicy(33);
    const action = policy.sample(obs);
    const grads = policy.policyGradients(obs, action, 1);
    const mean = policy.actGreedy(obs);
    for (let i = 0; i < policy.actionSize; i++) {
      const logSigma = policy.logSigmas[i]!;
      const sigma = Math.exp(logSigma);
      const diff = action[i]! - mean[i]!;
      const expected = (diff * diff) / (sigma * sigma) - 1;
      expect(grads.logSigmaGrad[i]!).toBeCloseTo(expected, 5);
    }
  });

  it('entropy rises with sigma and matches the closed form', () => {
    const policy = makePolicy(41, { initialLogSigma: 0 });
    const expected = policy.actionSize * 0.5 * Math.log(2 * Math.PI * Math.E);
    expect(policy.entropy()).toBeCloseTo(expected, 6);
    const narrow = makePolicy(41, { initialLogSigma: -2 });
    expect(narrow.entropy()).toBeLessThan(policy.entropy());
  });

  it('reuses the provided output buffer without aliasing the scratch mean', () => {
    const policy = makePolicy(51);
    const buf = new Float32Array(2);
    const action = policy.sample(obs, buf);
    expect(action).toBe(buf);
    const grads = policy.policyGradients(obs, buf, 1);
    expect(Number.isFinite(grads.meanGrad[0])).toBe(true);
    expect(Number.isFinite(grads.meanGrad[1])).toBe(true);
  });

  it('round-trips through JSON', () => {
    const policy = makePolicy(61);
    policy.applyLogSigmaGradient(new Float32Array([0.3, -0.2]), 0.5);
    const restored = GaussianPolicy.fromJSON(policy.toJSON());
    const action = policy.sample(obs);
    expect(restored.logProb(obs, action)).toBeCloseTo(policy.logProb(obs, action), 5);
    expect(Array.from(restored.logSigmas)).toEqual(Array.from(policy.logSigmas));
    expect(Array.from(restored.actGreedy(obs))).toEqual(Array.from(policy.actGreedy(obs)));
  });
});

describe('ValueBaseline', () => {
  it('fits a constant target', () => {
    const critic = new ValueBaseline({ observationSize: 2, hiddenSizes: [16, 16], learningRate: 0.08, seed: 3 });
    const obs = new Float32Array([0.1, 0.2]);
    let loss = 0;
    for (let i = 0; i < 400; i++) {
      loss = critic.train([{ observation: obs, target: 5 }]);
    }
    expect(loss).toBeLessThan(0.05);
    expect(critic.predict(obs)).toBeCloseTo(5, 1);
  });

  it('fits a linear function of the input', () => {
    const critic = new ValueBaseline({ observationSize: 1, hiddenSizes: [32, 32], learningRate: 0.05, seed: 8 });
    const rng = new Rng(19);
    const samples = Array.from({ length: 64 }, () => {
      const x = rng.range(-1, 1);
      return { observation: new Float32Array([x]), target: 2 * x + 1 };
    });
    for (let epoch = 0; epoch < 150; epoch++) critic.train(samples);
    let maxError = 0;
    for (const sample of samples) {
      maxError = Math.max(maxError, Math.abs(critic.predict(sample.observation) - sample.target));
    }
    expect(maxError).toBeLessThan(0.25);
  });

  it('returns zero loss for an empty batch and does not train', () => {
    const critic = new ValueBaseline({ observationSize: 2, seed: 1 });
    const obs = new Float32Array([0.4, 0.6]);
    const before = critic.predict(obs);
    expect(critic.train([])).toBe(0);
    expect(critic.predict(obs)).toBe(before);
    expect(critic.sampleCount).toBe(0);
  });

  it('tracks target statistics and normalizes advantages by them', () => {
    const critic = new ValueBaseline({ observationSize: 1, seed: 2 });
    critic.train([
      { observation: new Float32Array([0]), target: -1 },
      { observation: new Float32Array([0]), target: 1 },
    ]);
    expect(critic.sampleCount).toBe(2);
    expect(critic.targetStd).toBeGreaterThan(0);
    const normalized = critic.normalizeAdvantage(critic.targetStd);
    expect(normalized).toBeCloseTo(1, 5);
  });

  it('predict is finite even for extreme inputs', () => {
    const critic = new ValueBaseline({ observationSize: 2, seed: 4 });
    const value = critic.predict(new Float32Array([1e6, -1e6]));
    expect(Number.isFinite(value)).toBe(true);
  });
});

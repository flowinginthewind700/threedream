import { describe, expect, it } from 'vitest';

import { Mlp } from '../src/ai/mlp.js';
import { Rng } from '../src/core/rng.js';

function makeNet(sizes = [4, 8, 3], seed = 7): Mlp {
  return new Mlp({ sizes }, new Rng(seed));
}

describe('Mlp structure', () => {
  it('reports sizes and parameter count', () => {
    const net = makeNet([2, 5, 3]);
    expect(net.inputSize).toBe(2);
    expect(net.outputSize).toBe(3);
    expect(net.parameterCount).toBe(2 * 5 + 5 + 5 * 3 + 3);
  });

  it('rejects degenerate specs', () => {
    expect(() => new Mlp({ sizes: [3] })).toThrow(RangeError);
    expect(() => new Mlp({ sizes: [3, 0, 2] })).toThrow(RangeError);
    expect(() => new Mlp({ sizes: [3, 1.5, 2] })).toThrow(RangeError);
  });

  it('rejects a wrong-sized input', () => {
    const net = makeNet([4, 8, 3]);
    expect(() => net.forward(new Float32Array(5))).toThrow(/input size/);
  });

  it('initializes deterministically from a seed', () => {
    const a = makeNet([4, 8, 3], 42);
    const b = makeNet([4, 8, 3], 42);
    const input = new Float32Array([0.5, -0.3, 1.2, 0.1]);
    expect(Array.from(a.forward(input))).toEqual(Array.from(b.forward(input)));
  });
});

describe('Mlp forward', () => {
  it('computes the affine output for a single linear layer', () => {
    const net = new Mlp({ sizes: [2, 2] }, new Rng(1));
    // Overwrite with known weights: out0 = x0 - 2*x1, out1 = 3*x0 + x1.
    // Layout is [fanIn * fanOut], column = output unit.
    net.weights[0]!.set([1, 3, -2, 1]);
    net.biases[0]!.set([0.5, -0.25]);
    const out = net.forward([2, 1]);
    expect(out[0]).toBeCloseTo(2 - 2 + 0.5, 6);
    expect(out[1]).toBeCloseTo(6 + 1 - 0.25, 6);
  });

  it('applies ReLU on hidden layers and identity on the output', () => {
    const net = new Mlp({ sizes: [1, 2, 1] }, new Rng(1));
    // Hidden: one unit passes, one is clamped to zero by ReLU.
    net.weights[0]!.set([1, -1]);
    net.biases[0]!.set([0, 0]);
    net.weights[1]!.set([2, 5]);
    net.biases[1]!.set([0]);
    const out = net.forward([3]);
    // ReLU(3)*2 + ReLU(-3)*5 = 6 + 0
    expect(out[0]).toBeCloseTo(6, 6);
  });

  it('reuses the caller-provided output buffer', () => {
    const net = makeNet([4, 8, 3]);
    const buf = new Float32Array(3);
    const input = new Float32Array([1, 2, 3, 4]);
    const fresh = net.forward(input);
    const written = net.forward(input, buf);
    expect(written).toBe(buf);
    expect(Array.from(buf)).toEqual(Array.from(fresh));
  });
});

describe('Mlp backward (numeric gradient check)', () => {
  // Loss = sum of outputs; dL/dy = 1 for every output unit.
  const ones = (n: number) => new Float32Array(n).fill(1);

  function numericGrad(
    net: Mlp,
    input: Float32Array,
    paramIndex: number,
    layer: number,
    isBias: boolean,
    eps = 1e-3,
  ): number {
    const target = isBias ? net.biases[layer]! : net.weights[layer]!;
    const original = target[paramIndex]!;
    target[paramIndex] = original + eps;
    const plus = Array.from(net.forward(input)).reduce((s, v) => s + v, 0);
    target[paramIndex] = original - eps;
    const minus = Array.from(net.forward(input)).reduce((s, v) => s + v, 0);
    target[paramIndex] = original;
    return (plus - minus) / (2 * eps);
  }

  it('matches finite differences on every weight and bias', () => {
    const net = makeNet([3, 5, 2], 13);
    const input = new Float32Array([0.4, -0.8, 1.1]);
    net.zeroGradients();
    net.forward(input);
    net.backward(ones(2));

    let checked = 0;
    for (let l = 0; l < net.weights.length; l++) {
      for (let i = 0; i < net.weights[l]!.length; i++) {
        const analytic = net.weightGrads[l]![i]!;
        const numeric = numericGrad(net, input, i, l, false);
        expect(Math.abs(analytic - numeric)).toBeLessThan(2e-2);
        checked++;
      }
      for (let i = 0; i < net.biases[l]!.length; i++) {
        const analytic = net.biasGrads[l]![i]!;
        const numeric = numericGrad(net, input, i, l, true);
        expect(Math.abs(analytic - numeric)).toBeLessThan(2e-2);
        checked++;
      }
    }
    expect(checked).toBe(net.parameterCount);
  });

  it('accumulates gradients across samples until zeroed', () => {
    const net = makeNet([2, 4, 1], 3);
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    net.zeroGradients();
    net.forward(a);
    net.backward(ones(1));
    const afterFirst = net.weightGrads[0]![0]!;
    net.forward(b);
    net.backward(ones(1));
    const afterSecond = net.weightGrads[0]![0]!;
    expect(afterSecond).toBeCloseTo(afterFirst, 6); // x0=0 contributes nothing
    // Bias gradient adds 1 per sample.
    expect(net.biasGrads[1]![0]!).toBeCloseTo(2, 6);
  });

  it('applyGradients descends and clips', () => {
    const net = new Mlp({ sizes: [1, 1] }, new Rng(1));
    net.weights[0]!.set([1]);
    net.biases[0]!.set([0]);
    net.zeroGradients();
    net.weightGrads[0]!.set([1000]); // huge, should clip to 5
    net.applyGradients(0.1, 5);
    expect(net.weights[0]![0]).toBeCloseTo(1 - 0.1 * 5, 6);
  });

  it('rejects a wrong-sized output gradient', () => {
    const net = makeNet([4, 8, 3]);
    net.forward(new Float32Array(4));
    expect(() => net.backward(new Float32Array(2))).toThrow(/output grad size/);
  });
});

describe('Mlp serialisation', () => {
  it('round-trips through JSON', () => {
    const net = makeNet([3, 6, 2], 99);
    const input = new Float32Array([0.2, 0.9, -0.4]);
    const expected = Array.from(net.forward(input));
    const restored = Mlp.fromJSON(net.toJSON());
    expect(restored.sizes).toEqual(net.sizes);
    expect(Array.from(restored.forward(input))).toEqual(expected);
  });

  it('getParameters/setParameters round-trip', () => {
    const net = makeNet([4, 8, 3], 5);
    const flat = net.getParameters();
    expect(flat.length).toBe(net.parameterCount);
    const other = makeNet([4, 8, 3], 77);
    other.setParameters(flat);
    const input = new Float32Array([1, -1, 0.5, 0.25]);
    expect(Array.from(other.forward(input))).toEqual(Array.from(net.forward(input)));
  });

  it('setParameters rejects a wrong-length vector', () => {
    const net = makeNet([4, 8, 3]);
    expect(() => net.setParameters(new Float32Array(3))).toThrow(RangeError);
  });
});

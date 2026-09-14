/**
 * Multi-layer perceptron with hand-written backpropagation.
 *
 * No autograd dependency: weights live in flat `Float32Array`s, forward and
 * backward are explicit, and the whole network serialises to a plain object.
 * That keeps it small enough to run in a Worker, fast enough for headless
 * training loops, and inspectable, which matters when a policy misbehaves and
 * you need to dump the exact weights that produced it.
 *
 * Activations: ReLU on hidden layers, identity on the output (policies apply
 * their own squashing / Gaussian head on top).
 */

import { Rng } from '../core/rng.js';

export interface MlpSpec {
  /** Input dimension, hidden layer sizes, output dimension. */
  sizes: number[];
}

export interface MlpSnapshot {
  sizes: number[];
  /** Row-major weights per layer, length = fanIn * fanOut. */
  weights: number[][];
  biases: number[][];
}

export class Mlp {
  readonly sizes: number[];
  readonly weights: Float32Array[];
  readonly biases: Float32Array[];

  // Forward cache for backward.
  private readonly activations: Float32Array[];
  private readonly preActivations: Float32Array[];

  // Gradient accumulators.
  readonly weightGrads: Float32Array[];
  readonly biasGrads: Float32Array[];

  private lastInput: Float32Array;

  constructor(spec: MlpSpec, rng?: Rng) {
    if (spec.sizes.length < 2) {
      throw new RangeError('Mlp needs at least an input and an output size');
    }
    for (const size of spec.sizes) {
      if (!Number.isInteger(size) || size <= 0) {
        throw new RangeError(`invalid layer size: ${size}`);
      }
    }
    this.sizes = [...spec.sizes];
    const seed = rng ?? new Rng(0x51ed);
    this.weights = [];
    this.biases = [];
    this.weightGrads = [];
    this.biasGrads = [];
    this.activations = [];
    this.preActivations = [];
    for (let l = 0; l < this.sizes.length - 1; l++) {
      const fanIn = this.sizes[l]!;
      const fanOut = this.sizes[l + 1]!;
      // He initialisation for ReLU.
      const limit = Math.sqrt(6 / fanIn);
      const w = new Float32Array(fanIn * fanOut);
      seed.uniformInto(w, limit);
      this.weights.push(w);
      this.biases.push(new Float32Array(fanOut));
      this.weightGrads.push(new Float32Array(fanIn * fanOut));
      this.biasGrads.push(new Float32Array(fanOut));
    }
    this.lastInput = new Float32Array(this.sizes[0]!);
    // Reserve cache buffers sized to the widest layer.
    for (let l = 0; l < this.sizes.length; l++) {
      this.activations.push(new Float32Array(this.sizes[l]!));
      this.preActivations.push(new Float32Array(this.sizes[l]!));
    }
  }

  get inputSize(): number {
    return this.sizes[0]!;
  }

  get outputSize(): number {
    return this.sizes[this.sizes.length - 1]!;
  }

  get parameterCount(): number {
    let n = 0;
    for (const w of this.weights) n += w.length;
    for (const b of this.biases) n += b.length;
    return n;
  }

  /** Forward pass; caches internals for a following `backward`. */
  forward(input: ArrayLike<number>, out?: Float32Array): Float32Array {
    if (input.length !== this.inputSize) {
      throw new RangeError(
        `input size ${input.length} != expected ${this.inputSize}`,
      );
    }
    // Layer 0 input.
    const a0 = this.activations[0]!;
    for (let i = 0; i < this.inputSize; i++) {
      const v = input[i]!;
      a0[i] = v;
      this.lastInput[i] = v;
    }

    let current = a0;
    for (let l = 0; l < this.weights.length; l++) {
      const fanIn = this.sizes[l]!;
      const fanOut = this.sizes[l + 1]!;
      const w = this.weights[l]!;
      const b = this.biases[l]!;
      const z = this.preActivations[l + 1]!;
      for (let o = 0; o < fanOut; o++) {
        let sum = b[o]!;
        // Column-major walk matches weight layout [fanIn * fanOut], row = fanIn.
        for (let i = 0; i < fanIn; i++) {
          sum += current[i]! * w[i * fanOut + o]!;
        }
        z[o] = sum;
      }
      const a = this.activations[l + 1]!;
      const isOutput = l === this.weights.length - 1;
      for (let o = 0; o < fanOut; o++) {
        a[o] = isOutput ? z[o]! : Math.max(0, z[o]!);
      }
      current = a;
    }

    const result = out ?? new Float32Array(this.outputSize);
    for (let o = 0; o < this.outputSize; o++) result[o] = current[o]!;
    return result;
  }

  zeroGradients(): void {
    for (const g of this.weightGrads) g.fill(0);
    for (const g of this.biasGrads) g.fill(0);
  }

  /**
   * Backpropagate an output-gradient (dL/dy) computed by the caller's loss.
   * Gradients accumulate, so call `zeroGradients()` per batch, not per sample.
   */
  backward(outputGrad: ArrayLike<number>): void {
    if (outputGrad.length !== this.outputSize) {
      throw new RangeError(
        `output grad size ${outputGrad.length} != ${this.outputSize}`,
      );
    }
    let delta = new Float32Array(this.outputSize);
    for (let o = 0; o < this.outputSize; o++) delta[o] = outputGrad[o]!;

    for (let l = this.weights.length - 1; l >= 0; l--) {
      const fanIn = this.sizes[l]!;
      const fanOut = this.sizes[l + 1]!;
      const aIn = this.activations[l]!;
      const w = this.weights[l]!;
      const wg = this.weightGrads[l]!;
      const bg = this.biasGrads[l]!;

      // Accumulate weight and bias gradients.
      for (let o = 0; o < fanOut; o++) {
        const d = delta[o]!;
        bg[o] += d;
        for (let i = 0; i < fanIn; i++) {
          wg[i * fanOut + o] += aIn[i]! * d;
        }
      }

      // Propagate to previous layer's activations, applying ReLU' where needed.
      if (l > 0) {
        const nextDelta = new Float32Array(fanIn);
        const zPrev = this.preActivations[l]!;
        for (let i = 0; i < fanIn; i++) {
          let sum = 0;
          for (let o = 0; o < fanOut; o++) {
            sum += w[i * fanOut + o]! * delta[o]!;
          }
          nextDelta[i] = zPrev[i]! > 0 ? sum : 0;
        }
        delta = nextDelta;
      }
    }
  }

  /** In-place SGD update; `clip` bounds each gradient component. */
  applyGradients(learningRate: number, clip = 5): void {
    for (let l = 0; l < this.weights.length; l++) {
      const w = this.weights[l]!;
      const wg = this.weightGrads[l]!;
      const b = this.biases[l]!;
      const bg = this.biasGrads[l]!;
      for (let i = 0; i < w.length; i++) {
        const g = Math.max(-clip, Math.min(clip, wg[i]!));
        w[i] -= learningRate * g;
      }
      for (let i = 0; i < b.length; i++) {
        const g = Math.max(-clip, Math.min(clip, bg[i]!));
        b[i] -= learningRate * g;
      }
    }
  }

  /** Flatten all parameters into one vector (policy-gradient style updates). */
  getParameters(out?: Float32Array): Float32Array {
    const result = out ?? new Float32Array(this.parameterCount);
    let offset = 0;
    for (const w of this.weights) {
      result.set(w, offset);
      offset += w.length;
    }
    for (const b of this.biases) {
      result.set(b, offset);
      offset += b.length;
    }
    return result;
  }

  setParameters(flat: ArrayLike<number>): void {
    if (flat.length !== this.parameterCount) {
      throw new RangeError(
        `parameter vector length ${flat.length} != ${this.parameterCount}`,
      );
    }
    let offset = 0;
    for (const w of this.weights) {
      for (let i = 0; i < w.length; i++) w[i] = flat[offset + i]!;
      offset += w.length;
    }
    for (const b of this.biases) {
      for (let i = 0; i < b.length; i++) b[i] = flat[offset + i]!;
      offset += b.length;
    }
  }

  toJSON(): MlpSnapshot {
    return {
      sizes: [...this.sizes],
      weights: this.weights.map((w) => Array.from(w)),
      biases: this.biases.map((b) => Array.from(b)),
    };
  }

  static fromJSON(snapshot: MlpSnapshot): Mlp {
    const net = new Mlp({ sizes: snapshot.sizes });
    for (let l = 0; l < net.weights.length; l++) {
      const w = snapshot.weights[l]!;
      const b = snapshot.biases[l]!;
      for (let i = 0; i < net.weights[l]!.length; i++) net.weights[l]![i] = w[i]!;
      for (let i = 0; i < net.biases[l]!.length; i++) net.biases[l]![i] = b[i]!;
    }
    return net;
  }
}

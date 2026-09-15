/**
 * `gpu/softColoring.ts` -- greedy edge coloring, and the batches it implies.
 *
 * The claim under test is the one the GPU backend's `raceFree: true` rests on: no
 * two edges that share a node are ever dispatched together. That is asserted
 * structurally over every scene rather than by example, because a coloring that is
 * wrong on one edge produces a simulation that differs between runs by an amount
 * that looks like solver noise and is not.
 *
 * The second thing pinned here is the color count, because it is the dispatch cost
 * of a step: `iterations * colors` dispatches, each with a barrier. A greedy pass
 * that quietly used 20 colors on a degree-8 cloth would be correct and slow, and
 * nothing else in the suite would notice.
 */

import { describe, expect, it } from 'vitest';
import {
  MAX_COLORS,
  colorConstraints,
  coloringIsRaceFree,
  coloringWorkgroups,
  type SoftColoring,
} from '../src/gpu/softColoring.js';
import { SOFT_STRIDE, SoftMesh, emptyConstraints, type SoftScene } from '../src/gpu/softMesh.js';
import { softWorkgroups } from '../src/gpu/softIslands.js';

/** A hand-built graph, so the expected colors can be written down. */
function graph(count: number, pairs: readonly number[]): SoftMesh {
  const data = new Float32Array(count * SOFT_STRIDE);
  for (let i = 0; i < count; i++) {
    data[i * SOFT_STRIDE] = i;
    data[i * SOFT_STRIDE + 3] = 1;
    data[i * SOFT_STRIDE + 7] = 0.1;
  }
  const n = pairs.length / 2;
  return new SoftMesh(data, {
    count,
    constraints: {
      count: n,
      ends: new Uint32Array(pairs),
      rest: new Float32Array(n).fill(1),
      stiffness: new Float32Array(n).fill(1),
    },
  });
}

/** A star: one hub with `spokes` edges, i.e. a node of degree `spokes`. */
function star(spokes: number, hub = 0): SoftMesh {
  const pairs: number[] = [];
  for (let k = 0; k < spokes; k++) pairs.push(hub, spokes + 1 + k);
  return graph(2 * spokes + 2, pairs);
}

describe('greedy first-fit', () => {
  it('gives a path two colors and alternates along it', () => {
    const mesh = graph(6, [0, 1, 1, 2, 3, 4]);
    const coloring = colorConstraints(mesh);
    // Edges 0-1 and 1-2 share node 1, so they differ; 3-4 shares nothing with
    // either and takes the lowest free color again.
    expect(coloring.colors).toBe(2);
    expect(Array.from(coloring.colorOfConstraint)).toEqual([0, 1, 0]);
    expect(coloring.maxNodeDegree).toBe(2);
    expect(coloring.constraints).toBe(3);
    // Batch 0 holds constraints 0 and 2, batch 1 holds constraint 1: `order` maps
    // batch position back to constraint index, ascending within a color.
    expect(Array.from(coloring.order)).toEqual([0, 2, 1]);
    expect(coloring.batches).toEqual([
      { color: 0, base: 0, count: 2, workgroups: 1 },
      { color: 1, base: 2, count: 1, workgroups: 1 },
    ]);
  });

  it('needs three colors for a triangle, and no more', () => {
    const mesh = graph(3, [0, 1, 1, 2, 0, 2]);
    const coloring = colorConstraints(mesh);
    expect(coloring.maxNodeDegree).toBe(2);
    // Vizing bounds a graph at maxDegree + 1; a triangle is the case that reaches
    // it, and first-fit does so here because every edge meets the other two.
    expect(coloring.colors).toBe(3);
    expect(Array.from(coloring.colorOfConstraint)).toEqual([0, 1, 2]);
    expect(coloringIsRaceFree(coloring, mesh)).toBe(true);
  });

  it('uses close to the fewest colors a greedy pass can reach', () => {
    // The guarantee first-fit actually offers is 2 * maxDegree - 1; Vizing's
    // maxDegree + 1 needs a smarter algorithm than this one runs. In practice the
    // scenes here land on maxDegree, and the two large bent cloths pay one extra
    // color, so the checked bound is maxDegree + 1 and the exact counts are pinned
    // by the running sum. A regression that colors lazily moves the sum, not the
    // bound, which is why both are asserted.
    let sum = 0;
    let checked = 0;
    for (const scene of ['cloth', 'sheets', 'cube', 'rope'] as SoftScene[]) {
      for (const count of [7, 64, 100, 512, 1000]) {
        const mesh = new SoftMesh({ count, scene, seed: 5, groups: 4, bend: true });
        const coloring = colorConstraints(mesh);
        expect(coloring.colors, `${scene}@${count}`).toBeLessThanOrEqual(coloring.maxNodeDegree + 1);
        if (coloring.constraints > 0) {
          expect(coloring.colors, `${scene}@${count}`).toBeGreaterThanOrEqual(1);
        }
        sum += coloring.colors;
        checked += 1;
      }
    }
    expect(checked).toBe(20);
    expect(sum).toBe(141);
  });

  it('colors by ascending constraint index, so the result is a function of the graph', () => {
    const a = colorConstraints(new SoftMesh({ count: 1000, seed: 7 }));
    const b = colorConstraints(new SoftMesh({ count: 1000, seed: 7 }));
    expect(Array.from(a.colorOfConstraint)).toEqual(Array.from(b.colorOfConstraint));
    expect(Array.from(a.order)).toEqual(Array.from(b.order));
    // Reordering the edges legitimately changes the colors: first-fit is
    // order-dependent by definition. What must not change is the guarantee, so the
    // reversed graph is still race-free and still within the greedy color bound.
    const forward = graph(6, [0, 1, 1, 2, 3, 4]);
    const backward = graph(6, [3, 4, 1, 2, 0, 1]);
    expect(Array.from(colorConstraints(backward).colorOfConstraint)).not.toEqual(
      Array.from(colorConstraints(forward).colorOfConstraint),
    );
    expect(coloringIsRaceFree(colorConstraints(backward), backward)).toBe(true);
    const backwardColored = colorConstraints(backward);
    expect(backwardColored.colors).toBeLessThanOrEqual(
      2 * backwardColored.maxNodeDegree - 1,
    );
  });

  it('handles a graph with no edges', () => {
    const mesh = new SoftMesh(new Float32Array(8 * SOFT_STRIDE), {
      count: 8,
      constraints: emptyConstraints(),
    });
    const coloring = colorConstraints(mesh);
    expect(coloring.colors).toBe(0);
    expect(coloring.batches).toEqual([]);
    expect(coloring.order.length).toBe(0);
    expect(coloring.maxNodeDegree).toBe(0);
    expect(coloring.maxBatchSize).toBe(0);
    expect(coloring.dispatchesPerIteration).toBe(0);
    expect(coloringWorkgroups(coloring)).toBe(0);
    expect(coloringIsRaceFree(coloring, mesh)).toBe(true);
  });
});

describe('the batch table', () => {
  it('is contiguous, sums to the edge count, and sizes each dispatch at 64', () => {
    for (const scene of ['cloth', 'sheets', 'cube', 'rope'] as SoftScene[]) {
      for (const count of [7, 64, 100, 512, 1000]) {
        const mesh = new SoftMesh({ count, scene, seed: 5, groups: 4, bend: true });
        const coloring = colorConstraints(mesh);
        const label = `${scene}@${count}`;
        expect(coloring.batches.length, label).toBe(coloring.colors);
        let base = 0;
        for (const [i, batch] of coloring.batches.entries()) {
          expect(batch.color, label).toBe(i);
          expect(batch.base, label).toBe(base);
          expect(batch.workgroups, label).toBe(softWorkgroups(batch.count));
          expect(batch.count, label).toBeGreaterThan(0);
          base += batch.count;
        }
        expect(base, label).toBe(coloring.constraints);
        expect(coloring.dispatchesPerIteration, label).toBe(coloring.colors);
      }
    }
  });

  it('makes order a permutation of every constraint, ascending within a batch', () => {
    const mesh = new SoftMesh({ count: 1000, seed: 3, scene: 'cube' });
    const coloring = colorConstraints(mesh);
    const seen = new Uint32Array(coloring.constraints);
    for (const k of coloring.order) seen[k]++;
    expect(Array.from(seen).every((n) => n === 1)).toBe(true);
    for (const batch of coloring.batches) {
      const slice = coloring.order.subarray(batch.base, batch.base + batch.count);
      expect(slice.length).toBe(batch.count);
      expect(Array.from(slice)).toEqual(Array.from(slice).sort((a, b) => a - b));
      for (const k of slice) expect(coloring.colorOfConstraint[k]).toBe(batch.color);
    }
    expect(coloring.maxBatchSize).toBe(Math.max(...coloring.batches.map((b) => b.count)));
    expect(coloringWorkgroups(coloring)).toBe(
      coloring.batches.reduce((sum, b) => sum + b.workgroups, 0),
    );
  });

  it('records the golden color counts of the four scenes', () => {
    // Dispatches per step are iterations * colors, so these are the step cost.
    // Recorded because a change to a builder that raises the degree raises them
    // silently, and the demo would get slower with nothing failing.
    expect(colorConstraints(new SoftMesh({ count: 100, seed: 7 })).colors).toBe(8);
    expect(colorConstraints(new SoftMesh({ count: 100, seed: 7, shear: false })).colors).toBe(4);
    expect(colorConstraints(new SoftMesh({ count: 100, seed: 7, bend: true })).colors).toBe(12);
    expect(colorConstraints(new SoftMesh({ count: 1000, seed: 3, scene: 'cube' })).colors).toBe(6);
    expect(colorConstraints(new SoftMesh({ count: 50, seed: 2, scene: 'rope' })).colors).toBe(2);
    expect(
      colorConstraints(new SoftMesh({ count: 100, seed: 1, scene: 'sheets', groups: 3 })).colors,
    ).toBe(8);
  });

  it('splits a 100-node cloth into the batches the solver will dispatch', () => {
    const coloring = colorConstraints(new SoftMesh({ count: 100, seed: 7 }));
    expect(coloring.batches.map((b) => b.count)).toEqual([48, 48, 48, 47, 44, 36, 35, 36]);
    expect(coloring.maxNodeDegree).toBe(8);
    expect(coloring.maxBatchSize).toBe(48);
    // Every batch of 48 is still one workgroup: at this size the dispatch is
    // launch-bound rather than thread-bound, which is why the color count and not
    // the batch size is what a step costs.
    expect(coloringWorkgroups(coloring)).toBe(8);
  });
});

describe('the race-free guarantee', () => {
  it('holds for every scene at every count tried', () => {
    for (const scene of ['cloth', 'sheets', 'cube', 'rope'] as SoftScene[]) {
      for (const count of [1, 2, 3, 7, 17, 64, 101, 1000]) {
        const mesh = new SoftMesh({ count, scene, seed: 3, groups: 5, bend: count > 8 });
        const coloring = colorConstraints(mesh);
        expect(coloringIsRaceFree(coloring, mesh), `${scene}@${count}`).toBe(true);
      }
    }
  });

  it('holds at 10k nodes, where the dispatch count is the point', () => {
    const mesh = new SoftMesh({ count: 10_000, seed: 1 });
    const coloring = colorConstraints(mesh);
    expect(coloring.constraints).toBe(39_402);
    expect(coloring.colors).toBe(8);
    expect(coloring.batches.map((b) => b.count)).toEqual([
      4975, 4975, 4976, 4974, 4950, 4859, 4847, 4846,
    ]);
    expect(coloringWorkgroups(coloring)).toBe(618);
    expect(coloringIsRaceFree(coloring, mesh)).toBe(true);
    // Eight colors at three iterations is 24 solve dispatches a step, plus the
    // four fixed kernels: the number the HUD shows and the budget accounts for.
    expect(coloring.dispatchesPerIteration * 3 + 5).toBe(29);
  });

  it('detects a coloring that is wrong, so the validator is not decoration', () => {
    const mesh = graph(6, [0, 1, 1, 2, 3, 4]);
    const good = colorConstraints(mesh);
    expect(coloringIsRaceFree(good, mesh)).toBe(true);
    // Everything in one batch: edges 0-1 and 1-2 both write node 1, which is the
    // exact race the coloring exists to prevent.
    const broken: SoftColoring = {
      ...good,
      colors: 1,
      colorOfConstraint: new Uint32Array(3),
      order: new Uint32Array([0, 1, 2]),
      batches: [{ color: 0, base: 0, count: 3, workgroups: 1 }],
    };
    expect(coloringIsRaceFree(broken, mesh)).toBe(false);
    // Same edges, split so the shared node lands in different batches.
    const fixed: SoftColoring = {
      ...good,
      colors: 2,
      colorOfConstraint: new Uint32Array([0, 1, 0]),
      order: new Uint32Array([0, 2, 1]),
      batches: [
        { color: 0, base: 0, count: 2, workgroups: 1 },
        { color: 1, base: 2, count: 1, workgroups: 1 },
      ],
    };
    expect(coloringIsRaceFree(fixed, mesh)).toBe(true);
  });

  it('is per-batch, not global: a node may appear in many colors', () => {
    // The hub of a star is in every one of its 32 batches. A validator that checked
    // globally would reject a correct coloring, so this pins the scope.
    const mesh = star(32);
    const coloring = colorConstraints(mesh);
    expect(coloring.colors).toBe(32);
    expect(coloring.maxNodeDegree).toBe(32);
    expect(coloring.batches.every((b) => b.count === 1)).toBe(true);
    expect(coloringIsRaceFree(coloring, mesh)).toBe(true);
  });
});

describe('the 32-color ceiling', () => {
  it('is one word per node, and is refused rather than silently exceeded', () => {
    expect(MAX_COLORS).toBe(32);
    // A 33-spoke star has a node of degree 33. Mis-coloring it would put two edges
    // at the hub in one batch, which is a race, so the pass throws instead.
    expect(() => colorConstraints(star(33))).toThrow(
      /node 0 has degree 33, which needs more than the 32 colors a u32 mask holds/,
    );
    expect(() => colorConstraints(star(32))).not.toThrow();
  });

  it('names the offending node, because the fix is in the mesh, not the solver', () => {
    // Hub at node 4 rather than 0: the message has to say which node is too busy,
    // or a caller hitting this has to go and find it in a 45-node graph.
    const mesh = star(33, 4);
    expect(() => colorConstraints(mesh)).toThrow(/node 4 has degree 33, which needs more than the 32/);
  });
});

/**
 * Greedy edge coloring of the constraint graph, and the dispatch batches it
 * implies.
 *
 * # The race this removes
 *
 * A distance-constraint solve reads both endpoints and writes both endpoints. Two
 * edges that share a node, solved in the same dispatch, therefore write that node
 * concurrently -- and the result depends on which thread's store landed last. That
 * is not a small error: it is a different simulation on every run, on every
 * driver, and it is the reason a naive parallel PBD solver cannot be compared to
 * a reference at all.
 *
 * Coloring the graph so that no two edges of the same color share a node makes
 * every write inside one batch land on a distinct node. Batches then run in
 * sequence, one dispatch each, and the whole solve is race-free without a single
 * atomic. This is the M4 ask "约束图着色分批，避免数据竞争", and it is what lets
 * the GPU backend claim `raceFree: true` -- a stronger claim than the particle
 * layer can make, and the reason `deterministic: false` there does not have to
 * mean "different every run" here.
 *
 * # Why greedy first-fit
 *
 * Optimal edge coloring is NP-hard and irrelevant here. What matters is that the
 * color count stays small, because it is the number of dispatches per iteration.
 * First-fit in edge order guarantees `2 * maxDegree - 1` colors; reaching Vizing's
 * `maxDegree + 1` takes an algorithm with recoloring steps this does not run. In
 * practice the scenes here land on `maxDegree` exactly, and a cloth carrying bending
 * constraints pays one color over it. The shipped presets top out at degree 8 (cloth
 * with shear) or 12 (cloth with bending), so the 32-bit mask never comes close to
 * filling, and a graph that would exceed it is refused rather than mis-colored. One
 * loop over the edges with no allocation: that is what a scene change in the demo
 * can afford to pay.
 *
 * # Order stability
 *
 * `order` is the permutation from batch position to constraint index, and within a
 * color the constraints keep their original ascending index. That matters beyond
 * tidiness: the CPU reference walks the same `order`, so both tiers touch every
 * node's edges in the same sequence and accumulate the same f32 rounding, which is
 * what makes a CPU/GPU parity assertion meaningful rather than a tolerance game.
 */

import { softWorkgroups, type IslandInput } from './softIslands.js';

/**
 * Colors a single u32 per-node mask can hold.
 *
 * 32 because the mask is one word and one load. A graph whose maximum degree
 * exceeds this is refused rather than silently mis-colored: the alternative would
 * be a mask array of several words per node, which costs a load per node per edge
 * in the hot path to support a mesh none of the scenes here can produce.
 */
export const MAX_COLORS = 32;

/** One color's worth of edges, and the dispatch that runs them. */
export interface SoftBatch {
  readonly color: number;
  /** Index into `SoftColoring.order` where this batch starts. */
  readonly base: number;
  readonly count: number;
  /** `ceil(count / 64)`. One dispatch per batch per iteration. */
  readonly workgroups: number;
}

export interface SoftColoring {
  readonly constraints: number;
  /** Distinct colors used, which is also `batches.length`. Contiguous from 0. */
  readonly colors: number;
  /** Color per constraint, in the original constraint order. */
  readonly colorOfConstraint: Uint32Array;
  /**
   * Batch position -> constraint index. Concatenation of every batch's edges in
   * ascending original index, so `order[batch.base + i]` is the i-th edge of that
   * color. This is the permutation the GPU upload applies and the CPU walks.
   */
  readonly order: Uint32Array;
  readonly batches: readonly SoftBatch[];
  /** Largest node degree: the optimal color count, and within one of first-fit's. */
  readonly maxNodeDegree: number;
  /** The biggest batch. A per-batch allocation would size to this. */
  readonly maxBatchSize: number;
  /** Dispatches one solve iteration costs: one per color. */
  readonly dispatchesPerIteration: number;
}

/**
 * Color every edge so no two edges of one color share a node.
 *
 * First-fit in ascending constraint order: an edge takes the lowest color neither
 * endpoint has used yet. Ascending order rather than a random or degree-sorted one
 * because the result has to be reproducible from the graph alone -- a sorted-by-
 * degree pass would need a tie-break, and any tie-break that looks at insertion
 * order is a determinism bug waiting for a HashMap.
 */
export function colorConstraints(input: IslandInput): SoftColoring {
  const { count } = input;
  const { ends, count: edges } = input.constraints;

  // Bits of the colors already used at each endpoint. One word per node, so the
  // whole mask array is 4 bytes/node -- 40 KiB at 10k nodes.
  const used = new Uint32Array(count);
  const degree = new Uint32Array(count);
  const colorOfConstraint = new Uint32Array(edges);

  let colors = 0;
  let maxNodeDegree = 0;
  for (let k = 0; k < edges; k++) {
    const a = ends[k * 2];
    const b = ends[k * 2 + 1];
    const mask = used[a] | used[b];
    // Lowest clear bit. `~mask & 31` is not usable directly because JS bitwise
    // ops are 32-bit signed, so count the trailing ones instead.
    let color = 0;
    while (color < MAX_COLORS && (mask & (1 << color)) !== 0) color++;
    if (color >= MAX_COLORS) {
      throw new RangeError(
        `node ${a} has degree ${degree[a] + 1}, which needs more than the ${MAX_COLORS} colors a u32 mask holds`,
      );
    }
    const bit = 1 << color;
    used[a] |= bit;
    used[b] |= bit;
    colorOfConstraint[k] = color;
    if (color >= colors) colors = color + 1;
    degree[a]++;
    degree[b]++;
  }
  for (let i = 0; i < count; i++) {
    if (degree[i] > maxNodeDegree) maxNodeDegree = degree[i];
  }

  // Counting sort by color: one pass for the sizes, one for the positions, one to
  // scatter. Stable, so edges keep ascending original index within their color.
  const sizes = new Uint32Array(colors);
  for (let k = 0; k < edges; k++) sizes[colorOfConstraint[k]]++;
  const bases = new Uint32Array(colors);
  let total = 0;
  for (let c = 0; c < colors; c++) {
    bases[c] = total;
    total += sizes[c];
  }
  const order = new Uint32Array(edges);
  const cursor = bases.slice();
  for (let k = 0; k < edges; k++) {
    const c = colorOfConstraint[k];
    order[cursor[c]++] = k;
  }

  const batches: SoftBatch[] = [];
  let maxBatchSize = 0;
  for (let c = 0; c < colors; c++) {
    if (sizes[c] > maxBatchSize) maxBatchSize = sizes[c];
    batches.push({
      color: c,
      base: bases[c],
      count: sizes[c],
      workgroups: softWorkgroups(sizes[c]),
    });
  }

  return {
    constraints: edges,
    colors,
    colorOfConstraint,
    order,
    batches,
    maxNodeDegree,
    maxBatchSize,
    dispatchesPerIteration: colors,
  };
}

/**
 * Total workgroups one solve iteration dispatches, summed over its colors.
 *
 * Reported separately from the batch count because the two answer different
 * questions: the batch count is the number of dispatches (the thing the HUD shows
 * and the thing that costs a barrier), and this is the number of thread groups
 * behind them, which is what an occupancy estimate needs.
 */
export function coloringWorkgroups(coloring: SoftColoring): number {
  let total = 0;
  for (const batch of coloring.batches) total += batch.workgroups;
  return total;
}

/**
 * Check the property the whole race-free claim rests on.
 *
 * Not used by the solver -- re-verifying on every step would cost more than the
 * solve -- but the specs call it once per scene, and it is the assertion that
 * turns "coloring avoids data races" from a comment into a tested claim. O(edges +
 * nodes) with one array.
 */
export function coloringIsRaceFree(coloring: SoftColoring, input: IslandInput): boolean {
  const { ends } = input.constraints;
  // Walked batch by batch in dispatch order, because that is the grouping the
  // claim is about: two edges in *different* batches may share a node freely.
  // One generation-stamped array serves every batch without being cleared, so a
  // node touched by an earlier batch cannot be mistaken for one touched by this.
  const seenAt = new Uint32Array(input.count);
  let generation = 0;
  for (const batch of coloring.batches) {
    generation++;
    for (let i = 0; i < batch.count; i++) {
      const k = coloring.order[batch.base + i];
      for (let e = 0; e < 2; e++) {
        const node = ends[k * 2 + e];
        if (seenAt[node] === generation) return false;
        seenAt[node] = generation;
      }
    }
  }
  return true;
}

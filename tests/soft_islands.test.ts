/**
 * `gpu/softIslands.ts` -- connected components, and the padded node order the
 * island-mapped kernels dispatch against.
 *
 * The load-bearing property is that *no workgroup straddles two islands*, because
 * that is what turns "is this island asleep" into one load per workgroup instead
 * of one per node. Everything else here -- the numbering, the padding, the
 * sentinel -- exists to make that true and to make it reproducible, so the tests
 * go after the boundary cases: an island of exactly 64, one of 65, one of 200
 * across four workgroups, and a graph with no edges at all where every node is its
 * own island.
 */

import { describe, expect, it } from 'vitest';
import {
  NODE_SENTINEL,
  SOFT_WORKGROUP_SIZE,
  groupIslands,
  isSentinel,
  softWorkgroups,
} from '../src/gpu/softIslands.js';
import { SoftMesh, SOFT_STRIDE, emptyConstraints, type SoftScene } from '../src/gpu/softMesh.js';

/** A hand-built graph, so island structure is chosen rather than discovered. */
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

/** A chain of `n` nodes, i.e. the worst case for a naive union-find. */
function chain(n: number, first = 0): number[] {
  const out: number[] = [];
  for (let k = 0; k < n - 1; k++) out.push(first + k, first + k + 1);
  return out;
}

describe('the workgroup size', () => {
  it('is 64, the size every WebGPU implementation guarantees', () => {
    // The plan fixes this at 64 and forbids depending on subgroups: 64 is a
    // multiple of every common subgroup width, so nothing has to query one.
    expect(SOFT_WORKGROUP_SIZE).toBe(64);
    expect(NODE_SENTINEL).toBe(0xffffffff);
    expect(isSentinel(NODE_SENTINEL)).toBe(true);
    expect(isSentinel(0)).toBe(false);
    expect(isSentinel(65535)).toBe(false);
  });

  it('rounds up, because rounding down drops the tail workgroup', () => {
    // A floored division would leave the last 63 nodes of a mesh unintegrated,
    // which reads as a solver that leaks at one edge of the cloth.
    expect(softWorkgroups(0)).toBe(0);
    expect(softWorkgroups(1)).toBe(1);
    expect(softWorkgroups(63)).toBe(1);
    expect(softWorkgroups(64)).toBe(1);
    expect(softWorkgroups(65)).toBe(2);
    expect(softWorkgroups(10_000)).toBe(157);
    expect(softWorkgroups(100_000)).toBe(1563);
  });
});

describe('island numbering', () => {
  it('numbers islands by ascending smallest node index', () => {
    const mesh = graph(6, [0, 1, 1, 2, 3, 4]);
    const islands = groupIslands(mesh);
    expect(islands.islands).toBe(3);
    expect(Array.from(islands.islandSizes)).toEqual([3, 2, 1]);
    expect(Array.from(islands.islandOfNode)).toEqual([0, 0, 0, 1, 1, 2]);
    expect(islands.singletonIslands).toBe(1);
    expect(islands.maxIslandSize).toBe(3);
    expect(islands.nodes).toBe(6);
  });

  it('does not depend on the order the edges were united in', () => {
    // The partition is order-independent, and so is the numbering, because both
    // come out of one ascending scan of the nodes. This is what makes "island 3 is
    // asleep" mean the same thing on the CPU reference and on the device.
    const forward = groupIslands(graph(6, [0, 1, 1, 2, 3, 4]));
    const backward = groupIslands(graph(6, [3, 4, 1, 2, 0, 1]));
    expect(Array.from(backward.islandOfNode)).toEqual(Array.from(forward.islandOfNode));
    expect(Array.from(backward.islandSizes)).toEqual(Array.from(forward.islandSizes));
    expect(Array.from(backward.nodeOrder)).toEqual(Array.from(forward.nodeOrder));
  });

  it('treats an unconnected node as its own island', () => {
    const mesh = new SoftMesh(new Float32Array(8 * SOFT_STRIDE), {
      count: 8,
      constraints: emptyConstraints(),
    });
    const islands = groupIslands(mesh);
    expect(islands.islands).toBe(8);
    expect(islands.singletonIslands).toBe(8);
    expect(islands.maxIslandSize).toBe(1);
    expect(Array.from(islands.islandOfNode)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    // Eight islands of one node is eight workgroups: the degenerate case where
    // padding costs 63/64 of the dispatch, and the reason `sheets` is worth more
    // than a single connected cloth as a test scene.
    expect(islands.workgroups).toBe(8);
    expect(islands.paddedNodes).toBe(512);
    expect(islands.paddingNodes).toBe(504);
  });

  it('groups sheets into one island per sheet', () => {
    // jitter 0 so the sheet planes are exactly -3.2 / 0 / +3.2 and the island a
    // node belongs to can be read off its z.
    const mesh = new SoftMesh({ count: 100, seed: 1, scene: 'sheets', groups: 3, jitter: 0 });
    const islands = groupIslands(mesh);
    expect(islands.islands).toBe(3);
    expect(Array.from(islands.islandSizes)).toEqual([34, 33, 33]);
    expect(islands.islandSizes.reduce((a, b) => a + b, 0)).toBe(100);
    // Contiguous runs: island i owns node indices [offset, offset+size).
    expect(Array.from(islands.islandOffsets)).toEqual([0, 64, 128]);
    for (let i = 0; i < 100; i++) {
      const island = islands.islandOfNode[i];
      expect(mesh.position(i)[2], `node ${i}`).toBeCloseTo([-3.2, 0, 3.2][island], 1);
    }
  });

  it('finds one island in every connected scene', () => {
    for (const scene of ['cloth', 'cube', 'rope'] as SoftScene[]) {
      const islands = groupIslands(new SoftMesh({ count: 216, seed: 5, scene }));
      expect(islands.islands, scene).toBe(1);
      expect(islands.singletonIslands, scene).toBe(0);
    }
  });

  it('transitively connects through a shared node', () => {
    // 0-1, 1-2 and 2-3 are three separate unions that must land in one island, so
    // this is the case a find without path compression or a union by index gets
    // wrong at scale.
    const islands = groupIslands(graph(5, [0, 1, 1, 2, 2, 3]));
    expect(islands.islands).toBe(2);
    expect(Array.from(islands.islandSizes)).toEqual([4, 1]);
    expect(Array.from(islands.islandOfNode)).toEqual([0, 0, 0, 0, 1]);
  });
});

describe('the padded node order', () => {
  it('pads every island to a workgroup boundary with the sentinel', () => {
    const islands = groupIslands(graph(6, [0, 1, 1, 2, 3, 4]));
    expect(islands.paddedNodes).toBe(192);
    expect(islands.paddedNodes % SOFT_WORKGROUP_SIZE).toBe(0);
    expect(islands.nodeOrder.length).toBe(192);
    // Island 0: three real nodes then 61 sentinels.
    expect(Array.from(islands.nodeOrder.subarray(0, 4))).toEqual([0, 1, 2, NODE_SENTINEL]);
    // Island 1 starts exactly on the next workgroup.
    expect(Array.from(islands.nodeOrder.subarray(64, 67))).toEqual([3, 4, NODE_SENTINEL]);
    expect(Array.from(islands.nodeOrder.subarray(128, 130))).toEqual([5, NODE_SENTINEL]);
    const real = Array.from(islands.nodeOrder).filter((x) => !isSentinel(x));
    expect(real.length).toBe(6);
    expect([...real].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(islands.paddingNodes).toBe(186);
  });

  it('adds no padding when an island is an exact multiple of 64', () => {
    const mesh = new SoftMesh({ count: 64, seed: 7 });
    const islands = groupIslands(mesh);
    expect(islands.paddedNodes).toBe(64);
    expect(islands.workgroups).toBe(1);
    expect(islands.paddingNodes).toBe(0);
    expect(islands.nodeOrder.some(isSentinel)).toBe(false);
    // One node more and the whole second workgroup is paid for.
    const over = groupIslands(new SoftMesh({ count: 65, seed: 7 }));
    expect(over.paddedNodes).toBe(128);
    expect(over.workgroups).toBe(2);
    expect(over.paddingNodes).toBe(63);
  });

  it('spreads one island over several workgroups without mixing it with another', () => {
    const islands = groupIslands(graph(200, chain(200)));
    expect(islands.islands).toBe(1);
    expect(islands.paddedNodes).toBe(256);
    expect(islands.workgroups).toBe(4);
    expect(Array.from(islands.islandOfWorkgroup)).toEqual([0, 0, 0, 0]);
    expect(Array.from(islands.nodeOrder.subarray(0, 200).filter((x) => isSentinel(x)))).toEqual([]);
    expect(islands.paddingNodes).toBe(56);
    expect(islands.nodeOrder[199]).toBe(199);
    expect(islands.nodeOrder[200]).toBe(NODE_SENTINEL);
  });

  it('starts the next island on a workgroup boundary even after an exact fit', () => {
    // 64 nodes in one chain, then a separate 10-node chain: island 1 must begin at
    // slot 64, not at slot 64 of a shared workgroup.
    const islands = groupIslands(graph(74, [...chain(64), ...chain(10, 64)]));
    expect(islands.islands).toBe(2);
    expect(Array.from(islands.islandSizes)).toEqual([64, 10]);
    expect(Array.from(islands.islandOffsets)).toEqual([0, 64]);
    expect(islands.paddedNodes).toBe(128);
    expect(Array.from(islands.islandOfWorkgroup)).toEqual([0, 1]);
    expect(islands.nodeOrder[63]).toBe(63);
    expect(islands.nodeOrder[64]).toBe(64);
  });

  it('attributes every workgroup to exactly one island, for every scene', () => {
    // The invariant the sleep kernel is built on. Checked structurally rather than
    // by example: a workgroup that straddled two islands would have two different
    // islandOfNode values in it, and the one-load sleep test would be wrong.
    for (const scene of ['cloth', 'sheets', 'cube', 'rope'] as SoftScene[]) {
      for (const count of [1, 7, 64, 100, 257, 1000]) {
        const mesh = new SoftMesh({ count, scene, seed: 5, groups: 4 });
        const islands = groupIslands(mesh);
        expect(islands.paddedNodes % SOFT_WORKGROUP_SIZE, `${scene}@${count}`).toBe(0);
        expect(islands.workgroups, `${scene}@${count}`).toBe(islands.paddedNodes / SOFT_WORKGROUP_SIZE);
        expect(islands.islandOfWorkgroup.length).toBe(islands.workgroups);
        for (let w = 0; w < islands.workgroups; w++) {
          const slice = islands.nodeOrder.subarray(w * 64, w * 64 + 64);
          const owner = islands.islandOfWorkgroup[w];
          for (const slot of slice) {
            if (isSentinel(slot)) continue;
            expect(islands.islandOfNode[slot], `${scene}@${count} wg ${w}`).toBe(owner);
          }
          // The first slot of a workgroup is never a sentinel: padding only ever
          // trails an island, so a whole-sentinel workgroup would be dead dispatch.
          expect(isSentinel(slice[0]), `${scene}@${count} wg ${w}`).toBe(false);
        }
        // Every node appears exactly once, and the runs are contiguous.
        const seen = new Uint8Array(count);
        for (const slot of islands.nodeOrder) if (!isSentinel(slot)) seen[slot]++;
        expect(Array.from(seen), `${scene}@${count}`).toEqual(new Array(count).fill(1));
        for (let i = 0; i < islands.islands; i++) {
          const run = islands.nodeOrder.subarray(
            islands.islandOffsets[i],
            islands.islandOffsets[i] + islands.islandSizes[i],
          );
          expect(Array.from(run), `${scene}@${count} island ${i}`).toEqual(
            Array.from(run).sort((a, b) => a - b),
          );
        }
      }
    }
  });

  it('is reproducible run to run', () => {
    const build = () => groupIslands(new SoftMesh({ count: 100, seed: 1, scene: 'sheets', groups: 3 }));
    const a = build();
    const b = build();
    expect(Array.from(a.nodeOrder)).toEqual(Array.from(b.nodeOrder));
    expect(Array.from(a.islandOfNode)).toEqual(Array.from(b.islandOfNode));
    expect(Array.from(a.islandOfWorkgroup)).toEqual(Array.from(b.islandOfWorkgroup));
  });
});

describe('at the scale the plan asks for', () => {
  it('groups and pads a 10k-node cloth without a straddle', () => {
    const mesh = new SoftMesh({ count: 10_000, seed: 1 });
    const islands = groupIslands(mesh);
    expect(islands.islands).toBe(1);
    expect(islands.paddedNodes).toBe(10_048);
    expect(islands.workgroups).toBe(157);
    expect(islands.paddingNodes).toBe(48);
    expect(Array.from(islands.islandOfWorkgroup).every((w) => w === 0)).toBe(true);
    expect(islands.nodeOrder.length).toBe(10_048);
  });

  it('keeps a 10k-node chain from degenerating the union-find', () => {
    // Union by size rather than by index: an index-ordered union turns a chain
    // into a linked list, and 10k nodes then means O(n^2) finds. This is the rope
    // scene at target scale, and it has to come back quickly enough that the demo
    // can rebuild a scene without a visible hitch.
    const started = performance.now();
    const islands = groupIslands(new SoftMesh({ count: 10_000, seed: 1, scene: 'rope' }));
    const elapsed = performance.now() - started;
    expect(islands.islands).toBe(1);
    expect(islands.workgroups).toBe(157);
    expect(elapsed, 'grouping 10k chain nodes').toBeLessThan(250);
  });

  it('keeps padding waste under one workgroup per island', () => {
    for (const groups of [2, 4, 8, 16]) {
      const mesh = new SoftMesh({ count: 10_000, seed: 1, scene: 'sheets', groups });
      const islands = groupIslands(mesh);
      expect(islands.islands, `${groups} sheets`).toBe(groups);
      // At most 63 slots per island, and never a whole extra workgroup.
      expect(islands.paddingNodes, `${groups} sheets`).toBeLessThan(groups * SOFT_WORKGROUP_SIZE);
      expect(islands.paddedNodes - islands.nodes).toBe(islands.paddingNodes);
    }
  });
});

/**
 * Connected-component grouping over the constraint graph, and the node order
 * the island-mapped kernels dispatch against.
 *
 * # Why islands get their own pass
 *
 * Nothing in this layer makes two islands interact: no self-collision, no
 * contact against other bodies. So an island that comes to rest can stop being
 * simulated entirely, and `sheets` -- several disconnected cloths -- spends most
 * of its frames with some of them asleep. That is the M4 ask "island 分组", and
 * the saving is only available if "is this island asleep" costs one load per
 * workgroup rather than one per node.
 *
 * # Why the node order is repacked at all
 *
 * A workgroup of 64 that straddles two islands cannot answer that question with
 * one load, because its 64 nodes belong to two different sleep flags. So this
 * module emits `nodeOrder`: every island's nodes consecutively, each island
 * padded out to a multiple of the workgroup size with a sentinel. Workgroup *w*
 * then covers 64 slots of exactly one island, `islandOfWorkgroup[w]` names it,
 * and the kernel's first act is `if (asleep[islandOfWorkgroup[wg]]) { return; }`.
 *
 * The padding costs at most 63 slots per island, which at 64-node workgroups is
 * under one extra workgroup each. The alternative -- a per-node island lookup
 * and a ballot -- needs subgroups, and the plan forbids depending on them: 64 is
 * the one workgroup size every WebGPU implementation guarantees, and a subgroup
 * operation is not core.
 *
 * # Determinism
 *
 * Island indices are assigned in ascending order of each island's smallest node
 * index, and nodes within an island ascend. Both fall out of a single ascending
 * scan, so the numbering is a function of the graph and not of the union-find's
 * internal merge order -- which does vary with the order edges are united in.
 * That is what makes the CPU reference and the GPU backend agree on "island 3 is
 * asleep" rather than merely agreeing that *some* island is.
 */

import type { SoftConstraints } from './softMesh.js';

/**
 * Workgroup size for every soft-body kernel.
 *
 * 64 rather than 128 or 256: it is the largest size WebGPU guarantees without
 * querying `maxComputeInvocationsPerWorkgroup`, and it is a multiple of every
 * common subgroup size (32 on NVIDIA/Intel, 64 on AMD, and the Metal
 * quad/simdgroup widths), so a driver that does have subgroups still gets full
 * ones. Nothing here uses a subgroup operation -- the plan says not to depend on
 * them -- so 64 is chosen for the guarantee, and the padding below is sized to
 * it.
 */
export const SOFT_WORKGROUP_SIZE = 64;

/** The value `nodeOrder` pads an island out with. Not a legal node index. */
export const NODE_SENTINEL = 0xffffffff;

/** What the grouper reads: a node count and the graph over it. */
export interface IslandInput {
  readonly count: number;
  readonly constraints: SoftConstraints;
}

export interface SoftIslands {
  readonly nodes: number;
  /** Connected components, singletons included. Always >= 1 for a real mesh. */
  readonly islands: number;
  /** Island index per node, ascending node order. */
  readonly islandOfNode: Uint32Array;
  /** Nodes per island, ascending island index. Sums to `nodes`. */
  readonly islandSizes: Uint32Array;
  /** Where each island's run starts in `nodeOrder`. */
  readonly islandOffsets: Uint32Array;
  /**
   * The dispatch order for the island-mapped kernels: island 0's nodes, padding
   * to a workgroup boundary, island 1's nodes, padding, and so on. Length is
   * `paddedNodes`, which is a multiple of `SOFT_WORKGROUP_SIZE`.
   */
  readonly nodeOrder: Uint32Array;
  /** The island each workgroup belongs to. Length `workgroups`. */
  readonly islandOfWorkgroup: Uint32Array;
  readonly paddedNodes: number;
  readonly workgroups: number;
  /** Islands of exactly one node. A rope of one, or a `sheets` degenerate. */
  readonly singletonIslands: number;
  /** The largest island, which is what a per-island allocation would size to. */
  readonly maxIslandSize: number;
  /** Slots that exist only to pad a workgroup. Waste, reported so it is visible. */
  readonly paddingNodes: number;
}

/**
 * Group the constraint graph into islands and repack the node order.
 *
 * Union-find with path halving and union by size: two passes over the edges and
 * one over the nodes, no allocation per edge. At 40k edges and 10k nodes this is
 * well under a millisecond, which matters because it runs once per system
 * creation and again on every scene change in the demo.
 */
export function groupIslands(input: IslandInput): SoftIslands {
  const { count } = input;
  const { ends, count: edges } = input.constraints;

  const parent = new Uint32Array(count);
  const size = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    parent[i] = i;
    size[i] = 1;
  }

  for (let k = 0; k < edges; k++) {
    union(parent, size, ends[k * 2], ends[k * 2 + 1]);
  }

  // Island ids in first-encounter order, which is ascending smallest-node-index
  // because the scan below is ascending. `rootToIsland` is filled lazily rather
  // than by sorting representatives, so no comparator is involved and the result
  // cannot depend on a tie-break.
  const rootToIsland = new Int32Array(count).fill(-1);
  const islandOfNode = new Uint32Array(count);
  const members: number[][] = [];
  for (let i = 0; i < count; i++) {
    const root = find(parent, i);
    let island = rootToIsland[root];
    if (island < 0) {
      island = members.length;
      rootToIsland[root] = island;
      members.push([]);
    }
    islandOfNode[i] = island;
    members[island].push(i);
  }

  const islands = members.length;
  const islandSizes = new Uint32Array(islands);
  const islandOffsets = new Uint32Array(islands);
  let singletons = 0;
  let maxIslandSize = 0;
  let paddedNodes = 0;
  for (let i = 0; i < islands; i++) {
    const n = members[i].length;
    islandSizes[i] = n;
    islandOffsets[i] = paddedNodes;
    if (n === 1) singletons++;
    if (n > maxIslandSize) maxIslandSize = n;
    paddedNodes += Math.ceil(n / SOFT_WORKGROUP_SIZE) * SOFT_WORKGROUP_SIZE;
  }

  const nodeOrder = new Uint32Array(paddedNodes).fill(NODE_SENTINEL);
  const workgroups = paddedNodes / SOFT_WORKGROUP_SIZE;
  const islandOfWorkgroup = new Uint32Array(workgroups);
  for (let i = 0; i < islands; i++) {
    const base = islandOffsets[i];
    const list = members[i];
    for (let j = 0; j < list.length; j++) nodeOrder[base + j] = list[j];
    // Every workgroup this island occupies belongs to it, including the last
    // partly-sentinel one. A kernel reads this instead of the node it is on,
    // which is why the padding has to be attributed rather than left blank.
    const last = (base + list.length - 1) / SOFT_WORKGROUP_SIZE;
    for (let w = base / SOFT_WORKGROUP_SIZE; w <= last; w++) islandOfWorkgroup[w] = i;
  }

  return {
    nodes: count,
    islands,
    islandOfNode,
    islandSizes,
    islandOffsets,
    nodeOrder,
    islandOfWorkgroup,
    paddedNodes,
    workgroups,
    singletonIslands: singletons,
    maxIslandSize,
    paddingNodes: paddedNodes - count,
  };
}

/** True when every slot of `nodeOrder` is a real node or the sentinel. */
export function isSentinel(slot: number): boolean {
  return slot === NODE_SENTINEL;
}

/**
 * Workgroups needed for `n` items at the soft-body workgroup size.
 *
 * One function rather than `Math.ceil(n / 64)` at each call site, because the
 * batch dispatch and the node dispatch have to agree on it and a rounded-down
 * division silently drops the tail workgroup -- the last 63 nodes of a mesh
 * would never be integrated, which looks like a solver that leaks at one edge.
 */
export function softWorkgroups(n: number): number {
  return Math.ceil(n / SOFT_WORKGROUP_SIZE);
}

/** Path-halving find: no recursion, and it shortens the tree as it goes. */
function find(parent: Uint32Array, i: number): number {
  let root = i;
  while (parent[root] !== root) {
    parent[root] = parent[parent[root]];
    root = parent[root];
  }
  return root;
}

function union(parent: Uint32Array, size: Uint32Array, a: number, b: number): void {
  const ra = find(parent, a);
  const rb = find(parent, b);
  if (ra === rb) return;
  // By size rather than by index: index-ordered union degenerates into a list on
  // a chain graph, which is exactly what `rope` at 10k nodes is.
  if (size[ra] < size[rb]) {
    parent[ra] = rb;
    size[rb] += size[ra];
  } else {
    parent[rb] = ra;
    size[ra] += size[rb];
  }
}

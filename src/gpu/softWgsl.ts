/**
 * The WGSL for the soft-body pipeline, generated from the same constants the CPU
 * reference uses.
 *
 * # Why the source is a template and not a `.wgsl` file
 *
 * Every number in here -- the node sentinel, the flag bits, the bounds-mode
 * encoding, the stats word order, the `Params` member order, the workgroup size --
 * is interpolated from the TypeScript module that already owns it. A hand-written
 * shader file would be a second copy, and the failure mode of a second copy is not
 * a compile error: WGSL is perfectly happy to read `restitution` where the CPU
 * wrote `damping`, and to walk a node order that `softIslands.ts` padded
 * differently. Generating the text keeps one source of truth, and
 * `tests/soft_wgsl.test.ts` re-derives the struct offsets from the WGSL alignment
 * rules so that a *layout* drift fails on every machine rather than only on one
 * with a device attached.
 *
 * # Pass structure
 *
 * One step is `predict -> solve * (iterations * colors) -> finalize -> measure ->
 * sleep_update -> publish`, matching `CpuSoftSystem.step` pass for pass and in the
 * same order, over the same padded node order and the same colored constraint
 * order. The order is not a style choice: dispatches inside one compute pass are
 * ordered by the WebGPU spec, with the writes of one visible to the next, and that
 * ordering is the only barrier this pipeline has. `solve` reads and writes `predBuf`
 * in place, so iteration *k + 1* has to be a later dispatch, not a wider one.
 *
 * # Why the batch rides on the z axis
 *
 * Two constraints of one color are race-free, but two colors are not, so each color
 * is its own dispatch and the kernel still has to learn *which* batch it is
 * running. The alternatives are all worse: rewriting a uniform per dispatch needs a
 * submit per dispatch (24 submits a step at the default iteration count), and a
 * dynamic offset needs a 256-byte stride per slot for 8 bytes of data. Instead the
 * batch table is one static buffer of `(base, count)` pairs built when the mesh is
 * built, and the dispatcher puts the color index in the z dimension, where
 * `global_invocation_id.z` hands it to the kernel for free. A whole step is then one
 * command buffer with no per-dispatch CPU work at all. `softSolveDispatch` is the
 * one place that mapping is written down, because it is exactly the kind of
 * contract that drifts between a shader and a dispatcher.
 *
 * # Why `solve` contains no atomics
 *
 * `softColoring.ts` guarantees that no two constraints of one color share a node, so
 * every write inside one dispatch lands on a distinct `predBuf` slot and a plain
 * store is correct. That is what makes this layer's `raceFree: true` a stronger
 * claim than the particle layer's, and it is why the same seed on the same device
 * produces the same digest twice. The atomics that remain -- two `atomicMax` and
 * two `atomicAdd` -- are max and count reductions, which are order-independent by
 * construction, so the driver's freedom to schedule them costs nothing.
 *
 * # What parity means here
 *
 * Not bitwise across machines. The arithmetic association below is written to match
 * `softCpu.ts` term for term, but WGSL lets a driver contract `a * b + c` into a
 * fused multiply-add and two adapters may round `sqrt` differently. So the claim is
 * the one `softTypes.ts` makes: identical digests for the same seed on the same
 * device, and tolerance-based agreement with the CPU reference elsewhere.
 *
 * # Constraints this shader is written under
 *
 * - `@workgroup_size(64)` everywhere and no `subgroup` anything. 64 is the size
 *   every WebGPU implementation guarantees without querying a limit, it is the size
 *   `softIslands.ts` pads to, and the plan forbids depending on subgroups.
 * - Fifteen storage buffers in one pipeline layout, which is above the WebGPU
 *   baseline of 8. `device.ts` already asks for the limit the adapter reports
 *   (`REQUESTED_LIMITS`), so the shared device has it on any adapter that does;
 *   `SOFT_STORAGE_BINDINGS` is the number `softGpu.ts` has to gate on, because an
 *   adapter that reports 8 gets the CPU reference rather than a pipeline layout
 *   that fails validation at creation time.
 * - No prefix scan and no indirect dispatch: the counts are all known on the CPU
 *   when the mesh is built.
 * - f32 only, because WGSL has no f64 and `softCpu.ts` is f32 for the same reason.
 */

import type { SoftBatch } from './softColoring.js';
import { NODE_SENTINEL, SOFT_WORKGROUP_SIZE } from './softIslands.js';
import { SOFT_STRIDE } from './softMesh.js';
import {
  SOFT_BOUNDS_MODE_BITS,
  SOFT_FLAG,
  SOFT_PARAM_WORD,
} from './softOptions.js';

/**
 * Floats per node in the state buffer: `pos.xyz | invMass | vel.xyz | radius`.
 *
 * Taken from `SOFT_STRIDE` rather than restated, because the state buffer *is*
 * `SoftMesh.data`: one `writeBuffer` up, one copy down, and no repack in either
 * direction to get wrong.
 */
export const SOFT_STATE_FLOATS_PER_NODE = SOFT_STRIDE;

/**
 * `vec4`s per node in the state buffer. Two, so a node is `state[2i] =
 * (pos, invMass)` and `state[2i + 1] = (vel, radius)` -- the split `SOFT_OFFSET`
 * already declares, which is what lets the kernels read an inverse mass out of
 * `state[i * 2u].w` instead of from a fourth buffer.
 */
export const SOFT_STATE_VECS_PER_NODE = SOFT_STATE_FLOATS_PER_NODE / 4;

/**
 * `vec4`s per node in the prediction scratch. One, holding `xyz` with `w` written
 * as zero.
 *
 * `vec4` rather than `vec3` even though only three components are used: a
 * `vec3<f32>` in a storage array has a stride of 16 bytes anyway, so the tighter
 * type would buy nothing and would leave the stride to be inferred from an
 * alignment rule instead of stated.
 */
export const SOFT_PRED_VECS_PER_NODE = 1;

/**
 * Floats per node in the buffer the renderer reads: tight `xyz`.
 *
 * Three rather than four on purpose. `CpuSoftSystem.copyPublishedTo` writes tight
 * xyz, and one layout for both tiers is what lets `render/soft.ts` blit into a
 * three.js attribute without ever learning which tier produced it. A `vec4` here
 * would be 33% more bandwidth across the compute/graphics boundary every frame,
 * for a component nothing reads.
 */
export const SOFT_PUBLISH_FLOATS_PER_NODE = 3;

/** `u32`s per constraint in `endsBuf`: the two endpoint indices. */
export const SOFT_ENDS_U32_PER_CONSTRAINT = 2;

/** `u32`s per constraint in `orderBuf`: its position in the colored order. */
export const SOFT_ORDER_U32_PER_CONSTRAINT = 1;

/**
 * `f32`s per constraint across `restBuf` and `stiffBuf`, one each.
 *
 * Kept as two buffers in the mesh's own order rather than interleaved into one,
 * because neither is permuted: the shader applies the coloring through `orderBuf`,
 * so the upload is a straight copy of `constraints.rest` and
 * `constraints.stiffness` and there is no interleave step to get wrong.
 */
export const SOFT_EDGE_F32_PER_CONSTRAINT = 2;

/** `u32`s per color in `batchBuf`: `(base, count)`, packed as a `vec2<u32>`. */
export const SOFT_BATCH_U32_PER_COLOR = 2;

/**
 * Words per island across the three sleep buffers: `asleep`, `quiet`,
 * `islandSpeed`. One word each, mirroring the three arrays `softCpu.ts` keeps.
 */
export const SOFT_SLEEP_WORDS_PER_ISLAND = 3;

/**
 * Entry points, in dispatch order. `tests/soft_wgsl.test.ts` pins the list, the
 * order, and the fact that it is one longer than `SOFT_FIXED_DISPATCHES`.
 */
export const SOFT_KERNELS = [
  'predict',
  'solve',
  'finalize',
  'measure',
  'sleep_update',
  'publish',
] as const;

export type SoftKernel = (typeof SOFT_KERNELS)[number];

/**
 * What each kernel's dispatch count is derived from.
 *
 * `nodeWorkgroups` means the island-mapped kernels, which walk the padded node
 * order and so dispatch `plan.nodeWorkgroups` groups whose 64 lanes all belong to
 * one island. `batch` is `solve` alone: its x count comes from the color's batch
 * and its z from the color index, which is what makes the two dimensions of one
 * dispatch mean different things.
 */
export type SoftKernelDispatch =
  | 'nodeWorkgroups'
  | 'batch'
  | 'constraints'
  | 'islands'
  | 'count';

export const SOFT_KERNEL_DISPATCH: Readonly<Record<SoftKernel, SoftKernelDispatch>> = {
  predict: 'nodeWorkgroups',
  solve: 'batch',
  finalize: 'nodeWorkgroups',
  measure: 'constraints',
  sleep_update: 'islands',
  publish: 'count',
};

/**
 * Dispatch dimensions of one `solve`: `(workgroups, 1, color)`.
 *
 * The z dimension is the batch index the kernel reads out of `batchBuf`, and the y
 * dimension is always 1 because a batch is a flat run of constraints. Written down
 * once, here, next to the shader text that depends on it: a dispatcher that put the
 * color in y instead would compile, run, and solve color 0 `iterations * colors`
 * times.
 */
export function softSolveDispatch(batch: SoftBatch): readonly [number, number, number] {
  return [batch.workgroups, 1, batch.color];
}

/** Words in `statsBuf`. The order is pinned by the test against the kernel text. */
export const SOFT_STAT_WORD = {
  /** Nodes whose centre ended the step outside the box. Zero under `reflect`. */
  escaped: 0,
  /**
   * `bitcast<u32>` of the largest squared node speed this step.
   *
   * Tracked with `atomicMax`, which works because IEEE-754 non-negative floats
   * sort the same way as their bit patterns and a squared speed is non-negative by
   * construction -- the same trick `particleWgsl.ts` uses, so the max is taken on
   * integers and decoded on readback.
   */
  maxSpeedSq: 1,
  /**
   * `bitcast<u32>` of the largest `|length - rest| / rest` over every edge.
   *
   * Also non-negative, so also an integer max. This is the word that says whether
   * the solver is converging, and it is the cleanest CPU/GPU parity check in the
   * layer: a max is independent of the order the reduction happened in, so the two
   * tiers agree on it exactly rather than within a tolerance.
   */
  maxConstraintError: 2,
  /** Islands asleep at the end of the step, counted by `sleep_update`. */
  sleepingIslands: 3,
  /** Reserved, so the buffer stays a round five words. */
  unused: 4,
} as const;

export const SOFT_STAT_WORDS = 5;

/** Bind-group index for everything derived from the mesh's shape. */
export const SOFT_GROUP_STATIC = 0;
/** Bind-group index for everything that changes while simulating. */
export const SOFT_GROUP_STATE = 1;

export type SoftBindingKind = 'uniform' | 'storage-read' | 'storage-read-write';

export type SoftBufferType = 'uniform' | 'read-only-storage' | 'storage';

/**
 * One binding slot, as the shader declares it and as a bind-group layout entry
 * needs it.
 *
 * Structurally the same shape as `WgslBinding` in `particleWgsl.ts` and a superset
 * of `ComputeBinding` in `compute.ts`, and deliberately not an import of either:
 * the particle layer's descriptor growing a field for a particle-shaped reason
 * should not force an edit here, and `compute.ts` already documents the same choice
 * from the other direction.
 */
export interface SoftWgslBinding {
  readonly group: number;
  readonly binding: number;
  readonly name: string;
  readonly kind: SoftBindingKind;
  /** WGSL type as declared in the shader. */
  readonly type: string;
  /** What this binding is in a `GPUBindGroupLayoutEntry`. */
  readonly bufferType: SoftBufferType;
}

/**
 * Every binding the pipeline declares, and the single place the shader text, the
 * bind-group layouts and the bind groups are all derived from.
 *
 * Two groups, split by lifetime rather than by access mode. Group 0 is built once
 * when the mesh is built and never touched again: the uniform and the eight buffers
 * that describe the *shape* of the problem -- the padded node order, the two island
 * maps, the colored edge order, the graph and its per-edge constants, and the batch
 * table. Group 1 is everything that changes while simulating: the node state, the
 * prediction scratch, the three sleep arrays, the step counters and the buffer the
 * renderer reads. The consequence is that a step, a `wake()` and a state restore all
 * stay inside group 1, and a scene change is the only thing that reallocates group 0.
 */
export const SOFT_BINDINGS: readonly SoftWgslBinding[] = [
  {
    group: SOFT_GROUP_STATIC,
    binding: 0,
    name: 'params',
    kind: 'uniform',
    type: 'Params',
    bufferType: 'uniform',
  },
  {
    group: SOFT_GROUP_STATIC,
    binding: 1,
    name: 'nodeOrderBuf',
    kind: 'storage-read',
    type: 'array<u32>',
    bufferType: 'read-only-storage',
  },
  {
    group: SOFT_GROUP_STATIC,
    binding: 2,
    name: 'islandOfWgBuf',
    kind: 'storage-read',
    type: 'array<u32>',
    bufferType: 'read-only-storage',
  },
  {
    group: SOFT_GROUP_STATIC,
    binding: 3,
    name: 'islandOfNodeBuf',
    kind: 'storage-read',
    type: 'array<u32>',
    bufferType: 'read-only-storage',
  },
  {
    group: SOFT_GROUP_STATIC,
    binding: 4,
    name: 'orderBuf',
    kind: 'storage-read',
    type: 'array<u32>',
    bufferType: 'read-only-storage',
  },
  {
    group: SOFT_GROUP_STATIC,
    binding: 5,
    name: 'endsBuf',
    kind: 'storage-read',
    type: 'array<u32>',
    bufferType: 'read-only-storage',
  },
  {
    group: SOFT_GROUP_STATIC,
    binding: 6,
    name: 'restBuf',
    kind: 'storage-read',
    type: 'array<f32>',
    bufferType: 'read-only-storage',
  },
  {
    group: SOFT_GROUP_STATIC,
    binding: 7,
    name: 'stiffBuf',
    kind: 'storage-read',
    type: 'array<f32>',
    bufferType: 'read-only-storage',
  },
  {
    group: SOFT_GROUP_STATIC,
    binding: 8,
    name: 'batchBuf',
    kind: 'storage-read',
    type: 'array<vec2<u32>>',
    bufferType: 'read-only-storage',
  },
  {
    group: SOFT_GROUP_STATE,
    binding: 0,
    name: 'stateBuf',
    kind: 'storage-read-write',
    type: 'array<vec4<f32>>',
    bufferType: 'storage',
  },
  {
    group: SOFT_GROUP_STATE,
    binding: 1,
    name: 'predBuf',
    kind: 'storage-read-write',
    type: 'array<vec4<f32>>',
    bufferType: 'storage',
  },
  {
    group: SOFT_GROUP_STATE,
    binding: 2,
    name: 'asleepBuf',
    kind: 'storage-read-write',
    type: 'array<u32>',
    bufferType: 'storage',
  },
  {
    group: SOFT_GROUP_STATE,
    binding: 3,
    name: 'quietBuf',
    kind: 'storage-read-write',
    type: 'array<u32>',
    bufferType: 'storage',
  },
  {
    group: SOFT_GROUP_STATE,
    binding: 4,
    name: 'islandSpeedBuf',
    kind: 'storage-read-write',
    type: 'array<atomic<u32>>',
    bufferType: 'storage',
  },
  {
    group: SOFT_GROUP_STATE,
    binding: 5,
    name: 'statsBuf',
    kind: 'storage-read-write',
    type: 'array<atomic<u32>>',
    bufferType: 'storage',
  },
  {
    group: SOFT_GROUP_STATE,
    binding: 6,
    name: 'publishBuf',
    kind: 'storage-read-write',
    type: 'array<f32>',
    bufferType: 'storage',
  },
];

/** Bindings of one group, in binding order. */
export function softBindingsForGroup(group: number): readonly SoftWgslBinding[] {
  return SOFT_BINDINGS.filter((b) => b.group === group);
}

/**
 * Storage buffers the pipeline binds, which is the
 * `maxStorageBuffersPerShaderStage` a device has to expose before any of this can
 * be created.
 *
 * Derived from the table rather than written down, because the number is the kind
 * of thing that grows by one when a kernel needs one more array and nobody thinks
 * about limits. It counts every binding but the uniform, and it counts across both
 * groups: the limit is per shader stage, so splitting the table over more bind
 * groups buys nothing.
 *
 * The consequence is a real floor on the device tier. Packing the eight static
 * buffers into one `array<u32>` of sections addressed by uniform-supplied offsets
 * would clear it, at the cost of turning every read in every kernel into an
 * `base + i` that has to be right and that no type checks. Fifteen named buffers
 * whose names say what they hold is worth more than running on an adapter that
 * exposes the 2021 minimum, and the tier that adapter gets is the CPU reference,
 * which is a supported answer rather than a degraded one.
 */
export const SOFT_STORAGE_BINDINGS = SOFT_BINDINGS.filter(
  (b) => b.bufferType !== 'uniform',
).length;

/**
 * The WebGPU baseline for `maxStorageBuffersPerShaderStage`: what a device created
 * without asking gets, and the number the constant above is compared against.
 *
 * The particle pipeline binds exactly this many, which is why `device.ts` requests
 * the adapter's own value; this pipeline binds more, which is why the request is
 * not optional here.
 */
export const SOFT_BASELINE_STORAGE_BUFFERS = 8;

export type SoftParamKind = 'vec3' | 'f32' | 'u32';

export interface SoftWgslParamMember {
  readonly name: string;
  readonly kind: SoftParamKind;
  /** Offset in 4-byte words, taken from `SOFT_PARAM_WORD` so the two cannot drift. */
  readonly word: number;
}

/**
 * `Params` in declaration order.
 *
 * The struct body in the shader is generated from this list and each `word` comes
 * straight out of `SOFT_PARAM_WORD`, so what makes this a real check rather than a
 * tautology is that the *order* here is independent: `writeSoftParams` packs by word
 * index, this list declares by name, and the test recomputes every offset from the
 * WGSL alignment rules (`vec3` aligns to 16, everything else to 4) and compares. A
 * member moved in one place and not the other fails the test.
 *
 * Three members are never read by a kernel -- `paddedNodes` and the two pads. They
 * stay declared because the struct has to be the 96 bytes `writeSoftParams` fills,
 * and because `paddedNodes` is what the CPU dispatch counts are derived from; a
 * uniform that described only the fields the shader happens to use would be a second
 * layout to keep in step with the first.
 */
export const SOFT_WGSL_PARAMS_LAYOUT: readonly SoftWgslParamMember[] = [
  { name: 'gravity', kind: 'vec3', word: SOFT_PARAM_WORD.gravityX },
  { name: 'damping', kind: 'f32', word: SOFT_PARAM_WORD.damping },
  { name: 'boundsMin', kind: 'vec3', word: SOFT_PARAM_WORD.boundsMinX },
  { name: 'invDt', kind: 'f32', word: SOFT_PARAM_WORD.invDt },
  { name: 'boundsMax', kind: 'vec3', word: SOFT_PARAM_WORD.boundsMaxX },
  { name: 'maxSpeed', kind: 'f32', word: SOFT_PARAM_WORD.maxSpeed },
  { name: 'dt', kind: 'f32', word: SOFT_PARAM_WORD.dt },
  { name: 'stiffness', kind: 'f32', word: SOFT_PARAM_WORD.stiffness },
  { name: 'restitution', kind: 'f32', word: SOFT_PARAM_WORD.restitution },
  { name: 'sleepThresholdSq', kind: 'f32', word: SOFT_PARAM_WORD.sleepThresholdSq },
  { name: 'count', kind: 'u32', word: SOFT_PARAM_WORD.count },
  { name: 'islandCount', kind: 'u32', word: SOFT_PARAM_WORD.islandCount },
  { name: 'paddedNodes', kind: 'u32', word: SOFT_PARAM_WORD.paddedNodes },
  { name: 'sleepAfter', kind: 'u32', word: SOFT_PARAM_WORD.sleepAfter },
  { name: 'flags', kind: 'u32', word: SOFT_PARAM_WORD.flags },
  { name: 'constraintCount', kind: 'u32', word: SOFT_PARAM_WORD.constraintCount },
  { name: 'padA', kind: 'u32', word: SOFT_PARAM_WORD.padA },
  { name: 'padB', kind: 'u32', word: SOFT_PARAM_WORD.padB },
];

const WGSL_TYPE: Readonly<Record<SoftParamKind, string>> = {
  vec3: 'vec3<f32>',
  f32: 'f32',
  u32: 'u32',
};

const ACCESS: Readonly<Record<SoftBindingKind, string>> = {
  uniform: 'var<uniform>',
  'storage-read': 'var<storage, read>',
  'storage-read-write': 'var<storage, read_write>',
};

function paramsStruct(): string {
  const members = SOFT_WGSL_PARAMS_LAYOUT.map((m) => `  ${m.name}: ${WGSL_TYPE[m.kind]},`);
  return ['struct Params {', ...members, '};'].join('\n');
}

function bindingDecls(): string {
  return SOFT_BINDINGS.map(
    (b) => `@group(${b.group}) @binding(${b.binding}) ${ACCESS[b.kind]} ${b.name}: ${b.type};`,
  ).join('\n');
}

let cached: string | null = null;

/**
 * The whole soft-body pipeline as one WGSL module.
 *
 * One module with six entry points rather than six modules: they share `Params`, the
 * binding declarations and `sumSq`, and a single `createShaderModule` means a single
 * place for a compile error to surface. Memoised, because the text is constant for
 * the lifetime of the process.
 */
export function softShaderSource(): string {
  if (cached !== null) return cached;
  cached = `// Generated by src/gpu/softWgsl.ts -- edit the generator, not this text.

${paramsStruct()}

${bindingDecls()}

const SENTINEL: u32 = ${NODE_SENTINEL}u;
const WG: u32 = ${SOFT_WORKGROUP_SIZE}u;
const FLAG_SLEEP: u32 = ${SOFT_FLAG.sleep}u;
const BOUNDS_SHIFT: u32 = ${SOFT_FLAG.boundsShift}u;
const BOUNDS_REFLECT: u32 = ${SOFT_BOUNDS_MODE_BITS.reflect}u;
const BOUNDS_NONE: u32 = ${SOFT_BOUNDS_MODE_BITS.none}u;

const STAT_ESCAPED: u32 = ${SOFT_STAT_WORD.escaped}u;
const STAT_MAX_SPEED_SQ: u32 = ${SOFT_STAT_WORD.maxSpeedSq}u;
const STAT_MAX_ERROR: u32 = ${SOFT_STAT_WORD.maxConstraintError}u;
const STAT_SLEEPING: u32 = ${SOFT_STAT_WORD.sleepingIslands}u;

// Written out component by component rather than as dot(v, v) so the association
// order matches the CPU reference's f32 sum. A driver is still free to contract
// this into a multiply-add, which is one of the reasons parity against softCpu.ts
// is asserted with a tolerance rather than with a digest.
fn sumSq(v: vec3<f32>) -> f32 {
  return (v.x * v.x + v.y * v.y) + v.z * v.z;
}

fn hasFlag(flag: u32) -> bool {
  return (params.flags & flag) != 0u;
}

fn boundsMode() -> u32 {
  return (params.flags >> BOUNDS_SHIFT) & 3u;
}

// Pass 1: integrate velocity, then position, into the prediction scratch.
//
// Dispatched over the padded node order, so all 64 lanes of a workgroup belong to
// one island and "is this island asleep" is one load of islandOfWgBuf[wid.x]
// instead of one islandOfNodeBuf lookup per node.
@compute @workgroup_size(${SOFT_WORKGROUP_SIZE})
fn predict(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  // Invocation 0 owns the per-step reset, and it has to run before the sleep guard
  // below: island 0 can be the one that is asleep, and a step that skipped the
  // reset would report the previous step's counters forever.
  if (gid.x == 0u) {
    atomicStore(&statsBuf[STAT_ESCAPED], 0u);
    atomicStore(&statsBuf[STAT_MAX_SPEED_SQ], 0u);
    atomicStore(&statsBuf[STAT_MAX_ERROR], 0u);
    atomicStore(&statsBuf[STAT_SLEEPING], 0u);
  }
  let wg = wid.x;
  if (asleepBuf[islandOfWgBuf[wg]] != 0u) { return; }
  let i = nodeOrderBuf[wg * WG + lid.x];
  if (i == SENTINEL) { return; }
  let base = i * 2u;
  let prev = stateBuf[base];
  let pos = prev.xyz;
  // A pinned node predicts onto itself. Its velocity needs no case of its own,
  // because finalize derives velocity from the position change, which is zero.
  if (!(prev.w > 0.0)) {
    predBuf[i] = vec4<f32>(pos, 0.0);
    return;
  }
  let dt = params.dt;
  let damp = 1.0 - params.damping * dt;
  let v = (stateBuf[base + 1u].xyz + params.gravity * dt) * damp;
  predBuf[i] = vec4<f32>(pos + v * dt, 0.0);
}

// Passes 2..N+1: one color of the constraint graph, applied to the predictions.
//
// Which color arrives on the z axis, so the whole step is one command buffer with
// nothing written per dispatch. No atomics anywhere in here: the coloring guarantees
// that no two invocations of one dispatch touch the same node, which is what makes
// the parallel solve and the CPU's sequential walk of the same order compute the
// same thing.
@compute @workgroup_size(${SOFT_WORKGROUP_SIZE})
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = batchBuf[gid.z];
  if (gid.x >= batch.y) { return; }
  let e = orderBuf[batch.x + gid.x];
  let a = endsBuf[e * 2u];
  // Both endpoints are in the same island by construction, so testing one is
  // testing the edge. A sleeping island keeps its predictions from the step it
  // fell asleep on, and solving them would move nodes finalize will not write back.
  if (asleepBuf[islandOfNodeBuf[a]] != 0u) { return; }
  let b = endsBuf[e * 2u + 1u];
  let pa = predBuf[a];
  let pb = predBuf[b];
  let d = pb.xyz - pa.xyz;
  let d2 = sumSq(d);
  // Two nodes at exactly the same spot have no direction to push along. The negated
  // comparison also drops a NaN, which is the behaviour the CPU reference has.
  if (!(d2 > 0.0)) { return; }
  let wa = stateBuf[a * 2u].w;
  let wb = stateBuf[b * 2u].w;
  let wSum = wa + wb;
  // Both ends pinned: nothing can move, and this is the only divide in the solver
  // that could see a zero.
  if (!(wSum > 0.0)) { return; }
  let dist = sqrt(d2);
  let invDist = 1.0 / dist;
  let invW = 1.0 / wSum;
  let stiffEff = stiffBuf[e] * params.stiffness;
  let s = stiffEff * ((dist - restBuf[e]) * invDist) * invW;
  predBuf[a] = vec4<f32>(pa.xyz + d * (s * wa), 0.0);
  predBuf[b] = vec4<f32>(pb.xyz - d * (s * wb), 0.0);
}

// Pass N+2: bounds, velocity from the position change, and the speed clamp.
//
// Velocity is derived here rather than integrated in predict, which is the part of
// PBD that makes pinned nodes and wall contacts need no special case: whatever moved
// the node -- gravity, a constraint, a reflection -- is what its velocity ends up
// being. The clamp comes last and touches velocity only; clamping the position too
// would fight the constraints that just put it there.
@compute @workgroup_size(${SOFT_WORKGROUP_SIZE})
fn finalize(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let wg = wid.x;
  let island = islandOfWgBuf[wg];
  if (asleepBuf[island] != 0u) { return; }
  let i = nodeOrderBuf[wg * WG + lid.x];
  if (i == SENTINEL) { return; }
  let base = i * 2u;
  let prev = stateBuf[base];
  let radius = stateBuf[base + 1u].w;
  var p = predBuf[i].xyz;
  var v = (p - prev.xyz) * params.invDt;

  // Same shape as the particle integrator: the counting mode is the explicit one
  // and reflect is the else, so an encoding the flag bits cannot express lands in
  // the bounded branch instead of flying out of the box uncounted.
  if (boundsMode() == BOUNDS_NONE) {
    let inside = all(p >= params.boundsMin) && all(p <= params.boundsMax);
    if (!inside) { atomicAdd(&statsBuf[STAT_ESCAPED], 1u); }
  } else {
    let rest = params.restitution;
    let lo = params.boundsMin + vec3<f32>(radius);
    let hi = params.boundsMax - vec3<f32>(radius);
    if (p.x < lo.x) {
      p.x = lo.x + (lo.x - p.x) * rest;
      v.x = -v.x * rest;
    } else if (p.x > hi.x) {
      p.x = hi.x - (p.x - hi.x) * rest;
      v.x = -v.x * rest;
    }
    if (p.y < lo.y) {
      p.y = lo.y + (lo.y - p.y) * rest;
      v.y = -v.y * rest;
    } else if (p.y > hi.y) {
      p.y = hi.y - (p.y - hi.y) * rest;
      v.y = -v.y * rest;
    }
    if (p.z < lo.z) {
      p.z = lo.z + (lo.z - p.z) * rest;
      v.z = -v.z * rest;
    } else if (p.z > hi.z) {
      p.z = hi.z - (p.z - hi.z) * rest;
      v.z = -v.z * rest;
    }
    // A reflection can overshoot to the far wall when one step's penetration
    // exceeds the box. The clamp keeps "escaped === 0" a structural invariant under
    // reflect instead of a property of the numbers that happened to arrive.
    p = clamp(p, lo, hi);
  }

  let maxSpeed2 = params.maxSpeed * params.maxSpeed;
  var s2 = sumSq(v);
  if (s2 > maxSpeed2) {
    v = v * (params.maxSpeed / sqrt(s2));
    s2 = maxSpeed2;
  }
  stateBuf[base] = vec4<f32>(p, prev.w);
  stateBuf[base + 1u] = vec4<f32>(v, radius);
  // One integer max for the step and one for the island. Both are exact no matter
  // what order the driver schedules them in, because a max is.
  let bits = bitcast<u32>(s2);
  atomicMax(&statsBuf[STAT_MAX_SPEED_SQ], bits);
  atomicMax(&islandSpeedBuf[island], bits);
}

// Pass N+3: the largest relative constraint error over every edge.
//
// Per constraint rather than per node, over the whole graph and with no sleep check:
// a sleeping island contributes its frozen error, which is the honest number, and
// reading the final positions rather than the predictions is what makes that true.
@compute @workgroup_size(${SOFT_WORKGROUP_SIZE})
fn measure(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= params.constraintCount) { return; }
  let pa = stateBuf[endsBuf[e * 2u] * 2u].xyz;
  let pb = stateBuf[endsBuf[e * 2u + 1u] * 2u].xyz;
  let d = pb - pa;
  let d2 = sumSq(d);
  if (!(d2 > 0.0)) { return; }
  let r = restBuf[e];
  let err = abs(sqrt(d2) - r) / r;
  atomicMax(&statsBuf[STAT_MAX_ERROR], bitcast<u32>(err));
}

// Pass N+4: put an island to sleep once it has been quiet for sleepAfter steps.
//
// One invocation per island, which is why asleepBuf and quietBuf need no atomics:
// each word has exactly one writer. Per island rather than per node because a cloth
// is one body -- a node that stops while its neighbours keep pulling is not at rest
// -- and because that is what makes the saving real: predict, solve and finalize all
// drop the whole workgroup, not half its lanes.
@compute @workgroup_size(${SOFT_WORKGROUP_SIZE})
fn sleep_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.islandCount) { return; }
  // Read and clear in one operation. Clearing here rather than at the top of the
  // step is what lets the buffer start zero-initialised and still mean "every
  // island is awake": a fresh buffer is all zeros, zero means awake, and the first
  // sleep pass leaves it in the state the second step needs.
  let sq = bitcast<f32>(atomicExchange(&islandSpeedBuf[k], 0u));
  if (!hasFlag(FLAG_SLEEP)) { return; }
  if (asleepBuf[k] != 0u) {
    atomicAdd(&statsBuf[STAT_SLEEPING], 1u);
    return;
  }
  if (sq < params.sleepThresholdSq) {
    let q = quietBuf[k] + 1u;
    quietBuf[k] = q;
    if (q >= params.sleepAfter) {
      asleepBuf[k] = 1u;
      atomicAdd(&statsBuf[STAT_SLEEPING], 1u);
    }
  } else {
    quietBuf[k] = 0u;
  }
}

// Pass N+5: the tight xyz array the renderer reads.
//
// Over 0..count rather than over the padded node order, and with no sleep check:
// every node is drawn every frame, and a sleeping island's positions are frozen, not
// absent. This is the only buffer that crosses the compute/graphics boundary, and it
// is byte for byte what CpuSoftSystem.copyPublishedTo writes.
@compute @workgroup_size(${SOFT_WORKGROUP_SIZE})
fn publish(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let p = stateBuf[i * 2u].xyz;
  publishBuf[i * 3u] = p.x;
  publishBuf[i * 3u + 1u] = p.y;
  publishBuf[i * 3u + 2u] = p.z;
}
`;
  return cached;
}

/**
 * The GPU soft-body backend: the same solver as `softCpu.ts`, running as six
 * kernels and `5 + iterations * colors` dispatches per step.
 *
 * # What has to stay true
 *
 * M4's acceptance criterion is that island grouping and constraint coloring pass
 * a determinism check against the CPU reference, so this file is written to make
 * agreement structural rather than tested-into-existence:
 *
 * - The dispatch **order** comes from `SOFT_KERNELS` and the dispatch **basis**
 *   from `SOFT_KERNEL_DISPATCH`, both in `softWgsl.ts`. A kernel added to the
 *   shader shows up in the chain without a second edit here.
 * - The decomposition is `buildSoftLayout`, the same call the CPU reference
 *   makes on the same mesh, so `plan` is field-for-field the CPU tier's `plan`.
 *   The uploads are straight copies of the arrays that layout produced: no
 *   repack, no permutation, nothing to get subtly wrong.
 * - The uniform is packed by `writeSoftParams`, the one packer both tiers read.
 * - The state buffer is byte-for-byte `SoftMesh.data`: two `vec4`s per node.
 *   Upload is one `writeBuffer` and `readback()` is one copy.
 * - A color is chosen by its bind group, not by a dispatch dimension. Each `solve`
 *   runs with a group-0 binding whose view of the batch table is one slot wide, so
 *   `batchBuf[0u]` inside the kernel is that color's `(base, count)` and no other
 *   color's pair is in range to be read by mistake.
 *
 * # Why there is no ping-pong
 *
 * The particle integrator reads one state buffer and writes another, because
 * every particle is touched by every kernel and a read-write buffer would race
 * with itself. The soft-body solver is different: `solve` is dispatched one
 * color at a time, and a color is by construction a set of constraints that
 * share no nodes, so no two invocations of one dispatch write the same `vec4`.
 * `predBuf` is the only scratch, and it too is written by disjoint invocations.
 * One state buffer, one group-1 binding, built once.
 *
 * # What is deliberately not true
 *
 * `deterministic` is `false`. The reductions in `finalize` and `measure` use
 * `atomicMax` (exact, but a driver may still contract `a * b + c` into an fma)
 * and WGSL leaves the association of the constraint solve to the color order,
 * which is fixed -- but the *rounding* is not, across adapters. Parity against
 * `softCpu.ts` is asserted with a tolerance in `e2e/soft_gpu.spec.ts`.
 *
 * `raceFree` is `true`, and that is the stronger claim this layer can make: the
 * same seed on the same device produces the same bytes, twice, because nothing
 * in the chain has two writers for one word. That is a property the particle
 * layer does not have, and it is what the coloring pass buys.
 *
 * # Statistics
 *
 * Reading five words back costs a buffer map, which serialises the queue, so
 * `step()` never reads anything. `readStats()` polls the counters, `readback()`
 * pulls the whole state into `mesh`, and `stats()` reports whatever the last
 * poll produced. A frame loop that only draws pays for no readback at all.
 */

import {
  ComputeContext,
  rawBuffer,
  submitCopy,
  type ComputeBuffer,
  type ComputeBufferView,
  type ComputeDispatch,
  type ComputeProgram,
  type ComputeResource,
} from './compute.js';
import type { GpuBufferLike, SharedDevice } from './device.js';
import type { Bounds } from './particleField.js';
import { SOFT_WORKGROUP_SIZE, softWorkgroups } from './softIslands.js';
import { SOFT_BYTES, type SoftMesh } from './softMesh.js';
import {
  SOFT_PARAMS_BYTES,
  assertMeshFits,
  buildSoftLayout,
  resolveSoftOptions,
  writeSoftParams,
  type SoftLayout,
  type SoftParamsFrame,
} from './softOptions.js';
import type {
  ResolvedSoftOptions,
  SoftPlan,
  SoftSimOptions,
  SoftStepStats,
  SoftSystem,
} from './softTypes.js';
import {
  SOFT_BINDINGS,
  SOFT_BATCH_STRIDE_BYTES,
  SOFT_BATCH_U32_PER_COLOR,
  SOFT_ENDS_U32_PER_CONSTRAINT,
  SOFT_GROUP_STATIC,
  SOFT_GROUP_STATE,
  SOFT_KERNELS,
  SOFT_KERNEL_DISPATCH,
  SOFT_ORDER_U32_PER_CONSTRAINT,
  SOFT_PUBLISH_FLOATS_PER_NODE,
  SOFT_PRED_VECS_PER_NODE,
  SOFT_STATE_FLOATS_PER_NODE,
  SOFT_STAT_WORDS,
  SOFT_STAT_WORD,
  SOFT_STORAGE_BINDINGS,
  softBindingsForGroup,
  softShaderSource,
  softSolveDispatch,
  type SoftKernelDispatch,
} from './softWgsl.js';

const BYTES_PER_VEC4 = 16;
const BYTES_PER_U32 = 4;

/** Words from one color's slot in `batchBuf` to the next. */
const SOFT_BATCH_STRIDE_WORDS = SOFT_BATCH_STRIDE_BYTES / BYTES_PER_U32;

/** Bytes one color's slot exposes to the kernel: the `(base, count)` pair. */
const SOFT_BATCH_SLOT_BYTES = SOFT_BATCH_U32_PER_COLOR * BYTES_PER_U32;

/**
 * The binding `batchBuf` occupies, read out of the table rather than written down.
 *
 * It is the one binding a `solve` dispatch binds differently from the rest, so the
 * override has to name it, and naming it by index in two places is how a renumbered
 * binding ends up overriding the wrong one.
 */
const SOFT_BATCH_BINDING = SOFT_BINDINGS.find((b) => b.name === 'batchBuf')!.binding;

/**
 * The label a buffer gets, derived from the binding name in `SOFT_BINDINGS`.
 *
 * Derived rather than written down a second time: a binding renamed in the WGSL
 * table renames the buffer, the bind group follows, and a spec that looks for
 * the old label fails with the new one in the message. A hand-written label map
 * would instead keep the old name and bind the wrong buffer.
 */
function bufferLabel(bindingName: string): string {
  return bindingName.replace(/Buf$/, '');
}

/**
 * No buffer may be zero bytes, and an empty graph is legal.
 *
 * `emptyConstraints()` produces a mesh whose every per-edge array is empty, and
 * `checkedBytes` in `compute.ts` refuses a non-positive size because a
 * zero-length `GPUBuffer` is invalid in every implementation. Four bytes for a
 * buffer no invocation reads is cheaper than a special case at each of the nine
 * dispatch sites, and it stays visible in the budget rather than hidden in a
 * `Math.max` nobody reads.
 */
function atLeastOneWord(bytes: number): number {
  return Math.max(BYTES_PER_U32, bytes);
}

/**
 * Bytes for the batch table: one padded slot a color, and never fewer than one slot.
 *
 * The floor here is a slot and not `atLeastOneWord`'s four bytes. `batchBuf` is a
 * runtime-sized `array<vec2<u32>>`, so a binding's size has to be a whole number of
 * elements and a whole number of slots on top of that -- and an empty graph still
 * binds the buffer, because the group it lives in is the group the five fixed
 * kernels share. Four bytes is neither, and would fail at `createBindGroup` on a
 * real device rather than at a check that names the buffer.
 */
function batchSlotBytes(colors: number): number {
  return Math.max(1, colors) * SOFT_BATCH_STRIDE_BYTES;
}

/** Bytes each buffer needs, so a caller can refuse a mesh that will not fit. */
export interface SoftGpuBudget {
  readonly state: number;
  readonly pred: number;
  readonly publish: number;
  readonly nodeOrder: number;
  readonly islandOfWg: number;
  readonly islandOfNode: number;
  readonly order: number;
  readonly ends: number;
  readonly rest: number;
  readonly stiff: number;
  readonly batch: number;
  /** One of the three per-island sleep words. All three are this size. */
  readonly sleep: number;
  readonly stats: number;
  readonly params: number;
  /** The largest single buffer, which is what a binding-size limit constrains. */
  readonly largest: number;
  /** Everything at once, including all three sleep buffers. */
  readonly total: number;
}

/**
 * Size every buffer from the plan and nothing else.
 *
 * Taking `SoftPlan` rather than a mesh is the point: the plan is the object both
 * tiers agree on field by field, so a budget computed from it cannot describe a
 * different decomposition from the one the kernels run over.
 */
export function softGpuBudget(plan: SoftPlan): SoftGpuBudget {
  const nodes = plan.nodes;
  const constraints = plan.constraints;
  const colors = plan.colors;
  const islands = plan.islands;
  const workgroups = plan.nodeWorkgroups;
  const sleep = atLeastOneWord(islands * BYTES_PER_U32);
  const sizes = {
    state: atLeastOneWord(nodes * SOFT_BYTES),
    pred: atLeastOneWord(nodes * SOFT_PRED_VECS_PER_NODE * BYTES_PER_VEC4),
    publish: atLeastOneWord(nodes * SOFT_PUBLISH_FLOATS_PER_NODE * BYTES_PER_U32),
    nodeOrder: atLeastOneWord(workgroups * SOFT_WORKGROUP_SIZE * BYTES_PER_U32),
    islandOfWg: atLeastOneWord(workgroups * BYTES_PER_U32),
    islandOfNode: atLeastOneWord(nodes * BYTES_PER_U32),
    order: atLeastOneWord(constraints * SOFT_ORDER_U32_PER_CONSTRAINT * BYTES_PER_U32),
    ends: atLeastOneWord(constraints * SOFT_ENDS_U32_PER_CONSTRAINT * BYTES_PER_U32),
    rest: atLeastOneWord(constraints * BYTES_PER_U32),
    stiff: atLeastOneWord(constraints * BYTES_PER_U32),
    batch: batchSlotBytes(colors),
    sleep,
    stats: SOFT_STAT_WORDS * BYTES_PER_U32,
    params: SOFT_PARAMS_BYTES,
  };
  const values = Object.values(sizes);
  return {
    ...sizes,
    largest: Math.max(...values),
    // Three buffers share the `sleep` size: asleep, quiet and islandSpeed.
    total: values.reduce((sum, bytes) => sum + bytes, 0) + sleep * 2,
  };
}

export interface GpuSoftSystemOptions {
  /** The device to run on. The caller acquired it; this system retains its own. */
  shared: SharedDevice;
  mesh: SoftMesh;
  options?: SoftSimOptions;
}

export async function createGpuSoftSystem(
  options: GpuSoftSystemOptions,
): Promise<GpuSoftSystem> {
  return GpuSoftSystem.create(options);
}

export class GpuSoftSystem implements SoftSystem {
  readonly name = 'gpu';
  /** fma contraction and adapter-specific `sqrt` rounding. See the file header. */
  readonly deterministic = false;
  /** One writer per word everywhere in the chain, which the coloring guarantees. */
  readonly raceFree = true;

  readonly mesh: SoftMesh;
  readonly options: ResolvedSoftOptions;
  readonly shared: SharedDevice;
  readonly context: ComputeContext;
  readonly program: ComputeProgram;
  /** Islands, coloring and plan: the same decomposition the CPU tier builds. */
  readonly layout: SoftLayout;
  readonly plan: SoftPlan;
  readonly budget: SoftGpuBudget;

  /** The buffer the renderer reads. Copied out with `copyPublishedTo`. */
  readonly publish: ComputeBuffer;

  private readonly buffers = new Map<string, ComputeBuffer>();
  private readonly paramsBytes = new ArrayBuffer(SOFT_PARAMS_BYTES);
  private readonly stateBuffer: ComputeBuffer;
  private readonly statsBuffer: ComputeBuffer;
  private readonly sleepBuffers: readonly ComputeBuffer[];
  private readonly staticGroup: unknown;
  private readonly stateGroup: unknown;
  /**
   * One group-0 bind group a color, each seeing exactly that color's slot of the
   * batch table. Index `c` is the group for color `c`, asserted at construction.
   */
  private readonly solveGroups: readonly unknown[];

  private stepsTaken = 0;
  private lastStats: SoftStepStats;
  private disposedFlag = false;

  /**
   * Allocate, compile, bind, upload and publish frame zero.
   *
   * Async because compilation info is async, and a program built without reading
   * it is a program whose WGSL typo surfaces as a validation error at the first
   * dispatch instead of as a line number.
   */
  static async create({ shared, mesh, options }: GpuSoftSystemOptions): Promise<GpuSoftSystem> {
    if (!mesh || !Number.isInteger(mesh.count) || !(mesh.count > 0)) {
      throw new TypeError('GpuSoftSystem needs a SoftMesh with a positive count');
    }
    if (!shared.usable) {
      throw new Error(`the shared device is ${shared.lost ? 'lost' : 'destroyed'}`);
    }
    const resolved = resolveSoftOptions(options);
    assertMeshFits(mesh);
    const layout = buildSoftLayout(mesh, resolved);
    // Checked before anything is allocated: the failure mode it prevents is a
    // pipeline layout that fails validation at creation time, on a device that
    // has already been asked for fifteen buffers.
    assertStorageBuffers(shared);
    const budget = softGpuBudget(layout.plan);
    assertFits(shared, budget);

    const context = new ComputeContext(shared);
    try {
      const program = await context.program({
        label: 'soft',
        code: softShaderSource(),
        entryPoints: SOFT_KERNELS,
        bindings: SOFT_BINDINGS,
      });
      return new GpuSoftSystem(shared, context, program, mesh, resolved, layout, budget);
    } catch (error) {
      // A half-built system leaks device memory, which on a page that then falls
      // back to the CPU reference is a leak that outlives the reason for it.
      context.destroy();
      throw error;
    }
  }

  private constructor(
    shared: SharedDevice,
    context: ComputeContext,
    program: ComputeProgram,
    mesh: SoftMesh,
    resolved: ResolvedSoftOptions,
    layout: SoftLayout,
    budget: SoftGpuBudget,
  ) {
    this.shared = shared.retain();
    this.context = context;
    this.program = program;
    this.mesh = mesh;
    this.options = resolved;
    this.layout = layout;
    this.plan = layout.plan;
    this.budget = budget;

    // One pass over the binding table makes every buffer, so the set of buffers
    // and the set of bindings cannot disagree about a name or a size.
    for (const binding of SOFT_BINDINGS) {
      const label = bufferLabel(binding.name);
      const bytes = this.bytesFor(label, budget);
      const buffer =
        binding.bufferType === 'uniform'
          ? context.uniformBuffer(bytes, label)
          : context.storageBuffer(bytes, label);
      this.buffers.set(label, buffer);
    }
    this.stateBuffer = this.buffers.get('state')!;
    this.statsBuffer = this.buffers.get('stats')!;
    this.publish = this.buffers.get('publish')!;
    this.sleepBuffers = ['asleep', 'quiet', 'islandSpeed'].map((label) => this.buffers.get(label)!);

    this.staticGroup = this.bindGroup(SOFT_GROUP_STATIC, 'soft:static');
    this.stateGroup = this.bindGroup(SOFT_GROUP_STATE, 'soft:state');
    // Built once and reused for every iteration of every step: a bind group is a
    // device object, and a step records `iterations * colors` solve dispatches.
    this.solveGroups = layout.coloring.batches.map((batch, color) => {
      if (batch.color !== color) {
        throw new Error(
          `batch ${color} carries color ${batch.color}; the batch table is not tiled by color`,
        );
      }
      return this.bindGroup(SOFT_GROUP_STATIC, `soft:solve${color}`, this.batchSlot(color));
    });

    this.uploadMesh();
    this.lastStats = {
      escaped: 0,
      maxSpeed: mesh.maxSpeed(),
      maxConstraintError: mesh.maxConstraintError(),
      awakeIslands: layout.islands.islands,
      sleepingIslands: 0,
      kineticEnergy: mesh.kineticEnergy(),
    };

    // Frame zero has to be drawable before the first step, which is what the CPU
    // reference's constructor does with `publishPass()`. Without this a renderer
    // that draws before stepping reads a zero-initialised publish buffer and
    // puts every node at the origin.
    this.uploadParams(resolved.fixedDt);
    program.submit(
      [
        {
          entryPoint: 'publish',
          workgroups: softWorkgroups(mesh.count),
          groups: this.groups(),
        },
      ],
      'soft publish 0',
    );
  }

  get fixedDt(): number {
    return this.options.fixedDt;
  }

  get count(): number {
    return this.mesh.count;
  }

  get bounds(): Bounds {
    return this.mesh.bounds;
  }

  get steps(): number {
    return this.stepsTaken;
  }

  get time(): number {
    return this.stepsTaken * this.options.fixedDt;
  }

  /** True once the device behind this system is gone. `step()` throws then. */
  get lost(): boolean {
    return !this.shared.usable;
  }

  get disposed(): boolean {
    return this.disposedFlag;
  }

  /**
   * One fixed step: pack the uniform, upload it, record the whole chain.
   *
   * Nothing is awaited, because a step that waited for the GPU would run at
   * readback speed rather than at dispatch speed. The queue orders this step's
   * dispatches before the next step's uniform write, so a caller can step in a
   * tight loop and still get a serial simulation.
   */
  step(dt: number = this.options.fixedDt): void {
    this.assertLive();
    if (!Number.isFinite(dt) || !(dt > 0)) {
      throw new RangeError(`dt must be finite and positive, got ${dt}`);
    }
    this.uploadParams(dt);
    this.program.submit(this.chain(), `soft step ${this.stepsTaken}`);
    this.stepsTaken++;
  }

  advance(n: number): void {
    if (!Number.isInteger(n) || n < 0) {
      throw new RangeError(`advance needs a non-negative integer, got ${n}`);
    }
    for (let k = 0; k < n; k++) this.step();
  }

  /**
   * The last polled counters.
   *
   * Before the first `readStats()` this holds the seeded mesh's own speed, error
   * and energy with zero counters, which is the honest answer for a system that
   * has not stepped yet and better than zeros that look like a clean run.
   */
  stats(): SoftStepStats {
    return this.lastStats;
  }

  /**
   * Read the counter words off the GPU and cache them.
   *
   * `maxSpeedSq` and `maxConstraintError` arrive as the bit patterns of f32s,
   * because both kernels take their maximum with `atomicMax` on integers:
   * non-negative floats sort the same way as their bits, so the max is exact and
   * needs no atomic on floats. `awakeIslands` is derived rather than counted, so
   * the two island numbers always sum to `plan.islands`.
   */
  async readStats(): Promise<SoftStepStats> {
    this.assertLive();
    const [words] = await this.context.readBytes([
      { buffer: this.statsBuffer, bytes: SOFT_STAT_WORDS * BYTES_PER_U32 },
    ]);
    const ints = new Uint32Array(words.buffer, words.byteOffset, SOFT_STAT_WORDS);
    const floats = new Float32Array(words.buffer, words.byteOffset, SOFT_STAT_WORDS);
    const speedSq = floats[SOFT_STAT_WORD.maxSpeedSq]!;
    const error = floats[SOFT_STAT_WORD.maxConstraintError]!;
    const sleeping = ints[SOFT_STAT_WORD.sleepingIslands]!;
    this.lastStats = {
      escaped: ints[SOFT_STAT_WORD.escaped]!,
      maxSpeed: Number.isFinite(speedSq) && speedSq > 0 ? Math.sqrt(speedSq) : 0,
      maxConstraintError: Number.isFinite(error) && error > 0 ? error : 0,
      awakeIslands: this.layout.islands.islands - sleeping,
      sleepingIslands: sleeping,
      // Only `readback()` can refresh this: it is a sum over the whole mesh and
      // no kernel tracks it. Carrying the previous value forward keeps a counter
      // poll from silently reporting zero energy.
      kineticEnergy: this.lastStats.kineticEnergy,
    };
    return this.lastStats;
  }

  /**
   * Copy the current GPU state into `mesh.data`, so `digest()`, `kineticEnergy`
   * and any CPU-side inspection see the nodes the GPU is simulating.
   *
   * Deliberately not called by `step()`: it is a full readback, and a parity
   * check is the only thing that needs it.
   */
  async readback(): Promise<SoftMesh> {
    this.assertLive();
    const [bytes] = await this.context.readBytes([
      { buffer: this.stateBuffer, bytes: this.count * SOFT_BYTES },
    ]);
    this.mesh.data.set(
      new Float32Array(bytes.buffer, bytes.byteOffset, this.count * SOFT_STATE_FLOATS_PER_NODE),
    );
    this.lastStats = { ...this.lastStats, kineticEnergy: this.mesh.kineticEnergy() };
    return this.mesh;
  }

  /**
   * Blit the published positions into a buffer the caller owns -- in practice the
   * one behind a three.js position attribute, reached through
   * `renderer.backend.get(attribute).buffer`.
   *
   * @returns bytes copied, or 0 when there was nothing to copy.
   */
  copyPublishedTo(target: ComputeBuffer | GpuBufferLike, bytes = this.budget.publish): number {
    this.assertLive();
    const size = Math.min(bytes, this.budget.publish);
    const copies = submitCopy(
      this.shared.device,
      [{ from: this.publish, to: rawBuffer(target), bytes: size }],
      'soft publish',
    );
    return copies > 0 ? size : 0;
  }

  /**
   * `hex:count` over `mesh.data`, i.e. over the last `readback()`.
   *
   * A comparison tool, not a replay key: this backend is not deterministic, so
   * two runs of the same seed may differ in the last bits across adapters. On one
   * device it is stable, which is what `raceFree` promises and what
   * `e2e/soft_gpu.spec.ts` asserts.
   */
  digest(): string {
    return this.mesh.digest();
  }

  /**
   * Re-activate every island.
   *
   * Three zero writes rather than one, because the sleep state is three words per
   * island: a flag, a quiet-step counter and this step's speed accumulator. The
   * writes go through the queue, so they are ordered after any step already
   * recorded and before the next one -- the same guarantee the CPU tier gets for
   * free by clearing its arrays in place.
   */
  wake(): void {
    this.assertLive();
    const zeros = new Uint32Array(this.layout.islands.islands);
    for (const buffer of this.sleepBuffers) buffer.write(zeros);
    this.lastStats = {
      ...this.lastStats,
      awakeIslands: this.layout.islands.islands,
      sleepingIslands: 0,
    };
  }

  /** Release the buffers and this system's reference to the shared device. */
  dispose(): void {
    if (this.disposedFlag) return;
    this.disposedFlag = true;
    this.context.destroy();
    // The caller still holds the handle they acquired; this drops only ours.
    this.shared.release();
  }

  private assertLive(): void {
    if (this.disposedFlag) throw new Error('GpuSoftSystem has been disposed');
    if (!this.shared.usable) {
      throw new Error(`the shared device is ${this.shared.lost ? 'lost' : 'destroyed'}`);
    }
  }

  private bytesFor(label: string, budget: SoftGpuBudget): number {
    switch (label) {
      case 'state':
        return budget.state;
      case 'pred':
        return budget.pred;
      case 'publish':
        return budget.publish;
      case 'nodeOrder':
        return budget.nodeOrder;
      case 'islandOfWg':
        return budget.islandOfWg;
      case 'islandOfNode':
        return budget.islandOfNode;
      case 'order':
        return budget.order;
      case 'ends':
        return budget.ends;
      case 'rest':
        return budget.rest;
      case 'stiff':
        return budget.stiff;
      case 'batch':
        return budget.batch;
      case 'asleep':
      case 'quiet':
      case 'islandSpeed':
        return budget.sleep;
      case 'stats':
        return budget.stats;
      case 'params':
        return budget.params;
      default:
        throw new Error(`no budget line for a buffer labelled '${label}'`);
    }
  }

  /**
   * Bind one group, optionally with `batchBuf` narrowed to a single color's slot.
   *
   * `batchSlot` is the whole reason the solver is race-free across colors: with it,
   * the kernel's `batchBuf[0u]` is this dispatch's color and no other pair is in
   * range. Without it -- the group the five fixed kernels use -- the binding is the
   * whole table, which those kernels never read.
   */
  private bindGroup(group: number, label: string, batchSlot?: ComputeBufferView): unknown {
    const resources: Record<number, ComputeResource> = {};
    for (const binding of softBindingsForGroup(group)) {
      const buffer = this.buffers.get(bufferLabel(binding.name));
      if (!buffer) {
        throw new Error(`binding ${binding.name} has no buffer; the labels disagree`);
      }
      resources[binding.binding] = buffer;
    }
    if (batchSlot) resources[SOFT_BATCH_BINDING] = batchSlot;
    return this.program.bindGroup(group, resources, label);
  }

  /** The view of `batchBuf` that holds color `color`'s `(base, count)` pair. */
  private batchSlot(color: number): ComputeBufferView {
    return {
      buffer: this.buffers.get('batch')!,
      offset: color * SOFT_BATCH_STRIDE_BYTES,
      size: SOFT_BATCH_SLOT_BYTES,
    };
  }

  /**
   * Upload the shape of the problem.
   *
   * Every array here is the one `buildSoftLayout` or the mesh already produced:
   * the padded node order, the two island maps, the colored edge order, the graph
   * and its per-edge constants. Nothing is permuted on the way in, because the
   * coloring is applied by `orderBuf` inside the shader.
   *
   * The sleep buffers and the counters are deliberately absent. WebGPU
   * zero-initialises a buffer, zero means awake and zero means "no error yet", so
   * a fresh allocation is exactly the state the first step needs.
   */
  private uploadMesh(): void {
    const { islands, coloring } = this.layout;
    const { constraints } = this.mesh;
    // One slot a color, payload at the front of it and zeros in the padding. The
    // padding is not waste: it is what makes `color * SOFT_BATCH_STRIDE_BYTES` a
    // legal bind-group offset, and what keeps a color from reading its neighbour.
    const batches = new Uint32Array(coloring.batches.length * SOFT_BATCH_STRIDE_WORDS);
    coloring.batches.forEach((batch, color) => {
      const slot = color * SOFT_BATCH_STRIDE_WORDS;
      batches[slot] = batch.base;
      batches[slot + 1] = batch.count;
    });
    // The state buffer *is* the mesh's own layout, so the seed is one write of
    // `mesh.data` and the readback is one copy into it.
    this.upload('state', this.mesh.data);
    this.upload('nodeOrder', islands.nodeOrder);
    this.upload('islandOfWg', islands.islandOfWorkgroup);
    this.upload('islandOfNode', islands.islandOfNode);
    this.upload('order', coloring.order);
    this.upload('ends', constraints.ends);
    this.upload('rest', constraints.rest);
    this.upload('stiff', constraints.stiffness);
    this.upload('batch', batches);
  }

  private upload(label: string, data: ArrayBufferView): void {
    // An empty graph has nothing to say, and its buffers are the padded four bytes
    // `atLeastOneWord` gave them.
    if (data.byteLength === 0) return;
    const buffer = this.buffers.get(label);
    if (!buffer) throw new Error(`no buffer labelled '${label}' to upload into`);
    buffer.write(data);
  }

  private uploadParams(dt: number): void {
    const frame: SoftParamsFrame = {
      dt,
      count: this.count,
      islandCount: this.layout.islands.islands,
      paddedNodes: this.layout.islands.paddedNodes,
      constraintCount: this.mesh.constraints.count,
      sleepAfter: this.options.sleepAfter,
      bounds: this.mesh.bounds,
    };
    writeSoftParams(this.paramsBytes, this.options, frame);
    this.buffers.get('params')!.write(this.paramsBytes);
  }

  /** Both groups, for every dispatch: nothing here swaps state mid-step. */
  private groups(): Readonly<Record<number, unknown>> {
    return { 0: this.staticGroup, 1: this.stateGroup };
  }

  /**
   * The kernel chain for one step, in `SOFT_KERNELS` order.
   *
   * `solve` is the only kernel that expands: `iterations` copies of one dispatch
   * per color, each recorded with the bind group that exposes only that color's
   * batch. That is what makes the whole step a single command buffer with no
   * uniform rewrite between batches -- the alternative is `iterations * colors`
   * submits, and a submit is the expensive part of a frame that dispatches two
   * hundred times.
   *
   * Nothing is dropped for an option being off, unlike the particle chain: a
   * sleeping island is skipped inside the kernels by reading `asleepBuf`, and
   * `measure` must run over the whole graph either way so a frozen island still
   * reports its error.
   */
  private chain(): ComputeDispatch[] {
    const groups = this.groups();
    const chain: ComputeDispatch[] = [];
    for (const kernel of SOFT_KERNELS) {
      const basis = SOFT_KERNEL_DISPATCH[kernel];
      if (basis !== 'batch') {
        chain.push({ entryPoint: kernel, workgroups: this.sizing(basis), groups });
        continue;
      }
      for (let iteration = 0; iteration < this.options.iterations; iteration++) {
        for (const batch of this.layout.coloring.batches) {
          chain.push({
            entryPoint: kernel,
            workgroups: softSolveDispatch(batch),
            // The colors' tiling is asserted in the constructor, so this index is
            // total; an unbound group here would be a silent no-op dispatch.
            groups: { 0: this.solveGroups[batch.color], 1: this.stateGroup },
          });
        }
      }
    }
    return chain;
  }

  /** Workgroups for a kernel whose basis is a single count. */
  private sizing(basis: SoftKernelDispatch): number {
    switch (basis) {
      case 'nodeWorkgroups':
        return this.layout.islands.workgroups;
      case 'constraints':
        return softWorkgroups(this.mesh.constraints.count);
      case 'islands':
        return softWorkgroups(this.layout.islands.islands);
      case 'count':
        return softWorkgroups(this.count);
      case 'batch':
        // Unreachable from `chain()`, and a lie if reached: one number cannot
        // describe a dispatch that also has to say which color it is solving.
        throw new Error("'batch' sizing is softSolveDispatch's, not a single workgroup count");
    }
  }
}

/**
 * Refuse a device that cannot bind this pipeline's storage buffers.
 *
 * Fifteen, against a WebGPU baseline of eight. `device.ts` asks the adapter for
 * its own value, so a device that reports fewer genuinely has fewer, and the
 * alternative to this check is a `createPipelineLayout` that fails validation
 * with a message naming a binding rather than a tier. Throwing here is what lets
 * `soft.ts` turn it into a downgrade to the CPU reference with a reason attached.
 *
 * A limit reported as 0 was never reported at all, and guessing a ceiling there
 * would refuse adapters that could have run the solver.
 */
function assertStorageBuffers(shared: SharedDevice): void {
  const reported = shared.info.limits.maxStorageBuffersPerShaderStage;
  if (reported > 0 && reported < SOFT_STORAGE_BINDINGS) {
    throw new RangeError(
      `the soft-body pipeline binds ${SOFT_STORAGE_BINDINGS} storage buffers in one stage, ` +
        `above this device's maxStorageBuffersPerShaderStage=${reported}`,
    );
  }
}

/**
 * Refuse a mesh whose largest buffer will not bind.
 *
 * Checked before allocation, because the alternative is a `createBuffer` that
 * succeeds and a bind group that fails validation at dispatch time -- an error
 * with no numbers in it, on a page that has already built everything.
 */
function assertFits(shared: SharedDevice, budget: SoftGpuBudget): void {
  const { maxStorageBufferBindingSize, maxBufferSize } = shared.info.limits;
  const cap =
    maxBufferSize > 0
      ? Math.min(maxStorageBufferBindingSize, maxBufferSize)
      : maxStorageBufferBindingSize;
  if (cap > 0 && budget.largest > cap) {
    throw new RangeError(
      `the largest buffer here is ${budget.largest} bytes, above this device's ${cap}-byte storage limit`,
    );
  }
}

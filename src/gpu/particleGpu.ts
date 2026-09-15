/**
 * The GPU particle backend: the same simulation as `particleCpu.ts`, running as
 * six compute dispatches per step.
 *
 * # What has to stay true
 *
 * The plan's M3 acceptance criterion is that the GPU backend and the CPU
 * reference backend agree, so this file is written to make agreement the default
 * rather than a hope:
 *
 * - The dispatch **order** comes from `PARTICLE_KERNELS` and the dispatch
 *   **sizing** from `KERNEL_DISPATCH`, both in `particleWgsl.ts`. Nothing here
 *   re-derives either, so a kernel added to the shader shows up in the chain
 *   without a second edit.
 * - The uniform is packed by `writeParams`, the same function the CPU spec reads
 *   back. Two packers would drift on the first new field.
 * - The state buffers are byte-for-byte `ParticleField.data`: two `vec4`s per
 *   particle, `(pos, radius)` then `(vel, mass)`. Upload is one `writeBuffer`
 *   and readback is one copy, with no repack step in either direction to get
 *   wrong.
 *
 * # What is deliberately not true
 *
 * `deterministic` is `false`. The broadphase fills buckets with `atomicAdd` and
 * the n-body reduction order belongs to the driver, so two runs of the same seed
 * can differ in the last bits. Parity against the CPU backend is asserted with a
 * tolerance; exact digests stay a CPU-only property.
 *
 * # Statistics
 *
 * Reading five words back from the GPU costs a buffer map, which serialises the
 * queue. So `step()` never reads anything: it records commands and returns.
 * `readStats()` polls the counters, `readback()` pulls the whole state into
 * `field`, and `stats()` reports whatever the last poll produced. A caller that
 * wants a fresh number asks for it, and a caller that does not is not paying for
 * one every frame.
 */

import {
  ComputeContext,
  PingPong,
  rawBuffer,
  submitCopy,
  type ComputeBuffer,
  type ComputeDispatch,
  type ComputeProgram,
} from './compute.js';
import type { GpuBufferLike, SharedDevice } from './device.js';
import { PARTICLE_BYTES, type Bounds, type ParticleField } from './particleField.js';
import { nextPow2 } from './particleHash.js';
import {
  PARAMS_BYTES,
  assertFieldFits,
  effectiveCellSize,
  resolveParticleOptions,
  writeParams,
  type ParamsFrame,
} from './particleOptions.js';
import type {
  ParticleSimOptions,
  ParticleStepStats,
  ParticleSystem,
  ResolvedParticleOptions,
} from './particleTypes.js';
import {
  ACCEL_VECS_PER_PARTICLE,
  CONTACT_VECS_PER_PARTICLE,
  GROUP_STATIC,
  GROUP_STATE,
  KERNEL_DISPATCH,
  PARTICLE_BINDINGS,
  PARTICLE_KERNELS,
  PUBLISH_FLOATS_PER_PARTICLE,
  STATE_VECS_PER_PARTICLE,
  STAT_WORD,
  STAT_WORDS,
  particleShaderSource,
  workgroupsFor,
} from './particleWgsl.js';

const BYTES_PER_VEC4 = 16;
const FLOATS_PER_PARTICLE = PARTICLE_BYTES / 4;

/** Hash table size for `count` particles: what `SpatialHash` picks, restated. */
export function tableSizeFor(count: number): number {
  return nextPow2(count);
}

/** Bytes each buffer needs, so a caller can refuse a world that will not fit. */
export interface GpuBufferBudget {
  readonly state: number;
  readonly accel: number;
  readonly contact: number;
  readonly hashCounts: number;
  readonly hashSlots: number;
  readonly publish: number;
  readonly stats: number;
  readonly params: number;
  /** The largest single buffer, which is what a binding-size limit constrains. */
  readonly largest: number;
  /** Everything at once, which is what a memory budget constrains. */
  readonly total: number;
}

export function gpuBufferBudget(
  count: number,
  tableSize: number,
  bucketCapacity: number,
): GpuBufferBudget {
  const state = count * STATE_VECS_PER_PARTICLE * BYTES_PER_VEC4;
  const accel = count * ACCEL_VECS_PER_PARTICLE * BYTES_PER_VEC4;
  const contact = count * CONTACT_VECS_PER_PARTICLE * BYTES_PER_VEC4;
  const hashCounts = tableSize * 4;
  const hashSlots = tableSize * bucketCapacity * 4;
  const publish = count * PUBLISH_FLOATS_PER_PARTICLE * 4;
  const stats = STAT_WORDS * 4;
  return {
    state,
    accel,
    contact,
    hashCounts,
    hashSlots,
    publish,
    stats,
    params: PARAMS_BYTES,
    largest: Math.max(state, hashSlots, contact, accel, publish),
    // Two state buffers: the ping-pong pair is the whole point of the layout.
    total: state * 2 + accel + contact + hashCounts + hashSlots + publish + stats + PARAMS_BYTES,
  };
}

export interface GpuParticleSystemOptions {
  /** The device to run on. The caller acquired it; this system retains its own. */
  shared: SharedDevice;
  field: ParticleField;
  options?: ParticleSimOptions;
}

export async function createGpuParticleSystem(
  options: GpuParticleSystemOptions,
): Promise<GpuParticleSystem> {
  return GpuParticleSystem.create(options);
}

export class GpuParticleSystem implements ParticleSystem {
  readonly name = 'gpu';
  /** Atomic broadphase and driver-chosen reduction order. See the file header. */
  readonly deterministic = false;

  readonly field: ParticleField;
  readonly options: ResolvedParticleOptions;
  readonly shared: SharedDevice;
  readonly context: ComputeContext;
  readonly program: ComputeProgram;
  readonly cellSize: number;
  readonly tableSize: number;
  readonly budget: GpuBufferBudget;

  /** The buffer the renderer reads. Copied out with `copyPublishedTo`. */
  readonly publish: ComputeBuffer;

  private readonly paramsBuffer: ComputeBuffer;
  private readonly statsBuffer: ComputeBuffer;
  private readonly paramsBytes = new ArrayBuffer(PARAMS_BYTES);
  private readonly state: PingPong<ComputeBuffer>;
  /** Group 1 as bound on even steps and on odd steps, built once. */
  private readonly stateGroups: readonly [unknown, unknown];
  private readonly staticGroup: unknown;

  private stepsTaken = 0;
  private lastStats: ParticleStepStats;
  private disposedFlag = false;

  /**
   * Allocate, compile, bind and upload.
   *
   * Async because compilation info is async, and a program built without reading
   * it is a program whose WGSL typo surfaces as a validation error at the first
   * dispatch instead of as a line number.
   */
  static async create({
    shared,
    field,
    options,
  }: GpuParticleSystemOptions): Promise<GpuParticleSystem> {
    if (!field || !Number.isInteger(field.count) || !(field.count > 0)) {
      throw new TypeError('GpuParticleSystem needs a ParticleField with a positive count');
    }
    if (!shared.usable) {
      throw new Error(`the shared device is ${shared.lost ? 'lost' : 'destroyed'}`);
    }
    const resolved = resolveParticleOptions(options);
    assertFieldFits(field);
    const tableSize = tableSizeFor(field.count);
    const budget = gpuBufferBudget(field.count, tableSize, resolved.bucketCapacity);
    assertFits(shared, budget);

    const context = new ComputeContext(shared);
    try {
      const program = await context.program({
        label: 'particles',
        code: particleShaderSource(),
        entryPoints: PARTICLE_KERNELS,
        bindings: PARTICLE_BINDINGS,
      });
      return new GpuParticleSystem(shared, context, program, field, resolved, tableSize, budget);
    } catch (error) {
      // A half-built system leaks device memory, which on a page that then falls
      // back to the CPU backend is a leak that outlives the reason for it.
      context.destroy();
      throw error;
    }
  }

  private constructor(
    shared: SharedDevice,
    context: ComputeContext,
    program: ComputeProgram,
    field: ParticleField,
    resolved: ResolvedParticleOptions,
    tableSize: number,
    budget: GpuBufferBudget,
  ) {
    this.shared = shared.retain();
    this.context = context;
    this.program = program;
    this.field = field;
    this.options = resolved;
    this.tableSize = tableSize;
    this.budget = budget;
    this.cellSize = effectiveCellSize(field, resolved);

    this.paramsBuffer = context.uniformBuffer(PARAMS_BYTES, 'params');
    this.statsBuffer = context.storageBuffer(budget.stats, 'stats');
    const first = context.storageBuffer(budget.state, 'state-a');
    const second = context.storageBuffer(budget.state, 'state-b');
    const accel = context.storageBuffer(budget.accel, 'accel');
    const contact = context.storageBuffer(budget.contact, 'contact');
    const hashCounts = context.storageBuffer(budget.hashCounts, 'hashCounts');
    const hashSlots = context.storageBuffer(budget.hashSlots, 'hashSlots');
    this.publish = context.storageBuffer(budget.publish, 'publish');
    this.state = new PingPong(first, second);

    // Two pre-built group-1 bindings rather than one rebuilt per step: a step
    // swaps which of them is bound, and `createBindGroup` in a frame loop is
    // driver work nobody can amortise.
    this.stateGroups = [
      program.bindGroup(GROUP_STATE, { 0: first, 1: second, 2: this.publish }, 'particles:state-a'),
      program.bindGroup(GROUP_STATE, { 1: first, 0: second, 2: this.publish }, 'particles:state-b'),
    ];
    this.staticGroup = program.bindGroup(
      GROUP_STATIC,
      {
        0: this.paramsBuffer,
        1: accel,
        2: contact,
        3: hashCounts,
        4: hashSlots,
        5: this.statsBuffer,
      },
      'particles:static',
    );

    // The field is the seed and nothing more after this: the GPU owns the state
    // from here on, and `field.data` only becomes current again on `readback()`.
    first.write(field.data);
    this.lastStats = {
      contacts: 0,
      escaped: 0,
      hashOverflow: 0,
      maxSpeed: field.maxSpeed(),
      kineticEnergy: field.kineticEnergy(),
    };
  }

  get fixedDt(): number {
    return this.options.fixedDt;
  }

  get count(): number {
    return this.field.count;
  }

  get bounds(): Bounds {
    return this.field.bounds;
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

  /** Workgroups per particle-wide kernel, i.e. what most of the chain runs. */
  get workgroups(): number {
    return workgroupsFor(this.count);
  }

  /**
   * One fixed step: pack the uniform, upload it, record the kernel chain, swap.
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
    this.program.submit(this.chain(), `particles step ${this.stepsTaken}`);
    this.state.swap();
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
   * Before the first `readStats()` this holds the seeded field's own speeds and
   * energies with zero counters, which is the honest answer for a system that has
   * not stepped yet and better than zeros that look like a clean run.
   */
  stats(): ParticleStepStats {
    return this.lastStats;
  }

  /**
   * Read the five counter words off the GPU and cache them.
   *
   * `maxSpeedSq` arrives as the bit pattern of an f32, because the kernel takes
   * its maximum with `atomicMax` on integers: non-negative floats sort the same
   * way as their bits, so the max is exact and needs no atomic on floats.
   */
  async readStats(): Promise<ParticleStepStats> {
    this.assertLive();
    const [words] = await this.context.readBytes([
      { buffer: this.statsBuffer, bytes: STAT_WORDS * 4 },
    ]);
    const ints = new Uint32Array(words.buffer, words.byteOffset, STAT_WORDS);
    const floats = new Float32Array(words.buffer, words.byteOffset, STAT_WORDS);
    const squared = floats[STAT_WORD.maxSpeedSq];
    this.lastStats = {
      contacts: ints[STAT_WORD.contacts],
      escaped: ints[STAT_WORD.escaped],
      hashOverflow: ints[STAT_WORD.overflow],
      maxSpeed: Number.isFinite(squared) && squared > 0 ? Math.sqrt(squared) : 0,
      // Only `readback()` can refresh this: it is a sum over the whole field and
      // the GPU does not track it. Carrying the previous value forward keeps a
      // counter poll from silently reporting zero energy.
      kineticEnergy: this.lastStats.kineticEnergy,
    };
    return this.lastStats;
  }

  /**
   * Copy the current GPU state into `field.data`, so `digest()`, `kineticEnergy`
   * and any CPU-side inspection see the particles the GPU is simulating.
   *
   * Deliberately not called by `step()`: it is a full readback, and a parity
   * check is the only thing that needs it.
   */
  async readback(): Promise<ParticleField> {
    this.assertLive();
    const [bytes] = await this.context.readBytes([
      { buffer: this.state.src, bytes: this.count * PARTICLE_BYTES },
    ]);
    this.field.data.set(new Float32Array(bytes.buffer, bytes.byteOffset, this.count * FLOATS_PER_PARTICLE));
    this.lastStats = { ...this.lastStats, kineticEnergy: this.field.kineticEnergy() };
    return this.field;
  }

  /**
   * Blit the published positions and radii into a buffer the caller owns -- in
   * practice the one behind a three.js `InstancedBufferAttribute`, reached
   * through `renderer.backend.get(attribute).buffer`.
   *
   * @returns bytes copied, or 0 when there was nothing to copy.
   */
  copyPublishedTo(target: ComputeBuffer | GpuBufferLike, bytes = this.budget.publish): number {
    this.assertLive();
    const size = Math.min(bytes, this.budget.publish);
    const copies = submitCopy(
      this.shared.device,
      [{ from: this.publish, to: rawBuffer(target), bytes: size }],
      'particles publish',
    );
    return copies > 0 ? size : 0;
  }

  /**
   * `hex:count` over `field.data`, i.e. over the last `readback()`.
   *
   * A comparison tool, not a replay key: this backend is not deterministic, so
   * two runs of the same seed may differ in the last bits.
   */
  digest(): string {
    return this.field.digest();
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
    if (this.disposedFlag) throw new Error('GpuParticleSystem has been disposed');
    if (!this.shared.usable) {
      throw new Error(`the shared device is ${this.shared.lost ? 'lost' : 'destroyed'}`);
    }
  }

  private uploadParams(dt: number): void {
    const frame: ParamsFrame = {
      dt,
      cellSize: this.cellSize,
      tableSize: this.tableSize,
      bucketCapacity: this.options.bucketCapacity,
      count: this.count,
      bounds: this.field.bounds,
    };
    writeParams(this.paramsBytes, this.options, frame);
    this.paramsBuffer.write(this.paramsBytes);
  }

  /**
   * The kernel chain for one step.
   *
   * `nbody`, `hash_scatter` and `collide` are dropped when their flag is off:
   * each is a no-op in the shader without the flag, and a dispatch is not free
   * even when every invocation returns on its first line. `hash_clear` always
   * runs, because invocation 0 owns the counter reset the readback depends on.
   *
   * `publish` is the one dispatch that binds the *other* state group: it reads
   * `stateSrc`, and by the time it runs the state `integrate` wrote is in what
   * this step called `dst`. Both groups are legal to bind in the same pass
   * because only one is bound to index 1 at a time.
   */
  private chain(): ComputeDispatch[] {
    const perParticle = this.workgroups;
    const running = { 0: this.staticGroup, 1: this.stateGroups[this.state.parity] };
    const published = { 0: this.staticGroup, 1: this.stateGroups[this.state.parity ^ 1] };
    const chain: ComputeDispatch[] = [];
    for (const kernel of PARTICLE_KERNELS) {
      if (kernel === 'nbody' && !this.options.nbody) continue;
      if ((kernel === 'hash_scatter' || kernel === 'collide') && !this.options.collisions) continue;
      const basis = KERNEL_DISPATCH[kernel];
      chain.push({
        entryPoint: kernel,
        workgroups: basis === 'tableSize' ? workgroupsFor(this.tableSize) : perParticle,
        groups: kernel === 'publish' ? published : running,
      });
    }
    return chain;
  }
}

/**
 * Refuse a world whose largest buffer will not bind.
 *
 * Checked before allocation, because the alternative is a `createBuffer` that
 * succeeds and a bind group that fails validation at dispatch time -- an error
 * with no numbers in it, on a page that has already built everything. A limit
 * reported as 0 was never reported at all, and guessing a ceiling there would
 * reject adapters that could have run the simulation.
 */
function assertFits(shared: SharedDevice, budget: GpuBufferBudget): void {
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

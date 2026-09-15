/**
 * The compute half of the GPU layer: buffers, pipelines and readbacks on top of
 * the shared device manager.
 *
 * # Why this is a layer and not inline in the particle system
 *
 * `particleGpu.ts` is about one simulation: which buffers exist, which kernels
 * run in which order, how a step maps onto the CPU reference. Everything else
 * -- "make me a 96-byte uniform", "compile this WGSL and tell me if it failed",
 * "copy these bytes back to the CPU" -- is the same for every compute feature
 * this project will ever ship, and repeating it per system is how a codebase
 * ends up with four subtly different readback helpers, one of which forgets to
 * `unmap()`.
 *
 * # The two rules this file exists to enforce
 *
 * - **A shader that fails to compile is an exception, not a device error.**
 *   WebGPU reports compilation problems through `getCompilationInfo()` and then
 *   hands you a pipeline that fails validation at dispatch time, so the useful
 *   message is available exactly once and only if you ask. `ComputeProgram`
 *   asks, and turns `type: 'error'` messages into a `ShaderCompilationError`
 *   carrying the line numbers.
 * - **A dispatch that cannot be expressed is a bug, not a silent no-op.**
 *   `workgroups > maxComputeWorkgroupsPerDimension` is rejected by the driver
 *   after the command buffer is recorded, in a console message nobody reads.
 *   Checked here, against the snapshot the device was created with, so it fails
 *   at the call site with the number in the text.
 *
 * As everywhere else in `src/gpu`, the WebGPU surface comes from the structural
 * declarations in `device.ts` and nothing imports `@webgpu/types`, so all of
 * this runs in bare Node against `tests/stub_webgpu.ts`.
 */

import type {
  GpuBufferLike,
  GpuCompilationMessage,
  GpuConstants,
  GpuDeviceLike,
  GpuShaderModuleLike,
  SharedDevice,
} from './device.js';

// ---------------------------------------------------------------------------
// resources
// ---------------------------------------------------------------------------

/** What a binding is in a `GPUBindGroupLayoutEntry`, as the WGSL declares it. */
export type ComputeBufferType = 'uniform' | 'read-only-storage' | 'storage';

/**
 * One binding slot of one bind group.
 *
 * Deliberately a structural subset of `WgslBinding` in `particleWgsl.ts` rather
 * than an import of it: the shader generator owns the WGSL-specific fields
 * (`kind`, `type`) and this layer only needs the three that reach a layout
 * entry. `PARTICLE_BINDINGS` satisfies this shape as written.
 */
export interface ComputeBinding {
  readonly group: number;
  readonly binding: number;
  readonly bufferType: ComputeBufferType;
  /** Only ever used in error text, so it stays optional. */
  readonly name?: string;
}

/** A sub-range of a buffer, for bindings that do not own the whole thing. */
export interface ComputeBufferView {
  readonly buffer: ComputeBuffer | GpuBufferLike;
  readonly offset?: number;
  readonly size?: number;
}

/** Anything bindable: a wrapper, a raw buffer, or a range of either. */
export type ComputeResource = ComputeBuffer | GpuBufferLike | ComputeBufferView;

/**
 * The raw buffer behind a resource.
 *
 * Exported because the renderer has to do the same unwrap when it blits a
 * published buffer into a three.js-owned one.
 */
export function rawBuffer(resource: ComputeResource): GpuBufferLike {
  if (resource instanceof ComputeBuffer) return resource.raw;
  if (isBufferLike(resource)) return resource;
  return rawBuffer(resource.buffer);
}

function isBufferLike(value: unknown): value is GpuBufferLike {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as GpuBufferLike).getMappedRange === 'function' &&
    typeof (value as GpuBufferLike).mapAsync === 'function'
  );
}

/** The `GPUBindingResource` for one slot: a whole buffer, or an explicit range. */
function bindingResource(resource: ComputeResource): {
  readonly buffer: GpuBufferLike;
  readonly offset?: number;
  readonly size?: number;
} {
  if (resource instanceof ComputeBuffer) return { buffer: resource.raw };
  if (isBufferLike(resource)) return { buffer: resource };
  return {
    buffer: rawBuffer(resource.buffer),
    offset: resource.offset,
    size: resource.size,
  };
}

/**
 * The size rules WebGPU applies to every buffer, checked before `createBuffer`
 * so the failure names the buffer instead of arriving as a validation error.
 */
function checkedBytes(bytes: number, what: string): number {
  if (!Number.isFinite(bytes) || !Number.isInteger(bytes)) {
    throw new RangeError(`${what}: byte size ${bytes} is not an integer`);
  }
  if (bytes <= 0) throw new RangeError(`${what}: byte size must be positive, got ${bytes}`);
  // `COPY_BUFFER_ALIGNMENT` is 4; an unaligned size is rejected by every
  // implementation and the message never mentions which buffer it was.
  if (bytes % 4 !== 0) throw new RangeError(`${what}: byte size ${bytes} is not a multiple of 4`);
  return bytes;
}

function checkedOffset(offset: number, what: string): number {
  if (!Number.isInteger(offset) || offset < 0) {
    throw new RangeError(`${what}: offset ${offset} is not a non-negative integer`);
  }
  if (offset % 4 !== 0) throw new RangeError(`${what}: offset ${offset} is not a multiple of 4`);
  return offset;
}

// ---------------------------------------------------------------------------
// buffers
// ---------------------------------------------------------------------------

export interface ComputeBufferOptions {
  readonly label?: string;
  readonly bytes: number;
  /** A `GPUBufferUsage` bitmask, built from `shared.constants.bufferUsage`. */
  readonly usage: number;
  readonly mappedAtCreation?: boolean;
}

/**
 * A `GPUBuffer` with a name, a byte count and a destroyed flag.
 *
 * The wrapper earns its keep in two places. `write()` can say which buffer
 * overran, because it knows the size the buffer was created with -- the driver
 * only knows that a `writeBuffer` call was invalid. And `destroyed` makes
 * write-after-dispose a named error at the call site instead of a queue
 * operation on a dead object, which is what a resize path does wrong most
 * often.
 */
export class ComputeBuffer {
  readonly raw: GpuBufferLike;
  readonly label: string;
  readonly bytes: number;
  readonly usage: number;

  private destroyedFlag = false;

  /** @internal Created through `ComputeContext`, which tracks what it makes. */
  constructor(
    private readonly shared: SharedDevice,
    options: ComputeBufferOptions,
  ) {
    this.label = options.label ?? 'buffer';
    this.bytes = checkedBytes(options.bytes, this.label);
    this.usage = options.usage;
    this.raw = shared.device.createBuffer({
      label: this.label,
      size: this.bytes,
      usage: this.usage,
      mappedAtCreation: options.mappedAtCreation,
    });
  }

  get destroyed(): boolean {
    return this.destroyedFlag;
  }

  /**
   * `queue.writeBuffer` at an offset, with the bounds checked here.
   *
   * Returns `this` so seeding a buffer reads as one expression.
   */
  write(data: ArrayBufferView | ArrayBuffer, offset = 0): this {
    if (this.destroyedFlag) throw new Error(`${this.label}: cannot write to a destroyed buffer`);
    if (!(this.usage & this.shared.constants.bufferUsage.COPY_DST)) {
      throw new Error(`${this.label}: created without COPY_DST, so it cannot be written`);
    }
    checkedOffset(offset, this.label);
    const bytes = data.byteLength;
    if (offset + bytes > this.bytes) {
      throw new RangeError(
        `${this.label}: writing ${bytes} bytes at ${offset} overruns the ${this.bytes}-byte buffer`,
      );
    }
    this.shared.device.queue.writeBuffer(this.raw, offset, data);
    return this;
  }

  /** Idempotent: a teardown path that has to ask first is a teardown bug. */
  destroy(): void {
    if (this.destroyedFlag) return;
    this.destroyedFlag = true;
    this.raw.destroy();
  }
}

// ---------------------------------------------------------------------------
// programs
// ---------------------------------------------------------------------------

/** Thrown when `getCompilationInfo()` reports at least one `error` message. */
export class ShaderCompilationError extends Error {
  readonly label: string;
  readonly messages: readonly GpuCompilationMessage[];

  constructor(label: string, messages: readonly GpuCompilationMessage[]) {
    super(
      [
        `${label}: the shader failed to compile`,
        ...messages.map((m) => `  line ${m.lineNum || '?'}: ${m.message}`),
      ].join('\n'),
    );
    this.name = 'ShaderCompilationError';
    this.label = label;
    this.messages = messages;
  }
}

export interface ComputeProgramOptions {
  readonly label: string;
  readonly code: string;
  /**
   * The entry points this program may dispatch. Declared up front rather than
   * discovered, so a typo in a dispatch site is a thrown error naming the
   * declared set instead of a pipeline created for a function that does not
   * exist.
   */
  readonly entryPoints: readonly string[];
  readonly bindings: readonly ComputeBinding[];
}

/** One kernel to run, and the bind groups to have set when it runs. */
export interface ComputeDispatch {
  readonly entryPoint: string;
  /**
   * Workgroups, not invocations. Non-positive means "skip": `workgroupsFor()`
   * returns 0 for an empty input, and a caller should not have to filter.
   */
  readonly workgroups: number;
  /** Bind groups to set before the dispatch, by group index. */
  readonly groups?: Readonly<Record<number, unknown>>;
}

/**
 * A compiled module, its bind-group layouts and one memoized pipeline per entry
 * point.
 *
 * Built with `create()` rather than a constructor because compilation info is
 * async, and a program that skips that check is a program whose first symptom
 * is a validation error three dispatches later.
 */
export class ComputeProgram {
  readonly shared: SharedDevice;
  readonly label: string;
  readonly module: GpuShaderModuleLike;
  readonly entryPoints: readonly string[];
  readonly bindings: readonly ComputeBinding[];
  readonly layout: unknown;

  private readonly groupIndices: readonly number[];
  private readonly layouts: readonly unknown[];
  private readonly byGroup: ReadonlyMap<number, readonly ComputeBinding[]>;
  private readonly pipelines = new Map<string, unknown>();

  /**
   * Compile `code` and lay out its bind groups.
   *
   * @throws {ShaderCompilationError} when the module reports an error message.
   */
  static async create(
    shared: SharedDevice,
    options: ComputeProgramOptions,
  ): Promise<ComputeProgram> {
    assertUsable(shared, options.label);
    const module = shared.device.createShaderModule({ label: options.label, code: options.code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) throw new ShaderCompilationError(options.label, errors);
    return new ComputeProgram(shared, options, module);
  }

  private constructor(
    shared: SharedDevice,
    options: ComputeProgramOptions,
    module: GpuShaderModuleLike,
  ) {
    this.shared = shared;
    this.label = options.label;
    this.module = module;
    this.entryPoints = [...options.entryPoints];
    this.bindings = [...options.bindings];

    const byGroup = new Map<number, ComputeBinding[]>();
    for (const binding of this.bindings) {
      const list = byGroup.get(binding.group);
      if (list) list.push(binding);
      else byGroup.set(binding.group, [binding]);
    }
    for (const list of byGroup.values()) list.sort((a, b) => a.binding - b.binding);
    this.byGroup = byGroup;
    this.groupIndices = [...byGroup.keys()].sort((a, b) => a - b);
    this.layouts = this.buildLayouts(shared, options.label, this.groupIndices, byGroup);
    this.layout = shared.device.createPipelineLayout({
      label: options.label,
      bindGroupLayouts: this.layouts,
    });
  }

  /**
   * `createPipelineLayout` takes an array indexed by group number, so a gap or
   * a non-zero start would silently bind group 2's layout to index 0. Both are
   * caller bugs and both are cheap to catch here.
   */
  private buildLayouts(
    shared: SharedDevice,
    label: string,
    groups: readonly number[],
    byGroup: ReadonlyMap<number, readonly ComputeBinding[]>,
  ): readonly unknown[] {
    const maxBindGroups = shared.info.limits.maxBindGroups;
    if (maxBindGroups > 0 && groups.length > maxBindGroups) {
      throw new RangeError(
        `${label}: ${groups.length} bind groups exceeds maxBindGroups=${maxBindGroups}`,
      );
    }
    for (let i = 0; i < groups.length; i++) {
      if (groups[i] !== i) {
        throw new Error(
          `${label}: bind groups must start at 0 and be contiguous, got [${groups.join(', ')}]`,
        );
      }
    }
    return groups.map((group) =>
      shared.device.createBindGroupLayout({
        label: `${label}:group${group}`,
        entries: (byGroup.get(group) ?? []).map((b) => ({
          binding: b.binding,
          visibility: shared.constants.shaderStage.COMPUTE,
          buffer: { type: b.bufferType },
        })),
      }),
    );
  }

  /** Declared group indices, ascending. */
  get groups(): readonly number[] {
    return this.groupIndices;
  }

  /** The layout built for one group, for tests and for diagnostics. */
  layoutFor(group: number): unknown {
    const at = this.groupIndices.indexOf(group);
    if (at < 0) throw new Error(`${this.label}: group ${group} is not declared`);
    return this.layouts[at];
  }

  /** The pipeline for an entry point, created on first use and kept. */
  pipeline(entryPoint: string): unknown {
    const cached = this.pipelines.get(entryPoint);
    if (cached !== undefined) return cached;
    this.assertDeclared(entryPoint);
    const pipeline = this.shared.device.createComputePipeline({
      label: `${this.label}:${entryPoint}`,
      layout: this.layout,
      compute: { module: this.module, entryPoint },
    });
    this.pipelines.set(entryPoint, pipeline);
    return pipeline;
  }

  /**
   * Build one bind group.
   *
   * @throws when a declared binding has no resource, naming every missing one:
   *   a bind group short two buffers is one error, not two.
   */
  bindGroup(
    group: number,
    resources: Readonly<Record<number, ComputeResource>>,
    label?: string,
  ): unknown {
    const declared = this.byGroup.get(group);
    if (!declared) {
      throw new Error(
        `${this.label}: group ${group} is not declared (declared: [${this.groupIndices.join(', ')}])`,
      );
    }
    const missing = declared.filter((b) => resources[b.binding] === undefined);
    if (missing.length > 0) {
      throw new Error(
        `${this.label}: group ${group} is missing ${missing
          .map((b) => (b.name ? `binding ${b.binding} (${b.name})` : `binding ${b.binding}`))
          .join(', ')}`,
      );
    }
    return this.shared.device.createBindGroup({
      label: label ?? `${this.label}:group${group}`,
      layout: this.layoutFor(group),
      entries: declared.map((b) => ({
        binding: b.binding,
        resource: bindingResource(resources[b.binding]),
      })),
    });
  }

  /**
   * Run a chain of kernels in one compute pass, then submit once.
   *
   * One pass matters: separate passes would need a barrier between every pair
   * of kernels, and the particle chain reads what the previous kernel wrote.
   *
   * @returns how many dispatches were issued, i.e. the chain length minus the
   *   kernels skipped for a non-positive workgroup count.
   */
  submit(dispatches: readonly ComputeDispatch[], label?: string): number {
    const passLabel = label ?? this.label;
    assertUsable(this.shared, passLabel);
    const planned = dispatches.filter((d) => {
      if (!Number.isFinite(d.workgroups)) {
        throw new RangeError(`${passLabel}: ${d.entryPoint} got ${d.workgroups} workgroups`);
      }
      return d.workgroups > 0;
    });
    if (planned.length === 0) return 0;
    for (const dispatch of planned) this.validate(dispatch, passLabel);

    const device = this.shared.device;
    const encoder = device.createCommandEncoder({ label: passLabel });
    const pass = encoder.beginComputePass({ label: passLabel });
    for (const dispatch of planned) {
      if (dispatch.groups) {
        for (const [index, group] of sortedGroups(dispatch.groups)) pass.setBindGroup(index, group);
      }
      pass.setPipeline(this.pipeline(dispatch.entryPoint));
      pass.dispatchWorkgroups(dispatch.workgroups);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
    return planned.length;
  }

  private validate(dispatch: ComputeDispatch, passLabel: string): void {
    this.assertDeclared(dispatch.entryPoint);
    if (!Number.isInteger(dispatch.workgroups)) {
      throw new RangeError(
        `${passLabel}: ${dispatch.entryPoint} got ${dispatch.workgroups} workgroups, not an integer`,
      );
    }
    const limit = this.shared.info.limits.maxComputeWorkgroupsPerDimension;
    // 0 means the limit was never reported, and guessing a ceiling would fail
    // dispatches on an adapter that could have run them.
    if (limit > 0 && dispatch.workgroups > limit) {
      throw new RangeError(
        `${passLabel}: ${dispatch.entryPoint} needs ${dispatch.workgroups} workgroups, ` +
          `above maxComputeWorkgroupsPerDimension=${limit}`,
      );
    }
  }

  private assertDeclared(entryPoint: string): void {
    if (!this.entryPoints.includes(entryPoint)) {
      throw new Error(
        `${this.label}: '${entryPoint}' is not a declared entry point ` +
          `(${this.entryPoints.join(', ')})`,
      );
    }
  }
}

function sortedGroups(groups: Readonly<Record<number, unknown>>): Array<[number, unknown]> {
  return Object.entries(groups)
    .map(([key, value]) => [Number(key), value] as [number, unknown])
    .filter(([index]) => Number.isInteger(index))
    .sort((a, b) => a[0] - b[0]);
}

function assertUsable(shared: SharedDevice, what: string): void {
  if (shared.usable) return;
  throw new Error(`${what}: the shared device is ${shared.lost ? 'lost' : 'destroyed'}`);
}

// ---------------------------------------------------------------------------
// readback
// ---------------------------------------------------------------------------

/** One buffer to copy out, and how much of it. */
export interface ReadbackSource {
  readonly buffer: ComputeBuffer | GpuBufferLike;
  readonly bytes: number;
  /** Offset in the source. Defaults to 0. */
  readonly offset?: number;
}

// ---------------------------------------------------------------------------
// the context
// ---------------------------------------------------------------------------

/**
 * Buffer and program factory bound to one shared device.
 *
 * It does **not** touch the device refcount: whoever constructs a context
 * already holds a `SharedDevice` and releases it themselves. Splitting those
 * would make `dispose()` order a puzzle, and the manager's whole job is that
 * order does not matter.
 */
export class ComputeContext {
  readonly shared: SharedDevice;

  private readonly tracked = new Set<ComputeBuffer>();
  private destroyedFlag = false;

  constructor(shared: SharedDevice) {
    assertUsable(shared, 'compute context');
    this.shared = shared;
  }

  get device(): GpuDeviceLike {
    return this.shared.device;
  }

  get constants(): GpuConstants {
    return this.shared.constants;
  }

  get destroyed(): boolean {
    return this.destroyedFlag;
  }

  /** Buffers this context made and has not destroyed yet. */
  get buffers(): readonly ComputeBuffer[] {
    return [...this.tracked];
  }

  /** A buffer with an explicit usage mask, tracked so `destroy()` finds it. */
  buffer(options: ComputeBufferOptions): ComputeBuffer {
    this.assertLive();
    const buffer = new ComputeBuffer(this.shared, options);
    this.tracked.add(buffer);
    return buffer;
  }

  /**
   * A storage buffer that can also be written from the CPU and copied out.
   *
   * `COPY_SRC` is included even where nothing reads it back, because the two
   * things that do -- a parity check against the CPU backend and a blit into a
   * renderer-owned buffer -- are both added after the buffer is allocated, and
   * a missing usage bit is a validation error at exactly the worst moment.
   */
  storageBuffer(bytes: number, label?: string): ComputeBuffer {
    const u = this.constants.bufferUsage;
    return this.buffer({ label, bytes, usage: u.STORAGE | u.COPY_DST | u.COPY_SRC });
  }

  /** A uniform buffer, written with `queue.writeBuffer` and read by a shader. */
  uniformBuffer(bytes: number, label?: string): ComputeBuffer {
    const u = this.constants.bufferUsage;
    return this.buffer({ label, bytes, usage: u.UNIFORM | u.COPY_DST });
  }

  /** A buffer the CPU can map for reading. Not tracked: it is always short-lived. */
  stagingBuffer(bytes: number, label?: string): ComputeBuffer {
    const u = this.constants.bufferUsage;
    return new ComputeBuffer(this.shared, {
      label,
      bytes,
      usage: u.MAP_READ | u.COPY_DST,
    });
  }

  /** A buffer the CPU can map for writing, for one-shot uploads. */
  uploadBuffer(bytes: number, label?: string): ComputeBuffer {
    const u = this.constants.bufferUsage;
    return this.buffer({ label, bytes, usage: u.MAP_WRITE | u.COPY_SRC });
  }

  /** Compile a program against this context's device. */
  program(options: ComputeProgramOptions): Promise<ComputeProgram> {
    this.assertLive();
    return ComputeProgram.create(this.shared, options);
  }

  /**
   * Copy buffers out to the CPU, all of them in one submit.
   *
   * Each source gets its own staging buffer, mapped and unmapped here, because
   * a mapped range is invalid the moment the buffer is unmapped or destroyed --
   * returning the range instead of a copy is the classic way to hand back
   * memory that reads as zeros an hour later.
   */
  async readBytes(sources: readonly ReadbackSource[]): Promise<Uint8Array[]> {
    this.assertLive();
    assertUsable(this.shared, 'readback');
    const plans = sources.map((source) => ({
      source: rawBuffer(source.buffer),
      offset: checkedOffset(source.offset ?? 0, 'readback'),
      bytes: checkedBytes(source.bytes, 'readback'),
    }));
    if (plans.length === 0) return [];

    const device = this.shared.device;
    const staging = plans.map((plan, i) => this.stagingBuffer(plan.bytes, `readback staging ${i}`));
    try {
      const encoder = device.createCommandEncoder({ label: 'readback' });
      plans.forEach((plan, i) => {
        encoder.copyBufferToBuffer(plan.source, plan.offset, staging[i].raw, 0, plan.bytes);
      });
      device.queue.submit([encoder.finish()]);

      const out: Uint8Array[] = [];
      for (let i = 0; i < plans.length; i++) {
        const plan = plans[i];
        const stage = staging[i];
        await stage.raw.mapAsync(this.constants.mapMode.READ);
        const range = stage.raw.getMappedRange();
        const copy = new Uint8Array(plan.bytes);
        copy.set(new Uint8Array(range, 0, Math.min(plan.bytes, range.byteLength)));
        stage.raw.unmap();
        out.push(copy);
      }
      return out;
    } finally {
      for (const buffer of staging) buffer.destroy();
    }
  }

  /** Destroy every tracked buffer. Idempotent, and does not touch the device. */
  destroy(): void {
    if (this.destroyedFlag) return;
    this.destroyedFlag = true;
    for (const buffer of this.tracked) buffer.destroy();
    this.tracked.clear();
  }

  private assertLive(): void {
    if (this.destroyedFlag) throw new Error('the compute context is destroyed');
    assertUsable(this.shared, 'compute context');
  }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/**
 * Two buffers used alternately as source and destination.
 *
 * The particle integrator reads `stateSrc` and writes `stateDst`, so a step is
 * "dispatch, then swap" rather than "copy dst back into src". Swapping pointers
 * is free; the copy would be the largest transfer in the frame.
 */
export class PingPong<T> {
  private index = 0;

  constructor(
    readonly first: T,
    readonly second: T,
  ) {}

  /** What a kernel should read this step. */
  get src(): T {
    return this.index === 0 ? this.first : this.second;
  }

  /** What a kernel should write this step. */
  get dst(): T {
    return this.index === 0 ? this.second : this.first;
  }

  /** Which of the two is currently `src`: 0 for `first`, 1 for `second`. */
  get parity(): number {
    return this.index;
  }

  /** Call after a step, so the next one reads what this one wrote. */
  swap(): void {
    this.index = this.index === 0 ? 1 : 0;
  }
}

/** One buffer-to-buffer copy. Sizes and offsets must be multiples of 4. */
export interface CopyPair {
  readonly from: ComputeResource;
  readonly to: ComputeResource;
  readonly bytes: number;
  readonly fromOffset?: number;
  readonly toOffset?: number;
}

/**
 * Record and submit a batch of buffer copies.
 *
 * A free function rather than a context method because the interesting case is
 * the one where the destination is not ours: publishing GPU state into a buffer
 * three.js created for an `InstancedBufferAttribute`. That path has a
 * `ComputeContext` on one side and a raw device on the other, and taking the
 * device keeps it callable from both.
 *
 * @returns how many copies were recorded. A non-positive `bytes` skips the
 *   pair, so an empty particle system submits nothing.
 */
export function submitCopy(
  device: GpuDeviceLike,
  pairs: readonly CopyPair[],
  label = 'copy',
): number {
  const planned = pairs.filter((pair) => {
    if (!Number.isFinite(pair.bytes)) throw new RangeError(`${label}: ${pair.bytes} bytes to copy`);
    return pair.bytes > 0;
  });
  if (planned.length === 0) return 0;
  const prepared = planned.map((pair) => ({
    from: rawBuffer(pair.from),
    to: rawBuffer(pair.to),
    fromOffset: checkedOffset(pair.fromOffset ?? 0, label),
    toOffset: checkedOffset(pair.toOffset ?? 0, label),
    bytes: checkedBytes(pair.bytes, label),
  }));
  const encoder = device.createCommandEncoder({ label });
  for (const copy of prepared) {
    encoder.copyBufferToBuffer(copy.from, copy.fromOffset, copy.to, copy.toOffset, copy.bytes);
  }
  device.queue.submit([encoder.finish()]);
  return prepared.length;
}

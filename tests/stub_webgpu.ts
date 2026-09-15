/**
 * A recording WebGPU stub, shared by every spec that drives the GPU layer in
 * bare Node.
 *
 * `src/gpu` declares the WebGPU surface structurally so it runs headless, which
 * only pays off if the headless stand-in is faithful where it matters. So this
 * stub does three things a `vi.fn()`-shaped mock would not:
 *
 * - **Buffers hold bytes.** `queue.writeBuffer`, `copyBufferToBuffer` and
 *   `getMappedRange` move real data through a real `ArrayBuffer`, so an
 *   upload/readback round trip is asserted rather than assumed.
 * - **Command buffers record.** Every `setPipeline`, `setBindGroup` and
 *   `dispatchWorkgroups` lands in `submissions`, so a spec can check the
 *   dispatch chain an encoder produced instead of trusting that one was made.
 * - **Devices misbehave on cue.** `lose()` and `failLost()` resolve `device.lost`
 *   the way a crashed driver does, and `setCompilationMessages()` makes a shader
 *   fail to compile. None of that is producible against real hardware on demand.
 *
 * Not a spec of its own: `tests/tdd.test.ts` walks `src/`, and this file is only
 * reached through the specs that import it.
 */

import {
  type DeviceGpuLike,
  type GpuBufferLike,
  type GpuCommandEncoderLike,
  type GpuCompilationMessage,
  type GpuComputePassLike,
  type GpuConstants,
  type GpuDeviceAdapterLike,
  type GpuDeviceLike,
  type GpuDeviceLostInfo,
  type GpuQueueLike,
  type GpuShaderModuleLike,
} from '../src/gpu/device.js';

const MIB = 1024 * 1024;

/** The spec's bit values. Node has no WebGPU globals, so specs inject these. */
export const STUB_CONSTANTS: GpuConstants = {
  bufferUsage: {
    MAP_READ: 1,
    MAP_WRITE: 2,
    COPY_SRC: 4,
    COPY_DST: 8,
    INDEX: 16,
    VERTEX: 32,
    UNIFORM: 64,
    STORAGE: 128,
  },
  mapMode: { READ: 1, WRITE: 2 },
  shaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
};

/** Limits as a real core-mode desktop adapter reports them. */
export function stubLimits(overrides: Record<string, number> = {}): Record<string, number> {
  return {
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeWorkgroupsPerDimension: 65535,
    maxStorageBufferBindingSize: 128 * MIB,
    maxBufferSize: 256 * MIB,
    maxBindGroups: 4,
    maxStorageBuffersPerShaderStage: 8,
    maxUniformBufferBindingSize: 64 * 1024,
    ...overrides,
  };
}

export interface StubBufferInit {
  label?: string;
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

export class StubBuffer implements GpuBufferLike {
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  /** The bytes. `queue.writeBuffer` and copies land here. */
  readonly store: ArrayBuffer;
  mapCalls = 0;
  unmapCalls = 0;
  destroyCalls = 0;
  lastMapMode: number | null = null;

  private mappedFlag: boolean;

  constructor(init: StubBufferInit) {
    this.label = init.label ?? '';
    this.size = init.size;
    this.usage = init.usage;
    this.store = new ArrayBuffer(init.size);
    this.mappedFlag = init.mappedAtCreation === true;
  }

  get mapState(): string {
    return this.mappedFlag ? 'mapped' : 'unmapped';
  }

  get mapped(): boolean {
    return this.mappedFlag;
  }

  get destroyed(): boolean {
    return this.destroyCalls > 0;
  }

  bytes(): Uint8Array {
    return new Uint8Array(this.store);
  }

  floats(): Float32Array {
    return new Float32Array(this.store);
  }

  u32(): Uint32Array {
    return new Uint32Array(this.store);
  }

  async mapAsync(mode: number): Promise<void> {
    this.mapCalls++;
    this.lastMapMode = mode;
    this.mappedFlag = true;
  }

  /**
   * A copy, not a live view: the stub models readback, which is the only
   * direction this layer maps a buffer in.
   */
  getMappedRange(offset = 0, size = this.size - offset): ArrayBuffer {
    if (!this.mappedFlag) throw new Error(`getMappedRange on an unmapped buffer (${this.label})`);
    return this.store.slice(offset, offset + size);
  }

  unmap(): void {
    this.unmapCalls++;
    this.mappedFlag = false;
  }

  destroy(): void {
    this.destroyCalls++;
  }
}

export interface StubPassCall {
  readonly kind: 'pipeline' | 'bindGroup' | 'dispatch' | 'end';
  readonly pipeline?: StubPipeline;
  readonly group?: number;
  readonly bindGroup?: StubBindGroup;
  readonly workgroups?: number;
}

export class StubPass implements GpuComputePassLike {
  readonly label: string;
  readonly calls: StubPassCall[] = [];
  ended = false;

  constructor(label: string) {
    this.label = label;
  }

  setPipeline(pipeline: unknown): void {
    this.calls.push({ kind: 'pipeline', pipeline: pipeline as StubPipeline });
  }

  setBindGroup(index: number, group: unknown): void {
    this.calls.push({ kind: 'bindGroup', group: index, bindGroup: group as StubBindGroup });
  }

  dispatchWorkgroups(x: number): void {
    this.calls.push({ kind: 'dispatch', workgroups: x });
  }

  end(): void {
    this.ended = true;
    this.calls.push({ kind: 'end' });
  }

  get dispatches(): number[] {
    return this.calls.filter((c) => c.kind === 'dispatch').map((c) => c.workgroups!);
  }

  get entryPoints(): string[] {
    return this.calls.filter((c) => c.kind === 'pipeline').map((c) => c.pipeline!.entryPoint);
  }
}

export interface StubCopy {
  readonly from: StubBuffer;
  readonly fromOffset: number;
  readonly to: StubBuffer;
  readonly toOffset: number;
  readonly bytes: number;
}

export class StubCommandBuffer {
  constructor(
    readonly passes: StubPass[],
    readonly copies: StubCopy[],
  ) {}
}

export class StubEncoder implements GpuCommandEncoderLike {
  readonly passes: StubPass[] = [];
  readonly copies: StubCopy[] = [];
  finished = false;

  constructor(readonly label: string) {}

  beginComputePass(descriptor?: { label?: string }): StubPass {
    const pass = new StubPass(descriptor?.label ?? '');
    this.passes.push(pass);
    return pass;
  }

  copyBufferToBuffer(
    src: GpuBufferLike,
    srcOffset: number,
    dst: GpuBufferLike,
    dstOffset: number,
    size: number,
  ): void {
    this.copies.push({
      from: src as StubBuffer,
      fromOffset: srcOffset,
      to: dst as StubBuffer,
      toOffset: dstOffset,
      bytes: size,
    });
  }

  /** Sealing the encoder is where the stub applies its copies. */
  finish(): StubCommandBuffer {
    this.finished = true;
    for (const copy of this.copies) {
      const from = new Uint8Array(copy.from.store, copy.fromOffset, copy.bytes);
      const to = new Uint8Array(copy.to.store, copy.toOffset, copy.bytes);
      to.set(from);
    }
    return new StubCommandBuffer(this.passes, this.copies);
  }
}

export class StubShaderModule implements GpuShaderModuleLike {
  constructor(
    readonly label: string,
    readonly code: string,
    private readonly messages: () => readonly GpuCompilationMessage[],
  ) {}

  async getCompilationInfo(): Promise<{ readonly messages: readonly GpuCompilationMessage[] }> {
    return { messages: this.messages() };
  }
}

export class StubPipelineLayout {
  constructor(readonly bindGroupLayouts: readonly StubBindGroupLayout[]) {}
}

export class StubBindGroupLayout {
  constructor(
    readonly label: string,
    readonly entries: readonly Record<string, unknown>[],
  ) {}
}

export class StubPipeline {
  constructor(
    readonly label: string,
    readonly entryPoint: string,
    readonly module: StubShaderModule,
    readonly layout: StubPipelineLayout | 'auto',
  ) {}
}

export class StubBindGroup {
  constructor(
    readonly layout: StubBindGroupLayout,
    readonly entries: ReadonlyArray<{ binding: number; resource: StubBuffer }>,
  ) {}

  bufferAt(binding: number): StubBuffer | undefined {
    return this.entries.find((e) => e.binding === binding)?.resource;
  }
}

export interface StubDeviceInit {
  limits?: Record<string, number>;
  features?: string[];
}

export class StubDevice implements GpuDeviceLike {
  readonly features: { has(name: string): boolean };
  readonly limits: Record<string, number | undefined>;
  readonly queue: GpuQueueLike;
  readonly lost: Promise<GpuDeviceLostInfo>;

  readonly buffers: StubBuffer[] = [];
  readonly modules: StubShaderModule[] = [];
  readonly pipelines: StubPipeline[] = [];
  readonly bindGroups: StubBindGroup[] = [];
  readonly layouts: StubBindGroupLayout[] = [];
  readonly submissions: StubCommandBuffer[] = [];
  readonly writes: Array<{ buffer: StubBuffer; offset: number; bytes: number }> = [];
  destroyCalls = 0;
  submittedWorkDone = 0;

  private compilationMessages: readonly GpuCompilationMessage[] = [];
  private resolveLost!: (info: GpuDeviceLostInfo) => void;
  private rejectLost!: (error: unknown) => void;

  constructor(init: StubDeviceInit = {}) {
    const features = init.features ?? [];
    this.features = { has: (name: string) => features.includes(name) };
    this.limits = init.limits ?? stubLimits();
    this.lost = new Promise<GpuDeviceLostInfo>((resolve, reject) => {
      this.resolveLost = resolve;
      this.rejectLost = reject;
    });
    this.queue = {
      submit: (commandBuffers: readonly unknown[]) => {
        for (const cb of commandBuffers) this.submissions.push(cb as StubCommandBuffer);
      },
      writeBuffer: (
        buffer: GpuBufferLike,
        offset: number,
        data: ArrayBufferView | ArrayBuffer,
      ) => {
        const target = buffer as StubBuffer;
        const source =
          data instanceof ArrayBuffer
            ? new Uint8Array(data)
            : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        new Uint8Array(target.store, offset, source.byteLength).set(source);
        this.writes.push({ buffer: target, offset, bytes: source.byteLength });
      },
      onSubmittedWorkDone: async () => {
        this.submittedWorkDone++;
      },
    };
  }

  get destroyed(): boolean {
    return this.destroyCalls > 0;
  }

  /** Every compute pass this device has seen, in submission order. */
  get passes(): StubPass[] {
    return this.submissions.flatMap((cb) => cb.passes);
  }

  /** The last submitted pass, which is what most specs care about. */
  get lastPass(): StubPass | undefined {
    const all = this.passes;
    return all[all.length - 1];
  }

  setCompilationMessages(messages: readonly GpuCompilationMessage[]): void {
    this.compilationMessages = messages;
  }

  lose(info: GpuDeviceLostInfo = { reason: 'unknown', message: 'lost by the stub' }): void {
    this.resolveLost(info);
  }

  failLost(error: unknown): void {
    this.rejectLost(error);
  }

  createBuffer(descriptor: StubBufferInit): StubBuffer {
    const buffer = new StubBuffer(descriptor);
    this.buffers.push(buffer);
    return buffer;
  }

  createShaderModule(descriptor: { label?: string; code: string }): StubShaderModule {
    const module = new StubShaderModule(descriptor.label ?? '', descriptor.code, () =>
      this.compilationMessages.slice(),
    );
    this.modules.push(module);
    return module;
  }

  createBindGroupLayout(descriptor: {
    label?: string;
    entries: readonly unknown[];
  }): StubBindGroupLayout {
    const layout = new StubBindGroupLayout(
      descriptor.label ?? '',
      descriptor.entries as readonly Record<string, unknown>[],
    );
    this.layouts.push(layout);
    return layout;
  }

  createBindGroup(descriptor: {
    layout: unknown;
    entries: readonly unknown[];
  }): StubBindGroup {
    const entries = (
      descriptor.entries as ReadonlyArray<{
        binding: number;
        resource: { buffer?: StubBuffer } | StubBuffer;
      }>
    ).map((e) => ({
      binding: e.binding,
      resource: ('buffer' in e.resource ? e.resource.buffer : e.resource) as StubBuffer,
    }));
    const group = new StubBindGroup(descriptor.layout as StubBindGroupLayout, entries);
    this.bindGroups.push(group);
    return group;
  }

  createPipelineLayout(descriptor: { bindGroupLayouts: readonly unknown[] }): StubPipelineLayout {
    return new StubPipelineLayout(descriptor.bindGroupLayouts as readonly StubBindGroupLayout[]);
  }

  createComputePipeline(descriptor: {
    label?: string;
    layout: unknown;
    compute: { module: GpuShaderModuleLike; entryPoint: string };
  }): StubPipeline {
    const pipeline = new StubPipeline(
      descriptor.label ?? '',
      descriptor.compute.entryPoint,
      descriptor.compute.module as StubShaderModule,
      descriptor.layout as StubPipelineLayout | 'auto',
    );
    this.pipelines.push(pipeline);
    return pipeline;
  }

  createCommandEncoder(descriptor?: { label?: string }): StubEncoder {
    return new StubEncoder(descriptor?.label ?? '');
  }

  destroy(): void {
    this.destroyCalls++;
  }
}

export interface StubAdapterSpec {
  limits?: Record<string, number>;
  features?: string[];
  isCompatibilityMode?: boolean;
  info?: { vendor?: string; architecture?: string };
  /** What `requestDevice` hands back, or the error it throws. */
  device?: StubDevice | Error;
}

export interface StubDeviceRequest {
  readonly label?: string;
  readonly requiredFeatures?: readonly string[];
  readonly requiredLimits?: Readonly<Record<string, number>>;
}

export class StubAdapter implements GpuDeviceAdapterLike {
  readonly features: { has(name: string): boolean };
  readonly limits: Record<string, number>;
  readonly info: { vendor?: string; architecture?: string };
  readonly isCompatibilityMode?: boolean;
  readonly deviceRequests: StubDeviceRequest[] = [];
  readonly device: StubDevice;

  constructor(readonly spec: StubAdapterSpec = {}) {
    const features = spec.features ?? [];
    this.features = { has: (name: string) => features.includes(name) };
    this.limits = spec.limits ?? stubLimits();
    this.info = spec.info ?? { vendor: 'intel', architecture: 'gen-9' };
    if (spec.isCompatibilityMode !== undefined) {
      this.isCompatibilityMode = spec.isCompatibilityMode;
    }
    this.device =
      spec.device instanceof Error || spec.device === undefined
        ? new StubDevice({ limits: spec.limits, features: spec.features })
        : spec.device;
  }

  async requestDevice(descriptor?: StubDeviceRequest): Promise<StubDevice> {
    this.deviceRequests.push(descriptor ?? {});
    if (this.spec.device instanceof Error) throw this.spec.device;
    return this.device;
  }
}

export type StubLevelSpec = StubAdapterSpec | StubAdapter | Error | null;

export class StubGpu implements DeviceGpuLike {
  readonly asked: string[] = [];
  readonly adapters: StubAdapter[] = [];

  constructor(
    private readonly byLevel: Partial<Record<'core' | 'compatibility', StubLevelSpec>> = {},
  ) {}

  async requestAdapter(options?: {
    featureLevel?: 'core' | 'compatibility';
  }): Promise<StubAdapter | null> {
    const level = options?.featureLevel ?? 'core';
    this.asked.push(level);
    const spec = this.byLevel[level];
    if (!spec) return null;
    if (spec instanceof Error) throw spec;
    const adapter = spec instanceof StubAdapter ? spec : new StubAdapter(spec);
    this.adapters.push(adapter);
    return adapter;
  }
}

/** Let `device.lost` handlers run: a few microtask turns covers a `.then` chain. */
export async function flushMicrotasks(turns = 4): Promise<void> {
  for (let i = 0; i < turns; i++) await Promise.resolve();
}

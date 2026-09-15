/**
 * The shared `GPUDevice` manager: one device per page, handed to whoever needs
 * it, refcounted, and recovered from when it is lost.
 *
 * # Why this exists
 *
 * `docs/development-plan.md` M2 task 1 is "unify `requestDevice()` and
 * `WebGPURenderer` into a singleton device manager", and every milestone after
 * it depends on that being true rather than aspirational. Two devices means two
 * queues, two contexts and no way for a compute kernel to touch a buffer the
 * renderer created, which is exactly the zero-copy bridge M2 exists to prove.
 * `capabilities.ts` deliberately stops at the adapter -- "device lifetime
 * belongs to the shared device manager" -- and this file is that manager.
 *
 * # Contract
 *
 * - **Never throws on the way in.** `acquire()` resolves to `null` when there is
 *   no usable device and records why in `failure`. A boot path that has to
 *   `try/catch` a device request is a boot path that white-screens on the first
 *   blocklisted driver; the plan's acceptance criterion is that WebGPU being
 *   unavailable downgrades the app, not kills it.
 * - **One device, even under a race.** Two callers awaiting `acquire()` in the
 *   same tick share one in-flight request rather than creating two devices.
 * - **Loss is an event, not an exception.** `device.lost` is watched from the
 *   moment the device is created, the registration is dropped, and the next
 *   `acquire()` builds a fresh device.
 *
 * As in `capabilities.ts`, the WebGPU surface is declared structurally and
 * nothing imports `@webgpu/types`, so the whole manager runs in bare Node
 * against a stub adapter and stays inside the coverage floor.
 */

import {
  gpuFrom,
  snapshotLimits,
  unmetLimitsOf,
  type FeatureLevel,
  type GpuAdapterLike,
  type GpuLimits,
  type LimitName,
} from './capabilities.js';

// ---------------------------------------------------------------------------
// the WebGPU surface, declared narrowly
// ---------------------------------------------------------------------------

export interface GpuBufferLike {
  readonly size: number;
  readonly usage: number;
  readonly mapState: string;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

export interface GpuCompilationMessage {
  readonly type: string;
  readonly lineNum: number;
  readonly message: string;
}

export interface GpuShaderModuleLike {
  getCompilationInfo(): Promise<{ readonly messages: readonly GpuCompilationMessage[] }>;
}

export interface GpuComputePassLike {
  setPipeline(pipeline: unknown): void;
  setBindGroup(index: number, group: unknown, dynamicOffsets?: readonly number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface GpuCommandEncoderLike {
  beginComputePass(descriptor?: { label?: string }): GpuComputePassLike;
  copyBufferToBuffer(
    src: GpuBufferLike,
    srcOffset: number,
    dst: GpuBufferLike,
    dstOffset: number,
    size: number,
  ): void;
  finish(): unknown;
}

export interface GpuQueueLike {
  submit(commandBuffers: readonly unknown[]): void;
  writeBuffer(buffer: GpuBufferLike, offset: number, data: ArrayBufferView | ArrayBuffer): void;
  onSubmittedWorkDone(): Promise<void>;
}

export interface GpuDeviceLostInfo {
  readonly reason: string;
  readonly message: string;
}

/**
 * The subset of `GPUDevice` this layer uses.
 *
 * It covers compute only: no render pipelines, no textures, no query sets. The
 * renderer gets the same object and types it as three.js does, which is why the
 * manager hands out the raw object rather than a wrapper -- a wrapper would have
 * to grow a member for every three.js release.
 */
export interface GpuDeviceLike {
  readonly features: { has(name: string): boolean };
  readonly limits: { readonly [name: string]: number | undefined };
  readonly queue: GpuQueueLike;
  readonly lost: Promise<GpuDeviceLostInfo>;
  createBuffer(descriptor: {
    label?: string;
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): GpuBufferLike;
  createShaderModule(descriptor: { label?: string; code: string }): GpuShaderModuleLike;
  createBindGroupLayout(descriptor: { label?: string; entries: readonly unknown[] }): unknown;
  createBindGroup(descriptor: {
    label?: string;
    layout: unknown;
    entries: readonly unknown[];
  }): unknown;
  createPipelineLayout(descriptor: {
    label?: string;
    bindGroupLayouts: readonly unknown[];
  }): unknown;
  createComputePipeline(descriptor: {
    label?: string;
    layout: unknown;
    compute: { module: GpuShaderModuleLike; entryPoint: string };
  }): unknown;
  createCommandEncoder(descriptor?: { label?: string }): GpuCommandEncoderLike;
  destroy(): void;
}

/** `GPUAdapter` plus the one member `capabilities.ts` has no reason to know. */
export interface GpuDeviceAdapterLike extends GpuAdapterLike {
  requestDevice(descriptor?: {
    label?: string;
    requiredFeatures?: readonly string[];
    requiredLimits?: Readonly<Record<string, number>>;
  }): Promise<GpuDeviceLike>;
}

export interface DeviceGpuLike {
  requestAdapter(
    options?: { featureLevel?: FeatureLevel },
  ): Promise<GpuDeviceAdapterLike | null | undefined>;
}

/**
 * `navigator.gpu` typed for device requests.
 *
 * A downcast of `gpuFrom`: the adapter type here is a subtype of the one the
 * probe uses, so the assertion is the narrowest thing that works and keeps
 * `src/` free of `@webgpu/types`.
 */
export function deviceGpuFrom(scope: unknown = globalThis): DeviceGpuLike | undefined {
  return gpuFrom(scope) as DeviceGpuLike | undefined;
}

// ---------------------------------------------------------------------------
// enum constants
// ---------------------------------------------------------------------------

/**
 * The WebGPU bit constants, read from the globals rather than hardcoded.
 *
 * The values are fixed by the spec, but a copy of them in `src/` is a copy that
 * can be wrong in a way nothing detects until a buffer is created with the wrong
 * usage and every dispatch fails validation. Reading them costs one lookup at
 * acquisition time.
 */
export interface GpuConstants {
  readonly bufferUsage: {
    readonly MAP_READ: number;
    readonly MAP_WRITE: number;
    readonly COPY_SRC: number;
    readonly COPY_DST: number;
    readonly UNIFORM: number;
    readonly STORAGE: number;
    readonly VERTEX: number;
    readonly INDEX: number;
  };
  readonly mapMode: { readonly READ: number; readonly WRITE: number };
  readonly shaderStage: { readonly VERTEX: number; readonly FRAGMENT: number; readonly COMPUTE: number };
}

/** The constants, or `null` where WebGPU is not present at all. */
export function gpuConstantsFrom(scope: unknown = globalThis): GpuConstants | null {
  const s = scope as {
    readonly GPUBufferUsage?: GpuConstants['bufferUsage'];
    readonly GPUMapMode?: GpuConstants['mapMode'];
    readonly GPUShaderStage?: GpuConstants['shaderStage'];
  };
  if (!s.GPUBufferUsage || !s.GPUMapMode || !s.GPUShaderStage) return null;
  return {
    bufferUsage: s.GPUBufferUsage,
    mapMode: s.GPUMapMode,
    shaderStage: s.GPUShaderStage,
  };
}

// ---------------------------------------------------------------------------
// limits to request
// ---------------------------------------------------------------------------

/**
 * Limits the device is asked for, by name.
 *
 * A device created with no `requiredLimits` gets the *default* limits, which are
 * lower than the adapter's: `maxStorageBuffersPerShaderStage` defaults to 8 and
 * the particle pipeline binds exactly 8 storage buffers in the compute stage, so
 * one more buffer anywhere in the layer would fail validation on a device that
 * could have had more. Asking for what the adapter already exposes is always
 * legal, which makes this the cheap way to avoid the whole class of failure.
 */
export const REQUESTED_LIMITS: readonly LimitName[] = [
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupsPerDimension',
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxStorageBuffersPerShaderStage',
  'maxBindGroups',
  'maxUniformBufferBindingSize',
];

/**
 * `requiredLimits` for an adapter: every requested limit the adapter reports, at
 * the value it reports.
 *
 * A limit the adapter does not report is left out rather than guessed, because a
 * `requiredLimits` entry above what the adapter supports makes `requestDevice`
 * reject -- and a rejected device is a fallback to CPU, not a smaller request.
 */
export function requiredLimitsFor(adapter: GpuAdapterLike): Record<string, number> {
  const out: Record<string, number> = {};
  for (const name of REQUESTED_LIMITS) {
    const value = adapter.limits[name];
    if (typeof value === 'number' && Number.isFinite(value)) out[name] = value;
  }
  return out;
}

// ---------------------------------------------------------------------------
// failures
// ---------------------------------------------------------------------------

/** Why `acquire()` returned `null`. `capabilities.ts` reasons, extended. */
export type DeviceFailureReason =
  /** No `navigator.gpu`, or the WebGPU enum constants are missing. */
  | 'absent'
  /** `requestAdapter` returned null at every feature level asked for. */
  | 'no-adapter'
  /** Adapter or device limits are below `LIMIT_FLOOR`. */
  | 'unmet-limits'
  /** `requestAdapter` or `requestDevice` threw. */
  | 'error'
  /** The shared device was lost after it was handed out. */
  | 'lost';

export interface DeviceFailure {
  readonly reason: DeviceFailureReason;
  /** The message, never the stack: this ends up in a HUD and in a spec log. */
  readonly error?: string;
  readonly featureLevel?: FeatureLevel;
  readonly limits?: GpuLimits;
  readonly unmetLimits: readonly LimitName[];
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** `destroy()` on a device that is already gone can throw; nothing gains from it. */
function safeDestroy(device: GpuDeviceLike): void {
  try {
    device.destroy();
  } catch {
    // A lost or already-destroyed device. The caller wanted it gone; it is.
  }
}

// ---------------------------------------------------------------------------
// the handle
// ---------------------------------------------------------------------------

export interface SharedDeviceInfo {
  readonly featureLevel: FeatureLevel;
  readonly limits: GpuLimits;
  readonly adapterInfo: { readonly vendor: string; readonly architecture: string };
  /** True when the device came from somewhere else, e.g. three.js. */
  readonly adopted: boolean;
}

export interface SharedDeviceHooks {
  /** Last reference released. */
  zero(shared: SharedDevice): void;
  /** The device reported itself lost. */
  lost(shared: SharedDevice): void;
}

/**
 * A refcounted handle on the shared device.
 *
 * Callers get this rather than the raw `GPUDevice` so that "who is still using
 * it" is a number the manager can act on: a renderer and a compute layer that
 * each hold a handle can tear themselves down in any order, and the device
 * survives until both are done. `.device` is the raw object, and is what gets
 * passed to `new WebGPURenderer({ device })` and bound into pipelines.
 */
export class SharedDevice {
  readonly device: GpuDeviceLike;
  readonly constants: GpuConstants;
  readonly info: SharedDeviceInfo;

  private refs = 1;
  private lostFlag = false;
  private destroyedFlag = false;
  private lostInfo: GpuDeviceLostInfo | null = null;
  private listeners: Array<(shared: SharedDevice) => void> = [];

  /** @internal Built by `SharedDeviceManager`, which owns the hooks. */
  constructor(
    device: GpuDeviceLike,
    constants: GpuConstants,
    info: SharedDeviceInfo,
    private readonly hooks: SharedDeviceHooks,
  ) {
    this.device = device;
    this.constants = constants;
    this.info = info;
    this.watchLost();
  }

  get references(): number {
    return this.refs;
  }

  get lost(): boolean {
    return this.lostFlag;
  }

  get destroyed(): boolean {
    return this.destroyedFlag;
  }

  /** Safe to submit to. A lost device keeps its buffers and loses its meaning. */
  get usable(): boolean {
    return !this.lostFlag && !this.destroyedFlag;
  }

  /** What `device.lost` resolved with, once it has. */
  get lossInfo(): GpuDeviceLostInfo | null {
    return this.lostInfo;
  }

  /**
   * Take another reference.
   *
   * Throws rather than resurrecting: retaining a torn-down handle is a caller
   * bug, and the alternative -- handing back a dead device that silently fails
   * validation on every dispatch -- is the failure mode this layer exists to
   * prevent.
   */
  retain(): SharedDevice {
    if (!this.usable) {
      throw new Error(
        `shared device is ${this.destroyedFlag ? 'destroyed' : 'lost'}; acquire a new one`,
      );
    }
    this.refs++;
    return this;
  }

  /**
   * Drop a reference. Returns true when this was the last one and the manager
   * therefore owns no device any more.
   *
   * Releasing twice is a bug, and an unrecoverable one: the count cannot be
   * un-decremented. So it throws instead of clamping at zero, which would let a
   * leaked reference keep a device alive forever with nothing to show for it.
   */
  release(): boolean {
    if (this.refs <= 0) throw new Error('shared device released more often than retained');
    this.refs--;
    if (this.refs > 0) return false;
    this.hooks.zero(this);
    return true;
  }

  /**
   * Watch for device loss. Returns an unsubscribe function.
   *
   * Listeners registered after the loss fire immediately, so a late subscriber
   * is not left waiting for an event that already happened.
   */
  onLost(listener: (shared: SharedDevice) => void): () => void {
    if (this.lostFlag) {
      listener(this);
      return () => undefined;
    }
    this.listeners.push(listener);
    return () => {
      const at = this.listeners.indexOf(listener);
      if (at >= 0) this.listeners.splice(at, 1);
    };
  }

  /**
   * Destroy the underlying device now, whatever the refcount.
   *
   * Explicit teardown: page unload, a demo re-run, a test's `afterEach`. Handles
   * still outstanding become unusable, which is what the caller asked for.
   */
  destroy(): void {
    if (this.destroyedFlag) return;
    this.destroyedFlag = true;
    this.refs = 0;
    this.listeners = [];
    safeDestroy(this.device);
  }

  /**
   * Attach to `device.lost` exactly once, at construction.
   *
   * Both handlers are supplied: the spec says `lost` always resolves, but an
   * implementation that rejects instead would leave an unhandled rejection, and
   * in a browser that is a console error on top of a dead GPU. Watched from the
   * start rather than on first `onLost`, because a device lost before anybody
   * subscribed still has to drop the manager's registration.
   */
  private watchLost(): void {
    const settle = (info: GpuDeviceLostInfo | null): void => {
      // A device we destroyed ourselves resolves `lost` with reason
      // 'destroyed'. That is the teardown we asked for, not a failure.
      if (this.destroyedFlag) return;
      this.lostInfo = info;
      this.lostFlag = true;
      const listeners = this.listeners;
      this.listeners = [];
      this.hooks.lost(this);
      for (const listener of listeners) listener(this);
    };
    void this.device.lost.then(
      (info) => settle(info ?? null),
      (error: unknown) => settle({ reason: 'error', message: messageOf(error) }),
    );
  }
}

// ---------------------------------------------------------------------------
// the manager
// ---------------------------------------------------------------------------

export interface SharedDeviceOptions {
  /** Injectable, so the manager is testable in bare Node. Defaults to `navigator.gpu`. */
  readonly gpu?: DeviceGpuLike;
  /** Injectable enum constants. Defaults to a `globalThis` lookup. */
  readonly constants?: GpuConstants | null;
  /** Feature levels to try, in order. Defaults to `['core', 'compatibility']`. */
  readonly featureLevels?: readonly FeatureLevel[];
  /**
   * Features to require. Not filtered against the adapter on purpose: asking for
   * `subgroup` and quietly getting a device without it is worse than a reported
   * failure, and the caller can read `adapter.features` first.
   */
  readonly requiredFeatures?: readonly string[];
  readonly label?: string;
}

/**
 * One device per page, handed out by reference count.
 *
 * Construct one per app (the default export at the bottom of this file is the
 * process-wide one) or one per test. State is per instance, so a stub `gpu`
 * passed to a fresh manager cannot leak into another.
 */
export class SharedDeviceManager {
  private readonly gpu?: DeviceGpuLike;
  private readonly constantsOption?: GpuConstants | null;
  private readonly featureLevels: readonly FeatureLevel[];
  private readonly requiredFeatures: readonly string[];
  private readonly label: string;

  private currentShared: SharedDevice | null = null;
  private inflight: Promise<SharedDevice | null> | null = null;
  private lastFailure: DeviceFailure | null = null;
  private readonly hooks: SharedDeviceHooks;

  constructor(options: SharedDeviceOptions = {}) {
    this.gpu = options.gpu;
    this.constantsOption = options.constants;
    this.featureLevels = options.featureLevels ?? ['core', 'compatibility'];
    this.requiredFeatures = options.requiredFeatures ?? [];
    this.label = options.label ?? 'threedream';
    this.hooks = {
      zero: (shared) => this.unregister(shared, !shared.info.adopted),
      lost: (shared) => {
        // Recorded, not thrown: a device that dies mid-run has to leave a reason
        // behind, because the next `acquire()` silently making a new one is
        // indistinguishable from the loss never having happened.
        this.lastFailure = {
          reason: 'lost',
          error: shared.lossInfo?.message,
          featureLevel: shared.info.featureLevel,
          limits: shared.info.limits,
          unmetLimits: [],
        };
        this.unregister(shared, false);
      },
    };
  }

  /** The live registration, or `null`. Lost and destroyed handles are not returned. */
  get current(): SharedDevice | null {
    const shared = this.currentShared;
    if (!shared) return null;
    if (!shared.usable) this.unregister(shared, false);
    return this.currentShared;
  }

  /** The device request currently in flight, for tests and for diagnostics. */
  get pending(): Promise<SharedDevice | null> | null {
    return this.inflight;
  }

  /** Why the last `acquire()` failed, or `null` if it did not. */
  get failure(): DeviceFailure | null {
    return this.lastFailure;
  }

  /**
   * The shared device, created on first use. `null` when there is none, with the
   * reason in `failure`.
   *
   * Concurrent callers share one request: the second one awaits the first
   * caller's promise and then retains its handle, so a page that boots a
   * renderer and a compute layer in parallel ends up with one device and two
   * references rather than two devices.
   */
  async acquire(options: SharedDeviceOptions = {}): Promise<SharedDevice | null> {
    const current = this.current;
    if (current) {
      current.retain();
      this.lastFailure = null;
      return current;
    }
    if (this.inflight) {
      const shared = await this.inflight;
      if (!shared) return null;
      shared.retain();
      return shared;
    }
    const task = this.create(options);
    this.inflight = task;
    try {
      return await task;
    } finally {
      this.inflight = null;
    }
  }

  /**
   * Register a device this manager did not create.
   *
   * This is the three.js direction of M2: a `WebGPURenderer` that made its own
   * device can hand it over so compute shares it, instead of the app ending up
   * with two. Refcount-zero on an adopted device unregisters it without
   * destroying it, because its owner is the one who has to.
   */
  adopt(device: GpuDeviceLike, info?: Partial<SharedDeviceInfo>): SharedDevice {
    const existing = this.currentShared;
    if (existing && existing.device === device && existing.usable) return existing.retain();
    const constants = this.resolveConstants({});
    if (!constants) {
      throw new Error('cannot adopt a device: the WebGPU enum constants are missing');
    }
    const shared = new SharedDevice(
      device,
      constants,
      {
        // The adopter knows which feature level it asked for; this manager did
        // not create the device, so it cannot. `core` is the optimistic default
        // and the caller can correct it -- the level only ever reaches a log.
        featureLevel: info?.featureLevel ?? 'core',
        limits: info?.limits ?? snapshotLimits(device.limits),
        adapterInfo: info?.adapterInfo ?? { vendor: 'unknown', architecture: 'unknown' },
        adopted: true,
      },
      this.hooks,
    );
    this.currentShared = shared;
    this.lastFailure = null;
    return shared;
  }

  /**
   * Drop the registration and destroy a device this manager created.
   *
   * Awaited: the in-flight request is settled first, so a `releaseAll()` racing
   * an `acquire()` cannot leave a device behind that nobody is registered to
   * destroy.
   */
  async releaseAll(): Promise<void> {
    if (this.inflight) await this.inflight;
    const shared = this.currentShared;
    this.currentShared = null;
    if (shared && !shared.info.adopted) shared.destroy();
  }

  private unregister(shared: SharedDevice, destroy: boolean): void {
    if (this.currentShared === shared) this.currentShared = null;
    if (destroy) shared.destroy();
  }

  private resolveConstants(options: SharedDeviceOptions): GpuConstants | null {
    // An explicit `null` is a real answer ("there are no constants"), so it must
    // not fall through to the constructor option the way `??` would let it.
    const requested = options.constants !== undefined ? options.constants : this.constantsOption;
    if (requested !== undefined) return requested;
    return gpuConstantsFrom();
  }

  private fail(failure: DeviceFailure): null {
    this.lastFailure = failure;
    return null;
  }

  /** The create path. Split out so `acquire()` can dedupe on the promise. */
  private async create(options: SharedDeviceOptions): Promise<SharedDevice | null> {
    const gpu = options.gpu ?? this.gpu ?? deviceGpuFrom();
    if (!gpu || typeof gpu.requestAdapter !== 'function') {
      return this.fail({ reason: 'absent', unmetLimits: [] });
    }
    const constants = this.resolveConstants(options);
    if (!constants) {
      return this.fail({
        reason: 'absent',
        error: 'GPUBufferUsage, GPUMapMode or GPUShaderStage is missing',
        unmetLimits: [],
      });
    }
    const levels = options.featureLevels ?? this.featureLevels;
    const features = options.requiredFeatures ?? this.requiredFeatures;
    const label = options.label ?? this.label;
    // The *first* rejection is the diagnosis, matching `probeWebGpu`: it came
    // from the better feature level, so its limits are the informative ones.
    let first: DeviceFailure | null = null;

    for (const featureLevel of levels) {
      let adapter: GpuDeviceAdapterLike | null | undefined;
      try {
        adapter = await gpu.requestAdapter({ featureLevel });
      } catch (error) {
        return this.fail({ reason: 'error', error: messageOf(error), unmetLimits: [] });
      }
      if (!adapter) continue;

      const adapterLimits = snapshotLimits(adapter.limits);
      const adapterUnmet = unmetLimitsOf(adapterLimits);
      if (adapterUnmet.length > 0) {
        // Keep trying the other levels, but remember why this one was refused.
        first ??= {
          reason: 'unmet-limits',
          featureLevel,
          limits: adapterLimits,
          unmetLimits: adapterUnmet,
        };
        continue;
      }

      let device: GpuDeviceLike;
      try {
        device = await adapter.requestDevice({
          label,
          requiredLimits: requiredLimitsFor(adapter),
          ...(features.length > 0 ? { requiredFeatures: features } : {}),
        });
      } catch (error) {
        first ??= {
          reason: 'error',
          featureLevel,
          limits: adapterLimits,
          error: messageOf(error),
          unmetLimits: [],
        };
        continue;
      }

      // Re-check on the device, not just the adapter: a device is allowed to
      // expose less than its adapter, and `requiredLimits` is a floor we asked
      // for rather than one we were granted.
      const limits = snapshotLimits(device.limits);
      const unmet = unmetLimitsOf(limits);
      if (unmet.length > 0) {
        safeDestroy(device);
        first ??= {
          reason: 'unmet-limits',
          featureLevel,
          limits,
          unmetLimits: unmet,
          error: 'the device exposes lower limits than its adapter',
        };
        continue;
      }

      const compatibilityMode =
        adapter.isCompatibilityMode ?? featureLevel === 'compatibility';
      const shared = new SharedDevice(
        device,
        constants,
        {
          featureLevel: compatibilityMode ? 'compatibility' : 'core',
          limits,
          adapterInfo: {
            vendor: adapter.info?.vendor ?? 'unknown',
            architecture: adapter.info?.architecture ?? 'unknown',
          },
          adopted: false,
        },
        this.hooks,
      );
      this.currentShared = shared;
      this.lastFailure = null;
      return shared;
    }
    return this.fail(first ?? { reason: 'no-adapter', unmetLimits: [] });
  }
}

/**
 * The process-wide manager.
 *
 * A module-level singleton is what makes "one device per page" true across
 * modules that do not know about each other: `render/scene.ts` and the particle
 * system both `acquire()` from here and get the same object. Inject a stub
 * `SharedDeviceManager` in tests instead of touching this one.
 */
export const sharedDevices = new SharedDeviceManager();

/** The shared device, or `null`; see `sharedDevices.failure` for why. */
export function acquireSharedDevice(
  options: SharedDeviceOptions = {},
): Promise<SharedDevice | null> {
  return sharedDevices.acquire(options);
}

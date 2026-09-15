/**
 * Which backend runs the soft bodies, and how a frame loop drives it.
 *
 * # The decision layer, not the solver
 *
 * The arithmetic lives in `softCpu.ts` and `softGpu.ts`; the graph passes live in
 * `softIslands.ts` and `softColoring.ts`. What this file decides is which of the
 * two backends a page gets, why, and who owns the device afterwards. Those are
 * the parts that fail quietly: a demo that renders on WebGL2 while its own HUD
 * says WebGPU, or a device that outlives the system that asked for it.
 *
 * # Tier selection is a probe, not a build flag
 *
 * `docs/development-plan.md` requires the WebGL2 fallback to be the result of
 * asking the browser at runtime, so nothing here reads `import.meta.env` and
 * nothing branches on a constant. `probeSoft` asks, `selectRenderTier` decides,
 * and the reason comes back attached to the handle. `onFallback` is how a caller
 * hears about a downgrade, and `handle.reason` is how a log reader does.
 *
 * # What a tier does and does not choose
 *
 * Only `webgpu` changes the simulation, because only WebGPU has compute here.
 * `webgl2` and `cpu` both step `softCpu.ts`; the tier says how the result is
 * drawn, and the WebGL2 renderer draws the same mesh the CPU backend just wrote.
 * A transform-feedback position-based solver would put a third integrator in the
 * tree, and the plan's parity gate compares exactly two.
 *
 * # The one soft-body-specific downgrade
 *
 * The solver binds fifteen storage buffers in one stage, against a WebGPU
 * baseline of eight. `capabilities.ts` floors the limits a *probe* insists on,
 * and that floor is shared with the particle layer, which needs far fewer
 * bindings -- so an adapter with eight storage buffers probes as available and
 * then refuses to build. That refusal is a `RangeError` carrying the device's own
 * number, and it arrives here as a downgrade with the number in the reason rather
 * than as a page that stopped rendering. `softGpu.assertStorageBuffers` throws it
 * before a single buffer is allocated, which is what keeps the fallback cheap.
 *
 * # Replay and training never reach this file's GPU path
 *
 * Asking for `tier: 'cpu'` skips the probe entirely: no adapter request, no
 * canvas, no context. A headless trainer on a machine with no GPU must not have
 * to satisfy a browser first, and the deterministic reference is the tier replay
 * is built on anyway -- `GpuSoftSystem.deterministic` is false, and `softTypes.ts`
 * says why.
 *
 * # Frames are not steps
 *
 * `SoftRunner` puts a `FixedClock` between the display and the simulation. A
 * 144 Hz panel and a 30 Hz one running for the same wall-clock second take the
 * same number of steps, which is the decoupling the plan asks for and the reason a
 * recorded run is comparable across machines. The clock also caps the backlog, so
 * a tab restored after ten minutes does not try to simulate ten minutes of cloth.
 */

import { FixedClock, type FixedClockOptions } from '../core/clock.js';
import {
  describeCapabilities,
  probeWebGpu,
  probeWebgl2,
  selectRenderTier,
  type CanvasLike,
  type FeatureLevel,
  type RenderTier,
  type RenderTierDecision,
  type WebGpuCapabilities,
  type Webgl2Capabilities,
} from './capabilities.js';
import {
  SharedDeviceManager,
  deviceGpuFrom,
  type DeviceGpuLike,
  type GpuConstants,
  type SharedDevice,
} from './device.js';
import type { TierFallback } from './particles.js';
import { createCpuSoftSystem } from './softCpu.js';
import { createGpuSoftSystem, type GpuSoftSystem } from './softGpu.js';
import type { SoftMesh } from './softMesh.js';
import type { SoftSimOptions, SoftSystem } from './softTypes.js';

// ---------------------------------------------------------------------------
// probing
// ---------------------------------------------------------------------------

export interface SoftProbeRequest {
  /** Injectable `navigator.gpu`, so the decision logic runs in bare Node. */
  readonly gpu?: DeviceGpuLike;
  /** Injectable canvas factory for the WebGL2 probe. */
  readonly makeCanvas?: () => CanvasLike | null;
  readonly featureLevels?: readonly FeatureLevel[];
}

/** What the browser answered, and what was decided from it. */
export interface SoftProbe {
  readonly webgpu: WebGpuCapabilities;
  readonly webgl2: Webgl2Capabilities;
  readonly decision: RenderTierDecision;
  /** Ready-to-print lines, so a HUD and a spec cannot format them differently. */
  readonly lines: readonly string[];
}

/** The answer for a tier that was forced, where nothing was asked. */
const NOT_PROBED: SoftProbe = {
  webgpu: {
    available: false,
    compatibilityMode: false,
    features: [],
    optionalFeatures: [],
    unmetLimits: [],
  },
  webgl2: { available: false, renderer: '', vendor: '' },
  decision: { tier: 'cpu', reason: 'not probed: the CPU tier was requested' },
  lines: ['not probed: the CPU tier was requested'],
};

/**
 * Ask the browser what it can do, and what that implies.
 *
 * Exported on its own because the demo prints the answer before it builds
 * anything: "which tier am I about to get, and why" is a question worth being
 * able to ask without allocating a hundred thousand nodes to find out. Note that
 * a `true` here does not promise the solver fits -- see the file header on the
 * storage-buffer count.
 */
export async function probeSoft(request: SoftProbeRequest = {}): Promise<SoftProbe> {
  const webgpu = await probeWebGpu(request.gpu ?? deviceGpuFrom(), {
    featureLevels: request.featureLevels,
  });
  const webgl2 = probeWebgl2(request.makeCanvas);
  return {
    webgpu,
    webgl2,
    decision: selectRenderTier(webgpu, webgl2),
    lines: describeCapabilities(webgpu, webgl2),
  };
}

/**
 * The probe for a device that already exists.
 *
 * three.js asks for its own `GPUDevice`, and requesting a second adapter while
 * holding one is how a page ends up with two contexts and a driver that picks the
 * wrong one. So the WebGPU half of this answer is read off the device's own `info`
 * rather than re-requested. WebGL2 is still probed for real: it is the tier this
 * factory falls back to when the handed-in device turns out to be lost, or too
 * small for the mesh, or short of storage buffers.
 */
function probeFromDevice(shared: SharedDevice, request: SoftProbeRequest): SoftProbe {
  const webgpu: WebGpuCapabilities = {
    available: true,
    featureLevel: shared.info.featureLevel,
    compatibilityMode: shared.info.featureLevel === 'compatibility',
    features: [],
    optionalFeatures: [],
    limits: shared.info.limits,
    info: shared.info.adapterInfo,
    unmetLimits: [],
  };
  const webgl2 = probeWebgl2(request.makeCanvas);
  return {
    webgpu,
    webgl2,
    decision: selectRenderTier(webgpu, webgl2),
    lines: describeCapabilities(webgpu, webgl2),
  };
}

// ---------------------------------------------------------------------------
// the factory
// ---------------------------------------------------------------------------

export interface SoftSystemRequest extends SoftProbeRequest {
  readonly mesh: SoftMesh;
  readonly options?: SoftSimOptions;
  /**
   * Force a tier. Omit to probe and take the best available.
   *
   * `'cpu'` is special: it is honoured without probing at all, so a headless run
   * never asks a browser for a GPU it does not need.
   */
  readonly tier?: RenderTier;
  /**
   * Refuse to downgrade. A tier that cannot be honoured throws instead of quietly
   * becoming a slower one, which is what a benchmark and a CI gate want and what a
   * demo does not.
   */
  readonly strict?: boolean;
  /** Injectable enum constants, for Node. Defaults to a `globalThis` lookup. */
  readonly constants?: GpuConstants | null;
  /** A device that already exists -- the three.js direction of M2. */
  readonly shared?: SharedDevice;
  /** A manager to acquire through. One is created per call when omitted. */
  readonly manager?: SharedDeviceManager;
  readonly onFallback?: (event: TierFallback) => void;
}

export interface SoftSystemHandle {
  readonly system: SoftSystem;
  /**
   * The GPU backend, when the GPU backend is what won.
   *
   * `system` is the contract every tier satisfies; this is the extra surface a
   * renderer needs -- `copyPublishedTo` for the position attribute, `readStats`
   * for the HUD, `readback` for a parity check -- and it is null rather than a
   * stub so a caller cannot mistake a CPU run for a GPU one.
   */
  readonly gpu: GpuSoftSystem | null;
  /** The tier that is actually running. */
  readonly tier: RenderTier;
  /** The tier that was asked for, before any downgrade. */
  readonly requested: RenderTier;
  /** Human-readable, and the only record of a downgrade that reached nobody. */
  readonly reason: string;
  readonly probe: SoftProbe;
  /** Non-null exactly when `tier === 'webgpu'`: the device a renderer should adopt. */
  readonly shared: SharedDevice | null;
  dispose(): void;
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function cpuHandle(
  request: SoftSystemRequest,
  tier: RenderTier,
  reason: string,
  probe: SoftProbe,
): SoftSystemHandle {
  const system = createCpuSoftSystem({ mesh: request.mesh, options: request.options });
  return {
    system,
    gpu: null,
    tier,
    requested: request.tier ?? probe.decision.tier,
    reason,
    probe,
    shared: null,
    dispose: () => system.dispose(),
  };
}

/**
 * Build the soft-body system the machine can actually run.
 *
 * Never throws for a missing capability: the whole point is that a page with no
 * WebGPU still gets a simulation. It throws only for a caller bug -- no mesh, a
 * mesh that does not fit its bounds, a `'wrap'` bounds mode -- and, under
 * `strict`, for a tier that was asked for by name and could not be honoured.
 */
export async function createSoftSystem(request: SoftSystemRequest): Promise<SoftSystemHandle> {
  if (request.tier === 'cpu') {
    return cpuHandle(request, 'cpu', NOT_PROBED.decision.reason, NOT_PROBED);
  }

  const probe = request.shared
    ? probeFromDevice(request.shared, request)
    : await probeSoft(request);
  const requested = request.tier ?? probe.decision.tier;

  if (requested === 'webgpu') {
    let shared = request.shared ?? null;
    let acquired: SharedDevice | null = null;
    try {
      if (!shared) {
        acquired = await acquireDevice(request);
        shared = acquired;
      }
      let system: GpuSoftSystem;
      try {
        system = await createGpuSoftSystem({
          shared,
          mesh: request.mesh,
          options: request.options,
        });
      } finally {
        // The reference `acquire()` took is not ours to keep in either outcome:
        // on success the system holds its own, and holding both would leave the
        // device alive after `dispose()`. On failure nothing holds any, and a
        // page that has just decided to run on the CPU is the worst place to
        // leak a GPU.
        acquired?.release();
      }
      return {
        system,
        gpu: system,
        tier: 'webgpu',
        requested,
        reason: probe.decision.reason,
        probe,
        shared,
        dispose: () => system.dispose(),
      };
    } catch (error) {
      return downgrade(request, probe, 'webgpu', messageOf(error));
    }
  }

  if (requested === 'webgl2' && !probe.webgl2.available) {
    return downgrade(request, probe, 'webgl2', 'no WebGL2 context');
  }

  return cpuHandle(request, requested, probe.decision.reason, probe);
}

/**
 * A forced or probed tier that cannot be honoured.
 *
 * Downgrading is the default because the acceptance criterion is that the page
 * keeps working; `strict` exists because a benchmark that silently ran on the CPU
 * reports a number that means nothing, and a CI gate that cannot tell is worse
 * than no gate.
 */
function downgrade(
  request: SoftSystemRequest,
  probe: SoftProbe,
  from: RenderTier,
  reason: string,
): SoftSystemHandle {
  if (request.strict) {
    throw new Error(`the ${from} tier cannot be honoured: ${reason}`);
  }
  const webgl2 = from === 'webgpu' && probe.webgl2.available;
  const to: RenderTier = webgl2 ? 'webgl2' : 'cpu';
  request.onFallback?.({ from, to, reason });
  return cpuHandle(request, to, `${from} fell back to ${to}: ${reason}`, probe);
}

/**
 * The device to run on: the caller's, or one acquired through a manager.
 *
 * Acquiring rather than requesting a device directly is what makes the M2 claim
 * hold at this layer too: a renderer that adopted its device into a manager and a
 * factory that acquires from the same manager end up sharing one `GPUDevice`, and
 * the refcount decides when it dies.
 */
async function acquireDevice(request: SoftSystemRequest): Promise<SharedDevice> {
  const options = {
    gpu: request.gpu,
    constants: request.constants,
    featureLevels: request.featureLevels,
  };
  const manager = request.manager ?? new SharedDeviceManager({ ...options, label: 'soft' });
  const shared = await manager.acquire(options);
  if (!shared) {
    const failure = manager.failure;
    throw new Error(
      `no WebGPU device (${failure?.reason ?? 'unknown'}${failure?.error ? `: ${failure.error}` : ''})`,
    );
  }
  return shared;
}

// ---------------------------------------------------------------------------
// the frame loop
// ---------------------------------------------------------------------------

/**
 * A simulation driven by wall-clock frames at a fixed step.
 *
 * `frame()` is what a render loop calls; the number it returns is how many steps
 * that frame was worth, which is 0 for a 240 Hz panel mid-step and 4 for a frame
 * that took 66 ms. The simulation never sees `frameDt`, so a cloth's trajectory
 * does not depend on how fast the display is.
 */
export class SoftRunner {
  readonly system: SoftSystem;
  readonly clock: FixedClock;

  private framesRun = 0;
  private behindFlag = false;

  constructor(system: SoftSystem, options: FixedClockOptions = {}) {
    this.system = system;
    // The system's own step is the default, and a different one is refused rather
    // than accepted. `frame()` steps with the clock's dt while `advance()` steps
    // with the system's, and `system.time` is `steps * system.fixedDt`, so a
    // runner with two step sizes would simulate one rate and report another. The
    // step belongs on the soft-body options, where the backend can honour it --
    // and on the GPU tier it has to, because it is a word in the uniform.
    const fixedDt = options.fixedDt ?? system.fixedDt;
    if (fixedDt !== system.fixedDt) {
      throw new RangeError(
        `the clock step (${fixedDt}) disagrees with the system step (${system.fixedDt}); ` +
          'set fixedDt in the soft-body options instead',
      );
    }
    this.clock = new FixedClock({ ...options, fixedDt });
  }

  get fixedDt(): number {
    return this.clock.fixedDt;
  }

  /** Interpolation blend for rendering between two steps, in `[0, 1)`. */
  get alpha(): number {
    return this.clock.alpha;
  }

  get steps(): number {
    return this.system.steps;
  }

  get time(): number {
    return this.system.time;
  }

  get frames(): number {
    return this.framesRun;
  }

  /**
   * True when the last frame hit `maxStepsPerFrame`.
   *
   * The clock drops the backlog rather than growing it, so a runner that is
   * persistently behind is simulating slower than the wall clock and the display
   * is showing stale state. On a solver whose cost per step grows with
   * `iterations * colors`, that is the first symptom of a mesh too big for the
   * machine, and the HUD should say it out loud.
   */
  get behind(): boolean {
    return this.behindFlag;
  }

  /**
   * Consume `frameDt` seconds of wall-clock time and step that many fixed slices.
   *
   * @returns how many steps ran, which is what a frame-rate-independent counter of
   *   "simulation work" should count.
   */
  frame(frameDt: number): number {
    const steps = this.clock.update(frameDt);
    for (let i = 0; i < steps; i++) this.system.step(this.clock.fixedDt);
    this.framesRun++;
    this.behindFlag = steps >= this.clock.maxStepsPerFrame;
    return steps;
  }

  /**
   * Run exactly `n` steps, ignoring the wall clock.
   *
   * The headless path: replay advances by step count, and the clock is moved along
   * with it so `time` still agrees with the system's.
   */
  advance(n: number): void {
    this.system.advance(n);
    this.clock.advance(n);
  }

  reset(): void {
    this.clock.reset();
    this.framesRun = 0;
    this.behindFlag = false;
  }

  dispose(): void {
    this.system.dispose();
  }
}

/** A runner over a freshly built system, for callers that want one call. */
export async function createSoftRunner(
  request: SoftSystemRequest,
  clock?: FixedClockOptions,
): Promise<{ readonly handle: SoftSystemHandle; readonly runner: SoftRunner }> {
  const handle = await createSoftSystem(request);
  return { handle, runner: new SoftRunner(handle.system, clock) };
}

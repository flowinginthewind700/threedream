/**
 * Which backend runs the particles, and how a frame loop drives it.
 *
 * # Tier selection is a probe, not a build flag
 *
 * `docs/development-plan.md` requires the WebGL2 fallback to be the result of
 * asking the browser at runtime, so nothing here reads `import.meta.env` and
 * nothing branches on a constant. `probeParticles` asks, `selectRenderTier`
 * decides, and the reason comes back attached to the system. A silent downgrade
 * is the failure mode that makes GPU work look intermittent rather than broken:
 * a driver regresses, the page still runs -- on WebGL2 -- and nothing says so.
 * `onFallback` is how a caller hears about it, and `handle.reason` is how a log
 * reader does.
 *
 * # What a tier does and does not choose
 *
 * Only `webgpu` changes the simulation, because only WebGPU has compute here.
 * `webgl2` and `cpu` both step `particleCpu.ts`; the tier says how the result is
 * drawn, and the WebGL2 renderer draws the same field the CPU backend just
 * wrote. A transform-feedback simulation would put a third integrator in the
 * tree, and the plan's parity gate compares exactly two.
 *
 * # Replay and training never reach this file's GPU path
 *
 * Asking for `tier: 'cpu'` skips the probe entirely: no adapter request, no
 * canvas, no context. That is what "回放与训练不依赖 GPU 层" has to mean in
 * practice -- a headless trainer on a machine with no GPU must not have to
 * satisfy a browser first.
 *
 * # Frames are not steps
 *
 * `ParticleRunner` puts a `FixedClock` between the display and the simulation.
 * A 144 Hz panel and a 30 Hz one running for the same wall-clock second take the
 * same number of steps, which is the decoupling the plan asks for and the reason
 * a recorded run is comparable across machines. The clock also caps the backlog,
 * so a tab restored after ten minutes does not try to simulate ten minutes.
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
import type { ParticleField } from './particleField.js';
import { createCpuParticleSystem } from './particleCpu.js';
import { createGpuParticleSystem, type GpuParticleSystem } from './particleGpu.js';
import type { ParticleSimOptions, ParticleSystem } from './particleTypes.js';

// ---------------------------------------------------------------------------
// probing
// ---------------------------------------------------------------------------

export interface ParticleProbeRequest {
  /** Injectable `navigator.gpu`, so the decision logic runs in bare Node. */
  readonly gpu?: DeviceGpuLike;
  /** Injectable canvas factory for the WebGL2 probe. */
  readonly makeCanvas?: () => CanvasLike | null;
  readonly featureLevels?: readonly FeatureLevel[];
}

/** What the browser answered, and what was decided from it. */
export interface ParticleProbe {
  readonly webgpu: WebGpuCapabilities;
  readonly webgl2: Webgl2Capabilities;
  readonly decision: RenderTierDecision;
  /** Ready-to-print lines, so a HUD and a spec cannot format them differently. */
  readonly lines: readonly string[];
}

/** The answer for a tier that was forced, where nothing was asked. */
const NOT_PROBED: ParticleProbe = {
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
 * able to ask without allocating 100k particles to find out.
 */
export async function probeParticles(
  request: ParticleProbeRequest = {},
): Promise<ParticleProbe> {
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
 * holding one is how a page ends up with two contexts and a driver that picks
 * the wrong one. So the WebGPU half of this answer is read off the device's own
 * `info` rather than re-requested. WebGL2 is still probed for real: it is the
 * tier this factory falls back to when the handed-in device turns out to be lost
 * or too small for the field, and that decision should not rest on a guess.
 */
function probeFromDevice(
  shared: SharedDevice,
  request: ParticleProbeRequest,
): ParticleProbe {
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

/** One tier gave way to another. The reason is the part worth keeping. */
export interface TierFallback {
  readonly from: RenderTier;
  readonly to: RenderTier;
  readonly reason: string;
}

export interface ParticleSystemRequest extends ParticleProbeRequest {
  readonly field: ParticleField;
  readonly options?: ParticleSimOptions;
  /**
   * Force a tier. Omit to probe and take the best available.
   *
   * `'cpu'` is special: it is honoured without probing at all, so a headless run
   * never asks a browser for a GPU it does not need.
   */
  readonly tier?: RenderTier;
  /**
   * Refuse to downgrade. A tier that cannot be honoured rejects instead of
   * quietly becoming a slower one, which is what a benchmark and a CI gate want
   * and what a demo does not.
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

export interface ParticleSystemHandle {
  readonly system: ParticleSystem;
  /**
   * The GPU backend, when the GPU backend is what won.
   *
   * `system` is the contract every tier satisfies; this is the extra surface a
   * renderer needs -- `copyPublishedTo` for the instance attribute, `readStats`
   * for the HUD -- and it is null rather than a stub so a caller cannot mistake a
   * CPU run for a GPU one.
   */
  readonly gpu: GpuParticleSystem | null;
  /** The tier that is actually running. */
  readonly tier: RenderTier;
  /** The tier that was asked for, before any downgrade. */
  readonly requested: RenderTier;
  /** Human-readable, and the only record of a downgrade that reached nobody. */
  readonly reason: string;
  readonly probe: ParticleProbe;
  /** Non-null exactly when `tier === 'webgpu'`: the device a renderer should adopt. */
  readonly shared: SharedDevice | null;
  dispose(): void;
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function cpuHandle(
  request: ParticleSystemRequest,
  tier: RenderTier,
  reason: string,
  probe: ParticleProbe,
): ParticleSystemHandle {
  const system = createCpuParticleSystem({ field: request.field, options: request.options });
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
 * Build the particle system the machine can actually run.
 *
 * Never throws for a missing capability: the whole point is that a page with no
 * WebGPU still gets a simulation. It throws only for a caller bug -- no field, a
 * field that does not fit its bounds -- and, under `strict`, for a tier that was
 * asked for by name and could not be honoured.
 */
export async function createParticleSystem(
  request: ParticleSystemRequest,
): Promise<ParticleSystemHandle> {
  if (request.tier === 'cpu') {
    return cpuHandle(request, 'cpu', NOT_PROBED.decision.reason, NOT_PROBED);
  }

  const probe = request.shared
    ? probeFromDevice(request.shared, request)
    : await probeParticles(request);
  const requested = request.tier ?? probe.decision.tier;

  if (requested === 'webgpu') {
    let shared = request.shared ?? null;
    let acquired: SharedDevice | null = null;
    try {
      if (!shared) {
        acquired = await acquireDevice(request);
        shared = acquired;
      }
      let system: GpuParticleSystem;
      try {
        system = await createGpuParticleSystem({
          shared,
          field: request.field,
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
  request: ParticleSystemRequest,
  probe: ParticleProbe,
  from: RenderTier,
  reason: string,
): ParticleSystemHandle {
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
 * hold at the particle layer: a renderer that adopted its device into a manager
 * and a factory that acquires from the same manager end up sharing one
 * `GPUDevice`, and the refcount decides when it dies.
 */
async function acquireDevice(request: ParticleSystemRequest): Promise<SharedDevice> {
  const options = {
    gpu: request.gpu,
    constants: request.constants,
    featureLevels: request.featureLevels,
  };
  const manager = request.manager ?? new SharedDeviceManager({ ...options, label: 'particles' });
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
 * that took 66 ms. The simulation never sees `frameDt`, so its results do not
 * depend on how fast the display is.
 */
export class ParticleRunner {
  readonly system: ParticleSystem;
  readonly clock: FixedClock;

  private framesRun = 0;
  private behindFlag = false;

  constructor(system: ParticleSystem, options: FixedClockOptions = {}) {
    this.system = system;
    // The system's own step is the default, and a different one is refused
    // rather than accepted. `frame()` steps with the clock's dt while `advance()`
    // steps with the system's, and `system.time` is `steps * system.fixedDt`, so
    // a runner with two step sizes would simulate one rate and report another.
    // The step belongs on the particle options, where the backend can honour it.
    const fixedDt = options.fixedDt ?? system.fixedDt;
    if (fixedDt !== system.fixedDt) {
      throw new RangeError(
        `the clock step (${fixedDt}) disagrees with the system step (${system.fixedDt}); ` +
          'set fixedDt in the particle options instead',
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
   * is showing stale state. That is a number the HUD should say out loud.
   */
  get behind(): boolean {
    return this.behindFlag;
  }

  /**
   * Consume `frameDt` seconds of wall-clock time and step that many fixed slices.
   *
   * @returns how many steps ran, which is what a frame-rate-independent counter
   *   of "simulation work" should count.
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
   * The headless path: training and replay advance by step count, and the clock
   * is moved along with them so `time` still agrees with the system's.
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
export async function createParticleRunner(
  request: ParticleSystemRequest,
  clock?: FixedClockOptions,
): Promise<{ readonly handle: ParticleSystemHandle; readonly runner: ParticleRunner }> {
  const handle = await createParticleSystem(request);
  return { handle, runner: new ParticleRunner(handle.system, clock) };
}

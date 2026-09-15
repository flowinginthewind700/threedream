/**
 * `gpu/particles.ts` -- which backend runs, and how a frame loop drives it.
 *
 * The arithmetic belongs to `tests/particle_cpu.test.ts` and
 * `tests/particle_gpu.test.ts`. What is pinned here is the decision layer: the
 * probe, the downgrade paths and the reasons they leave behind, who owns the
 * device reference, and the guarantee that a frame is not a step. Those are the
 * parts that fail quietly -- a page that renders on WebGL2 while the log says
 * WebGPU, or a device that outlives the system that asked for it.
 */

import { describe, expect, it } from 'vitest';

import type { CanvasLike } from '../src/gpu/capabilities.js';
import { SharedDeviceManager } from '../src/gpu/device.js';
import { createCpuParticleSystem } from '../src/gpu/particleCpu.js';
import { GpuParticleSystem } from '../src/gpu/particleGpu.js';
import { ParticleField } from '../src/gpu/particleField.js';
import {
  ParticleRunner,
  createParticleRunner,
  createParticleSystem,
  probeParticles,
  type ParticleSystemRequest,
  type TierFallback,
} from '../src/gpu/particles.js';
import type { ParticleSystem } from '../src/gpu/particleTypes.js';
import {
  STUB_CONSTANTS as CONSTANTS,
  StubAdapter,
  StubDevice,
  StubGpu,
  stubLimits,
} from './stub_webgpu.js';

// ---------------------------------------------------------------------------
// rig
// ---------------------------------------------------------------------------

/** The minimum a `probeWebgl2` asks of a context, including the debug extension. */
const UNMASKED_RENDERER = 0x9246;
const UNMASKED_VENDOR = 0x9245;
const GL = {
  getExtension: (name: string) =>
    name === 'WEBGL_debug_renderer_info'
      ? { UNMASKED_RENDERER_WEBGL: UNMASKED_RENDERER, UNMASKED_VENDOR_WEBGL: UNMASKED_VENDOR }
      : null,
  getParameter: (pname: number) =>
    pname === UNMASKED_RENDERER ? 'stub renderer' : 'stub vendor',
};

/** A canvas factory that answers `webgl2` and counts how often it was asked. */
function countingCanvas(webgl2 = true): {
  readonly make: () => CanvasLike | null;
  readonly state: { calls: number };
} {
  const state = { calls: 0 };
  const make = (): CanvasLike | null => {
    state.calls++;
    return {
      getContext: (type: string) => (webgl2 && type === 'webgl2' ? GL : null),
    };
  };
  return { make, state };
}

/** A canvas factory that answers nothing, for the headless case. */
function noCanvas(): () => CanvasLike | null {
  return () => null;
}

function seededField(count = 64): ParticleField {
  return new ParticleField({ count, scene: 'sphere', seed: 7, radius: [0.1, 0.1], speed: 2 });
}

/** A `navigator.gpu` that answers the core level with a real-looking adapter. */
function gpuStub(device = new StubDevice({ limits: stubLimits() })): StubGpu {
  return new StubGpu({ core: new StubAdapter({ device }) });
}

/** The request every WebGPU path in Node needs: an injected `gpu` and enums. */
function webgpuRequest(over: Partial<ParticleSystemRequest> = {}): ParticleSystemRequest {
  return { field: seededField(), gpu: gpuStub(), constants: CONSTANTS, ...over };
}

function managerFor(gpu: StubGpu): SharedDeviceManager {
  return new SharedDeviceManager({ gpu, constants: CONSTANTS, label: 'spec' });
}

/** Every fallback a call reported, in order. */
function fallbackSink(): {
  readonly events: TierFallback[];
  readonly onFallback: (event: TierFallback) => void;
} {
  const events: TierFallback[] = [];
  return { events, onFallback: (event) => events.push(event) };
}

/** A CPU system over a fresh seeded field: what the runner specs drive. */
function cpuSystem(fixedDt?: number, count = 32): ParticleSystem {
  return createCpuParticleSystem({
    field: seededField(count),
    ...(fixedDt === undefined ? {} : { options: { fixedDt } }),
  });
}

// ---------------------------------------------------------------------------
// the probe
// ---------------------------------------------------------------------------

describe('probeParticles', () => {
  it('reports the CPU tier when the browser answers with neither API', async () => {
    const probe = await probeParticles({ gpu: new StubGpu({}), makeCanvas: noCanvas() });
    expect(probe.webgpu.available).toBe(false);
    expect(probe.webgpu.reason).toBe('no-adapter');
    expect(probe.webgl2.available).toBe(false);
    expect(probe.decision.tier).toBe('cpu');
    expect(probe.lines.join('\n')).toContain('webgpu: unavailable');
  });

  it('prefers WebGPU when an adapter answers, and says which feature level', async () => {
    const probe = await probeParticles({ gpu: gpuStub(), makeCanvas: countingCanvas().make });
    expect(probe.decision.tier).toBe('webgpu');
    expect(probe.decision.reason).toContain('core feature level');
    expect(probe.webgpu.featureLevel).toBe('core');
    expect(probe.lines[0]).toContain('core adapter');
    expect(probe.lines[0]).toContain('intel');
  });

  it('falls back to WebGL2 when only a canvas answers', async () => {
    const probe = await probeParticles({ gpu: new StubGpu({}), makeCanvas: countingCanvas().make });
    expect(probe.decision.tier).toBe('webgl2');
    expect(probe.webgl2.available).toBe(true);
    expect(probe.webgl2.renderer).toBe('stub renderer');
    expect(probe.webgl2.vendor).toBe('stub vendor');
    expect(probe.decision.reason).toContain('WebGL2 fallback');
    expect(probe.lines.join('\n')).toContain('stub renderer');
  });

  it('reports unmet limits as the reason, not as an available adapter', async () => {
    const small = new StubDevice({ limits: stubLimits({ maxStorageBufferBindingSize: 1024 }) });
    const probe = await probeParticles({
      gpu: new StubGpu({ core: new StubAdapter({ device: small, limits: stubLimits({ maxStorageBufferBindingSize: 1024 }) }) }),
      makeCanvas: countingCanvas().make,
    });
    expect(probe.webgpu.available).toBe(false);
    expect(probe.webgpu.reason).toBe('unmet-limits');
    expect(probe.webgpu.unmetLimits).toContain('maxStorageBufferBindingSize');
    expect(probe.decision.tier).toBe('webgl2');
  });
});

// ---------------------------------------------------------------------------
// tier selection
// ---------------------------------------------------------------------------

describe('createParticleSystem: which backend won', () => {
  it('builds the GPU backend when WebGPU answers', async () => {
    const handle = await createParticleSystem(webgpuRequest());
    expect(handle.tier).toBe('webgpu');
    expect(handle.requested).toBe('webgpu');
    expect(handle.gpu).toBeInstanceOf(GpuParticleSystem);
    expect(handle.system).toBe(handle.gpu);
    expect(handle.system.name).toBe('gpu');
    expect(handle.system.deterministic).toBe(false);
    expect(handle.shared).not.toBeNull();
    expect(handle.shared?.references).toBe(1);
    handle.dispose();
  });

  it('steps the backend it built', async () => {
    const handle = await createParticleSystem(webgpuRequest());
    handle.system.step();
    expect(handle.system.steps).toBe(1);
    expect(handle.system.stats().escaped).toBe(0);
    handle.dispose();
  });

  it('builds the CPU backend when the browser answers with nothing', async () => {
    const handle = await createParticleSystem({
      field: seededField(),
      gpu: new StubGpu({}),
      constants: CONSTANTS,
      makeCanvas: noCanvas(),
    });
    expect(handle.tier).toBe('cpu');
    expect(handle.system.name).toBe('cpu');
    expect(handle.system.deterministic).toBe(true);
    expect(handle.gpu).toBeNull();
    expect(handle.shared).toBeNull();
    handle.dispose();
  });

  it('keeps the CPU simulation when only WebGL2 answers', async () => {
    // The tier says how the field is drawn. Only WebGPU has compute here, so a
    // WebGL2 machine still simulates on the CPU -- a transform-feedback
    // integrator would be a third backend, and the parity gate compares two.
    const handle = await createParticleSystem({
      field: seededField(),
      gpu: new StubGpu({}),
      constants: CONSTANTS,
      makeCanvas: countingCanvas().make,
    });
    expect(handle.tier).toBe('webgl2');
    expect(handle.system.name).toBe('cpu');
    expect(handle.gpu).toBeNull();
    expect(handle.shared).toBeNull();
    handle.dispose();
  });

  it('honours a forced CPU tier without asking the browser anything', async () => {
    const gpu = gpuStub();
    const canvas = countingCanvas();
    const handle = await createParticleSystem({
      field: seededField(),
      gpu,
      constants: CONSTANTS,
      makeCanvas: canvas.make,
      tier: 'cpu',
    });
    expect(gpu.asked).toEqual([]);
    expect(canvas.state.calls).toBe(0);
    expect(handle.tier).toBe('cpu');
    expect(handle.requested).toBe('cpu');
    expect(handle.probe.decision.reason).toContain('not probed');
    expect(handle.system.name).toBe('cpu');
    handle.dispose();
  });

  it('honours a forced WebGL2 tier even when WebGPU is available', async () => {
    const handle = await createParticleSystem(
      webgpuRequest({ tier: 'webgl2', makeCanvas: countingCanvas().make }),
    );
    expect(handle.requested).toBe('webgl2');
    expect(handle.tier).toBe('webgl2');
    expect(handle.system.name).toBe('cpu');
    expect(handle.gpu).toBeNull();
    handle.dispose();
  });
});

// ---------------------------------------------------------------------------
// downgrades
// ---------------------------------------------------------------------------

describe('createParticleSystem: downgrades', () => {
  it('falls from webgpu to webgl2, and reports it', async () => {
    const sink = fallbackSink();
    const handle = await createParticleSystem({
      field: seededField(),
      gpu: new StubGpu({}),
      constants: CONSTANTS,
      makeCanvas: countingCanvas().make,
      tier: 'webgpu',
      onFallback: sink.onFallback,
    });
    expect(handle.requested).toBe('webgpu');
    expect(handle.tier).toBe('webgl2');
    expect(handle.reason).toContain('webgpu fell back to webgl2');
    expect(handle.reason).toContain('no WebGPU device');
    expect(sink.events).toHaveLength(1);
    expect(sink.events[0].from).toBe('webgpu');
    expect(sink.events[0].to).toBe('webgl2');
    expect(sink.events[0].reason).toContain('no WebGPU device');
    handle.dispose();
  });

  it('rejects instead of downgrading under strict', async () => {
    await expect(
      createParticleSystem({
        field: seededField(),
        gpu: new StubGpu({}),
        constants: CONSTANTS,
        makeCanvas: countingCanvas().make,
        tier: 'webgpu',
        strict: true,
      }),
    ).rejects.toThrow(/the webgpu tier cannot be honoured/);
  });

  it('falls from webgl2 to cpu when there is no canvas', async () => {
    const sink = fallbackSink();
    const handle = await createParticleSystem({
      field: seededField(),
      gpu: new StubGpu({}),
      constants: CONSTANTS,
      makeCanvas: noCanvas(),
      tier: 'webgl2',
      onFallback: sink.onFallback,
    });
    expect(handle.tier).toBe('cpu');
    expect(handle.reason).toContain('webgl2 fell back to cpu');
    expect(sink.events.map((e) => [e.from, e.to])).toEqual([['webgl2', 'cpu']]);
    handle.dispose();
  });

  it('rejects a forced webgl2 tier with no canvas under strict', async () => {
    await expect(
      createParticleSystem({
        field: seededField(),
        gpu: gpuStub(),
        constants: CONSTANTS,
        makeCanvas: noCanvas(),
        tier: 'webgl2',
        strict: true,
      }),
    ).rejects.toThrow(/the webgl2 tier cannot be honoured: no WebGL2 context/);
  });

  it('destroys the device it acquired when the backend fails to build', async () => {
    // The reference `acquire()` took has to go back even on the failure path: a
    // page that has just decided to run on the CPU is the worst place to leave a
    // GPU pinned for the lifetime of the tab.
    const device = new StubDevice({ limits: stubLimits() });
    device.setCompilationMessages([
      { type: 'error', message: 'unknown identifier', lineNum: 3 },
    ]);
    const gpu = new StubGpu({ core: new StubAdapter({ device }) });
    const manager = managerFor(gpu);
    const sink = fallbackSink();
    const handle = await createParticleSystem({
      field: seededField(),
      gpu,
      constants: CONSTANTS,
      manager,
      makeCanvas: countingCanvas().make,
      onFallback: sink.onFallback,
    });
    expect(handle.tier).toBe('webgl2');
    expect(handle.system.name).toBe('cpu');
    expect(handle.reason).toContain('line 3');
    expect(device.buffers).toHaveLength(0);
    expect(device.destroyCalls).toBe(1);
    expect(manager.current).toBeNull();
    expect(sink.events.map((e) => e.to)).toEqual(['webgl2']);
    handle.dispose();
  });

  it('rejects under strict when the shader fails to compile', async () => {
    const device = new StubDevice({ limits: stubLimits() });
    device.setCompilationMessages([{ type: 'error', message: 'bad', lineNum: 1 }]);
    const gpu = new StubGpu({ core: new StubAdapter({ device }) });
    await expect(
      createParticleSystem({
        field: seededField(),
        gpu,
        constants: CONSTANTS,
        manager: managerFor(gpu),
        strict: true,
      }),
    ).rejects.toThrow(/cannot be honoured/);
    expect(device.destroyCalls).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// the shared device
// ---------------------------------------------------------------------------

describe('createParticleSystem: a device handed in', () => {
  it('runs on the caller device without requesting an adapter', async () => {
    const gpu = gpuStub();
    const manager = new SharedDeviceManager({ constants: CONSTANTS });
    const device = new StubDevice({ limits: stubLimits() });
    const shared = manager.adopt(device);
    const handle = await createParticleSystem({
      field: seededField(),
      gpu,
      constants: CONSTANTS,
      shared,
      makeCanvas: countingCanvas().make,
    });
    // The three.js direction: it already asked for a device, and asking for a
    // second adapter is how a page ends up with two contexts.
    expect(gpu.asked).toEqual([]);
    expect(handle.tier).toBe('webgpu');
    expect(handle.shared).toBe(shared);
    expect(handle.gpu).toBeInstanceOf(GpuParticleSystem);
    expect(shared.references).toBe(2);
    handle.dispose();
    expect(shared.references).toBe(1);
    expect(device.destroyCalls).toBe(0);
  });

  it('reads the WebGPU probe off the device and still probes WebGL2', async () => {
    const manager = new SharedDeviceManager({ constants: CONSTANTS });
    const device = new StubDevice({ limits: stubLimits() });
    const shared = manager.adopt(device, {
      featureLevel: 'compatibility',
      adapterInfo: { vendor: 'qualcomm', architecture: 'adreno' },
    });
    const canvas = countingCanvas();
    const handle = await createParticleSystem({
      field: seededField(),
      constants: CONSTANTS,
      shared,
      makeCanvas: canvas.make,
    });
    expect(handle.probe.webgpu.available).toBe(true);
    expect(handle.probe.webgpu.compatibilityMode).toBe(true);
    expect(handle.probe.webgpu.limits).toBe(shared.info.limits);
    expect(handle.probe.webgpu.info).toEqual({ vendor: 'qualcomm', architecture: 'adreno' });
    expect(handle.probe.decision.reason).toContain('compatibility feature level');
    expect(handle.probe.lines[0]).toContain('compatibility adapter');
    // WebGL2 is the tier this falls back to, so it is asked for real.
    expect(canvas.state.calls).toBeGreaterThan(0);
    expect(handle.probe.webgl2.available).toBe(true);
    handle.dispose();
  });

  it('gives the caller device back untouched when the field does not fit it', async () => {
    const manager = new SharedDeviceManager({ constants: CONSTANTS });
    const device = new StubDevice({ limits: stubLimits({ maxStorageBufferBindingSize: 1024 }) });
    const shared = manager.adopt(device);
    const handle = await createParticleSystem({
      field: seededField(64),
      constants: CONSTANTS,
      shared,
      makeCanvas: noCanvas(),
    });
    expect(handle.tier).toBe('cpu');
    expect(handle.system.name).toBe('cpu');
    expect(handle.reason).toContain('webgpu fell back to cpu');
    // The caller's reference is not ours to release, and its device is not ours
    // to destroy: it belongs to whoever handed it over.
    expect(shared.references).toBe(1);
    expect(device.destroyCalls).toBe(0);
    expect(device.buffers).toHaveLength(0);
    handle.dispose();
  });

  it('rejects under strict when the caller device is too small', async () => {
    const manager = new SharedDeviceManager({ constants: CONSTANTS });
    const device = new StubDevice({ limits: stubLimits({ maxStorageBufferBindingSize: 1024 }) });
    const shared = manager.adopt(device);
    await expect(
      createParticleSystem({ field: seededField(64), constants: CONSTANTS, shared, strict: true }),
    ).rejects.toThrow(/the webgpu tier cannot be honoured/);
    expect(shared.references).toBe(1);
    expect(device.destroyCalls).toBe(0);
  });

  it('downgrades when the caller device has already been destroyed', async () => {
    const manager = new SharedDeviceManager({ constants: CONSTANTS });
    const shared = manager.adopt(new StubDevice({ limits: stubLimits() }));
    shared.destroy();
    const handle = await createParticleSystem({
      field: seededField(),
      constants: CONSTANTS,
      shared,
      makeCanvas: noCanvas(),
    });
    expect(handle.tier).toBe('cpu');
    expect(handle.reason).toContain('destroyed');
  });

  it('shares one device between two systems acquired through one manager', async () => {
    const device = new StubDevice({ limits: stubLimits() });
    const adapter = new StubAdapter({ device });
    const gpu = new StubGpu({ core: adapter });
    const manager = managerFor(gpu);
    const first = await createParticleSystem({
      field: seededField(32),
      gpu,
      constants: CONSTANTS,
      manager,
    });
    const second = await createParticleSystem({
      field: seededField(32),
      gpu,
      constants: CONSTANTS,
      manager,
    });
    expect(first.shared).toBe(second.shared);
    expect(adapter.deviceRequests).toHaveLength(1);
    expect(first.shared?.references).toBe(2);
    first.dispose();
    expect(device.destroyCalls).toBe(0);
    expect(second.shared?.references).toBe(1);
    second.dispose();
    expect(device.destroyCalls).toBe(1);
    expect(manager.current).toBeNull();
  });

  it('acquires its own device when no manager is supplied', async () => {
    const gpu = gpuStub();
    const handle = await createParticleSystem({ field: seededField(), gpu, constants: CONSTANTS });
    expect(gpu.asked).toContain('core');
    expect(handle.shared?.info.adopted).toBe(false);
    handle.dispose();
  });
});

// ---------------------------------------------------------------------------
// the frame loop
// ---------------------------------------------------------------------------

describe('ParticleRunner', () => {
  /**
   * Dyadic step and frame sizes on purpose.
   *
   * 1/60 accumulated 144 times as 1/144 lands on 59 steps rather than 60, which
   * is float drift and not a property of the clock. Halves and quarters are
   * exact in binary, so what this asserts is the decoupling itself.
   */
  const FIXED = 1 / 64;

  function cpuRunner(clock: { fixedDt?: number; maxStepsPerFrame?: number } = {}): ParticleRunner {
    return new ParticleRunner(cpuSystem(FIXED), { fixedDt: FIXED, ...clock });
  }

  it('takes the same number of steps at 32 fps and at 128 fps', () => {
    const slow = cpuRunner();
    const fast = cpuRunner();
    let slowSteps = 0;
    let fastSteps = 0;
    for (let i = 0; i < 32; i++) slowSteps += slow.frame(1 / 32);
    for (let i = 0; i < 128; i++) fastSteps += fast.frame(1 / 128);
    expect(slowSteps).toBe(64);
    expect(fastSteps).toBe(64);
    expect(slow.steps).toBe(fast.steps);
    expect(slow.time).toBeCloseTo(fast.time, 12);
    // The acceptance criterion behind the decoupling: same wall-clock second,
    // same simulated bytes, whatever the panel did in between.
    expect(slow.system.digest()).toBe(fast.system.digest());
    expect(slow.frames).toBe(32);
    expect(fast.frames).toBe(128);
  });

  it('runs nothing for a frame shorter than a step and reports the blend', () => {
    const runner = cpuRunner();
    expect(runner.frame(FIXED / 4)).toBe(0);
    expect(runner.steps).toBe(0);
    expect(runner.alpha).toBeCloseTo(0.25, 12);
    expect(runner.frame(FIXED)).toBe(1);
    expect(runner.steps).toBe(1);
  });

  it('caps the backlog and says so, then recovers', () => {
    const runner = cpuRunner({ maxStepsPerFrame: 3 });
    expect(runner.frame(10)).toBe(3);
    expect(runner.behind).toBe(true);
    // A tab restored after ten minutes must not simulate ten minutes.
    expect(runner.time).toBeCloseTo(3 * FIXED, 12);
    expect(runner.clock.time).toBeCloseTo(3 * FIXED, 12);
    expect(runner.frame(FIXED)).toBeGreaterThanOrEqual(1);
    expect(runner.behind).toBe(false);
  });

  it('defaults maxStepsPerFrame to the clock default of eight', () => {
    const runner = cpuRunner();
    expect(runner.frame(10)).toBe(8);
    expect(runner.behind).toBe(true);
  });

  it('advances by step count, ignoring the wall clock', () => {
    const runner = cpuRunner();
    runner.advance(5);
    expect(runner.steps).toBe(5);
    expect(runner.time).toBeCloseTo(5 * FIXED, 12);
    expect(runner.clock.time).toBeCloseTo(runner.time, 12);
    expect(runner.frames).toBe(0);
    expect(runner.alpha).toBe(0);
  });

  it('defaults its step to the system step, so both report one time', () => {
    const system = cpuSystem();
    const runner = new ParticleRunner(system);
    expect(runner.fixedDt).toBe(system.fixedDt);
    runner.frame(system.fixedDt);
    expect(runner.steps).toBe(1);
    expect(runner.time).toBeCloseTo(runner.clock.time, 12);
  });

  it('refuses a clock step the system was not built for', () => {
    // `frame()` steps with the clock's dt and `advance()` with the system's, so
    // a runner holding two step sizes would simulate one rate and report another.
    const system = cpuSystem();
    expect(() => new ParticleRunner(system, { fixedDt: system.fixedDt / 2 })).toThrow(
      /disagrees with the system step/,
    );
    expect(() => new ParticleRunner(system, { fixedDt: system.fixedDt })).not.toThrow();
  });

  it('resets the frame counters', () => {
    const runner = cpuRunner({ maxStepsPerFrame: 2 });
    runner.frame(10);
    expect(runner.behind).toBe(true);
    runner.reset();
    expect(runner.behind).toBe(false);
    expect(runner.frames).toBe(0);
    expect(runner.clock.steps).toBe(0);
  });

  it('disposes the system it drives', () => {
    const runner = cpuRunner();
    runner.dispose();
    expect(() => runner.frame(FIXED)).toThrow(/disposed/);
  });

  it('drives the GPU backend through the same frame loop', async () => {
    const device = new StubDevice({ limits: stubLimits() });
    const gpu = new StubGpu({ core: new StubAdapter({ device }) });
    const handle = await createParticleSystem({
      field: seededField(64),
      gpu,
      constants: CONSTANTS,
      manager: managerFor(gpu),
    });
    const runner = new ParticleRunner(handle.system);
    const before = device.passes.length;
    expect(runner.frame(runner.fixedDt)).toBe(1);
    expect(runner.steps).toBe(1);
    expect(device.passes.length).toBeGreaterThan(before);
    runner.dispose();
    expect(device.destroyCalls).toBe(1);
  });

  it('builds a runner and its handle in one call', async () => {
    const { handle, runner } = await createParticleRunner(
      {
        field: seededField(),
        gpu: new StubGpu({}),
        constants: CONSTANTS,
        makeCanvas: noCanvas(),
        options: { fixedDt: FIXED },
      },
      { fixedDt: FIXED },
    );
    expect(handle.tier).toBe('cpu');
    expect(runner.system).toBe(handle.system);
    expect(runner.fixedDt).toBe(FIXED);
    expect(runner.frame(FIXED)).toBe(1);
    runner.dispose();
  });
});

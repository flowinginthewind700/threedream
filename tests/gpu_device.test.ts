/**
 * Device lifetime is exactly what a real-GPU test is worst at: a lost device, a
 * refcount reaching zero, two callers racing for the same adapter -- none of
 * those can be produced on cue against real hardware. So the manager is driven
 * here by stubs, and `e2e/shared_device.spec.ts` keeps the real-adapter half
 * honest. What is pinned below is the contract the rest of the GPU layer leans
 * on: one device, a reason for every failure, and recovery after loss.
 */

import { describe, expect, it } from 'vitest';

import { LIMIT_FLOOR } from '../src/gpu/capabilities.js';
import {
  REQUESTED_LIMITS,
  SharedDeviceManager,
  acquireSharedDevice,
  deviceGpuFrom,
  gpuConstantsFrom,
  requiredLimitsFor,
  sharedDevices,
  type GpuConstants,
} from '../src/gpu/device.js';
import {
  STUB_CONSTANTS as CONSTANTS,
  StubAdapter,
  StubDevice,
  StubGpu,
  stubLimits as coreLimits,
  type StubAdapterSpec,
  type StubLevelSpec,
} from './stub_webgpu.js';

const MIB = 1024 * 1024;

/** Stub constructors, so the cases below read as data rather than as plumbing. */
const stubDevice = (
  init: { limits?: Record<string, number>; features?: string[] } = {},
): StubDevice => new StubDevice(init);
const stubAdapter = (spec: StubAdapterSpec = {}): StubAdapter => new StubAdapter(spec);
const stubGpu = (
  byLevel: Partial<Record<'core' | 'compatibility', StubLevelSpec>>,
): StubGpu => new StubGpu(byLevel);

function manager(gpu: StubGpu | undefined, constants: GpuConstants | null = CONSTANTS): SharedDeviceManager {
  return new SharedDeviceManager({ gpu, constants });
}

/** `acquire` plus the assertion that it worked, so the tests read as intent. */
async function mustAcquire(m: SharedDeviceManager) {
  const shared = await m.acquire();
  expect(shared, `acquire failed: ${JSON.stringify(m.failure)}`).not.toBeNull();
  return shared!;
}

// ---------------------------------------------------------------------------

describe('the WebGPU globals', () => {
  it('reads navigator.gpu through the probe helper, and says when there is none', () => {
    const gpu = { requestAdapter: async () => null };
    expect(deviceGpuFrom({ navigator: { gpu } })).toBe(gpu);
    expect(deviceGpuFrom({})).toBeUndefined();
    expect(deviceGpuFrom({ navigator: {} })).toBeUndefined();
  });

  it('reads the enum constants, and refuses to guess them', () => {
    expect(gpuConstantsFrom({})).toBeNull();
    expect(
      gpuConstantsFrom({ GPUBufferUsage: CONSTANTS.bufferUsage, GPUMapMode: CONSTANTS.mapMode }),
    ).toBeNull();
    const scope = {
      GPUBufferUsage: CONSTANTS.bufferUsage,
      GPUMapMode: CONSTANTS.mapMode,
      GPUShaderStage: CONSTANTS.shaderStage,
    };
    expect(gpuConstantsFrom(scope)).toEqual(CONSTANTS);
  });

  it('requests exactly the limits the adapter reports, at the value it reports', () => {
    const adapter = stubAdapter({
      limits: coreLimits({ maxStorageBuffersPerShaderStage: 12, maxBindGroups: 6 }),
    });
    const required = requiredLimitsFor(adapter);
    expect(Object.keys(required).sort()).toEqual([...REQUESTED_LIMITS].sort());
    expect(required['maxStorageBuffersPerShaderStage']).toBe(12);
    expect(required['maxBindGroups']).toBe(6);
    expect(required['maxStorageBufferBindingSize']).toBe(128 * MIB);
  });

  it('leaves out a limit the adapter does not report, rather than inventing one', () => {
    const limits = coreLimits();
    delete limits['maxBufferSize'];
    limits['maxBindGroups'] = Number.NaN;
    const required = requiredLimitsFor(stubAdapter({ limits }));
    expect('maxBufferSize' in required).toBe(false);
    expect('maxBindGroups' in required).toBe(false);
    expect(required['maxComputeInvocationsPerWorkgroup']).toBe(256);
  });

  it('asks for at least the floor on every limit that decides usability', () => {
    // A device that cannot hold a 16 MiB binding cannot hold the field, so the
    // request has to say so rather than accept whatever the default is.
    const required = requiredLimitsFor(stubAdapter());
    for (const name of Object.keys(LIMIT_FLOOR)) {
      expect(required[name], `${name} must be requested`).toBeGreaterThanOrEqual(LIMIT_FLOOR[name as keyof typeof LIMIT_FLOOR]);
    }
  });
});

describe('acquire', () => {
  it('reports "absent" when there is no navigator.gpu at all', async () => {
    const m = manager(undefined);
    expect(await m.acquire()).toBeNull();
    expect(m.failure).toMatchObject({ reason: 'absent', unmetLimits: [] });
  });

  it('reports "absent" when the enum constants are missing, because buffers could not be created', async () => {
    const m = manager(stubGpu({ core: {} }), null);
    expect(await m.acquire()).toBeNull();
    expect(m.failure?.reason).toBe('absent');
    expect(m.failure?.error).toMatch(/GPUBufferUsage/);
  });

  it('reports "no-adapter" after trying every feature level', async () => {
    const gpu = stubGpu({ core: null, compatibility: null });
    const m = manager(gpu);
    expect(await m.acquire()).toBeNull();
    expect(m.failure?.reason).toBe('no-adapter');
    expect(gpu.asked).toEqual(['core', 'compatibility']);
  });

  it('reports "error" when requestAdapter throws, and does not retry', async () => {
    const gpu = stubGpu({ core: new Error('driver blocklisted'), compatibility: {} });
    const m = manager(gpu);
    expect(await m.acquire()).toBeNull();
    expect(m.failure).toMatchObject({ reason: 'error', error: 'driver blocklisted' });
    expect(gpu.asked).toEqual(['core']);
  });

  it('reports the unmet limits when the adapter is too small, then falls back a level', async () => {
    const gpu = stubGpu({
      core: { limits: coreLimits({ maxStorageBufferBindingSize: 4 * MIB }) },
      compatibility: { isCompatibilityMode: true },
    });
    const m = manager(gpu);
    const shared = await mustAcquire(m);
    expect(shared.info.featureLevel).toBe('compatibility');
    expect(gpu.asked).toEqual(['core', 'compatibility']);
  });

  it('keeps the first rejection as the diagnosis when no level is usable', async () => {
    const gpu = stubGpu({
      core: { limits: coreLimits({ maxBufferSize: MIB }) },
      compatibility: { limits: coreLimits({ maxBufferSize: MIB, maxStorageBufferBindingSize: MIB }) },
    });
    const m = manager(gpu);
    expect(await m.acquire()).toBeNull();
    expect(m.failure?.reason).toBe('unmet-limits');
    // The core adapter's rejection, not the compatibility one: the first level
    // tried is the one whose numbers say the most about the hardware.
    expect(m.failure?.unmetLimits).toEqual(['maxBufferSize']);
    expect(m.failure?.featureLevel).toBe('core');
  });

  it('hands back the device, its limits and its adapter info', async () => {
    const device = stubDevice({ limits: coreLimits({ maxStorageBuffersPerShaderStage: 12 }) });
    const adapter = stubAdapter({ device, info: { vendor: 'nvidia', architecture: 'ampere' } });
    const m = manager(stubGpu({ core: adapter }));
    const shared = await mustAcquire(m);
    expect(shared.device).toBe(device);
    expect(shared.constants).toBe(CONSTANTS);
    expect(shared.info).toEqual({
      featureLevel: 'core',
      limits: expect.objectContaining({ maxStorageBuffersPerShaderStage: 12 }),
      adapterInfo: { vendor: 'nvidia', architecture: 'ampere' },
      adopted: false,
    });
    expect(shared.references).toBe(1);
    expect(shared.usable).toBe(true);
    expect(m.failure).toBeNull();
    expect(m.current).toBe(shared);
  });

  it('asks the device for the adapter limits and the label, and for no features by default', async () => {
    const adapter = stubAdapter();
    const m = manager(stubGpu({ core: adapter }));
    await mustAcquire(m);
    expect(adapter.deviceRequests).toHaveLength(1);
    const request = adapter.deviceRequests[0]!;
    expect(request.label).toBe('threedream');
    expect(request.requiredLimits).toEqual(requiredLimitsFor(adapter));
    expect('requiredFeatures' in request).toBe(false);
  });

  it('passes requiredFeatures through when asked, and honours a per-call label', async () => {
    const device = stubDevice({ features: ['timestamp-query'] });
    const adapter = stubAdapter({ features: ['timestamp-query'], device });
    const m = new SharedDeviceManager({ gpu: stubGpu({ core: adapter }), constants: CONSTANTS });
    const shared = await m.acquire({ requiredFeatures: ['timestamp-query'], label: 'particles' });
    expect(shared?.device).toBe(device);
    expect(device.features.has('timestamp-query')).toBe(true);
    expect(adapter.deviceRequests[0]).toMatchObject({
      label: 'particles',
      requiredFeatures: ['timestamp-query'],
    });
  });

  it('rejects a device that exposes less than its adapter promised', async () => {
    // The case the re-check exists for: `requiredLimits` is a request, and an
    // implementation is allowed to hand back a device that ignored it.
    const small = stubDevice({ limits: coreLimits({ maxBufferSize: 2 * MIB }) });
    const gpu = stubGpu({ core: { device: small }, compatibility: {} });
    const m = manager(gpu);
    const shared = await mustAcquire(m);
    expect(small.destroyed, 'the unusable device must be destroyed').toBe(true);
    expect(shared.device).not.toBe(small);
    expect(shared.info.featureLevel).toBe('compatibility');
  });

  it('reports "error" when requestDevice throws and no level is left', async () => {
    const gpu = stubGpu({ core: { device: new Error('device limit exceeded') } });
    const m = manager(gpu);
    expect(await m.acquire()).toBeNull();
    expect(m.failure).toMatchObject({ reason: 'error', error: 'device limit exceeded' });
    expect(m.failure?.featureLevel).toBe('core');
  });

  it('honours a per-call feature level list', async () => {
    const gpu = stubGpu({ core: {}, compatibility: {} });
    const m = manager(gpu);
    const shared = await m.acquire({ featureLevels: ['compatibility'] });
    expect(shared?.info.featureLevel).toBe('compatibility');
    expect(gpu.asked).toEqual(['compatibility']);
  });

  it('trusts the adapter over the request when it says it is compatibility mode', async () => {
    const gpu = stubGpu({ core: { isCompatibilityMode: true } });
    const shared = await mustAcquire(manager(gpu));
    expect(shared.info.featureLevel).toBe('compatibility');
  });
});

describe('one device, shared', () => {
  it('returns the same handle on a second acquire, and asks for one adapter', async () => {
    const gpu = stubGpu({ core: {} });
    const m = manager(gpu);
    const first = await mustAcquire(m);
    const second = await mustAcquire(m);
    expect(second).toBe(first);
    expect(first.references).toBe(2);
    expect(gpu.asked).toEqual(['core']);
    expect(gpu.adapters).toHaveLength(1);
  });

  it('creates one device for two callers that arrive in the same tick', async () => {
    const gpu = stubGpu({ core: {} });
    const m = manager(gpu);
    const [a, b] = await Promise.all([m.acquire(), m.acquire()]);
    expect(a).not.toBeNull();
    expect(b).toBe(a);
    expect(a?.references).toBe(2);
    expect(gpu.asked).toEqual(['core']);
  });

  it('exposes the in-flight request while it is outstanding, and clears it after', async () => {
    const m = manager(stubGpu({ core: {} }));
    const task = m.acquire();
    const inflight = m.pending;
    expect(inflight, 'a concurrent caller has to be able to see the request').not.toBeNull();
    const shared = await task;
    expect(await inflight).toBe(shared);
    expect(m.pending).toBeNull();
  });

  it('destroys the device only when the last reference goes', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({ core: { device } }));
    const shared = await mustAcquire(m);
    await m.acquire();
    expect(shared.release()).toBe(false);
    expect(shared.references).toBe(1);
    expect(device.destroyed).toBe(false);
    expect(m.current).toBe(shared);
    expect(shared.release()).toBe(true);
    expect(device.destroyed).toBe(true);
    expect(shared.destroyed).toBe(true);
    expect(m.current).toBeNull();
  });

  it('treats an over-release as the caller bug it is', async () => {
    const m = manager(stubGpu({ core: {} }));
    const shared = await mustAcquire(m);
    shared.release();
    expect(() => shared.release()).toThrow(/more often than retained/);
  });

  it('refuses to hand out a reference to a dead device', async () => {
    const m = manager(stubGpu({ core: {} }));
    const shared = await mustAcquire(m);
    shared.destroy();
    expect(() => shared.retain()).toThrow(/destroyed/);
    expect(shared.destroy(), 'destroy is idempotent').toBeUndefined();
  });

  it('acquires a fresh device after the previous one was destroyed', async () => {
    const gpu = stubGpu({ core: {} });
    const m = manager(gpu);
    const first = await mustAcquire(m);
    first.destroy();
    expect(m.current).toBeNull();
    const second = await mustAcquire(m);
    expect(second).not.toBe(first);
    expect(gpu.adapters).toHaveLength(2);
  });
});

describe('device loss', () => {
  it('marks the handle unusable, notifies listeners and drops the registration', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({ core: { device } }));
    const shared = await mustAcquire(m);
    const seen: string[] = [];
    shared.onLost((s) => seen.push(s.info.featureLevel));

    device.lose({ reason: 'destroyed', message: 'the driver crashed' });
    await Promise.resolve();
    await Promise.resolve();

    expect(shared.lost).toBe(true);
    expect(shared.usable).toBe(false);
    expect(shared.lossInfo).toEqual({ reason: 'destroyed', message: 'the driver crashed' });
    expect(seen).toEqual(['core']);
    expect(m.current).toBeNull();
    expect(m.failure).toMatchObject({ reason: 'lost', error: 'the driver crashed' });
    expect(device.destroyed, 'a lost device is not ours to destroy again').toBe(false);
  });

  it('recovers: the next acquire builds a new device', async () => {
    const gpu = stubGpu({ core: {} });
    const m = manager(gpu);
    const first = await mustAcquire(m);
    (first.device as StubDevice).lose();
    await Promise.resolve();
    await Promise.resolve();
    const second = await mustAcquire(m);
    expect(second).not.toBe(first);
    expect(second.usable).toBe(true);
    expect(m.failure).toBeNull();
  });

  it('fires a listener that subscribes after the loss, and stops one that unsubscribes', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({ core: { device } }));
    const shared = await mustAcquire(m);
    let late = 0;
    let cancelled = 0;
    const unsubscribe = shared.onLost(() => cancelled++);
    unsubscribe();
    device.lose();
    await Promise.resolve();
    await Promise.resolve();
    shared.onLost(() => late++);
    expect(late).toBe(1);
    expect(cancelled).toBe(0);
  });

  it('treats a rejecting lost promise as a loss, not as an unhandled rejection', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({ core: { device } }));
    const shared = await mustAcquire(m);
    let notified = 0;
    shared.onLost(() => notified++);
    device.failLost(new Error('device went away'));
    await Promise.resolve();
    await Promise.resolve();
    expect(shared.lost).toBe(true);
    expect(shared.lossInfo?.message).toBe('device went away');
    expect(notified).toBe(1);
  });

  it('does not report our own teardown as a loss', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({ core: { device } }));
    const shared = await mustAcquire(m);
    let notified = 0;
    shared.onLost(() => notified++);
    shared.destroy();
    device.lose({ reason: 'destroyed', message: 'destroy() was called' });
    await Promise.resolve();
    await Promise.resolve();
    expect(shared.lost).toBe(false);
    expect(shared.destroyed).toBe(true);
    expect(notified).toBe(0);
    expect(m.failure).toBeNull();
  });

  it('survives a destroy() that throws, because the device is already gone', async () => {
    const device = stubDevice();
    device.destroy = () => {
      throw new Error('already destroyed');
    };
    const m = manager(stubGpu({ core: { device } }));
    const shared = await mustAcquire(m);
    expect(() => shared.destroy()).not.toThrow();
    expect(m.current).toBeNull();
  });

  it('stops returning a lost handle from current, even before anyone looks', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({ core: { device } }));
    const shared = await mustAcquire(m);
    device.lose();
    await Promise.resolve();
    await Promise.resolve();
    expect(m.current).toBeNull();
    expect(shared.usable).toBe(false);
  });
});

describe('adoption', () => {
  it('registers a device from elsewhere and does not destroy it at zero refs', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({ core: {} }));
    const shared = m.adopt(device);
    expect(shared.info.adopted).toBe(true);
    expect(shared.info.featureLevel).toBe('core');
    expect(shared.info.adapterInfo).toEqual({ vendor: 'unknown', architecture: 'unknown' });
    expect(shared.info.limits.maxStorageBufferBindingSize).toBe(128 * MIB);
    expect(m.current).toBe(shared);
    expect(shared.release()).toBe(true);
    expect(device.destroyed).toBe(false);
    expect(m.current).toBeNull();
  });

  it('destroys an adopted device when asked to explicitly', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({}));
    m.adopt(device).destroy();
    expect(device.destroyed).toBe(true);
  });

  it('returns the same handle when the same object is adopted twice', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({}));
    const first = m.adopt(device);
    const second = m.adopt(device);
    expect(second).toBe(first);
    expect(first.references).toBe(2);
  });

  it('accepts the info the adopter knows and this manager cannot', async () => {
    const m = manager(stubGpu({}));
    const shared = m.adopt(stubDevice(), {
      featureLevel: 'compatibility',
      adapterInfo: { vendor: 'apple', architecture: 'metal-3' },
    });
    expect(shared.info.featureLevel).toBe('compatibility');
    expect(shared.info.adapterInfo.vendor).toBe('apple');
  });

  it('replaces a previous registration, leaving the old handle to its owner', async () => {
    const created = stubDevice();
    const adopted = stubDevice();
    const m = manager(stubGpu({ core: { device: created } }));
    const first = await mustAcquire(m);
    const second = m.adopt(adopted);
    expect(m.current).toBe(second);
    expect(first.usable, 'a displaced handle keeps working until released').toBe(true);
    expect(created.destroyed).toBe(false);
    first.release();
    expect(created.destroyed).toBe(true);
  });

  it('needs the enum constants, and says so', () => {
    const m = manager(stubGpu({}), null);
    expect(() => m.adopt(stubDevice())).toThrow(/enum constants are missing/);
  });
});

describe('releaseAll', () => {
  it('destroys a device this manager created', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({ core: { device } }));
    const shared = await mustAcquire(m);
    await m.acquire();
    await m.releaseAll();
    expect(device.destroyed).toBe(true);
    expect(shared.destroyed).toBe(true);
    expect(m.current).toBeNull();
  });

  it('leaves an adopted device alive', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({}));
    m.adopt(device);
    await m.releaseAll();
    expect(device.destroyed).toBe(false);
    expect(m.current).toBeNull();
  });

  it('waits for a request in flight, so nothing is left behind', async () => {
    const device = stubDevice();
    const m = manager(stubGpu({ core: { device } }));
    const task = m.acquire();
    const done = m.releaseAll();
    await Promise.all([task, done]);
    expect(device.destroyed).toBe(true);
    expect(m.current).toBeNull();
  });

  it('is a no-op when nothing was ever acquired', async () => {
    const m = manager(stubGpu({ core: {} }));
    await expect(m.releaseAll()).resolves.toBeUndefined();
    expect(m.current).toBeNull();
  });
});

describe('the process-wide manager', () => {
  it('is a singleton the convenience function delegates to', async () => {
    expect(sharedDevices).toBeInstanceOf(SharedDeviceManager);
    const shared = await acquireSharedDevice();
    // Node has no WebGPU, which is the point: this must resolve to null with a
    // reason rather than throw and take the caller's boot path with it.
    expect(shared).toBeNull();
    expect(sharedDevices.failure?.reason).toBe('absent');
  });
});

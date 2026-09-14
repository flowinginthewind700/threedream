/**
 * Capability probing has to be right in the cases where it is cheapest to be
 * wrong: no WebGPU at all, an adapter that exists but is too small, and
 * compatibility mode, which reports *lower* limits than the code was written
 * against. Those are exactly the cases a real-GPU test cannot produce on
 * demand, so they are stubbed here and the real adapter is checked in `e2e/`.
 */

import { describe, expect, it } from 'vitest';

import {
  LIMIT_FLOOR,
  LIMIT_NAMES,
  OPTIONAL_FEATURE_NAMES,
  clampWorkgroupSize,
  describeCapabilities,
  gpuFrom,
  maxElements,
  probeWebGpu,
  probeWebgl2,
  selectRenderTier,
  snapshotLimits,
  unmetLimitsOf,
  type CanvasLike,
  type GpuAdapterLike,
  type GpuLike,
  type WebGpuCapabilities,
} from '../src/gpu/capabilities.js';

const MIB = 1024 * 1024;

/** Limits as a real core-mode desktop adapter reports them. */
function coreLimits(overrides: Record<string, number> = {}): Record<string, number> {
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

interface StubAdapterOptions {
  limits?: Record<string, number>;
  features?: string[];
  /** Omit `keys` entirely to exercise the `has`-only fallback path. */
  enumerableFeatures?: boolean;
  isCompatibilityMode?: boolean;
  info?: { vendor?: string; architecture?: string };
}

function stubAdapter(options: StubAdapterOptions = {}): GpuAdapterLike {
  const features = options.features ?? [];
  const enumerable = options.enumerableFeatures ?? true;
  return {
    features: enumerable
      ? {
          has: (name: string) => features.includes(name),
          keys: () => features,
        }
      : { has: (name: string) => features.includes(name) },
    limits: options.limits ?? coreLimits(),
    info: options.info ?? { vendor: 'intel', architecture: 'gen-9' },
    ...(options.isCompatibilityMode === undefined
      ? {}
      : { isCompatibilityMode: options.isCompatibilityMode }),
  };
}

/** A `GPU` whose `requestAdapter` answers per feature level. */
function stubGpu(
  byLevel: Partial<Record<'core' | 'compatibility', GpuAdapterLike | null>>,
): GpuLike & { readonly asked: string[] } {
  const asked: string[] = [];
  return {
    asked,
    requestAdapter: async (options) => {
      const level = options?.featureLevel ?? 'core';
      asked.push(level);
      return byLevel[level] ?? null;
    },
  };
}

describe('gpuFrom', () => {
  it('finds nothing in Node, which is the environment most of the suite runs in', () => {
    expect(gpuFrom()).toBeUndefined();
    expect(gpuFrom({})).toBeUndefined();
    expect(gpuFrom({ navigator: {} })).toBeUndefined();
  });

  it('returns the injected navigator.gpu', async () => {
    const gpu = stubGpu({ core: stubAdapter() });
    expect(gpuFrom({ navigator: { gpu } })).toBe(gpu);
    expect((await probeWebGpu(gpuFrom({ navigator: { gpu } }))).available).toBe(true);
  });
});

describe('probeWebGpu', () => {
  it('reports "absent" rather than throwing when there is no WebGPU', async () => {
    const caps = await probeWebGpu(undefined);
    expect(caps.available).toBe(false);
    expect(caps.reason).toBe('absent');
    expect(caps.limits).toBeUndefined();
    expect(caps.features).toEqual([]);
  });

  it('reports "absent" for a gpu object with no requestAdapter', async () => {
    const caps = await probeWebGpu({} as unknown as GpuLike);
    expect(caps.reason).toBe('absent');
  });

  it('reports "no-adapter" when every level is refused', async () => {
    const gpu = stubGpu({});
    const caps = await probeWebGpu(gpu);
    expect(caps.available).toBe(false);
    expect(caps.reason).toBe('no-adapter');
    // Both levels were tried: a core-only probe would call Safari's WebGPU
    // unavailable, which is a false negative on a supported browser.
    expect(gpu.asked).toEqual(['core', 'compatibility']);
  });

  it('probes a core adapter and records its features, limits, and info', async () => {
    const adapter = stubAdapter({ features: ['subgroup', 'timestamp-query'] });
    const caps = await probeWebGpu(stubGpu({ core: adapter }));
    expect(caps.available).toBe(true);
    expect(caps.reason).toBeUndefined();
    expect(caps.featureLevel).toBe('core');
    expect(caps.compatibilityMode).toBe(false);
    expect(caps.features).toEqual(['subgroup', 'timestamp-query']);
    expect(caps.optionalFeatures).toEqual(['subgroup', 'timestamp-query']);
    expect(caps.info).toEqual({ vendor: 'intel', architecture: 'gen-9' });
    expect(caps.limits?.maxComputeInvocationsPerWorkgroup).toBe(256);
    expect(caps.unmetLimits).toEqual([]);
  });

  it('falls back to compatibility mode when core is refused', async () => {
    const gpu = stubGpu({
      compatibility: stubAdapter({
        limits: coreLimits({
          maxComputeInvocationsPerWorkgroup: 128,
          maxStorageBufferBindingSize: 128 * MIB,
          maxBufferSize: 128 * MIB,
        }),
        isCompatibilityMode: true,
      }),
    });
    const caps = await probeWebGpu(gpu);
    expect(gpu.asked).toEqual(['core', 'compatibility']);
    expect(caps.available).toBe(true);
    expect(caps.featureLevel).toBe('compatibility');
    expect(caps.compatibilityMode).toBe(true);
  });

  it('believes the adapter over the request when they disagree', async () => {
    // A core request that is answered with a compat-mode adapter is allowed; the
    // flag on the adapter is the only trustworthy answer.
    const caps = await probeWebGpu(
      stubGpu({ core: stubAdapter({ isCompatibilityMode: true }) }),
    );
    expect(caps.featureLevel).toBe('compatibility');
    expect(caps.compatibilityMode).toBe(true);
    // ... and vice versa.
    const other = await probeWebGpu(
      stubGpu({ compatibility: stubAdapter({ isCompatibilityMode: false }) }),
      { featureLevels: ['compatibility'] },
    );
    expect(other.featureLevel).toBe('core');
  });

  it('honours an explicit feature-level list and does not probe past it', async () => {
    const gpu = stubGpu({ core: stubAdapter(), compatibility: stubAdapter() });
    const caps = await probeWebGpu(gpu, { featureLevels: ['core'] });
    expect(gpu.asked).toEqual(['core']);
    expect(caps.available).toBe(true);
  });

  it('rejects an adapter whose storage binding is below the floor', async () => {
    const caps = await probeWebGpu(
      stubGpu({
        core: stubAdapter({ limits: coreLimits({ maxStorageBufferBindingSize: 4 * MIB }) }),
      }),
    );
    expect(caps.available).toBe(false);
    expect(caps.reason).toBe('unmet-limits');
    expect(caps.unmetLimits).toEqual(['maxStorageBufferBindingSize']);
    // The numbers are still reported: an unusable adapter is a diagnosis, not a
    // black box, and the HUD prints why the fallback happened.
    expect(caps.limits?.maxStorageBufferBindingSize).toBe(4 * MIB);
  });

  it('keeps probing after a rejected adapter but reports the first rejection', async () => {
    const gpu = stubGpu({
      core: stubAdapter({ limits: coreLimits({ maxComputeInvocationsPerWorkgroup: 32 }) }),
      compatibility: stubAdapter({ limits: coreLimits({ maxBufferSize: MIB }) }),
    });
    const caps = await probeWebGpu(gpu);
    expect(gpu.asked).toEqual(['core', 'compatibility']);
    expect(caps.reason).toBe('unmet-limits');
    expect(caps.unmetLimits).toEqual(['maxComputeInvocationsPerWorkgroup']);
  });

  it('stops at the first usable adapter', async () => {
    const gpu = stubGpu({ core: stubAdapter(), compatibility: stubAdapter() });
    await probeWebGpu(gpu);
    expect(gpu.asked).toEqual(['core']);
  });

  it('turns a rejecting driver into a reason instead of an exception', async () => {
    const gpu: GpuLike = {
      requestAdapter: async () => {
        throw new Error('adapter request denied by blocklist');
      },
    };
    const caps = await probeWebGpu(gpu);
    expect(caps.available).toBe(false);
    expect(caps.reason).toBe('error');
    expect(caps.error).toBe('adapter request denied by blocklist');
  });

  it('stringifies a non-Error rejection', async () => {
    const gpu: GpuLike = {
      requestAdapter: async () => {
        throw 'gpu process crashed';
      },
    };
    expect((await probeWebGpu(gpu)).error).toBe('gpu process crashed');
  });

  it('enumerates features by name when the adapter exposes no keys()', async () => {
    const adapter = stubAdapter({
      features: ['subgroup', 'float32-filterable', 'texture-compression-bc'],
      enumerableFeatures: false,
    });
    const caps = await probeWebGpu(stubGpu({ core: adapter }));
    // Only the known-optional ones are reported, sorted; an unknown feature is
    // still on the adapter, this probe just does not claim to list everything.
    expect(caps.features).toEqual(['float32-filterable', 'subgroup']);
    expect(caps.optionalFeatures).toEqual(['float32-filterable', 'subgroup']);
    expect(OPTIONAL_FEATURE_NAMES).toContain('timestamp-query');
  });

  it('defaults vendor and architecture when info is missing', async () => {
    const adapter = stubAdapter();
    const bare = { ...adapter, info: undefined } as unknown as GpuAdapterLike;
    const caps = await probeWebGpu(stubGpu({ core: bare }));
    expect(caps.info).toEqual({ vendor: 'unknown', architecture: 'unknown' });
  });
});

describe('limit snapshotting', () => {
  it('copies every limit the bridge reads', () => {
    const limits = snapshotLimits(coreLimits());
    expect(Object.keys(limits).sort()).toEqual([...LIMIT_NAMES].sort());
    expect(limits.maxComputeWorkgroupsPerDimension).toBe(65535);
  });

  it('treats a missing, NaN, or infinite limit as zero, i.e. below every floor', () => {
    const limits = snapshotLimits({
      maxStorageBufferBindingSize: Number.NaN,
      maxBufferSize: Number.POSITIVE_INFINITY,
      maxBindGroups: 4,
    });
    expect(limits.maxStorageBufferBindingSize).toBe(0);
    expect(limits.maxBufferSize).toBe(0);
    expect(limits.maxBindGroups).toBe(4);
    expect(limits.maxComputeInvocationsPerWorkgroup).toBe(0);
    expect(unmetLimitsOf(limits).sort()).toEqual(
      ['maxBufferSize', 'maxComputeInvocationsPerWorkgroup', 'maxStorageBufferBindingSize'].sort(),
    );
  });

  it('reports unmet limits in LIMIT_NAMES order, not object order', () => {
    const limits = snapshotLimits(coreLimits({ maxBufferSize: 1, maxStorageBufferBindingSize: 1 }));
    const unmet = unmetLimitsOf(limits);
    expect(unmet).toEqual(['maxStorageBufferBindingSize', 'maxBufferSize']);
    expect(LIMIT_NAMES.indexOf(unmet[0]!)).toBeLessThan(LIMIT_NAMES.indexOf(unmet[1]!));
  });

  it('meets its own floor with the measured compatibility-mode numbers', () => {
    // 128 invocations and 128MiB are what headless Chromium on this machine
    // reports in compatibility mode (docs/feasibility-rust-wasm-webgpu.md).
    const limits = snapshotLimits(
      coreLimits({
        maxComputeInvocationsPerWorkgroup: 128,
        maxStorageBufferBindingSize: 128 * MIB,
        maxBufferSize: 128 * MIB,
      }),
    );
    expect(unmetLimitsOf(limits)).toEqual([]);
    expect(LIMIT_FLOOR.maxComputeInvocationsPerWorkgroup).toBe(64);
  });
});

/** A usable adapter, for the derived-quantity tests. */
async function usable(overrides: StubAdapterOptions = {}): Promise<WebGpuCapabilities> {
  return probeWebGpu(stubGpu({ core: stubAdapter(overrides) }));
}

describe('clampWorkgroupSize', () => {
  it('clamps to the smaller of invocations-per-workgroup and sizeX', async () => {
    expect(clampWorkgroupSize(256, await usable())).toBe(256);
    expect(clampWorkgroupSize(1024, await usable())).toBe(256);
    const compat = await usable({
      limits: coreLimits({ maxComputeInvocationsPerWorkgroup: 128, maxComputeWorkgroupSizeX: 128 }),
      isCompatibilityMode: true,
    });
    expect(clampWorkgroupSize(256, compat)).toBe(128);
  });

  it('never returns zero for a usable adapter, and never a size at all for an unusable one', async () => {
    expect(clampWorkgroupSize(0, await usable())).toBe(1);
    expect(clampWorkgroupSize(64, await probeWebGpu(undefined))).toBe(0);
    const tiny = await probeWebGpu(
      stubGpu({ core: stubAdapter({ limits: coreLimits({ maxComputeInvocationsPerWorkgroup: 0 }) }) }),
    );
    expect(clampWorkgroupSize(64, tiny)).toBe(0);
  });
});

describe('maxElements', () => {
  it('divides the binding limit, which is the smaller constraint', async () => {
    const caps = await usable();
    // 12 doubles per body is the state stride in `rust/crates/physics-wasm`.
    expect(maxElements(caps, 96)).toBe(Math.floor((128 * MIB) / 96));
  });

  it('uses maxBufferSize when it is the smaller of the two', async () => {
    // 24MiB rather than smaller still: below `LIMIT_FLOOR.maxBufferSize` the
    // adapter is rejected outright and `maxElements` answers 0 for a different
    // reason than the one this test is about.
    const caps = await usable({ limits: coreLimits({ maxBufferSize: 24 * MIB }) });
    expect(maxElements(caps, 64)).toBe((24 * MIB) / 64);
  });

  it('is zero for an unusable adapter or a nonsensical element size', async () => {
    expect(maxElements(await probeWebGpu(undefined), 96)).toBe(0);
    expect(maxElements(await usable(), 0)).toBe(0);
    expect(maxElements(await usable(), -4)).toBe(0);
  });
});

describe('probeWebgl2', () => {
  const canvasWith = (gl: unknown): (() => CanvasLike | null) => () => ({
    getContext: () => gl,
  });

  it('is unavailable in Node, where there is no document', () => {
    expect(probeWebgl2()).toEqual({ available: false, renderer: '', vendor: '' });
  });

  it('is unavailable when the canvas or the context is missing', () => {
    expect(probeWebgl2(() => null)).toEqual({ available: false, renderer: '', vendor: '' });
    expect(probeWebgl2(canvasWith(null)).available).toBe(false);
  });

  it('reads the unmasked renderer when the debug extension is exposed', () => {
    const caps = probeWebgl2(
      canvasWith({
        getExtension: (name: string) =>
          name === 'WEBGL_debug_renderer_info'
            ? { UNMASKED_RENDERER_WEBGL: 37446, UNMASKED_VENDOR_WEBGL: 37445 }
            : null,
        getParameter: (pname: number) => (pname === 37446 ? 'ANGLE (SwiftShader)' : 'Google'),
      }),
    );
    expect(caps).toEqual({ available: true, renderer: 'ANGLE (SwiftShader)', vendor: 'Google' });
  });

  it('reports availability without a renderer when the extension is blocked', () => {
    const caps = probeWebgl2(
      canvasWith({ getExtension: () => null, getParameter: () => 'never called' }),
    );
    expect(caps).toEqual({ available: true, renderer: '', vendor: '' });
  });

  it('tolerates a null parameter value', () => {
    const caps = probeWebgl2(
      canvasWith({
        getExtension: () => ({ UNMASKED_RENDERER_WEBGL: 1, UNMASKED_VENDOR_WEBGL: 2 }),
        getParameter: () => null,
      }),
    );
    expect(caps).toEqual({ available: true, renderer: '', vendor: '' });
  });
});

describe('selectRenderTier', () => {
  it('chooses WebGPU and says which feature level it is', async () => {
    const core = selectRenderTier(await usable(), { available: false, renderer: '', vendor: '' });
    expect(core).toEqual({ tier: 'webgpu', reason: 'WebGPU adapter (core feature level)' });
    const compat = selectRenderTier(
      await usable({
        limits: coreLimits({ maxComputeInvocationsPerWorkgroup: 128 }),
        isCompatibilityMode: true,
      }),
      { available: false, renderer: '', vendor: '' },
    );
    expect(compat.reason).toContain('compatibility feature level');
  });

  it('falls back to WebGL2 with the reason WebGPU was refused', async () => {
    const decision = selectRenderTier(await probeWebGpu(undefined), {
      available: true,
      renderer: 'ANGLE (SwiftShader)',
      vendor: 'Google',
    });
    expect(decision.tier).toBe('webgl2');
    expect(decision.reason).toBe('WebGL2 fallback: WebGPU unavailable (absent)');
  });

  it('names the unmet limits when that is why WebGPU was refused', async () => {
    const caps = await probeWebGpu(
      stubGpu({ core: stubAdapter({ limits: coreLimits({ maxBufferSize: 1024 }) }) }),
    );
    const decision = selectRenderTier(caps, { available: true, renderer: '', vendor: '' });
    expect(decision.tier).toBe('webgl2');
    expect(decision.reason).toContain('limits below floor: maxBufferSize');
  });

  it('falls all the way back to CPU, which is the tier that always works', async () => {
    const decision = selectRenderTier(await probeWebGpu(stubGpu({})), {
      available: false,
      renderer: '',
      vendor: '',
    });
    expect(decision).toEqual({
      tier: 'cpu',
      reason: 'CPU fallback: no WebGPU (no-adapter) and no WebGL2',
    });
  });

  it('explains a missing reason rather than printing undefined', () => {
    const decision = selectRenderTier(
      {
        available: false,
        compatibilityMode: false,
        features: [],
        optionalFeatures: [],
        unmetLimits: [],
      },
      { available: false, renderer: '', vendor: '' },
    );
    expect(decision.reason).toContain('unknown');
  });
});

describe('describeCapabilities', () => {
  it('prints adapter, limits, and features for a usable adapter', async () => {
    const caps = await usable({ features: ['subgroup'] });
    const lines = describeCapabilities(caps);
    expect(lines[0]).toBe('webgpu: core adapter, intel gen-9');
    expect(lines[1]).toContain('invocations/workgroup=256');
    expect(lines[1]).toContain('workgroup=256x256x64');
    expect(lines[2]).toContain('storageBinding=128MiB');
    expect(lines[3]).toContain('bindGroups=4');
    expect(lines[4]).toBe('  features: subgroup');
  });

  it('says "(core only)" when the adapter has no optional features', async () => {
    const lines = describeCapabilities(await usable());
    expect(lines[4]).toBe('  features: (core only)');
  });

  it('prints the reason when WebGPU is unavailable, and the WebGL2 line when asked', async () => {
    expect(describeCapabilities(await probeWebGpu(undefined))).toEqual(['webgpu: unavailable (absent)']);
    const lines = describeCapabilities(await probeWebGpu(undefined), {
      available: true,
      renderer: '',
      vendor: '',
    });
    expect(lines[1]).toBe('webgl2: available (unknown renderer)');
    const none = describeCapabilities(await probeWebGpu(undefined), {
      available: false,
      renderer: '',
      vendor: '',
    });
    expect(none[1]).toBe('webgl2: unavailable');
  });

  it('prints the driver error message when the probe threw', async () => {
    const caps = await probeWebGpu({
      requestAdapter: async () => {
        throw new Error('lost device');
      },
    });
    expect(describeCapabilities(caps)).toEqual(['webgpu: unavailable (lost device)']);
  });

  it('prints the unmet limits when that is why the adapter was rejected', async () => {
    const caps = await probeWebGpu(
      stubGpu({ core: stubAdapter({ limits: coreLimits({ maxStorageBufferBindingSize: MIB }) }) }),
    );
    expect(describeCapabilities(caps)[0]).toBe(
      'webgpu: unavailable (limits below floor: maxStorageBufferBindingSize)',
    );
  });
});

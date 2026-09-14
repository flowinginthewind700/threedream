/**
 * Runtime capability probing for the GPU layer.
 *
 * Two rules from `docs/development-plan.md` shape this file:
 *
 * - The WebGL2 fallback has to be the *result of probing at runtime*, never a
 *   compile-time branch or a build flag. So nothing here reads `import.meta.env`,
 *   and `selectRenderTier` takes probe results as arguments.
 * - A GPU backend is not deterministic and must not claim to be. That is a
 *   property of the backend (`deterministic: false`), and it is why the tiers
 *   below are ordered by *capability* only: choosing WebGPU never changes a
 *   simulation result, because simulation stays on the CPU backends. What the
 *   tier selects is rendering and, later, presentation-side bulk compute.
 *
 * Everything is injectable and nothing touches `three` or `@webgpu/types`, so the
 * whole module runs in bare Node under vitest against a stub adapter. That is
 * deliberate: the decision logic is the part worth testing, and a test that needs
 * a real GPU is a test that only runs on some machines. The browser half of the
 * story is `e2e/`, which asserts the same functions against a real adapter.
 *
 * # Compatibility mode
 *
 * WebGPU ships in two feature levels. `core` is what desktop Chrome/Edge expose;
 * `compatibility` is what Safari and Chrome-on-Android expose, and it is not a
 * subset in the harmless sense -- it lowers real limits (measured in
 * `docs/feasibility-rust-wasm-webgpu.md`: `maxComputeInvocationsPerWorkgroup`
 * 256 -> 128) and forbids a few bindings. A probe that does not report which
 * level it got is a probe whose numbers cannot be trusted, so the level is part of
 * the result and the adapter's own `isCompatibilityMode` wins over what we asked
 * for.
 */

/** WebGPU feature level. `compatibility` is the constrained one. */
export type FeatureLevel = 'core' | 'compatibility';

/** Render tiers, best first. `cpu` always works, which is the point of having it. */
export type RenderTier = 'webgpu' | 'webgl2' | 'cpu';

/** The subset of `GPU` this module touches. */
export interface GpuLike {
  requestAdapter(
    options?: { featureLevel?: FeatureLevel },
  ): Promise<GpuAdapterLike | null | undefined>;
}

/**
 * The subset of `GPUAdapter`.
 *
 * `features` is a WebIDL `setlike`, so `has` is always there; `keys` is optional
 * in this type because enumerating is a convenience rather than a requirement, and
 * a stub should not have to implement it. Without it the probe falls back to
 * asking about the features it knows by name (`OPTIONAL_FEATURE_NAMES`).
 */
export interface GpuAdapterLike {
  readonly features: {
    has(feature: string): boolean;
    keys?(): Iterable<string>;
  };
  /** Core limits are all numbers on a real adapter; indexed so a stub is easy. */
  readonly limits: { readonly [name: string]: number | undefined };
  readonly info?: {
    readonly vendor?: string;
    readonly architecture?: string;
    readonly device?: string;
    readonly description?: string;
  };
  readonly isCompatibilityMode?: boolean;
}

/** The limits the compute bridge reads, snapshotted as plain numbers. */
export interface GpuLimits {
  readonly maxComputeInvocationsPerWorkgroup: number;
  readonly maxComputeWorkgroupSizeX: number;
  readonly maxComputeWorkgroupSizeY: number;
  readonly maxComputeWorkgroupSizeZ: number;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
  readonly maxBindGroups: number;
  readonly maxStorageBuffersPerShaderStage: number;
  readonly maxUniformBufferBindingSize: number;
}

export type LimitName = keyof GpuLimits;

export const LIMIT_NAMES: readonly LimitName[] = [
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxBindGroups',
  'maxStorageBuffersPerShaderStage',
  'maxUniformBufferBindingSize',
];

/** Limits that decide whether an adapter is usable at all. */
export type FlooredLimit =
  | 'maxComputeInvocationsPerWorkgroup'
  | 'maxStorageBufferBindingSize'
  | 'maxBufferSize';

/**
 * Floors, not aspirations: every kernel the bridge dispatches has to run on an
 * adapter that meets these.
 *
 * They are set from the *compatibility*-mode numbers measured in the feasibility
 * study, because that is the weakest adapter we intend to support.
 * `maxComputeInvocationsPerWorkgroup` is 64 rather than the 128 compatibility mode
 * allows so a workgroup size of 64 stays portable without a check at every
 * dispatch site.
 */
export const LIMIT_FLOOR: Readonly<Record<FlooredLimit, number>> = {
  maxComputeInvocationsPerWorkgroup: 64,
  // 16 MiB: 100k bodies at 12 f64 each is 9.6 MiB, and the bridge wants one
  // binding to be able to hold a whole world.
  maxStorageBufferBindingSize: 16 * 1024 * 1024,
  maxBufferSize: 16 * 1024 * 1024,
};

/**
 * Features that unlock a fast path when present. None is required: an adapter
 * with no optional features can still run every kernel, just without them.
 */
export const OPTIONAL_FEATURES = {
  subgroup: 'lane-level reductions instead of workgroup shared memory',
  'timestamp-query': 'GPU-side timing for the perf readout',
  'float32-filterable': 'sampling f32 state textures without a manual fetch',
} as const;

export type OptionalFeature = keyof typeof OPTIONAL_FEATURES;

export const OPTIONAL_FEATURE_NAMES: readonly string[] = Object.keys(OPTIONAL_FEATURES);

/** Why a probe came back unusable. Absent and refused are different problems. */
export type UnavailableReason =
  /** No `navigator.gpu` at all: Node, an old browser, or an insecure context. */
  | 'absent'
  /** `requestAdapter` returned null at every feature level asked for. */
  | 'no-adapter'
  /** An adapter exists but a limit is below `LIMIT_FLOOR`. */
  | 'unmet-limits'
  /** `requestAdapter` threw (blocklisted driver, denied, lost during the probe). */
  | 'error';

export interface WebGpuCapabilities {
  readonly available: boolean;
  readonly reason?: UnavailableReason;
  /** Set when `reason === 'error'`: the message, not the stack. */
  readonly error?: string;
  /** The level the adapter actually is, not the one that was asked for. */
  readonly featureLevel?: FeatureLevel;
  readonly compatibilityMode: boolean;
  /** Every feature the adapter reports, sorted so results are comparable. */
  readonly features: readonly string[];
  readonly optionalFeatures: readonly OptionalFeature[];
  /** Present whenever an adapter was obtained, including a rejected one. */
  readonly limits?: GpuLimits;
  readonly info?: {
    readonly vendor: string;
    readonly architecture: string;
  };
  /** Limit names below `LIMIT_FLOOR`, in `LIMIT_NAMES` order. */
  readonly unmetLimits: readonly LimitName[];
}

export interface Webgl2Capabilities {
  readonly available: boolean;
  readonly renderer: string;
  readonly vendor: string;
}

export interface RenderTierDecision {
  readonly tier: RenderTier;
  /** Human-readable, because this ends up in logs and in the demo's HUD. */
  readonly reason: string;
}

const NOTHING: WebGpuCapabilities = {
  available: false,
  compatibilityMode: false,
  features: [],
  optionalFeatures: [],
  unmetLimits: [],
};

/**
 * `navigator.gpu`, or `undefined` where there is no navigator and no WebGPU.
 *
 * The parameter is `unknown` because TypeScript's DOM lib still ships no WebGPU
 * types: a structural `{ navigator?: { gpu?: GpuLike } }` parameter would not
 * accept `globalThis`, whose `Navigator` declares no `gpu`. The cast inside is
 * the narrowest thing that works, and it keeps `src/` free of `@webgpu/types`.
 */
export function gpuFrom(scope: unknown = globalThis): GpuLike | undefined {
  const scopeWithNavigator = scope as { readonly navigator?: { readonly gpu?: GpuLike } };
  return scopeWithNavigator.navigator?.gpu;
}

/**
 * Copy the limits we care about into a total record.
 *
 * A limit the adapter does not report becomes `0`, which is below every floor.
 * That is the conservative reading: an adapter we cannot interrogate is not an
 * adapter we can dispatch 100k workgroups to.
 */
export function snapshotLimits(limits: GpuAdapterLike['limits']): GpuLimits {
  const out = {} as Record<LimitName, number>;
  for (const name of LIMIT_NAMES) {
    const value = limits[name];
    out[name] = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  }
  return out;
}

/** Limit names whose snapshotted value is below `LIMIT_FLOOR`. */
export function unmetLimitsOf(limits: GpuLimits): LimitName[] {
  const unmet: LimitName[] = [];
  for (const name of LIMIT_NAMES) {
    if (isFlooredLimit(name) && !(limits[name] >= LIMIT_FLOOR[name])) unmet.push(name);
  }
  return unmet;
}

function isFlooredLimit(name: LimitName): name is FlooredLimit {
  return name in LIMIT_FLOOR;
}

function featuresOf(adapter: GpuAdapterLike): string[] {
  const { features } = adapter;
  if (typeof features.keys === 'function') return [...features.keys()].sort();
  return OPTIONAL_FEATURE_NAMES.filter((name) => features.has(name)).sort();
}

function describeAdapter(adapter: GpuAdapterLike, askedFor: FeatureLevel): WebGpuCapabilities {
  const limits = snapshotLimits(adapter.limits);
  const unmet = unmetLimitsOf(limits);
  // Prefer the adapter's own flag: an implementation may hand back a
  // compat-mode adapter for a core request, and `askedFor` would then lie.
  const compatibilityMode = adapter.isCompatibilityMode ?? askedFor === 'compatibility';
  const features = featuresOf(adapter);
  // Derived from the already-sorted `features`, so the two lists always agree on
  // order and a diff between two adapters reads as a diff.
  const known = OPTIONAL_FEATURE_NAMES as readonly string[];
  return {
    available: unmet.length === 0,
    reason: unmet.length === 0 ? undefined : 'unmet-limits',
    featureLevel: compatibilityMode ? 'compatibility' : 'core',
    compatibilityMode,
    features,
    optionalFeatures: features.filter((name): name is OptionalFeature => known.includes(name)),
    limits,
    info: {
      vendor: adapter.info?.vendor ?? 'unknown',
      architecture: adapter.info?.architecture ?? 'unknown',
    },
    unmetLimits: unmet,
  };
}

export interface ProbeWebGpuOptions {
  /**
   * Feature levels to try, in order. Compatibility mode is in the default list
   * because Safari and Chrome-on-Android return `null` for a core request, where
   * reporting "no adapter" would be a false negative.
   */
  readonly featureLevels?: readonly FeatureLevel[];
}

/**
 * Probe WebGPU.
 *
 * Never throws: an adapter request can reject on a crashed or blocklisted driver,
 * and a probe that throws takes the caller's boot path with it. The failure is
 * reported through `reason`/`error`, which is also what makes the WebGL2 fallback
 * a decision rather than a try/catch at every call site.
 *
 * Only an adapter is requested, never a device: features, limits, and info all
 * live on the adapter, and creating a device costs a real GPU context this probe
 * has no business holding. Device lifetime belongs to the shared device manager.
 */
export async function probeWebGpu(
  gpu: GpuLike | undefined = gpuFrom(),
  options: ProbeWebGpuOptions = {},
): Promise<WebGpuCapabilities> {
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    return { ...NOTHING, reason: 'absent' };
  }
  const levels = options.featureLevels ?? ['core', 'compatibility'];
  let rejected: WebGpuCapabilities | undefined;
  for (const featureLevel of levels) {
    let adapter: GpuAdapterLike | null | undefined;
    try {
      adapter = await gpu.requestAdapter({ featureLevel });
    } catch (error) {
      return {
        ...NOTHING,
        reason: 'error',
        error: error instanceof Error ? error.message : String(error),
      };
    }
    if (!adapter) continue;
    const caps = describeAdapter(adapter, featureLevel);
    if (caps.available) return caps;
    // Keep probing while the adapter is unusable, but report the first
    // rejection: it came from the better feature level and its `unmetLimits`
    // are the more informative diagnosis.
    rejected ??= caps;
  }
  return rejected ?? { ...NOTHING, reason: 'no-adapter' };
}

/** Anything that can hand out a WebGL2 context. */
export interface CanvasLike {
  getContext(type: string, attributes?: unknown): unknown;
}

interface WebGL2Like {
  getExtension(
    name: string,
  ): { readonly UNMASKED_RENDERER_WEBGL?: number; readonly UNMASKED_VENDOR_WEBGL?: number } | null;
  getParameter(pname: number): string | null;
}

function defaultCanvasFactory(): CanvasLike | null {
  const scope = globalThis as {
    readonly document?: { createElement(tag: 'canvas'): CanvasLike };
  };
  return scope.document?.createElement('canvas') ?? null;
}

/**
 * Probe WebGL2.
 *
 * The context is asked for and dropped: holding one just to answer "can we?"
 * would consume a browser-level context slot, and browsers cap those per page
 * while the demo already owns one. The renderer string comes from
 * `WEBGL_debug_renderer_info` when it is exposed, because "is WebGL2 there" is
 * not the only question the fallback matrix asks -- SwiftShader and a real GPU
 * arrive at the same tier by very different routes, and the difference is worth
 * logging.
 */
export function probeWebgl2(
  makeCanvas: () => CanvasLike | null = defaultCanvasFactory,
): Webgl2Capabilities {
  const canvas = makeCanvas();
  if (!canvas) return { available: false, renderer: '', vendor: '' };
  const gl = canvas.getContext('webgl2') as WebGL2Like | null | undefined;
  if (!gl) return { available: false, renderer: '', vendor: '' };
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  const read = (pname: number | undefined): string =>
    pname === undefined ? '' : gl.getParameter(pname) ?? '';
  return {
    available: true,
    renderer: read(debug?.UNMASKED_RENDERER_WEBGL),
    vendor: read(debug?.UNMASKED_VENDOR_WEBGL),
  };
}

/**
 * Pick a render tier from probe results.
 *
 * The order is fixed and the reason comes back with the tier, because a silent
 * downgrade is the failure mode that makes GPU work look "flaky" rather than
 * broken: if a driver regresses, the app still runs -- on WebGL2 -- and nothing
 * says so.
 */
export function selectRenderTier(
  webgpu: WebGpuCapabilities,
  webgl2: Webgl2Capabilities,
): RenderTierDecision {
  if (webgpu.available) {
    return {
      tier: 'webgpu',
      reason: `WebGPU adapter (${webgpu.compatibilityMode ? 'compatibility' : 'core'} feature level)`,
    };
  }
  const why = describeAbsence(webgpu);
  if (webgl2.available) {
    return { tier: 'webgl2', reason: `WebGL2 fallback: WebGPU unavailable (${why})` };
  }
  return { tier: 'cpu', reason: `CPU fallback: no WebGPU (${why}) and no WebGL2` };
}

function describeAbsence(
  webgpu: Pick<WebGpuCapabilities, 'reason' | 'unmetLimits'>,
): string {
  if (webgpu.reason === 'unmet-limits') {
    return `limits below floor: ${webgpu.unmetLimits.join(', ')}`;
  }
  return webgpu.reason ?? 'unknown';
}

/**
 * Largest workgroup size that is safe on this adapter, or 0 when there is none.
 *
 * Dispatch sites need this because a size of 256 compiles on core and fails on
 * compatibility mode; clamping once here beats a surprise per kernel.
 */
export function clampWorkgroupSize(desired: number, caps: WebGpuCapabilities): number {
  const limits = caps.limits;
  if (!caps.available || !limits) return 0;
  const max = Math.min(
    limits.maxComputeInvocationsPerWorkgroup,
    limits.maxComputeWorkgroupSizeX,
  );
  if (!(max > 0)) return 0;
  return Math.max(1, Math.min(desired, max));
}

/**
 * How many `bytesPerElement` records fit in one storage buffer binding, or 0 when
 * the adapter cannot be used.
 *
 * This decides whether a world has to be split across bindings, so it is derived
 * from the *binding* limit rather than the buffer limit: the smaller of the two
 * is what actually constrains a single `var<storage>` array. A `maxBufferSize` of
 * 0 means the adapter did not report it, and is treated as "no extra constraint"
 * rather than as a zero-sized buffer, because `snapshotLimits` already put
 * `maxBufferSize` in `unmetLimits` if the floor was not met.
 */
export function maxElements(caps: WebGpuCapabilities, bytesPerElement: number): number {
  const limits = caps.limits;
  if (!caps.available || !limits || !(bytesPerElement > 0)) return 0;
  const binding = Math.min(
    limits.maxStorageBufferBindingSize,
    limits.maxBufferSize > 0 ? limits.maxBufferSize : limits.maxStorageBufferBindingSize,
  );
  return Math.floor(binding / bytesPerElement);
}

/**
 * Probe results as log lines.
 *
 * The demo HUD, the e2e specs, and `scripts/bench_shared_device.mjs` all print
 * the same facts, and three hand-rolled formatters would drift. Sizes are printed
 * in MiB because that is the unit people compare limits in.
 */
export function describeCapabilities(
  webgpu: WebGpuCapabilities,
  webgl2?: Webgl2Capabilities,
): string[] {
  const limits = webgpu.limits;
  if (!webgpu.available || !limits) {
    const detail = webgpu.reason === 'error' ? (webgpu.error ?? 'error') : describeAbsence(webgpu);
    const lines = [`webgpu: unavailable (${detail})`];
    if (webgl2) lines.push(describeWebgl2(webgl2));
    return lines;
  }
  const mib = (bytes: number): string => `${(bytes / (1024 * 1024)).toFixed(0)}MiB`;
  const lines = [
    `webgpu: ${webgpu.featureLevel} adapter, ${webgpu.info?.vendor ?? 'unknown'} ${webgpu.info?.architecture ?? 'unknown'}`,
    `  invocations/workgroup=${limits.maxComputeInvocationsPerWorkgroup} ` +
      `workgroup=${limits.maxComputeWorkgroupSizeX}x${limits.maxComputeWorkgroupSizeY}x${limits.maxComputeWorkgroupSizeZ} ` +
      `workgroups/dim=${limits.maxComputeWorkgroupsPerDimension}`,
    `  storageBinding=${mib(limits.maxStorageBufferBindingSize)} ` +
      `bufferSize=${mib(limits.maxBufferSize)} ` +
      `uniformBinding=${mib(limits.maxUniformBufferBindingSize)}`,
    `  bindGroups=${limits.maxBindGroups} storageBuffers/stage=${limits.maxStorageBuffersPerShaderStage}`,
    `  features: ${webgpu.features.length > 0 ? webgpu.features.join(', ') : '(core only)'}`,
  ];
  if (webgl2) lines.push(describeWebgl2(webgl2));
  return lines;
}

function describeWebgl2(webgl2: Webgl2Capabilities): string {
  return webgl2.available
    ? `webgl2: available (${webgl2.renderer || 'unknown renderer'})`
    : 'webgl2: unavailable';
}

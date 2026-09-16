/**
 * Browser proof of the shared-`GPUDevice` architecture.
 *
 * This is `scripts/bench_shared_device.mjs` turned into a page. The script
 * exists because the whole GPU roadmap rests on three facts that no unit test
 * can reach, and that a future three.js release could quietly break:
 *
 *   1. We can own the `GPUDevice` and inject it into `WebGPURenderer`, so
 *      simulation and rendering share one device and one queue.
 *   2. A TSL compute node writes a `StorageBufferAttribute`, and the `GPUBuffer`
 *      three.js created for it stays reachable from outside the renderer.
 *   3. A *raw* WGSL pipeline can bind that same three.js-managed buffer and be
 *      interleaved with `renderer.render()`. This is the bridge a native
 *      (Rust/wgpu) compute layer crosses with zero copies.
 *
 * Keeping it as a page rather than only a script matters: the script proves the
 * claims on a maintainer's machine, the page proves them on the reader's, and
 * `e2e/shared_device.spec.ts` proves them in CI against the built bundle. All
 * three read the same assertions, so they cannot drift.
 *
 * Capability probing goes through `src/gpu/capabilities.ts` rather than calling
 * `navigator.gpu` directly, because the plan requires the fallback to be a
 * runtime decision. When the probe says there is no usable adapter, this page
 * says so and stops checking -- the claim rows then read `not run` rather than
 * `FAIL`, because nothing failed, and the viewport presents whatever fallback
 * tier the probe landed on. It never pretends to have checked anything.
 */

import * as THREE from 'three/webgpu';
import { Fn, compute, instanceIndex, storage, uniform, vec4 } from 'three/tsl';
import { Cpu, Gauge, Play, ShieldCheck, Terminal, createElement, type IconNode } from 'lucide';

import {
  LIMIT_FLOOR,
  describeCapabilities,
  probeWebGpu,
  probeWebgl2,
  selectRenderTier,
} from '@threedream/gpu/capabilities.js';

// ---------------------------------------------------------------------------
// WebGPU surface used here
// ---------------------------------------------------------------------------

/**
 * The slice of WebGPU this page touches, declared locally.
 *
 * TypeScript's DOM lib still ships no WebGPU types, and `src/` stays free of
 * `@webgpu/types` on purpose so the probe logic runs in bare Node. Raw device
 * objects are only handled here, so the declarations live here too: narrower
 * than the real API, which is the point, since a page that only needs
 * `createComputePipeline` should not typecheck against all of it.
 */
interface BufferLike {
  readonly size: number;
  readonly usage: number;
  mapAsync(mode: number): Promise<void>;
}

interface ShaderModuleLike {
  getCompilationInfo(): Promise<{
    readonly messages: readonly { type: string; lineNum: number; message: string }[];
  }>;
}

interface ComputePassLike {
  setPipeline(pipeline: unknown): void;
  setBindGroup(index: number, group: unknown): void;
  dispatchWorkgroups(x: number): void;
  end(): void;
}

interface EncoderLike {
  beginComputePass(): ComputePassLike;
  finish(): unknown;
}

interface DeviceLike {
  readonly features: { has(name: string): boolean };
  readonly limits: { readonly [name: string]: number };
  readonly queue: {
    writeBuffer(buffer: BufferLike, offset: number, data: ArrayBufferView): void;
    submit(commandBuffers: readonly unknown[]): void;
    onSubmittedWorkDone(): Promise<void>;
  };
  createShaderModule(descriptor: { code: string }): ShaderModuleLike;
  createBuffer(descriptor: { size: number; usage: number }): BufferLike;
  createBindGroupLayout(descriptor: { entries: readonly unknown[] }): unknown;
  createBindGroup(descriptor: { layout: unknown; entries: readonly unknown[] }): unknown;
  createPipelineLayout(descriptor: { bindGroupLayouts: readonly unknown[] }): unknown;
  createComputePipeline(descriptor: {
    layout: unknown;
    compute: { module: ShaderModuleLike; entryPoint: string };
  }): unknown;
  createCommandEncoder(): EncoderLike;
  destroy(): void;
}

interface AdapterLike {
  readonly info?: { readonly vendor?: string; readonly architecture?: string };
  requestDevice(): Promise<DeviceLike>;
}

interface GpuNamespace {
  requestAdapter(options?: { featureLevel?: string }): Promise<AdapterLike | null>;
}

/** Usage/stage bit constants, read from the globals rather than hardcoded. */
interface WebGpuConstants {
  readonly GPUBufferUsage?: {
    readonly STORAGE: number;
    readonly VERTEX: number;
    readonly COPY_SRC: number;
    readonly UNIFORM: number;
    readonly COPY_DST: number;
  };
  readonly GPUShaderStage?: { readonly COMPUTE: number };
}

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

export interface SharedDeviceReport {
  /**
   * `unavailable` is its own status rather than a flavour of `error`: no adapter
   * means the three claims were never evaluated, which is a different statement
   * from having evaluated them and lost. Readers and the e2e suite tell them
   * apart on this field.
   */
  status: 'running' | 'done' | 'error' | 'unavailable';
  error?: string;
  threeRevision: string;
  tier: string;
  tierReason: string;
  webgpuAvailable: boolean;
  featureLevel?: string;
  limits: {
    invocationsPerWorkgroup: number;
    storageBufferBindingSize: number;
    maxBufferSize: number;
  };
  /** Every limit below `LIMIT_FLOOR`, as reported by the shared probe. */
  unmetLimits: readonly string[];
  claims: {
    sameDeviceObject: boolean;
    threeBufferReachable: boolean;
    rawPipelineBound: boolean;
    interleaved: boolean;
  };
  /**
   * Which claims were actually evaluated. A claim reads `false` both when it
   * failed and when the page never got far enough to run it, so the report keeps
   * the two apart here and the UI can say `not run` instead of painting rose.
   */
  claimsRun: {
    sameDeviceObject: boolean;
    threeBufferReachable: boolean;
    rawPipelineBound: boolean;
    interleaved: boolean;
  };
  points: number;
  sentinelPoints: number;
  frames: number;
  msPerFrame: number;
  canvasBytes: number;
  backendIsWebGpu: boolean;
  lines: readonly string[];
}

declare global {
  interface Window {
    __sharedDevice?: SharedDeviceReport;
  }
}

/**
 * Scene size and the sentinel the raw kernel writes.
 *
 * `WORKGROUP` is the floor from `LIMIT_FLOOR`, not a tuned value: 64 invocations
 * is what compatibility-mode adapters guarantee, so a dispatch sized to it needs
 * no per-adapter check. `POINTS` is a multiple of it, which the dispatch count
 * relies on -- a remainder would leave a tail of unwritten points that the
 * sentinel count would then report as a raw-kernel failure.
 */
const POINTS = 8192;
const SENTINEL = 7.5;
const FRAMES = 20;
const WORKGROUP = LIMIT_FLOOR.maxComputeInvocationsPerWorkgroup;

const report: SharedDeviceReport = {
  status: 'running',
  threeRevision: THREE.REVISION,
  tier: 'cpu',
  tierReason: 'not probed yet',
  webgpuAvailable: false,
  limits: { invocationsPerWorkgroup: 0, storageBufferBindingSize: 0, maxBufferSize: 0 },
  unmetLimits: [],
  claims: {
    sameDeviceObject: false,
    threeBufferReachable: false,
    rawPipelineBound: false,
    interleaved: false,
  },
  claimsRun: {
    sameDeviceObject: false,
    threeBufferReachable: false,
    rawPipelineBound: false,
    interleaved: false,
  },
  points: POINTS,
  sentinelPoints: 0,
  frames: 0,
  msPerFrame: 0,
  canvasBytes: 0,
  backendIsWebGpu: false,
  lines: [],
};
window.__sharedDevice = report;

// ---------------------------------------------------------------------------
// dom
// ---------------------------------------------------------------------------

const ICONS = {
  cpu: Cpu,
  gauge: Gauge,
  play: Play,
  'shield-check': ShieldCheck,
  terminal: Terminal,
} as const;

type IconName = keyof typeof ICONS;

function paintIcon(host: Element, name: IconName): void {
  host.replaceChildren(createElement(ICONS[name] as IconNode));
}

for (const el of Array.from(document.querySelectorAll<HTMLElement>('[data-icon]'))) {
  const name = el.dataset['icon'] as IconName | undefined;
  if (name && name in ICONS) paintIcon(el, name);
}
const brandLogo = document.getElementById('brand-logo');
if (brandLogo) paintIcon(brandLogo, 'cpu');

function byId(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}

const viewport = byId('viewport');
const loading = byId('loading');
const logList = byId('log');
const verdict = byId('verdict');
const tierBadge = byId('tier-badge');
const frameBadge = byId('frame-badge');
const runBtn = byId('run-btn') as HTMLButtonElement;

const lines: string[] = [];

function log(text: string, kind: 'plain' | 'head' | 'good' | 'warn' = 'plain'): void {
  lines.push(text);
  const li = document.createElement('li');
  li.textContent = text;
  if (kind !== 'plain') li.className = `is-${kind}`;
  logList.appendChild(li);
  logList.scrollTop = logList.scrollHeight;
}

/** A claim row: `ok` paints teal, `false` paints rose, `null` stays neutral. */
function showClaim(id: string, ok: boolean | null, text?: string): void {
  const el = byId(id);
  el.textContent = text ?? (ok === null ? '\u2014' : ok ? 'OK' : 'FAIL');
  el.classList.toggle('is-good', ok === true);
  el.classList.toggle('is-warn', ok === false);
}

/** The label for a claim the page never got far enough to evaluate. */
const NOT_RUN = 'not run';

/** A claim that was not evaluated reads `not run` and stays neutral, not rose. */
function showRanClaim(id: string, ran: boolean, ok: boolean): void {
  showClaim(id, ran ? ok : null, ran ? undefined : NOT_RUN);
}

function showStat(id: string, text: string, good = false): void {
  const el = byId(id);
  el.textContent = text;
  el.classList.toggle('is-good', good);
}

function paintReport(): void {
  const c = report.claims;
  const ran = report.claimsRun;
  showRanClaim('cl-device', ran.sameDeviceObject, c.sameDeviceObject);
  showRanClaim('cl-buffer', ran.threeBufferReachable, c.threeBufferReachable);
  showRanClaim('cl-pipeline', ran.rawPipelineBound, c.rawPipelineBound);
  showRanClaim('cl-interleaved', ran.interleaved, c.interleaved);
  // The sentinel count and the canvas bytes are both consequences of the
  // interleaved frames, so they stay neutral until those frames really ran.
  const sentinelOk =
    report.sentinelPoints === POINTS ? true : report.sentinelPoints > 0 ? false : null;
  showClaim(
    'cl-sentinel',
    ran.interleaved ? sentinelOk : null,
    ran.interleaved ? `${report.sentinelPoints} / ${POINTS}` : NOT_RUN,
  );
  showClaim(
    'cl-pixels',
    ran.interleaved ? (report.canvasBytes > 0 ? true : null) : null,
    ran.interleaved ? `${report.canvasBytes} B` : NOT_RUN,
  );
  showStat('ad-tier', report.tier, report.tier === 'webgpu');
  showStat('ad-level', report.featureLevel ?? '\u2014');
  const l = report.limits;
  showStat(
    'ad-invocations',
    l.invocationsPerWorkgroup > 0 ? String(l.invocationsPerWorkgroup) : '\u2014',
    l.invocationsPerWorkgroup >= LIMIT_FLOOR.maxComputeInvocationsPerWorkgroup,
  );
  showStat('ad-storage', mib(l.storageBufferBindingSize));
  showStat('ad-buffer', mib(l.maxBufferSize));
  showStat('ad-revision', report.threeRevision);
  showStat('ad-ms', report.msPerFrame > 0 ? report.msPerFrame.toFixed(2) : '\u2014');
  report.lines = [...lines];
}

function mib(bytes: number): string {
  return bytes > 0 ? `${(bytes / (1024 * 1024)).toFixed(0)} MiB` : '\u2014';
}

// ---------------------------------------------------------------------------
// the run
// ---------------------------------------------------------------------------

interface RendererInternals {
  readonly device: unknown;
  readonly isWebGPUBackend?: boolean;
  get(attribute: unknown): { readonly buffer?: BufferLike };
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/**
 * Raw WGSL that writes the same curve the TSL kernel writes, plus a sentinel in
 * `w`.
 *
 * The sentinel is the actual evidence: three.js created this buffer, a TSL node
 * filled it, and afterwards every point carries a value only this shader
 * writes. Counting them proves the raw pipeline was bound to three.js's buffer
 * rather than to a look-alike of our own.
 */
function rawKernelSource(): string {
  return `
@group(0) @binding(0) var<storage, read_write> data: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: vec4<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u32(params.x)) { return; }
  let t = params.y;
  let f = f32(i) / params.x;
  data[i] = vec4<f32>(
    cos(f * 6.2831853 * 3.0 + t) * 1.6,
    sin(f * 6.2831853 * 2.0 + t * 1.3) * 1.1,
    cos(f * 6.2831853 * 5.0 - t * 0.7) * 1.6,
    ${SENTINEL});
}`;
}

let idleFrame = 0;

function stopIdle(): void {
  if (idleFrame) cancelAnimationFrame(idleFrame);
  idleFrame = 0;
}

/**
 * The device, renderer and canvas the last successful run left behind.
 *
 * Kept alive on purpose: after the check finishes the page keeps presenting the
 * same frames through the same device, which is a stronger statement than a
 * screenshot -- it shows the device did not go invalid the moment the assertions
 * passed. Torn down at the start of the next run and on any error, so a page
 * left open holds exactly one GPU context rather than one per click.
 */
interface ActiveRun {
  renderer: THREE.WebGPURenderer;
  /**
   * Absent on the fallback tier, where three.js owns the context and there is no
   * `GPUDevice` of ours left to destroy.
   */
  device?: DeviceLike;
  canvas: HTMLCanvasElement;
}

let active: ActiveRun | null = null;

function teardown(): void {
  stopIdle();
  active?.renderer.dispose();
  active?.device?.destroy();
  active?.canvas.remove();
  active = null;
}

/** Keep presenting after the check finishes, so the page is not a still image. */
function startIdle(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  box: THREE.Mesh,
  countFrames = false,
): void {
  let frame = 0;
  const loop = (): void => {
    idleFrame = requestAnimationFrame(loop);
    frame += 1;
    box.rotation.y = frame * 0.008;
    renderer.render(scene, camera);
    if (countFrames) {
      // The fallback path never ran the measured 20 frames, so the badge counts
      // the frames it is presenting instead of sitting on a stale zero.
      report.frames = frame;
      frameBadge.textContent = `${frame} frames`;
    }
  };
  idleFrame = requestAnimationFrame(loop);
}

/** A canvas stretched to the viewport by CSS, which is what both paths want. */
function makeCanvas(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  viewport.appendChild(canvas);
  return canvas;
}

/** The box the check presents, so the fallback frame shows the same subject. */
function boxScene(width: number, height: number): {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  box: THREE.Mesh;
} {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1013);
  const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshNormalMaterial());
  scene.add(box);
  const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);
  camera.position.set(0, 0, 3.2);
  return { scene, camera, box };
}

/**
 * Present the tier the probe landed on instead of leaving an empty black box.
 *
 * The `forceWebGL` renderer is the house fallback pattern (`demo/particles.ts`,
 * `demo/soft.ts` both reach for it on the tiers with no device). None of this is
 * evidence for the three claims, and the page says so -- but a viewport that
 * renders is the honest picture of "this browser granted no WebGPU adapter; here
 * is what it does have".
 */
async function presentFallback(): Promise<void> {
  const canvas = makeCanvas();
  // Without `forceWebGL` three.js would request an adapter of its own here, which
  // is the exact request the probe just answered "no" to.
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, forceWebGL: true });
  const width = Math.max(1, viewport.clientWidth);
  const height = Math.max(1, viewport.clientHeight);
  renderer.setSize(width, height, false);
  await renderer.init();
  active = { renderer, canvas };
  loading.classList.add('is-hidden');
  const { scene, camera, box } = boxScene(width, height);
  startIdle(renderer, scene, camera, box, true);
}

async function run(): Promise<void> {
  teardown();
  logList.replaceChildren();
  lines.length = 0;
  report.status = 'running';
  report.error = undefined;
  report.sentinelPoints = 0;
  report.canvasBytes = 0;
  report.msPerFrame = 0;
  report.frames = 0;
  report.claims = {
    sameDeviceObject: false,
    threeBufferReachable: false,
    rawPipelineBound: false,
    interleaved: false,
  };
  report.claimsRun = {
    sameDeviceObject: false,
    threeBufferReachable: false,
    rawPipelineBound: false,
    interleaved: false,
  };
  loading.textContent = 'Probing adapters\u2026';
  loading.classList.remove('is-hidden');
  frameBadge.textContent = '0 frames';
  paintReport();

  let device: DeviceLike | undefined;
  let renderer: THREE.WebGPURenderer | undefined;
  try {
    // --- capability probe: the same code path the fallback matrix uses -------
    const webgpu = await probeWebGpu();
    const webgl2 = probeWebgl2();
    const decision = selectRenderTier(webgpu, webgl2);
    report.tier = decision.tier;
    report.tierReason = decision.reason;
    report.webgpuAvailable = webgpu.available;
    report.featureLevel = webgpu.featureLevel;
    report.unmetLimits = webgpu.unmetLimits;
    if (webgpu.limits) {
      report.limits = {
        invocationsPerWorkgroup: webgpu.limits.maxComputeInvocationsPerWorkgroup,
        storageBufferBindingSize: webgpu.limits.maxStorageBufferBindingSize,
        maxBufferSize: webgpu.limits.maxBufferSize,
      };
    }
    for (const line of describeCapabilities(webgpu, webgl2)) log(line);
    log(`tier: ${decision.tier} -- ${decision.reason}`, decision.tier === 'webgpu' ? 'good' : 'warn');
    paintReport();

    if (!webgpu.available) {
      // No adapter means none of the three claims can be evaluated, which is
      // not the same as having evaluated them and lost. Say that plainly, leave
      // the claim rows neutral, and still show the reader the tier their
      // browser landed on rather than an empty black viewport.
      report.status = 'unavailable';
      report.error = `no usable WebGPU adapter (${decision.reason})`;
      log(`not checked: ${decision.reason}`, 'warn');
      log(`presenting the ${decision.tier} fallback tier`);
      verdict.textContent =
        `Not checked -- ${decision.reason}. The three claims need one GPUDevice shared ` +
        `with three.js, so none of them ran; the viewport presents the ${decision.tier} ` +
        `fallback tier instead.`;
      verdict.classList.add('is-warn');
      tierBadge.textContent = decision.tier;
      paintReport();
      await presentFallback();
      return;
    }
    tierBadge.textContent = `${decision.tier} / ${webgpu.featureLevel ?? '?'}`;

    const constants = globalThis as WebGpuConstants;
    const usage = constants.GPUBufferUsage;
    const stage = constants.GPUShaderStage;
    if (!usage || !stage) throw new Error('WebGPU enums are missing although an adapter was found');

    // --- CLAIM 1: we create the device, three.js accepts it ------------------
    const gpu = (navigator as { gpu?: GpuNamespace }).gpu;
    if (!gpu) throw new Error('navigator.gpu disappeared between probe and request');
    const adapter = await gpu.requestAdapter({ featureLevel: webgpu.featureLevel });
    if (!adapter) throw new Error(`requestAdapter returned null at ${webgpu.featureLevel}`);
    device = await adapter.requestDevice();
    log(
      `adapter: ${adapter.info?.vendor ?? 'unknown'} ${adapter.info?.architecture ?? ''}`.trimEnd(),
      'head',
    );
    log(`three REVISION: ${THREE.REVISION}`);
    log(
      `device: subgroup=${device.features.has('subgroup')} ` +
        `timestampQuery=${device.features.has('timestamp-query')} ` +
        `maxWorkgroupInvocations=${device.limits['maxComputeInvocationsPerWorkgroup']} ` +
        `maxStorageBinding=${mib(device.limits['maxStorageBufferBindingSize'] ?? 0)}`,
    );

    const canvas = makeCanvas();
    renderer = new THREE.WebGPURenderer({ canvas, antialias: false, device });
    // Sized to the viewport rather than a fixed 640x360: `updateStyle` is false
    // because the canvas is stretched by CSS, and a drawing buffer whose aspect
    // disagrees with its box would make the rendered frame look like a bug.
    const width = Math.max(1, viewport.clientWidth);
    const height = Math.max(1, viewport.clientHeight);
    renderer.setSize(width, height, false);
    await renderer.init();
    active = { renderer, device, canvas };
    loading.classList.add('is-hidden');

    const backend = renderer.backend as unknown as RendererInternals;
    report.backendIsWebGpu = backend.isWebGPUBackend === true;
    report.claims.sameDeviceObject = backend.device === device;
    report.claimsRun.sameDeviceObject = true;
    log(`backend is WebGPU: ${String(report.backendIsWebGpu)}`);
    log(`CLAIM1 same device object: ${String(report.claims.sameDeviceObject)}`,
      report.claims.sameDeviceObject ? 'good' : 'warn');
    paintReport();

    // --- CLAIM 2: TSL compute writes it, the GPUBuffer stays reachable -------
    const attr = new THREE.StorageBufferAttribute(POINTS, 4);
    const gpuAttr = storage(attr, 'vec4', POINTS);
    const time = uniform(0);
    // `Fn(...)` builds the node; invoking it produces the callable the compute
    // node wants. Forgetting the second call is the classic TSL footgun.
    const kernel = Fn(() => {
      const i = instanceIndex;
      const f = i.toFloat().div(POINTS);
      const x = f.mul(6.2831853).mul(3).add(time).cos().mul(1.6);
      const y = f.mul(6.2831853).mul(2).add(time.mul(1.3)).sin().mul(1.1);
      const z = f.mul(6.2831853).mul(5).sub(time.mul(0.7)).cos().mul(1.6);
      gpuAttr.element(i).assign(vec4(x, y, z, 1));
    })();
    await renderer.computeAsync(compute(kernel, POINTS, [WORKGROUP]));
    log('CLAIM2 TSL computeAsync: OK', 'good');

    const threeBuffer = backend.get(attr).buffer;
    report.claims.threeBufferReachable =
      !!threeBuffer && typeof threeBuffer.mapAsync === 'function';
    report.claimsRun.threeBufferReachable = true;
    log(
      `CLAIM2 three-managed GPUBuffer reachable: ${String(report.claims.threeBufferReachable)}`,
      report.claims.threeBufferReachable ? 'good' : 'warn',
    );
    if (!threeBuffer) throw new Error('three.js exposed no GPUBuffer for the storage attribute');
    log(
      `CLAIM2 usage STORAGE=${(threeBuffer.usage & usage.STORAGE) !== 0} ` +
        `VERTEX=${(threeBuffer.usage & usage.VERTEX) !== 0} ` +
        `COPY_SRC=${(threeBuffer.usage & usage.COPY_SRC) !== 0}`,
    );
    const first = new Float32Array(await renderer.getArrayBufferAsync(attr));
    log(`CLAIM2 readback p[0] = [${fmt3(first)}]`);
    paintReport();

    // --- CLAIM 3: a raw WGSL pipeline binds three.js's own buffer ------------
    const module = device.createShaderModule({ code: rawKernelSource() });
    for (const message of (await module.getCompilationInfo()).messages) {
      if (message.type === 'error') log(`WGSL error line ${message.lineNum}: ${message.message}`, 'warn');
    }
    const params = device.createBuffer({
      size: 16,
      usage: usage.UNIFORM | usage.COPY_DST,
    });
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: stage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: stage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const bindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: threeBuffer } },
        { binding: 1, resource: { buffer: params } },
      ],
    });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'main' },
    });
    report.claims.rawPipelineBound = true;
    report.claimsRun.rawPipelineBound = true;
    log("CLAIM3 raw pipeline bound to THREE's buffer: OK", 'good');

    const { scene, camera, box } = boxScene(width, height);

    // One interleaved pair per animation frame rather than a tight loop: the
    // frames actually present, so what is on screen is what was measured.
    const started = performance.now();
    for (let frame = 0; frame < FRAMES; frame++) {
      await nextFrame();
      device.queue.writeBuffer(params, 0, new Float32Array([POINTS, frame * 0.05, 0, 0]));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(POINTS / WORKGROUP);
      pass.end();
      device.queue.submit([encoder.finish()]);
      box.rotation.y = frame * 0.18;
      box.rotation.x = frame * 0.07;
      renderer.render(scene, camera);
      report.frames = frame + 1;
      frameBadge.textContent = `${frame + 1} frames`;
    }
    const elapsed = performance.now() - started;
    await device.queue.onSubmittedWorkDone();
    report.msPerFrame = elapsed / FRAMES;
    report.claims.interleaved = true;
    report.claimsRun.interleaved = true;
    log(
      `CLAIM3 ${FRAMES}x raw-compute + three.render interleaved: OK ` +
        `(${report.msPerFrame.toFixed(2)} ms/frame, compute+render+present)`,
      'good',
    );

    const readback = new Float32Array(await renderer.getArrayBufferAsync(attr));
    let sentinel = 0;
    for (let i = 0; i < POINTS; i++) {
      if (Math.abs((readback[i * 4 + 3] ?? 0) - SENTINEL) < 1e-3) sentinel += 1;
    }
    report.sentinelPoints = sentinel;
    log(
      `CLAIM3 points carrying the raw-kernel sentinel (w=${SENTINEL}): ${sentinel}/${POINTS}`,
      sentinel === POINTS ? 'good' : 'warn',
    );
    log(`CLAIM3 sample = [${fmt3(readback)}]`);

    report.canvasBytes = await pngBytes(canvas);
    log(`canvas PNG bytes (render produced pixels): ${report.canvasBytes}`);
    paintReport();

    const failed = Object.entries(report.claims).filter(([, ok]) => !ok).map(([name]) => name);
    if (sentinel !== POINTS) failed.push('sentinel');
    report.status = failed.length === 0 ? 'done' : 'error';
    if (failed.length > 0) report.error = `claims failed: ${failed.join(', ')}`;
    verdict.textContent =
      failed.length === 0
        ? `All three claims hold on this ${report.featureLevel ?? '?'} adapter.`
        : `Failed: ${failed.join(', ')}`;
    verdict.classList.toggle('is-warn', failed.length > 0);

    startIdle(renderer, scene, camera, box);
  } catch (error) {
    report.status = 'error';
    report.error = error instanceof Error ? error.message : String(error);
    log(`error: ${report.error}`, 'warn');
    verdict.textContent = `Check failed: ${report.error}`;
    verdict.classList.add('is-warn');
    loading.textContent = report.error;
    loading.classList.remove('is-hidden');
    tierBadge.textContent = report.tier;
    teardown();
  } finally {
    paintReport();
    if (report.status === 'done') {
      frameBadge.textContent = `${report.frames} frames`;
    }
  }
}

function fmt3(values: Float32Array): string {
  return [values[0] ?? 0, values[1] ?? 0, values[2] ?? 0]
    .map((v) => v.toFixed(4))
    .join(', ');
}

async function pngBytes(canvas: HTMLCanvasElement): Promise<number> {
  const blob = await new Promise<Blob | null>((resolve) => {
    void canvas.toBlob(resolve, 'image/png');
  });
  return blob?.size ?? 0;
}

let running: Promise<void> | null = null;

runBtn.addEventListener('click', () => {
  if (running) return;
  running = run().finally(() => {
    running = null;
  });
});

running = run().finally(() => {
  running = null;
});

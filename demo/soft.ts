/**
 * Browser demo for the M4 GPU soft-body layer.
 *
 * The particle page proves a device can be shared and a buffer can be filled on
 * it. This page has to prove three more things, and each one has a row in the
 * sidebar because a claim nobody can see is a claim nobody can check:
 *
 *   1. The constraint graph is *coloured*, and the solve runs one colour at a
 *      time. That is what makes a GPU solve race-free without being
 *      bit-reproducible, and the "color batches" and "race-free solve" rows are
 *      the two halves of the claim. The colours are the same on every tier,
 *      because both tiers compute the plan with the same two passes -- so a
 *      number in the Graph panel that changes with the tier is a bug, not a
 *      difference of opinion.
 *   2. Nodes are grouped into islands, and each workgroup covers 64 consecutive
 *      nodes of exactly one island. `sheets` is in the scene picker for this
 *      reason: a single connected cloth is one island, which is the case that
 *      proves nothing about a grouping pass.
 *   3. Positions reach the screen without crossing the bus. The publish buffer is
 *      copied into the `GPUBuffer` three.js already allocated for the position
 *      attribute, and the surface and its wire overlay share that one attribute,
 *      so a single blit fills both. "blitted / frame" is that copy's byte count.
 *
 * And the M4 acceptance criterion about WebGL2 is checked the same way M3's was:
 * pick a worse tier on a good machine and watch the counters keep moving. Here
 * the downgrade means a real change of solver -- the WebGL2 and CPU tiers run the
 * deterministic reference -- so the page is also the place where "the reference is
 * a specification, not a fallback" is visible: same scene, same seed, same graph,
 * different tier.
 *
 * Everything here is the real code path. `createSoftSystem` does the probing,
 * `SoftRunner` does the fixed-step decoupling, and `SoftView` does the drawing.
 * Nothing in this file special-cases the browser.
 *
 * # Two ways to run
 *
 * Without parameters the page is a live demo on a rAF loop. With `?steps=N` it
 * runs exactly `N` simulation steps, timed, then stops and publishes a final
 * report -- that is the mode `e2e/soft_gpu.spec.ts` and
 * `scripts/bench_gpu_soft.mjs` drive, because a benchmark that measures a
 * display's refresh rate is measuring the wrong thing. Every parameter is also a
 * way to reproduce a reported bug without touching the UI.
 */

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  Cpu,
  Gauge,
  Grid3x3,
  Layers,
  Moon,
  Network,
  Pause,
  Play,
  RotateCcw,
  ShieldCheck,
  Spline,
  Terminal,
  Waypoints,
  Zap,
  createElement,
  type IconNode,
} from 'lucide';

import type { RenderTier } from '@threedream/gpu/capabilities.js';
import { SharedDeviceManager, type SharedDevice } from '@threedream/gpu/device.js';
import {
  SoftRunner,
  createSoftSystem,
  type SoftSystemHandle,
} from '@threedream/gpu/soft.js';
import { SOFT_WORKGROUP_SIZE } from '@threedream/gpu/softIslands.js';
import { SOFT_SCENES, SoftMesh, type SoftScene } from '@threedream/gpu/softMesh.js';
import type { SoftSimOptions, SoftStepStats } from '@threedream/gpu/softTypes.js';
import type { RendererLike } from '@threedream/render/particles.js';
import { SoftView, type SoftDrawMode } from '@threedream/render/soft.js';

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

/** What the tier bar offers. `'auto'` means "probe and take the best". */
export type TierChoice = RenderTier | 'auto';

/** The Graph panel, minus the two per-island/per-color arrays it has no row for. */
export interface SoftGraphReport {
  nodes: number;
  constraints: number;
  islands: number;
  colors: number;
  iterations: number;
  nodeWorkgroups: number;
  dispatchesPerStep: number;
}

/**
 * The published state of the page.
 *
 * Exported for the reason `ParticlesReport` is: the e2e specs import the type
 * rather than re-declaring it, so a field renamed here breaks the spec at compile
 * time instead of arriving as `undefined` at assert time. `stats` is typed as the
 * engine's own `SoftStepStats` for the same reason twice over -- the HUD and the
 * specs are reading one struct, not two that happen to agree today.
 */
export interface SoftReport {
  /** `booting` until the first build settles; the specs wait on this. */
  status: 'booting' | 'live' | 'done' | 'error';
  error?: string;
  threeRevision: string;
  requested: TierChoice;
  strict: boolean;
  /** The tier that is actually running, after any downgrade. */
  tier: RenderTier;
  reason: string;
  featureLevel?: string;
  /**
   * Whether the page could get a WebGPU adapter, independent of which tier it
   * ended up on. The e2e specs skip on this rather than probing `navigator.gpu`
   * themselves: a freshly launched Playwright page is `about:blank`, whose opaque
   * origin is not a secure context, so `navigator.gpu` is missing there and a
   * probe answers "no" on every machine.
   */
  webgpuAvailable: boolean;
  /** Which three.js backend drew the frame. */
  backend: 'webgpu' | 'webgl2' | 'none';
  /** `renderer.backend.device === shared.device`: the M2 claim at this layer. */
  deviceShared: boolean;
  /**
   * The GPU system's own `raceFree`, and `false` on every other tier.
   *
   * Read off the backend rather than asserted from the tier name, because the
   * flag is the one place the engine states the property this milestone exists
   * for. A page that hardcoded `true` here would keep saying so the day a kernel
   * stopped being colour-batched.
   */
  raceFree: boolean;
  viewMode: 'gpu' | 'cpu';
  frameMode: 'gpu-blit' | 'cpu-upload';
  blitBytes: number;
  gpuError: string | null;
  seed: number;
  scene: SoftScene;
  count: number;
  iterations: number;
  stiffness: number;
  sleep: boolean;
  wire: boolean;
  drawMode: SoftDrawMode;
  fixedDt: number;
  frames: number;
  steps: number;
  stepsPerFrame: number;
  msPerFrame: number;
  /**
   * Per-step cost, and the distribution behind it.
   *
   * In a scripted run `msPerStep` is the *median* timed chunk; in the live loop
   * it stays the EMA it always was, since that loop has no chunk boundaries. The
   * median replaced a mean for a measured reason: a rung of 30 steps produces
   * four chunks, and a mean over four samples is a number one contended chunk
   * can triple. The same rung on the same machine read 2.9 ms/step and 10.0
   * ms/step minutes apart, and the run that produced the 10.0 also reported
   * 20k nodes *faster* than 10k, which is not a solver property. `msPerStepP95`
   * keeps the spread visible rather than averaging it away, `msPerStepMean` is
   * the statistic it replaced, and `stepSamples` says how much either rests on.
   */
  msPerStep: number;
  msPerStepMean: number;
  msPerStepP95: number;
  stepSamples: number;
  fps: number;
  behind: boolean;
  drawCalls: number;
  triangles: number;
  plan: SoftGraphReport;
  stats: SoftStepStats;
  digest: string;
  canvasBytes: number;
  lines: readonly string[];
}

declare global {
  interface Window {
    __soft?: SoftReport;
  }
}

const EMPTY_STATS: SoftStepStats = {
  escaped: 0,
  maxSpeed: 0,
  maxConstraintError: 0,
  awakeIslands: 0,
  sleepingIslands: 0,
  kineticEnergy: 0,
};

const EMPTY_PLAN: SoftGraphReport = {
  nodes: 0,
  constraints: 0,
  islands: 0,
  colors: 0,
  iterations: 0,
  nodeWorkgroups: 0,
  dispatchesPerStep: 0,
};

const report: SoftReport = {
  status: 'booting',
  threeRevision: THREE.REVISION,
  requested: 'auto',
  strict: false,
  tier: 'cpu',
  reason: 'not built yet',
  webgpuAvailable: false,
  backend: 'none',
  deviceShared: false,
  raceFree: false,
  viewMode: 'cpu',
  frameMode: 'cpu-upload',
  blitBytes: 0,
  gpuError: null,
  seed: 0,
  scene: 'cloth',
  count: 0,
  iterations: 8,
  stiffness: 1,
  sleep: false,
  wire: false,
  drawMode: 'surface',
  fixedDt: 1 / 60,
  frames: 0,
  steps: 0,
  stepsPerFrame: 0,
  msPerFrame: 0,
  msPerStep: 0,
  msPerStepMean: 0,
  msPerStepP95: 0,
  stepSamples: 0,
  fps: 0,
  behind: false,
  drawCalls: 0,
  triangles: 0,
  plan: { ...EMPTY_PLAN },
  stats: { ...EMPTY_STATS },
  digest: '',
  canvasBytes: 0,
  lines: [],
};
window.__soft = report;

// ---------------------------------------------------------------------------
// dom
// ---------------------------------------------------------------------------

const ICONS = {
  cpu: Cpu,
  gauge: Gauge,
  'grid-3x3': Grid3x3,
  layers: Layers,
  moon: Moon,
  network: Network,
  pause: Pause,
  play: Play,
  'rotate-ccw': RotateCcw,
  'shield-check': ShieldCheck,
  spline: Spline,
  terminal: Terminal,
  waypoints: Waypoints,
  zap: Zap,
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
if (brandLogo) paintIcon(brandLogo, 'spline');

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
const sceneBadge = byId('scene-badge');
const countBadge = byId('count-badge');
const pathBadge = byId('path-badge');
const fpsBadge = byId('fps-badge');
const runBtn = byId('run-btn') as HTMLButtonElement;
const runIcon = runBtn.querySelector<HTMLElement>('[data-icon]')!;
const runLabel = runBtn.querySelector<HTMLElement>('.btn-label')!;
const resetBtn = byId('reset-btn') as HTMLButtonElement;
const wireBtn = byId('wire-btn') as HTMLButtonElement;
const sleepBtn = byId('sleep-btn') as HTMLButtonElement;
const iterationsSlider = byId('iterations-slider') as HTMLInputElement;
const iterationsValue = byId('iterations-value') as HTMLOutputElement;
const stiffnessSlider = byId('stiffness-slider') as HTMLInputElement;
const stiffnessValue = byId('stiffness-value') as HTMLOutputElement;
const tierSelect = byId('tier-select');
const sceneSelect = byId('scene-select');
const countSelect = byId('count-select');

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

function showStat(id: string, text: string, good = false, warn = false): void {
  const el = byId(id);
  el.textContent = text;
  el.classList.toggle('is-good', good);
  el.classList.toggle('is-warn', warn);
}

// ---------------------------------------------------------------------------
// settings
// ---------------------------------------------------------------------------

/**
 * The scenes the picker offers, read off the markup and checked against the
 * engine's own list.
 *
 * A picker that offers a scene the builder does not know produces a URL parameter
 * that silently builds a cloth, and the report then says `cloth` while the address
 * bar says `rope`. Refusing the mismatch at load is louder than either.
 */
const SCENE_CHOICES: SoftScene[] = Array.from(
  sceneSelect.querySelectorAll<HTMLElement>('[data-scene]'),
  (el) => el.dataset['scene'] as SoftScene,
).filter((scene) => (SOFT_SCENES as readonly string[]).includes(scene));

const MIN_COUNT = 2;
const MAX_COUNT = 200000;
const DEFAULT_SEED = 4242;
/**
 * Initial node speed, in units/second.
 *
 * Not zero, and not a knob: a cloth released from a perfect lattice with no
 * velocity hangs straight down and stays there, which hides every asymmetry a
 * solver bug would show up as. A little motion makes the drape visible and costs
 * nothing. It is a constant rather than a slider because it is a property of the
 * picture, not of the physics under test.
 */
const INITIAL_SPEED = 2;

interface Settings {
  scene: SoftScene;
  count: number;
  seed: number;
  tier: TierChoice;
  strict: boolean;
  iterations: number;
  stiffness: number;
  sleep: boolean;
  wire: boolean;
  /** Above zero: run exactly this many steps, timed, then report. Benchmark mode. */
  steps: number;
}

/**
 * Whichever `.seg` carries `is-active` IS the default.
 *
 * Restating it here would give the page two answers to one question, and only one
 * of them is visible to the person reading the markup.
 */
function initialSelected(
  host: HTMLElement,
  key: 'tier' | 'count' | 'scene',
  fallback: string,
): string {
  return host.querySelector<HTMLElement>('.seg.is-active')?.dataset[key] ?? fallback;
}

/**
 * Every knob, read once.
 *
 * A URL is how a bug report arrives. `?scene=sheets&count=20000&iterations=8&steps=600`
 * has to be reproducible by pasting it, not by clicking four controls in the order
 * somebody happened to click them.
 */
function readSettings(): Settings {
  const params = new URLSearchParams(location.search);

  const int = (name: string, fallback: number, min: number, max: number): number => {
    const raw = params.get(name);
    if (raw === null) return fallback;
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isInteger(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };
  const num = (name: string, fallback: number, min: number, max: number): number => {
    const raw = params.get(name);
    if (raw === null) return fallback;
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };
  const flag = (name: string, fallback: boolean): boolean => {
    const raw = params.get(name);
    if (raw === null) return fallback;
    return raw !== '0' && raw.toLowerCase() !== 'false';
  };
  const tierOf = (value: string, fallback: TierChoice): TierChoice =>
    value === 'auto' || value === 'webgpu' || value === 'webgl2' || value === 'cpu'
      ? value
      : fallback;
  const sceneOf = (value: string, fallback: SoftScene): SoftScene =>
    (SOFT_SCENES as readonly string[]).includes(value) ? (value as SoftScene) : fallback;

  const markupCount = Number(initialSelected(countSelect, 'count', '10000'));
  const markupScene = initialSelected(sceneSelect, 'scene', 'cloth');
  const sliderIterations = Number(iterationsSlider.value);
  const sliderStiffness = Number(stiffnessSlider.value);
  return {
    scene: sceneOf(
      params.get('scene')?.toLowerCase() ?? '',
      SCENE_CHOICES.includes(markupScene as SoftScene) ? (markupScene as SoftScene) : 'cloth',
    ),
    count: int(
      'count',
      Number.isInteger(markupCount) && markupCount > 0 ? markupCount : 10000,
      MIN_COUNT,
      MAX_COUNT,
    ),
    seed: int('seed', DEFAULT_SEED, 0, 2147483647),
    tier: tierOf(
      params.get('tier')?.toLowerCase() ?? '',
      tierOf(initialSelected(tierSelect, 'tier', 'auto'), 'auto'),
    ),
    strict: flag('strict', false),
    iterations: int(
      'iterations',
      Number.isInteger(sliderIterations) && sliderIterations > 0 ? sliderIterations : 8,
      1,
      16,
    ),
    stiffness: num(
      'stiffness',
      Number.isFinite(sliderStiffness) ? sliderStiffness : 1,
      0,
      1,
    ),
    sleep: flag('sleep', sleepBtn.classList.contains('is-on')),
    wire: flag('wire', wireBtn.classList.contains('is-on')),
    steps: int('steps', 0, 0, 1000000),
  };
}

const settings = readSettings();

// ---------------------------------------------------------------------------
// live state
// ---------------------------------------------------------------------------

/**
 * The two renderer fields three.js does not type but this page needs.
 *
 * `device` is how the M2 claim gets checked here: a pointer comparison between the
 * backend's device and the `SharedDevice` the solver kernels were built on. And
 * `isWebGPUBackend` is the honest tier test, because `three/webgpu` handed
 * `forceWebGL` runs a WebGL backend under a WebGPU-named renderer -- reading the
 * import would report the wrong thing.
 */
interface RendererInternals {
  readonly device: unknown;
  readonly isWebGPUBackend?: boolean;
}

interface Live {
  readonly handle: SoftSystemHandle;
  readonly runner: SoftRunner;
  readonly view: SoftView;
  readonly renderer: THREE.WebGPURenderer;
  readonly controls: OrbitControls;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly canvas: HTMLCanvasElement;
  readonly shared: SharedDevice | null;
  /** Benchmark mode: a fixed step count, then a report instead of a frame loop. */
  readonly scripted: boolean;
}

let live: Live | null = null;
/** One manager for the life of the page, so rebuilds share a device and its history. */
let manager: SharedDeviceManager | null = null;
let loopFrame = 0;
let paused = false;
let lastTime = 0;
/** Exponential moving average: a per-frame number nobody can read is a number nobody reads. */
let frameMs = 0;
let fpsWindowStart = 0;
let fpsWindowFrames = 0;
let counterDue = 0;
let refreshInFlight = false;
let refreshDue = false;
let canvasSampled = false;
let observer: ResizeObserver | null = null;

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function stopLoop(): void {
  if (loopFrame) cancelAnimationFrame(loopFrame);
  loopFrame = 0;
}

function kib(bytes: number): string {
  return `${(bytes / 1024).toFixed(1)} KiB`;
}

function compact(n: number): string {
  return n.toLocaleString('en-US');
}

/**
 * Run `fn` and swallow what it throws.
 *
 * Teardown reports, it does not fail: every call here is a best-effort release of
 * something the next boot replaces, and one that threw halfway would leave the
 * page holding a canvas and a device it could no longer name.
 */
function safe(fn: () => void): void {
  try {
    fn();
  } catch {
    // Deliberately empty; see above.
  }
}

/**
 * Take the page apart, in the order that does not leak.
 *
 * The view first, because it holds geometries and materials against the device.
 * Then the system, which releases its own reference to that device -- and with it
 * every solver buffer, which is most of the memory on this page. Then the controls
 * and the renderer, and only then our reference: three.js does not destroy a device
 * it was handed, so the final `release()` is what actually frees the GPU, and doing
 * it earlier would drop the device out from under a copy still in flight.
 */
function teardown(): void {
  stopLoop();
  const current = live;
  live = null;
  if (!current) return;
  observer?.disconnect();
  observer = null;
  safe(() => current.view.dispose());
  safe(() => current.handle.dispose());
  safe(() => current.controls.dispose());
  // `dispose()` is async on the WebGPU renderer; nothing here can wait for it, and
  // a rejection from a page that is already rebuilding would surface as noise.
  safe(() => void current.renderer.dispose().catch(() => undefined));
  safe(() => current.shared?.release());
  current.canvas.remove();
}

// ---------------------------------------------------------------------------
// the frame loop
// ---------------------------------------------------------------------------

/**
 * Pull the GPU's own numbers into the report.
 *
 * Two awaits, and both matter. `readStats()` reads the counters buffer the kernels
 * write, which is the only source of `escaped`, `maxConstraintError` and the
 * island counts on this tier -- without it `system.stats()` returns whatever the
 * last read left behind. `readback()` copies node positions back, and it is what
 * makes `maxSpeed`, `kineticEnergy` and the digest mean anything here.
 *
 * Rate limited rather than per-frame, because a map/unmap pair is a stall and a HUD
 * that costs frames ends up measuring itself. That trade is worth spelling out for
 * this page in particular: `readback()` is `count * 32` bytes, so at 20k nodes it
 * is 640 KiB per read, and reading it every frame would be a bigger transfer than
 * the blit the milestone exists to avoid.
 */
async function refreshCounters(): Promise<void> {
  const current = live;
  if (!current) return;
  if (refreshInFlight) {
    // Coalesce. A HUD is a view of the latest state, not a queue of old ones.
    refreshDue = true;
    return;
  }
  refreshInFlight = true;
  try {
    const gpu = current.handle.gpu;
    if (gpu) {
      await gpu.readStats();
      await gpu.readback();
    }
    // The page can be rebuilt while those two promises are in the air, and
    // painting a disposed system's numbers over the new one's would be a lie.
    if (live !== current) return;
    report.stats = { ...current.handle.system.stats() };
    report.digest = current.handle.system.digest();
    paintCounters();
  } catch (error) {
    log(`counter read failed: ${messageOf(error)}`, 'warn');
  } finally {
    refreshInFlight = false;
    if (refreshDue) {
      refreshDue = false;
      void refreshCounters();
    }
  }
}

/**
 * Encode the canvas once, to show the frame is pixels and not a cleared buffer.
 *
 * A presented frame cannot be read back with `readPixels`, so the only honest
 * sample is an encode; `toBlob` is the browser's own path, and its size is
 * non-zero exactly when something was drawn. Zero on failure rather than a throw,
 * because a sampling problem is not a page problem.
 */
async function sampleCanvas(canvas: HTMLCanvasElement): Promise<void> {
  try {
    const blob = await new Promise<Blob | null>((resolve) => {
      void canvas.toBlob(resolve, 'image/png');
    });
    report.canvasBytes = blob?.size ?? 0;
  } catch {
    report.canvasBytes = 0;
  }
}

/**
 * Drive the simulation from the wall clock and draw it.
 *
 * `runner.frame(dt)` decides how many fixed steps this frame was worth, so a
 * 240 Hz panel and a 30 Hz one simulate the same trajectory at different
 * smoothness instead of at different speeds -- which matters more here than it did
 * for particles, because a solver's `maxConstraintError` depends on the step it
 * was given and not on how often it was displayed. The view's `update()` sits
 * between the step and the draw, because filling the position attribute is a
 * rendering concern and nothing in it writes the mesh.
 */
function startLoop(): void {
  stopLoop();
  const started = performance.now();
  lastTime = started;
  fpsWindowStart = started;
  fpsWindowFrames = 0;
  counterDue = started;
  // Closure rather than module state: a fresh loop is a fresh throttle, and boot()
  // already has enough flags to reset.
  let paintDueAt = 0;

  const tick = (now: number): void => {
    const current = live;
    if (!current) return;
    loopFrame = requestAnimationFrame(tick);

    // Clamped: a backgrounded tab hands back a multi-second dt, and stepping 60
    // fixed slices to catch up is how a demo turns into a hang on focus. For a
    // solver it is worse than a hang -- a cloth stretched that far in one frame
    // can come back with edges longer than the box.
    const dt = Math.min(0.25, Math.max(0, (now - lastTime) / 1000));
    lastTime = now;
    const frameStart = performance.now();

    if (!paused) {
      report.stepsPerFrame = current.runner.frame(dt);
      current.view.update();
      report.steps = current.runner.steps;
      report.frames = current.runner.frames;
    }
    current.controls.update();
    // Owned here rather than left to three's internal animation loop, which only
    // resets `info` on its own rAF and is not what drives this page.
    current.renderer.info.reset();
    current.renderer.render(current.scene, current.camera);

    const elapsed = performance.now() - frameStart;
    frameMs = frameMs === 0 ? elapsed : frameMs * 0.9 + elapsed * 0.1;
    report.msPerFrame = frameMs;
    report.msPerStep = report.steps > 0 ? (frameMs * report.frames) / report.steps : 0;
    report.behind = current.runner.behind;
    report.viewMode = current.view.viewMode;
    report.frameMode = current.view.frameMode;
    report.blitBytes = current.view.blittedBytes;
    report.gpuError = current.view.gpuError;
    report.drawCalls = current.renderer.info.render.drawCalls;
    report.triangles = current.renderer.info.render.triangles;

    fpsWindowFrames++;
    if (now - fpsWindowStart >= 500) {
      report.fps = (fpsWindowFrames * 1000) / (now - fpsWindowStart);
      fpsWindowStart = now;
      fpsWindowFrames = 0;
    }

    if (!paused) {
      // The CPU tier's stats are free and exact, so it does not have to wait for a
      // map/unmap pair; painting them every throttle window keeps the HUD live on
      // the tier where the numbers cost nothing.
      if (now - counterDue >= 500) {
        counterDue = now;
        if (current.handle.gpu) {
          void refreshCounters();
        } else {
          report.stats = { ...current.handle.system.stats() };
          report.digest = current.handle.system.digest();
        }
      }
      // Paint more often than we read. The runner's numbers are free, and a HUD
      // that only moves twice a second looks broken even when it is not.
      if (now - paintDueAt >= 120) {
        paintDueAt = now;
        // The tier panel as well as the counters. It is the half that carries the
        // M4 claim, and on a live run the vertex path is decided on the *second*
        // frame -- the first one legitimately finds no buffer, because three.js
        // creates it during the first render. Painting the tier once at boot would
        // leave the page asserting `cpu-upload` next to a mesh that is blitting.
        paintTier();
        paintCounters();
      }
    }

    // Late enough that the blit path has had a frame to attach, early enough that
    // a page which never draws still gets sampled.
    if (!canvasSampled && report.frames >= 8) {
      canvasSampled = true;
      void sampleCanvas(current.canvas);
    }
  };

  loopFrame = requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// benchmark mode
// ---------------------------------------------------------------------------

/**
 * The `p`th percentile of `samples`, nearest rank, and 0 when there are none.
 *
 * Nearest rank rather than interpolated: every number a benchmark prints should
 * be one it actually measured, and with the handful of chunks a rung produces an
 * interpolated midpoint is a cost no step ever had.
 */
function percentile(samples: readonly number[], p: number): number {
  if (samples.length === 0) return 0;
  const sorted = [...samples].sort((a, b) => a - b);
  const rank = Math.min(sorted.length, Math.max(1, Math.ceil((p / 100) * sorted.length)));
  return sorted[rank - 1];
}

/**
 * Run exactly `target` steps, timed, then stop and report.
 *
 * This is what `e2e/soft_gpu.spec.ts` and `scripts/bench_gpu_soft.mjs` drive, and
 * it exists because a frame loop measures a display. Work is submitted in chunks,
 * each one rendered and then awaited through `queue.onSubmittedWorkDone()`, so the
 * timing covers the GPU rather than the rAF cadence; the yields between chunks sit
 * outside the timed region.
 *
 * Chunked rather than one `advance(target)` for two reasons. A single call would
 * queue `target * dispatchesPerStep` dispatches with nothing presented in between,
 * which is a fair stress test and a meaningless benchmark. And `onSubmittedWorkDone`
 * on an empty queue returns immediately, so without a chunk boundary the number
 * reported would be how long it took to *record* commands -- the one measurement
 * that makes a GPU look as fast as a CPU.
 *
 * A chunk is also the unit of timing, which makes the sample count a function of
 * `target`: 30 steps is four samples, and a mean over four samples is a number one
 * contended chunk can triple. So the headline is the median chunk with the spread
 * printed next to it, and the bench ladder asks for enough steps for both to mean
 * something.
 */
async function runScripted(target: number): Promise<void> {
  const current = live;
  if (!current) return;
  const CHUNK = 8;
  const wallStart = performance.now();
  let chunks = 0;
  let workMs = 0;
  // One per-step cost per chunk, so the headline number can be a median. See
  // `msPerStep` on SoftReport for the measurement that made this necessary.
  const stepSamples: number[] = [];

  while (current.runner.steps < target && live === current) {
    const n = Math.min(CHUNK, target - current.runner.steps);
    const t0 = performance.now();
    current.runner.advance(n);
    current.view.update();
    current.renderer.info.reset();
    current.renderer.render(current.scene, current.camera);
    if (current.shared) await current.shared.device.queue.onSubmittedWorkDone();
    const chunkMs = performance.now() - t0;
    workMs += chunkMs;
    stepSamples.push(chunkMs / n);
    chunks++;
    // Yield so a long run stays interruptible and the canvas is actually
    // presented; excluded from the timing above on purpose.
    if (chunks % 4 === 0) await nextFrame();
  }
  // Rebuilt mid-run: the new boot owns the report now.
  if (live !== current) return;

  const wallMs = performance.now() - wallStart;
  const done = current.runner.steps;
  // One more `update()` after the last step. There is no pipeline to wait for --
  // unlike the particle view, this one blits the first frame it finds a
  // destination for -- but the last frame inside the loop ran before the final
  // step's positions were published, and `frameMode` should describe the path the
  // page is on rather than the path one stale frame happened to take.
  report.frameMode = current.view.update();
  report.viewMode = current.view.viewMode;
  report.blitBytes = current.view.blittedBytes;
  report.gpuError = current.view.gpuError;
  report.steps = done;
  report.frames = Math.max(1, chunks);
  report.stepsPerFrame = done / report.frames;
  report.msPerFrame = workMs / report.frames;
  report.msPerStep = percentile(stepSamples, 50);
  report.msPerStepMean = done > 0 ? workMs / done : 0;
  report.msPerStepP95 = percentile(stepSamples, 95);
  report.stepSamples = stepSamples.length;
  report.fps = wallMs > 0 ? (report.frames * 1000) / wallMs : 0;

  // Last, so the report describes the final state rather than the state the first
  // chunk happened to leave behind.
  const gpu = current.handle.gpu;
  if (gpu) {
    await gpu.readStats();
    await gpu.readback();
  }
  report.stats = { ...current.handle.system.stats() };
  report.digest = current.handle.system.digest();

  // The presented frame, and the renderer's own counters for it. `info` is reset
  // once per frame, and this loop is the frame, so the counters describe what is
  // on the canvas rather than whatever the last chunk happened to leave behind.
  current.renderer.info.reset();
  current.renderer.render(current.scene, current.camera);
  report.drawCalls = current.renderer.info.render.drawCalls;
  report.triangles = current.renderer.info.render.triangles;
  await sampleCanvas(current.canvas);
  report.status = 'done';

  log(
    `${compact(report.steps)} steps | ${wallMs.toFixed(0)} ms wall | ${chunks} submits | ` +
      `${report.msPerStep.toFixed(3)} ms/step p50 | p95 ${report.msPerStepP95.toFixed(3)} | ` +
      `mean ${report.msPerStepMean.toFixed(3)} | ${report.msPerFrame.toFixed(2)} ms/submit`,
    'head',
  );
  log(
    `vertex path ${report.frameMode}` +
      `${report.blitBytes > 0 ? ` (${kib(report.blitBytes)} per frame)` : ''} | ` +
      `draw calls ${report.drawCalls} | triangles ${compact(report.triangles)}`,
    report.frameMode === 'gpu-blit' ? 'good' : 'plain',
  );
  log(
    `escaped ${report.stats.escaped} | max stretch ${(report.stats.maxConstraintError * 100).toFixed(1)}% | ` +
      `max speed ${report.stats.maxSpeed.toFixed(3)} | ` +
      `islands ${report.stats.awakeIslands} awake / ${report.stats.sleepingIslands} asleep | ` +
      `kinetic energy ${compact(Math.round(report.stats.kineticEnergy))}`,
    report.stats.escaped === 0 ? 'good' : 'warn',
  );
  log(`digest ${report.digest || '(none)'}`);
  log(`canvas PNG bytes ${report.canvasBytes}`);

  verdict.textContent =
    report.tier === 'webgpu' && report.frameMode === 'gpu-blit'
      ? `${compact(report.steps)} steps on the GPU tier, positions blitted on the device.`
      : `${compact(report.steps)} steps on the ${report.tier} tier: ${report.reason}`;
  verdict.classList.toggle('is-warn', report.tier !== 'webgpu');
  paintTier();
  paintCounters();
}

// ---------------------------------------------------------------------------
// painting
// ---------------------------------------------------------------------------

/** The tier panel and the badges. Cheap enough to call on every change. */
function paintTier(): void {
  showStat('ti-tier', report.tier, report.tier === 'webgpu', report.tier === 'cpu');
  showStat('ti-requested', `${report.requested}${report.strict ? ' (strict)' : ''}`);
  showStat('ti-backend', report.backend);

  // The M2 claim. Only a WebGPU backend has a GPUDevice to compare, so a WebGL2
  // renderer gets "n/a": painting FAIL there would be a false alarm about a tier
  // that was never able to share one.
  showClaim(
    'ti-device',
    report.backend === 'webgpu' ? report.deviceShared : null,
    report.backend === 'webgpu' ? undefined : 'n/a (WebGL renderer)',
  );

  // The M4 claim. `null` means "not a device solve" rather than "failed": the
  // CPU reference walks its colour batches in order, so there is no race to be
  // free of, and FAIL is reserved for a GPU backend that says it is not.
  showClaim(
    'ti-race',
    report.tier === 'webgpu' ? report.raceFree : null,
    report.tier === 'webgpu' ? undefined : 'sequential solver',
  );

  // The vertex path. Unlike the particle page there is no pipeline to compile, so
  // `null` never means "still warming up" here -- only "this tier has no device".
  const pathOk = report.viewMode === 'gpu' ? true : report.tier === 'webgpu' ? false : null;
  // The reason rides along with the claim it explains: a row that says
  // "cpu-upload" and nothing else leaves the reader guessing whether the page
  // chose that path or lost a fight with the driver.
  showClaim(
    'ti-path',
    pathOk,
    report.gpuError ? `${report.frameMode} (${report.gpuError})` : report.frameMode,
  );
  showClaim('ti-blit', report.blitBytes > 0 ? true : null, kib(report.blitBytes));
  showStat('ti-level', report.featureLevel ?? '\u2014', false, report.gpuError !== null);

  tierBadge.textContent =
    report.status === 'error'
      ? 'failed'
      : report.featureLevel
        ? `${report.tier} / ${report.featureLevel}`
        : report.tier;
  sceneBadge.textContent = report.scene;
  countBadge.textContent = `${compact(report.count)} nodes`;
  pathBadge.textContent = report.frameMode;
  pathBadge.classList.toggle('is-learned', report.frameMode === 'gpu-blit');
  report.lines = [...lines];
}

/** The Graph panel: the shape of the problem, which no tier is allowed to differ on. */
function paintGraph(): void {
  const plan = report.plan;
  showStat('gr-nodes', compact(plan.nodes), plan.nodes === report.count);
  showStat('gr-constraints', compact(plan.constraints));
  // `sheets` is the scene that makes this row mean something: one connected cloth
  // is one island, and a grouping pass with one group proves nothing.
  showStat('gr-islands', compact(plan.islands), plan.islands > 1);
  showStat('gr-colors', compact(plan.colors), plan.colors > 1);
  showStat('gr-workgroups', compact(plan.nodeWorkgroups));
  showStat('gr-dispatches', compact(plan.dispatchesPerStep));
  showStat('gr-draw', report.drawMode);
  // Pinned at 64 and asserted by the unit suite; shown here so a reader can check
  // the page against the plan's "workgroup size 为 64，不依赖 subgroup" without
  // opening a source file.
  showStat('gr-wgsize', `${SOFT_WORKGROUP_SIZE}`, SOFT_WORKGROUP_SIZE === 64);
}

/**
 * The counters panel.
 *
 * Good/warn are set from thresholds rather than left neutral, because a HUD where
 * every number is the same colour is a HUD nobody reads: `escaped` is supposed to
 * be zero.
 *
 * `max stretch` is graded against stability, not against stiffness. It is a relative
 * error -- the worst `|length - rest| / rest` over every edge -- and Gauss-Seidel
 * converges about one row of the mesh per iteration, so a 20k-node cloth hung from
 * its top edge reports several hundred percent at three iterations with `escaped`
 * still zero and every constraint solved exactly as batched. Painting that rose
 * would teach the reader that a correct run is a failing one, so the warn colour is
 * reserved for the mesh actually coming apart. The iterations slider is the control
 * that buys reach, and the footnote beside it says so.
 */
function paintCounters(): void {
  showStat('st-steps', `${compact(report.steps)} / ${compact(report.frames)}`);
  // More than two fixed steps per frame means the simulation is not keeping up
  // with the display, which is the number that decides whether a tier is usable.
  showStat('st-spf', `${report.stepsPerFrame}`, report.stepsPerFrame <= 2, report.behind);
  showStat('st-ms', `${report.msPerFrame.toFixed(2)} ms`);
  showStat('st-step-ms', `${report.msPerStep.toFixed(3)} ms`);
  showStat(
    'st-draws',
    `${report.drawCalls}`,
    report.drawCalls >= 1 && report.drawCalls <= 4,
    report.drawCalls > 4,
  );
  showStat('st-tris', compact(report.triangles));
  showStat(
    'st-escaped',
    compact(report.stats.escaped),
    report.stats.escaped === 0,
    report.stats.escaped > 0,
  );
  const stretch = report.stats.maxConstraintError;
  // Shown as a percentage because that is what it is: a relative error, and a bare
  // 0.85 sitting in a column of unitful numbers reads as a length.
  showStat('st-stretch', `${(stretch * 100).toFixed(1)}%`, stretch < 0.1, stretch >= 1);
  showStat('st-speed', report.stats.maxSpeed.toFixed(3));
  // A sleeping island while sleep is off is not a preference, it is a bug: nothing
  // in the layer puts an island to sleep that was not asked to.
  showStat(
    'st-islands',
    `${compact(report.stats.awakeIslands)} / ${compact(report.stats.sleepingIslands)}`,
    report.stats.sleepingIslands > 0 === settings.sleep,
    report.stats.sleepingIslands > 0 && !settings.sleep,
  );
  showStat('st-energy', compact(Math.round(report.stats.kineticEnergy)));
  showStat('st-digest', report.digest ? report.digest.slice(0, 16) : '\u2014');
  fpsBadge.textContent = `${report.fps.toFixed(0)} fps`;
  report.lines = [...lines];
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------

let building: Promise<void> | null = null;
let rebuildQueued = false;

/**
 * Rebuild, coalescing.
 *
 * Every control that changes the simulation rebuilds the page, and clicking
 * through the scene buttons fires several of those in a row. Running them serially
 * would allocate and free a device per click; holding one build behind the
 * in-flight one means the last click wins and the ones in between are dropped,
 * which is what clicking them meant.
 */
function rebuild(): void {
  if (building) {
    rebuildQueued = true;
    return;
  }
  building = boot().finally(() => {
    building = null;
    if (rebuildQueued) {
      rebuildQueued = false;
      rebuild();
    }
  });
}

/** Move `is-active` and `aria-selected` together, so the widget cannot disagree with itself. */
function selectSeg(host: HTMLElement, key: 'tier' | 'count' | 'scene', value: string): void {
  for (const el of Array.from(host.querySelectorAll<HTMLElement>('.seg'))) {
    const selected = el.dataset[key] === value;
    el.classList.toggle('is-active', selected);
    el.setAttribute('aria-selected', selected ? 'true' : 'false');
  }
}

/**
 * Push `settings` back into the controls.
 *
 * Called at the top of every build, so the buttons describe the run that is about
 * to happen rather than the last one that finished -- including when a URL
 * parameter overrode what the markup said.
 */
function syncControls(): void {
  selectSeg(tierSelect, 'tier', settings.tier);
  selectSeg(sceneSelect, 'scene', settings.scene);
  selectSeg(countSelect, 'count', String(settings.count));
  iterationsSlider.value = String(settings.iterations);
  iterationsValue.textContent = `${settings.iterations}`;
  stiffnessSlider.value = String(settings.stiffness);
  stiffnessValue.textContent = settings.stiffness.toFixed(2);
  wireBtn.classList.toggle('is-on', settings.wire);
  wireBtn.setAttribute('aria-pressed', String(settings.wire));
  sleepBtn.classList.toggle('is-on', settings.sleep);
  sleepBtn.setAttribute('aria-pressed', String(settings.sleep));
  paintIcon(runIcon, paused ? 'play' : 'pause');
  runLabel.textContent = paused ? 'Run' : 'Pause';
  // Benchmark mode has no loop to pause, and a button that looks like it does
  // something is worse than one that is visibly not applicable.
  runBtn.disabled = settings.steps > 0;
  runBtn.title =
    settings.steps > 0
      ? `Running a fixed ${compact(settings.steps)} steps`
      : 'Pause / resume the frame loop';
}

for (const el of Array.from(tierSelect.querySelectorAll<HTMLButtonElement>('.seg'))) {
  el.addEventListener('click', () => {
    const raw = el.dataset['tier'] ?? '';
    const next: TierChoice =
      raw === 'auto' || raw === 'webgpu' || raw === 'webgl2' || raw === 'cpu' ? raw : 'auto';
    if (next === settings.tier) return;
    settings.tier = next;
    rebuild();
  });
}

for (const el of Array.from(sceneSelect.querySelectorAll<HTMLButtonElement>('.seg'))) {
  el.addEventListener('click', () => {
    const raw = el.dataset['scene'] as SoftScene | undefined;
    if (!raw || raw === settings.scene) return;
    settings.scene = raw;
    rebuild();
  });
}

for (const el of Array.from(countSelect.querySelectorAll<HTMLButtonElement>('.seg'))) {
  el.addEventListener('click', () => {
    const value = Number(el.dataset['count']);
    if (!Number.isInteger(value) || value === settings.count) return;
    settings.count = Math.min(MAX_COUNT, Math.max(MIN_COUNT, value));
    rebuild();
  });
}

/**
 * Both sliders update their readout on `input` and rebuild on `change`.
 *
 * The split is the point. `iterations` and `stiffness` are words in the solver's
 * uniform and fields of the plan, so neither can change under a running system --
 * a rebuild is the only honest response. But rebuilding on every `input` event
 * would allocate a device per pixel of slider travel, so the number moves under
 * the thumb and the mesh is rebuilt once, when the thumb lets go.
 */
iterationsSlider.addEventListener('input', () => {
  const value = Number.parseInt(iterationsSlider.value, 10);
  if (!Number.isInteger(value) || value < 1) return;
  iterationsValue.textContent = `${value}`;
});
iterationsSlider.addEventListener('change', () => {
  const value = Number.parseInt(iterationsSlider.value, 10);
  if (!Number.isInteger(value) || value < 1 || value === settings.iterations) return;
  settings.iterations = value;
  rebuild();
});

stiffnessSlider.addEventListener('input', () => {
  const value = Number.parseFloat(stiffnessSlider.value);
  if (!Number.isFinite(value)) return;
  stiffnessValue.textContent = value.toFixed(2);
});
stiffnessSlider.addEventListener('change', () => {
  const value = Number.parseFloat(stiffnessSlider.value);
  if (!Number.isFinite(value) || value === settings.stiffness) return;
  settings.stiffness = value;
  rebuild();
});

/**
 * The wire overlay is a visibility change, not a rebuild.
 *
 * `SoftView` is built with the overlay in place either way, so this is
 * `setWireVisible` and nothing else -- which is what makes it free, and what makes
 * it the one control here that shows the shared-attribute claim: the lines and the
 * surface move together because they are reading one buffer, and a page that had
 * to rebuild to show that would hide the fact behind a loading veil.
 *
 * In `'edges'` mode the line view *is* the constraint graph, so hiding it would
 * blank the viewport. The button is disabled there rather than left to do that.
 */
wireBtn.addEventListener('click', () => {
  settings.wire = !settings.wire;
  syncControls();
  const view = live?.view;
  if (view && !view.disposed && view.drawMode === 'surface') view.setWireVisible(settings.wire);
  report.wire = view ? view.wireVisible : settings.wire;
});

sleepBtn.addEventListener('click', () => {
  settings.sleep = !settings.sleep;
  rebuild();
});

runBtn.addEventListener('click', () => {
  if (!live || live.scripted) return;
  paused = !paused;
  // Reset the clock on resume. The interval spent paused is wall time the
  // simulation should not be asked to catch up on, and the clamp in `tick` would
  // only hide a quarter second of it.
  lastTime = performance.now();
  fpsWindowStart = lastTime;
  fpsWindowFrames = 0;
  syncControls();
});

resetBtn.addEventListener('click', () => {
  // Same seed and same settings, so the mesh is regenerated byte-identically.
  // That is the property which makes a replay a replay rather than a rerun.
  rebuild();
});

syncControls();
rebuild();

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

/**
 * The box the mesh is confined to, as a wireframe.
 *
 * Worth drawing because the alternative is trusting the numbers: a cloth that
 * escaped its bounds looks identical to one that filled the screen, unless the
 * screen shows where the bounds were. It matters more here than it did for
 * particles, because `escaped` is a per-step count over *nodes* and one node
 * outside the box drags every edge it is connected to with it. Depth test off so
 * the far edges stay visible through the mesh.
 */
function boundsBox(bounds: {
  readonly min: readonly number[];
  readonly max: readonly number[];
}): THREE.Box3Helper {
  const box = new THREE.Box3(
    new THREE.Vector3(bounds.min[0] ?? 0, bounds.min[1] ?? 0, bounds.min[2] ?? 0),
    new THREE.Vector3(bounds.max[0] ?? 0, bounds.max[1] ?? 0, bounds.max[2] ?? 0),
  );
  const helper = new THREE.Box3Helper(box, new THREE.Color(0x39414d));
  // `material` is typed as a union with the array case, which a helper never
  // uses. Naming the one type it actually constructs is shorter than an
  // `Array.isArray` branch that cannot be taken.
  const material = helper.material as THREE.LineBasicMaterial;
  material.depthTest = false;
  material.transparent = true;
  material.opacity = 0.55;
  helper.renderOrder = -1;
  return helper;
}

/**
 * Keep the drawing buffer the size of the viewport.
 *
 * `setSize(..., false)` because the canvas is stretched by CSS: letting three.js
 * write inline styles too would fight the stylesheet, and a drawing buffer whose
 * aspect disagrees with its element is a scene that looks squashed.
 */
function watchViewport(
  renderer: THREE.WebGPURenderer,
  camera: THREE.PerspectiveCamera,
  canvas: HTMLCanvasElement,
): void {
  observer?.disconnect();
  observer = new ResizeObserver(() => {
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });
  observer.observe(canvas);
}

/**
 * Build the page.
 *
 * One function owns the whole assembly because the order inside it is the
 * argument: a device, then a renderer handed that device, then a soft-body system
 * handed the same `SharedDevice`, then a view handed that renderer. In any other
 * order `deviceShared` comes out false and the page still looks like it works --
 * which is precisely the failure this page exists to make visible.
 */
async function boot(): Promise<void> {
  teardown();
  logList.replaceChildren();
  lines.length = 0;
  paused = false;
  frameMs = 0;
  fpsWindowStart = 0;
  fpsWindowFrames = 0;
  counterDue = 0;
  refreshInFlight = false;
  refreshDue = false;
  canvasSampled = false;
  report.canvasBytes = 0;

  const count = settings.count;
  Object.assign(report, {
    status: 'booting',
    error: undefined,
    requested: settings.tier,
    strict: settings.strict,
    tier: 'cpu',
    reason: 'building',
    featureLevel: undefined,
    webgpuAvailable: false,
    backend: 'none',
    deviceShared: false,
    raceFree: false,
    viewMode: 'cpu',
    frameMode: 'cpu-upload',
    blitBytes: 0,
    gpuError: null,
    seed: settings.seed,
    scene: settings.scene,
    count,
    iterations: settings.iterations,
    stiffness: settings.stiffness,
    sleep: settings.sleep,
    wire: false,
    drawMode: 'surface',
    frames: 0,
    steps: 0,
    stepsPerFrame: 0,
    msPerFrame: 0,
    msPerStep: 0,
    msPerStepMean: 0,
    msPerStepP95: 0,
    stepSamples: 0,
    fps: 0,
    behind: false,
    drawCalls: 0,
    triangles: 0,
    plan: { ...EMPTY_PLAN },
    stats: { ...EMPTY_STATS },
    digest: '',
  } satisfies Partial<SoftReport>);
  syncControls();
  paintTier();
  paintGraph();
  paintCounters();
  loading.textContent = 'Probing adapters\u2026';
  loading.classList.remove('is-hidden');
  verdict.textContent = 'Building\u2026';
  verdict.classList.remove('is-warn');

  try {
    // Ask for a device only where a device could be used. Probing `navigator.gpu`
    // on a forced-CPU run would leave an adapter request in the log for a tier
    // that never wanted one, and the log is the evidence on this page.
    const wantsDevice = settings.tier === 'auto' || settings.tier === 'webgpu';
    let shared: SharedDevice | null = null;
    if (wantsDevice) {
      manager ??= new SharedDeviceManager({ label: 'soft-demo' });
      shared = await manager.acquire();
      if (!shared) {
        const failure = manager.failure;
        log(
          `no WebGPU device: ${failure?.reason ?? 'unknown'}${failure?.error ? ` (${failure.error})` : ''}`,
          'warn',
        );
      }
    }
    // Before the system is built, because `strict` throws inside that call and a
    // report that never learns whether an adapter existed is one the specs cannot
    // skip on.
    report.webgpuAvailable = shared !== null;

    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    viewport.appendChild(canvas);

    // No device means no WebGPU, and `three/webgpu` will otherwise request one of
    // its own and hand the page two adapters. `forceWebGL` asks it not to.
    const renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: false,
      ...(shared ? { device: shared.device } : { forceWebGL: true }),
    });
    renderer.setSize(
      Math.max(1, viewport.clientWidth),
      Math.max(1, viewport.clientHeight),
      false,
    );
    await renderer.init();

    // Built before the system, and on the CPU, on every tier: the graph is the
    // input to both solvers, not a product of either. Same seed and same options
    // is the same bytes, which is what makes the CPU/GPU comparison below a
    // comparison of solvers rather than of meshes.
    loading.textContent = `Building ${count.toLocaleString('en-US')}-node ${settings.scene}\u2026`;
    const mesh = new SoftMesh({
      count,
      scene: settings.scene,
      seed: settings.seed,
      speed: INITIAL_SPEED,
    });
    const options: SoftSimOptions = {
      iterations: settings.iterations,
      stiffness: settings.stiffness,
      sleep: settings.sleep,
    };
    const handle = await createSoftSystem({
      mesh,
      options,
      ...(settings.tier !== 'auto' ? { tier: settings.tier } : {}),
      ...(shared ? { shared } : {}),
      ...(settings.strict ? { strict: true } : {}),
      manager: manager ?? undefined,
      onFallback: (event) => log(`${event.from} fell back to ${event.to}: ${event.reason}`, 'warn'),
    });
    for (const line of handle.probe.lines) log(line);

    const backend = renderer.backend as unknown as RendererInternals;
    report.backend = backend.isWebGPUBackend === true ? 'webgpu' : 'webgl2';
    report.tier = handle.tier;
    report.reason = handle.reason;
    report.featureLevel = handle.probe.webgpu.featureLevel;
    report.fixedDt = handle.system.fixedDt;
    report.raceFree = handle.gpu?.raceFree ?? false;
    // The M2 claim at this layer, checked as a pointer comparison rather than
    // believed because both halves were built from one manager.
    report.deviceShared = shared !== null && backend.device === shared.device;

    const plan = handle.system.plan;
    report.plan = {
      nodes: plan.nodes,
      constraints: plan.constraints,
      islands: plan.islands,
      colors: plan.colors,
      iterations: plan.iterations,
      nodeWorkgroups: plan.nodeWorkgroups,
      dispatchesPerStep: plan.dispatchesPerStep,
    };

    const view = new SoftView({
      system: handle.system,
      renderer: renderer as unknown as RendererLike,
      // Built whether or not it is showing, so the Wire button is a visibility
      // change rather than a rebuild. See the handler for why that matters.
      wireframe: true,
    });
    report.drawMode = view.drawMode;
    view.setWireVisible(view.drawMode === 'edges' ? true : settings.wire);
    report.wire = view.wireVisible;
    wireBtn.disabled = view.drawMode === 'edges';
    wireBtn.title =
      view.drawMode === 'edges'
        ? 'This scene is drawn as the constraint graph already; there is no surface to overlay'
        : 'Draw the constraint graph over the surface';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1013);
    scene.add(new THREE.HemisphereLight(0xbfd4e6, 0x191c21, 1.1));
    const key = new THREE.DirectionalLight(0xffe3bd, 1.5);
    key.position.set(14, 20, 12);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x63c6b6, 0.7);
    rim.position.set(-16, -6, -12);
    scene.add(rim);
    scene.add(view.object);
    scene.add(boundsBox(handle.system.bounds));

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 400);
    camera.position.set(10, 4, 16);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
    controls.update();

    live = {
      handle,
      runner: new SoftRunner(handle.system),
      view,
      renderer,
      controls,
      scene,
      camera,
      canvas,
      shared,
      scripted: settings.steps > 0,
    };
    watchViewport(renderer, camera, canvas);
    loading.classList.add('is-hidden');
    paintTier();
    paintGraph();
    verdict.textContent = report.deviceShared
      ? `Running on ${report.tier}: one GPUDevice, shared with three.js.`
      : `Running on ${report.tier}: ${report.reason}`;
    verdict.classList.toggle('is-warn', report.tier !== 'webgpu');
    log(
      `${compact(count)} ${settings.scene} nodes on the ${handle.tier} tier | ` +
        `three.js r${THREE.REVISION} | backend ${report.backend} | ` +
        `fixedDt ${handle.system.fixedDt.toFixed(4)}s | ` +
        `deterministic: ${handle.system.deterministic ? 'yes' : 'no'}`,
      'head',
    );
    log(
      `${compact(plan.constraints)} constraints | ${compact(plan.islands)} islands | ` +
        `${compact(plan.colors)} color batches | ${plan.iterations} iterations | ` +
        `${compact(plan.nodeWorkgroups)} node workgroups | ` +
        `${compact(plan.dispatchesPerStep)} dispatches/step | draw ${view.drawMode}`,
    );
    if (handle.gpu) {
      log(
        `solver buffers ${kib(handle.gpu.budget.total)}, largest ${kib(handle.gpu.budget.largest)}`,
      );
    }

    if (live.scripted) {
      await runScripted(settings.steps);
    } else {
      report.status = 'live';
      startLoop();
    }
  } catch (error) {
    report.status = 'error';
    report.error = messageOf(error);
    log(`build failed: ${report.error}`, 'warn');
    verdict.textContent = report.error;
    verdict.classList.add('is-warn');
    loading.textContent = report.error;
    loading.classList.remove('is-hidden');
    tierBadge.textContent = 'failed';
    teardown();
  } finally {
    paintTier();
    paintGraph();
    paintCounters();
  }
}

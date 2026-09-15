/**
 * Browser demo for the M3 GPU particle layer.
 *
 * The page exists to make three claims visible rather than merely tested:
 *
 *   1. Simulation and rendering share one `GPUDevice`. The page acquires the
 *      device through `SharedDeviceManager`, hands the raw device to
 *      `WebGPURenderer`, and hands the same `SharedDevice` to
 *      `createParticleSystem`. The "device shared" row is a pointer comparison,
 *      not a hope.
 *   2. The instance matrices never round-trip through the CPU. On the WebGPU
 *      tier the publish buffer is expanded into `mat4`s on the device and
 *      blitted into the `GPUBuffer` three.js already allocated for
 *      `instanceMatrix`; the "blitted / frame" row is that copy's byte count.
 *   3. A worse tier still runs. Picking WebGL2 or CPU in the bar rebuilds the
 *      same scene on `particleCpu.ts` and the counters keep moving, which is the
 *      acceptance criterion about automatic downgrade seen from the reader's
 *      side of the machine.
 *
 * Everything here is the real code path: `createParticleSystem` does the
 * probing, `ParticleRunner` does the fixed-step decoupling, and `ParticleView`
 * does the drawing. Nothing in this file special-cases the browser.
 *
 * # Two ways to run
 *
 * Without parameters the page is a live demo on a rAF loop. With `?steps=N` it
 * runs exactly `N` simulation steps, timed, then stops and publishes a final
 * report -- that is the mode `e2e/particles_gpu.spec.ts` and
 * `scripts/bench_gpu_particles.mjs` drive, because a benchmark that measures a
 * display's refresh rate is measuring the wrong thing. Every parameter is also a
 * way to reproduce a reported bug without touching the UI.
 */

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  Atom,
  Cpu,
  Gauge,
  Layers,
  Orbit,
  Pause,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Terminal,
  Waypoints,
  Zap,
  createElement,
  type IconNode,
} from 'lucide';

import type { RenderTier } from '@threedream/gpu/capabilities.js';
import { SharedDeviceManager, type SharedDevice } from '@threedream/gpu/device.js';
import { ParticleField } from '@threedream/gpu/particleField.js';
import {
  ParticleRunner,
  createParticleSystem,
  type ParticleSystemHandle,
} from '@threedream/gpu/particles.js';
import type { ParticleSimOptions } from '@threedream/gpu/particleTypes.js';
import { ParticleView, type RendererLike } from '@threedream/render/particles.js';

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

/** What the tier bar offers. `'auto'` means "probe and take the best". */
export type TierChoice = RenderTier | 'auto';

/**
 * The published state of the page.
 *
 * Exported for the same reason `SharedDeviceReport` is: the e2e specs import the
 * type rather than re-declaring it, so a field renamed here breaks the spec at
 * compile time instead of arriving as `undefined` at assert time.
 */
export interface ParticlesReport {
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
  viewMode: 'gpu' | 'cpu';
  frameMode: 'gpu-blit' | 'cpu-upload';
  blitBytes: number;
  gpuError: string | null;
  seed: number;
  count: number;
  collisions: boolean;
  nbody: boolean;
  radiusScale: number;
  fixedDt: number;
  frames: number;
  steps: number;
  stepsPerFrame: number;
  msPerFrame: number;
  /**
   * Per-step cost as a distribution: `msPerStep` is the median timed chunk of a
   * scripted run, `msPerStepMean` the mean it replaced, `msPerStepP95` the spread,
   * `stepSamples` how many chunks that rests on. A mean over the four chunks a
   * 30-step rung produced was a number one contended chunk could triple; see the
   * same fields on `SoftReport` in demo/soft.ts for the measurement.
   */
  msPerStep: number;
  msPerStepMean: number;
  msPerStepP95: number;
  stepSamples: number;
  fps: number;
  behind: boolean;
  drawCalls: number;
  triangles: number;
  stats: {
    contacts: number;
    escaped: number;
    hashOverflow: number;
    maxSpeed: number;
    kineticEnergy: number;
  };
  digest: string;
  canvasBytes: number;
  lines: readonly string[];
}

declare global {
  interface Window {
    __particles?: ParticlesReport;
  }
}

const report: ParticlesReport = {
  status: 'booting',
  threeRevision: THREE.REVISION,
  requested: 'auto',
  strict: false,
  tier: 'cpu',
  reason: 'not built yet',
  webgpuAvailable: false,
  backend: 'none',
  deviceShared: false,
  viewMode: 'cpu',
  frameMode: 'cpu-upload',
  blitBytes: 0,
  gpuError: null,
  seed: 0,
  count: 0,
  collisions: true,
  nbody: false,
  radiusScale: 1,
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
  stats: { contacts: 0, escaped: 0, hashOverflow: 0, maxSpeed: 0, kineticEnergy: 0 },
  digest: '',
  canvasBytes: 0,
  lines: [],
};
window.__particles = report;

// ---------------------------------------------------------------------------
// dom
// ---------------------------------------------------------------------------

const ICONS = {
  atom: Atom,
  cpu: Cpu,
  gauge: Gauge,
  layers: Layers,
  orbit: Orbit,
  pause: Pause,
  play: Play,
  'rotate-ccw': RotateCcw,
  'shield-check': ShieldCheck,
  sparkles: Sparkles,
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
if (brandLogo) paintIcon(brandLogo, 'orbit');

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
const countBadge = byId('count-badge');
const pathBadge = byId('path-badge');
const fpsBadge = byId('fps-badge');
const runBtn = byId('run-btn') as HTMLButtonElement;
const runIcon = runBtn.querySelector<HTMLElement>('[data-icon]')!;
const runLabel = runBtn.querySelector<HTMLElement>('.btn-label')!;
const resetBtn = byId('reset-btn') as HTMLButtonElement;
const collisionsBtn = byId('collisions-btn') as HTMLButtonElement;
const nbodyBtn = byId('nbody-btn') as HTMLButtonElement;
const radiusSlider = byId('radius-slider') as HTMLInputElement;
const radiusValue = byId('radius-value') as HTMLOutputElement;
const tierSelect = byId('tier-select');
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

/** The counts the segmented control offers, read off the markup. */
const COUNT_CHOICES: number[] = Array.from(
  countSelect.querySelectorAll<HTMLElement>('[data-count]'),
  (el) => Number(el.dataset['count']),
).filter((n) => Number.isInteger(n) && n > 0);

/**
 * All-pairs gravity is O(n^2) and there is no way around it. 4096^2 is 16M pairs,
 * which a GPU eats in a few milliseconds; 100k^2 is ten billion, which it does
 * not. So the count is capped while n-body is on, and the disabled buttons say why
 * in their title -- a page that quietly hung would read as a bug in the layer
 * under test rather than a fact about the algorithm.
 */
const NBODY_MAX_COUNT = 4096;
const MIN_COUNT = 1;
const MAX_COUNT = 250000;
const DEFAULT_SEED = 1234;

interface Settings {
  count: number;
  seed: number;
  tier: TierChoice;
  strict: boolean;
  collisions: boolean;
  nbody: boolean;
  radiusScale: number;
  /** Above zero: run exactly this many steps, timed, then report. Benchmark mode. */
  steps: number;
}

/**
 * Whichever `.seg` carries `is-active` IS the default.
 *
 * Restating it here would give the page two answers to one question, and only one
 * of them is visible to the person reading the markup.
 */
function initialSelected(host: HTMLElement, key: 'tier' | 'count', fallback: string): string {
  return host.querySelector<HTMLElement>('.seg.is-active')?.dataset[key] ?? fallback;
}

/**
 * Every knob, read once.
 *
 * A URL is how a bug report arrives. `?count=100000&tier=webgpu&steps=600` has to
 * be reproducible by pasting it, not by clicking four controls in the order
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

  const markupCount = Number(initialSelected(countSelect, 'count', '10000'));
  const sliderRadius = Number(radiusSlider.value);
  const sliderMin = Number(radiusSlider.min);
  const sliderMax = Number(radiusSlider.max);
  return {
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
    collisions: flag('collisions', collisionsBtn.classList.contains('is-on')),
    nbody: flag('nbody', nbodyBtn.classList.contains('is-on')),
    radiusScale: num(
      'radius',
      Number.isFinite(sliderRadius) && sliderRadius > 0 ? sliderRadius : 1,
      Number.isFinite(sliderMin) && sliderMin > 0 ? sliderMin : 0.1,
      Number.isFinite(sliderMax) && sliderMax > sliderMin ? sliderMax : 4,
    ),
    steps: int('steps', 0, 0, 1000000),
  };
}

const settings = readSettings();

/**
 * What actually gets simulated.
 *
 * Everything reads the count through here rather than through `settings.count`, so
 * the n-body cap is one rule in one place instead of a `Math.min` repeated at four
 * call sites that would eventually disagree.
 */
function effectiveCount(): number {
  return settings.nbody ? Math.min(settings.count, NBODY_MAX_COUNT) : settings.count;
}

// ---------------------------------------------------------------------------
// live state
// ---------------------------------------------------------------------------

/**
 * The two renderer fields three.js does not type but this page needs.
 *
 * `device` is how the M2 claim gets checked here: a pointer comparison between the
 * backend's device and the `SharedDevice` the compute kernels were built on. And
 * `isWebGPUBackend` is the honest tier test, because `three/webgpu` handed
 * `forceWebGL` runs a WebGL backend under a WebGPU-named renderer -- reading the
 * import would report the wrong thing.
 */
interface RendererInternals {
  readonly device: unknown;
  readonly isWebGPUBackend?: boolean;
}

interface Live {
  readonly handle: ParticleSystemHandle;
  readonly runner: ParticleRunner;
  readonly view: ParticleView;
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
 * The view first, because it holds the expansion pipeline against the device.
 * Then the system, which releases its own reference to that device. Then the
 * controls and the renderer, and only then our reference -- three.js does not
 * destroy a device it was handed, since `WebGPUBackend.dispose()` destroys one
 * only when it requested that device itself. So the final `release()` is what
 * actually frees the GPU, and doing it earlier would drop the device out from
 * under a pipeline still in flight.
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
 * write; on the GPU those are per-step because the clear dispatch zeroes them at
 * the start of every step, which is what makes them comparable to the CPU tier's
 * `stats()` at all. `readback()` copies positions back, and it is the only thing
 * that makes `maxSpeed`, `kineticEnergy` and the digest mean anything on this tier
 * -- without it `field.data` is still the initial state.
 *
 * Rate limited rather than per-frame, because a map/unmap pair is a stall and a HUD
 * that costs frames ends up measuring itself.
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
 * `runner.frame(dt)` is the whole point of the decoupling: it decides how many
 * fixed steps this frame was worth, so a 240 Hz panel and a 30 Hz one simulate the
 * same trajectory at different smoothness instead of at different speeds. The
 * view's `update()` sits between the step and the draw, because filling
 * `instanceMatrix` is a rendering concern and nothing in it writes the field.
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
    // fixed slices to catch up is how a demo turns into a hang on focus.
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
      if (now - counterDue >= 500) {
        counterDue = now;
        void refreshCounters();
      }
      // Paint more often than we read. The runner's numbers are free, and a HUD
      // that only moves twice a second looks broken even when it is not.
      if (now - paintDueAt >= 120) {
        paintDueAt = now;
        // The tier panel as well as the counters. It is the half that carries the
        // M3 claim, and on a live run the claim is decided a frame or two *after*
        // the boot-time paint -- the expansion pipeline is still compiling then.
        // Painting it once and never again leaves the page asserting `cpu-upload`
        // next to a field that is blitting, which is worse than showing nothing.
        paintTier();
        paintCounters();
      }
    }

    // Late enough that the GPU path has had a frame to attach, early enough that
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
 * This is what `e2e/particles_gpu.spec.ts` and `scripts/bench_gpu_particles.mjs`
 * drive, and it exists because a frame loop measures a display. Work is submitted
 * in chunks, each one rendered and then awaited through
 * `queue.onSubmittedWorkDone()`, so the timing covers the GPU rather than the rAF
 * cadence; the yields between chunks sit outside the timed region.
 *
 * Chunked rather than one `advance(target)` for two reasons. A single call would
 * queue thousands of dispatches with nothing presented in between, which is a fair
 * stress test and a meaningless benchmark. And `onSubmittedWorkDone` on an empty
 * queue returns immediately, so without a chunk boundary the number reported would
 * be how long it took to *record* commands -- the one measurement that makes a GPU
 * look as fast as a CPU.
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
  // `msPerStep` on ParticlesReport for why a mean over four chunks is not enough.
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
  report.viewMode = await current.view.settled();
  // `frameMode` describes one frame, not a mode, and the last frame inside the
  // loop may well have run before the expansion pipeline landed. One more pass
  // after settling is what makes the report describe the path the page is on.
  report.frameMode = current.view.update();
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
    `instance path ${report.frameMode}` +
      `${report.blitBytes > 0 ? ` (${kib(report.blitBytes)} per frame)` : ''} | ` +
      `draw calls ${report.drawCalls} | triangles ${compact(report.triangles)}`,
    report.frameMode === 'gpu-blit' ? 'good' : 'plain',
  );
  log(
    `escaped ${report.stats.escaped} | contacts ${compact(report.stats.contacts)} | ` +
      `hash overflow ${compact(report.stats.hashOverflow)} | ` +
      `max speed ${report.stats.maxSpeed.toFixed(3)} | ` +
      `kinetic energy ${compact(Math.round(report.stats.kineticEnergy))}`,
    report.stats.escaped === 0 ? 'good' : 'warn',
  );
  log(`digest ${report.digest || '(none)'}`);
  log(`canvas PNG bytes ${report.canvasBytes}`);

  verdict.textContent =
    report.tier === 'webgpu' && report.frameMode === 'gpu-blit'
      ? `${compact(report.steps)} steps on the GPU tier, matrices blitted on the device.`
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

  // The M3 claim. `null` means "not settled yet" rather than "failed", which is
  // what the first frames of a WebGPU run look like while the pipeline compiles;
  // FAIL is reserved for a WebGPU backend that ended up on the CPU path.
  const pathOk =
    report.viewMode === 'gpu' ? true : report.backend === 'webgpu' ? false : null;
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
  countBadge.textContent = `${compact(report.count)} particles`;
  pathBadge.textContent = report.frameMode;
  pathBadge.classList.toggle('is-learned', report.frameMode === 'gpu-blit');
  report.lines = [...lines];
}

/**
 * The counters panel.
 *
 * Good/warn are set from thresholds rather than left neutral, because a HUD where
 * every number is the same colour is a HUD nobody reads: `escaped` and `hash
 * overflow` are supposed to be zero, and draw calls are supposed to be small.
 */
function paintCounters(): void {
  showStat('st-count', compact(report.count));
  showStat('st-steps', `${compact(report.steps)} / ${compact(report.frames)}`);
  // More than two fixed steps per frame means the simulation is not keeping up
  // with the display, which is the number that decides whether a tier is usable.
  showStat('st-spf', `${report.stepsPerFrame}`, report.stepsPerFrame <= 2, report.behind);
  showStat('st-ms', `${report.msPerFrame.toFixed(2)} ms`);
  showStat(
    'st-draws',
    `${report.drawCalls}`,
    report.drawCalls >= 1 && report.drawCalls <= 4,
    report.drawCalls > 4,
  );
  showStat('st-tris', compact(report.triangles));
  showStat('st-contacts', compact(report.stats.contacts));
  showStat(
    'st-escaped',
    compact(report.stats.escaped),
    report.stats.escaped === 0,
    report.stats.escaped > 0,
  );
  showStat(
    'st-overflow',
    compact(report.stats.hashOverflow),
    report.stats.hashOverflow === 0,
    report.stats.hashOverflow > 0,
  );
  showStat('st-speed', report.stats.maxSpeed.toFixed(3));
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
 * through the count buttons fires several of those in a row. Running them serially
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
function selectSeg(host: HTMLElement, key: 'tier' | 'count', value: string): void {
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
  selectSeg(countSelect, 'count', String(settings.count));
  for (const el of Array.from(countSelect.querySelectorAll<HTMLButtonElement>('.seg'))) {
    const value = Number(el.dataset['count']);
    const blocked = settings.nbody && value > NBODY_MAX_COUNT;
    el.disabled = blocked;
    if (blocked) {
      el.title = `n-body is all-pairs: capped at ${compact(NBODY_MAX_COUNT)} while it is on`;
    } else {
      el.removeAttribute('title');
    }
  }
  collisionsBtn.classList.toggle('is-on', settings.collisions);
  collisionsBtn.setAttribute('aria-pressed', String(settings.collisions));
  nbodyBtn.classList.toggle('is-on', settings.nbody);
  nbodyBtn.setAttribute('aria-pressed', String(settings.nbody));
  radiusSlider.value = String(settings.radiusScale);
  radiusValue.textContent = `${settings.radiusScale.toFixed(1)}\u00d7`;
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

for (const el of Array.from(countSelect.querySelectorAll<HTMLButtonElement>('.seg'))) {
  el.addEventListener('click', () => {
    const value = Number(el.dataset['count']);
    if (!Number.isInteger(value) || value === settings.count) return;
    settings.count = Math.min(MAX_COUNT, Math.max(MIN_COUNT, value));
    rebuild();
  });
}

collisionsBtn.addEventListener('click', () => {
  settings.collisions = !settings.collisions;
  rebuild();
});

nbodyBtn.addEventListener('click', () => {
  settings.nbody = !settings.nbody;
  rebuild();
});

/**
 * Radius is a draw-time scale, not a simulation parameter.
 *
 * `ParticleView.setRadiusScale` changes the matrices being written and nothing
 * else, so this must not rebuild: rebuilding would restart the simulation to make
 * the particles look bigger, and every drag position would be a different run. The
 * physical radius lives in `field.data`, and only a rebuild changes that.
 */
radiusSlider.addEventListener('input', () => {
  const value = Number(radiusSlider.value);
  if (!Number.isFinite(value) || value <= 0) return;
  settings.radiusScale = value;
  radiusValue.textContent = `${value.toFixed(1)}\u00d7`;
  live?.view.setRadiusScale(value);
  report.radiusScale = value;
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
  // Same seed and same settings, so the field is regenerated byte-identically.
  // That is the property which makes a replay a replay rather than a rerun.
  rebuild();
});

syncControls();
rebuild();

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

/**
 * The box the particles are confined to, as a wireframe.
 *
 * Worth drawing because the alternative is trusting the numbers: a field that
 * escaped its bounds looks identical to a field that filled the screen, unless
 * the screen shows where the bounds were. Depth test off so the far edges stay
 * visible through the cloud.
 */
function boundsBox(bounds: { readonly min: readonly number[]; readonly max: readonly number[] }): THREE.Box3Helper {
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
 * argument: a device, then a renderer handed that device, then a particle system
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

  const count = effectiveCount();
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
    viewMode: 'cpu',
    frameMode: 'cpu-upload',
    blitBytes: 0,
    gpuError: null,
    seed: settings.seed,
    count,
    collisions: settings.collisions,
    nbody: settings.nbody,
    radiusScale: settings.radiusScale,
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
    stats: { contacts: 0, escaped: 0, hashOverflow: 0, maxSpeed: 0, kineticEnergy: 0 },
    digest: '',
  } satisfies Partial<ParticlesReport>);
  syncControls();
  paintTier();
  paintCounters();
  loading.textContent = 'Probing adapters\u2026';
  loading.classList.remove('is-hidden');
  verdict.textContent = 'Building\u2026';
  verdict.classList.remove('is-warn');
  if (settings.nbody && settings.count > NBODY_MAX_COUNT) {
    log(
      `n-body is all-pairs: the count is capped at ${compact(NBODY_MAX_COUNT)} while it is on`,
      'warn',
    );
  }

  try {
    // Ask for a device only where a device could be used. Probing `navigator.gpu`
    // on a forced-CPU run would leave an adapter request in the log for a tier
    // that never wanted one, and the log is the evidence on this page.
    const wantsDevice = settings.tier === 'auto' || settings.tier === 'webgpu';
    let shared: SharedDevice | null = null;
    if (wantsDevice) {
      manager ??= new SharedDeviceManager({ label: 'particles-demo' });
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

    const field = new ParticleField({
      count,
      scene: 'sphere',
      seed: settings.seed,
      speed: 3,
    });
    const options: ParticleSimOptions = {
      collisions: settings.collisions,
      nbody: settings.nbody,
      // A little drag under n-body, or the cloud collapses to a point and the
      // frame after that is a division by a softening constant.
      damping: settings.nbody ? 0.05 : 0,
      nbodyStrength: 6,
      restitution: 0.55,
    };
    const handle = await createParticleSystem({
      field,
      options,
      ...(settings.tier !== 'auto' ? { tier: settings.tier } : {}),
      ...(shared ? { shared } : {}),
      ...(settings.strict ? { strict: true } : {}),
      onFallback: (event) => log(`${event.from} fell back to ${event.to}: ${event.reason}`, 'warn'),
    });
    for (const line of handle.probe.lines) log(line);

    const backend = renderer.backend as unknown as RendererInternals;
    report.backend = backend.isWebGPUBackend === true ? 'webgpu' : 'webgl2';
    report.tier = handle.tier;
    report.reason = handle.reason;
    report.featureLevel = handle.probe.webgpu.featureLevel;
    report.fixedDt = handle.system.fixedDt;
    // The M2 claim at this layer, checked as a pointer comparison rather than
    // believed because both halves were built from one manager.
    report.deviceShared = shared !== null && backend.device === shared.device;

    const view = new ParticleView({
      system: handle.system,
      renderer: renderer as unknown as RendererLike,
      radiusScale: settings.radiusScale,
    });

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1013);
    scene.add(new THREE.HemisphereLight(0xbfd4e6, 0x191c21, 1.1));
    const key = new THREE.DirectionalLight(0xffe3bd, 1.5);
    key.position.set(14, 20, 12);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x63c6b6, 0.7);
    rim.position.set(-16, -6, -12);
    scene.add(rim);
    scene.add(view.mesh);
    scene.add(boundsBox(handle.system.bounds));

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 400);
    camera.position.set(13, 9, 24);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
    controls.update();

    live = {
      handle,
      runner: new ParticleRunner(handle.system),
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
    verdict.textContent = report.deviceShared
      ? `Running on ${report.tier}: one GPUDevice, shared with three.js.`
      : `Running on ${report.tier}: ${report.reason}`;
    verdict.classList.toggle('is-warn', report.tier !== 'webgpu');
    log(
      `${compact(count)} particles on the ${handle.tier} tier | three.js r${THREE.REVISION} | ` +
        `backend ${report.backend} | fixedDt ${handle.system.fixedDt.toFixed(4)}s | ` +
        `deterministic: ${handle.system.deterministic ? 'yes' : 'no'}`,
      'head',
    );

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
    paintCounters();
  }
}

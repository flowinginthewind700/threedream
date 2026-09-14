/**
 * Browser half of the M1 determinism claim.
 *
 * `tests/wasm_backend.test.ts` proves the Rust/wasm kernel and `builtin.ts`
 * agree bit for bit *in Node*. That leaves a gap the milestone explicitly names:
 * Node, the browser, and a replay must all land on the same digest. wasm in a
 * browser is a different instantiation path -- fetched over HTTP, compiled by a
 * different engine, SIMD lowering decided by a different backend -- and "it is
 * the same bytes" is an assumption, not a result, until it runs here.
 *
 * So this page does three runs of `runReference` and compares all of them to
 * `REFERENCE_GOLDEN_DIGEST`:
 *
 *   builtin  -- the normative TypeScript solver, in the browser
 *   wasm     -- the Rust kernel, in the browser
 *   replay   -- a second wasm world, which is what "reproducible" means
 *
 * The viewport is not decoration. It is driven by the same
 * `REFERENCE_SCRIPT`/`REFERENCE_BODIES` the digests were computed from, so the
 * run being hashed and the run being watched cannot drift apart, and switching
 * the backend tab swaps the solver under a scene that must look identical.
 *
 * `window.__physicsCheck` carries the whole report; `e2e/wasm_physics.spec.ts`
 * reads it rather than scraping text out of the DOM.
 */

import * as THREE from 'three';
import {
  Activity,
  Boxes,
  Cpu,
  Fingerprint,
  Pause,
  Play,
  RotateCcw,
  createElement,
  type IconNode,
} from 'lucide';

import { createBuiltinPhysics } from '@threedream/physics/builtin.js';
import { createWasmPhysics } from '@threedream/physics/wasm.js';
import {
  REFERENCE_BODIES,
  REFERENCE_DT,
  REFERENCE_GOLDEN_DIGEST,
  REFERENCE_OPTIONS,
  REFERENCE_STEPS,
  applyReferenceEvents,
  buildReferenceScene,
  digestValues,
  runReference,
} from '@threedream/physics/reference.js';
import { ThreeRenderer, type VisualBinding } from '@threedream/render/scene.js';
import type { BodyDescriptor, PhysicsBackend } from '@threedream/physics/types.js';

// ---------------------------------------------------------------------------
// report shape
// ---------------------------------------------------------------------------

export interface PhysicsCheckReport {
  /** `running` until every measurement finished; `error` never leaves it empty. */
  status: 'running' | 'done' | 'error';
  error?: string;
  golden: string;
  digests: { builtin: string; wasm: string; replay: string };
  matches: {
    builtin: boolean;
    wasm: boolean;
    replay: boolean;
    /** Replay-vs-run, independent of the golden value. */
    replayOfWasm: boolean;
  };
  counts: {
    steps: number;
    bodies: number;
    samples: number;
    contacts: number;
    rays: number;
    /** Doubles folded into the digest. Guards against a truncated hash matching. */
    values: number;
  };
  timings: { builtinMs: number; wasmMs: number; speedup: number };
  liveBackend: string;
}

declare global {
  interface Window {
    __physicsCheck?: PhysicsCheckReport;
  }
}

const report: PhysicsCheckReport = {
  status: 'running',
  golden: REFERENCE_GOLDEN_DIGEST,
  digests: { builtin: '', wasm: '', replay: '' },
  matches: { builtin: false, wasm: false, replay: false, replayOfWasm: false },
  counts: { steps: 0, bodies: REFERENCE_BODIES.length, samples: 0, contacts: 0, rays: 0, values: 0 },
  timings: { builtinMs: 0, wasmMs: 0, speedup: 0 },
  liveBackend: 'builtin',
};
window.__physicsCheck = report;

// ---------------------------------------------------------------------------
// dom
// ---------------------------------------------------------------------------

const ICONS = {
  activity: Activity,
  boxes: Boxes,
  cpu: Cpu,
  fingerprint: Fingerprint,
  pause: Pause,
  play: Play,
  'rotate-ccw': RotateCcw,
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
const backendBadge = byId('backend-badge');
const stepBadge = byId('step-badge');
const verdict = byId('verdict');
const pauseBtn = byId('pause-btn') as HTMLButtonElement;
const restartBtn = byId('restart-btn') as HTMLButtonElement;

/** Write a digest and colour it by whether it matched the golden value. */
function showDigest(id: string, digest: string, golden: string): void {
  const el = byId(id);
  el.textContent = digest || '\u2014';
  el.classList.toggle('is-good', digest !== '' && digest === golden);
  el.classList.toggle('is-warn', digest !== '' && digest !== golden);
}

function showStat(id: string, text: string, good = false): void {
  const el = byId(id);
  el.textContent = text;
  el.classList.toggle('is-good', good);
  el.classList.toggle('is-warn', false);
}

function paintReport(): void {
  showDigest('dg-golden', report.golden, report.golden);
  showDigest('dg-builtin', report.digests.builtin, report.golden);
  showDigest('dg-wasm', report.digests.wasm, report.golden);
  showDigest('dg-replay', report.digests.replay, report.golden);
  const c = report.counts;
  showStat('st-steps', c.steps > 0 ? String(c.steps) : '\u2014');
  showStat('st-bodies', String(c.bodies));
  showStat('st-samples', c.samples.toLocaleString('en-US'));
  showStat('st-contacts', c.contacts.toLocaleString('en-US'));
  showStat('st-rays', String(c.rays));
  showStat('st-values', c.values.toLocaleString('en-US'));
  const t = report.timings;
  showStat('st-builtin-ms', t.builtinMs > 0 ? `${t.builtinMs.toFixed(1)} ms` : '\u2014');
  showStat('st-wasm-ms', t.wasmMs > 0 ? `${t.wasmMs.toFixed(1)} ms` : '\u2014');
  showStat('st-speedup', t.speedup > 0 ? `${t.speedup.toFixed(2)}x` : '\u2014', t.speedup >= 2);
}

// ---------------------------------------------------------------------------
// the three runs
// ---------------------------------------------------------------------------

function timed(backend: PhysicsBackend): { ms: number; run: ReturnType<typeof runReference> } {
  const t0 = performance.now();
  const run = runReference(backend, REFERENCE_STEPS);
  return { ms: performance.now() - t0, run };
}

async function measure(): Promise<void> {
  const builtin = timed(createBuiltinPhysics(REFERENCE_OPTIONS));
  report.digests.builtin = builtin.run.digest;
  report.timings.builtinMs = builtin.ms;
  report.counts = {
    ...report.counts,
    steps: builtin.run.steps,
    samples: builtin.run.samples.length,
    contacts: builtin.run.contacts.length,
    rays: builtin.run.rays.length,
    values: digestValues(builtin.run).length,
  };
  // One invariant the readouts depend on: a sample per body per step. If this
  // ever stops holding, the counts below are describing a different run than the
  // digest is, and the page would look healthy while proving nothing.
  if (builtin.run.samples.length !== REFERENCE_BODIES.length * REFERENCE_STEPS) {
    fail(`expected ${REFERENCE_BODIES.length * REFERENCE_STEPS} samples, got ${builtin.run.samples.length}`);
  }
  paintReport();

  const wasm = timed(await createWasmPhysics(REFERENCE_OPTIONS));
  report.digests.wasm = wasm.run.digest;
  report.timings.wasmMs = wasm.ms;
  report.timings.speedup = wasm.ms > 0 ? builtin.ms / wasm.ms : 0;
  paintReport();

  // A second world from the same module: same bytes, fresh state. If wasm were
  // leaking anything between worlds -- a static, a cached scratch buffer -- this
  // is where it would show up.
  const replay = timed(await createWasmPhysics(REFERENCE_OPTIONS));
  report.digests.replay = replay.run.digest;

  report.matches = {
    builtin: report.digests.builtin === report.golden,
    wasm: report.digests.wasm === report.golden,
    replay: report.digests.replay === report.golden,
    replayOfWasm: report.digests.replay === report.digests.wasm,
  };
  report.status = 'done';
  const m = report.matches;
  const all = m.builtin && m.wasm && m.replay && m.replayOfWasm;
  verdict.textContent = all
    ? 'All three runs reproduce the digest recorded in Node.'
    : `Divergence: builtin=${m.builtin} wasm=${m.wasm} replay=${m.replay}`;
  verdict.classList.toggle('is-warn', !all);
  paintReport();
}

function fail(message: string): never {
  throw new Error(message);
}

// ---------------------------------------------------------------------------
// live view
// ---------------------------------------------------------------------------

/** Per-label colour, from the demo palette. Unknown labels fall back to steel. */
const TINTS: Record<string, number> = {
  floor: 0x2b3038,
  wall: 0x2b3038,
  bouncy: 0xdd8f3c,
  slider: 0x4fb3a5,
  doomed: 0xd9705f,
  'pair-a': 0x8b95a3,
  'pair-b': 0x8b95a3,
  crate: 0xa2662a,
  frozen: 0x6d7480,
};

function geometryFor(descriptor: BodyDescriptor): THREE.BufferGeometry {
  const { shape } = descriptor;
  if (shape.kind === 'sphere') return new THREE.SphereGeometry(shape.radius, 28, 18);
  if (shape.kind === 'box') {
    const [hx, hy, hz] = shape.halfExtents;
    return new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2);
  }
  throw new Error(`reference scene has no capsule body; got ${shape.kind}`);
}

/**
 * The animated reference run.
 *
 * Steps on a fixed accumulator rather than once per frame, because the digest is
 * defined at `REFERENCE_DT` and a display at 120Hz or a stalled tab must not
 * change what is simulated. Frames are capped so a long pause cannot turn into a
 * catch-up spiral that locks the page.
 */
class LiveRun {
  private readonly renderer: ThreeRenderer;
  private readonly backend: PhysicsBackend;
  private readonly handles: number[];
  private readonly bindings: VisualBinding[] = [];
  private step = 0;
  private accumulator = 0;
  private last = performance.now();
  private paused = false;
  private finished = false;

  constructor(backend: PhysicsBackend) {
    this.backend = backend;
    this.renderer = new ThreeRenderer({
      container: viewport,
      backend,
      background: 0x0e1013,
      maxPixelRatio: 2,
    });
    this.renderer.camera.position.set(0.5, 4.4, 9);
    this.renderer.camera.lookAt(0, 0.6, 0);
    this.handles = buildReferenceScene(backend);
    this.buildMeshes();
    this.syncVisibility();
    this.renderer.start(() => this.tick());
  }

  /**
   * One mesh per descriptor, bound to the handle that descriptor was given.
   *
   * Both lists come from `REFERENCE_BODIES`, so geometry and physics cannot
   * disagree about how many bodies the scene has -- a mismatch throws rather than
   * drawing the first N of them and looking fine.
   */
  private buildMeshes(): void {
    if (this.handles.length !== REFERENCE_BODIES.length) {
      fail(`built ${this.handles.length} bodies for ${REFERENCE_BODIES.length} descriptors`);
    }
    REFERENCE_BODIES.forEach((descriptor, i) => {
      const label = descriptor.label ?? 'body';
      const mesh = new THREE.Mesh(
        geometryFor(descriptor),
        new THREE.MeshStandardMaterial({
          color: TINTS[label] ?? 0x8b95a3,
          roughness: descriptor.kind === 'static' ? 0.95 : 0.45,
          metalness: descriptor.kind === 'static' ? 0.05 : 0.15,
        }),
      );
      mesh.name = label;
      mesh.castShadow = descriptor.kind !== 'static';
      mesh.receiveShadow = true;
      const binding: VisualBinding = { handle: this.handles[i]!, object: mesh };
      this.bindings.push(binding);
      this.renderer.bind(binding);
    });
  }

  /** The renderer mirrors transforms; a body destroyed mid-run must vanish too. */
  private syncVisibility(): void {
    for (const binding of this.bindings) {
      binding.object.visible = this.backend.getBodyState(binding.handle) !== undefined;
    }
  }

  private tick(): void {
    const now = performance.now();
    const elapsed = Math.min((now - this.last) / 1000, 0.25);
    this.last = now;
    if (this.paused || this.finished) return;
    this.accumulator += elapsed;
    let taken = 0;
    while (this.accumulator >= REFERENCE_DT && taken < 8) {
      applyReferenceEvents(this.backend, this.handles, this.step);
      this.backend.step(REFERENCE_DT);
      // Drained for parity with `runReference`: an undrained backend keeps every
      // contact it ever generated, which is not the run being hashed.
      this.backend.drainContacts();
      this.accumulator -= REFERENCE_DT;
      this.step += 1;
      taken += 1;
    }
    this.finished = this.step >= REFERENCE_STEPS;
    this.syncVisibility();
    stepBadge.textContent = this.finished
      ? `run complete (${REFERENCE_STEPS} steps)`
      : `step ${this.step} / ${REFERENCE_STEPS}`;
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    // Reset the clock: otherwise the paused interval is simulated on resume.
    this.last = performance.now();
    this.accumulator = 0;
  }

  dispose(): void {
    this.renderer.dispose();
    // `PhysicsBackend.dispose` clears the bodies but keeps the world allocated so
    // it can be reused. This page builds a fresh world on every restart, so the
    // wasm backend's stronger teardown is used when it is there; one leaked world
    // per click adds up in a page people leave open.
    const strong = this.backend as { destroy?: () => void };
    if (typeof strong.destroy === 'function') strong.destroy();
    else this.backend.dispose();
  }
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

const backendFactories: Record<string, () => Promise<PhysicsBackend>> = {
  builtin: async () => createBuiltinPhysics(REFERENCE_OPTIONS),
  wasm: async () => createWasmPhysics(REFERENCE_OPTIONS),
};

const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>('#backend-select .seg'));

let live: LiveRun | null = null;
let liveName = 'builtin';
let paused = false;
let mounting: Promise<void> | null = null;

/**
 * Build a live run on `name`, replacing whatever was mounted.
 *
 * The veil is only lifted on success: if the kernel fails to load, the message
 * stays on screen instead of revealing an empty viewport.
 */
async function mount(name: string): Promise<void> {
  const factory = backendFactories[name];
  if (!factory) return;
  loading.textContent = `Loading ${name} kernel\u2026`;
  loading.classList.remove('is-hidden');
  live?.dispose();
  live = null;
  live = new LiveRun(await factory());
  live.setPaused(paused);
  loading.classList.add('is-hidden');
}

/**
 * Serialised mount.
 *
 * Two quick clicks would otherwise dispose a renderer that is still being built
 * and leave two rAF loops driving one canvas. Chaining costs one promise and
 * makes the visible order match the clicked order.
 */
function remount(name: string): Promise<void> {
  const next = async (): Promise<void> => mount(name);
  mounting = (mounting ?? Promise.resolve()).then(next, next);
  return mounting;
}

function selectBackend(name: string): void {
  if (!backendFactories[name]) return;
  for (const button of tabs) {
    const on = button.dataset['backend'] === name;
    button.classList.toggle('is-active', on);
    button.setAttribute('aria-selected', on ? 'true' : 'false');
  }
  liveName = name;
  report.liveBackend = name;
  backendBadge.textContent = name;
  void remount(name);
}

for (const button of tabs) {
  button.addEventListener('click', () => selectBackend(button.dataset['backend'] ?? 'builtin'));
}

pauseBtn.addEventListener('click', () => {
  paused = !paused;
  live?.setPaused(paused);
  const label = pauseBtn.querySelector('.btn-label');
  if (label) label.textContent = paused ? 'Resume' : 'Pause';
  const icon = pauseBtn.querySelector('[data-icon]');
  if (icon) paintIcon(icon, paused ? 'play' : 'pause');
});

restartBtn.addEventListener('click', () => {
  void remount(liveName);
});

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function main(): Promise<void> {
  stepBadge.textContent = `step 0 / ${REFERENCE_STEPS}`;
  paintReport();

  // The live view is the illustration; the digests are the result. A machine with
  // no WebGL should still get a verdict, so a failed mount is reported on the
  // veil and measurement continues regardless.
  try {
    await mount('builtin');
  } catch (error) {
    loading.textContent = `live view unavailable: ${describeError(error)}`;
  }

  try {
    await measure();
  } catch (error) {
    report.status = 'error';
    report.error = describeError(error);
    verdict.textContent = `Check failed: ${report.error}`;
    verdict.classList.add('is-warn');
    paintReport();
  }
}

void main();

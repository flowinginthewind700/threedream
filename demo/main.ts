/**
 * Browser demo entry.
 *
 * Two things have to stay honest here, and they shape the whole file:
 *
 *  1. The physics/learning code is *the real code*. `ThreeRenderer` mirrors body
 *     transforms out of the same backend the env steps; `Trainer.train()` is the
 *     same synchronous call that drives `npm run train` and CI. Nothing in this
 *     file special-cases the browser. What you watch learn here is what trains
 *     headless.
 *
 *  2. `train()` is synchronous, so calling it for thousands of episodes on the
 *     main thread would freeze the render loop. We chunk it: each frame we run one
 *     `train()` call sized to a whole number of policy updates (`episodesPerUpdate`
 *     per update), which keeps the UI responsive while the agent visibly improves.
 *     The speed button just changes how many updates land per frame.
 */

import * as THREE from 'three';
import {
  Activity,
  Bot,
  Boxes,
  Brain,
  Cpu,
  Gauge,
  Pause,
  Play,
  RotateCcw,
  Shuffle,
  Target,
  Zap,
  createElement,
  type IconNode,
} from 'lucide';

import { GaussianPolicy } from '@threedream/ai/policy.js';
import { Trainer } from '@threedream/ai/trainer.js';
import { Rng } from '@threedream/core/rng.js';
import { DriveEnv } from '@threedream/envs/drive.js';
import { ReachEnv } from '@threedream/envs/reach.js';
import type { LearningEnvironment } from '@threedream/envs/types.js';
import { ThreeRenderer } from '@threedream/render/scene.js';
import { vec3, type Vec3 } from '@threedream/physics/types.js';

// ---------------------------------------------------------------------------
// icons
// ---------------------------------------------------------------------------

const ICONS = {
  activity: Activity,
  bot: Bot,
  brain: Brain,
  cpu: Cpu,
  gauge: Gauge,
  pause: Pause,
  play: Play,
  'rotate-ccw': RotateCcw,
  shuffle: Shuffle,
  target: Target,
  zap: Zap,
} as const;

type IconName = keyof typeof ICONS;

function paintIcon(host: Element, name: IconName): void {
  host.replaceChildren(createElement(ICONS[name] as IconNode));
}

function paintAllIcons(root: ParentNode): void {
  for (const el of Array.from(root.querySelectorAll<HTMLElement>('[data-icon]'))) {
    const name = el.dataset['icon'] as IconName | undefined;
    if (name && name in ICONS) paintIcon(el, name);
  }
}

const brandLogo = document.getElementById('brand-logo');
if (brandLogo) paintIcon(brandLogo, 'cpu');
paintAllIcons(document);

// ---------------------------------------------------------------------------
// dom handles
// ---------------------------------------------------------------------------

const viewport = document.getElementById('viewport')!;
const loading = document.getElementById('loading')!;
const overlayBadge = document.getElementById('mode-badge')!;

const trainBtn = document.getElementById('train-btn') as HTMLButtonElement;
const speedBtn = document.getElementById('speed-btn') as HTMLButtonElement;
const pauseBtn = document.getElementById('pause-btn') as HTMLButtonElement;
const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;

const spark = document.getElementById('sparkline') as HTMLCanvasElement;
const sparkCtx = spark.getContext('2d')!;

const statEpisodes = document.getElementById('stat-episodes')!;
const statUpdates = document.getElementById('stat-updates')!;
const statSuccess = document.getElementById('stat-success')!;
const statReturn = document.getElementById('stat-return')!;
const statSigma = document.getElementById('stat-sigma')!;
const statFps = document.getElementById('stat-fps')!;

const taskButtons = Array.from(
  document.querySelectorAll<HTMLButtonElement>('#task-select .seg'),
);
const modeButtons = Array.from(
  document.querySelectorAll<HTMLButtonElement>('#mode-select .seg'),
);

/**
 * Read the initially selected tab out of the markup.
 *
 * The HTML already says which tab is active (`aria-selected="true"` plus the
 * `is-active` class). Re-declaring that fact in a `let` initialiser gives two
 * sources of truth for one state, and they drift silently: the Playback control
 * was showing "Learned" while the loop below was still sampling random actions,
 * because the markup selected `policy` and the variable initialised to `random`.
 * Nothing renders wrong and nothing throws -- the demo simply lies about what it
 * is doing, which is the failure mode a viewer cannot detect.
 */
function initialSelected<T extends string>(
  buttons: HTMLButtonElement[],
  dataKey: string,
): T {
  const active = buttons.find((b) => b.getAttribute('aria-selected') === 'true');
  if (!active?.dataset[dataKey]) {
    throw new Error(`no tab marked aria-selected for data-${dataKey}`);
  }
  return active.dataset[dataKey] as T;
}

// ---------------------------------------------------------------------------
// shared scene furniture
// ---------------------------------------------------------------------------

const COPPER = 0xdd8f3c;
const TEAL = 0x4fb3a5;
const ROSE = 0xd9705f;

function standardMaterial(color: number, roughness = 0.5, metalness = 0.1): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}

function makeFloor(size: number): THREE.Mesh {
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshStandardMaterial({ color: 0x171a1f, roughness: 0.95, metalness: 0 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.06;
  floor.receiveShadow = true;
  return floor;
}

function makeGrid(size: number, divisions: number): THREE.GridHelper {
  const grid = new THREE.GridHelper(size, divisions, 0x2b3038, 0x1f242b);
  grid.position.y = -0.055;
  const mat = grid.material as THREE.Material | THREE.Material[];
  for (const m of Array.isArray(mat) ? mat : [mat]) {
    m.transparent = true;
    m.opacity = 0.55;
  }
  return grid;
}

/** A flat goal marker: a copper ring with a faint disc, lying in the play plane. */
function makeGoalMarker(): THREE.Group {
  const g = new THREE.Group();
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.16, 0.018, 12, 40),
    standardMaterial(COPPER, 0.4, 0.3),
  );
  ring.rotation.x = -Math.PI / 2;
  g.add(ring);
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(0.16, 40),
    new THREE.MeshStandardMaterial({
      color: COPPER,
      transparent: true,
      opacity: 0.18,
      roughness: 0.6,
      side: THREE.DoubleSide,
    }),
  );
  disc.rotation.x = -Math.PI / 2;
  disc.position.y = 0.002;
  g.add(disc);
  return g;
}

function makeSphere(radius: number, color: number): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.SphereGeometry(radius, 24, 18), standardMaterial(color, 0.45, 0.15));
  m.castShadow = true;
  return m;
}

// ---------------------------------------------------------------------------
// task scene
// ---------------------------------------------------------------------------

interface TaskScene {
  renderer: ThreeRenderer;
  goalMarker: THREE.Group;
  /** The live env this scene renders; stepped by playback and training alike. */
  env: LearningEnvironment;
  /** Read goal position out of the env for the current task. */
  readGoal(): Vec3;
  /** Read whether the current episode reached the goal. */
  readReached(): boolean;
  dispose(): void;
}

function buildDrive(): TaskScene {
  const env = new DriveEnv({ seed: 11 });
  const renderer = new ThreeRenderer({
    container: viewport,
    backend: env.backend,
    background: 0x0e1013,
    shadows: true,
  });
  renderer.camera.position.set(0, 2.6, 2.2);
  renderer.camera.lookAt(0, 0, 0);

  renderer.add(makeFloor(3.2));
  renderer.add(makeGrid(3, 12));

  const agent = makeSphere(0.08, TEAL);
  renderer.bind({ handle: env.handle, object: agent });

  const goalMarker = makeGoalMarker();
  renderer.add(goalMarker);

  const scene: TaskScene = {
    renderer,
    goalMarker,
    env,
    readGoal: () => env.diagnostics().goalPosition,
    readReached: () => env.diagnostics().reached,
    dispose: () => {
      renderer.dispose();
      env.dispose();
    },
  };
  return scene;
}

function buildReach(): TaskScene {
  const env = new ReachEnv({ seed: 7 });
  const renderer = new ThreeRenderer({
    container: viewport,
    backend: env.backend,
    background: 0x0e1013,
    shadows: true,
  });
  renderer.camera.position.set(0, 2.5, 2.1);
  renderer.camera.lookAt(0, 0, 0);

  renderer.add(makeFloor(2.8));
  renderer.add(makeGrid(2.4, 12));

  // The four arena walls are static bodies; draw them so the table reads as a
  // table. Their geometry mirrors ReachEnv.buildStaticGeometry() (half=1, t=0.05).
  const wallMat = standardMaterial(0x2a2f37, 0.8, 0.05);
  const h = 1.0;
  const t = 0.05;
  const wallSpecs: { pos: [number, number, number]; ext: [number, number, number] }[] = [
    { pos: [0, 0, -h - t], ext: [h + t, t, t] },
    { pos: [0, 0, h + t], ext: [h + t, t, t] },
    { pos: [-h - t, 0, 0], ext: [t, t, h + t] },
    { pos: [h + t, 0, 0], ext: [t, t, h + t] },
  ];
  for (const w of wallSpecs) {
    const wall = new THREE.Mesh(
      new THREE.BoxGeometry(w.ext[0] * 2, 0.12, w.ext[2] * 2),
      wallMat,
    );
    wall.position.set(w.pos[0], 0, w.pos[2]);
    wall.receiveShadow = true;
    renderer.add(wall);
  }

  const agent = makeSphere(0.09, TEAL);
  const puck = makeSphere(0.08, ROSE);
  renderer.bind({ handle: env.handles.agent, object: agent });
  renderer.bind({ handle: env.handles.puck, object: puck });

  const goalMarker = makeGoalMarker();
  renderer.add(goalMarker);

  const scene: TaskScene = {
    renderer,
    goalMarker,
    env,
    readGoal: () => env.diagnostics().goalPosition,
    readReached: () => env.diagnostics().reached,
    dispose: () => {
      renderer.dispose();
      env.dispose();
    },
  };
  return scene;
}

// ---------------------------------------------------------------------------
// runtime state
// ---------------------------------------------------------------------------

type TaskName = 'drive' | 'reach';
type Mode = 'policy' | 'random';

const SPEED_STEPS = [1, 2, 4, 8];
let speedIndex = 0;

let taskName: TaskName = initialSelected<TaskName>(taskButtons, 'task');
let mode: Mode = initialSelected<Mode>(modeButtons, 'mode');
let training = false;
let paused = false;

let scene = buildDrive();
let env = scene.env;

function policyConfigFor(name: TaskName): {
  hiddenSizes: number[];
  learningRate: number;
  criticLr: number;
  entropyCoefficient: number;
  episodesPerUpdate: number;
  initialLogSigma: number;
  trainerSeed: number;
  policySeed: number;
} {
  // These match the budgets in tests/trainer.test.ts, which are the measured
  // "this stack still learns" regression configs.
  return name === 'drive'
    ? {
        hiddenSizes: [64, 64],
        learningRate: 0.05,
        criticLr: 0.1,
        entropyCoefficient: 0,
        episodesPerUpdate: 16,
        initialLogSigma: -0.7,
        trainerSeed: 78,
        policySeed: 78,
      }
    : {
        hiddenSizes: [64, 64],
        learningRate: 0.02,
        criticLr: 0.04,
        entropyCoefficient: 0.001,
        episodesPerUpdate: 16,
        initialLogSigma: -0.5,
        trainerSeed: 7,
        policySeed: 7,
      };
}

let cfg = policyConfigFor(taskName);
let policy = newPolicy();
let trainer = newTrainer();

function newPolicy(): GaussianPolicy {
  return new GaussianPolicy({
    observationSize: env.observationSize,
    actionSize: env.actionSize,
    hiddenSizes: cfg.hiddenSizes,
    initialLogSigma: cfg.initialLogSigma,
    seed: cfg.policySeed,
  });
}

function newTrainer(): Trainer {
  return new Trainer({
    learningRate: cfg.learningRate,
    criticLr: cfg.criticLr,
    entropyCoefficient: cfg.entropyCoefficient,
    episodesPerUpdate: cfg.episodesPerUpdate,
    seed: cfg.trainerSeed,
  });
}

// training bookkeeping
let episodesTrained = 0;
let updates = 0;
const returnSeries: number[] = [];
const reachedWindow: boolean[] = [];
const WINDOW = 40;
let lastBatchReturn: number | null = null;

// fps bookkeeping
let fpsAccum = 0;
let fpsFrames = 0;
let fpsValue = 0;
let lastFrameMs = performance.now();

// scratch buffers reused every step
const actionBuffer = new Float32Array(2);
let observation: Float32Array;
let episodeDone = false;

const evalRng = new Rng(0xfeed);

function resetEpisode(): void {
  observation = env.reset(evalRng);
  episodeDone = false;
  const goal = scene.readGoal();
  scene.goalMarker.position.set(goal[0], goal[1], goal[2]);
}

function rebuildForTask(name: TaskName): void {
  scene.dispose();
  scene = name === 'drive' ? buildDrive() : buildReach();
  env = (scene as unknown as { env: LearningEnvironment }).env;
  cfg = policyConfigFor(name);
  policy = newPolicy();
  trainer = newTrainer();

  episodesTrained = 0;
  updates = 0;
  returnSeries.length = 0;
  reachedWindow.length = 0;
  lastBatchReturn = null;

  resetEpisode();
  drawSpark();
  paintStats();
}

// ---------------------------------------------------------------------------
// playback step: advance the simulation by one env step under the live policy
// ---------------------------------------------------------------------------

function stepPlayback(): void {
  if (episodeDone) {
    resetEpisode();
    return;
  }
  let action: ArrayLike<number>;
  if (mode === 'random') {
    actionBuffer[0] = evalRng.range(-1, 1);
    actionBuffer[1] = evalRng.range(-1, 1);
    action = actionBuffer;
  } else {
    action = policy.actGreedy(observation);
  }
  const result = env.step(action);
  episodeDone = result.done;
  if (episodeDone) {
    reachedWindow.push(scene.readReached());
    if (reachedWindow.length > WINDOW) reachedWindow.shift();
  }
  observation = env.observe(observation);
}

// ---------------------------------------------------------------------------
// training chunk: one train() call == a whole number of policy updates
// ---------------------------------------------------------------------------

function trainChunk(): void {
  const updatesPerFrame = SPEED_STEPS[speedIndex]!;
  const episodes = updatesPerFrame * cfg.episodesPerUpdate;
  const before = updates;
  trainer.train(policy, env, episodes, (r) => {
    episodesTrained++;
    returnSeries.push(r.batchMeanReturn);
    if (returnSeries.length > 240) returnSeries.shift();
    lastBatchReturn = r.batchMeanReturn;
  });
  const after = Math.floor(episodesTrained / cfg.episodesPerUpdate);
  if (after > before) updates = after;
  // Training resets the env repeatedly; restore a clean playback episode so the
  // viewer does not inherit a mid-rollout state.
  resetEpisode();
  drawSpark();
  paintStats();
}

// ---------------------------------------------------------------------------
// sparkline
// ---------------------------------------------------------------------------

function drawSpark(): void {
  const w = spark.width;
  const h = spark.height;
  sparkCtx.clearRect(0, 0, w, h);
  // background grid line at zero-ish baseline
  sparkCtx.strokeStyle = '#232830';
  sparkCtx.lineWidth = 1;
  sparkCtx.beginPath();
  sparkCtx.moveTo(0, h - 0.5);
  sparkCtx.lineTo(w, h - 0.5);
  sparkCtx.stroke();

  if (returnSeries.length < 2) {
    sparkCtx.fillStyle = '#6d7480';
    sparkCtx.font = '10px ui-monospace, monospace';
    sparkCtx.fillText('no data yet', 8, h / 2 + 3);
    return;
  }
  let min = Infinity;
  let max = -Infinity;
  for (const v of returnSeries) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!isFinite(min) || !isFinite(max)) return;
  const span = max - min || 1;
  const pad = 6;
  const n = returnSeries.length;
  const xFor = (i: number): number => (i / (n - 1)) * (w - pad * 2) + pad;
  const yFor = (v: number): number => h - pad - ((v - min) / span) * (h - pad * 2);

  sparkCtx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xFor(i);
    const y = yFor(returnSeries[i]!);
    if (i === 0) sparkCtx.moveTo(x, y);
    else sparkCtx.lineTo(x, y);
  }
  sparkCtx.strokeStyle = '#dd8f3c';
  sparkCtx.lineWidth = 1.5;
  sparkCtx.stroke();

  // fill under the curve, faint
  sparkCtx.lineTo(xFor(n - 1), h - pad);
  sparkCtx.lineTo(xFor(0), h - pad);
  sparkCtx.closePath();
  sparkCtx.fillStyle = 'rgba(221,143,60,0.10)';
  sparkCtx.fill();
}

// ---------------------------------------------------------------------------
// stats panel
// ---------------------------------------------------------------------------

function paintStats(): void {
  statEpisodes.textContent = String(episodesTrained);
  statUpdates.textContent = String(updates);

  if (reachedWindow.length === 0) {
    statSuccess.textContent = '\u2014';
    statSuccess.className = '';
  } else {
    const wins = reachedWindow.filter(Boolean).length;
    const pct = Math.round((wins / reachedWindow.length) * 100);
    statSuccess.textContent = `${pct}%`;
    statSuccess.className = pct >= 60 ? 'is-good' : pct > 0 ? 'is-warn' : '';
  }

  statReturn.textContent = lastBatchReturn === null ? '\u2014' : lastBatchReturn.toFixed(2);

  const sig = Math.exp(policy.logSigmas[0] ?? 0);
  statSigma.textContent = sig.toFixed(3);

  statFps.textContent = fpsValue > 0 ? fpsValue.toFixed(0) : '\u2014';

  // overlay badge reflects what is actually driving the agent right now
  const reached = scene.readReached();
  overlayBadge.className = 'badge' + (reached ? ' is-won' : mode === 'policy' ? ' is-learned' : '');
  overlayBadge.textContent = reached
    ? 'goal reached'
    : mode === 'policy'
      ? episodesTrained > 0
        ? `learned policy - ${episodesTrained} eps`
        : 'learned policy (untrained)'
      : 'random policy';
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------

function setActive(buttons: HTMLButtonElement[], active: HTMLButtonElement): void {
  for (const b of buttons) {
    const on = b === active;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  }
}

/**
 * Paint the tab that matches the state we already hold, rather than the tab at
 * some fixed index. Booting with `buttons[0]` looks harmless and is not: the
 * moment the markup's selected tab stops being the first child, boot silently
 * overrides what the markup declared and the two sources diverge again.
 */
function syncActive<T extends string>(
  buttons: HTMLButtonElement[],
  dataKey: string,
  value: T,
): void {
  const match = buttons.find((b) => b.dataset[dataKey] === value);
  if (match) setActive(buttons, match);
}

for (const b of taskButtons) {
  b.addEventListener('click', () => {
    const name = b.dataset['task'] as TaskName;
    if (name === taskName) return;
    setActive(taskButtons, b);
    taskName = name;
    rebuildForTask(name);
  });
}

for (const b of modeButtons) {
  b.addEventListener('click', () => {
    const m = b.dataset['mode'] as Mode;
    if (m === mode) return;
    setActive(modeButtons, b);
    mode = m;
    resetEpisode();
    paintStats();
  });
}

trainBtn.addEventListener('click', () => {
  training = !training;
  const label = trainBtn.querySelector('.btn-label')!;
  if (training) {
    label.textContent = 'Stop';
    trainBtn.classList.add('is-busy');
    paintIcon(trainBtn.querySelector('[data-icon]')!, 'pause');
  } else {
    label.textContent = 'Train';
    trainBtn.classList.remove('is-busy');
    paintIcon(trainBtn.querySelector('[data-icon]')!, 'play');
  }
});

speedBtn.addEventListener('click', () => {
  speedIndex = (speedIndex + 1) % SPEED_STEPS.length;
  const label = speedBtn.querySelector('.btn-label')!;
  label.textContent = `${SPEED_STEPS[speedIndex]}x`;
});

function setPaused(next: boolean): void {
  paused = next;
  const label = pauseBtn.querySelector('.btn-label')!;
  label.textContent = paused ? 'Resume' : 'Pause';
  paintIcon(pauseBtn.querySelector('[data-icon]')!, paused ? 'play' : 'pause');
}

pauseBtn.addEventListener('click', () => setPaused(!paused));

resetBtn.addEventListener('click', () => {
  resetEpisode();
  paintStats();
});

// ---------------------------------------------------------------------------
// main loop
// ---------------------------------------------------------------------------

function loop(now: number): void {
  const dtMs = now - lastFrameMs;
  lastFrameMs = now;
  fpsAccum += dtMs;
  fpsFrames++;
  if (fpsAccum >= 500) {
    fpsValue = (fpsFrames / fpsAccum) * 1000;
    fpsAccum = 0;
    fpsFrames = 0;
    statFps.textContent = fpsValue.toFixed(0);
  }

  if (!paused) {
    if (training) {
      trainChunk();
    }
    // Advance playback one env step per animation frame; the env is a 1/60
    // fixed-step sim and rAF is ~60Hz, so this is real-time play. When training
    // we still step once so the viewer tracks the freshly updated policy.
    stepPlayback();
    const goal = scene.readGoal();
    scene.goalMarker.position.set(goal[0], goal[1], goal[2]);
    scene.renderer.frame();
    paintStats();
  } else {
    scene.renderer.render();
  }

  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

resetEpisode();
drawSpark();
paintStats();
syncActive(taskButtons, 'task', taskName);
syncActive(modeButtons, 'mode', mode);
requestAnimationFrame(() => {
  loading.classList.add('is-hidden');
});
requestAnimationFrame(loop);

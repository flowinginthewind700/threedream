/**
 * Browser spec for the M4 soft-body layer on the tier that actually matters.
 *
 * `soft.spec.ts` owns the fallback tiers; this file owns the claims the milestone
 * exists to make:
 *
 *   - one `GPUDevice`, shared between the solve and the renderer -- the M2
 *     architecture claim, re-asserted where it pays for itself. `deviceShared` is
 *     `renderer.backend.device === shared.device`, not a proxy for it.
 *   - a *race-free* solve. `raceFree` is the engine's own flag, and it is true
 *     because the constraint graph is coloured and one colour is dispatched at a
 *     time, so no two invocations of one dispatch write the same node. The visible
 *     consequence is the one this file pins: the same seed on the same device gives
 *     the same bytes twice.
 *   - positions reach the screen without crossing the bus. `frameMode ===
 *     'gpu-blit'` means no readback fed the frame, and `blitBytes` is the publish
 *     buffer that was copied into the `GPUBuffer` three.js already owned.
 *   - the WGSL kernels agree with the CPU reference, and the plan agrees exactly.
 *     Tolerances on the numbers, equality on the graph: both tiers build the plan
 *     with `buildSoftLayout`, so islands, colours and workgroup counts are not
 *     allowed to differ, while fma contraction and adapter-specific `sqrt`
 *     rounding are.
 *
 * Runs in the `chromium-webgpu` project only (see playwright.config.ts). Every test
 * skips itself when the runner cannot produce an adapter, and the second describe is
 * the mirror image: it only runs when the adapter is missing, so a GPU-less runner
 * still gets real assertions out of this file.
 */

import { expect, test, type Page } from '@playwright/test';

import type { SoftReport } from '../demo/soft.js';
import { SOFT_WORKGROUP_SIZE } from '../src/gpu/softIslands.js';
import { SOFT_FIXED_DISPATCHES } from '../src/gpu/softOptions.js';
import { SOFT_PUBLISH_FLOATS_PER_NODE } from '../src/gpu/softWgsl.js';

import { distinctColors } from './pixels.js';

/** The demo serves from a subpath when built for Pages; resolve either way. */
const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173/threedream';
const PAGE_URL = `${BASE}/soft.html`;

/** Deep enough that the workgroup mapping matters, cheap enough to run twice. */
const COUNT = 2500;
const STEPS = 60;
const SEED = 1234;
/** The acceptance criterion: GPU 层能稳定处理 10,000 级别的软体或布料粒子. */
const LARGE_COUNT = 10_000;

/** Bytes the publish buffer holds per node: three f32 positions. */
const PUBLISH_BYTES_PER_NODE = SOFT_PUBLISH_FLOATS_PER_NODE * 4;

/**
 * `hex:count` -- the shape `SoftMesh.digest()` returns.
 *
 * The count rides along, so a digest can never be mistaken for one from a mesh of a
 * different size. The *value* is pinned exactly on the CPU tier in `soft.spec.ts`;
 * here it is asserted well-formed and stable on one device, which is all `raceFree`
 * promises, because WGSL leaves fma contraction and `sqrt` rounding to the adapter.
 */
const DIGEST = /^[0-9a-f]{8,}:[0-9]+$/;

/**
 * Collect page and console errors for the life of the page.
 *
 * Idempotent, because `open()` attaches it and a test that probes the adapter first
 * has already been handed the array. Registering twice would report every error
 * twice, which reads like two failures where there is one.
 */
function watchErrors(page: Page): string[] {
  const bag = page as unknown as { __errors?: string[] };
  if (bag.__errors) return bag.__errors;
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console: ${m.text()}`);
  });
  bag.__errors = errors;
  return errors;
}

function errorsOf(page: Page): string[] {
  return (page as unknown as { __errors?: string[] }).__errors ?? [];
}

async function report(page: Page): Promise<SoftReport> {
  const value = await page.evaluate(() => (window as unknown as { __soft?: SoftReport }).__soft);
  expect(value, 'demo/soft.ts must publish window.__soft').toBeDefined();
  return value as SoftReport;
}

/** The whole report plus the page log: the only useful context on a failure. */
function dump(r: SoftReport): string {
  const { lines, ...rest } = r;
  return `${JSON.stringify(rest, null, 2)}\nlog:\n  ${lines.join('\n  ')}`;
}

/**
 * Load the page and wait for the build to settle.
 *
 * `status !== 'booting'` rather than the wanted value, so an `error` outcome reaches
 * the assertions with its message attached instead of becoming a timeout that says
 * nothing about why. The default is generous because the GPU project compiles the
 * soft-body pipelines through ANGLE before the first frame.
 */
async function open(page: Page, query: string, timeout = 120_000): Promise<SoftReport> {
  watchErrors(page);
  await page.goto(`${PAGE_URL}${query}`, { waitUntil: 'load' });
  await page.waitForFunction(
    () => {
      const r = (window as unknown as { __soft?: { status: string } }).__soft;
      return !!r && r.status !== 'booting';
    },
    undefined,
    { timeout },
  );
  return report(page);
}

/**
 * Whether this runner has an adapter, read off the page's own report.
 *
 * Probing `navigator.gpu` from the spec instead would answer "no" everywhere: a
 * freshly launched Playwright page is `about:blank`, and its opaque origin is not a
 * secure context, so `navigator.gpu` does not exist there. That skips the GPU claims
 * on a machine that has a GPU and runs the no-adapter mirror on it -- both wrong,
 * and both quietly.
 */
function hasAdapter(r: SoftReport): boolean {
  return r.webgpuAvailable;
}

/** Relative difference, for two numbers that should agree within a tolerance. */
function rel(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(1e-9, (Math.abs(a) + Math.abs(b)) / 2);
}

test.describe('one device, one colour-batched solve, no round trip', () => {
  test('the blit path runs entirely on the shared device', async ({ page }) => {
    // `strict=1` so a silent downgrade fails the test instead of passing it with the
    // CPU path -- the whole point of this file.
    const r = await open(
      page,
      `?tier=webgpu&strict=1&scene=cloth&count=${COUNT}&seed=${SEED}&steps=${STEPS}`,
    );
    // No adapter is an environment fact rather than a regression, and the mirror
    // describe below owns that case. Everything from here assumes hardware.
    test.skip(!hasAdapter(r), `no WebGPU adapter here: ${r.reason}`);

    expect(r.status, `the scripted run did not finish: ${dump(r)}`).toBe('done');
    expect(r.error, dump(r)).toBeUndefined();
    expect(r.tier, dump(r)).toBe('webgpu');
    expect(r.backend, `three.js did not draw with WebGPU: ${dump(r)}`).toBe('webgpu');
    expect(r.gpuError, `a GPU pass failed and the view fell back: ${dump(r)}`).toBeNull();

    // M2's claim, where it earns its keep: the renderer is drawing with the very
    // device the solve was dispatched on.
    expect(r.deviceShared, `renderer.backend.device is not the acquired device: ${dump(r)}`).toBe(
      true,
    );
    // M4's claim, read off the backend rather than off the tier name.
    expect(r.raceFree, `the GPU system does not state a race-free solve: ${dump(r)}`).toBe(true);
    expect(r.viewMode, dump(r)).toBe('gpu');
    expect(r.frameMode, `the frame was uploaded from CPU copies: ${dump(r)}`).toBe('gpu-blit');
    expect(r.blitBytes, `expected the publish buffer for ${COUNT} nodes: ${dump(r)}`).toBe(
      COUNT * PUBLISH_BYTES_PER_NODE,
    );

    expect(r.count, dump(r)).toBe(COUNT);
    expect(r.steps, dump(r)).toBe(STEPS);
    expect(r.msPerStep, dump(r)).toBeGreaterThan(0);
    expect(r.msPerStep, dump(r)).toBeLessThan(Number.POSITIVE_INFINITY);

    // The plan, on the device: one island for a cloth, more than one colour because
    // adjacent edges share a node, and a workgroup count that is the ceiling at 64.
    expect(r.plan.nodes, dump(r)).toBe(COUNT);
    expect(r.plan.islands, dump(r)).toBe(1);
    expect(r.plan.colors, `adjacent edges landed in one batch: ${dump(r)}`).toBeGreaterThan(1);
    expect(r.plan.nodeWorkgroups, dump(r)).toBe(Math.ceil(COUNT / SOFT_WORKGROUP_SIZE));
    expect(r.plan.dispatchesPerStep, dump(r)).toBe(
      SOFT_FIXED_DISPATCHES + r.plan.iterations * r.plan.colors,
    );

    // One surface, one wire overlay, and the scene's bounds helper: a handful of draw
    // calls, not one per node.
    expect(r.drawCalls, `the mesh was not drawn in a handful of calls: ${dump(r)}`).toBeGreaterThan(
      0,
    );
    expect(r.drawCalls, dump(r)).toBeLessThanOrEqual(4);
    expect(r.triangles, dump(r)).toBeGreaterThan(0);

    // Physics invariants: nothing left the box, and the mesh is doing something.
    // `maxConstraintError` is relative, so a bound on it is a stability claim rather
    // than a stiffness one -- see the footnote beside the iterations slider.
    expect(r.stats.escaped, dump(r)).toBe(0);
    expect(r.stats.maxSpeed, dump(r)).toBeGreaterThan(0);
    expect(r.stats.kineticEnergy, dump(r)).toBeGreaterThan(0);
    expect(r.digest, dump(r)).toMatch(DIGEST);
    expect(r.canvasBytes, 'the final frame encoded to nothing').toBeGreaterThan(0);
    expect(errorsOf(page), `page or console errors: ${dump(r)}`).toEqual([]);
  });

  test('the live page keeps blitting and keeps moving', async ({ page }) => {
    // No `steps`: the real loop, which is what a visitor gets.
    const r = await open(page, `?tier=webgpu&scene=cloth&count=${COUNT}&seed=${SEED}`);
    test.skip(!hasAdapter(r), `no WebGPU adapter here: ${r.reason}`);
    expect(r.status, dump(r)).toBe('live');
    expect(r.deviceShared, dump(r)).toBe(true);

    // The first frame may be served before the publish buffer exists to copy from;
    // the blit path has to engage, not merely be intended.
    await expect
      .poll(async () => (await report(page)).frameMode, {
        timeout: 30_000,
        message: 'the blit path never engaged',
      })
      .toBe('gpu-blit');

    const live = await report(page);
    expect(live.viewMode, dump(live)).toBe('gpu');
    expect(live.raceFree, dump(live)).toBe(true);
    expect(live.gpuError, dump(live)).toBeNull();
    expect(live.steps, dump(live)).toBeGreaterThan(0);
    expect(live.blitBytes, dump(live)).toBe(COUNT * PUBLISH_BYTES_PER_NODE);

    const canvas = page.locator('#viewport canvas');
    await expect(canvas).toBeVisible();
    await expect(page.locator('#loading')).toBeHidden();
    const unique = await distinctColors(page, await canvas.screenshot());
    expect(unique, `the viewport is one flat colour: ${dump(live)}`).toBeGreaterThan(8);

    // Motion, not a still image that happens to be detailed: two frames 400ms apart
    // must differ. This is also the only check that catches a loop that rendered once
    // and stopped.
    const before = await canvas.screenshot();
    await page.waitForTimeout(400);
    const after = await canvas.screenshot();
    expect(after.equals(before), `the viewport did not change in 400ms: ${dump(live)}`).toBe(false);

    // What the page claims about itself has to match what it did.
    const device = page.locator('#ti-device');
    await expect(device).toHaveText('OK');
    await expect(device).toHaveClass(/is-good/);
    const race = page.locator('#ti-race');
    await expect(race).toHaveText('OK');
    await expect(race).toHaveClass(/is-good/);
    const path = page.locator('#ti-path');
    await expect(path).toHaveText('gpu-blit');
    await expect(path).toHaveClass(/is-good/);
    await expect(page.locator('#path-badge')).toHaveClass(/is-learned/);
    await expect(page.locator('#gr-wgsize')).toHaveText(String(SOFT_WORKGROUP_SIZE));
    await expect(page.locator('#verdict')).not.toHaveClass(/is-warn/);
    expect(errorsOf(page), `page or console errors: ${dump(live)}`).toEqual([]);
  });

  test('the WGSL kernels agree with the CPU reference', async ({ page }) => {
    // A converged case. 900 nodes is a 30x30 cloth, and 16 sweeps is enough for a
    // correction to travel from the pinned row to the middle of it, because
    // Gauss-Seidel reaches about one row per sweep. Comparing the two tiers on a
    // mesh that is still falling apart would be comparing noise.
    const n = 900;
    const steps = 60;
    const base = `scene=cloth&count=${n}&seed=${SEED}&steps=${steps}&iterations=16`;

    // One throwaway boot to answer the adapter question before spending minutes on
    // the comparison. It asks for a device, because `webgpuAvailable` is only
    // meaningful on a run that tried to get one.
    const probe = await open(page, '?tier=webgpu&strict=1&count=64&steps=1', 60_000);
    test.skip(!hasAdapter(probe), `no WebGPU adapter here: ${probe.reason}`);

    const cpu = await open(page, `?tier=cpu&${base}`);
    const gpu = await open(page, `?tier=webgpu&strict=1&${base}`);

    expect(cpu.status, `the CPU reference did not finish: ${dump(cpu)}`).toBe('done');
    expect(gpu.status, `the GPU tier did not finish: ${dump(gpu)}`).toBe('done');
    expect(gpu.tier, dump(gpu)).toBe('webgpu');
    expect(gpu.raceFree, dump(gpu)).toBe(true);
    expect(cpu.stats.escaped, dump(cpu)).toBe(0);
    expect(gpu.stats.escaped, dump(gpu)).toBe(0);
    expect(gpu.steps, dump(gpu)).toBe(cpu.steps);

    // The graph is exact. Both tiers get it from `buildSoftLayout`, so a difference
    // here is a bug in one of them and not a rounding story.
    expect(
      gpu.plan,
      `the device disagrees with the reference about the graph: ${dump(gpu)} / ${dump(cpu)}`,
    ).toEqual(cpu.plan);

    // The numbers are close, not equal: fma contraction and `sqrt` rounding are
    // adapter-specific, which is why `deterministic` is false on the GPU system and
    // true on the reference.
    expect(
      rel(gpu.stats.kineticEnergy, cpu.stats.kineticEnergy),
      `kinetic energy diverged: gpu=${gpu.stats.kineticEnergy} cpu=${cpu.stats.kineticEnergy}`,
    ).toBeLessThan(0.05);
    expect(
      rel(gpu.stats.maxSpeed, cpu.stats.maxSpeed),
      `speed envelope diverged: gpu=${gpu.stats.maxSpeed} cpu=${cpu.stats.maxSpeed}`,
    ).toBeLessThan(0.1);
    // A worst-edge error is a max over thousands of edges, so it is the least stable
    // number in the struct. The slack is absolute as well as relative, because two
    // runs that both converge towards zero can differ by a large ratio while
    // agreeing to within a few percent of stretch.
    expect(
      Math.abs(gpu.stats.maxConstraintError - cpu.stats.maxConstraintError),
      `constraint error diverged: gpu=${gpu.stats.maxConstraintError} cpu=${cpu.stats.maxConstraintError}`,
    ).toBeLessThan(0.1 + 0.25 * cpu.stats.maxConstraintError);
    expect(gpu.stats.awakeIslands + gpu.stats.sleepingIslands, dump(gpu)).toBe(gpu.plan.islands);

    // Deliberately not comparing digests across tiers. Both sides produce one and
    // both are stable run to run on their own backend, but bit equality between an
    // f32 GPU pipeline and an f32 CPU reference is a promise the hardware never
    // made. The digest is pinned exactly on the CPU tier in `soft.spec.ts`, which is
    // where determinism actually lives.
    expect(cpu.digest, dump(cpu)).toMatch(DIGEST);
    expect(gpu.digest, dump(gpu)).toMatch(DIGEST);
   expect(errorsOf(page), dump(gpu)).toEqual([]);
 });

  test('the same seed on the same device is the same bytes twice', async ({ page }) => {
    // This is what `raceFree` buys, and it is the claim a data race would break
    // first: two invocations writing one node would leave the result depending on
    // which won, so the digest would move between runs of the same seed on the same
    // device. Colour batching makes that impossible, and the reduction passes use
    // `atomicMax`, which is order-independent by definition.
    const query = `?tier=webgpu&strict=1&scene=sheets&count=${COUNT}&seed=${SEED}&steps=${STEPS}`;
    const first = await open(page, query);
    test.skip(!hasAdapter(first), `no WebGPU adapter here: ${first.reason}`);
    expect(first.status, `the first run did not finish: ${dump(first)}`).toBe('done');

    const second = await open(page, query);
    expect(second.status, `the second run did not finish: ${dump(second)}`).toBe('done');
    expect(second.plan, dump(second)).toEqual(first.plan);
    expect(second.digest, `the device solve is not reproducible: ${dump(second)}`).toBe(
      first.digest,
    );
    expect(second.stats, dump(second)).toEqual(first.stats);
    expect(second.stats.escaped, dump(second)).toBe(0);
    // `sheets` on the device: the island mapping is what this run is really for, and
    // a grouping pass with one group would prove nothing about it.
    expect(first.plan.islands, dump(first)).toBeGreaterThan(1);
    expect(first.plan.nodeWorkgroups, dump(first)).toBeGreaterThanOrEqual(
      Math.ceil(COUNT / SOFT_WORKGROUP_SIZE),
    );
   expect(errorsOf(page), dump(second)).toEqual([]);
 });

  test('10,000 nodes hold up on the device', async ({ page }) => {
    // The acceptance criterion, run for real. Fewer steps than the small case,
    // because the budget here is allocation, pipeline setup and the plan's two
    // passes over a 40k-edge graph.
    test.setTimeout(300_000);

    const r = await open(
      page,
      `?tier=webgpu&strict=1&scene=cloth&count=${LARGE_COUNT}&seed=${SEED}&steps=20`,
      240_000,
    );
    test.skip(!hasAdapter(r), `no WebGPU adapter here: ${r.reason}`);

    expect(r.status, `10k did not finish: ${dump(r)}`).toBe('done');
    expect(r.error, dump(r)).toBeUndefined();
    expect(r.tier, dump(r)).toBe('webgpu');
    expect(r.count, dump(r)).toBe(LARGE_COUNT);
    expect(r.plan.nodes, dump(r)).toBe(LARGE_COUNT);
    expect(r.steps, dump(r)).toBe(20);
    expect(r.frameMode, dump(r)).toBe('gpu-blit');
    expect(r.blitBytes, dump(r)).toBe(LARGE_COUNT * PUBLISH_BYTES_PER_NODE);
    expect(r.plan.nodeWorkgroups, dump(r)).toBe(Math.ceil(LARGE_COUNT / SOFT_WORKGROUP_SIZE));
    expect(r.stats.escaped, `a node left the box at 10k: ${dump(r)}`).toBe(0);
    expect(r.drawCalls, dump(r)).toBeLessThanOrEqual(4);
    expect(r.triangles, dump(r)).toBeGreaterThan(0);
    expect(r.canvasBytes, 'the final frame encoded to nothing').toBeGreaterThan(0);
    expect(errorsOf(page), dump(r)).toEqual([]);
  });
});

test.describe('without a WebGPU adapter', () => {
  test('auto downgrades and still draws', async ({ page }) => {
    // The acceptance criterion, in the environment that exercises it: no tier forced,
    // nothing available, and a page that must still put a cloth on the screen.
    const r = await open(page, `?scene=cloth&count=${COUNT}&seed=${SEED}`);
    test.skip(hasAdapter(r), 'this runner has an adapter; see the describe above');

    expect(r.status, `the page did not come up without a GPU: ${dump(r)}`).toBe('live');
    expect(r.error, dump(r)).toBeUndefined();
    expect(['webgl2', 'cpu'], `unexpected tier: ${dump(r)}`).toContain(r.tier);
    expect(r.deviceShared, dump(r)).toBe(false);
    // Race-freedom is a property of a parallel solve; the reference walks its colour
    // batches in order, so the row must say "sequential solver" rather than claim OK.
    expect(r.raceFree, dump(r)).toBe(false);
    expect(r.frameMode, dump(r)).toBe('cpu-upload');
    expect(r.backend, dump(r)).not.toBe('webgpu');

    await expect(page.locator('#loading')).toBeHidden();
    const canvas = page.locator('#viewport canvas');
    await expect(canvas).toBeVisible();
    const unique = await distinctColors(page, await canvas.screenshot());
    expect(unique, `the downgraded tier drew nothing: ${dump(r)}`).toBeGreaterThan(8);

    // The page says why, in the log a visitor reads. A downgrade that happened
    // silently would leave the operator guessing which tier they are on.
    const log = await page.locator('#log').innerText();
    expect(log, `the downgrade was not explained: ${dump(r)}`).toMatch(/no WebGPU device/i);
    await expect(page.locator('#ti-race')).toHaveText('sequential solver');
    await expect(page.locator('#ti-device')).toHaveText('n/a (WebGL renderer)');
    expect(errorsOf(page), dump(r)).toEqual([]);
  });

  test('strict refuses instead of downgrading', async ({ page }) => {
    // `strict=1` is the escape hatch for a caller that would rather fail loudly than
    // silently solve on the CPU -- a benchmark, or a training run whose numbers must
    // be comparable across machines. Refusing is the correct behaviour, and it has to
    // be visible on the page rather than only in the console.
    const r = await open(page, `?tier=webgpu&strict=1&count=${COUNT}&steps=10`);
    test.skip(hasAdapter(r), 'this runner has an adapter; see the describe above');

    expect(r.status, dump(r)).toBe('error');
    expect(r.error, `no reason was reported: ${dump(r)}`).toMatch(/cannot be honoured/i);
    await expect(page.locator('#tier-badge')).toHaveText('failed');
    await expect(page.locator('#loading')).toBeVisible();
    await expect(page.locator('#verdict')).toHaveClass(/is-warn/);
  });
});

/**
 * Browser spec for the M3 particle layer, on the tiers that need no GPU.
 *
 * `particles_gpu.spec.ts` owns the WebGPU claims; this file owns the other half
 * of the acceptance criteria, and it is the half that is easy to believe without
 * ever testing:
 *
 *   - "WebGPU 不可用时自动降级到 WebGL2 或 CPU" -- a page with no adapter must
 *     still draw. Not a blank canvas, not a thrown error: pixels.
 *   - "回放与训练不依赖 GPU 层" -- the CPU tier is the deterministic reference,
 *     so the same seed and step count must produce the same digest twice.
 *   - "渲染帧率与仿真步长解耦" -- over a second of wall clock the simulation
 *     advances by `1/fixedDt` steps, whatever the display managed to draw.
 *
 * Runs in the `chromium` project (SwiftShader), which has no WebGPU adapter and
 * is therefore exactly the environment the fallback tiers exist for. The tier is
 * pinned by URL rather than left to the probe, so a runner that does have an
 * adapter still tests the tier under the same name.
 */

import { expect, test, type Page } from '@playwright/test';

import type { ParticlesReport } from '../demo/particles.js';

import { distinctColors } from './pixels.js';

/** The demo serves from a subpath when built for Pages; resolve either way. */
const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173/threedream';
const PAGE_URL = `${BASE}/particles.html`;

/** Small enough that SwiftShader stays quick, big enough that contacts happen. */
const COUNT = 2000;
const SEED = 7;

/**
 * Steps a scripted run submits and times as one sample: `CHUNK` in
 * demo/particles.ts.
 *
 * Restated rather than imported, because importing a *value* from the page would
 * execute it. The point of pinning it is the sample count: `msPerStep` is a median
 * over chunks, and a median over one chunk is the mean it replaced.
 */
const SCRIPT_CHUNK_STEPS = 8;

/**
 * Collect page and console errors for the life of the page.
 *
 * Idempotent, because `open()` attaches it and a test that probes the adapter
 * first has already been handed the array. Registering twice would report every
 * error twice, which reads like two failures where there is one.
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

async function report(page: Page): Promise<ParticlesReport> {
  const value = await page.evaluate(
    () => (window as unknown as { __particles?: ParticlesReport }).__particles,
  );
  expect(value, 'demo/particles.ts must publish window.__particles').toBeDefined();
  return value as ParticlesReport;
}

/** The whole report plus the page log: the only useful context on a failure. */
function dump(r: ParticlesReport): string {
  const { lines, ...rest } = r;
  return `${JSON.stringify(rest, null, 2)}\nlog:\n  ${lines.join('\n  ')}`;
}

/**
 * Load the page and wait for the build to settle.
 *
 * `status !== 'booting'` rather than the wanted value, so an `error` outcome
 * reaches the assertions with its message attached instead of becoming a timeout
 * that says nothing about why.
 */
async function open(page: Page, query: string, timeout = 90_000): Promise<ParticlesReport> {
  watchErrors(page);
  await page.goto(`${PAGE_URL}${query}`, { waitUntil: 'load' });
  await page.waitForFunction(
    () => {
      const r = (window as unknown as { __particles?: { status: string } }).__particles;
      return !!r && r.status !== 'booting';
    },
    undefined,
    { timeout },
  );
  return report(page);
}

/** What a tier with no `GPUDevice` must look like, on every row that could lie. */
function expectNoDevice(r: ParticlesReport): void {
  expect(r.deviceShared, `no device was acquired, so none can be shared:\n${dump(r)}`).toBe(false);
  expect(r.viewMode, `a CPU tier has nothing to blit from:\n${dump(r)}`).toBe('cpu');
  expect(r.frameMode, dump(r)).toBe('cpu-upload');
  expect(r.blitBytes, dump(r)).toBe(0);
  expect(r.backend, `three.js should not have picked WebGPU:\n${dump(r)}`).not.toBe('webgpu');
}

test.describe('the particle layer without a GPU', () => {
  test('the WebGL2 tier simulates and draws', async ({ page }) => {
    const r = await open(page, `?tier=webgl2&count=${COUNT}&seed=${SEED}`);

    expect(r.status, `the page did not come up:\n${dump(r)}`).toBe('live');
    expect(r.error, dump(r)).toBeUndefined();
    expect(r.tier, `the forced tier was not honoured:\n${dump(r)}`).toBe('webgl2');
    expect(r.count, dump(r)).toBe(COUNT);
    expectNoDevice(r);

    const canvas = page.locator('#viewport canvas');
    await expect(canvas).toBeVisible();
    await expect(page.locator('#loading')).toBeHidden();

    // `canvasBytes` would be non-zero for a solid clear colour too; distinct
    // colours are what prove 2000 instances were drawn. See `pixels.ts` for why
    // the screenshot carries them and the live canvas cannot.
    const shot = await canvas.screenshot();
    const unique = await distinctColors(page, shot);
    expect(unique, `the viewport is one flat colour:\n${dump(r)}`).toBeGreaterThan(8);

    // The loop, not just the first frame.
    const first = r.steps;
    await page.waitForTimeout(600);
    const later = await report(page);
    expect(later.steps, `the simulation stopped at step ${first}:\n${dump(later)}`).toBeGreaterThan(
      first,
    );
    expect(errorsOf(page), `page or console errors:\n${dump(later)}`).toEqual([]);
  });

  test('the CPU tier is the deterministic reference', async ({ page }) => {
    const query = `?tier=cpu&count=500&seed=${SEED}&steps=60`;
    const first = await open(page, query);

    expect(first.status, `the scripted run did not finish:\n${dump(first)}`).toBe('done');
    expect(first.tier, dump(first)).toBe('cpu');
    expect(first.steps, dump(first)).toBe(60);
    expect(first.msPerStep, dump(first)).toBeGreaterThan(0);
    // The distribution behind the headline number. A p95 under the p50 would mean
    // the percentile is not one, no samples at all would mean the median is the 0
    // it is initialised to, and one sample would mean the median is the mean.
    expect(first.stepSamples, dump(first)).toBe(Math.ceil(60 / SCRIPT_CHUNK_STEPS));
    expect(first.msPerStepP95, dump(first)).toBeGreaterThanOrEqual(first.msPerStep);
    expect(first.msPerStepMean, dump(first)).toBeGreaterThan(0);
    expect(first.canvasBytes, 'the final frame encoded to nothing').toBeGreaterThan(0);
    // `hex:count`: the count rides along, so a digest from a field of a
    // different size can never be mistaken for this one.
    expect(first.digest, dump(first)).toMatch(/^[0-9a-f]{8,}:\d+$/);
    expectNoDevice(first);

    // Replay independence, tested the only way that means anything: the same
    // seed and step count must land on the same digest, twice, with no GPU
    // involved anywhere. A renderer that wrote back into the field -- the thing
    // `ParticleView` is careful not to do -- would break this.
    const second = await open(page, query);
    expect(second.digest, `the CPU tier is not reproducible:\n${dump(second)}`).toBe(first.digest);
    expect(second.stats, dump(second)).toEqual(first.stats);
    expect(errorsOf(page), dump(second)).toEqual([]);
  });

  test('the step is fixed while the display is not', async ({ page }) => {
    const r = await open(page, `?tier=webgl2&count=${COUNT}&seed=${SEED}`);
    expect(r.status, dump(r)).toBe('live');

    // The step size belongs to the simulation. Nothing about a browser, a
    // refresh rate or a frame time is allowed to reach it.
    expect(r.fixedDt, dump(r)).toBeCloseTo(1 / 60, 6);

    const stepsBefore = r.steps;
    const framesBefore = r.frames;
    const t0 = Date.now();
    await page.waitForTimeout(1000);
    const later = await report(page);
    const elapsed = (Date.now() - t0) / 1000;

    const stepsDelta = later.steps - stepsBefore;
    const framesDelta = later.frames - framesBefore;
    expect(stepsDelta, `no steps ran in ${elapsed}s:\n${dump(later)}`).toBeGreaterThan(0);
    expect(framesDelta, dump(later)).toBeGreaterThan(0);
    expect(later.fixedDt, 'the step drifted mid-run').toBe(r.fixedDt);

    if (later.behind) {
      // The clock drops a backlog rather than growing it, so a runner that
      // cannot keep up simulates fewer seconds than passed. That is the design
      // and the HUD says so; asserting wall-clock agreement here would be
      // asserting that SwiftShader is fast.
      expect(stepsDelta * later.fixedDt, dump(later)).toBeLessThanOrEqual(elapsed * 1.2);
    } else {
      // Decoupling, measured: about one second of simulation per second of wall
      // clock, at whatever frame rate the display managed.
      expect(
        stepsDelta * later.fixedDt,
        `simulated ${stepsDelta * later.fixedDt}s in ${elapsed}s wall:\n${dump(later)}`,
      ).toBeGreaterThan(elapsed * 0.5);
      expect(stepsDelta * later.fixedDt, dump(later)).toBeLessThan(elapsed * 1.6);
    }
    expect(errorsOf(page), dump(later)).toEqual([]);
  });

  test('an unforced page downgrades instead of failing', async ({ page }) => {
    // No `tier` parameter: the probe decides, which is what a visitor gets.
    const r = await open(page, `?count=${COUNT}&seed=${SEED}`);

    expect(r.status, `auto tier did not come up:\n${dump(r)}`).toBe('live');
    expect(r.error, dump(r)).toBeUndefined();
    expect(['webgpu', 'webgl2', 'cpu'], `unknown tier:\n${dump(r)}`).toContain(r.tier);
    if (r.tier !== 'webgpu') expectNoDevice(r);

    const canvas = page.locator('#viewport canvas');
    await expect(canvas).toBeVisible();
    const unique = await distinctColors(page, await canvas.screenshot());
    expect(unique, `the auto tier drew nothing:\n${dump(r)}`).toBeGreaterThan(8);
    await expect(page.locator('#tier-badge')).toHaveText(new RegExp(`^${r.tier}`));
    expect(errorsOf(page), dump(r)).toEqual([]);
  });

  test('changing the tier rebuilds without leaking a canvas', async ({ page }) => {
    const r = await open(page, `?tier=webgl2&count=1000&seed=${SEED}`);
    expect(r.status, dump(r)).toBe('live');

    await page.click('#tier-select .seg[data-tier="cpu"]');
    await page.waitForFunction(
      () => {
        const p = (window as unknown as { __particles?: ParticlesReport }).__particles;
        return !!p && p.status === 'live' && p.tier === 'cpu';
      },
      undefined,
      { timeout: 90_000 },
    );
    const after = await report(page);
    expect(after.tier, dump(after)).toBe('cpu');
    expect(after.requested, dump(after)).toBe('cpu');

    // One canvas and one loop: a teardown that left the old renderer alive would
    // leave two canvases stacked in the viewport, both being drawn.
    await expect(page.locator('#viewport canvas')).toHaveCount(1);
    expect(after.steps, dump(after)).toBeGreaterThan(0);
    expect(errorsOf(page), `errors across a rebuild:\n${dump(after)}`).toEqual([]);
  });
});

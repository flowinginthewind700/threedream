/**
 * Browser spec for the M3 particle layer on the tier that actually matters.
 *
 * `particles.spec.ts` owns the fallback tiers; this file owns the three claims
 * the milestone exists to make:
 *
 *   - one `GPUDevice`, shared between simulation and rendering -- the M2
 *     architecture claim, re-asserted where it pays off. `deviceShared` is
 *     `renderer.backend.device === shared.device`, not a proxy for it.
 *   - one draw call for the whole field, fed from a buffer the compute passes
 *     wrote on the device. `frameMode === 'gpu-blit'` means no readback fed the
 *     frame, and `blitBytes` is the expanded instance stream that went into it.
 *   - the WGSL kernels agree with the CPU reference. Tolerances, not equality:
 *     the GPU resolves contacts with atomics, so the order two overlapping
 *     particles are pushed apart in is not specified, and the digest is allowed
 *     to differ while the field's energy and speed envelope may not.
 *
 * Runs in the `chromium-webgpu` project only (see playwright.config.ts). Every
 * test skips itself when the runner cannot produce an adapter, and the second
 * describe is the mirror image: it only runs when the adapter is missing, so a
 * GPU-less runner still gets real assertions out of this file.
 */

import { expect, test, type Page } from '@playwright/test';

import type { ParticlesReport } from '../demo/particles.js';
import { INSTANCE_BYTES } from '../src/gpu/particleInstances.js';

import { distinctColors } from './pixels.js';

/** The demo serves from a subpath when built for Pages; resolve either way. */
const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173/threedream';
const PAGE_URL = `${BASE}/particles.html`;

/** Big enough that a per-particle CPU upload would be visibly slow, small enough to be quick. */
const COUNT = 4000;
const STEPS = 120;
const SEED = 1234;
/** The acceptance criterion's lower bound: 50k particles must hold up. */
const LARGE_COUNT = 50_000;

/**
 * Steps a scripted run submits and times as one sample: `CHUNK` in
 * demo/particles.ts.
 *
 * Restated rather than imported, because importing a *value* from the page would
 * execute it. What it pins is the sample count behind `msPerStep`, which is a
 * median: a median over one chunk is the mean it replaced, and a mean over four
 * was the statistic that let one contended chunk triple a rung's reported cost.
 */
const SCRIPT_CHUNK_STEPS = 8;

/**
 * `hex:count` -- the shape `ParticleField.digest()` returns.
 *
 * The count is part of the string, so a digest can never be mistaken for one
 * from a field of a different size. Asserting the shape rather than a value is
 * deliberate here: which of two overlapping particles the GPU pushes apart first
 * is not specified, so the bytes are not reproducible across backends and
 * `particles.spec.ts` is where a digest gets pinned exactly.
 */
const DIGEST = /^[0-9a-f]{8,}:\d+$/;

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
 * that says nothing about why. The default is generous because the GPU project
 * compiles six WGSL pipelines through ANGLE before the first frame.
 */
async function open(page: Page, query: string, timeout = 120_000): Promise<ParticlesReport> {
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

/**
 * Whether this runner has an adapter, read off the page's own report.
 *
 * Probing `navigator.gpu` from the spec instead would answer "no" everywhere: a
 * freshly launched Playwright page is `about:blank`, and its opaque origin is not
 * a secure context, so `navigator.gpu` does not exist there. That skips the GPU
 * claims on a machine that has a GPU and runs the no-adapter mirror on it --
 * both wrong, and both quietly. `shared_device.spec.ts` reads the same fact off
 * its own report for the same reason.
 */
function hasAdapter(r: ParticlesReport): boolean {
  return r.webgpuAvailable;
}

/** Relative difference, for two numbers that should agree within a tolerance. */
function rel(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(1e-9, (Math.abs(a) + Math.abs(b)) / 2);
}

test.describe('one GPUDevice, one draw call, no round trip', () => {
  test('the blit path runs entirely on the shared device', async ({ page }) => {
    // `strict=1` so a silent downgrade fails the test instead of passing it with
    // the CPU path -- the whole point of this file.
    const r = await open(
      page,
      `?tier=webgpu&strict=1&count=${COUNT}&seed=${SEED}&steps=${STEPS}&collisions=1`,
    );
    // No adapter is an environment fact rather than a regression, and the mirror
    // describe below owns that case. Everything from here assumes hardware.
    test.skip(!hasAdapter(r), `no WebGPU adapter here: ${r.reason}`);

    expect(r.status, `the scripted run did not finish:\n${dump(r)}`).toBe('done');
    expect(r.error, dump(r)).toBeUndefined();
    expect(r.tier, dump(r)).toBe('webgpu');
    expect(r.backend, `three.js did not draw with WebGPU:\n${dump(r)}`).toBe('webgpu');
    expect(r.gpuError, `a GPU pass failed and the view fell back:\n${dump(r)}`).toBeNull();

    // M2's claim, where it earns its keep: the renderer is drawing with the very
    // device the compute passes were dispatched on.
    expect(r.deviceShared, `renderer.backend.device is not the acquired device:\n${dump(r)}`).toBe(
      true,
    );
    expect(r.viewMode, dump(r)).toBe('gpu');
    expect(r.frameMode, `the frame was uploaded from CPU copies:\n${dump(r)}`).toBe('gpu-blit');
    expect(
      r.blitBytes,
      `expected the full instance stream for ${COUNT} particles:\n${dump(r)}`,
    ).toBe(COUNT * INSTANCE_BYTES);

    expect(r.count, dump(r)).toBe(COUNT);
    expect(r.steps, dump(r)).toBe(STEPS);
    expect(r.msPerStep, dump(r)).toBeGreaterThan(0);
    expect(r.msPerStep, dump(r)).toBeLessThan(Number.POSITIVE_INFINITY);
    // The spread, next to the headline: on a device that is also running a
    // compositor, p95 over p50 is the difference between "the kernels cost this"
    // and "the machine was busy". A p95 below the p50 means the percentile is not
    // one; zero samples means the median is the 0 it is initialised to.
    expect(r.stepSamples, dump(r)).toBe(Math.ceil(STEPS / SCRIPT_CHUNK_STEPS));
    expect(r.msPerStepP95, dump(r)).toBeGreaterThanOrEqual(r.msPerStep);
    expect(r.msPerStepMean, dump(r)).toBeGreaterThan(0);

    // One draw call for the field; the scene's bounds helper and any clear pass
    // account for the small headroom above one.
    expect(r.drawCalls, `the field was not drawn in a single call:\n${dump(r)}`).toBeGreaterThan(0);
    expect(r.drawCalls, dump(r)).toBeLessThanOrEqual(4);
    expect(r.triangles, dump(r)).toBeGreaterThan(0);

    // Physics invariants: nothing left the box, and the hash did not saturate.
    expect(r.stats.escaped, dump(r)).toBe(0);
    expect(r.stats.hashOverflow, dump(r)).toBe(0);
    expect(r.stats.contacts, `collisions were on but nothing touched:\n${dump(r)}`).toBeGreaterThan(
      0,
    );
    expect(r.digest, dump(r)).toMatch(DIGEST);
    expect(r.canvasBytes, 'the final frame encoded to nothing').toBeGreaterThan(0);
    expect(errorsOf(page), `page or console errors:\n${dump(r)}`).toEqual([]);
  });

  test('the live page keeps blitting and keeps moving', async ({ page }) => {
    // No `steps`: the real loop, which is what a visitor gets.
    const r = await open(page, `?tier=webgpu&count=${COUNT}&seed=${SEED}`);
    test.skip(!hasAdapter(r), `no WebGPU adapter here: ${r.reason}`);
    expect(r.status, dump(r)).toBe('live');
    expect(r.deviceShared, dump(r)).toBe(true);

    // The first frames may be served while the compute attach is still settling;
    // the blit path has to engage, not merely be intended.
    await expect
      .poll(async () => (await report(page)).frameMode, {
        timeout: 30_000,
        message: 'the blit path never engaged',
      })
      .toBe('gpu-blit');

    const live = await report(page);
    expect(live.viewMode, dump(live)).toBe('gpu');
    expect(live.steps, dump(live)).toBeGreaterThan(0);

    const canvas = page.locator('#viewport canvas');
    await expect(canvas).toBeVisible();
    await expect(page.locator('#loading')).toBeHidden();
    const unique = await distinctColors(page, await canvas.screenshot());
    expect(unique, `the viewport is one flat colour:\n${dump(live)}`).toBeGreaterThan(8);

    // Motion, not a still image that happens to be detailed: two frames 400ms
    // apart must differ. This is also the only check that catches a loop that
    // rendered once and stopped.
    const before = await canvas.screenshot();
    await page.waitForTimeout(400);
    const after = await canvas.screenshot();
    expect(after.equals(before), `the viewport did not change in 400ms:\n${dump(live)}`).toBe(false);

    // What the page claims about itself has to match what it did.
    const device = page.locator('#ti-device');
    await expect(device).toHaveText('OK');
    await expect(device).toHaveClass(/is-good/);
    const path = page.locator('#ti-path');
    await expect(path).toHaveText('gpu-blit');
    await expect(path).toHaveClass(/is-good/);
    await expect(page.locator('#path-badge')).toHaveClass(/is-learned/);
    await expect(page.locator('#verdict')).not.toHaveClass(/is-warn/);
    expect(errorsOf(page), `page or console errors:\n${dump(live)}`).toEqual([]);
  });

  test('the WGSL kernels agree with the CPU reference', async ({ page }) => {
    const n = 2000;
    const steps = 60;
    const base = `count=${n}&seed=${SEED}&steps=${steps}`;

    // One throwaway boot to answer the adapter question before spending minutes
    // on the comparison. It asks for a device, because `webgpuAvailable` is only
    // meaningful on a run that tried to get one.
    const probe = await open(page, '?tier=webgpu&strict=1&count=64&steps=1', 60_000);
    test.skip(!hasAdapter(probe), `no WebGPU adapter here: ${probe.reason}`);

    // Free flight first: no contacts, so the integrator is the only thing that
    // can disagree, and it must not.
    const cpu = await open(page, `?tier=cpu&${base}&collisions=0`);
    const gpu = await open(page, `?tier=webgpu&strict=1&${base}&collisions=0`);

    expect(cpu.status, `the CPU reference did not finish:\n${dump(cpu)}`).toBe('done');
    expect(gpu.status, `the GPU tier did not finish:\n${dump(gpu)}`).toBe('done');
    expect(cpu.stats.escaped, dump(cpu)).toBe(0);
    expect(gpu.stats.escaped, dump(gpu)).toBe(0);
    expect(gpu.steps, dump(gpu)).toBe(cpu.steps);

    expect(
      rel(gpu.stats.kineticEnergy, cpu.stats.kineticEnergy),
      `kinetic energy diverged: gpu=${gpu.stats.kineticEnergy} cpu=${cpu.stats.kineticEnergy}\n${dump(gpu)}\n${dump(cpu)}`,
    ).toBeLessThan(0.05);
    expect(
      rel(gpu.stats.maxSpeed, cpu.stats.maxSpeed),
      `speed envelope diverged: gpu=${gpu.stats.maxSpeed} cpu=${cpu.stats.maxSpeed}\n${dump(gpu)}`,
    ).toBeLessThan(0.1);

    // Deliberately not comparing digests. Both sides produce one, and both are
    // stable run to run, but the GPU resolves contacts and hash inserts with
    // atomics: which of two overlapping particles gets pushed first is not
    // specified, so bit equality is a promise the hardware never made. The
    // digest is asserted well-formed here and pinned exactly on the CPU tier in
    // `particles.spec.ts`, which is where determinism actually lives.
    expect(cpu.digest, dump(cpu)).toMatch(DIGEST);
    expect(gpu.digest, dump(gpu)).toMatch(DIGEST);

    // Now with contacts on. The tolerance is loose on purpose: the resolution
    // order is unspecified, and two bodies touching in a different order settle
    // into a genuinely different configuration. What must still hold is that
    // both sides found contacts at all, neither lost a body out of the box, and
    // neither blew the hash table.
    const cpuHit = await open(page, `?tier=cpu&${base}&collisions=1`);
    const gpuHit = await open(page, `?tier=webgpu&strict=1&${base}&collisions=1`);
    expect(cpuHit.status, dump(cpuHit)).toBe('done');
    expect(gpuHit.status, dump(gpuHit)).toBe('done');

    // Both count each touching pair once (i < j), so the numbers are comparable
    // in magnitude even when the exact set differs.
    expect(cpuHit.stats.contacts, dump(cpuHit)).toBeGreaterThan(0);
    expect(gpuHit.stats.contacts, dump(gpuHit)).toBeGreaterThan(0);
    expect(gpuHit.stats.escaped, dump(gpuHit)).toBe(0);
    expect(gpuHit.stats.hashOverflow, dump(gpuHit)).toBe(0);
    expect(
      rel(gpuHit.stats.kineticEnergy, cpuHit.stats.kineticEnergy),
      `colliding energy diverged: gpu=${gpuHit.stats.kineticEnergy} cpu=${cpuHit.stats.kineticEnergy}\n${dump(gpuHit)}\n${dump(cpuHit)}`,
    ).toBeLessThan(0.25);

    expect(errorsOf(page), dump(gpuHit)).toEqual([]);
  });

  test('50,000 particles hold up on the device', async ({ page }) => {
    // The acceptance criterion's lower bound, run for real. Fewer steps than the
    // small case, because the budget here is allocation and pipeline setup.
    test.setTimeout(300_000);

    const r = await open(
      page,
      `?tier=webgpu&strict=1&count=${LARGE_COUNT}&seed=${SEED}&steps=20`,
      240_000,
    );
    test.skip(!hasAdapter(r), `no WebGPU adapter here: ${r.reason}`);

    expect(r.status, `50k did not finish:\n${dump(r)}`).toBe('done');
    expect(r.error, dump(r)).toBeUndefined();
    expect(r.tier, dump(r)).toBe('webgpu');
    expect(r.count, dump(r)).toBe(LARGE_COUNT);
    expect(r.steps, dump(r)).toBe(20);
    expect(r.frameMode, dump(r)).toBe('gpu-blit');
    expect(r.blitBytes, dump(r)).toBe(LARGE_COUNT * INSTANCE_BYTES);
    expect(r.stats.escaped, dump(r)).toBe(0);
    expect(r.stats.hashOverflow, `the spatial hash saturated at 50k:\n${dump(r)}`).toBe(0);
    expect(r.drawCalls, dump(r)).toBeLessThanOrEqual(4);
    expect(r.triangles, dump(r)).toBeGreaterThan(0);
    expect(errorsOf(page), dump(r)).toEqual([]);
  });
});

test.describe('without a WebGPU adapter', () => {
  test('auto downgrades and still draws', async ({ page }) => {
    // The acceptance criterion, in the environment that exercises it: no tier
    // forced, nothing available, and a page that must still put pixels up.
    const r = await open(page, `?count=2000&seed=${SEED}`);
    test.skip(hasAdapter(r), 'this runner has an adapter; see the describe above');

    expect(r.status, `the page did not come up without a GPU:\n${dump(r)}`).toBe('live');
    expect(r.error, dump(r)).toBeUndefined();
    expect(['webgl2', 'cpu'], `unexpected tier:\n${dump(r)}`).toContain(r.tier);
    expect(r.deviceShared, dump(r)).toBe(false);
    expect(r.frameMode, dump(r)).toBe('cpu-upload');
    expect(r.backend, dump(r)).not.toBe('webgpu');

    await expect(page.locator('#loading')).toBeHidden();
    const canvas = page.locator('#viewport canvas');
    await expect(canvas).toBeVisible();
    const unique = await distinctColors(page, await canvas.screenshot());
    expect(unique, `the downgraded tier drew nothing:\n${dump(r)}`).toBeGreaterThan(8);

    // The page says why, in the log a visitor reads. A downgrade that happened
    // silently would leave the operator guessing which tier they are on.
    const log = await page.locator('#log').innerText();
    expect(log, `the downgrade was not explained:\n${dump(r)}`).toMatch(/no WebGPU device/i);
    expect(errorsOf(page), dump(r)).toEqual([]);
  });

  test('strict refuses instead of downgrading', async ({ page }) => {
    // `strict=1` is the escape hatch for a caller that would rather fail loudly
    // than silently simulate on the CPU -- a training run whose numbers must be
    // comparable across machines, say. Refusing is the correct behaviour, and it
    // has to be visible on the page rather than only in the console.
    const r = await open(page, '?tier=webgpu&strict=1&count=1000&steps=10');
    test.skip(hasAdapter(r), 'this runner has an adapter; see the describe above');

    expect(r.status, dump(r)).toBe('error');
    expect(r.error, `no reason was reported:\n${dump(r)}`).toMatch(/cannot be honoured/i);
    await expect(page.locator('#tier-badge')).toHaveText('failed');
    await expect(page.locator('#loading')).toBeVisible();
    await expect(page.locator('#verdict')).toHaveClass(/is-warn/);
  });
});

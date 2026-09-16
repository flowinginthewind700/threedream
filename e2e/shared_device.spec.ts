/**
 * Browser spec for the shared-GPUDevice architecture claim.
 *
 * Everything from M2 onwards -- GPU broadphase, GPU integration, the Rust/wgpu
 * bridge -- rests on one structural fact: the page can own the `GPUDevice`, hand
 * it to `THREE.WebGPURenderer`, and then bind a raw WGSL pipeline to a buffer
 * three.js allocated. If that stops being true in some future three.js release
 * the architecture has to change, so it is gated rather than assumed.
 * `scripts/bench_shared_device.mjs` proved it once by hand;
 * `demo/shared-device.ts` is that proof as a page, and this spec is that page as
 * a test.
 *
 * Runs in both projects (see playwright.config.ts), because the two halves need
 * opposite machines. The claim half needs ANGLE's Vulkan backend with the WebGPU
 * service enabled; the fallback half needs a browser that grants no adapter at
 * all, which is exactly what the SwiftShader project provides. Each half skips
 * itself where its environment is missing, so a GPU-less runner still gets real
 * assertions out of this file instead of a silently skipped one.
 *
 * Two describes, one per environment, because both outcomes are worth pinning:
 * with an adapter every claim must hold; without one the probe must say so, mark
 * the claims as never run, and present the tier it fell back to.
 */

import { expect, test, type Page } from '@playwright/test';

import type { SharedDeviceReport } from '../demo/shared-device.js';
import { LIMIT_FLOOR } from '../src/gpu/capabilities.js';

import { distinctColors } from './pixels.js';

/** The demo serves from a subpath when built for Pages; resolve either way. */
const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173/threedream';
const PAGE_URL = `${BASE}/shared-device.html`;

/** Points the raw kernel must have stamped; mirrors the page's own constant. */
const POINTS = 8192;

function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console: ${m.text()}`);
  });
  return errors;
}

async function report(page: Page): Promise<SharedDeviceReport> {
  const value = await page.evaluate(
    () => (window as unknown as { __sharedDevice?: SharedDeviceReport }).__sharedDevice,
  );
  expect(value, 'demo/shared-device.ts must publish window.__sharedDevice').toBeDefined();
  return value as SharedDeviceReport;
}

/** The whole report plus the page log: the only useful context on a failure. */
function dump(r: SharedDeviceReport): string {
  const { lines, ...rest } = r;
  return `${JSON.stringify(rest, null, 2)}\nlog:\n  ${lines.join('\n  ')}`;
}

/**
 * Load the page and wait for its auto-run to settle.
 *
 * `status !== 'running'` rather than `done`, so an `error` outcome reaches the
 * assertions with its message attached instead of turning into a timeout.
 */
async function openAndWait(page: Page): Promise<SharedDeviceReport> {
  await page.goto(PAGE_URL, { waitUntil: 'load' });
  await page.waitForFunction(
    () => {
      const r = (window as unknown as { __sharedDevice?: { status: string } }).__sharedDevice;
      return !!r && r.status !== 'running';
    },
    undefined,
    { timeout: 90_000 },
  );
  return report(page);
}

/** Whether this browser, with these flags, can produce an adapter at all. */
async function webgpuPresent(page: Page): Promise<boolean> {
  return page.evaluate(async () => {
    const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (!gpu) return false;
    try {
      return (await gpu.requestAdapter()) !== null;
    } catch {
      return false;
    }
  });
}

test.describe('one GPUDevice, shared by three.js and raw WGSL', () => {
  test('all three claims hold', async ({ page }) => {
    const errors = watchErrors(page);
    const r = await openAndWait(page);
    // No adapter is an environment fact, not a code regression; the fallback
    // path is the second describe's subject. Everything below assumes hardware.
    test.skip(!r.webgpuAvailable, `no WebGPU adapter here: ${r.tierReason}`);

    expect(r.status, `check did not complete:\n${dump(r)}`).toBe('done');
    expect(r.error, `check reported an error:\n${dump(r)}`).toBeUndefined();
    expect(r.tier, dump(r)).toBe('webgpu');
    expect(r.backendIsWebGpu, `three.js did not pick its WebGPU backend:\n${dump(r)}`).toBe(true);

    // CLAIM 1: the device we created is the device the renderer uses. If three.js
    // ever copies or re-requests it internally the zero-copy bridge is gone, and
    // this is the line that says so.
    expect(r.claims.sameDeviceObject, `renderer ignored the injected device:\n${dump(r)}`).toBe(true);
    // CLAIM 2: the GPUBuffer behind a three.js StorageBufferAttribute is reachable
    // from outside the renderer.
    expect(r.claims.threeBufferReachable, `three.js buffer not reachable:\n${dump(r)}`).toBe(true);
    // CLAIM 3: a raw pipeline bound to that buffer, interleaved with render().
    expect(r.claims.rawPipelineBound, `raw pipeline not bound:\n${dump(r)}`).toBe(true);
    expect(r.claims.interleaved, `compute and render did not interleave:\n${dump(r)}`).toBe(true);

    // The sentinel is the real evidence for claim 3: only the raw WGSL kernel
    // writes 7.5 into `w`, so a full count means the dispatch landed on three.js's
    // buffer rather than on a look-alike of our own.
    expect(r.points, dump(r)).toBe(POINTS);
    expect(r.sentinelPoints, `the raw kernel did not write every point:\n${dump(r)}`).toBe(POINTS);

    expect(r.frames, dump(r)).toBeGreaterThan(0);
    expect(r.msPerFrame, dump(r)).toBeGreaterThan(0);
    expect(Number.isFinite(r.msPerFrame), dump(r)).toBe(true);
    expect(r.canvasBytes, 'the canvas encoded to nothing').toBeGreaterThan(0);

    // An adapter missing a floored limit would have been rejected by
    // `probeWebGpu`, so reaching here with unmet limits means the probe and the
    // tier selection disagree -- the subtlest failure this page can have.
    expect(r.unmetLimits, `an available adapter is below LIMIT_FLOOR:\n${dump(r)}`).toEqual([]);
    expect(r.limits.invocationsPerWorkgroup, dump(r)).toBeGreaterThanOrEqual(
      LIMIT_FLOOR.maxComputeInvocationsPerWorkgroup,
    );

    expect(errors, `page or console errors:\n${dump(r)}`).toEqual([]);
  });

  test('what is on screen came from that device, and keeps coming', async ({ page }) => {
    const r = await openAndWait(page);
    test.skip(!r.webgpuAvailable, `no WebGPU adapter here: ${r.tierReason}`);
    expect(r.status, dump(r)).toBe('done');

    const canvas = page.locator('#viewport canvas');
    await expect(canvas).toBeVisible();

    // `report.canvasBytes` only proves the canvas encoded to *something*, and a
    // solid clear colour encodes fine. Distinct colours prove the box was drawn.
    const first = await canvas.screenshot();
    const unique = await distinctColors(page, first);
    expect(unique, `the viewport is one flat colour:\n${dump(r)}`).toBeGreaterThan(8);

    // The page keeps presenting after the check finishes, on the same device. A
    // frozen frame would mean the device went invalid the moment the assertions
    // passed, which is exactly the failure mode this check exists for.
    await page.waitForTimeout(500);
    const second = await canvas.screenshot();
    expect(first.equals(second), `the shared device stopped presenting:\n${dump(r)}`).toBe(false);

    await expect(page.locator('#frame-badge')).toHaveText(`${r.frames} frames`);
    await expect(page.locator('#tier-badge')).toHaveText(/^webgpu/);
  });

  test('the sidebar reports the claims it is showing', async ({ page }) => {
    const r = await openAndWait(page);
    test.skip(!r.webgpuAvailable, `no WebGPU adapter here: ${r.tierReason}`);

    for (const id of ['#cl-device', '#cl-buffer', '#cl-pipeline', '#cl-interleaved']) {
      await expect(page.locator(id), `${id} should read OK`).toHaveText('OK');
      await expect(page.locator(id)).toHaveClass(/is-good/);
    }
    await expect(page.locator('#cl-sentinel')).toHaveText(`${POINTS} / ${POINTS}`);
    await expect(page.locator('#verdict')).toHaveText(/All three claims hold/);
    await expect(page.locator('#verdict')).not.toHaveClass(/is-warn/);
    await expect(page.locator('#loading')).toBeHidden();

    // The log is what a human reads when this goes red, so it has to survive the
    // build: one line for the capability probe and one per claim.
    const logged = await page.locator('#log li').allTextContents();
    expect(logged.join('\n'), 'the probe log should describe the adapter').toMatch(/webgpu:/);
    expect(logged.length, dump(r)).toBeGreaterThan(6);
  });

  test('re-running tears the old device down and reaches the same verdict', async ({ page }) => {
    const errors = watchErrors(page);
    const first = await openAndWait(page);
    test.skip(!first.webgpuAvailable, `no WebGPU adapter here: ${first.tierReason}`);
    expect(first.status, dump(first)).toBe('done');

    await page.click('#run-btn');
    const second = await page
      .waitForFunction(
        () => {
          const r = (window as unknown as { __sharedDevice?: SharedDeviceReport }).__sharedDevice;
          return r && r.status !== 'running' ? r : null;
        },
        undefined,
        { timeout: 90_000 },
      )
      .then((handle) => handle.jsonValue() as Promise<SharedDeviceReport>);

    expect(second.status, `the second run failed:\n${dump(second)}`).toBe('done');
    expect(second.claims, dump(second)).toEqual(first.claims);
    expect(second.sentinelPoints, dump(second)).toBe(POINTS);
    // One canvas per run: a leaked renderer would leave two in the viewport and
    // two rAF loops driving them, which is what `teardown()` prevents.
    await expect(page.locator('#viewport canvas')).toHaveCount(1);

    expect(errors, `page or console errors across two runs:\n${dump(second)}`).toEqual([]);
  });
});

test.describe('without a WebGPU adapter', () => {
  test('the probe says so, and the page presents the tier it landed on', async ({ page }) => {
    const errors = watchErrors(page);
    const r = await openAndWait(page);
    test.skip(r.webgpuAvailable, 'this runner has WebGPU; the claims above are the real gate');

    // Not a free pass: `selectRenderTier` is production code, and this is the
    // only place it runs against a real browser rather than a stub.
    expect(r.status, dump(r)).toBe('unavailable');
    expect(r.error, dump(r)).toMatch(/no usable WebGPU adapter/);
    expect(['webgl2', 'cpu'], `unexpected fallback tier:\n${dump(r)}`).toContain(r.tier);
    expect(r.tierReason, dump(r)).toMatch(/fallback/i);
    expect(r.claims, `no claim may pass without a device:\n${dump(r)}`).toEqual({
      sameDeviceObject: false,
      threeBufferReachable: false,
      rawPipelineBound: false,
      interleaved: false,
    });
    // The stronger half of the same statement. `claims` alone cannot tell "we
    // checked and it failed" from "we never got to check", and a page that shows
    // the second as the first tells readers their machine is broken.
    expect(r.claimsRun, `nothing here was evaluated:\n${dump(r)}`).toEqual({
      sameDeviceObject: false,
      threeBufferReachable: false,
      rawPipelineBound: false,
      interleaved: false,
    });
    expect(r.sentinelPoints, dump(r)).toBe(0);

    await expect(page.locator('#tier-badge')).toHaveText(r.tier);
    await expect(page.locator('#verdict')).toHaveText(/Not checked/);
    await expect(page.locator('#verdict')).toHaveClass(/is-warn/);

    const logged = await page.locator('#log li').allTextContents();
    const logText = logged.join('\n');
    expect(logText, 'the log should explain the absence').toMatch(/webgpu: unavailable/);
    expect(logText, 'the log should say the claims were not run').toMatch(/not checked/);

    // Every row that could imply a result reads `not run`, and none of them wears
    // the failure colour: a wall of FAIL is a claim about the reader's machine
    // this page never earned the right to make.
    const claimRows = [
      'cl-device',
      'cl-buffer',
      'cl-pipeline',
      'cl-interleaved',
      'cl-sentinel',
      'cl-pixels',
    ];
    for (const id of claimRows) {
      const row = page.locator(`#${id}`);
      await expect(row, `${id} should not claim a result`).toHaveText('not run');
      await expect(row, `${id} is painted as a failure`).not.toHaveClass(/is-warn/);
    }

    // And the viewport is not a black hole. On the `cpu` tier there is no GL
    // context to present with either, so that case stays the veil's job; here the
    // tier exists and has to be visibly running.
    if (r.tier === 'webgl2') {
      await expect(page.locator('#loading')).toBeHidden();
      const canvas = page.locator('#viewport canvas');
      await expect(canvas).toBeVisible();
      const unique = await distinctColors(page, await canvas.screenshot());
      expect(unique, `the fallback viewport is one flat colour:\n${dump(r)}`).toBeGreaterThan(8);

      const first = r.frames;
      await page.waitForTimeout(500);
      const later = await report(page);
      expect(
        later.frames,
        `the fallback stopped presenting at frame ${first}:\n${dump(later)}`,
      ).toBeGreaterThan(first);
      expect(errors, `page or console errors on the fallback path:\n${dump(later)}`).toEqual([]);
    }
  });

  test('the browser really has no adapter, rather than the page failing to ask', async ({ page }) => {
    const r = await openAndWait(page);
    test.skip(r.webgpuAvailable, 'this runner has WebGPU');

    // Separates "the environment has no GPU" from "the page broke before it
    // asked". Without this, a regression in `probeWebGpu` would look exactly like
    // a GPU-less runner and pass.
    expect(await webgpuPresent(page), `the page saw no adapter but one exists:\n${dump(r)}`).toBe(
      false,
    );
  });
});

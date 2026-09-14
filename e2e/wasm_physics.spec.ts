/**
 * Browser spec for the wasm physics kernel.
 *
 * `tests/wasm_backend.test.ts` already proves the kernel is bit-exact with
 * `builtin.ts` -- in Node, on V8, with the wasm loaded through a Node path.
 * That leaves the claim the plan actually makes untested: the *same bytes*
 * produce the *same digest* in a real browser, where the glue is fetched over
 * HTTP, compiled by the browser's own wasm compiler, and run against a live
 * `WebAssembly.Memory`. A wasm-bindgen version skew or a float difference
 * between engines shows up here and nowhere else.
 *
 * The page (`demo/physics-check.ts`) runs the reference scene three times --
 * builtin, wasm, and a second wasm world as a replay -- and publishes the whole
 * report on `window.__physicsCheck`. This spec reads that object rather than
 * scraping the sidebar, so a cosmetic change to the demo cannot flip the gate.
 * The sidebar is asserted separately, once, as the thing a human looks at.
 */

import { expect, test, type Page } from '@playwright/test';

import type { PhysicsCheckReport } from '../demo/physics-check.js';
import {
  REFERENCE_BODIES,
  REFERENCE_GOLDEN_DIGEST,
  REFERENCE_STEPS,
} from '../src/physics/reference.js';

/** The demo serves from a subpath when built for Pages; resolve either way. */
const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173/threedream';
const PAGE_URL = `${BASE}/physics-check.html`;

/** Collect page and console errors so a silent failure can be reported loudly. */
function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console: ${m.text()}`);
  });
  return errors;
}

async function report(page: Page): Promise<PhysicsCheckReport> {
  const value = await page.evaluate(
    () => (window as unknown as { __physicsCheck?: PhysicsCheckReport }).__physicsCheck,
  );
  expect(value, 'demo/physics-check.ts must publish window.__physicsCheck').toBeDefined();
  return value as PhysicsCheckReport;
}

/** The full report, formatted, for the failure message of every assertion. */
function dump(r: PhysicsCheckReport): string {
  return JSON.stringify(r, null, 2);
}

/**
 * Open the page and wait for the measurement run to finish.
 *
 * Waiting on `status !== 'running'` rather than on `done`: an `error` status is
 * a legitimate outcome of the wait and must reach the assertion, where its
 * `error` field gets printed. Waiting for `done` alone would time out on a
 * failure and hide the reason.
 */
async function openAndWait(page: Page): Promise<PhysicsCheckReport> {
  await page.goto(PAGE_URL, { waitUntil: 'load' });
  await page.waitForFunction(
    () => {
      const r = (window as unknown as { __physicsCheck?: { status: string } }).__physicsCheck;
      return !!r && r.status !== 'running';
    },
    undefined,
    { timeout: 90_000 },
  );
  return report(page);
}

test.describe('the wasm kernel in a browser', () => {
  test('reproduces the digest recorded in Node', async ({ page }) => {
    const errors = watchErrors(page);
    const r = await openAndWait(page);

    expect(r.status, `check did not complete:\n${dump(r)}`).toBe('done');
    expect(r.error, `check reported an error:\n${dump(r)}`).toBeUndefined();

    expect(r.golden, 'the page must carry the committed golden digest').toBe(REFERENCE_GOLDEN_DIGEST);
    expect(r.digests.builtin, `builtin diverged:\n${dump(r)}`).toBe(REFERENCE_GOLDEN_DIGEST);
    expect(r.digests.wasm, `wasm diverged:\n${dump(r)}`).toBe(REFERENCE_GOLDEN_DIGEST);
    expect(r.digests.replay, `wasm replay diverged:\n${dump(r)}`).toBe(REFERENCE_GOLDEN_DIGEST);
    expect(r.matches, `match flags disagree with the digests:\n${dump(r)}`).toEqual({
      builtin: true,
      wasm: true,
      replay: true,
      replayOfWasm: true,
    });

    expect(errors, `page or console errors:\n${dump(r)}`).toEqual([]);
  });

  test('ran the whole reference scene, not a truncated one', async ({ page }) => {
    const errors = watchErrors(page);
    const r = await openAndWait(page);

    // A digest can match by accident if the run stopped early: fewer steps
    // hashed is a different, shorter input. These counts are what make the
    // equality above mean "the same 600 steps of the same 9 bodies".
    expect(r.counts.steps, dump(r)).toBe(REFERENCE_STEPS);
    expect(r.counts.bodies, dump(r)).toBe(REFERENCE_BODIES.length);
    expect(r.counts.samples, dump(r)).toBe(REFERENCE_BODIES.length * REFERENCE_STEPS);
    expect(r.counts.values, 'a truncated hash would still match').toBeGreaterThan(10_000);
    // Both must be non-zero: a scene that never collided and was never raycast
    // would exercise neither the narrowphase nor the query path in wasm.
    expect(r.counts.contacts, dump(r)).toBeGreaterThan(0);
    expect(r.counts.rays, dump(r)).toBeGreaterThan(0);

    expect(errors, dump(r)).toEqual([]);
  });

  test('the wasm world is faster than the builtin one, in the browser too', async ({ page }) => {
    const r = await openAndWait(page);

    // Not a benchmark. SwiftShader and CI runners make absolute numbers
    // meaningless, and a fixed speedup floor would be flaky, so what is asserted
    // is that both runs were really timed and that wasm did not come out slower
    // than the JS solver -- the one outcome that would mean the kernel is being
    // reached through a pathological boundary. The 2.5x measured locally lives
    // in docs/, not in a gate.
    expect(r.timings.builtinMs, dump(r)).toBeGreaterThan(0);
    expect(r.timings.wasmMs, dump(r)).toBeGreaterThan(0);
    expect(r.timings.speedup, dump(r)).toBeGreaterThan(1);
    expect(Number.isFinite(r.timings.speedup)).toBe(true);
  });

  test('the sidebar shows the same result the report carries', async ({ page }) => {
    const errors = watchErrors(page);
    const r = await openAndWait(page);
    await expect(page.locator('#loading')).toBeHidden({ timeout: 30_000 });

    for (const [id, digest] of [
      ['#dg-golden', r.golden],
      ['#dg-builtin', r.digests.builtin],
      ['#dg-wasm', r.digests.wasm],
      ['#dg-replay', r.digests.replay],
    ] as const) {
      await expect(page.locator(id), `${id} should show its digest`).toHaveText(digest);
      // The colour is the at-a-glance verdict; a green row over a wrong digest
      // is worse than no colour at all.
      await expect(page.locator(id)).toHaveClass(/is-good/);
    }

    await expect(page.locator('#st-steps')).toHaveText(String(REFERENCE_STEPS));
    await expect(page.locator('#st-bodies')).toHaveText(String(REFERENCE_BODIES.length));
    await expect(page.locator('#verdict')).toHaveText(/reproduce the digest recorded in Node/);
    await expect(page.locator('#verdict')).not.toHaveClass(/is-warn/);

    expect(errors, dump(r)).toEqual([]);
  });
});

test.describe('the live view runs on either backend', () => {
  test('switching to wasm drives the same scene from the wasm kernel', async ({ page }) => {
    const errors = watchErrors(page);
    await openAndWait(page);
    await expect(page.locator('#loading')).toBeHidden({ timeout: 30_000 });

    const canvas = page.locator('#viewport canvas');
    await expect(canvas, 'the builtin live view should already be mounted').toBeVisible();

    await page.click('[data-backend="wasm"]');
    await expect(page.locator('[data-backend="wasm"]')).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#backend-badge')).toHaveText('wasm');
    // The veil only lifts on a successful mount, so a kernel that failed to
    // instantiate shows up here rather than as an empty viewport.
    await expect(page.locator('#loading')).toBeHidden({ timeout: 60_000 });
    await expect.poll(async () => (await report(page)).liveBackend).toBe('wasm');

    // The run must actually advance: a mounted-but-frozen loop looks identical
    // to a working one in a screenshot.
    await expect
      .poll(
        async () => {
          const text = (await page.locator('#step-badge').textContent()) ?? '';
          return Number(/step (\d+)/.exec(text)?.[1] ?? -1);
        },
        { timeout: 30_000 },
      )
      .toBeGreaterThan(10);

    const first = await canvas.screenshot();
    await page.waitForTimeout(400);
    const second = await canvas.screenshot();
    expect(first.equals(second), 'the wasm-driven scene should be animating').toBe(false);

    expect(errors, 'no page or console errors from the wasm live view').toEqual([]);
  });

  test('pause holds the live run and resume continues it', async ({ page }) => {
    const errors = watchErrors(page);
    await openAndWait(page);
    await page.click('[data-backend="wasm"]');
    await expect(page.locator('#loading')).toBeHidden({ timeout: 60_000 });

    const badge = page.locator('#step-badge');
    await page.click('#pause-btn');
    await expect(page.locator('#pause-btn .btn-label')).toHaveText('Resume');
    const held = await badge.textContent();
    await page.waitForTimeout(500);
    expect(await badge.textContent(), 'a paused run must not step').toBe(held);

    await page.click('#pause-btn');
    await expect(page.locator('#pause-btn .btn-label')).toHaveText('Pause');
    await expect.poll(async () => badge.textContent(), { timeout: 20_000 }).not.toBe(held);

    expect(errors, 'no page or console errors').toEqual([]);
  });
});

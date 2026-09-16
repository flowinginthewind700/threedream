/**
 * Browser spec for the render layer and the demo that hosts it.
 *
 * What is asserted here is exactly what unit tests cannot reach: a real WebGL
 * context produces a canvas with actual pixels, the render loop keeps mirroring
 * physics state into scene objects, and the in-page trainer learns. If `render/`
 * regresses, this file is the only thing that goes red.
 */

import { expect, test, type Page } from '@playwright/test';

import { distinctColors } from './pixels.js';

/** The demo serves from a subpath when built for Pages; resolve either way. */
const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173/threedream';

async function openDemo(page: Page): Promise<void> {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console: ${m.text()}`);
  });
  (page as unknown as { __errors: string[] }).__errors = errors;
  await page.goto(`${BASE}/`, { waitUntil: 'load' });
  // The demo builds its scene asynchronously; wait for the loading veil to go.
  await expect(page.locator('#loading')).toBeHidden({ timeout: 30_000 });
}

function errorsOf(page: Page): string[] {
  return (page as unknown as { __errors: string[] }).__errors ?? [];
}

test.describe('the demo renders', () => {
  test('produces a WebGL canvas that is not blank', async ({ page }) => {
    await openDemo(page);

    const canvas = page.locator('#viewport canvas');
    await expect(canvas).toBeVisible();

    // A canvas can be present and still be cleared to one flat colour if the
    // scene never drew. Sample the rendered pixels via a screenshot of the
    // element and require real variation; see `distinctColors` for why the
    // screenshot and not the live canvas is the thing that carries them.
    const shot = await canvas.screenshot();
    expect(shot.length, 'canvas screenshot should carry pixel data').toBeGreaterThan(5_000);

    const unique = await distinctColors(page, shot);
    expect(unique, 'scene should contain more than one colour').toBeGreaterThan(8);

    expect(errorsOf(page), 'no page or console errors').toEqual([]);
  });

  test('the simulation advances over time', async ({ page }) => {
    await openDemo(page);

    // Two screenshots of the same canvas must differ: the agent is moving under
    // a live policy, so a frozen frame means the render loop stopped mirroring
    // physics state.
    const canvas = page.locator('#viewport canvas');
    const first = await canvas.screenshot();
    await page.waitForTimeout(700);
    const second = await canvas.screenshot();

    expect(first.equals(second), 'canvas should change between frames').toBe(false);
    expect(errorsOf(page), 'no page or console errors').toEqual([]);
  });

  test('reports a live simulation frame rate', async ({ page }) => {
    await openDemo(page);
    const fps = page.locator('#stat-fps');
    await expect(fps).not.toHaveText('\u2014', { timeout: 20_000 });
    const value = Number(await fps.textContent());
    // SwiftShader is slow, but a running loop still clears single digits.
    expect(Number.isFinite(value)).toBe(true);
    expect(value, 'render loop should be ticking').toBeGreaterThan(3);
  });
});

test.describe('in-page training', () => {
  test('trains episodes and updates the stats it claims to', async ({ page }) => {
    await openDemo(page);

    const episodes = page.locator('#stat-episodes');
    const updates = page.locator('#stat-updates');
    await expect(episodes).toHaveText('0');

    await page.click('#train-btn');
    // The button flips to Stop while training; that is the visible state change.
    await expect(page.locator('#train-btn .btn-label')).toHaveText('Stop');

    await expect
      .poll(async () => Number(await episodes.textContent()), { timeout: 60_000 })
      .toBeGreaterThan(0);
    await expect
      .poll(async () => Number(await updates.textContent()), { timeout: 60_000 })
      .toBeGreaterThan(0);

    // A batch return should have been computed and painted.
    const batchReturn = await page.locator('#stat-return').textContent();
    expect(batchReturn).not.toBe('\u2014');
    expect(Number.isFinite(Number(batchReturn))).toBe(true);

    await page.click('#train-btn'); // stop
    await expect(page.locator('#train-btn .btn-label')).toHaveText('Train');
    expect(errorsOf(page), 'no page or console errors').toEqual([]);
  });

  test('switching task resets the training counters', async ({ page }) => {
    await openDemo(page);

    await page.click('#train-btn');
    await expect
      .poll(async () => Number(await page.locator('#stat-episodes').textContent()), {
        timeout: 60_000,
      })
      .toBeGreaterThan(0);
    await page.click('#train-btn');

    await page.click('[data-task="reach"]');
    await expect(page.locator('#stat-episodes')).toHaveText('0');
    await expect(page.locator('#stat-updates')).toHaveText('0');
    // Reach is a different env, so the viewport must have been rebuilt.
    await expect(page.locator('#viewport canvas')).toBeVisible();
    expect(errorsOf(page), 'no page or console errors').toEqual([]);
  });

  test('playback mode switching is reflected in the segmented control', async ({ page }) => {
    await openDemo(page);

    await page.click('[data-mode="random"]');
    await expect(page.locator('[data-mode="random"]')).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('[data-mode="policy"]')).toHaveAttribute('aria-selected', 'false');

    await page.click('[data-mode="policy"]');
    await expect(page.locator('[data-mode="policy"]')).toHaveAttribute('aria-selected', 'true');
    expect(errorsOf(page), 'no page or console errors').toEqual([]);
  });

  test('pause stops the simulation and resume restarts it', async ({ page }) => {
    await openDemo(page);
    const canvas = page.locator('#viewport canvas');

    await page.click('#pause-btn');
    await expect(page.locator('#pause-btn .btn-label')).toHaveText('Resume');
    const pausedA = await canvas.screenshot();
    await page.waitForTimeout(500);
    const pausedB = await canvas.screenshot();
    expect(pausedA.equals(pausedB), 'paused canvas must not change').toBe(true);

    await page.click('#pause-btn');
    await expect(page.locator('#pause-btn .btn-label')).toHaveText('Pause');
    const resumedA = await canvas.screenshot();
    await page.waitForTimeout(700);
    const resumedB = await canvas.screenshot();
    expect(resumedA.equals(resumedB), 'resumed canvas must change').toBe(false);
    expect(errorsOf(page), 'no page or console errors').toEqual([]);
  });
});

/**
 * The nav strip, asserted against the built site rather than the dev server.
 *
 * This is a browser spec and not a unit test because the property worth having
 * only exists in a browser under the deployment's URL shape: the site is served
 * from `https://<user>.github.io/<repo>/` on Pages and from the domain root
 * locally, and an href that is domain-absolute works in exactly one of the two.
 * `tests/demo_nav.test.ts` covers the pure half (the page list, the pathname
 * logic, and that the list matches what vite builds); what is left is that five
 * real pages really do render the strip, mark the right link, and navigate.
 */
test.describe('the five pages navigate to each other', () => {
  /** Path relative to BASE, and the label its own nav link must carry. */
  const PAGES = [
    { path: '/', label: 'Trainer' },
    { path: '/physics-check.html', label: 'Physics check' },
    { path: '/shared-device.html', label: 'Shared device' },
    { path: '/particles.html', label: 'Particles' },
    { path: '/soft.html', label: 'Soft bodies' },
  ] as const;

  test('every page carries the strip, marks itself current, and links relatively', async ({
    page,
  }) => {
    for (const p of PAGES) {
      await page.goto(`${BASE}${p.path}`, { waitUntil: 'load' });
      const where = `on ${p.path}`;

      const nav = page.locator('.pagenav');
      await expect(nav, `nav ${where}`).toBeVisible();
      await expect(nav, `nav is labelled ${where}`).toHaveAttribute('aria-label', 'Demos');

      const links = nav.locator('a');
      await expect(links, `five links ${where}`).toHaveCount(PAGES.length);

      const hrefs = await links.evaluateAll((els) =>
        els.map((el) => el.getAttribute('href') ?? ''),
      );
      for (const href of hrefs) {
        // A leading slash is the Pages bug class: it asks the domain root for a
        // page that lives under a subpath. It would pass under `npm run dev`.
        expect(href.startsWith('/'), `domain-absolute href "${href}" ${where}`).toBe(false);
        expect(href.length, `empty href ${where}`).toBeGreaterThan(0);
      }

      const current = nav.locator('a[aria-current="page"]');
      await expect(current, `exactly one current link ${where}`).toHaveCount(1);
      await expect(current, `the current link ${where}`).toHaveText(p.label);

      // "You are here" has to be true and not merely labelled: the link marked
      // current must resolve to the URL the browser is actually on. Resolving
      // against `page.url()` keeps this correct under any base.
      const currentHref = await current.getAttribute('href');
      expect(
        new URL(currentHref ?? '', page.url()).href,
        `current link resolves to this page ${where}`,
      ).toBe(page.url());

      // The five hrefs must resolve to five distinct URLs, exactly one of which
      // is the page being looked at. Without this the strip could be five copies
      // of one link, or four links plus a dead one, and both look fine by eye.
      const resolved = hrefs.map((href) => new URL(href, page.url()).href);
      expect(new Set(resolved).size, `five distinct targets ${where}`).toBe(PAGES.length);
      expect(
        resolved.filter((url) => url === page.url()).length,
        `exactly one link resolves to this page ${where}`,
      ).toBe(1);
    }
  });

  test('a nav click lands on another page and that page renders', async ({ page }) => {
    await page.goto(`${BASE}/`, { waitUntil: 'load' });
    await expect(page.locator('#loading')).toBeHidden({ timeout: 30_000 });

    await page.locator('.pagenav a', { hasText: 'Soft bodies' }).click();
    await expect(page).toHaveURL(/soft\.html$/);
    await expect(page.locator('.pagenav a[aria-current="page"]')).toHaveText('Soft bodies');

    // Arriving is not the same as working. The other four pages are demos rather
    // than documents, so the claim the nav makes -- "there is more here" -- is
    // only honest if the page it lands on lifts its loading veil and draws.
    await expect(page.locator('#loading')).toBeHidden({ timeout: 30_000 });
    await expect(page.locator('#viewport canvas')).toBeVisible();

    // And the strip survives the trip, so the nav is not one-directional.
    await page.locator('.pagenav a', { hasText: 'Trainer' }).click();
    await expect(page.locator('.pagenav a[aria-current="page"]')).toHaveText('Trainer');
    await expect(page.locator('#viewport canvas')).toBeVisible();
  });
});

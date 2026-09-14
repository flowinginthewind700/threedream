/**
 * Browser spec for the render layer and the demo that hosts it.
 *
 * What is asserted here is exactly what unit tests cannot reach: a real WebGL
 * context produces a canvas with actual pixels, the render loop keeps mirroring
 * physics state into scene objects, and the in-page trainer learns. If `render/`
 * regresses, this file is the only thing that goes red.
 */

import { expect, test, type Page } from '@playwright/test';

/**
 * Decode a PNG screenshot into distinct RGB triples, inside the page.
 *
 * This is the only readback path that works on this renderer. `scene.ts` creates
 * its WebGL context with `preserveDrawingBuffer: false`, so by the time any JS
 * runs the drawing buffer has been cleared for compositing: `canvas.toDataURL()`,
 * `ctx.drawImage(canvas, ...)`, and `gl.readPixels()` all return exactly one
 * flat colour even while the page is visibly animating (measured: 1 distinct
 * colour from all three, 1512 from the decoded screenshot, same frame). Reading
 * the compositor's own capture is what actually contains the pixels.
 *
 * Decoding via an in-page `<img>` keeps this dependency-free: no pngjs, no
 * sharp, just the browser's own PNG decoder.
 */
async function distinctColors(page: Page, png: Buffer): Promise<number> {
  return page.evaluate(async (b64: string) => {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('screenshot did not decode'));
      img.src = `data:image/png;base64,${b64}`;
    });
    const off = document.createElement('canvas');
    off.width = img.naturalWidth;
    off.height = img.naturalHeight;
    const ctx = off.getContext('2d');
    if (!ctx) throw new Error('no 2d context');
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, off.width, off.height).data;
    const seen = new Set<string>();
    for (let i = 0; i < data.length; i += 4) {
      seen.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
    }
    return seen.size;
  }, png.toString('base64'));
}

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

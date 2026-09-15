/**
 * Browser spec for the M4 soft-body layer, on the tiers that need no GPU.
 *
 * `soft_gpu.spec.ts` owns the WebGPU claims; this file owns the other half of the
 * acceptance criteria, and the half that is easiest to believe without testing:
 *
 *   - "WebGL2 机器上仍能跑" -- and on this page the downgrade is a real change of
 *     renderer over an *unchanged* solver, which makes the claim testable exactly
 *     rather than approximately: the WebGL2 and CPU tiers both run the
 *     deterministic reference, so the same scene, seed and step count must produce
 *     the same digest on both. That is "the reference is a specification, not a
 *     fallback" as an assertion.
 *   - The plan -- islands, colors, workgroups, dispatches -- is one function of the
 *     mesh, built once in `buildSoftLayout`, and no tier gets its own opinion of it.
 *   - "渲染帧率与仿真步长解耦" -- over a second of wall clock the simulation
 *     advances by `1/fixedDt` steps, whatever the display managed to draw.
 *   - All four scenes draw, each in the mode its mesh allows. Cloth and sheets have
 *     triangles; a cube lattice and a rope do not, and `SoftView` refuses to invent
 *     a surface for them, so those two draw the constraint graph instead.
 *
 * Runs in the `chromium` project (SwiftShader), which has no WebGPU adapter and is
 * therefore exactly the environment the fallback tiers exist for. The tier is pinned
 * by URL rather than left to the probe, so a runner that does have an adapter still
 * tests the tier under the same name.
 */

import { expect, test, type Page } from '@playwright/test';

import type { SoftReport } from '../demo/soft.js';
import { SOFT_WORKGROUP_SIZE } from '../src/gpu/softIslands.js';
import { SOFT_FIXED_DISPATCHES } from '../src/gpu/softOptions.js';

import { distinctColors } from './pixels.js';

/** The demo serves from a subpath when built for Pages; resolve either way. */
const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173/threedream';
const PAGE_URL = `${BASE}/soft.html`;

/**
 * Small enough that SwiftShader stays quick, big enough that the lattice is deeper
 * than one workgroup -- 1000 nodes is 16 workgroups of 64, so a mapping bug that
 * only shows up past the first workgroup still shows up.
 */
const COUNT = 1000;
const SEED = 7;

/** The page's default, asserted rather than assumed: see the plan test. */
const DEFAULT_ITERATIONS = 8;

/** `hex:count`, the shape `SoftMesh.digest()` returns. */
const DIGEST = /^[0-9a-f]{8,}:\d+$/;

/** What each scene's mesh allows `SoftView` to draw. */
const SCENES = [
  { scene: 'cloth', drawMode: 'surface' },
  { scene: 'sheets', drawMode: 'surface' },
  { scene: 'cube', drawMode: 'edges' },
  { scene: 'rope', drawMode: 'edges' },
] as const;

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

async function report(page: Page): Promise<SoftReport> {
  const value = await page.evaluate(
    () => (window as unknown as { __soft?: SoftReport }).__soft,
  );
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
 * `status !== 'booting'` rather than the wanted value, so an `error` outcome
 * reaches the assertions with its message attached instead of becoming a timeout
 * that says nothing about why.
 */
async function open(page: Page, query: string, timeout = 90_000): Promise<SoftReport> {
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

/** What a tier with no `GPUDevice` must look like, on every row that could lie. */
function expectNoDevice(r: SoftReport): void {
  expect(r.deviceShared, `no device was acquired, so none can be shared:\n${dump(r)}`).toBe(false);
  // Race-freedom is a property of a parallel solve. The reference walks its color
  // batches in order, so there is no race to be free of and claiming one here would
  // be the row lying to make the page look better.
  expect(r.raceFree, `a sequential solver has nothing to be race-free about:\n${dump(r)}`).toBe(
    false,
  );
  expect(r.viewMode, `a CPU tier has nothing to blit from:\n${dump(r)}`).toBe('cpu');
  expect(r.frameMode, dump(r)).toBe('cpu-upload');
  expect(r.blitBytes, dump(r)).toBe(0);
  expect(r.backend, `three.js should not have picked WebGPU:\n${dump(r)}`).not.toBe('webgpu');
  expect(r.gpuError, `a tier with no device cannot have a GPU error:\n${dump(r)}`).toBeNull();
}

/** Whether this runner has an adapter, read off the page's own report. */
function hasAdapter(r: SoftReport): boolean {
  return r.webgpuAvailable;
}

test.describe('the soft-body layer without a GPU', () => {
  test('the WebGL2 tier simulates and draws', async ({ page }) => {
    const r = await open(page, `?tier=webgl2&scene=cloth&count=${COUNT}&seed=${SEED}`);

    expect(r.status, `the page did not come up:\n${dump(r)}`).toBe('live');
    expect(r.error, dump(r)).toBeUndefined();
    expect(r.tier, `the forced tier was not honoured:\n${dump(r)}`).toBe('webgl2');
    expect(r.count, dump(r)).toBe(COUNT);
    expect(r.plan.nodes, `the mesh is not the size it was asked for:\n${dump(r)}`).toBe(COUNT);
    expectNoDevice(r);

    const canvas = page.locator('#viewport canvas');
    await expect(canvas).toBeVisible();
    await expect(page.locator('#loading')).toBeHidden();

    // `canvasBytes` would be non-zero for a solid clear colour too; distinct
    // colours are what prove a surface was drawn. See `pixels.ts` for why the
    // screenshot carries them and the live canvas cannot.
    const unique = await distinctColors(page, await canvas.screenshot());
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
    const query = `?tier=cpu&scene=cloth&count=${COUNT}&seed=${SEED}&steps=60`;
    const first = await open(page, query);

    expect(first.status, `the scripted run did not finish:\n${dump(first)}`).toBe('done');
    expect(first.tier, dump(first)).toBe('cpu');
    expect(first.steps, dump(first)).toBe(60);
    expect(first.msPerStep, dump(first)).toBeGreaterThan(0);
    expect(first.canvasBytes, 'the final frame encoded to nothing').toBeGreaterThan(0);
    expect(first.digest, dump(first)).toMatch(DIGEST);
    expect(first.stats.escaped, `a node left the box:\n${dump(first)}`).toBe(0);
    expectNoDevice(first);

    // Replay independence, tested the only way that means anything: the same seed
    // and step count must land on the same digest, twice, with no GPU involved
    // anywhere. A renderer that wrote back into the mesh -- the thing `SoftView`
    // is careful not to do -- would break this.
    const second = await open(page, query);
    expect(second.digest, `the CPU tier is not reproducible:\n${dump(second)}`).toBe(first.digest);
    expect(second.stats, dump(second)).toEqual(first.stats);
    expect(errorsOf(page), dump(second)).toEqual([]);
  });

  test('the WebGL2 downgrade changes the renderer and nothing else', async ({ page }) => {
    const base = `scene=cloth&count=${COUNT}&seed=${SEED}&steps=30`;
    const cpu = await open(page, `?tier=cpu&${base}`);
    const gl = await open(page, `?tier=webgl2&${base}`);

    expect(cpu.status, dump(cpu)).toBe('done');
    expect(gl.status, dump(gl)).toBe('done');
    expect(gl.tier, dump(gl)).toBe('webgl2');

    // One plan, two tiers. `buildSoftLayout` is the only place either is allowed to
    // get it from, so a difference here is not a difference of opinion.
    expect(
      gl.plan,
      `the tiers disagree about the shape of the problem:\n${dump(gl)}\n${dump(cpu)}`,
    ).toEqual(cpu.plan);
    // And one solver: the WebGL2 tier runs the same reference the CPU tier does, so
    // the bytes are the same bytes. This is the assertion that makes "WebGL2 机器上
    // 仍能跑" mean something -- the cloth you get is the cloth the reference makes.
    expect(gl.digest, `the two fallback tiers diverged:\n${dump(gl)}\n${dump(cpu)}`).toBe(
      cpu.digest,
    );
    expect(gl.stats, dump(gl)).toEqual(cpu.stats);
    expect(errorsOf(page), dump(gl)).toEqual([]);
  });

  test('the plan is a function of the mesh and the knobs', async ({ page }) => {
    const r = await open(page, `?tier=cpu&scene=cloth&count=${COUNT}&seed=${SEED}&steps=4`);
    expect(r.status, dump(r)).toBe('done');

    expect(r.plan.nodes, dump(r)).toBe(COUNT);
    // A cloth is a lattice: structural, shear and bend edges, so comfortably more
    // constraints than nodes.
    expect(r.plan.constraints, `a lattice with no bend or shear edges:\n${dump(r)}`).toBeGreaterThan(
      COUNT,
    );
    // One connected cloth is one island, and the coloring needs more than one batch
    // because adjacent edges share a node.
    expect(r.plan.islands, dump(r)).toBe(1);
    expect(r.plan.colors, `adjacent edges landed in one batch:\n${dump(r)}`).toBeGreaterThan(1);
    // 64 nodes per workgroup, no subgroup dependency, and a single island means no
    // padding: the count is exactly the ceiling.
    expect(r.plan.nodeWorkgroups, dump(r)).toBe(Math.ceil(COUNT / SOFT_WORKGROUP_SIZE));
    expect(r.plan.iterations, `the page default moved:\n${dump(r)}`).toBe(DEFAULT_ITERATIONS);
    expect(r.plan.dispatchesPerStep, dump(r)).toBe(
      SOFT_FIXED_DISPATCHES + DEFAULT_ITERATIONS * r.plan.colors,
    );
    await expect(page.locator('#gr-wgsize')).toHaveText(String(SOFT_WORKGROUP_SIZE));

    // Iterations buy sweeps, and sweeps cost dispatches: one per color per sweep.
    // Asserting the relation on the page is what stops the slider from becoming a
    // control that changes a number in the HUD and nothing in the solver.
    const one = await open(page, `?tier=cpu&scene=cloth&count=${COUNT}&seed=${SEED}&steps=4&iterations=1`);
    expect(one.plan.iterations, dump(one)).toBe(1);
    expect(one.plan.dispatchesPerStep, dump(one)).toBe(SOFT_FIXED_DISPATCHES + one.plan.colors);
    expect(one.plan.colors, 'the coloring must not depend on the iteration count').toBe(
      r.plan.colors,
    );
    expect(errorsOf(page), dump(one)).toEqual([]);
  });

  test('sheets is the scene that proves the island pass', async ({ page }) => {
    const r = await open(page, `?tier=cpu&scene=sheets&count=${COUNT}&seed=${SEED}&steps=4`);
    expect(r.status, dump(r)).toBe('done');

    // One connected cloth is one island, which proves nothing about a grouping pass.
    expect(r.plan.islands, `sheets came out connected:\n${dump(r)}`).toBeGreaterThan(1);
    // Every island is accounted for exactly once, and nothing sleeps unasked.
    expect(r.stats.awakeIslands + r.stats.sleepingIslands, dump(r)).toBe(r.plan.islands);
    expect(r.stats.sleepingIslands, `sleep is off:\n${dump(r)}`).toBe(0);
    // Islands are padded to whole workgroups, so more islands means more workgroups
    // than the ceiling -- the cost of the grouping, visible in the HUD.
    expect(r.plan.nodeWorkgroups, dump(r)).toBeGreaterThanOrEqual(
      Math.ceil(COUNT / SOFT_WORKGROUP_SIZE),
    );
    await expect(page.locator('#gr-islands')).toHaveClass(/is-good/);
    expect(r.stats.escaped, dump(r)).toBe(0);
    expect(errorsOf(page), dump(r)).toEqual([]);
  });

  test('every scene draws, in the mode its mesh allows', async ({ page }) => {
    for (const { scene, drawMode } of SCENES) {
      const r = await open(page, `?tier=cpu&scene=${scene}&count=${COUNT}&seed=${SEED}&steps=8`);
      expect(r.status, `${scene} did not finish:\n${dump(r)}`).toBe('done');
      expect(r.scene, dump(r)).toBe(scene);
      expect(r.plan.nodes, `${scene} is not the size it was asked for:\n${dump(r)}`).toBe(COUNT);
      expect(
        r.drawMode,
        `${scene} drew as ${r.drawMode}, expected ${drawMode}:\n${dump(r)}`,
      ).toBe(drawMode);
      expect(r.stats.escaped, `${scene} lost a node out of the box:\n${dump(r)}`).toBe(0);
      expect(r.canvasBytes, `${scene} encoded nothing`).toBeGreaterThan(0);
      expect(r.digest, `${scene}:\n${dump(r)}`).toMatch(DIGEST);
      await expect(page.locator('#scene-badge')).toHaveText(scene);

      if (drawMode === 'surface') {
        expect(r.triangles, `${scene} has a surface and drew no triangles:\n${dump(r)}`).toBeGreaterThan(
          0,
        );
        expect(r.wire, `${scene} shows the overlay unasked:\n${dump(r)}`).toBe(false);
        await expect(page.locator('#wire-btn')).toBeEnabled();
      } else {
        // An edges scene *is* the constraint graph, so the overlay is not a choice:
        // the button is disabled rather than offering a change it cannot make.
        expect(r.wire, `${scene} should be drawing its graph:\n${dump(r)}`).toBe(true);
        await expect(page.locator('#wire-btn')).toBeDisabled();
      }
      expect(errorsOf(page), `${scene}:\n${dump(r)}`).toEqual([]);
    }
  });

  test('the step is fixed while the display is not', async ({ page }) => {
    const r = await open(page, `?tier=webgl2&scene=cloth&count=${COUNT}&seed=${SEED}`);
    expect(r.status, dump(r)).toBe('live');

    // The step size belongs to the simulation. Nothing about a browser, a refresh
    // rate or a frame time is allowed to reach it.
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
      // The clock drops a backlog rather than growing it, so a runner that cannot
      // keep up simulates fewer seconds than passed. That is the design and the HUD
      // says so; asserting wall-clock agreement here would be asserting that
      // SwiftShader is fast.
      expect(stepsDelta * later.fixedDt, dump(later)).toBeLessThanOrEqual(elapsed * 1.2);
    } else {
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
    const r = await open(page, `?scene=cloth&count=${COUNT}&seed=${SEED}`);

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
    const r = await open(page, `?tier=webgl2&scene=cloth&count=${COUNT}&seed=${SEED}`);
    expect(r.status, dump(r)).toBe('live');

    await page.click('#tier-select .seg[data-tier="cpu"]');
    await page.waitForFunction(
      () => {
        const p = (window as unknown as { __soft?: SoftReport }).__soft;
        return !!p && p.status === 'live' && p.tier === 'cpu';
      },
      undefined,
      { timeout: 90_000 },
    );
    const after = await report(page);
    expect(after.tier, dump(after)).toBe('cpu');
    expect(after.requested, dump(after)).toBe('cpu');
    expectNoDevice(after);

    // One canvas and one loop: a teardown that left the old renderer alive would
    // leave two canvases stacked in the viewport, both being drawn.
    await expect(page.locator('#viewport canvas')).toHaveCount(1);
    expect(after.steps, dump(after)).toBeGreaterThan(0);
    expect(errorsOf(page), `errors across a rebuild:\n${dump(after)}`).toEqual([]);
  });

  test('the controls rebuild the page rather than reloading it', async ({ page }) => {
    const r = await open(page, `?tier=cpu&scene=cloth&count=${COUNT}&seed=${SEED}`);
    expect(r.status, dump(r)).toBe('live');
    const url = page.url();

    // The scene picker. `sheets` is chosen because it changes the plan's island
    // count, so a rebuild that silently kept the old mesh is caught by the plan
    // rather than by a pixel diff.
    await page.click('#scene-select .seg[data-scene="sheets"]');
    await page.waitForFunction(
      () => {
        const p = (window as unknown as { __soft?: SoftReport }).__soft;
        return !!p && p.status === 'live' && p.scene === 'sheets';
      },
      undefined,
      { timeout: 90_000 },
    );
    const sheets = await report(page);
    expect(sheets.plan.islands, `the picker did not rebuild the mesh:\n${dump(sheets)}`).toBeGreaterThan(
      1,
    );
    expect(page.url(), 'a control reload is a control that lost its state').toBe(url);
    await expect(page.locator('#viewport canvas')).toHaveCount(1);

    // The iterations slider, which is a rebuild because the solver's dispatch count
    // is baked into its pipeline layout.
    await page.locator('#iterations-slider').fill('16');
    await page.waitForFunction(
      () => {
        const p = (window as unknown as { __soft?: SoftReport }).__soft;
        return !!p && p.status === 'live' && p.iterations === 16;
      },
      undefined,
      { timeout: 90_000 },
    );
    const sixteen = await report(page);
    expect(sixteen.plan.iterations, dump(sixteen)).toBe(16);
    expect(sixteen.plan.dispatchesPerStep, dump(sixteen)).toBe(
      SOFT_FIXED_DISPATCHES + 16 * sixteen.plan.colors,
    );
    await expect(page.locator('#iterations-value')).toHaveText('16');
    expect(page.url(), 'a slider reload is a slider that lost its value').toBe(url);
    expect(errorsOf(page), `errors across two rebuilds:\n${dump(sixteen)}`).toEqual([]);
  });

  test('a strict page refuses to guess', async ({ page }) => {
    // The mirror of the GPU spec's refusal test, on a runner with no adapter: asking
    // for WebGPU with `strict` must fail loudly rather than quietly solve on the CPU.
    const r = await open(page, `?tier=webgpu&strict=1&count=${COUNT}&steps=4`);
    test.skip(hasAdapter(r), 'this runner has an adapter; see soft_gpu.spec.ts');

    expect(r.status, dump(r)).toBe('error');
    expect(r.error, `no reason was reported:\n${dump(r)}`).toMatch(/cannot be honoured/i);
    await expect(page.locator('#tier-badge')).toHaveText('failed');
    await expect(page.locator('#loading')).toBeVisible();
    await expect(page.locator('#verdict')).toHaveClass(/is-warn/);
  });
});

#!/usr/bin/env node
/**
 * The M3 performance gate: how much does one simulation step cost, per particle
 * count, on the tier that is supposed to scale.
 *
 * `bench_shared_device.mjs` proved the architecture once and
 * `bench_gpu_compute.mjs` measured the compute layer in isolation. This one
 * measures the whole thing the way a visitor gets it: the real built page, the
 * real renderer, six WGSL dispatches a step, and the instance blit that feeds
 * three.js -- with no readback anywhere in the frame. The acceptance criterion
 * is "50,000-100,000 particles stable on target hardware", and stable is a
 * number, not a vibe, so this prints the number.
 *
 * It is a ladder and not a single run for a reason: a per-step cost that grows
 * roughly linearly is a simulation doing O(n) work on the device, which is the
 * design. One that jumps between rungs is a cliff -- the spatial hash table
 * saturating, a buffer crossing the driver's allocation limit, a workgroup count
 * that stops fitting -- and a cliff is the thing worth knowing about before it
 * reaches a user.
 *
 * Run:  npm run build:pages && node scripts/bench_gpu_particles.mjs
 * Args: optional `count:steps` rungs, e.g. `... 5000:60 100000:20`
 * Env:  CHROME_PATH (optional executable), BENCH_PORT (default 8892),
 *       BENCH_SEED (default 1234), BENCH_COLLISIONS (default 1)
 *
 * Exits non-zero when a rung errors, when `strict` refused the WebGPU tier (no
 * adapter, or ANGLE without Vulkan -- see the flag list below), or when a
 * particle escaped the box. Exits zero and prints the table otherwise. The
 * numbers are a floor in device terms, because headless Chromium hands
 * `requestAdapter()` whichever GPU the driver stack prefers, which on a laptop
 * with an iGPU and no Vulkan ICD for the discrete card is the integrated one.
 *
 * They are not a floor in run-to-run terms, and the table says so rather than
 * implying otherwise. `p50 ms/step` is the median of the run's chunk samples,
 * with `p95` and the `mean` printed beside it, because a mean over the four
 * chunks a 30-step rung produced was a number one contended chunk could triple:
 * the sibling soft-body gate measured the same rung at 2.97 and then 10.07
 * ms/step four minutes apart on one machine, in a run whose 20k rung reported
 * *faster* than its 10k one. `n` is the sample count, since a median is only as
 * good as the chunks behind it. Read the acceptance criterion off p50, and how
 * busy the machine was off the gap between p50 and p95.
 */
import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { chromium } from '@playwright/test';

const PORT = Number(process.env.BENCH_PORT ?? 8892);
const CHROME = process.env.CHROME_PATH;
const SEED = Number(process.env.BENCH_SEED ?? 1234);
const COLLISIONS = process.env.BENCH_COLLISIONS ?? '1';

/**
 * The ladder: `[particles, steps]`.
 *
 * Every rung runs the same number of steps, for two reasons. The fixed cost of
 * pipeline creation and buffer allocation is what the small rungs mostly show,
 * and comparing that against the large rungs only means something at a common
 * run length. And steps are the sample count: a chunk is eight steps, so 160
 * steps is twenty samples, where the 20 and 30 this ladder used to ask for were
 * three and four -- too few for a median to filter anything, which is how one
 * contended chunk came to be able to triple a rung's reported per-step cost.
 * Twenty is also the smallest count at which a nearest-rank p95 is not simply
 * the maximum. It costs about five seconds on the 100k rung and a fraction of
 * that below it.
 */
const DEFAULT_LADDER = [
  [1_000, 160],
  [10_000, 160],
  [50_000, 160],
  [100_000, 160],
];

const LADDER =
  process.argv.length > 2
    ? process.argv.slice(2).map((arg) => {
        const [count, steps] = arg.split(':').map(Number);
        if (!Number.isInteger(count) || count <= 0 || !Number.isInteger(steps) || steps <= 0) {
          throw new Error(`bad rung "${arg}", expected count:steps, e.g. 50000:30`);
        }
        return [count, steps];
      })
    : DEFAULT_LADDER;

const ROOT = resolve(import.meta.dirname, '..');
const DIST = join(ROOT, 'dist');
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.map': 'application/json',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

if (!existsSync(join(DIST, 'particles.html'))) {
  console.error('!! dist/particles.html is missing. Build the site first:');
  console.error('     npm run build:pages');
  process.exit(1);
}

/**
 * Serve `dist/` under the subpath the Pages build was made for.
 *
 * `build:pages` sets `VITE_BASE=/threedream/`, so every asset URL in the built
 * HTML starts with that prefix. Serving the same files from `/` would 404 every
 * one of them and produce a blank page -- which, on a bench, looks exactly like
 * "the GPU is slow" rather than "the server is wrong".
 */
const server = createServer((req, res) => {
  const url = (req.url ?? '/').split('?')[0];
  const rel = url.startsWith('/threedream/') ? url.slice('/threedream/'.length) : url.replace(/^\//, '');
  const file = join(DIST, rel === '' || rel === '/' ? 'index.html' : rel);
  if (!file.startsWith(DIST) || !existsSync(file)) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('not found');
    return;
  }
  res.writeHead(200, { 'Content-Type': MIME[extname(file)] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(PORT, '127.0.0.1', r));
const BASE = `http://127.0.0.1:${PORT}/threedream/particles.html`;

/** One rung's worth of numbers, read out of the page's own report object. */
async function runRung(page, count, steps) {
  const query =
    `?tier=webgpu&strict=1&count=${count}&steps=${steps}` +
    `&seed=${SEED}&collisions=${COLLISIONS}`;
  // The 100k rung compiles six pipelines and allocates ~50 MB of buffers before
  // the first step; on SwiftShader-as-Vulkan that is minutes, not seconds.
  const timeout = 600_000;
  await page.goto(`${BASE}${query}`, { waitUntil: 'load', timeout });
  await page.waitForFunction(
    () => {
      const r = window.__particles;
      return !!r && r.status !== 'booting';
    },
    undefined,
    { timeout },
  );
  return page.evaluate(() => window.__particles);
}

const header = [
  'particles',
  'steps',
  'n',
  'p50 ms/step',
  'p95',
  'mean',
  'us/step/particle',
  'ms/frame',
  'blit KiB',
  'draws',
  'tris',
  'escaped',
];
const rows = [header];

let exitCode = 0;
try {
  const browser = await chromium.launch({
    executablePath: CHROME && existsSync(CHROME) ? CHROME : undefined,
    args: [
      '--headless=new',
      '--no-sandbox',
      '--ignore-gpu-blocklist',
      // `--enable-unsafe-swiftshader` is what lets ANGLE's Vulkan backend fall
      // back to SwiftShader's ICD on a machine with no GPU at all. Where real
      // hardware exists it changes nothing: the fallback is only consulted once
      // the real adapter has been ruled out.
      '--enable-unsafe-swiftshader',
      '--enable-features=Vulkan,DefaultANGLEVulkan,WebGPUService',
      `--use-angle=${process.env.BENCH_ANGLE ?? 'vulkan'}`,
    ],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message.slice(0, 400)));
  page.on('console', (m) => {
    if (m.type() === 'error') console.error('[console]', m.text().slice(0, 300));
  });

  for (const [count, steps] of LADDER) {
    const r = await runRung(page, count, steps);
    if (r.status === 'error') {
      console.error(`!! ${count} particles: ${r.error}`);
      console.error(`   log:\n     ${(r.lines ?? []).join('\n     ')}`);
      exitCode = 1;
      break;
    }
    if (r.status !== 'done') {
      console.error(`!! ${count} particles ended in status "${r.status}", not "done"`);
      exitCode = 1;
      break;
    }
    // `strict=1` means a downgrade is impossible: either the tier is webgpu or
    // the page threw, and the throw was handled above. Asserting anyway keeps a
    // future change to the demo honest.
    if (r.tier !== 'webgpu' || r.frameMode !== 'gpu-blit') {
      console.error(`!! ${count} particles ran on ${r.tier}/${r.frameMode}, not webgpu/gpu-blit`);
      // The view falls back to a CPU upload rather than throwing when a GPU pass
      // fails, so the reason only lives in the report. Without it this line says
      // "it was slow" and not "the instance expander could not be created".
      if (r.gpuError) console.error(`   gpuError: ${r.gpuError}`);
      console.error('   log:');
      for (const line of r.lines ?? []) console.error(`     ${line}`);
      exitCode = 1;
      break;
    }
    if (r.stats.escaped > 0) {
      console.error(`!! ${count} particles: ${r.stats.escaped} escaped the bounds`);
      exitCode = 1;
    }
    rows.push([
      String(r.count),
      String(r.steps),
      String(r.stepSamples),
      r.msPerStep.toFixed(3),
      r.msPerStepP95.toFixed(3),
      r.msPerStepMean.toFixed(3),
      ((r.msPerStep * 1000) / r.count).toFixed(3),
      r.msPerFrame.toFixed(2),
      (r.blitBytes / 1024).toFixed(0),
      String(r.drawCalls),
      String(r.triangles),
      String(r.stats.escaped),
    ]);
  }

  // Pad every column to its widest cell, so the table stays readable when the
  // rung sizes differ by two orders of magnitude.
  const widths = header.map((_, c) => Math.max(...rows.map((row) => row[c].length)));
  console.log('');
  console.log(`WebGPU particle step cost (seed ${SEED}, collisions ${COLLISIONS})`);
  for (const [i, row] of rows.entries()) {
    console.log(
      row.map((cell, c) => (i === 0 ? cell.padEnd(widths[c]) : cell.padStart(widths[c]))).join('  '),
    );
    if (i === 0) console.log(widths.map((w) => '-'.repeat(w)).join('  '));
  }
  console.log('');
  await browser.close();
} catch (error) {
  console.error(`!! bench failed: ${error instanceof Error ? error.message : String(error)}`);
  exitCode = 1;
} finally {
  server.close();
}
process.exit(exitCode);

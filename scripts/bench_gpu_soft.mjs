#!/usr/bin/env node
/**
 * The M4 performance gate: how much does one soft-body step cost, per node count,
 * on the tier that is supposed to scale.
 *
 * `bench_gpu_particles.mjs` measures the particle layer the same way, and this is
 * the same measurement over a harder problem. The step here is
 * `SOFT_FIXED_DISPATCHES + iterations * colors` dispatches rather than six; the
 * solve is a Gauss-Seidel sweep whose *ordering* is the only barrier the pipeline
 * has; and the frame ends in a blit of the publish buffer into the `GPUBuffer`
 * three.js already allocated for the position attribute, with no readback feeding
 * it. The acceptance criterion is "GPU 层能稳定处理 10,000 级别的软体或布料粒子", and
 * stable is a number rather than a vibe, so this prints the number.
 *
 * It is a ladder and not a single run for the reason the particle bench is one: a
 * per-step cost that grows roughly linearly is a solver doing O(n) work on the
 * device, which is the design. One that jumps between rungs is a cliff -- a color
 * batch that stops fitting, a buffer crossing the driver's allocation limit, a
 * workgroup count that stops being efficient -- and a soft body has one more place
 * to fall off than a particle system does, because `colors` grows with the graph
 * and every color is `iterations` more dispatches. The two columns that make a jump
 * readable are therefore in the table: dispatches a step and colors.
 *
 * Run:  npm run build:pages && node scripts/bench_gpu_soft.mjs
 * Args: optional `count:steps` rungs, e.g. `... 10000:120 20000:30`
 * Env:  CHROME_PATH (optional executable), BENCH_PORT (default 8893),
 *       BENCH_SEED (default 4242), BENCH_SCENE (default cloth),
 *       BENCH_ITERATIONS (default 8), BENCH_ANGLE (default vulkan)
 *
 * Exits non-zero when a rung errors, when `strict` refused the WebGPU tier (no
 * adapter, or ANGLE without Vulkan), when the backend reports its solve is not
 * race-free, when the frame did not blit, or when a node escaped the box. Exits
 * zero and prints the table otherwise. The numbers are a floor in device terms,
 * because headless Chromium hands `requestAdapter()` whichever GPU the driver
 * stack prefers, which on a laptop with an iGPU and no Vulkan ICD for the
 * discrete card is the integrated one.
 *
 * They are not a floor in run-to-run terms, and the table says so rather than
 * implying otherwise. `p50 ms/step` is the median of the run's chunk samples,
 * with `p95` and the `mean` printed beside it, because a mean over the four
 * chunks a 30-step rung produced was a number one contended chunk could triple:
 * the same rung on the same machine read 2.97 and then 10.07 ms/step four
 * minutes apart, in a run whose 20k rung reported *faster* than its 10k one.
 * `n` is the sample count, since a median is only as good as the chunks behind
 * it. Read the acceptance criterion off p50, and how busy the machine was off
 * the gap between p50 and p95.
 *
 * The `stretch %` column is `stats.maxConstraintError * 100`, and it is a
 * convergence diagnostic rather than a gate: nothing here fails on it. One
 * Gauss-Seidel sweep propagates a correction about one row of the mesh, so a
 * 100x100 cloth hung from its top edge needs on the order of 100 sweeps before
 * the hem stops stretching -- which is what a double-digit percentage at 5k-20k
 * nodes is reporting, at 8 iterations per step. The column exists so that number
 * is visible next to the cost of producing it, since "the step is fast" and "the
 * step converged" are different claims and only the first one is this bench's job.
 * The gate on correctness is `raceFree` plus `escaped`, and the gate on the
 * physics is the CPU/GPU parity assertion in `e2e/soft_gpu.spec.ts`.
 */
import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { chromium } from '@playwright/test';

const PORT = Number(process.env.BENCH_PORT ?? 8893);
const CHROME = process.env.CHROME_PATH;
const SEED = Number(process.env.BENCH_SEED ?? 4242);
const SCENE = process.env.BENCH_SCENE ?? 'cloth';
const ITERATIONS = Number(process.env.BENCH_ITERATIONS ?? 8);

/**
 * The ladder: `[nodes, steps]`.
 *
 * Every rung runs the same number of steps, for two reasons. The fixed cost of
 * pipeline creation, buffer allocation and the two graph passes is what the small
 * rungs mostly show, and comparing that against the large rungs only means
 * something at a common run length. And steps are the sample count: a chunk is
 * eight steps, so 160 steps is twenty samples, where the 20 and 30 this ladder
 * used to ask for were three and four -- too few for a median to filter anything,
 * which is how one contended chunk came to be able to triple a rung's reported
 * per-step cost. Twenty is also the smallest count at which a nearest-rank p95 is
 * not simply the maximum, and it costs about a second a rung.
 */
const DEFAULT_LADDER = [
  [1_000, 160],
  [5_000, 160],
  [10_000, 160],
  [20_000, 160],
];

const LADDER =
  process.argv.length > 2
    ? process.argv.slice(2).map((arg) => {
        const [count, steps] = arg.split(':').map(Number);
        if (!Number.isInteger(count) || count <= 0 || !Number.isInteger(steps) || steps <= 0) {
          throw new Error(`bad rung "${arg}", expected count:steps, e.g. 10000:30`);
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

if (!existsSync(join(DIST, 'soft.html'))) {
  console.error('!! dist/soft.html is missing. Build the site first:');
  console.error('     npm run build:pages');
  process.exit(1);
}

/**
 * Serve `dist/` under the subpath the Pages build was made for.
 *
 * `build:pages` sets `VITE_BASE=/threedream/`, so every asset URL in the built HTML
 * starts with that prefix. Serving the same files from `/` would 404 every one of
 * them and produce a blank page -- which, on a bench, looks exactly like "the GPU is
 * slow" rather than "the server is wrong".
 */
const server = createServer((req, res) => {
  const url = (req.url ?? '/').split('?')[0];
  const rel = url.startsWith('/threedream/')
    ? url.slice('/threedream/'.length)
    : url.replace(/^\//, '');
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
const BASE = `http://127.0.0.1:${PORT}/threedream/soft.html`;

/**
 * One rung's worth of numbers, read out of the page's own report object.
 *
 * `steps=N` puts the page in its scripted mode, which submits work in chunks and
 * awaits `queue.onSubmittedWorkDone()` on each one. That is the difference between
 * measuring the GPU and measuring how long it took to record commands: the callback
 * on an empty queue returns immediately, and a number from an empty queue is the one
 * measurement that makes a GPU look as fast as a CPU.
 */
async function runRung(page, count, steps) {
  const query =
    `?tier=webgpu&strict=1&scene=${SCENE}&count=${count}&steps=${steps}` +
    `&seed=${SEED}&iterations=${ITERATIONS}&sleep=0&wire=0`;
  // The 20k rung compiles six pipelines, runs both graph passes and allocates every
  // buffer before the first step; on SwiftShader-as-Vulkan that is minutes.
  const timeout = 600_000;
  await page.goto(`${BASE}${query}`, { waitUntil: 'load', timeout });
  await page.waitForFunction(
    () => {
      const r = window.__soft;
      return !!r && r.status !== 'booting';
    },
    undefined,
    { timeout },
  );
  return page.evaluate(() => window.__soft);
}

const header = [
  'nodes',
  'steps',
  'n',
  'p50 ms/step',
  'p95',
  'mean',
  'us/step/node',
  'ms/submit',
  'dispatches',
  'colors',
  'blit KiB',
  'draws',
  'tris',
  'stretch %',
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
      // `--enable-unsafe-swiftshader` is what lets ANGLE's Vulkan backend fall back
      // to SwiftShader's ICD on a machine with no GPU at all. Where real hardware
      // exists it changes nothing: the fallback is only consulted once the real
      // adapter has been ruled out.
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
      console.error(`!! ${count} nodes: ${r.error}`);
      console.error(`   log:\n     ${(r.lines ?? []).join('\n     ')}`);
      exitCode = 1;
      break;
    }
    if (r.status !== 'done') {
      console.error(`!! ${count} nodes ended in status "${r.status}", not "done"`);
      exitCode = 1;
      break;
    }
    // `strict=1` means a downgrade is impossible: either the tier is webgpu or the
    // page threw, and the throw was handled above. Asserting anyway keeps a future
    // change to the demo honest.
    if (r.tier !== 'webgpu' || r.frameMode !== 'gpu-blit') {
      console.error(`!! ${count} nodes ran on ${r.tier}/${r.frameMode}, not webgpu/gpu-blit`);
      // The view falls back to a CPU upload rather than throwing when a GPU pass
      // fails, so the reason only lives in the report. Without it this line says
      // "it was slow" and not "three.js exposed no buffer to blit into".
      if (r.gpuError) console.error(`   gpuError: ${r.gpuError}`);
      console.error('   log:');
      for (const line of r.lines ?? []) console.error(`     ${line}`);
      exitCode = 1;
      break;
    }
    // The engine's own flag, read off the backend rather than inferred from the tier
    // name. A bench that timed a solve which was silently racing would be measuring
    // the thing this milestone exists to rule out.
    if (!r.raceFree) {
      console.error(`!! ${count} nodes: the ${r.tier} backend reports raceFree=false`);
      exitCode = 1;
      break;
    }
    if (r.stats.escaped > 0) {
      console.error(`!! ${count} nodes: ${r.stats.escaped} escaped the bounds`);
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
      String(r.plan.dispatchesPerStep),
      String(r.plan.colors),
      (r.blitBytes / 1024).toFixed(0),
      String(r.drawCalls),
      String(r.triangles),
      (r.stats.maxConstraintError * 100).toFixed(1),
      String(r.stats.escaped),
    ]);
  }

  // Pad every column to its widest cell, so the table stays readable when the rung
  // sizes differ by an order of magnitude.
  const widths = header.map((_, c) => Math.max(...rows.map((row) => row[c].length)));
  console.log('');
  console.log(
    `WebGPU soft-body step cost (${SCENE}, ${ITERATIONS} iterations, seed ${SEED})`,
  );
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

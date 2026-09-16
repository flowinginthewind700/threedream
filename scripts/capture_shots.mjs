#!/usr/bin/env node
/**
 * Regenerates every screenshot the README and the share cards use, from the
 * built site.
 *
 * The images are committed, so without this they are a claim nobody can check:
 * a shot of the particle page at 100k could be six months old, from a tier the
 * browser no longer picks, or of a layout that has since changed. This script is
 * the answer to "is that still what it looks like" -- run it after a visual
 * change and `git diff --stat` says whether the site moved. Same discipline as
 * `wasm/pkg`: an artifact in the repo plus the tool that mints it.
 *
 * They are real captures of the real pages, not illustrations. Every shot is
 * gated on the page's own report before the shutter: `strict=1` on the two scale
 * pages, so a shot that silently fell back to the CPU tier fails the run instead
 * of publishing a picture of the fallback under a WebGPU caption, and the
 * trainer shot waits for episodes to actually train rather than for a timer.
 * A screenshot is the one artifact in this repo that can lie by looking right.
 *
 * Run:  npm run build:pages && node scripts/capture_shots.mjs
 * Args: optional filter, e.g. `node scripts/capture_shots.mjs og/particles`
 * Env:  CHROME_PATH (optional executable), SHOT_PORT (default 8894),
 *       SHOT_ANGLE (default vulkan), SHOT_QUALITY (default 78),
 *       SHOT_EPISODES (default 80)
 *
 * Exits non-zero when a page errored, landed on the wrong tier, or produced a
 * file too small to be a rendered frame.
 */
import { createServer } from 'node:http';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { dirname, extname, join, resolve } from 'node:path';
import { chromium } from '@playwright/test';

const PORT = Number(process.env.SHOT_PORT ?? 8894);
const CHROME = process.env.CHROME_PATH;
const QUALITY = Number(process.env.SHOT_QUALITY ?? 78);
/** Episodes the trainer shot waits for: enough that the sparkline has a trend. */
const EPISODES = Number(process.env.SHOT_EPISODES ?? 80);

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
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

if (!existsSync(join(DIST, 'index.html'))) {
  console.error('!! dist/index.html is missing. Build the site first:');
  console.error('     npm run build:pages');
  process.exit(1);
}

/**
 * Serve `dist/` under the subpath the Pages build was made for.
 *
 * Same reason the benches serve it this way: `build:pages` sets
 * `VITE_BASE=/threedream/`, so from `/` every asset 404s and the result is a
 * blank page -- which in a screenshot is indistinguishable from "the GPU did
 * not draw anything", the one failure this script exists to catch.
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
const BASE = `http://127.0.0.1:${PORT}/threedream`;

/**
 * README gallery size, matching the shot already in `docs/`.
 *
 * The three README images sit in one column, so they share one aspect rather
 * than each being whatever viewport happened to be open.
 */
const README_SHOT = { width: 1000, height: 625 };
/**
 * Share-card size: the 1.91:1 that `summary_large_image` and every unfurler
 * crops to. Captured at this viewport rather than cropped afterwards, because a
 * crop of a 1000x625 page shot loses the sidebar -- the half of the page where
 * the numbers are.
 */
const OG_SHOT = { width: 1200, height: 630 };

/**
 * Wait for the page's own report to reach the state the shot wants, and hand the
 * report back for the gate to read.
 *
 * `error` satisfies the wait too. Waiting only for the wanted state turns a page
 * that failed into a ten-minute timeout with nothing in it, where the report
 * already carries the reason.
 */
async function settled(page, globalName, want = 'live') {
  await page.waitForFunction(
    ([name, wanted]) => {
      const r = window[name];
      return !!r && (r.status === wanted || r.status === 'error');
    },
    [globalName, want],
    // The GPU pages compile their pipelines through ANGLE before the first frame,
    // and a cold SwiftShader-as-Vulkan fallback is minutes, not seconds.
    { timeout: 600_000 },
  );
  return page.evaluate((name) => window[name], globalName);
}

/**
 * Wait for the blit path to engage on the two scale pages, then hand back the
 * report.
 *
 * `frameMode` describes one frame, and the frames right after boot can still be
 * served from a CPU upload while the compute attach settles -- the live-page test
 * in `e2e/particles_gpu.spec.ts` polls for exactly this reason. Reading it the
 * moment the page reports `live` would gate on a transition instead of a state,
 * and fail a shot of a page that is working.
 */
async function settledOnDevice(page, globalName) {
  const first = await settled(page, globalName);
  if (first.status === 'error') return first;
  await page.waitForFunction(
    (name) => {
      const r = window[name];
      return r.frameMode === 'gpu-blit' || r.status === 'error';
    },
    globalName,
    { timeout: 120_000 },
  );
  return page.evaluate((name) => window[name], globalName);
}

/**
 * Put the trainer in the state worth photographing: training, with a sparkline
 * long enough to have a trend in it.
 *
 * Waiting on the episode counter rather than on a timer, because the shot is
 * captioned "mid-training" and on a slow runner a fixed wait publishes an empty
 * sparkline under that caption.
 */
async function trainForAWhile(page) {
  await page.locator('#loading').waitFor({ state: 'hidden', timeout: 120_000 });
  await page.click('#train-btn');
  await page.waitForFunction(
    (episodes) => Number(document.querySelector('#stat-episodes')?.textContent) >= episodes,
    EPISODES,
    { timeout: 300_000 },
  );
}

/**
 * The shots, in the order they are taken.
 *
 * `ready` waits for whatever the page says it is ready, and returns the page's
 * own report object; `gate` then takes that report and returns the reason not to
 * take the picture, or null. The pair is what makes these artifacts trustworthy:
 * without the gate a regression that drops the particle page to the CPU tier
 * still produces a pretty JPEG with 100,000 points in it.
 */
const SHOTS = [
  {
    out: 'docs/demo-drive.jpg',
    url: '/',
    viewport: README_SHOT,
    alt: 'the trainer, mid-run',
    ready: trainForAWhile,
    async gate(_report, page) {
      const fps = Number(await page.locator('#stat-fps').textContent());
      if (!Number.isFinite(fps) || fps <= 0) return `sim fps reads ${fps}`;
      return null;
    },
  },
  {
    out: 'docs/demo-particles.jpg',
    url: '/particles.html?tier=webgpu&strict=1&count=100000&seed=7&collisions=1',
    viewport: README_SHOT,
    settleMs: 4_000,
    alt: 'the particle page at 100k',
    ready: (page) => settledOnDevice(page, '__particles'),
    gate: (report) => scaleGate(report, 'particles', 100_000),
  },
  {
    out: 'docs/demo-soft.jpg',
    url: '/soft.html?tier=webgpu&strict=1&scene=sheets&count=20000&iterations=8&seed=7',
    viewport: README_SHOT,
    // Four sheets rather than one cloth: a single cloth hangs in a plane the
    // camera sees edge-on and photographs as a ribbon, where the staggered hems
    // of the sheets show the solve working. A few seconds of sweeps before the
    // shutter, so they are draped rather than mid-spawn.
    settleMs: 4_000,
    alt: 'the soft-body page, 20k nodes in four sheets',
    ready: (page) => settledOnDevice(page, '__soft'),
    gate: (report) => scaleGate(report, 'soft', 20_000),
  },
  {
    out: 'demo/public/og/index.jpg',
    url: '/',
    viewport: OG_SHOT,
    alt: 'share card: the trainer',
    ready: trainForAWhile,
  },
  {
    out: 'demo/public/og/physics-check.jpg',
    url: '/physics-check.html',
    viewport: OG_SHOT,
    // The loading veil fades rather than unmounts; shooting the instant the
    // report says `done` catches it half-transparent across the viewport.
    settleMs: 900,
    alt: 'share card: the determinism check',
    ready: (page) => settled(page, '__physicsCheck', 'done'),
    gate: (report) =>
      report.status === 'done'
        ? null
        : `physics check ended "${report.status}": ${report.error ?? ''}`,
  },
  {
    out: 'demo/public/og/shared-device.jpg',
    url: '/shared-device.html',
    viewport: OG_SHOT,
    settleMs: 900,
    alt: 'share card: the shared-device check',
    ready: (page) => settled(page, '__sharedDevice', 'done'),
    gate: (report) =>
      report.status === 'done'
        ? null
        : `shared-device check ended "${report.status}": ${report.error ?? ''}`,
  },
  {
    out: 'demo/public/og/particles.jpg',
    url: '/particles.html?tier=webgpu&strict=1&count=100000&seed=7&collisions=1',
    viewport: OG_SHOT,
    settleMs: 4_000,
    alt: 'share card: the particle page at 100k',
    ready: (page) => settledOnDevice(page, '__particles'),
    gate: (report) => scaleGate(report, 'particles', 100_000),
  },
  {
    out: 'demo/public/og/soft.jpg',
    url: '/soft.html?tier=webgpu&strict=1&scene=sheets&count=20000&iterations=8&seed=7',
    viewport: OG_SHOT,
    settleMs: 4_000,
    alt: 'share card: the soft-body page',
    ready: (page) => settledOnDevice(page, '__soft'),
    gate: (report) => scaleGate(report, 'soft', 20_000),
  },
];

/**
 * The claim both scale pages are pictured making, read off the report.
 *
 * `strict=1` already refuses a downgrade, so this is the belt to its braces and
 * the part that checks the *frame* rather than the tier name: a page can be on
 * the WebGPU tier and still feed the renderer from a CPU upload when the blit
 * path fails, or draw with a second device, and either picture looks identical
 * to the one this shot exists to show.
 */
function scaleGate(report, kind, count) {
  if (!report) return `${kind}: no report`;
  if (report.status === 'error') return `${kind}: ${report.error ?? 'errored'}`;
  if (report.tier !== 'webgpu') return `${kind}: ran on ${report.tier}, pictured as webgpu`;
  if (report.frameMode !== 'gpu-blit') return `${kind}: frame mode ${report.frameMode}`;
  if (!report.deviceShared) return `${kind}: three.js drew on a different device`;
  if (report.count !== count) return `${kind}: ${report.count} elements, not ${count}`;
  if (kind === 'soft' && !report.raceFree) return 'soft: backend reports raceFree=false';
  return null;
}

const FILTER = process.argv[2];
const shots = FILTER ? SHOTS.filter((s) => s.out.includes(FILTER)) : SHOTS;
if (shots.length === 0) {
  console.error(`!! no shot matches "${FILTER}"`);
  process.exit(1);
}

let exitCode = 0;
try {
  const browser = await chromium.launch({
    executablePath: CHROME && existsSync(CHROME) ? CHROME : undefined,
    args: [
      '--headless=new',
      '--no-sandbox',
      '--ignore-gpu-blocklist',
      '--enable-unsafe-swiftshader',
      '--enable-features=Vulkan,DefaultANGLEVulkan,WebGPUService',
      `--use-angle=${process.env.SHOT_ANGLE ?? 'vulkan'}`,
    ],
  });

  for (const shot of shots) {
    const page = await browser.newPage({ viewport: shot.viewport });
    const errors = [];
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(`console: ${m.text()}`);
    });

    const label = shot.out.padEnd(30);
    try {
      await page.goto(`${BASE}${shot.url}`, { waitUntil: 'load', timeout: 600_000 });
      const report = await shot.ready?.(page);
      const blocked = await shot.gate?.(report, page);
      if (blocked) throw new Error(blocked);
      if (errors.length > 0) throw new Error(errors[0].slice(0, 300));
      if (shot.settleMs) await page.waitForTimeout(shot.settleMs);

      const out = join(ROOT, shot.out);
      await mkdir(dirname(out), { recursive: true });
      await page.screenshot({ path: out, type: 'jpeg', quality: QUALITY });

      // A blank frame still encodes, and encodes small: one flat colour is a few
      // kilobytes where a rendered scene is tens. Crude, but it is the failure
      // mode that has actually happened, and it is invisible in a diff of paths.
      const bytes = statSync(out).size;
      if (bytes < 20_000) throw new Error(`${bytes} bytes is too small to be a rendered frame`);
      console.log(`ok ${label} ${(bytes / 1024).toFixed(0)} KiB  ${shot.alt}`);
    } catch (error) {
      console.error(`!! ${label} ${error instanceof Error ? error.message : String(error)}`);
      exitCode = 1;
    } finally {
      await page.close();
    }
  }

  await browser.close();
} catch (error) {
  console.error(`!! capture failed: ${error instanceof Error ? error.message : String(error)}`);
  exitCode = 1;
} finally {
  server.close();
}
process.exit(exitCode);

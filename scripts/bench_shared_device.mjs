#!/usr/bin/env node
/**
 * Proves, in a real browser, the three claims the whole GPU architecture rests
 * on. This is an architecture gate, not a micro-benchmark: if any claim stops
 * holding in a future three.js release, the script exits non-zero.
 *
 *   1. We can own the GPUDevice ourselves and inject it into
 *      THREE.WebGPURenderer, so simulation and rendering share one device.
 *   2. A TSL compute node writes a StorageBufferAttribute, and the GPUBuffer
 *      three.js created for it stays reachable from outside the renderer.
 *   3. A RAW WGSL pipeline can bind that same three.js-managed buffer and be
 *      interleaved with renderer.render() on one device and queue. This is the
 *      bridge a native (Rust/wgpu) compute layer would cross with zero copies.
 *
 * Run:  node scripts/bench_shared_device.mjs
 * Env:  CHROME_PATH (optional executable), BENCH_PORT (default 8890)
 *
 * Headless Chromium only exposes an adapter with Vulkan flags:
 *   --headless=new --no-sandbox --ignore-gpu-blocklist
 *   --enable-features=Vulkan,DefaultANGLEVulkan,WebGPUService --use-angle=vulkan
 *
 * Note on results: `navigator.gpu.requestAdapter()` picks whatever the driver
 * stack prefers, which on a laptop with an iGPU and no Vulkan ICD for the
 * discrete card means the *integrated* GPU. Treat the numbers as a floor.
 */
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { chromium } from '@playwright/test';

const PORT = Number(process.env.BENCH_PORT ?? 8890);
const CHROME = process.env.CHROME_PATH;

// Shared with the page below and with the assertions at the bottom, so the two
// cannot disagree about what was measured.
const N = 8192;          // points
const SENTINEL = 7.5;    // written into .w by the raw kernel as proof of authorship
const FRAMES = 20;       // interleaved compute+render frames

// three.js is resolved from node_modules so the page imports the exact build
// the repo pins, and the importmap satisfies three.tsl.js's bare 'three/webgpu'.
const IMPORTMAP = {
  imports: {
    'three/webgpu': '/three/build/three.webgpu.js',
    'three/tsl': '/three/build/three.tsl.js',
    three: '/three/build/three.module.js',
  },
};

const PAGE = `<!doctype html><html><head><meta charset="utf-8">
<script type="importmap">${JSON.stringify(IMPORTMAP)}</script>
</head><body><script type="module">
import * as THREE from '/three/build/three.webgpu.js';
import { storage, compute, instanceIndex, uniform, vec4, Fn } from '/three/build/three.tsl.js';

const out = [];
const L = (...a) => out.push(a.join(' '));
window.__out = out;

// Interpolated from the host constants above so page and assertions agree.
const N = ${N};
const SENTINEL = ${SENTINEL};
const FRAMES = ${FRAMES};

window.__run = async () => {
  // ---- CLAIM 1: we create the device, three.js accepts it. ----
  const adapter = await navigator.gpu.requestAdapter({ featureLevel: 'compatibility' });
  if (!adapter) { L('NO_ADAPTER'); return; }
  const device = await adapter.requestDevice();
  const inf = adapter.info || {};
  L('adapter: vendor=' + inf.vendor + ' arch=' + inf.architecture);
  L('three REVISION: ' + THREE.REVISION);
  // Feature gaps that constrain kernel design; recorded because they vary.
  L('device: subgroup=' + device.features.has('subgroup')
    + ' timestampQuery=' + device.features.has('timestamp-query')
    + ' maxWorkgroupInvocations=' + device.limits.maxComputeInvocationsPerWorkgroup
    + ' maxStorageBinding=' + (device.limits.maxStorageBufferBindingSize / 1048576).toFixed(0) + 'MB');

  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, device });
  renderer.setSize(640, 360, false);
  await renderer.init();
  L('backend is WebGPU: ' + !!renderer.backend.isWebGPUBackend);
  L('CLAIM1 same device object: ' + (renderer.backend.device === device));

  // ---- CLAIM 2: TSL compute writes it; the GPUBuffer stays reachable. ----
  const attr = new THREE.StorageBufferAttribute(N, 4);
  const gpuAttr = storage(attr, 'vec4', N);
  const tU = uniform(0);
  // Fn(...) must be invoked to become a callable node.
  const kernel = Fn(() => {
    const i = instanceIndex;
    const f = i.toFloat().div(N);
    const x = f.mul(6.2831853).mul(3).add(tU).cos().mul(1.6);
    const y = f.mul(6.2831853).mul(2).add(tU.mul(1.3)).sin().mul(1.1);
    const z = f.mul(6.2831853).mul(5).sub(tU.mul(0.7)).cos().mul(1.6);
    gpuAttr.element(i).assign(vec4(x, y, z, 1));
  })();
  await renderer.computeAsync(compute(kernel, N, [64]));
  L('CLAIM2 TSL computeAsync: OK');

  const threeBuf = renderer.backend.get(attr).buffer;
  L('CLAIM2 three-managed GPUBuffer reachable: '
    + !!(threeBuf && typeof threeBuf.mapAsync === 'function'));
  L('CLAIM2 usage STORAGE=' + ((threeBuf.usage & GPUBufferUsage.STORAGE) !== 0)
    + ' VERTEX=' + ((threeBuf.usage & GPUBufferUsage.VERTEX) !== 0)
    + ' COPY_SRC=' + ((threeBuf.usage & GPUBufferUsage.COPY_SRC) !== 0));
  const rb0 = new Float32Array(await renderer.getArrayBufferAsync(attr));
  L('CLAIM2 readback p[0] = ['
    + rb0[0].toFixed(4) + ', ' + rb0[1].toFixed(4) + ', ' + rb0[2].toFixed(4) + ']');

  // ---- CLAIM 3: raw WGSL binds three.js's own buffer. ----
  const module = device.createShaderModule({ code: \`
    @group(0) @binding(0) var<storage, read_write> data: array<vec4<f32>>;
    @group(0) @binding(1) var<uniform> params: vec4<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= u32(params.x)) { return; }
      let t = params.y;
      let f = f32(i) / params.x;
      data[i] = vec4<f32>(
        cos(f * 6.2831853 * 3.0 + t) * 1.6,
        sin(f * 6.2831853 * 2.0 + t * 1.3) * 1.1,
        cos(f * 6.2831853 * 5.0 - t * 0.7) * 1.6,
        ${SENTINEL});   // sentinel: proves this kernel, not TSL, wrote the buffer
    }\` });
  const ci = await module.getCompilationInfo();
  for (const m of ci.messages) if (m.type === 'error') L('WGSL ERROR line ' + m.lineNum + ': ' + m.message);
  const uni = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ]});
  const bg = device.createBindGroup({ layout: bgl, entries: [
    { binding: 0, resource: { buffer: threeBuf } },
    { binding: 1, resource: { buffer: uni } },
  ]});
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: 'main' },
  });
  L("CLAIM3 raw pipeline bound to THREE's buffer: OK");

  // Interleave raw compute and three.js rendering on one device and queue.
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshNormalMaterial()));
  const camera = new THREE.PerspectiveCamera(55, 640 / 360, 0.1, 100);
  camera.position.set(0, 0, 3.2);

  const t0 = performance.now();
  for (let f = 0; f < FRAMES; f++) {
    device.queue.writeBuffer(uni, 0, new Float32Array([N, f * 0.05, 0, 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(N / 64);
    pass.end();
    device.queue.submit([enc.finish()]);
    renderer.render(scene, camera);
  }
  await device.queue.onSubmittedWorkDone();
  const t1 = performance.now();
  L('CLAIM3 ' + FRAMES + 'x raw-compute + three.render interleaved: OK ('
    + ((t1 - t0) / FRAMES).toFixed(2) + ' ms/frame, compute+render+present)');

  const rb = new Float32Array(await renderer.getArrayBufferAsync(attr));
  let sentinel = 0;
  for (let i = 0; i < N; i++) if (Math.abs(rb[i * 4 + 3] - ${SENTINEL}) < 1e-3) sentinel++;
  L('CLAIM3 points carrying raw-kernel sentinel (w=${SENTINEL}): ' + sentinel + '/' + N);
  L('CLAIM3 sample = [' + rb[0].toFixed(4) + ', ' + rb[1].toFixed(4) + ', ' + rb[2].toFixed(4) + ']');

  const blob = await new Promise((r) => canvas.toBlob(r, 'image/png'));
  L('canvas PNG bytes (render produced pixels): ' + (blob ? blob.size : 0));
  device.destroy();
};
<\/script></body></html>`;

const { readFileSync } = await import('node:fs');
const { extname, join, resolve } = await import('node:path');
const ROOT = resolve(import.meta.dirname, '..');
const THREE_DIR = join(ROOT, 'node_modules', 'three');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript' };

const server = createServer((req, res) => {
  const url = (req.url ?? '/').split('?')[0];
  if (url === '/' || url === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(PAGE);
    return;
  }
  if (url === '/favicon.ico') { res.writeHead(204); res.end(); return; }
  if (!url.startsWith('/three/')) { res.writeHead(404); res.end('not found'); return; }
  const file = join(THREE_DIR, url.slice('/three/'.length));
  if (!file.startsWith(THREE_DIR) || !existsSync(file)) { res.writeHead(404); res.end('not found'); return; }
  res.writeHead(200, { 'Content-Type': MIME[extname(file)] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(PORT, '127.0.0.1', r));

let exitCode = 0;
try {
  const browser = await chromium.launch({
    executablePath: CHROME && existsSync(CHROME) ? CHROME : undefined,
    args: [
      '--headless=new', '--no-sandbox', '--ignore-gpu-blocklist',
      '--enable-features=Vulkan,DefaultANGLEVulkan,WebGPUService',
      '--use-angle=vulkan',
    ],
  });
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('[pageerror]', e.message.slice(0, 500)));
  page.on('console', (m) => {
    if (m.type() === 'error') console.error('[console]', m.text().slice(0, 400));
  });
  await page.goto(`http://127.0.0.1:${PORT}/`);
  await page.evaluate(() => window.__run());
  const out = await page.evaluate(() => window.__out);
  console.log(out.join('\n'));

  for (const must of [
    'CLAIM1 same device object: true',
    'CLAIM2 three-managed GPUBuffer reachable: true',
    "CLAIM3 raw pipeline bound to THREE's buffer: OK",
    'CLAIM3 ' + FRAMES + 'x raw-compute + three.render interleaved: OK',
  ]) {
    if (!out.some((l) => l.includes(must))) { console.error('!! FAILED: ' + must); exitCode = 1; }
  }
  const s = out.find((l) => l.includes('sentinel (w='));
  if (!s || !/: (\d+)\/\1$/.test(s)) {
    console.error('!! FAILED: raw kernel did not author every point -> ' + s);
    exitCode = 1;
  }
  await browser.close();
} finally {
  server.close();
}
process.exit(exitCode);

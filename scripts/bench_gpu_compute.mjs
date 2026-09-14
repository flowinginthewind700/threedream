#!/usr/bin/env node
/**
 * WebGPU compute throughput for an all-pairs O(N^2) n-body step -- the shape of
 * a broadphase-less particle or dense-contact kernel -- alongside the device
 * limits that constrain how such a kernel may be written.
 *
 * scripts/bench_cpu_nbody.mjs runs the SAME workload in single-threaded JS. The
 * two must stay semantically identical or the ratio means nothing: same
 * softening, same dt, same speed clamp, same integration order, same seed. A
 * previous attempt compared Jacobi against Gauss-Seidel and reported a 5.83x
 * speedup that did not exist. Keep them in lockstep.
 *
 * Run:  node scripts/bench_gpu_compute.mjs
 * Env:  CHROME_PATH, BENCH_PORT (default 8891)
 *
 * Which GPU is used is the driver's choice: on a machine whose discrete card
 * has no Vulkan ICD, requestAdapter() returns the integrated one. Read the
 * absolute numbers as a floor and the N-scaling as the real finding.
 */
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { chromium } from '@playwright/test';

const PORT = Number(process.env.BENCH_PORT ?? 8891);
const CHROME = process.env.CHROME_PATH;

// Shared with bench_cpu_nbody.mjs by construction: these four numbers define
// the workload. Change one and the CPU script must change with it.
const DT = 1 / 120;
const SOFTENING = 0.01;
const MAX_SPEED = 50;
const STRIDE = 8; // pos:vec4 (w unused) + vel:vec4 (w = mass)
// [N, iterations]: small N gets more iterations so timing noise stays low.
const LADDER = [[1024, 200], [2048, 100], [4096, 40], [8192, 20], [16384, 8]];

// Held on the host and injected as a string literal, so there is exactly one
// level of template escaping in this file. Params arrive through a uniform
// buffer rather than string substitution, so the shader source is a constant.
const WGSL = String.raw`
struct Body { pos: vec4<f32>, vel: vec4<f32> };
@group(0) @binding(0) var<storage, read> src: array<Body>;
@group(0) @binding(1) var<storage, read_write> dst: array<Body>;
// params: x = n, y = dt, z = maxSpeed, w = softening
@group(0) @binding(2) var<uniform> params: vec4<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = u32(params.x);
  if (i >= n) { return; }
  let me = src[i].pos.xyz;
  let soft = params.w;
  var acc = vec3<f32>(0.0);
  for (var j = 0u; j < n; j = j + 1u) {
    let d = src[j].pos.xyz - me;
    let r2 = dot(d, d) + soft;
    acc = acc + d * (src[j].vel.w / (r2 * sqrt(r2)));
  }
  let dt = params.y;
  var v = src[i].vel.xyz + acc * dt;
  let s2 = dot(v, v);
  let mx = params.z;
  if (s2 > mx * mx) { v = v * (mx / sqrt(s2)); }
  dst[i] = Body(vec4<f32>(me + v * dt, src[i].pos.w), vec4<f32>(v, src[i].vel.w));
}`;

const PAGE = `<!doctype html><html><head><meta charset="utf-8"></head><body>
<script type="module">
const out = [];
const L = (...a) => out.push(a.join(' '));
window.__out = out;

const WGSL = ${JSON.stringify(WGSL)};
const DT = ${DT};
const SOFTENING = ${SOFTENING};
const MAX_SPEED = ${MAX_SPEED};
const STRIDE = ${STRIDE};
const LADDER = ${JSON.stringify(LADDER)};

window.__run = async () => {
  if (!navigator.gpu) { L('NO_WEBGPU'); return; }
  const adapter = await navigator.gpu.requestAdapter({ featureLevel: 'compatibility' });
  if (!adapter) { L('NO_ADAPTER'); return; }
  const device = await adapter.requestDevice();
  device.onuncapturederror = (e) => L('[uncaptured] ' + e.error.message);
  const inf = adapter.info || {};
  L('adapter: vendor=' + inf.vendor + ' arch=' + inf.architecture
    + ' deviceId=0x' + (inf.deviceId || 0).toString(16));
  L('limits: workgroupsPerDim=' + device.limits.maxComputeWorkgroupsPerDimension
    + ' maxStorageBinding=' + (device.limits.maxStorageBufferBindingSize / 1048576).toFixed(0) + 'MB'
    + ' maxInvocationsPerWorkgroup=' + device.limits.maxComputeInvocationsPerWorkgroup);
  L('features: [' + [...device.features].join(',') + ']'
    + ' subgroup=' + device.features.has('subgroup'));

  const runN = async (N, iters) => {
    const bytes = N * STRIDE * 4;
    const SU = GPUBufferUsage.STORAGE, CS = GPUBufferUsage.COPY_SRC, CD = GPUBufferUsage.COPY_DST;
    const A = device.createBuffer({ size: bytes, usage: SU | CS | CD });
    const B = device.createBuffer({ size: bytes, usage: SU | CS | CD });
    // MAP_READ is only legal with COPY_DST. Pairing it with COPY_SRC yields a
    // buffer that is invalid from creation and fails later at mapAsync with
    // "Invalid Buffer due to previous error", far from the actual mistake.
    const staging = device.createBuffer({ size: bytes, usage: CD | GPUBufferUsage.MAP_READ });
    const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | CD });

    const module = device.createShaderModule({ code: WGSL });
    const ci = await module.getCompilationInfo();
    for (const m of ci.messages) if (m.type === 'error') {
      L('WGSL ERROR line ' + m.lineNum + ': ' + m.message);
    }
    const bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ]});
    // Double-buffered ping-pong: a step is one dispatch with no barrier stalls.
    const bgAB = device.createBindGroup({ layout: bgl, entries: [
      { binding: 0, resource: { buffer: A } }, { binding: 1, resource: { buffer: B } },
      { binding: 2, resource: { buffer: uni } } ]});
    const bgBA = device.createBindGroup({ layout: bgl, entries: [
      { binding: 0, resource: { buffer: B } }, { binding: 1, resource: { buffer: A } },
      { binding: 2, resource: { buffer: uni } } ]});
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module, entryPoint: 'main' },
    });
    device.queue.writeBuffer(uni, 0, new Float32Array([N, DT, MAX_SPEED, SOFTENING]));

    // Same LCG and same draw order as bench_cpu_nbody.mjs: identical state.
    const seed = new Float32Array(N * STRIDE);
    let s = 1;
    const r = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    for (let i = 0; i < N; i++) {
      seed[i*STRIDE+0] = r()*20-10; seed[i*STRIDE+1] = r()*20-10; seed[i*STRIDE+2] = r()*20-10;
      seed[i*STRIDE+3] = 0;
      seed[i*STRIDE+4] = r()*2-1;   seed[i*STRIDE+5] = r()*2-1;   seed[i*STRIDE+6] = r()*2-1;
      seed[i*STRIDE+7] = 1;
    }
    device.queue.writeBuffer(A, 0, seed);

    let parity = 0;
    const step = () => {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, parity ? bgBA : bgAB);
      pass.dispatchWorkgroups(Math.ceil(N / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
      parity ^= 1;
    };
    for (let i = 0; i < 3; i++) step();
    await device.queue.onSubmittedWorkDone();

    const t0 = performance.now();
    for (let i = 0; i < iters; i++) step();
    await device.queue.onSubmittedWorkDone();
    const ms = (performance.now() - t0) / iters;

    // Read back whichever buffer holds the newest state and require finiteness:
    // a kernel that silently diverged would still time perfectly cleanly.
    const live = parity ? B : A;
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(live, 0, staging, 0, bytes);
    device.queue.submit([enc2.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    let finite = 0;
    for (let i = 0; i < N; i++) if (Number.isFinite(res[i*STRIDE])) finite++;

    A.destroy(); B.destroy(); staging.destroy(); uni.destroy();
    return { N, ms, gips: (N * N) / (ms / 1000) / 1e9, finite };
  };

  L('');
  L('all-pairs O(N^2) n-body, softening=' + SOFTENING + ' dt=' + DT.toFixed(6)
    + ' maxSpeed=' + MAX_SPEED);
  L('N | per-step ms | G pair-int/s | bodies in a 16.6ms frame | finite');
  for (const [N, iters] of LADDER) {
    const r0 = await runN(N, iters);
    L(String(r0.N).padEnd(6) + '| ' + r0.ms.toFixed(3).padStart(9)
      + ' | ' + r0.gips.toFixed(2).padStart(9)
      + ' | ' + String(Math.floor((16.6 / r0.ms) * r0.N)).padStart(10)
      + ' | ' + r0.finite + '/' + r0.N);
  }
  device.destroy();
};
<\/script></body></html>`;

const server = createServer((req, res) => {
  const url = (req.url ?? '/').split('?')[0];
  if (url === '/favicon.ico') { res.writeHead(204); res.end(); return; }
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end(PAGE);
});
await new Promise((r) => server.listen(PORT, '127.0.0.1', r));

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
  if (out.some((l) => l.startsWith('NO_ADAPTER') || l.startsWith('NO_WEBGPU'))) process.exitCode = 2;
  await browser.close();
} finally {
  server.close();
}

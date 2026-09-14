#!/usr/bin/env node
/**
 * Single-threaded JS baseline for scripts/bench_gpu_compute.mjs: the same
 * all-pairs O(N^2) n-body step, the same constants, the same seed and draw
 * order, and the same ping-pong so the comparison is about the hardware and
 * not about one side doing less work.
 *
 * This runs on the main thread, which is exactly the point. GitHub Pages sends
 * no COOP/COEP headers, so SharedArrayBuffer is unavailable there and a wasm
 * physics kernel cannot use rayon/threads -- one core is the production
 * reality for the deterministic backend.
 *
 * Run: node scripts/bench_cpu_nbody.mjs
 */

// Shared with bench_gpu_compute.mjs by construction. Change one, change both.
const DT = 1 / 120;
const SOFTENING = 0.01;
const MAX_SPEED = 50;
const STRIDE = 8;
const LADDER = [[1024, 20], [2048, 6], [4096, 2], [8192, 1]];

function step(px, py, pz, m, vx, vy, vz, N) {
  const mx2 = MAX_SPEED * MAX_SPEED;
  // Read the whole step from the pre-update state, then write: the same
  // double-buffered semantics the GPU ping-pong has, expressed in two arrays.
  const outPx = new Float32Array(N), outPy = new Float32Array(N), outPz = new Float32Array(N);
  const outVx = new Float32Array(N), outVy = new Float32Array(N), outVz = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let ax = 0, ay = 0, az = 0;
    const xi = px[i], yi = py[i], zi = pz[i];
    for (let j = 0; j < N; j++) {
      const dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;
      const r2 = dx * dx + dy * dy + dz * dz + SOFTENING;
      const inv = m[j] / (r2 * Math.sqrt(r2));
      ax += dx * inv; ay += dy * inv; az += dz * inv;
    }
    let nvx = vx[i] + ax * DT, nvy = vy[i] + ay * DT, nvz = vz[i] + az * DT;
    const s2 = nvx * nvx + nvy * nvy + nvz * nvz;
    if (s2 > mx2) { const k = MAX_SPEED / Math.sqrt(s2); nvx *= k; nvy *= k; nvz *= k; }
    outVx[i] = nvx; outVy[i] = nvy; outVz[i] = nvz;
    outPx[i] = xi + nvx * DT; outPy[i] = yi + nvy * DT; outPz[i] = zi + nvz * DT;
  }
  px.set(outPx); py.set(outPy); pz.set(outPz);
  vx.set(outVx); vy.set(outVy); vz.set(outVz);
}

function makeState(N) {
  const px = new Float32Array(N), py = new Float32Array(N), pz = new Float32Array(N);
  const m = new Float32Array(N);
  const vx = new Float32Array(N), vy = new Float32Array(N), vz = new Float32Array(N);
  // Same LCG and draw order as the GPU script's seed buffer.
  let s = 1;
  const r = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  for (let i = 0; i < N; i++) {
    px[i] = r() * 20 - 10; py[i] = r() * 20 - 10; pz[i] = r() * 20 - 10;
    m[i] = 1;
    vx[i] = r() * 2 - 1; vy[i] = r() * 2 - 1; vz[i] = r() * 2 - 1;
  }
  return { px, py, pz, m, vx, vy, vz };
}

function bench(fn, warm, iters) {
  for (let i = 0; i < warm; i++) fn();
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  return Number(process.hrtime.bigint() - t0) / 1e6 / iters;
}

console.log(`single-threaded JS, all-pairs O(N^2) n-body, softening=${SOFTENING} dt=${DT.toFixed(6)} maxSpeed=${MAX_SPEED}`);
console.log(`node ${process.versions.node}, ${process.platform} ${process.arch}`);
console.log('N | per-step ms | G pair-int/s | finite');
for (const [N, iters] of LADDER) {
  const st = makeState(N);
  step(st.px, st.py, st.pz, st.m, st.vx, st.vy, st.vz, N); // warm
  const ms = bench(
    () => step(st.px, st.py, st.pz, st.m, st.vx, st.vy, st.vz, N),
    0, iters,
  );
  let finite = 0;
  for (let i = 0; i < N; i++) if (Number.isFinite(st.px[i])) finite++;
  const gips = (N * N) / (ms / 1000) / 1e9;
  console.log(
    String(N).padEnd(6) + '| ' + ms.toFixed(3).padStart(9)
    + ' | ' + gips.toFixed(3).padStart(9)
    + ' | ' + finite + '/' + N,
  );
}

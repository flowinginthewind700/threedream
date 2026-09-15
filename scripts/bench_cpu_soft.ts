/**
 * The M4 fallback budget: what one soft-body step costs on the tier a visitor
 * lands on when there is no WebGPU.
 *
 *   npx tsx scripts/bench_cpu_soft.ts                # the default ladder
 *   npx tsx scripts/bench_cpu_soft.ts 2000 10000     # your own rungs
 *   BENCH_ITERATIONS=4 npx tsx scripts/bench_cpu_soft.ts
 *
 * M4's acceptance criterion is "WebGL2 回退路径可用，性能目标明确降级": the
 * fallback has to exist *and* its target has to be a number rather than an
 * apology. `bench_gpu_soft.mjs` prints the number for the device tier and this
 * prints the one for the tier below it, and the pair is comparable by
 * construction -- same scene, same seed, same iteration count, same ladder, same
 * 160 steps a rung, same statistic. Both read `BENCH_SEED` / `BENCH_SCENE` /
 * `BENCH_ITERATIONS`, so one env prefix retargets the two together.
 *
 * It measures `src/gpu/softCpu.ts` itself rather than a reimplementation of it,
 * which is what separates it from `bench_cpu_nbody.mjs`: that script is a JS
 * baseline for a WGSL kernel whose constants it restates by hand, while the
 * reference tier measured here *is* the shipped fallback. `webgl2` and `cpu` are
 * one tier with two renderers, so this is the simulation cost of both, and what
 * drawing adds is the e2e suite's business rather than a column here.
 *
 * Single-threaded, which is the production reality and not an oversight: GitHub
 * Pages sends no COOP/COEP headers, so no worker can share a `Float32Array` with
 * the page, and a solver that split nodes across workers would have to
 * synchronize on every color batch -- the same batches the device tier hands to
 * the GPU, which is where the parallelism already went.
 *
 * The spread is the column worth reading and it is small: no driver submission,
 * no queue flush and no compositor inside the timed region, so p95 over p50 here
 * is close to the machine's own noise floor. That is what makes the GPU bench's
 * wider gap interpretable -- whatever in it is not in this table is the device
 * path rather than the machine.
 *
 * Exits non-zero when a rung throws or a node leaves the box. No digest and no
 * parity assertion: the tier's determinism claim is pinned by
 * `tests/soft_cpu.test.ts` against golden bytes, which says more than two runs
 * of this script agreeing with each other.
 */

import { SOFT_SCENES, SoftMesh, type SoftScene } from '../src/gpu/softMesh.js';
import { createCpuSoftSystem } from '../src/gpu/softCpu.js';

const SEED = Number(process.env.BENCH_SEED ?? 4242);
const ITERATIONS = Number(process.env.BENCH_ITERATIONS ?? 8);

/**
 * Steps in one timed sample, and samples in one rung.
 *
 * 8 x 20 = 160 steps, the run length the GPU ladder uses, so a rung's p50 here
 * and its p50 there describe the same amount of simulated cloth. Twenty is also
 * the smallest sample count at which a nearest-rank p95 is not simply the
 * maximum, which is the property the GPU bench was rebuilt around.
 */
const STEPS_PER_SAMPLE = 8;
const SAMPLES = 20;
/** Discarded, so the first sample is not also paying for allocation and warm caches. */
const WARMUP_STEPS = 8;

const DEFAULT_LADDER = [1_000, 5_000, 10_000, 20_000];

function sceneOf(name: string): SoftScene {
  const found = SOFT_SCENES.find((scene) => scene === name);
  if (!found) {
    throw new Error(`bad BENCH_SCENE "${name}", expected one of ${SOFT_SCENES.join(', ')}`);
  }
  return found;
}

const SCENE = sceneOf(process.env.BENCH_SCENE ?? 'cloth');

const RUNGS =
  process.argv.length > 2
    ? process.argv.slice(2).map((arg) => {
        const count = Number(arg);
        if (!Number.isInteger(count) || count <= 0) {
          throw new Error(`bad rung "${arg}", expected a node count, e.g. 10000`);
        }
        return count;
      })
    : DEFAULT_LADDER;

/** Nearest rank, and 0 for no samples: a bench should only print what it measured. */
function percentile(samples: readonly number[], p: number): number {
  if (samples.length === 0) return 0;
  const sorted = [...samples].sort((a, b) => a - b);
  const rank = Math.min(sorted.length, Math.max(1, Math.ceil((p / 100) * sorted.length)));
  return sorted[rank - 1]!;
}

interface Rung {
  count: number;
  steps: number;
  samples: number;
  p50: number;
  p95: number;
  min: number;
  max: number;
  constraints: number;
  colors: number;
  escaped: number;
  outOfBounds: number;
}

function runRung(count: number): Rung {
  const mesh = new SoftMesh({ scene: SCENE, count, seed: SEED });
  const system = createCpuSoftSystem({ mesh, options: { iterations: ITERATIONS } });
  try {
    system.advance(WARMUP_STEPS);
    const timed: number[] = [];
    for (let i = 0; i < SAMPLES; i++) {
      const t0 = performance.now();
      system.advance(STEPS_PER_SAMPLE);
      timed.push((performance.now() - t0) / STEPS_PER_SAMPLE);
    }
    const stats = system.stats();
    return {
      count,
      steps: system.steps,
      samples: timed.length,
      p50: percentile(timed, 50),
      p95: percentile(timed, 95),
      min: Math.min(...timed),
      max: Math.max(...timed),
      constraints: system.plan.constraints,
      colors: system.plan.colors,
      escaped: stats.escaped,
      outOfBounds: mesh.outOfBounds(),
    };
  } finally {
    system.dispose();
  }
}

const header = [
  'nodes',
  'steps',
  'n',
  'p50 ms/step',
  'p95',
  'min',
  'max',
  'us/step/node',
  'constraints',
  'colors',
  'escaped',
];
const rows: string[][] = [header];

let exitCode = 0;
const rungs: Rung[] = [];
try {
  for (const count of RUNGS) rungs.push(runRung(count));

  for (const r of rungs) {
    if (r.escaped > 0 || r.outOfBounds > 0) {
      console.error(
        `!! ${r.count} nodes: ${r.escaped} escaped, ${r.outOfBounds} out of bounds -- ` +
          'a benchmark of a solver that lost its mesh measures nothing',
      );
      exitCode = 1;
    }
    rows.push([
      String(r.count),
      String(r.steps),
      String(r.samples),
      r.p50.toFixed(3),
      r.p95.toFixed(3),
      r.min.toFixed(3),
      r.max.toFixed(3),
      ((r.p50 * 1000) / r.count).toFixed(3),
      String(r.constraints),
      String(r.colors),
      String(r.escaped),
    ]);
  }

  const widths = header.map((_, c) => Math.max(...rows.map((row) => row[c]!.length)));
  console.log('');
  console.log(
    `CPU reference-tier soft-body step cost (${SCENE}, ${ITERATIONS} iterations, seed ${SEED})`,
  );
  for (const [i, row] of rows.entries()) {
    console.log(
      row.map((cell, c) => (i === 0 ? cell.padEnd(widths[c]!) : cell.padStart(widths[c]!))).join('  '),
    );
    if (i === 0) console.log(widths.map((w) => '-'.repeat(w)).join('  '));
  }

  // The model the docs quote, fitted off the two ends of the ladder rather than
  // asserted: a fixed cost that does not depend on the node count, plus a
  // per-node marginal. Two rungs are the minimum a line can be drawn through,
  // and the middle rungs printed above are the check that it is a line.
  if (rungs.length >= 2) {
    const lo = rungs[0]!;
    const hi = rungs[rungs.length - 1]!;
    const slope = (hi.p50 - lo.p50) / (hi.count - lo.count);
    const intercept = lo.p50 - slope * lo.count;
    console.log('');
    console.log(
      `fit ${lo.count}-${hi.count}: ${intercept.toFixed(2)} ms fixed + ` +
        `${(slope * 1000).toFixed(2)} us per node per step`,
    );
  }
  console.log('');
} catch (error) {
  console.error(`!! bench failed: ${error instanceof Error ? error.message : String(error)}`);
  exitCode = 1;
}
process.exit(exitCode);

/**
 * `src/index.ts` -- the public entry point, checked against the claims its own
 * header makes.
 *
 * A barrel is documentation that compiles, and it rots in one specific way: a
 * new layer lands, the demo imports it through a deep path, and the barrel
 * quietly stops being "the only entry point consumers should need". Nothing
 * fails, because everything still works. This spec fails instead.
 *
 * Two claims, both from the header comment:
 *
 *   1. Importing the barrel must not pull three.js into a training script. That
 *      is what makes `npm run train` and CI work from a bare checkout, and it is
 *      the reason `render/` is excluded from the re-exports. Checked statically
 *      over the transitive import graph, because a runtime check in Node would
 *      only catch a module that touches the DOM at load time -- the failure worth
 *      catching is three.js entering the graph at all.
 *   2. The M3 particle stack is reachable from the barrel and runs headless. The
 *      CPU tier is the deterministic reference, so a replay driven entirely
 *      through public exports must reproduce itself.
 */

import { readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import * as api from '../src/index.js';

const ROOT = resolve(import.meta.dirname, '..');
const SRC = resolve(ROOT, 'src');
const BARREL = resolve(SRC, 'index.ts');

/**
 * Every import specifier in a module: static `from`, side-effect and dynamic.
 *
 * Anchored to a statement start rather than scanning for any `from '...'`,
 * because prose gets in the way: `ai/trainer.ts` carries a comment reading
 * 'distinguishes "the task ended" from "the clock ran out"', and a loose regex
 * reports `the clock ran out` as a dependency.
 */
const SPECIFIER_PATTERNS = [
  /^\s*(?:import|export)\b[^;'"]*?\bfrom\s*['"]([^'"]+)['"]/gm,
  /^\s*import\s*['"]([^'"]+)['"]/gm,
  /\bimport\(\s*['"]([^'"]+)['"]\s*\)/g,
];

function specifiersOf(file: string): string[] {
  const text = readFileSync(file, 'utf8');
  const out: string[] = [];
  for (const pattern of SPECIFIER_PATTERNS) {
    pattern.lastIndex = 0;
    for (const m of text.matchAll(pattern)) out.push(m[1]);
  }
  return out;
}

/**
 * Resolve a relative specifier to the `.ts` file it names, or null when it is
 * not ours to follow: a bare package name, or a path that leaves `src/`.
 *
 * The second case is real rather than hypothetical -- `physics/wasm.ts` reaches
 * the wasm-pack glue at `../../wasm/pkg/`, a generated artifact outside the
 * source tree. It lands in `bare` alongside package names, which is where the
 * assertion below can see it and say whether it is allowed.
 */
function resolveSpecifier(from: string, specifier: string): string | null {
  if (!specifier.startsWith('.')) return null;
  const base = resolve(dirname(from), specifier).replace(/\.js$/, '.ts');
  const file = base.endsWith('.ts') ? base : `${base}.ts`;
  return file.startsWith(`${SRC}/`) ? file : null;
}

/** Transitive graph under `src/`. Bare specifiers are collected, not followed. */
function importGraph(entry: string): { files: string[]; bare: string[] } {
  const seen = new Set<string>();
  const bare = new Set<string>();
  const queue = [entry];
  while (queue.length > 0) {
    const file = queue.pop() as string;
    if (seen.has(file)) continue;
    seen.add(file);
    for (const specifier of specifiersOf(file)) {
      const next = resolveSpecifier(file, specifier);
      if (next === null) bare.add(specifier);
      else queue.push(next);
    }
  }
  return {
    files: [...seen].map((f) => relative(ROOT, f)).sort(),
    bare: [...bare].sort(),
  };
}

describe('the barrel stays three.js-free', () => {
  const graph = importGraph(BARREL);

  it('reaches the layers it claims to, and only those', () => {
    // Sanity: a walker that silently found nothing would pass every assertion
    // below. The barrel re-exports five layers; all five must show up.
    for (const layer of ['core', 'physics', 'gpu', 'ai', 'envs']) {
      const prefix = `src/${layer}/`;
      expect(
        graph.files.some((f) => f.startsWith(prefix)),
        `${prefix} is not reachable from the barrel`,
      ).toBe(true);
    }
    expect(graph.files.length, 'the walk found almost nothing').toBeGreaterThan(20);
  });

  it('imports no three.js, statically or dynamically', () => {
    const three = graph.bare.filter((s) => s === 'three' || s.startsWith('three/'));
    expect(three, 'the barrel pulls three.js into a training script').toEqual([]);
  });

  it('re-exports nothing from render/', () => {
    const render = graph.files.filter((f) => f.startsWith('src/render/'));
    expect(render, 'render/ is the three.js boundary and must stay out').toEqual([]);
  });

  it('takes no bare dependency but node builtins and the two lazy solvers', () => {
    // `@dimforge/rapier3d-compat` and the wasm-pack glue are both dynamic imports
    // inside their factories, so a script that never asks for that world never
    // loads it. Anything else bare and unaccounted for is a new import-time
    // dependency on the headless path, which is what this barrel exists to
    // protect.
    const allowed = /^(node:|@dimforge\/rapier3d-compat|\.\.\/\.\.\/wasm\/pkg\/)/;
    const unexpected = graph.bare.filter((s) => !allowed.test(s));
    expect(unexpected, 'new import-time dependency on the headless path').toEqual([]);
  });
});

describe('the M3 particle stack is public API', () => {
  it('exports the container, both backends, the factory and the frame loop', () => {
    const names = [
      'ParticleField',
      'SpatialHash',
      'CpuParticleSystem',
      'GpuParticleSystem',
      'InstanceExpander',
      'ParticleRunner',
      'SharedDeviceManager',
      'ComputeContext',
      'createCpuParticleSystem',
      'createGpuParticleSystem',
      'createParticleSystem',
      'createParticleRunner',
      'probeParticles',
      'particleShaderSource',
      'instanceShaderSource',
      'resolveParticleOptions',
    ] as const;
    for (const name of names) {
      const value = (api as Record<string, unknown>)[name];
      expect(typeof value, `${name} is not exported from the barrel`).toBe('function');
    }
    // The constants a consumer needs to size a buffer or a dispatch.
    expect(api.PARTICLE_STRIDE, 'floats per particle').toBe(8);
    expect(api.INSTANCE_BYTES, 'bytes per instance').toBe(64);
    expect(api.WORKGROUP_SIZE, 'the plan pins the workgroup size at 64').toBe(64);
  });

  it('runs a deterministic replay through nothing but barrel exports', async () => {
    // The acceptance criterion "回放与训练不依赖 GPU 层", at the surface a consumer
    // actually touches: no deep imports, no device, no DOM.
    const run = async (): Promise<{ digest: string; escaped: number; steps: number }> => {
      const field = new api.ParticleField({ count: 600, seed: 1234, scene: 'sphere', speed: 3 });
      const handle = await api.createParticleSystem({
        field,
        tier: 'cpu',
        options: { collisions: true },
      });
      expect(handle.gpu, 'the CPU tier must not have acquired a device').toBeNull();
      expect(handle.system.deterministic, 'the reference backend is the deterministic one').toBe(
        true,
      );
      handle.system.advance(40);
      const stats = handle.system.stats();
      const out = { digest: handle.system.digest(), escaped: stats.escaped, steps: handle.system.steps };
      handle.dispose();
      return out;
    };

    const first = await run();
    const second = await run();
    expect(first.steps, 'advance(40) did not advance').toBe(40);
    expect(first.escaped, 'a particle left the box').toBe(0);
    expect(first.digest, 'the same seed produced a different field').toBe(second.digest);
  });

  it('drives the simulation from a frame loop whose rate is not the step rate', async () => {
    // The other half of the decoupling criterion, at the public surface: a frame
    // is worth a whole number of fixed steps, and a frame that lasts four and a
    // half steps simulates four of them -- not one long one.
    const field = new api.ParticleField({ count: 200, seed: 7 });
    const handle = await api.createParticleSystem({ field, tier: 'cpu' });
    const runner = new api.ParticleRunner(handle.system);
    const dt = handle.system.fixedDt;

    expect(runner.frame(dt * 0.4), 'a frame shorter than a step must not step').toBe(0);
    expect(handle.system.steps).toBe(0);
    expect(runner.frame(dt * 4.5), 'a long frame is worth four whole steps').toBe(4);
    expect(handle.system.steps).toBe(4);
    expect(handle.system.time, 'simulated time is steps * fixedDt').toBeCloseTo(dt * 4, 9);

    handle.dispose();
  });
});

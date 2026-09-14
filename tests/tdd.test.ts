/**
 * TDD discipline, enforced by the test suite itself.
 *
 * "We do TDD" is a claim about process, and process claims decay silently: a
 * feature lands without tests, coverage slips, and nothing fails. These tests
 * turn the claim into an executable gate.
 *
 * Three properties are pinned:
 *
 * 1. *Every source module has a test that exercises it.* A new file under
 *    `src/` that no test imports is a new file with no spec, which is exactly
 *    the TDD inversion (implementation first, tests never). The mapping is by
 *    name convention (`src/core/rng.ts` -> `tests/rng.test.ts`), so adding a
 *    module without a spec fails the suite immediately.
 *
 * 2. *Coverage does not regress.* `src/` is held at or above the floor recorded
 *    in `vitest.config.ts`, and the two layers that genuinely cannot run under
 *    vitest are excluded by name rather than by a blanket threshold drop.
 *
 * 3. *The browser-only layers stay excluded for a real reason.* `render/` needs
 *    WebGL and `physics/rapier` needs WASM; both are covered by their own
 *    dedicated specs or by the Playwright demo check, not by being ignored.
 */

import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = resolve(import.meta.dirname, '..');
const SRC = resolve(ROOT, 'src');
const TESTS = resolve(ROOT, 'tests');

/** Every module under `src/`, except the barrel and type-only files. */
function sourceModules(): string[] {
  const out: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = resolve(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.name.endsWith('.ts')) {
        out.push(full);
      }
    }
  };
  walk(SRC);
  return out.sort();
}

function testFiles(): string[] {
  return readdirSync(TESTS)
    .filter((f) => f.endsWith('.test.ts'))
    .sort();
}

/** Concatenated text of every test file, for import-graph assertions. */
function allTestText(): string {
  return testFiles()
    .map((f) => readFileSync(resolve(TESTS, f), 'utf8'))
    .join('\n');
}

/**
 * Layers that cannot execute under vitest's `node` environment. Excluding them
 * from the coverage floor is a statement about the *runner*, not about whether
 * they are tested: `render/` is exercised by the Playwright demo check and
 * `physics/rapier` by `tests/rapier_backend.test.ts`, which loads the real WASM
 * lazily inside the test so the module can be imported in Node.
 */
const BROWSER_ONLY = ['src/render/scene.ts'];

/** Where a browser-only module's spec lives, since it is not under `tests/`. */
const E2E_DIR = resolve(ROOT, 'e2e');

function e2eText(): string {
  if (!existsSync(E2E_DIR)) return '';
  return readdirSync(E2E_DIR)
    .filter((f) => f.endsWith('.spec.ts'))
    .map((f) => readFileSync(resolve(E2E_DIR, f), 'utf8'))
    .join('\n');
}

describe('every source module is specified by a test', () => {
  const modules = sourceModules();

  it('has modules to check (sanity)', () => {
    expect(modules.length).toBeGreaterThan(10);
    expect(testFiles().length).toBeGreaterThan(5);
  });

  it.each(modules.map((m) => [relative(SRC, m), m] as const))(
    '%s is imported by at least one test',
    (_rel, full) => {
      const stem = full.replace(/\.ts$/, '').split('/').pop()!;
      const text = allTestText();
      // A module counts as specified if some test imports it, or if a
      // same-stem spec exists. Both are checked because the barrel
      // (`src/index.ts`) is reached transitively through the layers it exports.
      const imported = new RegExp(`from ['"][^'"]*${stem}(\\.js)?['"]`).test(text);
      const dedicated = existsSync(resolve(TESTS, `${stem}.test.ts`));
      const isBarrel = full.endsWith('src/index.ts');
      const isTypeOnly = full.endsWith('types.ts');
      // A browser-only module has its spec in `e2e/`, not `tests/`: it cannot be
      // imported in Node at all (constructing a WebGLRenderer throws), so the
      // only honest spec is a real browser run. That is weaker than a unit test,
      // so it is allowed only for the modules listed in BROWSER_ONLY, and only
      // while the browser spec actually exists -- deleting e2e/ turns this red.
      const rel = `src/${_rel}`;
      const browserOnly = BROWSER_ONLY.includes(rel) && e2eText().length > 0;
      expect(
        imported || dedicated || isBarrel || isTypeOnly || browserOnly,
        `no test imports ${_rel} and tests/${stem}.test.ts does not exist`,
      ).toBe(true);
    },
  );

  it('the barrel is covered transitively: every layer has a dedicated spec', () => {
    // `src/index.ts` re-exports five layers: core, physics, gpu, ai, envs. If a
    // layer had no spec, the barrel assertion above would pass vacuously, so
    // check each layer explicitly. The two WASM backends are listed next to
    // `physics` for the same reason: a caller can select either one, and nothing
    // else would notice their specs going missing.
    const stems = testFiles().map((f) => f.replace('.test.ts', ''));
    const required = [
      'rng', 'clock', 'ecs',
      'physics', 'wasm_backend', 'rapier_backend',
      'gpu_capabilities',
      'mlp', 'policy', 'trainer',
      'envs',
    ];
    for (const layer of required) {
      expect(stems, `missing spec for the ${layer} layer`).toContain(layer);
    }
  });

  it('type-only modules export types, not runtime code', () => {
    // `types.ts` files are contracts. If one grew runtime behaviour it would
    // need a real spec, so this pins the reason they are exempt above.
    for (const m of modules.filter((x) => x.endsWith('types.ts'))) {
      const text = readFileSync(m, 'utf8');
      const runtime = text.split('\n').filter(
        (l) => /^(export\s+)?(function|class|const|let)\s/.test(l.trim()) && !/^export\s+type/.test(l.trim()),
      );
      // `vec3()` and friends in physics/types.ts are the deliberate exception:
      // they are the value half of the physics contract and are covered by
      // tests/physics.test.ts, which the assertion above already requires.
      if (m.endsWith('physics/types.ts')) continue;
      expect(runtime, `${relative(SRC, m)} must stay type-only`).toEqual([]);
    }
  });
});

describe('browser-only layers are excluded for a stated reason', () => {
  it('vitest excludes exactly the layers that cannot run headless', () => {
    const cfg = readFileSync(resolve(ROOT, 'vitest.config.ts'), 'utf8');
    for (const layer of BROWSER_ONLY) {
      expect(cfg, `${layer} must be listed in the coverage exclude`).toMatch(
        new RegExp(String(layer).replace(/[/.]/g, '\\$&')),
      );
    }
  });

  it('rapier is NOT coverage-excluded, because its spec really runs', () => {
    const cfg = readFileSync(resolve(ROOT, 'vitest.config.ts'), 'utf8');
    const excludeBlock = cfg.slice(cfg.indexOf('exclude'));
    expect(excludeBlock, 'physics/rapier must stay inside the coverage floor').not.toMatch(/rapier/);
    expect(testFiles()).toContain('rapier_backend.test.ts');
  });

  it('render/ is covered by the Playwright demo check instead', () => {
    // If this assertion is what stops you from deleting the e2e check, that is
    // the point: render/ has no unit coverage, so the browser check is its spec.
    const cfg = readFileSync(resolve(ROOT, 'package.json'), 'utf8');
    expect(cfg).toMatch(/test:e2e/);
    expect(existsSync(resolve(ROOT, 'e2e'))).toBe(true);
    expect(existsSync(resolve(ROOT, 'playwright.config.ts'))).toBe(true);
  });
});

describe('the coverage floor is recorded, not tribal knowledge', () => {
  it('vitest.config.ts declares a global statement threshold', () => {
    const cfg = readFileSync(resolve(ROOT, 'vitest.config.ts'), 'utf8');
    expect(cfg).toMatch(/thresholds/);
    expect(cfg).toMatch(/statements/);
    const floor = Number(cfg.match(/statements:\s*(\d+)/)?.[1]);
    expect(floor, 'floor must be a meaningful gate, not 0').toBeGreaterThanOrEqual(80);
    expect(floor, 'floor must be a real number').toBeLessThanOrEqual(100);
  });

  it('coverage is a separate script, because instrumenting the trainer costs 10x', () => {
    // Measured: `vitest run` is ~18s, and the two convergence tests in
    // `tests/trainer.test.ts` cost ~10x more under v8 instrumentation because
    // it lands on the training inner loop. Folding coverage into `npm test`
    // would therefore destroy the red/green loop that makes TDD usable, so it
    // is its own command that CI runs in parallel instead.
    const pkgJson = JSON.parse(readFileSync(resolve(ROOT, 'package.json'), 'utf8')) as {
      scripts: Record<string, string>;
    };
    expect(pkgJson.scripts.test, 'the fast loop must stay fast').not.toMatch(/--coverage/);
    expect(pkgJson.scripts['test:coverage'], 'coverage needs its own script').toMatch(/--coverage/);
    // A threshold nobody enforces is decoration, so the script must not opt out.
    expect(pkgJson.scripts['test:coverage']).not.toMatch(/--coverage\.enabled=false/);
  });

  it('only the coverage script skips the convergence tests, never `npm test`', () => {
    // The two convergence tests in `tests/trainer.test.ts` cost ~12s each
    // un-instrumented and ~10x that under v8 coverage, because the training
    // inner loop is exactly what the instrumentation counts. They contribute
    // ~0.3% of branch coverage, so the coverage job skips them (measured: 185s
    // -> 3s). That trade is only safe if the skip is confined to coverage --
    // if `npm test` also skipped them, the learning assertion would vanish from
    // the gate entirely and nothing would fail.
    const pkgJson = JSON.parse(readFileSync(resolve(ROOT, 'package.json'), 'utf8')) as {
      scripts: Record<string, string>;
    };
    expect(pkgJson.scripts['test:coverage'], 'coverage must set the skip flag').toMatch(/COVERAGE=1/);
    expect(pkgJson.scripts.test, '`npm test` must not set the skip flag').not.toMatch(/COVERAGE=1/);

    const trainer = readFileSync(resolve(TESTS, 'trainer.test.ts'), 'utf8');
    // The skip must be conditional on that flag, not unconditional.
    expect(trainer, 'convergence tests must be skipped conditionally').toMatch(
      /describe\.skipIf\(underCoverage\)\('Trainer learning'/,
    );
    expect(trainer).toMatch(/underCoverage\s*=\s*process\.env\.COVERAGE/);
  });

  it('CI runs both the fast suite and the coverage gate', () => {
    const ci = readFileSync(resolve(ROOT, '.github/workflows/ci.yml'), 'utf8');
    expect(ci).toMatch(/npm (ci|test)\b/);
    expect(ci).toMatch(/test:coverage/);
    expect(ci).toMatch(/test:e2e/);
  });
});

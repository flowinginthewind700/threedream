/**
 * Asset-path contract for the deployed demo.
 *
 * GitHub Pages serves the site from `https://<user>.github.io/<repo>/`, so a
 * bundle that emits `/assets/index.js` asks the CDN for the *domain* root and
 * 404s. The build has to know its own prefix. `vite.config.ts` reads `VITE_BASE`
 * and pages.yml sets it to `/<repo>/`; locally the default `/` is correct
 * because `npm run dev` serves from the domain root.
 *
 * This is worth a test rather than a code review note because the failure is
 * invisible locally: the demo works on a dev server, and only breaks once
 * deployed, where the blank page gives no console hint that a path was wrong.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = resolve(import.meta.dirname, '..');
const VITE_CONFIG = readFileSync(resolve(ROOT, 'vite.config.ts'), 'utf8');

/** Every src/href/url() target in a file, i.e. the paths a browser will fetch. */
function referencedPaths(html: string): string[] {
  return [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map((m) => m[1]);
}

describe('vite.config.ts resolves its base path from the environment', () => {
  it('reads VITE_BASE so CI can inject the Pages subpath', () => {
    expect(VITE_CONFIG).toMatch(/VITE_BASE/);
  });

  it('defaults to "/" when VITE_BASE is unset, so local dev still works', () => {
    // The default must be a real fallback, not `process.env.VITE_BASE` which is
    // `undefined` locally and would emit `undefinedassets/...`.
    expect(VITE_CONFIG).toMatch(/VITE_BASE\s*(\?\?|\|\|)\s*['"`]\//);
  });

  it('keeps root at demo/ and outDir at ../dist', () => {
    // pages.yml uploads `dist`, so the output directory is part of the contract.
    expect(VITE_CONFIG).toMatch(/root:\s*['"]demo['"]/);
    expect(VITE_CONFIG).toMatch(/outDir:\s*['"]\.\.\/dist['"]/);
  });
});

describe('a Pages build emits subpath-relative asset refs', () => {
  it('produces no domain-absolute /assets/ refs under a subpath base', async () => {
    // Drive the real config file through vite's own loader rather than
    // inlining a `base`, so what is asserted is what CI actually builds.
    const { build } = await import('vite');
    const outDir = resolve(ROOT, 'artifacts/pages-build');
    const base = '/threedream/';
    // try/finally: a failing build must not leak VITE_BASE into later tests,
    // where it would make unrelated assertions pass for the wrong reason.
    process.env.VITE_BASE = base;
    try {
      await build({
        configFile: resolve(ROOT, 'vite.config.ts'),
        mode: 'production',
        build: { outDir, emptyOutDir: true, write: true },
        logLevel: 'silent',
      });
    } finally {
      delete process.env.VITE_BASE;
    }

    const html = readFileSync(resolve(outDir, 'index.html'), 'utf8');
    const paths = referencedPaths(html);
    expect(paths.length, 'index.html should reference its bundle').toBeGreaterThan(0);

    for (const p of paths) {
      // The bundle and CSS must carry the prefix (or be relative). A leading
      // slash without the prefix is the exact bug this test exists to catch.
      expect(p.startsWith('/assets/'), `domain-absolute asset ref: ${p}`).toBe(false);
      if (p.startsWith('/')) expect(p.startsWith('/threedream/'), `missing base prefix: ${p}`).toBe(true);
    }
    expect(paths.some((p) => p.includes('/threedream/assets/')), 'expected prefixed asset refs').toBe(true);

    // The CSS/JS files it points at must actually exist next to index.html.
    for (const p of paths.filter((p) => p.includes('assets/'))) {
      const rel = p.replace(/^\/threedream\//, '');
      expect(() => readFileSync(resolve(outDir, rel)), `missing emitted file: ${rel}`).not.toThrow();
    }
  }, 120000);
});

import { defineConfig, loadEnv } from 'vite';

import { DEMO_PAGES } from './demo/pages.js';

/**
 * Every page the site ships.
 *
 * Listed explicitly rather than left to Vite's single-`index.html` default:
 * `demo/physics-check.html`, `demo/shared-device.html`, `demo/particles.html` and
 * `demo/soft.html` are the browser half of four milestone gates, and a page that
 * is not in `rollupOptions.input` is not built, not deployed, and not testable --
 * while still looking perfectly fine on the dev server, which serves any HTML it
 * finds. That asymmetry is exactly the kind of failure that reaches CI before it
 * reaches anyone's eyes, so the list is here where `tests/build_base.test.ts` and
 * the e2e specs can see it.
 *
 * The list itself is `demo/pages.ts`, the same array `demo/nav.ts` renders as the
 * nav strip every page carries, so "built but unreachable" and "linked but not
 * built" are both unrepresentable rather than merely unlikely. This file derives
 * its inputs from that array instead of restating it, and `tests/demo_nav.test.ts`
 * loads this config and compares the two, which is what makes the derivation a
 * contract rather than a comment.
 */
function pageInputs(): Record<string, string> {
  const inputs: Record<string, string> = {};
  for (const page of DEMO_PAGES) {
    const name = page.file.replace(/\.html$/, '');
    inputs[name] = new URL(`./demo/${page.file}`, import.meta.url).pathname;
  }
  return inputs;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', 'VITE_');

  return {
    /**
     * Resolved from the environment because the same build config has to serve
     * two different URL shapes:
     *
     * - local dev/preview, from the domain root, so `/`
     * - GitHub Pages, from `https://<user>.github.io/<repo>/`, so `/<repo>/`
     *
     * A bundle built with the wrong base emits `/assets/...`, which asks the CDN
     * for the *domain* root and 404s under the subpath -- and does so silently,
     * because the page still loads and the failure is a blank viewport.
     * `tests/build_base.test.ts` pins both halves of this contract.
     */
    base: env.VITE_BASE ?? '/',
    root: 'demo',
    resolve: {
      alias: {
        '@threedream': new URL('./src', import.meta.url).pathname,
      },
    },
    build: {
      outDir: '../dist',
      emptyOutDir: true,
      target: 'es2022',
      rollupOptions: {
        input: pageInputs(),
      },
    },
    server: {
      port: 5173,
    },
  };
});

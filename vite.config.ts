import { defineConfig, loadEnv } from 'vite';

/**
 * `base` is resolved from the environment because the same build config has to
 * serve two different URL shapes:
 *
 * - local dev/preview, from the domain root, so `/`
 * - GitHub Pages, from `https://<user>.github.io/<repo>/`, so `/<repo>/`
 *
 * A bundle built with the wrong base emits `/assets/...`, which asks the CDN
 * for the *domain* root and 404s under the subpath -- and does so silently,
 * because the page still loads and the failure is a blank viewport.
 * `tests/build_base.test.ts` pins both halves of this contract.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', 'VITE_');

  return {
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
    },
    server: {
      port: 5173,
    },
  };
});

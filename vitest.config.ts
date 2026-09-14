import { defineConfig } from 'vitest/config';

/**
 * Node environment on purpose: `render/` needs DOM/WebGL and the rapier backend
 * needs WASM, so they are handled by their own dedicated specs rather than
 * faked with stubs. Everything else (core, physics/builtin, ai, envs) runs
 * headless, which is the property that makes `npm run train` and CI work from a
 * bare checkout.
 */
export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    exclude: ['node_modules', 'dist', 'thirdparty'],
    testTimeout: 30000,
    hookTimeout: 30000,
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      // Named, not blanket-dropped. `render/scene.ts` constructs a real
      // WebGLRenderer, which cannot exist in Node; its spec is the Playwright
      // demo check (`e2e/demo.spec.ts`), run by `npm run test:e2e` and by CI.
      exclude: ['src/render/scene.ts'],
      reporter: ['text', 'html', 'lcov', 'json-summary'],
      reportsDirectory: 'coverage',
      // Floor, not aspiration: `npm run test:coverage` fails below it, so
      // coverage can only be raised deliberately, never lost by accident. A
      // floor set far below what the suite actually reaches is decoration, so
      // these sit a few points under what this command actually measures (97.05
      // / 86.70 / 97.54 / 98.47 with the two convergence tests skipped, see
      // tests/trainer.test.ts) -- enough headroom for a refactor to move code
      // around, not enough to delete a module's tests unnoticed. Raise them
      // when measured coverage rises.
      thresholds: {
        statements: 93,
        branches: 82,
        functions: 93,
        lines: 94,
      },
    },
  },
});

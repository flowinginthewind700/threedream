import { defineConfig } from 'vitest/config';

/**
 * Node environment on purpose: `render/` and `physics/rapier` need DOM/WASM,
 * so they are excluded from unit tests rather than faked. Everything under
 * test (core, physics/builtin, ai, envs) runs headless, which is the property
 * that makes `npm run train` and CI work from a bare checkout.
 */
export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    exclude: ['node_modules', 'dist', 'thirdparty'],
    testTimeout: 30000,
    hookTimeout: 30000,
  },
});

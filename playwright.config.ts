import { defineConfig, devices } from '@playwright/test';

/**
 * Browser spec for the `render/` layer.
 *
 * `render/scene.ts` needs WebGL, so it is excluded from the vitest coverage
 * floor rather than faked with a stub canvas. This is its spec: it loads the
 * real demo page against a software GL context and asserts the things a unit
 * test cannot -- that a canvas is produced, that it is non-blank, that the
 * render loop mirrors physics state, and that training in the page works.
 *
 * Kept out of `npm test` on purpose: a browser run costs seconds of setup and
 * would destroy the sub-second red/green loop the unit suite exists to provide.
 * `npm run test:e2e` and CI run it.
 */

/** Where the demo is served from, and where the tests go to find it. */
const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173/threedream';

/**
 * Pages serves from `/<repo>/`, so the local preview has to reproduce that
 * subpath -- built *and* served with the same base. Serving a subpath build
 * from `/` would 404 every asset and prove nothing about what ships.
 */
const SERVE = 'npm run build:pages && npx vite preview --base /threedream/ --port 4173 --host 127.0.0.1 --strictPort';

export default defineConfig({
  testDir: './e2e',
  // The demo trains for real in-page; give slow CI runners room.
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['list']] : [['list']],
  use: {
    baseURL: BASE,
    trace: 'retain-on-failure',
    // Headless Chromium has no GPU, so the demo runs on SwiftShader. That is the
    // point of the exercise: render/ must work without real hardware.
    launchOptions: {
      args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'],
    },
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  // Serve the built site (not the dev server) so e2e tests the shipped bundle.
  // Point E2E_BASE_URL at a live deploy to test that instead; nothing is built.
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : {
        command: SERVE,
        url: `${BASE}/`,
        reuseExistingServer: !process.env.CI,
        timeout: 240_000,
        stdout: 'pipe',
        stderr: 'pipe',
      },
});

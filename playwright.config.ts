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

/**
 * Two projects, because the pages need two different GPUs.
 *
 * `demo/`, `physics-check` and the fallback half of `particles` only need *a* GL
 * context, and headless Chromium has none, so they run on SwiftShader: that the
 * render layer works without hardware is the property worth testing.
 * `shared-device` and `particles_gpu` need a real `GPUDevice`, which
 * SwiftShader-as-GL cannot provide -- they need ANGLE's Vulkan backend with the
 * WebGPU service enabled. Same browser, different flags, and a flag set that
 * works for one silently downgrades the other, so they are separate projects
 * rather than one `launchOptions` compromise.
 */
const SWIFTSHADER_ARGS = ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'];

/**
 * `--enable-unsafe-swiftshader` is in here too, and it is not redundant: on a
 * runner with no GPU at all it is what lets ANGLE's Vulkan backend fall back to
 * SwiftShader's Vulkan ICD instead of failing `requestAdapter()`. Where real
 * hardware exists it changes nothing, because the fallback is only consulted
 * once the real device is ruled out.
 *
 * `E2E_ANGLE` is the escape hatch for a machine where `vulkan` is the wrong
 * answer (macOS wants `metal`, some Windows setups want `d3d11`).
 */
const WEBGPU_ARGS = [
  '--no-sandbox',
  '--ignore-gpu-blocklist',
  '--enable-unsafe-swiftshader',
  '--enable-features=Vulkan,DefaultANGLEVulkan,WebGPUService',
  `--use-angle=${process.env.E2E_ANGLE ?? 'vulkan'}`,
];

/** Specs that need the WebGPU project, and must not run under SwiftShader. */
const WEBGPU_SPECS = /(shared_device|particles_gpu)\.spec\.ts/;

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
  },
  projects: [
    {
      name: 'chromium',
      testIgnore: WEBGPU_SPECS,
      use: { ...devices['Desktop Chrome'], launchOptions: { args: SWIFTSHADER_ARGS } },
    },
    {
      name: 'chromium-webgpu',
      testMatch: WEBGPU_SPECS,
      use: { ...devices['Desktop Chrome'], launchOptions: { args: WEBGPU_ARGS } },
    },
  ],
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

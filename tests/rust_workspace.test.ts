/**
 * Contract tests for the Rust workspace's *configuration*.
 *
 * `cargo test` checks that the solver is correct. Nothing checks that the solver
 * is still built the way the determinism claim requires, because that lives in
 * TOML: a `wgpu = "25"` that quietly becomes `"30"`, an `lto` that gets dropped,
 * a `rust-toolchain.toml` that says `stable`. Each of those compiles, each of
 * them passes `cargo test`, and each of them invalidates something the docs
 * assert:
 *
 *   - `docs/feasibility-rust-wasm-webgpu.md` measured wgpu 25 -> 30 changing
 *     `InstanceDescriptor`, `push_constant_ranges` and `request_adapter`'s
 *     return type. The M2 bridge is written against v25's API.
 *   - `opt-level = 3` + fat LTO + one codegen unit was worth 2.43x over the TS
 *     solver. The default release profile is not, and M1's acceptance criterion
 *     is "no slower than 2x".
 *   - `deterministic: true` on the wasm backend is only meaningful if the binary
 *     is reproducible, which is what pinning the toolchain channel buys.
 *
 * Asserted as literal TOML text rather than parsed, because the point is what a
 * reviewer sees in the diff: `wgpu = "=25.0.2"` and `wgpu = "25"` resolve
 * identically today and mean different things tomorrow.
 */

import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import pkg from '../package.json';

const ROOT = resolve(import.meta.dirname, '..');
const RUST = resolve(ROOT, 'rust');

const read = (...parts: string[]) => readFileSync(resolve(...parts), 'utf8');
const workspaceToml = read(RUST, 'Cargo.toml');
const cargoLock = read(RUST, 'Cargo.lock');
const toolchain = read(RUST, 'rust-toolchain.toml');
const scripts = pkg.scripts as Record<string, string>;

/** The version a `[[package]]` block resolves to in Cargo.lock. */
function lockedVersion(crate: string): string | undefined {
  for (const block of cargoLock.split(/\n?\[\[package\]\]\n/).slice(1)) {
    const name = /^name = "([^"]+)"/m.exec(block)?.[1];
    if (name === crate) return /^version = "([^"]+)"/m.exec(block)?.[1];
  }
  return undefined;
}

/** The `=x.y.z` pin for a crate in `[workspace.dependencies]`. */
function workspacePin(crate: string): string | undefined {
  const section = workspaceToml.slice(workspaceToml.indexOf('[workspace.dependencies]'));
  return new RegExp(`^${crate} = "=([^"]+)"`, 'm').exec(section)?.[1];
}

describe('workspace shape', () => {
  it('has exactly the three crates, each with a manifest on disk', () => {
    const members = /\bmembers = \[([^\]]+)\]/.exec(workspaceToml)?.[1] ?? '';
    const paths = members
      .split(',')
      .map((m) => m.trim().replace(/^"|"$/g, ''))
      .filter(Boolean)
      .sort();
    expect(paths).toEqual(['crates/gpu', 'crates/physics', 'crates/physics-wasm']);
    for (const path of paths) {
      expect(existsSync(resolve(RUST, path, 'Cargo.toml')), `${path} has no manifest`).toBe(true);
    }
  });

  it('keeps the solver dependency-free, so nothing can perturb its arithmetic', () => {
    // `builtin.ts` is the normative spec and `crates/physics` is a port of it. A
    // dependency here (rand, nalgebra, approx) would be a second opinion about
    // how a float rounds, which is exactly what the bit-exactness test cannot
    // tolerate.
    const manifest = read(RUST, 'crates/physics/Cargo.toml');
    const deps = manifest.slice(manifest.indexOf('[dependencies]'));
    expect(deps.split('\n').filter((l) => /^[a-z0-9_-]+ *=/.test(l))).toEqual([]);
  });

  it('ships physics-wasm as a cdylib over the solver and wasm-bindgen only', () => {
    const manifest = read(RUST, 'crates/physics-wasm/Cargo.toml');
    expect(manifest).toMatch(/crate-type = \["cdylib", "rlib"\]/);
    expect(manifest).toMatch(/^threedream-physics = \{ workspace = true \}/m);
    expect(manifest).toMatch(/^wasm-bindgen = \{ workspace = true \}/m);
    // `workspace = true` for both, so the versions live in one place and the
    // pins below are the whole story.
    expect(manifest).not.toMatch(/wasm-bindgen = "/);
  });

  it('keeps wgpu out of the native build of the gpu crate', () => {
    // wgpu behind `cfg(target_arch = "wasm32")` is what lets `cargo test
    // --workspace` stay fast: a native wgpu build pulls in Vulkan/GL/DX12
    // loaders that no test here exercises.
    const manifest = read(RUST, 'crates/gpu/Cargo.toml');
    expect(manifest).toMatch(/\[target\.'cfg\(target_arch = "wasm32"\)'\.dependencies\]/);
    expect(manifest).toMatch(/^wgpu = \{ workspace = true \}/m);
    const plain = manifest.slice(0, manifest.indexOf('[target.'));
    expect(plain.slice(plain.indexOf('[dependencies]'))).not.toMatch(/^wgpu/m);
  });
});

describe('version pins are exact, and Cargo.lock honours them', () => {
  it.each([
    ['wasm-bindgen', '0.2.128'],
    ['wgpu', '25.0.2'],
  ])('%s is pinned with "=" to %s', (crate, version) => {
    // `=` and not `^`: a caret would let `cargo update` move to the next
    // breaking release, and for wgpu the feasibility study measured that v30
    // changes the API the M2 bridge is written against.
    expect(workspacePin(crate), `${crate} must be pinned with "=x.y.z"`).toBe(version);
    expect(lockedVersion(crate), `Cargo.lock must resolve ${crate}`).toBe(version);
  });

  it('wasm-bindgen in the lockfile is the version the artifact was built with', () => {
    // `scripts/check_wasm_artifact.mjs` reads the binary's `producers` section
    // and compares it against this. Asserting the other end here means a
    // lockfile bump fails at `npm test`, not only in the artifact gate.
    expect(workspaceToml).toMatch(/wasm-bindgen = "=0\.2\.128"/);
    expect(cargoLock).toMatch(/name = "wasm-bindgen"\nversion = "0\.2\.128"/);
  });
});

describe('the release profile is the measured one, not the cargo default', () => {
  const profile = workspaceToml.slice(workspaceToml.indexOf('[profile.release]'));

  it.each([
    ['opt-level = 3', 'the 2.43x speedup was measured at opt-level 3'],
    ['lto = "fat"', 'cross-crate inlining is where most of the win came from'],
    ['codegen-units = 1', 'more units means less inlining and different codegen'],
    ['panic = "abort"', 'unwinding tables are dead weight in a wasm kernel'],
    ['debug = false', 'committed bytes should not carry a build machine path'],
  ])('%s (%s)', (line) => {
    expect(profile).toMatch(new RegExp(`^${line.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'm'));
  });

  it('keeps a debug-info profile for profiling the kernel', () => {
    // `release-debug` inherits release, so a profile run measures the same
    // codegen the shipped artifact uses.
    expect(workspaceToml).toMatch(/\[profile\.release-debug\]/);
    expect(profile).toMatch(/^inherits = "release"/m);
  });
});

describe('the toolchain is pinned, because the artifact is committed', () => {
  it('names an exact channel rather than a moving one', () => {
    const channel = /^\s*channel\s*=\s*"([^"]+)"/m.exec(toolchain)?.[1];
    expect(channel, 'rust-toolchain.toml must pin a channel').toBeTruthy();
    expect(channel, 'a channel like "stable" is not reproducible').toMatch(/^\d+\.\d+\.\d+$/);
  });

  it('installs the wasm32 target the artifact is built for', () => {
    expect(toolchain).toMatch(/targets = \[[^\]]*"wasm32-unknown-unknown"[^\]]*\]/);
  });

  it('brings clippy and rustfmt, so CI can gate on them later', () => {
    expect(toolchain).toMatch(/components = \[[^\]]*"clippy"[^\]]*"rustfmt"[^\]]*\]/);
  });
});

describe('the ABI version is declared in exactly two places, and they agree', () => {
  it('src/physics/wasm.ts matches rust/crates/physics/src/lib.rs', () => {
    // `kernelFrom()` also refuses to load a mismatched artifact at runtime, but
    // that only fires once something loads the kernel. This fails at `npm test`.
    const ts = /export const WASM_ABI_VERSION = (\d+)/.exec(read(ROOT, 'src/physics/wasm.ts'))?.[1];
    const rs = /pub const ABI_VERSION: u32 = (\d+)/.exec(
      read(RUST, 'crates/physics/src/lib.rs'),
    )?.[1];
    expect(ts, 'WASM_ABI_VERSION must be a literal').toBeTruthy();
    expect(rs, 'ABI_VERSION must be a literal').toBeTruthy();
    expect(ts).toBe(rs);
  });
});

describe('npm scripts keep the Rust invocations in one place', () => {
  it('every cargo and wasm-pack command runs from rust/', () => {
    // The workspace root is `rust/`, not the repo root: without `cd rust`, cargo
    // finds no manifest and fails with a confusing "could not find Cargo.toml".
    for (const name of ['build:wasm', 'build:wasm:gpu', 'test:rust']) {
      expect(scripts[name], `${name} must exist`).toBeTruthy();
      expect(scripts[name], `${name} must run in rust/`).toMatch(/^cd rust && /);
    }
  });

  it('build:wasm targets the web, releases, and writes into the committed wasm/pkg', () => {
    const build = scripts['build:wasm'];
    expect(build).toMatch(/wasm-pack build crates\/physics-wasm/);
    // `--target web` is what emits the ESM glue `src/physics/wasm.ts` imports;
    // `--target nodejs` would emit CommonJS and the browser build would break.
    expect(build).toMatch(/--target web/);
    expect(build).toMatch(/--release/);
    expect(build).toMatch(/--out-dir \.\.\/\.\.\/\.\.\/wasm\/pkg/);
  });

  it('build:wasm:gpu builds the wgpu crate for wasm32 only', () => {
    // The M0 claim is that wgpu at the pinned version compiles for the web
    // target. It is a *build* gate: there is no runtime path to it until M2.
    expect(scripts['build:wasm:gpu']).toMatch(/--target wasm32-unknown-unknown/);
    expect(scripts['build:wasm:gpu']).toMatch(/-p threedream-gpu/);
    expect(scripts['build:wasm:gpu']).toMatch(/--release/);
  });

  it('test:rust covers the whole workspace, not just the solver crate', () => {
    // `-p threedream-physics` would silently skip tests added to the other two
    // crates, and a skipped test is indistinguishable from a passing one.
    expect(scripts['test:rust']).toMatch(/cargo test --workspace/);
  });

  it('check:wasm runs the artifact gate', () => {
    expect(scripts['check:wasm']).toMatch(/node scripts\/check_wasm_artifact\.mjs/);
    expect(existsSync(resolve(ROOT, 'scripts/check_wasm_artifact.mjs'))).toBe(true);
  });
});

#!/usr/bin/env node
/**
 * Gate on the committed wasm artifact in `wasm/pkg`.
 *
 * `wasm/pkg` is a build artifact that is *checked in*, because the browser demo
 * and GitHub Pages both load it and neither runner has a Rust toolchain. That
 * buys reproducibility and costs a failure mode nothing else in the repo has:
 * the artifact can drift from `rust/crates/physics` and still typecheck, still
 * build, and still load -- it just computes something else. `td_abi_version()`
 * catches a *deliberate* ABI change, and nothing catches an accidental one.
 *
 * So this script checks the four things a version number cannot:
 *
 *   shape        the files a `--target web` build emits are all there, and the
 *                manifest points at them.
 *   abi          `WasmBindings` in `src/physics/wasm.ts` and the binary's export
 *                table are the same set in both directions. One-sided checks
 *                pass on a stale TS interface *or* a dropped Rust `#[wasm_bindgen]`.
 *   provenance   the binary's `producers` section names the toolchain and the
 *                wasm-bindgen version the workspace pins, and no build machine's
 *                filesystem leaked into it.
 *   behaviour    the artifact is instantiated and stepped here, in Node, against
 *                closed-form expectations. This is the only place the *shipped*
 *                bytes are executed outside a browser.
 *
 * With `--baseline <dir>`, a fifth check runs: every file is hashed and compared
 * against that directory. CI uses it as `committed pkg` vs `freshly rebuilt pkg`,
 * which turns "someone forgot to run `npm run build:wasm`" from a review
 * comment into a red build.
 *
 * Run:   node scripts/check_wasm_artifact.mjs [--pkg wasm/pkg] [--baseline DIR]
 * Exit:  0 when every check passed, 1 otherwise (with each failure printed).
 */
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');

const CRATE = 'threedream_physics_wasm';
/** What a `wasm-pack build --target web` emits, and what `wasm/pkg` must hold. */
const ARTIFACT_FILES = [
  `${CRATE}.js`,
  `${CRATE}.d.ts`,
  `${CRATE}_bg.wasm`,
  `${CRATE}_bg.wasm.d.ts`,
  'package.json',
];
/**
 * Also emitted by wasm-pack, but not part of the ABI surface: its contents are
 * `*`, i.e. "ignore this whole directory", which the repo deliberately
 * overrides by tracking `wasm/pkg` explicitly. Listed so the shape check does
 * not report it as an unexpected extra file.
 */
const WASM_PACK_EXTRA_FILES = ['.gitignore'];
/**
 * Exports the binary is allowed to carry beyond the `td_*` ABI.
 *
 * `memory` is the linear memory the adapter builds `Float64Array` views over, so
 * its absence would be fatal; the `__wbindgen_*` pair is wasm-bindgen's own
 * externref-table bootstrap; `__abort_handler` / `__instance_terminated` come
 * with `panic = "abort"`. Anything else here is unexplained surface.
 */
const RUNTIME_EXPORTS = new Set([
  'memory',
  '__wbindgen_externrefs',
  '__wbindgen_start',
  '__abort_handler',
  '__instance_terminated',
]);
/**
 * Import module names that would mean the artifact cannot run in a browser.
 *
 * `wasi_snapshot_preview1` shows up when a crate pulls in `std` IO or a
 * `panic = "abort"` handler that formats to stderr; the web glue has no WASI
 * shim, so such a module fails at instantiation with a confusing
 * "module not found" in the page and passes every Node test here.
 */
const FORBIDDEN_IMPORT_MODULES = ['wasi_snapshot_preview1', 'wasi_unstable', 'env'];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const opts = { pkg: join(ROOT, 'wasm', 'pkg'), baseline: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--pkg' || arg === '--baseline') {
      const value = argv[++i];
      if (!value) throw new Error(`${arg} needs a directory argument`);
      const key = arg === '--pkg' ? 'pkg' : 'baseline';
      opts[key] = resolve(ROOT, value);
    } else if (arg === '--help' || arg === '-h') {
      process.stdout.write(
        'Usage: node scripts/check_wasm_artifact.mjs [--pkg DIR] [--baseline DIR]\n',
      );
      process.exit(0);
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  return opts;
}

// ---------------------------------------------------------------------------
// Reporting harness
// ---------------------------------------------------------------------------

const failures = [];
let passed = 0;
let section = '';

function begin(name) {
  section = name;
  process.stdout.write(`\n${name}\n`);
}

/**
 * Run one assertion. Failures are collected rather than thrown, so a single run
 * reports every problem instead of stopping at the first.
 *
 * `fn` may return a short note (a version string, a count) which is printed
 * after the check name -- that is what makes the output usable as a build log.
 */
async function check(name, fn) {
  try {
    const note = await fn();
    passed++;
    process.stdout.write(`  ok    ${name}${note ? ` (${note})` : ''}\n`);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    failures.push({ section, name, detail });
    process.stdout.write(`  FAIL  ${name}\n`);
    for (const line of detail.split('\n')) process.stdout.write(`          ${line}\n`);
  }
}

class ArtifactError extends Error {}

function assert(condition, message) {
  if (!condition) throw new ArtifactError(message);
}

function assertEqual(actual, expected, label) {
  if (!Object.is(actual, expected)) {
    throw new ArtifactError(`${label}: expected ${fmt(expected)}, got ${fmt(actual)}`);
  }
}

function fmt(value) {
  if (typeof value === 'bigint') return `${value}n`;
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

// ---------------------------------------------------------------------------
// Binary inspection helpers
// ---------------------------------------------------------------------------

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

/** A path as it should appear in output: repo-relative inside the checkout. */
function displayPath(dir) {
  const rel = relative(ROOT, dir);
  return rel.startsWith('..') ? dir : rel;
}

/**
 * Basenames git tracks under `dir`, or `null` when git cannot answer.
 *
 * `null` is a skip, not a failure: the script also runs from an exported
 * tarball, where there is no index to consult.
 */
function trackedBasenames(dir) {
  try {
    const out = execFileSync('git', ['ls-files', '--', relative(ROOT, dir) || '.'], {
      cwd: ROOT,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    return out.split('\n').filter(Boolean).map((p) => basename(p));
  } catch {
    return null;
  }
}

/** Decode a `producers` custom section: `vec(field{name, vec(string, string)})`. */
function decodeProducers(buffer) {
  const bytes = new Uint8Array(buffer);
  let at = 0;
  const uleb = () => {
    let value = 0;
    let shift = 1;
    for (;;) {
      const byte = bytes[at++];
      if (byte === undefined) throw new ArtifactError('producers section truncated');
      value += (byte & 0x7f) * shift;
      if ((byte & 0x80) === 0) return value;
      shift *= 128;
    }
  };
  const text = () => {
    const length = uleb();
    const slice = bytes.subarray(at, at + length);
    if (slice.length !== length) throw new ArtifactError('producers string truncated');
    at += length;
    return new TextDecoder().decode(slice);
  };

  const fields = {};
  const count = uleb();
  for (let i = 0; i < count; i++) {
    const name = text();
    const values = [];
    const valueCount = uleb();
    for (let j = 0; j < valueCount; j++) values.push({ name: text(), version: text() });
    fields[name] = values;
  }
  return fields;
}

/** The `td_*` method names declared by `WasmBindings` in `src/physics/wasm.ts`. */
function bindingsFromSource(source) {
  const marker = 'export interface WasmBindings';
  const start = source.indexOf(marker);
  if (start < 0) throw new ArtifactError(`${marker} not found in src/physics/wasm.ts`);
  const open = source.indexOf('{', start);
  let depth = 0;
  let close = -1;
  for (let i = open; i < source.length; i++) {
    if (source[i] === '{') depth++;
    else if (source[i] === '}' && --depth === 0) {
      close = i;
      break;
    }
  }
  if (close < 0) throw new ArtifactError('WasmBindings interface is not terminated');
  const body = source
    .slice(open + 1, close)
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\n]*/g, '');
  const names = new Set();
  for (const match of body.matchAll(/^[ \t]*(td_[A-Za-z0-9_]+)[ \t]*\(/gm)) names.add(match[1]);
  return [...names].sort();
}

/** Names declared by a generated `.d.ts` / `.js` glue file. */
function exportedNames(source) {
  const names = new Set();
  for (const match of source.matchAll(/^export (?:function|const) (td_[A-Za-z0-9_]+)/gm)) {
    names.add(match[1]);
  }
  return [...names].sort();
}

function sortedDiff(left, right) {
  const l = new Set(left);
  const r = new Set(right);
  return {
    onlyLeft: [...l].filter((x) => !r.has(x)).sort(),
    onlyRight: [...r].filter((x) => !l.has(x)).sort(),
  };
}

function describeDiff(label, left, right, leftName, rightName) {
  const { onlyLeft, onlyRight } = sortedDiff(left, right);
  if (onlyLeft.length === 0 && onlyRight.length === 0) return;
  const lines = [`${label}: ${leftName} and ${rightName} disagree`];
  if (onlyLeft.length) lines.push(`  only in ${leftName}: ${onlyLeft.join(', ')}`);
  if (onlyRight.length) lines.push(`  only in ${rightName}: ${onlyRight.join(', ')}`);
  throw new ArtifactError(lines.join('\n'));
}

/** Every `[[package]]` entry in a Cargo.lock, as `name -> version`. */
function parseCargoLock(source) {
  const packages = new Map();
  for (const block of source.split(/\n?\[\[package\]\]\n/).slice(1)) {
    const name = /^name = "([^"]+)"/m.exec(block)?.[1];
    const version = /^version = "([^"]+)"/m.exec(block)?.[1];
    if (name && version) packages.set(name, version);
  }
  return packages;
}

/** The `=x.y.z` pin for a crate in `[workspace.dependencies]`. */
function workspacePin(source, crate) {
  const block = source.slice(source.indexOf('[workspace.dependencies]'));
  const line = new RegExp(`^${crate} = "=([^"]+)"`, 'm').exec(block);
  return line ? line[1] : null;
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

async function checkShape(pkgDir) {
  begin('shape');

  await check(`wasm/pkg exists (${displayPath(pkgDir)})`, () => {
    assert(
      existsSync(pkgDir) && statSync(pkgDir).isDirectory(),
      `${displayPath(pkgDir)} is missing. Run \`npm run build:wasm\` and commit the result.`,
    );
  });

  await check('the five emitted files are present and non-empty', () => {
    const missing = [];
    const empty = [];
    const extra = readdirSync(pkgDir)
      .filter((f) => !ARTIFACT_FILES.includes(f) && !WASM_PACK_EXTRA_FILES.includes(f))
      .sort();
    for (const name of ARTIFACT_FILES) {
      const path = join(pkgDir, name);
      if (!existsSync(path)) missing.push(name);
      else if (statSync(path).size === 0) empty.push(name);
    }
    const problems = [];
    if (missing.length) problems.push(`missing: ${missing.join(', ')}`);
    if (empty.length) problems.push(`empty: ${empty.join(', ')}`);
    if (extra.length) {
      problems.push(
        `unexpected: ${extra.join(', ')} (wasm-pack does not emit these; ` +
          'delete them or explain them in ARTIFACT_FILES)',
      );
    }
    assert(problems.length === 0, problems.join('\n'));
  });

  await check('package.json names the crate and resolves main/types', () => {
    const manifest = JSON.parse(readFileSync(join(pkgDir, 'package.json'), 'utf8'));
    assertEqual(manifest.name, 'threedream-physics-wasm', 'manifest name');
    assertEqual(manifest.type, 'module', 'manifest type');
    assertEqual(manifest.main, `${CRATE}.js`, 'manifest main');
    assertEqual(manifest.types, `${CRATE}.d.ts`, 'manifest types');
    for (const file of manifest.files ?? []) {
      assert(ARTIFACT_FILES.includes(file), `manifest lists unknown file: ${file}`);
    }
  });

  await check('.wasm starts with the wasm magic and version', () => {
    const bytes = readFileSync(join(pkgDir, `${CRATE}_bg.wasm`));
    assert(bytes.length > 8, 'wasm binary is too small to be a module');
    assertEqual(bytes[0], 0x00, 'magic[0]');
    assertEqual(bytes[1], 0x61, 'magic[1]');
    assertEqual(bytes[2], 0x73, 'magic[2]');
    assertEqual(bytes[3], 0x6d, 'magic[3]');
    assertEqual(bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24), 1, 'version');
  });

  await check('every artifact file on disk is committed', () => {
    // wasm/pkg/.gitignore holds `*`, so a file wasm-pack starts emitting after a
    // toolchain bump is invisible to `git add wasm/pkg`: the build stays green
    // locally and the deployed bundle 404s. Comparing the directory against the
    // index is the only way to see it, and it needs `git add -f` to fix.
    const tracked = trackedBasenames(pkgDir);
    if (!tracked) return 'no git index; skipped';
    // Unfiltered, unlike `digests()`: `wasm/pkg/.gitignore` is itself committed,
    // and has to be, or the `*` inside it would ignore the artifact too.
    const onDisk = readdirSync(pkgDir).sort();
    const uncommitted = onDisk.filter((f) => !tracked.includes(f));
    const absent = tracked.filter((f) => !onDisk.includes(f));
    const problems = [];
    if (uncommitted.length) {
      problems.push(
        `on disk but not tracked: ${uncommitted.join(', ')} ` +
          '(wasm/pkg/.gitignore is `*`; commit them with `git add -f wasm/pkg/<file>`)',
      );
    }
    if (absent.length) problems.push(`tracked but missing on disk: ${absent.join(', ')}`);
    assert(problems.length === 0, problems.join('\n'));
    return `${tracked.length} tracked files`;
  });
}

async function checkAbi(pkgDir) {
  begin('abi');

  const wasmPath = join(pkgDir, `${CRATE}_bg.wasm`);
  if (!existsSync(wasmPath)) {
    failures.push({
      section,
      name: 'binary is readable',
      detail: `${displayPath(pkgDir)}/${CRATE}_bg.wasm is missing; nothing else can be checked.`,
    });
    return null;
  }
  const binary = readFileSync(wasmPath);

  // Compiled inside a `check` on purpose: a truncated or corrupt artifact is a
  // finding this script should *report*, not a stack trace it should die with.
  let mod = null;
  await check('the committed .wasm compiles', () => {
    mod = new WebAssembly.Module(binary);
    return `${binary.length} bytes`;
  });
  if (!mod) return null;

  const exports = WebAssembly.Module.exports(mod);
  const imports = WebAssembly.Module.imports(mod);
  const declared = bindingsFromSource(readFileSync(join(ROOT, 'src', 'physics', 'wasm.ts'), 'utf8'));
  const tdExports = exports.filter((e) => e.name.startsWith('td_')).map((e) => e.name);

  await check('WasmBindings and the binary export the same td_* surface', () => {
    describeDiff('ABI', declared, tdExports, 'WasmBindings', 'wasm exports');
    return `${declared.length} functions, both directions`;
  });

  await check('every td_* export is a function', () => {
    for (const entry of exports.filter((e) => e.name.startsWith('td_'))) {
      assertEqual(entry.kind, 'function', `export ${entry.name} kind`);
    }
  });

  await check('non-ABI exports are only the known wasm-bindgen runtime set', () => {
    const unknown = exports
      .map((e) => e.name)
      .filter((n) => !n.startsWith('td_') && !RUNTIME_EXPORTS.has(n))
      .sort();
    assert(unknown.length === 0, `unexplained exports: ${unknown.join(', ')}`);
    const missing = [...RUNTIME_EXPORTS].filter((n) => !exports.some((e) => e.name === n));
    // __abort_handler/__instance_terminated are wasm-bindgen version specific.
    const required = ['memory', '__wbindgen_start'].filter((n) => missing.includes(n));
    assert(required.length === 0, `expected runtime exports missing: ${required.join(', ')}`);
  });

  await check('memory is exported, so the adapter can view its buffers', () => {
    const memory = exports.find((e) => e.name === 'memory');
    assertEqual(memory?.kind, 'memory', 'memory export kind');
  });

  await check('no WASI/env imports (the web glue has no shim for them)', () => {
    const bad = imports
      .filter((i) => FORBIDDEN_IMPORT_MODULES.includes(i.module))
      .map((i) => `${i.module}::${i.name}`);
    assert(
      bad.length === 0,
      `imports a host module the browser cannot satisfy: ${bad.join(', ')}`,
    );
  });

  await check('all imports resolve to wasm-bindgen glue the loader provides', () => {
    // The module name here is a label, not a file: wasm-bindgen 0.2.128 with
    // --target web inlines the whole import object into the entry .js, and
    // `__wbindgen_init_externref_table` is one of its own functions.
    const external = imports.filter((i) => i.module !== `./${CRATE}_bg.js`);
    assert(
      external.length === 0,
      `imports from unexpected modules: ${external.map((i) => i.module).join(', ')}`,
    );
    for (const entry of imports) {
      assertEqual(entry.kind, 'function', `import ${entry.name} kind`);
    }
  });

  await check('generated .js glue and .d.ts declare the same surface', () => {
    const glue = exportedNames(readFileSync(join(pkgDir, `${CRATE}.js`), 'utf8'));
    const types = exportedNames(readFileSync(join(pkgDir, `${CRATE}.d.ts`), 'utf8'));
    describeDiff('glue', glue, tdExports, '.js glue', 'wasm exports');
    describeDiff('types', types, tdExports, '.d.ts', 'wasm exports');
    const bgTypes = exportedNames(readFileSync(join(pkgDir, `${CRATE}_bg.wasm.d.ts`), 'utf8'));
    describeDiff('raw types', bgTypes, tdExports, '_bg.wasm.d.ts', 'wasm exports');
  });

  await check('src/physics/wasm.ts loads the artifact by the committed path', () => {
    const source = readFileSync(join(ROOT, 'src', 'physics', 'wasm.ts'), 'utf8');
    const specifier = '../../wasm/pkg/threedream_physics_wasm.js';
    assert(
      source.includes(`'${specifier}'`),
      `no static import of '${specifier}'; Vite cannot bundle a dynamic specifier`,
    );
  });

  return { mod, binary };
}

async function checkProvenance(pkgDir, binary, mod) {
  begin('provenance');

  const sections = WebAssembly.Module.customSections(mod, 'producers');
  let producers = {};
  await check('the binary carries a producers section', () => {
    assertEqual(sections.length, 1, 'producers section count');
    producers = decodeProducers(sections[0]);
  });

  const toolchain = readFileSync(join(ROOT, 'rust', 'rust-toolchain.toml'), 'utf8');
  const channel = /^\s*channel\s*=\s*"([^"]+)"/m.exec(toolchain)?.[1];
  await check('built by the toolchain rust-toolchain.toml pins', () => {
    assert(channel, 'rust/rust-toolchain.toml declares no channel');
    const rustc = producers['processed-by']?.find((p) => p.name === 'rustc');
    assert(rustc, `producers names no rustc; got ${JSON.stringify(producers)}`);
    assert(
      rustc.version.startsWith(`${channel} `),
      `rustc ${rustc.version} does not match pinned channel ${channel}`,
    );
    return `rustc ${rustc.version}`;
  });

  const cargoToml = readFileSync(join(ROOT, 'rust', 'Cargo.toml'), 'utf8');
  const cargoLock = parseCargoLock(readFileSync(join(ROOT, 'rust', 'Cargo.lock'), 'utf8'));
  await check('wasm-bindgen version agrees with Cargo.toml pin and Cargo.lock', () => {
    const pin = workspacePin(cargoToml, 'wasm-bindgen');
    assert(pin, 'rust/Cargo.toml does not pin wasm-bindgen with "=x.y.z"');
    assertEqual(cargoLock.get('wasm-bindgen'), pin, 'Cargo.lock wasm-bindgen');
    const bindgen = producers['processed-by']?.find((p) => p.name === 'wasm-bindgen');
    assert(bindgen, 'producers names no wasm-bindgen: the artifact was not wasm-pack built');
    assertEqual(bindgen.version, pin, 'binary wasm-bindgen');
    return `wasm-bindgen ${bindgen.version}`;
  });

  await check('wgpu stays at the pinned major (M0 feasibility claim)', () => {
    const pin = workspacePin(cargoToml, 'wgpu');
    assert(pin, 'rust/Cargo.toml does not pin wgpu with "=x.y.z"');
    assertEqual(cargoLock.get('wgpu'), pin, 'Cargo.lock wgpu');
  });

  await check('language is Rust', () => {
    const language = producers.language ?? [];
    assert(language.some((l) => l.name === 'Rust'), `producers language: ${fmt(language)}`);
  });

  await check('no build machine path leaked into the binary or glue', () => {
    const home = process.env.HOME ?? process.env.USERPROFILE ?? '';
    const needles = ['/home/', '/Users/', 'C:\\Users\\', '/root/', '/builds/'];
    if (home && home !== '/') needles.push(home);
    const latin1 = binary.toString('latin1');
    const glue = readFileSync(join(pkgDir, `${CRATE}.js`), 'utf8');
    for (const needle of needles) {
      if (needle && needle !== '/') {
        assert(!latin1.includes(needle), `wasm binary contains "${needle}"`);
        assert(!glue.includes(needle), `glue contains "${needle}"`);
      }
    }
    assert(!latin1.includes(ROOT), 'wasm binary embeds an absolute path to this checkout');
  });

  await check('SIMD128 is a declared target feature (matches td_simd_enabled)', () => {
    const features = WebAssembly.Module.customSections(mod, 'target_features');
    assert(features.length === 1, `target_features section count: ${features.length}`);
    const text = new TextDecoder('latin1').decode(new Uint8Array(features[0]));
    assert(text.includes('simd128'), `target_features lacks simd128: ${fmt(text)}`);
  });
}

async function checkBehaviour(pkgDir) {
  begin('behaviour');

  const glue = await import(pathToFileURL(join(pkgDir, `${CRATE}.js`)).href);
  const bytes = readFileSync(join(pkgDir, `${CRATE}_bg.wasm`));
  let memory;

  await check('initSync instantiates the committed bytes', () => {
    memory = glue.initSync({ module: bytes }).memory;
    assert(memory instanceof WebAssembly.Memory, 'initSync returned no memory');
  });

  const expectedAbi = Number(
    /export const WASM_ABI_VERSION = (\d+)/.exec(
      readFileSync(join(ROOT, 'src', 'physics', 'wasm.ts'), 'utf8'),
    )?.[1],
  );
  await check('td_abi_version matches src/physics/wasm.ts', () => {
    assert(Number.isInteger(expectedAbi), 'could not read WASM_ABI_VERSION');
    assertEqual(glue.td_abi_version(), expectedAbi, 'td_abi_version');
  });

  await check('the shipped binary carries SIMD', () => {
    assertEqual(glue.td_simd_enabled(), true, 'td_simd_enabled');
  });

  await check('SIMD path is bit-identical to the scalar path, in this binary', () => {
    assertEqual(glue.td_selftest_simd_parity(), true, 'td_selftest_simd_parity');
  });

  // The reference scene's world options, restated: this script cannot import
  // src/physics/reference.ts (TypeScript, and Node would need the .js specifiers
  // rewritten), so the numbers are written down here and asserted against
  // REFERENCE_OPTIONS by tests/wasm_backend.test.ts.
  const DT = 1 / 60;
  const GRAVITY_Y = -9.81;
  const LINEAR_DAMPING = 0.05;
  const ANGULAR_DAMPING = 0.2;
  const ITERATIONS = 8;

  const world = glue.td_world_create(0, GRAVITY_Y, 0, DT, ITERATIONS, LINEAR_DAMPING, ANGULAR_DAMPING);

  await check('td_world_create returns a usable id and echoes fixedDt', () => {
    assert(Number.isInteger(world) && world >= 0, `world id: ${world}`);
    assertEqual(glue.td_world_fixed_dt(world), DT, 'td_world_fixed_dt');
    assertEqual(glue.td_world_body_count(world), 0, 'body count of a new world');
  });

  let scratch = 0;
  await check('td_alloc hands back 8-byte-aligned, writable, freeable memory', () => {
    scratch = glue.td_alloc(12 * 8);
    assert(scratch > 0, `td_alloc returned ${scratch}`);
    assertEqual(scratch % 8, 0, 'td_alloc alignment');
    const view = new Float64Array(memory.buffer, scratch, 12);
    view[0] = -1.5;
    view[11] = Number.MAX_SAFE_INTEGER;
    assertEqual(view[0], -1.5, 'scratch write/read');
    assertEqual(view[11], Number.MAX_SAFE_INTEGER, 'scratch write/read');
    assertEqual(glue.td_alloc(0), 0, 'td_alloc(0) must fail rather than alias');
  });

  await check('free fall matches builtin.ts integration bit for bit', () => {
    const handle = glue.td_body_create(
      world, 0, 0.5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, true, 1, 0.2, 0.5, 1, 1,
    );
    assert(handle > 0, `td_body_create returned ${handle}`);
    // Same order of operations as `BuiltinPhysics.step`: v += g*dt; v *= damp;
    // p += v*dt. Bit-exactness here is the M1 claim in the strongest form this
    // script can state without importing the TypeScript reference.
    let y = 2;
    let vy = 0;
    for (let i = 0; i < 60; i++) {
      glue.td_world_step(world, DT);
      vy = vy + GRAVITY_Y * DT;
      vy = vy * Math.max(0, 1 - LINEAR_DAMPING * DT);
      y = y + vy * DT;
    }
    const state = new Float64Array(memory.buffer, scratch, 12);
    assertEqual(glue.td_body_get_state(world, handle, scratch), 1, 'td_body_get_state');
    assertEqual(state[1], y, 'position.y after 60 free-fall steps');
    assertEqual(state[7], vy, 'velocity.y after 60 free-fall steps');
  });

  await check('write_states and write_handles agree in length and order', () => {
    glue.td_body_create(world, 1, 0, 0.25, 0.25, 0.25, 4, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0.2, 0.5, 1, 1);
    glue.td_body_create(world, 0, 0.25, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, true, 2, 0.2, 0.5, 1, 1);
    const count = glue.td_world_body_count(world);
    assertEqual(count, 3, 'body count');
    const states = glue.td_alloc(12 * count);
    const handles = glue.td_alloc(count);
    assertEqual(glue.td_world_write_states(world, states), count, 'write_states length');
    assertEqual(glue.td_world_write_handles(world, handles), count, 'write_handles length');
    const handleView = new Uint32Array(memory.buffer, handles, count);
    const stateView = new Float64Array(memory.buffer, states, 12 * count);
    for (let i = 1; i < count; i++) {
      assert(handleView[i] > handleView[i - 1], 'handles are not in ascending order');
    }
    // Body 3 was created at x=8; row order must match handle order.
    assertEqual(stateView[12 * 2], 8, 'third body position.x');
    glue.td_free(states, 12 * count);
    glue.td_free(handles, count);
  });

  await check('a settling sphere rests on the ground and reports one contact', () => {
    const settle = glue.td_world_create(0, GRAVITY_Y, 0, DT, ITERATIONS, LINEAR_DAMPING, ANGULAR_DAMPING);
    const ball = glue.td_body_create(settle, 0, 0.5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, true, 1, 0.2, 0.5, 1, 1);
    glue.td_body_create(settle, 1, 0, 5, 0.5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0.2, 0.5, 1, 1);
    for (let i = 0; i < 600; i++) glue.td_world_step(settle, DT);

    const state = new Float64Array(memory.buffer, scratch, 12);
    glue.td_body_get_state(settle, ball, scratch);
    const restY = state[1];
    assert(restY > 0.5 && restY < 1.05, `sphere did not rest on the ground: y=${restY}`);
    assertEqual(state[7], 0, 'settled sphere still has vertical velocity');

    const contactStride = 7;
    const contacts = glue.td_alloc(contactStride * 16);
    assertEqual(glue.td_world_contact_count(settle), 1, 'contact count');
    assertEqual(glue.td_world_drain_contacts(settle, contacts), 1, 'drained contacts');
    const record = new Float64Array(memory.buffer, contacts, contactStride);
    assertEqual(record[0], ball, 'contact body a');
    assert(record[1] !== ball, 'contact body b must be the other body');
    assertEqual(record[5] > 0, true, 'penetration depth must be positive');
    // Destructive by contract: the adapter relies on the second read being empty.
    assertEqual(glue.td_world_contact_count(settle), 0, 'contacts survive a drain');
    assertEqual(glue.td_world_drain_contacts(settle, contacts), 0, 'second drain');
    glue.td_free(contacts, contactStride * 16);
    glue.td_world_destroy(settle);
  });

  await check('raycast reports a hit whose distance matches its point', () => {
    const settle = glue.td_world_create(0, GRAVITY_Y, 0, DT, ITERATIONS, LINEAR_DAMPING, ANGULAR_DAMPING);
    glue.td_body_create(settle, 1, 0, 5, 0.5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0.2, 0.5, 1, 1);
    const hit = glue.td_alloc(8);
    assertEqual(glue.td_world_raycast(settle, 0, 5, 0, 0, -1, 0, 20, hit), 1, 'raycast hit');
    const record = new Float64Array(memory.buffer, hit, 8);
    const distance = record[7];
    assert(distance > 0 && distance < 20, `raycast distance: ${distance}`);
    assertEqual(5 - record[2], distance, 'distance must equal origin.y - point.y');
    const miss = glue.td_world_raycast(settle, 0, 5, 0, 0, 1, 0, 20, hit);
    assertEqual(miss, 0, 'upward ray must miss');
    const degenerate = glue.td_world_raycast(settle, 0, 5, 0, 0, 0, 0, 20, hit);
    assertEqual(degenerate, 0, 'zero direction must miss');
    glue.td_free(hit, 8);
    glue.td_world_destroy(settle);
  });

  await check('impulses and forces move a body, and static bodies ignore them', () => {
    const id = glue.td_world_create(0, 0, 0, DT, ITERATIONS, 0, 0);
    const moving = glue.td_body_create(id, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true, 2, 0.2, 0.5, 1, 1);
    const still = glue.td_body_create(id, 0, 0.5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0.2, 0.5, 1, 1);
    glue.td_body_apply_impulse(id, moving, 4, 0, 0);
    glue.td_body_apply_impulse(id, still, 4, 0, 0);
    const state = new Float64Array(memory.buffer, scratch, 12);
    glue.td_body_get_state(id, moving, scratch);
    assertEqual(state[6], 2, 'impulse/mass = 4/2');
    glue.td_body_apply_force(id, moving, 6, 0, 0);
    glue.td_body_get_state(id, still, scratch);
    assertEqual(state[6], 0, 'static body must ignore an impulse');
    glue.td_body_set_state(id, moving, 1 | 4, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0);
    glue.td_body_get_state(id, moving, scratch);
    assertEqual(state[0], 0, 'set_state position.x');
    assertEqual(state[1], 1, 'set_state position.y');
    assertEqual(state[6], 3, 'set_state velocity.x');
    glue.td_world_destroy(id);
  });

  await check('dispose empties a world but keeps its id valid', () => {
    glue.td_world_dispose(world);
    assertEqual(glue.td_world_body_count(world), 0, 'body count after dispose');
    assertEqual(glue.td_world_fixed_dt(world), DT, 'fixedDt after dispose');
    glue.td_world_step(world, DT);
    assertEqual(glue.td_world_contact_count(world), 0, 'contacts after dispose');
  });

  await check('destroy and dead-world calls are inert, not trapping', () => {
    glue.td_world_destroy(world);
    glue.td_world_destroy(world);
    assertEqual(glue.td_world_body_count(world), 0, 'body count of a destroyed world');
    assert(Number.isNaN(glue.td_world_fixed_dt(world)), 'destroyed world must report NaN dt');
    assertEqual(glue.td_world_step(world, DT), undefined, 'step on a destroyed world');
    assertEqual(glue.td_world_body_count(4242), 0, 'unknown world id');
  });

  await check('two identical runs produce identical states (replayability)', () => {
    const run = () => {
      const id = glue.td_world_create(0, GRAVITY_Y, 0, DT, ITERATIONS, LINEAR_DAMPING, ANGULAR_DAMPING);
      glue.td_body_create(id, 0, 0.5, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, true, 1, 0.4, 0.6, 1, 1);
      glue.td_body_create(id, 1, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0.4, 0.6, 1, 1);
      const out = glue.td_alloc(12 * 2);
      for (let i = 0; i < 240; i++) glue.td_world_step(id, DT);
      glue.td_world_write_states(id, out);
      // Copied, not viewed: the buffer is freed before the second run starts.
      const values = Array.from(new Float64Array(memory.buffer, out, 24));
      glue.td_free(out, 12 * 2);
      glue.td_world_destroy(id);
      return values;
    };
    const [first, second] = [run(), run()];
    for (let i = 0; i < first.length; i++) {
      assertEqual(second[i], first[i], `state[${i}]`);
    }
  });

  glue.td_free(scratch, 12 * 8);
}

async function checkFreshness(pkgDir, baselineDir) {
  begin('freshness');

  await check(`baseline directory exists (${displayPath(baselineDir)})`, () => {
    assert(
      existsSync(baselineDir) && statSync(baselineDir).isDirectory(),
      `${displayPath(baselineDir)} is not a directory`,
    );
  });

  // Every file in the directory is hashed, `.gitignore` included: the baseline
  // in CI is `cp -r wasm/pkg`, so both sides carry the same set.
  const digests = (dir) => {
    const map = new Map();
    for (const name of readdirSync(dir).sort()) {
      const path = join(dir, name);
      if (statSync(path).isFile()) map.set(name, sha256(readFileSync(path)));
    }
    return map;
  };

  await check('committed artifact is byte-identical to a rebuild', () => {
    const mine = digests(pkgDir);
    const theirs = digests(baselineDir);
    const { onlyLeft, onlyRight } = sortedDiff([...mine.keys()], [...theirs.keys()]);
    const problems = [];
    if (onlyLeft.length) problems.push(`only in ${displayPath(pkgDir)}: ${onlyLeft.join(', ')}`);
    if (onlyRight.length) {
      problems.push(`only in ${displayPath(baselineDir)}: ${onlyRight.join(', ')}`);
    }
    for (const name of [...mine.keys()].filter((n) => theirs.has(n))) {
      if (mine.get(name) !== theirs.get(name)) problems.push(`differs: ${name}`);
    }
    assert(
      problems.length === 0,
      `${problems.join('\n')}\n` +
        'The committed wasm/pkg was not produced by the current rust/ sources.\n' +
        'Run `npm run build:wasm` and commit the result, or revert the Rust change.',
    );
    return `${mine.size} files, sha256 matched`;
  });
}

// ---------------------------------------------------------------------------

async function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    process.stderr.write(`check_wasm_artifact: ${err.message}\n`);
    process.exit(1);
  }

  process.stdout.write(`threedream wasm artifact check\n  pkg:      ${displayPath(opts.pkg)}\n`);
  if (opts.baseline) process.stdout.write(`  baseline: ${displayPath(opts.baseline)}\n`);

  try {
    await checkShape(opts.pkg);
    const abi = await checkAbi(opts.pkg);
    if (abi) {
      await checkProvenance(opts.pkg, abi.binary, abi.mod);
      await checkBehaviour(opts.pkg);
    }
    if (opts.baseline) await checkFreshness(opts.pkg, opts.baseline);
  } catch (err) {
    // Anything reaching here is a bug in this script or an unreadable checkout,
    // not a finding about the artifact. Reported distinctly, still exit 1.
    failures.push({
      section: 'fatal',
      name: 'check run',
      detail: err instanceof Error ? `${err.message}\n${err.stack}` : String(err),
    });
  }

  process.stdout.write(
    `\n${passed} passed, ${failures.length} failed${opts.baseline ? '' : ' (no --baseline: freshness skipped)'}\n`,
  );
  if (failures.length) {
    process.stdout.write('\nfailures:\n');
    for (const f of failures) {
      process.stdout.write(`  [${f.section}] ${f.name}\n    ${f.detail.split('\n').join('\n    ')}\n`);
    }
    process.exit(1);
  }
}

await main();

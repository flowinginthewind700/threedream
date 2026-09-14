#!/usr/bin/env node
/**
 * Re-measures every Unreal Engine number quoted in
 * docs/feasibility-rust-wasm-webgpu.md, straight from the reference submodule.
 *
 * Why this exists: the first draft of that report quoted counts from an earlier
 * ad-hoc session, and half of them were wrong because a "count" is only
 * meaningful next to the exact pattern and file filter that produced it. Prose
 * numbers have no way to fail, so they drift. This script is the single source
 * of truth: run it, paste the table, and every figure in the report is
 * reproducible by one command.
 *
 * It also turns the three findings that actually decide the architecture into
 * hard invariants, checked with a non-zero exit:
 *
 *   1. every Chaos source file carries Epic's copyright header,
 *   2. no Chaos source file carries an OSI licence header, and
 *   3. Chaos has no GPU/RHI path at all.
 *
 * 1 + 2 are why "extract the Unreal kernel" is legally impossible and the work
 * has to be clean-room. 3 is why our GPU layer is a scale story rather than a
 * port of something Epic already built.
 *
 * Requires thirdparty/UnrealEngine (see scripts/clone_unreal_reference.sh, and
 * EULA access to Epic's private repo). Exits 0 when the submodule is absent so
 * a normal checkout and CI are unaffected -- CI has no EULA access by design.
 *
 * Run: node scripts/audit_unreal_reference.mjs [--ue <path>] [--json]
 */

import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import { resolve, relative, join } from 'node:path';

const CODE_EXT = new Set(['.h', '.cpp', '.inl']);

/**
 * One row of the report table. `mode` mirrors how the original number was
 * taken, because "how many files contain X" and "how many times does X occur"
 * are different questions and conflating them is what produced the bad draft.
 *   files - number of code files with at least one match (rg -l | wc -l)
 *   hits  - total matches across all lines           (rg -o | wc -l)
 * `regex: false` means `pattern` is matched literally (rg -F).
 */
const METRICS = [
  { key: 'coreminimal',    label: 'Files including `CoreMinimal.h`',                  mode: 'files', pattern: 'CoreMinimal.h', regex: false },
  { key: 'core_include',   label: 'Files including a Core-family header',             mode: 'files', pattern: '#include "(CoreMinimal|CoreTypes|Containers|HAL|Templates|Misc)' },
  { key: 'platform_macro', label: 'Files using `PLATFORM_*` / `UE_*` macros',         mode: 'files', pattern: 'PLATFORM_[A-Z]|UE_[A-Z]' },
  { key: 'thread_files',   label: 'Files using thread primitives',                    mode: 'files', pattern: 'FRunnableThread|ParallelFor|FTaskGraph|FScopeLock|TQueue|FRWLock|TAtomic|FPlatformAtomics|LowLevelTasks|FThreadSafeCounter' },
  { key: 'simd_files',     label: 'Files using `VectorRegister` / SIMD',              mode: 'files', pattern: 'VectorRegister|SIMD' },
  { key: 'template_files', label: 'Files with `template<`',                           mode: 'files', pattern: 'template<', regex: false },
  { key: 'template_hits',  label: 'Occurrences of `template<`',                       mode: 'hits',  pattern: 'template<', regex: false },
  { key: 'typename_t',     label: 'Occurrences of `template<typename T`',             mode: 'hits',  pattern: 'template<typename T', regex: false },
  { key: 'ispc_files',     label: 'Files referencing ISPC',                           mode: 'files', pattern: 'ISPC', regex: false },
  { key: 'ispc_hits',      label: 'Occurrences of `ISPC`',                            mode: 'hits',  pattern: 'ISPC', regex: false },
  { key: 'cwus',           label: 'Occurrences of `COMPILE_WITHOUT_UNREAL_SUPPORT`',  mode: 'hits',  pattern: 'COMPILE_WITHOUT_UNREAL_SUPPORT', regex: false },
  { key: 'gpu_hits',       label: 'Files touching GPU / RHI / RDG',                   mode: 'files', pattern: 'FRDGBuffer|RDG_BUFFER|AddPass|FComputeShaderUtils|ENQUEUE_RENDER_COMMAND|RenderCore|RHI\\.h|FRHICommandList' },
];

/** Headers that would make a clean-room rewrite unnecessary. Expected: none. */
const OSI_PATTERN = 'SPDX-License-Identifier|Apache License|MIT License|BSD License|GNU General Public';
/** The Epic header every file is expected to carry. */
const COPYRIGHT = 'Copyright Epic Games';

const CORE_DEP_LABELS = ['Core', 'CoreUObject', 'GeometryCore', 'AutoRTFM'];

function parseArgs(argv) {
  const out = { ue: null, json: false };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--ue') out.ue = argv[++i];
    else if (argv[i] === '--json') out.json = true;
    else { console.error(`unknown argument: ${argv[i]}`); process.exit(2); }
  }
  return out;
}

/** Every regular file under `dir`, as paths relative to `dir`. */
function walk(dir, base = dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) walk(full, base, out);
    else if (entry.isFile()) out.push(relative(base, full));
  }
  return out;
}

function readText(root, rel) {
  return readFileSync(resolve(root, rel), 'utf8');
}

/**
 * Line count with the same semantics as `rg -c ''` / `wc -l`, because a
 * trailing newline ends a line rather than starting an empty one. Counting
 * `split('\n').length` instead inflates every file by one and quietly disagrees
 * with the tool the original figures came from.
 */
function countLines(text) {
  if (text.length === 0) return 0;
  let n = 0;
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) n++;
  return text.charCodeAt(text.length - 1) === 10 ? n : n + 1;
}

/** Count matches per line, the same shape as `rg -o | wc -l`. */
function countHits(text, pattern, literal) {
  let total = 0;
  for (const line of text.split('\n')) {
    if (literal) {
      let idx = line.indexOf(pattern);
      while (idx !== -1) { total++; idx = line.indexOf(pattern, idx + pattern.length); }
    } else {
      const m = line.match(new RegExp(pattern, 'g'));
      if (m) total += m.length;
    }
  }
  return total;
}

function locateUnreal(explicit) {
  const repoRoot = resolve(import.meta.dirname, '..');
  const candidates = [
    explicit,
    process.env.UE_ROOT,
    join(repoRoot, 'thirdparty', 'UnrealEngine'),
    // A sibling checkout, for when this repo is not the one holding the tree.
    join(resolve(repoRoot, '..'), 'robotworld', 'thirdparty', 'UnrealEngine'),
  ].filter(Boolean);
  for (const c of candidates) {
    const chaos = join(c, 'Engine/Source/Runtime/Experimental/Chaos');
    if (existsSync(chaos) && statSync(chaos).isDirectory()) return { root: c, chaos };
  }
  return null;
}

const args = parseArgs(process.argv.slice(2));
const found = locateUnreal(args.ue);

if (!found) {
  console.log('thirdparty/UnrealEngine not present; skipping the reference audit.');
  console.log('Fetch it with: bash scripts/clone_unreal_reference.sh');
  console.log('(Needs EULA access to Epic\'s private repo. CI skips this by design.)');
  process.exit(0);
}

const { root, chaos } = found;
const allFiles = walk(chaos);
const codeFiles = allFiles.filter((f) => CODE_EXT.has(f.slice(f.lastIndexOf('.')))).sort();

// Read each code file once; every metric below is derived from this same pass so
// the numbers cannot disagree with each other about which files were scanned.
const texts = new Map();
let loc = 0;
for (const rel of codeFiles) {
  const t = readText(chaos, rel);
  texts.set(rel, t);
  loc += countLines(t);
}

const metrics = {};
for (const m of METRICS) {
  let n = 0;
  for (const [rel, t] of texts) {
    if (m.mode === 'files') {
      if (countHits(t, m.pattern, m.regex === false) > 0) n++;
    } else {
      n += countHits(t, m.pattern, m.regex === false);
    }
  }
  metrics[m.key] = n;
}

const copyrightFiles = [...texts].filter(([, t]) => countHits(t, COPYRIGHT, true) > 0).length;
// Case-insensitive, and deliberately not /g: with the global flag .test() is
// stateful across calls, so reusing one regex over a file's lines can skip a
// real match. Presence is all this needs.
const osiRe = new RegExp(OSI_PATTERN, 'i');
const osiFiles = [...texts].filter(([, t]) => osiRe.test(t)).length;

// Dependency modules are measured over their own code files, to size the
// transitive surface Chaos pulls in. `Core` is the one that matters: it is the
// module Chaos cannot be separated from without a rewrite.
const coreDeps = {};
for (const name of CORE_DEP_LABELS) {
  const dir = join(root, 'Engine/Source/Runtime', name);
  if (!existsSync(dir)) { coreDeps[name] = null; continue; }
  let n = 0;
  for (const rel of walk(dir)) {
    if (!CODE_EXT.has(rel.slice(rel.lastIndexOf('.')))) continue;
    n += countLines(readText(dir, rel));
  }
  coreDeps[name] = n;
}

const invariantFailures = [];
if (copyrightFiles !== codeFiles.length) {
  invariantFailures.push(
    `expected all ${codeFiles.length} Chaos source files to carry Epic's copyright, found ${copyrightFiles}`,
  );
}
if (osiFiles !== 0) {
  invariantFailures.push(
    `expected 0 Chaos source files with an OSI licence header, found ${osiFiles}`,
  );
}
if (metrics.gpu_hits !== 0) {
  invariantFailures.push(
    `expected Chaos to have no GPU/RHI path, found ${metrics.gpu_hits} files touching it`,
  );
}

const summary = {
  ueRoot: root,
  ueCommit: readGitCommit(root),
  allFiles: allFiles.length,
  codeFiles: codeFiles.length,
  loc,
  copyrightFiles,
  osiFiles,
  metrics,
  coreDeps,
};

function readGitCommit(ueRoot) {
  // Submodules store their git dir in a file, not a directory; we only want the
  // checked-out commit, so read HEAD and follow a ref if it is symbolic.
  try {
    const dotGit = join(ueRoot, '.git');
    let gitDir = statSync(dotGit).isDirectory() ? dotGit : null;
    if (!gitDir) {
      const pointer = readFileSync(dotGit, 'utf8').match(/^gitdir:\s*(.+)$/m);
      if (!pointer) return null;
      gitDir = resolve(ueRoot, pointer[1].trim());
    }
    let head = readFileSync(join(gitDir, 'HEAD'), 'utf8').trim();
    if (head.startsWith('ref:')) {
      const refPath = join(gitDir, head.slice(4).trim());
      head = existsSync(refPath) ? readFileSync(refPath, 'utf8').trim() : head;
    }
    return /^[0-9a-f]{40}$/.test(head) ? head : null;
  } catch {
    return null;
  }
}

if (args.json) {
  console.log(JSON.stringify(summary, null, 2));
} else {
  const commit = summary.ueCommit ? summary.ueCommit.slice(0, 8) : 'unknown';
  console.log(`Unreal Engine reference: ${root}`);
  console.log(`commit: ${commit}`);
  console.log('');
  console.log('## Chaos (Engine/Source/Runtime/Experimental/Chaos)');
  console.log('');
  console.log('| Metric | Value |');
  console.log('|--------|-------|');
  console.log(`| All files | ${summary.allFiles} |`);
  console.log(`| Code files (\\*.h / \\*.cpp / \\*.inl) | ${summary.codeFiles} |`);
  console.log(`| Lines of code (code files) | ${summary.loc.toLocaleString('en-US')} |`);
  console.log(`| Files with Epic copyright | ${summary.copyrightFiles} |`);
  console.log(`| Files with an OSI licence header | **${summary.osiFiles}** |`);
  for (const m of METRICS) {
    console.log(`| ${m.label} | ${metrics[m.key].toLocaleString('en-US')} |`);
  }
  console.log('');
  console.log('## Transitive dependency surface (lines of code)');
  console.log('');
  console.log('| Module | Lines |');
  console.log('|--------|-------|');
  for (const [name, n] of Object.entries(coreDeps)) {
    console.log(`| ${name} | ${n === null ? 'not at this path' : n.toLocaleString('en-US')} |`);
  }
  console.log('');
  if (invariantFailures.length) {
    console.log('INVARIANTS: FAILED');
    for (const f of invariantFailures) console.log(`  - ${f}`);
  } else {
    console.log('INVARIANTS: OK');
    console.log('  - every Chaos source file carries Epic copyright');
    console.log('  - no Chaos source file carries an OSI licence header');
    console.log('  - Chaos has no GPU / RHI / RDG path');
  }
}

process.exit(invariantFailures.length ? 1 : 0);

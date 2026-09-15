/**
 * `gpu/softWgsl.ts` -- the generated soft-body shader text and the tables around it.
 *
 * Text-level assertions, for the reason `particle_wgsl.test.ts` gives: the shader is
 * only compiled on a machine with a WebGPU implementation, so the drift a compiler
 * would catch *there* has to be caught here, everywhere. What is checked is what a
 * compiler cannot check for us -- that the `Params` member order still produces the
 * byte offsets `writeSoftParams` fills, that the constants in the WGSL are the ones
 * `softIslands.ts` and `softOptions.ts` own, that `solve` reads the batch the
 * dispatcher puts on the z axis, and that the six entry points are the six passes
 * `softCpu.ts` calls, in its order.
 *
 * The last one is why this file reads `softCpu.ts` as text. Both tiers are generated
 * from the same constants, so a formula that drifts surfaces as a parity failure in
 * `e2e/soft_gpu.spec.ts`; a *pass order* that drifts surfaces as a simulation that is
 * subtly wrong in a way no tolerance names. Extracting the CPU's `step()` and
 * comparing its call sequence to `SOFT_KERNELS` makes the dispatch order a checked
 * property of the two files rather than a claim in a comment.
 *
 * Actual compilation is asserted against a real device in `e2e/soft_gpu.spec.ts`.
 * This file is what runs in the sub-second unit loop.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { ComputeBinding } from '../src/gpu/compute.js';
import { REQUESTED_LIMITS } from '../src/gpu/device.js';
import { DEFAULT_BOUNDS } from '../src/gpu/particleField.js';
import { PARTICLE_BINDINGS } from '../src/gpu/particleWgsl.js';
import { MAX_COLORS, type SoftBatch } from '../src/gpu/softColoring.js';
import { createCpuSoftSystem } from '../src/gpu/softCpu.js';
import { NODE_SENTINEL, SOFT_WORKGROUP_SIZE, softWorkgroups } from '../src/gpu/softIslands.js';
import { SOFT_OFFSET, SOFT_SCENES, SOFT_STRIDE, SoftMesh } from '../src/gpu/softMesh.js';
import {
  SOFT_BOUNDS_MODE_BITS,
  SOFT_FIXED_DISPATCHES,
  SOFT_FLAG,
  SOFT_PARAMS_BYTES,
  SOFT_PARAM_WORD,
  buildSoftLayout,
  resolveSoftOptions,
  writeSoftParams,
} from '../src/gpu/softOptions.js';
import {
  SOFT_BASELINE_STORAGE_BUFFERS,
  SOFT_BATCH_U32_PER_COLOR,
  SOFT_BINDINGS,
  SOFT_EDGE_F32_PER_CONSTRAINT,
  SOFT_ENDS_U32_PER_CONSTRAINT,
  SOFT_GROUP_STATE,
  SOFT_GROUP_STATIC,
  SOFT_KERNELS,
  SOFT_KERNEL_DISPATCH,
  SOFT_ORDER_U32_PER_CONSTRAINT,
  SOFT_PRED_VECS_PER_NODE,
  SOFT_PUBLISH_FLOATS_PER_NODE,
  SOFT_SLEEP_WORDS_PER_ISLAND,
  SOFT_STATE_FLOATS_PER_NODE,
  SOFT_STATE_VECS_PER_NODE,
  SOFT_STAT_WORDS,
  SOFT_STAT_WORD,
  SOFT_STORAGE_BINDINGS,
  SOFT_WGSL_PARAMS_LAYOUT,
  softBindingsForGroup,
  softShaderSource,
  softSolveDispatch,
  type SoftKernel,
  type SoftKernelDispatch,
} from '../src/gpu/softWgsl.js';

const src = softShaderSource();

/**
 * `src` with the line comments stripped.
 *
 * Several assertions below are "this construct must not appear", and the prose
 * explaining *why* it must not appear names the construct -- the module header alone
 * says "subgroup" four times. WGSL has no string literals, so a comment cannot be
 * hiding inside one, and the generated text uses no block comments, so stripping `//`
 * to end of line strips all of it.
 */
const code = src.replace(/\/\/[^\n]*/g, '');

/** WGSL alignment and size, in bytes. `vec3` is the interesting one: 12 in 16. */
const ALIGN: Record<string, number> = { vec3: 16, f32: 4, u32: 4 };
const SIZE: Record<string, number> = { vec3: 12, f32: 4, u32: 4 };
const WGSL_TYPE: Record<string, string> = { vec3: 'vec3<f32>', f32: 'f32', u32: 'u32' };

/**
 * Body of one `fn`, from its declaration to the matching closing brace.
 *
 * Sliced by braces rather than by line numbers, because the whole point is that the
 * assertions survive an edit that moves a kernel.
 */
function bodyOf(name: string): string {
  const start = src.indexOf(`fn ${name}(`);
  expect(start, `kernel ${name} is missing`).toBeGreaterThan(-1);
  return tsBodyOf(src, start);
}

/**
 * One kernel's body with the comments taken out.
 *
 * Half the assertions below are "this construct does not appear", and the prose in
 * the shader explaining *why* it does not appear names the construct -- `solve`
 * carries a comment saying "No atomics anywhere in here".
 */
function bareBody(name: SoftKernel): string {
  return bodyOf(name).replace(/\/\/[^\n]*/g, '');
}

/** The same brace walk, over TypeScript as well as WGSL. */
function tsBodyOf(text: string, from: number): string {
  const open = text.indexOf('{', from);
  expect(open, 'no opening brace after the signature').toBeGreaterThan(-1);
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    if (text[i] === '{') depth++;
    else if (text[i] === '}') {
      depth--;
      if (depth === 0) return text.slice(open, i + 1);
    }
  }
  throw new Error('unbalanced braces');
}

/** Every member `Params` declares, in the order and with the type the shader gives it. */
function declaredMembers(): readonly (readonly [string, string])[] {
  const start = src.indexOf('struct Params {');
  expect(start, 'struct Params is missing').toBeGreaterThan(-1);
  const end = src.indexOf('};', start);
  return [...src.slice(start, end).matchAll(/^ {2}(\w+): (vec3<f32>|f32|u32),$/gm)].map(
    (m) => [m[1], m[2]] as const,
  );
}

describe('Params layout', () => {
  it('matches the offsets WGSL alignment rules produce', () => {
    // Recomputed here rather than read from SOFT_PARAM_WORD: the whole value of the
    // check is that the two derivations are independent.
    let offset = 0;
    for (const m of SOFT_WGSL_PARAMS_LAYOUT) {
      offset = Math.ceil(offset / ALIGN[m.kind]) * ALIGN[m.kind];
      expect(offset, `${m.name} offset`).toBe(m.word * 4);
      offset += SIZE[m.kind];
    }
    // A struct's size rounds up to its own alignment, which is the largest member
    // alignment -- 16 here, from the three vec3s. 96 is already a multiple of it,
    // which is what makes the uniform 24 whole words with no tail the CPU must know.
    expect(Math.ceil(offset / 16) * 16).toBe(SOFT_PARAMS_BYTES);
    expect(SOFT_PARAMS_BYTES).toBe(96);
  });

  it('covers every word of the uniform exactly once', () => {
    const words: number[] = [];
    for (const m of SOFT_WGSL_PARAMS_LAYOUT) {
      const span = m.kind === 'vec3' ? 3 : 1;
      for (let k = 0; k < span; k++) words.push(m.word + k);
    }
    const every = Array.from({ length: SOFT_PARAMS_BYTES / 4 }, (_, i) => i);
    expect([...words].sort((a, b) => a - b)).toEqual(every);
    expect(Object.values(SOFT_PARAM_WORD).sort((a, b) => a - b)).toEqual(every);
  });

  it('declares the members in this order and with these types', () => {
    expect(declaredMembers()).toEqual(
      SOFT_WGSL_PARAMS_LAYOUT.map((m) => [m.name, WGSL_TYPE[m.kind]] as const),
    );
  });

  it('reads the integers as u32 and keeps them in the tail', () => {
    // writeSoftParams fills words 0-15 through a Float32Array and words 16-23
    // through a Uint32Array. Declaring one of the integers f32 would reinterpret the
    // bit pattern, and declaring one of the floats u32 would read a normal number as
    // a very large one. Both are silent and both are fatal to the simulation.
    const ints = [
      'count',
      'islandCount',
      'paddedNodes',
      'sleepAfter',
      'flags',
      'constraintCount',
      'padA',
      'padB',
    ];
    for (const name of ints) {
      const member = SOFT_WGSL_PARAMS_LAYOUT.find((m) => m.name === name)!;
      expect(member.kind, name).toBe('u32');
      expect(member.word, `${name} word`).toBeGreaterThanOrEqual(16);
    }
    expect(SOFT_WGSL_PARAMS_LAYOUT.filter((m) => m.kind === 'u32')).toHaveLength(ints.length);
    for (const m of SOFT_WGSL_PARAMS_LAYOUT) {
      if (m.kind !== 'u32') expect(m.word, `${m.name} word`).toBeLessThan(16);
    }
  });

  it('agrees with writeSoftParams word for word', () => {
    // The check both derivations exist for: pack a uniform the CPU way, then read
    // every member back at the offset its WGSL declaration order implies, through
    // the view its WGSL type implies. A member moved in either file fails here, on a
    // machine with no device attached.
    const resolved = resolveSoftOptions({
      gravity: [1.5, -9.81, -0.25],
      damping: 0.35,
      restitution: 0.45,
      maxSpeed: 17.5,
      stiffness: 0.75,
      iterations: 4,
      boundsMode: 'none',
      fixedDt: 1 / 120,
      sleep: true,
      sleepThreshold: 0.25,
      sleepAfter: 9,
    });
    const frame = {
      dt: 1 / 90,
      count: 4200,
      islandCount: 13,
      paddedNodes: 4224,
      constraintCount: 16_000,
      sleepAfter: resolved.sleepAfter,
      bounds: DEFAULT_BOUNDS,
    };
    const buffer = new ArrayBuffer(SOFT_PARAMS_BYTES);
    writeSoftParams(buffer, resolved, frame);
    const floats = new Float32Array(buffer);
    const ints = new Uint32Array(buffer);
    const f = Math.fround;
    const want: Record<string, number | readonly number[]> = {
      gravity: resolved.gravity,
      damping: resolved.damping,
      boundsMin: DEFAULT_BOUNDS.min,
      invDt: 1 / frame.dt,
      boundsMax: DEFAULT_BOUNDS.max,
      maxSpeed: resolved.maxSpeed,
      dt: frame.dt,
      stiffness: resolved.stiffness,
      restitution: resolved.restitution,
      sleepThresholdSq: resolved.sleepThreshold * resolved.sleepThreshold,
      count: frame.count,
      islandCount: frame.islandCount,
      paddedNodes: frame.paddedNodes,
      sleepAfter: frame.sleepAfter,
      flags: SOFT_FLAG.sleep | (SOFT_BOUNDS_MODE_BITS.none << SOFT_FLAG.boundsShift),
      constraintCount: frame.constraintCount,
      padA: 0,
      padB: 0,
    };
    // Bidirectional, so a new member without an expectation fails instead of being
    // silently skipped.
    expect(Object.keys(want).sort()).toEqual(SOFT_WGSL_PARAMS_LAYOUT.map((m) => m.name).sort());
    for (const m of SOFT_WGSL_PARAMS_LAYOUT) {
      const expected = want[m.name];
      // Narrowed on typeof rather than Array.isArray, which does not exclude a
      // readonly array. The view each member is read through is the one its WGSL
      // type implies, so an integer declared f32 fails here instead of reading a
      // bit pattern back as a very small float.
      if (typeof expected === 'number') {
        if (m.kind === 'u32') {
          expect(ints[m.word], m.name).toBe(expected);
        } else {
          expect(m.kind, m.name).not.toBe('vec3');
          expect(floats[m.word], m.name).toBe(f(expected));
        }
      } else {
        expect(m.kind, m.name).toBe('vec3');
        for (let k = 0; k < 3; k++) {
          expect(floats[m.word + k], `${m.name}[${k}]`).toBe(f(expected[k]));
        }
      }
    }
  });

  it('declares the three words no kernel reads', () => {
    // Documented as unread, so pinned as unread: a kernel that started reading a pad
    // would be a kernel whose counts the shader and the CPU now disagree about.
    expect(code).not.toContain('params.paddedNodes');
    expect(code).not.toContain('params.padA');
    expect(code).not.toContain('params.padB');
    expect(SOFT_WGSL_PARAMS_LAYOUT.filter((m) => m.name === 'paddedNodes')).toHaveLength(1);
  });
});

describe('constants cannot drift from the CPU reference', () => {
  const constOf = (name: string): string => {
    const m = src.match(new RegExp('^const ' + name + ': (\\w+) = ([^;]+);$', 'm'));
    expect(m, `const ${name} is missing`).not.toBeNull();
    return m![2];
  };

  it('uses the same sentinel and workgroup size', () => {
    expect(constOf('SENTINEL')).toBe(`${NODE_SENTINEL}u`);
    expect(NODE_SENTINEL).toBe(0xffffffff);
    // WG is the stride the island-mapped kernels walk the padded node order with,
    // and softIslands pads to the same number. Disagreeing here reads one island's
    // nodes as another's, which no bounds check would catch.
    expect(constOf('WG')).toBe(`${SOFT_WORKGROUP_SIZE}u`);
    expect(SOFT_WORKGROUP_SIZE).toBe(64);
  });

  it('uses the same flag bits and bounds-mode encoding', () => {
    expect(constOf('FLAG_SLEEP')).toBe(`${SOFT_FLAG.sleep}u`);
    expect(constOf('BOUNDS_SHIFT')).toBe(`${SOFT_FLAG.boundsShift}u`);
    expect(constOf('BOUNDS_REFLECT')).toBe(`${SOFT_BOUNDS_MODE_BITS.reflect}u`);
    expect(constOf('BOUNDS_NONE')).toBe(`${SOFT_BOUNDS_MODE_BITS.none}u`);
    expect(SOFT_FLAG.sleep).toBe(1);
  });

  it('knows two bounds modes, because wrapping a node tears its edges', () => {
    // The particle layer offers wrap. A wrapped node moves away from every edge
    // attached to it, so there is no third constant here, no third branch in
    // finalize, and resolveSoftOptions refuses the mode by name.
    expect(Object.keys(SOFT_BOUNDS_MODE_BITS).sort()).toEqual(['none', 'reflect']);
    expect(src).not.toContain('BOUNDS_WRAP');
    expect(code).not.toContain('wrap');
  });

  it('uses the same stats words it exports', () => {
    expect(constOf('STAT_ESCAPED')).toBe(`${SOFT_STAT_WORD.escaped}u`);
    expect(constOf('STAT_MAX_SPEED_SQ')).toBe(`${SOFT_STAT_WORD.maxSpeedSq}u`);
    expect(constOf('STAT_MAX_ERROR')).toBe(`${SOFT_STAT_WORD.maxConstraintError}u`);
    expect(constOf('STAT_SLEEPING')).toBe(`${SOFT_STAT_WORD.sleepingIslands}u`);
    expect(Object.values(SOFT_STAT_WORD).sort((a, b) => a - b)).toEqual(
      Array.from({ length: SOFT_STAT_WORDS }, (_, i) => i),
    );
    // Every index into the buffer is one of those four names: no literal word
    // numbers, and nothing writing the reserved fifth.
    const indexed = [...code.matchAll(/statsBuf\[(\w+)\]/g)].map((m) => m[1]);
    expect([...new Set(indexed)].sort()).toEqual([
      'STAT_ESCAPED',
      'STAT_MAX_ERROR',
      'STAT_MAX_SPEED_SQ',
      'STAT_SLEEPING',
    ]);
    expect(indexed).toHaveLength(9);
  });
});

describe('entry points', () => {
  it('declares all six, in dispatch order', () => {
    expect(SOFT_KERNELS).toEqual([
      'predict',
      'solve',
      'finalize',
      'measure',
      'sleep_update',
      'publish',
    ]);
    const declared = [...src.matchAll(/^@compute[^\n]*\nfn (\w+)\(/gm)].map((m) => m[1]);
    expect(declared).toEqual([...SOFT_KERNELS]);
  });

  it('is one longer than the fixed dispatch count', () => {
    // `solve` is the only kernel that scales with the graph -- `iterations * colors`
    // of it, and one of each of the other five. SOFT_FIXED_DISPATCHES is the number
    // the HUD subtracts to show the part the iteration slider does not move, so the
    // two have to stay one apart.
    expect(SOFT_KERNELS.length).toBe(SOFT_FIXED_DISPATCHES + 1);
    expect(SOFT_FIXED_DISPATCHES).toBe(5);
  });

  it('gives every kernel a dispatch rule', () => {
    expect(Object.keys(SOFT_KERNEL_DISPATCH).sort()).toEqual([...SOFT_KERNELS].sort());
    expect(SOFT_KERNEL_DISPATCH.solve).toBe('batch');
    // The two island-mapped kernels walk the padded node order, so their 64 lanes
    // all belong to one island and the sleep test is one load per workgroup.
    expect(SOFT_KERNEL_DISPATCH.predict).toBe('nodeWorkgroups');
    expect(SOFT_KERNEL_DISPATCH.finalize).toBe('nodeWorkgroups');
    expect(SOFT_KERNEL_DISPATCH.measure).toBe('constraints');
    expect(SOFT_KERNEL_DISPATCH.sleep_update).toBe('islands');
    expect(SOFT_KERNEL_DISPATCH.publish).toBe('count');
  });

  it('guards each kernel with the bound its dispatch rule implies', () => {
    // A rule and a guard that disagree either drops the tail of a mesh or reads past
    // the end of a buffer, and neither is visible until a scene changes size.
    expect(bodyOf('predict')).toContain('let i = nodeOrderBuf[wg * WG + lid.x];');
    expect(bodyOf('predict')).toContain('if (i == SENTINEL) { return; }');
    expect(bodyOf('finalize')).toContain('let i = nodeOrderBuf[wg * WG + lid.x];');
    expect(bodyOf('finalize')).toContain('if (i == SENTINEL) { return; }');
    expect(bodyOf('solve')).toContain('if (gid.x >= batch.y) { return; }');
    expect(bodyOf('measure')).toContain('if (e >= params.constraintCount) { return; }');
    expect(bodyOf('sleep_update')).toContain('if (k >= params.islandCount) { return; }');
    expect(bodyOf('publish')).toContain('if (i >= params.count) { return; }');
  });

  it('sizes every workgroup at 64 and nothing else', () => {
    const sizes = [...src.matchAll(/@workgroup_size\((\d+)\)/g)].map((m) => Number(m[1]));
    expect(sizes).toHaveLength(SOFT_KERNELS.length);
    expect(new Set(sizes)).toEqual(new Set([SOFT_WORKGROUP_SIZE]));
    expect(SOFT_WORKGROUP_SIZE).toBe(64);
  });

  it('uses no subgroups, no f64 and no workgroup memory', () => {
    // Subgroups are unavailable on the adapters the feasibility study measured and
    // fail at pipeline-creation time on some drivers; f64 does not exist in WGSL at
    // all. Workgroup memory would add a barrier, and the only barrier this pipeline
    // has is the ordering between dispatches.
    expect(code).not.toMatch(/subgroup/);
    expect(code).not.toMatch(/f64/);
    expect(code).not.toMatch(/var<workgroup>/);
    expect(code).not.toMatch(/barrier|workgroupUniform|textureGather|textureStore/);
  });

  it('sums componentwise instead of calling dot', () => {
    // A dot product is free to be contracted into a multiply-add, and softCpu.ts
    // writes its sums out explicitly in the same association. Matching it is what
    // keeps the two tiers within tolerance instead of merely close.
    expect(code).not.toMatch(/\bdot\(/);
    expect(bodyOf('sumSq')).toContain('(v.x * v.x + v.y * v.y) + v.z * v.z');
    // Both callers of the helper are the two places a squared length is needed.
    expect([...code.matchAll(/sumSq\(/g)]).toHaveLength(4);
  });

  it('reads the batch off the z axis, where the dispatcher puts it', () => {
    // The one line softSolveDispatch exists to document. A dispatcher that put the
    // color in y would compile, run, and solve color 0 iterations * colors times.
    expect(bodyOf('solve')).toContain('let batch = batchBuf[gid.z];');
    expect(bodyOf('solve')).toContain('let e = orderBuf[batch.x + gid.x];');
    expect(code).not.toContain('gid.y');
  });
});

describe('bindings', () => {
  it('assigns each group/binding pair once', () => {
    const pairs = SOFT_BINDINGS.map((b) => `${b.group}:${b.binding}`);
    expect(new Set(pairs).size).toBe(pairs.length);
    const names = SOFT_BINDINGS.map((b) => b.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it('declares exactly the table, with the matching access mode', () => {
    const access = {
      uniform: 'var<uniform>',
      'storage-read': 'var<storage, read>',
      'storage-read-write': 'var<storage, read_write>',
    } as const;
    for (const b of SOFT_BINDINGS) {
      const decl = `@group(${b.group}) @binding(${b.binding}) ${access[b.kind]} ${b.name}: ${b.type};`;
      expect(src, decl).toContain(decl);
    }
    expect([...src.matchAll(/^@group\(\d+\) @binding\(\d+\)/gm)]).toHaveLength(
      SOFT_BINDINGS.length,
    );
    // One module, one Params, one uniform: a second uniform would be a second layout
    // for writeSoftParams to keep in step with.
    expect(SOFT_BINDINGS.filter((b) => b.bufferType === 'uniform')).toHaveLength(1);
    expect(src.match(/^struct Params \{$/gm)).toHaveLength(1);
  });

  it('splits the mesh-shaped buffers from the ones that change', () => {
    expect(SOFT_GROUP_STATIC).toBe(0);
    expect(SOFT_GROUP_STATE).toBe(1);
    // Group 0 is built once per mesh and never touched again, so a step, a wake()
    // and a state restore all stay inside group 1 and a scene change is the only
    // thing that reallocates group 0.
    expect(softBindingsForGroup(SOFT_GROUP_STATIC).map((b) => b.name)).toEqual([
      'params',
      'nodeOrderBuf',
      'islandOfWgBuf',
      'islandOfNodeBuf',
      'orderBuf',
      'endsBuf',
      'restBuf',
      'stiffBuf',
      'batchBuf',
    ]);
    expect(softBindingsForGroup(SOFT_GROUP_STATE).map((b) => b.name)).toEqual([
      'stateBuf',
      'predBuf',
      'asleepBuf',
      'quietBuf',
      'islandSpeedBuf',
      'statsBuf',
      'publishBuf',
    ]);
    expect(softBindingsForGroup(2)).toEqual([]);
    // The state buffer is the mesh's own f32 array, so one writeBuffer up and one
    // copy down need no repack in either direction.
    expect(softBindingsForGroup(SOFT_GROUP_STATE)[0].type).toBe('array<vec4<f32>>');
  });

  it('marks the eight static buffers read-only and nothing else', () => {
    const readOnly = SOFT_BINDINGS.filter((b) => b.kind === 'storage-read');
    expect(readOnly).toHaveLength(8);
    for (const b of readOnly) {
      expect(b.group, b.name).toBe(SOFT_GROUP_STATIC);
      expect(b.bufferType, b.name).toBe('read-only-storage');
    }
    for (const b of SOFT_BINDINGS.filter((x) => x.group === SOFT_GROUP_STATE)) {
      expect(b.kind, b.name).toBe('storage-read-write');
      expect(b.bufferType, b.name).toBe('storage');
    }
  });

  it('keeps the two reduction buffers atomic, and reduces with max and count only', () => {
    const atomics = SOFT_BINDINGS.filter((b) => b.type.includes('atomic'));
    expect(atomics.map((b) => b.name)).toEqual(['islandSpeedBuf', 'statsBuf']);
    for (const b of atomics) {
      // An atomic has to live in a read_write binding; a read-only one is a
      // pipeline-creation error on every implementation.
      expect(b.kind, b.name).toBe('storage-read-write');
      expect(b.type, b.name).toBe('array<atomic<u32>>');
    }
    const targets = [...code.matchAll(/atomic\w+\(&(\w+)\[/g)].map((m) => m[1]);
    expect([...new Set(targets)].sort()).toEqual(['islandSpeedBuf', 'statsBuf']);
    // Max and add are order-independent, so the driver's freedom to schedule them
    // costs nothing. Anything else here -- a compare-and-swap loop, an accumulate
    // into a float -- would make the digest depend on scheduling.
    const ops = [...code.matchAll(/\b(atomic\w+)\(/g)].map((m) => m[1]);
    expect([...new Set(ops)].sort()).toEqual([
      'atomicAdd',
      'atomicExchange',
      'atomicMax',
      'atomicStore',
    ]);
    // The two f32 maxima ride the bitcast trick: non-negative floats sort the same
    // way as their bit patterns, so an integer max is a float max.
    expect(bodyOf('finalize')).toContain('let bits = bitcast<u32>(s2);');
    expect(bodyOf('measure')).toContain('bitcast<u32>(err)');
  });

  it('never writes a read-only binding', () => {
    for (const b of SOFT_BINDINGS.filter((x) => x.kind === 'storage-read')) {
      expect(code, `${b.name} is assigned`).not.toMatch(
        new RegExp(`\\b${b.name}\\[[^\\]]*\\]\\s*=(?!=)`),
      );
      // Taking an element's address is how an atomic -- or an inout call -- would
      // write through a read-only binding.
      expect(code, `${b.name} is addressed`).not.toContain(`&${b.name}[`);
    }
  });

  it('uses every binding it declares', () => {
    // A declared-but-unused binding is a buffer softGpu.ts would allocate, upload and
    // bind for nothing, and the allocation is the part that shows up in a budget.
    for (const b of SOFT_BINDINGS) {
      const uses = [...code.matchAll(new RegExp(`\\b${b.name}\\b`, 'g'))].length;
      expect(uses, `${b.name} is declared but never read`).toBeGreaterThan(1);
    }
  });

  it('needs more storage buffers than a device gets by default', () => {
    expect(SOFT_STORAGE_BINDINGS).toBe(
      SOFT_BINDINGS.filter((b) => b.bufferType !== 'uniform').length,
    );
    expect(SOFT_STORAGE_BINDINGS).toBe(15);
    expect(SOFT_BASELINE_STORAGE_BUFFERS).toBe(8);
    expect(SOFT_STORAGE_BINDINGS).toBeGreaterThan(SOFT_BASELINE_STORAGE_BUFFERS);
    // The limit is per shader stage and both groups count towards it, so splitting
    // the table over more bind groups would not help. Requesting what the adapter
    // already reports is the fix, and device.ts already asks for this limit --
    // because the particle pipeline sits exactly on the baseline.
    expect(REQUESTED_LIMITS).toContain('maxStorageBuffersPerShaderStage');
    expect(PARTICLE_BINDINGS.filter((b) => b.bufferType !== 'uniform')).toHaveLength(
      SOFT_BASELINE_STORAGE_BUFFERS,
    );
  });

  it('is shaped the way compute.ts wants a binding', () => {
    // ComputePipeline builds one bind-group layout per declared group and requires
    // the groups to start at zero and be contiguous, or it binds group 2's layout to
    // index 1. SoftWgslBinding is a structural superset of ComputeBinding by design;
    // the assignment is the compile-time half of that claim.
    const asCompute: readonly ComputeBinding[] = SOFT_BINDINGS;
    expect(asCompute).toHaveLength(SOFT_BINDINGS.length);
    expect([...new Set(SOFT_BINDINGS.map((b) => b.group))].sort()).toEqual([0, 1]);
    for (const b of SOFT_BINDINGS) {
      expect(['uniform', 'read-only-storage', 'storage'], b.name).toContain(b.bufferType);
      expect(Number.isInteger(b.binding), b.name).toBe(true);
    }
  });
});

describe('buffer sizing constants', () => {
  it('agrees with the interleaved node record', () => {
    // The state buffer *is* SoftMesh.data, so these are not a second layout the two
    // tiers could disagree about: they are the one layout, and the vec4 split below
    // is only how the shader addresses it.
    expect(SOFT_STATE_FLOATS_PER_NODE).toBe(SOFT_STRIDE);
    expect(SOFT_STRIDE).toBe(8);
    expect(SOFT_STATE_VECS_PER_NODE).toBe(SOFT_STATE_FLOATS_PER_NODE / 4);
    expect(SOFT_STATE_VECS_PER_NODE).toBe(2);
    expect(SOFT_PRED_VECS_PER_NODE).toBe(1);
    expect(SOFT_PUBLISH_FLOATS_PER_NODE).toBe(3);
  });

  it('splits the node record where SOFT_OFFSET says it splits', () => {
    // invMass is the w of vec 0 and radius the w of vec 1, which is what lets the
    // solver read a mass and the bounds read a radius without a fourth buffer.
    const perVec = SOFT_STATE_FLOATS_PER_NODE / SOFT_STATE_VECS_PER_NODE;
    expect(perVec).toBe(4);
    expect(SOFT_OFFSET.position).toBe(0);
    expect(SOFT_OFFSET.invMass).toBe(perVec - 1);
    expect(SOFT_OFFSET.velocity).toBe(perVec);
    expect(SOFT_OFFSET.radius).toBe(perVec * SOFT_STATE_VECS_PER_NODE - 1);
    expect(SOFT_OFFSET.radius).toBe(SOFT_STATE_FLOATS_PER_NODE - 1);
    const predict = bodyOf('predict');
    expect(predict).toContain('let prev = stateBuf[base];');
    expect(predict).toContain('let pos = prev.xyz;');
    expect(predict).toContain('if (!(prev.w > 0.0)) {');
    expect(predict).toContain('let v = (stateBuf[base + 1u].xyz + params.gravity * dt) * damp;');
    const finalize = bodyOf('finalize');
    expect(finalize).toContain('let radius = stateBuf[base + 1u].w;');
    expect(finalize).toContain('stateBuf[base] = vec4<f32>(p, prev.w);');
    expect(finalize).toContain('stateBuf[base + 1u] = vec4<f32>(v, radius);');
    // Both writes carry through the field they did not compute, so no step can lose
    // a node's inverse mass or its radius.
    expect(finalize).not.toContain('vec4<f32>(p, 0.0)');
    expect(finalize).not.toContain('vec4<f32>(v, 0.0)');
  });

  it('addresses every buffer with the stride it declares', () => {
    const V = SOFT_STATE_VECS_PER_NODE;
    expect(bodyOf('predict')).toContain(`let base = i * ${V}u;`);
    expect(bodyOf('finalize')).toContain(`let base = i * ${V}u;`);
    expect(bodyOf('publish')).toContain(`stateBuf[i * ${V}u].xyz`);
    expect(bodyOf('solve')).toContain(`stateBuf[a * ${V}u].w`);
    expect(bodyOf('solve')).toContain(`stateBuf[b * ${V}u].w`);
    const E = SOFT_ENDS_U32_PER_CONSTRAINT;
    expect(bodyOf('solve')).toContain(`let a = endsBuf[e * ${E}u];`);
    expect(bodyOf('solve')).toContain(`let b = endsBuf[e * ${E}u + 1u];`);
    expect(bodyOf('solve')).toContain('let batch = batchBuf[gid.z];');
    expect(bodyOf('solve')).toContain('let e = orderBuf[batch.x + gid.x];');
    expect(bodyOf('publish')).toContain(
      `publishBuf[i * ${SOFT_PUBLISH_FLOATS_PER_NODE}u] = p.x;`,
    );
    // A vec2<u32> is 8 bytes at an 8-byte alignment, so the batch table is the one
    // buffer whose WGSL stride is not its element count.
    expect(SOFT_BINDINGS.find((b) => b.name === 'batchBuf')!.type).toBe('array<vec2<u32>>');
    expect(SOFT_BATCH_U32_PER_COLOR).toBe(2);
  });

  it('sizes the per-constraint and per-island buffers', () => {
    expect(SOFT_ENDS_U32_PER_CONSTRAINT).toBe(2);
    expect(SOFT_ORDER_U32_PER_CONSTRAINT).toBe(1);
    // Rest length and stiffness stay in two buffers in the mesh's own order rather
    // than interleaved into one, so each upload is a straight copy of the array
    // SoftMesh already built and the coloring is applied through orderBuf alone.
    expect(SOFT_EDGE_F32_PER_CONSTRAINT).toBe(2);
    expect(
      softBindingsForGroup(SOFT_GROUP_STATIC)
        .filter((b) => b.type === 'array<f32>')
        .map((b) => b.name),
    ).toEqual(['restBuf', 'stiffBuf']);
    // Three words per island, one each for the three arrays softCpu.ts keeps.
    expect(SOFT_SLEEP_WORDS_PER_ISLAND).toBe(3);
    const sleepBuffers = ['asleepBuf', 'quietBuf', 'islandSpeedBuf'];
    expect(
      softBindingsForGroup(SOFT_GROUP_STATE).filter((b) => sleepBuffers.includes(b.name)),
    ).toHaveLength(SOFT_SLEEP_WORDS_PER_ISLAND);
  });

  it('publishes the tight xyz the CPU tier publishes', () => {
    // Run the reference and read its published array back against the mesh's own
    // record. This is the one layout both tiers share by contract, and the reason
    // render/soft.ts never has to know which tier it is holding.
    const mesh = new SoftMesh({ count: 100, scene: 'rope' });
    const cpu = createCpuSoftSystem({ mesh, options: { iterations: 2 } });
    cpu.step();
    const out = new Float32Array(mesh.count * SOFT_PUBLISH_FLOATS_PER_NODE);
    expect(cpu.copyPublishedTo(out)).toBe(out.byteLength);
    for (let i = 0; i < mesh.count; i++) {
      for (let k = 0; k < SOFT_PUBLISH_FLOATS_PER_NODE; k++) {
        expect(out[i * 3 + k], `node ${i} axis ${k}`).toBe(
          mesh.data[i * SOFT_STRIDE + SOFT_OFFSET.position + k],
        );
      }
    }
    // 100 is not a multiple of 64, so this is also the tail the publish guard and
    // the last workgroup of the island-mapped kernels have to survive.
    expect(mesh.count % SOFT_WORKGROUP_SIZE).not.toBe(0);
  });
});

describe('softSolveDispatch', () => {
  it('puts the workgroup count on x and the color on z', () => {
    const batch: SoftBatch = { color: 5, base: 320, count: 129, workgroups: 3 };
    expect(softSolveDispatch(batch)).toEqual([3, 1, 5]);
    expect(softSolveDispatch({ color: 0, base: 0, count: 0, workgroups: 0 })).toEqual([0, 1, 0]);
  });

  it('tiles the colored order for every scene', () => {
    for (const scene of SOFT_SCENES) {
      const mesh = new SoftMesh({ count: 600, scene });
      const layout = buildSoftLayout(mesh, resolveSoftOptions({ iterations: 2 }));
      const { batches, order, colors } = layout.coloring;
      expect(batches, scene).toHaveLength(colors);
      let base = 0;
      for (let c = 0; c < batches.length; c++) {
        const batch = batches[c]!;
        expect(batch.color, `${scene} batch ${c}`).toBe(c);
        expect(batch.base, `${scene} batch ${c}`).toBe(base);
        expect(batch.count, `${scene} batch ${c}`).toBeGreaterThan(0);
        const [x, y, z] = softSolveDispatch(batch);
        expect(x, `${scene} batch ${c} x`).toBe(softWorkgroups(batch.count));
        expect(batch.workgroups, `${scene} batch ${c}`).toBe(x);
        expect(y, `${scene} batch ${c} y`).toBe(1);
        expect(z, `${scene} batch ${c} z`).toBe(c);
        base += batch.count;
      }
      // The batches tile `order` with no gap and no overlap, which is what makes one
      // GPU iteration -- `colors` dispatches -- walk exactly the constraints the
      // CPU's single solvePass(order, 0, order.length) call walks.
      expect(base, scene).toBe(order.length);
      expect(order.length, scene).toBe(mesh.constraints.count);
      expect(layout.plan.colors, scene).toBe(colors);
      expect(layout.plan.batchSizes, scene).toEqual(batches.map((b) => b.count));
    }
  });

  it('keeps both dimensions inside what a dispatch allows', () => {
    // WebGPU caps every dispatch dimension at 65535. MAX_COLORS is far under it, so
    // z always holds a color; x is the one a large mesh could overflow, and it is
    // bounded by the biggest single color rather than by the graph.
    expect(MAX_COLORS).toBe(32);
    expect(MAX_COLORS).toBeLessThanOrEqual(65535);
    for (const scene of SOFT_SCENES) {
      const mesh = new SoftMesh({ count: 600, scene });
      const coloring = buildSoftLayout(mesh, resolveSoftOptions({ iterations: 2 })).coloring;
      for (const batch of coloring.batches) {
        const [x, y, z] = softSolveDispatch(batch);
        expect(z, scene).toBeLessThan(MAX_COLORS);
        expect(x, scene).toBeLessThanOrEqual(65535);
        expect(y, scene).toBe(1);
      }
      expect(coloring.maxBatchSize, scene).toBe(
        Math.max(...coloring.batches.map((b) => b.count)),
      );
    }
  });
});

describe('pass semantics', () => {
  const bare = bareBody;

  /** Index of `needle`, failing the test with the needle if it is not there. */
  const at = (text: string, needle: string): number => {
    const i = text.indexOf(needle);
    expect(i, `missing: ${needle}`).toBeGreaterThan(-1);
    return i;
  };

  it('resets the step counters before the sleep guard can skip them', () => {
    const p = bare('predict');
    // Island 0 can be the one that is asleep. A reset placed after the guard would
    // then never run, and the HUD would report the previous step's numbers forever.
    expect(at(p, 'atomicStore(&statsBuf[STAT_ESCAPED], 0u);')).toBeLessThan(
      at(p, 'if (asleepBuf[islandOfWgBuf[wg]] != 0u) { return; }'),
    );
    expect(at(p, 'if (gid.x == 0u) {')).toBeLessThan(at(p, 'let wg = wid.x;'));
    expect([...p.matchAll(/atomicStore\(&statsBuf\[(\w+)\], 0u\);/g)].map((m) => m[1])).toEqual([
      'STAT_ESCAPED',
      'STAT_MAX_SPEED_SQ',
      'STAT_MAX_ERROR',
      'STAT_SLEEPING',
    ]);
    // Four stores, one owner, nowhere else: a second reset would be a second place
    // the two tiers can disagree about what "this step" means.
    expect(code.match(/atomicStore/g)).toHaveLength(4);
  });

  it('damps, then integrates, then predicts -- in that order', () => {
    const p = bare('predict');
    const damp = at(p, 'let damp = 1.0 - params.damping * dt;');
    const integrate = at(p, 'let v = (stateBuf[base + 1u].xyz + params.gravity * dt) * damp;');
    const write = at(p, 'predBuf[i] = vec4<f32>(pos + v * dt, 0.0);');
    expect(damp).toBeLessThan(integrate);
    expect(integrate).toBeLessThan(write);
    // predict owns the scratch and nothing else. Writing stateBuf here would make
    // the solve read positions the constraints of the previous iteration moved.
    expect(p).not.toMatch(/stateBuf\[[^\]]*\]\s*=(?!=)/);
  });

  it('predicts a pinned node onto itself, with no velocity case of its own', () => {
    const p = bare('predict');
    expect(at(p, 'if (!(prev.w > 0.0)) {')).toBeLessThan(at(p, 'let dt = params.dt;'));
    expect(p).toContain('predBuf[i] = vec4<f32>(pos, 0.0);');
    // The negated comparison is the NaN case: a node whose inverse mass arrived NaN
    // takes the pinned branch here and in softCpu.ts, which writes the same !(x > 0).
    expect(p).not.toContain('if (prev.w == 0.0)');
    expect(p.match(/predBuf\[i\] = vec4<f32>\([^,]+, 0\.0\);/g)).toHaveLength(2);
  });

  it('solves with plain stores, which is what makes the digest reproducible', () => {
    const s = bare('solve');
    // The coloring guarantees no two invocations of one dispatch share a node, so a
    // plain store is correct and an atomic would only add a scheduling dependency.
    expect(s).not.toMatch(/atomic/);
    expect(s).toContain('let pa = predBuf[a];');
    expect(s).toContain('let pb = predBuf[b];');
    expect(s).toContain('predBuf[a] = vec4<f32>(pa.xyz + d * (s * wa), 0.0);');
    expect(s).toContain('predBuf[b] = vec4<f32>(pb.xyz - d * (s * wb), 0.0);');
    // Read and written in place, so iteration k + 1 has to be a later dispatch and
    // cannot be a wider one: the ordering between dispatches is the only barrier
    // this pipeline has.
    expect(at(s, 'let pa = predBuf[a];')).toBeLessThan(at(s, 'predBuf[a] = vec4<f32>'));
    expect(s).not.toMatch(/stateBuf\[[^\]]*\]\s*=(?!=)/);
  });

  it('writes the solver arithmetic the way the CPU reference writes it', () => {
    const s = bare('solve');
    expect(s).toContain('if (!(d2 > 0.0)) { return; }');
    expect(s).toContain('if (!(wSum > 0.0)) { return; }');
    // Both divides are written as reciprocals and multiplied afterwards, matching
    // softCpu.ts term for term; two divisions inside the correction would round
    // differently and the parity check would need a tolerance wide enough to hide a
    // real bug.
    expect(s).toContain('let invDist = 1.0 / dist;');
    expect(s).toContain('let invW = 1.0 / wSum;');
    expect(s).toContain('let stiffEff = stiffBuf[e] * params.stiffness;');
    expect(s).toContain('let s = stiffEff * ((dist - restBuf[e]) * invDist) * invW;');
    expect(s).toContain('let dist = sqrt(d2);');
    expect(s.match(/\//g)).toHaveLength(2);
  });

  it('tests the island before it loads the second endpoint', () => {
    const s = bare('solve');
    const guard = at(s, 'if (asleepBuf[islandOfNodeBuf[a]] != 0u) { return; }');
    expect(at(s, 'let a = endsBuf[e * 2u];')).toBeLessThan(guard);
    expect(guard).toBeLessThan(at(s, 'let b = endsBuf[e * 2u + 1u];'));
    expect(guard).toBeLessThan(at(s, 'let pa = predBuf[a];'));
    // Both endpoints are in one island by construction, so testing one tests the
    // edge -- and per-node is the only option here, because a solve workgroup's 64
    // lanes are 64 unrelated constraints.
    expect(s.match(/asleepBuf/g)).toHaveLength(1);
  });

  it('derives velocity from the position change and clamps speed last', () => {
    const f = bare('finalize');
    const derive = at(f, 'var v = (p - prev.xyz) * params.invDt;');
    const clamp = at(f, 'if (s2 > maxSpeed2) {');
    const write = at(f, 'stateBuf[base] = vec4<f32>(p, prev.w);');
    expect(derive).toBeLessThan(clamp);
    expect(clamp).toBeLessThan(write);
    expect(f).toContain('let maxSpeed2 = params.maxSpeed * params.maxSpeed;');
    expect(f).toContain('v = v * (params.maxSpeed / sqrt(s2));');
    // The clamp touches velocity only, and clamps the reported maximum down with it,
    // so the stat cannot claim a speed the state does not have.
    expect(at(f, 'v = v * (params.maxSpeed / sqrt(s2));')).toBeLessThan(at(f, 's2 = maxSpeed2;'));
    expect(f).not.toContain('p = p * (params.maxSpeed');
  });

  it('clamps after reflecting, so escaped is zero by structure under reflect', () => {
    const f = bare('finalize');
    for (const axis of ['x', 'y', 'z']) {
      expect(f.match(new RegExp(`v\\.${axis} = -v\\.${axis} \\* rest;`, 'g'))).toHaveLength(2);
      expect(f, axis).toContain(`p.${axis} = lo.${axis} + (lo.${axis} - p.${axis}) * rest;`);
      expect(f, axis).toContain(`p.${axis} = hi.${axis} - (p.${axis} - hi.${axis}) * rest;`);
    }
    expect(at(f, 'p.x = lo.x + (lo.x - p.x) * rest;')).toBeLessThan(at(f, 'p = clamp(p, lo, hi);'));
    // The walls are inset by the node radius, so "inside" means the whole sphere is.
    expect(f).toContain('let lo = params.boundsMin + vec3<f32>(radius);');
    expect(f).toContain('let hi = params.boundsMax - vec3<f32>(radius);');
    expect(at(f, 'let lo = params.boundsMin + vec3<f32>(radius);')).toBeGreaterThan(
      at(f, '} else {'),
    );
  });

  it('counts an escape only in the unbounded mode', () => {
    const f = bare('finalize');
    const none = at(f, 'if (boundsMode() == BOUNDS_NONE) {');
    const escaped = at(f, 'if (!inside) { atomicAdd(&statsBuf[STAT_ESCAPED], 1u); }');
    expect(none).toBeLessThan(escaped);
    expect(escaped).toBeLessThan(at(f, '} else {'));
    expect(f.match(/STAT_ESCAPED/g)).toHaveLength(1);
    expect(f).toContain('let inside = all(p >= params.boundsMin) && all(p <= params.boundsMax);');
    // The counting mode is the explicit one, so an encoding the flag bits cannot
    // express lands in the bounded branch instead of leaving the box uncounted.
    expect(code).not.toContain('BOUNDS_WRAP');
    expect([...code.matchAll(/atomicAdd\(&statsBuf\[STAT_ESCAPED\]/g)]).toHaveLength(1);
  });

  it('drops a whole sleeping workgroup, not half its lanes', () => {
    const load = 'let i = nodeOrderBuf[wg * WG + lid.x];';
    const predict = bare('predict');
    expect(at(predict, 'if (asleepBuf[islandOfWgBuf[wg]] != 0u) { return; }')).toBeLessThan(
      at(predict, load),
    );
    const finalize = bare('finalize');
    expect(finalize).toContain('let island = islandOfWgBuf[wg];');
    expect(at(finalize, 'if (asleepBuf[island] != 0u) { return; }')).toBeLessThan(
      at(finalize, load),
    );
    // One island map lookup per workgroup rather than one per node, which is the
    // whole reason softIslands.ts pads every island to a multiple of 64.
    expect(predict).not.toContain('islandOfNodeBuf');
    expect(finalize).not.toContain('islandOfNodeBuf');
    expect(at(finalize, 'atomicMax(&islandSpeedBuf[island], bits);')).toBeGreaterThan(
      at(finalize, 'atomicMax(&statsBuf[STAT_MAX_SPEED_SQ], bits);'),
    );
  });

  it('measures the final positions, sleeping islands included', () => {
    const m = bare('measure');
    // No sleep check and no predictions: a sleeping island contributes its frozen
    // error, which is the honest number, and reading stateBuf is what makes that so.
    expect(m).not.toContain('asleepBuf');
    expect(m).not.toContain('predBuf');
    expect(m).toContain('let pa = stateBuf[endsBuf[e * 2u] * 2u].xyz;');
    expect(m).toContain('let pb = stateBuf[endsBuf[e * 2u + 1u] * 2u].xyz;');
    expect(m).toContain('let err = abs(sqrt(d2) - r) / r;');
    expect(m).toContain('atomicMax(&statsBuf[STAT_MAX_ERROR], bitcast<u32>(err));');
    expect(m).toContain('if (!(d2 > 0.0)) { return; }');
    expect(m.match(/atomic/g)).toHaveLength(1);
    // Relative error, so a max over it is scale-free and the two tiers agree on it
    // exactly rather than within a tolerance.
    expect(m).not.toContain('abs(sqrt(d2) - r);');
  });

  it('reads and clears the island speed in one atomic, before the flag check', () => {
    const s = bare('sleep_update');
    const exchange = at(s, 'let sq = bitcast<f32>(atomicExchange(&islandSpeedBuf[k], 0u));');
    expect(exchange).toBeLessThan(at(s, 'if (!hasFlag(FLAG_SLEEP)) { return; }'));
    // Clearing here rather than at the top of the step is what lets the buffer start
    // zero-initialised and still mean "every island is awake".
    expect(s).toContain('if (sq < params.sleepThresholdSq) {');
    expect(s).toContain('let q = quietBuf[k] + 1u;');
    expect(s).toContain('if (q >= params.sleepAfter) {');
    expect(s).toContain('asleepBuf[k] = 1u;');
    expect(s).toContain('quietBuf[k] = 0u;');
    expect(s.match(/atomicAdd\(&statsBuf\[STAT_SLEEPING\], 1u\);/g)).toHaveLength(2);
    // One invocation per island, so exactly one writer per word: the two sleep arrays
    // are plain stores and an atomic on either would be a wasted lock.
    expect(s).not.toContain('&asleepBuf');
    expect(s).not.toContain('&quietBuf');
    expect(s.match(/atomic/g)).toHaveLength(3);
  });

  it('publishes every node, including the sleeping ones', () => {
    const p = bare('publish');
    expect(p).not.toContain('asleepBuf');
    expect(p).not.toContain('predBuf');
    expect(p).not.toContain('nodeOrderBuf');
    expect(p).toContain('if (i >= params.count) { return; }');
    expect(p).toContain('let p = stateBuf[i * 2u].xyz;');
    expect(p).toContain('publishBuf[i * 3u] = p.x;');
    expect(p).toContain('publishBuf[i * 3u + 1u] = p.y;');
    expect(p).toContain('publishBuf[i * 3u + 2u] = p.z;');
    // Over 0..count rather than over the padded node order: a sleeping island's
    // positions are frozen, not absent, and every node is drawn every frame.
    expect(p.match(/publishBuf/g)).toHaveLength(3);
  });
});

describe('cross-tier pass order', () => {
  const cpuSrc = readFileSync(resolve(import.meta.dirname, '../src/gpu/softCpu.ts'), 'utf8');

  /** Both comment kinds, because softCpu.ts documents itself with block comments. */
  const bareTs = (text: string): string =>
    text.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');

  /** CPU pass name to kernel name. `sleep` is the only one not spelled the same. */
  const CPU_TO_KERNEL: Readonly<Record<string, SoftKernel>> = {
    predict: 'predict',
    solve: 'solve',
    finalize: 'finalize',
    measure: 'measure',
    sleep: 'sleep_update',
    publish: 'publish',
  };

  function stepBody(): string {
    const start = cpuSrc.indexOf('step(dt: number = this.options.fixedDt): void');
    expect(start, 'CpuSoftSystem.step is missing').toBeGreaterThan(-1);
    return tsBodyOf(cpuSrc, start);
  }

  it('has one private pass per kernel, declared in kernel order', () => {
    // Both tiers are generated from the same constants, so a formula that drifts
    // surfaces as a parity failure in e2e/soft_gpu.spec.ts. A *pass order* that
    // drifts surfaces as a simulation that is subtly wrong in a way no tolerance
    // names, which is why this file reads softCpu.ts as text.
    const passes = [...cpuSrc.matchAll(/private (\w+)Pass\(/g)].map((m) => m[1]);
    expect(passes).toEqual(Object.keys(CPU_TO_KERNEL));
    expect(Object.values(CPU_TO_KERNEL)).toEqual([...SOFT_KERNELS]);
  });

  it('calls them in that order from step()', () => {
    const calls = [...stepBody().matchAll(/this\.(\w+)Pass\(/g)].map((m) => m[1]);
    expect(calls).toEqual(Object.keys(CPU_TO_KERNEL));
    // Dispatches inside one compute pass are ordered by the WebGPU spec, with the
    // writes of one visible to the next. That ordering is the only barrier this
    // pipeline has, so it has to be the same ordering the reference walks in.
    expect(calls.map((c) => CPU_TO_KERNEL[c])).toEqual([...SOFT_KERNELS]);
  });

  it('runs solve once per iteration, which is the count the plan reports', () => {
    const body = stepBody();
    const head = body.indexOf('for (let it = 0; it < this.options.iterations; it++) {');
    expect(head, 'the iteration loop is missing from step()').toBeGreaterThan(-1);
    const loop = bareTs(tsBodyOf(body, head));
    // Exactly one pass call in the loop, over the whole colored order: one walk is
    // `colors` batches, so the CPU's loop body and the GPU's z dimension are the
    // same decomposition of the same permutation.
    expect([...loop.matchAll(/this\.(\w+)Pass\(/g)].map((m) => m[1])).toEqual(['solve']);
    expect(loop).toContain('this.solvePass(order, 0, order.length);');
    const iterations = 5;
    const mesh = new SoftMesh({ count: 400, scene: 'cloth' });
    const coloring = buildSoftLayout(mesh, resolveSoftOptions({ iterations })).coloring;
    expect(coloring.order.length).toBe(coloring.batches.reduce((n, b) => n + b.count, 0));
    expect(coloring.dispatchesPerIteration).toBe(coloring.colors);
  });

  it('adds up to the dispatch count the plan reports, for every scene', () => {
    const iterations = 2;
    for (const scene of SOFT_SCENES) {
      const mesh = new SoftMesh({ count: 600, scene });
      const plan = buildSoftLayout(mesh, resolveSoftOptions({ iterations })).plan;
      // What each rule in SOFT_KERNEL_DISPATCH costs in dispatches. `batch` is the
      // only one that scales, and the HUD subtracts SOFT_FIXED_DISPATCHES to show
      // the part the iteration slider does not move.
      const perRule: Readonly<Record<SoftKernelDispatch, number>> = {
        nodeWorkgroups: 1,
        batch: iterations * plan.colors,
        constraints: 1,
        islands: 1,
        count: 1,
      };
      const total = SOFT_KERNELS.reduce((n, k) => n + perRule[SOFT_KERNEL_DISPATCH[k]], 0);
      expect(total, scene).toBe(plan.dispatchesPerStep);
      const scaled = SOFT_FIXED_DISPATCHES + iterations * plan.colors;
      expect(plan.dispatchesPerStep, scene).toBe(scaled);
      expect(plan.batchSizes.reduce((a, b) => a + b, 0), scene).toBe(plan.constraints);
    }
  });

  it('checks sleep in the same passes on both tiers', () => {
    // A pass that skips sleeping islands on one tier and not the other is not a
    // performance difference: the sleeping nodes keep different positions, and the
    // divergence shows up in the digest rather than in a frame time.
    const sleepingCpu = Object.entries(CPU_TO_KERNEL)
      .filter(([pass]) => {
        const start = cpuSrc.indexOf(`private ${pass}Pass(`);
        expect(start, `${pass}Pass is missing`).toBeGreaterThan(-1);
        return /\basleep\b/.test(bareTs(tsBodyOf(cpuSrc, start)));
      })
      .map(([, kernel]) => kernel);
    const sleepingGpu = SOFT_KERNELS.filter((k) => bareBody(k).includes('asleepBuf'));
    expect(sleepingCpu).toEqual([...sleepingGpu]);
    expect([...sleepingGpu].sort()).toEqual(['finalize', 'predict', 'sleep_update', 'solve']);
    // measure and publish deliberately do not, and the comments in both shaders say
    // why: a frozen error is the honest number and every node is drawn every frame.
    expect(bareBody('measure')).not.toContain('asleepBuf');
    expect(bareBody('publish')).not.toContain('asleepBuf');
  });
});

describe('softShaderSource', () => {
  it('is memoised, so the module is created once per process', () => {
    expect(softShaderSource()).toBe(src);
  });

  it('interpolates nothing that looks like an unresolved template', () => {
    expect(src).not.toContain('${');
    expect(code).not.toContain('undefined');
    expect(code).not.toContain('NaN');
    expect(code).not.toContain('[object Object]');
  });

  it('produces a module a WGSL parser accepts', () => {
    // Structural smoke test: balanced braces and no stray TypeScript in the output.
    // The real compile happens against a device in e2e/soft_gpu.spec.ts.
    let depth = 0;
    for (const ch of code) {
      if (ch === '{') depth++;
      else if (ch === '}') depth--;
      expect(depth, 'brace depth went negative').toBeGreaterThanOrEqual(0);
    }
    expect(depth).toBe(0);
    expect(code).not.toMatch(/\bimport\b|\bexport\b|=>/);
    // Ends where a module should, with no trailing whitespace anywhere in it: this
    // text is read verbatim in a device profile when a compile error points at it.
    expect(src.endsWith('}\n')).toBe(true);
    expect(src).not.toMatch(/[ \t]+\n/);
  });

  it('says where it came from, on the first line', () => {
    // The text is checked in nowhere and is read by whoever debugs a compile error
    // in a browser profile, so the generated module has to point back at its source.
    expect(src.split('\n')[0]).toBe(
      '// Generated by src/gpu/softWgsl.ts -- edit the generator, not this text.',
    );
  });

  it('declares three helpers before the six kernels, and nothing else', () => {
    expect([...src.matchAll(/^fn (\w+)\(/gm)].map((m) => m[1])).toEqual([
      'sumSq',
      'hasFlag',
      'boundsMode',
      ...SOFT_KERNELS,
    ]);
    // Helpers are plain fns and kernels are the only @compute entry points, so a
    // seventh entry point would be a dispatch softGpu.ts does not know it owes.
    expect(src.match(/^@compute/gm)).toHaveLength(SOFT_KERNELS.length);
    // One caller each: sleep_update is the only pass the sleep flag can turn off,
    // and finalize is the only pass whose shape the bounds mode changes.
    expect(bareBody('sleep_update')).toContain('if (!hasFlag(FLAG_SLEEP)) { return; }');
    expect(bareBody('finalize')).toContain('if (boundsMode() == BOUNDS_NONE) {');
    expect([...code.matchAll(/\bhasFlag\(/g)]).toHaveLength(2);
    expect([...code.matchAll(/\bboundsMode\(\)/g)]).toHaveLength(2);
    expect(bareBody('predict')).not.toMatch(/\bhasFlag\(|\bboundsMode\(/);
    expect(bareBody('solve')).not.toMatch(/\bhasFlag\(|\bboundsMode\(/);
  });
});

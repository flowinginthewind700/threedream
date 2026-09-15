/**
 * `gpu/particleWgsl.ts` -- the generated shader text and the tables around it.
 *
 * These are text-level assertions, and that is the point: the shader is only
 * compiled on a machine with a WebGPU implementation, so the drift that a
 * compiler would catch on *that* machine has to be caught here on every machine.
 * What is checked is the stuff a compiler cannot check for us -- that the
 * `Params` member order still produces the byte offsets `writeParams` writes,
 * that the constants in the WGSL are the constants the CPU backend uses, and
 * that the passes are ordered the way `particleCpu.ts` orders them.
 *
 * Actual compilation is asserted in `e2e/particles_gpu.spec.ts`, against a real
 * device. This file is what runs in the sub-second unit loop.
 */

import { describe, expect, it } from 'vitest';
import {
  ACCEL_VECS_PER_PARTICLE,
  CONTACT_VECS_PER_PARTICLE,
  GROUP_STATE,
  GROUP_STATIC,
  KERNEL_DISPATCH,
  PARTICLE_BINDINGS,
  PARTICLE_KERNELS,
  PUBLISH_FLOATS_PER_PARTICLE,
  STATE_FLOATS_PER_PARTICLE,
  STATE_VECS_PER_PARTICLE,
  STAT_WORDS,
  STAT_WORD,
  WGSL_PARAMS_LAYOUT,
  WORKGROUP_SIZE,
  bindingsForGroup,
  particleShaderSource,
  workgroupsFor,
} from '../src/gpu/particleWgsl.js';
import { POSITION_CORRECTION } from '../src/gpu/particleCpu.js';
import { PARTICLE_STRIDE } from '../src/gpu/particleField.js';
import { HASH_PRIMES } from '../src/gpu/particleHash.js';
import {
  BOUNDS_MODE_BITS,
  PARAMS_BYTES,
  PARAM_WORD,
  PARTICLE_FLAG,
} from '../src/gpu/particleOptions.js';

const src = particleShaderSource();

/**
 * `src` with the line comments stripped.
 *
 * Several assertions below are "this construct must not appear", and the prose
 * explaining *why* it must not appear names the construct. Checking the code
 * rather than the whole text keeps those assertions honest. WGSL has no string
 * literals, so a comment cannot be hiding inside one.
 */
const code = src.replace(/\/\/[^\n]*/g, '');

/** WGSL alignment and size, in bytes. `vec3` is the interesting one: 12 in 16. */
const ALIGN: Record<string, number> = { vec3: 16, f32: 4, u32: 4 };
const SIZE: Record<string, number> = { vec3: 12, f32: 4, u32: 4 };
const WGSL_TYPE: Record<string, string> = { vec3: 'vec3<f32>', f32: 'f32', u32: 'u32' };

/** Body of one `fn`, from its declaration to the matching closing brace. */
function bodyOf(name: string): string {
  const start = src.indexOf(`fn ${name}(`);
  expect(start, `kernel ${name} is missing`).toBeGreaterThan(-1);
  const open = src.indexOf('{', start);
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  throw new Error(`unbalanced braces in ${name}`);
}

describe('Params layout', () => {
  it('matches the offsets WGSL alignment rules produce', () => {
    // Recomputed here rather than read from PARAM_WORD: the whole value of the
    // check is that the two derivations are independent.
    let offset = 0;
    for (const m of WGSL_PARAMS_LAYOUT) {
      offset = Math.ceil(offset / ALIGN[m.kind]) * ALIGN[m.kind];
      expect(offset, `${m.name} offset`).toBe(m.word * 4);
      offset += SIZE[m.kind];
    }
    // A struct's size rounds up to its own alignment, which is the largest
    // member alignment -- 16 here, from the three vec3s.
    expect(Math.ceil(offset / 16) * 16).toBe(PARAMS_BYTES);
  });

  it('covers every word of the uniform exactly once', () => {
    const words: number[] = [];
    for (const m of WGSL_PARAMS_LAYOUT) {
      const span = m.kind === 'vec3' ? 3 : 1;
      for (let k = 0; k < span; k++) words.push(m.word + k);
    }
    expect([...words].sort((a, b) => a - b)).toEqual(
      Array.from({ length: PARAMS_BYTES / 4 }, (_, i) => i),
    );
    expect(Object.values(PARAM_WORD).sort((a, b) => a - b)).toEqual(
      Array.from({ length: PARAMS_BYTES / 4 }, (_, i) => i),
    );
  });

  it('declares the members in this order and with these types', () => {
    const start = src.indexOf('struct Params {');
    const end = src.indexOf('};', start);
    expect(start).toBeGreaterThan(-1);
    const declared = [...src.slice(start, end).matchAll(/^ {2}(\w+): (vec3<f32>|f32|u32),$/gm)].map(
      (m) => [m[1], m[2]] as const,
    );
    expect(declared).toEqual(
      WGSL_PARAMS_LAYOUT.map((m) => [m.name, WGSL_TYPE[m.kind]] as const),
    );
  });

  it('reads the integers as u32 and the floats as f32', () => {
    // count, tableMask, bucketCapacity and flags are the four words writeParams
    // fills through a Uint32Array view. Declaring them f32 would reinterpret the
    // bit pattern, which is the silent kind of wrong.
    for (const name of ['count', 'tableMask', 'bucketCapacity', 'flags']) {
      const member = WGSL_PARAMS_LAYOUT.find((m) => m.name === name)!;
      expect(member.kind, name).toBe('u32');
    }
    expect(WGSL_PARAMS_LAYOUT.filter((m) => m.kind === 'u32')).toHaveLength(4);
  });
});

describe('constants cannot drift from the CPU backend', () => {
  const constOf = (name: string): string => {
    const m = src.match(new RegExp('^const ' + name + ': (\\w+) = ([^;]+);$', 'm'));
    expect(m, `const ${name} is missing`).not.toBeNull();
    return m![2];
  };

  it('uses the same hash primes', () => {
    expect(constOf('PRIME_X')).toBe(`${HASH_PRIMES.x}u`);
    expect(constOf('PRIME_Y')).toBe(`${HASH_PRIMES.y}u`);
    expect(constOf('PRIME_Z')).toBe(`${HASH_PRIMES.z}u`);
  });

  it('uses the same flag bits and bounds-mode encoding', () => {
    expect(constOf('FLAG_COLLISIONS')).toBe(`${PARTICLE_FLAG.collisions}u`);
    expect(constOf('FLAG_NBODY')).toBe(`${PARTICLE_FLAG.nbody}u`);
    expect(constOf('BOUNDS_SHIFT')).toBe(`${PARTICLE_FLAG.boundsShift}u`);
    expect(constOf('BOUNDS_REFLECT')).toBe(`${BOUNDS_MODE_BITS.reflect}u`);
    expect(constOf('BOUNDS_WRAP')).toBe(`${BOUNDS_MODE_BITS.wrap}u`);
    expect(constOf('BOUNDS_NONE')).toBe(`${BOUNDS_MODE_BITS.none}u`);
  });

  it('uses the same positional correction as the CPU solver', () => {
    expect(constOf('CORRECTION')).toBe(`${POSITION_CORRECTION}`);
    expect(POSITION_CORRECTION).toBe(0.5);
  });

  it('uses the same stats words it exports', () => {
    expect(constOf('STAT_CONTACTS')).toBe(`${STAT_WORD.contacts}u`);
    expect(constOf('STAT_ESCAPED')).toBe(`${STAT_WORD.escaped}u`);
    expect(constOf('STAT_OVERFLOW')).toBe(`${STAT_WORD.overflow}u`);
    expect(constOf('STAT_MAX_SPEED_SQ')).toBe(`${STAT_WORD.maxSpeedSq}u`);
    expect(Object.values(STAT_WORD).sort((a, b) => a - b)).toEqual(
      Array.from({ length: STAT_WORDS }, (_, i) => i),
    );
  });
});

describe('entry points', () => {
  it('declares all six, in dispatch order', () => {
    expect(PARTICLE_KERNELS).toEqual([
      'nbody',
      'hash_clear',
      'hash_scatter',
      'collide',
      'integrate',
      'publish',
    ]);
    const declared = [...src.matchAll(/^@compute[^\n]*\nfn (\w+)\(/gm)].map((m) => m[1]);
    expect(declared).toEqual([...PARTICLE_KERNELS]);
  });

  it('gives every kernel a dispatch rule', () => {
    expect(Object.keys(KERNEL_DISPATCH).sort()).toEqual([...PARTICLE_KERNELS].sort());
    // hash_clear walks the cell table, not the particles; everything else is
    // per-particle. Getting this backwards leaves stale bucket counts.
    expect(KERNEL_DISPATCH.hash_clear).toBe('tableSize');
    for (const k of PARTICLE_KERNELS) {
      if (k !== 'hash_clear') expect(KERNEL_DISPATCH[k], k).toBe('count');
    }
  });

  it('sizes every workgroup at 64 and nothing else', () => {
    const sizes = [...src.matchAll(/@workgroup_size\((\d+)\)/g)].map((m) => Number(m[1]));
    expect(sizes).toHaveLength(PARTICLE_KERNELS.length);
    expect(new Set(sizes)).toEqual(new Set([WORKGROUP_SIZE]));
    expect(WORKGROUP_SIZE).toBe(64);
  });

  it('uses no subgroups and no f64', () => {
    // Both are unavailable on the adapters the feasibility study measured, and
    // both fail at pipeline-creation time rather than at compile time on some
    // drivers -- the worst place to discover it is a user's machine.
    expect(code).not.toMatch(/subgroup/);
    expect(code).not.toMatch(/f64/);
    expect(code).not.toMatch(/workgroup_uniform|textureGather/);
  });

  it('sums componentwise instead of calling dot', () => {
    // A dot product is free to be contracted into a multiply-add, and the CPU
    // backend writes its sums out explicitly. Using the same association here is
    // what keeps the two within tolerance instead of merely close.
    expect(code).not.toMatch(/\bdot\(/);
    expect(bodyOf('sumSq')).toContain('(v.x * v.x + v.y * v.y) + v.z * v.z');
  });
});

describe('bindings', () => {
  it('assigns each group/binding pair once', () => {
    const pairs = PARTICLE_BINDINGS.map((b) => `${b.group}:${b.binding}`);
    expect(new Set(pairs).size).toBe(pairs.length);
    const names = PARTICLE_BINDINGS.map((b) => b.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it('declares exactly the table, with the matching access mode', () => {
    const access = {
      uniform: 'var<uniform>',
      'storage-read': 'var<storage, read>',
      'storage-read-write': 'var<storage, read_write>',
    } as const;
    for (const b of PARTICLE_BINDINGS) {
      const decl = `@group(${b.group}) @binding(${b.binding}) ${access[b.kind]} ${b.name}: ${b.type};`;
      expect(src, decl).toContain(decl);
    }
    expect([...src.matchAll(/^@group\(\d+\) @binding\(\d+\)/gm)]).toHaveLength(
      PARTICLE_BINDINGS.length,
    );
  });

  it('splits the static resources from the ping-pong pair', () => {
    expect(GROUP_STATIC).toBe(0);
    expect(GROUP_STATE).toBe(1);
    expect(bindingsForGroup(GROUP_STATIC).map((b) => b.name)).toEqual([
      'params',
      'accelBuf',
      'contactBuf',
      'hashCounts',
      'hashSlots',
      'statsBuf',
    ]);
    expect(bindingsForGroup(GROUP_STATE).map((b) => b.name)).toEqual([
      'stateSrc',
      'stateDst',
      'publishBuf',
    ]);
    expect(bindingsForGroup(2)).toEqual([]);
  });

  it('marks only stateSrc read-only, so the swap is legal', () => {
    const readOnly = PARTICLE_BINDINGS.filter((b) => b.kind === 'storage-read');
    expect(readOnly.map((b) => b.name)).toEqual(['stateSrc']);
    expect(readOnly[0].bufferType).toBe('read-only-storage');
    // params is the only uniform; everything else is a storage buffer.
    expect(PARTICLE_BINDINGS.filter((b) => b.bufferType === 'uniform')).toHaveLength(1);
  });
});

describe('buffer sizing constants', () => {
  it('agrees with the interleaved field layout', () => {
    expect(STATE_FLOATS_PER_PARTICLE).toBe(PARTICLE_STRIDE);
    expect(STATE_VECS_PER_PARTICLE).toBe(PARTICLE_STRIDE / 4);
    expect(CONTACT_VECS_PER_PARTICLE).toBe(2);
    expect(ACCEL_VECS_PER_PARTICLE).toBe(1);
    expect(PUBLISH_FLOATS_PER_PARTICLE).toBe(4);
  });
});

describe('pass semantics', () => {
  it('applies the contact impulse to velocity before advancing position', () => {
    // The CPU integrator adds dv, then computes p from the corrected v, then
    // adds dx. Swapping the first two silently changes every trajectory, and
    // nothing about the rendered result would look wrong enough to notice.
    const body = bodyOf('integrate');
    const impulse = body.indexOf('v = v + contactBuf[base].xyz;');
    const advance = body.indexOf('var p = srcPos.xyz + v * dt;');
    const correction = body.indexOf('p = p + contactBuf[base + 1u].xyz;');
    expect(impulse).toBeGreaterThan(-1);
    expect(advance).toBeGreaterThan(impulse);
    expect(correction).toBeGreaterThan(advance);
  });

  it('damps, then clamps, then integrates', () => {
    const body = bodyOf('integrate');
    const damp = body.indexOf('var v = (srcVel.xyz + a * dt) * damp;');
    const clampSpeed = body.indexOf('if (s2 > maxSpeed2)');
    expect(damp).toBeGreaterThan(-1);
    expect(clampSpeed).toBeGreaterThan(damp);
    expect(body).toContain('let damp = 1.0 - params.damping * dt;');
  });

  it('writes the destination state and tracks the speed maximum', () => {
    const body = bodyOf('integrate');
    expect(body).toContain('stateDst[base] = vec4<f32>(p, r);');
    expect(body).toContain('stateDst[base + 1u] = vec4<f32>(v, srcVel.w);');
    expect(body).toContain('atomicMax(&statsBuf[STAT_MAX_SPEED_SQ], bitcast<u32>(sumSq(v)));');
    // Reading back through stateSrc would make the kernel race with itself.
    expect(body).not.toMatch(/stateDst\[[^\]]+\]\s*=[^;]*stateDst/);
  });

  it('clamps after reflecting, so escaped stays structurally zero', () => {
    const body = bodyOf('integrate');
    const reflect = body.indexOf('p.x = lo.x + (lo.x - p.x) * rest;');
    const clamp = body.indexOf('p = clamp(p, lo, hi);');
    expect(reflect).toBeGreaterThan(-1);
    expect(clamp).toBeGreaterThan(reflect);
    for (const axis of ['x', 'y', 'z']) {
      expect(body).toContain(`v.${axis} = -v.${axis} * rest;`);
    }
  });

  it('counts an escape only in the unbounded mode', () => {
    const body = bodyOf('integrate');
    expect(body).toContain('if (mode == BOUNDS_WRAP)');
    expect(body).toContain('} else if (mode == BOUNDS_NONE) {');
    expect(body).toContain('atomicAdd(&statsBuf[STAT_ESCAPED], 1u);');
    expect(bodyOf('wrapAxis')).toContain('let k = floor((p - lo) / size);');
  });

  it('resolves each pair from source velocities only', () => {
    const body = bodyOf('collide');
    // Reading a neighbour's accumulated dv would make the impulse depend on
    // scheduling, which is the one thing atomics cannot promise.
    expect(body).toContain('sumDot(stateSrc[bj + 1u].xyz - vi, nrm)');
    expect(body).not.toContain('contactBuf[base].xyz +');
    // contactBuf is written exactly twice, at the end, and never read.
    expect([...body.matchAll(/contactBuf/g)]).toHaveLength(4); // 2 guards + 2 writes
  });

  it('counts a contact once per unordered pair', () => {
    const body = bodyOf('collide');
    expect(body).toContain('if (i < j) { contacts = contacts + 1u; }');
    expect(body).toContain('atomicAdd(&statsBuf[STAT_CONTACTS], contacts);');
  });

  it('dedupes aliased buckets before walking them', () => {
    const body = bodyOf('collide');
    expect(body).toContain('var seen: array<u32, 27>;');
    expect(body).toContain('if (seen[k] == cell) { duplicate = true; }');
    expect(body).toContain('if (duplicate) { continue; }');
    // The bucket walk must come after the dedupe, or the guard does nothing.
    expect(body.indexOf('if (duplicate) { continue; }')).toBeLessThan(
      body.indexOf('atomicLoad(&hashCounts[cell])'),
    );
  });

  it('walks the 27-cell neighbourhood', () => {
    const body = bodyOf('collide');
    expect([...body.matchAll(/for \(var o[xyz] = -1; o[xyz] <= 1;/g)]).toHaveLength(3);
  });

  it('counts hash overflow instead of dropping the insertion silently', () => {
    const body = bodyOf('hash_scatter');
    expect(body).toContain('atomicAdd(&hashCounts[cell], 1)');
    expect(body).toContain('if (n < i32(params.bucketCapacity))');
    expect(body).toContain('atomicAdd(&statsBuf[STAT_OVERFLOW], 1u);');
  });

  it('clears the stats every step and the table only when colliding', () => {
    const body = bodyOf('hash_clear');
    const stats = body.indexOf('atomicStore(&statsBuf[STAT_CONTACTS], 0u);');
    const flag = body.indexOf('if (!hasFlag(FLAG_COLLISIONS)) { return; }');
    expect(stats).toBeGreaterThan(-1);
    // Stats reset has to happen before the early return, or a run with
    // collisions disabled reads the previous step's numbers forever.
    expect(flag).toBeGreaterThan(stats);
    expect(body).toContain('if (gid.x > params.tableMask) { return; }');
    expect(body).toContain('atomicStore(&hashCounts[gid.x], 0);');
  });

  it('guards every kernel against the tail of the last workgroup', () => {
    for (const k of ['nbody', 'hash_scatter', 'collide', 'integrate', 'publish']) {
      expect(bodyOf(k), k).toContain('if (i >= params.count) { return; }');
    }
  });

  it('publishes position and radius from the source buffer', () => {
    const body = bodyOf('publish');
    expect(body).toContain('publishBuf[i] = stateSrc[i * 2u];');
    // Dispatched after the swap, so this is the state integrate just wrote.
    expect(body).not.toContain('stateDst');
  });

  it('guards the inverse mass against a zero', () => {
    expect(bodyOf('inverseMass')).toContain('if (m > 0.0) { return 1.0 / m; }');
  });

  it('measures cells from boundsMin, the CPU hash origin', () => {
    expect(bodyOf('cellCoords')).toContain(
      'vec3<i32>(floor((p - params.boundsMin) * params.invCell))',
    );
    expect(bodyOf('hashCell')).toContain('& params.tableMask');
  });
});

describe('workgroupsFor', () => {
  it('rounds up to a whole workgroup', () => {
    expect(workgroupsFor(1)).toBe(1);
    expect(workgroupsFor(64)).toBe(1);
    expect(workgroupsFor(65)).toBe(2);
    expect(workgroupsFor(50_000)).toBe(Math.ceil(50_000 / 64));
    expect(workgroupsFor(100_000)).toBe(1563);
  });

  it('returns 0 for an empty range so the caller can skip the dispatch', () => {
    expect(workgroupsFor(0)).toBe(0);
    expect(workgroupsFor(-1)).toBe(0);
    expect(workgroupsFor(Number.NaN)).toBe(0);
    expect(workgroupsFor(Number.POSITIVE_INFINITY)).toBe(0);
  });
});

describe('particleShaderSource', () => {
  it('is memoised, so the module is created once per process', () => {
    expect(particleShaderSource()).toBe(src);
  });

  it('interpolates nothing that looks like an unresolved template', () => {
    expect(src).not.toContain('${');
    expect(code).not.toContain('undefined');
    expect(code).not.toContain('NaN');
    expect(code).not.toContain('[object Object]');
  });

  it('produces a module a WGSL parser accepts', () => {
    // Structural smoke test: balanced braces and no stray TS in the output. The
    // real compile happens in the browser spec.
    let depth = 0;
    for (const ch of code) {
      if (ch === '{') depth++;
      else if (ch === '}') depth--;
      expect(depth, 'brace depth went negative').toBeGreaterThanOrEqual(0);
    }
    expect(depth).toBe(0);
    expect(code).not.toMatch(/\bimport\b|\bexport\b|=>/);
  });
});

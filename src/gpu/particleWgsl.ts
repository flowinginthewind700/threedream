/**
 * The WGSL for the GPU particle pipeline, generated from the same constants the
 * CPU reference backend uses.
 *
 * # Why the source is a template and not a `.wgsl` file
 *
 * Every magic number in here -- the hash primes, the flag bits, the bounds-mode
 * encoding, the positional-correction fraction, the `Params` member order -- is
 * interpolated from the TypeScript module that already owns it. A hand-written
 * shader file would be a second copy of those numbers, and the failure mode of a
 * second copy is not a compile error: WGSL is perfectly happy to read
 * `restitution` where the CPU wrote `damping`. Generating the text keeps one
 * source of truth, and `tests/particle_wgsl.test.ts` re-derives the struct
 * offsets from the WGSL alignment rules to catch the case where the *layout*
 * drifts rather than a constant.
 *
 * # Constraints this shader is written under
 *
 * - `@workgroup_size(64)` everywhere, and no `subgroup` anything. The
 *   feasibility study measured an adapter with an empty feature set and
 *   `maxComputeInvocationsPerWorkgroup` as low as 128 in compatibility mode, so
 *   64 is the floor every target clears.
 * - No prefix scan. The broadphase uses fixed-capacity buckets filled with
 *   `atomicAdd`, which needs no workgroup-shared reduction; overflow is counted
 *   into `statsBuf` instead of being silently dropped.
 * - f32 only, because WGSL has no f64. The CPU backend is f32 for the same
 *   reason, and parity between them is tolerance-based wherever a `pow`, a
 *   `sqrt` or a driver-contracted multiply-add is involved.
 *
 * # Pass structure
 *
 * One simulation step is `nbody? -> hash_clear -> hash_scatter -> collide? ->
 * integrate -> publish`, matching `CpuParticleSystem.step` pass for pass.
 * `integrate` reads `stateSrc` and writes `stateDst`; the caller swaps the two
 * bindings between steps and then dispatches `publish`, which is why `publish`
 * reads `stateSrc` too -- after the swap, `stateSrc` *is* the new state.
 */

import { POSITION_CORRECTION } from './particleCpu.js';
import { PARTICLE_STRIDE } from './particleField.js';
import { HASH_PRIMES } from './particleHash.js';
import { BOUNDS_MODE_BITS, PARAM_WORD, PARTICLE_FLAG } from './particleOptions.js';

/**
 * Invocations per workgroup, for every kernel.
 *
 * One value on purpose: a mixed pipeline means a mixed set of occupancy limits,
 * and the plan pins 64 as the portable choice. Dispatch counts are therefore
 * `ceil(n / 64)` and every kernel guards with `if (i >= params.count)`.
 */
export const WORKGROUP_SIZE = 64;

/** Floats per particle in the state buffers: `pos.xyz | radius | vel.xyz | mass`. */
export const STATE_FLOATS_PER_PARTICLE = PARTICLE_STRIDE;

/**
 * `vec4`s per particle in the state buffers. Two, so a particle is
 * `state[2i] = (pos, radius)` and `state[2i + 1] = (vel, mass)` -- byte for byte
 * the layout `ParticleField` already uses, which means a field can be uploaded
 * with one `writeBuffer` and read back with one `copyBufferToBuffer` instead of
 * being repacked.
 */
export const STATE_VECS_PER_PARTICLE = STATE_FLOATS_PER_PARTICLE / 4;

/** `vec4`s per particle in the contact accumulator: `(dv, 0)` and `(dx, 0)`. */
export const CONTACT_VECS_PER_PARTICLE = 2;

/** `vec4`s per particle in the n-body accumulator: `(accel, 0)`. */
export const ACCEL_VECS_PER_PARTICLE = 1;

/**
 * Floats per particle in the buffer the renderer reads: `pos.xyz | radius`.
 *
 * Velocity is deliberately not published. The renderer's job is to place
 * instances, and a second `vec4` per particle would double the bandwidth of the
 * only buffer that crosses the compute/graphics boundary every frame.
 */
export const PUBLISH_FLOATS_PER_PARTICLE = 4;

/** Entry points, in dispatch order. `tests/particle_wgsl.test.ts` pins the set. */
export const PARTICLE_KERNELS = [
  'nbody',
  'hash_clear',
  'hash_scatter',
  'collide',
  'integrate',
  'publish',
] as const;

export type ParticleKernel = (typeof PARTICLE_KERNELS)[number];

/**
 * What each kernel's dispatch count is derived from.
 *
 * `hash_clear` is the odd one out: it walks the cell table rather than the
 * particles, because a table sized for a previous count would otherwise keep
 * stale bucket counts. Invocation 0 also zeroes `statsBuf`, so it must run every
 * step even when collisions are off.
 */
export const KERNEL_DISPATCH: Readonly<Record<ParticleKernel, 'count' | 'tableSize'>> = {
  nbody: 'count',
  hash_clear: 'tableSize',
  hash_scatter: 'count',
  collide: 'count',
  integrate: 'count',
  publish: 'count',
};

/** Words in `statsBuf`. The order is pinned by the test against the kernel text. */
export const STAT_WORD = {
  /** Contact pairs resolved this step, i.e. pairs with `i < j`. */
  contacts: 0,
  /** Particles whose centre ended the step outside the box. */
  escaped: 1,
  /** Hash insertions that found a full bucket. */
  overflow: 2,
  /**
   * `bitcast<u32>` of the largest squared speed this step, tracked with
   * `atomicMax`. That works because IEEE-754 non-negative floats sort the same
   * way as their bit patterns, and a squared speed is non-negative by
   * construction -- so the max is taken on integers and decoded on readback.
   */
  maxSpeedSq: 3,
  /** Reserved, so the buffer stays a round five words. */
  unused: 4,
} as const;

export const STAT_WORDS = 5;

/** Bind-group index that never changes between steps. */
export const GROUP_STATIC = 0;
/** Bind-group index that ping-pongs between the two state buffers. */
export const GROUP_STATE = 1;

export type WgslBindingKind = 'uniform' | 'storage-read' | 'storage-read-write';

export type WgslBufferType = 'uniform' | 'read-only-storage' | 'storage';

export interface WgslBinding {
  readonly group: number;
  readonly binding: number;
  readonly name: string;
  readonly kind: WgslBindingKind;
  /** WGSL type as declared in the shader. */
  readonly type: string;
  /** What this binding is in a `GPUBindGroupLayoutEntry`. */
  readonly bufferType: WgslBufferType;
}

/**
 * Every binding the pipeline declares, and the single place the shader text, the
 * bind-group layouts and the bind groups are all derived from.
 *
 * Two groups rather than one: everything in group 0 is allocated once for the
 * lifetime of the system, while group 1 holds the ping-pong pair. Keeping them
 * apart means a step swaps two pre-built bind groups instead of rebuilding one,
 * which is the difference between two `createBindGroup` calls per frame and none.
 */
export const PARTICLE_BINDINGS: readonly WgslBinding[] = [
  {
    group: GROUP_STATIC,
    binding: 0,
    name: 'params',
    kind: 'uniform',
    type: 'Params',
    bufferType: 'uniform',
  },
  {
    group: GROUP_STATIC,
    binding: 1,
    name: 'accelBuf',
    kind: 'storage-read-write',
    type: 'array<vec4<f32>>',
    bufferType: 'storage',
  },
  {
    group: GROUP_STATIC,
    binding: 2,
    name: 'contactBuf',
    kind: 'storage-read-write',
    type: 'array<vec4<f32>>',
    bufferType: 'storage',
  },
  {
    group: GROUP_STATIC,
    binding: 3,
    name: 'hashCounts',
    kind: 'storage-read-write',
    type: 'array<atomic<i32>>',
    bufferType: 'storage',
  },
  {
    group: GROUP_STATIC,
    binding: 4,
    name: 'hashSlots',
    kind: 'storage-read-write',
    type: 'array<u32>',
    bufferType: 'storage',
  },
  {
    group: GROUP_STATIC,
    binding: 5,
    name: 'statsBuf',
    kind: 'storage-read-write',
    type: 'array<atomic<u32>>',
    bufferType: 'storage',
  },
  {
    group: GROUP_STATE,
    binding: 0,
    name: 'stateSrc',
    kind: 'storage-read',
    type: 'array<vec4<f32>>',
    bufferType: 'read-only-storage',
  },
  {
    group: GROUP_STATE,
    binding: 1,
    name: 'stateDst',
    kind: 'storage-read-write',
    type: 'array<vec4<f32>>',
    bufferType: 'storage',
  },
  {
    group: GROUP_STATE,
    binding: 2,
    name: 'publishBuf',
    kind: 'storage-read-write',
    type: 'array<vec4<f32>>',
    bufferType: 'storage',
  },
];

/** Bindings of one group, in binding order. */
export function bindingsForGroup(group: number): readonly WgslBinding[] {
  return PARTICLE_BINDINGS.filter((b) => b.group === group);
}

export type WgslParamKind = 'vec3' | 'f32' | 'u32';

export interface WgslParamMember {
  readonly name: string;
  readonly kind: WgslParamKind;
  /** Offset in 4-byte words, taken from `PARAM_WORD` so the two cannot drift. */
  readonly word: number;
}

/**
 * `Params` in declaration order.
 *
 * The struct body in the shader is generated from this list, and each `word`
 * comes straight out of `PARAM_WORD`. What makes that a real check rather than a
 * tautology is that the *order* here is independent: `writeParams` packs by word
 * index, this list declares by name, and the test recomputes every offset from
 * the WGSL alignment rules (`vec3` aligns to 16, everything else to 4) and
 * compares. A member moved in one place and not the other fails the test.
 */
export const WGSL_PARAMS_LAYOUT: readonly WgslParamMember[] = [
  { name: 'gravity', kind: 'vec3', word: PARAM_WORD.gravityX },
  { name: 'damping', kind: 'f32', word: PARAM_WORD.damping },
  { name: 'boundsMin', kind: 'vec3', word: PARAM_WORD.boundsMinX },
  { name: 'restitution', kind: 'f32', word: PARAM_WORD.restitution },
  { name: 'boundsMax', kind: 'vec3', word: PARAM_WORD.boundsMaxX },
  { name: 'maxSpeed', kind: 'f32', word: PARAM_WORD.maxSpeed },
  { name: 'dt', kind: 'f32', word: PARAM_WORD.dt },
  { name: 'nbodyStrength', kind: 'f32', word: PARAM_WORD.nbodyStrength },
  { name: 'softening', kind: 'f32', word: PARAM_WORD.softening },
  { name: 'cutoffSquared', kind: 'f32', word: PARAM_WORD.cutoffSquared },
  { name: 'cellSize', kind: 'f32', word: PARAM_WORD.cellSize },
  { name: 'invCell', kind: 'f32', word: PARAM_WORD.invCell },
  { name: 'padA', kind: 'f32', word: PARAM_WORD.padA },
  { name: 'padB', kind: 'f32', word: PARAM_WORD.padB },
  { name: 'count', kind: 'u32', word: PARAM_WORD.count },
  { name: 'tableMask', kind: 'u32', word: PARAM_WORD.tableMask },
  { name: 'bucketCapacity', kind: 'u32', word: PARAM_WORD.bucketCapacity },
  { name: 'flags', kind: 'u32', word: PARAM_WORD.flags },
];

const WGSL_TYPE: Readonly<Record<WgslParamKind, string>> = {
  vec3: 'vec3<f32>',
  f32: 'f32',
  u32: 'u32',
};

const ACCESS: Readonly<Record<WgslBindingKind, string>> = {
  uniform: 'var<uniform>',
  'storage-read': 'var<storage, read>',
  'storage-read-write': 'var<storage, read_write>',
};

/** Workgroups needed to cover `n` invocations. `0` means "skip the dispatch". */
export function workgroupsFor(n: number): number {
  if (!Number.isFinite(n) || n <= 0) return 0;
  return Math.ceil(n / WORKGROUP_SIZE);
}

function paramsStruct(): string {
  const members = WGSL_PARAMS_LAYOUT.map((m) => `  ${m.name}: ${WGSL_TYPE[m.kind]},`);
  return ['struct Params {', ...members, '};'].join('\n');
}

function bindingDecls(): string {
  return PARTICLE_BINDINGS.map(
    (b) => `@group(${b.group}) @binding(${b.binding}) ${ACCESS[b.kind]} ${b.name}: ${b.type};`,
  ).join('\n');
}

let cached: string | null = null;

/**
 * The whole particle pipeline as one WGSL module.
 *
 * One module with six entry points rather than six modules: they share `Params`,
 * the hash helpers and the binding declarations, and a single `createShaderModule`
 * means a single place for a compile error to surface. Memoised, because the text
 * is constant for the lifetime of the process.
 */
export function particleShaderSource(): string {
  if (cached !== null) return cached;
  cached = `// Generated by src/gpu/particleWgsl.ts -- edit the generator, not this text.

${paramsStruct()}

${bindingDecls()}

const FLAG_COLLISIONS: u32 = ${PARTICLE_FLAG.collisions}u;
const FLAG_NBODY: u32 = ${PARTICLE_FLAG.nbody}u;
const BOUNDS_SHIFT: u32 = ${PARTICLE_FLAG.boundsShift}u;
const BOUNDS_REFLECT: u32 = ${BOUNDS_MODE_BITS.reflect}u;
const BOUNDS_WRAP: u32 = ${BOUNDS_MODE_BITS.wrap}u;
const BOUNDS_NONE: u32 = ${BOUNDS_MODE_BITS.none}u;

const STAT_CONTACTS: u32 = ${STAT_WORD.contacts}u;
const STAT_ESCAPED: u32 = ${STAT_WORD.escaped}u;
const STAT_OVERFLOW: u32 = ${STAT_WORD.overflow}u;
const STAT_MAX_SPEED_SQ: u32 = ${STAT_WORD.maxSpeedSq}u;

const PRIME_X: u32 = ${HASH_PRIMES.x}u;
const PRIME_Y: u32 = ${HASH_PRIMES.y}u;
const PRIME_Z: u32 = ${HASH_PRIMES.z}u;

const CORRECTION: f32 = ${POSITION_CORRECTION};

// Written out component by component rather than as dot(v, v) so the association
// order matches the CPU backend's f32 sum. A driver is still free to contract
// this into a multiply-add, which is one of the reasons parity is asserted with a
// tolerance rather than with a digest.
fn sumSq(v: vec3<f32>) -> f32 {
  return (v.x * v.x + v.y * v.y) + v.z * v.z;
}

fn sumDot(a: vec3<f32>, b: vec3<f32>) -> f32 {
  return (a.x * b.x + a.y * b.y) + a.z * b.z;
}

// mass === 0 means immovable. Dividing by it would produce Inf and then NaN the
// whole neighbourhood, so the reciprocal is guarded rather than left to whatever
// the driver decides 1/0 is.
fn inverseMass(m: f32) -> f32 {
  if (m > 0.0) { return 1.0 / m; }
  return 0.0;
}

fn hasFlag(flag: u32) -> bool {
  return (params.flags & flag) != 0u;
}

fn boundsMode() -> u32 {
  return (params.flags >> BOUNDS_SHIFT) & 3u;
}

// Integer cell coordinates, measured from boundsMin because that is the origin
// the CPU hash uses. floor then truncate is exact: the value is already integral.
fn cellCoords(p: vec3<f32>) -> vec3<i32> {
  return vec3<i32>(floor((p - params.boundsMin) * params.invCell));
}

// u32 multiply wraps the same way Math.imul does on the CPU side, so the two
// produce the same bucket for the same cell.
fn hashCell(c: vec3<i32>) -> u32 {
  let hx = bitcast<u32>(c.x) * PRIME_X;
  let hy = bitcast<u32>(c.y) * PRIME_Y;
  let hz = bitcast<u32>(c.z) * PRIME_Z;
  return (hx ^ hy ^ hz) & params.tableMask;
}

// Fold one axis into [lo, lo + size). floor rather than a modulo, because WGSL
// has no floored float remainder and this form is bit-identical to the CPU one.
fn wrapAxis(p: f32, lo: f32, size: f32) -> f32 {
  if (!(size > 0.0)) { return p; }
  let k = floor((p - lo) / size);
  return p - k * size;
}

// All-pairs gravity. O(n^2) and off unless the flag says otherwise; this is the
// pass that justifies running the simulation on a GPU at all.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn nbody(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  if (!hasFlag(FLAG_NBODY)) {
    accelBuf[i] = vec4<f32>(0.0);
    return;
  }
  let pi = stateSrc[i * 2u].xyz;
  let soft2 = params.softening * params.softening;
  var acc = vec3<f32>(0.0);
  for (var j = 0u; j < params.count; j = j + 1u) {
    if (j == i) { continue; }
    let bj = j * 2u;
    let d = stateSrc[bj].xyz - pi;
    let d2 = sumSq(d);
    if (params.cutoffSquared > 0.0 && d2 > params.cutoffSquared) { continue; }
    let mj = stateSrc[bj + 1u].w;
    let inv = (params.nbodyStrength * mj) / pow(d2 + soft2, 1.5);
    acc = acc + d * inv;
  }
  accelBuf[i] = vec4<f32>(acc, 0.0);
}

// Zero the cell table and the step counters. Runs every step, collisions or not,
// because invocation 0 owns the stats reset the readback depends on.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn hash_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x == 0u) {
    atomicStore(&statsBuf[STAT_CONTACTS], 0u);
    atomicStore(&statsBuf[STAT_ESCAPED], 0u);
    atomicStore(&statsBuf[STAT_OVERFLOW], 0u);
    atomicStore(&statsBuf[STAT_MAX_SPEED_SQ], 0u);
  }
  if (!hasFlag(FLAG_COLLISIONS)) { return; }
  if (gid.x > params.tableMask) { return; }
  atomicStore(&hashCounts[gid.x], 0);
}

// Fixed-capacity buckets: atomicAdd hands out the slot, and a bucket that is full
// counts the miss instead of growing. No prefix scan, so no subgroups.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn hash_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!hasFlag(FLAG_COLLISIONS)) { return; }
  let i = gid.x;
  if (i >= params.count) { return; }
  let cell = hashCell(cellCoords(stateSrc[i * 2u].xyz));
  let n = atomicAdd(&hashCounts[cell], 1);
  if (n < i32(params.bucketCapacity)) {
    hashSlots[cell * params.bucketCapacity + u32(n)] = i;
  } else {
    atomicAdd(&statsBuf[STAT_OVERFLOW], 1u);
  }
}

// Narrow phase. Each particle accumulates only its own share of every pair it
// finds, reading velocities from the source state and never from the accumulator,
// which is what makes the result independent of the order atomics filled the
// buckets in.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn collide(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let base = i * 2u;
  if (!hasFlag(FLAG_COLLISIONS)) {
    contactBuf[base] = vec4<f32>(0.0);
    contactBuf[base + 1u] = vec4<f32>(0.0);
    return;
  }
  let pi = stateSrc[base].xyz;
  let ri = stateSrc[base].w;
  let vi = stateSrc[base + 1u].xyz;
  let invi = inverseMass(stateSrc[base + 1u].w);
  let c = cellCoords(pi);
  var dv = vec3<f32>(0.0);
  var dx = vec3<f32>(0.0);
  var contacts = 0u;
  // Buckets already visited by this invocation. Two of the 27 offsets can hash to
  // the same bucket whenever the table is smaller than the neighbourhood, and
  // resolving that pair once per alias multiplies the impulse by the alias count.
  var seen: array<u32, 27>;
  var seenCount = 0u;
  for (var oz = -1; oz <= 1; oz = oz + 1) {
    for (var oy = -1; oy <= 1; oy = oy + 1) {
      for (var ox = -1; ox <= 1; ox = ox + 1) {
        let cell = hashCell(c + vec3<i32>(ox, oy, oz));
        var duplicate = false;
        for (var k = 0u; k < seenCount; k = k + 1u) {
          if (seen[k] == cell) { duplicate = true; }
        }
        if (duplicate) { continue; }
        seen[seenCount] = cell;
        seenCount = seenCount + 1u;
        let n = atomicLoad(&hashCounts[cell]);
        let slot = cell * params.bucketCapacity;
        for (var s = 0; s < n; s = s + 1) {
          let j = hashSlots[slot + u32(s)];
          if (j == i) { continue; }
          let bj = j * 2u;
          let d = stateSrc[bj].xyz - pi;
          let rsum = ri + stateSrc[bj].w;
          let d2 = sumSq(d);
          if (d2 >= rsum * rsum) { continue; }
          // Both i and j see this pair; only the lower index counts it, so the
          // total is one per pair no matter how the buckets were filled.
          if (i < j) { contacts = contacts + 1u; }
          // Exactly coincident centres have no normal to push along.
          if (d2 <= 0.0) { continue; }
          let dist = sqrt(d2);
          let nrm = d * (1.0 / dist);
          let invj = inverseMass(stateSrc[bj + 1u].w);
          let invSum = invi + invj;
          // A pair of immovables has nothing to solve.
          if (invSum <= 0.0) { continue; }
          let vn = sumDot(stateSrc[bj + 1u].xyz - vi, nrm);
          if (vn < 0.0) {
            let imp = ((-(1.0 + params.restitution)) * vn) / invSum;
            dv = dv - nrm * (imp * invi);
          }
          // Applied even when the pair is separating: overlapping spheres still
          // have to be pushed apart, or a pile under gravity sinks into itself.
          let corr = (rsum - dist) * ((invi / invSum) * CORRECTION);
          dx = dx - nrm * corr;
        }
      }
    }
  }
  atomicAdd(&statsBuf[STAT_CONTACTS], contacts);
  contactBuf[base] = vec4<f32>(dv, 0.0);
  contactBuf[base + 1u] = vec4<f32>(dx, 0.0);
}

// Semi-implicit Euler plus bounds, src -> dst. The order of operations is the CPU
// backend's order: damp, clamp, contact impulse, advance, positional correction,
// bounds. Reordering any two of those changes the trajectory.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let base = i * 2u;
  let srcPos = stateSrc[base];
  let srcVel = stateSrc[base + 1u];
  let r = srcPos.w;
  let dt = params.dt;
  var a = params.gravity;
  if (hasFlag(FLAG_NBODY)) { a = a + accelBuf[i].xyz; }
  let damp = 1.0 - params.damping * dt;
  var v = (srcVel.xyz + a * dt) * damp;
  let maxSpeed2 = params.maxSpeed * params.maxSpeed;
  let s2 = sumSq(v);
  if (s2 > maxSpeed2) {
    v = v * (params.maxSpeed / sqrt(s2));
  }
  if (hasFlag(FLAG_COLLISIONS)) {
    v = v + contactBuf[base].xyz;
  }
  var p = srcPos.xyz + v * dt;
  if (hasFlag(FLAG_COLLISIONS)) {
    p = p + contactBuf[base + 1u].xyz;
  }

  let mode = boundsMode();
  if (mode == BOUNDS_WRAP) {
    let size = params.boundsMax - params.boundsMin;
    p = vec3<f32>(
      wrapAxis(p.x, params.boundsMin.x, size.x),
      wrapAxis(p.y, params.boundsMin.y, size.y),
      wrapAxis(p.z, params.boundsMin.z, size.z),
    );
  } else if (mode == BOUNDS_NONE) {
    let inside = all(p >= params.boundsMin) && all(p <= params.boundsMax);
    if (!inside) { atomicAdd(&statsBuf[STAT_ESCAPED], 1u); }
  } else {
    let rest = params.restitution;
    let lo = params.boundsMin + vec3<f32>(r);
    let hi = params.boundsMax - vec3<f32>(r);
    if (p.x < lo.x) {
      p.x = lo.x + (lo.x - p.x) * rest;
      v.x = -v.x * rest;
    } else if (p.x > hi.x) {
      p.x = hi.x - (p.x - hi.x) * rest;
      v.x = -v.x * rest;
    }
    if (p.y < lo.y) {
      p.y = lo.y + (lo.y - p.y) * rest;
      v.y = -v.y * rest;
    } else if (p.y > hi.y) {
      p.y = hi.y - (p.y - hi.y) * rest;
      v.y = -v.y * rest;
    }
    if (p.z < lo.z) {
      p.z = lo.z + (lo.z - p.z) * rest;
      v.z = -v.z * rest;
    } else if (p.z > hi.z) {
      p.z = hi.z - (p.z - hi.z) * rest;
      v.z = -v.z * rest;
    }
    // A step whose penetration exceeds the box would reflect straight through to
    // the far wall. The clamp keeps "escaped === 0" a structural invariant under
    // reflect instead of a property of the numbers that happened to arrive.
    p = clamp(p, lo, hi);
  }

  stateDst[base] = vec4<f32>(p, r);
  stateDst[base + 1u] = vec4<f32>(v, srcVel.w);
  atomicMax(&statsBuf[STAT_MAX_SPEED_SQ], bitcast<u32>(sumSq(v)));
}

// Copy what the renderer needs out of the state buffer. Dispatched after the
// ping-pong swap, so stateSrc here is the state integrate just wrote.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn publish(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  publishBuf[i] = stateSrc[i * 2u];
}
`;
  return cached;
}

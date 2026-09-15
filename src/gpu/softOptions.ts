/**
 * Defaults, validation, the shared plan, and the uniform the soft-body kernels read.
 *
 * The CPU reference and the device backend have to compute the *same* function of
 * the same options over the *same* decomposition of the same mesh, and the cheapest
 * way to guarantee that is to have exactly one place resolve the options, exactly
 * one place build the island/coloring layout, and exactly one place pack the
 * uniform. So this module owns all three. The CPU sim reads the resolved struct
 * field by field, the GPU sim uploads it as a 96-byte uniform that `softWgsl.ts`
 * declares as `Params`, and neither backend builds its own plan.
 *
 * # Why the plan lives here and not in the backends
 *
 * `SoftPlan` is the claim M4 has to make testable: island grouping and constraint
 * coloring agree between a deterministic CPU reference and the GPU. If each tier
 * called `groupIslands` and `colorConstraints` itself, the claim would rest on both
 * call sites passing the same arguments, which is exactly the kind of thing that
 * drifts. `buildSoftLayout` is called once per system and hands back the islands,
 * the coloring and the plan together, so a spec can compare the two tiers' plans
 * field by field and the comparison means something.
 *
 * # Why `'wrap'` is refused rather than mapped
 *
 * The particle layer offers `reflect | wrap | none`. Wrapping a lone particle moves
 * it and nothing else. Wrapping a node moves it away from every edge attached to
 * it, so one step stretches each of those constraints across the whole box, the
 * next solve yanks the neighbourhood with it, and the mesh detonates. There is no
 * iteration count that fixes it, so `resolveSoftOptions` throws and names the mode
 * instead of quietly substituting `'reflect'` -- a silent substitution would show
 * up as "the boundsMode slider does nothing", which reads like a UI bug.
 */

import { boundsSize, type Bounds } from './particleField.js';
import { SOFT_STRIDE, type SoftMesh } from './softMesh.js';
import { SOFT_WORKGROUP_SIZE, groupIslands, type SoftIslands } from './softIslands.js';
import { colorConstraints, type SoftColoring } from './softColoring.js';
import type {
  ResolvedSoftOptions,
  SoftBoundsMode,
  SoftPlan,
  SoftSimOptions,
} from './softTypes.js';

export const DEFAULT_SOFT_GRAVITY = [0, -9.81, 0] as const;

export const DEFAULT_SOFT_OPTIONS: ResolvedSoftOptions = {
  gravity: DEFAULT_SOFT_GRAVITY,
  damping: 0.4,
  restitution: 0.3,
  maxSpeed: 50,
  stiffness: 1,
  iterations: 3,
  boundsMode: 'reflect',
  fixedDt: 1 / 60,
  sleep: false,
  sleepThreshold: 0.05,
  sleepAfter: 60,
};

/** `flags` bits in the uniform. Order is pinned by `tests/soft_options.test.ts`. */
export const SOFT_FLAG = {
  sleep: 1,
  /** Bounds mode occupies bits 1-2 so `reflect` stays the zero value. */
  boundsShift: 1,
} as const;

export const SOFT_BOUNDS_MODE_BITS: Readonly<Record<SoftBoundsMode, number>> = {
  reflect: 0,
  none: 1,
};

/**
 * Dispatches every step pays no matter what the graph looks like.
 *
 * `predict`, `finalize`, `measure`, `sleepUpdate` and `publish`. Only the solve
 * scales with the mesh, as `iterations * colors`, so the HUD can show the fixed
 * overhead separately from the part the iteration slider actually moves.
 */
export const SOFT_FIXED_DISPATCHES = 5;

/**
 * Uniform size in floats. 24 f32 = 96 bytes, which is `sizeOf(Params)` in WGSL.
 *
 * The layout is not free: WGSL gives `vec3<f32>` an alignment of 16 bytes, so the
 * three vectors below each start on a word that is a multiple of four and the
 * scalar that follows fills the fourth word of its group. Packed this way the
 * struct is 96 bytes with no interior padding the CPU has to know about, and
 * `softWgsl.ts` declares the members in this exact order.
 */
export const SOFT_PARAMS_FLOATS = 24;
export const SOFT_PARAMS_BYTES = SOFT_PARAMS_FLOATS * 4;

/**
 * Offsets into the uniform, in 4-byte words.
 *
 * Words 0-15 are f32, words 16-23 are u32. Keeping the integers in the tail means
 * the float half is one contiguous run `writeSoftParams` fills without a DataView,
 * and it leaves the two pad words visible rather than implicit. `invDt` sits in the
 * fourth word of the `boundsMin` group because `finalize` derives velocity from
 * `(p - prev) * invDt` and a per-node division by `dt` would be a reciprocal the
 * shader would have to compute anyway.
 */
export const SOFT_PARAM_WORD = {
  gravityX: 0,
  gravityY: 1,
  gravityZ: 2,
  damping: 3,
  boundsMinX: 4,
  boundsMinY: 5,
  boundsMinZ: 6,
  invDt: 7,
  boundsMaxX: 8,
  boundsMaxY: 9,
  boundsMaxZ: 10,
  maxSpeed: 11,
  dt: 12,
  stiffness: 13,
  restitution: 14,
  sleepThresholdSq: 15,
  count: 16,
  islandCount: 17,
  paddedNodes: 18,
  sleepAfter: 19,
  flags: 20,
  constraintCount: 21,
  padA: 22,
  padB: 23,
} as const;

/** The highest solver-iteration count accepted, i.e. `64 * 32` worst-case dispatches. */
export const MAX_SOFT_ITERATIONS = 64;

function finite(name: string, value: number, min: number, max: number): number {
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new RangeError(`${name} must be finite and within [${min}, ${max}], got ${value}`);
  }
  return value;
}

/**
 * Fill in every default and reject anything a kernel could not honour.
 *
 * Two checks are cross-field rather than per-option, because both are only wrong in
 * combination. `damping * fixedDt >= 1` flips the sign of the predict factor
 * `1 - damping*dt`, which looks like the solver injecting energy; and a global
 * `stiffness` above 1 overshoots the rest length every iteration, which looks like
 * a mesh that will not settle. Neither is a bug in the kernels, so both are refused
 * here with a message that names the pair.
 */
export function resolveSoftOptions(options: SoftSimOptions = {}): ResolvedSoftOptions {
  const gravity = options.gravity ?? DEFAULT_SOFT_OPTIONS.gravity;
  if (gravity.length !== 3 || gravity.some((v) => !Number.isFinite(v))) {
    throw new RangeError(`gravity must be three finite numbers, got [${gravity}]`);
  }
  const boundsMode = options.boundsMode ?? DEFAULT_SOFT_OPTIONS.boundsMode;
  if (!(boundsMode in SOFT_BOUNDS_MODE_BITS)) {
    throw new RangeError(
      `boundsMode must be one of reflect|none, got "${boundsMode}"; wrapping a node tears every edge attached to it`,
    );
  }
  const fixedDt = finite('fixedDt', options.fixedDt ?? DEFAULT_SOFT_OPTIONS.fixedDt, 1e-5, 1);
  const damping = finite('damping', options.damping ?? DEFAULT_SOFT_OPTIONS.damping, 0, 60);
  if (damping * fixedDt >= 1) {
    throw new RangeError(
      `damping ${damping} with fixedDt ${fixedDt} reverses velocity every step; damping * fixedDt must be < 1`,
    );
  }
  return {
    gravity: [gravity[0], gravity[1], gravity[2]],
    damping,
    restitution: finite('restitution', options.restitution ?? 0.3, 0, 1),
    maxSpeed: finite('maxSpeed', options.maxSpeed ?? 50, 1e-6, 1e6),
    stiffness: finite('stiffness', options.stiffness ?? 1, 0, 1),
    iterations: Math.trunc(
      finite('iterations', options.iterations ?? 3, 1, MAX_SOFT_ITERATIONS),
    ),
    boundsMode,
    fixedDt,
    sleep: options.sleep ?? false,
    sleepThreshold: finite('sleepThreshold', options.sleepThreshold ?? 0.05, 0, 1e3),
    sleepAfter: Math.trunc(finite('sleepAfter', options.sleepAfter ?? 60, 1, 1e6)),
  };
}

/**
 * Reject a mesh the bounds handling could not honour.
 *
 * A box thinner than one node diameter turns `reflect` into a fight between the two
 * walls: the node is clamped to `min + r`, which is past `max - r`, so every step
 * teleports it and the constraints attached to it stretch to match. No solver
 * iteration can fix that, so it is an argument error. The buffer-length check is
 * here too rather than at each dispatch site, because a buffer that is not
 * `count * SOFT_STRIDE` floats makes every GPU binding lie about its length.
 */
export function assertMeshFits(mesh: SoftMesh): void {
  const diameter = mesh.maxRadius() * 2;
  const size = boundsSize(mesh.bounds);
  for (let axis = 0; axis < 3; axis++) {
    if (!(size[axis] > diameter)) {
      throw new RangeError(
        `bounds axis ${axis} is ${size[axis]} wide, which cannot contain a node of diameter ${diameter}`,
      );
    }
  }
  if (mesh.data.length !== mesh.count * SOFT_STRIDE) {
    throw new RangeError('mesh data does not match its count and stride');
  }
}

/**
 * The islands, the coloring and the plan one mesh produces.
 *
 * Handed back together because the backends need all three and must not disagree
 * about any of them: the GPU uploads `islands.nodeOrder` as the node map and
 * `coloring.order` as the constraint order, and both tiers report `plan`.
 */
export interface SoftLayout {
  readonly islands: SoftIslands;
  readonly coloring: SoftColoring;
  readonly plan: SoftPlan;
}

/**
 * Decompose a mesh once, identically on every tier.
 *
 * The workgroup counts come from the two passes rather than being recomputed here,
 * so `nodeWorkgroups` is by construction the padded node order divided by 64 and
 * `colors` is the number of solve dispatches per iteration.
 */
export function buildSoftLayout(mesh: SoftMesh, resolved: ResolvedSoftOptions): SoftLayout {
  const islands = groupIslands(mesh);
  const coloring = colorConstraints(mesh);
  const plan: SoftPlan = {
    nodes: mesh.count,
    constraints: mesh.constraints.count,
    islands: islands.islands,
    colors: coloring.colors,
    iterations: resolved.iterations,
    nodeWorkgroups: islands.paddedNodes / SOFT_WORKGROUP_SIZE,
    dispatchesPerStep: SOFT_FIXED_DISPATCHES + resolved.iterations * coloring.colors,
    islandSizes: Array.from(islands.islandSizes),
    batchSizes: coloring.batches.map((batch) => batch.count),
  };
  return { islands, coloring, plan };
}

/** Everything `writeSoftParams` needs besides the resolved options. */
export interface SoftParamsFrame {
  /** Step size for this dispatch. Must be finite and > 0. */
  dt: number;
  count: number;
  islandCount: number;
  /** Padded node order length, i.e. `nodeWorkgroups * 64`. */
  paddedNodes: number;
  constraintCount: number;
  sleepAfter: number;
  /** The box the bounds kernels clamp against. */
  bounds: Bounds;
}

/**
 * Pack one frame's uniform into `out`.
 *
 * `out` must be at least `SOFT_PARAMS_BYTES`; the same call fills the CPU-side
 * scratch a test reads back, so what the shader sees is what the spec asserts.
 */
export function writeSoftParams(
  out: ArrayBuffer,
  resolved: ResolvedSoftOptions,
  frame: SoftParamsFrame,
): void {
  if (out.byteLength < SOFT_PARAMS_BYTES) {
    throw new RangeError(`params buffer is ${out.byteLength} bytes, needs ${SOFT_PARAMS_BYTES}`);
  }
  if (!Number.isFinite(frame.dt) || frame.dt <= 0) {
    throw new RangeError(`dt must be finite and > 0, got ${frame.dt}`);
  }
  const floats = new Float32Array(out, 0, 16);
  const ints = new Uint32Array(out, 16 * 4, 8);
  floats.fill(0);
  ints.fill(0);
  floats[SOFT_PARAM_WORD.gravityX] = resolved.gravity[0];
  floats[SOFT_PARAM_WORD.gravityY] = resolved.gravity[1];
  floats[SOFT_PARAM_WORD.gravityZ] = resolved.gravity[2];
  floats[SOFT_PARAM_WORD.damping] = resolved.damping;
  floats[SOFT_PARAM_WORD.boundsMinX] = frame.bounds.min[0];
  floats[SOFT_PARAM_WORD.boundsMinY] = frame.bounds.min[1];
  floats[SOFT_PARAM_WORD.boundsMinZ] = frame.bounds.min[2];
  floats[SOFT_PARAM_WORD.invDt] = 1 / frame.dt;
  floats[SOFT_PARAM_WORD.boundsMaxX] = frame.bounds.max[0];
  floats[SOFT_PARAM_WORD.boundsMaxY] = frame.bounds.max[1];
  floats[SOFT_PARAM_WORD.boundsMaxZ] = frame.bounds.max[2];
  floats[SOFT_PARAM_WORD.maxSpeed] = resolved.maxSpeed;
  floats[SOFT_PARAM_WORD.dt] = frame.dt;
  floats[SOFT_PARAM_WORD.stiffness] = resolved.stiffness;
  floats[SOFT_PARAM_WORD.restitution] = resolved.restitution;
  // Squared, because the sleep test compares against a per-island `max(dot(v,v))`
  // and a sqrt per island per step would only be thrown away by the comparison.
  floats[SOFT_PARAM_WORD.sleepThresholdSq] = resolved.sleepThreshold * resolved.sleepThreshold;
  let flags = 0;
  if (resolved.sleep) flags |= SOFT_FLAG.sleep;
  flags |= SOFT_BOUNDS_MODE_BITS[resolved.boundsMode] << SOFT_FLAG.boundsShift;
  ints[SOFT_PARAM_WORD.count - 16] = frame.count >>> 0;
  ints[SOFT_PARAM_WORD.islandCount - 16] = frame.islandCount >>> 0;
  ints[SOFT_PARAM_WORD.paddedNodes - 16] = frame.paddedNodes >>> 0;
  ints[SOFT_PARAM_WORD.sleepAfter - 16] = frame.sleepAfter >>> 0;
  ints[SOFT_PARAM_WORD.flags - 16] = flags >>> 0;
  ints[SOFT_PARAM_WORD.constraintCount - 16] = frame.constraintCount >>> 0;
}

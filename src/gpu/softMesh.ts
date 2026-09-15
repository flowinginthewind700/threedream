/**
 * The soft-body state container: one interleaved `Float32Array` of nodes, plus
 * the constraint graph that connects them.
 *
 * # Why this is a sibling of `particleField.ts` and not a generalisation of it
 *
 * A particle field is a set of independent bodies: nothing in the M3 kernels
 * reads particle *i* and writes particle *j*, so the only structure the
 * container needs is "count times stride". A soft body is the opposite: the
 * graph *is* the simulation. Two cloths with identical node bytes and different
 * edges are different objects, and a container that held only the bytes would
 * let a caller swap one for the other and get a wrong answer with no error. So
 * the edges live here, next to the state they describe, and every backend reads
 * both from the same place.
 *
 * The node record itself is deliberately the same 32 bytes as a particle:
 * `pos.xyz | invMass | vel.xyz | radius`, two `vec4<f32>`s, no padding. That is
 * what lets `softGpu.ts` upload `mesh.data` with one `writeBuffer` and read it
 * back with one copy -- no repack in either direction to get wrong. `invMass`
 * rather than `mass` is the one field that differs in meaning, and it differs
 * because a pinned node is `invMass === 0`, which is a value the kernels can
 * test directly instead of dividing by.
 *
 * # Why f32
 *
 * Same reason as the particle field: WGSL has no f64, the GPU backend declares
 * `deterministic: false`, and a CPU reference that computed in f64 could never
 * be compared to the shader that has to match it. `softCpu.ts` is the
 * deterministic replay key for this layer, and it is f32 all the way down.
 *
 * Nothing here imports `three` or WebGPU types, so the scenes, the graph and the
 * statistics all run in bare Node under vitest.
 */

import { Rng } from '../core/rng.js';
import { digestWithCount } from '../core/digest.js';
import {
  DEFAULT_BOUNDS,
  boundsCenter,
  boundsSize,
  type Bounds,
  type Vec3Tuple,
} from './particleField.js';

/** Floats per node: `pos.xyz, invMass, vel.xyz, radius`. */
export const SOFT_STRIDE = 8;

/** Bytes per node. The number a budget needs to size a binding. */
export const SOFT_BYTES = SOFT_STRIDE * 4;

/** Offsets into one node record, in floats. */
export const SOFT_OFFSET = {
  position: 0,
  invMass: 3,
  velocity: 4,
  radius: 7,
} as const;

/**
 * Initial topologies. All are deterministic for a given seed.
 *
 * Four rather than one because the plan's M4 wording names three different
 * things -- 软体、布料、质点弹簧 -- and they are three different *graphs*, not
 * three solvers. `cloth` is a 2D lattice with diagonals, `cube` is a 3D lattice
 * without them, `rope` is a chain, and `sheets` is several disconnected cloths.
 * That last one exists because a single connected mesh is one island, and the
 * island pass would then have exactly one thing to group -- the case that proves
 * nothing. `sheets` produces many islands and is what the island-mapped kernels
 * are actually tested against.
 */
export type SoftScene = 'cloth' | 'sheets' | 'cube' | 'rope';

export const SOFT_SCENES: readonly SoftScene[] = ['cloth', 'sheets', 'cube', 'rope'];

/**
 * Stiffness by edge role.
 *
 * Structural edges hold the mesh together, shear edges stop a square cell
 * folding into a diamond, and bend edges stop two adjacent cells folding flat.
 * Giving the latter two a lower rate is what makes a cloth drape instead of
 * snapping straight: at equal stiffness a bend edge fights its two structural
 * neighbours and the mesh ends up creased. All three are multipliers in
 * `[0, 1]`, scaled again by the per-step global stiffness.
 */
export const SOFT_STIFFNESS = {
  structural: 1,
  shear: 0.7,
  bend: 0.3,
} as const;

/**
 * The constraint graph: `count` edges, each an unordered pair of node indices
 * with a rest length and a stiffness.
 *
 * Three parallel arrays rather than one array of objects, because the GPU upload
 * reorders all of them by the same permutation and a struct-of-arrays makes that
 * three `for` loops instead of an allocation per edge. At 20k edges the
 * difference is the whole upload cost.
 */
export interface SoftConstraints {
  readonly count: number;
  /** `2 * count` node indices: `ends[2k]` and `ends[2k + 1]`. */
  readonly ends: Uint32Array;
  /** Natural length of each edge. Positive by construction and by validation. */
  readonly rest: Float32Array;
  /** Per-edge rate in `[0, 1]`, multiplied by the global stiffness each step. */
  readonly stiffness: Float32Array;
}

/** A graph with no edges. Legal: a cloud of free nodes still simulates. */
export function emptyConstraints(): SoftConstraints {
  return {
    count: 0,
    ends: new Uint32Array(0),
    rest: new Float32Array(0),
    stiffness: new Float32Array(0),
  };
}

/** Everything a pre-built mesh needs besides its bytes. */
export interface SoftMeshSpec {
  readonly count: number;
  readonly constraints?: SoftConstraints;
  readonly bounds?: Bounds;
  /** Triangle vertex indices, `3 * t` of them. Empty for a rope or a lattice. */
  readonly triangles?: Uint32Array;
}

export interface SoftMeshOptions {
  /** Node count. Must be a positive integer. Exactly this many nodes result. */
  count: number;
  /** Seed for the initial layout. Same seed, same bytes. */
  seed?: number;
  scene?: SoftScene;
  bounds?: Bounds;
  /** Node radius, used by the bounds kernels. Defaults to a fraction of the spacing. */
  radius?: number;
  /** Inverse mass for every unpinned node. `0` is refused; use `pin()` instead. */
  invMass?: number;
  /** Scale of the initial velocity, in units/second. */
  speed?: number;
  /**
   * Node displacement as a fraction of the lattice spacing.
   *
   * Zero gives a perfect lattice, which is the worst case for a solver: every
   * edge has exactly the same rest length and every node exactly the same
   * neighbours, so a bug that only shows up on an irregular graph stays hidden.
   * A little noise costs nothing and makes the mesh behave like one.
   */
  jitter?: number;
  /** How many disconnected cloths `sheets` builds. Ignored by other scenes. */
  groups?: number;
  /** Add cell diagonals. On by default: without them a cloth shears into a rhombus. */
  shear?: boolean;
  /** Add two-cell skip edges. Off by default: they cost 2n edges for visible drape. */
  bend?: boolean;
}

/** How wide a scene is, as a fraction of the smallest box axis. */
export const SCENE_EXTENT_FRACTION = 0.4;

/** Default node radius as a fraction of the lattice spacing. */
export const RADIUS_FRACTION = 0.35;

/**
 * Ceiling on the default node radius, as a fraction of the smallest box axis.
 *
 * The spacing-derived default is right for a dense mesh and wrong for a sparse
 * one: a three-node rope across a 16-unit box has a spacing of 5.6, and a radius
 * of 2 would put its endpoints outside the reflection box they were built
 * inside. Capping by the box keeps "a freshly built scene is inside its own
 * bounds" true for every count, which is what lets `outOfBounds()` mean
 * something on frame zero.
 */
export const RADIUS_BOX_FRACTION = 0.05;

/** Per-scene defaults for `jitter`, as a fraction of the spacing. */
export const DEFAULT_JITTER: Readonly<Record<SoftScene, number>> = {
  cloth: 0.02,
  sheets: 0.02,
  cube: 0.02,
  // A rope is a visual: jittering it makes the chain look broken rather than loose.
  rope: 0,
};

/** The extent a scene fills, in world units. */
export function defaultExtent(bounds: Bounds): number {
  return Math.min(...boundsSize(bounds)) * SCENE_EXTENT_FRACTION;
}

/** How many nodes, edges and triangles one scene produces, without keeping any. */
export interface SoftSceneSize {
  readonly nodes: number;
  readonly constraints: number;
  readonly triangles: number;
}

/**
 * The shape of a scene before anything is allocated.
 *
 * A demo that offers a 10k-node picker has to size its buffers before it builds
 * the mesh, and a budget that has to build a throwaway 10k-node cloth to learn
 * its edge count is a budget that runs twice. This builds the throwaway once and
 * hands back the three numbers.
 */
export function sizeForScene(
  scene: SoftScene,
  nodes: number,
  options: Omit<SoftMeshOptions, 'count' | 'scene'> = {},
): SoftSceneSize {
  const mesh = new SoftMesh({ ...options, count: nodes, scene });
  return {
    nodes: mesh.count,
    constraints: mesh.constraints.count,
    triangles: mesh.triangles.length / 3,
  };
}

/**
 * The nodes, the graph and the box, as one flat f32 buffer plus typed arrays.
 *
 * A plain container rather than an entity in the ECS, for the reason
 * `ParticleField` gives: at 10k nodes the per-entity overhead of a component map
 * dominates, and the GPU needs a contiguous buffer anyway.
 */
export class SoftMesh {
  readonly count: number;
  readonly bounds: Bounds;
  /** `count * SOFT_STRIDE` f32, laid out as documented at the top. */
  readonly data: Float32Array;
  readonly constraints: SoftConstraints;
  /** `3 * t` vertex indices. Empty for scenes with no surface. */
  readonly triangles: Uint32Array;

  constructor(options: SoftMeshOptions);
  constructor(data: Float32Array, spec: SoftMeshSpec);
  constructor(
    first: SoftMeshOptions | Float32Array,
    second?: SoftMeshSpec,
  ) {
    if (first instanceof Float32Array) {
      const spec = second ?? { count: 0 };
      const count = spec.count;
      if (!Number.isInteger(count) || count <= 0) {
        throw new RangeError(`count must be a positive integer, got ${count}`);
      }
      if (first.length !== count * SOFT_STRIDE) {
        throw new RangeError(
          `data holds ${first.length} floats, but ${count} nodes need ${count * SOFT_STRIDE}`,
        );
      }
      this.data = first;
      this.count = count;
      this.bounds = spec.bounds ?? DEFAULT_BOUNDS;
      assertBounds(this.bounds);
      this.constraints = spec.constraints ?? emptyConstraints();
      this.triangles = spec.triangles ?? new Uint32Array(0);
      assertConstraints(this.constraints, count);
      assertTriangles(this.triangles, count);
      return;
    }

    const options = first;
    if (!Number.isInteger(options.count) || options.count <= 0) {
      throw new RangeError(`count must be a positive integer, got ${options.count}`);
    }
    const scene = options.scene ?? 'cloth';
    if (!SOFT_SCENES.includes(scene)) {
      throw new RangeError(
        `unknown scene "${scene}", expected one of ${SOFT_SCENES.join(', ')}`,
      );
    }
    this.count = options.count;
    this.bounds = options.bounds ?? DEFAULT_BOUNDS;
    assertBounds(this.bounds);
    this.data = new Float32Array(options.count * SOFT_STRIDE);

    const invMass = options.invMass ?? 1;
    if (!(invMass > 0) || !Number.isFinite(invMass)) {
      // Zero would be a pinned node with no way to say which ones, and a
      // negative one would make the solver push two nodes apart by pulling.
      throw new RangeError(`invMass must be finite and > 0, got ${invMass}`);
    }
    const speed = options.speed ?? 0;
    if (!(speed >= 0) || !Number.isFinite(speed)) {
      throw new RangeError(`speed must be finite and >= 0, got ${speed}`);
    }
    const jitter = options.jitter ?? DEFAULT_JITTER[scene];
    if (!(jitter >= 0) || !Number.isFinite(jitter) || jitter >= 0.5) {
      // At 0.5 a node can land on its neighbour, which makes the rest length of
      // that edge zero and the solve step divide by it.
      throw new RangeError(`jitter must be finite and within [0, 0.5), got ${jitter}`);
    }
    if (options.radius !== undefined && !(options.radius > 0)) {
      throw new RangeError(`radius must be positive, got ${options.radius}`);
    }

    const built = fillScene(this, scene, options, invMass, speed, jitter);
    this.constraints = built.constraints;
    this.triangles = built.triangles;
  }

  /** Bytes the GPU needs for this many nodes. */
  static bytesFor(count: number): number {
    return count * SOFT_BYTES;
  }

  /** An all-zero mesh with no edges. Used by tests and by readback targets. */
  static empty(count: number, bounds: Bounds = DEFAULT_BOUNDS): SoftMesh {
    return new SoftMesh(new Float32Array(count * SOFT_STRIDE), { count, bounds });
  }

  clone(): SoftMesh {
    return new SoftMesh(this.data.slice(), {
      count: this.count,
      bounds: this.bounds,
      constraints: this.constraints,
      triangles: this.triangles,
    });
  }

  /**
   * Copy another mesh's node bytes into this one.
   *
   * The graph is *not* copied: two meshes with the same node count can have
   * different edges, and a solver holding a plan built from this mesh's graph
   * would then be running the wrong one. Restoring a saved state is a node-bytes
   * operation, which is exactly what this does.
   */
  copyFrom(other: SoftMesh): void {
    if (other.count !== this.count) {
      throw new RangeError(`cannot copy ${other.count} nodes into ${this.count}`);
    }
    this.data.set(other.data);
  }

  private at(i: number): number {
    if (!Number.isInteger(i) || i < 0 || i >= this.count) {
      throw new RangeError(`node index ${i} out of range [0, ${this.count})`);
    }
    return i * SOFT_STRIDE;
  }

  position(i: number): Vec3Tuple {
    const o = this.at(i);
    return [this.data[o], this.data[o + 1], this.data[o + 2]];
  }

  velocity(i: number): Vec3Tuple {
    const o = this.at(i) + SOFT_OFFSET.velocity;
    return [this.data[o], this.data[o + 1], this.data[o + 2]];
  }

  invMass(i: number): number {
    return this.data[this.at(i) + SOFT_OFFSET.invMass];
  }

  radius(i: number): number {
    return this.data[this.at(i) + SOFT_OFFSET.radius];
  }

  setPosition(i: number, p: Vec3Tuple): void {
    const o = this.at(i);
    this.data[o] = p[0];
    this.data[o + 1] = p[1];
    this.data[o + 2] = p[2];
  }

  setVelocity(i: number, v: Vec3Tuple): void {
    const o = this.at(i) + SOFT_OFFSET.velocity;
    this.data[o] = v[0];
    this.data[o + 1] = v[1];
    this.data[o + 2] = v[2];
  }

  setInvMass(i: number, w: number): void {
    if (!(w >= 0) || !Number.isFinite(w)) {
      throw new RangeError(`invMass must be finite and >= 0, got ${w}`);
    }
    this.data[this.at(i) + SOFT_OFFSET.invMass] = w;
  }

  setRadius(i: number, r: number): void {
    if (!(r > 0)) throw new RangeError(`radius must be positive, got ${r}`);
    this.data[this.at(i) + SOFT_OFFSET.radius] = r;
  }

  /**
   * Nail a node in place.
   *
   * `invMass = 0` rather than a flag, because that is the form the solver needs:
   * the correction is already weighted by inverse mass, so a pinned node
   * receives none of it without a single branch anywhere in the hot loop.
   */
  pin(i: number): void {
    this.setInvMass(i, 0);
  }

  /**
   * Release a pinned node.
   *
   * The mass is not recoverable from `invMass === 0`, so it is an argument. The
   * default of 1 matches what every scene builder writes, which is the case a
   * demo's "unpin" button is always in.
   */
  unpin(i: number, invMass = 1): void {
    if (!(invMass > 0)) {
      throw new RangeError(`unpin needs invMass > 0, got ${invMass}`);
    }
    this.setInvMass(i, invMass);
  }

  isPinned(i: number): boolean {
    return this.invMass(i) === 0;
  }

  /** How many nodes are nailed down. The number a sleeping-island test keys on. */
  pinnedCount(): number {
    let pinned = 0;
    for (let i = 0; i < this.count; i++) {
      if (this.data[i * SOFT_STRIDE + SOFT_OFFSET.invMass] === 0) pinned++;
    }
    return pinned;
  }

  /** The largest radius present, which is what the bounds kernels clamp against. */
  maxRadius(): number {
    let max = 0;
    for (let i = 0; i < this.count; i++) {
      const r = this.data[i * SOFT_STRIDE + SOFT_OFFSET.radius];
      if (r > max) max = r;
    }
    return max;
  }

  maxSpeed(): number {
    let max = 0;
    for (let i = 0; i < this.count; i++) {
      const o = i * SOFT_STRIDE + SOFT_OFFSET.velocity;
      const vx = this.data[o];
      const vy = this.data[o + 1];
      const vz = this.data[o + 2];
      const s2 = vx * vx + vy * vy + vz * vz;
      if (s2 > max) max = s2;
    }
    return Math.sqrt(max);
  }

  /** Sum of 0.5*m*v^2, with pinned nodes contributing zero. */
  kineticEnergy(): number {
    let total = 0;
    for (let i = 0; i < this.count; i++) {
      const o = i * SOFT_STRIDE;
      const w = this.data[o + SOFT_OFFSET.invMass];
      if (!(w > 0)) continue;
      const vx = this.data[o + 4];
      const vy = this.data[o + 5];
      const vz = this.data[o + 6];
      total += 0.5 * (1 / w) * (vx * vx + vy * vy + vz * vz);
    }
    return total;
  }

  /**
   * Nodes whose centre has left the box.
   *
   * A boundary kernel that silently clamps instead of reflecting still "works"
   * for a few hundred frames, so the demo and the specs count escapes rather
   * than trusting the picture.
   */
  outOfBounds(): number {
    const { min, max } = this.bounds;
    let escaped = 0;
    for (let i = 0; i < this.count; i++) {
      const o = i * SOFT_STRIDE;
      const x = this.data[o];
      const y = this.data[o + 1];
      const z = this.data[o + 2];
      if (
        !(
          x >= min[0] && x <= max[0] &&
          y >= min[1] && y <= max[1] &&
          z >= min[2] && z <= max[2]
        )
      ) {
        escaped++;
      }
    }
    return escaped;
  }

  /**
   * The largest `|length - rest| / rest` over every edge.
   *
   * The number that says whether the solver is converging. Computed in f64 here
   * because this is a CPU-side inspection helper and nothing about it has to
   * match a kernel; `softCpu.ts` and the GPU `measure` pass each keep their own
   * f32 version, and a test compares the two against each other with a tolerance.
   */
  maxConstraintError(): number {
    const { ends, rest, count } = this.constraints;
    let worst = 0;
    for (let k = 0; k < count; k++) {
      const r = rest[k];
      if (!(r > 0)) continue;
      const oa = ends[k * 2] * SOFT_STRIDE;
      const ob = ends[k * 2 + 1] * SOFT_STRIDE;
      const dx = this.data[ob] - this.data[oa];
      const dy = this.data[ob + 1] - this.data[oa + 1];
      const dz = this.data[ob + 2] - this.data[oa + 2];
      const err = Math.abs(Math.sqrt(dx * dx + dy * dy + dz * dz) - r) / r;
      if (err > worst) worst = err;
    }
    return worst;
  }

  /** `hex:count` over the raw f32 bytes of every node. */
  digest(): string {
    return digestWithCount(this.data);
  }
}

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

function assertBounds(bounds: Bounds): void {
  for (let axis = 0; axis < 3; axis++) {
    const lo = bounds.min[axis];
    const hi = bounds.max[axis];
    if (!(Number.isFinite(lo) && Number.isFinite(hi)) || !(hi > lo)) {
      throw new RangeError(
        `bounds axis ${axis} must be finite with max > min, got [${lo}, ${hi}]`,
      );
    }
  }
}

/**
 * Reject a graph the solver could not honour.
 *
 * Checked at construction rather than at first step, because every one of these
 * is a silent corruption downstream: an out-of-range endpoint reads another
 * node's memory, a zero rest length divides by zero in the relative-error
 * measure, and a stiffness above 1 overshoots the rest length every iteration
 * and never settles -- which looks like a solver bug and is an argument bug.
 */
export function assertConstraints(constraints: SoftConstraints, count: number): void {
  const { ends, rest, stiffness } = constraints;
  if (!Number.isInteger(constraints.count) || constraints.count < 0) {
    throw new RangeError(`constraint count must be a non-negative integer, got ${constraints.count}`);
  }
  if (ends.length !== constraints.count * 2) {
    throw new RangeError(
      `ends holds ${ends.length} indices, but ${constraints.count} constraints need ${constraints.count * 2}`,
    );
  }
  if (rest.length !== constraints.count || stiffness.length !== constraints.count) {
    throw new RangeError(
      `rest (${rest.length}) and stiffness (${stiffness.length}) must each hold ${constraints.count} entries`,
    );
  }
  for (let k = 0; k < constraints.count; k++) {
    const a = ends[k * 2];
    const b = ends[k * 2 + 1];
    if (a >= count || b >= count) {
      throw new RangeError(
        `constraint ${k} references node ${a >= count ? a : b}, but the mesh has ${count}`,
      );
    }
    if (a === b) {
      throw new RangeError(`constraint ${k} connects node ${a} to itself`);
    }
    if (!(rest[k] > 0)) {
      throw new RangeError(`constraint ${k} has rest length ${rest[k]}, which must be > 0`);
    }
    if (!(stiffness[k] >= 0 && stiffness[k] <= 1)) {
      throw new RangeError(`constraint ${k} has stiffness ${stiffness[k]}, outside [0, 1]`);
    }
  }
}

export function assertTriangles(triangles: Uint32Array, count: number): void {
  if (triangles.length % 3 !== 0) {
    throw new RangeError(
      `triangles holds ${triangles.length} indices, which is not a multiple of 3`,
    );
  }
  for (let t = 0; t < triangles.length; t++) {
    if (triangles[t] >= count) {
      throw new RangeError(
        `triangle index ${t} references node ${triangles[t]}, but the mesh has ${count}`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// scene construction
// ---------------------------------------------------------------------------

/** Edges collected during a build, before their rest lengths are known. */
class EdgeList {
  private readonly a: number[] = [];
  private readonly b: number[] = [];
  private readonly rate: number[] = [];

  get count(): number {
    return this.a.length;
  }

  add(a: number, b: number, stiffness: number): void {
    this.a.push(a);
    this.b.push(b);
    this.rate.push(stiffness);
  }

  /**
   * Freeze into a `SoftConstraints`, measuring every rest length off `data`.
   *
   * Measured rather than declared, because the builders jitter the nodes after
   * deciding the lattice spacing. A rest length taken from the *intended*
   * spacing would put every edge in a pre-stressed state, and a mesh that starts
   * under tension spends its first few steps exploding outward instead of
   * draping.
   */
  finish(data: Float32Array): SoftConstraints {
    const n = this.a.length;
    const ends = new Uint32Array(n * 2);
    const rest = new Float32Array(n);
    const stiffness = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const ia = this.a[k];
      const ib = this.b[k];
      const oa = ia * SOFT_STRIDE;
      const ob = ib * SOFT_STRIDE;
      const dx = data[ob] - data[oa];
      const dy = data[ob + 1] - data[oa + 1];
      const dz = data[ob + 2] - data[oa + 2];
      ends[k * 2] = ia;
      ends[k * 2 + 1] = ib;
      rest[k] = Math.sqrt(dx * dx + dy * dy + dz * dz);
      stiffness[k] = this.rate[k];
    }
    return { count: n, ends, rest, stiffness };
  }
}

interface SceneBuild {
  readonly constraints: SoftConstraints;
  readonly triangles: Uint32Array;
}

/**
 * Lay out every node and collect the graph.
 *
 * Two passes over the same data, in this order, and the order is load-bearing:
 * nodes first (positions, jitter and velocities, consuming the RNG in a fixed
 * sequence), edges second (rest lengths measured from the positions that came
 * out). A builder that computed rest lengths as it went would need the jitter to
 * be decided before the position it jitters.
 */
function fillScene(
  mesh: SoftMesh,
  scene: SoftScene,
  options: SoftMeshOptions,
  invMass: number,
  speed: number,
  jitter: number,
): SceneBuild {
  const rng = new Rng(options.seed);
  const edges = new EdgeList();
  const triangles: number[] = [];
  let spacing = 0;

  switch (scene) {
    case 'cloth':
      // The plane's z is the box centre, not zero: a cloth built inside a box
      // whose centre is not the origin would otherwise start on the far wall,
      // and half its nodes would be outside the bounds they were built in.
      spacing = layoutGrid(mesh, 0, options.count, boundsCenter(mesh.bounds)[2], invMass, edges, triangles, {
        shear: options.shear ?? true,
        bend: options.bend ?? false,
      });
      break;
    case 'sheets':
      spacing = buildSheets(mesh, options.count, options.groups ?? 3, invMass, edges, triangles, {
        shear: options.shear ?? true,
        bend: options.bend ?? false,
      });
      break;
    case 'cube':
      spacing = buildCube(mesh, options.count, invMass, edges);
      break;
    case 'rope':
      spacing = buildRope(mesh, options.count, invMass, edges);
      break;
  }

  // One pass over the nodes for the values that are the same everywhere: radius
  // and initial velocity. Done after the builders so a scene cannot forget one,
  // and so the RNG is consumed in a single documented order -- per node
  // ascending, jitter then velocity -- rather than interleaved with the layout.
  const boxCap = Math.min(...boundsSize(mesh.bounds)) * RADIUS_BOX_FRACTION;
  const radius =
    options.radius ?? Math.min(Math.max(1e-3, spacing * RADIUS_FRACTION), Math.max(1e-3, boxCap));
  const { data } = mesh;
  for (let i = 0; i < mesh.count; i++) {
    const o = i * SOFT_STRIDE;
    if (jitter > 0) {
      const amp = jitter * spacing;
      data[o] += rng.range(-amp, amp);
      data[o + 1] += rng.range(-amp, amp);
      data[o + 2] += rng.range(-amp, amp);
    }
    data[o + SOFT_OFFSET.velocity] = speed === 0 ? 0 : rng.range(-speed, speed);
    data[o + SOFT_OFFSET.velocity + 1] = speed === 0 ? 0 : rng.range(-speed, speed);
    data[o + SOFT_OFFSET.velocity + 2] = speed === 0 ? 0 : rng.range(-speed, speed);
    data[o + SOFT_OFFSET.radius] = radius;
  }

  // Rest lengths are measured after the jitter pass, so what the solver is
  // asked to converge on is the mesh that actually exists.
  const constraints = edges.finish(data);
  assertConstraints(constraints, mesh.count);
  const tri = new Uint32Array(triangles);
  assertTriangles(tri, mesh.count);
  return { constraints, triangles: tri };
}

/** The layout knobs shared by the surface builders. */
interface LatticeOptions {
  readonly shear: boolean;
  readonly bend: boolean;
}

/**
 * Lay out one `cols x rows` grid of nodes in the x-y plane at `z`.
 *
 * @returns the lattice spacing, which is also the node radius's basis.
 *
 * The node count is the caller's, not `cols * rows`: the last row is allowed to
 * be short, and every edge and triangle below is guarded by an index bound
 * rather than by a row bound. A grid that rounded the count up to a full
 * rectangle would build more nodes than it was asked for, and "exactly n" is the
 * property a demo's node-count picker promises.
 */
function layoutGrid(
  mesh: SoftMesh,
  first: number,
  nodes: number,
  z: number,
  invMass: number,
  edges: EdgeList,
  triangles: number[],
  options: LatticeOptions,
): number {
  const cols = Math.max(1, Math.ceil(Math.sqrt(nodes)));
  const rows = Math.max(1, Math.ceil(nodes / cols));
  const extent = defaultExtent(mesh.bounds);
  const spacing = extent / Math.max(1, cols - 1);
  const [cx, cy] = boundsCenter(mesh.bounds);
  const { data } = mesh;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const k = r * cols + c;
      if (k >= nodes) break;
      const i = first + k;
      const o = i * SOFT_STRIDE;
      data[o] = cx - extent / 2 + c * spacing;
      // Row 0 is the top row, which is the one that gets pinned: a cloth hangs
      // from its top edge, and a cloth pinned along the bottom is a cloth that
      // immediately folds over itself.
      data[o + 1] = cy + extent / 2 - r * spacing;
      data[o + 2] = z;
      data[o + SOFT_OFFSET.invMass] = r === 0 ? 0 : invMass;

      const right = c + 1 < cols && k + 1 < nodes ? i + 1 : -1;
      const down = r + 1 < rows && k + cols < nodes ? i + cols : -1;
      if (right >= 0) edges.add(i, right, SOFT_STIFFNESS.structural);
      if (down >= 0) edges.add(i, down, SOFT_STIFFNESS.structural);
      // A full cell needs all four corners, and the fourth one is not implied by
      // the other two existing: on a short last row `i + 1` and `i + cols` can
      // both be nodes while `i + cols + 1` is past the end of the mesh.
      const diag = k + cols + 1 < nodes ? i + cols + 1 : -1;
      if (options.shear && right >= 0 && down >= 0 && diag >= 0) {
        edges.add(i, diag, SOFT_STIFFNESS.shear);
        edges.add(right, down, SOFT_STIFFNESS.shear);
        // Two triangles per full cell, wound consistently so a single-sided
        // material would still show the surface from one side.
        triangles.push(i, down, right, right, down, diag);
      }
      if (options.bend) {
        if (c + 2 < cols && k + 2 < nodes) edges.add(i, i + 2, SOFT_STIFFNESS.bend);
        if (r + 2 < rows && k + 2 * cols < nodes) {
          edges.add(i, i + 2 * cols, SOFT_STIFFNESS.bend);
        }
      }
    }
  }
  return spacing;
}

/**
 * Several disconnected cloths, spread along z.
 *
 * The multi-island scene. Node counts are spread as evenly as they divide, so
 * the last sheet may be one node smaller than the rest -- which is the point: an
 * island grouper that only ever saw equal-sized islands would pass a test it
 * could not pass in a real scene.
 */
function buildSheets(
  mesh: SoftMesh,
  nodes: number,
  groups: number,
  invMass: number,
  edges: EdgeList,
  triangles: number[],
  options: LatticeOptions,
): number {
  if (!Number.isInteger(groups) || groups <= 0) {
    throw new RangeError(`groups must be a positive integer, got ${groups}`);
  }
  const sheets = Math.min(groups, nodes);
  const base = Math.floor(nodes / sheets);
  const extra = nodes % sheets;
  const extent = defaultExtent(mesh.bounds);
  const [, , cz] = boundsCenter(mesh.bounds);
  let first = 0;
  let spacing = 0;
  for (let s = 0; s < sheets; s++) {
    const count = base + (s < extra ? 1 : 0);
    if (count <= 0) continue;
    const z = sheets === 1 ? cz : cz - extent / 2 + (s * extent) / (sheets - 1);
    spacing = layoutGrid(mesh, first, count, z, invMass, edges, triangles, options);
    first += count;
  }
  return spacing;
}

/**
 * An axis-aligned lattice with edge connections only: no diagonals.
 *
 * That is a deliberate choice, and it is what makes the cube a good test of the
 * colouring pass. Maximum degree is 6, so a greedy first-fit needs at most 7
 * colours and typically finds 3; adding face diagonals would raise the degree to
 * 18 and the colour count with it, for a mesh that jitters into a soft ball
 * anyway. The lattice is pinned along its top y-layer, so it hangs and wobbles
 * rather than falling through the floor.
 */
function buildCube(mesh: SoftMesh, nodes: number, invMass: number, edges: EdgeList): number {
  const s = Math.max(1, Math.ceil(Math.cbrt(nodes)));
  const extent = defaultExtent(mesh.bounds);
  const spacing = extent / Math.max(1, s - 1);
  const [cx, cy, cz] = boundsCenter(mesh.bounds);
  const { data } = mesh;
  for (let i = 0; i < nodes; i++) {
    const ix = i % s;
    const iy = Math.floor(i / s) % s;
    const iz = Math.floor(i / (s * s));
    const o = i * SOFT_STRIDE;
    data[o] = cx - extent / 2 + ix * spacing;
    data[o + 1] = cy - extent / 2 + iy * spacing;
    data[o + 2] = cz - extent / 2 + iz * spacing;
    data[o + SOFT_OFFSET.invMass] = iy === s - 1 ? 0 : invMass;
    // Each index bound is checked against `s` *and* against `nodes`: the last
    // z-layer is usually partial, and `i + s < nodes` alone would connect the top
    // node of one layer to the first node of the next.
    if (ix + 1 < s && i + 1 < nodes) edges.add(i, i + 1, SOFT_STIFFNESS.structural);
    if (iy + 1 < s && i + s < nodes) edges.add(i, i + s, SOFT_STIFFNESS.structural);
    if (iz + 1 < s && i + s * s < nodes) {
      edges.add(i, i + s * s, SOFT_STIFFNESS.structural);
    }
  }
  return spacing;
}

/**
 * A chain along x, pinned at both ends.
 *
 * Both ends rather than one. A rope pinned at node 0 only swings under gravity
 * until damping stops it, which is a fine picture and a poor test: the whole
 * chain ends up hanging straight down and every edge reaches the same rest
 * length, so a solver that got the weighting wrong would still look right. Two
 * pinned ends produce a catenary, whose shape *is* the weighting -- an edge
 * solved at half strength sags visibly further, and the max-constraint-error
 * counter has something real to converge towards.
 */
function buildRope(mesh: SoftMesh, nodes: number, invMass: number, edges: EdgeList): number {
  const [sx, sy] = boundsSize(mesh.bounds);
  const [cx, cy, cz] = boundsCenter(mesh.bounds);
  const span = sx * 0.7;
  const spacing = span / Math.max(1, nodes - 1);
  const { data } = mesh;
  for (let i = 0; i < nodes; i++) {
    const o = i * SOFT_STRIDE;
    data[o] = cx - span / 2 + i * spacing;
    // Above centre, so there is somewhere to sag into.
    data[o + 1] = cy + sy * 0.2;
    data[o + 2] = cz;
    const pinned = i === 0 || i === nodes - 1;
    data[o + SOFT_OFFSET.invMass] = pinned ? 0 : invMass;
    if (i + 1 < nodes) edges.add(i, i + 1, SOFT_STIFFNESS.structural);
  }
  return spacing;
}

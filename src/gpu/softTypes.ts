/**
 * The contract every soft-body backend satisfies, and the knobs they share.
 *
 * Type-only for the same reason `particleTypes.ts` is: `tests/tdd.test.ts` pins
 * that a `types.ts` module carries no runtime code, because a contract that grows
 * behaviour needs a spec of its own. Keeping the interface here is what lets
 * `softCpu.ts` and `softGpu.ts` both declare `implements SoftSystem` without
 * importing the factory that imports them -- and what lets the deterministic
 * reference be swapped for the device backend at a call site that only ever
 * names this file.
 *
 * # What "soft body" means here
 *
 * Position-based dynamics over a distance-constraint graph: nodes carry a
 * position, a velocity and an inverse mass, and edges carry a rest length and a
 * stiffness. Cloth, sheets, mass-spring lattices and ropes are all the same
 * solver over a different graph, which is why the plan's "软体、布料、质点弹簧"
 * is one module and not three. Bending resistance, self-collision and contact
 * against other bodies are deliberately out: the graph is the whole input, so
 * the solver's cost is O(edges) per iteration and nothing in it is a special
 * case per scene.
 *
 * # Why `deterministic` is not the whole story
 *
 * The plan's layering rule says the GPU backend declares `deterministic: false`,
 * and it does. That flag answers "may replay and training be built on this", and
 * the answer is no for anything on a device: WGSL allows a driver to contract
 * `a * b + c` into one fused multiply-add, and two adapters may round
 * `sqrt` differently, so bit-equality across machines is not offerable.
 *
 * The soft-body GPU path is nonetheless *race-free*, which the particle layer is
 * not, and that difference is worth a name of its own: `SoftPlan` and the
 * `raceFree` field on the GPU system say so, and `e2e/soft_gpu.spec.ts` asserts
 * the consequence -- the same seed on the same device produces the same digest,
 * twice. Coloring the constraint graph is what buys that, and it is the reason
 * M4 asks for a coloring pass at all.
 */

import type { Bounds, Vec3Tuple } from './particleField.js';
import type { SoftMesh } from './softMesh.js';

/**
 * How a node that reaches the box is handled.
 *
 * `'wrap'` is deliberately absent, unlike the particle layer's `BoundsMode`.
 * Wrapping a lone particle moves it and nothing else; wrapping a node moves it
 * away from every edge it is connected to, so one step turns a cloth into a
 * field of springs each stretched across the whole box. The energy that injects
 * explodes the mesh, and no solver iteration can undo it. `softOptions.ts`
 * rejects the value rather than quietly mapping it to something else.
 */
export type SoftBoundsMode = 'reflect' | 'none';

export interface SoftSimOptions {
  /** Uniform acceleration, units/s^2. `[0, -9.81, 0]` by default. */
  gravity?: Vec3Tuple;
  /** Linear damping per second, applied as `v *= 1 - damping*dt`. */
  damping?: number;
  /** Wall bounce for `boundsMode: 'reflect'`, in `[0, 1]`. */
  restitution?: number;
  /** Speed clamp. Keeps one bad step from throwing a node across the box. */
  maxSpeed?: number;
  /**
   * Global stiffness multiplier in `[0, 1]`, scaled against each edge's own
   * stiffness. 0 makes the mesh limp, 1 solves every edge at its declared rate.
   */
  stiffness?: number;
  /**
   * Solver iterations per step. Each iteration walks every color batch, so the
   * GPU dispatch count grows as `iterations * colors`; this is the knob that
   * trades accuracy for step cost, and the demo exposes it as a slider.
   */
  iterations?: number;
  boundsMode?: SoftBoundsMode;
  /** Simulation step. The renderer is decoupled from it; see `soft.ts`. */
  fixedDt?: number;
  /**
   * Let islands that come to rest stop being simulated. Off by default: nothing
   * in this layer makes islands interact, so a sleeping island stays asleep
   * until `wake()` is called, which is a behaviour a demo should opt into
   * rather than inherit.
   */
  sleep?: boolean;
  /** An island quieter than this speed for `sleepAfter` steps goes to sleep. */
  sleepThreshold?: number;
  /** Consecutive quiet steps required before an island sleeps. */
  sleepAfter?: number;
}

/** Options after defaults and validation, so kernels never re-check them. */
export interface ResolvedSoftOptions {
  readonly gravity: Vec3Tuple;
  readonly damping: number;
  readonly restitution: number;
  readonly maxSpeed: number;
  readonly stiffness: number;
  readonly iterations: number;
  readonly boundsMode: SoftBoundsMode;
  readonly fixedDt: number;
  readonly sleep: boolean;
  readonly sleepThreshold: number;
  readonly sleepAfter: number;
}

/** Per-step instrumentation. The demo HUD and the specs read the same struct. */
export interface SoftStepStats {
  /** Nodes whose centre left the box. Must stay 0 under `reflect`. */
  readonly escaped: number;
  readonly maxSpeed: number;
  /**
   * Largest `|length - rest| / rest` over every edge, measured after the solve.
   *
   * The number that says whether the solver is converging: a cloth at 0.02 is
   * stiff, at 0.5 is stretching badly, and rising over steps means the
   * iteration count or the stiffness is too low for the load. It is also the
   * cleanest CPU/GPU parity check, because it is a max rather than a sum and so
   * is independent of the order the reduction happened in.
   */
  readonly maxConstraintError: number;
  readonly awakeIslands: number;
  readonly sleepingIslands: number;
  readonly kineticEnergy: number;
}

/**
 * The shape of the problem, as opposed to its state.
 *
 * Both backends compute this from the same mesh with the same two passes --
 * `softIslands.ts` and `softColoring.ts` -- so every field here is identical on
 * every tier. That is what makes "island 并行与约束着色通过确定性对照测试" a
 * testable claim rather than a hope: the CPU system's plan *is* the GPU
 * system's plan, and a spec can compare them field by field.
 */
export interface SoftPlan {
  readonly nodes: number;
  readonly constraints: number;
  /** Connected components of the constraint graph. Singletons count. */
  readonly islands: number;
  /** Color batches. One dispatch per color per iteration on the GPU. */
  readonly colors: number;
  readonly iterations: number;
  /**
   * Workgroups in the island-mapped kernels (`predict`, `finalize`): the padded
   * node order divided by the workgroup size. Each workgroup covers 64
   * consecutive nodes of exactly one island and never straddles two, which is
   * what makes "is this island asleep" one load per workgroup instead of one per
   * node.
   */
  readonly nodeWorkgroups: number;
  /**
   * Dispatches one step records on the GPU: the five fixed kernels (predict,
   * finalize, measure, sleep update, publish) plus `iterations * colors` solves.
   * Reported on the CPU tier too, where it is the number of passes the reference
   * walks, so the HUD does not change meaning when the tier does.
   */
  readonly dispatchesPerStep: number;
  /** Nodes per island, ascending island index. */
  readonly islandSizes: readonly number[];
  /** Edges per color, ascending color. */
  readonly batchSizes: readonly number[];
}

/**
 * A soft-body simulation the rest of the engine can drive without knowing which
 * device, if any, is running it.
 */
export interface SoftSystem {
  readonly name: string;
  /** See the file header: false on a device, true for the reference. */
  readonly deterministic: boolean;
  readonly fixedDt: number;
  readonly mesh: SoftMesh;
  readonly options: ResolvedSoftOptions;
  readonly plan: SoftPlan;
  readonly count: number;
  readonly bounds: Bounds;
  /** Simulation steps taken. Independent of frames rendered. */
  readonly steps: number;
  /** Simulated seconds, i.e. `steps * fixedDt`. */
  readonly time: number;

  /** Advance one fixed step. `dt` defaults to `fixedDt`. */
  step(dt?: number): void;
  /** Advance exactly `n` steps, ignoring wall-clock time. Headless path. */
  advance(n: number): void;
  stats(): SoftStepStats;
  /** `hex:count` over the raw f32 state bytes. */
  digest(): string;
  /** Re-activate every island. Cheap, and the only way to undo a sleep. */
  wake(): void;
  /** Release device or heap resources. Idempotent. */
  dispose(): void;
}

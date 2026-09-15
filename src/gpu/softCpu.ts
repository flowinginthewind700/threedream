/**
 * The CPU soft-body backend: the reference every GPU kernel is measured against.
 *
 * # Why this file exists
 *
 * M4 asks for a deterministic CPU reference alongside the GPU layer, and the reason
 * is the one M3 already established: a solver that only runs on a device cannot be
 * tested on a machine without one, cannot be replayed, and cannot be argued about.
 * This is not a slow fallback that happens to exist, it is the specification.
 * `softWgsl.ts` computes the same function in the same pass order over the same
 * decomposition, so a parity spec compares numbers rather than eyeballing a cloth.
 *
 * # Why f32 arithmetic
 *
 * WGSL has no f64. If this file computed in f64 and stored to a `Float32Array`,
 * every intermediate would differ from the shader's by up to one rounding, and a
 * cloth is a stiff system: those roundings feed straight back through the
 * constraints on the next iteration. So every arithmetic step is `Math.fround`-ed
 * in the association the WGSL uses, and a divide is written as a reciprocal
 * followed by multiplies wherever the shader writes it that way. The verbosity is
 * the contract.
 *
 * # Why the pass structure mirrors the dispatch chain
 *
 * One `step()` here is exactly the GPU's `predict -> solve * (iterations * colors)
 * -> finalize -> measure -> sleepUpdate -> publish` chain, walking the same padded
 * node order and the same colored constraint order, with the same scratch: `pred`
 * for the predicted positions and one speed/quiet/asleep triple per island. A bug
 * found on one side is reproducible on the other by running the same pass, which is
 * the point of keeping them structurally identical rather than writing the CPU
 * version in whatever shape is most natural for scalar code.
 *
 * # Why the solve is race-free here too
 *
 * The CPU could solve every constraint in index order and get a perfectly
 * deterministic answer, but it would be a *different* answer from the GPU's, since
 * a constraint solved early changes the positions a later one reads. Walking the
 * colored batches in dispatch order means each batch touches a set of nodes no
 * other constraint in that batch touches, so the sequential walk and the parallel
 * dispatch perform the same arithmetic in the same order. That is what makes
 * `raceFree` a claim about both tiers rather than a hope about one.
 */

import type { Bounds } from './particleField.js';
import { isSentinel } from './softIslands.js';
import { SOFT_OFFSET, SOFT_STRIDE, type SoftMesh } from './softMesh.js';
import {
  assertMeshFits,
  buildSoftLayout,
  resolveSoftOptions,
  type SoftLayout,
} from './softOptions.js';
import type {
  ResolvedSoftOptions,
  SoftPlan,
  SoftSimOptions,
  SoftStepStats,
  SoftSystem,
} from './softTypes.js';

/** Shorthand, so the op order in the hot loops stays readable. */
const f = Math.fround;

export interface CpuSoftSystemOptions {
  mesh: SoftMesh;
  options?: SoftSimOptions;
}

export function createCpuSoftSystem(options: CpuSoftSystemOptions): CpuSoftSystem {
  return new CpuSoftSystem(options);
}

export class CpuSoftSystem implements SoftSystem {
  readonly name = 'cpu';
  /** Bit-reproducible: same mesh, same options, same bytes. */
  readonly deterministic = true;
  /**
   * The batches this walks are the colored ones, so no two constraints solved in
   * the same batch share a node. True on this tier by inspection and on the device
   * tier by construction; `coloringIsRaceFree` checks the coloring itself.
   */
  readonly raceFree = true;
  readonly mesh: SoftMesh;
  readonly options: ResolvedSoftOptions;
  /** Islands, coloring and plan in one object, shared with the device backend. */
  readonly layout: SoftLayout;
  readonly plan: SoftPlan;

  /** Predicted positions, tight xyz per node. The solve reads and writes these. */
  private readonly pred: Float32Array;
  /** Tight xyz positions for the renderer, filled by the publish pass. */
  private readonly published: Float32Array;
  /** Per-island sleep flags. 1 is asleep, matching the GPU's inverted word. */
  private readonly asleep: Uint8Array;
  /** Per-island consecutive-quiet-step counters. */
  private readonly quiet: Uint32Array;
  /** Per-island max squared speed this step. Cleared by the sleep pass. */
  private readonly speedSq: Float32Array;

  private stepsTaken = 0;
  private lastEscaped = 0;
  private lastMaxSpeedSq = 0;
  private lastMaxError = 0;
  private lastSleeping = 0;
  private disposed = false;

  constructor({ mesh, options }: CpuSoftSystemOptions) {
    if (!mesh || !Number.isInteger(mesh.count) || !(mesh.count > 0)) {
      throw new TypeError('CpuSoftSystem needs a SoftMesh with a positive count');
    }
    this.mesh = mesh;
    this.options = resolveSoftOptions(options);
    assertMeshFits(mesh);
    this.layout = buildSoftLayout(mesh, this.options);
    this.plan = this.layout.plan;
    const n3 = mesh.count * 3;
    this.pred = new Float32Array(n3);
    this.published = new Float32Array(n3);
    const islands = this.layout.islands.islands;
    this.asleep = new Uint8Array(islands);
    this.quiet = new Uint32Array(islands);
    this.speedSq = new Float32Array(islands);
    // Frame zero has to be publishable before the first step, or a renderer that
    // draws before stepping shows every node at the origin.
    this.publishPass();
  }

  get fixedDt(): number {
    return this.options.fixedDt;
  }

  get count(): number {
    return this.mesh.count;
  }

  get bounds(): Bounds {
    return this.mesh.bounds;
  }

  get steps(): number {
    return this.stepsTaken;
  }

  get time(): number {
    return this.stepsTaken * this.options.fixedDt;
  }

  step(dt: number = this.options.fixedDt): void {
    this.assertLive();
    if (!Number.isFinite(dt) || !(dt > 0)) {
      throw new RangeError(`dt must be finite and positive, got ${dt}`);
    }
    this.lastEscaped = 0;
    this.lastMaxSpeedSq = 0;
    this.predictPass(dt);
    const { order } = this.layout.coloring;
    for (let it = 0; it < this.options.iterations; it++) {
      // One walk of the whole colored order is exactly `colors` batches in
      // dispatch order, because `order` is their concatenation.
      this.solvePass(order, 0, order.length);
    }
    this.finalizePass(dt);
    this.measurePass();
    this.sleepPass();
    this.publishPass();
    this.stepsTaken++;
  }

  advance(n: number): void {
    if (!Number.isInteger(n) || n < 0) {
      throw new RangeError(`advance needs a non-negative integer, got ${n}`);
    }
    for (let k = 0; k < n; k++) this.step();
  }

  stats(): SoftStepStats {
    return {
      escaped: this.lastEscaped,
      maxSpeed: Math.sqrt(this.lastMaxSpeedSq),
      maxConstraintError: this.lastMaxError,
      awakeIslands: this.layout.islands.islands - this.lastSleeping,
      sleepingIslands: this.lastSleeping,
      kineticEnergy: this.mesh.kineticEnergy(),
    };
  }

  digest(): string {
    return this.mesh.digest();
  }

  wake(): void {
    this.assertLive();
    this.asleep.fill(0);
    this.quiet.fill(0);
    this.speedSq.fill(0);
    this.lastSleeping = 0;
  }

  /**
   * Blit the published positions into a buffer the caller owns.
   *
   * The same tight-xyz layout the device backend's publish buffer uses, so the
   * renderer has one code path for both tiers and `render/soft.ts` never has to
   * know which one it is holding. Returns the number of bytes written.
   */
  copyPublishedTo(target: Float32Array): number {
    this.assertLive();
    const n = Math.min(target.length, this.published.length);
    target.set(this.published.subarray(0, n));
    return n * 4;
  }

  dispose(): void {
    // Nothing to free: the scratch arrays belong to this object and the mesh
    // belongs to the caller. Idempotent, and it makes `step()` after `dispose()` a
    // loud error instead of a silent no-op on a half-torn-down sim.
    this.disposed = true;
  }

  private assertLive(): void {
    if (this.disposed) throw new Error('CpuSoftSystem has been disposed');
  }

  /**
   * Pass 1: integrate velocity, then position, into `pred`.
   *
   * Walks the padded node order rather than `0..count`, because that is what the
   * GPU dispatches over and the sentinel slots are where a straddling workgroup
   * would otherwise read a node belonging to another island. A sleeping island is
   * skipped whole: `finalize` skips it too, so its nodes keep their positions,
   * velocities and last-known constraint error untouched.
   *
   * A pinned node (`invMass === 0`) predicts onto itself. Its velocity needs no
   * special case here, because `finalize` derives velocity from the position
   * change, which is zero, so a pinned node's velocity comes out zero for free.
   */
  private predictPass(dt: number): void {
    const { data } = this.mesh;
    const { nodeOrder, islandOfWorkgroup } = this.layout.islands;
    const opts = this.options;
    const fdt = f(dt);
    const gx = f(opts.gravity[0]);
    const gy = f(opts.gravity[1]);
    const gz = f(opts.gravity[2]);
    const damp = f(1 - f(f(opts.damping) * fdt));
    const asleep = this.asleep;
    const pred = this.pred;

    for (let wg = 0; wg < islandOfWorkgroup.length; wg++) {
      if (asleep[islandOfWorkgroup[wg]!] === 1) continue;
      const base = wg * 64;
      for (let lane = 0; lane < 64; lane++) {
        const i = nodeOrder[base + lane]!;
        if (isSentinel(i)) continue;
        const o = i * SOFT_STRIDE;
        const o3 = i * 3;
        const px = data[o]!;
        const py = data[o + 1]!;
        const pz = data[o + 2]!;
        if (!(data[o + SOFT_OFFSET.invMass]! > 0)) {
          pred[o3] = px;
          pred[o3 + 1] = py;
          pred[o3 + 2] = pz;
          continue;
        }
        const vx = f(f(data[o + 4]! + f(gx * fdt)) * damp);
        const vy = f(f(data[o + 5]! + f(gy * fdt)) * damp);
        const vz = f(f(data[o + 6]! + f(gz * fdt)) * damp);
        pred[o3] = f(px + f(vx * fdt));
        pred[o3 + 1] = f(py + f(vy * fdt));
        pred[o3 + 2] = f(pz + f(vz * fdt));
      }
    }
  }

  /**
   * Passes 2..N+1: one distance constraint, applied to the predicted positions.
   *
   * Position-based dynamics rather than force accumulation: the correction is the
   * whole stretch, split by inverse mass and scaled by the constraint's own
   * stiffness times the global multiplier. No mass matrix, no substepping, and
   * nothing that goes unstable when a constraint is badly scaled -- which is why
   * one kernel serves cloth, rope and a 3D lattice.
   *
   * Both divides are written as a reciprocal followed by multiplies because that
   * is what the WGSL does, and `1/dist` used twice is one division instead of two.
   */
  private solvePass(order: Uint32Array, from: number, to: number): void {
    const { ends, rest, stiffness } = this.mesh.constraints;
    const { data } = this.mesh;
    const { islandOfNode } = this.layout.islands;
    const globalStiff = f(this.options.stiffness);
    const asleep = this.asleep;
    const pred = this.pred;

    for (let k = from; k < to; k++) {
      const e = order[k]!;
      const a = ends[e * 2]!;
      if (asleep[islandOfNode[a]!] === 1) continue;
      const b = ends[e * 2 + 1]!;
      const oa = a * 3;
      const ob = b * 3;
      const dx = f(pred[ob]! - pred[oa]!);
      const dy = f(pred[ob + 1]! - pred[oa + 1]!);
      const dz = f(pred[ob + 2]! - pred[oa + 2]!);
      const d2 = f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz));
      // Two nodes at exactly the same spot have no direction to push along, so
      // the constraint is skipped rather than normalised by zero.
      if (!(d2 > 0)) continue;
      const wa = data[a * SOFT_STRIDE + SOFT_OFFSET.invMass]!;
      const wb = data[b * SOFT_STRIDE + SOFT_OFFSET.invMass]!;
      const w = f(wa + wb);
      // Both ends pinned: nothing can move, and dividing by a zero total weight
      // would be the only NaN source in the solver.
      if (!(w > 0)) continue;
      const dist = f(Math.sqrt(d2));
      const invDist = f(1 / dist);
      const invW = f(1 / w);
      const stiffEff = f(f(stiffness[e]!) * globalStiff);
      const s = f(f(stiffEff * f(f(dist - rest[e]!) * invDist)) * invW);
      const sa = f(s * wa);
      const sb = f(s * wb);
      pred[oa] = f(pred[oa]! + f(dx * sa));
      pred[oa + 1] = f(pred[oa + 1]! + f(dy * sa));
      pred[oa + 2] = f(pred[oa + 2]! + f(dz * sa));
      pred[ob] = f(pred[ob]! - f(dx * sb));
      pred[ob + 1] = f(pred[ob + 1]! - f(dy * sb));
      pred[ob + 2] = f(pred[ob + 2]! - f(dz * sb));
    }
  }

  /**
   * Pass N+2: bounds, velocity from the position change, and the speed clamp.
   *
   * Velocity is *derived* here rather than integrated, which is the part of PBD
   * that makes pinned nodes and wall contacts work without special cases: whatever
   * moved the node -- gravity, a constraint, a reflection -- is what its velocity
   * ends up being. The reflection therefore has to be applied to the position and
   * to the derived velocity together, with the same restitution the particle layer
   * uses, so a wall absorbs energy identically on both tiers.
   *
   * The clamp comes last and only touches velocity. Clamping the position too
   * would fight the constraints, which just put it there.
   */
  private finalizePass(dt: number): void {
    const { data, bounds } = this.mesh;
    const { nodeOrder, islandOfWorkgroup } = this.layout.islands;
    const opts = this.options;
    const invDt = f(1 / f(dt));
    const rest = f(opts.restitution);
    const maxSpeed = f(opts.maxSpeed);
    const maxSpeed2 = f(maxSpeed * maxSpeed);
    const reflect = opts.boundsMode === 'reflect';
    const [minX, minY, minZ] = bounds.min;
    const [maxX, maxY, maxZ] = bounds.max;
    const asleep = this.asleep;
    const pred = this.pred;
    const speedSq = this.speedSq;

    for (let wg = 0; wg < islandOfWorkgroup.length; wg++) {
      const island = islandOfWorkgroup[wg]!;
      if (asleep[island] === 1) continue;
      const base = wg * 64;
      for (let lane = 0; lane < 64; lane++) {
        const i = nodeOrder[base + lane]!;
        if (isSentinel(i)) continue;
        const o = i * SOFT_STRIDE;
        const o3 = i * 3;
        const r = data[o + SOFT_OFFSET.radius]!;
        let px = pred[o3]!;
        let py = pred[o3 + 1]!;
        let pz = pred[o3 + 2]!;
        let vx = f(f(px - data[o]!) * invDt);
        let vy = f(f(py - data[o + 1]!) * invDt);
        let vz = f(f(pz - data[o + 2]!) * invDt);

        if (reflect) {
          const loX = f(minX + r);
          const hiX = f(maxX - r);
          const loY = f(minY + r);
          const hiY = f(maxY - r);
          const loZ = f(minZ + r);
          const hiZ = f(maxZ - r);
          if (px < loX) {
            px = f(loX + f(f(loX - px) * rest));
            vx = f(-vx * rest);
          } else if (px > hiX) {
            px = f(hiX - f(f(px - hiX) * rest));
            vx = f(-vx * rest);
          }
          if (py < loY) {
            py = f(loY + f(f(loY - py) * rest));
            vy = f(-vy * rest);
          } else if (py > hiY) {
            py = f(hiY - f(f(py - hiY) * rest));
            vy = f(-vy * rest);
          }
          if (pz < loZ) {
            pz = f(loZ + f(f(loZ - pz) * rest));
            vz = f(-vz * rest);
          } else if (pz > hiZ) {
            pz = f(hiZ - f(f(pz - hiZ) * rest));
            vz = f(-vz * rest);
          }
          // A reflection can overshoot to the far wall when one step's penetration
          // exceeds the box, which only an unclamped dt can do. The clamp keeps
          // "escaped === 0" a structural invariant under reflect rather than a
          // property of the numbers that happened to arrive.
          px = clamp(px, loX, hiX);
          py = clamp(py, loY, hiY);
          pz = clamp(pz, loZ, hiZ);
        } else if (
          !(px >= minX && px <= maxX && py >= minY && py <= maxY && pz >= minZ && pz <= maxZ)
        ) {
          this.lastEscaped++;
        }

        let s2 = f(f(f(vx * vx) + f(vy * vy)) + f(vz * vz));
        if (s2 > maxSpeed2) {
          const k = f(maxSpeed / f(Math.sqrt(s2)));
          vx = f(vx * k);
          vy = f(vy * k);
          vz = f(vz * k);
          s2 = maxSpeed2;
        }
        data[o] = px;
        data[o + 1] = py;
        data[o + 2] = pz;
        data[o + 4] = vx;
        data[o + 5] = vy;
        data[o + 6] = vz;
        if (s2 > this.lastMaxSpeedSq) this.lastMaxSpeedSq = s2;
        if (s2 > speedSq[island]!) speedSq[island] = s2;
      }
    }
  }

  /**
   * Pass N+3: the largest relative constraint error over every edge.
   *
   * Relative rather than absolute, because a rope with a 0.5 unit rest length and
   * a cloth with a 0.05 one are equally broken at 10% stretch and unequally broken
   * at 0.01 units. Read off the final positions in `data`, not off `pred`, so a
   * sleeping island contributes its frozen error instead of a stale prediction.
   *
   * Unlike every other pass this one is order-independent: it is a max, so the CPU
   * walks the edges in index order while the GPU reduces it with an atomic and the
   * two still agree exactly.
   */
  private measurePass(): void {
    const { ends, rest, count } = this.mesh.constraints;
    const { data } = this.mesh;
    let worst = 0;
    for (let e = 0; e < count; e++) {
      const oa = ends[e * 2]! * SOFT_STRIDE;
      const ob = ends[e * 2 + 1]! * SOFT_STRIDE;
      const dx = f(data[ob]! - data[oa]!);
      const dy = f(data[ob + 1]! - data[oa + 1]!);
      const dz = f(data[ob + 2]! - data[oa + 2]!);
      const d2 = f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz));
      if (!(d2 > 0)) continue;
      const dist = f(Math.sqrt(d2));
      const r = rest[e]!;
      const err = f(f(Math.abs(f(dist - r))) / r);
      if (err > worst) worst = err;
    }
    this.lastMaxError = worst;
  }

  /**
   * Pass N+4: put an island to sleep once it has been quiet for `sleepAfter` steps.
   *
   * Per island rather than per node, because a cloth is one body: a node that stops
   * moving while its neighbours keep pulling is not at rest, and sleeping it alone
   * would freeze one corner of a falling sheet. Per island is also what makes the
   * saving real -- the whole workgroup is skipped, not half its lanes.
   *
   * The speed accumulator is cleared here rather than at the top of the step,
   * which is what lets the GPU leave the flag word zero-initialised and still
   * start with every island awake: a fresh buffer is all zeros, zero means awake,
   * and the first sleep pass leaves it in the state the second step needs.
   */
  private sleepPass(): void {
    const opts = this.options;
    const thresholdSq = f(f(opts.sleepThreshold) * f(opts.sleepThreshold));
    const speedSq = this.speedSq;
    const asleep = this.asleep;
    const quiet = this.quiet;
    let sleeping = 0;
    for (let k = 0; k < asleep.length; k++) {
      const sq = speedSq[k]!;
      speedSq[k] = 0;
      if (!opts.sleep) continue;
      if (asleep[k] === 1) {
        sleeping++;
        continue;
      }
      if (sq < thresholdSq) {
        const q = quiet[k]! + 1;
        quiet[k] = q;
        if (q >= opts.sleepAfter) {
          asleep[k] = 1;
          sleeping++;
        }
      } else {
        quiet[k] = 0;
      }
    }
    this.lastSleeping = sleeping;
  }

  /** Pass N+5: the tight xyz array the renderer reads. */
  private publishPass(): void {
    const { data, count } = this.mesh;
    const out = this.published;
    for (let i = 0; i < count; i++) {
      const o = i * SOFT_STRIDE;
      const o3 = i * 3;
      out[o3] = data[o]!;
      out[o3 + 1] = data[o + 1]!;
      out[o3 + 2] = data[o + 2]!;
    }
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * The particle renderer: one `InstancedMesh`, one draw call, and two ways to fill
 * it.
 *
 * # What the plan asks for, and what that rules out
 *
 * `docs/development-plan.md` says "用 three.js `InstancedMesh` 渲染，不回写仿真
 * 状态". The first half fixes the draw call count; the second half is the one
 * with teeth, and it is why this module has no reference to the integrator: a
 * renderer that wrote back into `field.data` would make the simulation's next
 * step depend on how the previous frame was drawn, and a replay would then be a
 * function of the display. `update()` reads and draws. Nothing here writes to the
 * field, and `tests/render_particles.test.ts` pins that by digest.
 *
 * # Two fill paths, one mesh
 *
 * Every tier draws the same `InstancedMesh` with the same stock material. What
 * differs is how the per-instance `mat4`s get into `instanceMatrix`:
 *
 * - `gpu-blit`: `gpu/particleInstances.ts` expands the publish buffer into
 *   matrices on the device and copies them into the `GPUBuffer` three.js already
 *   created. No particle data crosses the bus, and the CPU never sees positions.
 * - `cpu-upload`: the matrices are written into `instanceMatrix.array` here and
 *   three.js uploads them. This is the WebGL2 and CPU tier, and it is also the
 *   WebGPU tier's first frame and its failure path.
 *
 * One mesh rather than a per-tier scene graph, because the alternative is two
 * render paths to keep visually identical -- and a fallback that looks different
 * is a fallback nobody believes. It also means the tier can change at runtime
 * without rebuilding anything.
 *
 * # Why the GPU path attaches late, and quietly
 *
 * three.js creates an attribute's `GPUBuffer` when the mesh is first rendered, so
 * a freshly constructed view cannot see one yet. `update()` therefore starts the
 * expansion pipeline the first time a buffer appears, keeps uploading from the
 * CPU until it exists, and never throws from the frame loop: if compilation fails
 * or the device is lost, `gpuError` says why and the mesh keeps drawing. A page
 * that went blank because a *renderer* could not get a compute pipeline would be
 * the exact failure the capability matrix exists to prevent.
 */

import * as THREE from 'three';

import type { GpuBufferLike } from '../gpu/device.js';
import { PARTICLE_OFFSET, PARTICLE_STRIDE } from '../gpu/particleField.js';
import type { GpuParticleSystem } from '../gpu/particleGpu.js';
import {
  INSTANCE_BYTES,
  INSTANCE_FLOATS,
  InstanceExpander,
} from '../gpu/particleInstances.js';
import type { ParticleSystem } from '../gpu/particleTypes.js';

/**
 * Anything indexable that can hold 16 floats per instance.
 *
 * Declared structurally because `BufferAttribute.array` is three's `TypedArray`
 * union, and a parameter typed `Float32Array` would not accept it -- which would
 * push every caller into a cast and hide the one requirement that matters, that
 * the writes are indexable.
 */
export type MatrixTarget = ArrayLike<number> & { [index: number]: number };

/** Which fill path a frame used. Reported, because a silent downgrade is a lie. */
export type ParticleFrameMode = 'gpu-blit' | 'cpu-upload';

/** Which fill path the view is settled on. `cpu` until a blit has succeeded. */
export type ParticleViewMode = 'gpu' | 'cpu';

/**
 * Above this many instances the spheres get cheaper.
 *
 * An `InstancedMesh` multiplies one geometry by the count, so vertex throughput
 * is `count * triangles`. At 4k instances a 16x12 sphere is comfortable; at
 * 100k the same mesh is 35M triangles and the frame is vertex-bound before the
 * simulation has said anything. An icosahedron is 20 triangles and still reads
 * as a sphere at the size particles are drawn at.
 */
export const HIGH_DETAIL_MAX_COUNT = 4096;
export const MEDIUM_DETAIL_MAX_COUNT = 32768;

/** The geometry for `count` instances. Owned by the view unless one is injected. */
export function geometryForCount(count: number): THREE.BufferGeometry {
  if (!Number.isInteger(count) || count <= 0) {
    throw new RangeError(`instance geometry needs a positive integer count, got ${count}`);
  }
  if (count <= HIGH_DETAIL_MAX_COUNT) return new THREE.SphereGeometry(1, 16, 12);
  if (count <= MEDIUM_DETAIL_MAX_COUNT) return new THREE.SphereGeometry(1, 8, 6);
  return new THREE.IcosahedronGeometry(1, 0);
}

/**
 * Write `count` scale-plus-translate matrices into an `instanceMatrix` array.
 *
 * Column major, matching both three.js and the WGSL kernel: the first three
 * columns are the scale on the diagonal, the fourth is the translation. All
 * sixteen floats are written every time rather than the four that change,
 * because a function that assumes its caller zeroed twelve of every sixteen
 * slots is a function that breaks the first time it is handed a reused array --
 * and three.js initialises `instanceMatrix` to *identity*, not to zeros, so the
 * assumption would be wrong on the very first call.
 *
 * @returns how many instances were written.
 */
export function writeInstanceMatrices(
  target: MatrixTarget,
  data: ArrayLike<number>,
  count: number,
  radiusScale = 1,
): number {
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError(`writeInstanceMatrices needs a non-negative integer count, got ${count}`);
  }
  if (!Number.isFinite(radiusScale) || !(radiusScale > 0)) {
    throw new RangeError(`radiusScale must be finite and positive, got ${radiusScale}`);
  }
  if (target.length < count * INSTANCE_FLOATS) {
    throw new RangeError(
      `writeInstanceMatrices: the target holds ${target.length} floats, ${count} instances need ${count * INSTANCE_FLOATS}`,
    );
  }
  if (data.length < count * PARTICLE_STRIDE) {
    throw new RangeError(
      `writeInstanceMatrices: the field holds ${data.length} floats, ${count} particles need ${count * PARTICLE_STRIDE}`,
    );
  }
  for (let i = 0; i < count; i++) {
    const src = i * PARTICLE_STRIDE;
    const dst = i * INSTANCE_FLOATS;
    const r = data[src + PARTICLE_OFFSET.radius] * radiusScale;
    target[dst] = r;
    target[dst + 1] = 0;
    target[dst + 2] = 0;
    target[dst + 3] = 0;
    target[dst + 4] = 0;
    target[dst + 5] = r;
    target[dst + 6] = 0;
    target[dst + 7] = 0;
    target[dst + 8] = 0;
    target[dst + 9] = 0;
    target[dst + 10] = r;
    target[dst + 11] = 0;
    target[dst + 12] = data[src + PARTICLE_OFFSET.position];
    target[dst + 13] = data[src + PARTICLE_OFFSET.position + 1];
    target[dst + 14] = data[src + PARTICLE_OFFSET.position + 2];
    target[dst + 15] = 1;
  }
  return count;
}

// ---------------------------------------------------------------------------
// the renderer surface this layer is allowed to touch
// ---------------------------------------------------------------------------

/** What `backend.get(attribute)` hands back for an attribute it has created. */
export interface AttributeHandle {
  readonly buffer?: GpuBufferLike;
}

/**
 * The slice of `WebGPURenderer` this module uses, declared structurally.
 *
 * `WebGLRenderer` has no `.backend` at all -- it keeps its buffers behind
 * `.attributes`, and those are `WebGLBuffer`s, which a compute pipeline cannot
 * bind. Reading `.backend?.get(...)` is therefore also the tier test: if it
 * yields a buffer, the renderer is WebGPU and the blit path is available.
 */
export interface RendererBackendLike {
  get(attribute: unknown): AttributeHandle | undefined;
}

export interface RendererLike {
  readonly backend?: RendererBackendLike | undefined;
}

/** The `GPUBuffer` behind an attribute, or `null` while three.js has not made one. */
export function instanceBufferOf(
  renderer: RendererLike | null | undefined,
  attribute: unknown,
): GpuBufferLike | null {
  const handle = renderer?.backend?.get(attribute);
  const buffer = handle?.buffer;
  if (!buffer) return null;
  // A WebGLBuffer would land here too if a caller passed a WebGL renderer that
  // grew a `backend`. Asking for the two methods a copy needs is cheaper than
  // trusting the shape, and the failure it prevents is a validation error with
  // no context at submit time.
  if (typeof buffer.mapAsync !== 'function' || typeof buffer.getMappedRange !== 'function') {
    return null;
  }
  return buffer;
}

/** Whether a system is the GPU backend, without importing its class. */
function asGpuSystem(system: ParticleSystem): GpuParticleSystem | null {
  const candidate = system as Partial<GpuParticleSystem>;
  if (system.name !== 'gpu') return null;
  if (!candidate.publish || typeof candidate.copyPublishedTo !== 'function') return null;
  if (!candidate.shared) return null;
  return system as GpuParticleSystem;
}

// ---------------------------------------------------------------------------
// the view
// ---------------------------------------------------------------------------

export interface ParticleViewOptions {
  readonly system: ParticleSystem;
  /**
   * The renderer to blit into. Omit it and the view is CPU-upload only, which is
   * what a headless spec and a WebGL2 page both want.
   */
  readonly renderer?: RendererLike | null;
  readonly radiusScale?: number;
  /** Injected geometry and material are not disposed by the view. */
  readonly geometry?: THREE.BufferGeometry;
  readonly material?: THREE.Material;
  readonly color?: number;
}

/**
 * An `InstancedMesh` kept in step with a particle system.
 *
 * The mesh is the only thing a caller has to add to a scene; everything else is
 * `update()` once a frame, after the simulation has stepped and before
 * `renderer.render()`.
 */
export class ParticleView {
  readonly system: ParticleSystem;
  readonly mesh: THREE.InstancedMesh;
  /** The GPU backend behind `system`, or `null` for the CPU and WebGL2 tiers. */
  readonly gpu: GpuParticleSystem | null;

  private readonly renderer: RendererLike | null;
  private readonly ownsGeometry: boolean;
  private readonly ownsMaterial: boolean;
  private expander: InstanceExpander | null = null;
  private attaching: Promise<ParticleViewMode> | null = null;
  private gpuErrorText: string | null = null;
  private scale: number;
  private lastFrame: ParticleFrameMode = 'cpu-upload';
  private lastBytes = 0;
  private disposedFlag = false;

  constructor(options: ParticleViewOptions) {
    const system = options.system;
    if (!system || !Number.isInteger(system.count) || system.count <= 0) {
      throw new TypeError('ParticleView needs a particle system with a positive count');
    }
    this.system = system;
    this.gpu = asGpuSystem(system);
    this.renderer = options.renderer ?? null;
    this.scale = options.radiusScale ?? 1;
    if (!Number.isFinite(this.scale) || !(this.scale > 0)) {
      throw new RangeError(`radiusScale must be finite and positive, got ${this.scale}`);
    }

    this.ownsGeometry = options.geometry === undefined;
    this.ownsMaterial = options.material === undefined;
    const geometry = options.geometry ?? geometryForCount(system.count);
    const material =
      options.material ??
      new THREE.MeshLambertMaterial({ color: options.color ?? 0x7fd4ff, flatShading: true });

    this.mesh = new THREE.InstancedMesh(geometry, material, system.count);
    // The instances move on the GPU, so the geometry's bounding sphere describes
    // a mesh that no longer exists. Culling against it would drop the whole draw
    // call the moment the camera turned, which looks exactly like a simulation
    // that stopped.
    this.mesh.frustumCulled = false;
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    // One CPU pass up front: it makes the first frame correct on every tier, and
    // it means a re-upload by three.js -- after a context restore, say -- carries
    // positions rather than the identity matrices the constructor filled in.
    this.uploadFromField();
  }

  /**
   * The path in use: `gpu` once the expansion pipeline exists, `cpu` before that
   * and after any GPU failure. Derived rather than stored, so it cannot drift out
   * of step with the thing that actually decides it.
   */
  get viewMode(): ParticleViewMode {
    return this.expander !== null ? 'gpu' : 'cpu';
  }

  /** What the last `update()` did. */
  get frameMode(): ParticleFrameMode {
    return this.lastFrame;
  }

  /** Bytes blitted by the last `update()`, or 0 for a CPU frame. */
  get blittedBytes(): number {
    return this.lastBytes;
  }

  /** Why the GPU path is not being used, or `null`. Surfaced by the demo HUD. */
  get gpuError(): string | null {
    return this.gpuErrorText;
  }

  /** True once the expansion pipeline exists and a blit can happen. */
  get ready(): boolean {
    return this.expander !== null;
  }

  get radiusScale(): number {
    return this.scale;
  }

  get disposed(): boolean {
    return this.disposedFlag;
  }

  /** Instances drawn. Follows `mesh.count`, which a caller may lower. */
  get count(): number {
    return this.mesh.count;
  }

  /**
   * Wait for the GPU path to settle.
   *
   * Resolves to `viewMode`, so a spec can say "attached or definitively not"
   * instead of polling. Attachment only starts once `update()` has seen a
   * renderer-owned buffer, so the useful sequence is one render, one update, then
   * this.
   */
  async settled(): Promise<ParticleViewMode> {
    if (this.attaching) await this.attaching;
    return this.viewMode;
  }

  /**
   * Bring the mesh up to date with the simulation. Call once a frame.
   *
   * Never throws for a GPU problem: the frame falls back to a CPU upload and the
   * reason lands in `gpuError`. It throws only for use after `dispose()`, which
   * is a caller bug and not a device condition.
   */
  update(): ParticleFrameMode {
    if (this.disposedFlag) throw new Error('ParticleView has been disposed');

    const target = this.targetBuffer();
    if (target && !this.expander && !this.attaching && !this.gpuErrorText) {
      this.beginAttach();
    }

    if (this.expander && target) {
      try {
        // Capped by both what is drawn and what the destination holds. The
        // buffer belongs to three.js, so its size is a fact about the renderer
        // and not about `count`; a copy past the end is a validation error whose
        // message contains no numbers, and it would arrive at submit time.
        const capacity = Math.min(target.size, this.count * INSTANCE_BYTES);
        const bytes = this.expander.expandTo(target, capacity);
        if (bytes > 0) {
          this.lastFrame = 'gpu-blit';
          this.lastBytes = bytes;
          // Deliberately no `needsUpdate`: the buffer now holds matrices three.js
          // did not write, and asking it to upload the CPU array would overwrite
          // them with the last frame's -- or, on the first blit, with identities.
          return 'gpu-blit';
        }
      } catch (error) {
        this.failGpu(error);
      }
    }

    // The CPU path. For a GPU-backed system `field.data` is whatever the last
    // `readback()` left there, so this frame shows stale positions rather than
    // nothing -- the right trade for a fallback, and the reason `gpuError` exists
    // is that "stale" must never be silent.
    this.uploadFromField();
    return 'cpu-upload';
  }

  /** Change the drawn radius without touching the simulation. */
  setRadiusScale(radiusScale: number): void {
    if (this.disposedFlag) throw new Error('ParticleView has been disposed');
    if (!Number.isFinite(radiusScale) || !(radiusScale > 0)) {
      throw new RangeError(`radiusScale must be finite and positive, got ${radiusScale}`);
    }
    this.scale = radiusScale;
    // The uniform only matters once the kernel exists; before that the CPU pass
    // reads `this.scale` directly.
    this.expander?.setRadiusScale(radiusScale);
  }

  /** Draw how many of the instances. Lowering it does not resize any buffer. */
  setCount(count: number): void {
    if (this.disposedFlag) throw new Error('ParticleView has been disposed');
    if (!Number.isInteger(count) || count < 0 || count > this.system.count) {
      throw new RangeError(
        `count must be an integer in [0, ${this.system.count}], got ${count}`,
      );
    }
    this.mesh.count = count;
  }

  /** Release the mesh, and the expander's device reference with it. */
  dispose(): void {
    if (this.disposedFlag) return;
    this.disposedFlag = true;
    this.expander?.dispose();
    this.expander = null;
    this.mesh.dispose();
    if (this.ownsGeometry) this.mesh.geometry.dispose();
    if (this.ownsMaterial) (this.mesh.material as THREE.Material).dispose();
  }

  /** The `GPUBuffer` three.js owns for `instanceMatrix`, or `null`. */
  private targetBuffer(): GpuBufferLike | null {
    if (!this.gpu) return null;
    return instanceBufferOf(this.renderer, this.mesh.instanceMatrix);
  }

  /**
   * Start building the expansion pipeline. Fire-and-forget on purpose: `update()`
   * runs inside a render loop, and a loop that awaits a shader compile stalls the
   * page for the length of a driver call. Frames upload from the CPU until it
   * lands.
   */
  private beginAttach(): void {
    const gpu = this.gpu!;
    this.attaching = InstanceExpander.create({
      shared: gpu.shared,
      count: this.system.count,
      source: gpu.publish,
      radiusScale: this.scale,
      label: 'particle-view',
    })
      .then((expander) => {
        if (this.disposedFlag) {
          // Built while the page was tearing down. Dropping it is the leak-free
          // answer: nothing else holds a reference to that device.
          expander.dispose();
          return this.viewMode;
        }
        this.expander = expander;
        return this.viewMode;
      })
      .catch((error: unknown) => {
        this.failGpu(error);
        return this.viewMode;
      });
  }

  private failGpu(error: unknown): void {
    // The expander is useless once its device is gone, and keeping it would keep
    // a reference to that device alive for the life of the page. Dropping it can
    // itself fail: a device destroyed out from under the view zeroes its
    // refcount, and `release()` refuses to go negative. An error handler that
    // throws would break the one promise `update()` makes, so it does not. The
    // field is cleared first, so a failed dispose cannot leave it holding a dead
    // expander that the next frame would try to submit to.
    const expander = this.expander;
    this.expander = null;
    try {
      expander?.dispose();
    } catch {
      // The device is gone; there is nothing left to release and nowhere to put it.
    }
    this.gpuErrorText = error instanceof Error ? error.message : String(error);
  }

  private uploadFromField(): void {
    writeInstanceMatrices(
      this.mesh.instanceMatrix.array,
      this.system.field.data,
      this.count,
      this.scale,
    );
    this.mesh.instanceMatrix.needsUpdate = true;
    this.lastFrame = 'cpu-upload';
    this.lastBytes = 0;
  }
}

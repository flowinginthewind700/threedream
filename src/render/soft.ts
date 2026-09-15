/**
 * The soft-body renderer: one position attribute shared by every drawable, and
 * two ways to fill it.
 *
 * # What the plan asks for, and why it is stricter here than for particles
 *
 * M4 inherits M3's rule -- "不回写仿真状态" -- and for a soft body the rule has
 * more teeth. A particle's state is a row in a buffer, so the worst a careless
 * renderer can do is perturb one body. A soft body's state is a *graph*: writing
 * one node moves every edge that touches it, and the next solver iteration then
 * pulls on neighbours that were never drawn. A renderer that wrote into
 * `mesh.data` would not merely be inaccurate, it would inject energy into the
 * simulation at frame rate, and the parity gate the plan asks for would compare
 * two runs that differ by how often they were displayed. So `update()` reads and
 * draws, and `tests/render_soft.test.ts` pins that by digest.
 *
 * # Why a mesh and not instanced spheres
 *
 * `render/particles.ts` draws one `InstancedMesh`, because there each body is
 * independent and its own shape. Here the interesting thing is the surface the
 * nodes span, and drawing it is cheaper than drawing the nodes: a 10k-node cloth
 * is 20k triangles, which is one draw call over 10k vertices, where 10k instanced
 * spheres at even 20 triangles each is 200k. It also reads better -- a cloth you
 * can see drape is the whole point of the milestone -- and it is the only option
 * that keeps the CPU/GPU comparison honest, since both tiers draw the identical
 * geometry over identical positions.
 *
 * # Which attribute the position lives in, and why it decides everything
 *
 * A plain `THREE.BufferAttribute` with `itemSize: 3`, and that choice is load
 * bearing in three ways:
 *
 * - *Not a `StorageBufferAttribute`.* three pads a storage attribute's
 *   `itemSize` 3 to 4, because WGSL cannot pack a `vec3<f32>` in a storage
 *   buffer. The padding would break the byte-for-byte match with the publish
 *   buffer and force an expansion kernel of exactly the kind
 *   `gpu/particleInstances.ts` exists for -- a shader, a pipeline and a second
 *   buffer, all to re-space numbers that were already correctly spaced.
 * - *A vertex attribute is created with `VERTEX | COPY_SRC | COPY_DST`* by
 *   three's WebGPU backend (`createAttribute`), so `copyBufferToBuffer` into it
 *   is legal without asking three for anything unusual. That usage mask is the
 *   entire reason the blit path exists here at all.
 * - *Not `DynamicDrawUsage`.* three re-uploads a dynamic attribute every frame,
 *   which would overwrite blitted positions with the stale CPU array -- on the
 *   first blit, with zeros. The CPU path sets `needsUpdate` itself when it
 *   writes, so it still uploads, including the one pass the constructor makes.
 *
 * The publish buffer is tight xyz, three floats a node, so it and the attribute
 * are the same size and the same layout. The copy is one `copyBufferToBuffer`
 * with no stride, no offset and nothing to expand: `SOFT_POSITION_FLOATS` is
 * `SOFT_PUBLISH_FLOATS_PER_NODE` rather than a second `3` so that a change to the
 * kernel's publish layout resizes the attribute with it.
 *
 * # Two draw modes over one attribute
 *
 * `'surface'` is a `THREE.Mesh` indexed by `mesh.triangles`; `'edges'` is a
 * `THREE.LineSegments` indexed by `mesh.constraints.ends`, which is already laid
 * out as `[a0, b0, a1, b1, ...]` and so *is* a line index -- no rebuild, and no
 * second copy of the graph to fall out of step with the solver's. Which one is
 * the default is a property of the mesh, not of the caller: `cloth` and `sheets`
 * carry triangles, `cube` and `rope` do not.
 *
 * The optional `wireframe` overlay is a third drawable that shares the *same*
 * attribute object, so three creates one `GPUBuffer` for all of them and one
 * blit fills the surface and its wire at once. Sharing rather than duplicating is
 * what keeps the overlay exactly on the surface: a second attribute would be a
 * second copy to fill, and a frame that filled one and not the other would draw
 * lines floating off the cloth. In `'edges'` mode there is nothing to overlay, so
 * `wire` is the line view itself.
 *
 * # Two fill paths, one drawable set
 *
 * - `gpu-blit`: `system.copyPublishedTo(buffer)` copies the publish buffer into
 *   the `GPUBuffer` three already made for the position attribute. No node data
 *   crosses the bus and the CPU never sees positions.
 * - `cpu-upload`: positions are written into `position.array` here and three
 *   uploads them. This is the WebGL2 and CPU tiers, and the WebGPU tier's first
 *   frame and failure path.
 *
 * Both write the same attribute of the same geometries, so the tier can change at
 * runtime without rebuilding anything and a fallback looks like the thing it is
 * standing in for.
 *
 * # No async attach, and the diagnostic that replaces it
 *
 * Unlike the particle view there is nothing to compile: the publish buffer
 * already exists when the system does, so `update()` blits the first frame it
 * finds a destination for. What can still go wrong is three.js not exposing one,
 * which is not a device condition but this module's assumption about three being
 * wrong -- the silent form of the failure, and the expensive one to find. So the
 * second consecutive `update()` with a GPU system and no reachable buffer records
 * a `gpuError`, and clears it if the buffer turns up later: a slow first render
 * is not a broken assumption, and a diagnostic that outlives its cause would pin
 * the view to the CPU path on a page where the blit works.
 *
 * A device that fails mid-run is handled the same way the particle view handles
 * one: the frame falls back to a CPU upload, the reason lands in `gpuError`, and
 * `update()` never throws. A page that went blank because a *renderer* could not
 * reach a buffer would be exactly the failure the capability matrix exists to
 * prevent.
 */

// `three/webgpu`, not `three`: the renderer this view blits into is a
// `WebGPURenderer`, and its WebGL2 fallback is a *backend* of the same renderer
// rather than a different library. Both builds load the same `three.core.js`, so
// the scene-graph classes are the identical objects either way and a page may
// import `three` elsewhere -- as `src/render/scene.ts` does -- without ending up
// with two copies.
import * as THREE from 'three/webgpu';

import type { GpuBufferLike } from '../gpu/device.js';
import type { GpuSoftSystem } from '../gpu/softGpu.js';
import { SOFT_OFFSET, SOFT_STRIDE, type SoftMesh } from '../gpu/softMesh.js';
import type { SoftSystem } from '../gpu/softTypes.js';
import { SOFT_PUBLISH_FLOATS_PER_NODE } from '../gpu/softWgsl.js';
import { instanceBufferOf, type RendererLike } from './particles.js';

/**
 * Anything indexable that can hold three floats per node.
 *
 * Declared structurally for the reason `MatrixTarget` in `render/particles.ts`
 * is: `BufferAttribute.array` is three's `TypedArray` union, and a parameter
 * typed `Float32Array` would not accept it -- which would push every caller into
 * a cast and hide the one requirement that matters, that the writes are
 * indexable.
 */
export type PositionTarget = ArrayLike<number> & { [index: number]: number };

/** Which fill path a frame used. Reported, because a silent downgrade is a lie. */
export type SoftFrameMode = 'gpu-blit' | 'cpu-upload';

/** Which fill path the view is settled on. `cpu` until a blit has succeeded. */
export type SoftViewMode = 'gpu' | 'cpu';

/**
 * What the nodes are drawn as.
 *
 * `'surface'` needs `mesh.triangles`; `'edges'` needs `mesh.constraints`. Both
 * are properties of the mesh, so `defaultDrawMode` picks one and a caller only
 * names a mode to override it.
 */
export type SoftDrawMode = 'surface' | 'edges';

/** Floats per drawn node. The publish buffer's layout, which is tight xyz. */
export const SOFT_POSITION_FLOATS = SOFT_PUBLISH_FLOATS_PER_NODE;

/** Bytes per drawn node. The number a blit is sized by. */
export const SOFT_POSITION_BYTES = SOFT_POSITION_FLOATS * 4;

/** Default surface colour: lit, so folds in a draped cloth are visible. */
export const SOFT_SURFACE_COLOR = 0x79c9a6;

/** Default wire colour: darker than the surface it sits on, so lines read as lines. */
export const SOFT_WIRE_COLOR = 0x14202b;

/** The draw mode a mesh asks for, from what it carries. */
export function defaultDrawMode(mesh: SoftMesh): SoftDrawMode {
  return mesh.triangles.length > 0 ? 'surface' : 'edges';
}

/**
 * Write `count` tight xyz positions out of an interleaved node field.
 *
 * The CPU fallback for a system whose published positions are not reachable as a
 * `Float32Array` -- in practice a GPU system after its device failed, where
 * `mesh.data` holds whatever the last `readback()` left there. Stale is the right
 * trade for a fallback and the reason `gpuError` exists is that it must never be
 * silent.
 *
 * @returns how many nodes were written.
 */
export function writePositions(
  target: PositionTarget,
  data: ArrayLike<number>,
  count: number,
): number {
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError(`writePositions needs a non-negative integer count, got ${count}`);
  }
  if (target.length < count * SOFT_POSITION_FLOATS) {
    throw new RangeError(
      `writePositions: the target holds ${target.length} floats, ${count} nodes need ${count * SOFT_POSITION_FLOATS}`,
    );
  }
  if (data.length < count * SOFT_STRIDE) {
    throw new RangeError(
      `writePositions: the mesh holds ${data.length} floats, ${count} nodes need ${count * SOFT_STRIDE}`,
    );
  }
  for (let i = 0; i < count; i++) {
    const src = i * SOFT_STRIDE + SOFT_OFFSET.position;
    const dst = i * SOFT_POSITION_FLOATS;
    target[dst] = data[src];
    target[dst + 1] = data[src + 1];
    target[dst + 2] = data[src + 2];
  }
  return count;
}

/**
 * The `GPUBuffer` three.js owns for a vertex attribute, or `null`.
 *
 * The lookup is `render/particles.js`'s `instanceBufferOf`, which is generic
 * despite its name: `backend.get(attribute)` is keyed on the attribute object
 * whatever the geometry happens to call it. Re-exporting it under a name that
 * says what it is used for here keeps the call site honest, and reusing the body
 * is what stops this module growing a second copy of the "is this renderer
 * WebGPU" test -- the two would eventually disagree, and the one that disagreed
 * would be the one nobody looked at.
 */
export function positionBufferOf(
  renderer: RendererLike | null | undefined,
  attribute: unknown,
): GpuBufferLike | null {
  return instanceBufferOf(renderer, attribute);
}

/**
 * The position attribute every drawable in a view shares.
 *
 * See the file header for why this is a plain vertex attribute and not a storage
 * one. The array is allocated at exactly `count * SOFT_POSITION_BYTES`, which is
 * what makes the blit a whole-buffer copy with no clamp to reason about.
 */
function positionAttribute(count: number): THREE.BufferAttribute {
  return new THREE.BufferAttribute(
    new Float32Array(count * SOFT_POSITION_FLOATS),
    SOFT_POSITION_FLOATS,
  );
}

/**
 * Wrap an index array in an attribute.
 *
 * `BufferGeometry.setIndex` only builds an attribute for a plain `Array`; handed
 * a `Uint32Array` it stores the typed array itself as `geometry.index`, which no
 * renderer accepts. Building the attribute here and passing *that* to `setIndex`
 * takes the branch that stores it as given, and keeps the mesh's `Uint32Array`
 * rather than copying 80k indices into a fresh one.
 */
function indexAttribute(indices: Uint32Array): THREE.BufferAttribute {
  return new THREE.BufferAttribute(indices, 1);
}

/** A geometry over the shared position attribute, indexed by `indices`. */
function indexedGeometry(
  position: THREE.BufferAttribute,
  indices: Uint32Array,
  what: string,
): THREE.BufferGeometry {
  if (indices.length === 0) {
    throw new RangeError(
      `this mesh has no ${what}, so there is nothing to draw; build the mesh with a graph or with triangles`,
    );
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', position);
  geometry.setIndex(indexAttribute(indices));
  return geometry;
}

/** Whether a system is the GPU backend, without importing its class. */
function asGpuSystem(system: SoftSystem): GpuSoftSystem | null {
  const candidate = system as Partial<GpuSoftSystem>;
  if (system.name !== 'gpu') return null;
  if (!candidate.publish || typeof candidate.copyPublishedTo !== 'function') return null;
  if (!candidate.shared) return null;
  return system as GpuSoftSystem;
}

/** The CPU backend's tight-xyz blit, found structurally. */
interface CpuPublisher {
  copyPublishedTo(target: Float32Array): number;
}

/**
 * The system's `copyPublishedTo`, when it takes a `Float32Array`.
 *
 * Both backends have a method of that name and they are not interchangeable: the
 * GPU one takes a `GPUBuffer` and submits a device copy, so handing it the
 * attribute's CPU array would be a type error the runtime would happily swallow.
 * `gpu === null` is the discriminator, and it is checked first for exactly that
 * reason.
 */
function asCpuPublisher(system: SoftSystem, gpu: GpuSoftSystem | null): CpuPublisher | null {
  if (gpu !== null) return null;
  const candidate = system as Partial<CpuPublisher>;
  const copy = candidate.copyPublishedTo;
  // Bound into a fresh object rather than the system cast to `CpuPublisher`:
  // `SoftSystem` does not declare the method, so that cast would be unsound, and
  // a structural probe that lies about types is how the GPU system's
  // buffer-taking method of the same name would end up being called with an array.
  if (typeof copy !== 'function') return null;
  return { copyPublishedTo: (target: Float32Array): number => copy.call(system, target) };
}

/**
 * How many consecutive frames a GPU-backed view may go without seeing a
 * destination buffer before that stops being "three.js has not rendered yet".
 *
 * Two, for the reason the particle view gives: the buffer is created during the
 * first render and `update()` runs before it, so frame one legitimately sees
 * nothing and frame two does not.
 */
const STUCK_WITHOUT_BUFFER_FRAMES = 2;

/** The `gpuError` recorded when the buffer never shows up. Compared by identity. */
const NO_BUFFER_ERROR =
  `three.js exposed no GPUBuffer for the position attribute in ${STUCK_WITHOUT_BUFFER_FRAMES} frames; ` +
  'the blit path cannot attach and the view is uploading from the CPU';

/** The `gpuError` recorded when a buffer that was working stops being reachable. */
const LOST_BUFFER_ERROR =
  'the GPUBuffer behind the position attribute disappeared after a successful blit; ' +
  'the view is uploading from the CPU';

// ---------------------------------------------------------------------------
// the view
// ---------------------------------------------------------------------------

export interface SoftViewOptions {
  readonly system: SoftSystem;
  /**
   * The renderer to blit into. Omit it and the view is CPU-upload only, which is
   * what a headless spec and a WebGL2 page both want.
   */
  readonly renderer?: RendererLike | null;
  /** Defaults to what the mesh carries: a surface if it has triangles. */
  readonly drawMode?: SoftDrawMode;
  /**
   * Draw the constraint graph as lines. Over a surface this is an overlay
   * sharing the same position attribute; in `'edges'` mode the line view already
   * is the constraint graph, so `wire` names that view instead and nothing is
   * duplicated.
   */
  readonly wireframe?: boolean;
  readonly color?: number;
  readonly wireColor?: number;
  /** Injected materials are not disposed by the view. */
  readonly material?: THREE.Material;
  readonly wireMaterial?: THREE.Material;
}

/**
 * A scene-graph object kept in step with a soft-body system.
 *
 * `object` is the only thing a caller has to add to a scene; everything else is
 * `update()` once a frame, after the simulation has stepped and before
 * `renderer.render()`.
 */
export class SoftView {
  readonly system: SoftSystem;
  /** The GPU backend behind `system`, or `null` for the CPU and WebGL2 tiers. */
  readonly gpu: GpuSoftSystem | null;
  /** The root to add to a scene. Holds every drawable the view owns. */
  readonly object: THREE.Group;
  readonly drawMode: SoftDrawMode;
  /**
   * The one position attribute, shared by every drawable and the destination of
   * every fill path. Exposed because a caller that wants a different look -- a
   * custom material, a second view over the same simulation -- needs the object
   * three.js keyed a `GPUBuffer` on, not a copy of it.
   */
  readonly position: THREE.BufferAttribute;
  /** The surface, in `'surface'` mode only. */
  readonly surface: THREE.Mesh | null;
  /** The line view, in `'edges'` mode only. */
  readonly edges: THREE.LineSegments | null;
  /** The drawable that shows the constraint graph as lines. See `wireframe`. */
  readonly wire: THREE.LineSegments | null;

  private readonly renderer: RendererLike | null;
  private readonly publisher: CpuPublisher | null;
  private readonly ownedGeometries: THREE.BufferGeometry[] = [];
  private readonly ownedMaterials: THREE.Material[] = [];
  private gpuActive: boolean;
  /** A real device failure. Never retried, unlike a buffer that has not appeared. */
  private deviceFailed = false;
  private blitted = false;
  private gpuErrorText: string | null = null;
  private lastFrame: SoftFrameMode = 'cpu-upload';
  private lastBytes = 0;
  private framesWithoutBuffer = 0;
  private noBufferError = false;
  private disposedFlag = false;

  constructor(options: SoftViewOptions) {
    const system = options.system;
    if (!system || !system.mesh || !Number.isInteger(system.count) || system.count <= 0) {
      throw new TypeError('SoftView needs a soft-body system with a mesh and a positive count');
    }
    this.system = system;
    this.gpu = asGpuSystem(system);
    this.publisher = asCpuPublisher(system, this.gpu);
    this.renderer = options.renderer ?? null;
    this.gpuActive = this.gpu !== null;

    const mesh = system.mesh;
    const drawMode = options.drawMode ?? defaultDrawMode(mesh);
    if (drawMode !== 'surface' && drawMode !== 'edges') {
      throw new RangeError(`unknown draw mode "${drawMode}", expected 'surface' or 'edges'`);
    }
    if (drawMode === 'surface' && mesh.triangles.length === 0) {
      throw new RangeError(
        'this mesh has no triangles, so there is no surface to draw; ask for drawMode "edges"',
      );
    }
    this.drawMode = drawMode;

    const position = positionAttribute(system.count);
    this.position = position;
    this.object = new THREE.Group();
    this.object.name = 'soft-view';

    const drawables = this.buildDrawables(options, drawMode, position, mesh);
    this.surface = drawables.surface;
    this.edges = drawables.edges;
    this.wire = drawables.wire;

    // One CPU pass up front: it makes the first frame correct on every tier, and
    // it means a re-upload by three.js -- after a context restore, say -- carries
    // positions rather than the zeros the attribute was allocated with.
    this.uploadFromSystem();
  }

  /**
   * Build the drawables for `drawMode`, all over `position`.
   *
   * A method rather than constructor branches because the two modes share no
   * control flow, and a constructor with two `return` paths to reach one common
   * tail is a constructor whose tail is easy to skip.
   */
  private buildDrawables(
    options: SoftViewOptions,
    drawMode: SoftDrawMode,
    position: THREE.BufferAttribute,
    mesh: SoftMesh,
  ): {
    readonly surface: THREE.Mesh | null;
    readonly edges: THREE.LineSegments | null;
    readonly wire: THREE.LineSegments | null;
  } {
    const wantsWire = options.wireframe ?? false;

    if (drawMode === 'edges') {
      const lines = this.buildLines(position, mesh.constraints.ends, {
        material: options.wireMaterial,
        color: options.wireColor ?? options.color ?? SOFT_WIRE_COLOR,
        name: 'soft-edges',
      });
      // In this mode the view *is* the constraint graph, so a requested overlay
      // names it rather than duplicating it. `wire` is then never null when
      // `wireframe` was asked for, which is the only reading of the option that
      // does not silently drop it.
      return { surface: null, edges: lines, wire: wantsWire ? lines : null };
    }

    const geometry = indexedGeometry(position, mesh.triangles, 'triangles');
    this.ownedGeometries.push(geometry);
    const material =
      options.material ??
      new THREE.MeshLambertMaterial({
        color: options.color ?? SOFT_SURFACE_COLOR,
        // A cloth's cells fold through each other as it drapes, and the triangles
        // in `mesh.triangles` are wound for the mesh's initial orientation only.
        // Culling the backs would punch holes in the view at exactly the moment
        // it is most interesting.
        side: THREE.DoubleSide,
        // No `normal` attribute, on purpose: per-frame normals would be an
        // O(triangles) CPU pass, which is the one thing the blit path exists to
        // avoid. Flat shading derives them from screen-space derivatives instead,
        // on both backends of `WebGPURenderer`, so the cost is a `cross` in the
        // fragment shader and the look is identical across tiers.
        flatShading: true,
        // Pushes the surface back far enough that the coplanar wire overlay draws
        // on top of it instead of z-fighting with it.
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
    if (options.material === undefined) this.ownedMaterials.push(material);
    const surface = new THREE.Mesh(geometry, material);
    surface.frustumCulled = false;
    surface.name = 'soft-surface';
    this.object.add(surface);

    const wire = wantsWire
      ? this.buildLines(position, mesh.constraints.ends, {
          material: options.wireMaterial,
          color: options.wireColor ?? SOFT_WIRE_COLOR,
          name: 'soft-wire',
        })
      : null;
    return { surface, edges: null, wire };
  }

  /** A `LineSegments` over the shared position attribute, indexed by `ends`. */
  private buildLines(
    position: THREE.BufferAttribute,
    ends: Uint32Array,
    spec: { readonly material?: THREE.Material; readonly color: number; readonly name: string },
  ): THREE.LineSegments {
    const geometry = indexedGeometry(position, ends, 'constraints');
    this.ownedGeometries.push(geometry);
    const material = spec.material ?? new THREE.LineBasicMaterial({ color: spec.color });
    if (spec.material === undefined) this.ownedMaterials.push(material);
    const lines = new THREE.LineSegments(geometry, material);
    lines.frustumCulled = false;
    lines.name = spec.name;
    this.object.add(lines);
    return lines;
  }

  /**
   * The path in use: `gpu` once a blit has succeeded, `cpu` before that and after
   * any GPU failure. Derived rather than stored, so it cannot drift out of step
   * with the thing that actually decides it.
   */
  get viewMode(): SoftViewMode {
    return this.blitted ? 'gpu' : 'cpu';
  }

  /** What the last `update()` did. */
  get frameMode(): SoftFrameMode {
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

  /** True while a GPU system is still eligible to blit. */
  get ready(): boolean {
    return this.gpuActive;
  }

  get disposed(): boolean {
    return this.disposedFlag;
  }

  /** Nodes drawn. Fixed by the mesh, unlike an instance count a caller may lower. */
  get count(): number {
    return this.system.count;
  }

  /** Whether the constraint lines are showing. False when there is no overlay. */
  get wireVisible(): boolean {
    return this.wire !== null && this.wire.visible;
  }

  /** Show or hide the constraint lines. Does nothing when there is no overlay. */
  setWireVisible(visible: boolean): void {
    if (this.disposedFlag) throw new Error('SoftView has been disposed');
    if (this.wire !== null) this.wire.visible = visible;
  }

  /**
   * Bring the drawables up to date with the simulation. Call once a frame.
   *
   * Never throws for a GPU problem: the frame falls back to a CPU upload and the
   * reason lands in `gpuError`. It throws only for use after `dispose()`, which is
   * a caller bug and not a device condition.
   */
  update(): SoftFrameMode {
    if (this.disposedFlag) throw new Error('SoftView has been disposed');

    const target = this.targetBuffer();
    if (target === null) {
      this.noteMissingBuffer();
    } else {
      this.noteBufferFound();
    }

    if (this.gpu !== null && this.gpuActive && target !== null) {
      try {
        // Capped by both what is drawn and what the destination holds. The buffer
        // belongs to three.js, so its size is a fact about the renderer and not
        // about `count`; a copy past the end is a validation error whose message
        // contains no numbers, and it would arrive at submit time.
        const fits = Math.min(target.size, this.count * SOFT_POSITION_BYTES);
        // Floored to whole nodes. A truncated one would leave its last component
        // from the previous frame, which is a *wrong* position rather than a
        // stale one, and a wrong position is the kind of artefact that gets
        // reported as a solver bug.
        const capacity = fits - (fits % SOFT_POSITION_BYTES);
        const bytes = this.gpu.copyPublishedTo(target, capacity);
        if (bytes > 0) {
          this.blitted = true;
          this.lastFrame = 'gpu-blit';
          this.lastBytes = bytes;
          // Deliberately no `needsUpdate`: the buffer now holds positions three.js
          // did not write, and asking it to upload the CPU array would overwrite
          // them with the last frame's -- or, on the first blit, with zeros.
          return 'gpu-blit';
        }
      } catch (error) {
        this.failGpu(error);
      }
    }

    this.uploadFromSystem();
    return 'cpu-upload';
  }

  /** Release the drawables this view owns. Idempotent. */
  dispose(): void {
    if (this.disposedFlag) return;
    this.disposedFlag = true;
    // Both geometries can share the position attribute. That is safe: three's
    // backend keys the buffer on the attribute and its `delete()` guards on
    // whether any data was there, so the second geometry's teardown is a no-op
    // rather than a double free.
    for (const geometry of this.ownedGeometries) geometry.dispose();
    this.ownedGeometries.length = 0;
    for (const material of this.ownedMaterials) material.dispose();
    this.ownedMaterials.length = 0;
    this.object.removeFromParent();
    this.object.clear();
  }

  /** The `GPUBuffer` three.js owns for the position attribute, or `null`. */
  private targetBuffer(): GpuBufferLike | null {
    if (this.gpu === null) return null;
    return positionBufferOf(this.renderer, this.position);
  }

  /**
   * Count frames a GPU-backed view had nowhere to blit.
   *
   * Only before the first successful blit: once positions are reaching the buffer,
   * a missing one is a lost device and `copyPublishedTo` reports it. The failure
   * this catches is quieter -- a view that uploads from the CPU forever, with
   * `frameMode` saying `cpu-upload` and `gpuError` saying nothing, because
   * three.js keeps the buffer somewhere this module does not look.
   */
  private noteMissingBuffer(): void {
    if (this.gpu === null || this.renderer === null) return;
    // A view that has blitted, or that failed for a reason worth more than this
    // one, has nothing left to learn from a missing buffer.
    if (this.deviceFailed || this.noBufferError) return;
    if (this.gpuErrorText !== null) return;
    if (this.blitted) {
      // Positions *were* reaching this buffer, and now three.js has none to hand
      // back. That is a destroyed attribute or a lost device, not a slow first
      // render, and a view that quietly went on uploading from the CPU is the
      // exact silent downgrade this module exists to prevent.
      this.failGpu(LOST_BUFFER_ERROR);
      return;
    }
    this.framesWithoutBuffer++;
    if (this.framesWithoutBuffer < STUCK_WITHOUT_BUFFER_FRAMES) return;
    this.noBufferError = true;
    // Not `failGpu`: this verdict is withdrawable, and marking the device failed
    // would make the withdrawal below impossible.
    this.gpuActive = false;
    this.gpuErrorText = NO_BUFFER_ERROR;
  }

  /**
   * Undo the above when the buffer turns up after all.
   *
   * A slow first render -- a shader compile, a tab that was backgrounded -- is not
   * a broken assumption, and a diagnostic that outlives its cause is worse than
   * none: it would report a CPU path on a page where the blit works.
   */
  private noteBufferFound(): void {
    this.framesWithoutBuffer = 0;
    if (!this.noBufferError) return;
    this.noBufferError = false;
    // Retrying is free here, unlike the particle view: there is no pipeline to
    // rebuild, so re-arming just means the next frame attempts the copy again.
    if (!this.deviceFailed) this.gpuActive = this.gpu !== null;
    if (this.gpuErrorText === NO_BUFFER_ERROR) this.gpuErrorText = null;
  }

  private failGpu(error: unknown): void {
    this.deviceFailed = true;
    this.gpuActive = false;
    this.blitted = false;
    this.lastFrame = 'cpu-upload';
    this.lastBytes = 0;
    this.gpuErrorText = messageOf(error);
  }

  /**
   * Write positions into the attribute's CPU array and ask three to upload them.
   *
   * Prefers the system's own published positions, which the CPU backend keeps as
   * tight xyz; falls back to the interleaved field, which on a failed GPU system
   * is the last `readback()`. Both are stale by at most one frame relative to the
   * device, and both are better than a blank mesh.
   */
  private uploadFromSystem(): void {
    const array = this.position.array as Float32Array;
    if (this.publisher !== null) {
      this.publisher.copyPublishedTo(array);
    } else {
      writePositions(array, this.system.mesh.data, this.count);
    }
    this.position.needsUpdate = true;
    this.lastFrame = 'cpu-upload';
    this.lastBytes = 0;
  }
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

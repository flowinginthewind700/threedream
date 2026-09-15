/**
 * The compute kernel that turns published particles into instance matrices.
 *
 * # Why a kernel and not a readback
 *
 * The renderer draws `count` spheres with one `InstancedMesh`, whose per-instance
 * transform is a 64-byte `mat4`. Filling those from the CPU means reading the
 * simulation back over the PCIe bus every frame and writing 64 bytes per
 * particle into an upload buffer: at 100k particles that is 6.4 MB each way per
 * frame, and it is exactly the traffic M3 exists to remove. This module keeps the
 * data on the device -- the published positions never reach the CPU -- and the
 * only per-frame CPU work is a `dispatchWorkgroups` and a `copyBufferToBuffer`.
 *
 * # Why an intermediate buffer at all
 *
 * three.js creates the `instanceMatrix` buffer with `VERTEX | COPY_DST`. It
 * cannot be bound as a storage buffer, so a kernel cannot write it directly.
 * What it *can* be is the destination of a copy, which is why the kernel writes
 * `instanceBuf` here and `expandTo()` then blits that into the renderer's own
 * buffer. The extra hop costs one device-local copy, which stays on the GPU and
 * is the cheapest thing in the frame.
 *
 * # Layout
 *
 * Both WGSL `mat4x4` and three.js `InstancedBufferAttribute` are column-major,
 * so the sixteen floats are the four columns in order. A sphere instance needs
 * no rotation, which makes the matrix scale-plus-translate and leaves twelve of
 * the sixteen floats constant zero -- written once on the CPU side by
 * `render/particles.ts`, and written in full here because a kernel that skips
 * them would have to know the CPU had not already overwritten the buffer.
 *
 * As everywhere else in `src/gpu`, nothing here imports three.js: this module
 * hands back bytes in a `ComputeBuffer` and the render layer decides what to do
 * with them. That is also what keeps it testable against `tests/stub_webgpu.ts`
 * in bare Node.
 */

import {
  ComputeContext,
  rawBuffer,
  submitCopy,
  type ComputeBinding,
  type ComputeBuffer,
  type ComputeProgram,
  type ComputeResource,
} from './compute.js';
import type { SharedDevice } from './device.js';
import { PUBLISH_FLOATS_PER_PARTICLE, WORKGROUP_SIZE, workgroupsFor } from './particleWgsl.js';

/** Floats in one `mat4`, i.e. one instance's worth of `instanceMatrix`. */
export const INSTANCE_FLOATS = 16;

/** Bytes in one instance matrix. The number the copy is sized by. */
export const INSTANCE_BYTES = INSTANCE_FLOATS * 4;

/**
 * `vec4`s per instance in the expansion buffer.
 *
 * Four rather than sixteen: the matrix is written as its columns, and a column is
 * a `vec4`, so the buffer is `array<vec4<f32>>` like every other buffer in the
 * particle pipeline and needs no second layout rule.
 */
export const INSTANCE_VECS_PER_INSTANCE = INSTANCE_FLOATS / 4;

/**
 * `vec4`s per particle in the buffer this kernel reads: one, `(pos.xyz, radius)`.
 *
 * Derived from the publisher's own stride rather than restated, because the two
 * are the same fact seen from two sides: a `publish` kernel that stopped writing
 * the radius would otherwise leave this one reading a field nobody fills.
 */
export const PUBLISH_VECS_PER_PARTICLE = PUBLISH_FLOATS_PER_PARTICLE / 4;

/**
 * Words in the expansion uniform, in order.
 *
 * Two live values and two pads, because a uniform struct's size is rounded to its
 * largest member's alignment and a 16-byte struct is the smallest one that holds
 * a `u32` and an `f32` without a surprise at the end. The pads are declared
 * rather than left implicit so the WGSL and the writer cannot disagree about
 * where `radiusScale` is.
 */
export const INSTANCE_WORD = {
  count: 0,
  radiusScale: 1,
  padA: 2,
  padB: 3,
} as const;

export const INSTANCE_PARAMS_WORDS = 4;

export const INSTANCE_PARAMS_BYTES = INSTANCE_PARAMS_WORDS * 4;

/**
 * Multiplies every published radius on the way into the matrix.
 *
 * A render-side knob, not a simulation one: widening it makes a sparse field
 * readable without touching positions, which keeps the digest -- and therefore
 * any parity check against the CPU backend -- independent of how big the spheres
 * are drawn.
 */
export const DEFAULT_RADIUS_SCALE = 1;

/** One bind group, since nothing here swaps between steps. */
export const INSTANCE_GROUP = 0;

export const INSTANCE_BINDINGS: readonly ComputeBinding[] = [
  {
    group: INSTANCE_GROUP,
    binding: 0,
    name: 'publishBuf',
    bufferType: 'read-only-storage',
  },
  {
    group: INSTANCE_GROUP,
    binding: 1,
    name: 'instanceBuf',
    bufferType: 'storage',
  },
  {
    group: INSTANCE_GROUP,
    binding: 2,
    name: 'params',
    bufferType: 'uniform',
  },
];

/** Entry point. One kernel, one dispatch per frame. */
export const INSTANCE_ENTRY = 'expand';

/** Bytes the expansion buffer needs for `count` instances. */
export function instanceBufferBytes(count: number): number {
  checkedCount(count);
  return count * INSTANCE_BYTES;
}

/**
 * Bytes to copy for `count` instances, capped at what a target buffer holds and
 * floored to whole instances.
 *
 * Two rules, both because the destination is not ours. A copy larger than the
 * buffer is a validation error whose message contains no numbers, and a copy
 * whose size is not a multiple of 4 is rejected by every implementation for the
 * same reason; a partial matrix is not something a renderer can draw either. So
 * the answer is always a whole number of instances that fits.
 */
export function instanceCopyBytes(count: number, targetBytes: number): number {
  const wanted = instanceBufferBytes(count);
  if (!Number.isFinite(targetBytes) || targetBytes < 0) {
    throw new RangeError(`instance copy: target size ${targetBytes} is not a size`);
  }
  return Math.floor(Math.min(wanted, targetBytes) / INSTANCE_BYTES) * INSTANCE_BYTES;
}

function checkedCount(count: number): number {
  if (!Number.isInteger(count) || count <= 0) {
    throw new RangeError(`instance expansion needs a positive integer count, got ${count}`);
  }
  return count;
}

let cachedSource: string | null = null;

/**
 * The expansion shader.
 *
 * Generated rather than a `.wgsl` file for the reason given at the top of
 * `particleWgsl.ts`: every number in it -- the workgroup size, the vec4s per
 * instance, the publish stride, the word indices -- is interpolated from the
 * TypeScript constant that already owns it. Memoised because the text is constant
 * for the lifetime of the process.
 */
export function instanceShaderSource(): string {
  if (cachedSource !== null) return cachedSource;
  cachedSource = `// Generated by src/gpu/particleInstances.ts -- edit the generator, not this text.

struct InstanceParams {
  count: u32,
  radiusScale: f32,
  padA: f32,
  padB: f32,
};

// ${PUBLISH_VECS_PER_PARTICLE} vec4 per particle, i.e. (pos.xyz, radius).
@group(${INSTANCE_GROUP}) @binding(0) var<storage, read> publishBuf: array<vec4<f32>>;
@group(${INSTANCE_GROUP}) @binding(1) var<storage, read_write> instanceBuf: array<vec4<f32>>;
@group(${INSTANCE_GROUP}) @binding(2) var<uniform> params: InstanceParams;

// One invocation per particle: read the published (position, radius) pair and
// write the scale-plus-translate matrix three.js will instance with. Column
// major, matching InstancedBufferAttribute, so no transpose anywhere.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn ${INSTANCE_ENTRY}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let published = publishBuf[i];
  let r = published.w * params.radiusScale;
  let base = i * ${INSTANCE_VECS_PER_INSTANCE}u;
  instanceBuf[base] = vec4<f32>(r, 0.0, 0.0, 0.0);
  instanceBuf[base + 1u] = vec4<f32>(0.0, r, 0.0, 0.0);
  instanceBuf[base + 2u] = vec4<f32>(0.0, 0.0, r, 0.0);
  instanceBuf[base + 3u] = vec4<f32>(published.xyz, 1.0);
}
`;
  return cachedSource;
}

/**
 * Pack the uniform. Word indices come from `INSTANCE_WORD`, so moving a member in
 * the struct without moving it here is a change in one place that the WGSL text
 * and this writer both read.
 */
export function writeInstanceParams(
  target: Float32Array,
  count: number,
  radiusScale: number,
): Float32Array {
  if (target.length < INSTANCE_PARAMS_WORDS) {
    throw new RangeError(
      `instance params need ${INSTANCE_PARAMS_WORDS} words, the target holds ${target.length}`,
    );
  }
  checkedCount(count);
  if (!Number.isFinite(radiusScale) || !(radiusScale > 0)) {
    throw new RangeError(`radiusScale must be finite and positive, got ${radiusScale}`);
  }
  const words = new Uint32Array(target.buffer, target.byteOffset, INSTANCE_PARAMS_WORDS);
  words[INSTANCE_WORD.count] = count;
  target[INSTANCE_WORD.radiusScale] = radiusScale;
  target[INSTANCE_WORD.padA] = 0;
  target[INSTANCE_WORD.padB] = 0;
  return target;
}

export interface InstanceExpanderOptions {
  readonly shared: SharedDevice;
  readonly count: number;
  /** The publish buffer to read, i.e. `GpuParticleSystem.publish`. */
  readonly source: ComputeResource;
  readonly radiusScale?: number;
  readonly label?: string;
}

/**
 * One frame's worth of instance matrices, expanded on the device.
 *
 * Holds its own `ComputeContext` and its own reference to the shared device, so
 * `dispose()` cannot leave a buffer behind on a page that then falls back to the
 * CPU renderer. It does not own the publish buffer it reads: that belongs to the
 * simulation, and an expander that destroyed it would take the system with it.
 */
export class InstanceExpander {
  readonly count: number;
  readonly bytes: number;
  readonly label: string;
  /** The expansion buffer. Bound as `instanceBuf`, copied out by `expandTo`. */
  readonly instances: ComputeBuffer;

  private readonly shared: SharedDevice;
  private readonly context: ComputeContext;
  private readonly program: ComputeProgram;
  private readonly group: unknown;
  private readonly source: ComputeResource;
  private readonly paramsBytes: Float32Array;
  private readonly params: ComputeBuffer;
  private scale: number;
  private disposedFlag = false;

  /**
   * Compile the kernel and allocate.
   *
   * @throws {ShaderCompilationError} when the module reports an error.
   * @throws {RangeError} when the expansion buffer will not bind on this device.
   */
  static async create(options: InstanceExpanderOptions): Promise<InstanceExpander> {
    if (!options.shared.usable) {
      throw new Error(
        `the shared device is ${options.shared.lost ? 'lost' : 'destroyed'}`,
      );
    }
    const count = checkedCount(options.count);
    const bytes = count * INSTANCE_BYTES;
    assertBindable(options.shared, bytes, options.label ?? 'particle-instances');
    const context = new ComputeContext(options.shared);
    try {
      const program = await context.program({
        label: options.label ?? 'particle-instances',
        code: instanceShaderSource(),
        entryPoints: [INSTANCE_ENTRY],
        bindings: INSTANCE_BINDINGS,
      });
      return new InstanceExpander(options, context, program, count, bytes);
    } catch (error) {
      context.destroy();
      throw error;
    }
  }

  private constructor(
    options: InstanceExpanderOptions,
    context: ComputeContext,
    program: ComputeProgram,
    count: number,
    bytes: number,
  ) {
    this.shared = options.shared.retain();
    this.context = context;
    this.program = program;
    this.count = count;
    this.bytes = bytes;
    this.label = options.label ?? 'particle-instances';
    this.source = options.source;
    this.scale = options.radiusScale ?? DEFAULT_RADIUS_SCALE;
    this.paramsBytes = new Float32Array(INSTANCE_PARAMS_WORDS);

    this.params = context.uniformBuffer(INSTANCE_PARAMS_BYTES, `${this.label}:params`);
    this.instances = context.storageBuffer(bytes, this.label);
    writeInstanceParams(this.paramsBytes, count, this.scale);
    this.params.write(this.paramsBytes);
    this.group = program.bindGroup(
      INSTANCE_GROUP,
      { 0: this.source, 1: this.instances, 2: this.params },
      `${this.label}:group`,
    );
  }

  /** The scale every radius is multiplied by. */
  get radiusScale(): number {
    return this.scale;
  }

  get workgroups(): number {
    return workgroupsFor(this.count);
  }

  get disposed(): boolean {
    return this.disposedFlag;
  }

  /** True once the device behind this expander is gone. */
  get lost(): boolean {
    return !this.shared.usable;
  }

  /** Re-upload the uniform. Cheap, and the only way the scale changes. */
  setRadiusScale(radiusScale: number): void {
    this.assertLive();
    writeInstanceParams(this.paramsBytes, this.count, radiusScale);
    this.params.write(this.paramsBytes);
    this.scale = radiusScale;
  }

  /**
   * Expand into this expander's own buffer, without copying anywhere.
   *
   * @returns dispatches issued: 0 for an empty expansion, 1 otherwise.
   */
  expand(label?: string): number {
    this.assertLive();
    return this.program.submit(
      [{ entryPoint: INSTANCE_ENTRY, workgroups: this.workgroups, groups: { 0: this.group } }],
      label ?? `${this.label} expand`,
    );
  }

  /**
   * Expand, then blit the matrices into a buffer the caller owns.
   *
   * The destination is normally the `GPUBuffer` three.js created for
   * `instanceMatrix`, reached through `renderer.backend.get(attribute).buffer`.
   * Two submits rather than one because the copy's destination is not ours to
   * record a pass around, and the queue orders them either way.
   *
   * @returns bytes copied, or 0 when the target cannot hold one instance -- in
   *   which case nothing is dispatched either, because a frame that cannot be
   *   presented should not cost a kernel.
   */
  expandTo(target: ComputeResource, targetBytes = this.bytes): number {
    this.assertLive();
    const size = instanceCopyBytes(this.count, Math.min(targetBytes, this.bytes));
    if (size <= 0) return 0;
    this.expand();
    const copies = submitCopy(
      this.shared.device,
      [{ from: this.instances, to: rawBuffer(target), bytes: size }],
      `${this.label} blit`,
    );
    return copies > 0 ? size : 0;
  }

  /** Release the buffers and this expander's reference to the shared device. */
  dispose(): void {
    if (this.disposedFlag) return;
    this.disposedFlag = true;
    this.context.destroy();
    this.shared.release();
  }

  private assertLive(): void {
    if (this.disposedFlag) throw new Error('InstanceExpander has been disposed');
    if (!this.shared.usable) {
      throw new Error(
        `the shared device is ${this.shared.lost ? 'lost' : 'destroyed'}`,
      );
    }
  }
}

/**
 * Refuse an expansion that will not bind.
 *
 * Checked before allocation for the reason `particleGpu.ts` gives: the
 * alternative is a `createBuffer` that succeeds and a bind group that fails
 * validation later, on a page that has already built everything. A limit reported
 * as 0 was never reported, and guessing a ceiling there would reject devices that
 * could have run the expansion.
 */
function assertBindable(shared: SharedDevice, bytes: number, label: string): void {
  const { maxStorageBufferBindingSize, maxBufferSize } = shared.info.limits;
  const cap =
    maxBufferSize > 0
      ? Math.min(maxStorageBufferBindingSize, maxBufferSize)
      : maxStorageBufferBindingSize;
  if (cap > 0 && bytes > cap) {
    throw new RangeError(
      `${label}: ${bytes} bytes of instance matrices, above this device's ${cap}-byte storage limit`,
    );
  }
}

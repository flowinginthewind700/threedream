/**
 * `render/soft.ts` -- the soft-body renderer, driven headless.
 *
 * three.js runs fine in bare Node as long as nothing asks it for a WebGL
 * context: geometries, materials, `Mesh`, `LineSegments` and `Group` are plain
 * data, which is all this module builds. So the spec puts a real `SoftView`
 * around a real `GpuSoftSystem` running on the recording stub, with a fake
 * renderer in front of it -- one object whose `backend.get()` hands back a
 * `GPUBuffer` the way three.js's WebGPU backend does once it has created the
 * attribute. That is enough to drive both fill paths end to end.
 *
 * What is asserted here, and what is not:
 *
 * - `writePositions` is arithmetic, so it is checked numerically against the
 *   node field layout. It is the CPU-side twin of the publish kernel, and
 *   `e2e/soft_gpu.spec.ts` compares the two on a live device.
 * - the blit path is checked by *moving bytes*: the stub applies a copy when the
 *   encoder is sealed, so filling the publish buffer with a pattern and then
 *   calling `update()` proves the positions land in the buffer three.js owns.
 *   The stub cannot run WGSL, so it cannot prove the kernel wrote the right
 *   numbers into that buffer -- only that the copy is sized and aimed correctly.
 * - sharing one position attribute across drawables is checked by identity, not
 *   by value. Two attributes holding equal numbers would pass every value test
 *   and still draw a wire overlay floating off the cloth on the first frame that
 *   filled one and not the other.
 * - the plan's "不回写仿真状态" is checked by digest *and* by raw bytes, because
 *   a renderer that wrote into `mesh.data` would inject energy at frame rate and
 *   the parity gate would then compare two runs that differ by how often they
 *   were displayed.
 */

import { describe, expect, it } from 'vitest';

import * as THREE from 'three';

import { SharedDeviceManager, type SharedDevice } from '../src/gpu/device.js';
import { createCpuSoftSystem, type CpuSoftSystem } from '../src/gpu/softCpu.js';
import { createGpuSoftSystem, type GpuSoftSystem } from '../src/gpu/softGpu.js';
import {
  SOFT_OFFSET,
  SOFT_STRIDE,
  SoftMesh,
  emptyConstraints,
} from '../src/gpu/softMesh.js';
import type { SoftSystem } from '../src/gpu/softTypes.js';
import { SOFT_PUBLISH_FLOATS_PER_NODE } from '../src/gpu/softWgsl.js';
import {
  SOFT_POSITION_BYTES,
  SOFT_POSITION_FLOATS,
  SOFT_SURFACE_COLOR,
  SOFT_WIRE_COLOR,
  SoftView,
  defaultDrawMode,
  positionBufferOf,
  writePositions,
  type PositionTarget,
} from '../src/render/soft.js';
import {
  type AttributeHandle,
  type RendererBackendLike,
  type RendererLike,
} from '../src/render/particles.js';
import {
  STUB_CONSTANTS as CONSTANTS,
  StubBuffer,
  StubDevice,
  flushMicrotasks,
  stubLimits,
} from './stub_webgpu.js';

const U = CONSTANTS.bufferUsage;

/**
 * The stub reports the WebGPU baseline of 8 storage buffers, which the soft-body
 * pipeline cannot bind. Every GPU build here overrides it.
 */
function softLimits(over: Record<string, number> = {}): Record<string, number> {
  return stubLimits({ maxStorageBuffersPerShaderStage: 16, ...over });
}

// ---------------------------------------------------------------------------
// rig
// ---------------------------------------------------------------------------

/**
 * The slice of `WebGPURenderer` the view reads, faked.
 *
 * `asked` records which attribute the view inquired about, because the view has
 * to ask for *its own* position attribute and nothing else -- a renderer that
 * asked for the wrong one would blit into some other mesh's buffer and still
 * look correct from here. `handle` is mutable so a spec can withdraw the buffer
 * mid-run, which is what a destroyed attribute looks like from this side.
 */
class FakeRenderer implements RendererLike {
  readonly asked: unknown[] = [];
  readonly backend: RendererBackendLike;
  handle: AttributeHandle | undefined;

  constructor(handle?: AttributeHandle) {
    this.handle = handle;
    this.backend = {
      get: (attribute: unknown): AttributeHandle | undefined => {
        this.asked.push(attribute);
        return this.handle;
      },
    };
  }
}

/** A 10x10 cloth: has triangles, so it is the `'surface'` case. */
function cloth(count = 100): SoftMesh {
  // `speed: 2` matters -- a mesh seeded at rest is indistinguishable from one
  // the view never read.
  return new SoftMesh({ count, scene: 'cloth', seed: 7, speed: 2 });
}

/** A rope: no triangles, so it is the `'edges'` case. */
function rope(count = 32): SoftMesh {
  return new SoftMesh({ count, scene: 'rope', seed: 3, speed: 2 });
}

/** Free nodes with no graph at all: the mesh that has nothing to draw. */
function cloud(count = 8): SoftMesh {
  const data = new Float32Array(count * SOFT_STRIDE);
  for (let i = 0; i < count; i++) {
    const o = i * SOFT_STRIDE;
    data[o] = i - (count - 1) / 2;
    data[o + SOFT_OFFSET.invMass] = 1;
    data[o + SOFT_OFFSET.radius] = 0.1;
  }
  return new SoftMesh(data, { count, constraints: emptyConstraints() });
}

/** Raw node bytes with distinct positions and values that must *not* be drawn. */
function nodeData(count: number): Float32Array {
  const data = new Float32Array(count * SOFT_STRIDE);
  for (let i = 0; i < count; i++) {
    const o = i * SOFT_STRIDE;
    data[o + SOFT_OFFSET.position] = i + 1;
    data[o + SOFT_OFFSET.position + 1] = -(i + 1);
    data[o + SOFT_OFFSET.position + 2] = (i + 1) / 2;
    data[o + SOFT_OFFSET.velocity] = 1000 + i;
    data[o + SOFT_OFFSET.velocity + 1] = 2000 + i;
    data[o + SOFT_OFFSET.velocity + 2] = 3000 + i;
    data[o + SOFT_OFFSET.invMass] = 777;
    data[o + SOFT_OFFSET.radius] = 888;
  }
  return data;
}

interface GpuRig {
  readonly device: StubDevice;
  readonly manager: SharedDeviceManager;
  readonly shared: SharedDevice;
  readonly mesh: SoftMesh;
  readonly system: GpuSoftSystem;
  readonly target: StubBuffer;
  readonly renderer: FakeRenderer;
  readonly count: number;
}

/** A real GPU system on a stub device, plus the renderer-owned buffer to blit into. */
async function gpuRig(count = 100, targetBytes = count * SOFT_POSITION_BYTES): Promise<GpuRig> {
  const device = new StubDevice({ limits: softLimits() });
  const manager = new SharedDeviceManager({ constants: CONSTANTS });
  const shared = manager.adopt(device);
  const mesh = cloth(count);
  const system = await createGpuSoftSystem({ shared, mesh });
  // The usage mask three.js itself creates a vertex attribute with. Without
  // `COPY_DST` the blit this module exists for would be illegal.
  const target = device.createBuffer({
    label: 'three-position',
    size: targetBytes,
    usage: U.VERTEX | U.COPY_SRC | U.COPY_DST,
  });
  return {
    device,
    manager,
    shared,
    mesh,
    system,
    target,
    renderer: new FakeRenderer({ buffer: target }),
    count,
  };
}

/** The stub buffer behind `system.publish`, so a spec can seed what the kernel would write. */
function publishBuffer(system: GpuSoftSystem): StubBuffer {
  return system.publish.raw as StubBuffer;
}

/** Fill the publish buffer with a pattern no solver would produce. */
function seedPublish(system: GpuSoftSystem): Float32Array {
  const floats = publishBuffer(system).floats();
  for (let i = 0; i < floats.length; i++) floats[i] = i + 1;
  return floats;
}

/** Count the `dispose` events a three.js object fires, since it keeps no flag. */
function onDispose(object: { addEventListener(type: string, fn: () => void): void }): () => number {
  let fired = 0;
  object.addEventListener('dispose', () => {
    fired++;
  });
  return () => fired;
}

/** The tight xyz a view should be holding, straight out of the node field. */
function expectedPositions(mesh: SoftMesh): number[] {
  const out: number[] = [];
  for (let i = 0; i < mesh.count; i++) {
    const o = i * SOFT_STRIDE + SOFT_OFFSET.position;
    out.push(mesh.data[o], mesh.data[o + 1], mesh.data[o + 2]);
  }
  return out;
}

function snapshot(mesh: SoftMesh): Uint8Array {
  return new Uint8Array(mesh.data.buffer.slice(0));
}

// ---------------------------------------------------------------------------
// the pure functions
// ---------------------------------------------------------------------------

describe('the position layout', () => {
  it('is the publish layout, so a kernel change resizes the attribute with it', () => {
    expect(SOFT_POSITION_FLOATS).toBe(SOFT_PUBLISH_FLOATS_PER_NODE);
    expect(SOFT_POSITION_FLOATS).toBe(3);
    expect(SOFT_POSITION_BYTES).toBe(SOFT_POSITION_FLOATS * 4);
  });
});

describe('defaultDrawMode', () => {
  it('asks for a surface when the mesh carries triangles', () => {
    expect(defaultDrawMode(cloth())).toBe('surface');
    expect(new SoftMesh({ count: 64, scene: 'sheets', seed: 1 }).triangles.length).toBeGreaterThan(0);
    expect(defaultDrawMode(new SoftMesh({ count: 64, scene: 'sheets', seed: 1 }))).toBe('surface');
  });

  it('falls back to edges for a mesh that has a graph and no surface', () => {
    expect(rope().triangles.length).toBe(0);
    expect(defaultDrawMode(rope())).toBe('edges');
    expect(defaultDrawMode(new SoftMesh({ count: 64, scene: 'cube', seed: 1 }))).toBe('edges');
  });

  it('still says edges for free nodes, which the view then refuses to draw', () => {
    expect(defaultDrawMode(cloud())).toBe('edges');
  });
});

describe('writePositions', () => {
  it('writes tight xyz out of the interleaved node field', () => {
    const data = nodeData(4);
    const target = new Float32Array(4 * SOFT_POSITION_FLOATS);
    expect(writePositions(target, data, 4)).toBe(4);
    expect(Array.from(target)).toEqual([
      1, -1, 0.5,
      2, -2, 1,
      3, -3, 1.5,
      4, -4, 2,
    ]);
  });

  it('ignores velocity, inverse mass and radius, since a position has nowhere to put them', () => {
    const data = nodeData(3);
    const target = new Float32Array(3 * SOFT_POSITION_FLOATS);
    writePositions(target, data, 3);
    const written = Array.from(target);
    for (const stray of [1000, 2000, 3000, 777, 888]) {
      expect(written, `${stray} is state, not a position`).not.toContain(stray);
    }
  });

  it('overwrites what was there, so a recycled attribute is safe', () => {
    const target = new Float32Array(2 * SOFT_POSITION_FLOATS).fill(-12345);
    writePositions(target, nodeData(2), 2);
    expect(Array.from(target)).not.toContain(-12345);
    expect(Array.from(target)).toEqual([1, -1, 0.5, 2, -2, 1]);
  });

  it('writes nothing for a zero count', () => {
    const target = new Float32Array(SOFT_POSITION_FLOATS).fill(9);
    expect(writePositions(target, nodeData(1), 0)).toBe(0);
    expect(Array.from(target)).toEqual([9, 9, 9]);
  });

  it('accepts any indexable target, not only a typed array', () => {
    // Pre-sized, because the guard is on `length`: a plain array that grows as it
    // is written would report 0 floats and be refused, correctly.
    const target: number[] = new Array(2 * SOFT_POSITION_FLOATS).fill(0);
    const view: PositionTarget = target;
    expect(writePositions(view, nodeData(2), 2)).toBe(2);
    expect(target).toEqual([1, -1, 0.5, 2, -2, 1]);
  });

  it('refuses a count that is not a non-negative integer', () => {
    const target = new Float32Array(SOFT_POSITION_FLOATS);
    for (const count of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => writePositions(target, nodeData(1), count)).toThrow(RangeError);
    }
  });

  it('refuses a target too small for the nodes, naming both sizes', () => {
    const target = new Float32Array(2 * SOFT_POSITION_FLOATS);
    expect(() => writePositions(target, nodeData(4), 3)).toThrow(
      /holds 6 floats, 3 nodes need 9/,
    );
  });

  it('refuses a field too small for the count, rather than reading past it', () => {
    const target = new Float32Array(4 * SOFT_POSITION_FLOATS);
    expect(() => writePositions(target, nodeData(2), 4)).toThrow(
      /holds 16 floats, 4 nodes need 32/,
    );
  });
});

describe('positionBufferOf', () => {
  it('hands back the GPUBuffer three.js owns for an attribute', () => {
    const device = new StubDevice({ limits: stubLimits() });
    const buffer = device.createBuffer({
      label: 'three-position',
      size: SOFT_POSITION_BYTES,
      usage: U.VERTEX | U.COPY_DST,
    });
    const attribute = {};
    const renderer = new FakeRenderer({ buffer });
    expect(positionBufferOf(renderer, attribute)).toBe(buffer);
    expect(renderer.asked, 'it asks about the attribute it was given').toEqual([attribute]);
  });

  it('returns null with no renderer, no handle and no buffer', () => {
    const attribute = {};
    expect(positionBufferOf(null, attribute)).toBeNull();
    expect(positionBufferOf(undefined, attribute)).toBeNull();
    expect(positionBufferOf(new FakeRenderer(undefined), attribute)).toBeNull();
    expect(positionBufferOf(new FakeRenderer({}), attribute)).toBeNull();
  });

  it('returns null for a buffer that cannot be copied into', () => {
    // A WebGLBuffer would land here too if a caller passed a renderer that grew a
    // `backend`. The shape test is what turns a submit-time validation error with
    // no numbers in it into a clean CPU frame.
    const renderer = new FakeRenderer({ buffer: { size: 12 } as never });
    expect(positionBufferOf(renderer, {})).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------

describe('SoftView construction', () => {
  it('builds one indexed surface over a plain, non-dynamic position attribute', () => {
    const mesh = cloth();
    const view = new SoftView({ system: createCpuSoftSystem({ mesh }) });
    expect(view.drawMode).toBe('surface');
    expect(view.surface).toBeInstanceOf(THREE.Mesh);
    expect(view.edges).toBeNull();
    expect(view.wire, 'no overlay was asked for').toBeNull();
    expect(view.object.children).toEqual([view.surface]);

    const geometry = view.surface!.geometry;
    expect(geometry.attributes.position).toBe(view.position);
    expect(geometry.index!.array, 'the mesh triangles are wrapped, not copied').toBe(mesh.triangles);
    expect(view.position.itemSize).toBe(SOFT_POSITION_FLOATS);
    expect(view.position.count).toBe(mesh.count);
    expect(view.position.array).toBeInstanceOf(Float32Array);
    expect(view.position.array.length).toBe(mesh.count * SOFT_POSITION_FLOATS);
    // A dynamic attribute is re-uploaded every frame, which on a blit frame would
    // overwrite the positions the device just wrote with the stale CPU array.
    expect(view.position.usage).not.toBe(THREE.DynamicDrawUsage);
    expect(view.surface!.frustumCulled, 'the solver owns the bounds, not three.js').toBe(false);
    expect(view.surface!.name).toBe('soft-surface');
    expect(view.object.name).toBe('soft-view');
    view.dispose();
  });

  it('builds a line view over the constraint graph when the mesh has no surface', () => {
    const mesh = rope();
    const view = new SoftView({ system: createCpuSoftSystem({ mesh }) });
    expect(view.drawMode).toBe('edges');
    expect(view.surface).toBeNull();
    expect(view.edges).toBeInstanceOf(THREE.LineSegments);
    expect(view.edges!.geometry.attributes.position).toBe(view.position);
    expect(view.edges!.geometry.index!.array).toBe(mesh.constraints.ends);
    expect(view.edges!.frustumCulled).toBe(false);
    expect(view.edges!.name).toBe('soft-edges');
    view.dispose();
  });

  it('draws the wire overlay over the same attribute, so one blit fills both', () => {
    const mesh = cloth();
    const view = new SoftView({ system: createCpuSoftSystem({ mesh }), wireframe: true });
    expect(view.wire).toBeInstanceOf(THREE.LineSegments);
    expect(view.wire).not.toBe(view.edges);
    expect(view.object.children).toEqual([view.surface, view.wire]);
    expect(view.wire!.geometry.attributes.position, 'shared, not duplicated').toBe(view.position);
    expect(view.wire!.geometry.index!.array).toBe(mesh.constraints.ends);
    expect(view.wire!.geometry).not.toBe(view.surface!.geometry);
    expect(view.wireVisible).toBe(true);
    view.dispose();
  });

  it('names the line view as the overlay in edges mode instead of duplicating it', () => {
    const mesh = rope();
    const view = new SoftView({ system: createCpuSoftSystem({ mesh }), wireframe: true });
    expect(view.wire, 'the view already is the constraint graph').toBe(view.edges);
    expect(view.object.children).toEqual([view.edges]);
    expect(view.wireVisible).toBe(true);
    view.dispose();
  });

  it('toggles the overlay, and does nothing at all when there is none', () => {
    const mesh = cloth();
    const view = new SoftView({ system: createCpuSoftSystem({ mesh }), wireframe: true });
    view.setWireVisible(false);
    expect(view.wireVisible).toBe(false);
    expect(view.wire!.visible).toBe(false);
    view.setWireVisible(true);
    expect(view.wireVisible).toBe(true);
    view.dispose();

    const bare = new SoftView({ system: createCpuSoftSystem({ mesh: cloth() }) });
    expect(bare.wireVisible).toBe(false);
    expect(() => bare.setWireVisible(true)).not.toThrow();
    expect(bare.wireVisible).toBe(false);
    bare.dispose();
  });

  it('fills the attribute from the mesh before the first frame', () => {
    const mesh = cloth(16);
    const view = new SoftView({ system: createCpuSoftSystem({ mesh }) });
    expect(view.position.version, 'the constructor uploads once').toBe(1);
    expect(Array.from(view.position.array as Float32Array)).toEqual(expectedPositions(mesh));
    view.dispose();
  });

  it('reads the mesh rather than the device for that first fill', async () => {
    // The GPU tier has no `copyPublishedTo` that takes an array, so the
    // constructor has to go through the interleaved field -- which on a fresh
    // system is the mesh it was built from.
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    expect(Array.from(view.position.array as Float32Array)).toEqual(expectedPositions(rig.mesh));
    view.dispose();
    rig.system.dispose();
  });

  it('starts on the CPU path with nothing to report', () => {
    const view = new SoftView({ system: createCpuSoftSystem({ mesh: cloth(16) }) });
    expect(view.viewMode).toBe('cpu');
    expect(view.frameMode).toBe('cpu-upload');
    expect(view.ready).toBe(false);
    expect(view.blittedBytes).toBe(0);
    expect(view.gpuError).toBeNull();
    expect(view.count).toBe(16);
    expect(view.disposed).toBe(false);
    view.dispose();
  });

  it('lights a flat-shaded, two-sided, offset surface and a darker wire', () => {
    const view = new SoftView({
      system: createCpuSoftSystem({ mesh: cloth(16) }),
      wireframe: true,
    });
    const surface = view.surface!.material as THREE.MeshLambertMaterial;
    expect(surface).toBeInstanceOf(THREE.MeshLambertMaterial);
    expect(surface.color.getHex()).toBe(SOFT_SURFACE_COLOR);
    // Cells fold through each other as a cloth drapes, and the triangles are
    // wound for the initial orientation only.
    expect(surface.side).toBe(THREE.DoubleSide);
    // No normal attribute, so the shading has to come from derivatives.
    expect(view.surface!.geometry.attributes.normal).toBeUndefined();
    expect(surface.flatShading).toBe(true);
    expect(surface.polygonOffset, 'the coplanar overlay must not z-fight').toBe(true);
    const wire = view.wire!.material as THREE.LineBasicMaterial;
    expect(wire).toBeInstanceOf(THREE.LineBasicMaterial);
    expect(wire.color.getHex()).toBe(SOFT_WIRE_COLOR);
    view.dispose();
  });

  it('honours colours, and lets the wire colour fall back to the surface one in edges mode', () => {
    const surface = new SoftView({ system: createCpuSoftSystem({ mesh: cloth(16) }), color: 0x112233 });
    expect((surface.surface!.material as THREE.MeshLambertMaterial).color.getHex()).toBe(0x112233);
    surface.dispose();

    const wired = new SoftView({
      system: createCpuSoftSystem({ mesh: cloth(16) }),
      wireframe: true,
      color: 0x112233,
      wireColor: 0x445566,
    });
    expect((wired.wire!.material as THREE.LineBasicMaterial).color.getHex()).toBe(0x445566);
    wired.dispose();

    const edges = new SoftView({
      system: createCpuSoftSystem({ mesh: rope(16) }),
      color: 0x112233,
    });
    expect((edges.edges!.material as THREE.LineBasicMaterial).color.getHex()).toBe(0x112233);
    edges.dispose();
  });

  it('takes injected materials as given', () => {
    const material = new THREE.MeshBasicMaterial();
    const wireMaterial = new THREE.LineBasicMaterial({ color: 0xabcdef });
    const view = new SoftView({
      system: createCpuSoftSystem({ mesh: cloth(16) }),
      wireframe: true,
      material,
      wireMaterial,
    });
    expect(view.surface!.material).toBe(material);
    expect(view.wire!.material).toBe(wireMaterial);
    view.dispose();
  });

  it('recognises the GPU backend and nothing else', async () => {
    const rig = await gpuRig(16);
    const gpu = new SoftView({ system: rig.system, renderer: rig.renderer });
    expect(gpu.gpu).toBe(rig.system);
    expect(gpu.ready, 'a GPU system is eligible to blit from frame one').toBe(true);
    gpu.dispose();

    const cpu = new SoftView({ system: createCpuSoftSystem({ mesh: cloth(16) }) });
    expect(cpu.gpu).toBeNull();
    cpu.dispose();
    rig.system.dispose();
  });

  it('refuses a system it cannot draw', () => {
    const mesh = cloth(16);
    expect(() => new SoftView({ system: null as unknown as SoftSystem })).toThrow(TypeError);
    expect(() => new SoftView({ system: {} as SoftSystem })).toThrow(TypeError);
    const noMesh = { name: 'cpu', count: 4 } as unknown as SoftSystem;
    expect(() => new SoftView({ system: noMesh })).toThrow(TypeError);
    const noNodes = { name: 'cpu', mesh, count: 0 } as unknown as SoftSystem;
    expect(() => new SoftView({ system: noNodes })).toThrow(TypeError);
    const fractional = { name: 'cpu', mesh, count: 1.5 } as unknown as SoftSystem;
    expect(() => new SoftView({ system: fractional })).toThrow(TypeError);
  });

  it('refuses a draw mode it does not implement', () => {
    const system = createCpuSoftSystem({ mesh: cloth(16) });
    expect(() => new SoftView({ system, drawMode: 'points' as never })).toThrow(
      /unknown draw mode "points"/,
    );
  });

  it('refuses a surface over a mesh that has no triangles, naming the way out', () => {
    const system = createCpuSoftSystem({ mesh: rope(16) });
    expect(() => new SoftView({ system, drawMode: 'surface' })).toThrow(
      /no triangles.*drawMode "edges"/s,
    );
  });

  it('refuses a mesh with nothing to draw at all', () => {
    const system = createCpuSoftSystem({ mesh: cloud(8) });
    expect(() => new SoftView({ system })).toThrow(/no constraints/);
    expect(() => new SoftView({ system, drawMode: 'edges' })).toThrow(/no constraints/);
  });
});

// ---------------------------------------------------------------------------
// the CPU tier
// ---------------------------------------------------------------------------

describe('SoftView on the CPU tier', () => {
  it('never asks the backend for anything, even one that has buffers', () => {
    const device = new StubDevice({ limits: softLimits() });
    const target = device.createBuffer({
      label: 'three-position',
      size: 16 * SOFT_POSITION_BYTES,
      usage: U.VERTEX | U.COPY_SRC | U.COPY_DST,
    });
    const renderer = new FakeRenderer({ buffer: target });
    const mesh = cloth(16);
    const system = createCpuSoftSystem({ mesh });
    const view = new SoftView({ system, renderer });
    expect(view.gpu).toBeNull();
    // The constructor already filled the attribute, so `asked` starts empty and
    // stays empty: there is no buffer for this tier to look for.
    expect(view.update()).toBe('cpu-upload');
    expect(view.update()).toBe('cpu-upload');
    expect(renderer.asked, 'a CPU-tier view never asks the backend for anything').toEqual([]);
    expect(device.submissions, 'and never submits a copy').toEqual([]);
    expect(view.viewMode).toBe('cpu');
    expect(view.gpuError, 'this tier was never going to blit, so there is nothing to report').toBeNull();
    view.dispose();
  });

  it('is not fooled by a system that claims the gpu name', () => {
    const system = createCpuSoftSystem({ mesh: cloth(16) });
    const impostor = Object.create(system) as CpuSoftSystem;
    Object.assign(impostor, { name: 'gpu' });
    const view = new SoftView({ system: impostor });
    expect(view.gpu, 'a name is not a publish buffer').toBeNull();
    expect(view.update()).toBe('cpu-upload');
    view.dispose();
  });

  it('is not fooled by a system with a publish buffer but no device', () => {
    const device = new StubDevice({ limits: softLimits() });
    const buffer = device.createBuffer({ label: 'publish', size: 64, usage: U.STORAGE });
    const system = createCpuSoftSystem({ mesh: cloth(16) });
    const impostor = Object.create(system) as CpuSoftSystem;
    Object.assign(impostor, { name: 'gpu', publish: buffer, copyPublishedTo: () => 0 });
    const view = new SoftView({ system: impostor, renderer: new FakeRenderer({ buffer }) });
    expect(view.gpu, 'a publish buffer with no shared device cannot submit a copy').toBeNull();
    expect(view.update()).toBe('cpu-upload');
    view.dispose();
  });

  it('re-uploads every frame, so a stepped mesh is drawn where it moved to', () => {
    const mesh = cloth(16);
    const system = createCpuSoftSystem({ mesh });
    const view = new SoftView({ system });
    const before = Array.from(view.position.array as Float32Array);
    const version = view.position.version;
    system.step();
    expect(view.update()).toBe('cpu-upload');
    expect(Array.from(view.position.array as Float32Array)).not.toEqual(before);
    expect(Array.from(view.position.array as Float32Array)).toEqual(expectedPositions(mesh));
    expect(view.position.version, 'three.js has to be told to upload').toBe(version + 1);
    expect(view.blittedBytes).toBe(0);
    view.dispose();
  });

  it('uploads the published positions rather than the interleaved field', () => {
    // Both are the same numbers here, so the distinction is checked by taking the
    // publisher away: with no `copyPublishedTo` the view has to fall back to
    // `writePositions` over `mesh.data` and still draw the same thing.
    const mesh = cloth(16);
    const system = createCpuSoftSystem({ mesh });
    system.step();
    const published = new Float32Array(mesh.count * SOFT_POSITION_FLOATS);
    system.copyPublishedTo(published);

    const withPublisher = new SoftView({ system });
    expect(Array.from(withPublisher.position.array as Float32Array)).toEqual(
      Array.from(published),
    );
    withPublisher.dispose();

    const stripped = Object.create(system) as CpuSoftSystem;
    Object.defineProperty(stripped, 'copyPublishedTo', { value: undefined });
    const withoutPublisher = new SoftView({ system: stripped });
    expect(Array.from(withoutPublisher.position.array as Float32Array)).toEqual(
      Array.from(published),
    );
    withoutPublisher.dispose();
  });

  it('keeps uploading when the publisher is gone and the field has moved on', () => {
    const mesh = cloth(16);
    const system = createCpuSoftSystem({ mesh });
    const stripped = Object.create(system) as CpuSoftSystem;
    Object.defineProperty(stripped, 'copyPublishedTo', { value: undefined });
    const view = new SoftView({ system: stripped });
    system.step();
    expect(view.update()).toBe('cpu-upload');
    expect(Array.from(view.position.array as Float32Array)).toEqual(expectedPositions(mesh));
    view.dispose();
  });

  it('blames nothing when a GPU-backed view has no renderer at all', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system });
    for (let i = 0; i < 4; i++) expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError, 'there was no renderer to expose a buffer').toBeNull();
    expect(view.ready, 'and the verdict is not withdrawn either').toBe(true);
    expect(view.viewMode).toBe('cpu');
    view.dispose();
    rig.system.dispose();
  });
});

// ---------------------------------------------------------------------------
// the blit path
// ---------------------------------------------------------------------------

describe('SoftView blitting to the GPU', () => {
  it('blits on the first frame, because there is nothing to compile', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    const seeded = seedPublish(rig.system);
    expect(view.update()).toBe('gpu-blit');
    expect(view.viewMode).toBe('gpu');
    expect(view.frameMode).toBe('gpu-blit');
    expect(view.ready).toBe(true);
    expect(view.gpuError).toBeNull();
    expect(view.blittedBytes).toBe(16 * SOFT_POSITION_BYTES);
    expect(Array.from(rig.target.floats())).toEqual(
      Array.from(seeded.subarray(0, 16 * SOFT_POSITION_FLOATS)),
    );
    expect(rig.renderer.asked, 'it asks about its own attribute and nothing else').toEqual([
      view.position,
    ]);
    view.dispose();
    rig.system.dispose();
  });

  it('submits one copy, aimed at the buffer three.js owns and sized to the nodes', async () => {
    const rig = await gpuRig(100);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    const before = rig.device.submissions.length;
    view.update();
    expect(rig.device.submissions.length).toBe(before + 1);
    const copies = rig.device.submissions.at(-1)!.copies;
    expect(copies).toHaveLength(1);
    expect(copies[0].from).toBe(publishBuffer(rig.system));
    expect(copies[0].to).toBe(rig.target);
    expect(copies[0].fromOffset).toBe(0);
    expect(copies[0].toOffset).toBe(0);
    // The publish buffer is tight xyz over exactly the drawn nodes -- the padding
    // lives in the workgroup order, not here -- so source and destination are the
    // same size and the copy needs no stride, no offset and nothing to expand.
    expect(publishBuffer(rig.system).size).toBe(100 * SOFT_POSITION_BYTES);
    expect(copies[0].bytes).toBe(100 * SOFT_POSITION_BYTES);
    view.dispose();
    rig.system.dispose();
  });

  it('does not ask three.js to re-upload the array it just bypassed', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    seedPublish(rig.system);
    const version = view.position.version;
    expect(view.update()).toBe('gpu-blit');
    expect(view.position.version, 'needsUpdate would overwrite the blit with zeros').toBe(version);
    expect(view.update()).toBe('gpu-blit');
    expect(view.position.version).toBe(version);
    view.dispose();
    rig.system.dispose();
  });

  it('leaves the CPU array alone, so a later fallback is stale and not zero', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    seedPublish(rig.system);
    const cpuArray = Array.from(view.position.array as Float32Array);
    view.update();
    expect(Array.from(view.position.array as Float32Array)).toEqual(cpuArray);
    view.dispose();
    rig.system.dispose();
  });

  it('copies only what the destination holds, floored to whole nodes', async () => {
    // 5 bytes short of the full mesh: a truncated node would leave its last
    // component from the previous frame, which is a *wrong* position rather than
    // a stale one.
    const rig = await gpuRig(100, 100 * SOFT_POSITION_BYTES - 5);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    seedPublish(rig.system);
    expect(view.update()).toBe('gpu-blit');
    expect(view.blittedBytes).toBe(99 * SOFT_POSITION_BYTES);
    expect(rig.device.submissions.at(-1)!.copies[0].bytes).toBe(99 * SOFT_POSITION_BYTES);
    view.dispose();
    rig.system.dispose();
  });

  it('falls back to the CPU when the destination cannot hold one node', async () => {
    const rig = await gpuRig(100, 8);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    const submissions = rig.device.submissions.length;
    expect(view.update()).toBe('cpu-upload');
    expect(rig.device.submissions.length, 'a zero-byte copy is never submitted').toBe(submissions);
    expect(view.blittedBytes).toBe(0);
    expect(view.viewMode).toBe('cpu');
    expect(view.gpuError, 'a small buffer is the renderer\'s business, not a fault').toBeNull();
    expect(view.ready, 'and the view keeps trying').toBe(true);
    expect(Array.from(view.position.array as Float32Array)).toEqual(expectedPositions(rig.mesh));
    view.dispose();
    rig.system.dispose();
  });

  it('says so when three.js never exposes a destination buffer', async () => {
    const rig = await gpuRig(16);
    rig.renderer.handle = {};
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    // Frame one legitimately sees nothing: three.js creates the buffer during the
    // first render, and `update()` runs before it.
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toBeNull();
    expect(view.ready).toBe(true);
    // Frame two does not.
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toMatch(/no GPUBuffer for the position attribute in 2 frames/);
    expect(view.ready).toBe(false);
    expect(view.viewMode).toBe('cpu');
    view.dispose();
    rig.system.dispose();
  });

  it('withdraws that verdict when the buffer arrives late', async () => {
    const rig = await gpuRig(16);
    rig.renderer.handle = {};
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    view.update();
    view.update();
    expect(view.gpuError).not.toBeNull();

    rig.renderer.handle = { buffer: rig.target };
    seedPublish(rig.system);
    expect(view.update(), 'a slow first render is not a broken assumption').toBe('gpu-blit');
    expect(view.gpuError).toBeNull();
    expect(view.ready).toBe(true);
    expect(view.viewMode).toBe('gpu');
    view.dispose();
    rig.system.dispose();
  });

  it('replaces the missing-buffer verdict with the real reason once a buffer arrives', async () => {
    const rig = await gpuRig(16);
    rig.renderer.handle = {};
    rig.device.lose({ reason: 'unknown', message: 'driver went away' });
    await flushMicrotasks();
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    view.update();
    view.update();
    expect(view.gpuError, 'two frames with nowhere to blit').toMatch(/no GPUBuffer/);

    // Withdrawing that verdict is free here -- there is no pipeline to rebuild --
    // so the view re-arms, attempts the copy, and the lost device answers. The
    // point is which reason survives: a view that kept reporting the missing
    // buffer would be naming the symptom and hiding the cause.
    rig.renderer.handle = { buffer: rig.target };
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toMatch(/lost/);
    expect(view.ready, 'a lost device is not retried').toBe(false);
    expect(view.viewMode).toBe('cpu');
    view.dispose();
    rig.system.dispose();
  });

  it('reports a buffer that disappeared after a successful blit', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    seedPublish(rig.system);
    expect(view.update()).toBe('gpu-blit');
    expect(view.gpuError).toBeNull();

    // A destroyed attribute or a lost device, not a slow first render: positions
    // *were* reaching this buffer.
    rig.renderer.handle = {};
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toMatch(/disappeared after a successful blit/);
    expect(view.viewMode).toBe('cpu');
    expect(view.ready).toBe(false);
    // And it stays down rather than alternating between the two diagnostics.
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toMatch(/disappeared after a successful blit/);
    view.dispose();
    rig.system.dispose();
  });
});

// ---------------------------------------------------------------------------
// failure
// ---------------------------------------------------------------------------

describe('SoftView when the GPU path fails', () => {
  it('falls back mid-run when the device is lost, and stays down', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    seedPublish(rig.system);
    expect(view.update()).toBe('gpu-blit');

    rig.device.lose({ reason: 'unknown', message: 'driver went away' });
    await flushMicrotasks();
    const submissions = rig.device.submissions.length;
    expect(view.update()).toBe('cpu-upload');
    expect(view.viewMode).toBe('cpu');
    expect(view.ready).toBe(false);
    expect(view.gpuError).toMatch(/lost/);
    expect(rig.device.submissions.length, 'nothing is submitted to a dead device').toBe(submissions);
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toMatch(/lost/);
    view.dispose();
    rig.system.dispose();
  });

  it('still draws, from the mesh, on the frame the device dies', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    rig.device.lose({ reason: 'unknown', message: 'driver went away' });
    await flushMicrotasks();
    const version = view.position.version;
    expect(view.update()).toBe('cpu-upload');
    expect(Array.from(view.position.array as Float32Array)).toEqual(expectedPositions(rig.mesh));
    expect(view.position.version, 'three.js has to be told to upload the fallback').toBe(
      version + 1,
    );
    view.dispose();
    rig.system.dispose();
  });

  it('reports a device that is already gone before the first frame', async () => {
    const rig = await gpuRig(16);
    rig.shared.destroy();
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toMatch(/destroyed/);
    expect(view.viewMode).toBe('cpu');
    view.dispose();
  });

  it('reports a failure that is not an Error', async () => {
    const rig = await gpuRig(16);
    (rig.system as unknown as { copyPublishedTo: () => number }).copyPublishedTo = (): never => {
      throw 'a driver threw a string';
    };
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toBe('a driver threw a string');
    view.dispose();
    rig.system.dispose();
  });

  it('never throws out of update, whatever the device does', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    seedPublish(rig.system);
    expect(view.update()).toBe('gpu-blit');
    rig.shared.destroy();
    expect(() => view.update()).not.toThrow();
    expect(view.frameMode).toBe('cpu-upload');
    expect(view.gpuError).toMatch(/destroyed/);
    rig.renderer.handle = {};
    expect(() => view.update()).not.toThrow();
    expect(view.gpuError, 'a device failure outranks the missing-buffer diagnostic').toMatch(
      /destroyed/,
    );
    view.dispose();
    expect(view.disposed).toBe(true);
  });

  it('reports a disposed system rather than drawing nothing', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    rig.system.dispose();
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toMatch(/disposed/);
    view.dispose();
  });
});

// ---------------------------------------------------------------------------
// teardown
// ---------------------------------------------------------------------------

describe('SoftView teardown', () => {
  it('disposes the geometries and materials it owns, and empties the group', () => {
    const view = new SoftView({
      system: createCpuSoftSystem({ mesh: cloth(16) }),
      wireframe: true,
    });
    const parent = new THREE.Group();
    parent.add(view.object);
    const surfaceGeometry = onDispose(view.surface!.geometry);
    const wireGeometry = onDispose(view.wire!.geometry);
    const surfaceMaterial = onDispose(view.surface!.material as THREE.Material);
    const wireMaterial = onDispose(view.wire!.material as THREE.Material);

    view.dispose();
    expect(surfaceGeometry()).toBe(1);
    expect(wireGeometry()).toBe(1);
    expect(surfaceMaterial()).toBe(1);
    expect(wireMaterial()).toBe(1);
    expect(view.object.children, 'the drawables are gone from the group').toEqual([]);
    expect(parent.children, 'and the group is out of the scene').toEqual([]);
    expect(view.disposed).toBe(true);
    expect(view.viewMode).toBe('cpu');
  });

  it('shares the position attribute between two geometries without a double free', () => {
    const view = new SoftView({
      system: createCpuSoftSystem({ mesh: cloth(16) }),
      wireframe: true,
    });
    const position = view.position;
    expect(() => view.dispose()).not.toThrow();
    // three's backend keys the buffer on the attribute and guards its `delete()`,
    // so the second geometry's teardown is a no-op. What must not happen is the
    // attribute's own disposal being driven twice from here.
    expect(position.version, 'disposing a geometry does not touch the attribute').toBe(1);
  });

  it('leaves injected materials alone', () => {
    const material = new THREE.MeshBasicMaterial();
    const wireMaterial = new THREE.LineBasicMaterial();
    const materialGone = onDispose(material);
    const wireGone = onDispose(wireMaterial);
    const view = new SoftView({
      system: createCpuSoftSystem({ mesh: cloth(16) }),
      wireframe: true,
      material,
      wireMaterial,
    });
    view.dispose();
    expect(materialGone(), 'the caller owns what the caller passed').toBe(0);
    expect(wireGone()).toBe(0);
  });

  it('is idempotent', () => {
    const view = new SoftView({
      system: createCpuSoftSystem({ mesh: cloth(16) }),
      wireframe: true,
    });
    const geometryGone = onDispose(view.surface!.geometry);
    view.dispose();
    view.dispose();
    expect(geometryGone()).toBe(1);
  });

  it('refuses use after dispose, which is a caller bug and not a device condition', async () => {
    const rig = await gpuRig(16);
    const view = new SoftView({
      system: rig.system,
      renderer: rig.renderer,
      wireframe: true,
    });
    view.dispose();
    expect(() => view.update()).toThrow(/disposed/);
    expect(() => view.setWireVisible(false)).toThrow(/disposed/);
    rig.system.dispose();
  });
});

// ---------------------------------------------------------------------------
// the one rule that matters most
// ---------------------------------------------------------------------------

describe('the renderer never writes to the simulation', () => {
  it('leaves mesh.data byte-identical across CPU frames', () => {
    const mesh = cloth(32);
    const system = createCpuSoftSystem({ mesh });
    const view = new SoftView({ system, wireframe: true });
    const digest = system.digest();
    const bytes = snapshot(mesh);
    for (let i = 0; i < 5; i++) view.update();
    view.setWireVisible(false);
    view.setWireVisible(true);
    view.update();
    expect(system.digest()).toBe(digest);
    expect(snapshot(mesh)).toEqual(bytes);
    view.dispose();
  });

  it('leaves mesh.data byte-identical across GPU blits', async () => {
    const rig = await gpuRig(32);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer, wireframe: true });
    const digest = rig.system.digest();
    const bytes = snapshot(rig.mesh);
    seedPublish(rig.system);
    expect(view.update()).toBe('gpu-blit');
    expect(view.update()).toBe('gpu-blit');
    expect(rig.system.digest()).toBe(digest);
    expect(snapshot(rig.mesh)).toEqual(bytes);
    view.dispose();
    rig.system.dispose();
  });

  it('leaves mesh.data byte-identical across a GPU failure and its fallback', async () => {
    const rig = await gpuRig(32);
    const view = new SoftView({ system: rig.system, renderer: rig.renderer });
    const digest = rig.system.digest();
    const bytes = snapshot(rig.mesh);
    seedPublish(rig.system);
    view.update();
    rig.device.lose({ reason: 'unknown', message: 'driver went away' });
    await flushMicrotasks();
    view.update();
    rig.renderer.handle = {};
    view.update();
    expect(rig.system.digest()).toBe(digest);
    expect(snapshot(rig.mesh)).toEqual(bytes);
    view.dispose();
    rig.system.dispose();
  });

  it('does not disturb the solver: stepping after drawing gives the same digest', () => {
    const drawn = cloth(32);
    const plain = cloth(32);
    const drawnSystem = createCpuSoftSystem({ mesh: drawn });
    const plainSystem = createCpuSoftSystem({ mesh: plain });
    const view = new SoftView({ system: drawnSystem, wireframe: true });
    for (let i = 0; i < 8; i++) {
      drawnSystem.step();
      plainSystem.step();
      view.update();
    }
    expect(drawnSystem.digest(), 'a displayed run and an undisplayed one agree').toBe(
      plainSystem.digest(),
    );
    view.dispose();
  });
});

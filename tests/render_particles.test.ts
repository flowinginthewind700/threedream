/**
 * `render/particles.ts` -- the instance renderer, driven headless.
 *
 * three.js runs fine in bare Node as long as nothing asks it for a WebGL
 * context: geometries, materials and `InstancedMesh` are plain data, which is
 * what this module is made of. So the spec builds a real `InstancedMesh` around
 * a real `GpuParticleSystem` running on the recording stub, and puts a fake
 * renderer in front of it -- one object with a `backend.get()` that hands back a
 * `GPUBuffer` the way three.js's WebGPU backend does once it has created the
 * attribute. That is enough to drive both fill paths end to end, including the
 * attach, the failure modes and the teardown.
 *
 * What is asserted here, and what is not:
 *
 * - `writeInstanceMatrices` is arithmetic, so it is checked numerically against
 *   the field layout -- this is also the CPU-side twin of the WGSL kernel in
 *   `gpu/particleInstances.ts`, and `e2e/particles_gpu.spec.ts` compares the two
 *   on a live device.
 * - the blit path is checked by *moving bytes*: the stub applies a copy when the
 *   encoder is sealed, so filling the expansion buffer with a pattern and then
 *   calling `update()` proves the matrices land in the buffer three.js owns.
 *   The stub cannot run WGSL, so it cannot prove the kernel wrote the right
 *   numbers into that buffer -- only that the copy is sized and aimed correctly.
 * - the plan's "不回写仿真状态" is checked by digest, because a renderer that
 *   quietly wrote into `field.data` would still pass every visual test.
 */

import { describe, expect, it } from 'vitest';

import * as THREE from 'three';

import { SharedDeviceManager, type GpuBufferLike, type SharedDevice } from '../src/gpu/device.js';
import { createCpuParticleSystem, type CpuParticleSystem } from '../src/gpu/particleCpu.js';
import { PARTICLE_OFFSET, PARTICLE_STRIDE, ParticleField } from '../src/gpu/particleField.js';
import { createGpuParticleSystem, type GpuParticleSystem } from '../src/gpu/particleGpu.js';
import {
  INSTANCE_BYTES,
  INSTANCE_FLOATS,
  INSTANCE_WORD,
} from '../src/gpu/particleInstances.js';
import type { ParticleSystem } from '../src/gpu/particleTypes.js';
import {
  HIGH_DETAIL_MAX_COUNT,
  MEDIUM_DETAIL_MAX_COUNT,
  ParticleView,
  geometryForCount,
  instanceBufferOf,
  writeInstanceMatrices,
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

// ---------------------------------------------------------------------------
// rig
// ---------------------------------------------------------------------------

/**
 * The slice of `WebGPURenderer` the view reads, faked.
 *
 * `asked` records which attribute the view inquired about, because the view has
 * to ask for *its own* `instanceMatrix` and nothing else -- a renderer that
 * asked for the wrong attribute would blit into some other mesh's buffer and
 * still look correct from here.
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

/** The buffer three.js would have made for `instanceMatrix`: `VERTEX|COPY_DST`. */
function threeLikeTarget(device: StubDevice, count: number, bytes = count * INSTANCE_BYTES): StubBuffer {
  return device.createBuffer({
    label: 'three-instanceMatrix',
    size: bytes,
    usage: U.VERTEX | U.COPY_SRC | U.COPY_DST,
  });
}

function buf(device: StubDevice, label: string): StubBuffer {
  const found = device.buffers.find((b) => b.label === label);
  if (!found) {
    throw new Error(`no buffer labelled '${label}'; have [${device.buffers.map((b) => b.label)}]`);
  }
  return found;
}

/** A field with distinct positions and a radius that is not 1, so scale is visible. */
function seededField(count: number, radius: readonly [number, number] = [0.25, 0.25]): ParticleField {
  return new ParticleField({ count, scene: 'sphere', seed: 7, radius, speed: 2 });
}

interface GpuRig {
  readonly device: StubDevice;
  readonly manager: SharedDeviceManager;
  readonly shared: SharedDevice;
  readonly field: ParticleField;
  readonly system: GpuParticleSystem;
  readonly target: StubBuffer;
  readonly renderer: FakeRenderer;
  readonly count: number;
}

/** A real GPU system on a stub device, plus the renderer-owned buffer to blit into. */
async function gpuRig(count = 128, targetBytes = count * INSTANCE_BYTES): Promise<GpuRig> {
  const device = new StubDevice({ limits: stubLimits() });
  const manager = new SharedDeviceManager({ constants: CONSTANTS });
  const shared = manager.adopt(device);
  const field = seededField(count);
  const system = await createGpuParticleSystem({ shared, field });
  const target = threeLikeTarget(device, count, targetBytes);
  return { device, manager, shared, field, system, target, renderer: new FakeRenderer({ buffer: target }), count };
}

/** Count the `dispose` events a three.js object fires, since it keeps no flag. */
function onDispose(object: { addEventListener(type: string, fn: () => void): void }): () => number {
  let fired = 0;
  object.addEventListener('dispose', () => {
    fired++;
  });
  return () => fired;
}

/** Raw particle bytes with values that must *not* reach a matrix. */
function fieldData(count: number, radius = 0.5): Float32Array {
  const data = new Float32Array(count * PARTICLE_STRIDE);
  for (let i = 0; i < count; i++) {
    const o = i * PARTICLE_STRIDE;
    data[o + PARTICLE_OFFSET.position] = i + 1;
    data[o + PARTICLE_OFFSET.position + 1] = -(i + 1);
    data[o + PARTICLE_OFFSET.position + 2] = (i + 1) * 0.5;
    data[o + PARTICLE_OFFSET.radius] = radius;
    data[o + PARTICLE_OFFSET.velocity] = 1000 + i;
    data[o + PARTICLE_OFFSET.mass] = 9999;
  }
  return data;
}

// ---------------------------------------------------------------------------

describe('geometryForCount', () => {
  it('uses a smooth sphere up to the high-detail ceiling', () => {
    for (const count of [1, 64, HIGH_DETAIL_MAX_COUNT]) {
      const geometry = geometryForCount(count) as THREE.SphereGeometry;
      expect(geometry.type, `count ${count}`).toBe('SphereGeometry');
      expect(geometry.parameters.widthSegments).toBe(16);
      expect(geometry.parameters.heightSegments).toBe(12);
      expect(geometry.parameters.radius).toBe(1);
      geometry.dispose();
    }
  });

  it('drops to a coarse sphere past it, and to an icosahedron past the medium ceiling', () => {
    const medium = geometryForCount(HIGH_DETAIL_MAX_COUNT + 1) as THREE.SphereGeometry;
    expect(medium.parameters.widthSegments).toBe(8);
    expect(medium.parameters.heightSegments).toBe(6);
    medium.dispose();

    const coarse = geometryForCount(MEDIUM_DETAIL_MAX_COUNT) as THREE.SphereGeometry;
    expect(coarse.parameters.widthSegments).toBe(8);
    coarse.dispose();

    const low = geometryForCount(MEDIUM_DETAIL_MAX_COUNT + 1) as THREE.IcosahedronGeometry;
    expect(low.type).toBe('IcosahedronGeometry');
    expect(low.parameters.detail).toBe(0);
    expect(low.parameters.radius).toBe(1);
    low.dispose();
  });

  it('returns a fresh geometry each call, so a view can dispose the one it owns', () => {
    const a = geometryForCount(8);
    const b = geometryForCount(8);
    expect(a).not.toBe(b);
    a.dispose();
    b.dispose();
  });

  it('refuses a count that cannot be instanced', () => {
    for (const bad of [0, -1, 1.5, Number.NaN]) {
      expect(() => geometryForCount(bad), `count ${bad}`).toThrow(RangeError);
    }
  });
});

describe('writeInstanceMatrices', () => {
  it('writes scale on the diagonal and the position in the fourth column', () => {
    const target = new Float32Array(INSTANCE_FLOATS);
    const written = writeInstanceMatrices(target, fieldData(1), 1);
    expect(written).toBe(1);
    expect(Array.from(target)).toEqual([0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 1, -1, 0.5, 1]);
  });

  it('is column major, matching both three.js and the WGSL kernel', () => {
    // A row-major writer would put the translation at 3, 7 and 11 instead.
    const target = new Float32Array(INSTANCE_FLOATS);
    writeInstanceMatrices(target, fieldData(1), 1);
    expect([target[3], target[7], target[11]]).toEqual([0, 0, 0]);
    expect([target[12], target[13], target[14]]).toEqual([1, -1, 0.5]);
    expect(target[15]).toBe(1);
  });

  it('writes every instance from its own slice of the field', () => {
    const count = 4;
    const target = new Float32Array(count * INSTANCE_FLOATS);
    expect(writeInstanceMatrices(target, fieldData(count), count)).toBe(count);
    for (let i = 0; i < count; i++) {
      const o = i * INSTANCE_FLOATS;
      expect(target[o + 12]).toBe(i + 1);
      expect(target[o + 13]).toBe(-(i + 1));
      expect(target[o + 14]).toBe((i + 1) * 0.5);
      expect(target[o]).toBe(0.5);
      expect(target[o + 15]).toBe(1);
    }
  });

  it('multiplies the radius by radiusScale, and only the radius', () => {
    const target = new Float32Array(INSTANCE_FLOATS);
    writeInstanceMatrices(target, fieldData(1), 1, 4);
    expect([target[0], target[5], target[10]]).toEqual([2, 2, 2]);
    expect([target[12], target[13], target[14]]).toEqual([1, -1, 0.5]);
  });

  it('overwrites all sixteen floats, so an identity-initialised array is safe', () => {
    // three.js fills `instanceMatrix` with identities, not zeros: a writer that
    // skipped the twelve constant slots would leave a rotation in every instance.
    const target = new Float32Array(INSTANCE_FLOATS);
    for (let i = 0; i < INSTANCE_FLOATS; i++) target[i] = i === 15 ? 1 : 7;
    writeInstanceMatrices(target, fieldData(1), 1);
    expect(Array.from(target)).toEqual([0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 1, -1, 0.5, 1]);
  });

  it('ignores velocity and mass, since a matrix has nowhere to put them', () => {
    const target = new Float32Array(INSTANCE_FLOATS);
    writeInstanceMatrices(target, fieldData(1), 1);
    expect(Array.from(target).some((v) => v >= 1000)).toBe(false);
  });

  it('writes nothing for a zero count, which is what setCount(0) asks for', () => {
    const target = new Float32Array(INSTANCE_FLOATS).fill(3);
    expect(writeInstanceMatrices(target, fieldData(1), 0)).toBe(0);
    expect(Array.from(target)).toEqual(new Array(INSTANCE_FLOATS).fill(3));
  });

  it('accepts any indexable target, not only a typed array', () => {
    const target: number[] = new Array(INSTANCE_FLOATS).fill(0);
    expect(writeInstanceMatrices(target, Array.from(fieldData(1)), 1)).toBe(1);
    expect(target[12]).toBe(1);
    expect(target[0]).toBe(0.5);
  });

  it('refuses a count that is not a non-negative integer', () => {
    const target = new Float32Array(INSTANCE_FLOATS);
    for (const bad of [-1, 1.5, Number.NaN]) {
      expect(() => writeInstanceMatrices(target, fieldData(1), bad), `count ${bad}`).toThrow(
        RangeError,
      );
    }
  });

  it('refuses a radius scale that would collapse or invert every sphere', () => {
    const target = new Float32Array(INSTANCE_FLOATS);
    for (const bad of [0, -2, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => writeInstanceMatrices(target, fieldData(1), 1, bad), `scale ${bad}`).toThrow(
        RangeError,
      );
    }
  });

  it('refuses a target too small for the instances, naming both sizes', () => {
    const target = new Float32Array(INSTANCE_FLOATS);
    expect(() => writeInstanceMatrices(target, fieldData(2), 2)).toThrow(
      /holds 16 floats, 2 instances need 32/,
    );
  });

  it('refuses a field too small for the count, rather than reading past it', () => {
    const target = new Float32Array(2 * INSTANCE_FLOATS);
    expect(() => writeInstanceMatrices(target, fieldData(1), 2)).toThrow(
      /field holds 8 floats, 2 particles need 16/,
    );
  });
});

describe('instanceBufferOf', () => {
  const attribute = { name: 'instanceMatrix' };

  it('hands back the GPUBuffer behind an attribute', () => {
    const buffer = new StubBuffer({ label: 'attr', size: 64, usage: U.VERTEX | U.COPY_DST });
    const renderer = new FakeRenderer({ buffer });
    expect(instanceBufferOf(renderer, attribute)).toBe(buffer);
    expect(renderer.asked).toEqual([attribute]);
  });

  it('returns null with no renderer, no backend, no handle and no buffer', () => {
    const buffer = new StubBuffer({ label: 'attr', size: 64, usage: U.VERTEX });
    expect(instanceBufferOf(null, attribute)).toBeNull();
    expect(instanceBufferOf(undefined, attribute)).toBeNull();
    expect(instanceBufferOf({}, attribute), 'a WebGLRenderer has no .backend').toBeNull();
    expect(instanceBufferOf(new FakeRenderer(undefined), attribute)).toBeNull();
    expect(instanceBufferOf(new FakeRenderer({}), attribute)).toBeNull();
    expect(instanceBufferOf(new FakeRenderer({ buffer }), attribute)).toBe(buffer);
  });

  it('returns null for a buffer that cannot be copied into', () => {
    // A WebGLBuffer would arrive here if some renderer grew a `backend`: asking
    // for the two methods a copy needs beats trusting the shape.
    const webglLike = {} as unknown as GpuBufferLike;
    expect(instanceBufferOf(new FakeRenderer({ buffer: webglLike }), attribute)).toBeNull();
    const half = { mapAsync: async () => {} } as unknown as GpuBufferLike;
    expect(instanceBufferOf(new FakeRenderer({ buffer: half }), attribute)).toBeNull();
  });
});

describe('ParticleView construction', () => {
  it('builds one InstancedMesh sized to the system, uncullable and dynamic', () => {
    const field = seededField(64);
    const system = createCpuParticleSystem({ field });
    const view = new ParticleView({ system });
    expect(view.mesh).toBeInstanceOf(THREE.InstancedMesh);
    expect(view.mesh.count).toBe(64);
    expect(view.count).toBe(64);
    expect(view.mesh.instanceMatrix.count).toBe(64);
    expect(view.mesh.instanceMatrix.itemSize).toBe(16);
    expect(view.mesh.instanceMatrix.usage).toBe(THREE.DynamicDrawUsage);
    expect(view.mesh.frustumCulled, 'the instances move; the geometry bounds do not').toBe(false);
    expect(view.mesh.geometry.type).toBe('SphereGeometry');
    expect((view.mesh.material as THREE.MeshLambertMaterial).type).toBe('MeshLambertMaterial');
    view.dispose();
  });

  it('fills the matrices from the field before the first frame', () => {
    const field = seededField(8);
    const view = new ParticleView({ system: createCpuParticleSystem({ field }) });
    const array = view.mesh.instanceMatrix.array as Float32Array;
    expect(array[0]).toBeCloseTo(field.radius(0), 6);
    expect(array[12]).toBeCloseTo(field.position(0)[0], 6);
    expect(array[13]).toBeCloseTo(field.position(0)[1], 6);
    expect(array[14]).toBeCloseTo(field.position(0)[2], 6);
    expect(array[15]).toBe(1);
    expect(view.mesh.instanceMatrix.version, 'the constructor marks it for upload').toBe(1);
    view.dispose();
  });

  it('gives a GPU-backed view the attribute three.js keeps a buffer for', async () => {
    const rig = await gpuRig(32);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    const attribute = view.mesh.instanceMatrix as THREE.InstancedBufferAttribute & {
      isStorageInstancedBufferAttribute?: boolean;
    };
    // A plain `InstancedBufferAttribute` is wrapped in an `InstancedInterleavedBuffer`
    // by the time the shader node is built, so `backend.get(instanceMatrix).buffer`
    // is undefined and there is nothing to copy into. Only the storage attribute
    // stays reachable by itself.
    expect(attribute.isStorageInstancedBufferAttribute).toBe(true);
    expect(attribute.usage, 'a dynamic attribute is re-uploaded over the blit').toBe(
      THREE.StaticDrawUsage,
    );
    expect(attribute.count).toBe(32);
    expect(attribute.itemSize).toBe(INSTANCE_FLOATS);
    view.dispose();
    rig.system.dispose();
  });

  it('keeps the dynamic attribute on every tier that cannot blit', async () => {
    const rig = await gpuRig(16);
    const withoutRenderer = new ParticleView({ system: rig.system });
    expect(withoutRenderer.mesh.instanceMatrix.usage).toBe(THREE.DynamicDrawUsage);
    withoutRenderer.dispose();

    const cpu = new ParticleView({
      system: createCpuParticleSystem({ field: seededField(16) }),
      renderer: rig.renderer,
    });
    expect(cpu.mesh.instanceMatrix.usage).toBe(THREE.DynamicDrawUsage);
    cpu.dispose();
    rig.system.dispose();
  });

  it('starts on the CPU path with nothing to report', () => {
    const view = new ParticleView({ system: createCpuParticleSystem({ field: seededField(8) }) });
    expect(view.viewMode).toBe('cpu');
    expect(view.frameMode).toBe('cpu-upload');
    expect(view.ready).toBe(false);
    expect(view.blittedBytes).toBe(0);
    expect(view.gpuError).toBeNull();
    expect(view.radiusScale).toBe(1);
    expect(view.disposed).toBe(false);
    view.dispose();
  });

  it('recognises the GPU backend and nothing else', () => {
    const field = seededField(8);
    const cpu = new ParticleView({ system: createCpuParticleSystem({ field }) });
    expect(cpu.gpu).toBeNull();
    cpu.dispose();
  });

  it('takes injected geometry and material as given', () => {
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshBasicMaterial();
    const view = new ParticleView({
      system: createCpuParticleSystem({ field: seededField(4) }),
      geometry,
      material,
    });
    expect(view.mesh.geometry).toBe(geometry);
    expect(view.mesh.material).toBe(material);
    view.dispose();
    expect(geometry.type).toBe('BoxGeometry');
  });

  it('applies radiusScale to the first upload', () => {
    const field = seededField(4);
    const view = new ParticleView({
      system: createCpuParticleSystem({ field }),
      radiusScale: 3,
    });
    expect(view.radiusScale).toBe(3);
    expect((view.mesh.instanceMatrix.array as Float32Array)[0]).toBeCloseTo(field.radius(0) * 3, 6);
    view.dispose();
  });

  it('honours a colour for the default material', () => {
    const view = new ParticleView({
      system: createCpuParticleSystem({ field: seededField(4) }),
      color: 0xff8800,
    });
    const material = view.mesh.material as THREE.MeshLambertMaterial;
    expect(material.color.getHex()).toBe(0xff8800);
    view.dispose();
  });

  it('refuses a system it cannot draw', () => {
    expect(
      () => new ParticleView({ system: null as unknown as ParticleSystem }),
    ).toThrow(TypeError);
    const broken = { name: 'cpu', count: 0 } as unknown as ParticleSystem;
    expect(() => new ParticleView({ system: broken })).toThrow(TypeError);
    const fractional = { name: 'cpu', count: 2.5 } as unknown as ParticleSystem;
    expect(() => new ParticleView({ system: fractional })).toThrow(/positive count/);
  });

  it('refuses a radius scale that would draw nothing', () => {
    const system = createCpuParticleSystem({ field: seededField(4) });
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(
        () => new ParticleView({ system, radiusScale: bad }),
        `scale ${bad}`,
      ).toThrow(RangeError);
    }
  });
});

describe('ParticleView on the CPU tier', () => {
  it('never attaches, even with a renderer that has buffers', async () => {
    const device = new StubDevice({ limits: stubLimits() });
    const target = threeLikeTarget(device, 16);
    const renderer = new FakeRenderer({ buffer: target });
    const field = seededField(16);
    const system = createCpuParticleSystem({ field });
    const view = new ParticleView({ system, renderer });
    expect(view.gpu).toBeNull();
    expect(view.update()).toBe('cpu-upload');
    expect(await view.settled()).toBe('cpu');
    expect(view.update()).toBe('cpu-upload');
    expect(renderer.asked, 'a CPU-tier view never asks the backend for anything').toEqual([]);
    expect(device.submissions).toEqual([]);
    view.dispose();
  });

  it('is not fooled by a system that claims the gpu name', () => {
    const system = createCpuParticleSystem({ field: seededField(8) });
    const impostor = Object.create(system) as CpuParticleSystem;
    Object.assign(impostor, { name: 'gpu' });
    const view = new ParticleView({ system: impostor });
    expect(view.gpu, 'a name is not a publish buffer').toBeNull();
    expect(view.update()).toBe('cpu-upload');
    view.dispose();
  });

  it('is not fooled by a system with a publish buffer but no device', () => {
    const device = new StubDevice({ limits: stubLimits() });
    const buffer = device.createBuffer({ label: 'publish', size: 64, usage: U.STORAGE });
    const system = createCpuParticleSystem({ field: seededField(8) });
    const impostor = Object.create(system) as CpuParticleSystem;
    Object.assign(impostor, { name: 'gpu', publish: buffer, copyPublishedTo: () => 0 });
    const view = new ParticleView({ system: impostor, renderer: new FakeRenderer({ buffer }) });
    expect(view.gpu).toBeNull();
    view.dispose();
  });

  it('re-uploads every frame, so a stepped field is drawn where it moved to', () => {
    const field = seededField(8);
    const system = createCpuParticleSystem({ field });
    const view = new ParticleView({ system });
    const before = Array.from(view.mesh.instanceMatrix.array as Float32Array).slice(12, 15);
    system.step();
    expect(view.update()).toBe('cpu-upload');
    const after = Array.from(view.mesh.instanceMatrix.array as Float32Array).slice(12, 15);
    expect(after).not.toEqual(before);
    expect(view.blittedBytes).toBe(0);
    view.dispose();
  });

  it('draws fewer instances on request without resizing anything', () => {
    const field = seededField(16);
    const view = new ParticleView({ system: createCpuParticleSystem({ field }) });
    view.setCount(4);
    expect(view.count).toBe(4);
    expect(view.mesh.instanceMatrix.count, 'the attribute keeps its capacity').toBe(16);
    expect(view.update()).toBe('cpu-upload');
    // Only the first four instances were rewritten; the rest keep last frame's.
    const array = view.mesh.instanceMatrix.array as Float32Array;
    expect(array[4 * INSTANCE_FLOATS + 12]).toBeCloseTo(field.position(4)[0], 6);
    view.setCount(0);
    expect(view.update()).toBe('cpu-upload');
    view.dispose();
  });

  it('refuses a count outside the system', () => {
    const view = new ParticleView({ system: createCpuParticleSystem({ field: seededField(8) }) });
    for (const bad of [-1, 9, 1.5, Number.NaN]) {
      expect(() => view.setCount(bad), `count ${bad}`).toThrow(RangeError);
    }
    view.dispose();
  });

  it('changes the drawn radius without touching the simulation', () => {
    const field = seededField(4);
    const system = createCpuParticleSystem({ field });
    const view = new ParticleView({ system });
    const digest = system.digest();
    view.setRadiusScale(2.5);
    expect(view.radiusScale).toBe(2.5);
    view.update();
    expect((view.mesh.instanceMatrix.array as Float32Array)[0]).toBeCloseTo(field.radius(0) * 2.5, 6);
    expect(system.digest(), 'a render knob is not a simulation input').toBe(digest);
    expect(() => view.setRadiusScale(0)).toThrow(RangeError);
    expect(view.radiusScale).toBe(2.5);
    view.dispose();
  });
});

describe('ParticleView attaching to the GPU', () => {
  it('starts the attach on the first frame that sees a buffer, and keeps uploading', async () => {
    const rig = await gpuRig(128);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    expect(view.update(), 'the pipeline is not compiled yet').toBe('cpu-upload');
    expect(view.viewMode).toBe('cpu');
    expect(rig.renderer.asked).toContain(view.mesh.instanceMatrix);
    expect(await view.settled()).toBe('gpu');
    expect(view.ready).toBe(true);
    expect(view.gpuError).toBeNull();
    view.dispose();
    rig.system.dispose();
  });

  it('does not ask twice: one attach per view', async () => {
    const rig = await gpuRig(64);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    view.update();
    view.update();
    expect(await view.settled()).toBe('gpu');
    expect(rig.device.modules.filter((m) => m.code.includes('expand'))).toHaveLength(1);
    view.dispose();
    rig.system.dispose();
  });

  it('settles on cpu when there is no renderer at all', async () => {
    const rig = await gpuRig(32);
    const view = new ParticleView({ system: rig.system });
    expect(view.update()).toBe('cpu-upload');
    expect(await view.settled()).toBe('cpu');
    expect(view.gpu).not.toBeNull();
    view.dispose();
    rig.system.dispose();
  });

  it('settles on cpu while three.js has not created the attribute buffer', async () => {
    const rig = await gpuRig(32);
    rig.renderer.handle = undefined;
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    expect(view.update()).toBe('cpu-upload');
    expect(await view.settled()).toBe('cpu');
    // The buffer appears once the mesh has been rendered; the next frame attaches.
    rig.renderer.handle = { buffer: rig.target };
    expect(view.update()).toBe('cpu-upload');
    expect(await view.settled()).toBe('gpu');
    view.dispose();
    rig.system.dispose();
  });

  it('says so when three.js never exposes a destination buffer', async () => {
    const rig = await gpuRig(32);
    rig.renderer.handle = undefined;
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError, 'one frame is only "not rendered yet"').toBeNull();
    expect(view.update()).toBe('cpu-upload');
    expect(
      view.gpuError,
      'two frames means this module is looking somewhere three.js does not keep it',
    ).toMatch(/GPUBuffer/);
    expect(view.viewMode).toBe('cpu');
    const once = view.gpuError;
    view.update();
    view.update();
    expect(view.gpuError, 'one verdict, not one per frame').toBe(once);
    view.dispose();
    rig.system.dispose();
  });

  it('withdraws that verdict when the buffer arrives late', async () => {
    const rig = await gpuRig(32);
    rig.renderer.handle = undefined;
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    view.update();
    expect(view.gpuError).not.toBeNull();
    // A diagnostic that outlived its cause would be worse than none: `gpuError`
    // gates attachment, so a stale one pins a working page to the CPU path.
    rig.renderer.handle = { buffer: rig.target };
    expect(view.update()).toBe('cpu-upload');
    expect(view.gpuError).toBeNull();
    expect(await view.settled()).toBe('gpu');
    view.dispose();
    rig.system.dispose();
  });

  it('never blames three.js on a tier that was never going to blit', async () => {
    const rig = await gpuRig(16);
    const headless = new ParticleView({ system: rig.system });
    headless.update();
    headless.update();
    headless.update();
    expect(headless.gpuError, 'no renderer is a choice, not a failure').toBeNull();
    headless.dispose();

    const cpu = new ParticleView({
      system: createCpuParticleSystem({ field: seededField(16) }),
      renderer: new FakeRenderer(),
    });
    cpu.update();
    cpu.update();
    cpu.update();
    expect(cpu.gpuError).toBeNull();
    cpu.dispose();
    rig.system.dispose();
  });

  it('blits the matrices into the buffer three.js owns', async () => {
    const rig = await gpuRig(128);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();

    // The stub cannot run WGSL, so fill the expansion buffer by hand and check
    // that the copy moves exactly those bytes into the destination.
    const expansion = buf(rig.device, 'particle-view');
    const pattern = expansion.floats();
    for (let i = 0; i < pattern.length; i++) pattern[i] = i + 1;

    expect(view.update()).toBe('gpu-blit');
    expect(view.frameMode).toBe('gpu-blit');
    expect(view.blittedBytes).toBe(128 * INSTANCE_BYTES);
    expect(Array.from(rig.target.floats())).toEqual(Array.from(pattern));

    const copies = rig.device.submissions.at(-1)!.copies;
    expect(copies).toHaveLength(1);
    expect(copies[0]!.from).toBe(expansion);
    expect(copies[0]!.to).toBe(rig.target);
    expect(copies[0]!.bytes).toBe(128 * INSTANCE_BYTES);
    view.dispose();
    rig.system.dispose();
  });

  it('dispatches the expansion kernel once a frame, sized to the count', async () => {
    const rig = await gpuRig(1000);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    const passesBefore = rig.device.passes.length;
    view.update();
    const fresh = rig.device.passes.slice(passesBefore);
    expect(fresh).toHaveLength(1);
    expect(fresh[0]!.entryPoints).toEqual(['expand']);
    expect(fresh[0]!.dispatches).toEqual([Math.ceil(1000 / 64)]);
    expect(fresh[0]!.ended).toBe(true);
    view.dispose();
    rig.system.dispose();
  });

  it('does not ask three.js to re-upload the array it just bypassed', async () => {
    const rig = await gpuRig(64);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    const version = view.mesh.instanceMatrix.version;
    view.update();
    view.update();
    expect(view.mesh.instanceMatrix.version, 'a CPU upload would overwrite the blit').toBe(version);
    view.dispose();
    rig.system.dispose();
  });

  it('copies only what the destination holds, floored to whole instances', async () => {
    const rig = await gpuRig(128, 100);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    expect(view.update()).toBe('gpu-blit');
    expect(view.blittedBytes).toBe(64);
    view.dispose();
    rig.system.dispose();
  });

  it('falls back to the CPU when the destination cannot hold one instance', async () => {
    const rig = await gpuRig(128, 32);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    expect(view.ready).toBe(true);
    expect(view.update()).toBe('cpu-upload');
    expect(view.blittedBytes).toBe(0);
    expect(view.gpuError, 'a small buffer is not a device failure').toBeNull();
    view.dispose();
    rig.system.dispose();
  });

  it('follows setCount into a smaller blit', async () => {
    const rig = await gpuRig(128);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    view.setCount(32);
    expect(view.update()).toBe('gpu-blit');
    expect(view.blittedBytes).toBe(32 * INSTANCE_BYTES);
    view.dispose();
    rig.system.dispose();
  });

  it('carries radiusScale into the uniform, and a later change into it too', async () => {
    const rig = await gpuRig(64);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer, radiusScale: 2 });
    view.update();
    await view.settled();
    const params = buf(rig.device, 'particle-view:params');
    expect(params.floats()[INSTANCE_WORD.radiusScale]).toBe(2);
    expect(params.u32()[INSTANCE_WORD.count]).toBe(64);
    view.setRadiusScale(7.5);
    expect(params.floats()[INSTANCE_WORD.radiusScale]).toBe(7.5);
    view.dispose();
    rig.system.dispose();
  });

  it('takes a second reference on the shared device and gives it back on dispose', async () => {
    const rig = await gpuRig(32);
    const before = rig.shared.references;
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    expect(rig.shared.references).toBe(before + 1);
    view.dispose();
    expect(rig.shared.references).toBe(before);
    expect(buf(rig.device, 'particle-view').destroyed).toBe(true);
    expect(buf(rig.device, 'particle-view:params').destroyed).toBe(true);
    // `raw` is the structural `GpuBufferLike`, which has `destroy()` but not the
    // stub's `destroyed` flag. The rig builds every buffer on a `StubDevice`, so
    // naming the class here is the assertion rather than a weakening of it.
    expect(
      (rig.system.publish.raw as StubBuffer).destroyed,
      'the simulation outlives the renderer',
    ).toBe(false);
    rig.system.dispose();
  });

  it('drops an expander that finished building after the view was disposed', async () => {
    const rig = await gpuRig(32);
    const before = rig.shared.references;
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    view.dispose();
    expect(await view.settled()).toBe('cpu');
    await flushMicrotasks();
    expect(rig.shared.references, 'nothing may hold a device after teardown').toBe(before);
    expect(buf(rig.device, 'particle-view').destroyed).toBe(true);
    rig.system.dispose();
  });
});

describe('ParticleView when the GPU path fails', () => {
  it('reports a shader that will not compile and keeps drawing from the CPU', async () => {
    const rig = await gpuRig(64);
    // After the system is built: its own pipeline has to compile for the rig.
    rig.device.setCompilationMessages([
      { type: 'error', message: 'unknown identifier', lineNum: 12 },
    ]);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    expect(view.update()).toBe('cpu-upload');
    expect(await view.settled()).toBe('cpu');
    expect(view.gpuError).toMatch(/unknown identifier/);
    expect(view.ready).toBe(false);
    expect(view.mesh.instanceMatrix.version, 'the frame still drew').toBeGreaterThan(0);
    expect(
      rig.device.buffers.filter((b) => b.label.startsWith('particle-view')),
      'a failed create allocates nothing',
    ).toEqual([]);
    view.dispose();
    rig.system.dispose();
  });

  it('does not retry a failed attach on every frame', async () => {
    const rig = await gpuRig(64);
    rig.device.setCompilationMessages([{ type: 'error', message: 'bad syntax', lineNum: 3 }]);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    const modules = rig.device.modules.length;
    view.update();
    view.update();
    expect(rig.device.modules.length, 'a compile error is not transient').toBe(modules);
    view.dispose();
    rig.system.dispose();
  });

  it('keeps a device failure as the reason even if the buffer then disappears', async () => {
    const rig = await gpuRig(64);
    rig.device.setCompilationMessages([{ type: 'error', message: 'bad syntax', lineNum: 3 }]);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    expect(view.gpuError).toMatch(/bad syntax/);
    rig.renderer.handle = undefined;
    view.update();
    view.update();
    expect(
      view.gpuError,
      'the reason a caller can act on outranks the missing-buffer diagnostic',
    ).toMatch(/bad syntax/);
    view.dispose();
    rig.system.dispose();
  });

  it('reports a device that is already lost before the attach starts', async () => {
    const rig = await gpuRig(64);
    rig.device.lose({ reason: 'unknown', message: 'driver went away' });
    await flushMicrotasks();
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    expect(view.update()).toBe('cpu-upload');
    expect(await view.settled()).toBe('cpu');
    expect(view.gpuError).toMatch(/lost/);
    view.dispose();
    rig.system.dispose();
  });

  it('falls back mid-run when the device is lost after the attach', async () => {
    const rig = await gpuRig(64);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    expect(await view.settled()).toBe('gpu');
    expect(view.update()).toBe('gpu-blit');

    rig.device.lose({ reason: 'unknown', message: 'driver went away' });
    await flushMicrotasks();
    const submissions = rig.device.submissions.length;
    expect(view.update()).toBe('cpu-upload');
    expect(view.viewMode).toBe('cpu');
    expect(view.gpuError).toMatch(/lost/);
    expect(rig.device.submissions.length, 'nothing is submitted to a dead device').toBe(submissions);
    expect(buf(rig.device, 'particle-view').destroyed, 'the expander is dropped, not kept').toBe(
      true,
    );
    // And it stays down rather than thrashing.
    expect(view.update()).toBe('cpu-upload');
    view.dispose();
    rig.system.dispose();
  });

  it('reports a failure that is not an Error', async () => {
    const rig = await gpuRig(32);
    const device = rig.device as StubDevice & { createShaderModule: unknown };
    device.createShaderModule = (): never => {
      throw 'a driver threw a string';
    };
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    expect(await view.settled()).toBe('cpu');
    expect(view.gpuError).toBe('a driver threw a string');
    view.dispose();
    rig.system.dispose();
  });

  it('never throws out of update, whatever the device does', async () => {
    const rig = await gpuRig(32);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    rig.shared.destroy();
    expect(() => view.update()).not.toThrow();
    expect(view.frameMode).toBe('cpu-upload');
    expect(view.gpuError).toMatch(/destroyed/);
    // `destroy()` zeroes the refcount, so there is nothing left to release: the
    // view can still be torn down, but the documented order for the system is
    // dispose-before-destroy and this rig has already gone the other way.
    view.dispose();
    expect(view.disposed).toBe(true);
  });
});

describe('ParticleView teardown', () => {
  it('disposes the mesh and what it owns', () => {
    const view = new ParticleView({ system: createCpuParticleSystem({ field: seededField(8) }) });
    const geometryGone = onDispose(view.mesh.geometry);
    const materialGone = onDispose(view.mesh.material as THREE.Material);
    const meshGone = onDispose(view.mesh);
    view.dispose();
    expect(meshGone()).toBe(1);
    expect(geometryGone()).toBe(1);
    expect(materialGone()).toBe(1);
    expect(view.disposed).toBe(true);
    expect(view.viewMode).toBe('cpu');
  });

  it('leaves injected geometry and material alone', () => {
    const geometry = new THREE.SphereGeometry(1, 4, 3);
    const material = new THREE.MeshBasicMaterial();
    const geometryGone = onDispose(geometry);
    const materialGone = onDispose(material);
    const view = new ParticleView({
      system: createCpuParticleSystem({ field: seededField(4) }),
      geometry,
      material,
    });
    view.dispose();
    expect(geometryGone(), 'the caller owns what the caller passed').toBe(0);
    expect(materialGone()).toBe(0);
  });

  it('is idempotent', () => {
    const view = new ParticleView({ system: createCpuParticleSystem({ field: seededField(4) }) });
    const meshGone = onDispose(view.mesh);
    view.dispose();
    view.dispose();
    expect(meshGone()).toBe(1);
  });

  it('refuses use after dispose, which is a caller bug and not a device condition', async () => {
    const rig = await gpuRig(32);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    view.update();
    await view.settled();
    view.dispose();
    expect(() => view.update()).toThrow(/disposed/);
    expect(() => view.setCount(1)).toThrow(/disposed/);
    expect(() => view.setRadiusScale(2)).toThrow(/disposed/);
    rig.system.dispose();
  });
});

describe('the renderer never writes to the simulation', () => {
  it('leaves field.data byte-identical across CPU frames', () => {
    const field = seededField(32);
    const system = createCpuParticleSystem({ field });
    const view = new ParticleView({ system });
    const digest = system.digest();
    const bytes = Uint8Array.from(new Uint8Array(field.data.buffer.slice(0)));
    for (let i = 0; i < 5; i++) view.update();
    view.setRadiusScale(9);
    view.setCount(4);
    view.update();
    expect(system.digest()).toBe(digest);
    expect(new Uint8Array(field.data.buffer.slice(0))).toEqual(bytes);
    view.dispose();
  });

  it('leaves field.data byte-identical across GPU frames', async () => {
    const rig = await gpuRig(64);
    const view = new ParticleView({ system: rig.system, renderer: rig.renderer });
    const digest = rig.system.digest();
    const bytes = Uint8Array.from(new Uint8Array(rig.field.data.buffer.slice(0)));
    view.update();
    await view.settled();
    expect(view.update()).toBe('gpu-blit');
    expect(view.update()).toBe('gpu-blit');
    expect(rig.system.digest()).toBe(digest);
    expect(new Uint8Array(rig.field.data.buffer.slice(0))).toEqual(bytes);
    view.dispose();
    rig.system.dispose();
  });
});

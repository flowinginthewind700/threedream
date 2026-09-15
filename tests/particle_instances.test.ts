/**
 * `gpu/particleInstances.ts` -- the kernel that turns published particles into
 * instance matrices, driven by the recording stub.
 *
 * Nothing here can execute the WGSL: a stub device records calls, it does not run
 * shaders. So the spec pins the two things that are decisions rather than
 * arithmetic -- the *shape* of the generated shader (which bindings at which
 * slots, which column holds what, the count guard, the workgroup size) and the
 * *wiring* around it (buffer sizes and usages, dispatch sizing, what the copy
 * moves and where, and what teardown leaves behind). That the matrices come out
 * numerically right is asserted against a live GPU in `e2e/particles_gpu.spec.ts`,
 * and the CPU-side equivalent of the same layout is
 * `writeInstanceMatrices` in `render/particles.ts`, which the parity spec
 * compares against.
 */

import { describe, expect, it } from 'vitest';

import { ComputeContext, ShaderCompilationError } from '../src/gpu/compute.js';
import { SharedDeviceManager, type SharedDevice } from '../src/gpu/device.js';
import { ParticleField } from '../src/gpu/particleField.js';
import { createGpuParticleSystem } from '../src/gpu/particleGpu.js';
import {
  DEFAULT_RADIUS_SCALE,
  INSTANCE_BINDINGS,
  INSTANCE_BYTES,
  INSTANCE_ENTRY,
  INSTANCE_FLOATS,
  INSTANCE_GROUP,
  INSTANCE_PARAMS_BYTES,
  INSTANCE_PARAMS_WORDS,
  INSTANCE_VECS_PER_INSTANCE,
  INSTANCE_WORD,
  InstanceExpander,
  PUBLISH_VECS_PER_PARTICLE,
  instanceBufferBytes,
  instanceCopyBytes,
  instanceShaderSource,
  writeInstanceParams,
} from '../src/gpu/particleInstances.js';
import { PUBLISH_FLOATS_PER_PARTICLE, WORKGROUP_SIZE, workgroupsFor } from '../src/gpu/particleWgsl.js';
import {
  STUB_CONSTANTS as CONSTANTS,
  StubDevice,
  flushMicrotasks,
  stubLimits,
  type StubBuffer,
} from './stub_webgpu.js';

const U = CONSTANTS.bufferUsage;

// ---------------------------------------------------------------------------
// rig
// ---------------------------------------------------------------------------

/** A device, a manager and one adopted handle: the cheapest usable triple. */
function stubbed(limits: Record<string, number> = stubLimits()) {
  const device = new StubDevice({ limits });
  const manager = new SharedDeviceManager({ constants: CONSTANTS });
  return { device, manager, shared: manager.adopt(device) };
}

/** A buffer by label, with the labels in the error so a rename is diagnosable. */
function buf(device: StubDevice, label: string): StubBuffer {
  const found = device.buffers.find((b) => b.label === label);
  if (!found) {
    throw new Error(`no buffer labelled '${label}'; have [${device.buffers.map((b) => b.label)}]`);
  }
  return found;
}

/**
 * A stand-in for `GpuParticleSystem.publish`: same usage, same size, no
 * simulation behind it. The integration test below uses the real thing.
 */
function publishStub(shared: SharedDevice, count: number): { context: ComputeContext; buffer: StubBuffer } {
  const context = new ComputeContext(shared);
  context.storageBuffer(count * PUBLISH_FLOATS_PER_PARTICLE * 4, 'publish');
  return { context, buffer: buf(shared.device as StubDevice, 'publish') };
}

async function expanderFor(
  count = 128,
  over: { radiusScale?: number; limits?: Record<string, number>; label?: string } = {},
) {
  const { device, manager, shared } = stubbed(over.limits);
  const publish = publishStub(shared, count);
  const expander = await InstanceExpander.create({
    shared,
    count,
    source: publish.buffer,
    ...(over.radiusScale === undefined ? {} : { radiusScale: over.radiusScale }),
    ...(over.label === undefined ? {} : { label: over.label }),
  });
  return { device, manager, shared, publish, expander, count };
}

/** The destination three.js would hand over: `VERTEX | COPY_DST`, and nothing else. */
function threeLikeTarget(device: StubDevice, count: number, label = 'three-instanceMatrix'): StubBuffer {
  return device.createBuffer({ label, size: count * INSTANCE_BYTES, usage: U.VERTEX | U.COPY_DST });
}

// ---------------------------------------------------------------------------

describe('instance sizing', () => {
  it('a mat4 is sixteen floats and sixty-four bytes', () => {
    expect(INSTANCE_FLOATS).toBe(16);
    expect(INSTANCE_BYTES).toBe(64);
    expect(INSTANCE_VECS_PER_INSTANCE).toBe(4);
  });

  it('derives the publish stride instead of restating it', () => {
    expect(PUBLISH_VECS_PER_PARTICLE).toBe(PUBLISH_FLOATS_PER_PARTICLE / 4);
    expect(PUBLISH_VECS_PER_PARTICLE).toBe(1);
  });

  it('sizes the expansion buffer at 64 bytes a particle', () => {
    expect(instanceBufferBytes(1)).toBe(64);
    expect(instanceBufferBytes(100_000)).toBe(6_400_000);
  });

  it('refuses a count that cannot be dispatched', () => {
    for (const bad of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => instanceBufferBytes(bad), `count ${bad}`).toThrow(RangeError);
    }
  });

  it('caps the copy at the destination, because three.js owns that size', () => {
    expect(instanceCopyBytes(10, 640)).toBe(640);
    expect(instanceCopyBytes(10, 128)).toBe(128);
    expect(instanceCopyBytes(10, 0)).toBe(0);
  });

  it('floors to whole instances, since a partial matrix cannot be drawn', () => {
    expect(instanceCopyBytes(10, 130)).toBe(128);
    expect(instanceCopyBytes(10, 63)).toBe(0);
    expect(instanceCopyBytes(10, 130.7)).toBe(128);
    for (const target of [0, 63, 130, 640, 4096]) {
      expect(instanceCopyBytes(10, target) % INSTANCE_BYTES, `target ${target}`).toBe(0);
    }
  });

  it('refuses a destination size that is not a size', () => {
    expect(() => instanceCopyBytes(4, -1)).toThrow(RangeError);
    expect(() => instanceCopyBytes(4, Number.NaN)).toThrow(RangeError);
  });

  it('the uniform is four words, so it stays a round 16 bytes', () => {
    expect(INSTANCE_PARAMS_WORDS).toBe(4);
    expect(INSTANCE_PARAMS_BYTES).toBe(16);
    expect(Object.values(INSTANCE_WORD)).toEqual([0, 1, 2, 3]);
  });
});

describe('writeInstanceParams', () => {
  it('packs count as an integer and radiusScale as a float, in the declared words', () => {
    const words = new Float32Array(INSTANCE_PARAMS_WORDS);
    writeInstanceParams(words, 4096, 1.5);
    const ints = new Uint32Array(words.buffer);
    expect(ints[INSTANCE_WORD.count]).toBe(4096);
    expect(words[INSTANCE_WORD.radiusScale]).toBe(1.5);
    expect(words[INSTANCE_WORD.padA]).toBe(0);
    expect(words[INSTANCE_WORD.padB]).toBe(0);
  });

  it('returns the target, so seeding reads as one expression', () => {
    const words = new Float32Array(INSTANCE_PARAMS_WORDS);
    expect(writeInstanceParams(words, 8, 1)).toBe(words);
  });

  it('refuses a target too short to hold the struct', () => {
    expect(() => writeInstanceParams(new Float32Array(2), 8, 1)).toThrow(RangeError);
  });

  it('refuses a count or a scale that would silently draw nothing', () => {
    const words = new Float32Array(INSTANCE_PARAMS_WORDS);
    expect(() => writeInstanceParams(words, 0, 1)).toThrow(RangeError);
    expect(() => writeInstanceParams(words, 8, 0)).toThrow(RangeError);
    expect(() => writeInstanceParams(words, 8, -1)).toThrow(RangeError);
    expect(() => writeInstanceParams(words, 8, Number.NaN)).toThrow(RangeError);
  });
});

describe('the generated shader', () => {
  const source = instanceShaderSource();

  it('is memoised, because the text is constant for the process', () => {
    expect(instanceShaderSource()).toBe(source);
  });

  it('declares one entry point at the portable workgroup size', () => {
    expect(source).toContain(`@compute @workgroup_size(${WORKGROUP_SIZE})`);
    expect(source).toContain(`fn ${INSTANCE_ENTRY}(`);
    expect(WORKGROUP_SIZE).toBe(64);
    expect(source.match(/@compute/g)?.length, 'one kernel, one @compute').toBe(1);
  });

  it('uses no subgroup feature, so it runs on an adapter with an empty feature set', () => {
    expect(source).not.toMatch(/subgroup/);
  });

  it('binds publish read-only at 0, the expansion buffer read-write at 1, params at 2', () => {
    expect(source).toContain(
      `@group(${INSTANCE_GROUP}) @binding(0) var<storage, read> publishBuf: array<vec4<f32>>;`,
    );
    expect(source).toContain(
      `@group(${INSTANCE_GROUP}) @binding(1) var<storage, read_write> instanceBuf: array<vec4<f32>>;`,
    );
    expect(source).toContain(`@group(${INSTANCE_GROUP}) @binding(2) var<uniform> params: InstanceParams;`);
    // The layout entries the pipeline is built from have to say the same thing,
    // or the bind group is invalid and the failure arrives at dispatch time.
    expect(INSTANCE_BINDINGS).toEqual([
      { group: 0, binding: 0, name: 'publishBuf', bufferType: 'read-only-storage' },
      { group: 0, binding: 1, name: 'instanceBuf', bufferType: 'storage' },
      { group: 0, binding: 2, name: 'params', bufferType: 'uniform' },
    ]);
  });

  it('declares the uniform members in the order INSTANCE_WORD indexes them', () => {
    const body = source.slice(source.indexOf('struct InstanceParams'));
    const members = body.slice(0, body.indexOf('};')).match(/^\s+(\w+):/gm)?.map((m) => m.trim().replace(':', ''));
    expect(members).toEqual(['count', 'radiusScale', 'padA', 'padB']);
    expect(members?.indexOf('count')).toBe(INSTANCE_WORD.count);
    expect(members?.indexOf('radiusScale')).toBe(INSTANCE_WORD.radiusScale);
    expect(body).toContain('count: u32');
    expect(body).toContain('radiusScale: f32');
  });

  it('guards the tail of the last workgroup', () => {
    expect(source).toContain('if (i >= params.count) { return; }');
  });

  it('writes the matrix column by column: scale on the diagonal, translation last', () => {
    // Column major, matching three.js InstancedBufferAttribute. A transposed
    // write is the bug this pins: it renders as spheres stretched along the
    // wrong axis and, with the translation in the top row, as nothing at all.
    const writes = [...source.matchAll(/instanceBuf\[base(?: \+ (\d)u)?\] = vec4<f32>\(([^)]*)\);/g)].map(
      (m) => [Number(m[1] ?? 0), m[2].replace(/\s+/g, ' ')] as const,
    );
    expect(writes).toEqual([
      [0, 'r, 0.0, 0.0, 0.0'],
      [1, '0.0, r, 0.0, 0.0'],
      [2, '0.0, 0.0, r, 0.0'],
      [3, 'published.xyz, 1.0'],
    ]);
    expect(source).toContain(`let base = i * ${INSTANCE_VECS_PER_INSTANCE}u;`);
    expect(source).toContain('let r = published.w * params.radiusScale;');
  });

  it('reads one published vec4 per particle', () => {
    expect(source).toContain('let published = publishBuf[i];');
    expect(PUBLISH_VECS_PER_PARTICLE).toBe(1);
  });
});

describe('InstanceExpander allocation', () => {
  it('allocates a uniform and a 64-byte-an-instance storage buffer', async () => {
    const { device, expander, count } = await expanderFor(256);
    const params = buf(device, 'particle-instances:params');
    const instances = buf(device, 'particle-instances');
    expect(params.size).toBe(INSTANCE_PARAMS_BYTES);
    expect(params.usage).toBe(U.UNIFORM | U.COPY_DST);
    expect(instances.size).toBe(count * INSTANCE_BYTES);
    expect(instances.usage).toBe(U.STORAGE | U.COPY_DST | U.COPY_SRC);
    expect(expander.bytes).toBe(instances.size);
    expect(expander.instances.raw).toBe(instances);
  });

  it('honours a label, so two expanders on one device stay tellable apart', async () => {
    const { device } = await expanderFor(64, { label: 'dust' });
    expect(buf(device, 'dust').size).toBe(64 * INSTANCE_BYTES);
    expect(buf(device, 'dust:params').size).toBe(INSTANCE_PARAMS_BYTES);
  });

  it('compiles one module with one declared entry point', async () => {
    const { device } = await expanderFor(64);
    expect(device.modules.length).toBe(1);
    expect(device.modules[0]!.label).toBe('particle-instances');
    expect(device.modules[0]!.code).toBe(instanceShaderSource());
  });

  it('binds the publish buffer at 0, its own at 1, the uniform at 2', async () => {
    const { device, publish, expander } = await expanderFor(64);
    expect(device.bindGroups.length).toBe(1);
    const group = device.bindGroups[0]!;
    expect(group.entryAt(0)?.buffer).toBe(publish.buffer);
    expect(group.entryAt(1)?.buffer).toBe(expander.instances.raw);
    expect(group.entryAt(2)?.buffer).toBe(buf(device, 'particle-instances:params'));
  });

  it('uploads the uniform at construction, so the first frame is already sized', async () => {
    const { device } = await expanderFor(32, { radiusScale: 2 });
    const params = buf(device, 'particle-instances:params');
    expect(device.writes.some((w) => w.buffer === params)).toBe(true);
    expect(params.u32()[INSTANCE_WORD.count]).toBe(32);
    expect(params.floats()[INSTANCE_WORD.radiusScale]).toBe(2);
  });

  it('defaults the radius scale to 1', async () => {
    const { expander } = await expanderFor(16);
    expect(expander.radiusScale).toBe(DEFAULT_RADIUS_SCALE);
    expect(DEFAULT_RADIUS_SCALE).toBe(1);
  });

  it('takes its refcount on the shared device and gives it back on dispose', async () => {
    const { shared, expander } = await expanderFor(16);
    expect(shared.references).toBe(2);
    expander.dispose();
    expect(shared.references).toBe(1);
    expect(shared.usable).toBe(true);
  });

  it('destroys its own buffers on dispose, and never the publish buffer', async () => {
    const { device, publish, expander } = await expanderFor(16);
    expander.dispose();
    expect(buf(device, 'particle-instances').destroyed).toBe(true);
    expect(buf(device, 'particle-instances:params').destroyed).toBe(true);
    // The simulation owns that one. Destroying it here would take the system
    // down with the renderer, on a page that is only redrawing.
    expect(publish.buffer.destroyed).toBe(false);
    expect(expander.disposed).toBe(true);
  });

  it('dispose is idempotent, and a second call does not over-release', async () => {
    const { device, shared, expander } = await expanderFor(16);
    expander.dispose();
    expander.dispose();
    expect(shared.references).toBe(1);
    expect(buf(device, 'particle-instances').destroyCalls).toBe(1);
  });

  it('refuses a count that cannot be dispatched, before touching the device', async () => {
    const { device, manager, shared } = stubbed();
    const publish = publishStub(shared, 4);
    for (const bad of [0, -3, 4.5]) {
      await expect(
        InstanceExpander.create({ shared, count: bad, source: publish.buffer }),
        `count ${bad}`,
      ).rejects.toThrow(RangeError);
    }
    expect(device.buffers.filter((b) => b.label.startsWith('particle-instances'))).toEqual([]);
    expect(shared.references).toBe(1);
    shared.release();
  });

  it('refuses a device whose storage limit cannot hold the matrices', async () => {
    const count = 1024;
    const { shared } = stubbed(stubLimits({ maxStorageBufferBindingSize: count * INSTANCE_BYTES - 4 }));
    const publish = publishStub(shared, count);
    await expect(InstanceExpander.create({ shared, count, source: publish.buffer })).rejects.toThrow(
      /above this device's \d+-byte storage limit/,
    );
  });

  it('takes maxBufferSize into account when it is the tighter of the two', async () => {
    const count = 1024;
    const { shared } = stubbed(
      stubLimits({ maxStorageBufferBindingSize: 1 << 30, maxBufferSize: count * INSTANCE_BYTES - 4 }),
    );
    const publish = publishStub(shared, count);
    await expect(InstanceExpander.create({ shared, count, source: publish.buffer })).rejects.toThrow(
      RangeError,
    );
  });

  it('refuses a device that is already gone', async () => {
    const { shared } = stubbed();
    const publish = publishStub(shared, 8);
    shared.destroy();
    await expect(
      InstanceExpander.create({ shared, count: 8, source: publish.buffer }),
    ).rejects.toThrow(/destroyed/);
  });

  it('leaves nothing allocated when the shader fails to compile', async () => {
    const device = new StubDevice({ limits: stubLimits() });
    const manager = new SharedDeviceManager({ constants: CONSTANTS });
    const shared = manager.adopt(device);
    const publish = publishStub(shared, 8);
    device.setCompilationMessages([{ type: 'error', message: 'unknown identifier', lineNum: 12 }]);
    const error = await InstanceExpander.create({ shared, count: 8, source: publish.buffer }).catch(
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(ShaderCompilationError);
    expect((error as ShaderCompilationError).label).toBe('particle-instances');
    expect(device.buffers.filter((b) => b.label.startsWith('particle-instances'))).toEqual([]);
    expect(shared.references, 'a failed create must not keep a reference').toBe(1);
    expect(publish.buffer.destroyed).toBe(false);
    shared.release();
  });

  it('binds the real publish buffer of a real GPU system', async () => {
    const { device, shared } = stubbed();
    const field = new ParticleField({ count: 96, scene: 'sphere', seed: 3, radius: [0.05, 0.05] });
    const system = await createGpuParticleSystem({ shared, field });
    const expander = await InstanceExpander.create({
      shared,
      count: system.count,
      source: system.publish,
    });
    expect(buf(device, 'publish').size).toBe(system.count * PUBLISH_FLOATS_PER_PARTICLE * 4);
    expect(expander.bytes).toBe(system.count * INSTANCE_BYTES);
    // The system built two groups of its own first; the expander's is the one
    // that binds the expansion buffer at slot 1.
    const group = device.bindGroups.find((g) => g.entryAt(1)?.buffer === expander.instances.raw);
    expect(group, 'the expander bound a group').toBeDefined();
    expect(group!.entryAt(0)?.buffer).toBe(buf(device, 'publish'));
    expect(group!.entryAt(2)?.buffer).toBe(buf(device, 'particle-instances:params'));
    system.dispose();
    expander.dispose();
  });
});

describe('InstanceExpander dispatch', () => {
  it('covers the count in ceil(count / 64) workgroups', async () => {
    const { expander, count } = await expanderFor(1000);
    expect(expander.workgroups).toBe(workgroupsFor(count));
    expect(expander.workgroups).toBe(16);
  });

  it('expand records one pass, one pipeline and one dispatch', async () => {
    const { device, expander } = await expanderFor(1000);
    expect(expander.expand()).toBe(1);
    expect(device.passes.length).toBe(1);
    const pass = device.lastPass!;
    expect(pass.entryPoints).toEqual([INSTANCE_ENTRY]);
    expect(pass.dispatches).toEqual([16]);
    expect(pass.ended).toBe(true);
    expect(pass.calls.filter((c) => c.kind === 'bindGroup').map((c) => c.group)).toEqual([INSTANCE_GROUP]);
  });

  it('names the pass, so a frame capture says which layer it is in', async () => {
    const { device, expander } = await expanderFor(64);
    expander.expand('frame 7');
    expect(device.lastPass!.label).toBe('frame 7');
  });

  it('refuses a dispatch the device cannot express', async () => {
    const { expander } = await expanderFor(512, { limits: stubLimits({ maxComputeWorkgroupsPerDimension: 4 }) });
    expect(expander.workgroups).toBe(8);
    expect(() => expander.expand()).toThrow(/maxComputeWorkgroupsPerDimension=4/);
  });

  it('re-uploads the uniform when the radius scale changes', async () => {
    const { device, expander } = await expanderFor(32);
    const before = device.writes.length;
    expander.setRadiusScale(3.5);
    expect(expander.radiusScale).toBe(3.5);
    expect(device.writes.length).toBe(before + 1);
    expect(buf(device, 'particle-instances:params').floats()[INSTANCE_WORD.radiusScale]).toBe(3.5);
  });

  it('refuses a radius scale that would collapse or invert every sphere', async () => {
    const { expander } = await expanderFor(32);
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => expander.setRadiusScale(bad), `scale ${bad}`).toThrow(RangeError);
    }
    expect(expander.radiusScale).toBe(1);
  });

  it('throws after dispose rather than submitting to dead buffers', async () => {
    const { expander } = await expanderFor(32);
    expander.dispose();
    expect(() => expander.expand()).toThrow(/disposed/);
    expect(() => expander.setRadiusScale(2)).toThrow(/disposed/);
  });

  it('throws once the device is lost, and says so rather than submitting', async () => {
    const { device, expander } = await expanderFor(32);
    device.lose({ reason: 'unknown', message: 'driver went away' });
    await flushMicrotasks();
    expect(expander.lost).toBe(true);
    expect(() => expander.expand()).toThrow(/lost/);
    expect(device.submissions.length).toBe(0);
  });
});

describe('InstanceExpander blit', () => {
  it('copies the whole expansion into the renderer-owned buffer', async () => {
    const { device, expander, count } = await expanderFor(128);
    const target = threeLikeTarget(device, count);
    // The stub applies copies at `finish()`, so pre-filling the source is what
    // makes the destination checkable without executing the shader.
    const matrices = expander.instances.raw as StubBuffer;
    matrices.floats().set(Array.from({ length: count * INSTANCE_FLOATS }, (_, i) => i % 977));

    expect(expander.expandTo(target)).toBe(count * INSTANCE_BYTES);

    const copies = device.submissions.flatMap((cb) => cb.copies);
    expect(copies.length).toBe(1);
    expect(copies[0]!.from).toBe(matrices);
    expect(copies[0]!.to).toBe(target);
    expect(copies[0]!.bytes).toBe(count * INSTANCE_BYTES);
    expect(copies[0]!.fromOffset).toBe(0);
    expect(copies[0]!.toOffset).toBe(0);
    expect(Array.from(target.floats().slice(0, INSTANCE_FLOATS))).toEqual(
      Array.from(matrices.floats().slice(0, INSTANCE_FLOATS)),
    );
  });

  it('dispatches and copies in that order, in two submissions', async () => {
    const { device, expander, count } = await expanderFor(64);
    const target = threeLikeTarget(device, count);
    expander.expandTo(target);
    expect(device.submissions.length).toBe(2);
    expect(device.submissions[0]!.passes.length).toBe(1);
    expect(device.submissions[0]!.copies.length).toBe(0);
    expect(device.submissions[1]!.passes.length).toBe(0);
    expect(device.submissions[1]!.copies.length).toBe(1);
  });

  it('copies only what the destination holds', async () => {
    const { device, expander, count } = await expanderFor(64);
    const small = device.createBuffer({
      label: 'three-instanceMatrix',
      size: 8 * INSTANCE_BYTES,
      usage: U.VERTEX | U.COPY_DST,
    });
    expect(expander.expandTo(small, 8 * INSTANCE_BYTES)).toBe(8 * INSTANCE_BYTES);
    expect(device.submissions.flatMap((cb) => cb.copies)[0]!.bytes).toBe(8 * INSTANCE_BYTES);
    expect(count).toBe(64);
  });

  it('ignores a targetBytes larger than its own buffer, since that is all it wrote', async () => {
    const { device, expander, count } = await expanderFor(16);
    const target = threeLikeTarget(device, count);
    expect(expander.expandTo(target, 1 << 20)).toBe(count * INSTANCE_BYTES);
  });

  it('submits nothing at all when the destination cannot hold one instance', async () => {
    const { device, expander } = await expanderFor(64);
    const target = threeLikeTarget(device, 64);
    expect(expander.expandTo(target, 0)).toBe(0);
    expect(device.submissions.length, 'no pass and no copy for a frame that cannot be drawn').toBe(0);
  });

  it('accepts a wrapped buffer and a buffer view alike', async () => {
    const { device, expander, count } = await expanderFor(32);
    const target = threeLikeTarget(device, count);
    expect(expander.expandTo({ buffer: target })).toBe(count * INSTANCE_BYTES);
    expect(expander.expandTo({ buffer: target, offset: 0, size: count * INSTANCE_BYTES })).toBe(
      count * INSTANCE_BYTES,
    );
    expect(device.submissions.flatMap((cb) => cb.copies).length).toBe(2);
  });

  it('throws after dispose, before recording anything', async () => {
    const { device, expander, count } = await expanderFor(32);
    const target = threeLikeTarget(device, count);
    expander.dispose();
    expect(() => expander.expandTo(target)).toThrow(/disposed/);
    expect(device.submissions.length).toBe(0);
  });
});

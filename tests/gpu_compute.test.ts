/**
 * The compute layer, driven by the recording stub.
 *
 * What is pinned here is the part a real GPU hides: that a shader which fails
 * to compile is reported as a compile error rather than turning up later as a
 * validation failure, that a dispatch above the workgroup limit is rejected at
 * the call site, that readback copies bytes out and leaves nothing mapped, and
 * that teardown is idempotent. `e2e/particles_gpu.spec.ts` covers the same code
 * against a real adapter, where none of these failure modes can be produced on
 * cue.
 */

import { describe, expect, it } from 'vitest';

import {
  ComputeBuffer,
  ComputeContext,
  ComputeProgram,
  PingPong,
  ShaderCompilationError,
  rawBuffer,
  submitCopy,
  type ComputeBinding,
  type ComputeDispatch,
  type ComputeProgramOptions,
} from '../src/gpu/compute.js';
import { SharedDeviceManager, type SharedDevice } from '../src/gpu/device.js';
import {
  STUB_CONSTANTS as CONSTANTS,
  StubDevice,
  stubLimits,
  type StubBindGroup,
} from './stub_webgpu.js';

const U = CONSTANTS.bufferUsage;

interface Rig {
  readonly ctx: ComputeContext;
  readonly device: StubDevice;
  readonly shared: SharedDevice;
  readonly manager: SharedDeviceManager;
}

/** A context over an adopted stub device, which is the cheapest usable pair. */
function rig(limits: Record<string, number> = stubLimits()): Rig {
  const device = new StubDevice({ limits });
  const manager = new SharedDeviceManager({ constants: CONSTANTS });
  const shared = manager.adopt(device);
  return { ctx: new ComputeContext(shared), device, shared, manager };
}

/** Two groups, mirroring the particle layout: static resources plus a state pair. */
const BINDINGS: readonly ComputeBinding[] = [
  { group: 0, binding: 0, bufferType: 'uniform', name: 'params' },
  { group: 0, binding: 1, bufferType: 'storage', name: 'statsBuf' },
  { group: 1, binding: 0, bufferType: 'read-only-storage', name: 'stateSrc' },
  { group: 1, binding: 1, bufferType: 'storage', name: 'stateDst' },
];

const ENTRY_POINTS = ['main', 'clear'];

function programOptions(over: Partial<ComputeProgramOptions> = {}): ComputeProgramOptions {
  return {
    label: 'test-program',
    code: '@compute @workgroup_size(64) fn main() {}',
    entryPoints: ENTRY_POINTS,
    bindings: BINDINGS,
    ...over,
  };
}

/** A program plus the four buffers its groups need, so `submit` cases stay short. */
async function bound(ctx: ComputeContext) {
  const program = await ctx.program(programOptions());
  const params = ctx.uniformBuffer(96, 'params');
  const stats = ctx.storageBuffer(20, 'stats');
  const src = ctx.storageBuffer(64, 'src');
  const dst = ctx.storageBuffer(64, 'dst');
  return {
    program,
    params,
    stats,
    src,
    dst,
    staticGroup: program.bindGroup(0, { 0: params, 1: stats }) as StubBindGroup,
    stateGroup: program.bindGroup(1, { 0: src, 1: dst }) as StubBindGroup,
  };
}

// ---------------------------------------------------------------------------

describe('ComputeContext buffer factories', () => {
  it('builds a storage buffer that can be written, read back and copied out', () => {
    const { ctx, device } = rig();
    const buffer = ctx.storageBuffer(1024, 'state');
    expect(buffer.bytes).toBe(1024);
    expect(buffer.label).toBe('state');
    expect(buffer.usage).toBe(U.STORAGE | U.COPY_DST | U.COPY_SRC);
    expect(buffer.raw).toBe(device.buffers[0]);
    expect(ctx.buffers).toEqual([buffer]);
  });

  it('separates the uniform, staging and upload usages', () => {
    const { ctx } = rig();
    expect(ctx.uniformBuffer(96, 'p').usage).toBe(U.UNIFORM | U.COPY_DST);
    expect(ctx.stagingBuffer(16, 's').usage).toBe(U.MAP_READ | U.COPY_DST);
    expect(ctx.uploadBuffer(16, 'u').usage).toBe(U.MAP_WRITE | U.COPY_SRC);
  });

  it('tracks what it allocates, except the staging buffer it owns itself', () => {
    const { ctx } = rig();
    const kept = [ctx.storageBuffer(16), ctx.uniformBuffer(16), ctx.uploadBuffer(16)];
    ctx.stagingBuffer(16);
    expect(ctx.buffers).toEqual(kept);
  });

  it('refuses a size WebGPU would reject, before creating anything', () => {
    const { ctx, device } = rig();
    expect(() => ctx.storageBuffer(0)).toThrow(RangeError);
    expect(() => ctx.storageBuffer(-4)).toThrow(/positive/);
    expect(() => ctx.storageBuffer(6)).toThrow(/multiple of 4/);
    expect(() => ctx.storageBuffer(4.5)).toThrow(/not an integer/);
    expect(() => ctx.storageBuffer(Number.NaN)).toThrow(RangeError);
    expect(device.buffers).toHaveLength(0);
  });

  it('destroys every tracked buffer once, and refuses to allocate afterwards', () => {
    const { ctx, device } = rig();
    expect(ctx.device).toBe(device);
    expect(ctx.constants).toBe(CONSTANTS);
    const a = ctx.storageBuffer(16, 'a');
    const b = ctx.uniformBuffer(16, 'b');
    ctx.destroy();
    expect(a.destroyed && b.destroyed).toBe(true);
    expect(device.buffers.filter((buffer) => buffer.destroyed)).toHaveLength(2);
    expect(ctx.destroyed).toBe(true);
    ctx.destroy();
    expect(() => ctx.storageBuffer(16)).toThrow(/destroyed/);
    expect(() => ctx.program(programOptions())).toThrow(/destroyed/);
  });

  it('will not wrap a device that is already gone', () => {
    const device = new StubDevice();
    const shared = new SharedDeviceManager({ constants: CONSTANTS }).adopt(device);
    shared.destroy();
    expect(() => new ComputeContext(shared)).toThrow(/destroyed/);
  });
});

describe('ComputeBuffer.write', () => {
  it('lands the bytes in the buffer, at an offset when asked', () => {
    const { ctx, device } = rig();
    const buffer = ctx.storageBuffer(16, 'data');
    buffer.write(new Float32Array([1, 2]));
    expect(Array.from(device.buffers[0].floats())).toEqual([1, 2, 0, 0]);
    buffer.write(new Float32Array([9]), 8);
    expect(Array.from(device.buffers[0].floats())).toEqual([1, 2, 9, 0]);
    expect(device.writes).toEqual([
      { buffer: device.buffers[0], offset: 0, bytes: 8 },
      { buffer: device.buffers[0], offset: 8, bytes: 4 },
    ]);
    expect(buffer.write(new Float32Array([1]))).toBe(buffer);
  });

  it('names the buffer when a write would overrun it', () => {
    const { ctx } = rig();
    const buffer = ctx.storageBuffer(8, 'small');
    expect(() => buffer.write(new Float32Array([1, 2, 3]))).toThrow(/small: writing 12 bytes at 0/);
    expect(() => buffer.write(new Float32Array([1]), 8)).toThrow(/overruns the 8-byte buffer/);
    expect(() => buffer.write(new Float32Array([1]), 3)).toThrow(/not a multiple of 4/);
    expect(() => buffer.write(new Float32Array([1]), -4)).toThrow(/non-negative integer/);
  });

  it('rejects a write after destroy, and a write to a buffer with no COPY_DST', () => {
    const { ctx } = rig();
    const buffer = ctx.storageBuffer(16, 'gone');
    buffer.destroy();
    buffer.destroy();
    expect(() => buffer.write(new Float32Array([1]))).toThrow(/destroyed buffer/);
    // The upload buffer is the one made without COPY_DST: it is mapped, not
    // written. The staging buffer has COPY_DST, since it is a copy destination.
    expect(() => ctx.uploadBuffer(16, 'upload').write(new Float32Array([1]))).toThrow(
      /without COPY_DST/,
    );
    expect(ctx.stagingBuffer(16, 'stage').usage & CONSTANTS.bufferUsage.COPY_DST).not.toBe(0);
  });

  it('unwraps to the raw buffer from any of the three resource shapes', () => {
    const { ctx, device } = rig();
    const wrapped = ctx.storageBuffer(16, 'wrapped');
    expect(wrapped).toBeInstanceOf(ComputeBuffer);
    expect(rawBuffer(wrapped)).toBe(wrapped.raw);
    expect(rawBuffer(wrapped.raw)).toBe(wrapped.raw);
    expect(rawBuffer({ buffer: wrapped, offset: 4 })).toBe(wrapped.raw);
    expect(rawBuffer({ buffer: device.buffers[0] })).toBe(device.buffers[0]);
  });
});

describe('ComputeProgram.create', () => {
  it('compiles, then lays out one bind-group layout per group in group order', async () => {
    const { ctx, device } = rig();
    const program = await ctx.program(programOptions());
    expect(program.groups).toEqual([0, 1]);
    expect(device.modules).toHaveLength(1);
    expect(device.layouts).toHaveLength(2);
    expect(device.layouts.map((l) => l.label)).toEqual([
      'test-program:group0',
      'test-program:group1',
    ]);
    expect(device.layouts[0].entries).toEqual([
      { binding: 0, visibility: CONSTANTS.shaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: CONSTANTS.shaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);
    expect(device.layouts[1].entries[0]).toEqual({
      binding: 0,
      visibility: CONSTANTS.shaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' },
    });
    expect(program.layout).toBeDefined();
    expect(program.entryPoints).toEqual(ENTRY_POINTS);
  });

  it('sorts bindings by index, so a declaration written out of order still lays out in order', async () => {
    const { ctx, device } = rig();
    await ctx.program(
      programOptions({
        bindings: [
          { group: 0, binding: 2, bufferType: 'storage' },
          { group: 0, binding: 0, bufferType: 'uniform' },
        ],
      }),
    );
    expect(device.layouts[0].entries.map((e) => e.binding)).toEqual([0, 2]);
  });

  it('turns a compilation error into a ShaderCompilationError carrying the messages', async () => {
    const { ctx, device } = rig();
    device.setCompilationMessages([
      { type: 'warning', lineNum: 1, message: 'unused variable' },
      { type: 'error', lineNum: 12, message: "unknown identifier 'damping'" },
    ]);
    const caught: unknown = await ctx.program(programOptions()).then(
      () => null,
      (reason: unknown) => reason,
    );
    expect(caught).toBeInstanceOf(ShaderCompilationError);
    const error = caught as ShaderCompilationError;
    expect(error.name).toBe('ShaderCompilationError');
    expect(error.label).toBe('test-program');
    expect(error.messages).toHaveLength(1);
    expect(error.message).toContain('line 12');
    expect(error.message).toContain("unknown identifier 'damping'");
    expect(error.message).not.toContain('unused variable');
  });

  it('accepts a module that only warns', async () => {
    const { ctx, device } = rig();
    device.setCompilationMessages([{ type: 'warning', lineNum: 3, message: 'narrowing' }]);
    await expect(ctx.program(programOptions())).resolves.toBeInstanceOf(ComputeProgram);
  });

  it('refuses bind groups that do not start at zero, or that exceed the limit', async () => {
    const gapOnly = rig();
    await expect(
      gapOnly.ctx.program(
        programOptions({ bindings: [{ group: 1, binding: 0, bufferType: 'storage' }] }),
      ),
    ).rejects.toThrow(/must start at 0 and be contiguous/);

    const sparse = rig();
    await expect(
      sparse.ctx.program(
        programOptions({
          bindings: [
            { group: 0, binding: 0, bufferType: 'storage' },
            { group: 2, binding: 0, bufferType: 'storage' },
          ],
        }),
      ),
    ).rejects.toThrow(/\[0, 2\]/);

    const tight = rig(stubLimits({ maxBindGroups: 1 }));
    await expect(tight.ctx.program(programOptions())).rejects.toThrow(
      /2 bind groups exceeds maxBindGroups=1/,
    );
  });

  it('refuses to compile on a device that is gone', async () => {
    const { ctx, shared } = rig();
    shared.destroy();
    // The context checks the handle before it starts compiling, so this is a
    // synchronous throw rather than a rejection.
    expect(() => ctx.program(programOptions())).toThrow(/shared device is destroyed/);
    await expect(ComputeProgram.create(shared, programOptions())).rejects.toThrow(/destroyed/);
  });
});

describe('ComputeProgram.pipeline', () => {
  it('creates one pipeline per entry point and keeps it', async () => {
    const { ctx, device } = rig();
    const program = await ctx.program(programOptions());
    const first = program.pipeline('main');
    expect(program.pipeline('main')).toBe(first);
    expect(program.pipeline('clear')).not.toBe(first);
    expect(program.pipeline('clear')).toBe(program.pipeline('clear'));
    expect(device.pipelines.map((p) => p.entryPoint)).toEqual(['main', 'clear']);
    expect(device.pipelines.map((p) => p.label)).toEqual([
      'test-program:main',
      'test-program:clear',
    ]);
    expect(device.pipelines[0].module).toBe(program.module);
    expect(device.pipelines[0].layout).toBeDefined();
  });

  it('names the declared entry points when asked for one that is not there', async () => {
    const { ctx } = rig();
    const program = await ctx.program(programOptions());
    expect(() => program.pipeline('nbody')).toThrow(/'nbody' is not a declared entry point/);
    expect(() => program.pipeline('nbody')).toThrow(/main, clear/);
  });
});

describe('ComputeProgram.bindGroup', () => {
  it('binds each declared binding to its buffer, in binding order', async () => {
    const { ctx, device } = rig();
    const { staticGroup, params, stats } = await bound(ctx);
    expect(device.bindGroups[0]).toBe(staticGroup);
    expect(staticGroup.entries.map((e) => e.binding)).toEqual([0, 1]);
    expect(staticGroup.bufferAt(0)).toBe(params.raw);
    expect(staticGroup.bufferAt(1)).toBe(stats.raw);
    expect(staticGroup.layout).toBe(device.layouts[0]);
  });

  it('passes an explicit range through, and labels the group when asked', async () => {
    const { ctx, device } = rig();
    const program = await ctx.program(programOptions());
    const big = ctx.storageBuffer(64, 'big');
    const group = program.bindGroup(
      0,
      { 0: { buffer: big, offset: 0, size: 16 }, 1: { buffer: big, offset: 16, size: 32 } },
      'custom',
    );
    const entry = device.bindGroups[0].entryAt(1);
    expect(entry?.buffer).toBe(big.raw);
    expect(entry?.offset).toBe(16);
    expect(entry?.size).toBe(32);
    expect(group).toBeDefined();
    expect(device.bindGroups[0].entryAt(0)?.offset).toBe(0);
  });

  it('lists every missing binding, by index and by name', async () => {
    const { ctx } = rig();
    const program = await ctx.program(programOptions());
    const stats = ctx.storageBuffer(20, 'stats');
    expect(() => program.bindGroup(0, { 1: stats })).toThrow(/missing binding 0 \(params\)/);
    expect(() => program.bindGroup(0, {})).toThrow(
      /missing binding 0 \(params\), binding 1 \(statsBuf\)/,
    );
    expect(() => program.bindGroup(1, {})).toThrow(/stateSrc.*stateDst/);
  });

  it('refuses a group the program never declared', async () => {
    const { ctx } = rig();
    const program = await ctx.program(programOptions());
    expect(() => program.bindGroup(2, {})).toThrow(/group 2 is not declared \(declared: \[0, 1\]\)/);
    expect(() => program.layoutFor(7)).toThrow(/group 7 is not declared/);
  });
});

describe('ComputeProgram.submit', () => {
  it('runs the chain in one pass, setting bind groups before each pipeline', async () => {
    const { ctx, device } = rig();
    const { program, staticGroup, stateGroup } = await bound(ctx);
    const dispatches: readonly ComputeDispatch[] = [
      { entryPoint: 'clear', workgroups: 2, groups: { 0: staticGroup } },
      { entryPoint: 'main', workgroups: 782, groups: { 1: stateGroup, 0: staticGroup } },
    ];
    expect(program.submit(dispatches)).toBe(2);
    expect(device.submissions).toHaveLength(1);
    const pass = device.lastPass!;
    expect(pass.ended).toBe(true);
    expect(pass.entryPoints).toEqual(['clear', 'main']);
    expect(pass.dispatches).toEqual([2, 782]);
    expect(pass.label).toBe('test-program');
    // Bind groups in ascending index order, whatever order the record lists them.
    expect(pass.calls.map((c) => c.kind)).toEqual([
      'bindGroup',
      'pipeline',
      'dispatch',
      'bindGroup',
      'bindGroup',
      'pipeline',
      'dispatch',
      'end',
    ]);
    expect(pass.calls.filter((c) => c.kind === 'bindGroup').map((c) => c.group)).toEqual([0, 0, 1]);
  });

  it('uses the label it is given for the encoder and the pass', async () => {
    const { ctx, device } = rig();
    const { program } = await bound(ctx);
    program.submit([{ entryPoint: 'main', workgroups: 1 }], 'step 4');
    expect(device.lastPass!.label).toBe('step 4');
  });

  it('skips a non-positive workgroup count, and submits nothing when the chain is empty', async () => {
    const { ctx, device } = rig();
    const { program } = await bound(ctx);
    expect(
      program.submit([
        { entryPoint: 'main', workgroups: 3 },
        { entryPoint: 'clear', workgroups: 0 },
        { entryPoint: 'clear', workgroups: -2 },
      ]),
    ).toBe(1);
    expect(device.lastPass!.dispatches).toEqual([3]);
    expect(device.lastPass!.entryPoints).toEqual(['main']);

    expect(program.submit([])).toBe(0);
    expect(program.submit([{ entryPoint: 'main', workgroups: 0 }])).toBe(0);
    expect(device.submissions).toHaveLength(1);
    expect(device.passes).toHaveLength(1);
  });

  it('rejects a dispatch above the workgroup limit, recording nothing', async () => {
    const { ctx, device } = rig(stubLimits({ maxComputeWorkgroupsPerDimension: 65535 }));
    const { program } = await bound(ctx);
    expect(() =>
      program.submit([
        { entryPoint: 'main', workgroups: 4 },
        { entryPoint: 'clear', workgroups: 65536 },
      ]),
    ).toThrow(/clear needs 65536 workgroups, above maxComputeWorkgroupsPerDimension=65535/);
    expect(device.submissions).toHaveLength(0);
    expect(device.passes).toHaveLength(0);
    // A limit that was never reported is not a ceiling of zero.
    const unknown = rig(stubLimits({ maxComputeWorkgroupsPerDimension: 0 }));
    const loose = await bound(unknown.ctx);
    expect(loose.program.submit([{ entryPoint: 'main', workgroups: 999999 }])).toBe(1);
  });

  it('rejects a workgroup count that is not a whole number', async () => {
    const { ctx, device } = rig();
    const { program } = await bound(ctx);
    expect(() => program.submit([{ entryPoint: 'main', workgroups: 1.5 }])).toThrow(
      /1\.5 workgroups, not an integer/,
    );
    expect(() => program.submit([{ entryPoint: 'main', workgroups: Number.NaN }])).toThrow(
      /NaN workgroups/,
    );
    expect(() => program.submit([{ entryPoint: 'main', workgroups: Number.POSITIVE_INFINITY }])).toThrow(
      /Infinity workgroups/,
    );
    expect(device.submissions).toHaveLength(0);
  });

  it('rejects an undeclared entry point at dispatch time, not at record time', async () => {
    const { ctx, device } = rig();
    const { program } = await bound(ctx);
    expect(() => program.submit([{ entryPoint: 'typo', workgroups: 1 }])).toThrow(
      /'typo' is not a declared entry point/,
    );
    expect(device.submissions).toHaveLength(0);
  });

  it('stops submitting once the device is lost or destroyed', async () => {
    const { ctx, shared } = rig();
    const { program } = await bound(ctx);
    expect(program.submit([{ entryPoint: 'main', workgroups: 1 }])).toBe(1);
    shared.destroy();
    expect(shared.usable).toBe(false);
    expect(() => program.submit([{ entryPoint: 'main', workgroups: 1 }])).toThrow(
      /shared device is destroyed/,
    );
  });
});

describe('ComputeContext.readBytes', () => {
  it('copies the buffer out, unmapped and staged nowhere afterwards', async () => {
    const { ctx, device } = rig();
    const stats = ctx.storageBuffer(20, 'stats');
    stats.write(new Uint32Array([7, 0, 1, 0x41200000, 0]));
    const before = device.buffers.length;

    const [words] = await ctx.readBytes([{ buffer: stats, bytes: 20 }]);
    expect(Array.from(new Uint32Array(words.buffer))).toEqual([7, 0, 1, 0x41200000, 0]);
    expect(words).toHaveLength(20);

    // One submit, one copy, and the staging buffer is gone again.
    expect(device.submissions).toHaveLength(1);
    const staging = device.buffers.slice(before);
    expect(staging).toHaveLength(1);
    expect(staging[0].usage).toBe(U.MAP_READ | U.COPY_DST);
    expect(staging[0].mapCalls).toBe(1);
    expect(staging[0].lastMapMode).toBe(CONSTANTS.mapMode.READ);
    expect(staging[0].unmapCalls).toBe(1);
    expect(staging[0].destroyed).toBe(true);
    expect(ctx.buffers.map((b) => b.label)).toEqual(['stats']);
  });

  it('reads several sources in one submit, honouring a source offset', async () => {
    const { ctx, device } = rig();
    const a = ctx.storageBuffer(32, 'a');
    const b = ctx.storageBuffer(32, 'b');
    a.write(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    b.write(new Float32Array([9, 9, 9, 9]));

    const [tail, head] = await ctx.readBytes([
      { buffer: a, bytes: 8, offset: 16 },
      { buffer: b.raw, bytes: 16 },
    ]);
    expect(Array.from(new Float32Array(tail.buffer))).toEqual([5, 6]);
    expect(Array.from(new Float32Array(head.buffer))).toEqual([9, 9, 9, 9]);

    expect(device.submissions).toHaveLength(1);
    const copies = device.submissions[0].copies;
    expect(copies.map((c) => [c.fromOffset, c.bytes])).toEqual([
      [16, 8],
      [0, 16],
    ]);
    expect(device.buffers.filter((buffer) => buffer.destroyed)).toHaveLength(2);
  });

  it('returns an empty result for no sources, without touching the device', async () => {
    const { ctx, device } = rig();
    await expect(ctx.readBytes([])).resolves.toEqual([]);
    expect(device.submissions).toHaveLength(0);
    expect(device.buffers).toHaveLength(0);
  });

  it('validates the size and offset, and refuses a dead context', async () => {
    const { ctx, shared } = rig();
    const source = ctx.storageBuffer(16, 's');
    await expect(ctx.readBytes([{ buffer: source, bytes: 6 }])).rejects.toThrow(/multiple of 4/);
    await expect(ctx.readBytes([{ buffer: source, bytes: 8, offset: 2 }])).rejects.toThrow(
      /not a multiple of 4/,
    );
    shared.destroy();
    await expect(ctx.readBytes([{ buffer: source, bytes: 8 }])).rejects.toThrow(/lost|destroyed/);
  });
});

describe('PingPong', () => {
  it('alternates src and dst without moving any data', () => {
    const pair = new PingPong('a', 'b');
    expect(pair.src).toBe('a');
    expect(pair.dst).toBe('b');
    expect(pair.parity).toBe(0);
    pair.swap();
    expect([pair.src, pair.dst, pair.parity]).toEqual(['b', 'a', 1]);
    pair.swap();
    expect([pair.src, pair.dst, pair.parity]).toEqual(['a', 'b', 0]);
    expect(pair.first).toBe('a');
    expect(pair.second).toBe('b');
  });
});

describe('submitCopy', () => {
  it('records the batch as one submission', () => {
    const { ctx, device } = rig();
    const from = ctx.storageBuffer(32, 'from');
    const to = ctx.storageBuffer(32, 'to');
    from.write(new Float32Array([1, 2, 3, 4]));
    expect(submitCopy(device, [{ from, to, bytes: 16 }])).toBe(1);
    const toStub = device.buffers.find((buffer) => buffer.label === 'to')!;
    expect(Array.from(toStub.floats())).toEqual([1, 2, 3, 4, 0, 0, 0, 0]);
    expect(device.submissions).toHaveLength(1);
    expect(device.submissions[0].copies).toHaveLength(1);
    expect(submitCopy(device, [{ from, to, bytes: 16 }, { from, to, bytes: 0 }])).toBe(1);
    expect(submitCopy(device, [])).toBe(0);
    expect(device.submissions).toHaveLength(2);
  });

  it('checks alignment, because the driver message never names the buffer', () => {
    const { ctx, device } = rig();
    const from = ctx.storageBuffer(32, 'from');
    const to = ctx.storageBuffer(32, 'to');
    expect(() => submitCopy(device, [{ from, to, bytes: 6 }])).toThrow(/multiple of 4/);
    expect(() => submitCopy(device, [{ from, to, bytes: 8, toOffset: 1 }])).toThrow(
      /not a multiple of 4/,
    );
    expect(() => submitCopy(device, [{ from, to, bytes: Number.NaN }])).toThrow(RangeError);
    expect(device.submissions).toHaveLength(0);
  });
});

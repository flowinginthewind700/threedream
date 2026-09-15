/**
 * `gpu/particleGpu.ts` -- the GPU particle backend, driven by the recording stub.
 *
 * A real adapter cannot be made to fail on cue, so this spec pins everything
 * that is a *decision* rather than a computation: which kernels run in which
 * order, how each is sized, which state buffer is bound on which step, what
 * lands in the uniform, and what happens when the world does not fit the
 * device. The arithmetic inside the kernels is specified by
 * `tests/particle_cpu.test.ts` and compared against a live GPU in
 * `e2e/particles_gpu.spec.ts`.
 */

import { describe, expect, it } from 'vitest';

import { ComputeContext, ShaderCompilationError } from '../src/gpu/compute.js';
import { SharedDeviceManager, type SharedDevice } from '../src/gpu/device.js';
import {
  GpuParticleSystem,
  createGpuParticleSystem,
  gpuBufferBudget,
  tableSizeFor,
} from '../src/gpu/particleGpu.js';
import { PARTICLE_BYTES, ParticleField } from '../src/gpu/particleField.js';
import { nextPow2 } from '../src/gpu/particleHash.js';
import {
  BOUNDS_MODE_BITS,
  PARAM_WORD,
  PARAMS_FLOATS,
  PARTICLE_FLAG,
  resolveParticleOptions,
} from '../src/gpu/particleOptions.js';
import type { ParticleSimOptions, ParticleSystem } from '../src/gpu/particleTypes.js';
import {
  GROUP_STATE,
  KERNEL_DISPATCH,
  PARTICLE_KERNELS,
  STAT_WORD,
  STAT_WORDS,
  workgroupsFor,
} from '../src/gpu/particleWgsl.js';
import {
  STUB_CONSTANTS as CONSTANTS,
  StubDevice,
  flushMicrotasks,
  stubLimits,
  type StubBindGroup,
  type StubBuffer,
  type StubPass,
} from './stub_webgpu.js';

const U = CONSTANTS.bufferUsage;

// ---------------------------------------------------------------------------
// rig
// ---------------------------------------------------------------------------

interface Built {
  readonly device: StubDevice;
  readonly manager: SharedDeviceManager;
  readonly shared: SharedDevice;
  readonly field: ParticleField;
  readonly system: GpuParticleSystem;
}

interface BuildOptions {
  readonly count?: number;
  readonly options?: ParticleSimOptions;
  readonly limits?: Record<string, number>;
}

/** A device, a manager and one adopted handle: the cheapest usable triple. */
function stubbed(limits: Record<string, number> = stubLimits()) {
  const device = new StubDevice({ limits });
  const manager = new SharedDeviceManager({ constants: CONSTANTS });
  return { device, manager, shared: manager.adopt(device) };
}

/**
 * A reproducible field with moving particles.
 *
 * `speed: 2` matters: a field seeded at rest reports `maxSpeed() === 0`, and
 * then "the stats were seeded from the field" is indistinguishable from "the
 * stats were never seeded".
 */
function seededField(count = 64): ParticleField {
  return new ParticleField({ count, scene: 'sphere', seed: 7, radius: [0.1, 0.1], speed: 2 });
}

async function build(over: BuildOptions = {}): Promise<Built> {
  const { device, manager, shared } = stubbed(over.limits);
  const field = seededField(over.count ?? 64);
  const system = await createGpuParticleSystem({ shared, field, options: over.options });
  return { device, manager, shared, field, system };
}

/** A buffer by label, with the labels in the error so a rename is diagnosable. */
function buf(device: StubDevice, label: string): StubBuffer {
  const found = device.buffers.find((b) => b.label === label);
  if (!found) {
    throw new Error(`no buffer labelled '${label}'; have [${device.buffers.map((b) => b.label)}]`);
  }
  return found;
}

/** The pass from one step. `device.passes` is flat, so index by step number. */
function passAt(device: StubDevice, step: number): StubPass {
  const pass = device.passes[step];
  if (!pass) throw new Error(`no pass ${step}; the device saw ${device.passes.length}`);
  return pass;
}

/** Every group-1 binding in a pass, in the order the encoder set them. */
function stateBindings(pass: StubPass): readonly StubBindGroup[] {
  return pass.calls
    .filter((c) => c.kind === 'bindGroup' && c.group === GROUP_STATE)
    .map((c) => c.bindGroup!);
}

/** The uploaded uniform, split the way `writeParams` packs it. */
function paramsOf(device: StubDevice) {
  const store = buf(device, 'params').store;
  return {
    floats: new Float32Array(store, 0, 20),
    ints: new Uint32Array(store, 20 * 4, PARAMS_FLOATS - 20),
  };
}

/** The u32 an f32's bits make, which is how `maxSpeedSq` crosses the boundary. */
function bitsOf(value: number): number {
  return new Uint32Array(new Float32Array([value]).buffer)[0]!;
}

// ---------------------------------------------------------------------------
// sizing
// ---------------------------------------------------------------------------

describe('sizing', () => {
  it('picks the hash table size the broadphase would pick', () => {
    for (const count of [1, 2, 5, 63, 64, 65, 50_000, 100_000]) {
      expect(tableSizeFor(count)).toBe(nextPow2(count));
    }
  });

  it('accounts for every buffer, and for the ping-pong pair twice', () => {
    const budget = gpuBufferBudget(64, 64, 8);
    expect(budget.state).toBe(64 * 2 * 16);
    expect(budget.accel).toBe(64 * 16);
    expect(budget.contact).toBe(64 * 2 * 16);
    expect(budget.hashCounts).toBe(64 * 4);
    expect(budget.hashSlots).toBe(64 * 8 * 4);
    expect(budget.publish).toBe(64 * 4 * 4);
    expect(budget.stats).toBe(STAT_WORDS * 4);
    expect(budget.params).toBe(96);
    expect(budget.total).toBe(
      budget.state * 2 +
        budget.accel +
        budget.contact +
        budget.hashCounts +
        budget.hashSlots +
        budget.publish +
        budget.stats +
        budget.params,
    );
  });

  it('reports the largest single buffer, because that is what a binding limit caps', () => {
    // At the default capacity the state buffers dominate...
    expect(gpuBufferBudget(1024, 1024, 8).largest).toBe(1024 * 2 * 16);
    // ...and a deeper bucket makes the slot table dominate instead.
    const deep = gpuBufferBudget(1024, 1024, 64);
    expect(deep.largest).toBe(deep.hashSlots);
    expect(deep.largest).toBeGreaterThan(deep.state);
  });
});

// ---------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------

describe('construction', () => {
  it('satisfies the shared contract, and is honest about determinism', async () => {
    const { system, field } = await build();
    const contract: ParticleSystem = system;
    expect(contract.name).toBe('gpu');
    // Atomics in the broadphase and a driver-chosen reduction order.
    expect(contract.deterministic).toBe(false);
    expect(contract).toBeInstanceOf(GpuParticleSystem);
    expect(contract.field).toBe(field);
    expect(contract.count).toBe(64);
    expect(contract.bounds).toBe(field.bounds);
    expect(contract.fixedDt).toBeCloseTo(1 / 60, 12);
    expect(contract.steps).toBe(0);
    expect(contract.time).toBe(0);
    expect(contract.options).toEqual(resolveParticleOptions());
  });

  it('derives the cell size and table size from the field, not from the caller', async () => {
    const { system } = await build({ count: 130 });
    expect(system.cellSize).toBeCloseTo(0.2, 6); // 2 * maxRadius
    expect(system.tableSize).toBe(256); // nextPow2(130)
    expect(system.workgroups).toBe(workgroupsFor(130));
  });

  it('honours an explicit cell size', async () => {
    const { system } = await build({ options: { cellSize: 1.5 } });
    expect(system.cellSize).toBe(1.5);
  });

  it('allocates every buffer at the size the layout promises', async () => {
    const { device, system } = await build({ count: 130 });
    const budget = system.budget;
    const expected: Record<string, number> = {
      params: budget.params,
      stats: budget.stats,
      'state-a': budget.state,
      'state-b': budget.state,
      accel: budget.accel,
      contact: budget.contact,
      hashCounts: budget.hashCounts,
      hashSlots: budget.hashSlots,
      publish: budget.publish,
    };
    for (const [label, bytes] of Object.entries(expected)) {
      expect(buf(device, label).size, label).toBe(bytes);
    }
    expect(device.buffers).toHaveLength(Object.keys(expected).length);
  });

  it('creates the uniform as a uniform and the rest as copy-capable storage', async () => {
    const { device } = await build();
    expect(buf(device, 'params').usage & U.UNIFORM).toBeTruthy();
    expect(buf(device, 'params').usage & U.COPY_DST).toBeTruthy();
    for (const label of ['state-a', 'state-b', 'accel', 'contact', 'publish', 'stats']) {
      const usage = buf(device, label).usage;
      expect(usage & U.STORAGE, label).toBeTruthy();
      // COPY_DST is what makes `write()` legal; COPY_SRC is what makes readback.
      expect(usage & U.COPY_DST, label).toBeTruthy();
      expect(usage & U.COPY_SRC, label).toBeTruthy();
    }
  });

  it('seeds state-a with the field bytes and leaves state-b empty', async () => {
    const { device, field } = await build();
    expect(buf(device, 'state-a').floats()).toEqual(field.data);
    expect(buf(device, 'state-b').bytes().every((b) => b === 0)).toBe(true);
    // One upload for the seed, and it went through the queue rather than a map.
    expect(device.writes).toHaveLength(1);
    expect(device.writes[0]!.buffer.label).toBe('state-a');
    expect(device.writes[0]!.bytes).toBe(field.count * PARTICLE_BYTES);
  });

  it('builds two state groups and one static group, over two layouts', async () => {
    const { device } = await build();
    expect(device.layouts.map((l) => l.label)).toEqual([
      'particles:group0',
      'particles:group1',
    ]);
    expect(device.bindGroups).toHaveLength(3);
  });

  it('binds the static resources to the slots the shader declares', async () => {
    const { device } = await build();
    const group = device.bindGroups[2]!;
    const bySlot: Record<number, string> = {
      0: 'params',
      1: 'accel',
      2: 'contact',
      3: 'hashCounts',
      4: 'hashSlots',
      5: 'stats',
    };
    for (const [binding, label] of Object.entries(bySlot)) {
      expect(group.bufferAt(Number(binding))?.label, `binding ${binding}`).toBe(label);
    }
    expect(group.entries).toHaveLength(Object.keys(bySlot).length);
  });

  it('binds the ping-pong pair in both orientations, with publish in both', async () => {
    const { device } = await build();
    const [even, odd] = device.bindGroups as [StubBindGroup, StubBindGroup];
    expect(even.bufferAt(0)?.label).toBe('state-a');
    expect(even.bufferAt(1)?.label).toBe('state-b');
    expect(odd.bufferAt(0)?.label).toBe('state-b');
    expect(odd.bufferAt(1)?.label).toBe('state-a');
    expect(even.bufferAt(2)?.label).toBe('publish');
    expect(odd.bufferAt(2)?.label).toBe('publish');
  });

  it('seeds stats() from the field, so an unstepped system is not all zeros', async () => {
    const { system, field } = await build();
    expect(system.stats()).toEqual({
      contacts: 0,
      escaped: 0,
      hashOverflow: 0,
      maxSpeed: field.maxSpeed(),
      kineticEnergy: field.kineticEnergy(),
    });
    expect(system.stats().maxSpeed).toBeGreaterThan(0);
    expect(system.stats().kineticEnergy).toBeGreaterThan(0);
  });

  it('takes exactly one more reference on the shared device', async () => {
    const { shared } = await build();
    expect(shared.references).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// the step chain
// ---------------------------------------------------------------------------

describe('the step chain', () => {
  it('runs the enabled kernels in one pass, in shader order', async () => {
    const { device, system } = await build();
    system.step();
    expect(device.passes).toHaveLength(1);
    expect(passAt(device, 0).label).toBe('particles step 0');
    expect(passAt(device, 0).entryPoints).toEqual([
      'hash_clear',
      'hash_scatter',
      'collide',
      'integrate',
      'publish',
    ]);
    expect(passAt(device, 0).ended).toBe(true);
    expect(device.submissions).toHaveLength(1);
  });

  it('sizes hash_clear from the table and everything else from the count', async () => {
    // 192 particles and a 256-slot table: the two sizings differ, so a mix-up
    // shows up instead of cancelling out.
    const { device, system } = await build({ count: 192 });
    expect(workgroupsFor(192)).not.toBe(workgroupsFor(system.tableSize));
    system.step();
    const pass = passAt(device, 0);
    const expected = pass.entryPoints.map((kernel) =>
      KERNEL_DISPATCH[kernel as keyof typeof KERNEL_DISPATCH] === 'tableSize'
        ? workgroupsFor(system.tableSize)
        : workgroupsFor(192),
    );
    expect(pass.dispatches).toEqual(expected);
    expect(pass.dispatches[0]).toBe(4);
    expect(pass.dispatches.slice(1)).toEqual([3, 3, 3, 3]);
  });

  it('puts nbody first when the option is on', async () => {
    const { device, system } = await build({ options: { nbody: true } });
    system.step();
    expect(passAt(device, 0).entryPoints).toEqual([...PARTICLE_KERNELS]);
    expect(system.options.nbody).toBe(true);
  });

  it('drops the broadphase kernels when collisions are off, but keeps hash_clear', async () => {
    const { device, system } = await build({ options: { collisions: false } });
    system.step();
    // Invocation 0 of hash_clear resets the counters the readback depends on.
    expect(passAt(device, 0).entryPoints).toEqual(['hash_clear', 'integrate', 'publish']);
  });

  it('counts steps and simulated seconds independently of frames', async () => {
    const { system } = await build({ options: { fixedDt: 0.01 } });
    system.advance(5);
    expect(system.steps).toBe(5);
    expect(system.time).toBeCloseTo(0.05, 10);
    system.step(0.25);
    expect(system.steps).toBe(6);
    // `time` is steps * fixedDt: a custom dt does not move the clock.
    expect(system.time).toBeCloseTo(0.06, 10);
  });

  it('refuses an advance that is not a non-negative integer', async () => {
    const { system } = await build();
    expect(() => system.advance(-1)).toThrow(RangeError);
    expect(() => system.advance(1.5)).toThrow(RangeError);
    system.advance(0);
    expect(system.steps).toBe(0);
  });

  it('memoizes one pipeline per entry point across steps', async () => {
    const { device, system } = await build();
    system.advance(4);
    expect(device.pipelines.map((p) => p.entryPoint)).toEqual([
      'hash_clear',
      'hash_scatter',
      'collide',
      'integrate',
      'publish',
    ]);
    expect(device.pipelines[0]!.label).toBe('particles:hash_clear');
    // One pass per step, but never a second pipeline for the same kernel.
    expect(device.passes).toHaveLength(4);
  });

  it('packs the uniform with writeParams, so the shader sees what the CPU spec reads', async () => {
    const { device, system } = await build({
      count: 130,
      options: { damping: 0.25, restitution: 0.4, maxSpeed: 12, cutoff: 3, gravity: [1, -2, 3] },
    });
    system.step(1 / 30);
    const { floats, ints } = paramsOf(device);
    expect(floats[PARAM_WORD.dt]).toBeCloseTo(1 / 30, 6);
    expect(floats[PARAM_WORD.gravityX]).toBeCloseTo(1, 6);
    expect(floats[PARAM_WORD.gravityY]).toBeCloseTo(-2, 6);
    expect(floats[PARAM_WORD.gravityZ]).toBeCloseTo(3, 6);
    expect(floats[PARAM_WORD.damping]).toBeCloseTo(0.25, 6);
    expect(floats[PARAM_WORD.restitution]).toBeCloseTo(0.4, 6);
    expect(floats[PARAM_WORD.maxSpeed]).toBeCloseTo(12, 6);
    expect(floats[PARAM_WORD.cutoffSquared]).toBeCloseTo(9, 6);
    expect(floats[PARAM_WORD.cellSize]).toBeCloseTo(system.cellSize, 6);
    expect(floats[PARAM_WORD.invCell]).toBeCloseTo(1 / system.cellSize, 4);
    expect(floats[PARAM_WORD.boundsMinX]).toBe(-8);
    expect(floats[PARAM_WORD.boundsMaxY]).toBe(8);
    expect(ints[0]).toBe(130);
    expect(ints[1]).toBe(system.tableSize - 1);
    expect(ints[2]).toBe(8);
  });

  it('re-uploads the uniform every step, so a new dt takes effect immediately', async () => {
    const { device, system } = await build();
    system.step(1 / 120);
    expect(paramsOf(device).floats[PARAM_WORD.dt]).toBeCloseTo(1 / 120, 8);
    system.step(1 / 20);
    expect(paramsOf(device).floats[PARAM_WORD.dt]).toBeCloseTo(1 / 20, 8);
    // One seed write plus one uniform write per step.
    expect(device.writes.filter((w) => w.buffer.label === 'params')).toHaveLength(2);
  });

  it('carries collisions, nbody and the bounds mode in the flags word', async () => {
    const plain = await build();
    plain.system.step();
    expect(paramsOf(plain.device).ints[3]).toBe(PARTICLE_FLAG.collisions);

    const wrapped = await build({ options: { nbody: true, collisions: false, boundsMode: 'wrap' } });
    wrapped.system.step();
    expect(paramsOf(wrapped.device).ints[3]).toBe(
      PARTICLE_FLAG.nbody | (BOUNDS_MODE_BITS.wrap << PARTICLE_FLAG.boundsShift),
    );

    const unbounded = await build({ options: { boundsMode: 'none' } });
    unbounded.system.step();
    expect(paramsOf(unbounded.device).ints[3]).toBe(
      PARTICLE_FLAG.collisions | (BOUNDS_MODE_BITS.none << PARTICLE_FLAG.boundsShift),
    );
  });

  it('refuses a dt that is not finite and positive, and records nothing', async () => {
    const { device, system } = await build();
    const writes = device.writes.length;
    for (const dt of [0, -0.01, NaN, Infinity, -Infinity]) {
      expect(() => system.step(dt), `dt=${dt}`).toThrow(RangeError);
    }
    expect(device.submissions).toHaveLength(0);
    expect(device.writes).toHaveLength(writes);
    expect(system.steps).toBe(0);
  });

  it('refuses a dispatch above the workgroup limit, and records nothing', async () => {
    const { device, system } = await build({
      count: 192,
      limits: stubLimits({ maxComputeWorkgroupsPerDimension: 2 }),
    });
    expect(system.workgroups).toBe(3);
    expect(() => system.step()).toThrow(RangeError);
    expect(device.submissions).toHaveLength(0);
    expect(device.passes).toHaveLength(0);
    // The uniform write already happened; the step counter did not.
    expect(system.steps).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// ping-pong
// ---------------------------------------------------------------------------

describe('ping-pong', () => {
  it('reads state-a and publishes state-b on step 0', async () => {
    const { device, system } = await build();
    system.step();
    const [even, odd] = device.bindGroups as [StubBindGroup, StubBindGroup];
    const bindings = stateBindings(passAt(device, 0));
    // `publish` is the odd one out: it reads the state `integrate` just wrote,
    // which this step called `dst`, so it binds the other orientation.
    expect(bindings.map((g) => g.bufferAt(0)?.label)).toEqual([
      'state-a',
      'state-a',
      'state-a',
      'state-a',
      'state-b',
    ]);
    expect(bindings[0]).toBe(even);
    expect(bindings[4]).toBe(odd);
    expect(system.steps).toBe(1);
  });

  it('inverts on step 1 without building a single bind group', async () => {
    const { device, system } = await build();
    system.advance(2);
    const [even, odd] = device.bindGroups as [StubBindGroup, StubBindGroup];
    const bindings = stateBindings(passAt(device, 1));
    expect(bindings.map((g) => g.bufferAt(0)?.label)).toEqual([
      'state-b',
      'state-b',
      'state-b',
      'state-b',
      'state-a',
    ]);
    expect(bindings[0]).toBe(odd);
    expect(bindings[4]).toBe(even);
    expect(system.steps).toBe(2);
    // Two pre-built groups are the whole point: nothing is created per step.
    expect(device.bindGroups).toHaveLength(3);
  });

  it('binds the static group before every dispatch, in ascending group order', async () => {
    const { device, system } = await build({ options: { collisions: false } });
    system.step();
    const staticGroup = device.bindGroups[2]!;
    const kinds = passAt(device, 0).calls.map((c) =>
      c.kind === 'bindGroup' ? `bind${c.group}` : c.kind,
    );
    expect(kinds).toEqual([
      'bind0',
      'bind1',
      'pipeline',
      'dispatch',
      'bind0',
      'bind1',
      'pipeline',
      'dispatch',
      'bind0',
      'bind1',
      'pipeline',
      'dispatch',
      'end',
    ]);
    expect(passAt(device, 0).calls.filter((c) => c.bindGroup === staticGroup)).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// readback
// ---------------------------------------------------------------------------

describe('readback', () => {
  it('decodes the counters and the bitcast max speed', async () => {
    const { device, system } = await build();
    const stats = buf(device, 'stats').u32();
    stats[STAT_WORD.contacts] = 5;
    stats[STAT_WORD.escaped] = 1;
    stats[STAT_WORD.overflow] = 2;
    stats[STAT_WORD.maxSpeedSq] = bitsOf(4);
    const energy = system.stats().kineticEnergy;
    expect(await system.readStats()).toEqual({
      contacts: 5,
      escaped: 1,
      hashOverflow: 2,
      maxSpeed: 2,
      kineticEnergy: energy,
    });
    // Cached, so `stats()` and the awaited value are the same object.
    expect(system.stats()).toBe(system.stats());
    expect(system.stats().contacts).toBe(5);
  });

  it('reports a zero max speed for a word that is not a positive f32', async () => {
    const { device, system } = await build();
    const stats = buf(device, 'stats').u32();
    stats[STAT_WORD.maxSpeedSq] = bitsOf(-1);
    expect((await system.readStats()).maxSpeed).toBe(0);
    stats[STAT_WORD.maxSpeedSq] = bitsOf(NaN);
    expect((await system.readStats()).maxSpeed).toBe(0);
    stats[STAT_WORD.maxSpeedSq] = bitsOf(0);
    expect((await system.readStats()).maxSpeed).toBe(0);
  });

  it('carries kinetic energy forward, because only a full readback can refresh it', async () => {
    const { device, system, field } = await build();
    buf(device, 'stats').u32()[STAT_WORD.contacts] = 9;
    const before = field.kineticEnergy();
    await system.readStats();
    expect(system.stats().kineticEnergy).toBeCloseTo(before, 6);
    expect(system.stats().kineticEnergy).not.toBe(0);
  });

  it('pulls the source buffer into field.data', async () => {
    const { device, system, field } = await build({ count: 8 });
    const src = buf(device, 'state-a').floats();
    for (let i = 0; i < src.length; i++) src[i] = (i + 1) / 16;
    const before = system.stats().kineticEnergy;
    expect(await system.readback()).toBe(field);
    expect(Array.from(field.data)).toEqual(Array.from(src));
    // The energy is recomputed from what came back, not carried over.
    expect(system.stats().kineticEnergy).toBeCloseTo(field.kineticEnergy(), 6);
    expect(system.stats().kineticEnergy).not.toBeCloseTo(before, 6);
  });

  it('follows the swap: after a step the source is state-b', async () => {
    const { device, system, field } = await build({ count: 8 });
    system.step();
    const a = buf(device, 'state-a').floats();
    const b = buf(device, 'state-b').floats();
    for (let i = 0; i < b.length; i++) b[i] = -1 - i;
    const snapshotA = a.slice();
    await system.readback();
    expect(Array.from(field.data)).toEqual(Array.from(b));
    expect(Array.from(a)).toEqual(Array.from(snapshotA));
  });

  it('destroys its staging buffer and leaves nothing mapped', async () => {
    const { device, system } = await build();
    await system.readStats();
    const staging = device.buffers.filter((b) => b.label.startsWith('readback staging'));
    expect(staging).toHaveLength(1);
    expect(staging[0]!.size).toBe(STAT_WORDS * 4);
    expect(staging[0]!.destroyed).toBe(true);
    expect(staging[0]!.mapCalls).toBe(1);
    expect(staging[0]!.unmapCalls).toBe(1);
    expect(staging[0]!.lastMapMode).toBe(CONSTANTS.mapMode.READ);
    // Staging is untracked, so teardown does not try to destroy it twice.
    expect(system.context.buffers.some((b) => b.label.startsWith('readback staging'))).toBe(false);
  });

  it('reports a digest that is only as fresh as the last readback', async () => {
    const { device, system, field } = await build({ count: 8 });
    const seeded = system.digest();
    expect(seeded).toBe(field.digest());
    const src = buf(device, 'state-a').floats();
    src[0] = 1234.5;
    // Stale until the state comes back: documented, and asserted so it stays so.
    expect(system.digest()).toBe(seeded);
    await system.readback();
    expect(system.digest()).not.toBe(seeded);
    expect(system.digest()).toBe(field.digest());
    // `hex:length`, where the length is in floats: the digest is over the raw
    // state buffer, so it says how many numbers went into it.
    expect(system.digest().endsWith(`:${field.data.length}`)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// publishing
// ---------------------------------------------------------------------------

describe('publishing to a renderer-owned buffer', () => {
  /** Fill the publish buffer with something recognisable. */
  function seedPublish(device: StubDevice): void {
    const out = buf(device, 'publish').floats();
    for (let i = 0; i < out.length; i++) out[i] = i * 0.5;
  }

  it('copies the published bytes into a ComputeBuffer the caller owns', async () => {
    const { device, shared, system } = await build({ count: 16 });
    seedPublish(device);
    const ctx = new ComputeContext(shared);
    const target = ctx.storageBuffer(system.budget.publish, 'instances');
    const submissions = device.submissions.length;
    expect(system.copyPublishedTo(target)).toBe(system.budget.publish);
    const copied = (target.raw as unknown as StubBuffer).floats();
    expect(copied).toEqual(buf(device, 'publish').floats());
    expect(copied[0]).toBe(0);
    expect(copied[1]).toBe(0.5);
    expect(device.submissions).toHaveLength(submissions + 1);
    // Recorded as a copy, not as a compute pass: no pipeline is touched here.
    const last = device.submissions[device.submissions.length - 1]!;
    expect(last.passes).toHaveLength(0);
    expect(last.copies).toHaveLength(1);
    expect(last.copies[0]!.bytes).toBe(system.budget.publish);
    expect(last.copies[0]!.from.label).toBe('publish');
    ctx.destroy();
  });

  it('copies into a raw GPUBuffer, which is what three.js hands back', async () => {
    const { device, system } = await build({ count: 16 });
    seedPublish(device);
    const attribute = device.createBuffer({
      label: 'three.js instance attribute',
      size: system.budget.publish,
      usage: U.VERTEX | U.COPY_DST,
    });
    expect(system.copyPublishedTo(attribute)).toBe(system.budget.publish);
    expect(new Float32Array(attribute.store)).toEqual(buf(device, 'publish').floats());
  });

  it('clamps an over-large request and skips an empty one', async () => {
    const { device, system } = await build({ count: 16 });
    seedPublish(device);
    const attribute = device.createBuffer({
      label: 'target',
      size: system.budget.publish,
      usage: U.COPY_DST,
    });
    expect(system.copyPublishedTo(attribute, system.budget.publish + 4096)).toBe(
      system.budget.publish,
    );
    const submissions = device.submissions.length;
    expect(system.copyPublishedTo(attribute, 0)).toBe(0);
    expect(device.submissions).toHaveLength(submissions);
  });
});

// ---------------------------------------------------------------------------
// refusing a world that will not fit
// ---------------------------------------------------------------------------

describe('refusing a world that will not fit', () => {
  it('rejects before allocating when the largest buffer exceeds the binding limit', async () => {
    const { device, shared } = stubbed(stubLimits({ maxStorageBufferBindingSize: 1024 }));
    const field = seededField(64);
    await expect(
      createGpuParticleSystem({ shared, field }),
    ).rejects.toThrow(/above this device's 1024-byte storage limit/);
    // Nothing allocated, and no reference taken: the caller still owns exactly
    // the handle they acquired, and the fallback path starts clean.
    expect(device.buffers).toHaveLength(0);
    expect(shared.references).toBe(1);
  });

  it('takes the smaller of maxBufferSize and the binding limit', async () => {
    const { shared } = stubbed(
      stubLimits({ maxStorageBufferBindingSize: 1 << 20, maxBufferSize: 1024 }),
    );
    await expect(
      createGpuParticleSystem({ shared, field: seededField(64) }),
    ).rejects.toThrow(/1024-byte/);
  });

  it('treats a zero limit as unreported rather than as a ceiling', async () => {
    const { shared } = stubbed(
      stubLimits({ maxStorageBufferBindingSize: 0, maxBufferSize: 0 }),
    );
    const system = await createGpuParticleSystem({ shared, field: seededField(64) });
    expect(system.budget.largest).toBeGreaterThan(0);
    system.dispose();
  });

  it('rejects a shader that fails to compile, and leaves nothing behind', async () => {
    const { device, shared } = stubbed();
    device.setCompilationMessages([
      { type: 'error', message: 'unknown identifier', lineNum: 12 },
    ]);
    const error = await createGpuParticleSystem({ shared, field: seededField(8) }).then(
      () => null,
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(ShaderCompilationError);
    expect((error as ShaderCompilationError).label).toBe('particles');
    expect((error as ShaderCompilationError).message).toContain('line 12');
    expect(device.buffers).toHaveLength(0);
    expect(shared.references).toBe(1);
  });

  it('refuses a device that is already gone', async () => {
    const { device, shared } = stubbed();
    shared.destroy();
    expect(device.destroyed).toBe(true);
    await expect(
      createGpuParticleSystem({ shared, field: seededField(8) }),
    ).rejects.toThrow(/the shared device is destroyed/);
  });

  it('refuses a device that reported itself lost', async () => {
    const { device, shared } = stubbed();
    device.lose({ reason: 'app-initiated', message: 'gone' });
    await flushMicrotasks();
    expect(shared.lost).toBe(true);
    await expect(
      createGpuParticleSystem({ shared, field: seededField(8) }),
    ).rejects.toThrow(/the shared device is lost/);
  });

  it('refuses something that is not a field', async () => {
    const { shared } = stubbed();
    for (const field of [null, undefined, {}, { count: 0 }, { count: 2.5 }]) {
      await expect(
        createGpuParticleSystem({ shared, field: field as unknown as ParticleField }),
      ).rejects.toThrow(TypeError);
    }
    expect(shared.references).toBe(1);
  });

  it('refuses a field whose box is thinner than a particle', async () => {
    const { shared } = stubbed();
    const thin = new ParticleField({
      count: 4,
      scene: 'grid',
      radius: [0.6, 0.6],
      bounds: { min: [-1, -0.5, -1], max: [1, 0.5, 1] },
    });
    await expect(createGpuParticleSystem({ shared, field: thin })).rejects.toThrow(
      /cannot contain a particle of diameter/,
    );
    expect(shared.references).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// teardown
// ---------------------------------------------------------------------------

describe('teardown', () => {
  it('destroys every buffer it made and drops exactly one reference', async () => {
    const { device, shared, system } = await build();
    expect(device.buffers.length).toBeGreaterThan(0);
    system.dispose();
    expect(system.disposed).toBe(true);
    for (const buffer of device.buffers) {
      expect(buffer.destroyed, buffer.label).toBe(true);
    }
    expect(system.context.destroyed).toBe(true);
    // The caller's own handle survives: only ours was dropped.
    expect(shared.references).toBe(1);
    expect(shared.usable).toBe(true);
    expect(device.destroyed).toBe(false);
  });

  it('is idempotent, and does not release twice', async () => {
    const { shared, system } = await build();
    system.dispose();
    system.dispose();
    system.dispose();
    expect(shared.references).toBe(1);
  });

  it('refuses to run anything after dispose', async () => {
    const { device, shared, system } = await build();
    const ctx = new ComputeContext(shared);
    const target = ctx.storageBuffer(system.budget.publish, 'target');
    system.dispose();
    expect(() => system.step()).toThrow(/disposed/);
    expect(() => system.advance(1)).toThrow(/disposed/);
    expect(() => system.copyPublishedTo(target)).toThrow(/disposed/);
    await expect(system.readStats()).rejects.toThrow(/disposed/);
    await expect(system.readback()).rejects.toThrow(/disposed/);
    expect(device.submissions).toHaveLength(0);
    ctx.destroy();
  });

  it('reports a device lost mid-run through `lost`, and refuses to step', async () => {
    const { device, system } = await build();
    expect(system.lost).toBe(false);
    system.step();
    device.lose({ reason: 'unknown', message: 'driver crashed' });
    await flushMicrotasks();
    expect(system.lost).toBe(true);
    expect(() => system.step()).toThrow(/the shared device is lost/);
    await expect(system.readStats()).rejects.toThrow(/lost/);
    // Disposal still runs on a dead device, and still drops the reference.
    system.dispose();
    expect(system.disposed).toBe(true);
  });

  it('reports a device destroyed under it, too', async () => {
    const { shared, system } = await build();
    shared.destroy();
    expect(system.lost).toBe(true);
    expect(() => system.step()).toThrow(/the shared device is destroyed/);
  });

  it('leaves the manager able to hand the same adopted device to a second system', async () => {
    const { shared, system } = await build({ count: 8 });
    const second = await createGpuParticleSystem({ shared, field: seededField(8) });
    expect(shared.references).toBe(3);
    system.dispose();
    expect(shared.references).toBe(2);
    second.step();
    expect(second.steps).toBe(1);
    second.dispose();
    expect(shared.references).toBe(1);
  });
});

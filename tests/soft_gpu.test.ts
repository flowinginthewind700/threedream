/**
 * `gpu/softGpu.ts` -- the GPU soft-body backend, driven by the recording stub.
 *
 * A real adapter cannot be made to fail on cue, so this spec pins everything that
 * is a *decision* rather than a computation: which kernels run in which order,
 * how each is sized along the one dispatch dimension it has, which color's batch
 * each `solve` binds, what is uploaded into which buffer, what lands in the
 * uniform, and what happens when the device cannot bind fifteen storage buffers.
 * The arithmetic inside the kernels is specified by `tests/soft_cpu.test.ts` and
 * compared against a live GPU in `e2e/soft_gpu.spec.ts`.
 *
 * The plan equality with the CPU tier is asserted here too, because it is the
 * claim M4 makes about this file and it does not need a device to be checked:
 * both tiers call `buildSoftLayout` on the same mesh, and a spec that compares
 * the two results field by field is what stops that from being an assumption.
 */

import { describe, expect, it } from 'vitest';

import { ComputeContext, ShaderCompilationError } from '../src/gpu/compute.js';
import { SharedDeviceManager, type SharedDevice } from '../src/gpu/device.js';
import { softWorkgroups } from '../src/gpu/softIslands.js';
import { createCpuSoftSystem } from '../src/gpu/softCpu.js';
import { GpuSoftSystem, createGpuSoftSystem, softGpuBudget } from '../src/gpu/softGpu.js';
import {
  SOFT_BYTES,
  SOFT_OFFSET,
  SOFT_STRIDE,
  SoftMesh,
  emptyConstraints,
  type SoftMeshOptions,
} from '../src/gpu/softMesh.js';
import {
  SOFT_BOUNDS_MODE_BITS,
  SOFT_FLAG,
  SOFT_PARAMS_BYTES,
  SOFT_PARAM_WORD,
  buildSoftLayout,
  resolveSoftOptions,
} from '../src/gpu/softOptions.js';
import type { SoftPlan, SoftSimOptions, SoftSystem } from '../src/gpu/softTypes.js';
import {
  SOFT_BATCH_STRIDE_BYTES,
  SOFT_BATCH_U32_PER_COLOR,
  SOFT_BINDINGS,
  SOFT_KERNELS,
  SOFT_STAT_WORDS,
  SOFT_STAT_WORD,
  SOFT_STORAGE_BINDINGS,
  softShaderSource,
} from '../src/gpu/softWgsl.js';
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

/**
 * The stub reports the WebGPU baseline of 8 storage buffers, which this pipeline
 * cannot bind. Every build here overrides it, and one spec below refuses a device
 * that does not.
 */
function softLimits(over: Record<string, number> = {}): Record<string, number> {
  return stubLimits({ maxStorageBuffersPerShaderStage: 16, ...over });
}

// ---------------------------------------------------------------------------
// rig
// ---------------------------------------------------------------------------

interface Built {
  readonly device: StubDevice;
  readonly manager: SharedDeviceManager;
  readonly shared: SharedDevice;
  readonly mesh: SoftMesh;
  readonly system: GpuSoftSystem;
}

interface BuildOptions {
  readonly mesh?: SoftMesh;
  readonly options?: SoftSimOptions;
  readonly limits?: Record<string, number>;
}

/** A device, a manager and one adopted handle: the cheapest usable triple. */
function stubbed(limits: Record<string, number> = softLimits()) {
  const device = new StubDevice({ limits });
  const manager = new SharedDeviceManager({ constants: CONSTANTS });
  return { device, manager, shared: manager.adopt(device) };
}

/**
 * A 10x10 cloth with moving nodes.
 *
 * `speed: 2` matters: a mesh seeded at rest reports `maxSpeed() === 0`, and then
 * "the stats were seeded from the mesh" is indistinguishable from "the stats were
 * never seeded".
 */
function seededMesh(spec: Partial<SoftMeshOptions> = {}): SoftMesh {
  return new SoftMesh({ count: 100, scene: 'cloth', seed: 7, speed: 2, ...spec });
}

/** Unconnected nodes: no constraints, no colors, and every node its own island. */
function cloud(count: number): SoftMesh {
  const data = new Float32Array(count * SOFT_STRIDE);
  for (let i = 0; i < count; i++) {
    const o = i * SOFT_STRIDE;
    data[o] = i - (count - 1) / 2;
    data[o + SOFT_OFFSET.invMass] = 1;
    data[o + SOFT_OFFSET.radius] = 0.1;
  }
  return new SoftMesh(data, { count, constraints: emptyConstraints() });
}

async function build(over: BuildOptions = {}): Promise<Built> {
  const { device, manager, shared } = stubbed(over.limits);
  const mesh = over.mesh ?? seededMesh();
  const system = await createGpuSoftSystem({ shared, mesh, options: over.options });
  return { device, manager, shared, mesh, system };
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
 * A pass by index. Index 0 is the frame-zero publish the constructor submits, so
 * step `n` is pass `n + 1` -- the offset is the price of a drawable first frame.
 */
function passAt(device: StubDevice, index: number): StubPass {
  const pass = device.passes[index];
  if (!pass) throw new Error(`no pass ${index}; the device saw ${device.passes.length}`);
  return pass;
}

/** The pass one `step()` recorded, given how many steps came before it. */
function passForStep(device: StubDevice, step: number): StubPass {
  return passAt(device, step + 1);
}

/** Every buffer label in `SOFT_BINDINGS` order, which is the allocation order. */
const LABELS: readonly string[] = SOFT_BINDINGS.map((b) => b.name.replace(/Buf$/, ''));

/** The uploaded uniform, split the way `writeSoftParams` packs it. */
function paramsOf(device: StubDevice) {
  const store = buf(device, 'params').store;
  return {
    floats: new Float32Array(store, 0, 16),
    ints: new Uint32Array(store, 16 * 4, SOFT_PARAMS_BYTES / 4 - 16),
  };
}

/** Integer word `w` of the uniform, addressed the way `SOFT_PARAM_WORD` does. */
function paramInt(device: StubDevice, word: number): number {
  return paramsOf(device).ints[word - 16]!;
}

/** The u32 an f32's bits make, which is how both maxima cross the boundary. */
function bitsOf(value: number): number {
  return new Uint32Array(new Float32Array([value]).buffer)[0]!;
}

function planOf(mesh: SoftMesh, options?: SoftSimOptions): SoftPlan {
  return buildSoftLayout(mesh, resolveSoftOptions(options)).plan;
}

/**
 * The binding `batchBuf` occupies, read out of the table the shader declares.
 *
 * It is the one binding a `solve` dispatch binds differently from every other
 * dispatch, so every spec that pins which color ran has to address it by name
 * rather than by an index written down a second time.
 */
const BATCH_BINDING = SOFT_BINDINGS.find((b) => b.name === 'batchBuf')!.binding;

/** Words from one color's slot in the batch table to the next. */
const BATCH_STRIDE_WORDS = SOFT_BATCH_STRIDE_BYTES / 4;

/** Bytes a solve's view of `batchBuf` exposes: one `(base, count)` pair. */
const BATCH_SLOT_BYTES = SOFT_BATCH_U32_PER_COLOR * 4;

/**
 * The group-0 binding each dispatch in a pass was recorded with, in dispatch order.
 *
 * Which color a `solve` ran is carried by this binding and nowhere else, so the only
 * way to observe it is to walk the pass's calls the way the encoder recorded them.
 */
function group0PerDispatch(pass: StubPass): StubBindGroup[] {
  const groups: StubBindGroup[] = [];
  let bound: StubBindGroup | undefined;
  for (const call of pass.calls) {
    if (call.kind === 'bindGroup' && call.group === 0) bound = call.bindGroup;
    if (call.kind === 'dispatch' && bound) groups.push(bound);
  }
  return groups;
}

// ---------------------------------------------------------------------------
// sizing
// ---------------------------------------------------------------------------

describe('sizing', () => {
  it('accounts for every buffer the pipeline binds', () => {
    const mesh = seededMesh();
    const plan = planOf(mesh);
    const budget = softGpuBudget(plan);
    expect(budget.state).toBe(plan.nodes * SOFT_BYTES);
    expect(budget.pred).toBe(plan.nodes * 16);
    expect(budget.publish).toBe(plan.nodes * 12);
    expect(budget.nodeOrder).toBe(plan.nodeWorkgroups * 64 * 4);
    expect(budget.islandOfWg).toBe(plan.nodeWorkgroups * 4);
    expect(budget.islandOfNode).toBe(plan.nodes * 4);
    expect(budget.order).toBe(plan.constraints * 4);
    expect(budget.ends).toBe(plan.constraints * 8);
    expect(budget.rest).toBe(plan.constraints * 4);
    expect(budget.stiff).toBe(plan.constraints * 4);
    expect(budget.batch).toBe(plan.colors * SOFT_BATCH_STRIDE_BYTES);
    expect(budget.sleep).toBe(plan.islands * 4);
    expect(budget.stats).toBe(SOFT_STAT_WORDS * 4);
    expect(budget.params).toBe(SOFT_PARAMS_BYTES);
  });

  it('sizes from the plan, so the budget cannot describe another decomposition', () => {
    // Two meshes with the same node count and different graphs: the per-edge
    // buffers differ, which they could only do if the plan drove them.
    const clothPlan = planOf(seededMesh({ scene: 'cloth' }));
    const ropePlan = planOf(seededMesh({ scene: 'rope' }));
    expect(clothPlan.nodes).toBe(ropePlan.nodes);
    expect(clothPlan.constraints).toBeGreaterThan(ropePlan.constraints);
    const cloth = softGpuBudget(clothPlan);
    const rope = softGpuBudget(ropePlan);
    // Equal node counts, so the per-node buffers match byte for byte...
    expect(cloth.state).toBe(rope.state);
    expect(cloth.pred).toBe(rope.pred);
    expect(cloth.publish).toBe(rope.publish);
    // ...while every per-edge buffer grows with the graph. A budget keyed on
    // anything other than the plan could not arrange that.
    for (const label of ['order', 'ends', 'rest', 'stiff'] as const) {
      expect(cloth[label], label).toBeGreaterThan(rope[label]);
    }
  });

  it('reports the largest single buffer, because that is what a binding limit caps', () => {
    const plan = planOf(seededMesh());
    const budget = softGpuBudget(plan);
    // 32 bytes a node beats the 16-byte prediction scratch and the 12-byte publish.
    expect(budget.largest).toBe(budget.state);
  });

  it('counts the three sleep buffers separately in the total', () => {
    const plan = planOf(seededMesh());
    const b = softGpuBudget(plan);
    expect(b.total).toBe(
      b.state +
        b.pred +
        b.publish +
        b.nodeOrder +
        b.islandOfWg +
        b.islandOfNode +
        b.order +
        b.ends +
        b.rest +
        b.stiff +
        b.batch +
        b.sleep * 3 +
        b.stats +
        b.params,
    );
    expect(b.total).toBeGreaterThan(b.largest);
  });

  it('pads an empty graph to one word per buffer, since zero bytes is not a buffer', () => {
    const plan = planOf(cloud(4));
    expect(plan.constraints).toBe(0);
    expect(plan.colors).toBe(0);
    const budget = softGpuBudget(plan);
    for (const label of ['order', 'ends', 'rest', 'stiff'] as const) {
      expect(budget[label], label).toBe(4);
    }
    // The batch table's floor is a whole slot and not a word: `batchBuf` is a
    // runtime-sized array, so a binding has to cover an integral number of pairs.
    expect(budget.batch).toBe(SOFT_BATCH_STRIDE_BYTES);
    // The per-node buffers are real, and the sleep buffers are one word an island.
    expect(budget.state).toBe(4 * SOFT_BYTES);
    expect(budget.sleep).toBe(4 * 4);
  });
});

// ---------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------

describe('construction', () => {
  it('satisfies the shared contract, and is honest about determinism', async () => {
    const { system, mesh } = await build();
    const contract: SoftSystem = system;
    expect(contract.name).toBe('gpu');
    // fma contraction and adapter-specific sqrt rounding. See the file header.
    expect(contract.deterministic).toBe(false);
    // The claim the coloring buys: one writer per word, so one device replays.
    expect(system.raceFree).toBe(true);
    expect(contract).toBeInstanceOf(GpuSoftSystem);
    expect(contract.mesh).toBe(mesh);
    expect(contract.count).toBe(mesh.count);
    expect(contract.bounds).toBe(mesh.bounds);
    expect(contract.fixedDt).toBeCloseTo(1 / 60, 12);
    expect(contract.steps).toBe(0);
    expect(contract.time).toBe(0);
    expect(contract.options).toEqual(resolveSoftOptions());
  });

  it('reports the same plan the CPU tier reports for the same mesh', async () => {
    for (const spec of [
      { scene: 'cloth' as const, count: 100 },
      { scene: 'sheets' as const, count: 400, groups: 4 },
      { scene: 'rope' as const, count: 512 },
      { scene: 'cube' as const, count: 216 },
    ]) {
      const options: SoftSimOptions = { iterations: 4 };
      const gpu = await build({ mesh: seededMesh(spec), options });
      const cpu = createCpuSoftSystem({ mesh: seededMesh(spec), options });
      expect(gpu.system.plan, spec.scene).toEqual(cpu.plan);
      expect(gpu.system.plan.iterations).toBe(4);
      expect(gpu.system.plan).toEqual(gpu.system.layout.plan);
      gpu.system.dispose();
    }
  });

  it('allocates one buffer per binding, at the size the budget promised', async () => {
    const { device, system } = await build();
    const budget = system.budget;
    expect(device.buffers.map((b) => b.label)).toEqual([...LABELS]);
    const expected: Record<string, number> = {
      params: budget.params,
      nodeOrder: budget.nodeOrder,
      islandOfWg: budget.islandOfWg,
      islandOfNode: budget.islandOfNode,
      order: budget.order,
      ends: budget.ends,
      rest: budget.rest,
      stiff: budget.stiff,
      batch: budget.batch,
      state: budget.state,
      pred: budget.pred,
      asleep: budget.sleep,
      quiet: budget.sleep,
      islandSpeed: budget.sleep,
      stats: budget.stats,
      publish: budget.publish,
    };
    for (const [label, bytes] of Object.entries(expected)) {
      expect(buf(device, label).size, label).toBe(bytes);
    }
    expect(Object.keys(expected)).toHaveLength(SOFT_BINDINGS.length);
  });

  it('creates the uniform as a uniform and the rest as copy-capable storage', async () => {
    const { device } = await build();
    expect(buf(device, 'params').usage & U.UNIFORM).toBeTruthy();
    expect(buf(device, 'params').usage & U.COPY_DST).toBeTruthy();
    for (const label of LABELS) {
      if (label === 'params') continue;
      const usage = buf(device, label).usage;
      expect(usage & U.STORAGE, label).toBeTruthy();
      // COPY_DST is what makes `write()` legal; COPY_SRC is what makes readback.
      expect(usage & U.COPY_DST, label).toBeTruthy();
      expect(usage & U.COPY_SRC, label).toBeTruthy();
    }
  });

  it('compiles the generated shader with the six declared entry points', async () => {
    const { device, system } = await build();
    expect(device.modules).toHaveLength(1);
    expect(device.modules[0]!.label).toBe('soft');
    expect(device.modules[0]!.code).toBe(softShaderSource());
    expect(system.program.entryPoints).toEqual([...SOFT_KERNELS]);
    expect(system.program.bindings).toEqual(SOFT_BINDINGS);
  });

  it('uploads the mesh, the two island maps and the colored order as-is', async () => {
    const { device, system, mesh } = await build();
    const { islands, coloring } = system.layout;
    // The state buffer *is* `mesh.data`: no repack in either direction.
    expect(buf(device, 'state').floats()).toEqual(mesh.data);
    expect(buf(device, 'nodeOrder').u32()).toEqual(islands.nodeOrder);
    expect(buf(device, 'islandOfWg').u32()).toEqual(islands.islandOfWorkgroup);
    expect(buf(device, 'islandOfNode').u32()).toEqual(islands.islandOfNode);
    expect(buf(device, 'order').u32()).toEqual(coloring.order);
    expect(buf(device, 'ends').u32()).toEqual(mesh.constraints.ends);
    expect(buf(device, 'rest').floats()).toEqual(mesh.constraints.rest);
    expect(buf(device, 'stiff').floats()).toEqual(mesh.constraints.stiffness);
  });

  it('packs the batch table as one padded slot a color, in color order', async () => {
    const { device, system } = await build();
    const table = buf(device, 'batch').u32();
    const batches = system.layout.coloring.batches;
    expect(table.length).toBe(batches.length * BATCH_STRIDE_WORDS);
    batches.forEach((batch, color) => {
      const slot = color * BATCH_STRIDE_WORDS;
      expect(table[slot], `batch ${color} base`).toBe(batch.base);
      expect(table[slot + 1], `batch ${color} count`).toBe(batch.count);
      // The padding is the mechanism, not waste: it puts every slot on a legal
      // bind-group offset, and it is what a neighbour's view has no way to reach.
      for (let word = SOFT_BATCH_U32_PER_COLOR; word < BATCH_STRIDE_WORDS; word++) {
        expect(table[slot + word], `batch ${color} padding ${word}`).toBe(0);
      }
    });
    // A slot only *means* a color if the table is tiled by color, which is the same
    // invariant the constructor throws on before it builds the per-color groups.
    expect(batches.map((b) => b.color)).toEqual(batches.map((_, i) => i));
  });

  it('leaves the sleep buffers and the counters un-uploaded, i.e. zero', async () => {
    const { device } = await build();
    // WebGPU zero-initialises, zero means awake and zero means "no error yet",
    // so a fresh allocation is exactly the state the first step needs.
    for (const label of ['asleep', 'quiet', 'islandSpeed', 'stats', 'pred', 'publish']) {
      expect(buf(device, label).bytes().every((b) => b === 0), label).toBe(true);
    }
    const uploaded = device.writes.map((w) => w.buffer.label);
    expect(uploaded).toEqual([
      'state',
      'nodeOrder',
      'islandOfWg',
      'islandOfNode',
      'order',
      'ends',
      'rest',
      'stiff',
      'batch',
      'params',
    ]);
  });

  it('builds the two shared groups plus one a color, and nothing else', async () => {
    const { device, system } = await build();
    const colors = system.plan.colors;
    expect(colors).toBeGreaterThan(1);
    expect(device.layouts.map((l) => l.label)).toEqual(['soft:group0', 'soft:group1']);
    // Two layouts for `2 + colors` groups: a solve group is a group-0 group, and
    // what differs between them is the range one binding covers, which is not part
    // of a layout. A layout a color would be a device object nothing needs.
    expect(device.bindGroups).toHaveLength(2 + colors);
  });

  it('narrows batchBuf to one slot a color, which is what makes a solve race-free', async () => {
    const { device, system } = await build();
    const batch = buf(device, 'batch');
    // The shared group-0 binds the whole table, and the five fixed kernels never
    // read it. Each color's group binds one slot at that color's offset, and a size
    // of one pair is what makes the kernel's `batchBuf[0u]` total: there is no
    // second element in range to index, so a color cannot solve another color.
    const shared = device.bindGroups[0]!.entryAt(BATCH_BINDING)!;
    expect(shared.buffer.label).toBe('batch');
    expect(shared.offset).toBeUndefined();
    expect(shared.size).toBeUndefined();
    for (let color = 0; color < system.plan.colors; color++) {
      const entry = device.bindGroups[2 + color]!.entryAt(BATCH_BINDING)!;
      expect(entry.buffer, `color ${color}`).toBe(batch);
      expect(entry.offset, `color ${color}`).toBe(color * SOFT_BATCH_STRIDE_BYTES);
      expect(entry.size, `color ${color}`).toBe(BATCH_SLOT_BYTES);
    }
  });

  it('binds every static resource to the slot the shader declares', async () => {
    const { device } = await build();
    const group = device.bindGroups[0]!;
    for (const binding of SOFT_BINDINGS.filter((b) => b.group === 0)) {
      const label = binding.name.replace(/Buf$/, '');
      expect(group.bufferAt(binding.binding)?.label, binding.name).toBe(label);
    }
    expect(group.entries).toHaveLength(9);
  });

  it('binds every state resource to the slot the shader declares', async () => {
    const { device } = await build();
    const group = device.bindGroups[1]!;
    for (const binding of SOFT_BINDINGS.filter((b) => b.group === 1)) {
      const label = binding.name.replace(/Buf$/, '');
      expect(group.bufferAt(binding.binding)?.label, binding.name).toBe(label);
    }
    expect(group.entries).toHaveLength(SOFT_STORAGE_BINDINGS - 8);
  });

  it('publishes frame zero, so a renderer that draws before stepping has positions', async () => {
    const { device, system } = await build();
    expect(device.passes).toHaveLength(1);
    const pass = passAt(device, 0);
    expect(pass.label).toBe('soft publish 0');
    expect(pass.entryPoints).toEqual(['publish']);
    expect(pass.dispatches).toEqual([softWorkgroups(system.count)]);
    expect(pass.ended).toBe(true);
  });

  it('seeds stats() from the mesh, so an unstepped system is not all zeros', async () => {
    const { system, mesh } = await build();
    expect(system.stats()).toEqual({
      escaped: 0,
      maxSpeed: mesh.maxSpeed(),
      maxConstraintError: mesh.maxConstraintError(),
      awakeIslands: system.plan.islands,
      sleepingIslands: 0,
      kineticEnergy: mesh.kineticEnergy(),
    });
    expect(system.stats().maxSpeed).toBeGreaterThan(0);
    expect(system.stats().kineticEnergy).toBeGreaterThan(0);
  });

  it('takes exactly one more reference on the shared device', async () => {
    const { shared } = await build();
    expect(shared.references).toBe(2);
  });

  it('uploads nothing for a graph with no edges', async () => {
    const { device, system } = await build({ mesh: cloud(4) });
    expect(system.plan.constraints).toBe(0);
    // Six uploads: the state, the two island maps, the padded node order and the
    // uniform. The four per-edge buffers have nothing to say.
    expect(device.writes.map((w) => w.buffer.label)).toEqual([
      'state',
      'nodeOrder',
      'islandOfWg',
      'islandOfNode',
      'params',
    ]);
  });
});

// ---------------------------------------------------------------------------
// the step chain
// ---------------------------------------------------------------------------

describe('the step chain', () => {
  it('runs the whole chain in one pass, in shader order', async () => {
    const { device, system } = await build();
    system.step();
    expect(device.passes).toHaveLength(2);
    const pass = passForStep(device, 0);
    expect(pass.label).toBe('soft step 0');
    expect(pass.ended).toBe(true);
    const solves = system.plan.iterations * system.plan.colors;
    expect(pass.entryPoints).toHaveLength(system.plan.dispatchesPerStep);
    expect(pass.entryPoints).toEqual([
      'predict',
      ...Array<string>(solves).fill('solve'),
      'finalize',
      'measure',
      'sleep_update',
      'publish',
    ]);
    // One submit for the whole step, which is the point of the per-color groups: a
    // uniform rewrite would only be visible after a submit, and this step has none.
    expect(device.submissions).toHaveLength(2);
  });

  it('sizes every kernel from the basis the shader declares', async () => {
    const { device, system } = await build();
    system.step();
    const pass = passForStep(device, 0);
    const dispatches = pass.dispatches;
    const solves = system.plan.iterations * system.plan.colors;
    expect(dispatches).toHaveLength(system.plan.dispatchesPerStep);
    expect(dispatches[0]).toBe(system.plan.nodeWorkgroups);
    // The tail after every solve, in `SOFT_KERNELS` order: finalize walks the same
    // padded node order predict did, then the graph, the islands and the nodes.
    expect(dispatches.slice(1 + solves)).toEqual([
      system.plan.nodeWorkgroups,
      softWorkgroups(system.mesh.constraints.count),
      softWorkgroups(system.plan.islands),
      softWorkgroups(system.count),
    ]);
  });

  it('carries each color on its own bind group, sized from its own batch', async () => {
    const { device, system } = await build();
    system.step();
    const pass = passForStep(device, 0);
    const batches = system.layout.coloring.batches;
    expect(batches.length).toBeGreaterThan(1);
    const groups = group0PerDispatch(pass);
    for (let k = 0; k < system.plan.iterations * batches.length; k++) {
      const batch = batches[k % batches.length]!;
      // x is the batch's own workgroup count, and the color rides on the binding.
      expect(pass.dispatches[1 + k], `solve ${k}`).toBe(softWorkgroups(batch.count));
      expect(groups[1 + k]!.entryAt(BATCH_BINDING)!.offset, `solve ${k}`).toBe(
        batch.color * SOFT_BATCH_STRIDE_BYTES,
      );
    }
    // The bug this pins shut: a dispatch dimension is a *parallel* axis, so a z of
    // `color + 1` would have run every color up to it inside one dispatch, with the
    // colors that share a node racing in exactly the way the coloring prevents --
    // and no implementation would have said a word about it.
    expect(groups[1]!.entryAt(BATCH_BINDING)!.offset).toBe(0);
    expect(pass.dispatches.every((x) => x > 0)).toBe(true);
  });

  it('repeats the batches once per solver iteration', async () => {
    const one = await build({ options: { iterations: 1 } });
    const five = await build({ options: { iterations: 5 } });
    one.system.step();
    five.system.step();
    const colors = one.system.plan.colors;
    expect(colors).toBe(five.system.plan.colors);
    expect(passForStep(one.device, 0).entryPoints.filter((k) => k === 'solve')).toHaveLength(colors);
    expect(passForStep(five.device, 0).entryPoints.filter((k) => k === 'solve')).toHaveLength(
      colors * 5,
    );
    one.system.dispose();
    five.system.dispose();
  });

  it('drops the measure dispatch for a graph with no edges, and nothing else', async () => {
    const { device, system } = await build({ mesh: cloud(5) });
    expect(system.plan.colors).toBe(0);
    system.step();
    const pass = passForStep(device, 0);
    // `measure` is sized from the constraint count, which is zero, and `submit`
    // skips a non-positive x rather than recording an empty dispatch.
    expect(pass.entryPoints).toEqual(['predict', 'finalize', 'sleep_update', 'publish']);
    // predict and finalize walk the padded node order, which gives every lone node
    // its own workgroup; sleep_update is a group an island, publish one per 64.
    expect(pass.dispatches).toEqual([5, 5, 1, 1]);
  });

  it('binds both groups before every dispatch, in ascending group order', async () => {
    const { device, system } = await build({ mesh: cloud(5) });
    system.step();
    const kinds = passForStep(device, 0).calls.map((c) =>
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
      'bind0',
      'bind1',
      'pipeline',
      'dispatch',
      'end',
    ]);
    // No ping-pong here: one group-1 binding serves the whole step, because the
    // coloring makes every write in it single-writer.
    expect(device.bindGroups).toHaveLength(2);
    expect(system.steps).toBe(1);
  });

  it('memoizes one pipeline per entry point across steps', async () => {
    const { device, system } = await build();
    system.advance(4);
    // The constructor publishes frame zero, so `publish` is compiled first and
    // every step reuses it. Six pipelines for six kernels, in the order the
    // device first saw them, and a set that is exactly `SOFT_KERNELS`.
    const compiled = device.pipelines.map((p) => p.entryPoint);
    expect(compiled).toEqual(['publish', ...SOFT_KERNELS.slice(0, 5)]);
    expect(new Set(compiled)).toEqual(new Set(SOFT_KERNELS));
    expect(device.pipelines.map((p) => p.label)).toEqual(compiled.map((k) => `soft:${k}`));
    // One pass per step, plus the frame-zero publish, and never a second pipeline.
    expect(device.passes).toHaveLength(5);
  });

  it('packs the uniform with writeSoftParams, so the shader sees what the CPU spec reads', async () => {
    const { device, system } = await build({
      options: {
        gravity: [1, -2, 3],
        damping: 0.25,
        restitution: 0.4,
        maxSpeed: 12,
        stiffness: 0.5,
        sleepThreshold: 0.5,
      },
    });
    system.step(1 / 30);
    const { floats } = paramsOf(device);
    expect(floats[SOFT_PARAM_WORD.dt]).toBeCloseTo(1 / 30, 6);
    expect(floats[SOFT_PARAM_WORD.invDt]).toBeCloseTo(30, 4);
    expect(floats[SOFT_PARAM_WORD.gravityX]).toBeCloseTo(1, 6);
    expect(floats[SOFT_PARAM_WORD.gravityY]).toBeCloseTo(-2, 6);
    expect(floats[SOFT_PARAM_WORD.gravityZ]).toBeCloseTo(3, 6);
    expect(floats[SOFT_PARAM_WORD.damping]).toBeCloseTo(0.25, 6);
    expect(floats[SOFT_PARAM_WORD.restitution]).toBeCloseTo(0.4, 6);
    expect(floats[SOFT_PARAM_WORD.maxSpeed]).toBeCloseTo(12, 6);
    expect(floats[SOFT_PARAM_WORD.stiffness]).toBeCloseTo(0.5, 6);
    expect(floats[SOFT_PARAM_WORD.sleepThresholdSq]).toBeCloseTo(0.25, 6);
    expect(floats[SOFT_PARAM_WORD.boundsMinX]).toBe(system.mesh.bounds.min[0]);
    expect(floats[SOFT_PARAM_WORD.boundsMaxY]).toBe(system.mesh.bounds.max[1]);
    expect(paramInt(device, SOFT_PARAM_WORD.count)).toBe(system.count);
    expect(paramInt(device, SOFT_PARAM_WORD.islandCount)).toBe(system.plan.islands);
    expect(paramInt(device, SOFT_PARAM_WORD.paddedNodes)).toBe(system.layout.islands.paddedNodes);
    expect(paramInt(device, SOFT_PARAM_WORD.constraintCount)).toBe(
      system.mesh.constraints.count,
    );
    expect(paramInt(device, SOFT_PARAM_WORD.sleepAfter)).toBe(system.options.sleepAfter);
  });

  it('re-uploads the uniform every step, so a new dt takes effect immediately', async () => {
    const { device, system } = await build();
    system.step(1 / 120);
    expect(paramsOf(device).floats[SOFT_PARAM_WORD.dt]).toBeCloseTo(1 / 120, 8);
    system.step(1 / 20);
    expect(paramsOf(device).floats[SOFT_PARAM_WORD.dt]).toBeCloseTo(1 / 20, 8);
    // One seed write plus one uniform write per step.
    expect(device.writes.filter((w) => w.buffer.label === 'params')).toHaveLength(3);
  });

  it('carries sleep and the bounds mode in the flags word', async () => {
    const plain = await build();
    plain.system.step();
    expect(paramInt(plain.device, SOFT_PARAM_WORD.flags)).toBe(
      SOFT_BOUNDS_MODE_BITS.reflect << SOFT_FLAG.boundsShift,
    );

    const sleeping = await build({ options: { sleep: true } });
    sleeping.system.step();
    expect(paramInt(sleeping.device, SOFT_PARAM_WORD.flags)).toBe(
      SOFT_FLAG.sleep | (SOFT_BOUNDS_MODE_BITS.reflect << SOFT_FLAG.boundsShift),
    );

    const unbounded = await build({ options: { boundsMode: 'none' } });
    unbounded.system.step();
    expect(paramInt(unbounded.device, SOFT_PARAM_WORD.flags)).toBe(
      SOFT_BOUNDS_MODE_BITS.none << SOFT_FLAG.boundsShift,
    );

    for (const built of [plain, sleeping, unbounded]) built.system.dispose();
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

  it('refuses a dt that is not finite and positive, and records nothing', async () => {
    const { device, system } = await build();
    const writes = device.writes.length;
    for (const dt of [0, -0.01, NaN, Infinity, -Infinity]) {
      expect(() => system.step(dt), `dt=${dt}`).toThrow(RangeError);
    }
    // The frame-zero publish is the only submission so far.
    expect(device.submissions).toHaveLength(1);
    expect(device.writes).toHaveLength(writes);
    expect(system.steps).toBe(0);
  });

  it('refuses a dispatch above the workgroup limit, and records nothing', async () => {
    const { device, system } = await build({
      // A ceiling of two admits the frame-zero publish, which is one workgroup
      // per 64 nodes, and so construction succeeds. The refusal then lands where
      // it belongs: on the step, whose `measure` is sized from a graph wide enough
      // that no color of it is.
      limits: softLimits({ maxComputeWorkgroupsPerDimension: 2 }),
    });
    expect(system.plan.nodeWorkgroups).toBeGreaterThan(1);
    // Every solve fits under the ceiling, so the dispatch that trips it is the one
    // sized from the constraint count, and the message names that kernel.
    for (const batch of system.layout.coloring.batches) {
      expect(batch.workgroups).toBeLessThanOrEqual(2);
    }
    expect(softWorkgroups(system.mesh.constraints.count)).toBe(6);
    expect(() => system.step()).toThrow(
      /soft step 0: measure needs 6 workgroups, above maxComputeWorkgroupsPerDimension=2/,
    );
    expect(device.submissions).toHaveLength(1);
    expect(device.passes).toHaveLength(1);
    // The uniform write already happened; the step counter did not.
    expect(system.steps).toBe(0);
  });

  it('refuses at construction when the frame-zero publish does not fit', async () => {
    // The publish that makes frame zero drawable is a dispatch like any other, so
    // an adapter that cannot run it fails while building rather than at the first
    // step -- and `soft.ts` turns that into a downgrade with a reason attached.
    await expect(
      build({ limits: softLimits({ maxComputeWorkgroupsPerDimension: 1 }) }),
    ).rejects.toThrow(
      /soft publish 0: publish needs 2 workgroups, above maxComputeWorkgroupsPerDimension=1/,
    );
  });
});

// ---------------------------------------------------------------------------
// readback
// ---------------------------------------------------------------------------

describe('readback', () => {
  it('decodes the counters and both bitcast maxima', async () => {
    const { device, system } = await build();
    const stats = buf(device, 'stats').u32();
    stats[SOFT_STAT_WORD.escaped] = 3;
    stats[SOFT_STAT_WORD.maxSpeedSq] = bitsOf(4);
    stats[SOFT_STAT_WORD.maxConstraintError] = bitsOf(0.25);
    stats[SOFT_STAT_WORD.sleepingIslands] = 1;
    const energy = system.stats().kineticEnergy;
    expect(await system.readStats()).toEqual({
      escaped: 3,
      maxSpeed: 2,
      maxConstraintError: 0.25,
      awakeIslands: system.plan.islands - 1,
      sleepingIslands: 1,
      kineticEnergy: energy,
    });
    // Cached, so `stats()` and the awaited value are the same object.
    expect(system.stats()).toBe(system.stats());
    expect(system.stats().escaped).toBe(3);
  });

  it('reports zero for a word that is not a positive f32', async () => {
    const { device, system } = await build();
    const stats = buf(device, 'stats').u32();
    for (const value of [-1, NaN, 0]) {
      stats[SOFT_STAT_WORD.maxSpeedSq] = bitsOf(value);
      stats[SOFT_STAT_WORD.maxConstraintError] = bitsOf(value);
      const read = await system.readStats();
      expect(read.maxSpeed, `maxSpeed of ${value}`).toBe(0);
      expect(read.maxConstraintError, `error of ${value}`).toBe(0);
    }
  });

  it('carries kinetic energy forward, because only a full readback can refresh it', async () => {
    const { device, system, mesh } = await build();
    buf(device, 'stats').u32()[SOFT_STAT_WORD.escaped] = 9;
    const before = mesh.kineticEnergy();
    await system.readStats();
    expect(system.stats().kineticEnergy).toBeCloseTo(before, 6);
    expect(system.stats().kineticEnergy).not.toBe(0);
  });

  it('pulls the state buffer into mesh.data', async () => {
    const { device, system, mesh } = await build();
    const state = buf(device, 'state').floats();
    for (let i = 0; i < state.length; i++) state[i] = (i + 1) / 64;
    const before = system.stats().kineticEnergy;
    expect(await system.readback()).toBe(mesh);
    expect(Array.from(mesh.data)).toEqual(Array.from(state));
    // The energy is recomputed from what came back, not carried over.
    expect(system.stats().kineticEnergy).toBeCloseTo(mesh.kineticEnergy(), 6);
    expect(system.stats().kineticEnergy).not.toBeCloseTo(before, 6);
  });

  it('destroys its staging buffer and leaves nothing mapped', async () => {
    const { device, system } = await build();
    await system.readStats();
    const staging = device.buffers.filter((b) => b.label.startsWith('readback staging'));
    expect(staging).toHaveLength(1);
    expect(staging[0]!.size).toBe(SOFT_STAT_WORDS * 4);
    expect(staging[0]!.destroyed).toBe(true);
    expect(staging[0]!.mapCalls).toBe(1);
    expect(staging[0]!.unmapCalls).toBe(1);
    expect(staging[0]!.lastMapMode).toBe(CONSTANTS.mapMode.READ);
    // Staging is untracked, so teardown does not try to destroy it twice.
    expect(system.context.buffers.some((b) => b.label.startsWith('readback staging'))).toBe(false);
  });

  it('reports a digest that is only as fresh as the last readback', async () => {
    const { device, system, mesh } = await build();
    const seeded = system.digest();
    expect(seeded).toBe(mesh.digest());
    buf(device, 'state').floats()[0] = 1234.5;
    // Stale until the state comes back: documented, and asserted so it stays so.
    expect(system.digest()).toBe(seeded);
    await system.readback();
    expect(system.digest()).not.toBe(seeded);
    expect(system.digest()).toBe(mesh.digest());
    expect(system.digest().endsWith(`:${mesh.data.length}`)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// sleeping
// ---------------------------------------------------------------------------

describe('wake', () => {
  it('zeroes all three sleep buffers and reports every island awake', async () => {
    const { device, system } = await build({ mesh: seededMesh({ scene: 'sheets', groups: 4, count: 400 }) });
    expect(system.plan.islands).toBeGreaterThan(1);
    for (const label of ['asleep', 'quiet', 'islandSpeed']) {
      buf(device, label).u32().fill(7);
    }
    buf(device, 'stats').u32()[SOFT_STAT_WORD.sleepingIslands] = system.plan.islands;
    await system.readStats();
    expect(system.stats().sleepingIslands).toBe(system.plan.islands);
    expect(system.stats().awakeIslands).toBe(0);

    system.wake();
    for (const label of ['asleep', 'quiet', 'islandSpeed']) {
      expect(buf(device, label).bytes().every((b) => b === 0), label).toBe(true);
    }
    expect(system.stats().sleepingIslands).toBe(0);
    expect(system.stats().awakeIslands).toBe(system.plan.islands);
    // The counters word is left alone: it belongs to the step that wrote it.
    expect(buf(device, 'stats').u32()[SOFT_STAT_WORD.sleepingIslands]).toBe(system.plan.islands);
  });

  it('writes through the queue, so the clear is ordered against recorded steps', async () => {
    const { device, system } = await build();
    system.step();
    const before = device.writes.length;
    system.wake();
    const wakes = device.writes.slice(before);
    expect(wakes.map((w) => w.buffer.label)).toEqual(['asleep', 'quiet', 'islandSpeed']);
    expect(wakes.every((w) => w.bytes === system.plan.islands * 4)).toBe(true);
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
    const { device, shared, system } = await build();
    seedPublish(device);
    const ctx = new ComputeContext(shared);
    const target = ctx.storageBuffer(system.budget.publish, 'positions');
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
    const { device, system } = await build();
    seedPublish(device);
    const attribute = device.createBuffer({
      label: 'three.js position attribute',
      size: system.budget.publish,
      usage: U.VERTEX | U.COPY_DST,
    });
    expect(system.copyPublishedTo(attribute)).toBe(system.budget.publish);
    expect(new Float32Array(attribute.store)).toEqual(buf(device, 'publish').floats());
  });

  it('clamps an over-large request and skips an empty one', async () => {
    const { device, system } = await build();
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

  it('publishes tight xyz, the layout the CPU tier publishes too', async () => {
    const { device, system, mesh } = await build();
    expect(system.budget.publish).toBe(mesh.count * 3 * 4);
    // Put a recognisable position into the state buffer, run the publish kernel's
    // copy on the CPU side, and compare: the layout is the contract render/soft.ts
    // depends on, and it is the same one `copyPublishedTo` on the CPU tier writes.
    const state = buf(device, 'state').floats();
    for (let i = 0; i < mesh.count; i++) {
      const o = i * SOFT_STRIDE;
      state[o] = i + 0.25;
      state[o + 1] = i + 0.5;
      state[o + 2] = i + 0.75;
    }
    const cpu = createCpuSoftSystem({ mesh });
    const target = new Float32Array(mesh.count * 3);
    expect(cpu.copyPublishedTo(target)).toBe(target.byteLength);
    expect(target[0]).toBeCloseTo(mesh.position(0)[0], 6);
    expect(buf(device, 'publish').size).toBe(target.byteLength);
    system.dispose();
  });
});

// ---------------------------------------------------------------------------
// refusing a device that cannot run this pipeline
// ---------------------------------------------------------------------------

describe('refusing a device that cannot run it', () => {
  it('rejects before allocating when the device binds fewer storage buffers', async () => {
    // The WebGPU baseline, which is what the stub reports by default.
    const { device, shared } = stubbed(stubLimits({ maxStorageBuffersPerShaderStage: 8 }));
    await expect(
      createGpuSoftSystem({ shared, mesh: seededMesh() }),
    ).rejects.toThrow(
      new RegExp(`binds ${SOFT_STORAGE_BINDINGS} storage buffers.*maxStorageBuffersPerShaderStage=8`),
    );
    // Nothing allocated, and no reference taken: the caller still owns exactly the
    // handle they acquired, and the fallback path starts clean.
    expect(device.buffers).toHaveLength(0);
    expect(shared.references).toBe(1);
  });

  it('accepts a device that reports exactly the binding count', async () => {
    const { shared } = stubbed(
      stubLimits({ maxStorageBuffersPerShaderStage: SOFT_STORAGE_BINDINGS }),
    );
    const system = await createGpuSoftSystem({ shared, mesh: seededMesh() });
    expect(SOFT_STORAGE_BINDINGS).toBe(15);
    expect(system.program.bindings.filter((b) => b.bufferType !== 'uniform')).toHaveLength(15);
    system.dispose();
  });

  it('treats a zero limit as unreported rather than as a ceiling', async () => {
    const { shared } = stubbed(stubLimits({ maxStorageBuffersPerShaderStage: 0 }));
    const system = await createGpuSoftSystem({ shared, mesh: seededMesh() });
    expect(system.budget.largest).toBeGreaterThan(0);
    system.dispose();
  });

  it('rejects before allocating when the largest buffer exceeds the binding limit', async () => {
    const { device, shared } = stubbed(softLimits({ maxStorageBufferBindingSize: 1024 }));
    await expect(createGpuSoftSystem({ shared, mesh: seededMesh() })).rejects.toThrow(
      /above this device's 1024-byte storage limit/,
    );
    expect(device.buffers).toHaveLength(0);
    expect(shared.references).toBe(1);
  });

  it('takes the smaller of maxBufferSize and the binding limit', async () => {
    const { shared } = stubbed(
      softLimits({ maxStorageBufferBindingSize: 1 << 20, maxBufferSize: 1024 }),
    );
    await expect(createGpuSoftSystem({ shared, mesh: seededMesh() })).rejects.toThrow(/1024-byte/);
  });

  it('rejects a shader that fails to compile, and leaves nothing behind', async () => {
    const { device, shared } = stubbed();
    device.setCompilationMessages([{ type: 'error', message: 'unknown identifier', lineNum: 12 }]);
    const error = await createGpuSoftSystem({ shared, mesh: seededMesh() }).then(
      () => null,
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(ShaderCompilationError);
    expect((error as ShaderCompilationError).label).toBe('soft');
    expect((error as ShaderCompilationError).message).toContain('line 12');
    expect(device.buffers).toHaveLength(0);
    expect(shared.references).toBe(1);
  });

  it('refuses a device that is already gone', async () => {
    const { device, shared } = stubbed();
    shared.destroy();
    expect(device.destroyed).toBe(true);
    await expect(createGpuSoftSystem({ shared, mesh: seededMesh() })).rejects.toThrow(
      /the shared device is destroyed/,
    );
  });

  it('refuses a device that reported itself lost', async () => {
    const { device, shared } = stubbed();
    device.lose({ reason: 'app-initiated', message: 'gone' });
    await flushMicrotasks();
    expect(shared.lost).toBe(true);
    await expect(createGpuSoftSystem({ shared, mesh: seededMesh() })).rejects.toThrow(
      /the shared device is lost/,
    );
  });

  it('refuses something that is not a mesh', async () => {
    const { shared } = stubbed();
    for (const mesh of [null, undefined, {}, { count: 0 }, { count: 2.5 }]) {
      await expect(
        createGpuSoftSystem({ shared, mesh: mesh as unknown as SoftMesh }),
      ).rejects.toThrow(TypeError);
    }
    expect(shared.references).toBe(1);
  });

  it('refuses a mesh whose box is thinner than a node', async () => {
    const { shared } = stubbed();
    const thin = new SoftMesh({
      count: 4,
      scene: 'rope',
      radius: 0.6,
      bounds: { min: [-1, -0.5, -1], max: [1, 0.5, 1] },
    });
    await expect(createGpuSoftSystem({ shared, mesh: thin })).rejects.toThrow(
      /cannot contain a node of diameter/,
    );
    expect(shared.references).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// teardown
// ---------------------------------------------------------------------------

describe('teardown', () => {
  it('destroys every buffer, drops one reference and is idempotent', async () => {
    const { device, shared, system } = await build();
    const buffers = device.buffers.length;
    expect(shared.references).toBe(2);
    system.dispose();
    expect(system.disposed).toBe(true);
    expect(device.buffers.filter((b) => b.destroyed)).toHaveLength(buffers);
    expect(shared.references).toBe(1);
    // The caller's handle is untouched: disposing a system is not destroying a device.
    expect(device.destroyed).toBe(false);
    expect(shared.usable).toBe(true);
    system.dispose();
    expect(shared.references).toBe(1);
  });

  it('makes every entry point after disposal a loud error', async () => {
    const { shared, system } = await build();
    system.dispose();
    expect(() => system.step()).toThrow(/has been disposed/);
    expect(() => system.wake()).toThrow(/has been disposed/);
    expect(() => system.advance(1)).toThrow(/has been disposed/);
    await expect(system.readStats()).rejects.toThrow(/has been disposed/);
    await expect(system.readback()).rejects.toThrow(/has been disposed/);
    expect(() => system.copyPublishedTo(system.publish)).toThrow(/has been disposed/);
    expect(shared.usable).toBe(true);
  });

  it('stops stepping once the device is lost', async () => {
    const { device, system } = await build();
    expect(system.lost).toBe(false);
    device.lose({ reason: 'unknown', message: 'gone' });
    await flushMicrotasks();
    expect(system.lost).toBe(true);
    expect(() => system.step()).toThrow(/the shared device is lost/);
  });

  it('leaves the buffers to the context, which is what dispose destroys', async () => {
    const { system } = await build();
    expect(system.context.buffers.map((b) => b.label)).toEqual([...LABELS]);
    expect(system.context.destroyed).toBe(false);
    system.dispose();
    expect(system.context.destroyed).toBe(true);
    expect(system.context.buffers).toHaveLength(0);
  });
});

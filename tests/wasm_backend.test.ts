/**
 * The wasm backend's spec: bit-exactness with `builtin`, backend-contract
 * parity, and the performance floor the milestone is justified by.
 *
 * "Deterministic" is only a useful claim if it is checked against something.
 * `builtin.ts` is that something -- it stays the normative implementation, and
 * every assertion here is `toBe`/bit-level rather than `toBeCloseTo`. A solver
 * that agrees to six decimals and diverges at the seventh still breaks replays
 * and reward-curve comparisons, so approximations are not accepted anywhere in
 * this file except the benchmark.
 *
 * The artifact under test is `wasm/pkg`, committed to the repo and rebuilt by
 * `npm run build:wasm`. `scripts/check_wasm_artifact.mjs` is what proves the
 * committed bytes still behave like the Rust source in CI.
 */

import { describe, expect, it } from 'vitest';

import { Rng } from '../src/core/rng.js';
import { Trainer } from '../src/ai/trainer.js';
import { GaussianPolicy } from '../src/ai/policy.js';
import { DriveEnv } from '../src/envs/drive.js';
import { ReachEnv } from '../src/envs/reach.js';
import { BuiltinPhysics, createBuiltinPhysics } from '../src/physics/builtin.js';
import {
  REFERENCE_GOLDEN_DIGEST,
  REFERENCE_OPTIONS,
  REFERENCE_STEPS,
  buildReferenceScene,
  runReference,
} from '../src/physics/reference.js';
import {
  WASM_ABI_VERSION,
  WasmPhysics,
  createWasmPhysics,
  kernelFrom,
  loadWasmKernel,
  memoryFromBrowser,
  memoryFromNode,
  wasmUrlFor,
  type WasmGlue,
  type WasmKernel,
} from '../src/physics/wasm.js';
import { vec3, type PhysicsBackend, type Vec3 } from '../src/physics/types.js';

const DT = 1 / 60;

/**
 * The reference digest, recorded in `src/physics/reference.ts` so the browser
 * spec asserts the very same string this file does.
 *
 * Both backends must produce exactly this. Pinning it (rather than only
 * comparing the two to each other) is what catches a change to `builtin.ts`:
 * without it, editing the reference solver and its port in the same commit would
 * keep this file green while silently invalidating every replay ever recorded.
 * Regenerate it only when the change to the solver is deliberate.
 */
const GOLDEN_DIGEST = REFERENCE_GOLDEN_DIGEST;

/** One kernel for the whole file: instantiating wasm per test would dominate. */
const kernel = await loadWasmKernel();

async function wasmBackend(options = REFERENCE_OPTIONS): Promise<WasmPhysics> {
  return createWasmPhysics({ ...options, kernel });
}

/** Bit-exact equality for a whole body state, including -0 vs 0 and NaN. */
function expectStateEqual(a: PhysicsBackend, b: PhysicsBackend, handle: number): void {
  const sa = a.getBodyState(handle);
  const sb = b.getBodyState(handle);
  expect(sa === undefined).toBe(sb === undefined);
  if (!sa || !sb) return;
  const keys = ['position', 'rotation', 'velocity', 'angularVelocity'] as const;
  for (const key of keys) {
    for (let i = 0; i < 3; i++) {
      // `toBe` is `Object.is`, which distinguishes -0 from 0 and matches NaN
      // with NaN -- i.e. it compares doubles the way the CPU stores them.
      expect(sa[key][i], `${key}[${i}]`).toBe(sb[key][i]);
    }
  }
}

describe('the committed wasm artifact', () => {
  it('loads, and reports the ABI version src/physics/wasm.ts expects', () => {
    expect(kernel.bindings.td_abi_version()).toBe(WASM_ABI_VERSION);
  });

  it('really carries SIMD128', () => {
    // The whole point of the Rust port is the vectorised integration pass. A
    // build without `+simd128` still passes every correctness test, so this is
    // the only thing that would notice the RUSTFLAGS went missing.
    expect(kernel.simdEnabled).toBe(true);
    expect(kernel.bindings.td_simd_enabled()).toBe(true);
  });

  it('passes its own SIMD/scalar parity self-test inside the shipped binary', () => {
    expect(kernel.bindings.td_selftest_simd_parity()).toBe(true);
  });

  it('caches one instantiation, so many worlds share one linear memory', async () => {
    expect(await loadWasmKernel()).toBe(kernel);
    expect(kernel.memory.buffer.byteLength).toBeGreaterThan(0);
  });

  it('resolves the paired .wasm next to any glue URL', () => {
    expect(wasmUrlFor().href).toMatch(/threedream_physics_wasm_bg\.wasm$/);
    const custom = wasmUrlFor('https://cdn.example/threedream/pkg/glue.js');
    expect(custom.href).toBe('https://cdn.example/threedream/pkg/glue_bg.wasm');
    expect(wasmUrlFor(new URL('https://cdn.example/a/b.js')).href).toBe(
      'https://cdn.example/a/b_bg.wasm',
    );
  });

  it('initialises through the browser path when the glue can fetch', async () => {
    // vitest runs in Node, so the browser branch is unreachable through
    // `loadWasmKernel`; this drives it directly with a stub glue, which is why
    // `memoryFromBrowser` takes the glue as a parameter.
    const memory = kernel.memory;
    let called = 0;
    const glue = {
      default: async () => {
        called++;
        return { memory };
      },
    } as unknown as WasmGlue;
    expect(await memoryFromBrowser(glue)).toBe(memory);
    expect(called).toBe(1);

    const namedInit = { init: async () => ({ memory }) } as unknown as WasmGlue;
    expect(await memoryFromBrowser(namedInit)).toBe(memory);

    const broken = {} as unknown as WasmGlue;
    await expect(memoryFromBrowser(broken)).rejects.toThrow(/no init function/);
  });

  it('initialises through the Node path by reading the bytes', async () => {
    const glue = (await import('../wasm/pkg/threedream_physics_wasm.js')) as unknown as WasmGlue;
    const memory = await memoryFromNode(glue);
    expect(memory).toBe(kernel.memory);
  });

  it('refuses a stale artifact instead of misreading a buffer layout', () => {
    const stale = {
      ...kernel.bindings,
      td_abi_version: () => WASM_ABI_VERSION + 1,
    } as unknown as WasmGlue;
    expect(() => kernelFrom(stale, kernel.memory)).toThrow(/ABI mismatch/);
    expect(() => kernelFrom(stale, kernel.memory)).toThrow(/build:wasm/);
    expect(kernelFrom(kernel.bindings as WasmGlue, kernel.memory).simdEnabled).toBe(true);
  });
});

describe('wasm is bit-identical to builtin', () => {
  it('assigns the same handles to the same descriptors', async () => {
    // Everything downstream compares bodies by handle, so handle agreement is
    // the precondition: if the kernel numbered differently, every later
    // assertion would be comparing two different bodies and still passing.
    const builtin = createBuiltinPhysics({ ...REFERENCE_OPTIONS });
    const wasm = await wasmBackend();
    expect(buildReferenceScene(wasm)).toEqual(buildReferenceScene(builtin));
    expect(wasm.bodyCount).toBe(builtin.bodyCount);
    wasm.destroy();
  });

  it('reproduces the recorded reference digest on builtin', () => {
    const builtin = createBuiltinPhysics({ ...REFERENCE_OPTIONS });
    const run = runReference(builtin);
    expect(run.digest).toBe(GOLDEN_DIGEST);
    expect(run.samples.length).toBe(REFERENCE_STEPS * 9);
    expect(run.contacts.length).toBeGreaterThan(0);
  });

  it('matches the wasm backend value for value over the whole reference run', async () => {
    const builtin = createBuiltinPhysics({ ...REFERENCE_OPTIONS });
    const wasm = await wasmBackend();
    const runA = runReference(builtin);
    const runB = runReference(wasm);

    expect(runB.digest).toBe(runA.digest);
    expect(runB.digest).toBe(GOLDEN_DIGEST);
    // The digest is the gate; these make a failure diagnosable rather than a
    // 16-character hex difference with no idea which phase diverged.
    expect(runB.samples.length).toBe(runA.samples.length);
    expect(runB.contacts.length).toBe(runA.contacts.length);
    expect(runB.rays.length).toBe(runA.rays.length);
    // Labels are strings and cannot live in the digest, so they get their own
    // assertion: they are what a reward function keys on.
    expect(runB.contacts.map((c) => c.labels)).toEqual(runA.contacts.map((c) => c.labels));
    expect(runB.contacts).toEqual(runA.contacts);
    expect(runB.rays).toEqual(runA.rays);
    for (let i = 0; i < runA.samples.length; i++) {
      const a = runA.samples[i]!;
      const b = runB.samples[i]!;
      expect(b.step, `sample ${i} step`).toBe(a.step);
      for (const key of ['position', 'rotation', 'velocity', 'angularVelocity'] as const) {
        for (let k = 0; k < 3; k++) expect(b[key][k], `sample ${i} ${key}[${k}]`).toBe(a[key][k]);
      }
    }
    wasm.destroy();
  });

  it('stays identical through free fall, bounce, rest, and slide', async () => {
    const options = { ...REFERENCE_OPTIONS, linearDamping: 0, angularDamping: 0 };
    const builtin = createBuiltinPhysics(options);
    const wasm = await wasmBackend(options);
    for (const world of [builtin, wasm]) {
      world.createBody({
        shape: { kind: 'box', halfExtents: vec3(5, 0.5, 5) },
        position: vec3(0, -0.5, 0),
        kind: 'static',
        friction: 0.8,
        label: 'floor',
      });
      world.createBody({
        shape: { kind: 'sphere', radius: 0.4 },
        position: vec3(0, 4, 0),
        velocity: vec3(2.5, 0, -1),
        mass: 1.5,
        restitution: 0.6,
        friction: 0.35,
        label: 'ball',
      });
    }
    for (let i = 0; i < 400; i++) {
      builtin.step(DT);
      wasm.step(DT);
      expectStateEqual(builtin, wasm, 2);
      if (i % 40 === 0) {
        expect(wasm.drainContacts()).toEqual(builtin.drainContacts());
      } else {
        builtin.drainContacts();
        wasm.drainContacts();
      }
    }
    wasm.destroy();
  });

  it('matches on every mutation the interface exposes', async () => {
    const builtin = createBuiltinPhysics({ ...REFERENCE_OPTIONS });
    const wasm = await wasmBackend();
    const build = (world: PhysicsBackend): number[] => [
      world.createBody({
        shape: { kind: 'box', halfExtents: vec3(4, 0.5, 4) },
        position: vec3(0, -0.5, 0),
        kind: 'static',
      }),
      world.createBody({
        shape: { kind: 'sphere', radius: 0.3 },
        position: vec3(0, 1, 0),
        mass: 2,
        label: 'a',
      }),
      world.createBody({
        shape: { kind: 'box', halfExtents: vec3(0.2, 0.2, 0.2) },
        position: vec3(1, 1, 0),
        mass: 0.5,
        friction: 0.2,
        restitution: 0.9,
        label: 'b',
      }),
    ];
    const [fa, a1, b1] = build(builtin);
    const [fb, a2, b2] = build(wasm);
    expect([fa, a1, b1]).toEqual([fb, a2, b2]);

    for (const w of [builtin, wasm]) w.applyImpulse(a1, vec3(0.5, 1, -0.25));
    for (const w of [builtin, wasm]) w.applyForce(b1, vec3(12, 0, 3));
    for (let i = 0; i < 30; i++) {
      builtin.step(DT);
      wasm.step(DT);
      expectStateEqual(builtin, wasm, a1);
      expectStateEqual(builtin, wasm, b1);
    }

    // Partial state writes: only the named triples may change.
    const partial = { position: vec3(-1, 2, 0.5), velocity: vec3(0.25, 0, 0) };
    builtin.setBodyState(b1, partial);
    wasm.setBodyState(b2, partial);
    expectStateEqual(builtin, wasm, b1);
    expect(wasm.getBodyState(b2)!.rotation).toEqual(builtin.getBodyState(b1)!.rotation);
    expect(wasm.getBodyState(b2)!.angularVelocity).toEqual(
      builtin.getBodyState(b1)!.angularVelocity,
    );

    // A no-op write must not touch anything, including the cached AABB.
    builtin.setBodyState(b1, {});
    wasm.setBodyState(b2, {});
    for (let i = 0; i < 10; i++) {
      builtin.step(DT);
      wasm.step(DT);
      expectStateEqual(builtin, wasm, b1);
    }

    // Destroying a middle body shifts every slot after it.
    builtin.destroyBody(a1);
    wasm.destroyBody(a2);
    expect(wasm.bodyCount).toBe(builtin.bodyCount);
    expect(wasm.getBodyState(a2)).toBeUndefined();
    for (let i = 0; i < 60; i++) {
      builtin.step(DT);
      wasm.step(DT);
      expectStateEqual(builtin, wasm, b1);
    }

    const rayArgs: [Vec3, Vec3, number][] = [
      [vec3(0, 6, 0), vec3(0, -1, 0), 20],
      [vec3(-6, 0.5, 3), vec3(1, 0, -0.4), 15],
      [vec3(0, 0, 0), vec3(0, 0, 0), 5],
      [vec3(0, 0, 0), vec3(0, -1, 0), 0.001],
    ];
    for (const [origin, dir, max] of rayArgs) {
      expect(wasm.raycast(origin, dir, max)).toEqual(builtin.raycast(origin, dir, max));
    }
    wasm.destroy();
  });

  it('agrees on contact labels, including the generated default', async () => {
    const builtin = createBuiltinPhysics({ ...REFERENCE_OPTIONS });
    const wasm = await wasmBackend();
    for (const world of [builtin, wasm]) {
      // One labelled, one not: the default is `body:{handle}`, which the wasm
      // side can only produce after the kernel assigns the handle.
      world.createBody({
        shape: { kind: 'sphere', radius: 0.5 },
        position: vec3(0, 0, 0),
        kind: 'static',
        label: 'ground',
      });
      world.createBody({
        shape: { kind: 'sphere', radius: 0.5 },
        position: vec3(0.6, 0, 0),
        mass: 1,
      });
    }
    builtin.step(DT);
    wasm.step(DT);
    const ca = builtin.drainContacts();
    const cb = wasm.drainContacts();
    expect(cb).toEqual(ca);
    expect(cb[0]!.labels).toEqual(['ground', 'body:2']);
    // Draining is destructive on both sides.
    expect(wasm.drainContacts()).toEqual([]);
    expect(builtin.drainContacts()).toEqual([]);
    wasm.destroy();
  });

  it('grows its contact buffer without losing exactness', async () => {
    // A pile of overlapping spheres produces many contacts in one step, which is
    // what forces the scratch buffer to realloc. Reallocating wrongly would show
    // up as garbage in the tail of the array rather than as a crash.
    const builtin = createBuiltinPhysics({ ...REFERENCE_OPTIONS });
    const wasm = await wasmBackend();
    for (const world of [builtin, wasm]) {
      for (let i = 0; i < 12; i++) {
        world.createBody({
          shape: { kind: 'sphere', radius: 0.5 },
          position: vec3(i * 0.4, 0, 0),
          mass: 1,
        });
      }
    }
    builtin.step(DT);
    wasm.step(DT);
    const cb = wasm.drainContacts();
    expect(cb.length).toBeGreaterThan(8);
    expect(cb).toEqual(builtin.drainContacts());
    wasm.destroy();
  });
});

describe('WasmPhysics honours the backend contract', () => {
  it('reports itself as deterministic and named', async () => {
    const wasm = await wasmBackend();
    expect(wasm.name).toBe('wasm');
    expect(wasm.deterministic).toBe(true);
    expect(wasm.simdEnabled).toBe(true);
    wasm.destroy();
  });

  it('resolves the same defaults as builtin', async () => {
    const wasm = await createWasmPhysics({ kernel });
    const builtin = createBuiltinPhysics();
    expect(wasm.fixedDt).toBe(builtin.fixedDt);
    expect(wasm.fixedDt).toBe(DT);
    // The kernel stores what the adapter resolved, so ask it rather than
    // assuming which world id this instance got.
    expect(kernel.bindings.td_world_fixed_dt(wasm.worldId)).toBe(DT);
    wasm.destroy();
  });

  it('rejects capsules with builtin\'s exact error', async () => {
    const wasm = await wasmBackend();
    const descriptor = {
      shape: { kind: 'capsule' as const, radius: 0.1, halfHeight: 0.2 },
      position: vec3(),
    };
    const builtin = createBuiltinPhysics();
    let builtinMessage = '';
    try {
      builtin.createBody(descriptor);
    } catch (e) {
      builtinMessage = (e as Error).message;
    }
    expect(() => wasm.createBody(descriptor)).toThrow(new RegExp(builtinMessage));
    expect(() => wasm.createBody(descriptor)).toThrow(/capsule/i);
    wasm.destroy();
  });

  it('rejects negative mass with a RangeError, like builtin', async () => {
    const wasm = await wasmBackend();
    expect(() =>
      wasm.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), mass: -1 }),
    ).toThrowError(RangeError);
    expect(() =>
      wasm.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), mass: -1 }),
    ).toThrow(/mass must be non-negative/);
    expect(wasm.bodyCount).toBe(0);
    wasm.destroy();
  });

  it('keeps handing out fresh handles across dispose, like builtin', async () => {
    const wasm = await wasmBackend();
    const first = wasm.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3() });
    expect(first).toBe(1);
    wasm.dispose();
    expect(wasm.bodyCount).toBe(0);
    expect(wasm.drainContacts()).toEqual([]);
    const second = wasm.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3() });
    expect(second).toBe(2);
    // Still usable after dispose, which is what `dispose` (not `destroy`) means.
    wasm.step(DT);
    expect(wasm.getBodyState(second)).toBeDefined();
    wasm.destroy();
  });

  it('ignores impulses, forces and state writes to unknown or static bodies', async () => {
    const builtin = createBuiltinPhysics({ ...REFERENCE_OPTIONS });
    const wasm = await wasmBackend();
    for (const world of [builtin, wasm]) {
      world.createBody({
        shape: { kind: 'sphere', radius: 0.3 },
        position: vec3(0, 0, 0),
        kind: 'static',
      });
    }
    for (const world of [builtin, wasm]) {
      world.applyImpulse(1, vec3(10, 10, 10));
      world.applyForce(1, vec3(10, 10, 10));
      world.setBodyState(999, { position: vec3(1, 1, 1) });
      world.applyImpulse(999, vec3(1, 1, 1));
      world.destroyBody(999);
    }
    builtin.step(DT);
    wasm.step(DT);
    expectStateEqual(builtin, wasm, 1);
    expect(wasm.getBodyState(999)).toBeUndefined();
    wasm.destroy();
  });

  it('keeps independent worlds in one kernel from interfering', async () => {
    const zeroG = await wasmBackend({ ...REFERENCE_OPTIONS, gravity: vec3(0, 0, 0) });
    const earthG = await wasmBackend();
    for (const world of [zeroG, earthG]) {
      world.createBody({ shape: { kind: 'sphere', radius: 0.2 }, position: vec3(0, 5, 0) });
    }
    for (let i = 0; i < 30; i++) {
      zeroG.step(DT);
      earthG.step(DT);
    }
    expect(zeroG.getBodyState(1)!.position[1]).toBe(5);
    expect(earthG.getBodyState(1)!.position[1]).toBeLessThan(5);
    zeroG.destroy();
    earthG.destroy();
  });

  it('destroy releases the world id, and later calls fail loudly', async () => {
    const wasm = await wasmBackend();
    wasm.createBody({ shape: { kind: 'sphere', radius: 0.2 }, position: vec3() });
    wasm.destroy();
    expect(wasm.bodyCount).toBe(0);
    expect(() =>
      wasm.createBody({ shape: { kind: 'sphere', radius: 0.2 }, position: vec3(1, 0, 0) }),
    ).toThrow(/not live/);
    // Reads degrade to "no such body" rather than throwing.
    expect(wasm.getBodyState(1)).toBeUndefined();
    expect(wasm.raycast(vec3(), vec3(0, -1, 0), 10)).toBeUndefined();
    expect(wasm.drainContacts()).toEqual([]);
    wasm.step(DT);
    wasm.destroy();
  });
});

describe('ABI sentinels are the documented ones', () => {
  // These call the raw bindings, which is the only way to observe the error
  // channel at all: the adapter translates every sentinel into an exception
  // before a caller can see it. If a sentinel value changes in Rust, the
  // adapter's translation silently becomes a wrong exception type.
  const DEAD_WORLD = 0xfffffffe;

  it('uses negative codes for create failures', async () => {
    const wasm = await wasmBackend();
    const sphere = (worldId: number, mass: number, shapeKind = 0) =>
      kernel.bindings.td_body_create(
        worldId, shapeKind, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true, mass,
        0.1, 0.7, 1, 0xffffffff,
      );
    expect(sphere(wasm.worldId, -2)).toBe(-1);
    expect(sphere(wasm.worldId, 1, 7)).toBe(-3);
    expect(sphere(DEAD_WORLD, 1)).toBe(-2);
    wasm.destroy();
  });

  it('degrades quietly on a dead world id rather than trapping', () => {
    expect(kernel.bindings.td_world_body_count(DEAD_WORLD)).toBe(0);
    expect(kernel.bindings.td_world_contact_count(DEAD_WORLD)).toBe(0);
    expect(Number.isNaN(kernel.bindings.td_world_fixed_dt(DEAD_WORLD))).toBe(true);
    expect(kernel.bindings.td_world_step(DEAD_WORLD, DT)).toBeUndefined();
    expect(kernel.bindings.td_world_drain_contacts(DEAD_WORLD, 0)).toBe(0);
    expect(kernel.bindings.td_world_write_states(DEAD_WORLD, 0)).toBe(0);
    expect(kernel.bindings.td_world_write_handles(DEAD_WORLD, 0)).toBe(0);
    expect(kernel.bindings.td_world_destroy(DEAD_WORLD)).toBeUndefined();
    expect(kernel.bindings.td_world_dispose(DEAD_WORLD)).toBeUndefined();
  });

  it('refuses a zero-sized allocation, so a caller bug cannot alias', () => {
    expect(kernel.bindings.td_alloc(0)).toBe(0);
    const ptr = kernel.bindings.td_alloc(4);
    expect(ptr).toBeGreaterThan(0);
    expect(ptr % 8).toBe(0);
    // Freeing with the wrong count would be unsound; freeing twice with the
    // right one is what the adapter does exactly once per buffer.
    kernel.bindings.td_free(ptr, 4);
    expect(kernel.bindings.td_free(0, 4)).toBeUndefined();
  });

  it('writes handles and states in matching order', async () => {
    const wasm = await wasmBackend();
    for (let i = 0; i < 5; i++) {
      wasm.createBody({ shape: { kind: 'sphere', radius: 0.2 }, position: vec3(i, 0, 0) });
    }
    wasm.destroyBody(3);
    const worldId = wasm.worldId;
    const count = kernel.bindings.td_world_body_count(worldId);
    expect(count).toBe(4);
    const ptr = kernel.bindings.td_alloc(count * 12 + count);
    const handles = kernel.bindings.td_world_write_handles(worldId, ptr);
    expect(handles).toBe(count);
    const writtenHandles = Array.from(new Uint32Array(kernel.memory.buffer, ptr, handles));
    expect(writtenHandles).toEqual([1, 2, 4, 5]);
    const states = kernel.bindings.td_world_write_states(worldId, ptr);
    expect(states).toBe(count);
    // Snapshot both buffers into plain arrays before calling back into the
    // adapter. `getBodyState` allocates from its own scratch, and if that
    // `td_alloc` has to `memory.grow`, every view over the old buffer detaches
    // and silently reads back `undefined` -- which would look like a kernel bug
    // rather than a test bug.
    const writtenStates = Array.from(new Float64Array(kernel.memory.buffer, ptr, states * 12));
    for (let i = 0; i < count; i++) {
      const state = wasm.getBodyState(writtenHandles[i]!)!;
      expect(writtenStates[i * 12]).toBe(state.position[0]);
      expect(writtenStates[i * 12 + 6]).toBe(state.velocity[0]);
    }
    kernel.bindings.td_free(ptr, count * 12 + count);
    wasm.destroy();
  });
});

describe('adapter error translation with a stubbed kernel', () => {
  function stubKernel(overrides: Partial<WasmKernel['bindings']>): WasmKernel {
    return {
      bindings: { ...kernel.bindings, ...overrides },
      memory: kernel.memory,
      simdEnabled: true,
    };
  }

  it('turns a dead-world sentinel into an Error', async () => {
    const stub = stubKernel({
      td_selftest_simd_parity: () => true,
      td_world_create: () => 3,
      td_body_create: () => -2,
    });
    const wasm = await WasmPhysics.create({ kernel: stub });
    expect(() =>
      wasm.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3() }),
    ).toThrow(/not live/);
  });

  it('turns an unknown negative code into an Error that names the code', async () => {
    const stub = stubKernel({
      td_selftest_simd_parity: () => true,
      td_world_create: () => 3,
      td_body_create: () => -9,
    });
    const wasm = await WasmPhysics.create({ kernel: stub });
    expect(() =>
      wasm.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3() }),
    ).toThrow(/code -9/);
  });

  it('refuses to build a world when the self-test fails', async () => {
    const stub = stubKernel({ td_selftest_simd_parity: () => false });
    await expect(WasmPhysics.create({ kernel: stub })).rejects.toThrow(/parity self-test/);
  });

  it('refuses to build a world when the id space is exhausted', async () => {
    const stub = stubKernel({
      td_selftest_simd_parity: () => true,
      td_world_create: () => 0xffffffff,
    });
    await expect(WasmPhysics.create({ kernel: stub })).rejects.toThrow(/world ids/);
  });

  it('surfaces an allocation failure instead of writing to address zero', async () => {
    const stub = stubKernel({ td_alloc: () => 0 });
    const wasm = await WasmPhysics.create({ kernel: stub });
    wasm.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3() });
    expect(() => wasm.getBodyState(1)).toThrow(/could not allocate/);
  });
});

describe('environments and training switch backends without changing callers', () => {
  it('drives DriveEnv to identical observations and rewards', async () => {
    const wasm = await wasmBackend({ ...REFERENCE_OPTIONS, linearDamping: 0.02 });
    const a = new DriveEnv({ seed: 11, backend: createBuiltinPhysics({ linearDamping: 0.02 }) });
    const b = new DriveEnv({ seed: 11, backend: wasm });
    const rngA = new Rng(11);
    const rngB = new Rng(11);
    let obsA = a.reset(rngA);
    let obsB = b.reset(rngB);
    for (let i = 0; i < 200; i++) {
      const action = [Math.sin(i * 0.7), Math.cos(i * 0.31)];
      const ra = a.step(action);
      const rb = b.step(action);
      expect(rb.reward, `step ${i} reward`).toBe(ra.reward);
      expect(rb.done, `step ${i} done`).toBe(ra.done);
      obsA = a.observe(obsA);
      obsB = b.observe(obsB);
      for (let k = 0; k < obsA.length; k++) expect(obsB[k], `step ${i} obs[${k}]`).toBe(obsA[k]);
      if (ra.done) {
        obsA = a.reset(rngA);
        obsB = b.reset(rngB);
      }
    }
    a.dispose();
    b.dispose();
    wasm.destroy();
  });

  it('drives ReachEnv, whose puck is contact-driven, identically', async () => {
    const wasm = await wasmBackend();
    const a = new ReachEnv({ seed: 7, backend: createBuiltinPhysics() });
    const b = new ReachEnv({ seed: 7, backend: wasm });
    const rng = () => new Rng(7);
    a.reset(rng());
    b.reset(rng());
    for (let i = 0; i < 120; i++) {
      const action = [Math.sin(i), Math.cos(i * 1.3)];
      const ra = a.step(action);
      const rb = b.step(action);
      expect(rb.reward).toBe(ra.reward);
      expect(b.diagnostics().puckDistance).toBe(a.diagnostics().puckDistance);
      expect(b.diagnostics().reached).toBe(a.diagnostics().reached);
    }
    a.dispose();
    b.dispose();
    wasm.destroy();
  });

  it('trains to an identical history on either backend', async () => {
    // The strongest form of "callers do not change": same policy seed, same env
    // config, same trainer, only the backend differs. Policy gradients amplify
    // tiny numeric differences, so an ulp-level divergence in the solver would
    // show up here as a different return within a few episodes.
    const train = async (backend: PhysicsBackend): Promise<number[]> => {
      const env = new DriveEnv({ seed: 3, backend, maxStepsPerEpisode: 30 });
      const policy = new GaussianPolicy({ observationSize: env.observationSize, actionSize: env.actionSize, seed: 5 });
      const trainer = new Trainer({ episodesPerUpdate: 2, seed: 5 });
      const result = trainer.train(policy, env, 6);
      env.dispose();
      return result.history.flatMap((h) => [h.return, h.entropy, h.valueLoss, h.advantageStd]);
    };
    const wasm = await wasmBackend();
    const fromWasm = await train(wasm);
    const fromBuiltin = await train(createBuiltinPhysics());
    // 6 episodes x the 4 scalars recorded per episode.
    expect(fromWasm.length).toBe(24);
    expect(fromWasm).toEqual(fromBuiltin);
    wasm.destroy();
  });
});

describe('performance', () => {
  /**
   * Best-of-N wall time for `steps` steps of an `n`-body scene, measuring only
   * `step()`. Reads are excluded on purpose: they cost one JS-to-wasm transition
   * per body either way, so including them would measure the boundary instead of
   * the solver and understate the difference the milestone is about.
   */
  function benchSteps(
    make: () => PhysicsBackend,
    n: number,
    steps: number,
    trials = 5,
  ): number {
    let best = Infinity;
    for (let t = 0; t < trials; t++) {
      const world = make();
      for (let i = 0; i < n; i++) {
        world.createBody({
          shape: { kind: 'sphere', radius: 0.3 },
          position: vec3((i % 20) * 0.7 - 7, Math.floor(i / 20) * 0.7, ((i * 7) % 13) * 0.3),
          mass: 1 + (i % 5),
        });
      }
      const start = performance.now();
      for (let k = 0; k < steps; k++) world.step(DT);
      best = Math.min(best, performance.now() - start);
      world.dispose();
    }
    return best;
  }

  it('is at least 2x faster than builtin on a broadphase-heavy scene', async () => {
    const wasm = await wasmBackend();
    // Measured on this machine: builtin 139.7ms, wasm 54.1ms -> 2.58x. The
    // 400-body scene is O(n^2) in the broadphase and memory-bound, which is the
    // hardest case for the port; the smaller scene below is the easier one.
    const builtinMs = benchSteps(() => createBuiltinPhysics({ ...REFERENCE_OPTIONS }), 400, 200);
    const wasmMs = benchSteps(() => wasm, 400, 200);
    expect(wasmMs).toBeLessThan(builtinMs);
    expect(builtinMs / wasmMs, `${builtinMs.toFixed(1)}ms vs ${wasmMs.toFixed(1)}ms`).toBeGreaterThanOrEqual(2);
    wasm.destroy();
  });

  it('is at least 2x faster on a small scene too, where call overhead matters most', async () => {
    const wasm = await wasmBackend();
    // Measured: builtin 59.9ms, wasm 12.4ms -> 4.85x.
    const builtinMs = benchSteps(() => createBuiltinPhysics({ ...REFERENCE_OPTIONS }), 60, 600);
    const wasmMs = benchSteps(() => wasm, 60, 600);
    expect(builtinMs / wasmMs).toBeGreaterThanOrEqual(2);
    wasm.destroy();
  });

  it('produces identical results at every scene size it is benched at', async () => {
    // A benchmark that measures a different computation than the one being
    // verified is worthless, so the two bench scenes are also checked for
    // exactness. This is the assertion that keeps the numbers above honest.
    for (const n of [60, 400]) {
      const builtin = createBuiltinPhysics({ ...REFERENCE_OPTIONS });
      const wasm = await wasmBackend();
      for (const world of [builtin, wasm]) {
        for (let i = 0; i < n; i++) {
          world.createBody({
            shape: { kind: 'sphere', radius: 0.3 },
            position: vec3((i % 20) * 0.7 - 7, Math.floor(i / 20) * 0.7, ((i * 7) % 13) * 0.3),
            mass: 1 + (i % 5),
          });
        }
      }
      for (let k = 0; k < 40; k++) {
        builtin.step(DT);
        wasm.step(DT);
      }
      for (let i = 1; i <= n; i++) expectStateEqual(builtin, wasm, i);
      wasm.destroy();
    }
  });
});

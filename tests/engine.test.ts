/**
 * Engine facade specs.
 *
 * The engine's contract is that `step(n)` and `frame(realDt)` drive the *same*
 * simulation. That equivalence is the property the whole project rests on: it is
 * why a policy trained headless in Node behaves identically in the browser demo.
 * These tests pin it, plus the ownership rules (who disposes the physics
 * backend) that a leak or a double-free would otherwise hide until CI runs long.
 */

import { describe, expect, it, vi } from 'vitest';

import { createEngine, Engine } from '../src/core/engine.js';
import { createBuiltinPhysics } from '../src/physics/builtin.js';
import { RigidBodyComponent } from '../src/physics/components.js';
import { vec3, type PhysicsBackend } from '../src/physics/types.js';

function makeEngine(options: { fixedDt?: number } = {}): { engine: Engine; physics: PhysicsBackend } {
  const physics = createBuiltinPhysics({
    gravity: vec3(0, -9.81, 0),
    fixedDt: options.fixedDt ?? 1 / 60,
  });
  return { engine: new Engine({ physics }), physics };
}

describe('createEngine defaults', () => {
  it('builds a deterministic built-in backend at 60 Hz with Earth gravity', () => {
    const engine = createEngine();
    expect(engine.physics.name).toBe('builtin');
    expect(engine.physics.deterministic).toBe(true);
    expect(engine.fixedDt).toBeCloseTo(1 / 60, 10);
    engine.dispose();
  });

  it('honours a custom fixedDt and gravity', () => {
    const engine = createEngine({ fixedDt: 1 / 120, gravity: vec3(0, -1, 0) });
    expect(engine.fixedDt).toBeCloseTo(1 / 120, 10);

    const h = engine.physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0, 0, 0),
      mass: 1,
    });
    engine.step(120);
    // One second under gravity -1: y ~ -0.5.
    const y = engine.physics.getBodyState(h)!.position[1];
    expect(y).toBeCloseTo(-0.5, 1);
    engine.dispose();
  });

  it('accepts an injected backend and attaches the physics system by default', () => {
    const physics = createBuiltinPhysics({ gravity: vec3(0, 0, 0) });
    const engine = new Engine({ physics });
    expect(engine.physics).toBe(physics);
    expect(engine.physicsSystem).toBeDefined();
    expect(engine.world.systemNames).toContain('physics');
    engine.dispose();
  });

  it('can be built without the physics system', () => {
    const physics = createBuiltinPhysics({ gravity: vec3(0, 0, 0) });
    const engine = new Engine({ physics, withPhysicsSystem: false });
    expect(engine.physicsSystem).toBeUndefined();
    expect(engine.world.systemNames).not.toContain('physics');
    engine.dispose({ disposePhysics: false });
  });
});

describe('step() and frame() are the same simulation', () => {
  it('a falling body reaches the same height either way', () => {
    // The core determinism claim. step(60) advances exactly one second of
    // simulated time; frame() must consume a second of wall time and land on the
    // same state, because it dispatches the same fixed steps through the clock.
    const a = makeEngine();
    const b = makeEngine();
    const spec = {
      shape: { kind: 'sphere', radius: 0.1 } as const,
      position: vec3(0, 10, 0),
      mass: 1,
    };
    const ha = a.physics.createBody(spec);
    const hb = b.physics.createBody(spec);

    a.engine.step(60);
    // 1.0s of real time at dt=1/60 is exactly 60 fixed steps, no remainder.
    let taken = 0;
    let guard = 0;
    while (taken < 1 - 1e-9 && guard++ < 1000) {
      taken += b.engine.frame(1 / 60).steps * b.engine.fixedDt;
    }

    const ya = a.physics.getBodyState(ha)!.position[1];
    const yb = b.physics.getBodyState(hb)!.position[1];
    expect(taken).toBeCloseTo(1, 6);
    expect(yb).toBeCloseTo(ya, 9);
    expect(a.engine.clock.steps).toBe(b.engine.clock.steps);

    a.engine.dispose();
    b.engine.dispose();
  });

  it('step(0) advances nothing and frame(0) dispatches no steps', () => {
    const { engine, physics } = makeEngine();
    const h = physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0, 10, 0),
      mass: 1,
    });
    const before = physics.getBodyState(h)!.position[1];

    engine.step(0);
    const r = engine.frame(0);

    expect(r.steps).toBe(0);
    expect(engine.clock.steps).toBe(0);
    expect(physics.getBodyState(h)!.position[1]).toBe(before);
    engine.dispose();
  });

  it('a sub-step frame defers work and reports the interpolation alpha', () => {
    // Renderers call frame() with real elapsed time, which rarely divides evenly
    // into fixedDt. The leftover is reported as alpha so the render layer can
    // interpolate instead of either skipping time or duplicating a step.
    const { engine } = makeEngine({ fixedDt: 1 / 60 });
    const half = engine.fixedDt / 2;

    expect(engine.frame(half).steps).toBe(0);
    const second = engine.frame(half);
    expect(second.steps).toBe(1);
    expect(second.alpha).toBeGreaterThanOrEqual(0);
    expect(second.alpha).toBeLessThan(1);
    expect(engine.clock.steps).toBe(1);
    engine.dispose();
  });

  it('the clock clamps a huge frame so one stall cannot explode into thousands of steps', () => {
    // A backgrounded tab can hand frame() several seconds at once. Without a
    // clamp that becomes a death spiral: catch-up work makes the next frame
    // longer still. The clock caps steps per update.
    const { engine } = makeEngine();
    const r = engine.frame(30); // 30 seconds at once
    expect(r.steps).toBeGreaterThan(0);
    expect(r.steps).toBeLessThan(30 * 60);
    engine.dispose();
  });
});

describe('world helpers and stats', () => {
  it('addGround() creates a static box that a falling body lands on', () => {
    const engine = createEngine({ gravity: vec3(0, -9.81, 0) });
    engine.addGround(20, 0.5, 0);
    const ball = engine.physics.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      position: vec3(0, 2, 0),
      mass: 1,
      restitution: 0,
      friction: 0.9,
    });

    engine.step(600); // 10 seconds, plenty to settle
    const y = engine.physics.getBodyState(ball)!.position[1];

    // Rests on the ground surface (y=0) plus its radius, within a tolerance.
    expect(y).toBeGreaterThan(0.1);
    expect(y).toBeLessThan(0.6);
    engine.dispose();
  });

  it('addStaticBox() returns a handle for a non-moving body', () => {
    const { engine, physics } = makeEngine();
    const h = engine.addStaticBox(vec3(1, 2, 3), vec3(0.5, 0.5, 0.5));

    engine.step(120);
    const after = physics.getBodyState(h)!;

    // A static body is exempt from gravity: it must not drift or gain velocity.
    expect(after.position).toEqual([1, 2, 3]);
    expect(after.velocity[1]).toBeCloseTo(0, 9);
    engine.dispose();
  });

  it('labels a body and surfaces the label on the contacts it takes part in', () => {
    // `label` lives on the descriptor, not on BodyState, and reaches gameplay
    // through ContactEvent -- that is how a reward function can tell "the puck"
    // from "a wall" without holding handles.
    const engine = createEngine({ gravity: vec3(0, -9.81, 0) });
    engine.addStaticBox(vec3(0, -0.25, 0), vec3(10, 0.25, 10), 'floor');
    engine.physics.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      position: vec3(0, 1, 0),
      mass: 1,
      restitution: 0,
      label: 'ball',
    });

    const labels = new Set<string>();
    engine.bus.on('physics:contact', ({ contact }) => {
      labels.add(contact.labels[0]);
      labels.add(contact.labels[1]);
    });
    engine.step(180);

    expect(labels.has('ball')).toBe(true);
    expect(labels.has('floor')).toBe(true);
    engine.dispose();
  });

  it('stats() reports fixedDt, step count, entities, bodies and system names', () => {
    const { engine, physics } = makeEngine();
    const e = engine.world.createEntity();
    const h = physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(),
      mass: 1,
    });
    engine.physicsSystem!.attach({ world: engine.world }, e, h, 'ball');

    engine.step(5);
    const s = engine.stats();

    expect(s.fixedDt).toBeCloseTo(1 / 60, 10);
    expect(s.steps).toBe(5);
    expect(s.simulationTime).toBeCloseTo(5 / 60, 9);
    expect(s.entities).toBe(1);
    expect(s.bodies).toBe(1);
    expect(s.systems).toContain('physics');
    expect(engine.world.query(RigidBodyComponent)).toEqual([e]);
    engine.dispose();
  });
});

describe('event bus wiring', () => {
  it('emits engine lifecycle events', () => {
    const { engine } = makeEngine();
    const seen: string[] = [];
    engine.bus.on('engine:paused', () => seen.push('paused'));
    engine.bus.on('engine:resumed', () => seen.push('resumed'));

    // start() is a no-op without requestAnimationFrame (Node), so pause/resume
    // are driven directly to keep this headless.
    engine.pause();
    expect(seen).toEqual([]); // never running, so nothing to pause
    engine.dispose();
  });

  it('surfaces physics contacts on the engine bus', () => {
    const engine = createEngine({ gravity: vec3(0, -9.81, 0) });
    const contacts: unknown[] = [];
    engine.bus.on('physics:contact', (p) => contacts.push(p));

    engine.addGround(20, 0.5, 0);
    engine.physics.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      position: vec3(0, 1, 0),
      mass: 1,
      restitution: 0,
    });

    engine.step(180);
    expect(contacts.length).toBeGreaterThan(0);
    engine.dispose();
  });
});

describe('dispose() ownership', () => {
  it('releases the physics backend by default', () => {
    const { engine, physics } = makeEngine();
    physics.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), mass: 1 });
    expect(physics.bodyCount).toBe(1);

    engine.dispose();

    // A disposed built-in backend forgets its bodies.
    expect(physics.bodyCount).toBe(0);
  });

  it('leaves an injected backend alive when asked to', () => {
    // Callers that own the backend (a test harness, or several engines sharing
    // one world) must be able to keep it after the facade goes away.
    const physics = createBuiltinPhysics({ gravity: vec3(0, 0, 0) });
    const engine = new Engine({ physics });
    physics.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), mass: 1 });

    engine.dispose({ disposePhysics: false });

    expect(physics.bodyCount).toBe(1);
    expect(engine.world.entityCount).toBe(0);
    physics.dispose();
  });

  it('is idempotent and stops stepping afterwards', () => {
    const { engine } = makeEngine();
    engine.step(3);
    engine.dispose();
    expect(() => engine.dispose()).not.toThrow();
    expect(engine.isRunning).toBe(false);
  });
});

/**
 * The rAF loop is the browser half of the facade: `step()` is headless and
 * synchronous, `start()`/`pause()`/`resume()` are driven by the display.
 * Nothing about them needs a browser, only the two globals they touch, so they
 * are specified here with a fake frame clock instead of being left to the e2e
 * demo -- a bug in the loop is far easier to localise from a 2 ms unit test
 * than from a blank viewport in Chromium.
 */
describe('the requestAnimationFrame loop', () => {
  /**
   * One frame of wall clock, just over a 60 Hz tick. Slightly over rather than
   * exactly `1000/60` because the accumulator compares against `fixedDt` in
   * seconds, and floating-point rounding of three exact frames lands one tick
   * short of three steps -- which would be a test bug, not an engine bug.
   */
  const FRAME_MS = 17;

  /** A manually pumped rAF: records callbacks, runs them with a chosen timestamp. */
  function fakeRaf() {
    const queue: Array<(now: number) => void> = [];
    let id = 0;
    let now = 0;
    const raf = (cb: (t: number) => void): number => {
      queue.push(cb);
      return ++id;
    };
    const cancel = (handle: number): void => {
      // Simplest faithful model: the id is a position, so drop that entry.
      if (handle > 0 && handle <= queue.length) queue[handle - 1] = () => {};
    };
    return {
      raf,
      cancel,
      /** Advance the fake clock by `ms` and run everything queued. */
      pump(ms: number): void {
        now += ms;
        const batch = queue.splice(0, queue.length);
        for (const cb of batch) cb(now);
      },
      pending(): number {
        return queue.length;
      },
      now: (): number => now,
    };
  }

  function withRaf(fn: (h: ReturnType<typeof fakeRaf>) => void): void {
    const h = fakeRaf();
    vi.stubGlobal('requestAnimationFrame', h.raf);
    vi.stubGlobal('cancelAnimationFrame', h.cancel);
    vi.stubGlobal('performance', { now: h.now });
    try {
      fn(h);
    } finally {
      vi.unstubAllGlobals();
    }
  }

  it('is a no-op without requestAnimationFrame, so Node hosts never schedule', () => {
    const { engine } = makeEngine();
    expect(typeof requestAnimationFrame).toBe('undefined');
    engine.start();
    expect(engine.isRunning, 'must not claim to run without a frame source').toBe(false);
    engine.dispose();
  });

  it('start() emits engine:started and steps the world once per frame', () => {
    withRaf((h) => {
      const { engine } = makeEngine({ fixedDt: 1 / 60 });
      const events: unknown[] = [];
      engine.bus.on('engine:started', (p) => events.push(p));

      engine.start();
      expect(engine.isRunning).toBe(true);
      expect(events).toEqual([{ fixedDt: 1 / 60 }]);

      // Three frames ~= three fixed steps at 60 Hz.
      h.pump(FRAME_MS);
      h.pump(FRAME_MS);
      h.pump(FRAME_MS);
      expect(engine.clock.steps).toBe(3);
      engine.dispose();
    });
  });

  it('a single huge frame is clamped, so a backgrounded tab cannot explode', () => {
    withRaf((h) => {
      const { engine } = makeEngine({ fixedDt: 1 / 60 });
      engine.start();
      h.pump(10_000); // ten seconds of wall clock in one frame
      // maxStepsPerFrame is 8 and maxFrameDt 0.25s -> at most 8 steps.
      expect(engine.clock.steps).toBeLessThanOrEqual(8);
      expect(engine.clock.steps).toBeGreaterThan(0);
      engine.dispose();
    });
  });

  it('start() twice does not double-schedule', () => {
    withRaf((h) => {
      const { engine } = makeEngine();
      engine.start();
      const pending = h.pending();
      engine.start();
      expect(h.pending(), 'the second start() must be ignored').toBe(pending);
      h.pump(FRAME_MS);
      expect(engine.clock.steps).toBeLessThanOrEqual(1);
      engine.dispose();
    });
  });

  it('pause() stops stepping and cancels the scheduled frame', () => {
    withRaf((h) => {
      const { engine } = makeEngine();
      const seen: string[] = [];
      engine.bus.on('engine:paused', () => seen.push('paused'));

      engine.start();
      h.pump(FRAME_MS);
      const stepsAtPause = engine.clock.steps;
      engine.pause();

      expect(engine.isRunning).toBe(false);
      expect(seen).toEqual(['paused']);
      h.pump(FRAME_MS);
      h.pump(FRAME_MS);
      expect(engine.clock.steps, 'a paused engine must not advance').toBe(stepsAtPause);
      engine.dispose();
    });
  });

  it('pause() when not running emits nothing', () => {
    withRaf(() => {
      const { engine } = makeEngine();
      const seen: string[] = [];
      engine.bus.on('engine:paused', () => seen.push('paused'));
      engine.pause();
      expect(seen).toEqual([]);
      engine.dispose();
    });
  });

  it('resume() restarts and re-anchors time so no backlog is simulated', () => {
    withRaf((h) => {
      const { engine } = makeEngine();
      const seen: string[] = [];
      engine.bus.on('engine:resumed', () => seen.push('resumed'));

      engine.start();
      h.pump(FRAME_MS);
      engine.pause();
      // Time passes while paused; resume() must not treat it as frame time.
      h.pump(5_000);
      engine.resume();

      expect(engine.isRunning).toBe(true);
      expect(seen).toEqual(['resumed']);
      const before = engine.clock.steps;
      h.pump(FRAME_MS);
      expect(engine.clock.steps - before, 'only the real frame is simulated').toBe(1);
      engine.dispose();
    });
  });

  it('resume() while already running is a no-op', () => {
    withRaf((h) => {
      const { engine } = makeEngine();
      engine.start();
      const pending = h.pending();
      engine.resume();
      expect(h.pending()).toBe(pending);
      engine.dispose();
    });
  });

  it('dispose() stops the loop for good', () => {
    withRaf((h) => {
      const { engine } = makeEngine();
      engine.start();
      h.pump(FRAME_MS);
      const steps = engine.clock.steps;
      engine.dispose();
      h.pump(FRAME_MS);
      expect(engine.isRunning).toBe(false);
      expect(engine.clock.steps).toBe(steps);
    });
  });
});

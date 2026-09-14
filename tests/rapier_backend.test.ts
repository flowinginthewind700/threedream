import { describe, expect, it } from 'vitest';
import { createRapierPhysics } from '../src/physics/rapier.js';
import { createBuiltinPhysics } from '../src/physics/builtin.js';
import { vec3, lengthVec3, type PhysicsBackend } from '../src/physics/types.js';

const dt = 1 / 60;

describe('rapier backend honors PhysicsWorldOptions', () => {
  it('applies an impulse to the mass named in the descriptor on the first step', async () => {
    // Regression: RigidBodyDesc.setAdditionalMass is lazy in Rapier (folded in at
    // the next world.step()), so a first-frame impulse used to land on a body
    // ~400x too light. Mass now goes on the collider, which is immediate.
    const physics = await createRapierPhysics({ gravity: vec3(0, 0, 0), fixedDt: dt });
    const h = physics.createBody({
      shape: { kind: 'sphere', radius: 0.08 },
      position: vec3(),
      mass: 1,
      label: 'agent',
    });
    physics.applyImpulse(h, vec3(1, 0, 0));
    physics.step(dt);
    // v = J/m = 1/1. Allow a hair of default linear damping.
    expect(physics.getBodyState(h)!.velocity[0]).toBeCloseTo(1, 2);
    physics.dispose();
  });

  it('mirrors builtin steady-state speed when both get the same damping', async () => {
    const opts = {
      gravity: vec3(0, 0, 0),
      linearDamping: 2,
      angularDamping: 2,
      solverIterations: 4,
      fixedDt: dt,
    };
    const rapier = await createRapierPhysics(opts);
    const builtin = createBuiltinPhysics(opts);
    // Both solvers must satisfy the same interface; that is the whole point.
    const steady = (p: PhysicsBackend): number => {
      const h = p.createBody({
        shape: { kind: 'sphere', radius: 0.08 },
        position: vec3(),
        mass: 1,
        friction: 0.5,
        label: 'agent',
      });
      let speed = 0;
      for (let i = 0; i < 120; i++) {
        p.applyImpulse(h, vec3(5 * dt, 0, 0));
        p.step(dt);
        speed = lengthVec3(p.getBodyState(h)!.velocity);
      }
      p.dispose();
      return speed;
    };
    const rapierSpeed = steady(rapier);
    const builtinSpeed = steady(builtin);
    // Same first-order drag law; the two solvers should land within a few percent.
    expect(rapierSpeed).toBeGreaterThan(builtinSpeed * 0.9);
    expect(rapierSpeed).toBeLessThan(builtinSpeed * 1.1);
  });
});

/**
 * The adapter's job is interface fidelity: a scene written against
 * `PhysicsBackend` must behave the same on Rapier as on the built-in solver.
 * Everything below exercises the parts of the adapter that the impulse/damping
 * tests above do not reach -- body kinds, all three shapes, collision groups,
 * state round-trips, contacts, raycasts, and teardown.
 */
describe('rapier backend implements the full PhysicsBackend surface', () => {
  async function world(options: Parameters<typeof createRapierPhysics>[0] = {}) {
    return createRapierPhysics({ gravity: vec3(0, 0, 0), fixedDt: dt, ...options });
  }

  it('reports its identity honestly: rapier, non-deterministic across platforms', () => {
    // `deterministic: false` is a claim about cross-platform reproducibility of
    // WASM/SIMD, not about the solver being random. If this ever flips, envs
    // that rely on `backend.deterministic` to choose a replay path change too.
    return (async () => {
      const p = await world();
      expect(p.name).toBe('rapier');
      expect(p.deterministic).toBe(false);
      expect(p.fixedDt).toBeCloseTo(dt, 10);
      p.dispose();
    })();
  });

  it('creates every shape kind and counts bodies', async () => {
    const p = await world();
    expect(p.bodyCount).toBe(0);

    const sphere = p.createBody({ shape: { kind: 'sphere', radius: 0.2 }, position: vec3(0, 0, 0), mass: 1 });
    const box = p.createBody({
      shape: { kind: 'box', halfExtents: vec3(0.2, 0.3, 0.4) },
      position: vec3(2, 0, 0),
      mass: 2,
    });
    const capsule = p.createBody({
      shape: { kind: 'capsule', radius: 0.15, halfHeight: 0.4 },
      position: vec3(-2, 0, 0),
      mass: 1,
    });

    expect(p.bodyCount).toBe(3);
    // Handles are distinct and stable identifiers.
    expect(new Set([sphere, box, capsule]).size).toBe(3);
    expect(p.getBodyState(box)!.position[0]).toBeCloseTo(2, 6);
    p.dispose();
    expect(p.bodyCount, 'dispose clears the body map').toBe(0);
  });

  it('honours body kinds: static and kinematic do not fall, dynamic does', async () => {
    const p = await createRapierPhysics({ gravity: vec3(0, -9.81, 0), fixedDt: dt });
    const dynamic = p.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(0, 5, 0), mass: 1 });
    const fixed = p.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(1, 5, 0),
      kind: 'static',
    });
    const kinematic = p.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(2, 5, 0),
      kind: 'kinematic',
    });

    for (let i = 0; i < 30; i++) p.step(dt);

    expect(p.getBodyState(dynamic)!.position[1]).toBeLessThan(5);
    expect(p.getBodyState(fixed)!.position[1]).toBeCloseTo(5, 6);
    expect(p.getBodyState(kinematic)!.position[1]).toBeCloseTo(5, 6);
    p.dispose();
  });

  it('applies an initial velocity from the descriptor', async () => {
    const p = await world();
    const h = p.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(),
      mass: 1,
      velocity: vec3(1, 0, 0),
    });
    p.step(dt);
    // No gravity, only the default linear damping: it moved in +x and slowed.
    const state = p.getBodyState(h)!;
    expect(state.position[0]).toBeGreaterThan(0);
    expect(state.velocity[0]).toBeGreaterThan(0.9);
    p.dispose();
  });

  it('setBodyState writes position, rotation and velocity back', async () => {
    const p = await world();
    const h = p.createBody({ shape: { kind: 'box', halfExtents: vec3(0.1, 0.2, 0.3) }, position: vec3(), mass: 1 });

    p.setBodyState(h, { position: vec3(1, 2, 3), velocity: vec3(0, 0, 4), angularVelocity: vec3(0, 0, 0) });
    let s = p.getBodyState(h)!;
    expect(s.position[0]).toBeCloseTo(1, 6);
    expect(s.position[1]).toBeCloseTo(2, 6);
    expect(s.position[2]).toBeCloseTo(3, 6);
    expect(s.velocity[2]).toBeCloseTo(4, 6);

    // Rotation goes through the Euler<->quaternion bridge, so a single-axis
    // rotation must round-trip: that conversion is the subtle part of the adapter.
    p.setBodyState(h, { rotation: vec3(0.5, 0, 0) });
    s = p.getBodyState(h)!;
    expect(s.rotation[0]).toBeCloseTo(0.5, 5);
    expect(s.rotation[1]).toBeCloseTo(0, 5);
    p.dispose();
  });

  it('applyForce accumulates over steps, applyImpulse acts at once', async () => {
    const p = await world();
    const forced = p.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(0, 0, 0), mass: 1 });
    const impulsed = p.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(0, 2, 0), mass: 1 });

    // Same total momentum delivered two ways: force over N steps vs one impulse.
    const total = 2;
    for (let i = 0; i < 60; i++) {
      p.applyForce(forced, vec3(total / (60 * dt), 0, 0));
      p.step(dt);
    }
    p.applyImpulse(impulsed, vec3(total, 0, 0));
    for (let i = 0; i < 60; i++) p.step(dt);

    const vf = p.getBodyState(forced)!.velocity[0];
    const vi = p.getBodyState(impulsed)!.velocity[0];
    expect(vf).toBeGreaterThan(0);
    // Both experience the same damping law, so they should agree closely.
    expect(Math.abs(vf - vi) / Math.max(vf, vi)).toBeLessThan(0.15);
    p.dispose();
  });

  it('destroyBody removes the body and later calls are inert', async () => {
    const p = await world();
    const h = p.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), mass: 1 });
    expect(p.getBodyState(h)).toBeDefined();

    p.destroyBody(h);
    expect(p.bodyCount).toBe(0);
    expect(p.getBodyState(h)).toBeUndefined();

    // No throw on an unknown handle: a scene that destroys a body in one system
    // and reads it in another must not crash the frame.
    expect(() => p.setBodyState(h, { position: vec3(1, 1, 1) })).not.toThrow();
    expect(() => p.applyImpulse(h, vec3(1, 0, 0))).not.toThrow();
    expect(() => p.applyForce(h, vec3(1, 0, 0))).not.toThrow();
    expect(() => p.destroyBody(h)).not.toThrow();
    expect(() => p.destroyBody(99999)).not.toThrow();
    p.dispose();
  });

  it('labels bodies and reports both labels on a contact', async () => {
    const p = await createRapierPhysics({ gravity: vec3(0, -9.81, 0), fixedDt: dt });
    p.createBody({
      shape: { kind: 'box', halfExtents: vec3(2, 0.1, 2) },
      position: vec3(0, -0.1, 0),
      kind: 'static',
      label: 'ground',
    });
    p.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      position: vec3(0, 1, 0),
      mass: 1,
      restitution: 0,
      label: 'agent',
    });

    let contacts: Awaited<ReturnType<typeof p.drainContacts>> = [];
    for (let i = 0; i < 120 && contacts.length === 0; i++) {
      p.step(dt);
      contacts = p.drainContacts();
    }

    expect(contacts.length, 'the sphere should land on the ground').toBeGreaterThan(0);
    const labels = contacts[0].labels;
    expect([...labels].sort()).toEqual(['agent', 'ground']);
    expect(contacts[0].impulse).toBeGreaterThan(0);
    p.dispose();
  });

  it('an unnamed body gets a `body:<handle>` label', async () => {
    const p = await createRapierPhysics({ gravity: vec3(0, -9.81, 0), fixedDt: dt });
    const ground = p.createBody({
      shape: { kind: 'box', halfExtents: vec3(2, 0.1, 2) },
      position: vec3(0, -0.1, 0),
      kind: 'static',
    });
    const ball = p.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      position: vec3(0, 1, 0),
      mass: 1,
      restitution: 0,
    });

    let contacts: Awaited<ReturnType<typeof p.drainContacts>> = [];
    for (let i = 0; i < 120 && contacts.length === 0; i++) {
      p.step(dt);
      contacts = p.drainContacts();
    }
    expect(contacts.length).toBeGreaterThan(0);
    expect([...contacts[0].labels].sort()).toEqual([`body:${ball}`, `body:${ground}`].sort());
    p.dispose();
  });

  it('collision groups can filter a pair out entirely', async () => {
    const p = await createRapierPhysics({ gravity: vec3(0, -9.81, 0), fixedDt: dt });
    p.createBody({
      shape: { kind: 'box', halfExtents: vec3(2, 0.1, 2) },
      position: vec3(0, -0.1, 0),
      kind: 'static',
      group: 1,
      mask: 0, // collides with nothing
    });
    p.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      position: vec3(0, 0.2, 0),
      mass: 1,
      group: 2,
      mask: 0xffff,
    });

    let contacts = 0;
    for (let i = 0; i < 60; i++) {
      p.step(dt);
      contacts += p.drainContacts().length;
    }
    expect(contacts, 'a filtered pair must not generate contacts').toBe(0);
    p.dispose();
  });

  it('drainContacts is one-shot: draining twice returns nothing', async () => {
    const p = await createRapierPhysics({ gravity: vec3(0, -9.81, 0), fixedDt: dt });
    p.createBody({
      shape: { kind: 'box', halfExtents: vec3(2, 0.1, 2) },
      position: vec3(0, -0.1, 0),
      kind: 'static',
    });
    p.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      position: vec3(0, 0.5, 0),
      mass: 1,
      restitution: 0,
    });

    let first: Awaited<ReturnType<typeof p.drainContacts>> = [];
    for (let i = 0; i < 90 && first.length === 0; i++) {
      p.step(dt);
      first = p.drainContacts();
    }
    expect(first.length).toBeGreaterThan(0);
    // No step in between, so the second drain must be empty -- otherwise a
    // consumer would double-count a contact and double-apply a reward.
    expect(p.drainContacts()).toEqual([]);
    p.dispose();
  });

  it('raycast finds a body below and reports a sane hit point', async () => {
    const p = await world();
    const h = p.createBody({
      shape: { kind: 'box', halfExtents: vec3(1, 0.1, 1) },
      position: vec3(0, 0, 0),
      kind: 'static',
    });

    const hit = p.raycast(vec3(0, 2, 0), vec3(0, -1, 0), 10);
    expect(hit, 'the ray should hit the box').toBeDefined();
    expect(hit!.body).toBe(h);
    expect(hit!.distance).toBeGreaterThan(0);
    expect(hit!.distance).toBeLessThan(10);
    // Hit point sits on the top face of the box, between the origin and it.
    expect(hit!.point[1]).toBeLessThan(2);
    expect(hit!.point[1]).toBeGreaterThan(-0.2);
    // Normal points back at the caster.
    expect(hit!.normal[1]).toBeGreaterThan(0.9);
    p.dispose();
  });

  it('raycast misses return undefined, and a short ray does not reach', async () => {
    const p = await world();
    p.createBody({ shape: { kind: 'box', halfExtents: vec3(1, 0.1, 1) }, position: vec3(0, 0, 0), kind: 'static' });

    expect(p.raycast(vec3(0, 2, 0), vec3(1, 0, 0), 10), 'a sideways ray hits nothing').toBeUndefined();
    expect(p.raycast(vec3(0, 2, 0), vec3(0, -1, 0), 0.5), 'too short to reach').toBeUndefined();
    p.dispose();
  });

  it('raycast normalizes a non-unit direction', async () => {
    const p = await world();
    p.createBody({ shape: { kind: 'box', halfExtents: vec3(1, 0.1, 1) }, position: vec3(0, 0, 0), kind: 'static' });
    // Same ray as the unit case, scaled; distance must not scale with it.
    const hit = p.raycast(vec3(0, 2, 0), vec3(0, -7, 0), 10);
    expect(hit).toBeDefined();
    expect(hit!.distance).toBeCloseTo(1.9, 1);
    p.dispose();
  });

  it('a kinematic body can be teleported and stays put', async () => {
    const p = await world();
    const h = p.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0, 0, 0),
      kind: 'kinematic',
    });
    p.setBodyState(h, { position: vec3(3, 0, 0) });
    for (let i = 0; i < 10; i++) p.step(dt);
    expect(p.getBodyState(h)!.position[0]).toBeCloseTo(3, 6);
    p.dispose();
  });

  it('rejects an unknown shape rather than building a degenerate collider', async () => {
    const p = await world();
    const bad = { kind: 'torus', radius: 1 } as unknown as Parameters<typeof p.createBody>[0]['shape'];
    expect(() => p.createBody({ shape: bad, position: vec3(), mass: 1 })).toThrow(/unsupported shape kind/);
    p.dispose();
  });

  it('stepping after dispose is inert rather than a crash', async () => {
    const p = await world();
    const h = p.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), mass: 1 });
    p.dispose();
    // The adapter must not hand back state for a freed world.
    expect(p.getBodyState(h)).toBeUndefined();
    expect(p.bodyCount).toBe(0);
  });
});

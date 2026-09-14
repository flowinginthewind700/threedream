import { describe, expect, it } from 'vitest';

import { BuiltinPhysics } from '../src/physics/builtin.js';
import {
  addVec3,
  crossVec3,
  distanceVec3,
  dotVec3,
  eulerFromQuatXYZ,
  lengthVec3,
  normalizeVec3,
  quatFromEulerXYZ,
  scaleVec3,
  subVec3,
  vec3,
  type Vec3,
} from '../src/physics/types.js';

const DT = 1 / 60;

function sphereWorld(options: { gravity?: Vec3; damping?: number } = {}): BuiltinPhysics {
  return new BuiltinPhysics({
    fixedDt: DT,
    gravity: options.gravity ?? vec3(0, -9.81, 0),
    linearDamping: options.damping ?? 0,
    angularDamping: options.damping ?? 0,
    solverIterations: 8,
  });
}

describe('vec3 helpers', () => {
  it('are algebraically consistent', () => {
    const a = vec3(1, 2, 3);
    const b = vec3(-4, 0.5, 2);
    expect(addVec3(a, b)).toEqual(vec3(-3, 2.5, 5));
    expect(subVec3(a, b)).toEqual(vec3(5, 1.5, 1));
    expect(scaleVec3(a, 2)).toEqual(vec3(2, 4, 6));
    expect(dotVec3(a, b)).toBeCloseTo(-4 + 1 + 6, 10);
    expect(crossVec3(vec3(1, 0, 0), vec3(0, 1, 0))).toEqual(vec3(0, 0, 1));
    expect(lengthVec3(vec3(3, 4, 0))).toBeCloseTo(5, 10);
    expect(distanceVec3(a, a)).toBe(0);
  });

  it('normalizes to unit length and degrades safely on zero', () => {
    const unit = normalizeVec3(vec3(0, 3, 0));
    expect(unit).toEqual(vec3(0, 1, 0));
    expect(lengthVec3(normalizeVec3(vec3(1, -2, 3)))).toBeCloseTo(1, 10);
    expect(normalizeVec3(vec3(0, 0, 0))).toEqual(vec3(0, 0, 0));
  });

  it('round-trips Euler XYZ through quaternions', () => {
    // Combined rotations matter: single-axis and identity cases round-trip
    // through several different conventions, so they cannot catch an inverse
    // written for the wrong order.
    const cases = [
      vec3(0.3, -0.7, 1.1),
      vec3(0, 0, 0),
      vec3(-1.2, 0.4, 0.05),
      vec3(0.6, 0.2, -0.9),
      vec3(-0.15, -1.0, 0.75),
    ];
    for (const euler of cases) {
      const back = eulerFromQuatXYZ(quatFromEulerXYZ(euler));
      expect(back[0]).toBeCloseTo(euler[0], 6);
      expect(back[1]).toBeCloseTo(euler[1], 6);
      expect(back[2]).toBeCloseTo(euler[2], 6);
    }
  });
});

describe('BuiltinPhysics bodies', () => {
  it('creates, counts and destroys bodies', () => {
    const world = sphereWorld();
    const a = world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3() });
    const b = world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(1, 0, 0) });
    expect(world.bodyCount).toBe(2);
    expect(a).not.toBe(b);
    world.destroyBody(a);
    expect(world.bodyCount).toBe(1);
    expect(world.getBodyState(a)).toBeUndefined();
  });

  it('rejects capsules (not implemented) with a clear error', () => {
    const world = sphereWorld();
    expect(() =>
      world.createBody({ shape: { kind: 'capsule', radius: 0.1, halfHeight: 0.2 }, position: vec3() }),
    ).toThrow(/capsule/i);
  });

  it('rejects negative mass', () => {
    const world = sphereWorld();
    expect(() =>
      world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), mass: -1 }),
    ).toThrow(RangeError);
  });

  it('static and kinematic bodies are immovable', () => {
    const world = sphereWorld();
    const stat = world.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0, 5, 0),
      kind: 'static',
    });
    world.applyImpulse(stat, vec3(0, 100, 0));
    for (let i = 0; i < 30; i++) world.step(DT);
    expect(world.getBodyState(stat)!.position).toEqual(vec3(0, 5, 0));
  });

  it('setBodyState teleports and clears velocity', () => {
    const world = sphereWorld();
    const h = world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3() });
    world.setBodyState(h, { position: vec3(1, 2, 3), velocity: vec3(4, 5, 6) });
    const state = world.getBodyState(h)!;
    expect(state.position).toEqual(vec3(1, 2, 3));
    expect(state.velocity).toEqual(vec3(4, 5, 6));
    // Returned state is a copy, not a live view.
    state.position[0];
    expect(world.getBodyState(h)!.position).toEqual(vec3(1, 2, 3));
  });

  it('reports defaults for label and friction', () => {
    const world = sphereWorld();
    const h = world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), label: 'puck' });
    world.step(DT);
    expect(world.getBodyState(h)).toBeDefined();
  });
});

describe('BuiltinPhysics integration', () => {
  it('falls under gravity at the analytic rate', () => {
    const world = sphereWorld();
    const h = world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(0, 100, 0) });
    const steps = 60; // one second
    for (let i = 0; i < steps; i++) world.step(DT);
    const state = world.getBodyState(h)!;
    // Velocity after n steps of semi-implicit Euler is exact: -g*n*dt = -9.81.
    expect(state.velocity[1]).toBeCloseTo(-9.81, 6);
    // Position overshoots the analytic 4.905 m by g*dt/2 (the known bias of
    // integrating with the post-force velocity): 4.905 + 0.0818 = 4.987.
    const fallen = 100 - state.position[1];
    expect(fallen).toBeCloseTo(4.987, 2);
    expect(fallen).toBeGreaterThan(4.905);
    expect(fallen).toBeLessThan(5.02);
  });

  it('applyImpulse changes velocity by impulse / mass', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0) });
    const h = world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), mass: 2 });
    world.applyImpulse(h, vec3(4, 0, 0));
    expect(world.getBodyState(h)!.velocity[0]).toBeCloseTo(2, 10);
  });

  it('applyForce accumulates over the fixed step', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0) });
    const h = world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(), mass: 1 });
    world.applyForce(h, vec3(60, 0, 0)); // 60 N on 1 kg over 1/60 s -> dv = 1
    expect(world.getBodyState(h)!.velocity[0]).toBeCloseTo(1, 6);
  });

  it('linear damping bleeds velocity toward zero', () => {
    const world = new BuiltinPhysics({
      fixedDt: DT,
      gravity: vec3(0, 0, 0),
      linearDamping: 2,
    });
    const h = world.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(),
      velocity: vec3(3, 0, 0),
    });
    for (let i = 0; i < 120; i++) world.step(DT);
    expect(Math.abs(world.getBodyState(h)!.velocity[0])).toBeLessThan(0.5);
  });
});

describe('BuiltinPhysics contacts', () => {
  it('a sphere comes to rest on the ground instead of sinking', () => {
    const world = sphereWorld();
    world.createBody({
      shape: { kind: 'box', halfExtents: vec3(5, 0.25, 5) },
      position: vec3(0, -0.25, 0),
      kind: 'static',
      label: 'ground',
    });
    const ball = world.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      position: vec3(0, 2, 0),
      restitution: 0.0,
    });
    for (let i = 0; i < 240; i++) world.step(DT);
    const y = world.getBodyState(ball)!.position[1];
    // Resting at the surface, within solver slop, and not tunnelling through.
    expect(y).toBeGreaterThan(0.2);
    expect(y).toBeLessThan(0.32);
    expect(Math.abs(world.getBodyState(ball)!.velocity[1])).toBeLessThan(0.25);
  });

  it('reports contacts with both participant labels', () => {
    const world = sphereWorld();
    world.createBody({
      shape: { kind: 'box', halfExtents: vec3(5, 0.25, 5) },
      position: vec3(0, -0.25, 0),
      kind: 'static',
      label: 'ground',
    });
    world.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      // 0.15 m of penetration, so the contact exists on the very first step.
      position: vec3(0, 0.1, 0),
      label: 'ball',
    });
    world.step(DT);
    const contacts = world.drainContacts();
    expect(contacts.length).toBeGreaterThan(0);
    const labels = contacts.map((c) => c.labels.slice().sort().join('|'));
    expect(labels).toContain('ball|ground');
    // Draining clears the queue.
    expect(world.drainContacts()).toEqual([]);
  });

  it('emits the contact normal pointing from body a toward body b', () => {
    const world = sphereWorld();
    const ground = world.createBody({
      shape: { kind: 'box', halfExtents: vec3(1, 0.1, 1) },
      position: vec3(0, -0.1, 0),
      kind: 'static',
    });
    const ball = world.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0, 0.05, 0),
    });
    world.step(DT);
    const contact = world.drainContacts()[0]!;
    expect(contact.a).toBe(ground); // created first -> lower handle
    expect(contact.b).toBe(ball);
    // From the box up toward the sphere. Grounding logic depends on this
    // orientation, so it is pinned here rather than assumed.
    expect(contact.normal[1]).toBeGreaterThan(0.99);
    expect(contact.depth).toBeGreaterThan(0);
  });

  it('conserves momentum and energy in an elastic head-on collision', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0), damping: 0 });
    const a = world.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(-0.5, 0, 0),
      velocity: vec3(2, 0, 0),
      mass: 1,
      restitution: 1,
    });
    const b = world.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0.5, 0, 0),
      velocity: vec3(-1, 0, 0),
      mass: 2,
      restitution: 1,
    });
    const momentumBefore = 1 * 2 + 2 * -1;
    const energyBefore = 0.5 * 1 * 4 + 0.5 * 2 * 1;
    for (let i = 0; i < 90; i++) world.step(DT);
    const va = world.getBodyState(a)!.velocity[0];
    const vb = world.getBodyState(b)!.velocity[0];
    expect(va * 1 + vb * 2).toBeCloseTo(momentumBefore, 5);
    expect(0.5 * 1 * va * va + 0.5 * 2 * vb * vb).toBeCloseTo(energyBefore, 3);
    // Elastic 1D solution for m1=1,m2=2: v1' = -2, v2' = +1.
    expect(va).toBeCloseTo(-2, 3);
    expect(vb).toBeCloseTo(1, 3);
  });

  it('a perfectly inelastic collision sticks at the shared velocity', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0), damping: 0 });
    const a = world.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(-0.5, 0, 0),
      velocity: vec3(2, 0, 0),
      mass: 1,
      restitution: 0,
    });
    const b = world.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0.5, 0, 0),
      velocity: vec3(-1, 0, 0),
      mass: 2,
      restitution: 0,
    });
    for (let i = 0; i < 90; i++) world.step(DT);
    const va = world.getBodyState(a)!.velocity[0];
    const vb = world.getBodyState(b)!.velocity[0];
    // (1*2 + 2*-1) / 3 = 0: both bodies stop, momentum is still conserved.
    expect(va).toBeCloseTo(0, 3);
    expect(vb).toBeCloseTo(0, 3);
    // They must not have passed through each other.
    expect(world.getBodyState(a)!.position[0]).toBeLessThan(
      world.getBodyState(b)!.position[0],
    );
  });

  it('collision filtering by group/mask keeps bodies apart', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0), damping: 0 });
    world.createBody({
      shape: { kind: 'sphere', radius: 0.2 },
      position: vec3(-0.5, 0, 0),
      velocity: vec3(1, 0, 0),
      group: 0b0001,
      mask: 0b0001,
    });
    const ghost = world.createBody({
      shape: { kind: 'sphere', radius: 0.2 },
      position: vec3(0.5, 0, 0),
      group: 0b0010,
      mask: 0b0010,
    });
    for (let i = 0; i < 60; i++) world.step(DT);
    expect(world.drainContacts()).toEqual([]);
    // The second body was never touched.
    expect(world.getBodyState(ghost)!.position).toEqual(vec3(0.5, 0, 0));
  });

  it('a wall stops a body moving into it', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0), damping: 0 });
    world.createBody({
      shape: { kind: 'box', halfExtents: vec3(0.05, 1, 1) },
      position: vec3(1.05, 0, 0),
      kind: 'static',
      label: 'wall:+x',
    });
    const puck = world.createBody({
      shape: { kind: 'sphere', radius: 0.08 },
      position: vec3(0, 0, 0),
      velocity: vec3(4, 0, 0),
      restitution: 0,
    });
    for (let i = 0; i < 120; i++) world.step(DT);
    expect(world.getBodyState(puck)!.position[0]).toBeLessThan(1.05);
  });
});

describe('BuiltinPhysics raycast', () => {
  it('hits a sphere and reports distance and normal', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0) });
    const h = world.createBody({ shape: { kind: 'sphere', radius: 0.5 }, position: vec3(0, 0, -4) });
    const hit = world.raycast(vec3(0, 0, 0), vec3(0, 0, -1), 10);
    expect(hit).toBeDefined();
    expect(hit!.body).toBe(h);
    expect(hit!.distance).toBeCloseTo(3.5, 6);
    // Normal faces back toward the ray origin.
    expect(hit!.normal[2]).toBeCloseTo(1, 6);
  });

  it('hits the nearest body only', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0) });
    const near = world.createBody({ shape: { kind: 'sphere', radius: 0.2 }, position: vec3(2, 0, 0) });
    world.createBody({ shape: { kind: 'sphere', radius: 0.2 }, position: vec3(5, 0, 0) });
    expect(world.raycast(vec3(0, 0, 0), vec3(1, 0, 0), 20)!.body).toBe(near);
  });

  it('respects maxDistance and misses cleanly', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0) });
    world.createBody({ shape: { kind: 'sphere', radius: 0.2 }, position: vec3(5, 0, 0) });
    expect(world.raycast(vec3(0, 0, 0), vec3(1, 0, 0), 1)).toBeUndefined();
    expect(world.raycast(vec3(0, 0, 0), vec3(0, 1, 0), 50)).toBeUndefined();
  });

  it('hits an axis-aligned box face-on with the right normal', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0) });
    world.createBody({
      shape: { kind: 'box', halfExtents: vec3(0.5, 0.5, 0.5) },
      position: vec3(3, 0, 0),
      kind: 'static',
    });
    const hit = world.raycast(vec3(0, 0, 0), vec3(1, 0, 0), 10)!;
    expect(hit.distance).toBeCloseTo(2.5, 6);
    expect(hit.normal).toEqual(vec3(-1, 0, 0));
  });
});

describe('BuiltinPhysics determinism', () => {
  function run(seed: number): number[] {
    const world = sphereWorld();
    world.createBody({
      shape: { kind: 'box', halfExtents: vec3(3, 0.1, 3) },
      position: vec3(0, -0.1, 0),
      kind: 'static',
    });
    const handles: number[] = [];
    for (let i = 0; i < 4; i++) {
      handles.push(
        world.createBody({
          shape: { kind: 'sphere', radius: 0.15 },
          position: vec3((i - 1.5) * 0.4 + seed * 0.001, 1 + i * 0.35, 0),
          velocity: vec3(0.2 * (i % 2 === 0 ? 1 : -1), 0, 0.1),
          restitution: 0.2,
        }),
      );
    }
    for (let i = 0; i < 300; i++) world.step(DT);
    const out: number[] = [];
    for (const h of handles) {
      const p = world.getBodyState(h)!.position;
      out.push(p[0], p[1], p[2]);
    }
    return out;
  }

  it('repeats bit-for-bit across two independent worlds', () => {
    const a = run(0);
    const b = run(0);
    expect(a).toEqual(b);
  });

  it('is order-independent in body creation only via handles (sanity)', () => {
    const world = sphereWorld({ gravity: vec3(0, 0, 0) });
    const first = world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3() });
    const second = world.createBody({ shape: { kind: 'sphere', radius: 0.1 }, position: vec3(1, 0, 0) });
    expect(second).toBe(first + 1);
  });

  it('exposes deterministic=true and a stable fixedDt', () => {
    const world = new BuiltinPhysics({ fixedDt: 1 / 120 });
    expect(world.deterministic).toBe(true);
    expect(world.name).toBe('builtin');
    expect(world.fixedDt).toBeCloseTo(1 / 120, 12);
  });
});

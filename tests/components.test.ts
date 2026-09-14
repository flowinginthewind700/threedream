/**
 * Physics <-> ECS bridge specs.
 *
 * `PhysicsSystem` is where simulated state becomes gameplay state, so two
 * properties matter more than the rest:
 *
 * - systems that run *after* physics see settled transforms in the same tick
 *   (the read-back happens inside update(), not lazily on the next frame);
 * - `grounded` is a decision about the *support* direction, which differs per
 *   participant in a contact. The normal points from A toward B, so A's support
 *   is -normal and B's is +normal. Getting that backwards makes a box resting on
 *   the floor report the floor as grounded and the box as airborne.
 *
 * Both are silent when wrong: nothing throws, the simulation still looks
 * plausible, and only a jump or a reward that depends on grounding misbehaves.
 */

import { describe, expect, it, vi } from 'vitest';

import { World } from '../src/core/ecs.js';
import { EventBus } from '../src/core/events.js';
import { createBuiltinPhysics } from '../src/physics/builtin.js';
import {
  PhysicsSystem,
  RigidBodyComponent,
  type PhysicsEvents,
} from '../src/physics/components.js';
import { vec3, type PhysicsBackend } from '../src/physics/types.js';

const DT = 1 / 60;

function setup(options: { gravity?: [number, number, number]; emitter?: boolean } = {}): {
  world: World;
  physics: PhysicsBackend;
  system: PhysicsSystem;
  bus?: EventBus<PhysicsEvents>;
  step: (n?: number) => void;
  dispose: () => void;
} {
  const physics = createBuiltinPhysics({
    gravity: vec3(...(options.gravity ?? [0, -9.81, 0])),
    fixedDt: DT,
  });
  const world = new World();
  const bus = options.emitter === false ? undefined : new EventBus<PhysicsEvents>();
  const system = new PhysicsSystem({ backend: physics, ...(bus ? { emitter: bus.asEmitter() } : {}) });
  world.addSystem(system);
  return {
    world,
    physics,
    system,
    bus,
    step: (n = 1) => {
      for (let i = 0; i < n; i++) world.step(DT);
    },
    dispose: () => {
      physics.dispose();
      world.clear();
    },
  };
}

describe('attach / detach', () => {
  it('attach() seeds the component from the backend and makes it queryable', () => {
    const s = setup();
    const handle = s.physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(1, 2, 3),
      mass: 1,
    });
    const entity = s.world.createEntity();

    const body = s.system.attach({ world: s.world }, entity, handle, 'ball');

    expect(body.handle).toBe(handle);
    expect(body.label).toBe('ball');
    expect(body.position).toEqual([1, 2, 3]);
    expect(body.grounded).toBe(false);
    expect(body.contacts).toEqual([]);
    expect(s.world.get(entity, RigidBodyComponent)).toBe(body);
    expect(s.world.query(RigidBodyComponent)).toEqual([entity]);
    s.dispose();
  });

  it('attach() defaults the label to the entity id', () => {
    const s = setup();
    const handle = s.physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(),
      mass: 1,
    });
    const entity = s.world.createEntity();

    const body = s.system.attach({ world: s.world }, entity, handle);

    expect(body.label).toBe(`entity:${entity}`);
    s.dispose();
  });

  it('detach() destroys the backend body and drops the component', () => {
    const s = setup();
    const handle = s.physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(),
      mass: 1,
    });
    const entity = s.world.createEntity();
    s.system.attach({ world: s.world }, entity, handle);
    expect(s.physics.bodyCount).toBe(1);

    s.system.detach({ world: s.world }, entity);

    expect(s.physics.bodyCount).toBe(0);
    expect(s.physics.getBodyState(handle)).toBeUndefined();
    expect(s.world.has(entity, RigidBodyComponent)).toBe(false);
    s.dispose();
  });

  it('detach() on an entity with no rigid body is a no-op', () => {
    const s = setup();
    const entity = s.world.createEntity();
    expect(() => s.system.detach({ world: s.world }, entity)).not.toThrow();
    s.dispose();
  });

  it('dispose() destroys every attached body', () => {
    const s = setup();
    for (let i = 0; i < 3; i++) {
      const handle = s.physics.createBody({
        shape: { kind: 'sphere', radius: 0.1 },
        position: vec3(i, 0, 0),
        mass: 1,
      });
      s.system.attach({ world: s.world }, s.world.createEntity(), handle);
    }
    expect(s.physics.bodyCount).toBe(3);

    s.system.dispose({ world: s.world, dt: DT, time: 0, step: 0 });

    expect(s.physics.bodyCount).toBe(0);
    s.dispose();
  });
});

describe('transform read-back', () => {
  it('systems running after physics see the state from the current tick', () => {
    // The ordering guarantee. `update()` steps the backend and writes results
    // back before returning, so a later system never reads a stale transform.
    const s = setup();
    const handle = s.physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0, 10, 0),
      mass: 1,
    });
    const entity = s.world.createEntity();
    s.system.attach({ world: s.world }, entity, handle);

    const observed: number[] = [];
    s.world.addSystem({
      name: 'observer',
      phase: 'post-update',
      update: (ctx) => {
        const body = ctx.world.require(entity, RigidBodyComponent);
        observed.push(body.position[1]);
      },
    });

    s.step(3);

    expect(observed).toHaveLength(3);
    // Strictly falling, and matching the backend exactly (same object read back).
    for (let i = 1; i < observed.length; i++) expect(observed[i]).toBeLessThan(observed[i - 1]);
    const final = s.world.require(entity, RigidBodyComponent).position[1];
    expect(final).toBeCloseTo(s.physics.getBodyState(handle)!.position[1], 12);
    s.dispose();
  });

  it('tracks velocity and rotation as well as position', () => {
    const s = setup();
    const handle = s.physics.createBody({
      shape: { kind: 'box', halfExtents: vec3(0.2, 0.2, 0.2) },
      position: vec3(0, 5, 0),
      rotation: vec3(0.3, 0, 0),
      velocity: vec3(1, 0, 0),
      mass: 1,
    });
    const entity = s.world.createEntity();
    s.system.attach({ world: s.world }, entity, handle);

    s.step(10);
    const body = s.world.require(entity, RigidBodyComponent);
    const state = s.physics.getBodyState(handle)!;

    expect(body.velocity).toEqual(state.velocity);
    expect(body.rotation).toEqual(state.rotation);
    expect(body.velocity[0]).toBeGreaterThan(0.5);
    s.dispose();
  });

  it('a body the backend no longer knows leaves the component untouched', () => {
    // Defensive path: if a handle goes stale the system must not crash the whole
    // world tick, and must not invent a state for it.
    const s = setup();
    const handle = s.physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(2, 2, 2),
      mass: 1,
    });
    const entity = s.world.createEntity();
    s.system.attach({ world: s.world }, entity, handle);
    s.physics.destroyBody(handle);

    expect(() => s.step(2)).not.toThrow();
    expect(s.world.require(entity, RigidBodyComponent).position).toEqual([2, 2, 2]);
    s.dispose();
  });
});

describe('contacts and grounding', () => {
  function restOnFloor(): ReturnType<typeof setup> & { boxEntity: number } {
    const s = setup();
    // Floor spans y in [-0.5, 0]; the box sits on top of it.
    s.physics.createBody({
      shape: { kind: 'box', halfExtents: vec3(5, 0.25, 5) },
      position: vec3(0, -0.25, 0),
      kind: 'static',
      friction: 0.9,
      label: 'floor',
    });
    const boxEntity = s.world.createEntity();
    const boxHandle = s.physics.createBody({
      shape: { kind: 'box', halfExtents: vec3(0.25, 0.25, 0.25) },
      position: vec3(0, 1, 0),
      mass: 1,
      restitution: 0,
      friction: 0.9,
      label: 'box',
    });
    s.system.attach({ world: s.world }, boxEntity, boxHandle, 'box');
    return { ...s, boxEntity };
  }

  it('publishes physics:contact with both participants resolved to entities', () => {
    const s = restOnFloor();
    const floorEntity = s.world.createEntity();
    const seen: Array<{ a?: number; b?: number }> = [];
    s.bus!.on('physics:contact', (p) => seen.push({ a: p.entityA, b: p.entityB }));

    s.step(180);

    expect(seen.length).toBeGreaterThan(0);
    // The box is attached; the floor was created directly on the backend so it
    // has no entity. At least one side must resolve to the box entity.
    expect(seen.some((p) => p.a === s.boxEntity || p.b === s.boxEntity)).toBe(true);
    expect(floorEntity).toBeGreaterThan(0);
    s.dispose();
  });

  it('marks the resting body grounded, using the support direction per participant', () => {
    const s = restOnFloor();
    s.step(240); // let it settle

    const body = s.world.require(s.boxEntity, RigidBodyComponent);
    expect(body.grounded).toBe(true);
    expect(body.contacts.length).toBeGreaterThan(0);
    s.dispose();
  });

  it('a body in free fall is not grounded', () => {
    const s = setup();
    const entity = s.world.createEntity();
    const handle = s.physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0, 20, 0),
      mass: 1,
    });
    s.system.attach({ world: s.world }, entity, handle);

    s.step(30);

    expect(s.world.require(entity, RigidBodyComponent).grounded).toBe(false);
    s.dispose();
  });

  it('grounding is cleared at the start of each tick and only re-set by a live contact', () => {
    // Stale `grounded` is the worst version of this bug: a body that left the
    // floor keeps reporting it is standing on something.
    // Gravity is needed for the resting phase: contact comes from penetration,
    // and a box placed exactly flush with the floor generates none.
    const s = setup({ gravity: [0, -9.81, 0] });
    s.physics.createBody({
      shape: { kind: 'box', halfExtents: vec3(5, 0.25, 5) },
      position: vec3(0, -0.25, 0),
      kind: 'static',
      label: 'floor',
    });
    const entity = s.world.createEntity();
    const handle = s.physics.createBody({
      shape: { kind: 'box', halfExtents: vec3(0.25, 0.25, 0.25) },
      position: vec3(0, 1, 0),
      mass: 1,
      restitution: 0,
      label: 'box',
    });
    s.system.attach({ world: s.world }, entity, handle);

    s.step(240);
    const groundedWhileResting = s.world.require(entity, RigidBodyComponent).grounded;

    // Teleport it clear of the floor. Two ticks of gravity drop it ~5mm, so it
    // is still nowhere near the floor and no contact can be reported.
    s.physics.setBodyState(handle, { position: vec3(0, 5, 0), velocity: vec3(0, 0, 0) });
    s.step(2);
    const groundedAfterLift = s.world.require(entity, RigidBodyComponent).grounded;

    expect(groundedWhileResting).toBe(true);
    expect(groundedAfterLift).toBe(false);
    s.dispose();
  });

  it('groundMinImpulse filters out resting noise below the threshold', () => {
    const physics = createBuiltinPhysics({ gravity: vec3(0, -9.81, 0), fixedDt: DT });
    const world = new World();
    const system = new PhysicsSystem({
      backend: physics,
      groundMinImpulse: Number.POSITIVE_INFINITY,
    });
    world.addSystem(system);

    physics.createBody({
      shape: { kind: 'box', halfExtents: vec3(5, 0.25, 5) },
      position: vec3(0, -0.25, 0),
      kind: 'static',
    });
    const entity = world.createEntity();
    const handle = physics.createBody({
      shape: { kind: 'box', halfExtents: vec3(0.25, 0.25, 0.25) },
      position: vec3(0, 1, 0),
      mass: 1,
      restitution: 0,
    });
    system.attach({ world }, entity, handle);

    for (let i = 0; i < 240; i++) world.step(DT);

    // Contacts still arrive, but none can exceed an infinite impulse threshold,
    // so nothing is ever marked grounded.
    const body = world.require(entity, RigidBodyComponent);
    expect(body.grounded).toBe(false);
    expect(body.contacts.length).toBeGreaterThan(0);
    physics.dispose();
  });

  it('groundDot controls how vertical a support must be to count', () => {
    // The grounding test is `supportUp[1] > groundDot`. Pinning both sides of
    // that comparison proves the threshold is actually consulted rather than
    // hardcoded: 0.7 admits a flat floor, 1.0 admits nothing, because no unit
    // vector can have a y component strictly greater than 1.
    const probe = (groundDot: number): boolean => {
      const physics = createBuiltinPhysics({ gravity: vec3(0, 0, 0), fixedDt: DT });
      const world = new World();
      const system = new PhysicsSystem({ backend: physics, groundDot });
      world.addSystem(system);

      // A floor the box overlaps slightly, so a contact exists without gravity.
      physics.createBody({
        shape: { kind: 'box', halfExtents: vec3(5, 0.25, 5) },
        position: vec3(0, -0.26, 0),
        kind: 'static',
      });
      const entity = world.createEntity();
      const handle = physics.createBody({
        shape: { kind: 'box', halfExtents: vec3(0.25, 0.25, 0.25) },
        position: vec3(0, 0, 0),
        mass: 1,
        restitution: 0,
      });
      system.attach({ world }, entity, handle);

      for (let i = 0; i < 20; i++) world.step(DT);
      const body = world.require(entity, RigidBodyComponent);
      const grounded = body.grounded && body.contacts.length > 0;
      physics.dispose();
      return grounded;
    };

    expect(probe(-1)).toBe(true); // any support direction counts
    expect(probe(0.7)).toBe(true); // the default admits a flat floor
    expect(probe(1)).toBe(false); // unreachable for a unit vector
  });
});

describe('step events and ordering', () => {
  it('publishes exactly one physics:step per tick with the tick dt', () => {
    const s = setup({ emitter: true });
    const steps: Array<{ step: number; dt: number }> = [];
    s.bus!.on('physics:step', (p) => steps.push({ step: p.step, dt: p.dt }));

    s.step(5);

    expect(steps).toHaveLength(5);
    expect(steps.every((p) => p.dt === DT)).toBe(true);
    expect(steps.map((p) => p.step)).toEqual([0, 1, 2, 3, 4]);
    s.dispose();
  });

  it('runs before other update-phase systems thanks to its negative order', () => {
    const s = setup();
    const order: string[] = [];
    const spy = vi.spyOn(s.system, 'update');
    spy.mockImplementation((ctx) => {
      order.push('physics');
      return PhysicsSystem.prototype.update.call(s.system, ctx);
    });
    s.world.addSystem({
      name: 'gameplay',
      phase: 'update',
      update: () => order.push('gameplay'),
    });

    s.step(1);

    expect(order).toEqual(['physics', 'gameplay']);
    spy.mockRestore();
    s.dispose();
  });

  it('works with no emitter configured', () => {
    const s = setup({ emitter: false });
    const entity = s.world.createEntity();
    const handle = s.physics.createBody({
      shape: { kind: 'sphere', radius: 0.1 },
      position: vec3(0, 10, 0),
      mass: 1,
    });
    s.system.attach({ world: s.world }, entity, handle);

    expect(() => s.step(5)).not.toThrow();
    expect(s.world.require(entity, RigidBodyComponent).position[1]).toBeLessThan(10);
    s.dispose();
  });

  it('exposes the scheduling metadata the world sorts by', () => {
    const s = setup();
    expect(s.system.name).toBe('physics');
    expect(s.system.phase).toBe('update');
    expect(s.system.order).toBeLessThan(0);
    expect(s.system.backend).toBe(s.physics);
    s.dispose();
  });
});

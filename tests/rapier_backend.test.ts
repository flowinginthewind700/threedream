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

/**
 * Physics <-> ECS bridge.
 *
 * `RigidBody` is the authoring component: it holds the physics handle and the
 * last state read back from the backend. `PhysicsSystem` steps the backend once
 * per fixed tick and writes results back into the world, so gameplay systems
 * that run after it always see settled transforms.
 *
 * Contact events are republished on the world event bus so gameplay code can
 * react ("player landed", "gripper touched object") without polling physics.
 */

import { defineComponent, type System, type SystemContext } from '../core/ecs.js';
import type { EventEmitter } from '../core/events.js';
import type {
  BodyState,
  ContactEvent,
  PhysicsBackend,
  Vec3,
} from './types.js';
import { vec3 } from './types.js';

export interface RigidBody {
  handle: number;
  position: Vec3;
  rotation: Vec3;
  velocity: Vec3;
  label: string;
  /** True while any contact was reported this tick. */
  grounded: boolean;
  /** Contacts observed this tick, cleared at the start of each step. */
  contacts: ContactEvent[];
}

export const RigidBodyComponent = defineComponent<RigidBody>('RigidBody');

export type PhysicsEvents = {
  'physics:contact': { contact: ContactEvent; entityA?: number; entityB?: number };
  'physics:step': { step: number; dt: number };
};

export interface PhysicsSystemOptions {
  backend: PhysicsBackend;
  /**
   * Publishes contact and step events. Typed as an emitter, not a full bus, so
   * an engine-level `EventBus<EngineEvents>` can be passed in even though
   * `EventBus` itself is invariant in its event map.
   */
  emitter?: EventEmitter<PhysicsEvents>;
  /** Contacts whose normal points up past this dot product mark a body grounded. */
  groundDot?: number;
  /** Contacts with impulse below this are ignored for grounding. */
  groundMinImpulse?: number;
}

export class PhysicsSystem implements System {
  readonly name = 'physics';
  readonly phase: SystemPhaseValue = 'update';
  readonly order = -100;

  readonly backend: PhysicsBackend;
  private readonly emitter?: EventEmitter<PhysicsEvents>;
  private readonly groundDot: number;
  private readonly groundMinImpulse: number;
  /** physics handle -> entity id */
  private readonly handleToEntity = new Map<number, number>();
  /** Set at the top of each update so contact handling can resolve components. */
  private lastWorld?: SystemContext['world'];

  constructor(options: PhysicsSystemOptions) {
    this.backend = options.backend;
    this.emitter = options.emitter;
    this.groundDot = options.groundDot ?? 0.7;
    this.groundMinImpulse = options.groundMinImpulse ?? 0;
  }

  /**
   * Bind an entity to a physics body. Call after `backend.createBody()`.
   * Returns the component so callers can chain.
   */
  attach(
    ctx: Pick<SystemContext, 'world'>,
    entity: number,
    handle: number,
    label = `entity:${entity}`,
  ): RigidBody {
    const state: BodyState | undefined = this.backend.getBodyState(handle);
    this.handleToEntity.set(handle, entity);
    const component: RigidBody = {
      handle,
      position: state?.position ?? vec3(),
      rotation: state?.rotation ?? vec3(),
      velocity: state?.velocity ?? vec3(),
      label,
      grounded: false,
      contacts: [],
    };
    return ctx.world.add(entity, RigidBodyComponent, component);
  }

  detach(ctx: Pick<SystemContext, 'world'>, entity: number): void {
    const body = ctx.world.get(entity, RigidBodyComponent);
    if (!body) return;
    this.handleToEntity.delete(body.handle);
    this.backend.destroyBody(body.handle);
    ctx.world.remove(entity, RigidBodyComponent);
  }

  update(ctx: SystemContext): void {
    const { world } = ctx;
    this.lastWorld = world;

    for (const entity of world.query(RigidBodyComponent)) {
      const body = world.require(entity, RigidBodyComponent);
      body.grounded = false;
      body.contacts.length = 0;
    }

    this.backend.step(ctx.dt);

    // Read back transforms.
    for (const entity of world.query(RigidBodyComponent)) {
      const body = world.require(entity, RigidBodyComponent);
      const state = this.backend.getBodyState(body.handle);
      if (!state) continue;
      body.position = state.position;
      body.rotation = state.rotation;
      body.velocity = state.velocity;
    }

    // Distribute contacts to both participants and publish events.
    for (const contact of this.backend.drainContacts()) {
      const entityA = this.handleToEntity.get(contact.a);
      const entityB = this.handleToEntity.get(contact.b);
      // `contact.normal` points from a toward b. The surface pushes each body
      // *away* from the other, so A's support direction is -normal and B's is
      // +normal. Grounding tests that support direction against world-up.
      this.markGrounded(entityA, contact, true);
      this.markGrounded(entityB, contact, false);
      this.emitter?.emit('physics:contact', { contact, entityA, entityB });
    }

    this.emitter?.emit('physics:step', { step: ctx.step, dt: ctx.dt });
  }

  private markGrounded(
    entity: number | undefined,
    contact: ContactEvent,
    flipNormal = false,
  ): void {
    if (entity === undefined) return;
    const body = this.bodyOf(entity);
    if (!body) return;
    body.contacts.push(contact);
    const up: Vec3 = flipNormal
      ? [-contact.normal[0], -contact.normal[1], -contact.normal[2]]
      : contact.normal;
    const strong = contact.impulse >= this.groundMinImpulse;
    if (strong && up[1] > this.groundDot) body.grounded = true;
  }

  private bodyOf(entity: number): RigidBody | undefined {
    return this.lastWorld?.get(entity, RigidBodyComponent);
  }

  dispose(ctx: SystemContext): void {
    for (const entity of ctx.world.query(RigidBodyComponent)) {
      const body = ctx.world.require(entity, RigidBodyComponent);
      this.backend.destroyBody(body.handle);
    }
    this.handleToEntity.clear();
  }
}

type SystemPhaseValue = System['phase'];

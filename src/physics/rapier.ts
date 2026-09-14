/**
 * Rapier backend adapter (`@dimforge/rapier3d-compat`).
 *
 * Rapier is the production path: a mature Rust rigid-body engine compiled to
 * WASM, with full rotational dynamics, joints, CCD and continuous contact
 * manifolds — none of which the built-in solver attempts. It is the same
 * `PhysicsBackend` interface, so scenes and agents are unchanged when swapping.
 *
 * Loading is dynamic on purpose. Rapier ships WASM that Node and the browser
 * both resolve differently, and headless training should not pay for it, so
 * `createRapierPhysics()` imports the module only when a caller asks for Rapier.
 */

import type RAPIER_NS from '@dimforge/rapier3d-compat';
import {
  eulerFromQuatXYZ,
  normalizeVec3,
  quatFromEulerXYZ,
  scaleVec3,
  vec3,
  type BodyDescriptor,
  type BodyState,
  type ContactEvent,
  type PhysicsBackend,
  type PhysicsWorldOptions,
  type RayHit,
  type Vec3,
} from './types.js';

type RAPIER_T = typeof RAPIER_NS;

interface RapierBody {
  rigid: RAPIER_NS.RigidBody;
  collider: RAPIER_NS.Collider;
  label: string;
}

export class RapierPhysics implements PhysicsBackend {
  readonly name = 'rapier';
  /**
   * Rapier is deterministic given identical inputs and identical step sizes, but
   * only within one build and one platform. Flagged false because cross-platform
   * reproducibility is not guaranteed by WASM/SIMD.
   */
  readonly deterministic = false;
  readonly fixedDt: number;
  /**
   * World-level damping, mirrored onto every dynamic body at creation time.
   * Rapier has no world damping, only per-body damping, so the adapter keeps the
   * configured values and applies them in `createBody()`. Without this a scene
   * authored against `BuiltinPhysics` (which damps globally) silently loses all
   * drag when swapped to Rapier — bodies coast forever and the same policy fails.
   *
   * The decay laws differ slightly: Rapier integrates `exp(-damping * dt)` while
   * the built-in solver uses the explicit `1 - damping * dt`. They agree to first
   * order in `dt`, which is the level of equivalence "same interface, same scene"
   * can honestly promise across two solvers.
   */
  readonly linearDamping: number;
  readonly angularDamping: number;

  private readonly RAPIER: RAPIER_T;
  private readonly world: RAPIER_NS.World;
  private readonly eventQueue: RAPIER_NS.EventQueue;
  private readonly bodies = new Map<number, RapierBody>();
  /** Rapier collider handle -> our body handle. Maintained on create/destroy. */
  private readonly colliderToHandle = new Map<number, number>();
  private nextHandle = 1;
  private pendingContacts: ContactEvent[] = [];

  private constructor(
    RAPIER: RAPIER_T,
    world: RAPIER_NS.World,
    fixedDt: number,
    linearDamping: number,
    angularDamping: number,
  ) {
    this.RAPIER = RAPIER;
    this.world = world;
    this.fixedDt = fixedDt;
    this.linearDamping = linearDamping;
    this.angularDamping = angularDamping;
    this.world.timestep = fixedDt;
    this.eventQueue = new RAPIER.EventQueue(true);
  }

  static async create(options: PhysicsWorldOptions = {}): Promise<RapierPhysics> {
    const RAPIER = (await import('@dimforge/rapier3d-compat')) as unknown as RAPIER_T;
    await RAPIER.init();
    const gravity = options.gravity ?? vec3(0, -9.81, 0);
    const world = new RAPIER.World({ x: gravity[0], y: gravity[1], z: gravity[2] });
    // `solverIterations` is a world property in Rapier (the built-in solver keeps
    // it per-instance), and it must be set before the first step.
    if (options.solverIterations !== undefined) {
      world.numSolverIterations = Math.max(1, Math.round(options.solverIterations));
    }
    return new RapierPhysics(
      RAPIER,
      world,
      options.fixedDt ?? 1 / 60,
      options.linearDamping ?? 0.05,
      options.angularDamping ?? 0.2,
    );
  }

  get bodyCount(): number {
    return this.bodies.size;
  }

  createBody(descriptor: BodyDescriptor): number {
    const R = this.RAPIER;
    const position = descriptor.position;
    const kind = descriptor.kind ?? 'dynamic';
    const bodyDesc =
      kind === 'static'
        ? R.RigidBodyDesc.fixed()
        : kind === 'kinematic'
          ? R.RigidBodyDesc.kinematicPositionBased()
          : R.RigidBodyDesc.dynamic();
    bodyDesc.setTranslation(position[0], position[1], position[2]);
    // Damping is per-body in Rapier; see the class note on why the world options
    // are mirrored here.
    if (this.linearDamping > 0) bodyDesc.setLinearDamping(this.linearDamping);
    if (this.angularDamping > 0) bodyDesc.setAngularDamping(this.angularDamping);
    if (descriptor.velocity) {
      bodyDesc.setLinvel(
        descriptor.velocity[0],
        descriptor.velocity[1],
        descriptor.velocity[2],
      );
    }

    const rigid = this.world.createRigidBody(bodyDesc);
    const colliderDesc = this.colliderFor(descriptor);
    colliderDesc.setRestitution(descriptor.restitution ?? 0.1);
    colliderDesc.setFriction(descriptor.friction ?? 0.7);
    colliderDesc.setCollisionGroups(this.groupsFor(descriptor));
    // Mass goes on the *collider*, not the body. `RigidBodyDesc.setAdditionalMass`
    // is lazy in Rapier: the value is folded into the body's mass properties only
    // at the next `world.step()`, so an impulse applied on the first frame after
    // creation lands on a body whose mass is still the shape's default (measured
    // ~400x too light for a 0.08 sphere). `ColliderDesc.setMass` is immediate and
    // derives inertia from the shape, which matches the descriptor's meaning of
    // "total mass". Dynamic bodies default to 1 to match `BuiltinPhysics`.
    if (kind === 'dynamic') {
      colliderDesc.setMass(descriptor.mass ?? 1);
    }
    const collider = this.world.createCollider(colliderDesc, rigid);

    const handle = this.nextHandle++;
    this.bodies.set(handle, {
      rigid,
      collider,
      label: descriptor.label ?? `body:${handle}`,
    });
    this.colliderToHandle.set(collider.handle, handle);
    return handle;
  }

  private colliderFor(descriptor: BodyDescriptor): RAPIER_NS.ColliderDesc {
    const R = this.RAPIER;
    const shape = descriptor.shape;
    switch (shape.kind) {
      case 'sphere':
        return R.ColliderDesc.ball(shape.radius);
      case 'box':
        return R.ColliderDesc.cuboid(
          shape.halfExtents[0],
          shape.halfExtents[1],
          shape.halfExtents[2],
        );
      case 'capsule':
        return R.ColliderDesc.capsule(shape.halfHeight, shape.radius);
      default:
        throw new Error(`unsupported shape kind: ${JSON.stringify(shape)}`);
    }
  }

  private groupsFor(descriptor: BodyDescriptor): number {
    const group = (descriptor.group ?? 1) & 0xffff;
    const mask = (descriptor.mask ?? 0xffffffff) & 0xffff;
    // Rapier packs membership in the high half, filter in the low half.
    return ((group << 16) | mask) >>> 0;
  }

  destroyBody(handle: number): void {
    const body = this.bodies.get(handle);
    if (!body) return;
    this.world.removeRigidBody(body.rigid);
    this.colliderToHandle.delete(body.collider.handle);
    this.bodies.delete(handle);
  }

  getBodyState(handle: number): BodyState | undefined {
    const body = this.bodies.get(handle);
    if (!body) return undefined;
    const t = body.rigid.translation();
    const r = body.rigid.rotation();
    const v = body.rigid.linvel();
    const w = body.rigid.angvel();
    return {
      position: [t.x, t.y, t.z],
      // Rapier stores quaternions; BodyState.rotation is Euler XYZ.
      rotation: eulerFromQuatXYZ([r.x, r.y, r.z, r.w]),
      velocity: [v.x, v.y, v.z],
      angularVelocity: [w.x, w.y, w.z],
    };
  }

  setBodyState(handle: number, state: Partial<BodyState>): void {
    const body = this.bodies.get(handle);
    if (!body) return;
    if (state.position) {
      const [x, y, z] = state.position;
      body.rigid.setTranslation({ x, y, z }, true);
    }
    if (state.rotation) {
      const [x, y, z, w] = quatFromEulerXYZ(state.rotation);
      body.rigid.setRotation({ x, y, z, w }, true);
    }
    if (state.velocity) {
      const [x, y, z] = state.velocity;
      body.rigid.setLinvel({ x, y, z }, true);
    }
    if (state.angularVelocity) {
      const [x, y, z] = state.angularVelocity;
      body.rigid.setAngvel({ x, y, z }, true);
    }
  }

  applyImpulse(handle: number, impulse: Vec3): void {
    const body = this.bodies.get(handle);
    if (!body) return;
    body.rigid.applyImpulse({ x: impulse[0], y: impulse[1], z: impulse[2] }, true);
  }

  applyForce(handle: number, force: Vec3): void {
    const body = this.bodies.get(handle);
    if (!body) return;
    body.rigid.addForce({ x: force[0], y: force[1], z: force[2] }, true);
  }

  step(dt: number): void {
    this.world.timestep = dt;
    this.world.step(this.eventQueue);
    this.pendingContacts = [];
    const labels = (handle: number) => this.bodies.get(handle)?.label ?? `body:${handle}`;
    this.eventQueue.drainContactForceEvents((event) => {
      // Temp events are reused by Rapier, so read handles immediately and
      // never retain the event object itself.
      const c1 = event.collider1();
      const c2 = event.collider2();
      const h1 = this.handleForCollider(c1);
      const h2 = this.handleForCollider(c2);
      if (h1 === undefined || h2 === undefined) return;
      const direction = event.maxForceDirection();
      this.pendingContacts.push({
        a: h1,
        b: h2,
        normal: [direction.x, direction.y, direction.z],
        depth: 0,
        impulse: event.totalForceMagnitude(),
        labels: [labels(h1), labels(h2)],
      });
    });
  }

  /**
   * Map a Rapier collider handle back to ours. Handles are stable for the
   * lifetime of a collider, so the reverse index is built once per collider.
   */
  private handleForCollider(colliderHandle: number): number | undefined {
    return this.colliderToHandle.get(colliderHandle);
  }

  drainContacts(): ContactEvent[] {
    const out = this.pendingContacts;
    this.pendingContacts = [];
    return out;
  }

  raycast(origin: Vec3, direction: Vec3, maxDistance: number): RayHit | undefined {
    const dir = normalizeVec3(direction);
    const ray = new this.RAPIER.Ray(
      { x: origin[0], y: origin[1], z: origin[2] },
      { x: dir[0], y: dir[1], z: dir[2] },
    );
    const hit = this.world.castRayAndGetNormal(ray, maxDistance, true);
    if (!hit) return undefined;
    const handle = this.handleForCollider(hit.collider.handle);
    if (handle === undefined) return undefined;
    const toi = hit.timeOfImpact;
    const point = scaleVec3(dir, toi);
    const n = hit.normal;
    // A miss on the normal (degenerate hit) falls back to the ray direction.
    const normal: Vec3 =
      n && (n.x !== 0 || n.y !== 0 || n.z !== 0) ? [n.x, n.y, n.z] : dir;
    return {
      body: handle,
      point: [origin[0] + point[0], origin[1] + point[1], origin[2] + point[2]],
      normal,
      distance: toi,
    };
  }

  dispose(): void {
    this.bodies.clear();
    this.colliderToHandle.clear();
    this.pendingContacts = [];
    this.world.free();
  }
}

export async function createRapierPhysics(
  options?: PhysicsWorldOptions,
): Promise<RapierPhysics> {
  return RapierPhysics.create(options);
}

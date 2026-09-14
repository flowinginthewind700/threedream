/**
 * Built-in deterministic rigid-body solver.
 *
 * Scope is deliberately narrow: spheres and axis-aligned boxes, gravity,
 * sequential-impulse contact resolution with restitution and Coulomb friction,
 * and sphere/AABB raycasts. That is enough to train locomotion and
 * manipulation-style agents headless in Node, and it has no WASM or Worker
 * dependency, so `npm run train` and CI both work from a bare checkout.
 *
 * Determinism: fixed timestep, integer-keyed body iteration in handle order,
 * and no use of `Math.random`. Two runs with the same seed produce identical
 * trajectories, which is what makes reward curves comparable.
 *
 * Known limits (tracked in ROADMAP): box rotation is not solved (boxes keep
 * their authored orientation), capsules are declared but rejected at creation.
 */

import {
  addVec3,
  axisVec3,
  crossVec3,
  distanceVec3,
  dotVec3,
  lengthVec3,
  normalizeVec3,
  scaleVec3,
  subVec3,
  vec3,
  type BodyDescriptor,
  type BodyState,
  type ContactEvent,
  type PhysicsBackend,
  type PhysicsWorldOptions,
  type RayHit,
  type Vec3,
} from './types.js';

interface Body extends BodyDescriptor {
  handle: number;
  position: Vec3;
  rotation: Vec3;
  velocity: Vec3;
  angularVelocity: Vec3;
  invMass: number;
  restitution: number;
  friction: number;
  group: number;
  mask: number;
  label: string;
  aabbMin: Vec3;
  aabbMax: Vec3;
}

interface Contact {
  a: Body;
  b: Body;
  /** Points from a toward b. */
  normal: Vec3;
  depth: number;
  point: Vec3;
}

const PENETRATION_SLOP = 0.005;
const BAUMGARTE = 0.2;
const ZERO: Vec3 = [0, 0, 0];

export class BuiltinPhysics implements PhysicsBackend {
  readonly name = 'builtin';
  readonly deterministic = true;
  readonly fixedDt: number;

  private readonly gravity: Vec3;
  private readonly iterations: number;
  private readonly linearDamping: number;
  private readonly angularDamping: number;

  private readonly bodies = new Map<number, Body>();
  private nextHandle = 1;
  private contacts: ContactEvent[] = [];

  constructor(options: PhysicsWorldOptions = {}) {
    this.gravity = options.gravity ?? vec3(0, -9.81, 0);
    this.fixedDt = options.fixedDt ?? 1 / 60;
    this.iterations = options.solverIterations ?? 8;
    this.linearDamping = options.linearDamping ?? 0.05;
    this.angularDamping = options.angularDamping ?? 0.2;
  }

  get bodyCount(): number {
    return this.bodies.size;
  }

  createBody(descriptor: BodyDescriptor): number {
    if (descriptor.shape.kind === 'capsule') {
      throw new Error('BuiltinPhysics does not support capsule shapes yet; use a sphere or box');
    }
    const kind = descriptor.kind ?? 'dynamic';
    const mass = kind === 'dynamic' ? (descriptor.mass ?? 1) : 0;
    if (mass < 0) throw new RangeError('mass must be non-negative');
    const handle = this.nextHandle++;
    const body: Body = {
      ...descriptor,
      kind,
      mass,
      handle,
      position: [...descriptor.position] as Vec3,
      rotation: descriptor.rotation ? ([...descriptor.rotation] as Vec3) : ZERO,
      velocity: descriptor.velocity ? ([...descriptor.velocity] as Vec3) : ZERO,
      angularVelocity: ZERO,
      invMass: mass > 0 ? 1 / mass : 0,
      restitution: descriptor.restitution ?? 0.1,
      friction: descriptor.friction ?? 0.7,
      group: descriptor.group ?? 1,
      mask: descriptor.mask ?? 0xffffffff,
      label: descriptor.label ?? `body:${handle}`,
      aabbMin: ZERO,
      aabbMax: ZERO,
    };
    this.refreshAabb(body);
    this.bodies.set(handle, body);
    return handle;
  }

  destroyBody(handle: number): void {
    this.bodies.delete(handle);
  }

  getBodyState(handle: number): BodyState | undefined {
    const body = this.bodies.get(handle);
    if (!body) return undefined;
    return {
      position: [...body.position] as Vec3,
      rotation: [...body.rotation] as Vec3,
      velocity: [...body.velocity] as Vec3,
      angularVelocity: [...body.angularVelocity] as Vec3,
    };
  }

  setBodyState(handle: number, state: Partial<BodyState>): void {
    const body = this.bodies.get(handle);
    if (!body) return;
    if (state.position) body.position = [...state.position] as Vec3;
    if (state.rotation) body.rotation = [...state.rotation] as Vec3;
    if (state.velocity) body.velocity = [...state.velocity] as Vec3;
    if (state.angularVelocity) body.angularVelocity = [...state.angularVelocity] as Vec3;
    this.refreshAabb(body);
  }

  applyImpulse(handle: number, impulse: Vec3): void {
    const body = this.bodies.get(handle);
    if (!body || body.invMass === 0) return;
    body.velocity = addVec3(body.velocity, scaleVec3(impulse, body.invMass));
  }

  applyForce(handle: number, force: Vec3): void {
    const body = this.bodies.get(handle);
    if (!body || body.invMass === 0) return;
    body.velocity = addVec3(body.velocity, scaleVec3(force, body.invMass * this.fixedDt));
  }

  step(dt: number): void {
    const ordered = [...this.bodies.values()].sort((a, b) => a.handle - b.handle);
    this.contacts = [];

    // Integrate forces.
    for (const body of ordered) {
      if (body.invMass === 0) continue;
      body.velocity = addVec3(body.velocity, scaleVec3(this.gravity, dt));
      const damp = Math.max(0, 1 - this.linearDamping * dt);
      body.velocity = scaleVec3(body.velocity, damp);
      const spinDamp = Math.max(0, 1 - this.angularDamping * dt);
      body.angularVelocity = scaleVec3(body.angularVelocity, spinDamp);
    }

    // Broadphase: uniform sweep over handle-ordered pairs with AABB overlap.
    for (const body of ordered) this.refreshAabb(body);
    const pairs: [Body, Body][] = [];
    for (let i = 0; i < ordered.length; i++) {
      for (let j = i + 1; j < ordered.length; j++) {
        const a = ordered[i]!;
        const b = ordered[j]!;
        if (a.invMass === 0 && b.invMass === 0) continue;
        if (!this.mayCollide(a, b)) continue;
        if (!aabbOverlap(a, b)) continue;
        pairs.push([a, b]);
      }
    }

    // Narrowphase.
    const contacts: Contact[] = [];
    for (const [a, b] of pairs) {
      const contact = this.narrowphase(a, b);
      if (contact) contacts.push(contact);
    }

    // Solve.
    for (let iter = 0; iter < this.iterations; iter++) {
      for (const contact of contacts) this.solveContact(contact, dt);
    }
    for (const contact of contacts) this.positionalCorrection(contact);

    // Integrate velocities.
    for (const body of ordered) {
      if (body.invMass === 0) continue;
      body.position = addVec3(body.position, scaleVec3(body.velocity, dt));
      body.rotation = addVec3(body.rotation, scaleVec3(body.angularVelocity, dt));
      this.refreshAabb(body);
    }

    this.contacts = contacts.filter((c) => c.depth > 0).map((c) => ({
      a: c.a.handle,
      b: c.b.handle,
      normal: c.normal,
      depth: c.depth,
      impulse: 0,
      labels: [c.a.label, c.b.label] as [string, string],
    }));
  }

  drainContacts(): ContactEvent[] {
    const out = this.contacts;
    this.contacts = [];
    return out;
  }

  raycast(origin: Vec3, direction: Vec3, maxDistance: number): RayHit | undefined {
    const dir = normalizeVec3(direction);
    if (lengthVec3(dir) < 0.5) return undefined;
    let best: RayHit | undefined;
    for (const body of [...this.bodies.values()].sort((a, b) => a.handle - b.handle)) {
      const hit = rayVsBody(origin, dir, body);
      if (!hit || hit.distance > maxDistance) continue;
      if (!best || hit.distance < best.distance) {
        best = { body: body.handle, ...hit };
      }
    }
    return best;
  }

  dispose(): void {
    this.bodies.clear();
    this.contacts = [];
  }

  private mayCollide(a: Body, b: Body): boolean {
    return (a.group & b.mask) !== 0 && (b.group & a.mask) !== 0;
  }

  private refreshAabb(body: Body): void {
    const e = shapeHalfExtent(body.shape);
    body.aabbMin = subVec3(body.position, e);
    body.aabbMax = addVec3(body.position, e);
  }

  private narrowphase(a: Body, b: Body): Contact | undefined {
    if (a.shape.kind === 'sphere' && b.shape.kind === 'sphere') {
      return sphereSphere(a, b);
    }
    if (a.shape.kind === 'sphere' && b.shape.kind === 'box') return sphereBox(a, b);
    if (a.shape.kind === 'box' && b.shape.kind === 'sphere') {
      const flipped = sphereBox(b, a);
      if (!flipped) return undefined;
      return { ...flipped, a, b, normal: scaleVec3(flipped.normal, -1) };
    }
    return boxBox(a, b);
  }

  private solveContact(contact: Contact, dt: number): void {
    const { a, b, normal } = contact;
    const invMassSum = a.invMass + b.invMass;
    if (invMassSum === 0) return;

    const relative = subVec3(b.velocity, a.velocity);
    const normalVelocity = dotVec3(relative, normal);
    if (normalVelocity > 0) return;

    const restitution = Math.min(a.restitution, b.restitution);
    // Skip restitution for near-resting contacts so stacks do not jitter.
    const bounce = Math.abs(normalVelocity) < 1.5 / Math.max(dt, 1e-6) / 60 ? 0 : restitution;
    let lambda = (-(1 + bounce) * normalVelocity) / invMassSum;
    lambda = Math.max(lambda, 0);

    const impulse = scaleVec3(normal, lambda);
    a.velocity = subVec3(a.velocity, scaleVec3(impulse, a.invMass));
    b.velocity = addVec3(b.velocity, scaleVec3(impulse, b.invMass));
    this.spinFromFriction(a, normal, -1);
    this.spinFromFriction(b, normal, 1);

    // Friction: remove tangential relative velocity up to the Coulomb cone.
    const tangentVelocity = subVec3(relative, scaleVec3(normal, normalVelocity));
    const tangentSpeed = lengthVec3(tangentVelocity);
    if (tangentSpeed > 1e-6) {
      const tangent = scaleVec3(tangentVelocity, 1 / tangentSpeed);
      const mu = Math.sqrt(a.friction * b.friction);
      let jt = -tangentSpeed / invMassSum;
      const maxFriction = mu * lambda;
      jt = Math.max(-maxFriction, Math.min(maxFriction, jt));
      const frictionImpulse = scaleVec3(tangent, jt);
      a.velocity = subVec3(a.velocity, scaleVec3(frictionImpulse, a.invMass));
      b.velocity = addVec3(b.velocity, scaleVec3(frictionImpulse, b.invMass));
    }
  }

  /** Rolling approximation: spheres pick up spin from tangential contact impulses. */
  private spinFromFriction(body: Body, normal: Vec3, sign: number): void {
    if (body.invMass === 0 || body.shape.kind !== 'sphere') return;
    const rolling = crossVec3(normal, body.velocity);
    body.angularVelocity = addVec3(
      body.angularVelocity,
      scaleVec3(rolling, (sign * body.friction * 0.05) / body.shape.radius),
    );
  }

  private positionalCorrection(contact: Contact): void {
    const { a, b, normal, depth } = contact;
    const invMassSum = a.invMass + b.invMass;
    if (invMassSum === 0) return;
    const correction = (Math.max(depth - PENETRATION_SLOP, 0) * BAUMGARTE) / invMassSum;
    a.position = subVec3(a.position, scaleVec3(normal, correction * a.invMass));
    b.position = addVec3(b.position, scaleVec3(normal, correction * b.invMass));
    this.refreshAabb(a);
    this.refreshAabb(b);
  }
}

function shapeHalfExtent(shape: BodyDescriptor['shape']): Vec3 {
  switch (shape.kind) {
    case 'sphere':
      return vec3(shape.radius, shape.radius, shape.radius);
    case 'box':
      return shape.halfExtents;
    default:
      // Capsules are rejected at creation; keep a safe bound for typing.
      return vec3(shape.radius, shape.radius + shape.halfHeight, shape.radius);
  }
}

function aabbOverlap(a: Body, b: Body): boolean {
  return (
    a.aabbMin[0] <= b.aabbMax[0] &&
    a.aabbMax[0] >= b.aabbMin[0] &&
    a.aabbMin[1] <= b.aabbMax[1] &&
    a.aabbMax[1] >= b.aabbMin[1] &&
    a.aabbMin[2] <= b.aabbMax[2] &&
    a.aabbMax[2] >= b.aabbMin[2]
  );
}

function sphereSphere(a: Body, b: Body): Contact | undefined {
  const ra = (a.shape as { radius: number }).radius;
  const rb = (b.shape as { radius: number }).radius;
  const delta = subVec3(b.position, a.position);
  const dist = lengthVec3(delta);
  const sum = ra + rb;
  if (dist >= sum) return undefined;
  const normal = dist > 1e-9 ? scaleVec3(delta, 1 / dist) : vec3(0, 1, 0);
  return {
    a,
    b,
    normal,
    depth: sum - dist,
    point: addVec3(a.position, scaleVec3(normal, ra)),
  };
}

function sphereBox(sphere: Body, box: Body): Contact | undefined {
  const radius = (sphere.shape as { radius: number }).radius;
  const half = (box.shape as { halfExtents: Vec3 }).halfExtents;
  const local = subVec3(sphere.position, box.position);
  const clamped: Vec3 = [
    Math.max(-half[0], Math.min(half[0], local[0])),
    Math.max(-half[1], Math.min(half[1], local[1])),
    Math.max(-half[2], Math.min(half[2], local[2])),
  ];
  const delta = subVec3(local, clamped);
  const dist = lengthVec3(delta);
  const inside = dist < 1e-9;
  if (!inside && dist >= radius) return undefined;

  // Centre inside the box: push out along the shallowest axis.
  let normal: Vec3;
  let depth: number;
  if (inside) {
    const penetration: [number, Vec3][] = [0, 1, 2].map((axis) => {
      const posSide = half[axis] - local[axis];
      const negSide = half[axis] + local[axis];
      const best = Math.min(posSide, negSide);
      const sign = posSide < negSide ? 1 : -1;
      return [best + radius, axisVec3(axis, sign)] as [number, Vec3];
    });
    penetration.sort((x, y) => x[0] - y[0]);
    depth = penetration[0]![0];
    // Normal points from sphere toward box, i.e. into the surface we hit.
    normal = scaleVec3(penetration[0]![1], -1);
  } else {
    normal = normalizeVec3(scaleVec3(delta, -1));
    depth = radius - dist;
  }
  return {
    a: sphere,
    b: box,
    normal,
    depth,
    point: addVec3(box.position, clamped),
  };
}

function boxBox(a: Body, b: Body): Contact | undefined {
  const ha = (a.shape as { halfExtents: Vec3 }).halfExtents;
  const hb = (b.shape as { halfExtents: Vec3 }).halfExtents;
  const delta = subVec3(b.position, a.position);
  let bestAxis = -1;
  let bestDepth = Number.POSITIVE_INFINITY;
  for (let axis = 0; axis < 3; axis++) {
    const overlap = ha[axis] + hb[axis] - Math.abs(delta[axis]);
    if (overlap <= 0) return undefined;
    if (overlap < bestDepth) {
      bestDepth = overlap;
      bestAxis = axis;
    }
  }
  if (bestAxis < 0) return undefined;
  const normal = axisVec3(bestAxis, delta[bestAxis] >= 0 ? 1 : -1);
  return {
    a,
    b,
    normal,
    depth: bestDepth,
    point: addVec3(a.position, scaleVec3(normal, ha[bestAxis])),
  };
}

function rayVsBody(
  origin: Vec3,
  dir: Vec3,
  body: Body,
): { point: Vec3; normal: Vec3; distance: number } | undefined {
  if (body.shape.kind === 'sphere') {
    const toCenter = subVec3(body.position, origin);
    const tca = dotVec3(toCenter, dir);
    const radius = body.shape.radius;
    const d2 = dotVec3(toCenter, toCenter) - tca * tca;
    const r2 = radius * radius;
    if (d2 > r2) return undefined;
    const thc = Math.sqrt(r2 - d2);
    let t = tca - thc;
    if (t < 0) t = tca + thc;
    if (t < 0) return undefined;
    const point = addVec3(origin, scaleVec3(dir, t));
    return { point, normal: normalizeVec3(subVec3(point, body.position)), distance: t };
  }
  // Slab test for axis-aligned boxes.
  let tmin = 0;
  let tmax = Number.POSITIVE_INFINITY;
  let axis = -1;
  let axisSign = 1;
  for (let i = 0; i < 3; i++) {
    const lo = body.aabbMin[i];
    const hi = body.aabbMax[i];
    const d = dir[i];
    const o = origin[i];
    if (Math.abs(d) < 1e-9) {
      if (o < lo || o > hi) return undefined;
      continue;
    }
    let t1 = (lo - o) / d;
    let t2 = (hi - o) / d;
    let sign = -1;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
      sign = 1;
    }
    if (t1 > tmin) {
      tmin = t1;
      axis = i;
      axisSign = sign;
    }
    tmax = Math.min(tmax, t2);
    if (tmin > tmax) return undefined;
  }
  if (axis < 0 || tmin < 0) return undefined;
  const point = addVec3(origin, scaleVec3(dir, tmin));
  const normal = axisVec3(axis, axisSign);
  return { point, normal, distance: tmin };
}

export function createBuiltinPhysics(options?: PhysicsWorldOptions): BuiltinPhysics {
  return new BuiltinPhysics(options);
}

export { distanceVec3 };

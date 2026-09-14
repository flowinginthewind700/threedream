/**
 * The canonical physics scene, and a digest of it.
 *
 * Three claims in the development plan need one shared artifact to be checked
 * against: that `builtin` and the wasm kernel agree bit for bit, that Node and
 * the browser agree, and that a replay reproduces the original run. Comparing
 * two live backends only proves the first, and only inside one process. So the
 * scene is defined once here, and any runner -- vitest, the Playwright browser
 * harness, a recorded replay -- can produce the same digest from it.
 *
 * The digest is over raw IEEE-754 bit patterns, not formatted numbers. That is
 * deliberate: `toFixed(6)` would hide a one-ulp divergence, which is exactly the
 * thing a determinism gate exists to catch. Two runs that hash equal ran the
 * same arithmetic; a run that differs by one ulp hashes differently.
 *
 * Nothing here imports a backend. Callers pass one in, which is what makes the
 * same code path exercise `builtin`, `wasm`, or `rapier`.
 */

import {
  vec3,
  type ContactEvent,
  type PhysicsBackend,
  type PhysicsWorldOptions,
  type RayHit,
  type Vec3,
} from './types.js';

/** Steps in a full reference run. Long enough to settle, bounce, and slide. */
export const REFERENCE_STEPS = 600;

/** The fixed timestep the reference scene is authored against. */
export const REFERENCE_DT = 1 / 60;

/**
 * World options every reference runner must use, so runs are comparable.
 *
 * Typed as `PhysicsWorldOptions` rather than `as const`: callers spread it and
 * override individual fields, and literal-widened property types would make
 * `{ ...REFERENCE_OPTIONS, linearDamping: 0 }` a type error.
 */
export const REFERENCE_OPTIONS: PhysicsWorldOptions = {
  gravity: vec3(0, -9.81, 0),
  fixedDt: REFERENCE_DT,
  solverIterations: 8,
  linearDamping: 0.05,
  angularDamping: 0.2,
};

/** One body's full state at one step, sampled in creation order. */
export interface ReferenceSample {
  step: number;
  position: Vec3;
  rotation: Vec3;
  velocity: Vec3;
  angularVelocity: Vec3;
}

export interface ReferenceRun {
  readonly backend: string;
  readonly steps: number;
  readonly samples: ReferenceSample[];
  /** Every contact drained during the run, in drain order. */
  readonly contacts: ContactEvent[];
  /** Raycast results, one per probe, `undefined` for a miss. */
  readonly rays: (RayHit | undefined)[];
  readonly digest: string;
}

/**
 * Build the scene and return the body handles, in creation order.
 *
 * Handles are returned rather than looked up by label because the whole point is
 * that two backends assign the *same* handle to the same descriptor: if one
 * started numbering differently, every downstream assertion would be comparing
 * different bodies and still pass.
 */
export function buildReferenceScene(backend: PhysicsBackend): number[] {
  const floor = backend.createBody({
    shape: { kind: 'box', halfExtents: vec3(6, 0.5, 6) },
    position: vec3(0, -0.5, 0),
    kind: 'static',
    friction: 0.8,
    label: 'floor',
  });
  const wall = backend.createBody({
    shape: { kind: 'box', halfExtents: vec3(0.5, 3, 6) },
    position: vec3(-4, 2, 0),
    kind: 'static',
    friction: 0.4,
    label: 'wall',
  });
  const bodies = [
    floor,
    wall,
    // A bouncy ball dropped onto the floor: exercises restitution and the
    // resting-contact cutoff, which is the easiest place for two solvers to
    // diverge by an ulp and then amplify it.
    backend.createBody({
      shape: { kind: 'sphere', radius: 0.4 },
      position: vec3(0, 3, 0),
      mass: 1.5,
      restitution: 0.55,
      friction: 0.3,
      label: 'bouncy',
    }),
    // A heavy slider with sideways velocity: exercises Coulomb friction and the
    // `mu = sqrt(fa * fb)` combination rule.
    backend.createBody({
      shape: { kind: 'sphere', radius: 0.5 },
      position: vec3(-2.5, 0.5, 1),
      velocity: vec3(3, 0, -0.5),
      mass: 6,
      restitution: 0.05,
      friction: 0.9,
      label: 'slider',
    }),
    // A light ball that will be destroyed mid-run, which shifts the slot table
    // and therefore every iteration order after it.
    backend.createBody({
      shape: { kind: 'sphere', radius: 0.25 },
      position: vec3(1.5, 2, -1),
      mass: 0.4,
      restitution: 0.2,
      label: 'doomed',
    }),
    // A pair that starts overlapping, so the first step already has a contact
    // with real penetration depth and a positional correction to apply.
    backend.createBody({
      shape: { kind: 'sphere', radius: 0.5 },
      position: vec3(2, 0.5, 2),
      mass: 2,
      restitution: 0.1,
      friction: 0.6,
      label: 'pair-a',
    }),
    backend.createBody({
      shape: { kind: 'sphere', radius: 0.5 },
      position: vec3(2.6, 0.5, 2),
      mass: 2,
      restitution: 0.1,
      friction: 0.6,
      label: 'pair-b',
    }),
    // Dynamic boxes. The solver does not rotate them, but they still have to
    // produce identical AABBs and identical sphere-vs-box contact normals.
    backend.createBody({
      shape: { kind: 'box', halfExtents: vec3(0.3, 0.3, 0.3) },
      position: vec3(-1, 1.5, -2),
      rotation: vec3(0.2, 0.1, -0.3),
      mass: 3,
      friction: 0.5,
      label: 'crate',
    }),
    // A zero-mass dynamic body: `invMass` is 0, so it must not move at all.
    backend.createBody({
      shape: { kind: 'sphere', radius: 0.3 },
      position: vec3(3, 1, -3),
      kind: 'kinematic',
      label: 'frozen',
    }),
  ];
  return bodies;
}

/**
 * Run the reference scene to completion.
 *
 * The mutations are scheduled at fixed steps so the run covers more than free
 * fall: an impulse, a body destroyed (which renumbers the iteration order), a
 * teleported state, a force, and periodic raycasts. A backend that only gets
 * `step()` right would still pass a plain drop test.
 */
export function runReference(
  backend: PhysicsBackend,
  steps: number = REFERENCE_STEPS,
): ReferenceRun {
  const handles = buildReferenceScene(backend);
  const bouncy = handles[2]!;
  const slider = handles[3]!;
  const doomed = handles[4]!;
  const crate = handles[7]!;

  const samples: ReferenceSample[] = [];
  const contacts: ContactEvent[] = [];
  const rays: (RayHit | undefined)[] = [];

  for (let step = 0; step < steps; step++) {
    if (step === 90) backend.applyImpulse(bouncy, vec3(1.2, 0.4, -0.8));
    if (step === 150) backend.destroyBody(doomed);
    if (step === 210) {
      backend.setBodyState(slider, {
        position: vec3(-1, 0.5, 1.5),
        velocity: vec3(-2, 0.5, 0.25),
      });
    }
    if (step === 260) backend.applyForce(crate, vec3(40, 0, 12));
    if (step === 320) {
      // Partial write: rotation and angular velocity must be left untouched.
      backend.setBodyState(crate, { angularVelocity: vec3(0.1, -0.2, 0.3) });
    }

    backend.step(REFERENCE_DT);
    contacts.push(...backend.drainContacts());

    if (step % 25 === 0) {
      rays.push(backend.raycast(vec3(0, 8, 0), vec3(0.1, -1, 0), 30));
      rays.push(backend.raycast(vec3(-6, 1, 4), vec3(1, 0, -0.2), 20));
    }

    for (const handle of handles) {
      const state = backend.getBodyState(handle);
      // A destroyed body reports `undefined`; that transition is itself part of
      // the contract, so it is recorded as a sample of NaNs rather than skipped.
      samples.push({
        step,
        position: state?.position ?? vec3(Number.NaN, Number.NaN, Number.NaN),
        rotation: state?.rotation ?? vec3(Number.NaN, Number.NaN, Number.NaN),
        velocity: state?.velocity ?? vec3(Number.NaN, Number.NaN, Number.NaN),
        angularVelocity: state?.angularVelocity ?? vec3(Number.NaN, Number.NaN, Number.NaN),
      });
    }
  }

  return {
    backend: backend.name,
    steps,
    samples,
    contacts,
    rays,
    digest: digestRun({ samples, contacts, rays }),
  };
}

/** Everything a digest has to cover, as one flat sequence of doubles. */
export function digestValues(run: {
  samples: readonly ReferenceSample[];
  contacts: readonly ContactEvent[];
  rays: readonly (RayHit | undefined)[];
}): Float64Array {
  // 12 per sample, 7 per contact (a, b, normal, depth, impulse -- labels are
  // strings the digest cannot carry, and they are asserted separately), 8 per
  // ray hit plus one presence flag.
  const size =
    run.samples.length * 12 + run.contacts.length * 7 + run.rays.length * 9;
  const out = new Float64Array(size);
  let i = 0;
  for (const s of run.samples) {
    for (const v of [s.position, s.rotation, s.velocity, s.angularVelocity]) {
      out[i++] = v[0];
      out[i++] = v[1];
      out[i++] = v[2];
    }
  }
  for (const c of run.contacts) {
    out[i++] = c.a;
    out[i++] = c.b;
    out[i++] = c.normal[0];
    out[i++] = c.normal[1];
    out[i++] = c.normal[2];
    out[i++] = c.depth;
    out[i++] = c.impulse;
  }
  for (const r of run.rays) {
    out[i++] = r ? 1 : 0;
    out[i++] = r?.body ?? 0;
    out[i++] = r?.point[0] ?? 0;
    out[i++] = r?.point[1] ?? 0;
    out[i++] = r?.point[2] ?? 0;
    out[i++] = r?.normal[0] ?? 0;
    out[i++] = r?.normal[1] ?? 0;
    out[i++] = r?.normal[2] ?? 0;
    out[i++] = r?.distance ?? 0;
  }
  return out;
}

/**
 * FNV-1a over the raw bytes of every double, mixed with a second accumulator so
 * a reordering of two equal-magnitude values cannot hash to the same string.
 * Returned as hex plus the value count, because a digest that does not say how
 * much it covered makes a truncation bug look like a match.
 */
export function digestValuesHex(values: Float64Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < bytes.length; i++) {
    h1 ^= bytes[i]!;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 = (Math.imul(h2 ^ bytes[i]!, 0x85ebca6b) + (i & 0xff)) >>> 0;
  }
  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

export function digestRun(run: {
  samples: readonly ReferenceSample[];
  contacts: readonly ContactEvent[];
  rays: readonly (RayHit | undefined)[];
}): string {
  const values = digestValues(run);
  return `${digestValuesHex(values)}:${values.length}`;
}

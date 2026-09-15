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
  type BodyDescriptor,
  type ContactEvent,
  type PhysicsBackend,
  type PhysicsWorldOptions,
  type RayHit,
  type Vec3,
} from './types.js';
import { digestHex } from '../core/digest.js';

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

/**
 * The digest the reference scene produced when it was written, recorded.
 *
 * This is the anchor for all three determinism claims: vitest compares `builtin`
 * and the wasm kernel against it, `e2e/wasm_physics.spec.ts` compares a real
 * browser against it, and running the scene twice in one process compares a
 * replay against it. Anything reproducing this string ran the same arithmetic;
 * a one-ulp difference anywhere in 600 steps produces a different string.
 *
 * Regenerate it only when a change to the solver is deliberate, and only after
 * reading what changed: every replay recorded against the old digest is
 * invalidated by editing this line, which is why it is written down rather than
 * computed.
 */
export const REFERENCE_GOLDEN_DIGEST = '65e56ebb483e7923:90649';

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
 * The scene, as data.
 *
 * Descriptors rather than `createBody` calls for one reason: the demo pages have
 * to build renderable meshes for the *same* bodies the digest was computed over,
 * and a second hand-written copy of the scene is a second thing that can drift.
 * Reading the descriptors off this array keeps geometry, label and mass in one
 * place, so `demo/physics-check.ts` can draw exactly what it is hashing.
 *
 * Order is load-bearing: `runReference` addresses bodies by index, and two
 * backends must assign the same handle to the same descriptor.
 */
export const REFERENCE_BODIES: readonly BodyDescriptor[] = [
  {
    shape: { kind: 'box', halfExtents: vec3(6, 0.5, 6) },
    position: vec3(0, -0.5, 0),
    kind: 'static',
    friction: 0.8,
    label: 'floor',
  },
  {
    shape: { kind: 'box', halfExtents: vec3(0.5, 3, 6) },
    position: vec3(-4, 2, 0),
    kind: 'static',
    friction: 0.4,
    label: 'wall',
  },
  // A bouncy ball dropped onto the floor: exercises restitution and the
  // resting-contact cutoff, which is the easiest place for two solvers to
  // diverge by an ulp and then amplify it.
  {
    shape: { kind: 'sphere', radius: 0.4 },
    position: vec3(0, 3, 0),
    mass: 1.5,
    restitution: 0.55,
    friction: 0.3,
    label: 'bouncy',
  },
  // A heavy slider with sideways velocity: exercises Coulomb friction and the
  // `mu = sqrt(fa * fb)` combination rule.
  {
    shape: { kind: 'sphere', radius: 0.5 },
    position: vec3(-2.5, 0.5, 1),
    velocity: vec3(3, 0, -0.5),
    mass: 6,
    restitution: 0.05,
    friction: 0.9,
    label: 'slider',
  },
  // A light ball that will be destroyed mid-run, which shifts the slot table
  // and therefore every iteration order after it.
  {
    shape: { kind: 'sphere', radius: 0.25 },
    position: vec3(1.5, 2, -1),
    mass: 0.4,
    restitution: 0.2,
    label: 'doomed',
  },
  // A pair that starts overlapping, so the first step already has a contact
  // with real penetration depth and a positional correction to apply.
  {
    shape: { kind: 'sphere', radius: 0.5 },
    position: vec3(2, 0.5, 2),
    mass: 2,
    restitution: 0.1,
    friction: 0.6,
    label: 'pair-a',
  },
  {
    shape: { kind: 'sphere', radius: 0.5 },
    position: vec3(2.6, 0.5, 2),
    mass: 2,
    restitution: 0.1,
    friction: 0.6,
    label: 'pair-b',
  },
  // Dynamic boxes. The solver does not rotate them, but they still have to
  // produce identical AABBs and identical sphere-vs-box contact normals.
  {
    shape: { kind: 'box', halfExtents: vec3(0.3, 0.3, 0.3) },
    position: vec3(-1, 1.5, -2),
    rotation: vec3(0.2, 0.1, -0.3),
    mass: 3,
    friction: 0.5,
    label: 'crate',
  },
  // A zero-mass dynamic body: `invMass` is 0, so it must not move at all.
  {
    shape: { kind: 'sphere', radius: 0.3 },
    position: vec3(3, 1, -3),
    kind: 'kinematic',
    label: 'frozen',
  },
];

/**
 * Create every reference body and return the handles, in creation order.
 *
 * Handles are returned rather than looked up by label because the whole point is
 * that two backends assign the *same* handle to the same descriptor: if one
 * started numbering differently, every downstream assertion would be comparing
 * different bodies and still pass.
 */
export function buildReferenceScene(backend: PhysicsBackend): number[] {
  return REFERENCE_BODIES.map((descriptor) => backend.createBody(descriptor));
}

/**
 * One scheduled mutation of the running scene.
 *
 * `body` indexes `REFERENCE_BODIES` -- and therefore the handles
 * `buildReferenceScene` returns -- rather than naming a label, because handle
 * agreement across backends is itself under test: looking a body up by label
 * would quietly forgive a backend that numbered differently.
 */
export interface ReferenceEvent {
  readonly step: number;
  readonly body: number;
  apply(backend: PhysicsBackend, handle: number): void;
}

/**
 * The mutations the reference run performs, in step order.
 *
 * Exported alongside the descriptors for the same reason: `demo/physics-check.ts`
 * animates this scene live, and a demo that retyped the schedule would be showing
 * a different run than the one whose digest it prints. Reading both arrays from
 * here is what makes "the scene you see is the scene that was hashed" a property
 * of the code rather than a claim in a comment.
 */
export const REFERENCE_SCRIPT: readonly ReferenceEvent[] = [
  // An impulse on the bouncy ball, mid-flight.
  { step: 90, body: 2, apply: (b, h) => b.applyImpulse(h, vec3(1.2, 0.4, -0.8)) },
  // Destroying the light ball renumbers the iteration order for every step that
  // follows, which is the part a plain drop test never reaches.
  { step: 150, body: 4, apply: (b, h) => b.destroyBody(h) },
  // A full teleport of the slider: position and velocity both written.
  {
    step: 210,
    body: 3,
    apply: (b, h) =>
      b.setBodyState(h, { position: vec3(-1, 0.5, 1.5), velocity: vec3(-2, 0.5, 0.25) }),
  },
  // A sustained force on the crate.
  { step: 260, body: 7, apply: (b, h) => b.applyForce(h, vec3(40, 0, 12)) },
  // Partial write: position, rotation and velocity must be left untouched.
  {
    step: 320,
    body: 7,
    apply: (b, h) => b.setBodyState(h, { angularVelocity: vec3(0.1, -0.2, 0.3) }),
  },
];

/** Apply every event scheduled for `step`. Shared by the runner and the demo. */
export function applyReferenceEvents(
  backend: PhysicsBackend,
  handles: readonly number[],
  step: number,
): void {
  for (const event of REFERENCE_SCRIPT) {
    if (event.step === step) event.apply(backend, handles[event.body]!);
  }
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

  const samples: ReferenceSample[] = [];
  const contacts: ContactEvent[] = [];
  const rays: (RayHit | undefined)[] = [];

  for (let step = 0; step < steps; step++) {
    applyReferenceEvents(backend, handles, step);

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
 * The digest of a reference run's values.
 *
 * Kept exported under its original name because the golden digest in this file
 * and the specs that reproduce it are pinned to it, but the implementation is
 * `core/digest.ts`: the GPU particle layer needs the identical hash and must not
 * import `physics/` to get it.
 */
export function digestValuesHex(values: Float64Array): string {
  return digestHex(values);
}

export function digestRun(run: {
  samples: readonly ReferenceSample[];
  contacts: readonly ContactEvent[];
  rays: readonly (RayHit | undefined)[];
}): string {
  const values = digestValues(run);
  return `${digestValuesHex(values)}:${values.length}`;
}

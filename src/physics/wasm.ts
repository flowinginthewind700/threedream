/**
 * Wasm backend adapter: the first-party Rust kernel behind `PhysicsBackend`.
 *
 * This is the deterministic production path. `builtin.ts` stays the normative
 * specification — the Rust kernel in `rust/crates/physics` is a port of it, and
 * `tests/wasm_backend.test.ts` asserts the two agree *bit for bit* on the same
 * scene. That is a stronger contract than `rapier.ts` can offer: Rapier is a
 * different solver, so swapping it in changes trajectories, while swapping this
 * in does not. Same states, same contacts, same replay, ~2.4x faster.
 *
 * # Why the ABI looks like this
 *
 * Everything crossing the boundary is a number: world ids, body handles, and
 * `f64` buffers the *caller* allocates through `td_alloc`. No strings, no
 * structs, no `Result`. Two consequences worth knowing before editing:
 *
 * - Labels never cross. `ContactEvent.labels` is filled in here from a
 *   `Map<handle, string>`, which is also why `createBody` can reproduce
 *   builtin's `body:{handle}` default only *after* the kernel assigns a handle.
 * - Errors come back as negative sentinels and are translated into the exact
 *   `Error`/`RangeError` objects `BuiltinPhysics` throws, so a caller cannot
 *   tell the two backends apart by how they fail.
 *
 * # Loading
 *
 * The generated glue lives in `wasm/pkg`, committed as a build artifact and
 * rebuilt by `npm run build:wasm` (see `scripts/check_wasm_artifact.mjs` for the
 * CI freshness gate). It is imported dynamically, like `rapier.ts`, so a bare
 * `npm run train` never pays for instantiating a module it will not use.
 *
 * Node and the browser differ in one way that matters: wasm-bindgen's web glue
 * fetches the `.wasm` by URL, and Node's `fetch` does not speak `file:`. So on a
 * `file:` URL the bytes are read with `node:fs` and handed to `initSync`; in a
 * browser the glue fetches them itself. `import.meta.url`'s scheme is the
 * discriminator, which keeps `src/` free of Node types (`tsconfig.json` only
 * loads `vite/client`).
 */

import {
  vec3,
  type BodyDescriptor,
  type BodyState,
  type ContactEvent,
  type PhysicsBackend,
  type PhysicsWorldOptions,
  type RayHit,
  type Vec3,
} from './types.js';

/** Must equal `ABI_VERSION` in `rust/crates/physics/src/lib.rs`. */
export const WASM_ABI_VERSION = 1;

const ZERO: Vec3 = [0, 0, 0];
/** Kept identical to `builtin.ts` so a swap cannot change how a caller fails. */
const CAPSULE_MESSAGE =
  'BuiltinPhysics does not support capsule shapes yet; use a sphere or box';
const NEGATIVE_MASS_MESSAGE = 'mass must be non-negative';
/** `ERR_BAD_WORLD`: the id was destroyed, so no call against it can succeed. */
const DEAD_WORLD_MESSAGE =
  'wasm physics world is not live; it was destroyed, so create a new backend';

const SHAPE_SPHERE = 0;
const SHAPE_BOX = 1;

const MASK_POS = 1;
const MASK_ROT = 2;
const MASK_VEL = 4;
const MASK_ANG = 8;

const ERR_BAD_WORLD = -2;
const ERR_NEGATIVE_MASS = -1;
const ERR_BAD_SHAPE = -3;

/** Doubles per body in `td_world_write_states` / `td_body_get_state`. */
const STATE_STRIDE = 12;
/** Doubles per contact in `td_world_drain_contacts`. */
const CONTACT_STRIDE = 7;
/** Doubles in a `td_world_raycast` hit record. */
const RAY_STRIDE = 8;

/**
 * The ABI, restated in TS.
 *
 * Deliberately not `typeof import('../../wasm/pkg/...')`: the generated
 * `.d.ts` is an artifact, and pinning the surface here is what makes a
 * Rust-side signature change a *review* event rather than something the build
 * silently absorbs. `td_abi_version()` is checked at load, so a stale artifact
 * fails loudly instead of misreading a buffer layout.
 */
export interface WasmBindings {
  td_abi_version(): number;
  td_simd_enabled(): boolean;
  td_selftest_simd_parity(): boolean;
  td_alloc(count: number): number;
  td_free(ptr: number, count: number): void;
  td_world_create(
    gx: number,
    gy: number,
    gz: number,
    fixedDt: number,
    solverIterations: number,
    linearDamping: number,
    angularDamping: number,
  ): number;
  td_world_destroy(id: number): void;
  td_world_dispose(id: number): void;
  td_world_body_count(id: number): number;
  td_world_fixed_dt(id: number): number;
  td_world_step(id: number, dt: number): void;
  td_world_contact_count(id: number): number;
  td_world_drain_contacts(id: number, out: number): number;
  td_world_write_states(id: number, out: number): number;
  td_world_write_handles(id: number, out: number): number;
  td_world_raycast(
    id: number,
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    maxDistance: number,
    out: number,
  ): number;
  td_body_create(
    id: number,
    shapeKind: number,
    radius: number,
    hx: number,
    hy: number,
    hz: number,
    px: number,
    py: number,
    pz: number,
    rx: number,
    ry: number,
    rz: number,
    vx: number,
    vy: number,
    vz: number,
    dynamic: boolean,
    mass: number,
    restitution: number,
    friction: number,
    group: number,
    mask: number,
  ): number;
  td_body_destroy(id: number, handle: number): void;
  td_body_get_state(id: number, handle: number, out: number): number;
  td_body_set_state(
    id: number,
    handle: number,
    mask: number,
    px: number,
    py: number,
    pz: number,
    rx: number,
    ry: number,
    rz: number,
    vx: number,
    vy: number,
    vz: number,
    wx: number,
    wy: number,
    wz: number,
  ): void;
  td_body_apply_impulse(id: number, handle: number, x: number, y: number, z: number): void;
  td_body_apply_force(id: number, handle: number, x: number, y: number, z: number): void;
}

/**
 * Shape of the generated `--target web` glue, narrowed to what is used.
 *
 * Exported because the two initialisation paths below take it as a parameter:
 * the browser path cannot run under vitest, so the only way to test it is to
 * hand it a stub glue. That is also what keeps the platform split honest -- both
 * halves are ordinary functions over an ordinary object.
 */
export interface WasmGlue extends WasmBindings {
  initSync(module: { module: WebAssembly.Module } | BufferSource): {
    readonly memory: WebAssembly.Memory;
  };
  default?: (path?: unknown) => Promise<{ readonly memory: WebAssembly.Memory }>;
  init?: (path?: unknown) => Promise<{ readonly memory: WebAssembly.Memory }>;
}

/** An instantiated kernel: the ABI plus the linear memory the buffers live in. */
export interface WasmKernel {
  readonly bindings: WasmBindings;
  readonly memory: WebAssembly.Memory;
  /** True when the shipped binary really carries wasm SIMD128. */
  readonly simdEnabled: boolean;
}

const MODULE_URL = new URL('../../wasm/pkg/threedream_physics_wasm.js', import.meta.url);
const WASM_URL = new URL('../../wasm/pkg/threedream_physics_wasm_bg.wasm', import.meta.url);

/** True when running from a filesystem URL, i.e. Node/vitest rather than a page. */
function isNodeUrl(): boolean {
  return import.meta.url.startsWith('file:');
}

/** The `_bg.wasm` sibling of a glue URL; wasm-bindgen always emits them paired. */
export function wasmUrlFor(url: URL | string = MODULE_URL): URL {
  if (url === MODULE_URL) return WASM_URL;
  const href = typeof url === 'string' ? url : url.href;
  return new URL(href.replace(/\.js$/, '_bg.wasm'));
}

/**
 * The Node-only filesystem module, spelled so that nothing statically resolves
 * it.
 *
 * Both a string literal and a no-substitution template literal are resolvable
 * specifiers: `tsc -p tsconfig.build.json` loads no Node types and fails on
 * them, and Vite would try to bundle the module into the browser build.
 * Assembling it at runtime keeps the dependency behind `isNodeUrl()`, which is
 * the only place it can ever be reached.
 */
const FS_PROMISES = ['node:', 'fs/promises'].join('');

/**
 * Initialise the glue under Node: read the bytes and hand them to `initSync`.
 *
 * Node's `fetch` does not speak `file:`, which is the only reason this path
 * exists -- the web glue's own loader would work unchanged in a browser.
 */
export async function memoryFromNode(
  glue: WasmGlue,
  url: URL | string = MODULE_URL,
): Promise<WebAssembly.Memory> {
  const fs = (await import(/* @vite-ignore */ FS_PROMISES)) as {
    readFile(path: URL): Promise<Uint8Array>;
  };
  const bytes = await fs.readFile(wasmUrlFor(url));
  return glue.initSync({ module: bytes }).memory;
}

/** Initialise the glue in a page, letting it fetch its own wasm by URL. */
export async function memoryFromBrowser(glue: WasmGlue): Promise<WebAssembly.Memory> {
  const init = glue.default ?? glue.init;
  if (!init) throw new Error('wasm glue exposes no init function');
  return (await init()).memory;
}

/**
 * Wrap an initialised glue, refusing to run against a stale artifact.
 *
 * Checked after instantiation because every exported function traps on an
 * uninitialised module: the version gate has to come second or it reports a
 * confusing "unreachable" instead of "rebuild the wasm".
 */
export function kernelFrom(glue: WasmGlue, memory: WebAssembly.Memory): WasmKernel {
  const abi = glue.td_abi_version();
  if (abi !== WASM_ABI_VERSION) {
    throw new Error(
      `wasm ABI mismatch: artifact is v${abi}, src/physics/wasm.ts expects v${WASM_ABI_VERSION}. ` +
        'Rebuild with `npm run build:wasm`.',
    );
  }
  return { bindings: glue, memory, simdEnabled: glue.td_simd_enabled() };
}

/**
 * Instantiate the kernel.
 *
 * Cached per module instance because a second `init` of the same glue is a no-op
 * anyway, and because two `WasmPhysics` worlds in one process should share one
 * linear memory: the kernel keeps worlds in a thread-local table, so one
 * instantiation can host any number of them.
 */
let cached: Promise<WasmKernel> | undefined;

export function loadWasmKernel(url: URL | string = MODULE_URL): Promise<WasmKernel> {
  if (url === MODULE_URL && cached) return cached;
  const load = instantiate(url);
  if (url === MODULE_URL) cached = load;
  return load;
}

async function instantiate(url: URL | string): Promise<WasmKernel> {
  // A literal specifier keeps Vite able to bundle the glue for the browser
  // build; `url` is only used to decide *how* to hand it the wasm bytes.
  const glue = (await import('../../wasm/pkg/threedream_physics_wasm.js')) as unknown as WasmGlue;
  const memory = isNodeUrl()
    ? await memoryFromNode(glue, url)
    : await memoryFromBrowser(glue);
  return kernelFrom(glue, memory);
}

/**
 * A growable caller-owned buffer in wasm linear memory.
 *
 * One of these per shape of read (states, contacts, ray hit) rather than one
 * shared scratch, because `drainContacts` and `getBodyState` can legitimately be
 * interleaved by a caller and a shared buffer would let one clobber the other.
 *
 * Views are rebuilt on every access: `WebAssembly.Memory.grow` detaches the
 * previous `ArrayBuffer`, and a cached view over a detached buffer reads zeros
 * instead of throwing, which is the worst possible failure mode for a physics
 * readback.
 */
class Scratch {
  private ptr = 0;
  private count = 0;

  constructor(private readonly kernel: WasmKernel) {}

  /** Ensure room for `count` doubles; grows geometrically to avoid per-step reallocs. */
  ensure(count: number): void {
    if (count <= this.count) return;
    const next = Math.max(count, this.count * 2, 16);
    const ptr = this.kernel.bindings.td_alloc(next);
    if (ptr === 0) {
      throw new Error(`wasm kernel could not allocate ${next} doubles`);
    }
    if (this.ptr !== 0) this.kernel.bindings.td_free(this.ptr, this.count);
    this.ptr = ptr;
    this.count = next;
  }

  get offset(): number {
    return this.ptr;
  }

  f64(length: number): Float64Array {
    return new Float64Array(this.kernel.memory.buffer, this.ptr, length);
  }

  u32(length: number): Uint32Array {
    return new Uint32Array(this.kernel.memory.buffer, this.ptr, length);
  }

  free(): void {
    if (this.ptr !== 0) this.kernel.bindings.td_free(this.ptr, this.count);
    this.ptr = 0;
    this.count = 0;
  }
}

export interface WasmPhysicsOptions extends PhysicsWorldOptions {
  /** Inject an already-instantiated kernel; tests use this to avoid re-init. */
  kernel?: WasmKernel;
}

/**
 * Deterministic rigid-body backend running the Rust kernel.
 *
 * `deterministic` is true, and that claim is load-bearing: replays and reward
 * curves are only comparable if the same seed and inputs give the same numbers.
 * It holds because (a) the kernel is a phase-for-phase port of `builtin.ts` with
 * `Math.*` semantics reimplemented in `jsmath.rs`, (b) wasm SIMD is
 * deterministic by spec and `td_selftest_simd_parity()` proves the vectorised
 * passes match the scalar ones inside the shipped binary, and (c) every world
 * iterates bodies in ascending handle order, so there is no map-ordering
 * nondeterminism to leak.
 */
export class WasmPhysics implements PhysicsBackend {
  readonly name = 'wasm';
  readonly deterministic = true;
  readonly fixedDt: number;

  private readonly kernel: WasmKernel;
  private readonly bindings: WasmBindings;
  private readonly id: number;
  private readonly states: Scratch;
  private readonly contacts: Scratch;
  private readonly ray: Scratch;
  /** Labels stay on this side of the boundary; handles are the join key. */
  private readonly labels = new Map<number, string>();

  private constructor(kernel: WasmKernel, options: PhysicsWorldOptions, id: number) {
    this.kernel = kernel;
    this.bindings = kernel.bindings;
    this.id = id;
    this.fixedDt = options.fixedDt ?? 1 / 60;
    this.states = new Scratch(kernel);
    this.contacts = new Scratch(kernel);
    this.ray = new Scratch(kernel);
  }

  static async create(options: WasmPhysicsOptions = {}): Promise<WasmPhysics> {
    const kernel = options.kernel ?? (await loadWasmKernel());
    if (!kernel.bindings.td_selftest_simd_parity()) {
      throw new Error('wasm kernel failed its SIMD/scalar parity self-test');
    }
    // Defaults are resolved here rather than in Rust so they live in exactly one
    // place, next to `BuiltinPhysics`'s own; the kernel requires all five.
    const gravity = options.gravity ?? vec3(0, -9.81, 0);
    const id = kernel.bindings.td_world_create(
      gravity[0],
      gravity[1],
      gravity[2],
      options.fixedDt ?? 1 / 60,
      Math.max(1, Math.round(options.solverIterations ?? 8)),
      options.linearDamping ?? 0.05,
      options.angularDamping ?? 0.2,
    );
    if (id === 0xffffffff) throw new Error('wasm kernel ran out of world ids');
    return new WasmPhysics(kernel, options, id);
  }

  /** True when the shipped binary carries wasm SIMD128. */
  get simdEnabled(): boolean {
    return this.kernel.simdEnabled;
  }

  /**
   * The kernel-side world id, exposed for diagnostics and for tests that drive
   * the ABI directly. Nothing in `src/` needs it: the backend contract is
   * handles-in/handles-out, and leaking the id into app code would couple that
   * code to one backend.
   */
  get worldId(): number {
    return this.id;
  }

  get bodyCount(): number {
    return this.bindings.td_world_body_count(this.id);
  }

  createBody(descriptor: BodyDescriptor): number {
    if (descriptor.shape.kind === 'capsule') throw new Error(CAPSULE_MESSAGE);
    const kind = descriptor.kind ?? 'dynamic';
    const mass = kind === 'dynamic' ? (descriptor.mass ?? 1) : 0;
    if (mass < 0) throw new RangeError(NEGATIVE_MASS_MESSAGE);
    const position = descriptor.position;
    const rotation = descriptor.rotation ?? ZERO;
    const velocity = descriptor.velocity ?? ZERO;
    const shape = descriptor.shape;
    const radius = shape.kind === 'sphere' ? shape.radius : 0;
    const halfExtents = shape.kind === 'box' ? shape.halfExtents : ZERO;
    const out = this.bindings.td_body_create(
      this.id,
      shape.kind === 'sphere' ? SHAPE_SPHERE : SHAPE_BOX,
      radius,
      halfExtents[0],
      halfExtents[1],
      halfExtents[2],
      position[0],
      position[1],
      position[2],
      rotation[0],
      rotation[1],
      rotation[2],
      velocity[0],
      velocity[1],
      velocity[2],
      kind === 'dynamic',
      mass,
      descriptor.restitution ?? 0.1,
      descriptor.friction ?? 0.7,
      // `>>> 0` because the kernel takes a `u32` and JS bitwise ops are signed.
      (descriptor.group ?? 1) >>> 0,
      (descriptor.mask ?? 0xffffffff) >>> 0,
    );
    if (out === ERR_NEGATIVE_MASS) throw new RangeError(NEGATIVE_MASS_MESSAGE);
    if (out === ERR_BAD_SHAPE) throw new Error(CAPSULE_MESSAGE);
    if (out === ERR_BAD_WORLD) throw new Error(DEAD_WORLD_MESSAGE);
    if (out < 0) throw new Error(`wasm kernel rejected the body (code ${out})`);
    const handle = out;
    this.labels.set(handle, descriptor.label ?? `body:${handle}`);
    return handle;
  }

  destroyBody(handle: number): void {
    this.labels.delete(handle);
    this.bindings.td_body_destroy(this.id, handle);
  }

  getBodyState(handle: number): BodyState | undefined {
    this.states.ensure(STATE_STRIDE);
    if (this.bindings.td_body_get_state(this.id, handle, this.states.offset) === 0) {
      return undefined;
    }
    const v = this.states.f64(STATE_STRIDE);
    return {
      position: [v[0], v[1], v[2]],
      rotation: [v[3], v[4], v[5]],
      velocity: [v[6], v[7], v[8]],
      angularVelocity: [v[9], v[10], v[11]],
    };
  }

  setBodyState(handle: number, state: Partial<BodyState>): void {
    // Truthiness, not `!== undefined`, to match `builtin.ts` exactly: an absent
    // field is left alone, and `[0, 0, 0]` is a real write on both sides.
    let mask = 0;
    if (state.position) mask |= MASK_POS;
    if (state.rotation) mask |= MASK_ROT;
    if (state.velocity) mask |= MASK_VEL;
    if (state.angularVelocity) mask |= MASK_ANG;
    if (mask === 0) return;
    const p = state.position ?? ZERO;
    const r = state.rotation ?? ZERO;
    const v = state.velocity ?? ZERO;
    const w = state.angularVelocity ?? ZERO;
    this.bindings.td_body_set_state(
      this.id,
      handle,
      mask,
      p[0],
      p[1],
      p[2],
      r[0],
      r[1],
      r[2],
      v[0],
      v[1],
      v[2],
      w[0],
      w[1],
      w[2],
    );
  }

  applyImpulse(handle: number, impulse: Vec3): void {
    this.bindings.td_body_apply_impulse(this.id, handle, impulse[0], impulse[1], impulse[2]);
  }

  applyForce(handle: number, force: Vec3): void {
    this.bindings.td_body_apply_force(this.id, handle, force[0], force[1], force[2]);
  }

  step(dt: number): void {
    this.bindings.td_world_step(this.id, dt);
  }

  drainContacts(): ContactEvent[] {
    const pending = this.bindings.td_world_contact_count(this.id);
    if (pending === 0) return [];
    this.contacts.ensure(pending * CONTACT_STRIDE);
    const drained = this.bindings.td_world_drain_contacts(this.id, this.contacts.offset);
    const v = this.contacts.f64(drained * CONTACT_STRIDE);
    const out: ContactEvent[] = [];
    for (let i = 0; i < drained; i++) {
      const base = i * CONTACT_STRIDE;
      const a = v[base]!;
      const b = v[base + 1]!;
      out.push({
        a,
        b,
        normal: [v[base + 2]!, v[base + 3]!, v[base + 4]!],
        depth: v[base + 5]!,
        impulse: v[base + 6]!,
        labels: [this.labels.get(a) ?? `body:${a}`, this.labels.get(b) ?? `body:${b}`],
      });
    }
    return out;
  }

  raycast(origin: Vec3, direction: Vec3, maxDistance: number): RayHit | undefined {
    this.ray.ensure(RAY_STRIDE);
    const hit = this.bindings.td_world_raycast(
      this.id,
      origin[0],
      origin[1],
      origin[2],
      direction[0],
      direction[1],
      direction[2],
      maxDistance,
      this.ray.offset,
    );
    if (hit === 0) return undefined;
    const v = this.ray.f64(RAY_STRIDE);
    return {
      body: v[0]!,
      point: [v[1]!, v[2]!, v[3]!],
      normal: [v[4]!, v[5]!, v[6]!],
      distance: v[7]!,
    };
  }

  /**
   * `BuiltinPhysics.dispose`: empty the world, keep it usable. Options and the
   * handle counter survive, so a disposed backend still hands out fresh handles.
   */
  dispose(): void {
    this.bindings.td_world_dispose(this.id);
    this.labels.clear();
  }

  /**
   * Release the world id and every scratch buffer. Not part of
   * `PhysicsBackend` (which has no teardown stronger than `dispose`), so callers
   * that are done with an instance should use this: `dispose` leaves the world
   * allocated so it can be reused, which is correct but not free.
   */
  destroy(): void {
    this.states.free();
    this.contacts.free();
    this.ray.free();
    this.labels.clear();
    this.bindings.td_world_destroy(this.id);
  }
}

export async function createWasmPhysics(options?: WasmPhysicsOptions): Promise<WasmPhysics> {
  return WasmPhysics.create(options);
}

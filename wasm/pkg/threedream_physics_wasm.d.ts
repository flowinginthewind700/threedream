/* tslint:disable */
/* eslint-disable */

/**
 * ABI version. `src/physics/wasm.ts` checks this on init, so a stale `wasm/pkg`
 * from an older build fails at load instead of misreading a buffer layout.
 */
export function td_abi_version(): number;

/**
 * Allocate `count` 8-byte-aligned `f64`s and return the byte offset.
 *
 * `count == 0` returns `ERR_ALLOC` rather than a dangling-but-valid pointer:
 * a zero-sized scratch buffer is always a caller bug, and silently handing back
 * an offset that aliases the next allocation would corrupt it.
 */
export function td_alloc(count: number): number;

export function td_body_apply_force(id: number, handle: number, x: number, y: number, z: number): void;

export function td_body_apply_impulse(id: number, handle: number, x: number, y: number, z: number): void;

/**
 * Create a body. Returns its handle, or a negative sentinel from the `ERR_*`
 * constants above.
 *
 * `dynamic` arrives as a `bool` because the TS adapter has already collapsed
 * `kind === 'dynamic'`; `kinematic` is treated as static, exactly as
 * `BuiltinPhysics` does (its `mass` computation only special-cases `'dynamic'`).
 */
export function td_body_create(id: number, shape_kind: number, radius: number, hx: number, hy: number, hz: number, px: number, py: number, pz: number, rx: number, ry: number, rz: number, vx: number, vy: number, vz: number, dynamic: boolean, mass: number, restitution: number, friction: number, group: number, mask: number): number;

export function td_body_destroy(id: number, handle: number): void;

/**
 * Write 12 doubles for one body. Returns `1` if the handle exists, `0` if not
 * (and writes nothing).
 */
export function td_body_get_state(id: number, handle: number, out: number): number;

/**
 * `setBodyState`. `mask` is the `MASK_*` bitset: only the triples whose bit is
 * set are read, but all twelve arguments are always passed, because a varargs
 * ABI would cost more in JS marshalling than the unused registers do.
 */
export function td_body_set_state(id: number, handle: number, mask: number, px: number, py: number, pz: number, rx: number, ry: number, rz: number, vx: number, vy: number, vz: number, wx: number, wy: number, wz: number): void;

/**
 * Free a `td_alloc` buffer. `count` must be the value it was allocated with,
 * because `Layout` has to match exactly for `dealloc` to be sound.
 */
export function td_free(ptr: number, count: number): void;

/**
 * Run the vectorised integration passes against scalar references inside wasm.
 *
 * This is the only test in the repo that can prove the *shipped* artifact's SIMD
 * path is bit-identical to its scalar path, because it executes in that
 * artifact. `tests/wasm_backend.test.ts` calls it and fails the build on false.
 */
export function td_selftest_simd_parity(): boolean;

/**
 * Whether the shipped binary really carries wasm SIMD128.
 */
export function td_simd_enabled(): boolean;

/**
 * `PhysicsBackend.bodyCount`.
 */
export function td_world_body_count(id: number): number;

/**
 * Contacts produced by the most recent step, and destructive: calling this
 * empties the kernel's list, matching `PhysicsBackend.drainContacts`.
 */
export function td_world_contact_count(id: number): number;

/**
 * Create a world. All five options are required; the TS adapter resolves
 * `options.x ?? default` before calling, so the defaults live in exactly one
 * place (`src/physics/types.ts`) instead of being duplicated here.
 */
export function td_world_create(gx: number, gy: number, gz: number, fixed_dt: number, solver_iterations: number, linear_damping: number, angular_damping: number): number;

/**
 * Drop a world and free its id. Destroying an unknown id is a no-op.
 */
export function td_world_destroy(id: number): void;

/**
 * `PhysicsBackend.dispose`. Keeps the world id valid but empties it, so an
 * adapter that disposes and reuses does not have to re-create the id.
 */
export function td_world_dispose(id: number): void;

/**
 * Drain contacts into `out` as 7 doubles each. Returns the number drained.
 *
 * Draining rather than peeking is what keeps the TS adapter's
 * `drainContacts()` semantics exact: a second call in the same frame returns an
 * empty array on both backends.
 */
export function td_world_drain_contacts(id: number, out: number): number;

export function td_world_fixed_dt(id: number): number;

/**
 * Raycast. Returns `1` and writes 8 doubles on a hit, `0` on a miss or a
 * degenerate direction (writing nothing).
 */
export function td_world_raycast(id: number, ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, max_distance: number, out: number): number;

/**
 * Advance one fixed step. Contacts from this step stay readable until the next
 * one, which is what makes `step()` then `drainContacts()` work across a JS
 * turn boundary.
 */
export function td_world_step(id: number, dt: number): void;

/**
 * Handles matching `td_world_write_states`, one `u32` per body, same order.
 */
export function td_world_write_handles(id: number, out: number): number;

/**
 * Bulk snapshot: 12 doubles per body, in ascending handle order.
 *
 * Exists so the renderer and the training loop can read a whole world in one
 * call. `td_body_get_state` is one JS->wasm transition per body; this is one
 * transition per step, which at 500 bodies is the difference between a
 * measurable and an unmeasurable overhead.
 */
export function td_world_write_states(id: number, out: number): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly td_abi_version: () => number;
    readonly td_alloc: (a: number) => number;
    readonly td_body_apply_force: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly td_body_apply_impulse: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly td_body_create: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number) => number;
    readonly td_body_destroy: (a: number, b: number) => void;
    readonly td_body_get_state: (a: number, b: number, c: number) => number;
    readonly td_body_set_state: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => void;
    readonly td_free: (a: number, b: number) => void;
    readonly td_selftest_simd_parity: () => number;
    readonly td_simd_enabled: () => number;
    readonly td_world_body_count: (a: number) => number;
    readonly td_world_contact_count: (a: number) => number;
    readonly td_world_create: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly td_world_destroy: (a: number) => void;
    readonly td_world_dispose: (a: number) => void;
    readonly td_world_drain_contacts: (a: number, b: number) => number;
    readonly td_world_fixed_dt: (a: number) => number;
    readonly td_world_raycast: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => number;
    readonly td_world_step: (a: number, b: number) => void;
    readonly td_world_write_handles: (a: number, b: number) => number;
    readonly td_world_write_states: (a: number, b: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

//! The wasm ABI: numeric world ids, numeric body handles, and raw `f64`
//! buffers. See `Cargo.toml` for why nothing else crosses.
//!
//! # Layout contracts
//!
//! `src/physics/wasm.ts` is the only caller and the only place these layouts are
//! restated. They are written down here because a mismatch is silent: writing
//! twelve doubles where the reader expects thirteen does not fail, it just
//! produces a body whose rotation is its own velocity.
//!
//! * **State** (`td_world_write_states`, `td_body_get_state`): 12 `f64` per body
//!   in ascending *handle* order -- `px py pz rx ry rz vx vy vz wx wy wz`.
//! * **Handles** (`td_world_write_handles`): one `u32` per body, same order, so
//!   the TS side can zip them against the state buffer.
//! * **Contacts** (`td_world_drain_contacts`): 7 `f64` per contact --
//!   `a b nx ny nz depth impulse`, with `a`/`b` as handles widened to `f64`.
//! * **Ray hit** (`td_world_raycast`): 8 `f64` -- `body px py pz nx ny nz distance`.
//!
//! # Memory ownership
//!
//! Every out-buffer is allocated by the *caller*, through `td_alloc`, and freed
//! by the caller through `td_free`. wasm-bindgen exports no allocator of its own
//! for this crate (there are no strings or `Vec`s crossing), so without these two
//! functions JS would have no way to hand the kernel writable linear memory.
//! Passing a `u32` byte offset rather than a Rust pointer type keeps the whole
//! ABI numeric, which is what `Cargo.toml` promises.
//!
//! # Error signalling
//!
//! There is no `Result` across this boundary, because a wasm `Result` becomes a
//! JS exception carrying a string that has to be allocated and freed. Instead:
//! functions returning a handle use negative sentinels, and predicates return
//! `0`/`1`. `src/physics/wasm.ts` translates the sentinels into the same
//! `RangeError`/`Error` objects `BuiltinPhysics` throws, so a caller cannot tell
//! the two backends apart by how they fail.
//!
//! An invalid *world id* is a programming error in the adapter, not a runtime
//! condition, so those paths return a sentinel rather than panicking: a panic
//! under `panic = "abort"` traps the instance and every later call fails
//! confusingly, which is a much worse debugging experience than a `-2`.

// The ABI crate has no arithmetic of its own, so there is nothing here that
// could break bit-exactness. Keeping it `unsafe`-free except for the four
// pointer writes is what makes that claim auditable in one file.
#![deny(unsafe_code)]

use std::cell::RefCell;

use threedream_physics::{
    simd_parity_ok, BodyDescriptor, CreateError, Shape, World, WorldOptions, ABI_VERSION, SIMD128,
};
use wasm_bindgen::prelude::*;

/// Sentinels for `td_body_create`, which returns a handle as `f64`.
pub const ERR_BAD_WORLD: f64 = -2.0;
pub const ERR_NEGATIVE_MASS: f64 = -1.0;
/// Sentinel for `td_world_create` when the id space is exhausted. Cannot happen
/// in practice; present so the return type has no "success" ambiguity.
pub const ERR_NO_WORLD: u32 = u32::MAX;
/// Returned by `td_alloc` when the request cannot be satisfied. A `0` is never a
/// valid allocation on wasm32 (the first page holds the null guard), so callers
/// can branch on it without a separate error channel.
pub const ERR_ALLOC: u32 = 0;

/// Shape kinds, as agreed with `src/physics/wasm.ts`. `capsule` is deliberately
/// absent: `BuiltinPhysics.createBody` throws for it, and the wasm backend
/// throws the identical error in TS before ever reaching this crate.
pub const SHAPE_SPHERE: u32 = 0;
pub const SHAPE_BOX: u32 = 1;
pub const ERR_BAD_SHAPE: f64 = -3.0;

thread_local! {
    /// Live worlds, indexed by id. `None` slots are recycled ids, which is safe
    /// because an id is only reused after a `td_world_destroy` -- the same
    /// lifetime contract a JS object handle has.
    static WORLDS: RefCell<Vec<Option<World>>> = const { RefCell::new(Vec::new()) };
}

/// Allocate `count` 8-byte-aligned `f64`s and return the byte offset.
///
/// `count == 0` returns `ERR_ALLOC` rather than a dangling-but-valid pointer:
/// a zero-sized scratch buffer is always a caller bug, and silently handing back
/// an offset that aliases the next allocation would corrupt it.
#[wasm_bindgen]
#[allow(unsafe_code)]
pub fn td_alloc(count: u32) -> u32 {
    if count == 0 {
        return ERR_ALLOC;
    }
    let layout = match std::alloc::Layout::array::<f64>(count as usize) {
        Ok(layout) => layout,
        Err(_) => return ERR_ALLOC,
    };
    // SAFETY: `Layout::array` returns an align-8, non-zero-size layout for any
    // `count > 0` that fits in `isize`, and `alloc` requires exactly that. The
    // returned pointer is handed to JS as an offset and stays live until the
    // matching `td_free`; nothing in this crate dereferences it.
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return ERR_ALLOC;
    }
    ptr as u32
}

/// Free a `td_alloc` buffer. `count` must be the value it was allocated with,
/// because `Layout` has to match exactly for `dealloc` to be sound.
#[wasm_bindgen]
#[allow(unsafe_code)]
pub fn td_free(ptr: u32, count: u32) {
    if ptr == ERR_ALLOC || count == 0 {
        return;
    }
    let Ok(layout) = std::alloc::Layout::array::<f64>(count as usize) else {
        return;
    };
    // SAFETY: `ptr`/`count` come from a matching `td_alloc` call -- that pairing
    // is the documented contract with `src/physics/wasm.ts`, which stores both
    // in one object and frees it exactly once, from `dispose()`.
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
}

fn with_world<R>(id: u32, f: impl FnOnce(&World) -> R) -> Option<R> {
    WORLDS.with(|w| w.borrow().get(id as usize).and_then(|slot| slot.as_ref()).map(f))
}

fn with_world_mut<R>(id: u32, f: impl FnOnce(&mut World) -> R) -> Option<R> {
    WORLDS.with(|w| {
        let mut worlds = w.borrow_mut();
        match worlds.get_mut(id as usize).and_then(|slot| slot.as_mut()) {
            Some(world) => Some(f(world)),
            None => None,
        }
    })
}

/// ABI version. `src/physics/wasm.ts` checks this on init, so a stale `wasm/pkg`
/// from an older build fails at load instead of misreading a buffer layout.
#[wasm_bindgen]
pub fn td_abi_version() -> u32 {
    ABI_VERSION
}

/// Whether the shipped binary really carries wasm SIMD128.
#[wasm_bindgen]
pub fn td_simd_enabled() -> bool {
    SIMD128
}

/// Run the vectorised integration passes against scalar references inside wasm.
///
/// This is the only test in the repo that can prove the *shipped* artifact's SIMD
/// path is bit-identical to its scalar path, because it executes in that
/// artifact. `tests/wasm_backend.test.ts` calls it and fails the build on false.
#[wasm_bindgen]
pub fn td_selftest_simd_parity() -> bool {
    simd_parity_ok()
}

/// Create a world. All five options are required; the TS adapter resolves
/// `options.x ?? default` before calling, so the defaults live in exactly one
/// place (`src/physics/types.ts`) instead of being duplicated here.
#[wasm_bindgen]
#[allow(unsafe_code)]
pub fn td_world_create(
    gx: f64,
    gy: f64,
    gz: f64,
    fixed_dt: f64,
    solver_iterations: u32,
    linear_damping: f64,
    angular_damping: f64,
) -> u32 {
    let world = World::new(WorldOptions {
        gravity: [gx, gy, gz],
        fixed_dt,
        solver_iterations: solver_iterations as usize,
        linear_damping,
        angular_damping,
    });
    WORLDS.with(|w| {
        let mut worlds = w.borrow_mut();
        match worlds.iter_mut().position(|slot| slot.is_none()) {
            Some(id) => {
                worlds[id] = Some(world);
                id as u32
            }
            None => {
                if worlds.len() as u64 >= u64::from(u32::MAX) {
                    return ERR_NO_WORLD;
                }
                worlds.push(Some(world));
                (worlds.len() - 1) as u32
            }
        }
    })
}

/// Drop a world and free its id. Destroying an unknown id is a no-op.
#[wasm_bindgen]
pub fn td_world_destroy(id: u32) {
    WORLDS.with(|w| {
        if let Some(slot) = w.borrow_mut().get_mut(id as usize) {
            *slot = None;
        }
    });
}

/// `PhysicsBackend.bodyCount`.
#[wasm_bindgen]
pub fn td_world_body_count(id: u32) -> u32 {
    with_world(id, |w| w.body_count() as u32).unwrap_or(0)
}

#[wasm_bindgen]
pub fn td_world_fixed_dt(id: u32) -> f64 {
    with_world(id, |w| w.fixed_dt()).unwrap_or(f64::NAN)
}

/// Create a body. Returns its handle, or a negative sentinel from the `ERR_*`
/// constants above.
///
/// `dynamic` arrives as a `bool` because the TS adapter has already collapsed
/// `kind === 'dynamic'`; `kinematic` is treated as static, exactly as
/// `BuiltinPhysics` does (its `mass` computation only special-cases `'dynamic'`).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn td_body_create(
    id: u32,
    shape_kind: u32,
    radius: f64,
    hx: f64,
    hy: f64,
    hz: f64,
    px: f64,
    py: f64,
    pz: f64,
    rx: f64,
    ry: f64,
    rz: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    dynamic: bool,
    mass: f64,
    restitution: f64,
    friction: f64,
    group: u32,
    mask: u32,
) -> f64 {
    let shape = match shape_kind {
        SHAPE_SPHERE => Shape::sphere(radius),
        SHAPE_BOX => Shape::aabb_box([hx, hy, hz]),
        _ => return ERR_BAD_SHAPE,
    };
    let descriptor = BodyDescriptor {
        shape,
        position: [px, py, pz],
        rotation: [rx, ry, rz],
        velocity: [vx, vy, vz],
        dynamic,
        mass,
        restitution,
        friction,
        group,
        mask,
        // Labels never cross the boundary; the TS adapter keeps the map.
        label: None,
    };
    match with_world_mut(id, |w| w.create_body(descriptor)) {
        None => ERR_BAD_WORLD,
        Some(Ok(handle)) => f64::from(handle),
        Some(Err(CreateError::NegativeMass)) => ERR_NEGATIVE_MASS,
    }
}

#[wasm_bindgen]
pub fn td_body_destroy(id: u32, handle: u32) {
    if let Some(()) = with_world_mut(id, |w| w.destroy_body(handle)) {}
}

/// Write 12 doubles for one body. Returns `1` if the handle exists, `0` if not
/// (and writes nothing).
#[wasm_bindgen]
#[allow(unsafe_code)]
pub fn td_body_get_state(id: u32, handle: u32, out: *mut f64) -> u32 {
    // SAFETY: the contract with `src/physics/wasm.ts` is that `out` points at 12
    // contiguous, 8-byte-aligned `f64`s inside wasm linear memory that the
    // caller allocated and still owns. Nothing here can be made memory-safe by
    // the callee alone; the adapter is the only caller and it allocates from a
    // single growable scratch buffer, which is what keeps this checkable.
    // `with_world` already returns an `Option` for the world id, and
    // `get_body_state` returns one for the handle, so both have to be unwrapped.
    let Some(Some(state)) = with_world(id, |w| w.get_body_state(handle)) else {
        return 0;
    };
    let fields = [
        state.position,
        state.rotation,
        state.velocity,
        state.angular_velocity,
    ];
    unsafe {
        for (i, triple) in fields.iter().enumerate() {
            for (k, v) in triple.iter().enumerate() {
                out.add(i * 3 + k).write(*v);
            }
        }
    }
    1
}

/// `setBodyState`. `mask` is the `MASK_*` bitset: only the triples whose bit is
/// set are read, but all twelve arguments are always passed, because a varargs
/// ABI would cost more in JS marshalling than the unused registers do.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn td_body_set_state(
    id: u32,
    handle: u32,
    mask: u32,
    px: f64,
    py: f64,
    pz: f64,
    rx: f64,
    ry: f64,
    rz: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    wx: f64,
    wy: f64,
    wz: f64,
) {
    if let Some(()) = with_world_mut(id, |w| {
        w.set_body_state(
            handle,
            mask,
            [px, py, pz],
            [rx, ry, rz],
            [vx, vy, vz],
            [wx, wy, wz],
        )
    }) {}
}

#[wasm_bindgen]
pub fn td_body_apply_impulse(id: u32, handle: u32, x: f64, y: f64, z: f64) {
    if let Some(()) = with_world_mut(id, |w| w.apply_impulse(handle, [x, y, z])) {}
}

#[wasm_bindgen]
pub fn td_body_apply_force(id: u32, handle: u32, x: f64, y: f64, z: f64) {
    if let Some(()) = with_world_mut(id, |w| w.apply_force(handle, [x, y, z])) {}
}

/// Advance one fixed step. Contacts from this step stay readable until the next
/// one, which is what makes `step()` then `drainContacts()` work across a JS
/// turn boundary.
#[wasm_bindgen]
pub fn td_world_step(id: u32, dt: f64) {
    if let Some(()) = with_world_mut(id, |w| w.step(dt)) {}
}

/// Contacts produced by the most recent step, and destructive: calling this
/// empties the kernel's list, matching `PhysicsBackend.drainContacts`.
#[wasm_bindgen]
pub fn td_world_contact_count(id: u32) -> u32 {
    with_world(id, |w| w.contact_count() as u32).unwrap_or(0)
}

/// Drain contacts into `out` as 7 doubles each. Returns the number drained.
///
/// Draining rather than peeking is what keeps the TS adapter's
/// `drainContacts()` semantics exact: a second call in the same frame returns an
/// empty array on both backends.
#[wasm_bindgen]
#[allow(unsafe_code)]
pub fn td_world_drain_contacts(id: u32, out: *mut f64) -> u32 {
    // SAFETY: as `td_body_get_state` -- `out` addresses 7 * `f64` per contact of
    // caller-owned, 8-byte-aligned linear memory. The count the caller
    // allocated from is `td_world_contact_count`, and nothing between that call
    // and this one can add a contact (only `td_world_step` does, and the adapter
    // never interleaves them).
    let Some(contacts) = with_world_mut(id, |w| w.drain_contacts()) else {
        return 0;
    };
    let n = contacts.len();
    unsafe {
        for (i, c) in contacts.iter().enumerate() {
            let base = out.add(i * 7);
            base.write(f64::from(c.a));
            base.add(1).write(f64::from(c.b));
            base.add(2).write(c.normal[0]);
            base.add(3).write(c.normal[1]);
            base.add(4).write(c.normal[2]);
            base.add(5).write(c.depth);
            base.add(6).write(c.impulse);
        }
    }
    n as u32
}

/// Bulk snapshot: 12 doubles per body, in ascending handle order.
///
/// Exists so the renderer and the training loop can read a whole world in one
/// call. `td_body_get_state` is one JS->wasm transition per body; this is one
/// transition per step, which at 500 bodies is the difference between a
/// measurable and an unmeasurable overhead.
#[wasm_bindgen]
#[allow(unsafe_code)]
pub fn td_world_write_states(id: u32, out: *mut f64) -> u32 {
    // SAFETY: as above, `out` addresses 12 * `body_count` caller-owned doubles.
    let Some(snapshot) = with_world(id, |w| w.snapshot()) else { return 0 };
    let n = snapshot.len();
    unsafe {
        for (i, v) in snapshot.iter().enumerate() {
            out.add(i).write(*v);
        }
    }
    (n / 12) as u32
}

/// Handles matching `td_world_write_states`, one `u32` per body, same order.
#[wasm_bindgen]
#[allow(unsafe_code)]
pub fn td_world_write_handles(id: u32, out: *mut u32) -> u32 {
    // SAFETY: `out` addresses `body_count` caller-owned `u32`s.
    let Some(handles) = with_world(id, |w| w.handles()) else { return 0 };
    let n = handles.len();
    unsafe {
        for (i, h) in handles.iter().enumerate() {
            out.add(i).write(*h);
        }
    }
    n as u32
}

/// Raycast. Returns `1` and writes 8 doubles on a hit, `0` on a miss or a
/// degenerate direction (writing nothing).
#[wasm_bindgen]
#[allow(unsafe_code)]
pub fn td_world_raycast(
    id: u32,
    ox: f64,
    oy: f64,
    oz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    max_distance: f64,
    out: *mut f64,
) -> u32 {
    // SAFETY: `out` addresses 8 caller-owned doubles.
    let Some(hit) = with_world(id, |w| {
        w.raycast([ox, oy, oz], [dx, dy, dz], max_distance)
    }) else {
        return 0;
    };
    let Some(hit) = hit else { return 0 };
    unsafe {
        out.write(f64::from(hit.body));
        out.add(1).write(hit.point[0]);
        out.add(2).write(hit.point[1]);
        out.add(3).write(hit.point[2]);
        out.add(4).write(hit.normal[0]);
        out.add(5).write(hit.normal[1]);
        out.add(6).write(hit.normal[2]);
        out.add(7).write(hit.distance);
    }
    1
}

/// `PhysicsBackend.dispose`. Keeps the world id valid but empties it, so an
/// adapter that disposes and reuses does not have to re-create the id.
#[wasm_bindgen]
pub fn td_world_dispose(id: u32) {
    if let Some(()) = with_world_mut(id, |w| w.dispose()) {}
}

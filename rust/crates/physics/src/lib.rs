//! The deterministic rigid-body kernel: a Rust port of `src/physics/builtin.ts`.
//!
//! M1's acceptance bar is not "a physics engine that behaves similarly to the
//! TypeScript one". It is *bit-identical doubles*, so that swapping
//! `BuiltinPhysics` for `WasmPhysicsBackend` cannot move a training curve, a
//! replay, or a reward threshold. Everything in this crate is organised around
//! that constraint, which is why it looks the way it does:
//!
//! * [`jsmath`] re-implements the handful of JS arithmetic semantics Rust
//!   genuinely differs on (`Math.min`/`Math.max` NaN and signed-zero handling,
//!   and reciprocal-multiply instead of divide in `normalizeVec3`).
//! * [`simd`] is restricted to lane-exact `f64x2` in the *elementwise* passes
//!   only. The contact solver stays scalar because it is a sequential
//!   Gauss-Seidel chain and reordering it changes results.
//! * [`body`] stores state component-major so those elementwise passes are flat
//!   loops, while [`crate::body::BodyMeta`] keeps the read-only solver inputs in
//!   one small record per body.
//! * [`narrowphase`], [`solve`] and [`raycast`] are operation-for-operation
//!   ports, short-circuit order included.
//!
//! [`World`] is the only stateful type and the only thing the wasm ABI exposes.
//! Its `step` reproduces `BuiltinPhysics.step` phase for phase, including the
//! three points at which AABBs are refreshed -- those are observable, because a
//! box raycast reads the cached bounds rather than recomputing them.
//!
//! # Identity model
//!
//! Two index spaces, and keeping them straight is the main invariant here:
//!
//! * **Handles** (`u32`, from 1, never reused) are what callers see. They are the
//!   sort key for every ordered traversal, which is the determinism guarantee.
//! * **Slots** (`usize`, dense, swap-removed) are what the solver and the SIMD
//!   passes see, because a hole in the middle of a `Vec<f64>` channel would force
//!   a branch into the innermost loop.
//!
//! [`crate::narrowphase::Contact`] carries slots; [`ContactEvent`] carries
//! handles. `World::step` is the single place the translation happens.
//!
//! # What is deliberately *not* here
//!
//! Capsule support and the `descriptor.x ?? default` resolution both live at the
//! TypeScript boundary (`src/physics/wasm.ts`), not in the kernel. `builtin.ts`
//! rejects capsules with a thrown `Error` and rejects negative mass with a
//! `RangeError`; the wasm backend has to fail the same way, and the cheapest way
//! to guarantee that is to keep those checks in the one file that already owns
//! the other ABI details. [`CreateError::NegativeMass`] is still enforced here as
//! well, because a kernel that silently accepts a negative inverse mass would be
//! a much worse failure than a duplicate check.

// The only `unsafe` in this crate is the `v128` intrinsic wrapper in
// `crate::simd`, which is cfg-gated to `wasm32 + simd128` and carries its own
// `#[allow]`. Denying it everywhere else keeps that claim checkable by the
// compiler instead of by review.
#![deny(unsafe_code)]

pub mod body;
pub mod integrate;
pub mod jsmath;
pub mod narrowphase;
pub mod raycast;
pub mod simd;
pub mod solve;

use std::collections::HashMap;

// Public re-export: these are the vocabulary of the kernel's API (`Shape` and
// `ShapeKind` appear in `BodyDescriptor`, the `MASK_*` bits in `set_body_state`,
// `Aabb` in `aabb_of`), so a private import here would make them unnameable
// from `threedream-physics-wasm` even though they are `pub` in `crate::body`.
pub use body::{
    Aabb, BodyMeta, Shape, ShapeKind, StateArrays, MASK_ANG, MASK_POS, MASK_ROT, MASK_VEL, PX, RX,
    VX, WX,
};
use integrate::{integrate_forces, integrate_velocities};
use jsmath::{add, scale};
use narrowphase::narrowphase_pair;
pub use narrowphase::Contact;
use raycast::{prepare_dir, ray_vs_body};
use solve::{positional_correction, solve_contact};

pub use integrate::simd_parity_ok;
pub use raycast::RayHitGeom;
pub use simd::SIMD128;

/// Version of the wasm ABI this kernel implements.
///
/// Exported as `td_abi_version()` and checked by `src/physics/wasm.ts` on init,
/// so a stale `wasm/pkg` artifact from an older build fails loudly at load
/// instead of mis-reading a struct layout nobody versioned.
pub const ABI_VERSION: u32 = 1;

/// Why `World::create_body` refused.
///
/// Deliberately one variant. The wasm layer turns it into a `RangeError` with
/// the same message `builtin.ts` throws; adding variants here means adding
/// messages there, and the two lists have to stay in step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreateError {
    /// `mass < 0`. Mirrors `builtin.ts`'s `RangeError('mass must be non-negative')`.
    NegativeMass,
}

/// `PhysicsWorldOptions`, with the defaults already applied.
///
/// The TS side resolves `options.x ?? default` and passes concrete numbers, so
/// the defaults exist here only for native tests and for `World::default()`. They
/// must match `src/physics/types.ts` exactly; `tests/wasm_backend.test.ts`
/// compares the two by stepping a default-constructed world on both backends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldOptions {
    pub gravity: [f64; 3],
    pub fixed_dt: f64,
    pub solver_iterations: usize,
    pub linear_damping: f64,
    pub angular_damping: f64,
}

impl Default for WorldOptions {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0],
            fixed_dt: 1.0 / 60.0,
            solver_iterations: 8,
            linear_damping: 0.05,
            angular_damping: 0.2,
        }
    }
}

/// `BodyDescriptor` with every `??` already resolved by the caller.
///
/// `dynamic` is the resolved `kind === 'dynamic'` test and `mass` the resolved
/// `kind === 'dynamic' ? (descriptor.mass ?? 1) : 0`. Pushing that resolution to
/// the TS boundary keeps the kernel free of option-defaulting logic that would
/// otherwise have to be duplicated -- and kept in sync -- on both sides.
#[derive(Clone, Debug)]
pub struct BodyDescriptor {
    pub shape: Shape,
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub velocity: [f64; 3],
    pub dynamic: bool,
    /// Only meaningful when `dynamic`; a static body is created with mass 0.
    pub mass: f64,
    pub restitution: f64,
    pub friction: f64,
    pub group: u32,
    pub mask: u32,
    /// `None` becomes `body:{handle}`, which can only be formatted here because
    /// the handle is assigned here.
    pub label: Option<String>,
}

/// `BodyState`: the four triples a caller can read or write.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyState {
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
}

/// `ContactEvent`, minus the `labels` field.
///
/// Labels are strings, and strings are the one thing this crate should not be
/// pushing across the wasm boundary: it would mean allocating in linear memory
/// and handing out lengths for the TS side to decode, on every contact, every
/// step. The TS backend already keeps a `handle -> label` map for exactly this
/// reason, so the kernel reports handles and the caller decorates them.
///
/// `impulse` is always `0.0`, matching `builtin.ts`, which also emits a literal
/// zero. It is kept in the struct so the two backends produce the same
/// `ContactEvent` shape rather than one that has to be patched up afterwards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactEvent {
    pub a: u32,
    pub b: u32,
    pub normal: [f64; 3],
    pub depth: f64,
    pub impulse: f64,
}

/// `RayHit`: a hit plus the handle of the body that was hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub body: u32,
    pub point: [f64; 3],
    pub normal: [f64; 3],
    pub distance: f64,
}

/// The world: bodies, their state, and the contacts the last step produced.
///
/// Every per-body collection is indexed by **slot** and kept the same length:
/// `metas`, `state`, `aabbs`, `labels` and `dynamic_mask` all have `body_count()`
/// entries, and `slot_of` maps a handle back to its slot. `create_body` pushes to
/// all of them; `destroy_body` swap-removes from all of them and repairs the one
/// `slot_of` entry the swap displaced. That repair is the easy thing to forget,
/// and forgetting it turns into a body whose state is written to the wrong slot
/// -- hence the explicit test below.
#[derive(Clone, Debug)]
pub struct World {
    gravity: [f64; 3],
    fixed_dt: f64,
    iterations: usize,
    linear_damping: f64,
    angular_damping: f64,

    metas: Vec<BodyMeta>,
    state: StateArrays,
    aabbs: Vec<Aabb>,
    labels: Vec<String>,
    /// `-1` for dynamic bodies, `0` for static, as an all-ones/all-zeros lane
    /// mask for `v128_bitselect`. Rebuilt on create/destroy, never on step.
    dynamic_mask: Vec<i64>,
    slot_of: HashMap<u32, usize>,
    next_handle: u32,

    /// Slots in ascending handle order, recomputed at the top of each step.
    /// Cached on the struct rather than allocated per step so a hot loop does not
    /// churn; `step` takes it with `mem::take` to keep the borrows disjoint.
    order: Vec<usize>,
    contacts: Vec<ContactEvent>,
}

impl Default for World {
    fn default() -> Self {
        Self::new(WorldOptions::default())
    }
}

impl World {
    pub fn new(options: WorldOptions) -> Self {
        Self {
            gravity: options.gravity,
            fixed_dt: options.fixed_dt,
            iterations: options.solver_iterations,
            linear_damping: options.linear_damping,
            angular_damping: options.angular_damping,
            metas: Vec::new(),
            state: StateArrays::new(),
            aabbs: Vec::new(),
            labels: Vec::new(),
            dynamic_mask: Vec::new(),
            slot_of: HashMap::new(),
            next_handle: 1,
            order: Vec::new(),
            contacts: Vec::new(),
        }
    }

    /// Number of live bodies. `BuiltinPhysics.bodyCount`.
    #[inline]
    pub fn body_count(&self) -> usize {
        self.metas.len()
    }

    #[inline]
    pub fn fixed_dt(&self) -> f64 {
        self.fixed_dt
    }

    /// Whether *this binary* carries wasm SIMD128. False on a native build.
    #[inline]
    pub fn simd_enabled(&self) -> bool {
        SIMD128
    }

    /// `BuiltinPhysics.createBody`, minus the capsule rejection (TS-side).
    pub fn create_body(&mut self, descriptor: BodyDescriptor) -> Result<u32, CreateError> {
        let mass = if descriptor.dynamic { descriptor.mass } else { 0.0 };
        if mass < 0.0 {
            return Err(CreateError::NegativeMass);
        }
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };

        let handle = self.next_handle;
        self.next_handle += 1;
        let slot = self.metas.len();

        self.metas.push(BodyMeta {
            handle,
            shape: descriptor.shape,
            inv_mass,
            restitution: descriptor.restitution,
            friction: descriptor.friction,
            group: descriptor.group,
            mask: descriptor.mask,
        });
        self.state.resize(slot + 1);
        self.state.set(slot, PX, descriptor.position);
        self.state.set(slot, RX, descriptor.rotation);
        self.state.set(slot, VX, descriptor.velocity);
        // Angular velocity starts at zero, as in `builtin.ts`; `resize` already
        // zero-filled the channel, and there is no descriptor field for it.
        self.aabbs.push(Aabb::around(descriptor.position, &descriptor.shape));
        self.labels.push(descriptor.label.unwrap_or_else(|| format!("body:{handle}")));
        self.dynamic_mask.push(if inv_mass != 0.0 { -1 } else { 0 });
        self.slot_of.insert(handle, slot);

        Ok(handle)
    }

    /// `BuiltinPhysics.destroyBody`. Deleting an unknown handle is a no-op.
    pub fn destroy_body(&mut self, handle: u32) {
        let Some(slot) = self.slot_of.remove(&handle) else { return };
        let last = self.metas.len() - 1;
        if slot != last {
            self.state.swap(slot, last);
            self.metas.swap(slot, last);
            self.aabbs.swap(slot, last);
            self.labels.swap(slot, last);
            self.dynamic_mask.swap(slot, last);
            // The body that just moved into `slot` has a different handle.
            let moved = self.metas[slot].handle;
            self.slot_of.insert(moved, slot);
        }
        self.state.pop();
        self.metas.pop();
        self.aabbs.pop();
        self.labels.pop();
        self.dynamic_mask.pop();
    }

    /// `BuiltinPhysics.getBodyState`; `None` for an unknown handle.
    pub fn get_body_state(&self, handle: u32) -> Option<BodyState> {
        let slot = *self.slot_of.get(&handle)?;
        Some(BodyState {
            position: self.state.get(slot, PX),
            rotation: self.state.get(slot, RX),
            velocity: self.state.get(slot, VX),
            angular_velocity: self.state.get(slot, WX),
        })
    }

    /// `BuiltinPhysics.setBodyState`, with the `Partial<BodyState>` expressed as
    /// a mask. All four triples are always passed; `mask` decides which are read.
    ///
    /// The AABB refresh at the end is unconditional, exactly as in the TS
    /// original: writing only a velocity still recomputes the bounds, which is
    /// wasteful but observable through `raycast` against a box, so it stays.
    pub fn set_body_state(
        &mut self,
        handle: u32,
        mask: u32,
        position: [f64; 3],
        rotation: [f64; 3],
        velocity: [f64; 3],
        angular_velocity: [f64; 3],
    ) {
        let Some(slot) = self.slot_of.get(&handle).copied() else { return };
        if mask & MASK_POS != 0 {
            self.state.set(slot, PX, position);
        }
        if mask & MASK_ROT != 0 {
            self.state.set(slot, RX, rotation);
        }
        if mask & MASK_VEL != 0 {
            self.state.set(slot, VX, velocity);
        }
        if mask & MASK_ANG != 0 {
            self.state.set(slot, WX, angular_velocity);
        }
        self.refresh_aabb(slot);
    }

    /// `BuiltinPhysics.applyImpulse`: `vel += impulse * invMass`.
    pub fn apply_impulse(&mut self, handle: u32, impulse: [f64; 3]) {
        let Some(slot) = self.slot_of.get(&handle).copied() else { return };
        let inv_mass = self.metas[slot].inv_mass;
        if inv_mass == 0.0 {
            return;
        }
        let vel = add(self.state.get(slot, VX), scale(impulse, inv_mass));
        self.state.set(slot, VX, vel);
    }

    /// `BuiltinPhysics.applyForce`: `vel += force * (invMass * fixedDt)`.
    ///
    /// Uses the world's `fixedDt`, not a `dt` argument -- that is what the TS
    /// original does, and it is the reason a force applied between steps behaves
    /// identically on both backends regardless of the frame time.
    pub fn apply_force(&mut self, handle: u32, force: [f64; 3]) {
        let Some(slot) = self.slot_of.get(&handle).copied() else { return };
        let inv_mass = self.metas[slot].inv_mass;
        if inv_mass == 0.0 {
            return;
        }
        let vel = add(self.state.get(slot, VX), scale(force, inv_mass * self.fixed_dt));
        self.state.set(slot, VX, vel);
    }

    /// `BuiltinPhysics.step`, phase for phase.
    ///
    /// The order below is the contract. Integrating forces before the broadphase
    /// means contacts see post-gravity velocities; solving before integrating
    /// positions means the position pass uses resolved velocities; refreshing
    /// AABBs immediately before the broadphase and immediately after the position
    /// pass is what keeps `raycast` reading bounds that correspond to the
    /// positions a caller can observe.
    pub fn step(&mut self, dt: f64) {
        self.refresh_order();
        self.contacts.clear();

        integrate_forces(
            &mut self.state,
            &self.dynamic_mask,
            self.gravity,
            dt,
            self.linear_damping,
            self.angular_damping,
        );

        // Refreshing every body rather than only the dynamic ones. A static
        // body's position cannot have changed since the last refresh (nothing in
        // a step writes it -- `positionalCorrection` scales by `invMass == 0`),
        // and `Aabb::around` is a pure function of position, so the extra writes
        // produce the identical bounds.
        for slot in 0..self.metas.len() {
            self.refresh_aabb(slot);
        }

        // Broadphase and narrowphase fused. `builtin.ts` materialises a `pairs`
        // array between them; nothing reads it, and `narrowphase_pair` is a pure
        // function of `(metas, state)`, so skipping the intermediate cannot
        // change which contacts exist or in what order.
        let order = std::mem::take(&mut self.order);
        let mut contacts: Vec<Contact> = Vec::new();
        for i in 0..order.len() {
            for j in (i + 1)..order.len() {
                let a = order[i];
                let b = order[j];
                let ma = self.metas[a];
                let mb = self.metas[b];
                if !ma.is_dynamic() && !mb.is_dynamic() {
                    continue;
                }
                if !may_collide(&ma, &mb) {
                    continue;
                }
                if !self.aabbs[a].overlaps(&self.aabbs[b]) {
                    continue;
                }
                if let Some(contact) = narrowphase_pair(&self.metas, &self.state, a, b) {
                    contacts.push(contact);
                }
            }
        }

        for _ in 0..self.iterations {
            for contact in &contacts {
                solve_contact(&mut self.state, &self.metas, contact, dt);
            }
        }
        for contact in &contacts {
            positional_correction(&mut self.state, &self.metas, &mut self.aabbs, contact);
        }

        integrate_velocities(&mut self.state, &self.dynamic_mask, dt);

        for slot in 0..self.metas.len() {
            self.refresh_aabb(slot);
        }

        self.contacts = contacts
            .iter()
            .filter(|c| c.depth > 0.0)
            .map(|c| ContactEvent {
                a: self.metas[c.a as usize].handle,
                b: self.metas[c.b as usize].handle,
                normal: c.normal,
                depth: c.depth,
                impulse: 0.0,
            })
            .collect();

        self.order = order;
    }

    /// `BuiltinPhysics.drainContacts`: hand over the list and empty it.
    pub fn drain_contacts(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.contacts)
    }

    /// Contacts still held from the last step, without draining them.
    ///
    /// The wasm ABI needs this and `builtin.ts` has no equivalent, because a
    /// drain across the boundary is a bulk copy into a caller-sized buffer and
    /// the caller has to know how large to make that buffer first. In JS the
    /// array can simply be returned.
    #[inline]
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// Live handles in ascending order.
    ///
    /// This is the order every traversal in this crate uses, and the order the
    /// wasm ABI writes its bulk buffers in, so the two zip without the caller
    /// re-sorting.
    pub fn handles(&self) -> Vec<u32> {
        self.ordered_slots().into_iter().map(|slot| self.metas[slot].handle).collect()
    }

    /// Every body's state flattened to twelve doubles, in ascending handle
    /// order: `px py pz rx ry rz vx vy vz wx wy wz` per body.
    ///
    /// Allocating here looks like it defeats the point of a bulk ABI. It is
    /// still one allocation per step rather than one per body per step, which
    /// is the cost that actually mattered; a persistent scratch buffer on
    /// `World` would save the rest at the price of making `World` hold mutable
    /// state that no caller asked for. Revisit if profiling says otherwise.
    pub fn snapshot(&self) -> Vec<f64> {
        let slots = self.ordered_slots();
        let mut out = Vec::with_capacity(slots.len() * body::CHAN);
        for slot in slots {
            for base in [PX, RX, VX, WX] {
                let v = self.state.get(slot, base);
                out.push(v[0]);
                out.push(v[1]);
                out.push(v[2]);
            }
        }
        out
    }

    /// `BuiltinPhysics.raycast`.
    ///
    /// Tie-breaking is the TS one: strict `<` against the current best, in
    /// ascending handle order, so the lower handle wins an exact tie. Changing
    /// that to `<=` would silently swap which body a grazing ray reports.
    pub fn raycast(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        max_distance: f64,
    ) -> Option<RayHit> {
        let dir = prepare_dir(direction)?;
        let mut best: Option<RayHit> = None;
        for slot in self.ordered_slots() {
            let hit = ray_vs_body(
                origin,
                dir,
                &self.metas[slot],
                self.state.get(slot, PX),
                &self.aabbs[slot],
            );
            let Some(hit) = hit else { continue };
            if hit.distance > max_distance {
                continue;
            }
            let better = match best {
                None => true,
                Some(ref b) => hit.distance < b.distance,
            };
            if better {
                best = Some(RayHit {
                    body: self.metas[slot].handle,
                    point: hit.point,
                    normal: hit.normal,
                    distance: hit.distance,
                });
            }
        }
        best
    }

    /// `BuiltinPhysics.dispose`. Options and `next_handle` survive, so a
    /// disposed-and-reused world keeps handing out fresh handles.
    pub fn dispose(&mut self) {
        self.metas.clear();
        self.state.resize(0);
        self.aabbs.clear();
        self.labels.clear();
        self.dynamic_mask.clear();
        self.slot_of.clear();
        self.order.clear();
        self.contacts.clear();
    }

    /// Cached bounds for a body. Not part of the TS surface; exposed so the wasm
    /// backend can answer a debug query and so tests can pin the refresh points.
    pub fn aabb_of(&self, handle: u32) -> Option<Aabb> {
        self.slot_of.get(&handle).map(|&slot| self.aabbs[slot])
    }

    /// The label a body was created with, or the generated `body:{handle}`.
    pub fn label(&self, handle: u32) -> Option<&str> {
        self.slot_of.get(&handle).map(|&slot| self.labels[slot].as_str())
    }

    /// Inverse mass for a body; `0.0` means the solver will not move it.
    pub fn inv_mass(&self, handle: u32) -> Option<f64> {
        self.slot_of.get(&handle).map(|&slot| self.metas[slot].inv_mass)
    }

    /// Slots in ascending handle order, freshly allocated.
    ///
    /// `step` uses the cached `self.order` instead; this is for `&self` methods
    /// like `raycast`, where allocating one small `Vec` per call is cheaper than
    /// the interior mutability a shared cache would need.
    fn ordered_slots(&self) -> Vec<usize> {
        let mut slots: Vec<usize> = (0..self.metas.len()).collect();
        let metas = &self.metas;
        slots.sort_by_key(|&slot| metas[slot].handle);
        slots
    }

    fn refresh_order(&mut self) {
        let metas = &self.metas;
        let mut slots: Vec<usize> = (0..metas.len()).collect();
        slots.sort_by_key(|&slot| metas[slot].handle);
        self.order = slots;
    }

    fn refresh_aabb(&mut self, slot: usize) {
        let position = self.state.get(slot, PX);
        self.aabbs[slot] = Aabb::around(position, &self.metas[slot].shape);
    }
}

/// `BuiltinPhysics.mayCollide`. Both directions must pass.
#[inline]
fn may_collide(a: &BodyMeta, b: &BodyMeta) -> bool {
    (a.group & b.mask) != 0 && (b.group & a.mask) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use body::ShapeKind;

    fn sphere(position: [f64; 3], radius: f64) -> BodyDescriptor {
        BodyDescriptor {
            shape: Shape::sphere(radius),
            position,
            rotation: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            dynamic: true,
            mass: 1.0,
            restitution: 0.1,
            friction: 0.7,
            group: 1,
            mask: u32::MAX,
            label: None,
        }
    }

    fn static_box(position: [f64; 3], half: [f64; 3]) -> BodyDescriptor {
        BodyDescriptor {
            shape: Shape::aabb_box(half),
            position,
            rotation: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            dynamic: false,
            mass: 0.0,
            restitution: 0.1,
            friction: 0.7,
            group: 1,
            mask: u32::MAX,
            label: None,
        }
    }

    /// Zero gravity, no damping: a world where nothing moves unless a test says
    /// so. Lets the identity and bookkeeping tests ignore the integrator.
    fn still_world() -> World {
        World::new(WorldOptions {
            gravity: [0.0, 0.0, 0.0],
            linear_damping: 0.0,
            angular_damping: 0.0,
            ..WorldOptions::default()
        })
    }

    #[test]
    fn handles_start_at_one_and_are_never_reused() {
        let mut w = still_world();
        assert_eq!(w.create_body(sphere([0.0, 0.0, 0.0], 0.5)).unwrap(), 1);
        assert_eq!(w.create_body(sphere([2.0, 0.0, 0.0], 0.5)).unwrap(), 2);
        assert_eq!(w.create_body(sphere([4.0, 0.0, 0.0], 0.5)).unwrap(), 3);
        w.destroy_body(2);
        // The next handle continues upward; reusing 2 would alias a dead body in
        // any map a caller kept.
        assert_eq!(w.create_body(sphere([6.0, 0.0, 0.0], 0.5)).unwrap(), 4);
        assert_eq!(w.body_count(), 3);
    }

    #[test]
    fn labels_default_to_body_handle_and_honour_an_explicit_one() {
        let mut w = still_world();
        let plain = w.create_body(sphere([0.0, 0.0, 0.0], 0.5)).unwrap();
        let mut named = sphere([2.0, 0.0, 0.0], 0.5);
        named.label = Some("floor".to_string());
        let named = w.create_body(named).unwrap();

        // The generated label embeds the handle, which is only known inside
        // create_body -- the reason label defaulting cannot live in the TS layer.
        assert_eq!(w.label(plain), Some(format!("body:{plain}").as_str()));
        assert_eq!(w.label(named), Some("floor"));
        assert_eq!(w.label(999), None);
    }

    #[test]
    fn negative_mass_is_rejected_but_only_for_dynamic_bodies() {
        let mut w = still_world();
        let mut bad = sphere([0.0, 0.0, 0.0], 0.5);
        bad.mass = -1.0;
        assert_eq!(w.create_body(bad), Err(CreateError::NegativeMass));
        assert_eq!(w.body_count(), 0);

        // A static body's mass is forced to 0, so a negative authored mass is
        // discarded before the check -- same as `builtin.ts`.
        let mut odd = static_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        odd.mass = -5.0;
        let handle = w.create_body(odd).unwrap();
        assert_eq!(w.inv_mass(handle), Some(0.0));

        // mass 0 on a *dynamic* body is legal there and means immovable.
        let mut zero = sphere([0.0, 0.0, 0.0], 0.5);
        zero.mass = 0.0;
        let handle = w.create_body(zero).unwrap();
        assert_eq!(w.inv_mass(handle), Some(0.0));
    }

    #[test]
    fn static_bodies_do_not_move_under_gravity() {
        let mut w = World::default();
        let floor = w.create_body(static_box([0.0, 0.0, 0.0], [5.0, 0.5, 5.0])).unwrap();
        let before = w.get_body_state(floor).unwrap();
        for _ in 0..10 {
            w.step(1.0 / 60.0);
        }
        assert_eq!(w.get_body_state(floor).unwrap(), before);
        assert_eq!(before.velocity, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_falling_sphere_matches_the_closed_form_bit_for_bit() {
        let dt = 1.0 / 60.0;
        let mut w = World::default();
        let ball = w.create_body(sphere([0.0, 10.0, 0.0], 0.5)).unwrap();

        // Exactly the operations `builtin.ts::step` performs, in its order:
        // v += g*dt; v *= max(0, 1 - linearDamping*dt); p += v*dt.
        let g = -9.81;
        let damp = 0.0f64.max(1.0 - 0.05 * dt);
        let mut vy = 0.0 + g * dt;
        vy *= damp;
        let mut py = 10.0 + vy * dt;
        w.step(dt);
        let s = w.get_body_state(ball).unwrap();
        assert_eq!(s.velocity[1].to_bits(), vy.to_bits());
        assert_eq!(s.position[1].to_bits(), py.to_bits());

        // ...and again, to catch an integrator that is right once by accident.
        vy = vy + g * dt;
        vy *= damp;
        py = py + vy * dt;
        w.step(dt);
        let s = w.get_body_state(ball).unwrap();
        assert_eq!(s.velocity[1].to_bits(), vy.to_bits());
        assert_eq!(s.position[1].to_bits(), py.to_bits());
        // Nothing was touching, so no contact was emitted.
        assert!(w.drain_contacts().is_empty());
    }

    #[test]
    fn impulses_and_forces_scale_by_inverse_mass_and_fixed_dt() {
        // `fixedDt` 0.5 rather than the default 1/60, so the force contribution
        // is an exact binary number and the assertions below can be `==`.
        let mut w = World::new(WorldOptions {
            fixed_dt: 0.5,
            gravity: [0.0, 0.0, 0.0],
            linear_damping: 0.0,
            angular_damping: 0.0,
            ..WorldOptions::default()
        });
        let ball = w
            .create_body(BodyDescriptor { mass: 2.0, ..sphere([0.0, 0.0, 0.0], 0.5) })
            .unwrap();
        let floor = w.create_body(static_box([0.0, -5.0, 0.0], [1.0, 1.0, 1.0])).unwrap();

        // invMass = 0.5, so impulse [4,0,0] adds exactly 2.
        w.apply_impulse(ball, [4.0, 0.0, 0.0]);
        assert_eq!(w.get_body_state(ball).unwrap().velocity, [2.0, 0.0, 0.0]);
        // fixedDt is 0.5 here, so force [6,0,0] adds 6 * (invMass * fixedDt)
        // = 6 * (0.5 * 0.5) = 1.5.
        w.apply_force(ball, [6.0, 0.0, 0.0]);
        assert_eq!(w.get_body_state(ball).unwrap().velocity, [3.5, 0.0, 0.0]);

        // Static bodies ignore both, and neither call creates a contact.
        w.apply_impulse(floor, [100.0, 0.0, 0.0]);
        w.apply_force(floor, [100.0, 0.0, 0.0]);
        assert_eq!(w.get_body_state(floor).unwrap().velocity, [0.0, 0.0, 0.0]);
        // Unknown handles are no-ops rather than panics.
        w.apply_impulse(4242, [1.0, 1.0, 1.0]);
        w.apply_force(4242, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn set_body_state_applies_only_masked_channels_and_refreshes_the_aabb() {
        let mut w = still_world();
        let ball = w.create_body(sphere([0.0, 0.0, 0.0], 1.0)).unwrap();

        w.set_body_state(
            ball,
            MASK_POS,
            [5.0, 0.0, 0.0],
            [9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0],
        );
        let s = w.get_body_state(ball).unwrap();
        assert_eq!(s.position, [5.0, 0.0, 0.0]);
        assert_eq!(s.rotation, [0.0, 0.0, 0.0]);
        assert_eq!(s.velocity, [0.0, 0.0, 0.0]);

        // Bounds follow the new position, which is what makes a box raycast
        // correct after a teleport.
        let aabb = w.aabb_of(ball).unwrap();
        assert_eq!(aabb.min, [4.0, -1.0, -1.0]);
        assert_eq!(aabb.max, [6.0, 1.0, 1.0]);

        // A velocity-only write leaves position alone but still recomputes the
        // bounds, matching the TS original's unconditional refresh.
        w.set_body_state(ball, MASK_VEL, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        let s = w.get_body_state(ball).unwrap();
        assert_eq!(s.position, [5.0, 0.0, 0.0]);
        assert_eq!(s.velocity, [1.0, 2.0, 3.0]);
        assert_eq!(w.aabb_of(ball).unwrap().max, [6.0, 1.0, 1.0]);

        // Unknown handle: no-op.
        w.set_body_state(7, MASK_POS, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn overlapping_spheres_emit_one_contact_and_draining_clears_it() {
        let mut w = still_world();
        let a = w.create_body(sphere([0.0, 0.0, 0.0], 0.5)).unwrap();
        let b = w.create_body(sphere([0.5, 0.0, 0.0], 0.5)).unwrap();

        w.step(1.0 / 60.0);
        let contacts = w.drain_contacts();
        assert_eq!(contacts.len(), 1);
        let c = contacts[0];
        assert_eq!((c.a, c.b), (a, b));
        // Normal points from a toward b: delta = [0.5,0,0], scaled by 1/0.5.
        assert_eq!(c.normal, [1.0, 0.0, 0.0]);
        assert!((c.depth - 0.5).abs() < 1e-12);
        assert_eq!(c.impulse, 0.0);

        // Drain is destructive; a second call sees nothing.
        assert!(w.drain_contacts().is_empty());
    }

    #[test]
    fn group_and_mask_bits_filter_pairs() {
        let mut w = still_world();
        let mut a_desc = sphere([0.0, 0.0, 0.0], 0.5);
        a_desc.group = 1;
        a_desc.mask = 2;
        let mut b_desc = sphere([0.5, 0.0, 0.0], 0.5);
        b_desc.group = 4;
        b_desc.mask = 1;
        let a = w.create_body(a_desc).unwrap();
        w.create_body(b_desc).unwrap();

        // (a.group & b.mask) = 1 passes, (b.group & a.mask) = 0 does not.
        w.step(1.0 / 60.0);
        assert!(w.drain_contacts().is_empty());
        // The pair is still overlapping, so the filter is what suppressed it.
        assert!(w.aabb_of(a).unwrap().overlaps(&w.aabb_of(2).unwrap()));
    }

    #[test]
    fn two_static_bodies_never_produce_a_contact() {
        let mut w = still_world();
        w.create_body(static_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])).unwrap();
        w.create_body(static_box([0.5, 0.0, 0.0], [1.0, 1.0, 1.0])).unwrap();
        w.step(1.0 / 60.0);
        assert!(w.drain_contacts().is_empty());
    }

    #[test]
    fn raycast_returns_the_nearest_hit_within_max_distance() {
        let mut w = still_world();
        let near = w.create_body(sphere([0.0, 0.0, -5.0], 1.0)).unwrap();
        let far = w.create_body(sphere([0.0, 0.0, -10.0], 1.0)).unwrap();

        let hit = w.raycast([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 100.0).unwrap();
        assert_eq!(hit.body, near);
        assert_eq!(hit.distance, 4.0);
        assert_eq!(hit.point, [0.0, 0.0, -4.0]);
        assert_eq!(hit.normal, [0.0, 0.0, 1.0]);

        // Out of range: the near sphere is 4 away, so 3 reaches nothing.
        assert_eq!(w.raycast([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 3.0), None);
        // A degenerate direction is rejected before any body is tested.
        assert_eq!(w.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 100.0), None);
        // An unnormalised direction still works: prepare_dir normalises it.
        assert_eq!(w.raycast([0.0, 0.0, 0.0], [0.0, 0.0, -7.0], 100.0).unwrap().body, near);
        assert_ne!(far, near);
    }

    #[test]
    fn raycast_against_a_box_uses_the_cached_bounds() {
        let mut w = still_world();
        let wall = w.create_body(static_box([0.0, 0.0, -4.0], [1.0, 1.0, 1.0])).unwrap();
        let hit = w.raycast([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 100.0).unwrap();
        assert_eq!(hit.body, wall);
        assert_eq!(hit.distance, 3.0);
        assert_eq!(hit.normal, [0.0, 0.0, 1.0]);

        // Teleport the wall and confirm the raycast follows, i.e. that
        // set_body_state really did refresh the bounds the ray reads.
        w.set_body_state(
            wall,
            MASK_POS,
            [0.0, 0.0, -8.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        );
        assert_eq!(w.raycast([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 100.0).unwrap().distance, 7.0);
    }

    #[test]
    fn two_identically_built_worlds_stay_bit_identical() {
        // The determinism claim, stated as a test: same descriptors, same step
        // count, same bits. This is also the property the TS/Rust parity test in
        // `tests/wasm_backend.test.ts` depends on holding on the Rust side alone.
        fn build() -> World {
            let mut w = World::default();
            w.create_body(static_box([0.0, -1.0, 0.0], [10.0, 1.0, 10.0])).unwrap();
            for i in 0..6 {
                let x = (i as f64) * 0.7 - 1.5;
                let y = 1.0 + (i as f64) * 0.31;
                let desc = if i % 2 == 0 {
                    BodyDescriptor {
                        velocity: [0.13 * i as f64, 0.0, -0.07],
                        ..sphere([x, y, 0.1 * i as f64], 0.3)
                    }
                } else {
                    BodyDescriptor {
                        mass: 1.0 + i as f64,
                        restitution: 0.3,
                        friction: 0.4,
                        ..static_box([x, y, 0.0], [0.3, 0.3, 0.3])
                    }
                };
                // Half of them are static, so the mixed path is exercised too.
                let desc = if i == 4 { BodyDescriptor { dynamic: false, ..desc } } else { desc };
                w.create_body(desc).unwrap();
            }
            w
        }

        let (mut a, mut b) = (build(), build());
        for step in 0..120 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
            if step % 40 == 39 {
                a.drain_contacts();
                b.drain_contacts();
            }
        }
        for handle in 1..=7 {
            let (sa, sb) = (a.get_body_state(handle).unwrap(), b.get_body_state(handle).unwrap());
            for (x, y) in [
                (sa.position, sb.position),
                (sa.rotation, sb.rotation),
                (sa.velocity, sb.velocity),
                (sa.angular_velocity, sb.angular_velocity),
            ] {
                for (u, v) in x.iter().zip(y.iter()) {
                    assert_eq!(u.to_bits(), v.to_bits(), "handle {handle} diverged");
                }
            }
        }
    }

    #[test]
    fn destroying_a_middle_body_keeps_every_lookup_correct() {
        let mut w = still_world();
        let a = w.create_body(sphere([0.0, 0.0, 0.0], 0.5)).unwrap();
        let b = w.create_body(sphere([2.0, 0.0, 0.0], 0.5)).unwrap();
        let c = w.create_body(sphere([4.0, 0.0, 0.0], 0.5)).unwrap();
        w.set_body_state(b, MASK_VEL, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 0.0, 0.0]);

        // Swap-remove moves the last body into b's slot; its handle mapping has
        // to be repaired or every later write goes to the wrong body.
        w.destroy_body(b);
        assert_eq!(w.body_count(), 2);
        assert_eq!(w.get_body_state(b), None);
        assert_eq!(w.get_body_state(a).unwrap().position, [0.0, 0.0, 0.0]);
        assert_eq!(w.get_body_state(c).unwrap().position, [4.0, 0.0, 0.0]);
        assert_eq!(w.label(c), Some(format!("body:{c}").as_str()));

        // The surviving bodies still respond to writes at their new slots.
        w.apply_impulse(c, [0.0, 3.0, 0.0]);
        assert_eq!(w.get_body_state(c).unwrap().velocity, [0.0, 3.0, 0.0]);
        assert_eq!(w.get_body_state(a).unwrap().velocity, [0.0, 0.0, 0.0]);

        // Destroying an unknown handle (including the one just removed) is a
        // no-op, not a panic.
        w.destroy_body(b);
        w.destroy_body(9999);
        assert_eq!(w.body_count(), 2);

        // Order is still by handle, so a raycast picks a before c.
        assert_eq!(
            w.raycast([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0).unwrap().body,
            a
        );
    }

    #[test]
    fn a_sphere_comes_to_rest_on_a_static_box() {
        let mut w = World::default();
        w.create_body(static_box([0.0, 0.0, 0.0], [5.0, 0.5, 5.0])).unwrap();
        let ball = w.create_body(sphere([0.0, 3.0, 0.0], 0.5)).unwrap();

        for _ in 0..600 {
            w.step(1.0 / 60.0);
            w.drain_contacts();
        }
        let s = w.get_body_state(ball).unwrap();
        // Resting centre is box top (0.5) + radius (0.5), less the penetration
        // slop the Baumgarte term leaves in place.
        assert!(s.position[1] > 0.9 && s.position[1] < 1.01, "y = {}", s.position[1]);
        assert!(s.velocity[1].abs() < 0.05, "vy = {}", s.velocity[1]);
        // Horizontal drift stays bounded: friction removed the tangential slide.
        assert!(s.position[0].abs() < 0.2 && s.position[2].abs() < 0.2);
    }

    #[test]
    fn snapshot_and_handles_agree_on_order_and_layout() {
        // The wasm ABI writes these two buffers side by side and the TS adapter
        // zips them; a disagreement in order or stride is silent data corruption
        // rather than an error, so both halves are pinned here.
        let mut w = still_world();
        let a = w.create_body(sphere([1.0, 2.0, 3.0], 0.5)).unwrap();
        let b = w.create_body(sphere([4.0, 5.0, 6.0], 0.5)).unwrap();
        let c = w.create_body(sphere([7.0, 8.0, 9.0], 0.5)).unwrap();
        // Destroy the middle one so the surviving slots are non-contiguous and
        // the swap-remove really did reorder the internal arrays.
        w.destroy_body(b);

        assert_eq!(w.handles(), vec![a, c]);
        let snap = w.snapshot();
        assert_eq!(snap.len(), 2 * 12);
        assert_eq!(&snap[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&snap[12..15], &[7.0, 8.0, 9.0]);
        // Untouched channels stay zero rather than leaking a destroyed body.
        assert!(snap[3..12].iter().all(|v| *v == 0.0));
        assert!(snap[15..24].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn contact_count_peeks_without_draining() {
        let mut w = still_world();
        w.create_body(sphere([0.0, 0.0, 0.0], 0.5)).unwrap();
        w.create_body(sphere([0.5, 0.0, 0.0], 0.5)).unwrap();
        w.step(1.0 / 60.0);

        // Counting twice must give the same answer, and must not consume the
        // contacts -- that is the whole reason the wasm ABI has both calls.
        assert_eq!(w.contact_count(), 1);
        assert_eq!(w.contact_count(), 1);
        assert_eq!(w.drain_contacts().len(), 1);
        assert_eq!(w.contact_count(), 0);
    }

    #[test]
    fn dispose_clears_bodies_but_keeps_the_handle_counter() {
        let mut w = still_world();
        w.create_body(sphere([0.0, 0.0, 0.0], 0.5)).unwrap();
        w.create_body(sphere([2.0, 0.0, 0.0], 0.5)).unwrap();
        w.step(1.0 / 60.0);
        w.dispose();

        assert_eq!(w.body_count(), 0);
        assert!(w.drain_contacts().is_empty());
        assert_eq!(w.get_body_state(1), None);
        assert_eq!(w.raycast([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 10.0), None);
        // A reused world must not hand back a handle a caller may still hold.
        assert_eq!(w.create_body(sphere([0.0, 0.0, 0.0], 0.5)).unwrap(), 3);
        w.step(1.0 / 60.0);
    }

    #[test]
    fn shapes_report_their_kind_and_half_extent() {
        // Guards the descriptor plumbing: the wasm layer builds `Shape` from raw
        // numbers, so a mis-set `kind` would silently pick the wrong branch in
        // narrowphase and raycast.
        let s = Shape::sphere(0.25);
        assert_eq!(s.kind, ShapeKind::Sphere);
        assert_eq!(s.half_extent(), [0.25, 0.25, 0.25]);
        let b = Shape::aabb_box([1.0, 2.0, 3.0]);
        assert_eq!(b.kind, ShapeKind::Box);
        assert_eq!(b.half_extent(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn simd_parity_holds_for_the_integration_passes() {
        // `simd_parity_ok` runs the vectorised passes against independently
        // written scalar references over pseudo-random state. On a native build
        // both sides are the scalar fallback, so this asserts the harness itself
        // is wired correctly; the wasm build runs the same function through
        // `td_selftest_simd_parity()` where the two paths genuinely differ.
        assert!(simd_parity_ok());
        assert_eq!(World::default().simd_enabled(), SIMD128);
    }
}

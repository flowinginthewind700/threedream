//! What a body *is*, stored separately from where it *is*.
//!
//! The split is the load-bearing decision in this crate. `src/physics/builtin.ts`
//! keeps one `Body` object per body and walks an array of them; that is natural
//! in JS and hopeless to vectorise, because the twelve state doubles of one body
//! are interleaved with the next body's. Here the state lives component-major in
//! [`StateArrays`] -- all `vx` contiguous, then all `vy`, and so on -- which turns
//! the two integration passes into flat elementwise loops that [`crate::simd`] can
//! do two lanes at a time without changing a single result. Everything the solver
//! reads but never integrates (shape, mass, material, filter bits) stays in
//! [`BodyMeta`], one small record per body.
//!
//! Slots are dense: `destroy_body` swap-removes, so `0..len` is always the set of
//! live bodies and the integration passes never skip a hole. Handles are *not*
//! dense and are never reused, which is what `builtin.ts` does and what makes a
//! body's identity stable across the create/destroy churn an environment does.

use crate::jsmath;

/// Allowed penetration before Baumgarte pushes back, mirroring the TS constant.
pub const PENETRATION_SLOP: f64 = 0.005;
/// Positional correction strength. Same value, same name, same meaning.
pub const BAUMGARTE: f64 = 0.2;

/// Number of doubles in a body's state: position, rotation, velocity, spin.
pub const CHAN: usize = 12;

// Channel bases. State is addressed as `base + component`, so `PX..=PZ` is the
// position triple and `VX..=VZ` the velocity triple.
pub const PX: usize = 0;
pub const RX: usize = 3;
pub const VX: usize = 6;
pub const WX: usize = 9;

/// Which three channels a `setBodyState` mask bit refers to.
pub const MASK_POS: u32 = 1;
pub const MASK_ROT: u32 = 2;
pub const MASK_VEL: u32 = 4;
pub const MASK_ANG: u32 = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShapeKind {
    Sphere,
    Box,
}

/// Sphere or axis-aligned box.
///
/// Both fields are always present; which one is meaningful is decided by `kind`.
/// That is cheaper than an enum payload here because `Shape` is copied into
/// `BodyMeta` and read inside the contact loop, and a tagged union would put a
/// branch where the TS original has none. `capsule` does not exist: `builtin.ts`
/// rejects it in `createBody`, and the rejection is reproduced at the ABI
/// boundary so the two backends fail identically.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub kind: ShapeKind,
    pub radius: f64,
    pub half: [f64; 3],
}

impl Shape {
    pub fn sphere(radius: f64) -> Self {
        Self { kind: ShapeKind::Sphere, radius, half: [0.0; 3] }
    }

    pub fn aabb_box(half: [f64; 3]) -> Self {
        Self { kind: ShapeKind::Box, radius: 0.0, half }
    }

    /// `shapeHalfExtent`: the AABB half-size, which for a sphere is its radius.
    #[inline(always)]
    pub fn half_extent(&self) -> [f64; 3] {
        match self.kind {
            ShapeKind::Sphere => [self.radius, self.radius, self.radius],
            ShapeKind::Box => self.half,
        }
    }
}

/// Everything about a body that the integrators do not touch.
#[derive(Clone, Copy, Debug)]
pub struct BodyMeta {
    /// Stable, never reused, assigned from 1 upward -- the sort key for every
    /// ordered traversal, which is the determinism guarantee.
    pub handle: u32,
    pub shape: Shape,
    /// `mass > 0 ? 1 / mass : 0`. Zero means "the solver leaves it alone".
    pub inv_mass: f64,
    pub restitution: f64,
    pub friction: f64,
    pub group: u32,
    pub mask: u32,
}

impl BodyMeta {
    /// True when the integrators and the impulse paths should act on this body.
    ///
    /// `builtin.ts` spells this `body.invMass === 0 -> continue`, so the test is
    /// against the reciprocal and not against the body kind: a *dynamic* body
    /// authored with `mass: 0` is immovable there, and it has to be immovable
    /// here too.
    #[inline(always)]
    pub fn is_dynamic(&self) -> bool {
        self.inv_mass != 0.0
    }
}

/// Axis-aligned bounds, refreshed at exactly the points `builtin.ts` refreshes
/// them: before the broadphase, inside positional correction, and after the
/// position integrate. Those points are observable -- `raycast` against a box
/// reads the cached bounds rather than recomputing them -- so they are not free
/// to consolidate.
#[derive(Clone, Copy, Debug, Default)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    /// `refreshAabb`: centre +/- half-extent.
    #[inline(always)]
    pub fn around(position: [f64; 3], shape: &Shape) -> Self {
        let e = shape.half_extent();
        Self {
            min: jsmath::sub(position, e),
            max: jsmath::add(position, e),
        }
    }

    /// `aabbOverlap`, component order included: the TS original short-circuits
    /// in this order, and although `&&` on six comparisons cannot change a
    /// result, keeping the order makes the two files readable side by side.
    #[inline(always)]
    pub fn overlaps(&self, o: &Aabb) -> bool {
        self.min[0] <= o.max[0]
            && self.max[0] >= o.min[0]
            && self.min[1] <= o.max[1]
            && self.max[1] >= o.min[1]
            && self.min[2] <= o.max[2]
            && self.max[2] >= o.min[2]
    }
}

/// Per-component state arrays, component-major.
///
/// `c[PX + k][slot]` is the `k`-th position component of `slot`. All twelve
/// vectors always have the same length, which is the invariant that lets the
/// integration passes index them without bounds gymnastics.
#[derive(Clone, Debug, Default)]
pub struct StateArrays {
    pub c: [Vec<f64>; CHAN],
}

impl StateArrays {
    pub fn new() -> Self {
        Self::default()
    }

    /// Grow or shrink every channel to `len` slots, zero-filling new ones.
    pub fn resize(&mut self, len: usize) {
        for chan in self.c.iter_mut() {
            chan.resize(len, 0.0);
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.c[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read a triple at channel `base` for `slot`.
    #[inline(always)]
    pub fn get(&self, slot: usize, base: usize) -> [f64; 3] {
        [self.c[base][slot], self.c[base + 1][slot], self.c[base + 2][slot]]
    }

    /// Write a triple at channel `base` for `slot`.
    #[inline(always)]
    pub fn set(&mut self, slot: usize, base: usize, v: [f64; 3]) {
        self.c[base][slot] = v[0];
        self.c[base + 1][slot] = v[1];
        self.c[base + 2][slot] = v[2];
    }

    /// Swap two slots across every channel. Used by `destroy_body`'s swap-remove.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for chan in self.c.iter_mut() {
            chan.swap(a, b);
        }
    }

    /// Drop the last slot from every channel.
    pub fn pop(&mut self) {
        for chan in self.c.iter_mut() {
            chan.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_extent_is_the_radius_for_spheres_and_the_extents_for_boxes() {
        assert_eq!(Shape::sphere(0.25).half_extent(), [0.25, 0.25, 0.25]);
        assert_eq!(Shape::aabb_box([1.0, 2.0, 3.0]).half_extent(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn is_dynamic_follows_inv_mass_not_the_kind() {
        let meta = |inv_mass: f64| BodyMeta {
            handle: 1,
            shape: Shape::sphere(1.0),
            inv_mass,
            restitution: 0.1,
            friction: 0.7,
            group: 1,
            mask: u32::MAX,
        };
        assert!(meta(1.0).is_dynamic());
        assert!(!meta(0.0).is_dynamic());
        // A dynamic body authored with mass 0 lands here, and must not move.
        assert!(!meta(-0.0).is_dynamic());
    }

    #[test]
    fn state_arrays_keep_every_channel_the_same_length() {
        let mut s = StateArrays::new();
        assert!(s.is_empty());
        s.resize(3);
        assert_eq!(s.len(), 3);
        assert!(s.c.iter().all(|c| c.len() == 3));

        s.set(1, VX, [4.0, 5.0, 6.0]);
        assert_eq!(s.get(1, VX), [4.0, 5.0, 6.0]);
        assert_eq!(s.get(0, VX), [0.0, 0.0, 0.0]);

        // A grown channel must not inherit a previous body's velocity.
        s.resize(5);
        assert_eq!(s.get(4, VX), [0.0, 0.0, 0.0]);
        s.pop();
        s.pop();
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn swap_moves_every_channel() {
        let mut s = StateArrays::new();
        s.resize(2);
        s.set(0, PX, [1.0, 1.0, 1.0]);
        s.set(1, PX, [2.0, 2.0, 2.0]);
        s.set(0, WX, [7.0, 0.0, 0.0]);
        s.swap(0, 1);
        assert_eq!(s.get(0, PX), [2.0, 2.0, 2.0]);
        assert_eq!(s.get(1, PX), [1.0, 1.0, 1.0]);
        assert_eq!(s.get(1, WX), [7.0, 0.0, 0.0]);
        s.swap(0, 0);
        assert_eq!(s.get(0, PX), [2.0, 2.0, 2.0]);
    }

    #[test]
    fn aabb_is_centre_plus_minus_half_extent_and_overlaps_transitively() {
        let a = Aabb::around([0.0, 0.0, 0.0], &Shape::sphere(1.0));
        assert_eq!(a.min, [-1.0, -1.0, -1.0]);
        assert_eq!(a.max, [1.0, 1.0, 1.0]);

        let b = Aabb::around([1.5, 0.0, 0.0], &Shape::aabb_box([0.5, 0.5, 0.5]));
        assert!(a.overlaps(&b) && b.overlaps(&a));

        // Touching bounds overlap: the test is `<=`, not `<`.
        let c = Aabb::around([2.0, 0.0, 0.0], &Shape::sphere(1.0));
        assert!(a.overlaps(&c));
        let d = Aabb::around([2.1, 0.0, 0.0], &Shape::sphere(1.0));
        assert!(!a.overlaps(&d));
    }
}

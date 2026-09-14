//! The sequential-impulse contact solver, kept scalar on purpose.
//!
//! This is the half of `builtin.ts::step` that [`crate::simd`] must *not*
//! touch. `solveContact` runs `iterations` times over the same contact list,
//! and each pass reads the velocities the previous pass wrote -- a Gauss-Seidel
//! chain. Reordering it (which is what vectorising across contacts would do)
//! changes the answer, so every operation here is a plain scalar `f64` op in the
//! exact order the TS original writes it.
//!
//! Bit-exactness notes, all load-bearing:
//!
//! * `relative` is captured once, before any impulse is applied. The friction
//!   term reuses that original `relative`, not a re-read of the (now modified)
//!   velocities. Re-reading would drift.
//! * `spinFromFriction` runs *after* the normal impulse, so it reads the
//!   already-updated linear velocity -- matching the TS call order.
//! * Every `Math.min`/`Math.max`/`Math.abs` goes through [`crate::jsmath`] so
//!   NaN and signed-zero behave like JS, and `-(1 + bounce) * nv` keeps the
//!   unary-minus-then-multiply grouping the TS parser produces.

use crate::body::{
    Aabb, BodyMeta, ShapeKind, StateArrays, BAUMGARTE, PENETRATION_SLOP, PX, VX, WX,
};
use crate::jsmath::{add, cross, dot, js_abs, js_max, js_min, length, scale, sub};
use crate::narrowphase::Contact;

/// `BuiltinPhysics.solveContact`: one Gauss-Seidel pass over a single contact.
///
/// `dt` is the step's timestep, used only for the resting-contact restitution
/// cutoff (`1.5 / max(dt, 1e-6) / 60`).
pub fn solve_contact(state: &mut StateArrays, metas: &[BodyMeta], contact: &Contact, dt: f64) {
    let a = contact.a as usize;
    let b = contact.b as usize;
    let ma = metas[a];
    let mb = metas[b];
    let normal = contact.normal;

    let inv_mass_sum = ma.inv_mass + mb.inv_mass;
    if inv_mass_sum == 0.0 {
        return;
    }

    // Captured once; the friction block below reuses this exact value.
    let relative = sub(state.get(b, VX), state.get(a, VX));
    let normal_velocity = dot(relative, normal);
    if normal_velocity > 0.0 {
        return;
    }

    let restitution = js_min(ma.restitution, mb.restitution);
    // Skip restitution for near-resting contacts so stacks do not jitter.
    let bounce = if js_abs(normal_velocity) < 1.5 / js_max(dt, 1e-6) / 60.0 {
        0.0
    } else {
        restitution
    };
    let mut lambda = (-(1.0 + bounce) * normal_velocity) / inv_mass_sum;
    lambda = js_max(lambda, 0.0);

    let impulse = scale(normal, lambda);
    let va = sub(state.get(a, VX), scale(impulse, ma.inv_mass));
    state.set(a, VX, va);
    let vb = add(state.get(b, VX), scale(impulse, mb.inv_mass));
    state.set(b, VX, vb);
    spin_from_friction(state, &ma, a, normal, -1.0);
    spin_from_friction(state, &mb, b, normal, 1.0);

    // Friction: remove tangential relative velocity up to the Coulomb cone.
    let tangent_velocity = sub(relative, scale(normal, normal_velocity));
    let tangent_speed = length(tangent_velocity);
    if tangent_speed > 1e-6 {
        let tangent = scale(tangent_velocity, 1.0 / tangent_speed);
        let mu = (ma.friction * mb.friction).sqrt();
        let mut jt = -tangent_speed / inv_mass_sum;
        let max_friction = mu * lambda;
        jt = js_max(-max_friction, js_min(max_friction, jt));
        let friction_impulse = scale(tangent, jt);
        let va = sub(state.get(a, VX), scale(friction_impulse, ma.inv_mass));
        state.set(a, VX, va);
        let vb = add(state.get(b, VX), scale(friction_impulse, mb.inv_mass));
        state.set(b, VX, vb);
    }
}

/// `BuiltinPhysics.spinFromFriction`: spheres pick up spin from contact.
///
/// Reads the body's *current* linear velocity (already updated by the normal
/// impulse), so the call site ordering matters and is preserved.
fn spin_from_friction(
    state: &mut StateArrays,
    meta: &BodyMeta,
    slot: usize,
    normal: [f64; 3],
    sign: f64,
) {
    if meta.inv_mass == 0.0 || meta.shape.kind != ShapeKind::Sphere {
        return;
    }
    let rolling = cross(normal, state.get(slot, VX));
    let w = add(
        state.get(slot, WX),
        scale(rolling, (sign * meta.friction * 0.05) / meta.shape.radius),
    );
    state.set(slot, WX, w);
}

/// `BuiltinPhysics.positionalCorrection`: Baumgarte push-apart, then refresh the
/// two AABBs exactly where the TS refreshes them (observable: a box raycast
/// reads the cached bounds, so the refresh point is not free to move).
pub fn positional_correction(
    state: &mut StateArrays,
    metas: &[BodyMeta],
    aabbs: &mut [Aabb],
    contact: &Contact,
) {
    let a = contact.a as usize;
    let b = contact.b as usize;
    let ma = metas[a];
    let mb = metas[b];
    let normal = contact.normal;

    let inv_mass_sum = ma.inv_mass + mb.inv_mass;
    if inv_mass_sum == 0.0 {
        return;
    }
    let correction = (js_max(contact.depth - PENETRATION_SLOP, 0.0) * BAUMGARTE) / inv_mass_sum;
    let pa = sub(state.get(a, PX), scale(normal, correction * ma.inv_mass));
    state.set(a, PX, pa);
    let pb = add(state.get(b, PX), scale(normal, correction * mb.inv_mass));
    state.set(b, PX, pb);
    aabbs[a] = Aabb::around(pa, &ma.shape);
    aabbs[b] = Aabb::around(pb, &mb.shape);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Shape;

    fn meta(handle: u32, shape: Shape, inv_mass: f64) -> BodyMeta {
        BodyMeta {
            handle,
            shape,
            inv_mass,
            restitution: 0.1,
            friction: 0.7,
            group: 1,
            mask: u32::MAX,
        }
    }

    fn world2(
        ma: BodyMeta,
        pa: [f64; 3],
        va: [f64; 3],
        mb: BodyMeta,
        pb: [f64; 3],
        vb: [f64; 3],
    ) -> (StateArrays, Vec<BodyMeta>, Vec<Aabb>) {
        let mut state = StateArrays::new();
        state.resize(2);
        state.set(0, PX, pa);
        state.set(0, VX, va);
        state.set(1, PX, pb);
        state.set(1, VX, vb);
        let aabbs = vec![Aabb::around(pa, &ma.shape), Aabb::around(pb, &mb.shape)];
        (state, vec![ma, mb], aabbs)
    }

    #[test]
    fn approaching_equal_spheres_cancel_their_normal_velocity() {
        // Head-on along X, equal mass, restitution suppressed by the resting
        // cutoff at dt = 1/60 (|nv| = 2 > 1.5, so bounce = restitution = 0.1).
        let s = |h| meta(h, Shape::sphere(0.5), 1.0);
        let contact = Contact { a: 0, b: 1, normal: [1.0, 0.0, 0.0], depth: 0.1 };
        let (mut state, metas, _) =
            world2(s(1), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], s(2), [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        // relative = vb - va = (-1 - 1) = -2 along X; nv = -2.
        solve_contact(&mut state, &metas, &contact, 1.0 / 60.0);
        let va = state.get(0, VX);
        let vb = state.get(1, VX);
        // Symmetric: velocities swap-ish and stay antisymmetric about 0.
        assert!(va[0] < 0.0, "a should recoil backward, got {va:?}");
        assert!(vb[0] > 0.0, "b should recoil forward, got {vb:?}");
        assert!((va[0] + vb[0]).abs() < 1e-12, "momentum conserved: {va:?} {vb:?}");
    }

    #[test]
    fn separating_contact_is_left_alone() {
        // nv > 0 (moving apart): solveContact returns before any impulse.
        let s = |h| meta(h, Shape::sphere(0.5), 1.0);
        let contact = Contact { a: 0, b: 1, normal: [1.0, 0.0, 0.0], depth: 0.1 };
        let (mut state, metas, _) =
            world2(s(1), [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], s(2), [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let before_a = state.get(0, VX);
        let before_b = state.get(1, VX);
        solve_contact(&mut state, &metas, &contact, 1.0 / 60.0);
        assert_eq!(state.get(0, VX), before_a);
        assert_eq!(state.get(1, VX), before_b);
    }

    #[test]
    fn two_statics_do_nothing() {
        let s = |h| meta(h, Shape::sphere(0.5), 0.0);
        let contact = Contact { a: 0, b: 1, normal: [1.0, 0.0, 0.0], depth: 0.5 };
        let (mut state, metas, mut aabbs) =
            world2(s(1), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], s(2), [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let aabb_before = (aabbs[0].min, aabbs[1].max);
        solve_contact(&mut state, &metas, &contact, 1.0 / 60.0);
        positional_correction(&mut state, &metas, &mut aabbs, &contact);
        assert_eq!(state.get(0, PX), [0.0, 0.0, 0.0]);
        assert_eq!(state.get(1, PX), [1.0, 0.0, 0.0]);
        assert_eq!((aabbs[0].min, aabbs[1].max), aabb_before);
    }

    #[test]
    fn positional_correction_pushes_apart_by_baumgarte_and_refreshes_aabbs() {
        let s = |h| meta(h, Shape::sphere(0.5), 1.0);
        // depth 0.1, slop 0.005 -> correction = (0.095 * 0.2) / 2 per unit invMass.
        let contact = Contact { a: 0, b: 1, normal: [1.0, 0.0, 0.0], depth: 0.1 };
        let (mut state, metas, mut aabbs) =
            world2(s(1), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], s(2), [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        positional_correction(&mut state, &metas, &mut aabbs, &contact);
        let expected = (0.095 * 0.2) / 2.0; // * invMass (1.0) each
        assert!((state.get(0, PX)[0] + expected).abs() < 1e-12);
        assert!((state.get(1, PX)[0] - (1.0 + expected)).abs() < 1e-12);
        // AABB tracked the moved position.
        assert!((aabbs[0].max[0] - (state.get(0, PX)[0] + 0.5)).abs() < 1e-12);
    }

    #[test]
    fn correction_below_slop_does_not_move() {
        let s = |h| meta(h, Shape::sphere(0.5), 1.0);
        // depth == slop -> max(depth - slop, 0) == 0 -> no correction.
        let contact = Contact { a: 0, b: 1, normal: [1.0, 0.0, 0.0], depth: PENETRATION_SLOP };
        let (mut state, metas, mut aabbs) =
            world2(s(1), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], s(2), [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        positional_correction(&mut state, &metas, &mut aabbs, &contact);
        assert_eq!(state.get(0, PX), [0.0, 0.0, 0.0]);
        assert_eq!(state.get(1, PX), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn spin_from_friction_only_spins_dynamic_spheres() {
        // A dynamic sphere gains angular velocity; a box or a static does not.
        let sphere = meta(1, Shape::sphere(0.5), 1.0);
        let mut state = StateArrays::new();
        state.resize(1);
        state.set(0, VX, [1.0, 0.0, 0.0]);
        spin_from_friction(&mut state, &sphere, 0, [0.0, 1.0, 0.0], 1.0);
        assert_ne!(state.get(0, WX), [0.0, 0.0, 0.0], "sphere should spin");

        let boxy = meta(2, Shape::aabb_box([0.5; 3]), 1.0);
        let mut s2 = StateArrays::new();
        s2.resize(1);
        s2.set(0, VX, [1.0, 0.0, 0.0]);
        spin_from_friction(&mut s2, &boxy, 0, [0.0, 1.0, 0.0], 1.0);
        assert_eq!(s2.get(0, WX), [0.0, 0.0, 0.0], "box must not spin");

        let stat = meta(3, Shape::sphere(0.5), 0.0);
        let mut s3 = StateArrays::new();
        s3.resize(1);
        s3.set(0, VX, [1.0, 0.0, 0.0]);
        spin_from_friction(&mut s3, &stat, 0, [0.0, 1.0, 0.0], 1.0);
        assert_eq!(s3.get(0, WX), [0.0, 0.0, 0.0], "static must not spin");
    }

    #[test]
    fn tangential_friction_reduces_sliding() {
        // a slides along +X while closing on b along the contact normal (Y), so
        // the normal impulse is non-zero and the Coulomb cone has something to
        // clamp against. Friction must bleed off tangential speed without
        // reversing it.
        let s = |h| meta(h, Shape::sphere(0.5), 1.0);
        let contact = Contact { a: 0, b: 1, normal: [0.0, 1.0, 0.0], depth: 0.0 };
        let (mut state, metas, _) = world2(
            s(1),
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            s(2),
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        );
        // relative = [-2,-1,0], nv = -1, so bounce is suppressed by the resting
        // cutoff and lambda = 1 / invMassSum = 0.5.
        solve_contact(&mut state, &metas, &contact, 1.0 / 60.0);
        let va = state.get(0, VX);
        let vb = state.get(1, VX);
        // Normal axis: the 0.5 impulse splits equally across the two unit masses.
        assert!((va[1] + 0.5).abs() < 1e-12, "{va:?}");
        assert!((vb[1] + 0.5).abs() < 1e-12, "{vb:?}");
        // Tangent axis: jt is clamped to -mu*lambda = -0.7*0.5 = -0.35, so a
        // loses 0.35 of its 2.0 slide and b gains it. Equal and opposite, which
        // is the property that keeps momentum conserved under friction.
        assert!((va[0] - 1.65).abs() < 1e-12, "friction should slow the slide, got {va:?}");
        assert!((vb[0] - 0.35).abs() < 1e-12, "friction should drag b along, got {vb:?}");
    }

    #[test]
    fn friction_is_bounded_by_the_normal_impulse() {
        // A pure slide with no approach: nv = 0, lambda = 0, maxFriction = 0, so
        // jt clamps to -0.0 and nothing moves. That is `builtin.ts`'s behaviour
        // and it is worth pinning, because "friction with no normal force" is
        // exactly the kind of plausible-looking divergence a port introduces.
        let s = |h| meta(h, Shape::sphere(0.5), 1.0);
        let contact = Contact { a: 0, b: 1, normal: [0.0, 1.0, 0.0], depth: 0.0 };
        let (mut state, metas, _) =
            world2(s(1), [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], s(2), [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]);
        solve_contact(&mut state, &metas, &contact, 1.0 / 60.0);
        assert_eq!(state.get(0, VX), [2.0, 0.0, 0.0]);
        assert_eq!(state.get(1, VX), [0.0, 0.0, 0.0]);
    }
}

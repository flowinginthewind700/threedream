//! Contact generation, ported operation-for-operation from `builtin.ts`.
//!
//! Four shape pairs, one dispatcher. Every branch reproduces the TS original's
//! arithmetic *and* its short-circuit order, because the acceptance bar for M1
//! is bit-identical doubles, not "close enough". The one deliberate omission is
//! the contact `point`: `builtin.ts` computes it in all three helpers but never
//! reads it again -- `solveContact` and `positionalCorrection` use only `normal`
//! and `depth` -- so the port drops it rather than carry a value with no
//! observable effect. Dropping a *dead* computation cannot change a result;
//! dropping a live one would, which is why the parity tests compare every
//! emitted field that survives.
//!
//! `a`/`b` on [`Contact`] are **slot** indices (dense, swap-remove), not the
//! stable handles. The solver indexes state by slot, so slots are what it needs;
//! `World::step` maps them back to handles when it emits `ContactEvent`s.

use crate::body::{BodyMeta, ShapeKind, StateArrays, PX};
use crate::jsmath::{axis, js_abs, js_max, js_min, length, normalize, scale, sub};

/// A contact between two bodies. `normal` points from `a` toward `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Slot index of the first body (lower handle of the pair).
    pub a: u32,
    /// Slot index of the second body (higher handle of the pair).
    pub b: u32,
    pub normal: [f64; 3],
    pub depth: f64,
}

/// `BuiltinPhysics.narrowphase`: dispatch on the two shape kinds.
///
/// The box+sphere arm is the subtle one. `builtin.ts` calls `sphereBox(b, a)`
/// (sphere = `b`, box = `a`), which returns a normal pointing *sphere -> box*,
/// then re-labels the contact `a`/`b` back to the outer order and negates the
/// normal. The net effect is a normal pointing from the outer `a` (the box) to
/// the outer `b` (the sphere), with `depth` unchanged. Reproduced exactly below.
pub fn narrowphase_pair(
    metas: &[BodyMeta],
    state: &StateArrays,
    a: usize,
    b: usize,
) -> Option<Contact> {
    let ma = metas[a];
    let mb = metas[b];
    let pa = state.get(a, PX);
    let pb = state.get(b, PX);
    let (normal, depth) = match (ma.shape.kind, mb.shape.kind) {
        (ShapeKind::Sphere, ShapeKind::Sphere) => {
            sphere_sphere(pa, ma.shape.radius, pb, mb.shape.radius)?
        }
        (ShapeKind::Sphere, ShapeKind::Box) => sphere_box(pa, ma.shape.radius, pb, mb.shape.half)?,
        (ShapeKind::Box, ShapeKind::Sphere) => {
            let (n, d) = sphere_box(pb, mb.shape.radius, pa, ma.shape.half)?;
            (scale(n, -1.0), d)
        }
        (ShapeKind::Box, ShapeKind::Box) => box_box(pa, ma.shape.half, pb, mb.shape.half)?,
    };
    Some(Contact { a: a as u32, b: b as u32, normal, depth })
}

/// `sphereSphere`.
fn sphere_sphere(
    pa: [f64; 3],
    ra: f64,
    pb: [f64; 3],
    rb: f64,
) -> Option<([f64; 3], f64)> {
    let delta = sub(pb, pa);
    let dist = length(delta);
    let sum = ra + rb;
    if dist >= sum {
        return None;
    }
    // Degenerate (coincident centres): fall back to +Y, exactly as the TS does.
    let normal = if dist > 1e-9 { scale(delta, 1.0 / dist) } else { [0.0, 1.0, 0.0] };
    Some((normal, sum - dist))
}

/// `sphereBox`, with the sphere first and the box second.
///
/// Returns a normal pointing *from the sphere toward the box*, matching the TS
/// helper's convention (the dispatcher negates it for the box+sphere arm).
fn sphere_box(
    sphere_pos: [f64; 3],
    radius: f64,
    box_pos: [f64; 3],
    half: [f64; 3],
) -> Option<([f64; 3], f64)> {
    let local = sub(sphere_pos, box_pos);
    let clamped = [
        js_max(-half[0], js_min(half[0], local[0])),
        js_max(-half[1], js_min(half[1], local[1])),
        js_max(-half[2], js_min(half[2], local[2])),
    ];
    let delta = sub(local, clamped);
    let dist = length(delta);
    let inside = dist < 1e-9;
    if !inside && dist >= radius {
        return None;
    }

    if inside {
        // Centre is inside the box: push out along the shallowest axis. Three
        // (depth, normal) candidates, stable-sorted by depth ascending. The
        // comparator mirrors JS `sort((x, y) => x[0] - y[0])`: a stable sort on
        // `.0`, with NaN treated as "equal" (V8's SortCompare maps a NaN
        // comparator result to +0). `sort_by` is stable, so equal depths keep
        // their axis order 0,1,2 just like the TS array.
        let mut pen: [(f64, [f64; 3]); 3] = [(0.0, [0.0; 3]); 3];
        for ax in 0..3 {
            let pos_side = half[ax] - local[ax];
            let neg_side = half[ax] + local[ax];
            let best = js_min(pos_side, neg_side);
            let sign = if pos_side < neg_side { 1.0 } else { -1.0 };
            pen[ax] = (best + radius, axis(ax, sign));
        }
        pen.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(core::cmp::Ordering::Equal));
        let depth = pen[0].0;
        let normal = scale(pen[0].1, -1.0);
        Some((normal, depth))
    } else {
        let normal = normalize(scale(delta, -1.0));
        Some((normal, radius - dist))
    }
}

/// `boxBox`: axis-aligned, shallowest-overlap separating axis.
fn box_box(
    pa: [f64; 3],
    ha: [f64; 3],
    pb: [f64; 3],
    hb: [f64; 3],
) -> Option<([f64; 3], f64)> {
    let delta = sub(pb, pa);
    let mut best_axis: i32 = -1;
    let mut best_depth = f64::INFINITY;
    for ax in 0..3usize {
        let overlap = ha[ax] + hb[ax] - js_abs(delta[ax]);
        if overlap <= 0.0 {
            return None;
        }
        if overlap < best_depth {
            best_depth = overlap;
            best_axis = ax as i32;
        }
    }
    if best_axis < 0 {
        return None;
    }
    let ax = best_axis as usize;
    let sign = if delta[ax] >= 0.0 { 1.0 } else { -1.0 };
    Some((axis(ax, sign), best_depth))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Shape;

    fn meta(handle: u32, shape: Shape) -> BodyMeta {
        BodyMeta {
            handle,
            shape,
            inv_mass: 1.0,
            restitution: 0.1,
            friction: 0.7,
            group: 1,
            mask: u32::MAX,
        }
    }

    /// Build a two-slot world so `narrowphase_pair` can be driven end to end.
    fn pair(a: BodyMeta, pa: [f64; 3], b: BodyMeta, pb: [f64; 3]) -> Option<Contact> {
        let metas = [a, b];
        let mut state = StateArrays::new();
        state.resize(2);
        state.set(0, PX, pa);
        state.set(1, PX, pb);
        narrowphase_pair(&metas, &state, 0, 1)
    }

    #[test]
    fn spheres_separate_and_touching_is_not_a_contact() {
        let s = |r| meta(1, Shape::sphere(r));
        // Overlapping along +X: normal points a -> b, i.e. +X.
        let c = pair(s(1.0), [0.0, 0.0, 0.0], meta(2, Shape::sphere(1.0)), [1.5, 0.0, 0.0]).unwrap();
        assert_eq!(c.normal, [1.0, 0.0, 0.0]);
        assert!((c.depth - 0.5).abs() < 1e-12);
        assert_eq!((c.a, c.b), (0, 1));
        // Exactly touching (dist == sum) is NOT a contact: the test is `>=`.
        assert!(pair(s(1.0), [0.0, 0.0, 0.0], meta(2, Shape::sphere(1.0)), [2.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn coincident_spheres_fall_back_to_plus_y() {
        let c = pair(
            meta(1, Shape::sphere(1.0)),
            [0.0, 0.0, 0.0],
            meta(2, Shape::sphere(1.0)),
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        assert_eq!(c.normal, [0.0, 1.0, 0.0]);
        assert_eq!(c.depth, 2.0);
    }

    #[test]
    fn sphere_outside_box_pushes_back_along_the_delta() {
        // Sphere just outside the +X face of a unit box at the origin.
        let c = pair(
            meta(1, Shape::sphere(0.5)),
            [1.2, 0.0, 0.0],
            meta(2, Shape::aabb_box([1.0, 1.0, 1.0])),
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        // normal points sphere -> box = -X.
        assert_eq!(c.normal, [-1.0, 0.0, 0.0]);
        // depth = radius - dist = 0.5 - 0.2.
        assert!((c.depth - 0.3).abs() < 1e-12);
    }

    #[test]
    fn sphere_far_from_box_is_no_contact() {
        assert!(pair(
            meta(1, Shape::sphere(0.5)),
            [3.0, 0.0, 0.0],
            meta(2, Shape::aabb_box([1.0, 1.0, 1.0])),
            [0.0, 0.0, 0.0],
        )
        .is_none());
    }

    #[test]
    fn sphere_centre_inside_box_exits_shallowest_axis() {
        // Centre at x=0.9 inside a unit box: +X face is 0.1 away (shallowest).
        let c = pair(
            meta(1, Shape::sphere(0.5)),
            [0.9, 0.0, 0.0],
            meta(2, Shape::aabb_box([1.0, 1.0, 1.0])),
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        // shallowest axis is +X (pos_side 0.1 < neg_side 1.9), sign +1, then
        // the helper returns axis(0, +1) and the *dispatcher* would negate it
        // for box+sphere. Here sphere is `a`, so normal points sphere -> box.
        // pen[0] = (0.1 + 0.5, axis(0, +1)); normal = -axis(0,+1) = -X.
        assert_eq!(c.normal, [-1.0, 0.0, 0.0]);
        assert!((c.depth - 0.6).abs() < 1e-12);
    }

    #[test]
    fn box_plus_sphere_negates_the_normal_and_keeps_depth() {
        // Same geometry as `sphere_outside_box_pushes_back_along_the_delta` but
        // with the box as `a` and the sphere as `b`. The dispatcher must flip
        // the normal so it still points a -> b (box -> sphere = +X).
        let c = pair(
            meta(1, Shape::aabb_box([1.0, 1.0, 1.0])),
            [0.0, 0.0, 0.0],
            meta(2, Shape::sphere(0.5)),
            [1.2, 0.0, 0.0],
        )
        .unwrap();
        assert_eq!(c.normal, [1.0, 0.0, 0.0]);
        assert!((c.depth - 0.3).abs() < 1e-12);
    }

    #[test]
    fn boxes_pick_the_shallowest_separating_axis() {
        // Overlap 0.5 on X, 1.5 on Y: X is shallowest, delta.x > 0 so +X.
        let c = pair(
            meta(1, Shape::aabb_box([1.0, 1.0, 1.0])),
            [0.0, 0.0, 0.0],
            meta(2, Shape::aabb_box([1.0, 1.0, 1.0])),
            [1.5, 0.0, 0.0],
        )
        .unwrap();
        assert_eq!(c.normal, [1.0, 0.0, 0.0]);
        assert!((c.depth - 0.5).abs() < 1e-12);
    }

    #[test]
    fn separated_boxes_are_no_contact() {
        assert!(pair(
            meta(1, Shape::aabb_box([1.0, 1.0, 1.0])),
            [0.0, 0.0, 0.0],
            meta(2, Shape::aabb_box([1.0, 1.0, 1.0])),
            [3.0, 0.0, 0.0],
        )
        .is_none());
    }

    #[test]
    fn negative_delta_picks_the_negative_axis_sign() {
        // b is on the -X side of a: delta.x < 0 so the normal is -X.
        let c = pair(
            meta(1, Shape::aabb_box([1.0, 1.0, 1.0])),
            [0.0, 0.0, 0.0],
            meta(2, Shape::aabb_box([1.0, 1.0, 1.0])),
            [-1.5, 0.0, 0.0],
        )
        .unwrap();
        assert_eq!(c.normal, [-1.0, 0.0, 0.0]);
    }
}

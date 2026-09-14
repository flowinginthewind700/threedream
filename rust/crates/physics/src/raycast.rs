//! Sphere and axis-aligned-box raycasts, matching `rayVsBody` exactly.
//!
//! Two shapes, two analytic tests. The sphere test is the standard
//! closest-approach formulation; the box test is a slab intersection that also
//! reports *which* face was entered so the hit normal is an axis unit vector.
//! Both reproduce the TS control flow verbatim, including the `length(dir) < 0.5`
//! degeneracy gate in [`prepare_dir`] and the `Math.min`/`Math.abs` spellings
//! (via [`crate::jsmath`]) so a NaN or signed zero behaves the same.
//!
//! The per-body hit carries no body handle: [`crate::World::raycast`] owns the
//! handle-ordered iteration and best-hit selection, and stamps the handle on
//! after picking the winner, exactly as `builtin.ts::raycast` does.

use crate::body::{Aabb, BodyMeta, ShapeKind};
use crate::jsmath::{add, axis, dot, js_abs, js_min, length, normalize, scale, sub};

/// A ray hit against one body, without the body handle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHitGeom {
    pub point: [f64; 3],
    pub normal: [f64; 3],
    pub distance: f64,
}

/// `normalizeVec3(direction)` plus the `lengthVec3(dir) < 0.5` gate.
///
/// `builtin.ts::raycast` normalises the direction and bails if the result is
/// degenerate (a zero or near-zero direction normalises to the zero vector,
/// whose length is 0 < 0.5). Returning `None` here lets the caller short-circuit
/// before touching any body.
pub fn prepare_dir(direction: [f64; 3]) -> Option<[f64; 3]> {
    let dir = normalize(direction);
    if length(dir) < 0.5 {
        None
    } else {
        Some(dir)
    }
}

/// `rayVsBody`: dispatch to the sphere or box test for one body.
pub fn ray_vs_body(
    origin: [f64; 3],
    dir: [f64; 3],
    meta: &BodyMeta,
    position: [f64; 3],
    aabb: &Aabb,
) -> Option<RayHitGeom> {
    match meta.shape.kind {
        ShapeKind::Sphere => ray_vs_sphere(origin, dir, position, meta.shape.radius),
        ShapeKind::Box => ray_vs_box(origin, dir, aabb),
    }
}

/// Analytic ray-sphere intersection, closest positive root.
fn ray_vs_sphere(
    origin: [f64; 3],
    dir: [f64; 3],
    center: [f64; 3],
    radius: f64,
) -> Option<RayHitGeom> {
    let to_center = sub(center, origin);
    let tca = dot(to_center, dir);
    let d2 = dot(to_center, to_center) - tca * tca;
    let r2 = radius * radius;
    if d2 > r2 {
        return None;
    }
    let thc = (r2 - d2).sqrt();
    let mut t = tca - thc;
    // Origin inside the sphere: the near root is behind us, take the far one.
    if t < 0.0 {
        t = tca + thc;
    }
    if t < 0.0 {
        return None;
    }
    let point = add(origin, scale(dir, t));
    let normal = normalize(sub(point, center));
    Some(RayHitGeom { point, normal, distance: t })
}

/// Slab ray-AABB intersection that also records the entered face for the normal.
fn ray_vs_box(origin: [f64; 3], dir: [f64; 3], aabb: &Aabb) -> Option<RayHitGeom> {
    let mut tmin = 0.0;
    let mut tmax = f64::INFINITY;
    let mut axis_idx: i32 = -1;
    let mut axis_sign = 1.0;
    for i in 0..3usize {
        let lo = aabb.min[i];
        let hi = aabb.max[i];
        let d = dir[i];
        let o = origin[i];
        if js_abs(d) < 1e-9 {
            // Ray parallel to this slab: it either never enters or is always in.
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let mut t1 = (lo - o) / d;
        let mut t2 = (hi - o) / d;
        let mut sign = -1.0;
        if t1 > t2 {
            core::mem::swap(&mut t1, &mut t2);
            sign = 1.0;
        }
        if t1 > tmin {
            tmin = t1;
            axis_idx = i as i32;
            axis_sign = sign;
        }
        tmax = js_min(tmax, t2);
        if tmin > tmax {
            return None;
        }
    }
    if axis_idx < 0 || tmin < 0.0 {
        return None;
    }
    let point = add(origin, scale(dir, tmin));
    let normal = axis(axis_idx as usize, axis_sign);
    Some(RayHitGeom { point, normal, distance: tmin })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Shape;

    fn sphere_meta(center: [f64; 3], r: f64) -> (BodyMeta, Aabb) {
        let m = BodyMeta {
            handle: 1,
            shape: Shape::sphere(r),
            inv_mass: 1.0,
            restitution: 0.1,
            friction: 0.7,
            group: 1,
            mask: u32::MAX,
        };
        let aabb = Aabb::around(center, &m.shape);
        (m, aabb)
    }

    fn box_meta(center: [f64; 3], half: [f64; 3]) -> (BodyMeta, Aabb) {
        let m = BodyMeta {
            handle: 1,
            shape: Shape::aabb_box(half),
            inv_mass: 1.0,
            restitution: 0.1,
            friction: 0.7,
            group: 1,
            mask: u32::MAX,
        };
        let aabb = Aabb::around(center, &m.shape);
        (m, aabb)
    }

    #[test]
    fn prepare_dir_rejects_degenerate_directions() {
        assert_eq!(prepare_dir([0.0, 0.0, 0.0]), None);
        // A unit vector normalises to length 1 >= 0.5.
        let d = prepare_dir([0.0, 0.0, -5.0]).unwrap();
        assert!((d[2] + 1.0).abs() < 1e-12);
        assert!((length(d) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ray_hits_sphere_near_face_with_outward_normal() {
        let (m, aabb) = sphere_meta([0.0, 0.0, -5.0], 1.0);
        // Fire from origin toward -Z; sphere centre is 5 units away, radius 1.
        let dir = prepare_dir([0.0, 0.0, -1.0]).unwrap();
        let hit = ray_vs_body([0.0, 0.0, 0.0], dir, &m, [0.0, 0.0, -5.0], &aabb).unwrap();
        // Near face is the centre minus one radius along the ray: z = -5 + 1 = -4,
        // four units from an origin at z = 0.
        assert!((hit.distance - 4.0).abs() < 1e-12, "near face at z=-4: {hit:?}");
        assert!((hit.point[2] + 4.0).abs() < 1e-12, "{hit:?}");
        assert!((hit.point[0]).abs() < 1e-12 && (hit.point[1]).abs() < 1e-12, "{hit:?}");
        // Normal points from centre to hit point => +Z.
        assert!((hit.normal[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ray_missing_sphere_returns_none() {
        let (m, aabb) = sphere_meta([0.0, 5.0, 0.0], 1.0);
        let dir = prepare_dir([0.0, 0.0, -1.0]).unwrap();
        assert!(ray_vs_body([0.0, 0.0, 0.0], dir, &m, [0.0, 0.0, 5.0], &aabb).is_none());
    }

    #[test]
    fn ray_from_inside_sphere_uses_far_root() {
        let (m, aabb) = sphere_meta([0.0, 0.0, 0.0], 2.0);
        let dir = prepare_dir([1.0, 0.0, 0.0]).unwrap();
        // Origin at centre: tca = 0, near root negative, far root = +radius.
        let hit = ray_vs_body([0.0, 0.0, 0.0], dir, &m, [0.0, 0.0, 0.0], &aabb).unwrap();
        assert!((hit.distance - 2.0).abs() < 1e-12, "{hit:?}");
    }

    #[test]
    fn ray_behind_sphere_returns_none() {
        let (m, aabb) = sphere_meta([0.0, 0.0, 5.0], 1.0);
        // Firing toward -Z away from a sphere at +Z: both roots negative.
        let dir = prepare_dir([0.0, 0.0, -1.0]).unwrap();
        assert!(ray_vs_body([0.0, 0.0, 0.0], dir, &m, [0.0, 0.0, 5.0], &aabb).is_none());
    }

    #[test]
    fn ray_hits_box_face_with_axis_normal() {
        let (m, aabb) = box_meta([0.0, 0.0, -5.0], [1.0, 1.0, 1.0]);
        let dir = prepare_dir([0.0, 0.0, -1.0]).unwrap();
        let hit = ray_vs_body([0.0, 0.0, 0.0], dir, &m, [0.0, 0.0, -5.0], &aabb).unwrap();
        // Box spans z in [-6,-4]; near face at z=-4, distance 4.
        assert!((hit.distance - 4.0).abs() < 1e-12, "{hit:?}");
        // Entered through the +Z face (we came from +Z), so normal is +Z.
        assert!((hit.normal[2] - 1.0).abs() < 1e-12, "{hit:?}");
    }

    #[test]
    fn ray_parallel_to_slab_outside_returns_none() {
        let (m, aabb) = box_meta([0.0, 0.0, -5.0], [1.0, 1.0, 1.0]);
        // Travelling along X at y=0,z=0, but the box is at z=-5: the Z slab has
        // d ~ 0 and origin z=0 is outside [-6,-4], so it must miss.
        let dir = prepare_dir([1.0, 0.0, 0.0]).unwrap();
        assert!(ray_vs_body([0.0, 0.0, 0.0], dir, &m, [0.0, 0.0, -5.0], &aabb).is_none());
    }

    #[test]
    fn ray_from_inside_a_box_is_not_a_hit() {
        let (m, aabb) = box_meta([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let dir = prepare_dir([1.0, 0.0, 0.0]).unwrap();
        // `builtin.ts` reports no hit for an origin inside the box, and so must
        // this port. `tmin` starts at 0 and the slab entry for an interior origin
        // is negative, so `t1 > tmin` never fires, `axis` stays -1, and the
        // `axis < 0` guard returns undefined. Worth pinning: an agent spawned
        // inside geometry gets "no floor here", not "the far wall".
        assert!(ray_vs_body([0.0, 0.0, 0.0], dir, &m, [0.0, 0.0, 0.0], &aabb).is_none());
        // Off-centre but still inside behaves the same way.
        assert!(ray_vs_body([0.25, -0.5, 0.0], dir, &m, [0.0, 0.0, 0.0], &aabb).is_none());
    }

    #[test]
    fn ray_entering_a_box_reports_the_face_it_crossed() {
        let (m, aabb) = box_meta([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // From x = -3 toward +X: the X slab entry is at x = -1, t = 2, and it is
        // the only slab that sets `axis`, so the normal is -X (the face entered).
        let dir = prepare_dir([1.0, 0.0, 0.0]).unwrap();
        let hit = ray_vs_body([-3.0, 0.0, 0.0], dir, &m, [0.0, 0.0, 0.0], &aabb).unwrap();
        assert!((hit.distance - 2.0).abs() < 1e-12, "{hit:?}");
        assert!((hit.point[0] + 1.0).abs() < 1e-12, "{hit:?}");
        assert_eq!(hit.normal, [-1.0, 0.0, 0.0]);
    }
}

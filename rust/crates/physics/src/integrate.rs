//! The two integration passes, and the only place this crate vectorises.
//!
//! `builtin.ts::step` has three loops that touch every body: integrate forces,
//! integrate velocities, and (inside the solver) the contact passes. The first
//! two update each body from itself and loop-invariant scalars, so body order
//! cannot affect a result and lane 0 of an `f64x2` cannot affect lane 1. The
//! third is a sequential Gauss-Seidel chain: contact *k* reads the velocity
//! contact *k-1* just wrote, so vectorising it would mean reordering it, which
//! would mean a different answer. Hence the split -- SIMD here, scalar in
//! `crate::solve`.
//!
//! Three primitives cover both passes:
//!
//! | TS expression                                   | primitive            |
//! |-------------------------------------------------|----------------------|
//! | `scale(add(vel, scale(gravity, dt)), damp)`      | [`affine_inplace`]   |
//! | `scale(angVel, spinDamp)`                        | [`mul_inplace`]      |
//! | `add(pos, scale(vel, dt))`                       | [`add_scaled_inplace`] |
//!
//! `mul_inplace` is not spelled as `affine_inplace` with `add = 0`, and that is
//! not pedantry: `(-0.0 + 0.0) * m` is `+0.0 * m` while `-0.0 * m` is `-0.0`,
//! and a sign flip in an angular velocity is exactly the kind of one-bit drift
//! the parity tests exist to catch.
//!
//! Every primitive has a `_ref` twin: an obvious scalar loop written without
//! `f64x2` at all. [`simd_parity_ok`] runs both over a fixed pseudo-random
//! dataset and compares bit patterns. That comparison runs natively under
//! `cargo test` (exercising the scalar `F64x2` backend) *and* inside the shipped
//! wasm through `td_selftest_simd_parity()` (exercising the real `v128` lanes),
//! which is what makes "SIMD128 does not change results" a checked claim rather
//! than an assumption about LLVM.

use crate::body::{StateArrays, PX, RX, VX, WX};
use crate::jsmath::js_max;
use crate::simd::F64x2;

/// `v[i] = (v[i] + add) * mul` for active slots; untouched otherwise.
///
/// `dynamic[i]` is `0` or `-1` (all bits set), the shape `v128_bitselect` wants,
/// so the mask reaches the SIMD path without a conversion. "Untouched" means
/// the lane is copied verbatim, not multiplied by one -- see [`crate::simd`].
pub fn affine_inplace(v: &mut [f64], dynamic: &[i64], add: f64, mul: f64) {
    let n = v.len();
    let add2 = F64x2::splat(add);
    let mul2 = F64x2::splat(mul);
    let mut i = 0;
    while i + 1 < n {
        let old = F64x2::new(v[i], v[i + 1]);
        let next = old.add(add2).mul(mul2);
        let (lo, hi) = next.select(old, (dynamic[i], dynamic[i + 1])).lanes();
        v[i] = lo;
        v[i + 1] = hi;
        i += 2;
    }
    if i < n {
        if dynamic[i] != 0 {
            v[i] = (v[i] + add) * mul;
        }
    }
}

/// Scalar reference for [`affine_inplace`].
pub fn affine_inplace_ref(v: &mut [f64], dynamic: &[i64], add: f64, mul: f64) {
    for i in 0..v.len() {
        if dynamic[i] != 0 {
            v[i] = (v[i] + add) * mul;
        }
    }
}

/// `v[i] = v[i] * mul` for active slots; untouched otherwise.
pub fn mul_inplace(v: &mut [f64], dynamic: &[i64], mul: f64) {
    let n = v.len();
    let mul2 = F64x2::splat(mul);
    let mut i = 0;
    while i + 1 < n {
        let old = F64x2::new(v[i], v[i + 1]);
        let (lo, hi) = old.mul(mul2).select(old, (dynamic[i], dynamic[i + 1])).lanes();
        v[i] = lo;
        v[i + 1] = hi;
        i += 2;
    }
    if i < n && dynamic[i] != 0 {
        v[i] *= mul;
    }
}

/// Scalar reference for [`mul_inplace`].
pub fn mul_inplace_ref(v: &mut [f64], dynamic: &[i64], mul: f64) {
    for i in 0..v.len() {
        if dynamic[i] != 0 {
            v[i] *= mul;
        }
    }
}

/// `dst[i] = dst[i] + src[i] * mul` for active slots; untouched otherwise.
pub fn add_scaled_inplace(dst: &mut [f64], src: &[f64], dynamic: &[i64], mul: f64) {
    let n = dst.len();
    let mul2 = F64x2::splat(mul);
    let mut i = 0;
    while i + 1 < n {
        let old = F64x2::new(dst[i], dst[i + 1]);
        let step = F64x2::new(src[i], src[i + 1]).mul(mul2);
        let (lo, hi) = old.add(step).select(old, (dynamic[i], dynamic[i + 1])).lanes();
        dst[i] = lo;
        dst[i + 1] = hi;
        i += 2;
    }
    if i < n && dynamic[i] != 0 {
        dst[i] += src[i] * mul;
    }
}

/// Scalar reference for [`add_scaled_inplace`].
pub fn add_scaled_inplace_ref(dst: &mut [f64], src: &[f64], dynamic: &[i64], mul: f64) {
    for i in 0..dst.len() {
        if dynamic[i] != 0 {
            dst[i] += src[i] * mul;
        }
    }
}

/// "Integrate forces": gravity, then linear damping, then angular damping.
///
/// `damp` and `spin_damp` are computed once rather than per body because they
/// depend only on `dt` and the world's damping coefficients -- `builtin.ts`
/// recomputes the identical expression inside its loop, and hoisting a
/// loop-invariant cannot change a floating-point result. Both use `Math.max(0, ..)`
/// semantics through [`js_max`], so a huge `dt` clamps to `+0.0` and not `-0.0`.
pub fn integrate_forces(
    state: &mut StateArrays,
    dynamic: &[i64],
    gravity: [f64; 3],
    dt: f64,
    linear_damping: f64,
    angular_damping: f64,
) {
    let damp = js_max(0.0, 1.0 - linear_damping * dt);
    let spin_damp = js_max(0.0, 1.0 - angular_damping * dt);
    for k in 0..3 {
        affine_inplace(&mut state.c[VX + k], dynamic, gravity[k] * dt, damp);
    }
    for k in 0..3 {
        mul_inplace(&mut state.c[WX + k], dynamic, spin_damp);
    }
}

/// Scalar reference for [`integrate_forces`].
pub fn integrate_forces_ref(
    state: &mut StateArrays,
    dynamic: &[i64],
    gravity: [f64; 3],
    dt: f64,
    linear_damping: f64,
    angular_damping: f64,
) {
    let damp = js_max(0.0, 1.0 - linear_damping * dt);
    let spin_damp = js_max(0.0, 1.0 - angular_damping * dt);
    for k in 0..3 {
        affine_inplace_ref(&mut state.c[VX + k], dynamic, gravity[k] * dt, damp);
    }
    for k in 0..3 {
        mul_inplace_ref(&mut state.c[WX + k], dynamic, spin_damp);
    }
}

/// "Integrate velocities": `pos += vel * dt`, `rot += angVel * dt`.
///
/// Reads `vel`/`angVel` and writes `pos`/`rot`, so the two channel groups are
/// distinct slices and the borrow checker is satisfied without a temporary.
/// Euler angles are integrated as three independent scalars, which is what the
/// reference does; it is not a valid rotation integrator and is not claimed to
/// be one (`builtin.ts` documents the same limitation).
pub fn integrate_velocities(state: &mut StateArrays, dynamic: &[i64], dt: f64) {
    for k in 0..3 {
        // Split borrow: channels 0..6 are `pos`+`rot`, 6..12 are `vel`+`spin`.
        let (lo, hi) = state.c.split_at_mut(VX);
        add_scaled_inplace(&mut lo[PX + k], &hi[k], dynamic, dt);
    }
    for k in 0..3 {
        let (lo, hi) = state.c.split_at_mut(VX);
        add_scaled_inplace(&mut lo[RX + k], &hi[WX - VX + k], dynamic, dt);
    }
}

/// Scalar reference for [`integrate_velocities`].
pub fn integrate_velocities_ref(state: &mut StateArrays, dynamic: &[i64], dt: f64) {
    for k in 0..3 {
        let (lo, hi) = state.c.split_at_mut(VX);
        add_scaled_inplace_ref(&mut lo[PX + k], &hi[k], dynamic, dt);
    }
    for k in 0..3 {
        let (lo, hi) = state.c.split_at_mut(VX);
        add_scaled_inplace_ref(&mut lo[RX + k], &hi[WX - VX + k], dynamic, dt);
    }
}

// ---------------------------------------------------------------------------
// Parity harness
// ---------------------------------------------------------------------------

/// Deterministic doubles, including the values that expose a lane bug.
///
/// An LCG rather than `rand` because this crate has no dependencies by design,
/// and because the dataset has to be identical on every host for the parity
/// claim to mean anything. The specials are the interesting half: `-0.0` catches
/// a select implemented as a multiply-by-one, `f64::MAX`/`MIN` catch saturation
/// in a lane merge, and the subnormals catch any pass through a narrower type.
fn dataset(n: usize) -> Vec<f64> {
    const SPECIALS: [f64; 8] = [
        0.0,
        -0.0,
        1e-320,
        -1e-320,
        f64::MAX,
        f64::MIN,
        f64::EPSILON,
        -f64::EPSILON,
    ];
    let mut seed: u64 = 0x5deece66d;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if i % 7 == 0 {
            out.push(SPECIALS[(i / 7) % SPECIALS.len()]);
            continue;
        }
        // 53 significant bits scaled into [-8, 8): enough range to overflow a
        // `* mul` and enough resolution to see a last-bit difference.
        let unit = (seed >> 11) as f64 / (1u64 << 53) as f64;
        out.push(unit * 16.0 - 8.0);
    }
    out
}

/// Mask with both parities and both lane positions populated, so a select wired
/// to the wrong lane fails here instead of in production.
fn masks(n: usize) -> Vec<i64> {
    (0..n).map(|i| if i % 3 == 0 { 0 } else { -1 }).collect()
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// True when every vectorised pass agrees bit for bit with its scalar reference.
///
/// Lengths 0..=9 are all checked, not one large one: the primitives have a scalar
/// tail, and an off-by-one there only shows up on an odd length.
pub fn simd_parity_ok() -> bool {
    let scalars = [0.0f64, 1.0, -1.0, 0.375, 1.0 / 3.0, -9.81, 60.0, 1e-6];
    for &n in &[0usize, 1, 2, 3, 4, 5, 7, 8, 9] {
        let base = dataset(n);
        let dynamic = masks(n);
        let src = dataset(n).into_iter().rev().collect::<Vec<_>>();

        for &add in &scalars {
            for &mul in &scalars {
                let mut a = base.clone();
                let mut b = base.clone();
                affine_inplace(&mut a, &dynamic, add, mul);
                affine_inplace_ref(&mut b, &dynamic, add, mul);
                if !same_bits(&a, &b) {
                    return false;
                }

                let mut a = base.clone();
                let mut b = base.clone();
                mul_inplace(&mut a, &dynamic, mul);
                mul_inplace_ref(&mut b, &dynamic, mul);
                if !same_bits(&a, &b) {
                    return false;
                }

                let mut a = base.clone();
                let mut b = base.clone();
                add_scaled_inplace(&mut a, &src, &dynamic, mul);
                add_scaled_inplace_ref(&mut b, &src, &dynamic, mul);
                if !same_bits(&a, &b) {
                    return false;
                }
            }
        }
    }
    world_pass_parity_ok()
}

/// The two full passes, compared end to end on a populated [`StateArrays`].
fn world_pass_parity_ok() -> bool {
    let n = 33;
    let dynamic = masks(n);
    let gravity = [0.0, -9.81, 0.3];
    let build = || {
        let mut s = StateArrays::new();
        s.resize(n);
        for k in 0..crate::body::CHAN {
            s.c[k].copy_from_slice(&dataset(n));
            // Perturb per channel and per slot, so every one of the twelve
            // arrays holds different numbers. Two passes that read a
            // cross-wired channel index then cannot cancel out.
            let skew = (k as f64 + 1.0) * 0.125;
            for (i, v) in s.c[k].iter_mut().enumerate() {
                *v = (*v + i as f64 * skew) * (1.0 + k as f64 * 1e-3);
            }
        }
        s
    };

    for &dt in &[1.0 / 60.0, 1.0 / 120.0, 0.5, 0.0] {
        let (mut a, mut b) = (build(), build());
        integrate_forces(&mut a, &dynamic, gravity, dt, 0.05, 0.2);
        integrate_forces_ref(&mut b, &dynamic, gravity, dt, 0.05, 0.2);
        integrate_velocities(&mut a, &dynamic, dt);
        integrate_velocities_ref(&mut b, &dynamic, dt);
        for k in 0..crate::body::CHAN {
            if !same_bits(&a.c[k], &b.c[k]) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectorised_passes_are_bit_identical_to_the_scalar_reference() {
        assert!(simd_parity_ok(), "a SIMD lane disagreed with the scalar spelling");
    }

    #[test]
    fn inactive_slots_are_copied_verbatim_including_signed_zero() {
        let dynamic = [0i64];
        for start in [-0.0f64, 0.0, 3.5, f64::MAX] {
            let mut v = vec![start];
            affine_inplace(&mut v, &dynamic, 1e30, -1e30);
            assert_eq!(v[0].to_bits(), start.to_bits());
            let mut v = vec![start];
            mul_inplace(&mut v, &dynamic, -1e30);
            assert_eq!(v[0].to_bits(), start.to_bits());
            let mut v = vec![start];
            add_scaled_inplace(&mut v, &[1e30], &dynamic, 1e30);
            assert_eq!(v[0].to_bits(), start.to_bits());
        }
    }

    #[test]
    fn mul_inplace_is_not_affine_with_a_zero_add() {
        // The reason the two primitives are separate. `-0.0 * 2` is `-0.0`,
        // while `(-0.0 + 0.0) * 2` is `+0.0`.
        let dynamic = [-1i64];
        let mut a = vec![-0.0f64];
        mul_inplace(&mut a, &dynamic, 2.0);
        let mut b = vec![-0.0f64];
        affine_inplace(&mut b, &dynamic, 0.0, 2.0);
        assert_eq!(a[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(b[0].to_bits(), 0.0f64.to_bits());
        assert_ne!(a[0].to_bits(), b[0].to_bits());
    }

    #[test]
    fn the_scalar_tail_runs_on_odd_lengths() {
        let dynamic = [-1i64, -1, -1];
        let mut v = vec![1.0, 2.0, 3.0];
        affine_inplace(&mut v, &dynamic, 1.0, 2.0);
        assert_eq!(v, vec![4.0, 6.0, 8.0]);

        let mut dst = vec![1.0, 2.0, 3.0];
        add_scaled_inplace(&mut dst, &[10.0, 20.0, 30.0], &dynamic, 0.5);
        assert_eq!(dst, vec![6.0, 12.0, 18.0]);
    }

    #[test]
    fn damping_clamps_at_positive_zero_for_an_absurd_dt() {
        // `Math.max(0, 1 - 0.05 * 1e9)` is `0`, and `js_max` keeps it `+0.0`.
        let dynamic = [-1i64];
        let mut s = StateArrays::new();
        s.resize(1);
        s.set(0, VX, [5.0, 5.0, 5.0]);
        integrate_forces(&mut s, &dynamic, [0.0, 0.0, 0.0], 1e9, 0.05, 0.2);
        assert_eq!(s.get(0, VX), [0.0, 0.0, 0.0]);
        assert_eq!(s.c[VX][0].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn forces_match_the_reference_spelling_operation_by_operation() {
        let dynamic = [-1i64];
        let (dt, g) = (1.0 / 60.0, -9.81f64);
        let damp = js_max(0.0, 1.0 - 0.05 * dt);
        let mut s = StateArrays::new();
        s.resize(1);
        s.set(0, VX, [0.25, 1.5, -0.75]);
        s.set(0, WX, [0.5, -0.5, 0.0]);
        integrate_forces(&mut s, &dynamic, [0.0, g, 0.0], dt, 0.05, 0.2);
        // `addVec3(vel, scaleVec3(gravity, dt))` then `scaleVec3(.., damp)`.
        assert_eq!(s.c[VX + 1][0].to_bits(), ((1.5 + g * dt) * damp).to_bits());
        assert_eq!(s.c[VX][0].to_bits(), ((0.25 + 0.0 * dt) * damp).to_bits());
        let spin_damp = js_max(0.0, 1.0 - 0.2 * dt);
        assert_eq!(s.c[WX + 1][0].to_bits(), (-0.5 * spin_damp).to_bits());
        // Untouched by gravity, and `-0.0` must stay `-0.0` rather than flip.
        assert_eq!(s.c[WX + 2][0].to_bits(), (0.0 * spin_damp).to_bits());
    }

    #[test]
    fn velocities_integrate_position_from_the_current_velocity() {
        let dynamic = [-1i64, 0];
        let dt = 1.0 / 60.0;
        let mut s = StateArrays::new();
        s.resize(2);
        s.set(0, PX, [1.0, 2.0, 3.0]);
        s.set(0, VX, [60.0, -60.0, 0.0]);
        s.set(1, PX, [9.0, 9.0, 9.0]);
        s.set(1, VX, [60.0, 60.0, 60.0]);
        integrate_velocities(&mut s, &dynamic, dt);
        assert_eq!(s.get(0, PX), [1.0 + 60.0 * dt, 2.0 + -60.0 * dt, 3.0 + 0.0 * dt]);
        // Slot 1 is static: untouched even though its velocity is non-zero.
        assert_eq!(s.get(1, PX), [9.0, 9.0, 9.0]);
    }
}

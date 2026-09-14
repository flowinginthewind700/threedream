//! Arithmetic that has to agree with JavaScript bit for bit.
//!
//! The acceptance criterion for M1 is that this solver and `src/physics/builtin.ts`
//! produce *identical* doubles for identical inputs. That is only achievable if
//! every operation here is the same operation, in the same order, with the same
//! edge-case semantics as the TS original. Three places where Rust and JS
//! genuinely differ, all handled here:
//!
//! 1. `Math.min` / `Math.max` propagate NaN; `f64::min` / `f64::max` return the
//!    other operand instead. The solver clamps impulses with them, so a NaN
//!    would silently become 0 under the Rust spelling.
//! 2. `Math.sqrt` and `f64::sqrt` are both required by IEEE-754 to be correctly
//!    rounded, so they agree; no wrapper needed beyond documenting it.
//! 3. Multiplication by a reciprocal is not division. `normalizeVec3` scales by
//!    `1 / len` rather than dividing by `len`, and the two differ in the last
//!    bit for most inputs, so `scale` and `normalize` keep the reciprocal form.
//!
//! Nothing here is clever on purpose. Reassociation, FMA contraction and
//! `x * 1.0` elision are all things a compiler may do under fast-math flags;
//! Rust enables none of them by default, and `.cargo/config.toml` deliberately
//! does not turn on `relaxed-simd`, which would license exactly that.

/// `Math.max(a, b)` with JS NaN *and* signed-zero semantics.
///
/// Both deviations from `f64::max` matter here. NaN because the solver clamps
/// impulses with it, and signed zero because `Math.max(+0, -0)` is `+0` while a
/// "return the second operand on ties" spelling yields `-0` -- and `-0.0` is a
/// different bit pattern that then propagates through every later multiply.
#[inline(always)]
pub fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else if b > a {
        b
    } else if a == 0.0 && b == 0.0 {
        0.0
    } else {
        a
    }
}

/// `Math.min(a, b)` with JS NaN and signed-zero semantics (`-0.0` wins a tie).
#[inline(always)]
pub fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else if b < a {
        b
    } else if a == 0.0 && b == 0.0 {
        -0.0
    } else {
        a
    }
}

#[inline(always)]
pub fn js_abs(a: f64) -> f64 {
    // `Math.abs` is sign-bit clearing; -0.0 -> 0.0. `f64::abs` is the same.
    a.abs()
}

/// `dotVec3`: `a0*b0 + a1*b1 + a2*b2`, left-associated exactly as JS parses it.
#[inline(always)]
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `lengthVec3`: `Math.sqrt(dotVec3(a, a))`.
#[inline(always)]
pub fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[inline(always)]
pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `scaleVec3`: `a[i] * s`, per component.
#[inline(always)]
pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `crossVec3`, in the reference's component order.
#[inline(always)]
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `normalizeVec3`: zero vector when the length is not clearly positive.
///
/// Note the reciprocal: `scaleVec3(a, 1 / len)`, not `a / len`.
#[inline(always)]
pub fn normalize(a: [f64; 3]) -> [f64; 3] {
    let len = length(a);
    if len > 1e-9 {
        scale(a, 1.0 / len)
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// `axisVec3(axis, sign)`: a unit vector along one axis.
#[inline(always)]
pub fn axis(axis: usize, sign: f64) -> [f64; 3] {
    let mut out = [0.0, 0.0, 0.0];
    out[axis] = sign;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_max_propagate_nan_like_javascript() {
        assert!(js_max(f64::NAN, 1.0).is_nan());
        assert!(js_max(1.0, f64::NAN).is_nan());
        assert!(js_min(f64::NAN, 1.0).is_nan());
        // ...which is exactly where the stdlib differs.
        assert_eq!(f64::NAN.max(1.0), 1.0);
    }

    #[test]
    fn min_max_pick_the_right_operand_for_finite_values() {
        assert_eq!(js_min(3.0, 5.0), 3.0);
        assert_eq!(js_max(3.0, 5.0), 5.0);
        assert_eq!(js_max(4.0, 4.0), 4.0);
    }

    #[test]
    fn min_max_resolve_signed_zero_the_way_javascript_does() {
        // Math.max(+0, -0) === +0 and Math.min(+0, -0) === -0, in both argument
        // orders. `assert_eq!` would pass on -0.0 == 0.0, so compare bits.
        assert_eq!(js_max(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(js_max(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(js_min(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(js_min(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn dot_is_left_associative() {
        // (a0*b0 + a1*b1) + a2*b2. A right-associated spelling of these three
        // terms differs in the last bit, which is the kind of drift this module
        // exists to prevent.
        let a = [0.1, 0.2, 0.3];
        let b = [0.7, 0.11, 0.13];
        let expected = (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
        assert_eq!(dot(a, b).to_bits(), expected.to_bits());
    }

    #[test]
    fn normalize_scales_by_the_reciprocal_and_degenerates_to_zero() {
        let n = normalize([3.0, 0.0, 4.0]);
        assert_eq!(n, [3.0 * (1.0 / 5.0), 0.0, 4.0 * (1.0 / 5.0)]);
        assert_eq!(normalize([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
        assert_eq!(normalize([1e-10, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn axis_builds_unit_vectors() {
        assert_eq!(axis(1, -1.0), [0.0, -1.0, 0.0]);
        assert_eq!(axis(2, 1.0), [0.0, 0.0, 1.0]);
    }
}

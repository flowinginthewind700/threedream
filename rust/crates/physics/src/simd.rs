//! Lane-exact `f64x2`, the only SIMD this solver is allowed to use.
//!
//! M1 asks for SIMD128 without giving up the bit-exactness contract against
//! `src/physics/builtin.ts`. Those two goals are compatible for exactly one
//! reason: a wasm `f64x2` lane performs the same IEEE-754 double operation on
//! the same inputs as the scalar instruction it replaces. Adding two packed
//! pairs is two adds, not a different add. So vectorising a loop that is
//! *elementwise* preserves every bit, while vectorising the contact solve would
//! not -- that one is a sequential Gauss-Seidel chain, and any reordering of it
//! changes results. Hence: SIMD in the integration passes (`crate::integrate`),
//! scalar in the solver (`crate::solve`).
//!
//! The same reasoning is why `.cargo/config.toml` enables `+simd128` and
//! pointedly does not enable `relaxed-simd`: relaxed mode lets the engine
//! substitute fused multiply-add and reassociate, which is a licence to move the
//! last bit.
//!
//! The native (non-wasm) implementation below is the scalar fallback used by
//! `cargo test`. It is not an approximation: `tests/simd_parity.rs` asserts the
//! vectorised passes in `crate::integrate` agree bit for bit with an
//! independently written scalar reference, and the wasm build runs the same
//! comparison from JS through `td_selftest_simd_parity()`. Those two together
//! are what make a native test run say something true about the shipped artifact.

/// True when *this* binary really carries wasm SIMD128.
///
/// Exported through the wasm ABI as `td_simd_enabled()` so a test can assert the
/// shipped artifact carries it, instead of trusting a build flag nobody checks.
/// Conjunction with `target_arch` on purpose: a native build can also have
/// `simd128` in its feature set on some hosts, and reporting that as "the wasm
/// kernel is vectorised" would be a lie.
pub const SIMD128: bool = cfg!(all(target_arch = "wasm32", target_feature = "simd128"));

/// The `v128` backend, and the gate that decides whether it may be used at all.
///
/// `core::arch::wasm32::f64x2_*` are not merely unsafe, they are *uncompilable*
/// unless the target feature is on. So the gate has to be the same conjunction
/// the intrinsics need: without `+simd128` a wasm32 build falls through to the
/// scalar `imp` below rather than failing, which keeps `cargo build --target
/// wasm32-unknown-unknown` working for anyone who has not picked up
/// `.cargo/config.toml` (cargo reads it from the working directory, not from the
/// manifest -- an easy thing to get wrong). `SIMD128` then reports `false` and
/// the TS side can tell the difference at runtime.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32::{
        f64x2, f64x2_add, f64x2_extract_lane, f64x2_mul, f64x2_splat, i64x2, v128, v128_bitselect,
    };

    /// Two f64 lanes in one wasm `v128`.
    #[derive(Clone, Copy)]
    pub struct F64x2(v128);

    // No `unsafe` anywhere below, and that is a fact about the pinned toolchain
    // rather than an accident: the `core::arch::wasm32` register intrinsics were
    // reclassified as safe (they touch registers only, never memory, so there is
    // no invariant for a caller to uphold). Wrapping them in `unsafe` blocks now
    // produces `unused_unsafe` warnings, so the blocks are gone. `rust-toolchain.
    // toml` pins 1.98.1, which is what makes "this crate contains no unsafe" a
    // durable claim instead of a version-dependent one.
    //
    // The wrapper still earns its place: it is the single point where the `cfg`
    // gate, the scalar fallback, and the lane-ordering convention meet, so the
    // integration passes read `a.mul(b).select(c, mask)` with no target-specific
    // spelling leaking into them.
    impl F64x2 {
        #[inline(always)]
        pub fn new(a: f64, b: f64) -> Self {
            // Built from two scalars rather than a `v128_load` of adjacent
            // memory: a `Vec<f64>` is 8-byte aligned and `v128` wants 16, so the
            // load spelling would need a misaligned pointer. LLVM folds this
            // pair of inserts into lane loads itself.
            Self(f64x2(a, b))
        }

        #[inline(always)]
        pub fn splat(v: f64) -> Self {
            Self(f64x2_splat(v))
        }

        #[inline(always)]
        pub fn add(self, rhs: Self) -> Self {
            Self(f64x2_add(self.0, rhs.0))
        }

        #[inline(always)]
        pub fn mul(self, rhs: Self) -> Self {
            Self(f64x2_mul(self.0, rhs.0))
        }

        /// Lane-wise select: take `self` where `mask` is all-ones, `other` where
        /// it is all-zero.
        ///
        /// This is how the integration passes reproduce `builtin.ts`'s
        /// `if (body.invMass === 0) continue` without a branch in the loop, and
        /// why the skip is bit-exact rather than "multiply by one": a select
        /// copies the untouched lane verbatim, so a static body's `-0.0`
        /// velocity stays `-0.0` instead of becoming `0.0`.
        #[inline(always)]
        pub fn select(self, other: Self, mask: (i64, i64)) -> Self {
            Self(v128_bitselect(self.0, other.0, i64x2(mask.0, mask.1)))
        }

        #[inline(always)]
        pub fn lanes(self) -> (f64, f64) {
            (f64x2_extract_lane::<0>(self.0), f64x2_extract_lane::<1>(self.0))
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod imp {
    /// Scalar stand-in for the wasm `v128` path. Same operations, same order.
    #[derive(Clone, Copy)]
    pub struct F64x2([f64; 2]);

    impl F64x2 {
        #[inline(always)]
        pub fn new(a: f64, b: f64) -> Self {
            Self([a, b])
        }

        #[inline(always)]
        pub fn splat(v: f64) -> Self {
            Self([v, v])
        }

        #[inline(always)]
        pub fn add(self, rhs: Self) -> Self {
            Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
        }

        #[inline(always)]
        pub fn mul(self, rhs: Self) -> Self {
            Self([self.0[0] * rhs.0[0], self.0[1] * rhs.0[1]])
        }

        #[inline(always)]
        pub fn select(self, other: Self, mask: (i64, i64)) -> Self {
            Self([
                if mask.0 != 0 { self.0[0] } else { other.0[0] },
                if mask.1 != 0 { self.0[1] } else { other.0[1] },
            ])
        }

        #[inline(always)]
        pub fn lanes(self) -> (f64, f64) {
            (self.0[0], self.0[1])
        }
    }
}

pub use imp::F64x2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_mul_are_lane_independent() {
        // The property the whole SIMD policy rests on: a packed op is two scalar
        // ops, so each lane must equal what the scalar spelling produces.
        let a = F64x2::new(0.1, -2.5);
        let b = F64x2::new(0.3, 4.0);
        assert_eq!(a.add(b).lanes(), (0.1 + 0.3, -2.5 + 4.0));
        assert_eq!(a.mul(b).lanes(), (0.1 * 0.3, -2.5 * 4.0));
        assert_eq!(F64x2::splat(3.0).lanes(), (3.0, 3.0));
    }

    #[test]
    fn select_copies_the_untouched_lane_verbatim_including_signed_zero() {
        let kept = F64x2::new(1.0, 2.0);
        let other = F64x2::new(-0.0, f64::NAN);
        // Mask lane all-ones takes from `kept`, all-zeros takes from `other` --
        // and "takes" means a verbatim bit copy, not an arithmetic identity. That
        // is what lets the integration passes skip a static body without turning
        // its `-0.0` velocity into `0.0`, which a `* 1.0` spelling would do.
        let (lo, hi) = kept.select(other, (-1, 0)).lanes();
        assert_eq!(lo.to_bits(), 1.0f64.to_bits());
        assert!(hi.is_nan(), "lane 1 must come from `other` unchanged");

        let (lo, hi) = kept.select(other, (0, -1)).lanes();
        assert_eq!(lo.to_bits(), (-0.0f64).to_bits());
        assert_eq!(hi.to_bits(), 2.0f64.to_bits());
    }

    #[test]
    fn simd128_flag_matches_the_target_this_binary_was_built_for() {
        // Not a useful assertion on the host -- the point is that the constant
        // exists, is exported through the ABI, and is checked against the real
        // wasm bytes by `tests/wasm_backend.test.ts`.
        assert_eq!(SIMD128, cfg!(all(target_arch = "wasm32", target_feature = "simd128")));
        // ...which on a native `cargo test` run is necessarily false.
        #[cfg(not(target_arch = "wasm32"))]
        assert!(!SIMD128);
    }
}

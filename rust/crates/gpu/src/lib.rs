//! Placeholder for the M2 shared-`GPUDevice` bridge.
//!
//! This crate exists in M0 for one reason: to prove, in CI, that the pinned
//! `wgpu = "=25.0.2"` really compiles to `wasm32-unknown-unknown` and that the
//! artifact is the size the feasibility study measured (~1.04 MB raw). The
//! study's warning is that this is a *version* claim, not a general one --
//! v25 has no `from_web` interop, so the M2 bridge will share the device by
//! passing three.js's `GPUDevice` through `wgpu::Device::from_web`-style
//! plumbing that v25 spells differently. Getting that wrong in M2 is a rewrite;
//! finding out in M0 that the pin builds at all is cheap.
//!
//! Nothing here is reachable from the physics path, and nothing here is
//! reachable from a native build.

#![cfg_attr(not(target_arch = "wasm32"), forbid(unsafe_code))]

/// ABI version of the GPU bridge. Bumped when the exported surface changes.
pub const GPU_ABI_VERSION: u32 = 0;

/// The wgpu surface, compiled only for wasm32.
///
/// Kept behind `cfg` rather than behind a cargo feature because there is no
/// host build of this crate to speak of: a feature flag would imply a
/// combination (native + wgpu) that no test needs and that would drag Vulkan
/// loaders into every `cargo test` in the workspace.
#[cfg(target_arch = "wasm32")]
pub mod bridge {
    /// What the M2 bridge has to be able to do with a device three.js already
    /// owns. Enumerated now so the wgpu version question is answered against a
    /// concrete requirement list rather than discovered during implementation.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum BridgeCapability {
        /// Wrap an existing `GPUDevice` instead of requesting a second one.
        AdoptWebGpuDevice,
        /// Bind a `wgpu::Buffer` that three.js also has a `GPUBuffer` for.
        ShareBuffer,
        /// Run an externally supplied WGSL compute shader on that device.
        ExternalComputePipeline,
    }

    /// The wgpu entry point this crate will use. Referenced (not called) so a
    /// wgpu version bump that renames or retypes it fails the wasm build here,
    /// in M0, rather than in M2 when the bridge is half written.
    pub fn instance_descriptor() -> wgpu::InstanceDescriptor {
        wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_crate_compiles_on_the_host_without_wgpu() {
        // The only host-runnable assertion this crate can honestly make: the
        // workspace member exists and its non-wasm surface is inert.
        assert_eq!(GPU_ABI_VERSION, 0);
    }
}

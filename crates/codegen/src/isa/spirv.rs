//! Compatibility exports for callers of the former SPIR-V-named backend.
//!
//! Implementation and shader realization live exclusively in `isa::naga`.
//! Remove this path after Fe and downstream users migrate to the Naga API.

pub use super::naga::*;
pub use super::naga::{NagaBackend as SpirvBackend, ShaderArtifact as SpirvArtifact};

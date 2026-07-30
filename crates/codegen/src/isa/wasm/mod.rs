//! WASM backend: Sonatina IR → WebAssembly module via WAFFLE.
//!
//! Translates Sonatina IR to WAFFLE's SSA IR (block params, operators),
//! then WAFFLE handles control flow recovery (Ramsey's algorithm) and
//! WASM emission automatically.

mod translate;

use std::collections::HashMap;

use sonatina_ir::Module;

use crate::backend::Backend;

#[derive(Debug)]
pub enum WasmError {
    UnsupportedTarget(String),
    Translation(String),
    UnsupportedType(String),
    UnsupportedInstruction(String),
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedTarget(msg) => write!(f, "unsupported target: {msg}"),
            Self::Translation(msg) => write!(f, "wasm translation error: {msg}"),
            Self::UnsupportedType(msg) => write!(f, "unsupported type for wasm: {msg}"),
            Self::UnsupportedInstruction(msg) => write!(f, "unsupported wasm instruction: {msg}"),
        }
    }
}

pub struct WasmArtifact {
    pub bytes: Vec<u8>,
    pub func_names: Vec<String>,
}

/// Opt-in checked LIFO canonical-memory exports synthesized by the Wasm backend.
///
/// Post-return names are operation identities selected by the frontend. Each
/// export releases exactly one live result allocation and traps on stale,
/// out-of-order, malformed, or out-of-bounds descriptors. This is intentionally
/// a stack allocator, not a general-purpose `realloc`: only the newest live
/// allocation may be resized or released. It is suitable for generator-owned
/// result lowering that guarantees reverse-order post-return cleanup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalStackMemoryManifest {
    pub post_return_exports: Vec<String>,
}

impl CanonicalStackMemoryManifest {
    pub fn new(post_return_exports: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            post_return_exports: post_return_exports.into_iter().map(Into::into).collect(),
        }
    }
}

pub struct WasmBackend {
    /// Per-import-symbol wasm import MODULE names, keyed by the Sonatina function
    /// symbol (which becomes the import field name). An external declaration whose
    /// symbol is absent from this table falls back to the `"fe"` v0 convention, so
    /// an empty table reproduces the pre-attribute behavior exactly. This is a
    /// SIDE TABLE consulted only in import emission; it touches no Sonatina IR
    /// type and no symbol interning.
    import_modules: HashMap<String, String>,
    canonical_arena: bool,
    canonical_memory: Option<CanonicalStackMemoryManifest>,
}

impl WasmBackend {
    pub fn new() -> Self {
        Self {
            import_modules: HashMap::new(),
            canonical_arena: false,
            canonical_memory: None,
        }
    }

    /// Emit canonical browser-interface arena exports. Disabled by default.
    pub fn with_canonical_arena(mut self) -> Self {
        self.canonical_arena = true;
        self
    }

    /// Emit checked LIFO `cabi_realloc`/post-return exports for a
    /// generator-controlled result stack. Disabled by default and mutually
    /// exclusive with the legacy resettable arena.
    pub fn with_canonical_stack_memory(
        mut self,
        manifest: CanonicalStackMemoryManifest,
    ) -> Self {
        self.canonical_memory = Some(manifest);
        self
    }

    /// Attach a symbol -> import-module table. A frontend that names an
    /// `extern` block's import module (e.g. Fe's `#[wasm_import(module = "...")]`)
    /// passes it here so the emitted import lands as `(<module>, <symbol>)`
    /// instead of the flat `("fe", <symbol>)` default.
    pub fn with_import_modules(mut self, import_modules: HashMap<String, String>) -> Self {
        self.import_modules = import_modules;
        self
    }
}

impl Backend for WasmBackend {
    type Artifact = WasmArtifact;
    type Error = WasmError;

    fn compile_module(&self, module: &Module) -> Result<Self::Artifact, Vec<Self::Error>> {
        if self.canonical_arena && self.canonical_memory.is_some() {
            return Err(vec![WasmError::Translation(
                "legacy canonical arena and canonical-memory manifest are mutually exclusive"
                    .to_string(),
            )]);
        }
        let (wasm_module, func_names) =
            translate::translate_module(
                module,
                &self.import_modules,
                self.canonical_arena,
                self.canonical_memory.as_ref(),
            )
                .map_err(|e| vec![WasmError::Translation(e)])?;

        let bytes = wasm_module
            .to_wasm_bytes()
            .map_err(|e| vec![WasmError::Translation(format!("WAFFLE emission: {e}"))])?;

        Ok(WasmArtifact { bytes, func_names })
    }
}

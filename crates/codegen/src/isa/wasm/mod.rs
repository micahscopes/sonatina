#![doc = include_str!("../../../docs/wasm.md")]

mod arith;
mod translate;

use sonatina_ir::Module;
use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

use crate::backend::{Backend, BackendOptions};

/// The import module used for `extern` functions unless
/// [`WasmBackend::with_import_module`] selects another.
pub const DEFAULT_IMPORT_MODULE: &str = "env";

#[derive(Debug)]
pub enum WasmError {
    /// The module does not target `wasm32-unknown-native`.
    UnsupportedTarget(String),
    /// The module uses an instruction, type or declaration that this backend
    /// does not translate. The message names it.
    Unsupported(String),
    /// The module is malformed in a way that translation cannot repair, such
    /// as an operand of the wrong type or a call with the wrong arity.
    Invalid(String),
    /// WAFFLE failed to emit the module.
    Emission(String),
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedTarget(message) => write!(f, "unsupported target: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported by the wasm backend: {message}"),
            Self::Invalid(message) => write!(f, "invalid input for the wasm backend: {message}"),
            Self::Emission(message) => write!(f, "wasm emission failed: {message}"),
        }
    }
}

impl std::error::Error for WasmError {}

/// An encoded WebAssembly module.
#[derive(Debug)]
pub struct WasmArtifact {
    bytes: Vec<u8>,
    exports: Vec<String>,
}

impl WasmArtifact {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Names of the exported functions, in module order. The linear memory is
    /// also exported, as `memory`.
    pub fn exported_functions(&self) -> &[String] {
        &self.exports
    }
}

#[derive(Debug, Clone)]
pub struct WasmBackend {
    import_module: String,
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmBackend {
    pub fn new() -> Self {
        Self {
            import_module: DEFAULT_IMPORT_MODULE.to_string(),
        }
    }

    /// Select the import module under which every `extern` function is
    /// imported. The import field name is the function's symbol.
    pub fn with_import_module(mut self, module: impl Into<String>) -> Self {
        self.import_module = module.into();
        self
    }
}

impl Backend for WasmBackend {
    type Artifact = WasmArtifact;
    type Error = WasmError;

    fn compile_module(
        &self,
        module: &Module,
        _options: &BackendOptions,
    ) -> Result<Self::Artifact, Vec<Self::Error>> {
        ensure_wasm32_target(module.ctx.triple).map_err(|error| vec![error])?;

        let translated = translate::translate_module(module, &self.import_module)
            .map_err(|error| vec![error])?;
        let bytes = translated
            .module
            .to_wasm_bytes()
            .map_err(|error| vec![WasmError::Emission(error.to_string())])?;

        Ok(WasmArtifact {
            bytes,
            exports: translated.exports,
        })
    }
}

fn ensure_wasm32_target(triple: TargetTriple) -> Result<(), WasmError> {
    if triple.architecture == Architecture::Wasm32
        && triple.vendor == Vendor::Unknown
        && triple.operating_system == OperatingSystem::Native
    {
        Ok(())
    } else {
        Err(WasmError::UnsupportedTarget(format!(
            "the wasm backend requires wasm32-unknown-native, not {triple}"
        )))
    }
}

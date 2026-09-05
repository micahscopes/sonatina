//! Checked environment and output selection, independent of shader stage.

use super::SpirvError;
use sonatina_ir::module::FuncRef;

/// Baseline limit for the currently supported WebGPU profile. Includes
/// compiler-owned channels, not just authored resources.
pub const WEBGPU_STORAGE_BUFFERS_PER_STAGE: usize = 8;

#[cfg(all(test, feature = "spirv-backend"))]
mod resource_tests {
    use super::*;

    fn check(source: &str) -> Result<(), SpirvError> {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty()).validate(&module).unwrap();
        validate_resource_limits(&module, &info)
    }

    #[test]
    fn webgpu_resource_limits_use_entry_liveness_not_module_union() {
        let declarations: String = (0..16).map(|i| format!(
            "@group(0) @binding({i}) var<storage, read_write> b{i}: u32;\n"
        )).collect();
        let entry = |name: &str, first: usize, end: usize| format!(
            "@compute @workgroup_size(1) fn {name}() {{ b{first} = {}; }}\n",
            (first+1..end).map(|i| format!("b{i}")).collect::<Vec<_>>().join(" + ")
        );
        // Sixteen declared buffers, eight used by each entry. Unused globals
        // and another entry's resources cannot consume this entry's budget.
        check(&format!("{declarations}{}{}", entry("a", 0, 8), entry("b", 8, 16))).unwrap();
        let error = check(&format!("{declarations}{}", entry("a", 0, 9))).unwrap_err();
        assert!(error.to_string().contains("uses 9 storage buffers"));
        assert!(error.to_string().contains("(0, 8)"));
    }

    #[test]
    fn webgpu_resource_limits_include_helper_and_atomic_channels() {
        let declarations: String = (0..8).map(|i| format!(
            "@group(0) @binding({i}) var<storage, read_write> b{i}: u32;\n"
        )).collect();
        let source = format!("{declarations}
            @group(0) @binding(8) var<storage, read_write> status: atomic<u32>;
            fn helper() -> u32 {{ return atomicLoad(&status); }}
            @compute @workgroup_size(1) fn main() {{
                b0 = b1 + b2 + b3 + b4 + b5 + b6 + b7 + helper();
            }}");
        assert!(check(&source).unwrap_err().to_string().contains("uses 9 storage buffers"));
    }
}

#[cfg(feature = "spirv-backend")]
pub(super) fn validate_resource_limits(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<(), SpirvError> {
    for (index, entry) in module.entry_points.iter().enumerate() {
        let uses = info.get_entry_point(index);
        let bindings: Vec<_> = module.global_variables.iter().filter_map(|(handle, global)| {
            if uses[handle].is_empty() || !matches!(global.space, naga::AddressSpace::Storage { .. }) {
                return None;
            }
            global.binding.as_ref().map(|binding| (binding.group, binding.binding))
        }).collect();
        if bindings.len() > WEBGPU_STORAGE_BUFFERS_PER_STAGE {
            return Err(SpirvError::Validation(format!(
                "WebGPU entry `{}` ({:?}) uses {} storage buffers, exceeding the portable per-stage limit {}; bindings={bindings:?}",
                entry.name, entry.stage, bindings.len(), WEBGPU_STORAGE_BUFFERS_PER_STAGE,
            )));
        }
    }
    Ok(())
}

/// Semantic pipeline selection. Legacy envelopes are named adapters, not
/// additional hardware shader stages.
#[derive(Debug, Clone, Copy)]
pub enum ShaderPipeline {
    Compute { entry: FuncRef, workgroup_size: [u32; 3], dispatch_grid: [u32; 3] },
    Raster { vertex: FuncRef, fragment: FuncRef },
    Fullscreen { entry: FuncRef },
    LegacyScalar { entry: FuncRef, workgroup_size: [u32; 3] },
    LegacyGrid { entry: FuncRef, workgroup_size: [u32; 3] },
}

/// Complete shader request with no ambient backend stage-selection flags.
pub struct ShaderCompileRequest<'a> {
    pub target: &'a ShaderTargetContract,
    pub pipeline: ShaderPipeline,
    pub resources: &'a [super::SpirvExternalResource],
    pub builtin_arguments: &'a [super::SpirvBuiltinArgument],
    pub private_heap_words: u32,
    /// Shared, zero-initialized atomic status for one graph execution epoch.
    /// The caller owns reset between epochs, never between dependent dispatches.
    pub graph_failure: Option<GraphFailureBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphFailureBinding {
    pub group: u32,
    pub binding: u32,
}

impl<'a> ShaderCompileRequest<'a> {
    pub fn new(target: &'a ShaderTargetContract, pipeline: ShaderPipeline) -> Self {
        Self { target, pipeline, resources: &[], builtin_arguments: &[], private_heap_words: 8192, graph_failure: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderEnvironment {
    WebGpu,
    Vulkan,
    WebGl2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderEncoding {
    Wgsl,
    Spirv,
    GlslEs,
}

/// The currently supported WebGPU profile requires no optional capabilities.
/// Other environments remain explicit rejections until their profiles and
/// execution gates exist. Selecting an encoding does not select an environment.
#[derive(Debug, Clone)]
pub struct ShaderTargetContract {
    environment: ShaderEnvironment,
    encodings: Vec<ShaderEncoding>,
}

impl ShaderTargetContract {
    pub fn new(
        environment: ShaderEnvironment,
        encodings: impl IntoIterator<Item = ShaderEncoding>,
    ) -> Result<Self, SpirvError> {
        if environment != ShaderEnvironment::WebGpu {
            return Err(SpirvError::UnsupportedTarget(format!(
                "shader environment {environment:?} has no implemented capability profile"
            )));
        }
        let mut selected = Vec::new();
        for encoding in encodings {
            if encoding == ShaderEncoding::GlslEs {
                return Err(SpirvError::UnsupportedTarget(
                    "GLSL ES output is not supported by the WebGPU profile".to_owned(),
                ));
            }
            if !selected.contains(&encoding) {
                selected.push(encoding);
            }
        }
        if selected.is_empty() {
            return Err(SpirvError::UnsupportedTarget(
                "a shader target must request at least one encoding".to_owned(),
            ));
        }
        Ok(Self { environment, encodings: selected })
    }

    pub fn environment(&self) -> ShaderEnvironment {
        self.environment
    }

    pub fn encodings(&self) -> &[ShaderEncoding] {
        &self.encodings
    }

    pub(super) fn requests(&self, encoding: ShaderEncoding) -> bool {
        self.encodings.contains(&encoding)
    }
}

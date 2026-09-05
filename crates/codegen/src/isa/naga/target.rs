//! Checked environment and output selection, independent of shader stage.

use super::SpirvError;

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

//! Naga backend: Sonatina IR to shader modules.
//!
//! Translates Sonatina IR to Naga's expression DAG + statement tree IR,
//! then emits SPIR-V and browser WGSL from that validated representation.
//! The historical `spirv-backend` feature name is retained during migration.

use sonatina_ir::Module;

mod target;
pub use target::{ShaderCompileRequest, ShaderEncoding, ShaderEnvironment, ShaderPipeline, ShaderTargetContract};

#[cfg(feature = "spirv-backend")]
use sonatina_ir::ir_writer::FuncWriter;

use crate::backend::Backend;

#[cfg(feature = "spirv-backend")]
use crate::optim::dead_arg::analyze_live_arguments;

#[cfg(feature = "spirv-backend")]
mod authored_raster;

#[cfg(feature = "spirv-backend")]
mod helper_plan;

#[cfg(feature = "spirv-backend")]
pub use helper_plan::{HelperBodyPlan, analyze_helper_body};

#[derive(Debug)]
pub enum SpirvError {
    UnsupportedTarget(String),
    Translation(String),
    Validation(String),
}

impl std::fmt::Display for SpirvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedTarget(msg) => write!(f, "unsupported target: {msg}"),
            Self::Translation(msg) => write!(f, "spirv translation error: {msg}"),
            Self::Validation(msg) => write!(f, "spirv validation error: {msg}"),
        }
    }
}

/// The word scalar the shader was emitted at, content-derived from the kernel's
/// Sonatina return type. `U32` -> naga `Uint`/width 4 (browser profile, no
/// SHADER_INT64); `I64` -> naga `Sint`/width 8 (the original path, bit-for-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordKind {
    U32,
    I64,
}

impl WordKind {
    /// Word width in bytes (4 for u32, 8 for i64). Drives struct spans, array
    /// strides, member offsets and the result readback width.
    pub fn width_bytes(self) -> u32 {
        match self {
            WordKind::U32 => 4,
            WordKind::I64 => 8,
        }
    }
}

/// Output-buffer shape: a single-value `Output` struct (`Scalar`), a dynamic
/// `OutputArray` written per-invocation via `ObjAlloc` (`Batch`), or a dynamic
/// `OutputArray` written once per grid invocation at `output[gid.y * row_width +
/// gid.x]` (`Grid`). Grid mode is driver-declared (there is no content signal),
/// the same treatment as workgroup size: args 0,1 are the grid coordinates, args
/// 2.. are broadcast inputs, the return value is stored per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    Scalar,
    Batch,
    Grid,
    /// Unit-returning compute stage with explicit external resources and no
    /// implicit result buffer. Unlike Grid and Batch, this mode is never
    /// inferred from the function body; the compiler supplies the stage
    /// interface explicitly.
    Compute,
    /// Render mode: ONE SPIR-V module with two entry points, a fixed
    /// fullscreen-triangle `@vertex` and a `@fragment` that binds args 0,1 to
    /// `u32(position.xy)` (the analog of Grid's `global_invocation_id.xy`), runs
    /// the mode-blind body unchanged, and returns `unpack4x8unorm(result)` as an
    /// `@location(0) vec4<f32>` color. Driver-declared (`with_render()`), off by
    /// default; there is no output storage buffer.
    Render,
}

/// Storage access a binding is declared with, mirroring the emitted naga
/// `GlobalVariable` address space exactly (never re-derived downstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    ReadWrite,
}

/// Whether a binding carries the kernel result (`Output`) or the kernel inputs
/// (`Input`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Output,
    Input,
    Resource,
}

/// Shader stages that can reach one physical binding. This is measured while
/// lowering the same entry points that produce the binding, so downstream
/// runtimes do not need to widen visibility from the enclosing pass kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpirvShaderStage {
    Compute,
    Vertex,
    Fragment,
}

/// Logical Sonatina scalar type of one ABI value. This describes the source
/// argument, not necessarily the shader expression's physical type: for
/// example Grid coordinates are logically `I32` while
/// `global_invocation_id` is physically `u32`, and Render coordinates are
/// logically `I32` after conversion from physical fragment-position `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpirvScalarKind {
    I1,
    I32,
    U32,
    I64,
    F32,
}

/// One source-language argument materialized in a storage-buffer member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpirvBindingMember {
    pub arg_index: u32,
    pub offset: u32,
    pub width: u32,
    pub scalar: SpirvScalarKind,
}

/// One field in the element record of an externally bound storage resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpirvResourceField {
    pub name: String,
    pub scalar: SpirvScalarKind,
    pub offset: u32,
}

/// Compiler-supplied storage element layout. B4 deliberately starts with the
/// browser u32 carrier and POD records of browser words. This is enough for
/// packed f32 pairs and crypto word buffers without conflating external
/// storage with the private `Mem` heap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpirvResourceElement {
    Scalar(SpirvScalarKind),
    Record {
        fields: Vec<SpirvResourceField>,
        span: u32,
    },
}

/// One explicit storage resource rooted at a function argument. `arg_index`
/// is semantic compiler metadata; the translator maps that argument value to
/// the emitted Naga global before ordinary ObjIndex/ObjProj/ObjLoad/ObjStore
/// lowering runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpirvExternalResource {
    pub arg_index: u32,
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub access: Access,
    pub element: SpirvResourceElement,
    pub stride: u32,
    pub length: u32,
}

/// Physical shader builtin supplying a logical source-language argument without
/// buffer storage. [`SpirvBuiltinInput::scalar`] records the logical type after
/// the backend's builtin conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpirvBuiltinSource {
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    NumWorkgroupsX,
    NumWorkgroupsY,
    NumWorkgroupsZ,
    LocalInvocationIndex,
    FragmentPositionX,
    FragmentPositionY,
    VertexIndex,
    InstanceIndex,
}

/// A source-language scalar argument supplied by one physical shader builtin.
///
/// The backend validates the source against the selected stage, the argument
/// index against the lowered signature, and the argument's exact scalar type.
/// [`SpirvBuiltinInput`] is the resulting measured layout fact, including the
/// scalar type derived by that validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpirvBuiltinArgument {
    pub arg_index: u32,
    pub source: SpirvBuiltinSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpirvBuiltinInput {
    pub arg_index: u32,
    pub source: SpirvBuiltinSource,
    pub scalar: SpirvScalarKind,
}

/// One storage-buffer binding, as the compiler actually emitted it.
#[derive(Debug, Clone)]
pub struct SpirvBinding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub access: Access,
    pub role: Role,
    /// Exact entry-point stages that can reach this physical binding.
    pub stages: Vec<SpirvShaderStage>,
    /// Byte distance between consecutive binding elements. For the current
    /// structs this equals `span`; keeping it distinct makes array layout
    /// explicit if tail padding or a larger array stride is introduced later.
    pub stride: u32,
    /// Bytes occupied by one emitted binding element, including internal and
    /// tail padding but excluding any gap before the next array element.
    pub span: u32,
    /// Typed source arguments stored in this binding. Empty for outputs and
    /// padding-only inputs.
    pub members: Vec<SpirvBindingMember>,
    /// Element layout for an authored external resource. `None` for implicit
    /// input/output/diagnostic bindings.
    pub resource_element: Option<SpirvResourceElement>,
    /// Declared element count for an authored external resource.
    pub resource_length: Option<u32>,
    /// Kernel argument rooted at this resource global. `None` for implicit
    /// compiler bindings.
    pub resource_arg_index: Option<u32>,
}

/// Where the scalar result lands for readback.
#[derive(Debug, Clone, Copy)]
pub struct SpirvResult {
    pub group: u32,
    pub binding: u32,
    pub offset: u32,
    pub width: u32,
}

/// The compiler-stated ABI of a compiled SPIR-V compute module. It is populated
/// from the SAME values `translate_to_naga` used to emit the naga module, so the
/// artifact describes its own binding layout and nothing downstream re-derives
/// it. Plain Rust (no serde in the fork); the fe side serializes it.
#[derive(Debug, Clone)]
pub struct SpirvLayout {
    pub entry_point: String,
    pub mode: LayoutMode,
    pub workgroup_size: [u32; 3],
    pub word: WordKind,
    pub bindings: Vec<SpirvBinding>,
    /// Source arguments supplied by shader builtins rather than bindings.
    pub builtin_inputs: Vec<SpirvBuiltinInput>,
    /// Where the scalar result lands for readback (`Some` for Scalar and Batch
    /// modes). `None` in Grid and Render modes: the whole output array (Grid) or
    /// the color target (Render) is the result, so there is no single readback slot.
    pub result: Option<SpirvResult>,
    /// Where the trap-status lane lands for readback. Explicit compute owns one
    /// word per invocation across its fixed workgroup and dispatch shape, so
    /// independent invocations never race on a diagnostic store. Scalar mode
    /// retains one word. Grid exposes the whole binding rather than a single
    /// result descriptor. Kernels without a reachable trap omit the channel.
    pub trap: Option<SpirvResult>,
    /// Render mode: the `@vertex` entry point name (`None` for compute modes).
    pub vertex_entry: Option<String>,
    /// Render mode: the `@fragment` entry point name (`None` for compute modes).
    pub fragment_entry: Option<String>,
    /// Render mode: the color-target texture format the fragment writes
    /// (`Some("rgba8unorm")`); `None` for compute modes.
    pub color_target_format: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpirvRasterPipeline {
    pub vertex_entry: String,
    pub fragment_entry: String,
}

pub struct ShaderArtifact {
    /// Empty when SPIR-V was not requested by an explicit target contract.
    pub words: Vec<u32>,
    /// WGSL source for wgpu execution (available when spirv-backend feature is on)
    pub wgsl: Option<String>,
    /// The compiler-stated ABI: entry point, mode, workgroup size, word kind,
    /// bindings and result location. Emitted from the same values the naga module
    /// was built from.
    pub layout: SpirvLayout,
}

impl ShaderArtifact {
    pub fn as_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

pub struct NagaBackend {
    pub workgroup_size: [u32; 3],
    /// Fixed dispatch grid in workgroups. Explicit compute uses this together
    /// with `workgroup_size` to size compiler-owned per-invocation channels.
    /// The default is one workgroup in each dimension.
    pub dispatch_grid: [u32; 3],
    /// Grid mode: one invocation per pixel, args 0,1 bound to
    /// `global_invocation_id.xy`, the return value stored at
    /// `output[gid.y * (num_workgroups.x * workgroup_size[0]) + gid.x]`. A
    /// driver-declared envelope fact (no content signal), off by default.
    pub grid: bool,
    /// Render mode: emit ONE module with a fixed fullscreen-triangle `@vertex`
    /// and a `@fragment` that binds args 0,1 to `u32(position.xy)`, runs the
    /// mode-blind body, and returns `unpack4x8unorm(result)` as an
    /// `@location(0) vec4<f32>` color. Driver-declared, off by default; mutually
    /// exclusive with grid and batch.
    pub render: bool,
    /// Explicit unit-returning compute stage. This is distinct from legacy
    /// Grid, whose return value is written to an implicit output image buffer.
    pub compute: bool,
    /// Paired source-authored vertex and fragment bodies. Their flattened
    /// signatures are checked as one raster interface by the backend.
    pub authored_raster: Option<SpirvRasterPipeline>,
    /// Bound resource roots supplied by the Fe stage-interface derivation.
    pub external_resources: Vec<SpirvExternalResource>,
    /// Logical source arguments supplied by physical shader builtins. The
    /// stage-interface derivation owns this list; the SPIR-V backend validates
    /// it and records the exact admitted mapping in `SpirvLayout`.
    pub builtin_arguments: Vec<SpirvBuiltinArgument>,
    /// Word capacity of the emulated private-storage heap (`fe_heap`) declared
    /// for kernels using function-local `[u32; N]` arrays (`MemAllocDynamic` /
    /// `Mload` / `Mstore`). Default 8192 words (32KB), matching
    /// `RUNG3_SPIRV_ARRAYS_DESIGN.md`. Irrelevant (and the heap is undeclared)
    /// for kernels with no Mem ops.
    pub heap_words: u32,
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy)]
pub enum ShaderEntries {
    Single(sonatina_ir::module::FuncRef),
    Raster {
        vertex: sonatina_ir::module::FuncRef,
        fragment: sonatina_ir::module::FuncRef,
    },
}

impl NagaBackend {
    pub fn new() -> Self {
        Self {
            workgroup_size: [64, 1, 1],
            dispatch_grid: [1, 1, 1],
            grid: false,
            render: false,
            compute: false,
            authored_raster: None,
            external_resources: Vec::new(),
            builtin_arguments: Vec::new(),
            heap_words: 8192,
        }
    }

    pub fn with_workgroup_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroup_size = [x, y, z];
        self
    }

    pub fn with_dispatch_grid(mut self, x: u32, y: u32, z: u32) -> Self {
        self.dispatch_grid = [x, y, z];
        self
    }

    pub fn with_grid(mut self) -> Self {
        self.grid = true;
        self
    }

    pub fn with_render(mut self) -> Self {
        self.render = true;
        self
    }

    pub fn with_compute(mut self) -> Self {
        self.compute = true;
        self
    }

    pub fn with_authored_raster(
        mut self,
        vertex_entry: impl Into<String>,
        fragment_entry: impl Into<String>,
    ) -> Self {
        self.authored_raster = Some(SpirvRasterPipeline {
            vertex_entry: vertex_entry.into(),
            fragment_entry: fragment_entry.into(),
        });
        self
    }

    pub fn with_external_resource(mut self, resource: SpirvExternalResource) -> Self {
        self.external_resources.push(resource);
        self
    }

    pub fn with_builtin_argument(mut self, argument: SpirvBuiltinArgument) -> Self {
        self.builtin_arguments.push(argument);
        self
    }

    /// Override the private-heap word capacity for Mem-op-bearing kernels.
    pub fn with_private_heap_words(mut self, words: u32) -> Self {
        self.heap_words = words;
        self
    }
}

#[cfg(feature = "spirv-backend")]
fn naga_equality_case(
    expressions: &naga::Arena<naga::Expression>,
    condition: naga::Handle<naga::Expression>,
) -> Option<(naga::Handle<naga::Expression>, naga::SwitchValue)> {
    let naga::Expression::Binary {
        op: naga::BinaryOperator::Equal,
        left,
        right,
    } = expressions[condition]
    else {
        return None;
    };
    let literal = |expression| match expressions[expression] {
        naga::Expression::Literal(naga::Literal::I32(value)) => {
            Some(naga::SwitchValue::I32(value))
        }
        naga::Expression::Literal(naga::Literal::U32(value)) => {
            Some(naga::SwitchValue::U32(value))
        }
        _ => None,
    };
    if let Some(value) = literal(right) {
        return Some((left, value));
    }
    literal(left).map(|value| (right, value))
}

#[cfg(feature = "spirv-backend")]
fn naga_reject_is_next_equality<'a>(
    reject: &'a naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    selector: naga::Handle<naga::Expression>,
) -> Option<(
    naga::SwitchValue,
    &'a naga::Block,
    &'a naga::Block,
)> {
    let [naga::Statement::Emit(range), naga::Statement::If { condition, accept, reject }] = &reject[..] else {
        return None;
    };
    if range.first_and_last() != Some((*condition, *condition)) {
        return None;
    }
    let (next_selector, value) = naga_equality_case(expressions, *condition)?;
    (next_selector == selector).then_some((value, accept, reject))
}

/// Turn a compiler-generated `x == 0`, `x == 1`, ... rejection ladder into
/// one native switch. Only a pure, singly emitted equality may sit between
/// rungs, so authored work in a rejection block is never reordered or lost.
#[cfg(feature = "spirv-backend")]
fn compact_naga_equality_ladder(
    statement: naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
) -> (naga::Statement, bool) {
    let naga::Statement::If { condition, accept, reject } = statement else {
        return (statement, false);
    };
    let Some((selector, first_value)) = naga_equality_case(expressions, condition) else {
        return (naga::Statement::If { condition, accept, reject }, false);
    };

    let mut values = std::collections::HashSet::from([first_value]);
    let mut case_count = 1usize;
    let mut cursor = &reject;
    while let Some((value, _, next_reject)) =
        naga_reject_is_next_equality(cursor, expressions, selector)
    {
        if !values.insert(value) {
            break;
        }
        case_count += 1;
        cursor = next_reject;
    }
    if case_count < 3 {
        return (naga::Statement::If { condition, accept, reject }, false);
    }

    let mut cases = Vec::with_capacity(case_count + 1);
    cases.push(naga::SwitchCase {
        value: first_value,
        body: accept,
        fall_through: false,
    });
    let mut cursor = reject;
    for _ in 1..case_count {
        let mut entries = cursor.span_into_iter();
        let Some((naga::Statement::Emit(_), _)) = entries.next() else {
            unreachable!("validated equality ladder must retain its condition emit")
        };
        let Some((naga::Statement::If { condition, accept, reject }, _)) = entries.next() else {
            unreachable!("validated equality ladder must retain its conditional")
        };
        let (case_selector, case_value) = naga_equality_case(expressions, condition)
            .expect("validated equality ladder must retain its equality case");
        debug_assert_eq!(case_selector, selector);
        cases.push(naga::SwitchCase {
            value: case_value,
            body: accept,
            fall_through: false,
        });
        cursor = reject;
    }
    cases.push(naga::SwitchCase {
        value: naga::SwitchValue::Default,
        body: cursor,
        fall_through: false,
    });
    (naga::Statement::Switch { selector, cases }, true)
}

#[cfg(feature = "spirv-backend")]
fn compact_naga_control_in_block(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
) -> usize {
    let original = std::mem::take(block);
    let mut compacted = 0usize;
    for (statement, span) in original.span_into_iter() {
        let (mut statement, folded) = compact_naga_equality_ladder(statement, expressions);
        compacted += usize::from(folded);
        match &mut statement {
            naga::Statement::Block(nested) => {
                compacted += compact_naga_control_in_block(nested, expressions);
            }
            naga::Statement::If { accept, reject, .. } => {
                compacted += compact_naga_control_in_block(accept, expressions);
                compacted += compact_naga_control_in_block(reject, expressions);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    compacted += compact_naga_control_in_block(&mut case.body, expressions);
                }
            }
            naga::Statement::Loop { body, continuing, .. } => {
                compacted += compact_naga_control_in_block(body, expressions);
                compacted += compact_naga_control_in_block(continuing, expressions);
            }
            _ => {}
        }
        block.push(statement, span);
    }
    compacted
}

#[cfg(feature = "spirv-backend")]
fn compact_naga_control(module: &mut naga::Module) -> usize {
    let mut compacted = 0usize;
    for (_, function) in module.functions.iter_mut() {
        compacted += compact_naga_control_in_block(&mut function.body, &function.expressions);
    }
    for entry in &mut module.entry_points {
        compacted += compact_naga_control_in_block(
            &mut entry.function.body,
            &entry.function.expressions,
        );
    }
    compacted
}

#[cfg(feature = "spirv-backend")]
fn validate_naga_portable_wgsl_limits(module: &naga::Module) -> Result<(), String> {
    let validate_function = |kind: &str, name: &str, function: &naga::Function| {
        let parameter_count = function.arguments.len();
        if parameter_count > MAX_WGSL_FUNCTION_PARAMETERS {
            return Err(format!(
                "spirv: {kind} `{name}` has {parameter_count} physical parameters after ABI lowering, over the portable WGSL limit of {MAX_WGSL_FUNCTION_PARAMETERS}. Fail closed."
            ));
        }
        Ok(())
    };

    for (handle, function) in module.functions.iter() {
        let name = function
            .name
            .as_deref()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("function_{}", handle.index()));
        validate_function("helper", &name, function)?;
    }
    for entry in &module.entry_points {
        validate_function("entry point", &entry.name, &entry.function)?;
    }
    Ok(())
}

impl Backend for NagaBackend {
    type Artifact = ShaderArtifact;
    type Error = SpirvError;

    #[cfg(not(feature = "spirv-backend"))]
    fn compile_module(&self, _module: &Module) -> Result<Self::Artifact, Vec<Self::Error>> {
        Err(vec![SpirvError::Translation(
            "SPIR-V backend requires the spirv-backend feature".to_string(),
        )])
    }

    #[cfg(feature = "spirv-backend")]
    fn compile_module(&self, module: &Module) -> Result<Self::Artifact, Vec<Self::Error>> {
        self.compile_module_for_entry(module, None, None)
    }
}

#[cfg(feature = "spirv-backend")]
impl NagaBackend {
    /// Compile a specific entry in this module, independently of declaration order.
    /// Paired authored raster uses its pipeline interface instead.
    pub fn compile_entry(
        &self,
        module: &Module,
        entry: sonatina_ir::module::FuncRef,
    ) -> Result<ShaderArtifact, Vec<SpirvError>> {
        self.compile_module_for_entry(module, Some(ShaderEntries::Single(entry)), None)
    }

    /// Compile source-authored raster stages by module-local function identity.
    /// This selects the paired raster interface without a name lookup.
    pub fn compile_raster_entries(
        &self,
        module: &Module,
        vertex: sonatina_ir::module::FuncRef,
        fragment: sonatina_ir::module::FuncRef,
    ) -> Result<ShaderArtifact, Vec<SpirvError>> {
        self.compile_module_for_entry(module, Some(ShaderEntries::Raster { vertex, fragment }), None)
    }

    /// Compile under a checked environment profile with required outputs.
    /// Unlike the compatibility entry points, this requires a Shader ISA module.
    pub fn compile_for_target(
        &self,
        module: &Module,
        entries: ShaderEntries,
        target: &ShaderTargetContract,
    ) -> Result<ShaderArtifact, Vec<SpirvError>> {
        if module.ctx.triple.architecture != sonatina_triple::Architecture::Shader {
            return Err(vec![SpirvError::UnsupportedTarget(
                "an explicit shader target requires a Shader ISA module".to_owned(),
            )]);
        }
        self.compile_module_for_entry(module, Some(entries), Some(target))
    }

    fn resolve_legacy_pipeline(
        &self,
        module: &Module,
        entry: Option<ShaderEntries>,
    ) -> Result<ShaderPipeline, String> {
        if let Some(ShaderEntries::Raster { vertex, fragment }) = entry {
            if self.authored_raster.is_some() || self.grid || self.render || self.compute {
                return Err("spirv raster: explicit raster entries cannot be combined with another entry mode".to_owned());
            }
            if self.dispatch_grid != [1, 1, 1] {
                return Err("spirv raster: a fixed dispatch grid is invalid for authored raster".to_owned());
            }
            return Ok(ShaderPipeline::Raster { vertex, fragment });
        }
        if let Some(raster) = &self.authored_raster {
            if entry.is_some() {
                return Err("spirv raster: an individual entry cannot override the paired raster interface".to_owned());
            }
            if self.grid || self.render || self.compute {
                return Err("spirv raster: authored raster, grid, fullscreen render, and compute modes are mutually exclusive".to_owned());
            }
            if self.dispatch_grid != [1, 1, 1] {
                return Err("spirv raster: a fixed dispatch grid is invalid for authored raster".to_owned());
            }
            let find_entry = |stage: &str, name: &str| {
                module.funcs().into_iter().find(|function| {
                    module.ctx.get_sig(*function).is_some_and(|sig| sig.name() == name)
                }).ok_or_else(|| format!("spirv raster: {stage} entry `{name}` is absent"))
            };
            return Ok(ShaderPipeline::Raster {
                vertex: find_entry("vertex", &raster.vertex_entry)?,
                fragment: find_entry("fragment", &raster.fragment_entry)?,
            });
        }
        if [self.grid, self.render, self.compute].into_iter().filter(|value| *value).count() > 1 {
            return Err("spirv: grid, render, and explicit compute modes are mutually exclusive".to_owned());
        }
        if !self.compute && self.dispatch_grid != [1, 1, 1] {
            return Err("spirv: a fixed dispatch grid currently requires explicit compute mode".to_owned());
        }
        let entry = match entry {
            Some(ShaderEntries::Single(entry)) => entry,
            Some(ShaderEntries::Raster { .. }) => unreachable!("raster resolved above"),
            None => *module.funcs().first()
                .ok_or_else(|| "spirv: module has no functions to translate".to_owned())?,
        };
        Ok(if self.compute {
            ShaderPipeline::Compute { entry, workgroup_size: self.workgroup_size, dispatch_grid: self.dispatch_grid }
        } else if self.render {
            ShaderPipeline::Fullscreen { entry }
        } else if self.grid {
            ShaderPipeline::LegacyGrid { entry, workgroup_size: self.workgroup_size }
        } else {
            ShaderPipeline::LegacyScalar { entry, workgroup_size: self.workgroup_size }
        })
    }

    fn compile_module_for_entry(
        &self,
        module: &Module,
        entry: Option<ShaderEntries>,
        target: Option<&ShaderTargetContract>,
    ) -> Result<ShaderArtifact, Vec<SpirvError>> {
        let pipeline = self.resolve_legacy_pipeline(module, entry)
            .map_err(|error| vec![SpirvError::Translation(error)])?;
        Self::compile_pipeline(module, pipeline, &self.external_resources,
            &self.builtin_arguments, self.heap_words, target)
    }

    /// Compile a self-contained request. No legacy backend configuration is read.
    pub fn compile_request(
        module: &Module,
        request: &ShaderCompileRequest<'_>,
    ) -> Result<ShaderArtifact, Vec<SpirvError>> {
        if module.ctx.triple.architecture != sonatina_triple::Architecture::Shader {
            return Err(vec![SpirvError::UnsupportedTarget(
                "an explicit shader target requires a Shader ISA module".to_owned(),
            )]);
        }
        Self::compile_pipeline(module, request.pipeline, request.resources,
            request.builtin_arguments, request.private_heap_words, Some(request.target))
    }

    fn compile_pipeline(
        module: &Module,
        pipeline: ShaderPipeline,
        resources: &[SpirvExternalResource],
        builtin_arguments: &[SpirvBuiltinArgument],
        heap_words: u32,
        target: Option<&ShaderTargetContract>,
    ) -> Result<ShaderArtifact, Vec<SpirvError>> {
        let capabilities = match target.map(ShaderTargetContract::environment) {
            Some(ShaderEnvironment::WebGpu) => naga::valid::Capabilities::empty(),
            Some(environment) => return Err(vec![SpirvError::UnsupportedTarget(format!(
                "shader environment {environment:?} has no implemented capability profile"
            ))]),
            None => naga::valid::Capabilities::all(),
        };
        let trace = std::env::var_os("SONATINA_SPIRV_TRACE").is_some();
        let started = std::time::Instant::now();
        let (mut naga_mod, layout) = translate_to_naga(
            module,
            pipeline,
            resources,
            builtin_arguments,
            heap_words,
        )
        .map_err(|e| vec![SpirvError::Translation(e)])?;
        let compacted_equality_ladders = compact_naga_control(&mut naga_mod);
        if trace {
            eprintln!(
                "sonatina spirv: translated to naga, compacted_equality_ladders={}, elapsed_ms={}",
                compacted_equality_ladders,
                started.elapsed().as_millis()
            );
        }

        validate_naga_portable_wgsl_limits(&naga_mod)
            .map_err(|error| vec![SpirvError::Validation(error)])?;

        let phase = std::time::Instant::now();
        let validation = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            capabilities,
        )
        .validate(&naga_mod);
        let info = match validation {
            Ok(info) => info,
            Err(error) => {
                if trace {
                    eprintln!("sonatina spirv: naga validation failed: {error:?}");
                    if let naga::valid::ValidationError::Function {
                        handle,
                        name,
                        source:
                            naga::valid::FunctionError::InvalidStoreTypes { pointer, value },
                    } = error.as_inner()
                    {
                        let function = &naga_mod.functions[*handle];
                        eprintln!(
                            "sonatina spirv: invalid helper store, function={name}, pointer={pointer:?} value={value:?}"
                        );
                        let mut pending = vec![*pointer, *value];
                        let mut seen = Vec::new();
                        while let Some(expression) = pending.pop() {
                            if seen.contains(&expression) {
                                continue;
                            }
                            seen.push(expression);
                            eprintln!(
                                "sonatina spirv: function={name} expression={expression:?} value={:?}",
                                function.expressions[expression],
                            );
                            match &function.expressions[expression] {
                                naga::Expression::LocalVariable(local) => {
                                    let local = &function.local_variables[*local];
                                    eprintln!(
                                        "sonatina spirv: function={name} local={local:?} type={:?}",
                                        naga_mod.types[local.ty],
                                    );
                                }
                                naga::Expression::CallResult(callee) => {
                                    eprintln!(
                                        "sonatina spirv: function={name} callee={callee:?} name={:?} result={:?}",
                                        naga_mod.functions[*callee].name,
                                        naga_mod.functions[*callee].result,
                                    );
                                }
                                naga::Expression::FunctionArgument(index) => {
                                    let argument = &function.arguments[*index as usize];
                                    eprintln!(
                                        "sonatina spirv: function={name} argument={index} value={argument:?} type={:?}",
                                        naga_mod.types[argument.ty],
                                    );
                                }
                                naga::Expression::AccessIndex { base, .. }
                                | naga::Expression::Load { pointer: base } => {
                                    pending.push(*base);
                                }
                                _ => {}
                            }
                        }
                    }
                    for entry in &naga_mod.entry_points {
                        let expression_count = entry.function.expressions.len();
                        let invalid_expression = match error.as_inner() {
                            naga::valid::ValidationError::EntryPoint {
                                name,
                                source:
                                    naga::valid::EntryPointError::Function(
                                        naga::valid::FunctionError::Expression { handle, .. },
                                    ),
                                ..
                            } if name == &entry.name => Some(handle.index()),
                            _ => None,
                        };
                        let (first, end) = invalid_expression.map_or_else(
                            || (expression_count.saturating_sub(128), expression_count),
                            |index| {
                                (
                                    index.saturating_sub(32),
                                    index.saturating_add(33).min(expression_count),
                                )
                            },
                        );
                        eprintln!(
                            "sonatina spirv: entry={} expression_window={}..{} invalid={invalid_expression:?}",
                            entry.name, first, end,
                        );
                        for (handle, expression) in entry.function.expressions.iter() {
                            if (first..end).contains(&handle.index()) {
                                eprintln!(
                                    "sonatina spirv: entry={} expression={handle:?} value={expression:?}",
                                    entry.name,
                                );
                            }
                        }
                    }
                }
                return Err(vec![SpirvError::Validation(format!("{error:?}"))]);
            }
        };
        if trace {
            eprintln!(
                "sonatina spirv: validated naga, elapsed_ms={}",
                phase.elapsed().as_millis()
            );
        }

        let options = naga::back::spv::Options {
            lang_version: (1, 5),
            flags: naga::back::spv::WriterFlags::empty(),
            ..Default::default()
        };

        let phase = std::time::Instant::now();
        let words = if target.is_none_or(|target| target.requests(ShaderEncoding::Spirv)) {
            naga::back::spv::write_vec(&naga_mod, &info, &options, None)
                .map_err(|e| vec![SpirvError::Translation(format!("{e}"))])?
        } else {
            Vec::new()
        };
        if trace {
            eprintln!(
                "sonatina spirv: emitted spirv, words={}, elapsed_ms={}",
                words.len(),
                phase.elapsed().as_millis()
            );
        }

        // Also emit WGSL for wgpu execution
        let phase = std::time::Instant::now();
        let wgsl = if target.is_none_or(|target| target.requests(ShaderEncoding::Wgsl)) {
            let output = naga::back::wgsl::write_string(
                &naga_mod, &info, naga::back::wgsl::WriterFlags::empty()
            );
            if target.is_some() {
                Some(output.map_err(|error| vec![SpirvError::Translation(format!(
                    "required WGSL encoding failed: {error}"
                ))])?)
            } else {
                output.ok()
            }
        } else {
            None
        };
        if trace {
            eprintln!(
                "sonatina spirv: emitted wgsl, bytes={}, elapsed_ms={}, total_elapsed_ms={}",
                wgsl.as_ref().map_or(0, String::len),
                phase.elapsed().as_millis(),
                started.elapsed().as_millis()
            );
        }

        Ok(ShaderArtifact { words, wgsl, layout })
    }
}

#[cfg(feature = "spirv-backend")]
fn resolve_naga_value(
    vid: sonatina_ir::ValueId,
    function: &sonatina_ir::Function,
    word: WordKind,
    vm: &std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    func: &mut naga::Function,
) -> Option<naga::Handle<naga::Expression>> {
    if let Some(&h) = vm.get(&vid) {
        return Some(h);
    }
    // If this is a phi value with a LocalVariable, load from it
    if let Some(&local) = phi_locals.get(&vid) {
        let ptr = func.expressions.append(
            naga::Expression::LocalVariable(local),
            naga::Span::UNDEFINED,
        );
        let loaded = func.expressions.append(
            naga::Expression::Load { pointer: ptr },
            naga::Span::UNDEFINED,
        );
        return Some(loaded);
    }
    if let sonatina_ir::Value::Immediate { imm, .. } = function.dfg.value(vid) {
        let literal = match imm {
            sonatina_ir::Immediate::I1(v) => naga::Literal::Bool(*v),
            sonatina_ir::Immediate::I8(v) => match word {
                WordKind::U32 => naga::Literal::U32(*v as u32),
                WordKind::I64 => naga::Literal::I64(*v as i64),
            },
            sonatina_ir::Immediate::I32(v) => match word {
                WordKind::U32 => naga::Literal::U32(*v as u32),
                WordKind::I64 => naga::Literal::I64(*v as i64),
            },
            sonatina_ir::Immediate::I64(v) => naga::Literal::I64(*v),
            sonatina_ir::Immediate::F32(bits) => naga::Literal::F32(f32::from_bits(*bits)),
            _ => return None,
        };
        return Some(func.expressions.append(
            naga::Expression::Literal(literal),
            naga::Span::UNDEFINED,
        ));
    }
    None
}

/// Append a naga expression and immediately `Statement::Emit` it as its own
/// single-expression range (house style; mirrors the `F32ToU32` cast
/// lowering below).
#[cfg(feature = "spirv-backend")]
fn emit_expr(
    func: &mut naga::Function,
    target: &mut naga::Block,
    expr: naga::Expression,
) -> naga::Handle<naga::Expression> {
    let h = func.expressions.append(expr, naga::Span::UNDEFINED);
    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
    h
}

/// Append a `u32` literal constant expression (NOT `Emit`-ted: literals are
/// naga constant expressions, same convention as the `zero`/`low_f`/`high_f`/
/// etc. literals in the `F32ToU32` cast lowering below).
#[cfg(feature = "spirv-backend")]
fn lit_u32(func: &mut naga::Function, v: u32) -> naga::Handle<naga::Expression> {
    func.expressions.append(naga::Expression::Literal(naga::Literal::U32(v)), naga::Span::UNDEFINED)
}

#[cfg(feature = "spirv-backend")]
pub(super) fn append_external_resources(
    naga_mod: &mut naga::Module,
    resources: &[SpirvExternalResource],
    resource_stages: &[Vec<SpirvShaderStage>],
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
) -> Result<
    (
        Vec<(u32, naga::Handle<naga::GlobalVariable>)>,
        Vec<SpirvBinding>,
    ),
    String,
> {
    if resources.len() != resource_stages.len() {
        return Err(format!(
            "spirv: external resource/stage metadata length mismatch: {} resources, {} stage sets",
            resources.len(),
            resource_stages.len(),
        ));
    }
    if !resources.is_empty() && word != WordKind::U32 {
        return Err(
            "spirv: external resources currently require the u32 browser word"
                .to_string(),
        );
    }

    fn admitted_scalar(
        scalar: SpirvScalarKind,
        word_type: naga::Handle<naga::Type>,
        f32_type: naga::Handle<naga::Type>,
    ) -> Result<naga::Handle<naga::Type>, String> {
        match scalar {
            SpirvScalarKind::U32 => Ok(word_type),
            SpirvScalarKind::F32 => Ok(f32_type),
            other => Err(format!(
                "spirv: external storage scalar {other:?} is unsupported; portable resources admit u32 and f32"
            )),
        }
    }

    let mut roots = Vec::with_capacity(resources.len());
    let mut bindings = Vec::with_capacity(resources.len());
    let mut names = std::collections::HashSet::new();
    for (resource, stages) in resources.iter().zip(resource_stages) {
        if stages.is_empty() {
            return Err(format!(
                "spirv: external resource {} has no reachable shader stage",
                resource.name
            ));
        }
        if resource.name.is_empty() || !names.insert(resource.name.clone()) {
            return Err(format!(
                "spirv: external resource names must be nonempty and unique; got {:?}",
                resource.name
            ));
        }
        if resource.stride == 0 || resource.stride % 4 != 0 {
            return Err(format!(
                "spirv: external resource {} stride {} must be a nonzero multiple of 4",
                resource.name, resource.stride
            ));
        }

        let (element_type, element_span) = match &resource.element {
            SpirvResourceElement::Scalar(scalar) => {
                (admitted_scalar(*scalar, word_type, f32_type)?, 4)
            }
            SpirvResourceElement::Record { fields, span } => {
                if fields.is_empty() || *span == 0 || *span % 4 != 0 {
                    return Err(format!(
                        "spirv: external resource {} record must have fields and a 4-byte-aligned nonzero span",
                        resource.name
                    ));
                }
                let mut naga_fields = Vec::with_capacity(fields.len());
                let mut previous_end = 0;
                for field in fields {
                    if field.name.is_empty() || field.offset % 4 != 0 {
                        return Err(format!(
                            "spirv: external resource {} has an unnamed or unaligned record field",
                            resource.name
                        ));
                    }
                    if field.offset < previous_end || field.offset + 4 > *span {
                        return Err(format!(
                            "spirv: external resource {} record fields overlap or exceed span {}",
                            resource.name, span
                        ));
                    }
                    naga_fields.push(naga::StructMember {
                        name: Some(field.name.clone()),
                        ty: admitted_scalar(field.scalar, word_type, f32_type)?,
                        binding: None,
                        offset: field.offset,
                    });
                    previous_end = field.offset + 4;
                }
                let ty = naga_mod.types.insert(
                    naga::Type {
                        name: Some(format!("{}_element", resource.name)),
                        inner: naga::TypeInner::Struct {
                            members: naga_fields,
                            span: *span,
                        },
                    },
                    naga::Span::UNDEFINED,
                );
                (ty, *span)
            }
        };
        if resource.stride < element_span {
            return Err(format!(
                "spirv: external resource {} stride {} is smaller than element span {}",
                resource.name, resource.stride, element_span
            ));
        }
        let array_type = naga_mod.types.insert(
            naga::Type {
                name: Some(format!("{}_array", resource.name)),
                inner: naga::TypeInner::Array {
                    base: element_type,
                    size: naga::ArraySize::Dynamic,
                    stride: resource.stride,
                },
            },
            naga::Span::UNDEFINED,
        );
        let access = match resource.access {
            Access::Read => naga::StorageAccess::LOAD,
            Access::ReadWrite => naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
        };
        let global = naga_mod.global_variables.append(
            naga::GlobalVariable {
                name: Some(resource.name.clone()),
                space: naga::AddressSpace::Storage { access },
                binding: Some(naga::ResourceBinding {
                    group: resource.group,
                    binding: resource.binding,
                }),
                ty: array_type,
                init: None,
                memory_decorations: naga::ir::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        roots.push((resource.arg_index, global));
        bindings.push(SpirvBinding {
            group: resource.group,
            binding: resource.binding,
            name: resource.name.clone(),
            access: resource.access,
            role: Role::Resource,
            stages: stages.clone(),
            stride: resource.stride,
            span: element_span,
            members: Vec::new(),
            resource_element: Some(resource.element.clone()),
            resource_length: Some(resource.length),
            resource_arg_index: Some(resource.arg_index),
        });
    }
    Ok((roots, bindings))
}

/// The private-storage heap proper (`fe_heap`/`fe_bump`), present only for
/// kernels using function-local `[u32; N]` arrays (`has_mem`).
#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy)]
struct HeapCtx {
    /// Pointer to `var<function> fe_heap: array<u32, heap_words>`. Entry
    /// functions create it locally; outlined helpers receive the same pointer
    /// through their compiler-owned ABI suffix.
    heap: naga::Handle<naga::Expression>,
    /// Pointer to the shared `var<function> fe_bump: u32` allocator state.
    bump: naga::Handle<naga::Expression>,
    word_type: naga::Handle<naga::Type>,
    heap_words: u32,
}

/// Function-scoped state for kernels that can reach a poison path: either
/// they use the private-storage heap emulation of function-local `[u32; N]`
/// arrays (`RUNG3_SPIRV_ARRAYS_DESIGN.md` section 2, `has_mem`), or they
/// contain a Sonatina `Unreachable` trap with NO Mem ops at all (checked-usize
/// arithmetic overflow, a generic MIR trap terminator, or Sccp/DCE eliminating
/// every Mem op while the trap survives). Declared once per entry function
/// whenever the entry or a reachable helper needs it. Compiler-owned pointer
/// arguments let helpers mutate the same arena and trap state without creating
/// another allocation or host-visible resource.
///
/// Guards the has_mem==false shadow of review finding 4 (adversarial review
/// Finding A, 2026-08-08): declaring `heap` as `Option<HeapCtx>` rather than
/// requiring it unconditionally means a no-Mem trapping function still gets
/// a `MemCtx` (for `trapped`) without ever declaring the heap/bump locals it
/// has no use for -- has_mem kernels are unaffected (heap is always `Some`
/// there), and has_unreachable-only kernels finally get a real trap channel
/// instead of silently falling through to a zero/uninitialized result.
#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy)]
struct MemCtx {
    /// `None` when this context exists solely to carry the trap channel for
    /// a no-Mem trapping function.
    heap: Option<HeapCtx>,
    /// `var<function> fe_trapped: bool`, init false. An OR-accumulator: any
    /// guard site ORs its own failure condition in; never cleared once set.
    /// This is the externally-visible status channel that closes review finding 3
    /// (poison-sentinel collision): a consumer reads this flag instead of
    /// trying to infer failure from an in-band magic result value.
    trapped: naga::Handle<naga::Expression>,
}

#[cfg(feature = "spirv-backend")]
impl MemCtx {
    /// The heap context, or an explanatory panic if this `MemCtx` was
    /// constructed for a no-Mem trapping function (should be unreachable:
    /// Mem-op emission arms only run when the pre-scan proved `has_mem`,
    /// which always yields `Some`).
    #[cfg(feature = "spirv-backend")]
    fn heap(&self) -> HeapCtx {
        self.heap.expect("Mem op emission requires a HeapCtx (has_mem pre-scan gate)")
    }
}

/// Arena scopes are compiler-authored, nested lifetime proofs. This verifier
/// independently checks their control-flow shape before the SPIR-V backend
/// uses them to admit allocations in loops.
///
/// Each block has one exact stack of live checkpoint SSA values. A checkpoint
/// pushes, a rewind must pop that same top value, and all incoming edges must
/// agree on the complete stack. Consequently a loop backedge can reach its
/// header only after every checkpoint opened during that iteration has been
/// rewound. An allocation in a loop is admitted only when its live stack is
/// deeper than the stack at the innermost loop header.
#[cfg(feature = "spirv-backend")]
struct ArenaScopeAnalysis {
    scoped_loop_allocations: std::collections::HashSet<sonatina_ir::InstId>,
    allocation_scopes:
        std::collections::HashMap<sonatina_ir::InstId, Option<sonatina_ir::ValueId>>,
    checkpoint_parents:
        std::collections::HashMap<sonatina_ir::ValueId, Option<sonatina_ir::ValueId>>,
}

#[cfg(feature = "spirv-backend")]
impl ArenaScopeAnalysis {
    /// Conservative high-water bound for the verified arena tree. Direct
    /// allocations in one scope accumulate. Nested scopes overlap their
    /// parent, while sibling scopes cannot overlap because each must rewind
    /// before the next sibling is opened.
    fn high_water_bytes(
        &self,
        allocations: impl IntoIterator<Item = (sonatina_ir::InstId, u64)>,
    ) -> u64 {
        use std::collections::HashMap;

        let mut direct = HashMap::<Option<sonatina_ir::ValueId>, u64>::new();
        direct.insert(None, 0);
        for (instruction, bytes) in allocations {
            let scope = self
                .allocation_scopes
                .get(&instruction)
                .copied()
                .flatten();
            let total = direct.entry(scope).or_default();
            *total = total.saturating_add(bytes);
        }

        let mut children =
            HashMap::<Option<sonatina_ir::ValueId>, Vec<sonatina_ir::ValueId>>::new();
        for (&checkpoint, &parent) in &self.checkpoint_parents {
            children.entry(parent).or_default().push(checkpoint);
        }

        fn visit(
            scope: Option<sonatina_ir::ValueId>,
            direct: &HashMap<Option<sonatina_ir::ValueId>, u64>,
            children: &HashMap<Option<sonatina_ir::ValueId>, Vec<sonatina_ir::ValueId>>,
        ) -> u64 {
            let child_high_water = children
                .get(&scope)
                .into_iter()
                .flatten()
                .map(|&child| visit(Some(child), direct, children))
                .max()
                .unwrap_or(0);
            direct
                .get(&scope)
                .copied()
                .unwrap_or(0)
                .saturating_add(child_high_water)
        }

        visit(None, &direct, &children)
    }
}

#[cfg(feature = "spirv-backend")]
fn verify_arena_scopes(
    function: &sonatina_ir::Function,
    cfg: &sonatina_ir::ControlFlowGraph,
    loop_tree: &crate::loop_analysis::LoopTree,
) -> Result<ArenaScopeAnalysis, String> {
    use sonatina_ir::{InstDowncast, inst::data};
    use std::collections::{HashMap, HashSet, VecDeque};

    let Some(entry) = function.layout.entry_block() else {
        return Ok(ArenaScopeAnalysis {
            scoped_loop_allocations: HashSet::new(),
            allocation_scopes: HashMap::new(),
            checkpoint_parents: HashMap::new(),
        });
    };
    let inst_set = function.inst_set();
    let mut incoming = HashMap::<sonatina_ir::BlockId, Vec<sonatina_ir::ValueId>>::new();
    let mut allocation_stacks = HashMap::<sonatina_ir::InstId, Vec<sonatina_ir::ValueId>>::new();
    let mut checkpoint_parents =
        HashMap::<sonatina_ir::ValueId, Option<sonatina_ir::ValueId>>::new();
    let mut worklist = VecDeque::from([entry]);
    incoming.insert(entry, Vec::new());

    while let Some(block) = worklist.pop_front() {
        let mut stack = incoming
            .get(&block)
            .cloned()
            .expect("arena worklist blocks always have an incoming state");
        for inst in function.layout.iter_inst(block) {
            let inst_data = function.dfg.inst(inst);
            if <&data::MemCheckpoint as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                let checkpoint = function.dfg.inst_result(inst).ok_or_else(|| {
                    format!("spirv: mem.checkpoint in {block:?} has no SSA result. Fail closed.")
                })?;
                checkpoint_parents.insert(checkpoint, stack.last().copied());
                stack.push(checkpoint);
            } else if let Some(rewind) =
                <&data::MemRewind as InstDowncast>::downcast(inst_set, inst_data)
            {
                let checkpoint = *rewind.checkpoint();
                let Some(active) = stack.pop() else {
                    return Err(format!(
                        "spirv: mem.rewind {checkpoint:?} in {block:?} has no live arena checkpoint. Fail closed."
                    ));
                };
                if active != checkpoint {
                    return Err(format!(
                        "spirv: mem.rewind {checkpoint:?} in {block:?} does not match the innermost live checkpoint {active:?}. Fail closed."
                    ));
                }
            } else if <&data::MemAllocDynamic as InstDowncast>::downcast(inst_set, inst_data)
                .is_some()
            {
                allocation_stacks.insert(inst, stack.clone());
            }
        }

        for &successor in cfg.succs_of(block) {
            match incoming.get(&successor) {
                Some(expected) if expected != &stack => {
                    return Err(format!(
                        "spirv: arena checkpoint stack disagrees at {successor:?}: edge from {block:?} carries {stack:?}, existing incoming state is {expected:?}. Fail closed."
                    ));
                }
                Some(_) => {}
                None => {
                    incoming.insert(successor, stack.clone());
                    worklist.push_back(successor);
                }
            }
        }
    }

    let mut scoped_loop_allocations = HashSet::new();
    for (&inst, stack) in &allocation_stacks {
        let block = function.layout.inst_block(inst);
        let Some(loop_id) = loop_tree.loop_of_block(block) else {
            continue;
        };
        let header = loop_tree.loop_header(loop_id);
        let header_stack = incoming.get(&header).ok_or_else(|| {
            format!(
                "spirv: loop header {header:?} for allocation {inst:?} has no reachable arena state. Fail closed."
            )
        })?;
        if stack.len() > header_stack.len() && stack.starts_with(header_stack) {
            scoped_loop_allocations.insert(inst);
        }
    }

    let allocation_scopes = allocation_stacks
        .into_iter()
        .map(|(instruction, stack)| (instruction, stack.last().copied()))
        .collect();
    Ok(ArenaScopeAnalysis {
        scoped_loop_allocations,
        allocation_scopes,
        checkpoint_parents,
    })
}

/// OR-accumulate `mem_ctx.trapped` with a freshly computed boolean condition
/// (read-modify-write, so an EARLIER guard's `true` is never clobbered by a
/// LATER guard's `false`). Used at every dynamic (runtime-computed) guard:
/// heap-allocation overflow and misaligned access.
#[cfg(feature = "spirv-backend")]
fn mark_trapped_if(
    func: &mut naga::Function,
    target: &mut naga::Block,
    mem_ctx: MemCtx,
    cond: naga::Handle<naga::Expression>,
) {
    let cur = emit_expr(func, target, naga::Expression::Load { pointer: mem_ctx.trapped });
    let or = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::LogicalOr, left: cur, right: cond });
    target.push(naga::Statement::Store { pointer: mem_ctx.trapped, value: or }, naga::Span::UNDEFINED);
}

/// Unconditionally set `mem_ctx.trapped = true`. Used at Unreachable (trap)
/// sites, where the condition is already known statically (control flow
/// reached this point at all only because the guard failed), so no OR-load
/// is needed -- `true OR x == true` regardless of `x`.
#[cfg(feature = "spirv-backend")]
fn mark_trapped_always(func: &mut naga::Function, target: &mut naga::Block, mem_ctx: MemCtx) {
    let one = func.expressions.append(naga::Expression::Literal(naga::Literal::Bool(true)), naga::Span::UNDEFINED);
    target.push(naga::Statement::Store { pointer: mem_ctx.trapped, value: one }, naga::Span::UNDEFINED);
}

/// Compute the guarded, clamped `fe_heap` word pointer for a Mem access at
/// byte address `addr` (already resolved to a naga u32 expression). Emits,
/// in order:
///  1. When `require_word_alignment` is true, `addr & 3 != 0` is computed and
///     OR'd into `mem_ctx.trapped`. An I32 access requires this check because
///     shifting the address would otherwise silently alias a neighboring word.
///     I1 accesses deliberately pass false: they use the low address bits to
///     select one byte inside the containing word.
///  2. Review finding 1 (heap-exhaustion aliasing), the per-access half: the word
///     index is `Min`-clamped into `[0, heap_words)` via `Select` so a bad or
///     wrapped address can only ever produce an in-range `OpAccessChain`
///     (wrong-but-bounded, never UB), independent of the allocation-time
///     guard in the `MemAllocDynamic` arm above.
#[cfg(feature = "spirv-backend")]
fn emit_mem_access(
    func: &mut naga::Function,
    target: &mut naga::Block,
    mem_ctx: MemCtx,
    addr: naga::Handle<naga::Expression>,
    require_word_alignment: bool,
) -> naga::Handle<naga::Expression> {
    if require_word_alignment {
        let three = lit_u32(func, 3);
        let low_bits = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::And, left: addr, right: three });
        let zero = lit_u32(func, 0);
        let misaligned = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::NotEqual, left: low_bits, right: zero });
        mark_trapped_if(func, target, mem_ctx, misaligned);
    }

    let heap_ctx = mem_ctx.heap();
    let two = lit_u32(func, 2);
    let word_idx = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::ShiftRight, left: addr, right: two });
    let max_idx = lit_u32(func, heap_ctx.heap_words.saturating_sub(1));
    let heap_words_lit = lit_u32(func, heap_ctx.heap_words);
    let in_range = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::Less, left: word_idx, right: heap_words_lit });
    // Review finding (LOW, "clamp does not flag"): an out-of-range word index
    // is ALSO OR'd into `trapped`, not just silently clamped. Unreachable for
    // fe-generated IR without an earlier flagged event (the bounds check +
    // the compile-time capacity proof already prevent it), but a hand-built
    // or future adversarial address now leaves a visible trace instead of
    // diverging from wasm silently-but-bounded.
    let out_of_range = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::GreaterEqual, left: word_idx, right: heap_words_lit });
    mark_trapped_if(func, target, mem_ctx, out_of_range);
    let clamped = emit_expr(func, target, naga::Expression::Select { condition: in_range, accept: word_idx, reject: max_idx });
    let idx_i32 = emit_expr(func, target, naga::Expression::As { expr: clamped, kind: naga::ScalarKind::Sint, convert: Some(4) });

    // Access returns a pointer -- no Emit needed (matches the existing
    // ObjIndex convention above).
    func.expressions.append(naga::Expression::Access { base: heap_ctx.heap, index: idx_i32 }, naga::Span::UNDEFINED)
}

#[cfg(feature = "spirv-backend")]
fn emit_heap_byte_load(
    func: &mut naga::Function,
    target: &mut naga::Block,
    mem_ctx: MemCtx,
    addr: naga::Handle<naga::Expression>,
) -> naga::Handle<naga::Expression> {
    let word_ptr = emit_mem_access(func, target, mem_ctx, addr, false);
    let word = emit_expr(func, target, naga::Expression::Load { pointer: word_ptr });
    let three = lit_u32(func, 3);
    let lane = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::And,
            left: addr,
            right: three,
        },
    );
    let eight = lit_u32(func, 8);
    let shift = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: lane,
            right: eight,
        },
    );
    let shifted = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::ShiftRight,
            left: word,
            right: shift,
        },
    );
    let mask = lit_u32(func, 0xff);
    emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::And,
            left: shifted,
            right: mask,
        },
    )
}

#[cfg(feature = "spirv-backend")]
fn emit_heap_byte_store(
    func: &mut naga::Function,
    target: &mut naga::Block,
    mem_ctx: MemCtx,
    addr: naga::Handle<naga::Expression>,
    value: naga::Handle<naga::Expression>,
) {
    let word_ptr = emit_mem_access(func, target, mem_ctx, addr, false);
    let old_word = emit_expr(func, target, naga::Expression::Load { pointer: word_ptr });
    let three = lit_u32(func, 3);
    let lane = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::And,
            left: addr,
            right: three,
        },
    );
    let eight = lit_u32(func, 8);
    let shift = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: lane,
            right: eight,
        },
    );
    let byte_mask = lit_u32(func, 0xff);
    let shifted_mask = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::ShiftLeft,
            left: byte_mask,
            right: shift,
        },
    );
    let inverse_mask = emit_expr(
        func,
        target,
        naga::Expression::Unary {
            op: naga::UnaryOperator::BitwiseNot,
            expr: shifted_mask,
        },
    );
    let cleared = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::And,
            left: old_word,
            right: inverse_mask,
        },
    );
    let byte = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::And,
            left: value,
            right: byte_mask,
        },
    );
    let shifted_byte = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::ShiftLeft,
            left: byte,
            right: shift,
        },
    );
    let updated = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::InclusiveOr,
            left: cleared,
            right: shifted_byte,
        },
    );
    target.push(
        naga::Statement::Store {
            pointer: word_ptr,
            value: updated,
        },
        naga::Span::UNDEFINED,
    );
}

/// Store the final `mem_ctx.trapped` value (as u32 0/1) into the trap-status
/// output at `index`, alongside the ordinary result store. The consumer checks
/// this word after execution; the channel itself is part of the emitted ABI.
#[cfg(feature = "spirv-backend")]
fn emit_trap_store(
    func: &mut naga::Function,
    target: &mut naga::Block,
    mem_ctx: MemCtx,
    trap_var: naga::Handle<naga::GlobalVariable>,
    index: naga::Handle<naga::Expression>,
) {
    let trapped_bool = emit_expr(func, target, naga::Expression::Load { pointer: mem_ctx.trapped });
    let trapped_u32 = emit_expr(func, target, naga::Expression::As { expr: trapped_bool, kind: naga::ScalarKind::Uint, convert: Some(4) });
    let trap_global = func.expressions.append(naga::Expression::GlobalVariable(trap_var), naga::Span::UNDEFINED);
    let trap_ptr = func.expressions.append(naga::Expression::Access { base: trap_global, index }, naga::Span::UNDEFINED);
    target.push(naga::Statement::Store { pointer: trap_ptr, value: trapped_u32 }, naga::Span::UNDEFINED);
}

/// Linearize the physical global invocation id into the compiler-sized fixed
/// dispatch extent. Each invocation owns one trap word, so no shader write is
/// shared even when the authored kernel uses checked indexing or local arrays.
#[cfg(feature = "spirv-backend")]
fn emit_compute_invocation_index(
    func: &mut naga::Function,
    target: &mut naga::Block,
    global_argument: u32,
    extent: [u32; 3],
) -> naga::Handle<naga::Expression> {
    let global = func.expressions.append(
        naga::Expression::FunctionArgument(global_argument),
        naga::Span::UNDEFINED,
    );
    let x = emit_expr(
        func,
        target,
        naga::Expression::AccessIndex {
            base: global,
            index: 0,
        },
    );
    let y = emit_expr(
        func,
        target,
        naga::Expression::AccessIndex {
            base: global,
            index: 1,
        },
    );
    let z = emit_expr(
        func,
        target,
        naga::Expression::AccessIndex {
            base: global,
            index: 2,
        },
    );
    let width = lit_u32(func, extent[0]);
    let row = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: y,
            right: width,
        },
    );
    let xy = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: x,
            right: row,
        },
    );
    let plane_stride = lit_u32(
        func,
        extent[0]
            .checked_mul(extent[1])
            .expect("validated compute plane extent"),
    );
    let plane = emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: z,
            right: plane_stride,
        },
    );
    emit_expr(
        func,
        target,
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: xy,
            right: plane,
        },
    )
}

/// Whether `block`'s terminator is Sonatina `Unreachable` (an array/memory
/// bounds trap). Distinct from `find_block_return_value`: there is no value
/// to resolve, only a poison signal to raise via `mark_trapped_always`.
#[cfg(feature = "spirv-backend")]
fn block_ends_unreachable(
    block: sonatina_ir::BlockId,
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
) -> bool {
    use sonatina_ir::InstDowncast;
    function.layout.iter_inst(block).any(|inst_id| {
        <&sonatina_ir::inst::control_flow::Unreachable as InstDowncast>::downcast(
            inst_set,
            function.dfg.inst(inst_id),
        )
        .is_some()
    })
}

/// Exact, branch-free "WebAssembly rules" (IEEE 754-2019 `minimum`/`maximum`)
/// f32 min/max for the naga/SPIR-V and WGSL backends. Bitcasts both operands
/// to `u32` and does everything else in integer space -- no float compare, no
/// fast-math latitude, no control flow -- so it is a conforming pinned-exact
/// implementation on every naga target, matching wasm's `f32.min`/`f32.max`
/// and cranelift's `fmin`/`fmax` bit-for-bit (see
/// `docs/numeric-intrinsics-semantics.md`).
///
/// Algorithm: build a monotone total integer order over the u32 bit pattern
/// of an f32 (`key(x) = xu ^ (0x80000000 | (0xffffffff * signbit(xu)))`,
/// i.e. flip all bits of negative values and just the sign bit of
/// non-negative values). Under that key, unsigned integer comparison agrees
/// with IEEE float ordering, including `-0.0` sorting strictly below `+0.0`
/// regardless of argument order. `OpSelect`/`select()` on the key comparison
/// picks the min/max operand; a second select forces the canonical quiet NaN
/// `0x7fc0_0000` if either operand's biased exponent+mantissa exceeds
/// `0x7f80_0000` (i.e. either operand is NaN). `want_max` flips the key
/// comparison from `Less` to `Greater`; everything else is shared.
///
/// Same toolkit as the proven `F32ToU32` saturating-cast lowering (bitcast
/// `As { convert: None }`, integer binaries, literals, chained
/// `Expression::Select`, `Statement::Emit`-only) -- see that arm below.
#[cfg(feature = "spirv-backend")]
fn emit_exact_fminmax(
    func: &mut naga::Function,
    target: &mut naga::Block,
    lhs: naga::Handle<naga::Expression>,
    rhs: naga::Handle<naga::Expression>,
    want_max: bool,
) -> naga::Handle<naga::Expression> {
    // Bitcast both operands to u32. NO float ops after this point.
    let au = emit_expr(func, target, naga::Expression::As { expr: lhs, kind: naga::ScalarKind::Uint, convert: None });
    let bu = emit_expr(func, target, naga::Expression::As { expr: rhs, kind: naga::ScalarKind::Uint, convert: None });

    let thirty_one = lit_u32(func, 31);
    let zero_u = lit_u32(func, 0);
    let sign_mask = lit_u32(func, 0x8000_0000);
    let abs_mask = lit_u32(func, 0x7fff_ffff);
    let exp_mask = lit_u32(func, 0x7f80_0000);
    let qnan = lit_u32(func, 0x7fc0_0000);

    // key(x) = xu ^ (0x80000000 | (0xffffffff * signbit(xu)))
    let sa_shift = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::ShiftRight, left: au, right: thirty_one });
    let sa = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::Subtract, left: zero_u, right: sa_shift });
    let sa_or = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::InclusiveOr, left: sa, right: sign_mask });
    let ka = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::ExclusiveOr, left: au, right: sa_or });

    let sb_shift = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::ShiftRight, left: bu, right: thirty_one });
    let sb = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::Subtract, left: zero_u, right: sb_shift });
    let sb_or = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::InclusiveOr, left: sb, right: sign_mask });
    let kb = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::ExclusiveOr, left: bu, right: sb_or });

    // pick = Select(Less(ka, kb), au, bu) for min; Greater for max. naga's
    // `Expression::Select { condition, accept, reject }` (confirmed against
    // both call sites below and `back/spv/block.rs`'s `Instruction::select`,
    // which emits SPIR-V `OpSelect(result, condition, object1=accept,
    // object2=reject)`: "Result is object1 if condition is true") takes
    // `accept` when `condition` is true, `reject` otherwise.
    let key_cmp = if want_max { naga::BinaryOperator::Greater } else { naga::BinaryOperator::Less };
    let cmp = emit_expr(func, target, naga::Expression::Binary { op: key_cmp, left: ka, right: kb });
    let pick = emit_expr(func, target, naga::Expression::Select { condition: cmp, accept: au, reject: bu });

    // nan = (au & 0x7fffffff) > 0x7f800000 || (bu & 0x7fffffff) > 0x7f800000
    let a_abs = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::And, left: au, right: abs_mask });
    let a_is_nan = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::Greater, left: a_abs, right: exp_mask });
    let b_abs = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::And, left: bu, right: abs_mask });
    let b_is_nan = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::Greater, left: b_abs, right: exp_mask });
    let nan = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::LogicalOr, left: a_is_nan, right: b_is_nan });

    // out = As{ Select(nan, 0x7fc00000, pick), Float, convert: None }
    let result_u = emit_expr(func, target, naga::Expression::Select { condition: nan, accept: qnan, reject: pick });
    emit_expr(func, target, naga::Expression::As { expr: result_u, kind: naga::ScalarKind::Float, convert: None })
}

#[cfg(feature = "spirv-backend")]
fn immediate_index_u32(
    function: &sonatina_ir::Function,
    value: sonatina_ir::ValueId,
) -> Option<u32> {
    match function.dfg.value_imm(value)? {
        sonatina_ir::Immediate::I1(value) => Some(u32::from(value)),
        sonatina_ir::Immediate::I8(value) => u32::try_from(value).ok(),
        sonatina_ir::Immediate::I32(value) => u32::try_from(value).ok(),
        sonatina_ir::Immediate::I64(value) => u32::try_from(value).ok(),
        _ => None,
    }
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypedLocalProjection {
    root: sonatina_ir::InstId,
    indices: Vec<u32>,
}

/// Recover the constant structural path from one shader-local allocation to a
/// typed pointer. Dynamic indices deliberately have no path because their
/// aliasing is real and must keep explicit stores.
#[cfg(feature = "spirv-backend")]
fn typed_local_constant_projection(
    function: &sonatina_ir::Function,
    value: sonatina_ir::ValueId,
) -> Option<TypedLocalProjection> {
    use sonatina_ir::InstDowncast;

    let instruction = function.dfg.value_inst(value)?;
    let data = function.dfg.inst(instruction);
    if <&sonatina_ir::inst::data::Alloca as InstDowncast>::downcast(function.inst_set(), data).is_some() {
        return Some(TypedLocalProjection { root: instruction, indices: Vec::new() });
    }
    if let Some(gep) = <&sonatina_ir::inst::data::Gep as InstDowncast>::downcast(function.inst_set(), data) {
        let (&base, indices) = gep.values().split_first()?;
        let (&leading, indices) = indices.split_first()?;
        if immediate_index_u32(function, leading) != Some(0) {
            return None;
        }
        let mut projection = typed_local_constant_projection(function, base)?;
        projection.indices.extend(indices.iter().map(|&index| immediate_index_u32(function, index)).collect::<Option<Vec<_>>>()?);
        return Some(projection);
    }
    if let Some(bitcast) = <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(function.inst_set(), data) {
        let mut projection = typed_local_constant_projection(function, *bitcast.from())?;
        projection.indices.extend(typed_local_zero_projection_path(
            function.ctx(), function.dfg.value_ty(*bitcast.from()), *bitcast.ty(),
        )?);
        return Some(projection);
    }
    None
}

#[cfg(feature = "spirv-backend")]
fn typed_local_alloca_root(
    function: &sonatina_ir::Function,
    value: sonatina_ir::ValueId,
) -> Option<sonatina_ir::InstId> {
    use sonatina_ir::InstDowncast;

    let instruction = function.dfg.value_inst(value)?;
    let data = function.dfg.inst(instruction);
    if <&sonatina_ir::inst::data::Alloca as InstDowncast>::downcast(function.inst_set(), data).is_some() {
        return Some(instruction);
    }
    if let Some(gep) = <&sonatina_ir::inst::data::Gep as InstDowncast>::downcast(function.inst_set(), data) {
        return typed_local_alloca_root(function, *gep.values().first()?);
    }
    if let Some(bitcast) = <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(function.inst_set(), data) {
        return typed_local_alloca_root(function, *bitcast.from());
    }
    None
}

#[cfg(feature = "spirv-backend")]
fn immediate_is_shader_zero(value: sonatina_ir::Immediate) -> bool {
    matches!(
        value,
        sonatina_ir::Immediate::I1(false)
            | sonatina_ir::Immediate::I8(0)
            | sonatina_ir::Immediate::I16(0)
            | sonatina_ir::Immediate::I32(0)
            | sonatina_ir::Immediate::F32(0)
            | sonatina_ir::Immediate::I64(0)
            | sonatina_ir::Immediate::I128(0)
    )
}

#[cfg(feature = "spirv-backend")]
fn typed_local_projections_may_alias(left: &TypedLocalProjection, right: &TypedLocalProjection) -> bool {
    left.root == right.root
        && left.indices.iter().zip(&right.indices).all(|(left, right)| left == right)
}

/// Prove that no path from the typed allocation to `before` can have changed
/// `target`. The allocation must dominate the store and both blocks must be
/// acyclic. Those restrictions keep this first analysis exact for
/// compiler-authored initialization while loops, dynamic aliases, and borrowed
/// calls continue to fail closed.
#[cfg(feature = "spirv-backend")]
fn typed_local_projection_is_pristine_before(
    function: &sonatina_ir::Function,
    target: &TypedLocalProjection,
    before: sonatina_ir::InstId,
) -> bool {
    use sonatina_ir::InstDowncast;

    let root_block = function.layout.inst_block(target.root);
    let target_block = function.layout.inst_block(before);
    if block_is_in_cfg_cycle(function, root_block) || block_is_in_cfg_cycle(function, target_block) {
        return false;
    }

    let mut cfg = sonatina_ir::ControlFlowGraph::default();
    cfg.compute(function);
    let mut domtree = crate::domtree::DomTree::new();
    domtree.compute(&cfg);
    if !domtree.dominates(root_block, target_block) {
        return false;
    }

    let mut reachable_from_root = std::collections::HashSet::new();
    let mut pending = vec![root_block];
    while let Some(block) = pending.pop() {
        if reachable_from_root.insert(block) {
            pending.extend(cfg.succs_of(block).copied());
        }
    }

    let mut can_reach_target = std::collections::HashSet::new();
    pending.push(target_block);
    while let Some(block) = pending.pop() {
        if can_reach_target.insert(block) {
            pending.extend(cfg.preds_of(block).copied());
        }
    }

    if !reachable_from_root.contains(&target_block) || !can_reach_target.contains(&root_block) {
        return false;
    }

    for block in reachable_from_root.intersection(&can_reach_target).copied() {
        let mut root_seen = block != root_block;
        for instruction in function.layout.iter_inst(block) {
            if block == root_block && instruction == target.root {
                root_seen = true;
                continue;
            }
            if !root_seen {
                continue;
            }
            if block == target_block && instruction == before {
                break;
            }

            let data = function.dfg.inst(instruction);
            if let Some(previous) = <&sonatina_ir::inst::data::Mstore as InstDowncast>::downcast(function.inst_set(), data)
                && typed_local_alloca_root(function, *previous.addr()) == Some(target.root)
            {
                if typed_local_alloca_root(function, *previous.value()) == Some(target.root) {
                    return false;
                }
                let Some(previous_projection) = typed_local_constant_projection(function, *previous.addr()) else {
                    return false;
                };
                if typed_local_projections_may_alias(&previous_projection, target) {
                    return false;
                }
                continue;
            }
            if let Some(call) = <&sonatina_ir::inst::control_flow::Call as InstDowncast>::downcast(function.inst_set(), data)
                && call.args().iter().any(|&argument| typed_local_alloca_root(function, argument) == Some(target.root))
            {
                return false;
            }
            let consumes_rooted_pointer = data.collect_values().iter().any(|&value| {
                typed_local_alloca_root(function, value) == Some(target.root)
            });
            if consumes_rooted_pointer
                && <&sonatina_ir::inst::data::Mload as InstDowncast>::downcast(function.inst_set(), data).is_none()
                && <&sonatina_ir::inst::data::Gep as InstDowncast>::downcast(function.inst_set(), data).is_none()
                && <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(function.inst_set(), data).is_none()
            {
                return false;
            }
        }
    }
    true
}

/// Naga function locals are explicitly initialized with `ZeroValue`. A
/// zero store to a constant, non-escaping typed projection is redundant while
/// that projection still has its initial value. The exact CFG slice above
/// admits compiler-authored initialization in acyclic child blocks. Dynamic
/// projections, overlapping aggregate writes, loops, and borrowed calls retain
/// the store.
#[cfg(feature = "spirv-backend")]
fn typed_local_zero_store_is_redundant(
    function: &sonatina_ir::Function,
    store_instruction: sonatina_ir::InstId,
    store: &sonatina_ir::inst::data::Mstore,
) -> bool {
    use sonatina_ir::InstDowncast;

    let Some(value) = function.dfg.value_imm(*store.value()) else { return false };
    if !immediate_is_shader_zero(value) {
        return false;
    }
    let Some(target) = typed_local_constant_projection(function, *store.addr()) else { return false };
    if typed_local_projection_is_pristine_before(function, &target, store_instruction) {
        return true;
    }
    let Some(entry) = function.layout.entry_block() else { return false };
    let mut root_seen = false;
    let mut target_is_zero = false;
    for instruction in function.layout.iter_inst(entry) {
        if instruction == store_instruction {
            return root_seen && target_is_zero;
        }
        if instruction == target.root {
            root_seen = true;
            target_is_zero = true;
            continue;
        }
        if !root_seen {
            continue;
        }
        let data = function.dfg.inst(instruction);
        if let Some(previous) = <&sonatina_ir::inst::data::Mstore as InstDowncast>::downcast(function.inst_set(), data)
            && typed_local_alloca_root(function, *previous.addr()) == Some(target.root)
        {
            let Some(previous_projection) = typed_local_constant_projection(function, *previous.addr()) else {
                target_is_zero = false;
                continue;
            };
            if !typed_local_projections_may_alias(&previous_projection, &target) {
                continue;
            }
            target_is_zero = previous_projection == target
                && function.dfg.value_imm(*previous.value()).is_some_and(immediate_is_shader_zero);
            continue;
        }
        if let Some(call) = <&sonatina_ir::inst::control_flow::Call as InstDowncast>::downcast(function.inst_set(), data)
            && call.args().iter().any(|&argument| typed_local_alloca_root(function, argument) == Some(target.root))
        {
            target_is_zero = false;
        }
    }
    false
}

/// SCCP represents an all-zero typed `Gep` as a pointer `Bitcast` when the
/// source and projected pointer types differ. Naga has no pointer bitcast, but
/// it does have the exact structural operation: one zero `AccessIndex` per
/// leading array element or unpacked struct field. Admit only that derivable
/// path, never a general pointer reinterpretation.
#[cfg(feature = "spirv-backend")]
fn typed_local_zero_projection_path(
    ctx: &sonatina_ir::module::ModuleCtx,
    from_ty: sonatina_ir::Type,
    to_ty: sonatina_ir::Type,
) -> Option<Vec<u32>> {
    let sonatina_ir::types::CompoundType::Ptr(mut current) = from_ty.resolve_compound(ctx)? else {
        return None;
    };
    let sonatina_ir::types::CompoundType::Ptr(target) = to_ty.resolve_compound(ctx)? else {
        return None;
    };
    let mut path = Vec::new();
    while current != target {
        current = match current.resolve_compound(ctx)? {
            sonatina_ir::types::CompoundType::Array { elem, len } if len > 0 => elem,
            sonatina_ir::types::CompoundType::Struct(data) if !data.packed => {
                *data.fields.first()?
            }
            _ => return None,
        };
        path.push(0);
    }
    Some(path)
}

/// Rebuild a typed private pointer projection in the lexical Naga block that
/// consumes it. Sonatina SSA permits a pointer computed in a loop header to
/// dominate a later exit block. Once that CFG is structurized, the original
/// Naga `Access` expression lives inside the loop body and is not in scope at
/// the later call. The allocation or parameter root remains in function
/// scope, so replaying the exact typed projection is both equivalent and the
/// natural WGSL representation.
#[cfg(feature = "spirv-backend")]
fn rematerialize_typed_pointer_projection(
    value: sonatina_ir::ValueId,
    function: &sonatina_ir::Function,
    word: WordKind,
    value_map: &std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    func: &mut naga::Function,
    target: &mut naga::Block,
) -> Result<Option<naga::Handle<naga::Expression>>, String> {
    use sonatina_ir::InstDowncast;

    let Some(inst_id) = function.dfg.value_inst(value) else {
        return Ok(None);
    };
    let inst_data = function.dfg.inst(inst_id);

    if let Some(gep) = <&sonatina_ir::inst::data::Gep as InstDowncast>::downcast(
        function.inst_set(), inst_data,
    ) {
        let Some((&base_value, indices)) = gep.values().split_first() else {
            return Err("spirv: typed pointer rematerialization found a Gep with no base".to_string());
        };
        let Some((&leading_index, indices)) = indices.split_first() else {
            return Err("spirv: typed pointer rematerialization requires a leading object index".to_string());
        };
        if immediate_index_u32(function, leading_index) != Some(0) {
            return Err("spirv: typed pointer rematerialization requires a zero leading object index".to_string());
        }
        let mut base = if let Some(projected) = rematerialize_typed_pointer_projection(
            base_value, function, word, value_map, phi_locals, func, target,
        )? {
            projected
        } else {
            resolve_naga_value(base_value, function, word, value_map, phi_locals, func)
                .ok_or_else(|| format!("spirv: typed pointer rematerialization cannot resolve base {base_value:?}"))?
        };
        let base_ty = function.dfg.value_ty(base_value);
        let Some(sonatina_ir::types::CompoundType::Ptr(mut pointee)) =
            base_ty.resolve_compound(function.ctx())
        else {
            return Err(format!("spirv: typed pointer rematerialization found non-pointer base {base_ty:?}"));
        };
        for &index in indices {
            match pointee.resolve_compound(function.ctx()) {
                Some(sonatina_ir::types::CompoundType::Struct(data)) if !data.packed => {
                    let field = immediate_index_u32(function, index).ok_or_else(|| {
                        "spirv: rematerialized struct projection requires a constant field".to_string()
                    })?;
                    let &field_ty = data.fields.get(field as usize).ok_or_else(|| {
                        format!("spirv: rematerialized struct field {field} is out of bounds")
                    })?;
                    base = emit_expr(func, target, naga::Expression::AccessIndex { base, index: field });
                    pointee = field_ty;
                }
                Some(sonatina_ir::types::CompoundType::Array { elem, len }) => {
                    if let Some(constant) = immediate_index_u32(function, index) {
                        if constant as usize >= len {
                            return Err(format!("spirv: rematerialized array index {constant} is out of bounds for length {len}"));
                        }
                        base = emit_expr(func, target, naga::Expression::AccessIndex { base, index: constant });
                    } else {
                        if word != WordKind::U32 {
                            return Err("spirv: dynamic rematerialized array projection requires the u32 browser word".to_string());
                        }
                        let index = resolve_naga_value(
                            index, function, word, value_map, phi_locals, func,
                        ).ok_or_else(|| "spirv: rematerialized array projection index is unresolved".to_string())?;
                        base = emit_expr(func, target, naga::Expression::Access { base, index });
                    }
                    pointee = elem;
                }
                _ => return Err(format!("spirv: typed pointer rematerialization cannot project through {pointee:?}")),
            }
        }
        return Ok(Some(base));
    }

    if let Some(bitcast) = <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(
        function.inst_set(), inst_data,
    ) {
        let from_ty = function.dfg.value_ty(*bitcast.from());
        let Some(path) = typed_local_zero_projection_path(function.ctx(), from_ty, *bitcast.ty()) else {
            return Ok(None);
        };
        let mut base = if let Some(projected) = rematerialize_typed_pointer_projection(
            *bitcast.from(), function, word, value_map, phi_locals, func, target,
        )? {
            projected
        } else {
            resolve_naga_value(*bitcast.from(), function, word, value_map, phi_locals, func)
                .ok_or_else(|| format!(
                    "spirv: typed pointer rematerialization cannot resolve zero projection base {:?}",
                    bitcast.from(),
                ))?
        };
        for index in path {
            base = emit_expr(func, target, naga::Expression::AccessIndex { base, index });
        }
        return Ok(Some(base));
    }

    Ok(None)
}

/// Emit a single arithmetic/cmp instruction into the given target block.
/// Returns the expression handle if an instruction was emitted, None otherwise.
/// Skips Phi, Jump, Br, and Return instructions.
#[cfg(feature = "spirv-backend")]
fn emit_single_inst(
    inst_id: sonatina_ir::InstId,
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    func: &mut naga::Function,
    target: &mut naga::Block,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    result_expr: &mut Option<naga::Handle<naga::Expression>>,
    return_abi: Option<&NagaResultAbi>,
    naga_functions: &NagaFunctionMap,
    mem_ctx: Option<MemCtx>,
    // Hygiene ask (adversarial review, 2026-08-08): the Mem-op arms below
    // resolve operands that the has_mem pre-scan already proved exist
    // (a `MemAllocDynamic`/`Mload`/`Mstore` operand unresolvable here would
    // be an upstream compiler-invariant violation, not a user error), so an
    // unresolvable operand is vanishingly unlikely -- but a bare `.unwrap()`
    // would still crash the process instead of surfacing a named
    // `SpirvError::Translation`. On the (never-expected) `None` case, the
    // Mem arms record a message here and return `false` instead of
    // panicking; `translate_to_naga` checks this after emission completes.
    mem_error: &mut Option<String>,
) -> bool {
    use sonatina_ir::InstDowncast;
    let inst_data = function.dfg.inst(inst_id);

    // Skip phi/jump/br
    if <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst_data).is_some() { return false; }
    if <&sonatina_ir::inst::control_flow::Jump as InstDowncast>::downcast(inst_set, inst_data).is_some() { return false; }
    if <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, inst_data).is_some() { return false; }

    if let Some(add) = <&sonatina_ir::inst::arith::Add as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*add.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(*add.rhs(), function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga::BinaryOperator::Add, left: lhs, right: rhs },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(sub) = <&sonatina_ir::inst::arith::Sub as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*sub.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(*sub.rhs(), function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga::BinaryOperator::Subtract, left: lhs, right: rhs },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(mul) = <&sonatina_ir::inst::arith::Mul as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*mul.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(*mul.rhs(), function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga::BinaryOperator::Multiply, left: lhs, right: rhs },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Fneg as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let arg = resolve_naga_value(*op.arg(), function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(naga::Expression::Unary { op: naga::UnaryOperator::Negate, expr: arg }, naga::Span::UNDEFINED);
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Fsqrt as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let arg = resolve_naga_value(*op.arg(), function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(naga::Expression::Math { fun: naga::MathFunction::Sqrt, arg, arg1: None, arg2: None, arg3: None }, naga::Span::UNDEFINED);
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Fabs as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let arg = resolve_naga_value(*op.arg(), function, word, value_map, phi_locals, func).unwrap();
            // Exact, branch-free: bitcast to u32, clear the sign bit, bitcast
            // back. A pure bitwise op on every backend (matches wasm's
            // `f32.abs` and cranelift's `fabs`), so this is not a relaxation
            // -- MathFunction::Abs (GLSL.std.450 `FAbs`) is ALSO exactly the
            // bitwise sign-clear, but the explicit bitcast form here matches
            // the same integer toolkit as Fmin/Fmax/Fclamp below and avoids
            // depending on the extended-instruction set's semantics.
            let au = emit_expr(func, target, naga::Expression::As { expr: arg, kind: naga::ScalarKind::Uint, convert: None });
            let abs_mask = lit_u32(func, 0x7fff_ffff);
            let cleared = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::And, left: au, right: abs_mask });
            let h = emit_expr(func, target, naga::Expression::As { expr: cleared, kind: naga::ScalarKind::Float, convert: None });
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Ffloor as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let arg = resolve_naga_value(*op.arg(), function, word, value_map, phi_locals, func).unwrap();
            // Single native instruction, no bit-twiddling, no NaN/-0
            // subtlety: floor is monotone and sign-preserving at the
            // boundary. Matches wasm's `f32.floor`/cranelift's `floor`.
            let h = func.expressions.append(naga::Expression::Math { fun: naga::MathFunction::Floor, arg, arg1: None, arg2: None, arg3: None }, naga::Span::UNDEFINED);
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Fceil as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let arg = resolve_naga_value(*op.arg(), function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(naga::Expression::Math { fun: naga::MathFunction::Ceil, arg, arg1: None, arg2: None, arg3: None }, naga::Span::UNDEFINED);
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Ftrunc as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let arg = resolve_naga_value(*op.arg(), function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(naga::Expression::Math { fun: naga::MathFunction::Trunc, arg, arg1: None, arg2: None, arg3: None }, naga::Span::UNDEFINED);
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Fround as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let arg = resolve_naga_value(*op.arg(), function, word, value_map, phi_locals, func).unwrap();
            // `MathFunction::Round` lowers to GLSL.std.450 `RoundEven`
            // (verified against naga 29.0.4's SPIR-V backend source, NOT
            // the ties-away-from-zero `Round` ext inst), matching wasm's
            // `f32.nearest`/cranelift's `nearest` exactly. No divergence to
            // pin around, unlike `Fmin`/`Fmax`.
            let h = func.expressions.append(naga::Expression::Math { fun: naga::MathFunction::Round, arg, arg1: None, arg2: None, arg3: None }, naga::Span::UNDEFINED);
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Fmin as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*op.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(*op.rhs(), function, word, value_map, phi_locals, func).unwrap();
            // PINNED-EXACT: branch-free integer key compare + OpSelect (see
            // `emit_exact_fminmax`). Matches wasm's `f32.min`/cranelift's
            // `fmin` ("WebAssembly rules") bit-for-bit, including NaN and
            // -0.0/+0.0. Formerly `MathFunction::Min` (GLSL.std.450 `FMin`,
            // implementation-defined on NaN/-0.0) -- that was the resolved
            // OPEN DECISION; see docs/numeric-intrinsics-semantics.md.
            let h = emit_exact_fminmax(func, target, lhs, rhs, false);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Fmax as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*op.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(*op.rhs(), function, word, value_map, phi_locals, func).unwrap();
            // PINNED-EXACT: see Fmin above (want_max = true selects the
            // larger integer key instead of the smaller).
            let h = emit_exact_fminmax(func, target, lhs, rhs, true);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::FminRelaxed as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*op.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(*op.rhs(), function, word, value_map, phi_locals, func).unwrap();
            // RELAXED: the single GLSL.std.450 `FMin` op via naga
            // `MathFunction::Min` -- implementation-defined on NaN operands
            // and signed-zero ties, which is exactly the latitude
            // `FminRelaxed`'s contract grants (unlike `Fmin` above, which
            // must use the ~15-20-op exact integer expansion). This is the
            // pre-slice-0 `Fmin`/`Fmax` lowering relocated here: the whole
            // point of the relaxed op is to keep this cheap 1-op path
            // reachable on GPU for code that opts in via `Regular`.
            let h = func.expressions.append(
                naga::Expression::Math { fun: naga::MathFunction::Min, arg: lhs, arg1: Some(rhs), arg2: None, arg3: None },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::FmaxRelaxed as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*op.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(*op.rhs(), function, word, value_map, phi_locals, func).unwrap();
            // RELAXED: see FminRelaxed above (single GLSL.std.450 `FMax` via
            // `MathFunction::Max`).
            let h = func.expressions.append(
                naga::Expression::Math { fun: naga::MathFunction::Max, arg: lhs, arg1: Some(rhs), arg2: None, arg3: None },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(op) = <&sonatina_ir::inst::arith::Fclamp as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let arg = resolve_naga_value(*op.arg(), function, word, value_map, phi_locals, func).unwrap();
            let lo = resolve_naga_value(*op.lo(), function, word, value_map, phi_locals, func).unwrap();
            let hi = resolve_naga_value(*op.hi(), function, word, value_map, phi_locals, func).unwrap();
            // PINNED-EXACT: compose as min(max(arg, lo), hi) from the exact
            // Fmin/Fmax expansions above (not a single GLSL.std.450
            // `FClamp`, which is spec-undefined/poison when lo > hi). This
            // keeps `lo > hi` DEFINED as `hi` (matching wasm/cranelift's
            // composed clamp) on every input, not just finite ones, and
            // stays branch-free throughout (every op is integer
            // compare-and-select, no control flow).
            let max_h = emit_exact_fminmax(func, target, arg, lo, true);
            let h = emit_exact_fminmax(func, target, max_h, hi, false);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some((lhs_id, rhs_id, naga_op)) =
        <&sonatina_ir::inst::arith::Udiv as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Divide))
        .or_else(|| <&sonatina_ir::inst::arith::Umod as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Modulo)))
        .or_else(|| <&sonatina_ir::inst::arith::Fadd as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Add)))
        .or_else(|| <&sonatina_ir::inst::arith::Fsub as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Subtract)))
        .or_else(|| <&sonatina_ir::inst::arith::Fmul as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Multiply)))
        .or_else(|| <&sonatina_ir::inst::arith::Fdiv as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Divide)))
        .or_else(|| <&sonatina_ir::inst::cmp::Feq as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Equal)))
        .or_else(|| <&sonatina_ir::inst::cmp::Flt as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Less)))
        .or_else(|| <&sonatina_ir::inst::cmp::Fle as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::LessEqual)))
    {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(lhs_id, function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(rhs_id, function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(naga::Expression::Binary { op: naga_op, left: lhs, right: rhs }, naga::Span::UNDEFINED);
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some((from_id, signed, to_float)) =
        <&sonatina_ir::inst::cast::I32ToF32 as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.from(), true, true))
        .or_else(|| <&sonatina_ir::inst::cast::U32ToF32 as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.from(), false, true)))
        .or_else(|| <&sonatina_ir::inst::cast::F32ToI32 as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.from(), true, false)))
        .or_else(|| <&sonatina_ir::inst::cast::F32ToU32 as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.from(), false, false)))
    {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let from = resolve_naga_value(from_id, function, word, value_map, phi_locals, func).unwrap();
            let converted = if to_float {
                let numeric = if signed {
                    let signed_bits = func.expressions.append(naga::Expression::As { expr: from, kind: naga::ScalarKind::Sint, convert: None }, naga::Span::UNDEFINED);
                    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(signed_bits, signed_bits)), naga::Span::UNDEFINED);
                    signed_bits
                } else { from };
                func.expressions.append(naga::Expression::As { expr: numeric, kind: naga::ScalarKind::Float, convert: Some(4) }, naga::Span::UNDEFINED)
            } else {
                let kind = if signed { naga::ScalarKind::Sint } else { naga::ScalarKind::Uint };
                let numeric = func.expressions.append(naga::Expression::As { expr: from, kind, convert: Some(4) }, naga::Span::UNDEFINED);
                target.push(naga::Statement::Emit(naga::Range::new_from_bounds(numeric, numeric)), naga::Span::UNDEFINED);
                let raw = if signed {
                    let raw = func.expressions.append(naga::Expression::As { expr: numeric, kind: naga::ScalarKind::Uint, convert: None }, naga::Span::UNDEFINED);
                    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(raw, raw)), naga::Span::UNDEFINED);
                    raw
                } else { numeric }
                ;
                // Rust's float-to-int casts, which define the interpreter contract,
                // saturate and map NaN to zero. Shader conversion outside the target
                // range is implementation-defined, so select the boundary values
                // explicitly after the raw conversion.
                let zero = func.expressions.append(naga::Expression::Literal(naga::Literal::U32(0)), naga::Span::UNDEFINED);
                let (low_f, high_f, low_u, high_u) = if signed {
                    (
                        naga::Literal::F32(-2_147_483_648.0),
                        naga::Literal::F32(2_147_483_648.0),
                        naga::Literal::U32(0x8000_0000),
                        naga::Literal::U32(0x7fff_ffff),
                    )
                } else {
                    (
                        naga::Literal::F32(0.0),
                        naga::Literal::F32(4_294_967_296.0),
                        naga::Literal::U32(0),
                        naga::Literal::U32(u32::MAX),
                    )
                };
                let low_f = func.expressions.append(naga::Expression::Literal(low_f), naga::Span::UNDEFINED);
                let high_f = func.expressions.append(naga::Expression::Literal(high_f), naga::Span::UNDEFINED);
                let low_u = func.expressions.append(naga::Expression::Literal(low_u), naga::Span::UNDEFINED);
                let high_u = func.expressions.append(naga::Expression::Literal(high_u), naga::Span::UNDEFINED);
                let is_low = func.expressions.append(naga::Expression::Binary { op: naga::BinaryOperator::LessEqual, left: from, right: low_f }, naga::Span::UNDEFINED);
                target.push(naga::Statement::Emit(naga::Range::new_from_bounds(is_low, is_low)), naga::Span::UNDEFINED);
                let low_clamped = func.expressions.append(naga::Expression::Select { condition: is_low, accept: low_u, reject: raw }, naga::Span::UNDEFINED);
                target.push(naga::Statement::Emit(naga::Range::new_from_bounds(low_clamped, low_clamped)), naga::Span::UNDEFINED);
                let is_high = func.expressions.append(naga::Expression::Binary { op: naga::BinaryOperator::GreaterEqual, left: from, right: high_f }, naga::Span::UNDEFINED);
                target.push(naga::Statement::Emit(naga::Range::new_from_bounds(is_high, is_high)), naga::Span::UNDEFINED);
                let clamped = func.expressions.append(naga::Expression::Select { condition: is_high, accept: high_u, reject: low_clamped }, naga::Span::UNDEFINED);
                target.push(naga::Statement::Emit(naga::Range::new_from_bounds(clamped, clamped)), naga::Span::UNDEFINED);
                let ordered = func.expressions.append(naga::Expression::Binary { op: naga::BinaryOperator::Equal, left: from, right: from }, naga::Span::UNDEFINED);
                target.push(naga::Statement::Emit(naga::Range::new_from_bounds(ordered, ordered)), naga::Span::UNDEFINED);
                func.expressions.append(naga::Expression::Select { condition: ordered, accept: clamped, reject: zero }, naga::Span::UNDEFINED)
            };
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(converted, converted)), naga::Span::UNDEFINED);
            value_map.insert(result, converted);
            return true;
        }
    } else if let Some(trunc) =
        <&sonatina_ir::inst::cast::Trunc as InstDowncast>::downcast(inst_set, inst_data)
    {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let from = resolve_naga_value(
                *trunc.from(),
                function,
                word,
                value_map,
                phi_locals,
                func,
            )
            .unwrap();
            let one = lit_u32(func, 1);
            let low_bit = emit_expr(func, target, naga::Expression::Binary {
                op: naga::BinaryOperator::And,
                left: from,
                right: one,
            });
            let zero = lit_u32(func, 0);
            let value = emit_expr(func, target, naga::Expression::Binary {
                op: naga::BinaryOperator::NotEqual,
                left: low_bit,
                right: zero,
            });
            value_map.insert(result, value);
            return true;
        }
    } else if let Some(bitcast) =
        <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(inst_set, inst_data)
    {
        // A Sonatina Bitcast is a representation-preserving reinterpretation,
        // not a numeric conversion. The browser word admits the exact 32-bit
        // scalar pair needed by storage records: i32 bits <-> f32.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let from_ty = function.dfg.value_ty(*bitcast.from());
            if let Some(path) =
                typed_local_zero_projection_path(function.ctx(), from_ty, *bitcast.ty())
            {
                let Some(mut projected) = resolve_naga_value(
                    *bitcast.from(),
                    function,
                    word,
                    value_map,
                    phi_locals,
                    func,
                ) else {
                    *mem_error = Some(format!(
                        "spirv: typed-local zero projection source {:?} is unresolved. Fail closed.",
                        bitcast.from(),
                    ));
                    return false;
                };
                for index in path {
                    projected = func.expressions.append(
                        naga::Expression::AccessIndex {
                            base: projected,
                            index,
                        },
                        naga::Span::UNDEFINED,
                    );
                }
                value_map.insert(result, projected);
                return true;
            }
            let from = resolve_naga_value(
                *bitcast.from(),
                function,
                word,
                value_map,
                phi_locals,
                func,
            )
            .unwrap();
            let kind = match *bitcast.ty() {
                sonatina_ir::Type::I32 => naga::ScalarKind::Uint,
                sonatina_ir::Type::F32 => naga::ScalarKind::Float,
                _ => unreachable!("unsupported Bitcast rejected by SPIR-V pre-scan"),
            };
            let h = func.expressions.append(
                naga::Expression::As {
                    expr: from,
                    kind,
                    convert: None,
                },
                naga::Span::UNDEFINED,
            );
            target.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(h, h)),
                naga::Span::UNDEFINED,
            );
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(sar) = <&sonatina_ir::inst::arith::Sar as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let val = resolve_naga_value(*sar.value(), function, word, value_map, phi_locals, func).unwrap();
            let bits_u32 = if let Some(imm) = function.dfg.value_imm(*sar.bits()) {
                let shift_amount = match imm {
                    sonatina_ir::Immediate::I64(v) => v as u32,
                    sonatina_ir::Immediate::I32(v) => v as u32,
                    sonatina_ir::Immediate::I8(v) => v as u32,
                    _ => 0,
                };
                func.expressions.append(
                    naga::Expression::Literal(naga::Literal::U32(shift_amount)),
                    naga::Span::UNDEFINED,
                )
            } else {
                debug_assert_eq!(word, WordKind::U32);
                resolve_naga_value(
                    *sar.bits(),
                    function,
                    word,
                    value_map,
                    phi_locals,
                    func,
                )
                .unwrap()
            };
            match word {
                WordKind::U32 => {
                    // The u32 word carries a signed Q12 value; WGSL `>>` on a `u32`
                    // is a LOGICAL shift, so bitcast to i32 (arithmetic `>>`), shift,
                    // then bitcast back to u32. `convert: None` = reinterpret bits.
                    let as_sint = func.expressions.append(
                        naga::Expression::As { expr: val, kind: naga::ScalarKind::Sint, convert: None },
                        naga::Span::UNDEFINED,
                    );
                    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(as_sint, as_sint)), naga::Span::UNDEFINED);
                    let shifted = func.expressions.append(
                        naga::Expression::Binary { op: naga::BinaryOperator::ShiftRight, left: as_sint, right: bits_u32 },
                        naga::Span::UNDEFINED,
                    );
                    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(shifted, shifted)), naga::Span::UNDEFINED);
                    let as_uint = func.expressions.append(
                        naga::Expression::As { expr: shifted, kind: naga::ScalarKind::Uint, convert: None },
                        naga::Span::UNDEFINED,
                    );
                    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(as_uint, as_uint)), naga::Span::UNDEFINED);
                    value_map.insert(result, as_uint);
                }
                WordKind::I64 => {
                    // The i64 word operand is already `Sint`; `>>` is arithmetic.
                    // Byte-identical to the pre-word-aware emission.
                    let h = func.expressions.append(
                        naga::Expression::Binary { op: naga::BinaryOperator::ShiftRight, left: val, right: bits_u32 },
                        naga::Span::UNDEFINED,
                    );
                    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
                    value_map.insert(result, h);
                }
            }
            return true;
        }
    } else if let Some(shl) = <&sonatina_ir::inst::arith::Shl as InstDowncast>::downcast(inst_set, inst_data) {
        // Shift-left is sign-agnostic. Immediate and runtime u32 amounts both map
        // directly to WGSL's integer shift operand.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let val = resolve_naga_value(*shl.value(), function, word, value_map, phi_locals, func).unwrap();
            let bits_u32 = if let Some(imm) = function.dfg.value_imm(*shl.bits()) {
                let shift_amount = match imm {
                    sonatina_ir::Immediate::I64(v) => v as u32,
                    sonatina_ir::Immediate::I32(v) => v as u32,
                    sonatina_ir::Immediate::I8(v) => v as u32,
                    _ => 0,
                };
                func.expressions.append(
                    naga::Expression::Literal(naga::Literal::U32(shift_amount)),
                    naga::Span::UNDEFINED,
                )
            } else {
                debug_assert_eq!(word, WordKind::U32);
                resolve_naga_value(
                    *shl.bits(),
                    function,
                    word,
                    value_map,
                    phi_locals,
                    func,
                )
                .unwrap()
            };
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga::BinaryOperator::ShiftLeft, left: val, right: bits_u32 },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(shr) = <&sonatina_ir::inst::arith::Shr as InstDowncast>::downcast(inst_set, inst_data) {
        // Logical (unsigned) shift right. Fe lowers unsigned `>>` to `Shr`. Under
        // the u32 word this is the EASY case: WGSL `>>` on a `u32` IS a logical
        // shift, so no bitcast dance (unlike `Sar`), just shift the u32 value with
        // the resolved u32 amount. The i64 word fails closed in the pre-scan.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let val = resolve_naga_value(*shr.value(), function, word, value_map, phi_locals, func).unwrap();
            let bits_u32 = if let Some(imm) = function.dfg.value_imm(*shr.bits()) {
                let shift_amount = match imm {
                    sonatina_ir::Immediate::I64(v) => v as u32,
                    sonatina_ir::Immediate::I32(v) => v as u32,
                    sonatina_ir::Immediate::I8(v) => v as u32,
                    _ => 0,
                };
                func.expressions.append(
                    naga::Expression::Literal(naga::Literal::U32(shift_amount)),
                    naga::Span::UNDEFINED,
                )
            } else {
                resolve_naga_value(
                    *shr.bits(),
                    function,
                    word,
                    value_map,
                    phi_locals,
                    func,
                )
                .unwrap()
            };
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga::BinaryOperator::ShiftRight, left: val, right: bits_u32 },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(lt) = <&sonatina_ir::inst::cmp::Lt as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*lt.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(*lt.rhs(), function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga::BinaryOperator::Less, left: lhs, right: rhs },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some((lhs_id, rhs_id, naga_op)) =
        <&sonatina_ir::inst::cmp::Eq as InstDowncast>::downcast(inst_set, inst_data)
            .map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Equal))
            .or_else(|| <&sonatina_ir::inst::cmp::Ne as InstDowncast>::downcast(inst_set, inst_data)
                .map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::NotEqual)))
    {
        // Equality is sign-agnostic: the u32 browser word preserves the exact
        // bit pattern of Sonatina's signless i32 carrier, so no signed cast or
        // numeric substitution is needed.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(lhs_id, function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(rhs_id, function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga_op, left: lhs, right: rhs },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(is_zero) = <&sonatina_ir::inst::cmp::IsZero as InstDowncast>::downcast(inst_set, inst_data) {
        // `is_zero` is also Fe's logical-not carrier for i1. Naga requires both
        // equality operands to have the same type, so a boolean operand compares
        // with `false`; integer operands compare with their active word zero.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*is_zero.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let zero = func.expressions.append(
                naga::Expression::Literal(match function.dfg.value_ty(*is_zero.lhs()) {
                    sonatina_ir::Type::I1 => naga::Literal::Bool(false),
                    sonatina_ir::Type::I64 => naga::Literal::I64(0),
                    _ => match word {
                        WordKind::U32 => naga::Literal::U32(0),
                        WordKind::I64 => naga::Literal::I64(0),
                    },
                }),
                naga::Span::UNDEFINED,
            );
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga::BinaryOperator::Equal, left: lhs, right: zero },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some((lhs_id, rhs_id, naga_op)) =
        <&sonatina_ir::inst::logic::And as InstDowncast>::downcast(inst_set, inst_data)
            .map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::And))
            .or_else(|| <&sonatina_ir::inst::logic::Or as InstDowncast>::downcast(inst_set, inst_data)
                .map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::InclusiveOr)))
            .or_else(|| <&sonatina_ir::inst::logic::Xor as InstDowncast>::downcast(inst_set, inst_data)
                .map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::ExclusiveOr)))
    {
        // Bitwise and/or/xor are sign-agnostic per-bit ops: the u32 browser word
        // and the i64 word both carry the exact bit pattern, and naga defines
        // these operators for Uint and Sint alike, so no cast dance is needed
        // (unlike `Sar`/`Slt`). Direct `lhs/rhs` operand order.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(lhs_id, function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(rhs_id, function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga_op, left: lhs, right: rhs },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(slt) = <&sonatina_ir::inst::cmp::Slt as InstDowncast>::downcast(inst_set, inst_data) {
        // Signed less-than. Under the u32 word the operands carry signed values in
        // two's complement, so bitcast BOTH to i32 (`convert: None` = reinterpret)
        // before the compare; naga `Less` on `Sint` scalars is a signed compare.
        // Under the i64 word the operands are already `Sint`, so compare directly
        // (byte-identical shape to the unsigned `Lt` arm on that word).
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*slt.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let rhs = resolve_naga_value(*slt.rhs(), function, word, value_map, phi_locals, func).unwrap();
            let (left, right) = match word {
                WordKind::U32 => {
                    let ls = func.expressions.append(
                        naga::Expression::As { expr: lhs, kind: naga::ScalarKind::Sint, convert: None },
                        naga::Span::UNDEFINED,
                    );
                    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(ls, ls)), naga::Span::UNDEFINED);
                    let rs = func.expressions.append(
                        naga::Expression::As { expr: rhs, kind: naga::ScalarKind::Sint, convert: None },
                        naga::Span::UNDEFINED,
                    );
                    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(rs, rs)), naga::Span::UNDEFINED);
                    (ls, rs)
                }
                WordKind::I64 => (lhs, rhs),
            };
            let h = func.expressions.append(
                naga::Expression::Binary { op: naga::BinaryOperator::Less, left, right },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(alloca) =
        <&sonatina_ir::inst::data::Alloca as InstDowncast>::downcast(inst_set, inst_data)
    {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let Some(local_ty) = naga_functions.typed_local_type(*alloca.ty()) else {
                *mem_error = Some(format!(
                    "spirv: Alloca type {:?} has no prevalidated Naga function-local representation. Fail closed.",
                    alloca.ty(),
                ));
                return false;
            };
            let zero = func.expressions.append(
                naga::Expression::ZeroValue(local_ty.handle),
                naga::Span::UNDEFINED,
            );
            let local = func.local_variables.append(
                naga::LocalVariable {
                    name: Some(format!("fixed_local_{}", result.0)),
                    ty: local_ty.handle,
                    init: Some(zero),
                },
                naga::Span::UNDEFINED,
            );
            let pointer = func.expressions.append(
                naga::Expression::LocalVariable(local),
                naga::Span::UNDEFINED,
            );
            value_map.insert(result, pointer);
            return true;
        }
    } else if let Some(gep) =
        <&sonatina_ir::inst::data::Gep as InstDowncast>::downcast(inst_set, inst_data)
    {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let Some((&base_value, indices)) = gep.values().split_first() else {
                *mem_error = Some("spirv: Gep has no base pointer. Fail closed.".to_string());
                return false;
            };
            let Some((&leading_index, indices)) = indices.split_first() else {
                *mem_error = Some(
                    "spirv: typed-local Gep requires a leading object index. Fail closed."
                        .to_string(),
                );
                return false;
            };
            if immediate_index_u32(function, leading_index) != Some(0) {
                *mem_error = Some(
                    "spirv: typed-local Gep requires a zero leading object index. Fail closed."
                        .to_string(),
                );
                return false;
            }
            let Some(mut base) = resolve_naga_value(
                base_value,
                function,
                word,
                value_map,
                phi_locals,
                func,
            ) else {
                *mem_error = Some(format!(
                    "spirv: Gep base pointer {base_value:?} is unresolved. Fail closed."
                ));
                return false;
            };
            let base_ty = function.dfg.value_ty(base_value);
            let Some(sonatina_ir::types::CompoundType::Ptr(mut pointee)) =
                base_ty.resolve_compound(function.ctx())
            else {
                *mem_error = Some(format!(
                    "spirv: Gep base {base_value:?} has non-pointer type {base_ty:?}. Fail closed."
                ));
                return false;
            };

            let mut first_projection = None;
            for &index in indices {
                match pointee.resolve_compound(function.ctx()) {
                    Some(sonatina_ir::types::CompoundType::Struct(data)) if !data.packed => {
                        let Some(field) = immediate_index_u32(function, index) else {
                            *mem_error = Some(
                                "spirv: typed-local struct Gep requires a constant field index. Fail closed."
                                    .to_string(),
                            );
                            return false;
                        };
                        let Some(&field_ty) = data.fields.get(field as usize) else {
                            *mem_error = Some(format!(
                                "spirv: typed-local struct Gep field {field} is out of bounds. Fail closed."
                            ));
                            return false;
                        };
                        base = func.expressions.append(
                            naga::Expression::AccessIndex { base, index: field },
                            naga::Span::UNDEFINED,
                        );
                        first_projection.get_or_insert(base);
                        pointee = field_ty;
                    }
                    Some(sonatina_ir::types::CompoundType::Array { elem, len }) => {
                        if let Some(constant) = immediate_index_u32(function, index) {
                            if constant as usize >= len {
                                *mem_error = Some(format!(
                                    "spirv: typed-local array Gep index {constant} is out of bounds for length {len}. Fail closed."
                                ));
                                return false;
                            }
                            base = func.expressions.append(
                                naga::Expression::AccessIndex {
                                    base,
                                    index: constant,
                                },
                                naga::Span::UNDEFINED,
                            );
                            first_projection.get_or_insert(base);
                        } else {
                            if word != WordKind::U32 {
                                *mem_error = Some(
                                    "spirv: dynamic typed-local array Gep requires the u32 browser word. Fail closed."
                                        .to_string(),
                                );
                                return false;
                            }
                            let Some(index) = resolve_naga_value(
                                index,
                                function,
                                word,
                                value_map,
                                phi_locals,
                                func,
                            ) else {
                                *mem_error = Some(
                                    "spirv: typed-local array Gep index is unresolved. Fail closed."
                                        .to_string(),
                                );
                                return false;
                            };
                            base = func.expressions.append(
                                naga::Expression::Access { base, index },
                                naga::Span::UNDEFINED,
                            );
                            first_projection.get_or_insert(base);
                        }
                        pointee = elem;
                    }
                    _ => {
                        *mem_error = Some(format!(
                            "spirv: typed-local Gep cannot project through {pointee:?}. Fail closed."
                        ));
                        return false;
                    }
                }
            }

            let result_ty = function.dfg.value_ty(result);
            let result_pointee = match result_ty.resolve_compound(function.ctx()) {
                Some(sonatina_ir::types::CompoundType::Ptr(result_pointee)) => result_pointee,
                _ => {
                    *mem_error = Some(format!(
                        "spirv: Gep result has non-pointer type {result_ty:?}. Fail closed."
                    ));
                    return false;
                }
            };
            if result_pointee != pointee {
                *mem_error = Some(format!(
                    "spirv: Gep result pointee {result_pointee:?} does not match projected type {pointee:?}. Fail closed."
                ));
                return false;
            }
            if let Some(first) = first_projection {
                target.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(first, base)),
                    naga::Span::UNDEFINED,
                );
            }
            value_map.insert(result, base);
            return true;
        }
    } else if <&sonatina_ir::inst::data::ObjAlloc as InstDowncast>::downcast(inst_set, inst_data).is_some() {
        // ObjAlloc in SPIR-V: the output storage buffer IS the allocation.
        // Map the result to the output buffer global variable expression.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            if let Some(&buf_expr) = value_map.get(&sonatina_ir::ValueId(u32::MAX)) {
                value_map.insert(result, buf_expr);
                return true;
            }
        }
    } else if let Some(obj_store) = <&sonatina_ir::inst::data::ObjStore as InstDowncast>::downcast(inst_set, inst_data) {
        // ObjStore: store value at the pointer (which is an Access expression into the buffer)
        let dest = resolve_naga_value(*obj_store.object(), function, word, value_map, phi_locals, func).unwrap();
        let val = resolve_naga_value(*obj_store.value(), function, word, value_map, phi_locals, func).unwrap();
        target.push(naga::Statement::Store { pointer: dest, value: val }, naga::Span::UNDEFINED);
        return true;
    } else if let Some(obj_load) = <&sonatina_ir::inst::data::ObjLoad as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let ptr = resolve_naga_value(*obj_load.object(), function, word, value_map, phi_locals, func).unwrap();
            let h = func.expressions.append(
                naga::Expression::Load { pointer: ptr },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(obj_index) = <&sonatina_ir::inst::data::ObjIndex as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let base = resolve_naga_value(*obj_index.object(), function, word, value_map, phi_locals, func).unwrap();
            let index = resolve_naga_value(*obj_index.index(), function, word, value_map, phi_locals, func).unwrap();
            // Cast i64 index to i32 for array access
            let i32_idx = func.expressions.append(
                naga::Expression::As {
                    expr: index,
                    kind: naga::ScalarKind::Sint,
                    convert: Some(4),
                },
                naga::Span::UNDEFINED,
            );
            target.push(naga::Statement::Emit(naga::Range::new_from_bounds(i32_idx, i32_idx)), naga::Span::UNDEFINED);
            // Access returns a pointer, so no Emit is needed (like LocalVariable/GlobalVariable)
            let h = func.expressions.append(
                naga::Expression::Access { base, index: i32_idx },
                naga::Span::UNDEFINED,
            );
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(obj_proj) =
        <&sonatina_ir::inst::data::ObjProj as InstDowncast>::downcast(inst_set, inst_data)
    {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let Some((&object, indices)) = obj_proj.values().split_first() else {
                *mem_error = Some("spirv: ObjProj has no object operand".to_string());
                return false;
            };
            let Some(mut base) = resolve_naga_value(
                object,
                function,
                word,
                value_map,
                phi_locals,
                func,
            ) else {
                *mem_error = Some(format!(
                    "spirv: ObjProj object operand {object:?} is unresolved"
                ));
                return false;
            };
            for &index in indices {
                let Some(immediate) = function.dfg.value_imm(index) else {
                    *mem_error = Some(
                        "spirv: ObjProj requires compile-time field indices".to_string(),
                    );
                    return false;
                };
                let field = match immediate {
                    sonatina_ir::Immediate::I8(value) => value as u8 as u32,
                    sonatina_ir::Immediate::I32(value) => value as u32,
                    sonatina_ir::Immediate::I64(value) => value as u32,
                    _ => {
                        *mem_error = Some(
                            "spirv: ObjProj field index has an unsupported immediate type"
                                .to_string(),
                        );
                        return false;
                    }
                };
                base = func.expressions.append(
                    naga::Expression::AccessIndex { base, index: field },
                    naga::Span::UNDEFINED,
                );
            }
            value_map.insert(result, base);
            return true;
        }
    } else if <&sonatina_ir::inst::data::MemCheckpoint as InstDowncast>::downcast(
        inst_set, inst_data,
    )
    .is_some()
    {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let mem_ctx = mem_ctx.expect("MemCheckpoint requires mem_ctx (has_mem pre-scan gate)");
            let heap_ctx = mem_ctx.heap();
            let checkpoint = emit_expr(
                func,
                target,
                naga::Expression::Load {
                    pointer: heap_ctx.bump,
                },
            );
            value_map.insert(result, checkpoint);
            return true;
        }
    } else if let Some(rewind) =
        <&sonatina_ir::inst::data::MemRewind as InstDowncast>::downcast(inst_set, inst_data)
    {
        let mem_ctx = mem_ctx.expect("MemRewind requires mem_ctx (has_mem pre-scan gate)");
        let heap_ctx = mem_ctx.heap();
        let Some(checkpoint) = resolve_naga_value(
            *rewind.checkpoint(),
            function,
            word,
            value_map,
            phi_locals,
            func,
        ) else {
            *mem_error = Some(format!(
                "spirv: MemRewind checkpoint operand {:?} is unresolved (compiler invariant violation: the arena-scope pre-scan already proved this value exists)",
                rewind.checkpoint()
            ));
            return false;
        };
        let current = emit_expr(
            func,
            target,
            naga::Expression::Load {
                pointer: heap_ctx.bump,
            },
        );
        let invalid = emit_expr(
            func,
            target,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Greater,
                left: checkpoint,
                right: current,
            },
        );
        mark_trapped_if(func, target, mem_ctx, invalid);
        let rewound = emit_expr(
            func,
            target,
            naga::Expression::Select {
                condition: invalid,
                accept: current,
                reject: checkpoint,
            },
        );
        target.push(
            naga::Statement::Store {
                pointer: heap_ctx.bump,
                value: rewound,
            },
            naga::Span::UNDEFINED,
        );
        return true;
    } else if let Some(alloc) = <&sonatina_ir::inst::data::MemAllocDynamic as InstDowncast>::downcast(inst_set, inst_data) {
        // Private-storage heap emulation (RUNG3_SPIRV_ARRAYS_DESIGN.md section
        // 2): fe_bump is a monotone bump pointer into fe_heap.
        //
        // Guards review finding 1 (silent heap-exhaustion aliasing). The pre-scan in
        // `translate_to_naga` already PROVES, at compile time, that the scoped
        // high-water bound of every MemAllocDynamic in this function is <=
        // heap_ctx.heap_words*4. Loop allocations are admitted only when an
        // independently verified MemCheckpoint/MemRewind scope proves that
        // each iteration restores the arena before its backedge. So the
        // overflow this guard checks for is unreachable by construction in
        // any module this translator accepts.
        // The runtime check below is a second, independent line of defense:
        // if it is ever reached anyway (e.g. a future relaxation of the
        // pre-scan), the allocation is refused (bump frozen at its old value,
        // not silently advanced past capacity) and `mem_ctx.trapped` is
        // raised, rather than the old bump/no-check scheme where excess
        // allocations silently clamp-and-alias onto the SAME last heap word.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let mem_ctx = mem_ctx.expect("MemAllocDynamic requires mem_ctx (has_mem pre-scan gate)");
            let heap_ctx = mem_ctx.heap();
            let Some(size) = resolve_naga_value(*alloc.size(), function, word, value_map, phi_locals, func) else {
                *mem_error = Some(format!(
                    "spirv: MemAllocDynamic size operand {:?} is unresolved (compiler invariant \
                     violation: the has_mem pre-scan already proved this value exists)",
                    alloc.size()
                ));
                return false;
            };
            let old_bump = emit_expr(func, target, naga::Expression::Load { pointer: heap_ctx.bump });
            let new_bump = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::Add, left: old_bump, right: size });
            // Unsigned wraparound check (new_bump < old_bump means size wrapped
            // the u32 add) OR new_bump exceeds the declared heap capacity.
            let overflowed = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::Less, left: new_bump, right: old_bump });
            let cap = lit_u32(func, heap_ctx.heap_words.saturating_mul(4));
            let too_big = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::Greater, left: new_bump, right: cap });
            let bad = emit_expr(func, target, naga::Expression::Binary { op: naga::BinaryOperator::LogicalOr, left: overflowed, right: too_big });
            mark_trapped_if(func, target, mem_ctx, bad);
            // Freeze the bump pointer on overflow instead of advancing it past
            // capacity: keep_old = select(bad, old_bump, new_bump).
            let kept_bump = emit_expr(func, target, naga::Expression::Select { condition: bad, accept: old_bump, reject: new_bump });
            target.push(naga::Statement::Store { pointer: heap_ctx.bump, value: kept_bump }, naga::Span::UNDEFINED);
            value_map.insert(result, old_bump);
            return true;
        }
    } else if let Some(load) = <&sonatina_ir::inst::data::Mload as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let Some(addr) = resolve_naga_value(*load.addr(), function, word, value_map, phi_locals, func) else {
                *mem_error = Some(format!(
                    "spirv: Mload addr operand {:?} is unresolved (compiler invariant \
                     violation: the has_mem pre-scan already proved this value exists)",
                    load.addr()
                ));
                return false;
            };
            if function.dfg.value_ty(*load.addr()).is_pointer(function.ctx()) {
                let loaded = func.expressions.append(
                    naga::Expression::Load { pointer: addr },
                    naga::Span::UNDEFINED,
                );
                target.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)),
                    naga::Span::UNDEFINED,
                );
                value_map.insert(result, loaded);
                return true;
            }
            let mem_ctx = mem_ctx.expect("byte-arena Mload requires mem_ctx");
            let elem = emit_mem_access(
                func,
                target,
                mem_ctx,
                addr,
                *load.ty() == sonatina_ir::Type::I32,
            );
            let loaded_word = emit_expr(func, target, naga::Expression::Load { pointer: elem });
            let loaded = if *load.ty() == sonatina_ir::Type::I1 {
                let three = lit_u32(func, 3);
                let byte_lane = emit_expr(func, target, naga::Expression::Binary {
                    op: naga::BinaryOperator::And,
                    left: addr,
                    right: three,
                });
                let eight = lit_u32(func, 8);
                let shift = emit_expr(func, target, naga::Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    left: byte_lane,
                    right: eight,
                });
                let shifted = emit_expr(func, target, naga::Expression::Binary {
                    op: naga::BinaryOperator::ShiftRight,
                    left: loaded_word,
                    right: shift,
                });
                let byte_mask = lit_u32(func, 0xff);
                let byte = emit_expr(func, target, naga::Expression::Binary {
                    op: naga::BinaryOperator::And,
                    left: shifted,
                    right: byte_mask,
                });
                if function.dfg.value_ty(result) == sonatina_ir::Type::I1 {
                    let zero = lit_u32(func, 0);
                    emit_expr(func, target, naga::Expression::Binary {
                        op: naga::BinaryOperator::NotEqual,
                        left: byte,
                        right: zero,
                    })
                } else {
                    // Narrow memory loads use the backend's I32 register
                    // carrier. A following Sonatina `trunc i1` recovers the
                    // logical boolean, matching Wasm's `i32.load8_u` shape.
                    byte
                }
            } else {
                loaded_word
            };
            value_map.insert(result, loaded);
            return true;
        }
    } else if let Some(store) = <&sonatina_ir::inst::data::Mstore as InstDowncast>::downcast(inst_set, inst_data) {
        let Some(addr) = resolve_naga_value(*store.addr(), function, word, value_map, phi_locals, func) else {
            *mem_error = Some(format!(
                "spirv: Mstore addr operand {:?} is unresolved (compiler invariant violation: \
                 the has_mem pre-scan already proved this value exists)",
                store.addr()
            ));
            return false;
        };
        let Some(value) = resolve_naga_value(*store.value(), function, word, value_map, phi_locals, func) else {
            *mem_error = Some(format!(
                "spirv: Mstore value operand {:?} is unresolved (compiler invariant violation: \
                 the has_mem pre-scan already proved this value exists)",
                store.value()
            ));
            return false;
        };
        if function.dfg.value_ty(*store.addr()).is_pointer(function.ctx()) {
            if typed_local_zero_store_is_redundant(function, inst_id, store) {
                return true;
            }
            target.push(
                naga::Statement::Store {
                    pointer: addr,
                    value,
                },
                naga::Span::UNDEFINED,
            );
            return true;
        }
        let mem_ctx = mem_ctx.expect("byte-arena Mstore requires mem_ctx");
        let elem = emit_mem_access(
            func,
            target,
            mem_ctx,
            addr,
            *store.ty() == sonatina_ir::Type::I32,
        );
        if *store.ty() == sonatina_ir::Type::I1 {
            let old_word = emit_expr(func, target, naga::Expression::Load { pointer: elem });
            let three = lit_u32(func, 3);
            let byte_lane = emit_expr(func, target, naga::Expression::Binary {
                op: naga::BinaryOperator::And,
                left: addr,
                right: three,
            });
            let eight = lit_u32(func, 8);
            let shift = emit_expr(func, target, naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: byte_lane,
                right: eight,
            });
            let byte_mask = lit_u32(func, 0xff);
            let shifted_mask = emit_expr(func, target, naga::Expression::Binary {
                op: naga::BinaryOperator::ShiftLeft,
                left: byte_mask,
                right: shift,
            });
            let inverse_mask = emit_expr(func, target, naga::Expression::Unary {
                op: naga::UnaryOperator::BitwiseNot,
                expr: shifted_mask,
            });
            let cleared_word = emit_expr(func, target, naga::Expression::Binary {
                op: naga::BinaryOperator::And,
                left: old_word,
                right: inverse_mask,
            });
            let one = lit_u32(func, 1);
            let zero = lit_u32(func, 0);
            let bool_word = emit_expr(func, target, naga::Expression::Select {
                condition: value,
                accept: one,
                reject: zero,
            });
            let shifted_value = emit_expr(func, target, naga::Expression::Binary {
                op: naga::BinaryOperator::ShiftLeft,
                left: bool_word,
                right: shift,
            });
            let updated_word = emit_expr(func, target, naga::Expression::Binary {
                op: naga::BinaryOperator::InclusiveOr,
                left: cleared_word,
                right: shifted_value,
            });
            target.push(
                naga::Statement::Store { pointer: elem, value: updated_word },
                naga::Span::UNDEFINED,
            );
        } else {
            target.push(naga::Statement::Store { pointer: elem, value }, naga::Span::UNDEFINED);
        }
        return true;
    } else if let Some(copy) =
        <&sonatina_ir::inst::data::Memcopy as InstDowncast>::downcast(inst_set, inst_data)
    {
        let mem_ctx = mem_ctx.expect("Memcopy requires mem_ctx (has_mem pre-scan gate)");
        let Some(dest) =
            resolve_naga_value(*copy.dest(), function, word, value_map, phi_locals, func)
        else {
            *mem_error = Some(format!(
                "spirv: Memcopy destination {:?} is unresolved",
                copy.dest()
            ));
            return false;
        };
        let Some(src) =
            resolve_naga_value(*copy.src(), function, word, value_map, phi_locals, func)
        else {
            *mem_error = Some(format!(
                "spirv: Memcopy source {:?} is unresolved",
                copy.src()
            ));
            return false;
        };
        let Some(len) =
            resolve_naga_value(*copy.len(), function, word, value_map, phi_locals, func)
        else {
            *mem_error = Some(format!(
                "spirv: Memcopy length {:?} is unresolved",
                copy.len()
            ));
            return false;
        };

        // WebAssembly memory.copy is memmove, not memcpy. Select a backwards
        // byte order exactly when the destination begins inside the source
        // range. A compact generated loop avoids multiplying shader size by
        // the aggregate byte length.
        let src_end = emit_expr(
            func,
            target,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: src,
                right: len,
            },
        );
        let dest_end = emit_expr(
            func,
            target,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: dest,
                right: len,
            },
        );
        let src_wrapped = emit_expr(
            func,
            target,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Less,
                left: src_end,
                right: src,
            },
        );
        let dest_wrapped = emit_expr(
            func,
            target,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Less,
                left: dest_end,
                right: dest,
            },
        );
        let wrapped = emit_expr(
            func,
            target,
            naga::Expression::Binary {
                op: naga::BinaryOperator::LogicalOr,
                left: src_wrapped,
                right: dest_wrapped,
            },
        );
        mark_trapped_if(func, target, mem_ctx, wrapped);
        let dest_after_src = emit_expr(
            func,
            target,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Greater,
                left: dest,
                right: src,
            },
        );
        let dest_before_src_end = emit_expr(
            func,
            target,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Less,
                left: dest,
                right: src_end,
            },
        );
        let backwards = emit_expr(
            func,
            target,
            naga::Expression::Binary {
                op: naga::BinaryOperator::LogicalAnd,
                left: dest_after_src,
                right: dest_before_src_end,
            },
        );

        let heap_ctx = mem_ctx.heap();
        let index_ty = heap_ctx.word_type;
        let zero = lit_u32(func, 0);
        let index_local = func.local_variables.append(
            naga::LocalVariable {
                name: Some(format!("fe_memcopy_index_{}", inst_id.0)),
                ty: index_ty,
                init: Some(zero),
            },
            naga::Span::UNDEFINED,
        );
        let mut loop_body = naga::Block::new();
        let index_ptr = func.expressions.append(
            naga::Expression::LocalVariable(index_local),
            naga::Span::UNDEFINED,
        );
        let index = emit_expr(
            func,
            &mut loop_body,
            naga::Expression::Load { pointer: index_ptr },
        );
        let more = emit_expr(
            func,
            &mut loop_body,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Less,
                left: index,
                right: len,
            },
        );
        let mut copy_byte = naga::Block::new();
        let one = lit_u32(func, 1);
        let last = emit_expr(
            func,
            &mut copy_byte,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Subtract,
                left: len,
                right: one,
            },
        );
        let reverse_offset = emit_expr(
            func,
            &mut copy_byte,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Subtract,
                left: last,
                right: index,
            },
        );
        let offset = emit_expr(
            func,
            &mut copy_byte,
            naga::Expression::Select {
                condition: backwards,
                accept: reverse_offset,
                reject: index,
            },
        );
        let source_addr = emit_expr(
            func,
            &mut copy_byte,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: src,
                right: offset,
            },
        );
        let destination_addr = emit_expr(
            func,
            &mut copy_byte,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: dest,
                right: offset,
            },
        );
        let byte = emit_heap_byte_load(func, &mut copy_byte, mem_ctx, source_addr);
        emit_heap_byte_store(func, &mut copy_byte, mem_ctx, destination_addr, byte);
        let next = emit_expr(
            func,
            &mut copy_byte,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: index,
                right: one,
            },
        );
        let index_ptr = func.expressions.append(
            naga::Expression::LocalVariable(index_local),
            naga::Span::UNDEFINED,
        );
        copy_byte.push(
            naga::Statement::Store {
                pointer: index_ptr,
                value: next,
            },
            naga::Span::UNDEFINED,
        );
        let mut done = naga::Block::new();
        done.push(naga::Statement::Break, naga::Span::UNDEFINED);
        loop_body.push(
            naga::Statement::If {
                condition: more,
                accept: copy_byte,
                reject: done,
            },
            naga::Span::UNDEFINED,
        );
        target.push(
            naga::Statement::Loop {
                body: loop_body,
                continuing: naga::Block::new(),
                break_if: None,
            },
            naga::Span::UNDEFINED,
        );
        return true;
    } else if let Some(call) = <&sonatina_ir::inst::control_flow::Call as InstDowncast>::downcast(inst_set, inst_data) {
        let Some(callee) = naga_functions.call(&inst_id) else {
            *mem_error = Some(format!(
                "spirv: call {inst_id:?} to reachable callee {:?} has no lowered Naga function variant. Fail closed.",
                call.callee(),
            ));
            return false;
        };
        if call.args().len() != callee.argument_abi.len() {
            *mem_error = Some(format!(
                "spirv: helper call to {:?} has {} logical arguments but {} ABI entries. Fail closed.",
                call.callee(),
                call.args().len(),
                callee.argument_abi.len(),
            ));
            return false;
        }
        let physical_argument_count = callee
            .argument_abi
            .iter()
            .filter_map(|source| match source {
                NagaArgumentSource::Physical(index)
                | NagaArgumentSource::Packed {
                    physical_index: index,
                    ..
                } => Some(*index as usize + 1),
                NagaArgumentSource::ImplicitResource(_) | NagaArgumentSource::Dead => None,
            })
            .max()
            .unwrap_or(0);
        let mut physical_arguments = vec![None; physical_argument_count];
        let mut packed_components = callee
            .packed_arguments
            .as_ref()
            .map(|packed| {
                packed
                    .groups
                    .iter()
                    .map(|group| vec![None; group.member_count])
                    .collect::<Vec<_>>()
            });
        let mut logical_arguments = Vec::with_capacity(call.args().len());
        for (&logical_argument, source) in call.args().iter().zip(&callee.argument_abi) {
            if matches!(source, NagaArgumentSource::Dead) {
                logical_arguments.push(None);
                continue;
            }
            let argument = match rematerialize_typed_pointer_projection(
                logical_argument, function, word, value_map, phi_locals, func, target,
            ) {
                Ok(Some(projected)) => projected,
                Ok(None) => {
                    let Some(argument) = resolve_naga_value(
                        logical_argument, function, word, value_map, phi_locals, func,
                    ) else {
                        *mem_error = Some(format!(
                            "spirv: call argument {logical_argument:?} could not be resolved. Fail closed."
                        ));
                        return false;
                    };
                    argument
                }
                Err(error) => {
                    *mem_error = Some(error);
                    return false;
                }
            };
            logical_arguments.push(Some(argument));
            match source {
                NagaArgumentSource::Physical(physical_index) => {
                    let Some(slot) = physical_arguments.get_mut(*physical_index as usize) else {
                        *mem_error = Some(format!(
                            "spirv: helper call to {:?} has out-of-range physical argument index {physical_index}. Fail closed.",
                            call.callee(),
                        ));
                        return false;
                    };
                    if slot.replace(argument).is_some() {
                        *mem_error = Some(format!(
                            "spirv: helper call to {:?} aliases physical argument index {physical_index}. Fail closed.",
                            call.callee(),
                        ));
                        return false;
                    }
                }
                NagaArgumentSource::Packed {
                    physical_index,
                    group_index,
                    member_index,
                } => {
                    let Some(packed) = callee.packed_arguments.as_ref() else {
                        *mem_error = Some(format!(
                            "spirv: helper call to {:?} has a packed source without a packed ABI. Fail closed.",
                            call.callee(),
                        ));
                        return false;
                    };
                    if packed.physical_index != *physical_index {
                        *mem_error = Some(format!(
                            "spirv: helper call to {:?} disagrees on packed physical argument index. Fail closed.",
                            call.callee(),
                        ));
                        return false;
                    }
                    let Some(slot) = packed_components
                        .as_mut()
                        .and_then(|groups| groups.get_mut(*group_index as usize))
                        .and_then(|components| components.get_mut(*member_index as usize))
                    else {
                        *mem_error = Some(format!(
                            "spirv: helper call to {:?} has out-of-range packed location {group_index}:{member_index}. Fail closed.",
                            call.callee(),
                        ));
                        return false;
                    };
                    if slot.replace(argument).is_some() {
                        *mem_error = Some(format!(
                            "spirv: helper call to {:?} aliases packed member index {member_index}. Fail closed.",
                            call.callee(),
                        ));
                        return false;
                    }
                }
                NagaArgumentSource::ImplicitResource(_) | NagaArgumentSource::Dead => {}
            }
        }
        if let Some(packed) = callee.packed_arguments.as_ref() {
            let Some(group_components) = packed_components.take() else {
                *mem_error = Some(format!(
                    "spirv: helper call to {:?} lost its packed argument components. Fail closed.",
                    call.callee(),
                ));
                return false;
            };
            let mut components = Vec::with_capacity(packed.groups.len());
            for (group, group_components) in packed.groups.iter().zip(group_components) {
                let Some(group_components) =
                    group_components.into_iter().collect::<Option<Vec<_>>>()
                else {
                    *mem_error = Some(format!(
                        "spirv: helper call to {:?} did not initialize every packed argument. Fail closed.",
                        call.callee(),
                    ));
                    return false;
                };
                let group_value = func.expressions.append(
                    naga::Expression::Compose {
                        ty: group.ty,
                        components: group_components,
                    },
                    naga::Span::UNDEFINED,
                );
                target.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(
                        group_value,
                        group_value,
                    )),
                    naga::Span::UNDEFINED,
                );
                components.push(group_value);
            }
            let composed = func.expressions.append(
                naga::Expression::Compose {
                    ty: packed.ty,
                    components,
                },
                naga::Span::UNDEFINED,
            );
            target.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(composed, composed)),
                naga::Span::UNDEFINED,
            );
            let Some(slot) = physical_arguments.get_mut(packed.physical_index as usize) else {
                *mem_error = Some(format!(
                    "spirv: helper call to {:?} lost its packed physical argument slot. Fail closed.",
                    call.callee(),
                ));
                return false;
            };
            if slot.replace(composed).is_some() {
                *mem_error = Some(format!(
                    "spirv: helper call to {:?} aliases its packed physical argument slot. Fail closed.",
                    call.callee(),
                ));
                return false;
            }
        }
        let Some(mut arguments) = physical_arguments.into_iter().collect::<Option<Vec<_>>>() else {
            *mem_error = Some(format!(
                "spirv: helper call to {:?} did not initialize every physical argument. Fail closed.",
                call.callee(),
            ));
            return false;
        };
        if callee.memory_abi.heap {
            let Some(heap) = mem_ctx.and_then(|context| context.heap) else {
                *mem_error = Some(format!(
                    "spirv: helper call to {:?} requires the caller's private arena, but no arena context is available. Fail closed.",
                    call.callee()
                ));
                return false;
            };
            arguments.push(heap.heap);
            arguments.push(heap.bump);
        }
        if callee.memory_abi.trap {
            let Some(context) = mem_ctx else {
                *mem_error = Some(format!(
                    "spirv: helper call to {:?} requires the caller's trap channel, but no trap context is available. Fail closed.",
                    call.callee()
                ));
                return false;
            };
            arguments.push(context.trapped);
        }
        let results = function.dfg.inst_results(inst_id);
        if results.len() != callee.result_abi.logical.len() {
            *mem_error = Some(format!(
                "spirv: helper call produces {} values but its lowered callee returns {}. Fail closed.",
                results.len(),
                callee.result_abi.logical.len(),
            ));
            return false;
        }
        let result = (callee.result_abi.physical_arity != 0).then(|| {
            let expression = func.expressions.append(
                naga::Expression::CallResult(callee.handle),
                naga::Span::UNDEFINED,
            );
            expression
        });
        target.push(
            naga::Statement::Call {
                function: callee.handle,
                arguments,
                result,
            },
            naga::Span::UNDEFINED,
        );
        let mut first_component = None;
        let mut last_component = None;
        for (&value, source) in results.iter().zip(&callee.result_abi.logical) {
            let expression = match *source {
                NagaResultSource::Physical(physical_index) => {
                    let Some(physical_result) = result else {
                        *mem_error = Some(
                            "spirv: helper physical result has no call-result expression. Fail closed."
                                .to_string(),
                        );
                        return false;
                    };
                    if callee.result_abi.physical_arity == 1 {
                        physical_result
                    } else {
                        let component = func.expressions.append(
                            naga::Expression::AccessIndex {
                                base: physical_result,
                                index: physical_index,
                            },
                            naga::Span::UNDEFINED,
                        );
                        first_component.get_or_insert(component);
                        last_component = Some(component);
                        component
                    }
                }
                NagaResultSource::PassthroughArgument(argument_index) => {
                    let Some(Some(argument)) = logical_arguments.get(argument_index as usize) else {
                        *mem_error = Some(format!(
                            "spirv: helper passthrough result refers to missing or dead argument {argument_index}. Fail closed."
                        ));
                        return false;
                    };
                    *argument
                }
            };
            value_map.insert(value, expression);
        }
        if let (Some(first), Some(last)) = (first_component, last_component) {
            target.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(first, last)),
                naga::Span::UNDEFINED,
            );
        }
        return true;
    } else if let Some(ret) = <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(inst_set, inst_data) {
        let logical_results = ret.args().as_slice();
        let physical_results = if let Some(return_abi) = return_abi {
            if logical_results.len() != return_abi.logical.len() {
                *mem_error = Some(format!(
                    "spirv: helper return has {} logical values but its lowered ABI has {}. Fail closed.",
                    logical_results.len(),
                    return_abi.logical.len(),
                ));
                return false;
            }
            let mut physical = Vec::with_capacity(return_abi.physical_arity as usize);
            for (&value, source) in logical_results.iter().zip(&return_abi.logical) {
                let NagaResultSource::Physical(index) = source else {
                    continue;
                };
                if *index as usize != physical.len() {
                    *mem_error = Some(format!(
                        "spirv: helper return has noncanonical physical result index {index}. Fail closed."
                    ));
                    return false;
                }
                physical.push(value);
            }
            physical
        } else {
            logical_results.first().copied().into_iter().collect()
        };
        let mut components = Vec::with_capacity(physical_results.len());
        for value in physical_results {
            let was_cached = value_map.contains_key(&value);
            let Some(component) = resolve_naga_value(
                value,
                function,
                word,
                value_map,
                phi_locals,
                func,
            ) else {
                *mem_error = Some(format!(
                    "spirv: helper return component {value:?} could not be resolved. Fail closed."
                ));
                return false;
            };
            if !was_cached && matches!(func.expressions[component], naga::Expression::Load { .. }) {
                target.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(component, component)),
                    naga::Span::UNDEFINED,
                );
            }
            components.push(component);
        }
        *result_expr = match components.as_slice() {
            [] => None,
            [component] => Some(*component),
            _ => {
                let Some(result_type) = return_abi.and_then(|abi| abi.physical_type) else {
                    *mem_error = Some(
                        "spirv: multi-value return has no lowered result type. Fail closed."
                            .to_string(),
                    );
                    return false;
                };
                let tuple = func.expressions.append(
                    naga::Expression::Compose {
                        ty: result_type,
                        components,
                    },
                    naga::Span::UNDEFINED,
                );
                target.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(tuple, tuple)),
                    naga::Span::UNDEFINED,
                );
                Some(tuple)
            }
        };
        return true;
    }
    false
}

/// Emit all non-control-flow instructions from a block into the target naga::Block.
#[cfg(feature = "spirv-backend")]
fn emit_block_to_target(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    block: sonatina_ir::BlockId,
    func: &mut naga::Function,
    target: &mut naga::Block,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    result_expr: &mut Option<naga::Handle<naga::Expression>>,
    return_abi: Option<&NagaResultAbi>,
    naga_functions: &NagaFunctionMap,
    mem_ctx: Option<MemCtx>,
    mem_error: &mut Option<String>,
) {
    for inst_id in function.layout.iter_inst(block) {
        emit_single_inst(
            inst_id,
            function,
            inst_set,
            word,
            func,
            target,
            value_map,
            phi_locals,
            result_expr,
            return_abi,
            naga_functions,
            mem_ctx,
            mem_error,
        );
        // Preserve the first lowering invariant failure. Continuing after a
        // failed instruction only produces secondary unresolved-value errors
        // for its consumers and used to overwrite the actionable root cause.
        if mem_error.is_some() {
            break;
        }
    }
}

#[cfg(feature = "spirv-backend")]
fn emit_naga_block_instructions(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    block: sonatina_ir::BlockId,
    _word_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    result_expr: &mut Option<naga::Handle<naga::Expression>>,
    return_abi: Option<&NagaResultAbi>,
    naga_functions: &NagaFunctionMap,
    mem_ctx: Option<MemCtx>,
    mem_error: &mut Option<String>,
) {
    let mut target = naga::Block::new();
    emit_phi_loads_for_block(function, inst_set, block, func, &mut target, value_map, phi_locals);
    emit_block_to_target(
        function,
        inst_set,
        word,
        block,
        func,
        &mut target,
        value_map,
        phi_locals,
        result_expr,
        return_abi,
        naga_functions,
        mem_ctx,
        mem_error,
    );
    func.body.extend_block(target);
}

#[cfg(feature = "spirv-backend")]
fn emit_phi_loads_for_block(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    block: sonatina_ir::BlockId,
    func: &mut naga::Function,
    target: &mut naga::Block,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
) {
    use sonatina_ir::InstDowncast;

    for inst_id in function.layout.iter_inst(block) {
        let inst = function.dfg.inst(inst_id);
        if <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst).is_none() {
            break;
        }
        let Some(result) = function.dfg.inst_result(inst_id) else { continue };
        let Some(&local) = phi_locals.get(&result) else { continue };
        let pointer = func.expressions.append(naga::Expression::LocalVariable(local), naga::Span::UNDEFINED);
        let loaded = func.expressions.append(naga::Expression::Load { pointer }, naga::Span::UNDEFINED);
        target.push(naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)), naga::Span::UNDEFINED);
        value_map.insert(result, loaded);
    }
}

#[cfg(feature = "spirv-backend")]
fn structured_regions_may_fall_through(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    regions: &[crate::structurize::Region],
) -> bool {
    use sonatina_ir::InstDowncast;

    let block_may_fall_through = |block| {
        !function.layout.iter_inst(block).any(|instruction| {
            let instruction = function.dfg.inst(instruction);
            <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(
                inst_set,
                instruction,
            )
            .is_some()
                || <&sonatina_ir::inst::control_flow::Unreachable as InstDowncast>::downcast(
                    inst_set,
                    instruction,
                )
                .is_some()
        })
    };

    for region in regions {
        let may_fall_through = match region {
            crate::structurize::Region::Block(block) => block_may_fall_through(*block),
            crate::structurize::Region::IfThenElse {
                then_branch,
                else_branch,
                ..
            } => {
                let then_falls_through = then_branch.is_empty()
                    || structured_regions_may_fall_through(function, inst_set, then_branch);
                let else_falls_through = else_branch.is_empty()
                    || structured_regions_may_fall_through(function, inst_set, else_branch);
                then_falls_through || else_falls_through
            }
            // A canonical structured loop has one header edge to its sibling
            // continuation. Return, trap, break, and continue paths inside the
            // body do not remove that possible normal exit.
            crate::structurize::Region::Loop { .. } => true,
            crate::structurize::Region::LoopExit { .. }
            | crate::structurize::Region::LoopContinue { .. } => false,
        };
        if !may_fall_through {
            return false;
        }
    }
    true
}

#[cfg(feature = "spirv-backend")]
fn emit_naga_regions(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    regions: &[crate::structurize::Region],
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    result_expr: &mut Option<naga::Handle<naga::Expression>>,
    return_abi: Option<&NagaResultAbi>,
    naga_functions: &NagaFunctionMap,
    mem_ctx: Option<MemCtx>,
) -> Result<(), String> {
    let mut mem_error = None;
    let mut region_idx = 0;
    while region_idx < regions.len() {
        let region = &regions[region_idx];
        match region {
            crate::structurize::Region::Block(block_id) => {
                emit_naga_block_instructions(
                    function, inst_set, word, *block_id, word_type,
                    func, value_map, phi_locals, result_expr, return_abi, naga_functions, mem_ctx,
                    &mut mem_error,
                );
                if let Some(msg) = mem_error.take() {
                    return Err(msg);
                }
                // Review finding 4 (wrong value on unconditional trap): a block
                // reached unconditionally at THIS (unnested) level whose
                // terminator is `Unreachable` is, by construction of
                // `structurize_function` (Unreachable is classified
                // Return-like, always ending its region chain), the LAST
                // region at this nesting level -- nothing here needs to be
                // skipped afterward. Before this arm existed, `emit_single_inst`
                // had no case for `Unreachable` at all, so a top-level trap
                // silently produced no statements and the function fell
                // through to the store-zero fallback (`0x0000_0000`), never
                // reaching the poison contract the design specifies.
                if let Some(mem_ctx) = mem_ctx {
                    if block_ends_unreachable(*block_id, function, inst_set) {
                        let mut target = naga::Block::new();
                        mark_trapped_always(func, &mut target, mem_ctx);
                        func.body.extend_block(target);
                    }
                }
                region_idx += 1;
            }
            crate::structurize::Region::Loop { header, body } => {
                let mut loop_target = naga::Block::new();
                let loop_return = emit_recursive_loop_region(
                    function, inst_set, word, *header, body, word_type, f32_type, bool_type,
                    func, value_map, phi_locals, result_expr, &mut loop_target,
                    return_abi, naga_functions, mem_ctx,
                )?;
                func.body.extend_block(loop_target);
                region_idx += 1;
                if let Some(loop_return) = loop_return {
                    let saved_body = std::mem::replace(&mut func.body, naga::Block::new());
                    let mut continuation_result = None;
                    emit_naga_regions(
                        function, inst_set, word, &regions[region_idx..], word_type, f32_type,
                        bool_type, func, value_map, phi_locals, &mut continuation_result,
                        return_abi, naga_functions, mem_ctx,
                    )?;
                    let mut continuation = std::mem::replace(&mut func.body, saved_body);
                    if let Some(value) = continuation_result {
                        if !loop_return.has_result {
                            return Err(
                                "spirv: unit structured return continuation produced a result"
                                    .to_string(),
                            );
                        }
                        let pointer = func.expressions.append(
                            naga::Expression::LocalVariable(loop_return.result),
                            naga::Span::UNDEFINED,
                        );
                        continuation.push(
                            naga::Statement::Store { pointer, value },
                            naga::Span::UNDEFINED,
                        );
                    } else if loop_return.has_result
                        && !continuation.is_empty()
                        && structured_regions_may_fall_through(
                            function,
                            inst_set,
                            &regions[region_idx..],
                        )
                    {
                        return Err(
                            "spirv: value structured return continuation has no result"
                                .to_string(),
                        );
                    }
                    if !continuation.is_empty() {
                        let returned_pointer = func.expressions.append(
                            naga::Expression::LocalVariable(loop_return.did_return),
                            naga::Span::UNDEFINED,
                        );
                        let returned = func.expressions.append(
                            naga::Expression::Load { pointer: returned_pointer },
                            naga::Span::UNDEFINED,
                        );
                        func.body.push(
                            naga::Statement::Emit(naga::Range::new_from_bounds(returned, returned)),
                            naga::Span::UNDEFINED,
                        );
                        func.body.push(
                            naga::Statement::If {
                                condition: returned,
                                accept: naga::Block::new(),
                                reject: continuation,
                            },
                            naga::Span::UNDEFINED,
                        );
                    }
                    if loop_return.has_result {
                        let result_pointer = func.expressions.append(
                            naga::Expression::LocalVariable(loop_return.result),
                            naga::Span::UNDEFINED,
                        );
                        let loaded = func.expressions.append(
                            naga::Expression::Load { pointer: result_pointer },
                            naga::Span::UNDEFINED,
                        );
                        func.body.push(
                            naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)),
                            naga::Span::UNDEFINED,
                        );
                        *result_expr = Some(loaded);
                    }
                    return Ok(());
                }
            }
            crate::structurize::Region::IfThenElse { .. } => {
                let mut target = naga::Block::new();
                let transport = allocate_return_transport(
                    function, inst_set, word_type, f32_type, bool_type, func, return_abi,
                )?;
                let mut may_return = false;
                emit_if_region(
                    function, inst_set, word, region, word_type, f32_type, bool_type, func, &mut target,
                    value_map, phi_locals, transport, &mut may_return, return_abi,
                    naga_functions, mem_ctx,
                )?;
                func.body.extend_block(target);
                region_idx += 1;
                if may_return {
                    let saved_body = std::mem::replace(&mut func.body, naga::Block::new());
                    let mut continuation_result = None;
                    emit_naga_regions(
                        function, inst_set, word, &regions[region_idx..], word_type, f32_type,
                        bool_type, func, value_map, phi_locals, &mut continuation_result,
                        return_abi, naga_functions, mem_ctx,
                    )?;
                    let mut continuation = std::mem::replace(&mut func.body, saved_body);
                    if let Some(value) = continuation_result {
                        if !transport.has_result {
                            return Err(
                                "spirv: unit structured return continuation produced a result"
                                    .to_string(),
                            );
                        }
                        let pointer = func.expressions.append(
                            naga::Expression::LocalVariable(transport.result), naga::Span::UNDEFINED,
                        );
                        continuation.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
                    } else if transport.has_result
                        && !continuation.is_empty()
                        && structured_regions_may_fall_through(
                            function,
                            inst_set,
                            &regions[region_idx..],
                        )
                    {
                        return Err(
                            "spirv: value structured return continuation has no result"
                                .to_string(),
                        );
                    }
                    if !continuation.is_empty() {
                        let pointer = func.expressions.append(
                            naga::Expression::LocalVariable(transport.did_return), naga::Span::UNDEFINED,
                        );
                        let returned = func.expressions.append(
                            naga::Expression::Load { pointer }, naga::Span::UNDEFINED,
                        );
                        func.body.push(
                            naga::Statement::Emit(naga::Range::new_from_bounds(returned, returned)),
                            naga::Span::UNDEFINED,
                        );
                        func.body.push(
                            naga::Statement::If {
                                condition: returned,
                                accept: naga::Block::new(),
                                reject: continuation,
                            },
                            naga::Span::UNDEFINED,
                        );
                    }
                    if transport.has_result {
                        let pointer = func.expressions.append(
                            naga::Expression::LocalVariable(transport.result), naga::Span::UNDEFINED,
                        );
                        let loaded = func.expressions.append(
                            naga::Expression::Load { pointer }, naga::Span::UNDEFINED,
                        );
                        func.body.push(
                            naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)),
                            naga::Span::UNDEFINED,
                        );
                        *result_expr = Some(loaded);
                    }
                    return Ok(());
                }
            }
            crate::structurize::Region::LoopExit { from, target } => return Err(format!(
                "spirv: loop exit edge {from:?}->{target:?} appeared outside its loop"
            )),
            crate::structurize::Region::LoopContinue { from, target } => return Err(format!(
                "spirv: loop continue edge {from:?}->{target:?} appeared outside its loop"
            )),
        }
    }
    Ok(())
}

#[cfg(feature = "spirv-backend")]
fn ensure_phi_locals(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    block: sonatina_ir::BlockId,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
    value_map: &std::collections::HashMap<
        sonatina_ir::ValueId,
        naga::Handle<naga::Expression>,
    >,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
) -> Result<(), String> {
    use sonatina_ir::InstDowncast;
    for inst_id in function.layout.iter_inst(block) {
        let inst = function.dfg.inst(inst_id);
        if <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst).is_none() { break }
        let Some(result) = function.dfg.inst_result(inst_id) else { continue };
        if value_map.contains_key(&result) {
            continue;
        }
        let ty = match function.dfg.value_ty(result) {
            sonatina_ir::Type::F32 => f32_type,
            sonatina_ir::Type::I1 => bool_type,
            sonatina_ir::Type::I32 if word == WordKind::U32 => word_type,
            sonatina_ir::Type::I64 if word == WordKind::I64 => word_type,
            other => {
                return Err(format!(
                    "spirv structurize: phi {result:?} in {block:?} has non-scalar type {other:?} without one proven resource identity. Fail closed."
                ));
            }
        };
        // Control transport is compiler-internal. Keep its physical WGSL name
        // compact while Sonatina value IDs remain available in diagnostics.
        phi_locals.entry(result).or_insert_with(|| {
            let zero = func.expressions.append(
                naga::Expression::ZeroValue(ty),
                naga::Span::UNDEFINED,
            );
            func.local_variables.append(
                naga::LocalVariable { name: None, ty, init: Some(zero) },
                naga::Span::UNDEFINED,
            )
        });
    }
    Ok(())
}

#[cfg(feature = "spirv-backend")]
fn region_blocks(regions: &[crate::structurize::Region], out: &mut std::collections::HashSet<sonatina_ir::BlockId>) {
    for region in regions {
        match region {
            crate::structurize::Region::Block(block) => { out.insert(*block); }
            crate::structurize::Region::IfThenElse { header, then_branch, else_branch, .. } => {
                out.insert(*header);
                region_blocks(then_branch, out);
                region_blocks(else_branch, out);
            }
            crate::structurize::Region::Loop { header, body } => {
                out.insert(*header);
                region_blocks(body, out);
            }
            crate::structurize::Region::LoopExit { .. }
            | crate::structurize::Region::LoopContinue { .. } => {}
        }
    }
}

#[cfg(feature = "spirv-backend")]
fn emit_exact_phi_edge(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    from: sonatina_ir::BlockId,
    to: sonatina_ir::BlockId,
    func: &mut naga::Function,
    target: &mut naga::Block,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
) -> Result<(), String> {
    use sonatina_ir::InstDowncast;
    let acyclic_merge = !block_is_in_cfg_cycle(function, to);
    let mut transfers = Vec::new();
    for inst_id in function.layout.iter_inst(to) {
        let inst = function.dfg.inst(inst_id);
        let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst) else { break };
        let result = function.dfg.inst_result(inst_id).ok_or_else(|| "spirv structurize: phi has no result".to_string())?;
        let Some(&local) = phi_locals.get(&result) else {
            if value_map.contains_key(&result) {
                continue;
            }
            return Err(format!("spirv structurize: phi {result:?} has no local"));
        };
        let [(value, _)] = phi.args().iter().filter(|(_, pred)| *pred == from).collect::<Vec<_>>().as_slice() else {
            return Err(format!("spirv structurize: edge {from:?}->{to:?} does not have exactly one input for phi {result:?}"));
        };
        // A nested merge phi can feed an outer merge edge without appearing
        // in this arm's expression map. Materialize its local load in the
        // current edge block. Returning a bare Load expression here leaves it
        // outside Naga scope when the following Store consumes it.
        let source = *value;
        let value = if let Some(&value) = value_map.get(value) {
            value
        } else if let Some(&source_local) = phi_locals.get(value) {
            let pointer = func.expressions.append(
                naga::Expression::LocalVariable(source_local),
                naga::Span::UNDEFINED,
            );
            let loaded = func.expressions.append(
                naga::Expression::Load { pointer },
                naga::Span::UNDEFINED,
            );
            target.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)),
                naga::Span::UNDEFINED,
            );
            loaded
        } else {
            resolve_naga_value(*value, function, word, value_map, phi_locals, func).ok_or_else(
                || {
                    format!(
                        "spirv structurize: unresolved phi input on edge {from:?}->{to:?}"
                    )
                },
            )?
        };
        let already_initialized = acyclic_merge
            && function.dfg.value_imm(source).is_some_and(immediate_is_shader_zero);
        transfers.push((local, value, already_initialized));
    }
    // Every non-literal source above is already an emitted Naga SSA
    // expression in this edge's lexical scope. Store those values directly:
    // they retain parallel phi semantics even when destinations form a cycle,
    // while one temporary local per edge and phi merely re-materializes the
    // same already-snapshotted values.
    for (local, value, already_initialized) in transfers {
        if already_initialized {
            continue;
        }
        let pointer = func.expressions.append(naga::Expression::LocalVariable(local), naga::Span::UNDEFINED);
        target.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
    }
    Ok(())
}

#[cfg(feature = "spirv-backend")]
fn block_is_in_cfg_cycle(
    function: &sonatina_ir::Function,
    block: sonatina_ir::BlockId,
) -> bool {
    let mut cfg = sonatina_ir::ControlFlowGraph::default();
    cfg.compute(function);
    let mut seen = std::collections::HashSet::new();
    let mut pending = cfg.succs_of(block).copied().collect::<Vec<_>>();
    while let Some(candidate) = pending.pop() {
        if candidate == block {
            return true;
        }
        if seen.insert(candidate) {
            pending.extend(cfg.succs_of(candidate).copied());
        }
    }
    false
}

#[cfg(feature = "spirv-backend")]
fn emit_if_region(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    region: &crate::structurize::Region,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
    target: &mut naga::Block,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    transport: StructuredReturnTransport,
    may_return: &mut bool,
    return_abi: Option<&NagaResultAbi>,
    naga_functions: &NagaFunctionMap,
    mem_ctx: Option<MemCtx>,
) -> Result<(), String> {
    use sonatina_ir::InstDowncast;
    let crate::structurize::Region::IfThenElse { header, then_branch, else_branch, merge } = region else {
        return Err("spirv structurize: expected if region".to_string());
    };
    if let Some(merge) = merge {
        ensure_phi_locals(
            function, inst_set, word, *merge, word_type, f32_type, bool_type, func, value_map,
            phi_locals,
        )?;
    }
    emit_phi_loads_for_block(
        function, inst_set, *header, func, target, value_map, phi_locals,
    );
    let mut ignored_result = None;
    let mut mem_error = None;
    emit_block_to_target(function, inst_set, word, *header, func, target, value_map, phi_locals, &mut ignored_result, return_abi, naga_functions, mem_ctx, &mut mem_error);
    if let Some(msg) = mem_error.take() {
        return Err(msg);
    }
    let branch = function.layout.iter_inst(*header).find_map(|inst_id|
        <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, function.dfg.inst(inst_id))
    ).ok_or_else(|| format!("spirv structurize: if header {header:?} has no branch"))?;
    let condition = resolve_naga_value(*branch.cond(), function, word, value_map, phi_locals, func)
        .ok_or_else(|| format!("spirv structurize: unresolved condition in {header:?}"))?;

    let mut accept = naga::Block::new();
    let mut reject = naga::Block::new();
    // Each arm starts from the header scope. Values defined by one arm must not
    // leak into resolution of its sibling; only explicit merge-phi transfers
    // carry values out of the branch.
    let mut then_values = value_map.clone();
    let mut else_values = value_map.clone();
    let mut then_returns = false;
    let mut else_returns = false;
    let mut then_edge_emitted = false;
    let mut else_edge_emitted = false;
    emit_non_loop_regions(function, inst_set, word, then_branch, word_type, f32_type, bool_type, func, &mut accept, &mut then_values, phi_locals, transport, *merge, &mut then_returns, &mut then_edge_emitted, return_abi, naga_functions, mem_ctx)?;
    emit_non_loop_regions(function, inst_set, word, else_branch, word_type, f32_type, bool_type, func, &mut reject, &mut else_values, phi_locals, transport, *merge, &mut else_returns, &mut else_edge_emitted, return_abi, naga_functions, mem_ctx)?;
    *may_return |= then_returns || else_returns;
    let then_outcome = arm_outcome(function, inst_set, *header, then_branch);
    let else_outcome = arm_outcome(function, inst_set, *header, else_branch);
    if let Some(merge) = merge {
        if let ArmOutcome::Predecessor(from) = then_outcome && !then_edge_emitted {
            let mut edge = naga::Block::new();
            emit_exact_phi_edge(function, inst_set, word, from, *merge, func, &mut edge, &mut then_values, phi_locals)?;
            if then_returns {
                let pointer = func.expressions.append(
                    naga::Expression::LocalVariable(transport.did_return), naga::Span::UNDEFINED,
                );
                let returned = func.expressions.append(
                    naga::Expression::Load { pointer }, naga::Span::UNDEFINED,
                );
                accept.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(returned, returned)),
                    naga::Span::UNDEFINED,
                );
                accept.push(
                    naga::Statement::If {
                        condition: returned,
                        accept: naga::Block::new(),
                        reject: edge,
                    },
                    naga::Span::UNDEFINED,
                );
            } else {
                accept.extend_block(edge);
            }
        }
        if let ArmOutcome::Predecessor(from) = else_outcome && !else_edge_emitted {
            let mut edge = naga::Block::new();
            emit_exact_phi_edge(function, inst_set, word, from, *merge, func, &mut edge, &mut else_values, phi_locals)?;
            if else_returns {
                let pointer = func.expressions.append(
                    naga::Expression::LocalVariable(transport.did_return), naga::Span::UNDEFINED,
                );
                let returned = func.expressions.append(
                    naga::Expression::Load { pointer }, naga::Span::UNDEFINED,
                );
                reject.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(returned, returned)),
                    naga::Span::UNDEFINED,
                );
                reject.push(
                    naga::Statement::If {
                        condition: returned,
                        accept: naga::Block::new(),
                        reject: edge,
                    },
                    naga::Span::UNDEFINED,
                );
            } else {
                reject.extend_block(edge);
            }
        }
    } else if !matches!(then_outcome, ArmOutcome::Terminal)
        || !matches!(else_outcome, ArmOutcome::Terminal)
    {
        return Err(format!(
            "spirv structurize: merge-less conditional {header:?} has a fallthrough arm"
        ));
    }
    target.push(naga::Statement::If { condition, accept, reject }, naga::Span::UNDEFINED);
    Ok(())
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy)]
enum ArmOutcome {
    Predecessor(sonatina_ir::BlockId),
    AlreadyAtMerge,
    Terminal,
}

#[cfg(feature = "spirv-backend")]
fn arm_outcome(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    header: sonatina_ir::BlockId,
    regions: &[crate::structurize::Region],
) -> ArmOutcome {
    use sonatina_ir::InstDowncast;
    let Some(last) = regions.last() else { return ArmOutcome::Predecessor(header) };
    match last {
        crate::structurize::Region::Block(block) => {
            // A block ending in `Return` OR `Unreachable` is Terminal: neither
            // has a real CFG successor, so this arm never becomes a
            // predecessor of some later merge. Treating only `Return` as
            // Terminal (the pre-array-support state) made a trap arm report
            // `Predecessor`, which trips the "merge-less conditional has a
            // fallthrough arm" hard error for EVERY bounds-checked array
            // access, since `find_merge`'s postdominator walk already
            // (correctly) computes `merge: None` whenever one arm is a true
            // CFG dead end (review finding 4's structural precondition).
            let returns = function.layout.iter_inst(*block).any(|iid| {
                let inst = function.dfg.inst(iid);
                <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(inst_set, inst).is_some()
                    || <&sonatina_ir::inst::control_flow::Unreachable as InstDowncast>::downcast(inst_set, inst).is_some()
            });
            if returns { ArmOutcome::Terminal } else { ArmOutcome::Predecessor(*block) }
        }
        crate::structurize::Region::IfThenElse { merge: Some(_), .. } => ArmOutcome::AlreadyAtMerge,
        crate::structurize::Region::IfThenElse { merge: None, .. }
        | crate::structurize::Region::Loop { .. }
        | crate::structurize::Region::LoopExit { .. }
        | crate::structurize::Region::LoopContinue { .. } => ArmOutcome::Terminal,
    }
}

#[cfg(feature = "spirv-backend")]
fn emit_non_loop_regions(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    regions: &[crate::structurize::Region],
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
    target: &mut naga::Block,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    transport: StructuredReturnTransport,
    fallthrough_merge: Option<sonatina_ir::BlockId>,
    may_return: &mut bool,
    edge_emitted: &mut bool,
    return_abi: Option<&NagaResultAbi>,
    naga_functions: &NagaFunctionMap,
    mem_ctx: Option<MemCtx>,
) -> Result<(), String> {
    let mut region_idx = 0;
    while region_idx < regions.len() {
        let region = &regions[region_idx];
        match region {
            crate::structurize::Region::Block(block) => {
                emit_phi_loads_for_block(function, inst_set, *block, func, target, value_map, phi_locals);
                let mut block_result = None;
                let mut mem_error = None;
                emit_block_to_target(function, inst_set, word, *block, func, target, value_map, phi_locals, &mut block_result, return_abi, naga_functions, mem_ctx, &mut mem_error);
                if let Some(msg) = mem_error.take() {
                    return Err(msg);
                }
                if block_has_return(*block, function, inst_set) {
                    if transport.has_result {
                        let value = block_result.ok_or_else(|| format!("spirv: unresolved structured return in {block:?}"))?;
                        let pointer = func.expressions.append(
                            naga::Expression::LocalVariable(transport.result), naga::Span::UNDEFINED,
                        );
                        target.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
                    }
                    let returned = func.expressions.append(
                        naga::Expression::Literal(naga::Literal::Bool(true)), naga::Span::UNDEFINED,
                    );
                    let pointer = func.expressions.append(
                        naga::Expression::LocalVariable(transport.did_return), naga::Span::UNDEFINED,
                    );
                    target.push(naga::Statement::Store { pointer, value: returned }, naga::Span::UNDEFINED);
                    *may_return = true;
                } else if let Some(mem_ctx) = mem_ctx {
                    // Review finding 4 analog, nested (non-loop) arm: a bounds-trap
                    // arm (`if idx<len {...} else { unreachable }`, `merge:
                    // None` per `find_merge`'s postdominator walk) is, by
                    // `structurize_function`'s Unreachable-is-Return-like
                    // classification, ALWAYS the last region at this nesting
                    // level -- nothing here needs `transport`/`may_return`
                    // (no sibling region depends on it); only the externally
                    // visible trapped flag needs raising.
                    if block_ends_unreachable(*block, function, inst_set) {
                        mark_trapped_always(func, target, mem_ctx);
                    }
                }
                region_idx += 1;
            }
            crate::structurize::Region::IfThenElse { header, merge: nested_merge, .. } => {
                let mut nested_returns = false;
                emit_if_region(
                    function, inst_set, word, region, word_type, f32_type, bool_type,
                    func, target, value_map, phi_locals, transport, &mut nested_returns,
                    return_abi, naga_functions, mem_ctx,
                )?;
                region_idx += 1;
                if nested_returns {
                    let mut continuation = naga::Block::new();
                    let mut continuation_returns = false;
                    let mut continuation_edge_emitted = false;
                    emit_non_loop_regions(
                        function, inst_set, word, &regions[region_idx..], word_type,
                        f32_type, bool_type, func, &mut continuation, value_map,
                        phi_locals, transport, fallthrough_merge,
                        &mut continuation_returns, &mut continuation_edge_emitted,
                        return_abi, naga_functions, mem_ctx,
                    )?;
                    if regions[region_idx..].is_empty()
                        && *nested_merge == fallthrough_merge
                    {
                        *edge_emitted = true;
                    } else if !continuation_edge_emitted
                        && let Some(merge) = fallthrough_merge
                        && let ArmOutcome::Predecessor(from) =
                            arm_outcome(function, inst_set, *header, &regions[region_idx..])
                    {
                        emit_exact_phi_edge(
                            function, inst_set, word, from, merge, func,
                            &mut continuation, value_map, phi_locals,
                        )?;
                        *edge_emitted = true;
                    } else {
                        *edge_emitted |= continuation_edge_emitted;
                    }
                    let pointer = func.expressions.append(
                        naga::Expression::LocalVariable(transport.did_return),
                        naga::Span::UNDEFINED,
                    );
                    let returned = func.expressions.append(
                        naga::Expression::Load { pointer }, naga::Span::UNDEFINED,
                    );
                    target.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(returned, returned)),
                        naga::Span::UNDEFINED,
                    );
                    target.push(
                        naga::Statement::If {
                            condition: returned,
                            accept: naga::Block::new(),
                            reject: continuation,
                        },
                        naga::Span::UNDEFINED,
                    );
                    *may_return = true;
                    return Ok(());
                }
            }
            crate::structurize::Region::Loop { header, body } => {
                let mut no_result = None;
                let loop_return = emit_recursive_loop_region(
                    function, inst_set, word, *header, body, word_type, f32_type, bool_type,
                    func, value_map, phi_locals, &mut no_result, target, return_abi,
                    naga_functions, mem_ctx,
                )?;
                region_idx += 1;
                if let Some(loop_return) = loop_return {
                    let loop_exit = structured_loop_exit(function, inst_set, *header, body)?;
                    let remaining = &regions[region_idx..];
                    let mut continuation = naga::Block::new();
                    let mut _continuation_returns = false;
                    let mut continuation_edge_emitted = false;
                    emit_non_loop_regions(
                        function,
                        inst_set,
                        word,
                        remaining,
                        word_type,
                        f32_type,
                        bool_type,
                        func,
                        &mut continuation,
                        value_map,
                        phi_locals,
                        transport,
                        fallthrough_merge,
                        &mut _continuation_returns,
                        &mut continuation_edge_emitted,
                        return_abi,
                        naga_functions,
                        mem_ctx,
                    )?;
                    if remaining.is_empty() && Some(loop_exit) == fallthrough_merge {
                        // The loop emitter materialized the exact header-to-exit
                        // phi edge in its normal exit arm.
                        *edge_emitted = true;
                    } else if !continuation_edge_emitted
                        && let Some(merge) = fallthrough_merge
                    {
                        let normal_outcome = if remaining.is_empty() {
                            ArmOutcome::Predecessor(loop_exit)
                        } else {
                            arm_outcome(function, inst_set, loop_exit, remaining)
                        };
                        if let ArmOutcome::Predecessor(from) = normal_outcome {
                            emit_exact_phi_edge(
                                function,
                                inst_set,
                                word,
                                from,
                                merge,
                                func,
                                &mut continuation,
                                value_map,
                                phi_locals,
                            )?;
                            *edge_emitted = true;
                        }
                    } else {
                        *edge_emitted |= continuation_edge_emitted;
                    }

                    let returned = load_structured_return_flag(loop_return, func, target);
                    let mut accept = naga::Block::new();
                    forward_structured_return(loop_return, transport, func, &mut accept)?;
                    target.push(
                        naga::Statement::If {
                            condition: returned,
                            accept,
                            reject: continuation,
                        },
                        naga::Span::UNDEFINED,
                    );
                    *may_return = true;
                    return Ok(());
                }
            }
            crate::structurize::Region::LoopExit { from, target } => return Err(format!(
                "spirv: loop exit edge {from:?}->{target:?} appeared outside its loop"
            )),
            crate::structurize::Region::LoopContinue { from, target } => return Err(format!(
                "spirv: loop continue edge {from:?}->{target:?} appeared outside its loop"
            )),
        }
    }
    Ok(())
}

#[cfg(feature = "spirv-backend")]
enum RegionOutcome {
    Fallthrough(sonatina_ir::BlockId),
    Terminal,
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy)]
struct StructuredReturnTransport {
    result: naga::Handle<naga::LocalVariable>,
    did_return: naga::Handle<naga::LocalVariable>,
    has_result: bool,
}

#[cfg(feature = "spirv-backend")]
fn load_structured_return_flag(
    source: StructuredReturnTransport,
    func: &mut naga::Function,
    target: &mut naga::Block,
) -> naga::Handle<naga::Expression> {
    let pointer = func.expressions.append(
        naga::Expression::LocalVariable(source.did_return),
        naga::Span::UNDEFINED,
    );
    let returned = func.expressions.append(
        naga::Expression::Load { pointer },
        naga::Span::UNDEFINED,
    );
    target.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(returned, returned)),
        naga::Span::UNDEFINED,
    );
    returned
}

#[cfg(feature = "spirv-backend")]
fn forward_structured_return(
    source: StructuredReturnTransport,
    destination: StructuredReturnTransport,
    func: &mut naga::Function,
    target: &mut naga::Block,
) -> Result<(), String> {
    if source.has_result != destination.has_result {
        return Err(format!(
            "spirv structurize: incompatible nested return transports, source has_result={}, destination has_result={}",
            source.has_result, destination.has_result,
        ));
    }
    if source.has_result {
        let source_pointer = func.expressions.append(
            naga::Expression::LocalVariable(source.result),
            naga::Span::UNDEFINED,
        );
        let value = func.expressions.append(
            naga::Expression::Load {
                pointer: source_pointer,
            },
            naga::Span::UNDEFINED,
        );
        target.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(value, value)),
            naga::Span::UNDEFINED,
        );
        let destination_pointer = func.expressions.append(
            naga::Expression::LocalVariable(destination.result),
            naga::Span::UNDEFINED,
        );
        target.push(
            naga::Statement::Store {
                pointer: destination_pointer,
                value,
            },
            naga::Span::UNDEFINED,
        );
    }
    let returned = func.expressions.append(
        naga::Expression::Literal(naga::Literal::Bool(true)),
        naga::Span::UNDEFINED,
    );
    let destination_pointer = func.expressions.append(
        naga::Expression::LocalVariable(destination.did_return),
        naga::Span::UNDEFINED,
    );
    target.push(
        naga::Statement::Store {
            pointer: destination_pointer,
            value: returned,
        },
        naga::Span::UNDEFINED,
    );
    Ok(())
}

#[cfg(feature = "spirv-backend")]
fn structured_loop_exit(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    header: sonatina_ir::BlockId,
    body_regions: &[crate::structurize::Region],
) -> Result<sonatina_ir::BlockId, String> {
    use sonatina_ir::InstDowncast;

    let mut loop_blocks = std::collections::HashSet::new();
    loop_blocks.insert(header);
    region_blocks(body_regions, &mut loop_blocks);
    let branch = function
        .layout
        .iter_inst(header)
        .find_map(|iid| {
            <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(
                inst_set,
                function.dfg.inst(iid),
            )
        })
        .ok_or_else(|| format!("spirv: loop header {header:?} has no branch"))?;
    let nz_in = loop_blocks.contains(branch.nz_dest());
    let z_in = loop_blocks.contains(branch.z_dest());
    if nz_in == z_in {
        return Err(format!(
            "spirv: loop {header:?} must have exactly one in-loop successor"
        ));
    }
    Ok(if nz_in {
        *branch.z_dest()
    } else {
        *branch.nz_dest()
    })
}

#[cfg(feature = "spirv-backend")]
fn allocate_return_transport(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
    return_abi: Option<&NagaResultAbi>,
) -> Result<StructuredReturnTransport, String> {
    let source_return_ty = function.layout.iter_block().find_map(|block| {
        find_block_return_value(block, function, inst_set)
            .map(|value| function.dfg.value_ty(value))
    });
    let (return_naga_ty, has_result) = match return_abi {
        Some(return_abi) => (
            return_abi.physical_type.unwrap_or(word_type),
            return_abi.physical_arity != 0,
        ),
        None => (
            match source_return_ty {
                Some(sonatina_ir::Type::F32) => f32_type,
                Some(sonatina_ir::Type::I1) => bool_type,
                Some(_) | None => word_type,
            },
            source_return_ty.is_some(),
        ),
    };
    // Structured return transport is likewise a physical lowering detail.
    let result = func.local_variables.append(
        naga::LocalVariable { name: None, ty: return_naga_ty, init: None },
        naga::Span::UNDEFINED,
    );
    let returned_false = func.expressions.append(
        naga::Expression::Literal(naga::Literal::Bool(false)),
        naga::Span::UNDEFINED,
    );
    let did_return = func.local_variables.append(
        naga::LocalVariable {
            name: None,
            ty: bool_type,
            init: Some(returned_false),
        },
        naga::Span::UNDEFINED,
    );
    Ok(StructuredReturnTransport {
        result,
        did_return,
        has_result,
    })
}

#[cfg(feature = "spirv-backend")]
fn emit_regions_in_loop(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    regions: &[crate::structurize::Region],
    loop_header: sonatina_ir::BlockId,
    loop_exit: sonatina_ir::BlockId,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    return_local: naga::Handle<naga::LocalVariable>,
    did_return_local: naga::Handle<naga::LocalVariable>,
    may_return: &mut bool,
    func: &mut naga::Function,
    target: &mut naga::Block,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    result_expr: &mut Option<naga::Handle<naga::Expression>>,
    return_abi: Option<&NagaResultAbi>,
    naga_functions: &NagaFunctionMap,
    mem_ctx: Option<MemCtx>,
) -> Result<RegionOutcome, String> {
    use sonatina_ir::InstDowncast;
    let mut outcome = None;
    for (region_index, region) in regions.iter().enumerate() {
        match region {
            crate::structurize::Region::Block(block) => {
                emit_phi_loads_for_block(function, inst_set, *block, func, target, value_map, phi_locals);
                let mut mem_error = None;
                emit_block_to_target(function, inst_set, word, *block, func, target, value_map, phi_locals, result_expr, return_abi, naga_functions, mem_ctx, &mut mem_error);
                if let Some(msg) = mem_error.take() {
                    return Err(msg);
                }
                let mut terminated = false;
                for inst_id in function.layout.iter_inst(*block) {
                    let inst = function.dfg.inst(inst_id);
                    if let Some(jump) = <&sonatina_ir::inst::control_flow::Jump as InstDowncast>::downcast(inst_set, inst) {
                        if *jump.dest() == loop_header {
                            emit_exact_phi_edge(function, inst_set, word, *block, loop_header, func, target, value_map, phi_locals)?;
                            target.push(naga::Statement::Continue, naga::Span::UNDEFINED);
                            return Ok(RegionOutcome::Terminal);
                        }
                    } else if <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(inst_set, inst).is_some() {
                        if let Some(value) = *result_expr {
                            let pointer = func.expressions.append(naga::Expression::LocalVariable(return_local), naga::Span::UNDEFINED);
                            target.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
                        }
                        let returned = func.expressions.append(
                            naga::Expression::Literal(naga::Literal::Bool(true)),
                            naga::Span::UNDEFINED,
                        );
                        let pointer = func.expressions.append(
                            naga::Expression::LocalVariable(did_return_local),
                            naga::Span::UNDEFINED,
                        );
                        target.push(naga::Statement::Store { pointer, value: returned }, naga::Span::UNDEFINED);
                        *may_return = true;
                        target.push(naga::Statement::Break, naga::Span::UNDEFINED);
                        return Ok(RegionOutcome::Terminal);
                    } else if let Some(mem_ctx) = mem_ctx {
                        if <&sonatina_ir::inst::control_flow::Unreachable as InstDowncast>::downcast(inst_set, inst).is_some() {
                            // Review finding 4 analog, mid-loop bounds trap (e.g. a
                            // dynamically-indexed store inside a `while`):
                            // raise the flag and `Break` out of the loop
                            // rather than let the (undefined/poison) trap
                            // arm silently fall through as ordinary control
                            // flow. `Break` here also bounds how much
                            // garbage-but-safe computation runs after a
                            // detected fault, though correctness does not
                            // depend on that -- every access is independently
                            // clamped (see `emit_mem_access`) and the
                            // consumer is expected to check `mem_ctx.trapped`
                            // before trusting anything this invocation wrote.
                            mark_trapped_always(func, target, mem_ctx);
                            target.push(naga::Statement::Break, naga::Span::UNDEFINED);
                            terminated = true;
                            break;
                        }
                    }
                }
                if terminated {
                    return Ok(RegionOutcome::Terminal);
                }
                outcome = Some(RegionOutcome::Fallthrough(*block));
            }
            crate::structurize::Region::IfThenElse { header, then_branch, else_branch, merge } => {
                if let Some(merge) = merge {
                    ensure_phi_locals(
                        function, inst_set, word, *merge, word_type, f32_type, bool_type, func,
                        value_map, phi_locals,
                    )?;
                }
                emit_phi_loads_for_block(
                    function, inst_set, *header, func, target, value_map, phi_locals,
                );
                let mut mem_error = None;
                emit_block_to_target(function, inst_set, word, *header, func, target, value_map, phi_locals, result_expr, return_abi, naga_functions, mem_ctx, &mut mem_error);
                if let Some(msg) = mem_error.take() {
                    return Err(msg);
                }
                let branch = function.layout.iter_inst(*header).find_map(|iid|
                    <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, function.dfg.inst(iid))
                ).ok_or_else(|| format!("spirv: if header {header:?} has no branch"))?;
                let condition = resolve_naga_value(*branch.cond(), function, word, value_map, phi_locals, func)
                    .ok_or_else(|| format!("spirv: unresolved condition in {header:?}"))?;
                let mut accept = naga::Block::new();
                let mut reject = naga::Block::new();
                let mut accept_values = value_map.clone();
                let mut reject_values = value_map.clone();
                let then_outcome = if then_branch.is_empty() { RegionOutcome::Fallthrough(*header) } else {
                    emit_regions_in_loop(function, inst_set, word, then_branch, loop_header, loop_exit, word_type, f32_type, bool_type, return_local, did_return_local, may_return, func, &mut accept, &mut accept_values, phi_locals, result_expr, return_abi, naga_functions, mem_ctx)?
                };
                let else_outcome = if else_branch.is_empty() { RegionOutcome::Fallthrough(*header) } else {
                    emit_regions_in_loop(function, inst_set, word, else_branch, loop_header, loop_exit, word_type, f32_type, bool_type, return_local, did_return_local, may_return, func, &mut reject, &mut reject_values, phi_locals, result_expr, return_abi, naga_functions, mem_ctx)?
                };
                if let Some(merge) = merge {
                    // `from == merge` means the arm already LANDED at the merge
                    // (a trailing nested loop whose exit is the merge, or a
                    // trailing nested if merging directly at the outer merge);
                    // its phi transfer was emitted inside that region with the
                    // true CFG predecessor, so a second transfer keyed on the
                    // merge itself would be a bogus self-edge.
                    if let RegionOutcome::Fallthrough(from) = then_outcome && from != *merge {
                        emit_exact_phi_edge(function, inst_set, word, from, *merge, func, &mut accept, &mut accept_values, phi_locals)?;
                    }
                    if let RegionOutcome::Fallthrough(from) = else_outcome && from != *merge {
                        emit_exact_phi_edge(function, inst_set, word, from, *merge, func, &mut reject, &mut reject_values, phi_locals)?;
                    }
                    if *merge == loop_exit {
                        // A source-level `break` nested in a conditional can
                        // make the loop's canonical exit the conditional's
                        // immediate postdominator. Such a direct arm is empty
                        // in the region tree: its edge is represented by the
                        // arm's `Fallthrough(header)` outcome. Preserve the
                        // exact exit-phi transfer above, then terminate every
                        // still-falling-through arm with a Naga `Break`.
                        if matches!(then_outcome, RegionOutcome::Fallthrough(_)) {
                            accept.push(naga::Statement::Break, naga::Span::UNDEFINED);
                        }
                        if matches!(else_outcome, RegionOutcome::Fallthrough(_)) {
                            reject.push(naga::Statement::Break, naga::Span::UNDEFINED);
                        }
                        target.push(
                            naga::Statement::If { condition, accept, reject },
                            naga::Span::UNDEFINED,
                        );
                        return Ok(RegionOutcome::Terminal);
                    }
                    outcome = Some(RegionOutcome::Fallthrough(*merge));
                } else if matches!(then_outcome, RegionOutcome::Terminal) && matches!(else_outcome, RegionOutcome::Terminal) {
                    target.push(naga::Statement::If { condition, accept, reject }, naga::Span::UNDEFINED);
                    return Ok(RegionOutcome::Terminal);
                } else {
                    return Err(format!("spirv: divergent conditional {header:?} has a fallthrough arm but no merge"));
                }
                target.push(naga::Statement::If { condition, accept, reject }, naga::Span::UNDEFINED);
            }
            crate::structurize::Region::Loop { header, body } => {
                let mut nested_result = None;
                let inner_return = emit_recursive_loop_region(
                    function, inst_set, word, *header, body, word_type, f32_type, bool_type,
                    func, value_map, phi_locals, &mut nested_result, target, return_abi,
                    naga_functions, mem_ctx,
                )?;
                let inner_exit = structured_loop_exit(function, inst_set, *header, body)?;
                if let Some(inner_return) = inner_return {
                    let remaining = &regions[region_index + 1..];
                    let mut continuation = naga::Block::new();
                    let continuation_outcome = if remaining.is_empty() {
                        RegionOutcome::Fallthrough(inner_exit)
                    } else {
                        emit_regions_in_loop(
                            function,
                            inst_set,
                            word,
                            remaining,
                            loop_header,
                            loop_exit,
                            word_type,
                            f32_type,
                            bool_type,
                            return_local,
                            did_return_local,
                            may_return,
                            func,
                            &mut continuation,
                            value_map,
                            phi_locals,
                            result_expr,
                            return_abi,
                            naga_functions,
                            mem_ctx,
                        )?
                    };
                    let returned = load_structured_return_flag(inner_return, func, target);
                    let mut accept = naga::Block::new();
                    let outer_transport = StructuredReturnTransport {
                        result: return_local,
                        did_return: did_return_local,
                        has_result: inner_return.has_result,
                    };
                    forward_structured_return(
                        inner_return,
                        outer_transport,
                        func,
                        &mut accept,
                    )?;
                    accept.push(naga::Statement::Break, naga::Span::UNDEFINED);
                    target.push(
                        naga::Statement::If {
                            condition: returned,
                            accept,
                            reject: continuation,
                        },
                        naga::Span::UNDEFINED,
                    );
                    *may_return = true;
                    return Ok(continuation_outcome);
                }
                outcome = Some(RegionOutcome::Fallthrough(inner_exit));
            }
            crate::structurize::Region::LoopExit { from, target: exit } => {
                ensure_phi_locals(
                    function, inst_set, word, *exit, word_type, f32_type, bool_type, func, value_map,
                    phi_locals,
                )?;
                emit_exact_phi_edge(
                    function, inst_set, word, *from, *exit, func, target, value_map, phi_locals,
                )?;
                if block_has_return(*exit, function, inst_set) {
                    emit_phi_loads_for_block(
                        function, inst_set, *exit, func, target, value_map, phi_locals,
                    );
                    let mut mem_error = None;
                    emit_block_to_target(
                        function, inst_set, word, *exit, func, target, value_map, phi_locals,
                        result_expr, return_abi, naga_functions, mem_ctx, &mut mem_error,
                    );
                    if let Some(msg) = mem_error.take() {
                        return Err(msg);
                    }
                    if let Some(value) = *result_expr {
                        let pointer = func.expressions.append(
                            naga::Expression::LocalVariable(return_local),
                            naga::Span::UNDEFINED,
                        );
                        target.push(
                            naga::Statement::Store { pointer, value },
                            naga::Span::UNDEFINED,
                        );
                    }
                    let returned = func.expressions.append(
                        naga::Expression::Literal(naga::Literal::Bool(true)),
                        naga::Span::UNDEFINED,
                    );
                    let pointer = func.expressions.append(
                        naga::Expression::LocalVariable(did_return_local),
                        naga::Span::UNDEFINED,
                    );
                    target.push(
                        naga::Statement::Store { pointer, value: returned },
                        naga::Span::UNDEFINED,
                    );
                    *may_return = true;
                }
                target.push(naga::Statement::Break, naga::Span::UNDEFINED);
                return Ok(RegionOutcome::Terminal);
            }
            crate::structurize::Region::LoopContinue { from, target: header } => {
                if *header != loop_header {
                    return Err(format!(
                        "spirv: loop continue edge {from:?}->{header:?} targets a foreign header; expected {loop_header:?}"
                    ));
                }
                emit_exact_phi_edge(
                    function, inst_set, word, *from, *header, func, target, value_map, phi_locals,
                )?;
                target.push(naga::Statement::Continue, naga::Span::UNDEFINED);
                return Ok(RegionOutcome::Terminal);
            }
        }
    }
    outcome.ok_or_else(|| "spirv: empty region sequence has no control outcome".to_string())
}

#[cfg(feature = "spirv-backend")]
fn emit_recursive_loop_region(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    header: sonatina_ir::BlockId,
    body_regions: &[crate::structurize::Region],
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    _result_expr: &mut Option<naga::Handle<naga::Expression>>,
    // The Naga block this loop's statements are appended to. At top level the
    // caller passes a fresh block and extends `func.body` with it; inside a
    // conditional arm it is the arm's block, so a loop nested in an `if` lands in
    // the arm rather than the function root. (`func.body` cannot be passed here
    // directly, it would alias the `&mut func` this fn also needs.)
    target: &mut naga::Block,
    return_abi: Option<&NagaResultAbi>,
    naga_functions: &NagaFunctionMap,
    mem_ctx: Option<MemCtx>,
) -> Result<Option<StructuredReturnTransport>, String> {
    use sonatina_ir::InstDowncast;
    let mut loop_blocks = std::collections::HashSet::new();
    loop_blocks.insert(header);
    region_blocks(body_regions, &mut loop_blocks);
    ensure_phi_locals(
        function, inst_set, word, header, word_type, f32_type, bool_type, func, value_map,
        phi_locals,
    )?;

    let mut outside_preds = Vec::new();
    for iid in function.layout.iter_inst(header) {
        let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(
            inst_set,
            function.dfg.inst(iid),
        ) else {
            break;
        };
        for (_, pred) in phi.args() {
            if !loop_blocks.contains(pred) && !outside_preds.contains(pred) {
                outside_preds.push(*pred);
            }
        }
    }
    // A conventional preheader is the sole owner of loop initialization, so
    // materialize its phi edge here. With several outside predecessors, the
    // loop header is also the merge of preceding structured control flow. Its
    // branch arms have already materialized their exact phi edges. Selecting
    // one predecessor here would overwrite the path-dependent initialization
    // after the branch, making every invocation observe whichever edge was
    // encountered first.
    if let [outside_pred] = outside_preds.as_slice() {
        let mut init = naga::Block::new();
        emit_exact_phi_edge(
            function,
            inst_set,
            word,
            *outside_pred,
            header,
            func,
            &mut init,
            value_map,
            phi_locals,
        )?;
        target.extend_block(init);
    }

    let source_return_ty = function.layout.iter_block().find_map(|block| {
        find_block_return_value(block, function, inst_set).map(|value| function.dfg.value_ty(value))
    });
    let (return_naga_ty, has_result) = match return_abi {
        Some(return_abi) => (
            return_abi.physical_type.unwrap_or(word_type),
            return_abi.physical_arity != 0,
        ),
        None => (
            match source_return_ty {
                Some(sonatina_ir::Type::F32) => f32_type,
                Some(sonatina_ir::Type::I1) => bool_type,
                Some(_) | None => word_type,
            },
            source_return_ty.is_some(),
        ),
    };
    let return_local = func.local_variables.append(
        naga::LocalVariable { name: Some("loop_result".into()), ty: return_naga_ty, init: None },
        naga::Span::UNDEFINED,
    );
    let returned_false = func.expressions.append(
        naga::Expression::Literal(naga::Literal::Bool(false)),
        naga::Span::UNDEFINED,
    );
    let did_return_local = func.local_variables.append(
        naga::LocalVariable {
            name: Some("loop_did_return".into()),
            ty: bool_type,
            init: Some(returned_false),
        },
        naga::Span::UNDEFINED,
    );
    let mut header_carry_locals = Vec::new();
    for inst_id in function.layout.iter_inst(header) {
        let inst = function.dfg.inst(inst_id);
        if <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst).is_some() {
            continue;
        }
        let Some(result) = function.dfg.inst_result(inst_id) else { continue };
        let result_ty = function.dfg.value_ty(result);
        let rematerializable_pointer = result_ty.resolve_compound(function.ctx()).is_some_and(|ty| {
            matches!(ty, sonatina_ir::types::CompoundType::Ptr(_))
        }) && (
            <&sonatina_ir::inst::data::Gep as InstDowncast>::downcast(
                inst_set, function.dfg.inst(inst_id),
            ).is_some()
                || <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(
                    inst_set, function.dfg.inst(inst_id),
                ).is_some_and(|bitcast| {
                    typed_local_zero_projection_path(
                        function.ctx(),
                        function.dfg.value_ty(*bitcast.from()),
                        *bitcast.ty(),
                    ).is_some()
                })
        );
        if rematerializable_pointer {
            continue;
        }
        let ty = match result_ty {
            sonatina_ir::Type::F32 => f32_type,
            sonatina_ir::Type::I1 => bool_type,
            sonatina_ir::Type::I32 if word == WordKind::U32 => word_type,
            sonatina_ir::Type::I64 if word == WordKind::I64 => word_type,
            _ => return Err(format!(
                "spirv: loop-header result {result:?} has unsupported cross-scope type {result_ty:?}"
            )),
        };
        let local = func.local_variables.append(
            naga::LocalVariable {
                name: Some(format!("loop_header_carry_{}", result.0)),
                ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        header_carry_locals.push((result, local));
    }
    // Loop-internal Returns are transported through `return_local`; their
    // expression handles are scoped to nested Naga blocks and must never be
    // published to the enclosing function result sink.
    let mut nested_result_expr = None;
    let mut loop_body = naga::Block::new();
    emit_phi_loads_for_block(function, inst_set, header, func, &mut loop_body, value_map, phi_locals);
    let mut mem_error = None;
    emit_block_to_target(
        function, inst_set, word, header, func, &mut loop_body, value_map,
        phi_locals, &mut nested_result_expr, return_abi, naga_functions, mem_ctx, &mut mem_error,
    );
    if let Some(msg) = mem_error.take() {
        return Err(msg);
    }
    let branch = function.layout.iter_inst(header).find_map(|iid|
        <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, function.dfg.inst(iid))
    ).ok_or_else(|| format!("spirv: loop header {header:?} has no branch"))?;
    let condition = resolve_naga_value(*branch.cond(), function, word, value_map, phi_locals, func)
        .ok_or_else(|| format!("spirv: unresolved loop condition in {header:?}"))?;
    let nz_in = loop_blocks.contains(branch.nz_dest());
    let z_in = loop_blocks.contains(branch.z_dest());
    if nz_in == z_in {
        return Err(format!("spirv: loop {header:?} must have exactly one in-loop successor"));
    }
    let exit = if nz_in { *branch.z_dest() } else { *branch.nz_dest() };
    ensure_phi_locals(
        function, inst_set, word, exit, word_type, f32_type, bool_type, func, value_map,
        phi_locals,
    )?;
    let mut continue_arm = naga::Block::new();
    let mut continue_values = value_map.clone();
    let mut may_return = false;
    if body_regions.is_empty() {
        // A header-only loop: the header is its own latch, so the continue arm
        // is exactly the self back-edge phi transfer (each header phi has one
        // arg whose predecessor is the header itself).
        emit_exact_phi_edge(
            function, inst_set, word, header, header, func, &mut continue_arm,
            &mut continue_values, phi_locals,
        )?;
    } else {
        let body_outcome = emit_regions_in_loop(
            function, inst_set, word, body_regions, header, exit, word_type, f32_type, bool_type,
            return_local, did_return_local, &mut may_return, func, &mut continue_arm,
            &mut continue_values, phi_locals, &mut nested_result_expr, return_abi,
            naga_functions, mem_ctx,
        )?;
        // Fallthrough(header) is a trailing nested loop whose exit IS this
        // loop's back-edge: its exit arm already stored this header's phi
        // locals, and falling off the Naga loop body is an implicit continue.
        // Any other fallthrough would silently continue instead of breaking;
        // conditional break cascades are consumed above, so remaining shapes
        // still fail closed.
        if let RegionOutcome::Fallthrough(resume) = body_outcome && resume != header {
            return Err(format!(
                "spirv structurize: loop {header:?} body falls through to {resume:?} \
                 instead of terminating; canonical exit={exit:?}; body regions={body_regions:?}"
            ));
        }
    }
    let mut exit_arm = naga::Block::new();
    let mut exit_values = value_map.clone();
    emit_exact_phi_edge(
        function, inst_set, word, header, exit, func, &mut exit_arm, &mut exit_values, phi_locals,
    )?;
    if block_has_return(exit, function, inst_set) {
        emit_phi_loads_for_block(
            function, inst_set, exit, func, &mut exit_arm, &mut exit_values, phi_locals,
        );
        let mut mem_error = None;
        emit_block_to_target(
            function, inst_set, word, exit, func, &mut exit_arm, &mut exit_values,
            phi_locals, &mut nested_result_expr, return_abi, naga_functions, mem_ctx,
            &mut mem_error,
        );
        if let Some(msg) = mem_error.take() {
            return Err(msg);
        }
        if let Some(value) = nested_result_expr {
            let pointer = func.expressions.append(naga::Expression::LocalVariable(return_local), naga::Span::UNDEFINED);
            exit_arm.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
        }
        let returned = func.expressions.append(
            naga::Expression::Literal(naga::Literal::Bool(true)),
            naga::Span::UNDEFINED,
        );
        let pointer = func.expressions.append(
            naga::Expression::LocalVariable(did_return_local),
            naga::Span::UNDEFINED,
        );
        exit_arm.push(naga::Statement::Store { pointer, value: returned }, naga::Span::UNDEFINED);
        may_return = true;
    }
    for &(result, local) in &header_carry_locals {
        let value = *value_map.get(&result)
            .ok_or_else(|| format!("spirv: loop header result {result:?} was not emitted"))?;
        let pointer = func.expressions.append(naga::Expression::LocalVariable(local), naga::Span::UNDEFINED);
        exit_arm.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
    }
    exit_arm.push(naga::Statement::Break, naga::Span::UNDEFINED);
    let (accept, reject) = if nz_in { (continue_arm, exit_arm) } else { (exit_arm, continue_arm) };
    loop_body.push(naga::Statement::If { condition, accept, reject }, naga::Span::UNDEFINED);
    target.push(naga::Statement::Loop { body: loop_body, continuing: naga::Block::new(), break_if: None }, naga::Span::UNDEFINED);

    // Header-phi Loads created at the top of the loop are scoped inside the
    // Naga Loop statement. A normal loop exit may resume at a sibling block
    // that legally uses those SSA values, so replace every loop-scoped handle
    // with a fresh outer-block Load from its typed phi local.
    for inst_id in function.layout.iter_inst(header) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            value_map.remove(&result);
        }
    }
    let mut outer_phi_loads = naga::Block::new();
    emit_phi_loads_for_block(
        function, inst_set, header, func, &mut outer_phi_loads, value_map, phi_locals,
    );
    target.extend_block(outer_phi_loads);
    for (result, local) in header_carry_locals {
        value_map.remove(&result);
        let pointer = func.expressions.append(naga::Expression::LocalVariable(local), naga::Span::UNDEFINED);
        let loaded = func.expressions.append(naga::Expression::Load { pointer }, naga::Span::UNDEFINED);
        target.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)),
            naga::Span::UNDEFINED,
        );
        value_map.insert(result, loaded);
    }

    Ok(may_return.then_some(StructuredReturnTransport {
        result: return_local,
        did_return: did_return_local,
        has_result,
    }))
}

/// Find the return value (ValueId) of a block that contains a Return instruction.
#[cfg(feature = "spirv-backend")]
fn find_block_return_value(
    block: sonatina_ir::BlockId,
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
) -> Option<sonatina_ir::ValueId> {
    use sonatina_ir::InstDowncast;
    for inst_id in function.layout.iter_inst(block) {
        let inst_data = function.dfg.inst(inst_id);
        if let Some(ret) = <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(inst_set, inst_data) {
            return ret.args().as_slice().first().copied();
        }
    }
    None
}

#[cfg(feature = "spirv-backend")]
fn block_has_return(
    block: sonatina_ir::BlockId,
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
) -> bool {
    use sonatina_ir::InstDowncast;
    function.layout.iter_inst(block).any(|inst_id| {
        <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(
            inst_set,
            function.dfg.inst(inst_id),
        )
        .is_some()
    })
}

#[cfg(feature = "spirv-backend")]
fn structurize_error_with_block_ir(
    error: String,
    func_ref: sonatina_ir::module::FuncRef,
    function: &sonatina_ir::Function,
) -> String {
    let function_name = function
        .ctx()
        .get_sig(func_ref)
        .map(|signature| signature.name().to_string())
        .unwrap_or_else(|| format!("{func_ref:?}"));
    let error = format!(
        "spirv lowering failed in function `%{function_name}` ({func_ref:?}): {error}"
    );
    let mut labels = Vec::new();
    for token in error.split(|character: char| !character.is_ascii_alphanumeric()) {
        let Some(number) = token.strip_prefix("block") else {
            continue;
        };
        if number.is_empty() || !number.chars().all(|character| character.is_ascii_digit()) {
            continue;
        }
        if !labels.iter().any(|label| label == token) {
            labels.push(token.to_string());
        }
    }
    if labels.is_empty() {
        return error;
    }

    let function_ir = FuncWriter::new(func_ref, function).dump_string();
    let lines = function_ir.lines().collect::<Vec<_>>();
    let is_block_label = |line: &str| {
        let trimmed = line.trim();
        let Some(name) = trimmed.strip_suffix(':') else {
            return false;
        };
        let Some(number) = name.strip_prefix("block") else {
            return false;
        };
        !number.is_empty() && number.chars().all(|character| character.is_ascii_digit())
    };
    let mut snippets = Vec::new();
    let mut label_index = 0;
    while label_index < labels.len() && label_index < 32 {
        let label = labels[label_index].clone();
        label_index += 1;
        let Some(start) = lines.iter().position(|line| line.trim() == format!("{label}:")) else {
            snippets.push(format!("{label}: <unavailable>"));
            continue;
        };
        let end = lines
            .iter()
            .enumerate()
            .skip(start + 1)
            .find_map(|(index, line)| is_block_label(line).then_some(index))
            .unwrap_or(lines.len());
        let body = &lines[start..end];
        for line in body {
            for token in line.split(|character: char| !character.is_ascii_alphanumeric()) {
                let Some(number) = token.strip_prefix("block") else {
                    continue;
                };
                if number.is_empty()
                    || !number.chars().all(|character| character.is_ascii_digit())
                {
                    continue;
                }
                if !labels.iter().any(|known| known == token) {
                    labels.push(token.to_string());
                }
            }
        }
        let compact = if body.len() <= 16 {
            body.join(" | ")
        } else {
            format!(
                "{} | <{} lines omitted> | {}",
                body[..5].join(" | "),
                body.len() - 13,
                body[body.len() - 8..].join(" | ")
            )
        };
        snippets.push(compact);
    }
    format!("{error}; implicated IR: {}", snippets.join(" || "))
}

#[cfg(feature = "spirv-backend")]
fn instruction_ir_context(
    func_ref: sonatina_ir::module::FuncRef,
    function: &sonatina_ir::Function,
    block: sonatina_ir::BlockId,
    instruction: &str,
) -> String {
    let function_ir = FuncWriter::new(func_ref, function).dump_string();
    let lines = function_ir.lines().collect::<Vec<_>>();
    let block_label = format!("{block:?}:");
    let Some(block_start) = lines.iter().position(|line| line.trim() == block_label) else {
        return "<block unavailable>".to_string();
    };
    let block_end = lines
        .iter()
        .enumerate()
        .skip(block_start + 1)
        .find_map(|(index, line)| {
            let trimmed = line.trim();
            let name = trimmed.strip_suffix(':')?;
            let number = name.strip_prefix("block")?;
            (!number.is_empty() && number.chars().all(|character| character.is_ascii_digit()))
                .then_some(index)
        })
        .unwrap_or(lines.len());
    let instruction_index = lines[block_start..block_end]
        .iter()
        .position(|line| line.contains(instruction))
        .map(|index| block_start + index)
        .unwrap_or(block_end.saturating_sub(1));
    let start = instruction_index.saturating_sub(10).max(block_start);
    let end = (instruction_index + 4).min(block_end);
    lines[start..end].join(" | ")
}

/// Under a u32 word, these signedness-sensitive ops have no correct signless
/// lowering yet, so they fail closed. Returns the op's name if `inst_data` is one
/// of them, else `None`. (Add/Sub/Mul are sign-agnostic under wrapping and stay
/// enabled; the emitted `Lt` maps to naga `Less`, which is an unsigned compare on
/// a `Uint` scalar, so it is not in this set. `Sar` and `Slt` are now handled
/// word-aware via an i32 bitcast in `emit_single_inst`, so they are no longer in
/// this set; a non-immediate-bits `Sar` still fails closed in the pre-scan.)
#[cfg(feature = "spirv-backend")]
fn unsupported_signed_op_under_u32(
    is: &dyn sonatina_ir::InstSetBase,
    inst_data: &dyn sonatina_ir::Inst,
) -> Option<&'static str> {
    use sonatina_ir::{
        InstDowncast,
        inst::{arith, cmp},
    };
    if <&arith::Sdiv as InstDowncast>::downcast(is, inst_data).is_some() {
        return Some("sdiv");
    }
    if <&arith::Smod as InstDowncast>::downcast(is, inst_data).is_some() {
        return Some("smod");
    }
    if <&cmp::Sgt as InstDowncast>::downcast(is, inst_data).is_some() {
        return Some("sgt");
    }
    if <&cmp::Sle as InstDowncast>::downcast(is, inst_data).is_some() {
        return Some("sle");
    }
    if <&cmp::Sge as InstDowncast>::downcast(is, inst_data).is_some() {
        return Some("sge");
    }
    None
}

#[cfg(feature = "spirv-backend")]
fn spirv_instruction_is_lowered(
    is: &dyn sonatina_ir::InstSetBase,
    inst: &dyn sonatina_ir::Inst,
) -> bool {
    use sonatina_ir::{InstDowncast, inst::{arith, cmp, control_flow, data, logic}};

    inst.is_terminator()
        || <&control_flow::Phi as InstDowncast>::downcast(is, inst).is_some()
        || <&control_flow::Call as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Add as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Sub as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Mul as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Udiv as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Umod as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fneg as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fadd as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fsub as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fmul as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fdiv as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fsqrt as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fabs as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fmin as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fmax as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::FminRelaxed as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::FmaxRelaxed as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fclamp as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Ffloor as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fceil as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Ftrunc as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fround as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Sar as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Shl as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Shr as InstDowncast>::downcast(is, inst).is_some()
        || <&logic::And as InstDowncast>::downcast(is, inst).is_some()
        || <&logic::Or as InstDowncast>::downcast(is, inst).is_some()
        || <&logic::Xor as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::Lt as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::Eq as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::Ne as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::Slt as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::IsZero as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::Feq as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::Flt as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::Fle as InstDowncast>::downcast(is, inst).is_some()
        || <&sonatina_ir::inst::cast::I32ToF32 as InstDowncast>::downcast(is, inst).is_some()
        || <&sonatina_ir::inst::cast::U32ToF32 as InstDowncast>::downcast(is, inst).is_some()
        || <&sonatina_ir::inst::cast::F32ToI32 as InstDowncast>::downcast(is, inst).is_some()
        || <&sonatina_ir::inst::cast::F32ToU32 as InstDowncast>::downcast(is, inst).is_some()
        || <&sonatina_ir::inst::cast::Trunc as InstDowncast>::downcast(is, inst).is_some()
        || <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(is, inst).is_some()
        || <&data::Alloca as InstDowncast>::downcast(is, inst).is_some()
        || <&data::Gep as InstDowncast>::downcast(is, inst).is_some()
        || <&data::ObjAlloc as InstDowncast>::downcast(is, inst).is_some()
        || <&data::ObjStore as InstDowncast>::downcast(is, inst).is_some()
        || <&data::ObjLoad as InstDowncast>::downcast(is, inst).is_some()
        || <&data::ObjIndex as InstDowncast>::downcast(is, inst).is_some()
        || <&data::ObjProj as InstDowncast>::downcast(is, inst).is_some()
        || <&data::MemAllocDynamic as InstDowncast>::downcast(is, inst).is_some()
        || <&data::MemCheckpoint as InstDowncast>::downcast(is, inst).is_some()
        || <&data::MemRewind as InstDowncast>::downcast(is, inst).is_some()
        || <&data::Memcopy as InstDowncast>::downcast(is, inst).is_some()
        || <&data::Mload as InstDowncast>::downcast(is, inst).is_some()
        || <&data::Mstore as InstDowncast>::downcast(is, inst).is_some()
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone)]
struct NagaFunctionInfo {
    handle: naga::Handle<naga::Function>,
    argument_abi: Vec<NagaArgumentSource>,
    packed_arguments: Option<NagaPackedArguments>,
    result_abi: NagaResultAbi,
    memory_abi: NagaMemoryAbi,
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy)]
enum NagaArgumentSource {
    Physical(u32),
    Packed {
        physical_index: u32,
        group_index: u32,
        member_index: u32,
    },
    ImplicitResource(naga::Handle<naga::GlobalVariable>),
    /// A logical source argument proven unable to reach a result or effect.
    /// It remains in Sonatina's stable signature but has no physical Naga ABI.
    Dead,
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone)]
struct NagaPackedArguments {
    ty: naga::Handle<naga::Type>,
    physical_index: u32,
    groups: Vec<NagaPackedArgumentGroup>,
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone)]
struct NagaPackedArgumentGroup {
    ty: naga::Handle<naga::Type>,
    member_count: usize,
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy)]
enum NagaResultSource {
    Physical(u32),
    PassthroughArgument(u32),
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone)]
struct NagaResultAbi {
    logical: Vec<NagaResultSource>,
    physical_type: Option<naga::Handle<naga::Type>>,
    physical_arity: usize,
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy, Default)]
struct NagaMemoryAbi {
    heap: bool,
    trap: bool,
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy, Default)]
struct NagaMemoryAbiTypes {
    heap: Option<naga::Handle<naga::Type>>,
    word: Option<naga::Handle<naga::Type>>,
    trap: Option<naga::Handle<naga::Type>>,
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy)]
struct NagaTypedLocalType {
    handle: naga::Handle<naga::Type>,
    alignment: u32,
    size: u32,
}

#[cfg(feature = "spirv-backend")]
#[derive(Default)]
struct NagaTypedLocalUseClosure {
    allocation_types: Vec<sonatina_ir::Type>,
    borrowed_pointer_types: Vec<sonatina_ir::Type>,
}

// This is a conservative compiler policy, not a WebGPU hardware limit. It keeps
// the first typed-local slice from silently moving a large arena into private
// storage for every invocation. Device-tuned limits can replace it later.
#[cfg(feature = "spirv-backend")]
const MAX_NAGA_TYPED_PRIVATE_BYTES_PER_FUNCTION: u32 = 16 * 1024;

// Dawn's portable WGSL frontend limits one function declaration to 255
// parameters. Preserve helper calls beyond that logical arity by grouping
// store-type values into one function-local ABI struct.
#[cfg(feature = "spirv-backend")]
const MAX_WGSL_FUNCTION_PARAMETERS: usize = 255;

#[cfg(feature = "spirv-backend")]
#[derive(Default)]
struct NagaFunctionMap {
    call_sites:
        std::collections::HashMap<sonatina_ir::InstId, NagaFunctionInfo>,
    typed_local_types:
        std::collections::HashMap<sonatina_ir::Type, NagaTypedLocalType>,
}

#[cfg(feature = "spirv-backend")]
impl NagaFunctionMap {
    fn new() -> Self {
        Self::default()
    }

    fn with_typed_local_types(
        typed_local_types: std::collections::HashMap<
            sonatina_ir::Type,
            NagaTypedLocalType,
        >,
    ) -> Self {
        Self {
            call_sites: std::collections::HashMap::new(),
            typed_local_types,
        }
    }

    fn call(
        &self,
        instruction: &sonatina_ir::InstId,
    ) -> Option<&NagaFunctionInfo> {
        self.call_sites.get(instruction)
    }

    fn replace_call_sites(
        &mut self,
        call_sites: std::collections::HashMap<
            sonatina_ir::InstId,
            NagaFunctionInfo,
        >,
    ) {
        self.call_sites = call_sites;
    }

    fn typed_local_type(
        &self,
        ty: sonatina_ir::Type,
    ) -> Option<NagaTypedLocalType> {
        self.typed_local_types.get(&ty).copied()
    }
}

#[cfg(feature = "spirv-backend")]
type NagaResourceCapabilities =
    std::collections::HashSet<sonatina_ir::Type>;

#[cfg(feature = "spirv-backend")]
type NagaLogicalResultAbis = std::collections::HashMap<
    sonatina_ir::module::FuncRef,
    Vec<NagaResultSource>,
>;

#[cfg(feature = "spirv-backend")]
type NagaLiveArguments = rustc_hash::FxHashMap<
    sonatina_ir::module::FuncRef,
    Vec<bool>,
>;

#[cfg(feature = "spirv-backend")]
fn naga_argument_is_live(
    live_arguments: &NagaLiveArguments,
    function: sonatina_ir::module::FuncRef,
    argument_index: usize,
) -> bool {
    live_arguments
        .get(&function)
        .and_then(|mask| mask.get(argument_index))
        .copied()
        .unwrap_or(true)
}

#[cfg(feature = "spirv-backend")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NagaFunctionVariant {
    function: sonatina_ir::module::FuncRef,
    ordinal: u32,
}

#[cfg(feature = "spirv-backend")]
type NagaResourceVariantBindings = std::collections::HashMap<
    sonatina_ir::module::FuncRef,
    Vec<Vec<Option<naga::Handle<naga::GlobalVariable>>>>,
>;

#[cfg(feature = "spirv-backend")]
struct NagaResourceVariants {
    entry: NagaFunctionVariant,
    bindings: NagaResourceVariantBindings,
    calls: std::collections::HashMap<
        (NagaFunctionVariant, sonatina_ir::InstId),
        NagaFunctionVariant,
    >,
}

#[cfg(feature = "spirv-backend")]
impl NagaResourceVariants {
    fn intern(
        &mut self,
        function: sonatina_ir::module::FuncRef,
        bindings: Vec<Option<naga::Handle<naga::GlobalVariable>>>,
    ) -> Result<NagaFunctionVariant, String> {
        let variants = self.bindings.entry(function).or_default();
        let ordinal = if let Some(index) = variants
            .iter()
            .position(|existing| *existing == bindings)
        {
            index
        } else {
            variants.push(bindings);
            variants.len() - 1
        };
        Ok(NagaFunctionVariant {
            function,
            ordinal: u32::try_from(ordinal).map_err(|_| {
                format!(
                    "spirv: function {function:?} has more resource variants than fit in u32. Fail closed."
                )
            })?,
        })
    }

    fn variants(
        &self,
        function: sonatina_ir::module::FuncRef,
    ) -> &[Vec<Option<naga::Handle<naga::GlobalVariable>>>] {
        self.bindings.get(&function).map(Vec::as_slice).unwrap_or(&[])
    }
}

#[cfg(feature = "spirv-backend")]
fn resource_identity_graph(
    function: &sonatina_ir::Function,
    resource_capabilities: &NagaResourceCapabilities,
    logical_result_abis: &NagaLogicalResultAbis,
) -> Result<
    std::collections::HashMap<sonatina_ir::ValueId, Vec<sonatina_ir::ValueId>>,
    String,
> {
    use sonatina_ir::{
        InstDowncast,
        inst::control_flow::{Call, Phi},
    };

    let inst_set = function.inst_set();
    let mut graph = std::collections::HashMap::<_, Vec<_>>::new();
    let mut connect = |left, right| {
        graph.entry(left).or_default().push(right);
        graph.entry(right).or_default().push(left);
    };
    for block in function.layout.iter_block() {
        for instruction in function.layout.iter_inst(block) {
            if let Some(call_info) = function.dfg.call_info(instruction) {
                let call = <&Call as InstDowncast>::downcast(
                    inst_set,
                    function.dfg.inst(instruction),
                )
                .ok_or_else(|| {
                    "spirv: resource provenance reached a call form without arguments. Fail closed."
                        .to_string()
                })?;
                let callee = logical_result_abis.get(&call_info.callee()).ok_or_else(|| {
                    format!(
                        "spirv: resource provenance reached {:?} before its helper ABI was available. Fail closed.",
                        call_info.callee(),
                    )
                })?;
                for (&result, source) in function
                    .dfg
                    .inst_results(instruction)
                    .iter()
                    .zip(callee)
                {
                    let NagaResultSource::PassthroughArgument(argument_index) = source else {
                        continue;
                    };
                    let argument = *call.args().get(*argument_index as usize).ok_or_else(|| {
                        format!(
                            "spirv: resource result from {:?} refers to missing argument {argument_index}. Fail closed.",
                            call_info.callee(),
                        )
                    })?;
                    connect(result, argument);
                }
                continue;
            }
            let Some(phi) = <&Phi as InstDowncast>::downcast(
                inst_set,
                function.dfg.inst(instruction),
            ) else {
                continue;
            };
            let Some(result) = function.dfg.inst_result(instruction) else {
                continue;
            };
            if !resource_capabilities.contains(&function.dfg.value_ty(result)) {
                continue;
            }
            for &(value, _) in phi.args() {
                connect(result, value);
            }
        }
    }
    Ok(graph)
}

#[cfg(feature = "spirv-backend")]
fn propagate_resource_identities<Identity>(
    graph: &std::collections::HashMap<sonatina_ir::ValueId, Vec<sonatina_ir::ValueId>>,
    seeds: impl IntoIterator<Item = (sonatina_ir::ValueId, Identity)>,
) -> Result<std::collections::HashMap<sonatina_ir::ValueId, Identity>, String>
where
    Identity: Copy + Eq + std::fmt::Debug,
{
    let mut provenance = std::collections::HashMap::new();
    let mut pending = std::collections::VecDeque::new();
    for (value, identity) in seeds {
        match provenance.insert(value, identity) {
            Some(previous) if previous != identity => {
                return Err(format!(
                    "spirv: resource seed {value:?} has conflicting identities {previous:?} and {identity:?}. Fail closed."
                ));
            }
            Some(_) => {}
            None => pending.push_back(value),
        }
    }
    while let Some(value) = pending.pop_front() {
        let identity = provenance[&value];
        for &alias in graph.get(&value).into_iter().flatten() {
            match provenance.entry(alias) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(identity);
                    pending.push_back(alias);
                }
                std::collections::hash_map::Entry::Occupied(entry)
                    if *entry.get() != identity =>
                {
                    return Err(format!(
                        "spirv: resource aliases {value:?} and {alias:?} carry conflicting identities {identity:?} and {:?}. Fail closed.",
                        entry.get(),
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(provenance)
}

#[cfg(feature = "spirv-backend")]
fn bind_resource_identity_aliases(
    function: &sonatina_ir::Function,
    seeds: impl IntoIterator<Item = sonatina_ir::ValueId>,
    resource_capabilities: &NagaResourceCapabilities,
    logical_result_abis: &NagaLogicalResultAbis,
    value_map: &mut std::collections::HashMap<
        sonatina_ir::ValueId,
        naga::Handle<naga::Expression>,
    >,
) -> Result<(), String> {
    let graph = resource_identity_graph(function, resource_capabilities, logical_result_abis)?;
    let mut seed_expressions = Vec::new();
    for seed in seeds {
        let expression = *value_map.get(&seed).ok_or_else(|| {
            format!(
                "spirv: resource seed {seed:?} has no physical Naga capability. Fail closed."
            )
        })?;
        seed_expressions.push((seed, expression));
    }
    let provenance = propagate_resource_identities(&graph, seed_expressions)?;

    for (value, expression) in provenance {
        value_map.insert(value, expression);
    }
    Ok(())
}

/// Return the direct-call closure rooted at `entry`, with every callee before
/// its callers. Naga permits calls only to module functions, so this order lets
/// us populate every reserved function body before the entry point is emitted.
/// Recursive SCCs are rejected explicitly because WGSL does not admit recursion.
#[cfg(feature = "spirv-backend")]
fn reachable_call_postorder(
    module: &Module,
    entry: sonatina_ir::module::FuncRef,
) -> Result<Vec<sonatina_ir::module::FuncRef>, String> {
    use sonatina_ir::InstDowncast;

    fn visit(
        module: &Module,
        function_ref: sonatina_ir::module::FuncRef,
        states: &mut std::collections::HashMap<sonatina_ir::module::FuncRef, u8>,
        order: &mut Vec<sonatina_ir::module::FuncRef>,
    ) -> Result<(), String> {
        match states.get(&function_ref).copied() {
            Some(1) => {
                let name = module
                    .ctx
                    .get_sig(function_ref)
                    .map(|signature| signature.name().to_string())
                    .unwrap_or_else(|| format!("{function_ref:?}"));
                return Err(format!(
                    "spirv: recursive helper call reaches `{name}`; WGSL recursion is unsupported. Fail closed."
                ));
            }
            Some(2) => return Ok(()),
            _ => {}
        }
        states.insert(function_ref, 1);
        let callees = module
            .func_store
            .try_view(function_ref, |function| {
                let inst_set = function.inst_set();
                let mut callees = Vec::new();
                for block in function.layout.iter_block() {
                    for instruction in function.layout.iter_inst(block) {
                        if let Some(call) =
                            <&sonatina_ir::inst::control_flow::Call as InstDowncast>::downcast(
                                inst_set,
                                function.dfg.inst(instruction),
                            )
                        {
                            callees.push(*call.callee());
                        }
                    }
                }
                callees
            })
            .ok_or_else(|| {
                format!(
                    "spirv: reachable function {function_ref:?} has no body; imported helpers are unsupported. Fail closed."
                )
            })?;
        for callee in callees {
            visit(module, callee, states, order)?;
        }
        states.insert(function_ref, 2);
        order.push(function_ref);
        Ok(())
    }

    let mut states = std::collections::HashMap::new();
    let mut order = Vec::new();
    visit(module, entry, &mut states, &mut order)?;
    Ok(order)
}

#[cfg(feature = "spirv-backend")]
fn helper_naga_logical_result_abis(
    module: &Module,
    call_order: &[sonatina_ir::module::FuncRef],
    roots: &[sonatina_ir::module::FuncRef],
    resource_capabilities: &NagaResourceCapabilities,
) -> Result<NagaLogicalResultAbis, String> {
    let mut result_abis = NagaLogicalResultAbis::new();
    for &function_ref in call_order {
        if roots.contains(&function_ref) {
            continue;
        }
        let logical = helper_naga_logical_result_abi(
            module,
            function_ref,
            resource_capabilities,
            &result_abis,
        )?;
        result_abis.insert(function_ref, logical);
    }
    Ok(result_abis)
}

#[cfg(feature = "spirv-backend")]
fn helper_resource_variants(
    module: &Module,
    call_order: &[sonatina_ir::module::FuncRef],
    entry: sonatina_ir::module::FuncRef,
    external_roots: &[(u32, naga::Handle<naga::GlobalVariable>)],
    resource_capabilities: &NagaResourceCapabilities,
    logical_result_abis: &NagaLogicalResultAbis,
    live_arguments: &NagaLiveArguments,
) -> Result<NagaResourceVariants, String> {
    use sonatina_ir::{InstDowncast, inst::control_flow::Call};

    let entry_signature = module
        .ctx
        .get_sig(entry)
        .ok_or_else(|| format!("spirv: entry {entry:?} has no signature"))?;
    let mut entry_bindings = vec![None; entry_signature.args().len()];
    for &(argument_index, global) in external_roots {
        let binding = entry_bindings
            .get_mut(argument_index as usize)
            .ok_or_else(|| {
                format!(
                    "spirv: external resource argument {argument_index} disappeared while deriving call-graph resource identities. Fail closed."
                )
            })?;
        match *binding {
            None => *binding = Some(global),
            Some(previous) if previous == global => {}
            Some(previous) => {
                return Err(format!(
                    "spirv: entry argument {argument_index} has conflicting external resources {previous:?} and {global:?}. Fail closed."
                ));
            }
        }
    }

    let mut variants = NagaResourceVariants {
        entry: NagaFunctionVariant {
            function: entry,
            ordinal: 0,
        },
        bindings: std::collections::HashMap::new(),
        calls: std::collections::HashMap::new(),
    };
    variants.entry = variants.intern(entry, entry_bindings)?;
    for &function_ref in call_order.iter().rev() {
        let signature = module
            .ctx
            .get_sig(function_ref)
            .ok_or_else(|| format!("spirv: reachable function {function_ref:?} has no signature"))?;
        let function_variants = variants.variants(function_ref).to_vec();
        if function_variants.is_empty() {
            return Err(format!(
                "spirv: reachable function `{}` has no entry-rooted resource variant. Fail closed.",
                signature.name(),
            ));
        }
        for (variant_index, function_bindings) in function_variants.iter().enumerate() {
            if function_bindings.len() != signature.args().len() {
                return Err(format!(
                    "spirv: function `{}` variant {variant_index} has {} call-graph resource bindings for {} arguments. Fail closed.",
                    signature.name(),
                    function_bindings.len(),
                    signature.args().len(),
                ));
            }
            let caller_variant = NagaFunctionVariant {
                function: function_ref,
                ordinal: u32::try_from(variant_index).map_err(|_| {
                    format!(
                        "spirv: function `{}` has more resource variants than fit in u32. Fail closed.",
                        signature.name(),
                    )
                })?,
            };
            module
                .func_store
                .try_view(function_ref, |function| -> Result<(), String> {
                    let graph = resource_identity_graph(
                        function,
                        resource_capabilities,
                        logical_result_abis,
                    )?;
                    let mut seeds = Vec::new();
                    for (argument_index, ((&value, &ty), binding)) in function
                        .arg_values
                        .iter()
                        .zip(signature.args())
                        .zip(function_bindings)
                        .enumerate()
                    {
                        if !resource_capabilities.contains(&ty)
                            || !naga_argument_is_live(
                                live_arguments,
                                function_ref,
                                argument_index,
                            )
                        {
                            continue;
                        }
                        let global = binding.ok_or_else(|| {
                            format!(
                                "spirv: resource argument {argument_index} of `{}` has no entry-rooted identity. Fail closed.",
                                signature.name(),
                            )
                        })?;
                        seeds.push((value, global));
                    }
                    let provenance = propagate_resource_identities(&graph, seeds)?;
                    let inst_set = function.inst_set();
                    for block in function.layout.iter_block() {
                        for instruction in function.layout.iter_inst(block) {
                            let Some(call) = <&Call as InstDowncast>::downcast(
                                inst_set,
                                function.dfg.inst(instruction),
                            ) else {
                                continue;
                            };
                            let callee_signature =
                                module.ctx.get_sig(*call.callee()).ok_or_else(|| {
                                    format!(
                                        "spirv: call from `{}` reaches {:?} without a signature. Fail closed.",
                                        signature.name(),
                                        call.callee(),
                                    )
                                })?;
                            let mut callee_bindings = vec![None; callee_signature.args().len()];
                            if call.args().len() != callee_bindings.len() {
                                return Err(format!(
                                    "spirv: call from `{}` to `{}` has {} arguments but {} resource-binding slots. Fail closed.",
                                    signature.name(),
                                    callee_signature.name(),
                                    call.args().len(),
                                    callee_bindings.len(),
                                ));
                            }
                            for (argument_index, ((&argument, &ty), binding)) in call
                                .args()
                                .iter()
                                .zip(callee_signature.args())
                                .zip(callee_bindings.iter_mut())
                                .enumerate()
                            {
                                if !resource_capabilities.contains(&ty)
                                    || !naga_argument_is_live(
                                        live_arguments,
                                        *call.callee(),
                                        argument_index,
                                    )
                                {
                                    continue;
                                }
                                let global = provenance.get(&argument).copied().ok_or_else(|| {
                                    format!(
                                        "spirv: call from `{}` to `{}` passes resource argument {argument_index} without a proven entry-rooted identity. Fail closed.",
                                        signature.name(),
                                        callee_signature.name(),
                                    )
                                })?;
                                *binding = Some(global);
                            }
                            let callee_variant =
                                variants.intern(*call.callee(), callee_bindings)?;
                            if let Some(previous) = variants
                                .calls
                                .insert((caller_variant, instruction), callee_variant)
                                && previous != callee_variant
                            {
                                return Err(format!(
                                    "spirv: call {instruction:?} from `{}` derives conflicting resource variants {previous:?} and {callee_variant:?}. Fail closed.",
                                    signature.name(),
                                ));
                            }
                        }
                    }
                    Ok(())
                })
                .ok_or_else(|| {
                    format!(
                        "spirv: reachable function `{}` has no body while deriving call-graph resource identities. Fail closed.",
                        signature.name(),
                    )
                })??;
        }
    }
    Ok(variants)
}

#[cfg(feature = "spirv-backend")]
fn helper_variant_call_sites(
    caller: NagaFunctionVariant,
    resource_variants: &NagaResourceVariants,
    lowered_variants: &std::collections::HashMap<
        NagaFunctionVariant,
        NagaFunctionInfo,
    >,
) -> Result<
    std::collections::HashMap<sonatina_ir::InstId, NagaFunctionInfo>,
    String,
> {
    let mut call_sites = std::collections::HashMap::new();
    for (&(candidate, instruction), &callee) in &resource_variants.calls {
        if candidate != caller {
            continue;
        }
        let info = lowered_variants.get(&callee).cloned().ok_or_else(|| {
            format!(
                "spirv: call {instruction:?} from function variant {caller:?} reaches Naga function variant {callee:?} before it is lowered. Fail closed."
            )
        })?;
        call_sites.insert(instruction, info);
    }
    Ok(call_sites)
}

#[cfg(feature = "spirv-backend")]
fn helper_private_memory_abis(
    module: &Module,
    call_order: &[sonatina_ir::module::FuncRef],
    entry: sonatina_ir::module::FuncRef,
) -> Result<
    std::collections::HashMap<sonatina_ir::module::FuncRef, NagaMemoryAbi>,
    String,
> {
    use sonatina_ir::{
        InstDowncast,
        inst::{control_flow, data},
    };

    // `call_order` is postorder, so every direct callee's ABI is known before
    // its caller. A helper borrows only the capabilities it uses locally or
    // must forward transitively. This keeps pure arithmetic helpers pure.
    let mut abis = std::collections::HashMap::<
        sonatina_ir::module::FuncRef,
        NagaMemoryAbi,
    >::new();
    for &function_ref in call_order {
        if function_ref == entry {
            continue;
        }
        let abi = module
            .func_store
            .try_view(function_ref, |function| {
                let inst_set = function.inst_set();
                let mut abi = NagaMemoryAbi::default();
                for block in function.layout.iter_block() {
                    for instruction in function.layout.iter_inst(block) {
                        let instruction_data = function.dfg.inst(instruction);
                        if let Some(call) = <&control_flow::Call as InstDowncast>::downcast(
                            inst_set,
                            function.dfg.inst(instruction),
                        ) {
                            let callee = abis.get(&call.callee()).copied().ok_or_else(|| {
                                format!(
                                    "spirv: helper {function_ref:?} reaches callee {:?} before its private-memory ABI is available. Fail closed.",
                                    call.callee(),
                                )
                            })?;
                            abi.heap |= callee.heap;
                            abi.trap |= callee.trap;
                            continue;
                        }
                        let byte_arena_access = <&data::Mload as InstDowncast>::downcast(
                            inst_set,
                            instruction_data,
                        )
                        .is_some_and(|load| {
                            !function
                                .dfg
                                .value_ty(*load.addr())
                                .is_pointer(&module.ctx)
                        }) || <&data::Mstore as InstDowncast>::downcast(
                            inst_set,
                            instruction_data,
                        )
                        .is_some_and(|store| {
                            !function
                                .dfg
                                .value_ty(*store.addr())
                                .is_pointer(&module.ctx)
                        });
                        if byte_arena_access {
                            abi.heap = true;
                            abi.trap = true;
                        }
                        if <&control_flow::Unreachable as InstDowncast>::downcast(
                            inst_set,
                            instruction_data,
                        )
                        .is_some()
                        {
                            abi.trap = true;
                        }
                    }
                }
                Ok::<_, String>(abi)
            })
            .ok_or_else(|| {
                format!(
                    "spirv: reachable helper {function_ref:?} has no body while deriving its private-memory ABI. Fail closed."
                )
            })??;
        abis.insert(function_ref, abi);
    }
    Ok(abis)
}

#[cfg(feature = "spirv-backend")]
fn helper_naga_type(
    ctx: &sonatina_ir::module::ModuleCtx,
    ty: sonatina_ir::Type,
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    typed_local_types: &std::collections::HashMap<
        sonatina_ir::Type,
        NagaTypedLocalType,
    >,
) -> Result<naga::Handle<naga::Type>, String> {
    match ty {
        sonatina_ir::Type::I1 => Ok(bool_type),
        sonatina_ir::Type::I32 if word == WordKind::U32 => Ok(word_type),
        sonatina_ir::Type::I64 if word == WordKind::I64 => Ok(word_type),
        sonatina_ir::Type::F32 if word == WordKind::U32 => Ok(f32_type),
        other if other.resolve_compound(ctx).is_some_and(|compound| {
            matches!(
                compound,
                sonatina_ir::types::CompoundType::Ptr(_)
                    | sonatina_ir::types::CompoundType::Array { .. }
                    | sonatina_ir::types::CompoundType::Struct(_)
            )
        }) => {
            typed_local_types
                .get(&other)
                .map(|mapped| mapped.handle)
                .ok_or_else(|| {
                    format!(
                        "spirv: helper typed-local ABI type {other:?} has no prevalidated representation. Fail closed."
                    )
                })
        }
        other => Err(format!(
            "spirv: helper ABI type {other:?} is unsupported under the {word:?} word. Fail closed."
        )),
    }
}

#[cfg(feature = "spirv-backend")]
fn verify_naga_typed_local_use_closure(
    module: &Module,
    function_ref: sonatina_ir::module::FuncRef,
) -> Result<NagaTypedLocalUseClosure, String> {
    module
        .func_store
        .try_view(function_ref, |function| {
            use sonatina_ir::InstDowncast;

            let inst_set = function.inst_set();
            let signature = module
                .ctx
                .get_sig(function_ref)
                .ok_or_else(|| format!("spirv: reachable function {function_ref:?} has no signature. Fail closed."))?;
            if signature.ret_tys().iter().copied().any(|ty| {
                matches!(
                    ty.resolve_compound(function.ctx()),
                    Some(sonatina_ir::types::CompoundType::Ptr(_))
                )
            }) {
                return Err(format!(
                    "spirv: typed-local pointer escapes through the result of helper `{}`. Fail closed.",
                    signature.name(),
                ));
            }
            let mut closure = NagaTypedLocalUseClosure::default();
            let mut local_pointers = std::collections::HashSet::new();

            for (&argument, &ty) in function.arg_values.iter().zip(signature.args()) {
                if matches!(
                    ty.resolve_compound(function.ctx()),
                    Some(sonatina_ir::types::CompoundType::Ptr(_))
                ) {
                    local_pointers.insert(argument);
                    closure.borrowed_pointer_types.push(ty);
                }
            }

            for block in function.layout.iter_block() {
                for instruction in function.layout.iter_inst(block) {
                    let instruction_data = function.dfg.inst(instruction);
                    let Some(alloca) =
                        <&sonatina_ir::inst::data::Alloca as InstDowncast>::downcast(
                            inst_set,
                            instruction_data,
                        )
                    else {
                        continue;
                    };
                    let Some(result) = function.dfg.inst_result(instruction) else {
                        return Err(
                            "spirv: typed-local Alloca has no result. Fail closed."
                                .to_string(),
                        );
                    };
                    let result_ty = function.dfg.value_ty(result);
                    let result_pointee = match result_ty.resolve_compound(function.ctx()) {
                        Some(sonatina_ir::types::CompoundType::Ptr(pointee)) => pointee,
                        _ => {
                            return Err(format!(
                                "spirv: typed-local Alloca result has non-pointer type {result_ty:?}. Fail closed."
                            ));
                        }
                    };
                    if result_pointee != *alloca.ty() {
                        return Err(format!(
                            "spirv: typed-local Alloca result points to {result_pointee:?}, not its declared type {:?}. Fail closed.",
                            alloca.ty(),
                        ));
                    }
                    closure.allocation_types.push(*alloca.ty());
                    local_pointers.insert(result);
                }
            }

            // Discover the complete projection closure before validating uses,
            // so layout order does not affect legality.
            loop {
                let mut changed = false;
                for block in function.layout.iter_block() {
                    for instruction in function.layout.iter_inst(block) {
                        let instruction_data = function.dfg.inst(instruction);
                        let Some(gep) =
                            <&sonatina_ir::inst::data::Gep as InstDowncast>::downcast(
                                inst_set,
                                instruction_data,
                            )
                        else {
                            if let Some(bitcast) =
                                <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(
                                    inst_set,
                                    instruction_data,
                                )
                                && local_pointers.contains(bitcast.from())
                                && typed_local_zero_projection_path(
                                    function.ctx(),
                                    function.dfg.value_ty(*bitcast.from()),
                                    *bitcast.ty(),
                                )
                                .is_some()
                            {
                                let Some(result) = function.dfg.inst_result(instruction) else {
                                    return Err(
                                        "spirv: typed-local structural Bitcast has no result. Fail closed."
                                            .to_string(),
                                    );
                                };
                                changed |= local_pointers.insert(result);
                            }
                            continue;
                        };
                        let Some(&base) = gep.values().first() else {
                            continue;
                        };
                        if local_pointers.contains(&base) {
                            let Some(result) = function.dfg.inst_result(instruction) else {
                                return Err(
                                    "spirv: typed-local Gep has no result. Fail closed."
                                        .to_string(),
                                );
                            };
                            changed |= local_pointers.insert(result);
                        }
                    }
                }
                if !changed {
                    break;
                }
            }

            for block in function.layout.iter_block() {
                for instruction in function.layout.iter_inst(block) {
                    let instruction_data = function.dfg.inst(instruction);
                    if let Some(gep) =
                        <&sonatina_ir::inst::data::Gep as InstDowncast>::downcast(
                            inst_set,
                            instruction_data,
                        )
                    {
                        let Some((&base, indices)) = gep.values().split_first() else {
                            return Err(
                                "spirv: typed-local Gep has no base pointer. Fail closed."
                                    .to_string(),
                            );
                        };
                        if !local_pointers.contains(&base) {
                            return Err(format!(
                                "spirv: Gep base {base:?} is not rooted in typed private storage. Fail closed."
                            ));
                        }
                        if indices
                            .iter()
                            .any(|index| local_pointers.contains(index))
                        {
                            return Err(
                                "spirv: typed-local pointer used as a Gep index. Fail closed."
                                    .to_string(),
                            );
                        }
                        continue;
                    }

                    if let Some(bitcast) =
                        <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(
                            inst_set,
                            instruction_data,
                        )
                    {
                        let source_is_local = local_pointers.contains(bitcast.from());
                        let structural = typed_local_zero_projection_path(
                            function.ctx(),
                            function.dfg.value_ty(*bitcast.from()),
                            *bitcast.ty(),
                        )
                        .is_some();
                        if source_is_local != structural {
                            return Err(format!(
                                "spirv: typed-local pointer Bitcast {:?} is not an exact zero projection. Fail closed.",
                                bitcast.from(),
                            ));
                        }
                        if source_is_local {
                            continue;
                        }
                    }

                    if let Some(load) =
                        <&sonatina_ir::inst::data::Mload as InstDowncast>::downcast(
                            inst_set,
                            instruction_data,
                        )
                    {
                        if function.dfg.value_ty(*load.addr()).is_pointer(function.ctx())
                            && !local_pointers.contains(load.addr())
                        {
                            return Err(format!(
                                "spirv: Mload pointer {:?} is not rooted in typed private storage. Fail closed.",
                                load.addr(),
                            ));
                        }
                        continue;
                    }

                    if let Some(store) =
                        <&sonatina_ir::inst::data::Mstore as InstDowncast>::downcast(
                            inst_set,
                            instruction_data,
                        )
                    {
                        if function.dfg.value_ty(*store.addr()).is_pointer(function.ctx())
                            && !local_pointers.contains(store.addr())
                        {
                            return Err(format!(
                                "spirv: Mstore pointer {:?} is not rooted in typed private storage. Fail closed.",
                                store.addr(),
                            ));
                        }
                        if local_pointers.contains(store.value()) {
                            return Err(
                                "spirv: typed-local pointer stored as data. Fail closed."
                                    .to_string(),
                            );
                        }
                        continue;
                    }

                    if let Some(call) =
                        <&sonatina_ir::inst::control_flow::Call as InstDowncast>::downcast(
                            inst_set,
                            instruction_data,
                        )
                    {
                        let callee = module.ctx.get_sig(*call.callee()).ok_or_else(|| {
                            format!(
                                "spirv: typed-local pointer reaches undeclared callee {:?}. Fail closed.",
                                call.callee(),
                            )
                        })?;
                        if call.args().len() != callee.args().len() {
                            return Err(format!(
                                "spirv: typed-local pointer call to `{}` has an inconsistent arity. Fail closed.",
                                callee.name(),
                            ));
                        }
                        for (&argument, &parameter_ty) in
                            call.args().iter().zip(callee.args())
                        {
                            let argument_is_local = local_pointers.contains(&argument);
                            let argument_ty = function.dfg.value_ty(argument);
                            if argument_is_local {
                                if argument_ty != parameter_ty
                                    || !matches!(
                                        parameter_ty.resolve_compound(function.ctx()),
                                        Some(sonatina_ir::types::CompoundType::Ptr(_))
                                    )
                                {
                                    return Err(format!(
                                        "spirv: typed-local pointer {argument:?} crosses call `{}` through incompatible parameter type {parameter_ty:?}. Fail closed.",
                                        callee.name(),
                                    ));
                                }
                            } else if matches!(
                                argument_ty.resolve_compound(function.ctx()),
                                Some(sonatina_ir::types::CompoundType::Ptr(_))
                            ) {
                                return Err(format!(
                                    "spirv: call `{}` receives pointer {argument:?} outside the verified typed-local closure. Fail closed.",
                                    callee.name(),
                                ));
                            }
                        }
                        continue;
                    }

                    if let Some(pointer) = instruction_data
                        .collect_values()
                        .into_iter()
                        .find(|value| local_pointers.contains(value))
                    {
                        return Err(format!(
                            "spirv: typed-local pointer {pointer:?} escapes through {}. Fail closed.",
                            instruction_data.as_text(),
                        ));
                    }
                }
            }

            Ok(closure)
        })
        .ok_or_else(|| {
            format!(
                "spirv: reachable function {function_ref:?} has no body while verifying typed locals. Fail closed."
            )
        })?
}

#[cfg(feature = "spirv-backend")]
fn intern_naga_typed_local_type(
    module: &Module,
    ty: sonatina_ir::Type,
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    types: &mut naga::UniqueArena<naga::Type>,
    cache: &mut std::collections::HashMap<sonatina_ir::Type, NagaTypedLocalType>,
) -> Result<NagaTypedLocalType, String> {
    if let Some(&cached) = cache.get(&ty) {
        return Ok(cached);
    }

    let mapped = match ty {
        sonatina_ir::Type::I1 => NagaTypedLocalType {
            handle: bool_type,
            alignment: 4,
            size: 4,
        },
        sonatina_ir::Type::I32 if word == WordKind::U32 => NagaTypedLocalType {
            handle: word_type,
            alignment: 4,
            size: 4,
        },
        sonatina_ir::Type::I64 if word == WordKind::I64 => NagaTypedLocalType {
            handle: word_type,
            alignment: 8,
            size: 8,
        },
        sonatina_ir::Type::F32 if word == WordKind::U32 => NagaTypedLocalType {
            handle: f32_type,
            alignment: 4,
            size: 4,
        },
        sonatina_ir::Type::Compound(compound_ref) => {
            let compound = module
                .ctx
                .with_ty_store(|store| store.resolve_compound(compound_ref).clone());
            match compound {
                sonatina_ir::types::CompoundType::Ptr(pointee) => {
                    let pointee = intern_naga_typed_local_type(
                        module,
                        pointee,
                        word,
                        word_type,
                        f32_type,
                        bool_type,
                        types,
                        cache,
                    )?;
                    let handle = types.insert(
                        naga::Type {
                            name: None,
                            inner: naga::TypeInner::Pointer {
                                base: pointee.handle,
                                space: naga::AddressSpace::Function,
                            },
                        },
                        naga::Span::UNDEFINED,
                    );
                    NagaTypedLocalType {
                        handle,
                        alignment: 4,
                        size: 4,
                    }
                }
                sonatina_ir::types::CompoundType::Array { elem, len } => {
                    let elem = intern_naga_typed_local_type(
                        module, elem, word, word_type, f32_type, bool_type, types, cache,
                    )?;
                    let len = u32::try_from(len).map_err(|_| {
                        format!(
                            "spirv: typed local array length does not fit u32 for {ty:?}. Fail closed."
                        )
                    })?;
                    let len = std::num::NonZeroU32::new(len).ok_or_else(|| {
                        format!(
                            "spirv: zero-length typed local array {ty:?} is unsupported. Fail closed."
                        )
                    })?;
                    let stride = align_helper_offset(elem.size, elem.alignment);
                    let size = stride.checked_mul(len.get()).ok_or_else(|| {
                        format!(
                            "spirv: typed local array size overflows u32 for {ty:?}. Fail closed."
                        )
                    })?;
                    let handle = types.insert(
                        naga::Type {
                            name: None,
                            inner: naga::TypeInner::Array {
                                base: elem.handle,
                                size: naga::ArraySize::Constant(len),
                                stride,
                            },
                        },
                        naga::Span::UNDEFINED,
                    );
                    NagaTypedLocalType {
                        handle,
                        alignment: elem.alignment,
                        size,
                    }
                }
                sonatina_ir::types::CompoundType::Struct(data) if !data.packed => {
                    let mut members = Vec::with_capacity(data.fields.len());
                    let mut offset = 0u32;
                    let mut alignment = 1u32;
                    for (index, field_ty) in data.fields.iter().copied().enumerate() {
                        let field = intern_naga_typed_local_type(
                            module,
                            field_ty,
                            word,
                            word_type,
                            f32_type,
                            bool_type,
                            types,
                            cache,
                        )?;
                        offset = align_helper_offset(offset, field.alignment);
                        members.push(naga::StructMember {
                            name: Some(format!("f{index}")),
                            ty: field.handle,
                            binding: None,
                            offset,
                        });
                        offset = offset.checked_add(field.size).ok_or_else(|| {
                            format!(
                                "spirv: typed local struct size overflows u32 for {ty:?}. Fail closed."
                            )
                        })?;
                        alignment = alignment.max(field.alignment);
                    }
                    let size = align_helper_offset(offset, alignment);
                    let handle = types.insert(
                        naga::Type {
                            name: Some(format!("FeLocal_{}", data.name)),
                            inner: naga::TypeInner::Struct {
                                members,
                                span: size,
                            },
                        },
                        naga::Span::UNDEFINED,
                    );
                    NagaTypedLocalType {
                        handle,
                        alignment,
                        size,
                    }
                }
                other => {
                    return Err(format!(
                        "spirv: typed local type {ty:?} has unsupported shape {other:?}. Fail closed."
                    ));
                }
            }
        }
        other => {
            return Err(format!(
                "spirv: typed local type {other:?} is unsupported under the {word:?} word. Fail closed."
            ));
        }
    };
    cache.insert(ty, mapped);
    Ok(mapped)
}

#[cfg(feature = "spirv-backend")]
fn collect_naga_typed_local_types(
    module: &Module,
    call_order: &[sonatina_ir::module::FuncRef],
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    types: &mut naga::UniqueArena<naga::Type>,
) -> Result<
    std::collections::HashMap<sonatina_ir::Type, NagaTypedLocalType>,
    String,
> {
    let mut cache = std::collections::HashMap::new();
    for &function_ref in call_order {
        let mut private_bytes = 0u32;
        let mut private_allocations = Vec::new();
        let closure = verify_naga_typed_local_use_closure(module, function_ref)?;
        for ty in closure.borrowed_pointer_types {
            intern_naga_typed_local_type(
                module,
                ty,
                word,
                word_type,
                f32_type,
                bool_type,
                types,
                &mut cache,
            )?;
        }
        for ty in closure.allocation_types {
            let local = intern_naga_typed_local_type(
                module,
                ty,
                word,
                word_type,
                f32_type,
                bool_type,
                types,
                &mut cache,
            )?;
            private_bytes = private_bytes.checked_add(local.size).ok_or_else(|| {
                format!(
                    "spirv: typed private storage size overflows u32 in {function_ref:?}. Fail closed."
                )
            })?;
            let name = types[local.handle]
                .name
                .clone()
                .unwrap_or_else(|| format!("{ty:?}"));
            private_allocations.push((name, local.size));
        }
        if private_bytes > MAX_NAGA_TYPED_PRIVATE_BYTES_PER_FUNCTION {
            private_allocations.sort_by(|left, right| {
                right
                    .1
                    .cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let largest = private_allocations
                .iter()
                .take(8)
                .map(|(name, size)| format!("{name}={size}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "spirv: typed private storage in {function_ref:?} requires {private_bytes} bytes across {} allocations, over the conservative {MAX_NAGA_TYPED_PRIVATE_BYTES_PER_FUNCTION}-byte per-function budget; largest allocations: {largest}. Fail closed.",
                private_allocations.len(),
            ));
        }
    }
    Ok(cache)
}

#[cfg(feature = "spirv-backend")]
fn helper_naga_argument_abi(
    module: &Module,
    function_ref: sonatina_ir::module::FuncRef,
    signature: &sonatina_ir::Signature,
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    resource_capabilities: &NagaResourceCapabilities,
    resource_bindings: &[Option<naga::Handle<naga::GlobalVariable>>],
    live_arguments: &NagaLiveArguments,
    naga_functions: &NagaFunctionMap,
) -> Result<Vec<NagaArgumentSource>, String> {
    if resource_bindings.len() != signature.args().len() {
        return Err(format!(
            "spirv: helper `{}` has {} call-graph resource bindings for {} arguments. Fail closed.",
            signature.name(),
            resource_bindings.len(),
            signature.args().len(),
        ));
    }
    let mut physical_index = 0u32;
    signature
        .args()
        .iter()
        .copied()
        .zip(resource_bindings)
        .enumerate()
        .map(|(logical_index, (ty, binding))| {
            if resource_capabilities.contains(&ty) {
                match binding {
                    Some(global) => Ok(NagaArgumentSource::ImplicitResource(*global)),
                    None if !naga_argument_is_live(
                        live_arguments,
                        function_ref,
                        logical_index,
                    ) => Ok(NagaArgumentSource::Dead),
                    None => Err(
                        format!(
                            "spirv: helper `{}` resource argument {logical_index} has no call-graph identity. Fail closed.",
                            signature.name(),
                        )
                    ),
                }
            } else {
                helper_naga_type(
                    &module.ctx,
                    ty,
                    word,
                    word_type,
                    f32_type,
                    bool_type,
                    &naga_functions.typed_local_types,
                )?;
                let source = NagaArgumentSource::Physical(physical_index);
                physical_index += 1;
                Ok(source)
            }
        })
        .collect()
}

#[cfg(feature = "spirv-backend")]
#[allow(clippy::too_many_arguments)]
fn pack_wide_naga_helper_arguments(
    module: &Module,
    signature: &sonatina_ir::Signature,
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    memory_abi: NagaMemoryAbi,
    argument_abi: &mut [NagaArgumentSource],
    naga_functions: &NagaFunctionMap,
    types: &mut naga::UniqueArena<naga::Type>,
) -> Result<Option<NagaPackedArguments>, String> {
    let memory_parameters = usize::from(memory_abi.heap) * 2 + usize::from(memory_abi.trap);
    let physical_parameters = argument_abi
        .iter()
        .filter(|source| matches!(source, NagaArgumentSource::Physical(_)))
        .count();
    if physical_parameters + memory_parameters <= MAX_WGSL_FUNCTION_PARAMETERS {
        return Ok(None);
    }

    struct PackedGroupBuilder {
        element_ty: naga::Handle<naga::Type>,
        alignment: u32,
        stride: u32,
        logical_indices: Vec<usize>,
    }

    let mut groups = Vec::<PackedGroupBuilder>::new();
    let mut packed_locations = vec![None; signature.args().len()];
    for (logical_index, (&ty, source)) in
        signature.args().iter().zip(argument_abi.iter()).enumerate()
    {
        if !matches!(source, NagaArgumentSource::Physical(_)) {
            continue;
        }
        let (alignment, stride) = match ty {
            sonatina_ir::Type::I1 => (4, 4),
            sonatina_ir::Type::I32 | sonatina_ir::Type::F32
                if word == WordKind::U32 => (4, 4),
            sonatina_ir::Type::I64 if word == WordKind::I64 => (8, 8),
            _ => continue,
        };
        let element_ty = helper_naga_type(
            &module.ctx,
            ty,
            word,
            word_type,
            f32_type,
            bool_type,
            &naga_functions.typed_local_types,
        )?;
        let group_index = if let Some(index) = groups
            .iter()
            .position(|group| group.element_ty == element_ty)
        {
            index
        } else {
            groups.push(PackedGroupBuilder {
                element_ty,
                alignment,
                stride,
                logical_indices: Vec::new(),
            });
            groups.len() - 1
        };
        let member_index = groups[group_index].logical_indices.len();
        groups[group_index].logical_indices.push(logical_index);
        packed_locations[logical_index] = Some((group_index as u32, member_index as u32));
    }
    let packed_argument_count = packed_locations.iter().flatten().count();
    if packed_argument_count < 2 {
        return Err(format!(
            "spirv: helper `{}` requires {} physical parameters, over the portable WGSL limit of {MAX_WGSL_FUNCTION_PARAMETERS}, but its ABI has fewer than two packable scalar values. Fail closed.",
            signature.name(),
            physical_parameters + memory_parameters,
        ));
    }

    let direct_parameters = physical_parameters - packed_argument_count;
    let packed_physical_parameters = 1 + direct_parameters + memory_parameters;
    if packed_physical_parameters > MAX_WGSL_FUNCTION_PARAMETERS {
        return Err(format!(
            "spirv: helper `{}` still requires {packed_physical_parameters} physical parameters after scalar argument packing, over the portable WGSL limit of {MAX_WGSL_FUNCTION_PARAMETERS}. Fail closed.",
            signature.name(),
        ));
    }

    let physical_index = 0u32;
    let mut next_direct_index = 1u32;
    for (logical_index, source) in argument_abi.iter_mut().enumerate() {
        if !matches!(source, NagaArgumentSource::Physical(_)) {
            continue;
        }
        if let Some((group_index, member_index)) = packed_locations[logical_index] {
            *source = NagaArgumentSource::Packed {
                physical_index,
                group_index,
                member_index,
            };
        } else {
            *source = NagaArgumentSource::Physical(next_direct_index);
            next_direct_index += 1;
        }
    }

    let mut members = Vec::with_capacity(groups.len());
    let mut packed_groups = Vec::with_capacity(groups.len());
    let mut offset = 0u32;
    let mut struct_alignment = 1u32;
    for (group_index, group) in groups.into_iter().enumerate() {
        let member_count = u32::try_from(group.logical_indices.len()).map_err(|_| {
            format!(
                "spirv: packed helper argument group is too large in `{}`. Fail closed.",
                signature.name(),
            )
        })?;
        let member_count = std::num::NonZeroU32::new(member_count).ok_or_else(|| {
            format!(
                "spirv: packed helper argument group is empty in `{}`. Fail closed.",
                signature.name(),
            )
        })?;
        let group_span = group
            .stride
            .checked_mul(member_count.get())
            .ok_or_else(|| {
                format!(
                    "spirv: packed helper argument array overflows u32 in `{}`. Fail closed.",
                    signature.name(),
                )
            })?;
        let group_ty = types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base: group.element_ty,
                    size: naga::ArraySize::Constant(member_count),
                    stride: group.stride,
                },
            },
            naga::Span::UNDEFINED,
        );
        offset = align_helper_offset(offset, group.alignment);
        members.push(naga::StructMember {
            name: Some(format!("g{group_index}")),
            ty: group_ty,
            binding: None,
            offset,
        });
        offset = offset.checked_add(group_span).ok_or_else(|| {
            format!(
                "spirv: packed helper argument layout overflows u32 in `{}`. Fail closed.",
                signature.name(),
            )
        })?;
        struct_alignment = struct_alignment.max(group.alignment);
        packed_groups.push(NagaPackedArgumentGroup {
            ty: group_ty,
            member_count: member_count.get() as usize,
        });
    }
    let span = align_helper_offset(offset, struct_alignment);
    let ty = types.insert(
        naga::Type {
            name: Some(format!("{}_arguments", signature.name())),
            inner: naga::TypeInner::Struct { members, span },
        },
        naga::Span::UNDEFINED,
    );
    Ok(Some(NagaPackedArguments {
        ty,
        physical_index,
        groups: packed_groups,
    }))
}

#[cfg(feature = "spirv-backend")]
fn helper_resource_capabilities(
    module: &Module,
    entry_signature: &sonatina_ir::Signature,
    external_roots: &[(u32, naga::Handle<naga::GlobalVariable>)],
) -> Result<NagaResourceCapabilities, String> {
    let mut capabilities = NagaResourceCapabilities::new();
    for &(argument_index, _) in external_roots {
        let logical_type = *entry_signature
            .args()
            .get(argument_index as usize)
            .ok_or_else(|| {
                format!(
                    "spirv: external resource argument {argument_index} disappeared while deriving helper capabilities. Fail closed."
                )
            })?;
        let Some(sonatina_ir::types::CompoundType::ObjRef(referent)) =
            logical_type.resolve_compound(&module.ctx)
        else {
            return Err(format!(
                "spirv: external resource argument {argument_index} has non-object type {logical_type:?}. Fail closed."
            ));
        };
        if !matches!(
            referent.resolve_compound(&module.ctx),
            Some(sonatina_ir::types::CompoundType::Array { .. })
        ) {
            return Err(format!(
                "spirv: external resource argument {argument_index} must root an object array, got {logical_type:?}. Fail closed."
            ));
        }
        capabilities.insert(logical_type);
    }
    Ok(capabilities)
}

#[cfg(feature = "spirv-backend")]
fn helper_naga_logical_result_abi(
    module: &Module,
    function_ref: sonatina_ir::module::FuncRef,
    resource_capabilities: &NagaResourceCapabilities,
    logical_result_abis: &NagaLogicalResultAbis,
) -> Result<Vec<NagaResultSource>, String> {
    use sonatina_ir::{InstDowncast, inst::control_flow};

    let signature = module
        .ctx
        .get_sig(function_ref)
        .ok_or_else(|| format!("spirv: helper {function_ref:?} has no signature"))?;
    let mut logical = signature
        .ret_tys()
        .iter()
        .scan(0u32, |physical, ty| {
            if resource_capabilities.contains(ty) {
                Some(None)
            } else {
                let source = NagaResultSource::Physical(*physical);
                *physical += 1;
                Some(Some(source))
            }
        })
        .collect::<Vec<_>>();
    if logical.iter().all(Option::is_some) {
        return Ok(logical.into_iter().flatten().collect());
    }

    let sources = module
        .func_store
        .try_view(function_ref, |function| -> Result<Vec<u32>, String> {
            let inst_set = function.inst_set();
            let graph = resource_identity_graph(
                function,
                resource_capabilities,
                logical_result_abis,
            )?;
            let seeds = function
                .arg_values
                .iter()
                .zip(signature.args())
                .enumerate()
                .filter_map(|(argument_index, (&value, &ty))| {
                    resource_capabilities
                        .contains(&ty)
                        .then_some((value, argument_index as u32))
                });
            let provenance = propagate_resource_identities(&graph, seeds)?;

            let return_sites = function
                .layout
                .iter_block()
                .flat_map(|block| function.layout.iter_inst(block))
                .filter_map(|instruction| {
                    <&control_flow::Return as InstDowncast>::downcast(
                        inst_set,
                        function.dfg.inst(instruction),
                    )
                    .map(|return_| return_.args().as_slice().to_vec())
                })
                .collect::<Vec<_>>();
            if return_sites.is_empty() {
                return Err(format!(
                    "spirv: resource-carrying helper `{}` has no return site. Fail closed.",
                    signature.name(),
                ));
            }
            for return_values in &return_sites {
                if return_values.len() != signature.ret_tys().len() {
                    return Err(format!(
                        "spirv: resource-carrying helper `{}` returns {} values at one exit but declares {}. Fail closed.",
                        signature.name(),
                        return_values.len(),
                        signature.ret_tys().len(),
                    ));
                }
            }
            signature
                .ret_tys()
                .iter()
                .enumerate()
                .filter(|(_, ty)| resource_capabilities.contains(ty))
                .map(|(logical_index, _)| {
                    let mut source = None;
                    for return_values in &return_sites {
                        let value = return_values[logical_index];
                        let candidate = provenance.get(&value).copied().ok_or_else(|| {
                            format!(
                                "spirv: resource-carrying helper `{}` returns {value:?} without a proven argument identity. Fail closed.",
                                signature.name(),
                            )
                        })?;
                        match source {
                            None => source = Some(candidate),
                            Some(previous) if previous == candidate => {}
                            Some(previous) => {
                                return Err(format!(
                                    "spirv: resource-carrying helper `{}` returns different argument identities {previous} and {candidate} across exits. Fail closed.",
                                    signature.name(),
                                ));
                            }
                        }
                    }
                    source.ok_or_else(|| {
                        format!(
                            "spirv: resource-carrying helper `{}` has no resource return evidence. Fail closed.",
                            signature.name(),
                        )
                    })
                })
                .collect()
        })
        .ok_or_else(|| {
            format!(
                "spirv: resource-carrying helper {function_ref:?} has no body. Fail closed."
            )
        })??;
    let mut sources = sources.into_iter();
    for source in &mut logical {
        if source.is_none() {
            *source = Some(NagaResultSource::PassthroughArgument(
                sources.next().expect("one source per resource result"),
            ));
        }
    }
    Ok(logical.into_iter().flatten().collect())
}

#[cfg(feature = "spirv-backend")]
fn helper_naga_result_abi(
    module: &Module,
    function_ref: sonatina_ir::module::FuncRef,
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    resource_capabilities: &NagaResourceCapabilities,
    logical_result_abis: &NagaLogicalResultAbis,
    naga_functions: &NagaFunctionMap,
    naga_types: &mut naga::UniqueArena<naga::Type>,
) -> Result<NagaResultAbi, String> {
    let signature = module
        .ctx
        .get_sig(function_ref)
        .ok_or_else(|| format!("spirv: helper {function_ref:?} has no signature"))?;
    let physical_types = signature
        .ret_tys()
        .iter()
        .copied()
        .filter(|ty| !resource_capabilities.contains(ty))
        .collect::<Vec<_>>();
    for &ty in &physical_types {
        helper_naga_type(
            &module.ctx,
            ty,
            word,
            word_type,
            f32_type,
            bool_type,
            &naga_functions.typed_local_types,
        )?;
    }
    let physical_type = helper_naga_result_type(
        module,
        signature.name(),
        &physical_types,
        word,
        word_type,
        f32_type,
        bool_type,
        naga_functions,
        naga_types,
    )?;

    let logical = logical_result_abis.get(&function_ref).cloned().ok_or_else(|| {
        format!(
            "spirv: helper `{}` has no derived logical result ABI. Fail closed.",
            signature.name(),
        )
    })?;
    Ok(NagaResultAbi {
        logical,
        physical_type,
        physical_arity: physical_types.len(),
    })
}

#[cfg(feature = "spirv-backend")]
fn align_helper_offset(offset: u32, alignment: u32) -> u32 {
    offset.div_ceil(alignment) * alignment
}

/// Represent a scalar multi-result helper as one logical WGSL result struct.
/// The struct is function-local ABI only: it never crosses a storage, uniform,
/// arena, or host boundary.
#[cfg(feature = "spirv-backend")]
fn helper_naga_result_type(
    module: &Module,
    name: &str,
    return_types: &[sonatina_ir::Type],
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    naga_functions: &NagaFunctionMap,
    types: &mut naga::UniqueArena<naga::Type>,
) -> Result<Option<naga::Handle<naga::Type>>, String> {
    if return_types.is_empty() {
        return Ok(None);
    }
    if let [ty] = return_types {
        return helper_naga_type(
            &module.ctx,
            *ty,
            word,
            word_type,
            f32_type,
            bool_type,
            &naga_functions.typed_local_types,
        )
        .map(Some);
    }

    let mut members = Vec::with_capacity(return_types.len());
    let mut offset = 0u32;
    let mut struct_alignment = 1u32;
    for (index, &ty) in return_types.iter().enumerate() {
        let (alignment, size) = match ty {
            sonatina_ir::Type::I1 => (1, 1),
            sonatina_ir::Type::I32 | sonatina_ir::Type::F32
                if word == WordKind::U32 => (4, 4),
            sonatina_ir::Type::I64 if word == WordKind::I64 => (8, 8),
            other => {
                return Err(format!(
                    "spirv: helper `{name}` has unsupported multi-result ABI type {other:?}. Fail closed."
                ));
            }
        };
        offset = align_helper_offset(offset, alignment);
        members.push(naga::StructMember {
            name: Some(format!("r{index}")),
            ty: helper_naga_type(
                &module.ctx,
                ty,
                word,
                word_type,
                f32_type,
                bool_type,
                &naga_functions.typed_local_types,
            )?,
            binding: None,
            offset,
        });
        offset += size;
        struct_alignment = struct_alignment.max(alignment);
    }
    let span = align_helper_offset(offset, struct_alignment);
    Ok(Some(types.insert(
        naga::Type {
            name: Some(format!("{name}_result")),
            inner: naga::TypeInner::Struct { members, span },
        },
        naga::Span::UNDEFINED,
    )))
}

/// Lower one admitted helper as a real Naga function. External resources and
/// the private arena cross calls only through compiler-derived capabilities.
/// WGSL module resources are implicit helper capabilities because storage
/// pointers are not legal function arguments. Resource identity results are
/// erased only after the result ABI proves that they pass through an input
/// unchanged.
#[cfg(feature = "spirv-backend")]
fn lower_naga_helper(
    module: &Module,
    function_ref: sonatina_ir::module::FuncRef,
    body_plan: &HelperBodyPlan,
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    argument_abi: &[NagaArgumentSource],
    packed_arguments: Option<&NagaPackedArguments>,
    result_abi: &NagaResultAbi,
    memory_abi: NagaMemoryAbi,
    parameters: helper_plan::PhysicalHelperParameters,
    heap_words: u32,
    resource_capabilities: &NagaResourceCapabilities,
    logical_result_abis: &NagaLogicalResultAbis,
    naga_functions: &NagaFunctionMap,
) -> Result<naga::Function, String> {
    let signature = module
        .ctx
        .get_sig(function_ref)
        .ok_or_else(|| format!("spirv: helper {function_ref:?} has no signature"))?;
    let helper_plan::PhysicalHelperParameters {
        arguments,
        explicit_count: physical_argument_count,
    } = parameters;
    let result = result_abi
        .physical_type
        .map(|ty| naga::FunctionResult { ty, binding: None });
    let mut naga_function = naga::Function {
        name: Some(signature.name().to_string()),
        arguments,
        result,
        local_variables: naga::Arena::new(),
        expressions: naga::Arena::new(),
        named_expressions: Default::default(),
        body: naga::Block::new(),
        diagnostic_filter_leaf: None,
    };

    let mut lowering_error = None;
    module.func_store.try_view(function_ref, |function| {
        let inst_set = function.inst_set();
        let mut value_map = std::collections::HashMap::new();
        for (&argument, source) in function.arg_values.iter().zip(argument_abi) {
            let mut packed_emit_start = None;
            let expression = match *source {
                NagaArgumentSource::Physical(index) => {
                    naga::Expression::FunctionArgument(index)
                }
                NagaArgumentSource::Packed {
                    physical_index,
                    group_index,
                    member_index,
                } => {
                    let Some(packed) = packed_arguments else {
                        lowering_error = Some(format!(
                            "spirv: helper `{}` has a packed source without a packed ABI. Fail closed.",
                            signature.name(),
                        ));
                        return;
                    };
                    if packed.physical_index != physical_index
                        || group_index as usize >= packed.groups.len()
                    {
                        lowering_error = Some(format!(
                            "spirv: helper `{}` has an invalid packed argument location. Fail closed.",
                            signature.name(),
                        ));
                        return;
                    }
                    let packed_argument = naga_function.expressions.append(
                        naga::Expression::FunctionArgument(physical_index),
                        naga::Span::UNDEFINED,
                    );
                    let packed_group = naga_function.expressions.append(
                        naga::Expression::AccessIndex {
                            base: packed_argument,
                            index: group_index,
                        },
                        naga::Span::UNDEFINED,
                    );
                    packed_emit_start = Some(packed_group);
                    naga::Expression::AccessIndex {
                        base: packed_group,
                        index: member_index,
                    }
                }
                NagaArgumentSource::ImplicitResource(global) => {
                    naga::Expression::GlobalVariable(global)
                }
                NagaArgumentSource::Dead => continue,
            };
            let expression = naga_function
                .expressions
                .append(expression, naga::Span::UNDEFINED);
            if let Some(first) = packed_emit_start {
                naga_function.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(first, expression)),
                    naga::Span::UNDEFINED,
                );
            }
            value_map.insert(argument, expression);
        }
        let resource_seeds = function
            .arg_values
            .iter()
            .zip(argument_abi)
            .filter_map(|(&argument, source)| {
                matches!(source, NagaArgumentSource::ImplicitResource(_)).then_some(argument)
            })
            .collect::<Vec<_>>();
        if let Err(error) = bind_resource_identity_aliases(
            function,
            resource_seeds,
            resource_capabilities,
            logical_result_abis,
            &mut value_map,
        ) {
            lowering_error = Some(error);
            return;
        }
        let mut memory_argument = physical_argument_count;
        let heap = if memory_abi.heap {
            let heap = naga_function.expressions.append(
                naga::Expression::FunctionArgument(memory_argument),
                naga::Span::UNDEFINED,
            );
            memory_argument += 1;
            let bump = naga_function.expressions.append(
                naga::Expression::FunctionArgument(memory_argument),
                naga::Span::UNDEFINED,
            );
            memory_argument += 1;
            Some(HeapCtx {
                heap,
                bump,
                word_type,
                heap_words,
            })
        } else {
            None
        };
        let mem_ctx = if memory_abi.trap {
            let trapped = naga_function.expressions.append(
                naga::Expression::FunctionArgument(memory_argument),
                naga::Span::UNDEFINED,
            );
            Some(MemCtx { heap, trapped })
        } else {
            None
        };
        let mut phi_locals = std::collections::HashMap::new();
        let structured = &body_plan.structured;
        if std::env::var("SONATINA_SPIRV_TRACE_BODY")
            .is_ok_and(|needle| signature.name().contains(&needle))
        {
            eprintln!(
                "sonatina spirv: selected helper IR, function={}\n{}",
                signature.name(),
                FuncWriter::new(function_ref, function).dump_string(),
            );
            eprintln!(
                "sonatina spirv: selected helper regions, function={}, regions={:#?}",
                signature.name(),
                structured.regions,
            );
        }
        let mut result_expression = None;
        if let Err(error) = emit_naga_regions(
            function,
            inst_set,
            word,
            &structured.regions,
            word_type,
            f32_type,
            bool_type,
            &mut naga_function,
            &mut value_map,
            &mut phi_locals,
            &mut result_expression,
            Some(result_abi),
            naga_functions,
            mem_ctx,
        ) {
            lowering_error = Some(structurize_error_with_block_ir(
                error,
                function_ref,
                function,
            ));
            return;
        }
        if result_abi.physical_arity == 0 {
            naga_function.body.push(
                naga::Statement::Return { value: None },
                naga::Span::UNDEFINED,
            );
        } else {
            let Some(value) = result_expression else {
                lowering_error = Some(format!(
                    "spirv: helper `{}` produced no return value. Fail closed.",
                    signature.name(),
                ));
                return;
            };
            naga_function.body.push(
                naga::Statement::Return { value: Some(value) },
                naga::Span::UNDEFINED,
            );
        }
    });
    if let Some(error) = lowering_error {
        return Err(error);
    }
    Ok(naga_function)
}

/// Return `(physical builtin family, optional vector component)` for the
/// portable compute invocation vocabulary. Families use a compact private
/// ordinal so this stage-neutral API does not expose naga types.
#[cfg(feature = "spirv-backend")]
fn compute_builtin_slot(source: SpirvBuiltinSource) -> Option<(usize, Option<u32>)> {
    Some(match source {
        SpirvBuiltinSource::GlobalInvocationIdX => (0, Some(0)),
        SpirvBuiltinSource::GlobalInvocationIdY => (0, Some(1)),
        SpirvBuiltinSource::GlobalInvocationIdZ => (0, Some(2)),
        SpirvBuiltinSource::LocalInvocationIdX => (1, Some(0)),
        SpirvBuiltinSource::LocalInvocationIdY => (1, Some(1)),
        SpirvBuiltinSource::LocalInvocationIdZ => (1, Some(2)),
        SpirvBuiltinSource::WorkgroupIdX => (2, Some(0)),
        SpirvBuiltinSource::WorkgroupIdY => (2, Some(1)),
        SpirvBuiltinSource::WorkgroupIdZ => (2, Some(2)),
        SpirvBuiltinSource::NumWorkgroupsX => (3, Some(0)),
        SpirvBuiltinSource::NumWorkgroupsY => (3, Some(1)),
        SpirvBuiltinSource::NumWorkgroupsZ => (3, Some(2)),
        SpirvBuiltinSource::LocalInvocationIndex => (4, None),
        SpirvBuiltinSource::FragmentPositionX
        | SpirvBuiltinSource::FragmentPositionY
        | SpirvBuiltinSource::VertexIndex
        | SpirvBuiltinSource::InstanceIndex => return None,
    })
}

#[cfg(feature = "spirv-backend")]
fn translate_to_naga(
    module: &Module,
    pipeline: ShaderPipeline,
    external_resources: &[SpirvExternalResource],
    builtin_arguments: &[SpirvBuiltinArgument],
    heap_words: u32,
) -> Result<(naga::Module, SpirvLayout), String> {
    use std::collections::HashMap;
    let trace = std::env::var_os("SONATINA_SPIRV_TRACE").is_some();
    let started = std::time::Instant::now();

    let (first_func, workgroup_size, dispatch_grid) = match pipeline {
        ShaderPipeline::Raster { vertex, fragment } => {
            return authored_raster::translate_entries(
                module, vertex, fragment, external_resources, builtin_arguments,
            );
        }
        ShaderPipeline::Compute { entry, workgroup_size, dispatch_grid } =>
            (entry, workgroup_size, dispatch_grid),
        ShaderPipeline::Fullscreen { entry } => (entry, [0, 0, 0], [1, 1, 1]),
        ShaderPipeline::LegacyScalar { entry, workgroup_size }
        | ShaderPipeline::LegacyGrid { entry, workgroup_size } =>
            (entry, workgroup_size, [1, 1, 1]),
    };
    // Derived predicates describe one already-selected pipeline. They cannot
    // represent conflicting modes and do not choose the compilation target.
    let grid = matches!(pipeline, ShaderPipeline::LegacyGrid { .. });
    let render = matches!(pipeline, ShaderPipeline::Fullscreen { .. });
    let compute = matches!(pipeline, ShaderPipeline::Compute { .. });
    if !module.funcs().contains(&first_func) {
        return Err(format!("spirv: selected entry {first_func:?} is not defined in this module"));
    }

    let sig = module
        .ctx
        .get_sig(first_func)
        .ok_or_else(|| "spirv: selected entry has no declared signature".to_string())?;

    let word = match sig.single_ret_ty() {
        Some(sonatina_ir::Type::I32) => WordKind::U32,
        Some(sonatina_ir::Type::I64) => WordKind::I64,
        Some(other) => {
            return Err(format!(
                "spirv: unsupported kernel return type {other:?}; only i32 (u32 word) \
                 and i64 words are supported"
            ));
        }
        None if compute && sig.returns_unit() => WordKind::U32,
        None => {
            return Err(
                "spirv: kernel has no single return value; the word width cannot be derived"
                    .to_string(),
            );
        }
    };

    if !external_resources.is_empty() && !(compute || render) {
        return Err(
            "spirv: external resources require explicit compute or render mode"
                .to_string(),
        );
    }
    if !builtin_arguments.is_empty() && !compute {
        return Err("spirv: explicit builtin arguments currently require compute mode".to_string());
    }
    if compute && !sig.returns_unit() {
        return Err(
            "spirv compute: explicit compute stages must return unit; results belong in external resources"
                .to_string(),
        );
    }
    let compute_invocation_extent = if compute {
        if workgroup_size.contains(&0) || dispatch_grid.contains(&0) {
            return Err(
                "spirv compute: workgroup and dispatch dimensions must be nonzero".to_string(),
            );
        }
        [
            workgroup_size[0]
                .checked_mul(dispatch_grid[0])
                .ok_or_else(|| "spirv compute: x invocation extent overflows u32".to_string())?,
            workgroup_size[1]
                .checked_mul(dispatch_grid[1])
                .ok_or_else(|| "spirv compute: y invocation extent overflows u32".to_string())?,
            workgroup_size[2]
                .checked_mul(dispatch_grid[2])
                .ok_or_else(|| "spirv compute: z invocation extent overflows u32".to_string())?,
        ]
    } else {
        [1, 1, 1]
    };
    let compute_invocation_count = compute_invocation_extent
        .into_iter()
        .try_fold(1u32, |count, extent| count.checked_mul(extent))
        .ok_or_else(|| "spirv compute: total invocation count overflows u32".to_string())?;
    let compute_trap_span = compute_invocation_count
        .checked_mul(4)
        .ok_or_else(|| "spirv compute: trap channel byte span overflows u32".to_string())?;

    let mut resource_arg_indices = std::collections::HashSet::new();
    for (position, resource) in external_resources.iter().enumerate() {
        if resource.group != 0 || resource.binding != position as u32 {
            return Err(format!(
                "spirv: external resources must occupy contiguous group 0 bindings in declaration order; resource {} requested @group({}) @binding({})",
                resource.name, resource.group, resource.binding
            ));
        }
        if resource.length == 0 {
            return Err(format!(
                "spirv: external resource {} must have a nonzero length",
                resource.name
            ));
        }
        let arg_index = resource.arg_index as usize;
        if arg_index >= sig.args().len() {
            return Err(format!(
                "spirv: external resource {} refers to missing kernel arg {}",
                resource.name, resource.arg_index
            ));
        }
        if !resource_arg_indices.insert(arg_index) {
            return Err(format!(
                "spirv: kernel arg {} is rooted by more than one external resource",
                resource.arg_index
            ));
        }
    }

    // The source ABI keeps every declared resource argument so unused object
    // references are never mistaken for scalar parameter storage. The physical
    // shader interface, however, contains only resources that can reach the
    // selected entry's results or effects. Compact bindings are safe because
    // the layout preserves each resource's stable name and logical arg index.
    let live_arguments = analyze_live_arguments(module);
    let live_mask = live_arguments.get(&first_func).ok_or_else(|| {
        format!(
            "spirv: live-argument analysis has no entry for `{}`",
            sig.name()
        )
    })?;
    let emitted_external_resources = external_resources
        .iter()
        .filter(|resource| {
            live_mask
                .get(resource.arg_index as usize)
                .copied()
                .unwrap_or(true)
        })
        .cloned()
        .enumerate()
        .map(|(binding, mut resource)| {
            resource.group = 0;
            resource.binding = binding as u32;
            resource
        })
        .collect::<Vec<_>>();
    if trace && emitted_external_resources.len() != external_resources.len() {
        eprintln!(
            "sonatina spirv: pruned external resources, entry={}, declared={}, emitted={}, names=[{}]",
            sig.name(),
            external_resources.len(),
            emitted_external_resources.len(),
            emitted_external_resources
                .iter()
                .map(|resource| resource.name.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );
    }

    let mut builtin_arg_indices = std::collections::HashSet::new();
    let mut builtin_sources = std::collections::HashSet::new();
    for argument in builtin_arguments {
        let arg_index = argument.arg_index as usize;
        let Some(&arg_ty) = sig.args().get(arg_index) else {
            return Err(format!(
                "spirv compute: builtin {:?} refers to missing kernel arg {}",
                argument.source, argument.arg_index
            ));
        };
        if resource_arg_indices.contains(&arg_index) {
            return Err(format!(
                "spirv compute: kernel arg {} cannot be both an external resource and builtin {:?}",
                argument.arg_index, argument.source
            ));
        }
        if !builtin_arg_indices.insert(arg_index) {
            return Err(format!(
                "spirv compute: kernel arg {} is supplied by more than one builtin",
                argument.arg_index
            ));
        }
        if !builtin_sources.insert(argument.source) {
            return Err(format!(
                "spirv compute: builtin {:?} is mapped more than once",
                argument.source
            ));
        }
        if compute_builtin_slot(argument.source).is_none() {
            return Err(format!(
                "spirv compute: builtin {:?} is not valid for a compute stage",
                argument.source
            ));
        }
        if arg_ty != sonatina_ir::Type::I32 {
            return Err(format!(
                "spirv compute: builtin {:?} requires an i32/u32 carrier at arg {}, got {arg_ty:?}",
                argument.source, argument.arg_index
            ));
        }
    }

    for (i, &arg_ty) in sig.args().iter().enumerate() {
        if resource_arg_indices.contains(&i) || builtin_arg_indices.contains(&i) {
            continue;
        }
        if word == WordKind::I64 && arg_ty != sonatina_ir::Type::I64 {
            return Err(format!(
                "spirv i64: kernel arg {i} has type {arg_ty:?}; i64 kernels require homogeneous i64 arguments"
            ));
        }
        if !matches!(arg_ty, sonatina_ir::Type::I1 | sonatina_ir::Type::I32 | sonatina_ir::Type::I64 | sonatina_ir::Type::F32) {
            return Err(format!("spirv: kernel arg {i} has unsupported type {arg_ty:?}"));
        }
        if matches!(word, WordKind::U32) && arg_ty == sonatina_ir::Type::I64 {
            return Err(format!("spirv u32: kernel arg {i} is i64, which requires SHADER_INT64"));
        }
        let is_storage_arg = !(grid || render) || i >= 2;
        if is_storage_arg && arg_ty == sonatina_ir::Type::I1 {
            return Err(format!(
                "spirv: kernel arg {i} is i1; boolean storage-buffer arguments are unsupported"
            ));
        }
    }
    if (grid || render) && sig.args().get(0..2) != Some(&[sonatina_ir::Type::I32, sonatina_ir::Type::I32]) {
        return Err("spirv grid/render: coordinate args 0 and 1 must both be i32".to_string());
    }

    let word_width = word.width_bytes();

    let mut naga_mod = naga::Module::default();

    let word_type = naga_mod.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar {
                kind: match word {
                    WordKind::U32 => naga::ScalarKind::Uint,
                    WordKind::I64 => naga::ScalarKind::Sint,
                },
                width: word_width as u8,
            }),
        },
        naga::Span::UNDEFINED,
    );
    let f32_type = naga_mod.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Float,
                width: 4,
            }),
        },
        naga::Span::UNDEFINED,
    );
    let bool_type = naga_mod.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Bool,
                width: 1,
            }),
        },
        naga::Span::UNDEFINED,
    );

    let call_order = reachable_call_postorder(module, first_func)?;
    let typed_local_types = collect_naga_typed_local_types(
        module,
        &call_order,
        word,
        word_type,
        f32_type,
        bool_type,
        &mut naga_mod.types,
    )?;

    // Scan the first function for ObjAlloc (output mode). Under a u32 word, also
    // fail closed on any signedness-sensitive op (Sar / signed compares / signed
    // div|mod): Sonatina integers are signless, so u32 is exact for wrapping
    // Add/Sub/Mul but WRONG for these until a sign mapping is designed. We never
    // silently emit the signed WGSL operator.
    let phase = std::time::Instant::now();
    let (param_count, has_obj_alloc, has_mem, has_unreachable, mem_heap_bytes) = module
        .func_store
        .try_view(first_func, |f| -> Result<(usize, bool, bool, bool, u64), String> {
            let pc = f.arg_values.len();
            let is = f.inst_set();
            let mut has_alloc = false;
            let mut has_mem = false;
            // Finding A (adversarial review, 2026-08-08): whether the
            // function contains ANY `Unreachable` terminator, Mem ops or
            // not. `structurize.rs` classifies Unreachable Return-like
            // unconditionally (needed for the array-bounds-trap shape), so
            // a function that traps via `RTerminator::Trap`
            // (wasm_lower.rs:4769) or a checked-usize overflow
            // (`lower_checked_usize_arith` -> `trap_if`,
            // wasm_lower.rs:4439-4470) with NO arrays at all now
            // structurizes fine too -- without a trap channel gated on this
            // flag (not just `has_mem`), that path silently falls through
            // to a zero/uninitialized result exactly like the original
            // Review finding 4, just on the has_mem==false side. Also covers the
            // Sccp/DCE hazard: a kernel whose Mem ops all get eliminated
            // but whose trap survives still needs the channel.
            let mut has_unreachable = false;
            // Review finding 1 (heap-exhaustion aliasing), the compile-time half:
            // a conservative high-water bound over every MemAllocDynamic
            // instruction's constant size in this function. A loop allocation
            // is counted once only after `verify_arena_scopes` proves that every
            // iteration rewinds its compiler-authored frame before the
            // backedge. Verified sibling scopes reuse storage after rewind;
            // nested scopes add to their parent's live storage. This makes the
            // runtime overflow check in the translator's MemAllocDynamic arm
            // unreachable for accepted modules, rather than the only defense.
            let mut mem_alloc_census = Vec::new();
            let mut bounded_allocations = Vec::new();
            let mut cfg = sonatina_ir::cfg::ControlFlowGraph::default();
            cfg.compute(f);
            let mut domtree = crate::domtree::DomTree::new();
            domtree.compute(&cfg);
            let mut loop_tree = crate::loop_analysis::LoopTree::new();
            loop_tree.compute(&cfg, &domtree);
            let arena_scopes = verify_arena_scopes(f, &cfg, &loop_tree)?;
            if word == WordKind::I64
                && f.dfg
                    .value_ids()
                    .any(|value| f.dfg.value_ty(value) == sonatina_ir::Type::F32)
            {
                return Err(
                    "spirv i64: f32 values and 32-bit float conversions are unsupported"
                        .to_string(),
                );
            }
            // External f32 resources are admitted below through authored roots.
            // Locally allocated object storage is still selected by ObjAlloc's
            // batch path, where f32 object carriers have not been specified.
            // Keep that unsupported boundary fail-closed and diagnostic rather
            // than letting the later grid/batch mode conflict mask it.
            let has_local_obj_alloc = f.layout.iter_block().any(|bid| {
                f.layout.iter_inst(bid).any(|iid| {
                    <&sonatina_ir::inst::data::ObjAlloc as sonatina_ir::InstDowncast>::downcast(
                        is,
                        f.dfg.inst(iid),
                    )
                    .is_some()
                })
            });
            if has_local_obj_alloc {
                for bid in f.layout.iter_block() {
                    for iid in f.layout.iter_inst(bid) {
                        let inst_data = f.dfg.inst(iid);
                        if let Some(store) = <&sonatina_ir::inst::data::ObjStore as sonatina_ir::InstDowncast>::downcast(is, inst_data)
                            && f.dfg.value_ty(*store.value()) == sonatina_ir::Type::F32
                        {
                            return Err(
                                "spirv: f32 object storage is unsupported for private allocations"
                                    .to_string(),
                            );
                        }
                        if <&sonatina_ir::inst::data::ObjLoad as sonatina_ir::InstDowncast>::downcast(is, inst_data).is_some()
                            && f.dfg.inst_result(iid).is_some_and(|value| {
                                f.dfg.value_ty(value) == sonatina_ir::Type::F32
                            })
                        {
                            return Err(
                                "spirv: f32 object storage is unsupported for private allocations"
                                    .to_string(),
                            );
                        }
                    }
                }
            }
            for bid in f.layout.iter_block() {
                for iid in f.layout.iter_inst(bid) {
                    let inst_data = f.dfg.inst(iid);
                    for &result in f.dfg.inst_results(iid) {
                        let result_ty = f.dfg.value_ty(result);
                        let carrier_ty = match word {
                            WordKind::U32 => sonatina_ir::Type::I32,
                            WordKind::I64 => sonatina_ir::Type::I64,
                        };
                        if result_ty.is_integral()
                            && result_ty != sonatina_ir::Type::I1
                            && result_ty != carrier_ty
                        {
                            let function_ir = FuncWriter::new(first_func, f).dump_string();
                            let marker = format!("v{}.", result.0);
                            let lines = function_ir.lines().collect::<Vec<_>>();
                            let result_ir = lines
                                .iter()
                                .position(|line| line.contains(&marker))
                                .map(|line| {
                                    let start = line.saturating_sub(16);
                                    let end = (line + 3).min(lines.len());
                                    lines[start..end].join(" | ")
                                })
                                .unwrap_or_else(|| "<result instruction unavailable>".to_string());
                            return Err(format!(
                                "spirv: narrow or mixed integer instruction result {result:?} has unsupported type {result_ty:?}; expected {carrier_ty:?} carrier in `{}`; instruction `{}`; IR `{result_ir}`",
                                sig.name(),
                                inst_data.as_text(),
                            ));
                        }
                    }
                    if !spirv_instruction_is_lowered(is, inst_data) {
                        let instruction = inst_data.as_text();
                        let context = instruction_ir_context(
                            first_func, f, bid, &instruction,
                        );
                        return Err(format!(
                            "spirv: instruction `{instruction}` is unsupported by the SPIR-V \
                             translator in `{}` at {bid:?}; IR `{context}`",
                            sig.name(),
                        ));
                    }
                    if let Some(trunc) =
                        <&sonatina_ir::inst::cast::Trunc as sonatina_ir::InstDowncast>::downcast(
                            is, inst_data,
                        )
                    {
                        let from_ty = f.dfg.value_ty(*trunc.from());
                        let to_ty = *trunc.ty();
                        if word != WordKind::U32
                            || from_ty != sonatina_ir::Type::I32
                            || to_ty != sonatina_ir::Type::I1
                        {
                            let instruction = inst_data.as_text();
                            let context = instruction_ir_context(
                                first_func, f, bid, &instruction,
                            );
                            return Err(format!(
                                "spirv: Trunc supports exactly i32 -> i1 under the u32 browser \
                                 word; got {from_ty:?} -> {to_ty:?} in `{}` at {bid:?}; \
                                 instruction `{instruction}`; IR `{context}`. Fail closed.",
                                sig.name(),
                            ));
                        }
                    }
                    if let Some(bitcast) =
                        <&sonatina_ir::inst::cast::Bitcast as sonatina_ir::InstDowncast>::downcast(
                            is, inst_data,
                        )
                    {
                        let from_ty = f.dfg.value_ty(*bitcast.from());
                        let to_ty = *bitcast.ty();
                        let admitted = typed_local_zero_projection_path(
                            f.ctx(), from_ty, to_ty,
                        )
                        .is_some()
                            || (word == WordKind::U32
                                && matches!(
                                    (from_ty, to_ty),
                                    (sonatina_ir::Type::I32, sonatina_ir::Type::F32)
                                        | (sonatina_ir::Type::F32, sonatina_ir::Type::I32)
                                ));
                        if !admitted {
                            return Err(format!(
                                "spirv: Bitcast supports exactly i32 <-> f32 under the u32 browser word or a typed-local zero projection; got {from_ty:?} -> {to_ty:?}"
                            ));
                        }
                    }
                    if <&sonatina_ir::inst::data::ObjAlloc as sonatina_ir::InstDowncast>::downcast(
                        is, inst_data,
                    )
                    .is_some()
                    {
                        has_alloc = true;
                    }
                    if <&sonatina_ir::inst::control_flow::Unreachable as sonatina_ir::InstDowncast>::downcast(
                        is, inst_data,
                    )
                    .is_some()
                    {
                        has_unreachable = true;
                    }
                    if <&sonatina_ir::inst::data::MemCheckpoint as sonatina_ir::InstDowncast>::downcast(
                        is, inst_data,
                    )
                    .is_some()
                        || <&sonatina_ir::inst::data::MemRewind as sonatina_ir::InstDowncast>::downcast(
                            is, inst_data,
                        )
                        .is_some()
                    {
                        has_mem = true;
                    }
                    if let Some(copy) = <&sonatina_ir::inst::data::Memcopy as sonatina_ir::InstDowncast>::downcast(is, inst_data) {
                        has_mem = true;
                        let Some(len_imm) = f.dfg.value_imm(*copy.len()) else {
                            return Err(
                                "spirv: Memcopy with a runtime length is unsupported; \
                                 the generated byte loop requires a compile-time bound. \
                                 Fail closed."
                                    .to_string(),
                            );
                        };
                        let len_bytes: u64 = match len_imm {
                            sonatina_ir::Immediate::I1(v) => v as u64,
                            sonatina_ir::Immediate::I8(v) => v as u8 as u64,
                            sonatina_ir::Immediate::I32(v) => v as u32 as u64,
                            sonatina_ir::Immediate::I64(v) => v as u64,
                            _ => {
                                return Err(
                                    "spirv: Memcopy length has an unsupported immediate kind. \
                                     Fail closed."
                                        .to_string(),
                                );
                            }
                        };
                        let heap_capacity_bytes = (heap_words as u64) * 4;
                        if len_bytes > heap_capacity_bytes {
                            return Err(format!(
                                "spirv: Memcopy length ({len_bytes} bytes) exceeds the private heap capacity ({heap_capacity_bytes} bytes). Fail closed."
                            ));
                        }
                    }
                    if let Some(alloc) = <&sonatina_ir::inst::data::MemAllocDynamic as sonatina_ir::InstDowncast>::downcast(is, inst_data) {
                        has_mem = true;
                        let Some(size_imm) = f.dfg.value_imm(*alloc.size()) else {
                            return Err(
                                "spirv: MemAllocDynamic with a non-constant size is \
                                 unsupported (the private-heap capacity proof requires \
                                 every allocation size to be known at compile time). \
                                 Fail closed."
                                    .to_string(),
                            );
                        };
                        let size_bytes: u64 = match size_imm {
                            sonatina_ir::Immediate::I1(v) => v as u64,
                            sonatina_ir::Immediate::I8(v) => v as u8 as u64,
                            sonatina_ir::Immediate::I32(v) => v as u32 as u64,
                            sonatina_ir::Immediate::I64(v) => v as u64,
                            _ => {
                                return Err(
                                    "spirv: MemAllocDynamic size has an unsupported \
                                     immediate kind. Fail closed."
                                        .to_string(),
                                );
                            }
                        };
                        let mut executions = 1_u64;
                        if !arena_scopes.scoped_loop_allocations.contains(&iid) {
                            let mut containing_loop = loop_tree.loop_of_block(bid);
                            while let Some(loop_id) = containing_loop {
                                let Some(iterations) = crate::analysis::induction::u32_loop_iteration_upper_bound(
                                    f,
                                    &cfg,
                                    &loop_tree,
                                    loop_id,
                                ) else {
                                    let instruction = inst_data.as_text();
                                    let context = instruction_ir_context(
                                        first_func, f, bid, &instruction,
                                    );
                                    return Err(format!(
                                        "spirv: MemAllocDynamic inside a loop has neither a \
                                         balanced MemCheckpoint/MemRewind scope opened during \
                                         that iteration nor a compiler-proven static u32 trip \
                                         bound in `{}` at {bid:?}; instruction `{instruction}`; \
                                         IR `{context}`. Its total allocation would depend on \
                                         the runtime trip count. Fail closed.",
                                        sig.name(),
                                    ));
                                };
                                executions = executions.saturating_mul(iterations);
                                containing_loop = loop_tree.parent_loop(loop_id);
                            }
                        }
                        mem_alloc_census.push((
                            size_bytes.saturating_mul(executions),
                            size_bytes,
                            executions,
                            bid,
                            inst_data.as_text(),
                        ));
                        bounded_allocations
                            .push((iid, size_bytes.saturating_mul(executions)));
                    }
                    if let Some(load) = <&sonatina_ir::inst::data::Mload as sonatina_ir::InstDowncast>::downcast(is, inst_data) {
                        let address_ty = f.dfg.value_ty(*load.addr());
                        let pointer_pointee = match address_ty.resolve_compound(&module.ctx) {
                            Some(sonatina_ir::types::CompoundType::Ptr(pointee)) => Some(pointee),
                            _ => None,
                        };
                        has_mem |= pointer_pointee.is_none();
                        let result_ty = f
                            .dfg
                            .inst_result(iid)
                            .map(|result| f.dfg.value_ty(result));
                        let typed_local = pointer_pointee == Some(*load.ty())
                            && result_ty == Some(*load.ty());
                        let byte_arena = word == WordKind::U32
                            && matches!(
                                (load.ty(), result_ty),
                                (sonatina_ir::Type::I1, Some(sonatina_ir::Type::I1))
                                    | (sonatina_ir::Type::I1, Some(sonatina_ir::Type::I32))
                                    | (sonatina_ir::Type::I32, Some(sonatina_ir::Type::I32))
                            );
                        let admitted = typed_local || byte_arena;
                        if !admitted {
                            let instruction = inst_data.as_text();
                            let context = instruction_ir_context(
                                first_func, f, bid, &instruction,
                            );
                            return Err(format!(
                                "spirv: Mload address/memory/result types {address_ty:?}/{:?}/{result_ty:?} are unsupported \
                                 (typed-local pointers require an exact pointee/result match; the u32 byte arena admits I1 -> I1 or I32 and I32 -> I32) in `{}` at {bid:?}; \
                                 instruction `{instruction}`; IR `{context}`. Fail closed.",
                                load.ty(), sig.name(),
                            ));
                        }
                    }
                    if let Some(store) = <&sonatina_ir::inst::data::Mstore as sonatina_ir::InstDowncast>::downcast(is, inst_data) {
                        let address_ty = f.dfg.value_ty(*store.addr());
                        let pointer_pointee = match address_ty.resolve_compound(&module.ctx) {
                            Some(sonatina_ir::types::CompoundType::Ptr(pointee)) => Some(pointee),
                            _ => None,
                        };
                        has_mem |= pointer_pointee.is_none();
                        let value_ty = f.dfg.value_ty(*store.value());
                        let typed_local = pointer_pointee == Some(*store.ty())
                            && value_ty == *store.ty();
                        let byte_arena = word == WordKind::U32
                            && matches!(
                                (store.ty(), value_ty),
                                (sonatina_ir::Type::I1, sonatina_ir::Type::I1)
                                    | (sonatina_ir::Type::I32, sonatina_ir::Type::I32)
                            );
                        let admitted = typed_local || byte_arena;
                        if !admitted {
                            let instruction = inst_data.as_text();
                            let context = instruction_ir_context(
                                first_func, f, bid, &instruction,
                            );
                            return Err(format!(
                                "spirv: Mstore address/memory/value types {address_ty:?}/{:?}/{value_ty:?} are unsupported \
                                 (typed-local pointers require an exact pointee/value match; the u32 byte arena admits matching I1 or I32 values) in `{}` at {bid:?}; \
                                 instruction `{instruction}`; IR `{context}`. Fail closed.",
                                store.ty(), sig.name(),
                            ));
                        }
                    }
                    if word == WordKind::U32 {
                        if let Some(op) = unsupported_signed_op_under_u32(is, inst_data) {
                            return Err(format!(
                                "spirv u32: signedness-sensitive op `{op}` is unsupported under \
                                 a u32 word (Sonatina integers are signless; a sign mapping is \
                                 not yet designed). Fail closed."
                            ));
                        }
                        // WGSL accepts runtime u32 shift amounts directly. Sar
                        // still bitcasts only the shifted value through i32; Shl
                        // and Shr remain direct u32 operations.
                    }
                    // `Shr` (logical/unsigned shift) is only lowered for the u32
                    // browser word, where WGSL `>>` on a `u32` IS the logical shift.
                    // The i64 word is a `Sint` scalar whose `>>` is arithmetic; a
                    // logical shift would need a bitcast dance no path emits, so it
                    // fails closed rather than emitting the wrong (arithmetic) shift.
                    if word == WordKind::I64
                        && <&sonatina_ir::inst::arith::Shr as sonatina_ir::InstDowncast>::downcast(
                            is, inst_data,
                        )
                        .is_some()
                    {
                        return Err(
                            "spirv i64: logical shift `Shr` is unsupported under the i64 word \
                             (only the u32 browser word lowers `>>`). Fail closed."
                                .to_string(),
                        );
                    }
                }
            }
            let mem_heap_bytes = arena_scopes.high_water_bytes(bounded_allocations);
            if has_mem && word == WordKind::I64 {
                return Err(
                    "spirv i64: MemAllocDynamic/Memcopy/Mload/Mstore (function-local arrays) are \
                     unsupported under the i64 word (Mem ops admit the u32 browser word \
                     only). Fail closed."
                        .to_string(),
                );
            }
            if has_mem {
                let heap_capacity_bytes = (heap_words as u64) * 4;
                if mem_heap_bytes > heap_capacity_bytes {
                    mem_alloc_census.sort_unstable_by(|left, right| right.0.cmp(&left.0));
                    let largest = mem_alloc_census
                        .iter()
                        .take(8)
                        .map(|(total, size, executions, block, instruction)| {
                            format!(
                                "{total} bytes total = {size} bytes x {executions} at \
                                 {block:?}, instruction `{instruction}`"
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("; ");
                    return Err(format!(
                        "spirv: MemAllocDynamic static allocation high-water ({mem_heap_bytes} \
                         bytes) exceeds the private heap capacity ({heap_capacity_bytes} \
                         bytes = {heap_words} words); increase with_private_heap_words or \
                         reduce array usage. Largest contributors: {largest}. Fail closed."
                    ));
                }
            }
            Ok((pc, has_alloc, has_mem, has_unreachable, mem_heap_bytes))
        })
        .ok_or_else(|| "spirv: first function body is unavailable".to_string())??;
    if trace {
        eprintln!(
            "sonatina spirv: pre-scan complete, function={}, heap_bytes={}, elapsed_ms={}, total_elapsed_ms={}",
            sig.name(),
            mem_heap_bytes,
            phase.elapsed().as_millis(),
            started.elapsed().as_millis()
        );
    }

    // `heap_words` is a fail-closed capacity, not an allocation request. The
    // pre-scan above has already proved the exact static upper bound for this
    // call-free entry, so materialize only that many private words. Emitting
    // the full default capacity for every invocation turns a small local
    // array into 32 KiB of private storage and multiplies that cost by the
    // workgroup width. This was enough to make otherwise modest Fe workgroup
    // kernels pathological for browser shader compilers.
    let private_heap_words = if has_mem {
        u32::try_from(mem_heap_bytes.div_ceil(4))
            .map_err(|_| {
                format!(
                    "spirv: static private heap requirement ({mem_heap_bytes} bytes) does not fit in u32 words"
                )
            })?
            .max(1)
    } else {
        0
    };

    let external_resource_stages = vec![
        vec![if render {
            SpirvShaderStage::Fragment
        } else {
            SpirvShaderStage::Compute
        }];
        emitted_external_resources.len()
    ];
    let (external_roots, external_layout_bindings) = append_external_resources(
        &mut naga_mod,
        &emitted_external_resources,
        &external_resource_stages,
        word,
        word_type,
        f32_type,
    )?;
    let helper_resource_capabilities = helper_resource_capabilities(
        module,
        &sig,
        &external_roots,
    )?;
    let helper_logical_result_abis = helper_naga_logical_result_abis(
        module,
        &call_order,
        &[first_func],
        &helper_resource_capabilities,
    )?;
    let helper_resource_variants = helper_resource_variants(
        module,
        &call_order,
        first_func,
        &external_roots,
        &helper_resource_capabilities,
        &helper_logical_result_abis,
        &live_arguments,
    )?;
    let helper_memory_abis = helper_private_memory_abis(module, &call_order, first_func)?;
    let helper_memory = helper_memory_abis.values().any(|abi| abi.heap);
    let helper_trap = helper_memory_abis.values().any(|abi| abi.trap);
    if helper_memory && !has_mem {
        return Err(
            "spirv: a reachable helper accesses the private arena, but the entry function owns no proven arena allocation. Fail closed."
                .to_string(),
        );
    }
    let needs_trap_channel = has_mem || has_unreachable || helper_trap;
    let private_heap_type = if has_mem {
        let heap_len = std::num::NonZeroU32::new(private_heap_words)
            .ok_or_else(|| "spirv: derived private heap must be nonzero".to_string())?;
        Some(naga_mod.types.insert(
            naga::Type {
                name: Some("FeHeap".into()),
                inner: naga::TypeInner::Array {
                    base: word_type,
                    size: naga::ArraySize::Constant(heap_len),
                    stride: 4,
                },
            },
            naga::Span::UNDEFINED,
        ))
    } else {
        None
    };
    let helper_memory_types = NagaMemoryAbiTypes {
        heap: if helper_memory {
            Some(naga_mod.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Pointer {
                        base: private_heap_type.expect("helper memory requires an entry heap"),
                        space: naga::AddressSpace::Function,
                    },
                },
                naga::Span::UNDEFINED,
            ))
        } else {
            None
        },
        word: if helper_memory {
            Some(naga_mod.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Pointer {
                        base: word_type,
                        space: naga::AddressSpace::Function,
                    },
                },
                naga::Span::UNDEFINED,
            ))
        } else {
            None
        },
        trap: if helper_trap {
            Some(naga_mod.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Pointer {
                        base: bool_type,
                        space: naga::AddressSpace::Function,
                    },
                },
                naga::Span::UNDEFINED,
            ))
        } else {
            None
        },
    };

    let mut naga_functions =
        NagaFunctionMap::with_typed_local_types(typed_local_types);
    let helper_plans = helper_plan::plan_helper_abis(
        module,
        &call_order,
        &[first_func],
        word,
        word_type,
        f32_type,
        bool_type,
        &helper_resource_capabilities,
        &helper_logical_result_abis,
        &helper_resource_variants.bindings,
        &helper_memory_abis,
        helper_memory_types,
        &live_arguments,
        &naga_functions,
        &mut naga_mod.types,
    )?;
    let mut lowered_variants = std::collections::HashMap::<
        NagaFunctionVariant,
        NagaFunctionInfo,
    >::new();
    for helper_plan::PlannedHelperAbi {
        variant: helper_variant,
        arguments: argument_abi,
        packed_arguments,
        result: result_abi,
        memory: helper_memory_abi,
        body: body_plan,
        parameters,
    } in helper_plans
    {
        let helper_ref = helper_variant.function;
        let variant_index = helper_variant.ordinal as usize;
        let helper_signature = module
            .ctx
            .get_sig(helper_ref)
            .ok_or_else(|| format!("spirv: helper {helper_ref:?} has no signature"))?;
        let resource_variants = helper_resource_variants.variants(helper_ref);
        let resource_bindings = &resource_variants[variant_index];
        naga_functions.replace_call_sites(helper_variant_call_sites(
            helper_variant,
            &helper_resource_variants,
            &lowered_variants,
        )?);
        let mut helper = lower_naga_helper(
            module,
            helper_ref,
            &body_plan,
            word,
            word_type,
            f32_type,
            bool_type,
            &argument_abi,
            packed_arguments.as_ref(),
            &result_abi,
            helper_memory_abi,
            parameters,
            private_heap_words,
            &helper_resource_capabilities,
            &helper_logical_result_abis,
            &naga_functions,
        )
        .map_err(|error| {
            format!(
                "spirv: helper `{}` resource variant {variant_index} lowering failed: {error}",
                helper_signature.name(),
            )
        })?;
        if resource_variants.len() > 1 {
            helper.name = Some(format!(
                "{}_resource_variant_{variant_index}",
                helper_signature.name(),
            ));
        }
        let handle = naga_mod.functions.append(helper, naga::Span::UNDEFINED);
        if trace {
            let resources = resource_bindings
                .iter()
                .filter_map(|binding| binding.map(naga::Handle::index))
                .collect::<Vec<_>>();
            eprintln!(
                "sonatina spirv: lowered helper, naga_handle={}, sonatina_ref={helper_ref:?}, name={}, resource_variant={variant_index}, resources={resources:?}",
                handle.index(),
                helper_signature.name(),
            );
        }
        lowered_variants.insert(
            helper_variant,
            NagaFunctionInfo {
                handle,
                argument_abi,
                packed_arguments,
                result_abi,
                memory_abi: helper_memory_abi,
            },
        );
    }
    naga_functions.replace_call_sites(helper_variant_call_sites(
        helper_resource_variants.entry,
        &helper_resource_variants,
        &lowered_variants,
    )?);

    // Grid mode fail-closed checks. Grid is a driver-declared envelope fact: the
    // translator never guesses it, and when asked for it, every precondition is
    // enforced here so no wrong shader is ever emitted.
    if grid {
        if word != WordKind::U32 {
            return Err(
                "spirv grid: grid mode requires the u32 word (browser profile); got i64"
                    .to_string(),
            );
        }
        if param_count < 2 {
            return Err(format!(
                "spirv grid: a grid kernel needs at least (px, py) args; got {param_count}"
            ));
        }
        if has_obj_alloc {
            return Err(
                "spirv grid: grid and batch (ObjAlloc) modes are mutually exclusive".to_string(),
            );
        }
        if !(workgroup_size[0] >= 1 && workgroup_size[1] >= 1 && workgroup_size[2] == 1) {
            return Err("spirv grid: grid dispatch is 2D; workgroup z must be 1".to_string());
        }
    }

    // Batch (ObjAlloc) mode reuses the output storage buffer AS the
    // allocation (a completely different mechanism from the private-heap
    // array emulation, `RUNG3_SPIRV_ARRAYS_DESIGN.md` section 1, option B
    // REJECTED). Combining the two is out of scope here: fail closed with a
    // named error rather than silently picking one interpretation.
    if has_obj_alloc && has_mem {
        return Err(
            "spirv: function-local [u32; N] arrays (MemAllocDynamic/Mload/Mstore) inside a \
             batch (ObjAlloc) mode kernel are unsupported. Fail closed."
                .to_string(),
        );
    }

    // ======================================================================
    // Explicit compute mode. The stage returns unit and communicates only via
    // authored external resources. Ordinary scalar arguments, if any, occupy
    // one read-only parameter record after the authored resource bindings.
    // ======================================================================
    if compute {
        if has_obj_alloc {
            return Err(
                "spirv compute: external-resource compute must not contain ObjAlloc or select Batch"
                    .to_string(),
            );
        }

        let parameter_args = sig
            .args()
            .iter()
            .copied()
            .enumerate()
            .filter(|(index, _)| {
                !resource_arg_indices.contains(index) && !builtin_arg_indices.contains(index)
            })
            .collect::<Vec<_>>();
        let mut parameter_members = Vec::with_capacity(parameter_args.len());
        let mut layout_parameter_members = Vec::with_capacity(parameter_args.len());
        let mut parameter_span = 0;
        let mut parameter_align = 1;
        for (arg_index, ty) in &parameter_args {
            let (naga_ty, width, scalar) = match ty {
                sonatina_ir::Type::I32 => (word_type, 4, SpirvScalarKind::I32),
                sonatina_ir::Type::F32 => (f32_type, 4, SpirvScalarKind::F32),
                _ => {
                    return Err(format!(
                        "spirv compute: parameter arg {arg_index} has unsupported storage type {ty:?}"
                    ));
                }
            };
            parameter_span = (parameter_span + width - 1) & !(width - 1);
            parameter_members.push(naga::StructMember {
                name: Some(format!("p{arg_index}")),
                ty: naga_ty,
                binding: None,
                offset: parameter_span,
            });
            layout_parameter_members.push(SpirvBindingMember {
                arg_index: *arg_index as u32,
                offset: parameter_span,
                width,
                scalar,
            });
            parameter_span += width;
            parameter_align = parameter_align.max(width);
        }
        parameter_span = (parameter_span + parameter_align - 1) & !(parameter_align - 1);
        let parameter_binding = emitted_external_resources.len() as u32;
        let parameter_var = if parameter_members.is_empty() {
            None
        } else {
            let parameter_type = naga_mod.types.insert(
                naga::Type {
                    name: Some("Params".into()),
                    inner: naga::TypeInner::Struct {
                        members: parameter_members,
                        span: parameter_span,
                    },
                },
                naga::Span::UNDEFINED,
            );
            Some(naga_mod.global_variables.append(
                naga::GlobalVariable {
                    name: Some("params".into()),
                    space: naga::AddressSpace::Storage {
                        access: naga::StorageAccess::LOAD,
                    },
                    binding: Some(naga::ResourceBinding {
                        group: 0,
                        binding: parameter_binding,
                    }),
                    ty: parameter_type,
                    init: None,
                    memory_decorations: naga::ir::MemoryDecorations::empty(),
                },
                naga::Span::UNDEFINED,
            ))
        };

        let trap_binding = parameter_binding + u32::from(parameter_var.is_some());
        let trap_var = if needs_trap_channel {
            let trap_len = std::num::NonZeroU32::new(compute_invocation_count)
                .expect("validated compute invocation count is nonzero");
            let trap_type = naga_mod.types.insert(
                naga::Type {
                    name: Some("TrapArray".into()),
                    inner: naga::TypeInner::Array {
                        base: word_type,
                        size: naga::ArraySize::Constant(trap_len),
                        stride: 4,
                    },
                },
                naga::Span::UNDEFINED,
            );
            Some(naga_mod.global_variables.append(
                naga::GlobalVariable {
                    name: Some("trap".into()),
                    space: naga::AddressSpace::Storage {
                        access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
                    },
                    binding: Some(naga::ResourceBinding {
                        group: 0,
                        binding: trap_binding,
                    }),
                    ty: trap_type,
                    init: None,
                    memory_decorations: naga::ir::MemoryDecorations::empty(),
                },
                naga::Span::UNDEFINED,
            ))
        } else {
            None
        };

        let vec3_u32_type = naga_mod.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Tri,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: 4,
                    },
                },
            },
            naga::Span::UNDEFINED,
        );
        let mut physical_builtin_arguments = [None; 5];
        let mut entry_arguments = Vec::new();
        for family in 0..physical_builtin_arguments.len() {
            let authored = builtin_arguments.iter().any(|argument| {
                compute_builtin_slot(argument.source)
                    .is_some_and(|(candidate, _)| candidate == family)
            });
            let compiler_trap_index = needs_trap_channel
                && compute_invocation_count > 1
                && family == 0;
            if !authored && !compiler_trap_index {
                continue;
            }
            let (name, ty, binding) = match family {
                0 => ("global_invocation_id", vec3_u32_type, naga::BuiltIn::GlobalInvocationId),
                1 => ("local_invocation_id", vec3_u32_type, naga::BuiltIn::LocalInvocationId),
                2 => ("workgroup_id", vec3_u32_type, naga::BuiltIn::WorkGroupId),
                3 => ("num_workgroups", vec3_u32_type, naga::BuiltIn::NumWorkGroups),
                4 => ("local_invocation_index", word_type, naga::BuiltIn::LocalInvocationIndex),
                _ => unreachable!("portable compute builtin family"),
            };
            physical_builtin_arguments[family] = Some(entry_arguments.len() as u32);
            entry_arguments.push(naga::FunctionArgument {
                name: Some(name.to_string()),
                ty,
                binding: Some(naga::Binding::BuiltIn(binding)),
            });
        }

        let mut func = naga::Function {
            name: Some("main".into()),
            arguments: entry_arguments,
            result: None,
            local_variables: naga::Arena::new(),
            expressions: naga::Arena::new(),
            named_expressions: Default::default(),
            body: naga::Block::new(),
            diagnostic_filter_leaf: None,
        };
        let mem_ctx = if needs_trap_channel {
            let heap = if has_mem {
                let heap_type = private_heap_type
                    .expect("has_mem establishes one shared private heap type");
                let heap_zero = func.expressions.append(
                    naga::Expression::ZeroValue(heap_type),
                    naga::Span::UNDEFINED,
                );
                let heap = func.local_variables.append(
                    naga::LocalVariable {
                        name: Some("fe_heap".into()),
                        ty: heap_type,
                        init: Some(heap_zero),
                    },
                    naga::Span::UNDEFINED,
                );
                let bump_zero = func.expressions.append(
                    naga::Expression::Literal(naga::Literal::U32(0)),
                    naga::Span::UNDEFINED,
                );
                let bump = func.local_variables.append(
                    naga::LocalVariable {
                        name: Some("fe_bump".into()),
                        ty: word_type,
                        init: Some(bump_zero),
                    },
                    naga::Span::UNDEFINED,
                );
                let heap = func.expressions.append(
                    naga::Expression::LocalVariable(heap),
                    naga::Span::UNDEFINED,
                );
                let bump = func.expressions.append(
                    naga::Expression::LocalVariable(bump),
                    naga::Span::UNDEFINED,
                );
                Some(HeapCtx {
                    heap,
                    bump,
                    word_type,
                    heap_words: private_heap_words,
                })
            } else {
                None
            };
            let trapped_false = func.expressions.append(
                naga::Expression::Literal(naga::Literal::Bool(false)),
                naga::Span::UNDEFINED,
            );
            let trapped = func.local_variables.append(
                naga::LocalVariable {
                    name: Some("fe_trapped".into()),
                    ty: bool_type,
                    init: Some(trapped_false),
                },
                naga::Span::UNDEFINED,
            );
            let trapped = func.expressions.append(
                naga::Expression::LocalVariable(trapped),
                naga::Span::UNDEFINED,
            );
            Some(MemCtx { heap, trapped })
        } else {
            None
        };

        let mut body_error = None;
        module.func_store.try_view(first_func, |function| {
            let inst_set = function.inst_set();
            let mut value_map = HashMap::new();
            let mut phi_locals = HashMap::new();
            for argument in builtin_arguments {
                let Some(&arg_value) = function.arg_values.get(argument.arg_index as usize) else {
                    body_error = Some(format!(
                        "spirv compute: builtin arg {} disappeared during lowering",
                        argument.arg_index
                    ));
                    return;
                };
                let (family, component) = compute_builtin_slot(argument.source)
                    .expect("compute builtin source was validated");
                let physical_index = physical_builtin_arguments[family]
                    .expect("used compute builtin family has an entry argument");
                let physical = func.expressions.append(
                    naga::Expression::FunctionArgument(physical_index),
                    naga::Span::UNDEFINED,
                );
                let value = if let Some(component) = component {
                    let projected = func.expressions.append(
                        naga::Expression::AccessIndex {
                            base: physical,
                            index: component,
                        },
                        naga::Span::UNDEFINED,
                    );
                    func.body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(
                            projected, projected,
                        )),
                        naga::Span::UNDEFINED,
                    );
                    projected
                } else {
                    physical
                };
                value_map.insert(arg_value, value);
            }
            for &(arg_index, global) in &external_roots {
                let Some(&arg_value) = function.arg_values.get(arg_index as usize) else {
                    body_error = Some(format!(
                        "spirv compute: external resource arg {arg_index} disappeared during lowering"
                    ));
                    return;
                };
                let root = func.expressions.append(
                    naga::Expression::GlobalVariable(global),
                    naga::Span::UNDEFINED,
                );
                value_map.insert(arg_value, root);
            }
            let resource_seeds = external_roots
                .iter()
                .filter_map(|(arg_index, _)| {
                    function.arg_values.get(*arg_index as usize).copied()
                })
                .collect::<Vec<_>>();
            if let Err(error) = bind_resource_identity_aliases(
                function,
                resource_seeds,
                &helper_resource_capabilities,
                &helper_logical_result_abis,
                &mut value_map,
            ) {
                body_error = Some(error);
                return;
            }
            if let Some(parameter_var) = parameter_var {
                let params = func.expressions.append(
                    naga::Expression::GlobalVariable(parameter_var),
                    naga::Span::UNDEFINED,
                );
                for (member_index, (arg_index, _)) in parameter_args.iter().enumerate() {
                    let field = func.expressions.append(
                        naga::Expression::AccessIndex {
                            base: params,
                            index: member_index as u32,
                        },
                        naga::Span::UNDEFINED,
                    );
                    let loaded = func.expressions.append(
                        naga::Expression::Load { pointer: field },
                        naga::Span::UNDEFINED,
                    );
                    func.body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(field, field)),
                        naga::Span::UNDEFINED,
                    );
                    func.body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)),
                        naga::Span::UNDEFINED,
                    );
                    value_map.insert(function.arg_values[*arg_index], loaded);
                }
            }
            let phase = std::time::Instant::now();
            let scfg = match crate::structurize::structurize_function(function) {
                Ok(scfg) => scfg,
                Err(error) => {
                    body_error = Some(structurize_error_with_block_ir(
                        error, first_func, function,
                    ));
                    return;
                }
            };
            if trace {
                let stats = scfg.stats();
                eprintln!(
                    "sonatina spirv: structurized compute entry, top_regions={}, region_nodes={}, reachable_blocks={}, referenced_blocks={}, block_occurrences={}, duplicated_block_occurrences={}, loops={}, conditionals={}, loop_exits={}, loop_continues={}, elapsed_ms={}, total_elapsed_ms={}",
                    scfg.regions.len(),
                    stats.region_nodes,
                    stats.reachable_blocks,
                    stats.referenced_blocks,
                    stats.block_occurrences,
                    stats.duplicated_block_occurrences,
                    stats.loops,
                    stats.conditionals,
                    stats.loop_exits,
                    stats.loop_continues,
                    phase.elapsed().as_millis(),
                    started.elapsed().as_millis()
                );
            }
            let mut ignored_result = None;
            let phase = std::time::Instant::now();
            if let Err(error) = emit_naga_regions(
                function,
                inst_set,
                word,
                &scfg.regions,
                word_type,
                f32_type,
                bool_type,
                &mut func,
                &mut value_map,
                &mut phi_locals,
                &mut ignored_result,
                None,
                &naga_functions,
                mem_ctx,
            ) {
                body_error = Some(structurize_error_with_block_ir(
                    error, first_func, function,
                ));
            }
            if trace {
                eprintln!(
                    "sonatina spirv: emitted compute entry regions, expressions={}, elapsed_ms={}, total_elapsed_ms={}",
                    func.expressions.len(),
                    phase.elapsed().as_millis(),
                    started.elapsed().as_millis()
                );
            }
        });
        if let Some(error) = body_error {
            return Err(error);
        }
        if let (Some(mem_ctx), Some(trap_var)) = (mem_ctx, trap_var) {
            let mut tail = naga::Block::new();
            let trap_index = if compute_invocation_count == 1 {
                lit_u32(&mut func, 0)
            } else {
                let global_argument = physical_builtin_arguments[0]
                    .expect("multi-invocation trap channel requires global invocation id");
                emit_compute_invocation_index(
                    &mut func,
                    &mut tail,
                    global_argument,
                    compute_invocation_extent,
                )
            };
            emit_trap_store(&mut func, &mut tail, mem_ctx, trap_var, trap_index);
            func.body.extend_block(tail);
        }
        naga_mod.entry_points.push(naga::EntryPoint {
            name: "main".into(),
            stage: naga::ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size,
            workgroup_size_overrides: None,
            function: func,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let mut bindings = external_layout_bindings;
        if parameter_var.is_some() {
            bindings.push(SpirvBinding {
                group: 0,
                binding: parameter_binding,
                name: "params".to_string(),
                access: Access::Read,
                role: Role::Input,
                stages: vec![SpirvShaderStage::Compute],
                stride: parameter_span,
                span: parameter_span,
                members: layout_parameter_members,
                resource_element: None,
                resource_length: None,
                resource_arg_index: None,
            });
        }
        if needs_trap_channel {
            bindings.push(SpirvBinding {
                group: 0,
                binding: trap_binding,
                name: "trap".to_string(),
                access: Access::ReadWrite,
                role: Role::Output,
                stages: vec![SpirvShaderStage::Compute],
                stride: 4,
                span: compute_trap_span,
                members: Vec::new(),
                resource_element: None,
                resource_length: None,
                resource_arg_index: None,
            });
        }
        return Ok((
            naga_mod,
            SpirvLayout {
                entry_point: "main".to_string(),
                mode: LayoutMode::Compute,
                workgroup_size,
                word,
                bindings,
                builtin_inputs: builtin_arguments
                    .iter()
                    .map(|argument| SpirvBuiltinInput {
                        arg_index: argument.arg_index,
                        source: argument.source,
                        scalar: SpirvScalarKind::I32,
                    })
                    .collect(),
                result: None,
                trap: needs_trap_channel.then_some(SpirvResult {
                    group: 0,
                    binding: trap_binding,
                    offset: 0,
                    width: compute_trap_span,
                }),
                vertex_entry: None,
                fragment_entry: None,
                color_target_format: None,
            },
        ));
    }

    // ======================================================================
    // Render mode (fork push #3). ONE module, two entry points: a fixed
    // fullscreen-triangle `@vertex` and a `@fragment` that binds args 0,1 to
    // `u32(position.xy)`, runs the SAME mode-blind body translation Grid uses,
    // and returns `unpack4x8unorm(result)` as an `@location(0) vec4<f32>` color.
    // There is NO output storage buffer. Every precondition is enforced here so
    // no wrong shader is ever emitted; Scalar/Grid/Batch never reach this branch
    // and are byte-untouched.
    // ======================================================================
    if render {
        if grid {
            return Err(
                "spirv render: render and grid modes are mutually exclusive".to_string(),
            );
        }
        if word != WordKind::U32 {
            return Err(
                "spirv render: render mode requires the u32 word (browser profile); got i64"
                    .to_string(),
            );
        }
        if param_count < 2 {
            return Err(format!(
                "spirv render: a render kernel needs at least (px, py) args; got {param_count}"
            ));
        }
        if has_mem {
            return Err(
                "spirv render: function-local [u32; N] arrays (MemAllocDynamic/Mload/Mstore) \
                 are unsupported in render mode. Fail closed."
                    .to_string(),
            );
        }
        if has_unreachable {
            // Finding A (2026-08-08): render mode's fragment/vertex functions
            // are a separate naga::Function scope from the compute-mode
            // `func` the trap channel is threaded through below, so a trap
            // here has no `MemCtx` to raise. Rather than build a second,
            // parallel trap-channel for the render path (no current fixture
            // needs it), fail closed and preserve the pin's original
            // behavior for this mode: a trap in render is a named compile
            // error, never a silent zero pixel.
            return Err(
                "spirv render: a trap (Unreachable -- an array-bounds check, a checked-usize \
                 overflow, or a generic MIR trap) is unsupported in render mode (no trap \
                 channel is wired for the vertex/fragment functions). Fail closed."
                    .to_string(),
            );
        }
        if has_obj_alloc {
            return Err(
                "spirv render: render and batch (ObjAlloc) modes are mutually exclusive"
                .to_string(),
            );
        }
        if let Some(resource) = emitted_external_resources
            .iter()
            .find(|resource| resource.access != Access::Read)
        {
            return Err(format!(
                "spirv render: external resource {} must be read-only",
                resource.name
            ));
        }
        if let Some(resource) = emitted_external_resources
            .iter()
            .find(|resource| resource.arg_index < 2)
        {
            return Err(format!(
                "spirv render: external resource {} cannot replace fragment coordinate arg {}",
                resource.name, resource.arg_index
            ));
        }

        // ---- types: f32, vec4<f32> (the vertex/fragment stage I/O) ----------
        let f32_scalar = naga::Scalar { kind: naga::ScalarKind::Float, width: 4 };
        let vec4f = naga_mod.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector { size: naga::VectorSize::Quad, scalar: f32_scalar },
            },
            naga::Span::UNDEFINED,
        );

        // ---- broadcast input struct (args 2..), exactly the Grid shape -------
        // Binding 0 (the compute output buffer) is simply ABSENT in render mode;
        // the input storage buffer stays at @group(0) @binding(1), no renumbering.
        let parameter_args = sig
            .args()
            .iter()
            .copied()
            .enumerate()
            .skip(2)
            .filter(|(index, _)| !resource_arg_indices.contains(index))
            .collect::<Vec<_>>();
        let emit_parameter_binding =
            external_resources.is_empty() || !parameter_args.is_empty();
        let effective_params = parameter_args.len().max(1);
        let mut input_members = Vec::with_capacity(effective_params);
        let mut layout_input_members = Vec::with_capacity(parameter_args.len());
        let mut input_span = 0;
        let mut input_align = 1;
        for (arg_index, ty) in &parameter_args {
            let (naga_ty, width, scalar) = match ty {
                sonatina_ir::Type::I32 => (word_type, 4, SpirvScalarKind::I32),
                sonatina_ir::Type::F32 => (f32_type, 4, SpirvScalarKind::F32),
                _ => return Err(format!("spirv render: broadcast arg {arg_index} has unsupported storage type {ty:?}")),
            };
            input_members.push(naga::StructMember { name: Some(format!("p{arg_index}")), ty: naga_ty, binding: None, offset: input_span });
            layout_input_members.push(SpirvBindingMember { arg_index: *arg_index as u32, offset: input_span, width, scalar });
            input_span += width;
            input_align = input_align.max(width);
        }
        if input_members.is_empty() {
            input_members.push(naga::StructMember { name: Some("padding".into()), ty: word_type, binding: None, offset: 0 });
            input_span = word_width;
        }
        input_span = (input_span + input_align - 1) & !(input_align - 1);
        let parameter_binding = if external_resources.is_empty() {
            1
        } else {
            emitted_external_resources.len() as u32
        };
        let input_var = if emit_parameter_binding {
            let input_struct = naga_mod.types.insert(
                naga::Type {
                    name: Some("Input".into()),
                    inner: naga::TypeInner::Struct { members: input_members, span: input_span },
                },
                naga::Span::UNDEFINED,
            );
            Some(naga_mod.global_variables.append(
                naga::GlobalVariable {
                    name: Some("input".into()),
                    space: naga::AddressSpace::Storage { access: naga::StorageAccess::LOAD },
                    binding: Some(naga::ResourceBinding { group: 0, binding: parameter_binding }),
                    ty: input_struct,
                    init: None,
                    memory_decorations: naga::ir::MemoryDecorations::empty(),
                },
                naga::Span::UNDEFINED,
            ))
        } else {
            None
        };

        // ---- @vertex: fullscreen triangle from vertex_index -----------------
        // vi=0 -> (-1,-1); vi=1 -> (3,-1); vi=2 -> (-1,3). No vertex buffer, no
        // varyings: the fragment reads its pixel from the position builtin.
        //   x = f32((vi & 1u) << 2u) - 1.0 ;  y = f32((vi & 2u) << 1u) - 1.0
        let mut vs = naga::Function {
            name: Some("vs_fullscreen".into()),
            arguments: vec![naga::FunctionArgument {
                name: Some("vi".into()),
                ty: word_type, // u32 (word == U32 here)
                binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::VertexIndex)),
            }],
            result: Some(naga::FunctionResult {
                ty: vec4f,
                binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { invariant: false })),
            }),
            local_variables: naga::Arena::new(),
            expressions: naga::Arena::new(),
            named_expressions: Default::default(),
            body: naga::Block::new(),
            diagnostic_filter_leaf: None,
        };
        {
            let mut b = naga::Block::new();
            // Append an expression and Emit it as its own single-expression range
            // (the house style). Literals stay un-Emitted (const expressions).
            fn e(
                f: &mut naga::Function,
                body: &mut naga::Block,
                x: naga::Expression,
            ) -> naga::Handle<naga::Expression> {
                let h = f.expressions.append(x, naga::Span::UNDEFINED);
                body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(h, h)),
                    naga::Span::UNDEFINED,
                );
                h
            }
            let vi = vs.expressions.append(naga::Expression::FunctionArgument(0), naga::Span::UNDEFINED);
            let one = vs.expressions.append(naga::Expression::Literal(naga::Literal::U32(1)), naga::Span::UNDEFINED);
            let two = vs.expressions.append(naga::Expression::Literal(naga::Literal::U32(2)), naga::Span::UNDEFINED);
            let one_f = vs.expressions.append(naga::Expression::Literal(naga::Literal::F32(1.0)), naga::Span::UNDEFINED);
            let zero_f = vs.expressions.append(naga::Expression::Literal(naga::Literal::F32(0.0)), naga::Span::UNDEFINED);

            let xa = e(&mut vs, &mut b, naga::Expression::Binary { op: naga::BinaryOperator::And, left: vi, right: one });
            let xs = e(&mut vs, &mut b, naga::Expression::Binary { op: naga::BinaryOperator::ShiftLeft, left: xa, right: two });
            let xf = e(&mut vs, &mut b, naga::Expression::As { expr: xs, kind: naga::ScalarKind::Float, convert: Some(4) });
            let x = e(&mut vs, &mut b, naga::Expression::Binary { op: naga::BinaryOperator::Subtract, left: xf, right: one_f });

            let ya = e(&mut vs, &mut b, naga::Expression::Binary { op: naga::BinaryOperator::And, left: vi, right: two });
            let ys = e(&mut vs, &mut b, naga::Expression::Binary { op: naga::BinaryOperator::ShiftLeft, left: ya, right: one });
            let yf = e(&mut vs, &mut b, naga::Expression::As { expr: ys, kind: naga::ScalarKind::Float, convert: Some(4) });
            let y = e(&mut vs, &mut b, naga::Expression::Binary { op: naga::BinaryOperator::Subtract, left: yf, right: one_f });

            let pos = e(&mut vs, &mut b, naga::Expression::Compose { ty: vec4f, components: vec![x, y, zero_f, one_f] });
            b.push(naga::Statement::Return { value: Some(pos) }, naga::Span::UNDEFINED);
            vs.body = b;
        }

        // ---- @fragment: position builtin in, vec4<f32> color out ------------
        let mut fs = naga::Function {
            name: Some("fs_main".into()),
            arguments: vec![naga::FunctionArgument {
                name: Some("pos".into()),
                ty: vec4f,
                binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { invariant: false })),
            }],
            result: Some(naga::FunctionResult {
                ty: vec4f,
                binding: Some(naga::Binding::Location {
                    location: 0,
                    interpolation: None,
                    sampling: None,
                    blend_src: None,
                    per_primitive: false,
                }),
            }),
            local_variables: naga::Arena::new(),
            expressions: naga::Arena::new(),
            named_expressions: Default::default(),
            body: naga::Block::new(),
            diagnostic_filter_leaf: None,
        };

        let mut result_expr = None;
        let mut body_error = None;
        {
            module.func_store.try_view(first_func, |function| {
                let inst_set = function.inst_set();
                let mut value_map: HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>> =
                    HashMap::new();
                let mut phi_locals: HashMap<
                    sonatina_ir::ValueId,
                    naga::Handle<naga::LocalVariable>,
                > = HashMap::new();

                for &(arg_index, global) in &external_roots {
                    let Some(&arg_value) = function.arg_values.get(arg_index as usize) else {
                        body_error = Some(format!(
                            "spirv render: external resource arg {arg_index} disappeared during lowering"
                        ));
                        return;
                    };
                    let root = fs.expressions.append(
                        naga::Expression::GlobalVariable(global),
                        naga::Span::UNDEFINED,
                    );
                    value_map.insert(arg_value, root);
                }
                let resource_seeds = external_roots
                    .iter()
                    .filter_map(|(arg_index, _)| {
                        function.arg_values.get(*arg_index as usize).copied()
                    })
                    .collect::<Vec<_>>();
                if let Err(error) = bind_resource_identity_aliases(
                    function,
                    resource_seeds,
                    &helper_resource_capabilities,
                    &helper_logical_result_abis,
                    &mut value_map,
                ) {
                    body_error = Some(error);
                    return;
                }

                let input_expr = input_var.map(|input_var| {
                    fs.expressions.append(
                        naga::Expression::GlobalVariable(input_var),
                        naga::Span::UNDEFINED,
                    )
                });

                // Prologue: px = u32(pos.x), py = u32(pos.y). The fragment center
                // (px + 0.5) truncates to the pixel index, the render-mode analog
                // of Grid's gid.x/gid.y binding (same orientation: y=0 top row).
                let pos = fs.expressions.append(
                    naga::Expression::FunctionArgument(0),
                    naga::Span::UNDEFINED,
                );
                let posx = fs.expressions.append(
                    naga::Expression::AccessIndex { base: pos, index: 0 },
                    naga::Span::UNDEFINED,
                );
                fs.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(posx, posx)),
                    naga::Span::UNDEFINED,
                );
                let px = fs.expressions.append(
                    naga::Expression::As { expr: posx, kind: naga::ScalarKind::Uint, convert: Some(4) },
                    naga::Span::UNDEFINED,
                );
                fs.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(px, px)),
                    naga::Span::UNDEFINED,
                );
                let posy = fs.expressions.append(
                    naga::Expression::AccessIndex { base: pos, index: 1 },
                    naga::Span::UNDEFINED,
                );
                fs.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(posy, posy)),
                    naga::Span::UNDEFINED,
                );
                let py = fs.expressions.append(
                    naga::Expression::As { expr: posy, kind: naga::ScalarKind::Uint, convert: Some(4) },
                    naga::Span::UNDEFINED,
                );
                fs.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(py, py)),
                    naga::Span::UNDEFINED,
                );
                if let Some(&a0) = function.arg_values.first() {
                    value_map.insert(a0, px);
                }
                if let Some(&a1) = function.arg_values.get(1) {
                    value_map.insert(a1, py);
                }

                // Ordinary args load from the parameter record. Resource args
                // are already rooted as globals above and never enter params.
                for (member_index, (arg_index, _)) in parameter_args.iter().enumerate() {
                    let field = fs.expressions.append(
                        naga::Expression::AccessIndex {
                            base: input_expr.expect("parameter args require input binding"),
                            index: member_index as u32,
                        },
                        naga::Span::UNDEFINED,
                    );
                    let loaded = fs.expressions.append(
                        naga::Expression::Load { pointer: field },
                        naga::Span::UNDEFINED,
                    );
                    fs.body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(field, field)),
                        naga::Span::UNDEFINED,
                    );
                    fs.body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)),
                        naga::Span::UNDEFINED,
                    );
                    value_map.insert(function.arg_values[*arg_index], loaded);
                }

                // The mode-blind body: SAME structurizer + region emission Grid and
                // Scalar use (zero changes to emit_naga_regions / emit_single_inst).
                let scfg = match crate::structurize::structurize_function(function) {
                    Ok(scfg) => scfg,
                    Err(err) => {
                        body_error = Some(structurize_error_with_block_ir(
                            err, first_func, function,
                        ));
                        return;
                    }
                };
                if let Err(err) = emit_naga_regions(
                    function, inst_set, word, &scfg.regions, word_type, f32_type, bool_type,
                    &mut fs, &mut value_map, &mut phi_locals, &mut result_expr,
                    None, &naga_functions,
                    // Render mode fails closed on has_mem (checked above), so
                    // there is never a heap/trap context here.
                    None,
                ) {
                    body_error = Some(structurize_error_with_block_ir(
                        err, first_func, function,
                    ));
                }
            });
        }

        if let Some(err) = body_error { return Err(err); }

        // Epilogue: unpack4x8unorm(result) -> @location(0) vec4<f32>. A render
        // kernel that produced no result expression is a hard error, never a
        // silent store-0 (that fallback is scalar-only). The packed u32 is r|g<<8|
        // b<<16|a<<24, which unpack4x8unorm maps to the exact rgba8unorm color.
        let result_val = result_expr
            .ok_or_else(|| "spirv render: fragment kernel produced no result expression".to_string())?;
        let color = fs.expressions.append(
            naga::Expression::Math {
                fun: naga::MathFunction::Unpack4x8unorm,
                arg: result_val,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            naga::Span::UNDEFINED,
        );
        fs.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(color, color)),
            naga::Span::UNDEFINED,
        );
        fs.body.push(naga::Statement::Return { value: Some(color) }, naga::Span::UNDEFINED);

        // Two entry points into ONE module (spv-out with pipeline_options=None
        // writes both). Non-compute stages carry workgroup_size = [0,0,0].
        naga_mod.entry_points.push(naga::EntryPoint {
            name: "vs_fullscreen".into(),
            stage: naga::ShaderStage::Vertex,
            early_depth_test: None,
            workgroup_size: [0, 0, 0],
            workgroup_size_overrides: None,
            function: vs,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        naga_mod.entry_points.push(naga::EntryPoint {
            name: "fs_main".into(),
            stage: naga::ShaderStage::Fragment,
            early_depth_test: None,
            workgroup_size: [0, 0, 0],
            workgroup_size_overrides: None,
            function: fs,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        // The compiler states its own render ABI: mode Render, the two entry
        // names, the color-target format, the single input binding (@0/1, Read),
        // no output binding (binding 0 absent), no single-slot result.
        let layout = SpirvLayout {
            entry_point: "fs_main".to_string(),
            mode: LayoutMode::Render,
            workgroup_size: [0, 0, 0],
            word,
            bindings: {
                let mut bindings = external_layout_bindings;
                if emit_parameter_binding {
                    bindings.push(SpirvBinding {
                        group: 0,
                        binding: parameter_binding,
                        name: "input".to_string(),
                        access: Access::Read,
                        role: Role::Input,
                        stages: vec![SpirvShaderStage::Fragment],
                        stride: input_span,
                        span: input_span,
                        members: layout_input_members,
                        resource_element: None,
                        resource_length: None,
                        resource_arg_index: None,
                    });
                }
                bindings
            },
            builtin_inputs: vec![
                SpirvBuiltinInput { arg_index: 0, source: SpirvBuiltinSource::FragmentPositionX, scalar: SpirvScalarKind::I32 },
                SpirvBuiltinInput { arg_index: 1, source: SpirvBuiltinSource::FragmentPositionY, scalar: SpirvScalarKind::I32 },
            ],
            result: None,
            // Render mode fails closed on has_mem (checked above): never a
            // trap channel here.
            trap: None,
            vertex_entry: Some("vs_fullscreen".to_string()),
            fragment_entry: Some("fs_main".to_string()),
            color_target_format: Some("rgba8unorm".to_string()),
        };

        return Ok((naga_mod, layout));
    }

    // Output type: dynamic array for batch (ObjAlloc) or grid (per-pixel store),
    // or a single-value struct for scalar.
    let output_type = if has_obj_alloc || grid {
        naga_mod.types.insert(
            naga::Type {
                name: Some("OutputArray".into()),
                inner: naga::TypeInner::Array {
                    base: word_type,
                    size: naga::ArraySize::Dynamic,
                    stride: word_width,
                },
            },
            naga::Span::UNDEFINED,
        )
    } else {
        naga_mod.types.insert(
            naga::Type {
                name: Some("Output".into()),
                inner: naga::TypeInner::Struct {
                    members: vec![naga::StructMember {
                        name: Some("result".into()),
                        ty: word_type,
                        binding: None,
                        offset: 0,
                    }],
                    span: word_width,
                },
            },
            naga::Span::UNDEFINED,
        )
    };

    // In grid mode the first two args are the grid coordinates (delivered as
    // builtins, not loaded from the input buffer), so they are excluded from the
    // broadcast input struct. Args 2.. are the shared broadcast params.
    let broadcast = if grid { param_count - 2 } else { param_count };
    let effective_params = broadcast.max(1);
    let mut input_members = Vec::with_capacity(effective_params);
    let mut layout_input_members = Vec::with_capacity(broadcast);
    let mut input_span = 0;
    let mut input_align = 1;
    for (i, ty) in sig.args().iter().skip(if grid { 2 } else { 0 }).copied().enumerate() {
        let (naga_ty, width, scalar) = match ty {
            sonatina_ir::Type::I32 => (word_type, 4, SpirvScalarKind::I32),
            sonatina_ir::Type::I64 if word == WordKind::I64 => (word_type, 8, SpirvScalarKind::I64),
            sonatina_ir::Type::F32 => (f32_type, 4, SpirvScalarKind::F32),
            _ => return Err(format!("spirv: input arg {i} has unsupported storage type {ty:?}")),
        };
        input_span = (input_span + width - 1) & !(width - 1);
        input_members.push(naga::StructMember { name: Some(format!("p{i}")), ty: naga_ty, binding: None, offset: input_span });
        layout_input_members.push(SpirvBindingMember {
            arg_index: (i + if grid { 2 } else { 0 }) as u32,
            offset: input_span,
            width,
            scalar,
        });
        input_span += width;
        input_align = input_align.max(width);
    }
    if input_members.is_empty() {
        input_members.push(naga::StructMember { name: Some("padding".into()), ty: word_type, binding: None, offset: 0 });
        input_span = word_width;
    }
    input_span = (input_span + input_align - 1) & !(input_align - 1);

    let input_struct = naga_mod.types.insert(
        naga::Type {
            name: Some("Input".into()),
            inner: naga::TypeInner::Struct {
                members: input_members,
                span: input_span,
            },
        },
        naga::Span::UNDEFINED,
    );

    // For batch mode: input is an array of structs, one per invocation
    let input_type = if has_obj_alloc {
        naga_mod.types.insert(
            naga::Type {
                name: Some("InputArray".into()),
                inner: naga::TypeInner::Array {
                    base: input_struct,
                    size: naga::ArraySize::Dynamic,
                    stride: input_span,
                },
            },
            naga::Span::UNDEFINED,
        )
    } else {
        input_struct
    };

    let output_var = naga_mod.global_variables.append(
        naga::GlobalVariable {
            name: Some("output".into()),
            space: naga::AddressSpace::Storage {
                access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
            },
            binding: Some(naga::ResourceBinding { group: 0, binding: 0 }),
            ty: output_type,
            init: None,
            memory_decorations: naga::ir::MemoryDecorations::empty(),
        },
        naga::Span::UNDEFINED,
    );

    let input_var = naga_mod.global_variables.append(
        naga::GlobalVariable {
            name: Some("input".into()),
            space: naga::AddressSpace::Storage {
                access: naga::StorageAccess::LOAD,
            },
            binding: Some(naga::ResourceBinding { group: 0, binding: 1 }),
            ty: input_type,
            init: None,
            memory_decorations: naga::ir::MemoryDecorations::empty(),
        },
        naga::Span::UNDEFINED,
    );

    // Review finding 3 (poison-sentinel collision): an externally visible,
    // per-invocation trap-status output, declared whenever the function can
    // reach a poison path at all (has_mem, has_unreachable, or both --
    // Finding A, 2026-08-08: a no-Mem trapping function needs this exactly
    // as much as an array kernel does). A dynamic `array<u32>` in both
    // Scalar (written at index 0) and Grid (written per-pixel, same
    // `linear` index as the result) modes, so a consumer never has to infer
    // failure from an in-band magic result value -- it reads this word
    // instead. `0` = completed without a guard firing; `1` = `fe_trapped`
    // was set (heap exhaustion, a misaligned access, or ANY trap reached).
    let trap_var = if needs_trap_channel {
        let trap_array_ty = naga_mod.types.insert(
            naga::Type {
                name: Some("TrapArray".into()),
                inner: naga::TypeInner::Array {
                    base: word_type,
                    size: naga::ArraySize::Dynamic,
                    stride: word_width,
                },
            },
            naga::Span::UNDEFINED,
        );
        Some(naga_mod.global_variables.append(
            naga::GlobalVariable {
                name: Some("trap".into()),
                space: naga::AddressSpace::Storage {
                    access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
                },
                binding: Some(naga::ResourceBinding { group: 0, binding: 2 }),
                ty: trap_array_ty,
                init: None,
                memory_decorations: naga::ir::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        ))
    } else {
        None
    };

    // u32 vec3 type for global_invocation_id
    let vec3_u32_type = naga_mod.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Vector {
                size: naga::VectorSize::Tri,
                scalar: naga::Scalar { kind: naga::ScalarKind::Uint, width: 4 },
            },
        },
        naga::Span::UNDEFINED,
    );

    // Build the entry point function. Batch and grid modes both take the grid id
    // as FunctionArgument(0). Grid additionally takes num_workgroups as
    // FunctionArgument(1): the per-pixel store derives the row width as
    // num_workgroups.x * workgroup_size[0] (the dispatched width), so W is never a
    // kernel parameter.
    let arguments = if grid {
        vec![
            naga::FunctionArgument {
                name: Some("global_id".into()),
                ty: vec3_u32_type,
                binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)),
            },
            naga::FunctionArgument {
                name: Some("num_workgroups".into()),
                ty: vec3_u32_type,
                binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::NumWorkGroups)),
            },
        ]
    } else if has_obj_alloc {
        vec![naga::FunctionArgument {
            name: Some("global_id".into()),
            ty: vec3_u32_type,
            binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)),
        }]
    } else {
        vec![]
    };

    let mut func = naga::Function {
        name: Some("main".into()),
        arguments,
        result: None,
        local_variables: naga::Arena::new(),
        expressions: naga::Arena::new(),
        named_expressions: Default::default(),
        body: naga::Block::new(),
        diagnostic_filter_leaf: None,
    };

    // Private-storage heap emulation locals (RUNG3_SPIRV_ARRAYS_DESIGN.md
    // section 2). `fe_heap`/`fe_bump` are declared ONLY for has_mem kernels
    // so kernels with no Mem ops stay byte-identical to before this rung
    // landed. `fe_trapped` is declared whenever `needs_trap_channel`: an
    // entry or reachable helper that can trap needs the shared channel, or it
    // silently falls through to a zero/uninitialized result the same way the
    // original review finding 4 did.
    let mem_ctx = if needs_trap_channel {
        let heap = if has_mem {
            let heap_ty = private_heap_type
                .expect("has_mem establishes one shared private heap type");
            // Explicit ZeroValue init is load-bearing: it matches wasm's
            // zero-initialized arena, and removes a WGSL/SPIR-V divergence --
            // WGSL zero-inits `var<function>` implicitly; SPIR-V without an
            // initializer does not (`RUNG3_SPIRV_ARRAYS_DESIGN.md` section 2).
            let heap_zero = func.expressions.append(naga::Expression::ZeroValue(heap_ty), naga::Span::UNDEFINED);
            let heap_local = func.local_variables.append(
                naga::LocalVariable { name: Some("fe_heap".into()), ty: heap_ty, init: Some(heap_zero) },
                naga::Span::UNDEFINED,
            );
            let bump_zero = func.expressions.append(naga::Expression::Literal(naga::Literal::U32(0)), naga::Span::UNDEFINED);
            let bump = func.local_variables.append(
                naga::LocalVariable { name: Some("fe_bump".into()), ty: word_type, init: Some(bump_zero) },
                naga::Span::UNDEFINED,
            );
            let heap = func.expressions.append(
                naga::Expression::LocalVariable(heap_local),
                naga::Span::UNDEFINED,
            );
            let bump = func.expressions.append(
                naga::Expression::LocalVariable(bump),
                naga::Span::UNDEFINED,
            );
            Some(HeapCtx {
                heap,
                bump,
                word_type,
                heap_words: private_heap_words,
            })
        } else {
            None
        };
        let trapped_false = func.expressions.append(naga::Expression::Literal(naga::Literal::Bool(false)), naga::Span::UNDEFINED);
        let trapped = func.local_variables.append(
            naga::LocalVariable { name: Some("fe_trapped".into()), ty: bool_type, init: Some(trapped_false) },
            naga::Span::UNDEFINED,
        );
        let trapped = func.expressions.append(
            naga::Expression::LocalVariable(trapped),
            naga::Span::UNDEFINED,
        );
        Some(MemCtx { heap, trapped })
    } else {
        None
    };

    // Translate the selected Sonatina entry.
    let mut result_expr = None;
    // In grid mode, the gid.x / gid.y expressions bound to args 0,1 are emitted
    // inside the body closure but reused by the per-pixel store that follows it,
    // so their handles flow out here.
    let mut grid_gid: Option<(naga::Handle<naga::Expression>, naga::Handle<naga::Expression>)> =
        None;

    let mut body_error = None;
    {
        module.func_store.try_view(first_func, |function| {
            let inst_set = function.inst_set();
            let mut value_map: HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>> =
                HashMap::new();
            // Map phi values to LocalVariable handles for store/load in loops
            let mut phi_locals: HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>> =
                HashMap::new();

            // For batch mode (ObjAlloc), inject the output buffer into the value_map
            if has_obj_alloc {
                let output_expr = func.expressions.append(
                    naga::Expression::GlobalVariable(output_var),
                    naga::Span::UNDEFINED,
                );
                value_map.insert(sonatina_ir::ValueId(u32::MAX), output_expr);
            }

            // Load function args from input buffer
            let input_global = func.expressions.append(
                naga::Expression::GlobalVariable(input_var),
                naga::Span::UNDEFINED,
            );

            // In batch mode, index into the input array with global_invocation_id.x
            let input_expr = if has_obj_alloc {
                let gid_u32 = func.expressions.append(
                    naga::Expression::FunctionArgument(0),
                    naga::Span::UNDEFINED,
                );
                let gid_x = func.expressions.append(
                    naga::Expression::AccessIndex { base: gid_u32, index: 0 },
                    naga::Span::UNDEFINED,
                );
                func.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(gid_x, gid_x)),
                    naga::Span::UNDEFINED,
                );
                // Cast u32 to i32 for Access index
                let gid_i32 = func.expressions.append(
                    naga::Expression::As {
                        expr: gid_x,
                        kind: naga::ScalarKind::Sint,
                        convert: Some(4),
                    },
                    naga::Span::UNDEFINED,
                );
                func.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(gid_i32, gid_i32)),
                    naga::Span::UNDEFINED,
                );
                // input[gid.x] -> pointer to InputStruct for this invocation
                func.expressions.append(
                    naga::Expression::Access { base: input_global, index: gid_i32 },
                    naga::Span::UNDEFINED,
                )
            } else {
                input_global
            };

            if grid {
                // Grid mode: args 0,1 are the grid coordinates, bound castless to
                // global_invocation_id.x / .y (both u32, the check-1 guarantee).
                // Args 2.. load from the broadcast input struct at member idx - 2.
                let gid = func.expressions.append(
                    naga::Expression::FunctionArgument(0),
                    naga::Span::UNDEFINED,
                );
                let gid_x = func.expressions.append(
                    naga::Expression::AccessIndex { base: gid, index: 0 },
                    naga::Span::UNDEFINED,
                );
                func.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(gid_x, gid_x)),
                    naga::Span::UNDEFINED,
                );
                let gid_y = func.expressions.append(
                    naga::Expression::AccessIndex { base: gid, index: 1 },
                    naga::Span::UNDEFINED,
                );
                func.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(gid_y, gid_y)),
                    naga::Span::UNDEFINED,
                );
                if let Some(&a0) = function.arg_values.first() {
                    value_map.insert(a0, gid_x);
                }
                if let Some(&a1) = function.arg_values.get(1) {
                    value_map.insert(a1, gid_y);
                }
                grid_gid = Some((gid_x, gid_y));

                for (idx, &arg_val) in function.arg_values.iter().enumerate().skip(2) {
                    let field = func.expressions.append(
                        naga::Expression::AccessIndex {
                            base: input_expr,
                            index: (idx - 2) as u32,
                        },
                        naga::Span::UNDEFINED,
                    );
                    let loaded = func.expressions.append(
                        naga::Expression::Load { pointer: field },
                        naga::Span::UNDEFINED,
                    );
                    func.body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(field, field)),
                        naga::Span::UNDEFINED,
                    );
                    func.body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)),
                        naga::Span::UNDEFINED,
                    );
                    value_map.insert(arg_val, loaded);
                }
            } else {
                for (idx, &arg_val) in function.arg_values.iter().enumerate() {
                    let field = func.expressions.append(
                        naga::Expression::AccessIndex {
                            base: input_expr,
                            index: idx as u32,
                        },
                        naga::Span::UNDEFINED,
                    );
                    let loaded = func.expressions.append(
                        naga::Expression::Load { pointer: field },
                        naga::Span::UNDEFINED,
                    );
                    // Emit AccessIndex and Load individually to avoid range
                    // overlap issues when there are 3+ parameters
                    func.body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(field, field)),
                        naga::Span::UNDEFINED,
                    );
                    func.body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)),
                        naga::Span::UNDEFINED,
                    );
                    value_map.insert(arg_val, loaded);
                }
            }

            let scfg = match crate::structurize::structurize_function(function) {
                Ok(scfg) => scfg,
                Err(err) => {
                    body_error = Some(structurize_error_with_block_ir(
                        err, first_func, function,
                    ));
                    return;
                }
            };
            if let Err(err) = emit_naga_regions(
                function, inst_set, word, &scfg.regions, word_type, f32_type, bool_type,
                &mut func, &mut value_map, &mut phi_locals, &mut result_expr,
                None, &naga_functions, mem_ctx,
            ) {
                body_error = Some(structurize_error_with_block_ir(
                    err, first_func, function,
                ));
            }
        });
    }

    if let Some(err) = body_error { return Err(err); }

    // Result store, three-way by mode:
    //  - Scalar: store the single result into the output struct (store-0 fallback).
    //  - Batch (ObjAlloc): ObjStore already wrote to the buffer, nothing to do.
    //  - Grid: store the result at output[gid.y * (num_workgroups.x * wgx) + gid.x].
    if grid {
        // A grid kernel that produced no result expression is a hard error, never
        // a silent store-0 (that fallback is scalar-only).
        let result_val = result_expr.ok_or_else(|| {
            "spirv grid: kernel produced no result expression".to_string()
        })?;
        let (gid_x, gid_y) = grid_gid.ok_or_else(|| {
            "spirv grid: grid coordinate expressions were not bound".to_string()
        })?;

        // row_width = num_workgroups.x * workgroup_size[0] = the dispatched width.
        let wgx = func.expressions.append(
            naga::Expression::Literal(naga::Literal::U32(workgroup_size[0])),
            naga::Span::UNDEFINED,
        );
        let nwg = func.expressions.append(
            naga::Expression::FunctionArgument(1),
            naga::Span::UNDEFINED,
        );
        let nwg_x = func.expressions.append(
            naga::Expression::AccessIndex { base: nwg, index: 0 },
            naga::Span::UNDEFINED,
        );
        func.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(nwg_x, nwg_x)),
            naga::Span::UNDEFINED,
        );
        let row_width = func.expressions.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: nwg_x,
                right: wgx,
            },
            naga::Span::UNDEFINED,
        );
        func.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(row_width, row_width)),
            naga::Span::UNDEFINED,
        );
        let y_off = func.expressions.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: gid_y,
                right: row_width,
            },
            naga::Span::UNDEFINED,
        );
        func.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(y_off, y_off)),
            naga::Span::UNDEFINED,
        );
        let linear = func.expressions.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: y_off,
                right: gid_x,
            },
            naga::Span::UNDEFINED,
        );
        func.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(linear, linear)),
            naga::Span::UNDEFINED,
        );
        // As Sint index cast, mirroring the proven ObjIndex convention.
        let idx_i32 = func.expressions.append(
            naga::Expression::As {
                expr: linear,
                kind: naga::ScalarKind::Sint,
                convert: Some(4),
            },
            naga::Span::UNDEFINED,
        );
        func.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(idx_i32, idx_i32)),
            naga::Span::UNDEFINED,
        );
        let output_expr = func.expressions.append(
            naga::Expression::GlobalVariable(output_var),
            naga::Span::UNDEFINED,
        );
        // Access returns a pointer, so no Emit is needed.
        let ptr = func.expressions.append(
            naga::Expression::Access { base: output_expr, index: idx_i32 },
            naga::Span::UNDEFINED,
        );
        func.body.push(
            naga::Statement::Store { pointer: ptr, value: result_val },
            naga::Span::UNDEFINED,
        );
        if let (Some(mem_ctx), Some(trap_var)) = (mem_ctx, trap_var) {
            let mut trap_tail = naga::Block::new();
            emit_trap_store(&mut func, &mut trap_tail, mem_ctx, trap_var, idx_i32);
            func.body.extend_block(trap_tail);
        }
    } else if !has_obj_alloc {
        let output_expr = func.expressions.append(
            naga::Expression::GlobalVariable(output_var),
            naga::Span::UNDEFINED,
        );
        let result_field = func.expressions.append(
            naga::Expression::AccessIndex { base: output_expr, index: 0 },
            naga::Span::UNDEFINED,
        );

        let final_val = result_expr.unwrap_or_else(|| {
            let zero = match word {
                WordKind::U32 => naga::Literal::U32(0),
                WordKind::I64 => naga::Literal::I64(0),
            };
            func.expressions.append(
                naga::Expression::Literal(zero),
                naga::Span::UNDEFINED,
            )
        });

        func.body.push(
            naga::Statement::Store { pointer: result_field, value: final_val },
            naga::Span::UNDEFINED,
        );
        if let (Some(mem_ctx), Some(trap_var)) = (mem_ctx, trap_var) {
            let zero_idx = func.expressions.append(naga::Expression::Literal(naga::Literal::I32(0)), naga::Span::UNDEFINED);
            let mut trap_tail = naga::Block::new();
            emit_trap_store(&mut func, &mut trap_tail, mem_ctx, trap_var, zero_idx);
            func.body.extend_block(trap_tail);
        }
    }

    naga_mod.entry_points.push(naga::EntryPoint {
        name: "main".into(),
        stage: naga::ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size,
        workgroup_size_overrides: None,
        function: func,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    // The compiler states its own ABI, populated from the SAME values used above:
    // the two storage globals (output @0/0 LOAD|STORE, input @0/1 LOAD), the word
    // width, and the workgroup size passed to the entry point. Nothing downstream
    // re-derives this.
    let layout = SpirvLayout {
        entry_point: "main".to_string(),
        mode: if grid {
            LayoutMode::Grid
        } else if has_obj_alloc {
            LayoutMode::Batch
        } else {
            LayoutMode::Scalar
        },
        workgroup_size,
        word,
        bindings: {
            let mut bindings = vec![
                SpirvBinding {
                    group: 0,
                    binding: 0,
                    name: "output".to_string(),
                    access: Access::ReadWrite,
                    role: Role::Output,
                    stages: vec![SpirvShaderStage::Compute],
                    stride: word_width,
                    span: word_width,
                    members: Vec::new(),
                    resource_element: None,
                    resource_length: None,
                    resource_arg_index: None,
                },
                SpirvBinding {
                    group: 0,
                    binding: 1,
                    name: "input".to_string(),
                    access: Access::Read,
                    role: Role::Input,
                    stages: vec![SpirvShaderStage::Compute],
                    stride: input_span,
                    span: input_span,
                    members: layout_input_members,
                    resource_element: None,
                    resource_length: None,
                    resource_arg_index: None,
                },
            ];
            if needs_trap_channel {
                bindings.push(SpirvBinding {
                    group: 0,
                    binding: 2,
                    name: "trap".to_string(),
                    access: Access::ReadWrite,
                    role: Role::Output,
                    stages: vec![SpirvShaderStage::Compute],
                    stride: word_width,
                    span: word_width,
                    members: Vec::new(),
                    resource_element: None,
                    resource_length: None,
                    resource_arg_index: None,
                });
            }
            bindings
        },
        builtin_inputs: if grid {
            vec![
                SpirvBuiltinInput { arg_index: 0, source: SpirvBuiltinSource::GlobalInvocationIdX, scalar: SpirvScalarKind::I32 },
                SpirvBuiltinInput { arg_index: 1, source: SpirvBuiltinSource::GlobalInvocationIdY, scalar: SpirvScalarKind::I32 },
            ]
        } else {
            Vec::new()
        },
        // Grid mode has no single readback slot: the whole output array is the
        // result, written per pixel.
        result: if grid {
            None
        } else {
            Some(SpirvResult {
                group: 0,
                binding: 0,
                offset: 0,
                width: word_width,
            })
        },
        // Same "no single slot" reasoning as `result`: Grid writes `trap` per
        // pixel (the whole binding is the answer), Scalar writes one word at
        // index 0. `None` when the kernel can never reach a poison path (no
        // Mem ops AND no Unreachable -- `needs_trap_channel` is false, so no
        // `trap` binding was declared at all).
        trap: if needs_trap_channel && !grid {
            Some(SpirvResult { group: 0, binding: 2, offset: 0, width: word_width })
        } else {
            None
        },
        // Compute modes (Scalar/Grid/Batch) have no vertex/fragment stages and no
        // color target.
        vertex_entry: None,
        fragment_entry: None,
        color_target_format: None,
    };

    Ok((naga_mod, layout))
}

#[cfg(all(test, feature = "spirv-backend"))]
mod tests {
    use super::{MAX_WGSL_FUNCTION_PARAMETERS, validate_naga_portable_wgsl_limits};

    fn naga_function_with_parameters(
        name: &str,
        count: usize,
        ty: naga::Handle<naga::Type>,
    ) -> naga::Function {
        let mut function = naga::Function {
            name: Some(name.to_string()),
            ..Default::default()
        };
        function.arguments = (0..count)
            .map(|index| naga::FunctionArgument {
                name: Some(format!("a{index}")),
                ty,
                binding: None,
            })
            .collect();
        function
    }

    #[test]
    fn final_naga_module_enforces_portable_wgsl_parameter_limit() {
        let mut module = naga::Module::default();
        let word = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar {
                    kind: naga::ScalarKind::Uint,
                    width: 4,
                }),
            },
            naga::Span::UNDEFINED,
        );
        module.functions.append(
            naga_function_with_parameters(
                "at_the_limit",
                MAX_WGSL_FUNCTION_PARAMETERS,
                word,
            ),
            naga::Span::UNDEFINED,
        );
        validate_naga_portable_wgsl_limits(&module)
            .expect("the portable WGSL parameter limit must be inclusive");

        module.functions.append(
            naga_function_with_parameters(
                "over_the_limit",
                MAX_WGSL_FUNCTION_PARAMETERS + 1,
                word,
            ),
            naga::Span::UNDEFINED,
        );
        let error = validate_naga_portable_wgsl_limits(&module)
            .expect_err("the final physical module must reject an over-wide function");
        assert!(
            error.contains("over_the_limit")
                && error.contains("256 physical parameters")
                && error.contains("limit of 255"),
            "unexpected conformance error: {error}",
        );
    }
}

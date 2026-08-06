//! SPIR-V backend: Sonatina IR → SPIR-V compute shader modules via Naga.
//!
//! Translates Sonatina IR to Naga's expression DAG + statement tree IR,
//! then Naga emits SPIR-V. Optionally produces WGSL for debugging.

use sonatina_ir::Module;

use crate::backend::Backend;

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

/// Physical shader builtin supplying a logical source-language argument without
/// buffer storage. [`SpirvBuiltinInput::scalar`] records the logical type after
/// the backend's builtin conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpirvBuiltinSource {
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    FragmentPositionX,
    FragmentPositionY,
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
    /// Render mode: the `@vertex` entry point name (`None` for compute modes).
    pub vertex_entry: Option<String>,
    /// Render mode: the `@fragment` entry point name (`None` for compute modes).
    pub fragment_entry: Option<String>,
    /// Render mode: the color-target texture format the fragment writes
    /// (`Some("rgba8unorm")`); `None` for compute modes.
    pub color_target_format: Option<String>,
}

pub struct SpirvArtifact {
    pub words: Vec<u32>,
    /// WGSL source for wgpu execution (available when spirv-backend feature is on)
    pub wgsl: Option<String>,
    /// The compiler-stated ABI: entry point, mode, workgroup size, word kind,
    /// bindings and result location. Emitted from the same values the naga module
    /// was built from.
    pub layout: SpirvLayout,
}

impl SpirvArtifact {
    pub fn as_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

pub struct SpirvBackend {
    pub workgroup_size: [u32; 3],
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
}

impl SpirvBackend {
    pub fn new() -> Self {
        Self {
            workgroup_size: [64, 1, 1],
            grid: false,
            render: false,
        }
    }

    pub fn with_workgroup_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroup_size = [x, y, z];
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
}

impl Backend for SpirvBackend {
    type Artifact = SpirvArtifact;
    type Error = SpirvError;

    #[cfg(not(feature = "spirv-backend"))]
    fn compile_module(&self, _module: &Module) -> Result<Self::Artifact, Vec<Self::Error>> {
        Err(vec![SpirvError::Translation(
            "SPIR-V backend requires the spirv-backend feature".to_string(),
        )])
    }

    #[cfg(feature = "spirv-backend")]
    fn compile_module(&self, module: &Module) -> Result<Self::Artifact, Vec<Self::Error>> {
        let (naga_mod, layout) =
            translate_to_naga(module, self.workgroup_size, self.grid, self.render)
                .map_err(|e| vec![SpirvError::Translation(e)])?;

        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&naga_mod)
        .map_err(|e| vec![SpirvError::Validation(format!("{e:?}"))])?;

        let options = naga::back::spv::Options {
            lang_version: (1, 5),
            flags: naga::back::spv::WriterFlags::empty(),
            ..Default::default()
        };

        let words = naga::back::spv::write_vec(&naga_mod, &info, &options, None)
            .map_err(|e| vec![SpirvError::Translation(format!("{e}"))])?;

        // Also emit WGSL for wgpu execution
        let wgsl = naga::back::wgsl::write_string(
            &naga_mod, &info, naga::back::wgsl::WriterFlags::empty()
        ).ok();

        Ok(SpirvArtifact { words, wgsl, layout })
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
        <&sonatina_ir::inst::arith::Fadd as InstDowncast>::downcast(inst_set, inst_data).map(|i| (*i.lhs(), *i.rhs(), naga::BinaryOperator::Add))
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
    } else if let Some(sar) = <&sonatina_ir::inst::arith::Sar as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let val = resolve_naga_value(*sar.value(), function, word, value_map, phi_locals, func).unwrap();
            let shift_amount = if let Some(imm) = function.dfg.value_imm(*sar.bits()) {
                match imm {
                    sonatina_ir::Immediate::I64(v) => v as u32,
                    sonatina_ir::Immediate::I32(v) => v as u32,
                    sonatina_ir::Immediate::I8(v) => v as u32,
                    _ => 0,
                }
            } else { 0 };
            // WGSL requires the shift amount to be u32 even when the shifted value
            // is i32; keep the literal u32 for both words.
            let bits_u32 = func.expressions.append(
                naga::Expression::Literal(naga::Literal::U32(shift_amount)),
                naga::Span::UNDEFINED,
            );
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
        // Shift-left is sign-agnostic. As with the existing Shr path, materialize
        // the immediate amount as WGSL u32 while preserving the runtime value's
        // carrier bits exactly.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let val = resolve_naga_value(*shl.value(), function, word, value_map, phi_locals, func).unwrap();
            let shift_amount = if let Some(imm) = function.dfg.value_imm(*shl.bits()) {
                match imm {
                    sonatina_ir::Immediate::I64(v) => v as u32,
                    sonatina_ir::Immediate::I32(v) => v as u32,
                    sonatina_ir::Immediate::I8(v) => v as u32,
                    _ => 0,
                }
            } else { 0 };
            let bits_u32 = func.expressions.append(
                naga::Expression::Literal(naga::Literal::U32(shift_amount)),
                naga::Span::UNDEFINED,
            );
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
        // a u32 literal amount. The i64 word fails closed in the pre-scan (only the
        // u32 browser word lowers `>>`), and a non-immediate amount fails closed
        // there too, so `bits` is an immediate here.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let val = resolve_naga_value(*shr.value(), function, word, value_map, phi_locals, func).unwrap();
            let shift_amount = if let Some(imm) = function.dfg.value_imm(*shr.bits()) {
                match imm {
                    sonatina_ir::Immediate::I64(v) => v as u32,
                    sonatina_ir::Immediate::I32(v) => v as u32,
                    sonatina_ir::Immediate::I8(v) => v as u32,
                    _ => 0,
                }
            } else { 0 };
            let bits_u32 = func.expressions.append(
                naga::Expression::Literal(naga::Literal::U32(shift_amount)),
                naga::Span::UNDEFINED,
            );
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
        // `x == 0`, lowered like the Eq arm to a naga bool consumed by branches /
        // `select`. The zero literal matches the active word (the U32 browser word
        // or the i64 word), the same shape as the immediate materialization above.
        if let Some(result) = function.dfg.inst_result(inst_id) {
            let lhs = resolve_naga_value(*is_zero.lhs(), function, word, value_map, phi_locals, func).unwrap();
            let zero = func.expressions.append(
                naga::Expression::Literal(match word {
                    WordKind::U32 => naga::Literal::U32(0),
                    WordKind::I64 => naga::Literal::I64(0),
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
            // Access returns a pointer — no Emit needed (like LocalVariable/GlobalVariable)
            let h = func.expressions.append(
                naga::Expression::Access { base, index: i32_idx },
                naga::Span::UNDEFINED,
            );
            value_map.insert(result, h);
            return true;
        }
    } else if let Some(ret) = <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(inst_set, inst_data) {
        if let Some(&val_id) = ret.args().as_slice().first() {
            let was_cached = value_map.contains_key(&val_id);
            let resolved = resolve_naga_value(val_id, function, word, value_map, phi_locals, func);
            if let Some(h) = resolved {
                if !was_cached && matches!(func.expressions[h], naga::Expression::Load { .. }) {
                    target.push(naga::Statement::Emit(naga::Range::new_from_bounds(h, h)), naga::Span::UNDEFINED);
                }
            }
            *result_expr = resolved;
            return true;
        }
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
) {
    for inst_id in function.layout.iter_inst(block) {
        emit_single_inst(inst_id, function, inst_set, word, func, target, value_map, phi_locals, result_expr);
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
) {
    let mut target = naga::Block::new();
    emit_phi_loads_for_block(function, inst_set, block, func, &mut target, value_map, phi_locals);
    emit_block_to_target(function, inst_set, word, block, func, &mut target, value_map, phi_locals, result_expr);
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
) -> Result<(), String> {
    let mut region_idx = 0;
    while region_idx < regions.len() {
        let region = &regions[region_idx];
        match region {
            crate::structurize::Region::Block(block_id) => {
                emit_naga_block_instructions(
                    function, inst_set, word, *block_id, word_type,
                    func, value_map, phi_locals, result_expr,
                );
                region_idx += 1;
            }
            crate::structurize::Region::Loop { header, body } => {
                let mut loop_target = naga::Block::new();
                let loop_return = emit_recursive_loop_region(
                    function, inst_set, word, *header, body, word_type, f32_type, bool_type,
                    func, value_map, phi_locals, result_expr, &mut loop_target,
                )?;
                func.body.extend_block(loop_target);
                region_idx += 1;
                if let Some(loop_return) = loop_return {
                    let saved_body = std::mem::replace(&mut func.body, naga::Block::new());
                    let mut continuation_result = None;
                    emit_naga_regions(
                        function, inst_set, word, &regions[region_idx..], word_type, f32_type,
                        bool_type, func, value_map, phi_locals, &mut continuation_result,
                    )?;
                    let mut continuation = std::mem::replace(&mut func.body, saved_body);
                    if let Some(value) = continuation_result {
                        let pointer = func.expressions.append(
                            naga::Expression::LocalVariable(loop_return.result),
                            naga::Span::UNDEFINED,
                        );
                        continuation.push(
                            naga::Statement::Store { pointer, value },
                            naga::Span::UNDEFINED,
                        );
                    }
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
                    return Ok(());
                }
            }
            crate::structurize::Region::IfThenElse { .. } => {
                let mut target = naga::Block::new();
                let transport = allocate_return_transport(
                    function, inst_set, word_type, f32_type, bool_type, func,
                )?;
                let mut may_return = false;
                emit_if_region(
                    function, inst_set, word, region, word_type, f32_type, bool_type, func, &mut target,
                    value_map, phi_locals, transport, &mut may_return,
                )?;
                func.body.extend_block(target);
                region_idx += 1;
                if may_return {
                    let saved_body = std::mem::replace(&mut func.body, naga::Block::new());
                    let mut continuation_result = None;
                    emit_naga_regions(
                        function, inst_set, word, &regions[region_idx..], word_type, f32_type,
                        bool_type, func, value_map, phi_locals, &mut continuation_result,
                    )?;
                    let mut continuation = std::mem::replace(&mut func.body, saved_body);
                    if let Some(value) = continuation_result {
                        let pointer = func.expressions.append(
                            naga::Expression::LocalVariable(transport.result), naga::Span::UNDEFINED,
                        );
                        continuation.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
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
                    } else if !continuation.is_empty() {
                        return Err("spirv: structured return continuation has no result".to_string());
                    }
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
                    return Ok(());
                }
            }
            crate::structurize::Region::LoopExit { from, target } => return Err(format!(
                "spirv: loop exit edge {from:?}->{target:?} appeared outside its loop"
            )),
        }
    }
    Ok(())
}

#[cfg(feature = "spirv-backend")]
fn ensure_phi_locals(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    block: sonatina_ir::BlockId,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
) {
    use sonatina_ir::InstDowncast;
    for inst_id in function.layout.iter_inst(block) {
        let inst = function.dfg.inst(inst_id);
        if <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst).is_none() { break }
        let Some(result) = function.dfg.inst_result(inst_id) else { continue };
        let ty = match function.dfg.value_ty(result) {
            sonatina_ir::Type::F32 => f32_type,
            sonatina_ir::Type::I1 => bool_type,
            _ => word_type,
        };
        phi_locals.entry(result).or_insert_with(|| func.local_variables.append(
            naga::LocalVariable { name: Some(format!("phi_{}", result.0)), ty, init: None },
            naga::Span::UNDEFINED,
        ));
    }
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
            crate::structurize::Region::LoopExit { .. } => {}
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
    let mut transfers = Vec::new();
    for inst_id in function.layout.iter_inst(to) {
        let inst = function.dfg.inst(inst_id);
        let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst) else { break };
        let result = function.dfg.inst_result(inst_id).ok_or_else(|| "spirv structurize: phi has no result".to_string())?;
        let local = *phi_locals.get(&result).ok_or_else(|| format!("spirv structurize: phi {result:?} has no local"))?;
        let [(value, _)] = phi.args().iter().filter(|(_, pred)| *pred == from).collect::<Vec<_>>().as_slice() else {
            return Err(format!("spirv structurize: edge {from:?}->{to:?} does not have exactly one input for phi {result:?}"));
        };
        let value = resolve_naga_value(*value, function, word, value_map, phi_locals, func)
            .ok_or_else(|| format!("spirv structurize: unresolved phi input on edge {from:?}->{to:?}"))?;
        let temp = func.local_variables.append(
            naga::LocalVariable {
                name: Some(format!("edge_{}_{}_phi_{}", from.0, to.0, result.0)),
                ty: func.local_variables[local].ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        transfers.push((local, temp, value));
    }
    // Snapshot every RHS before changing any destination phi local.
    for (_, temp, value) in &transfers {
        let pointer = func.expressions.append(naga::Expression::LocalVariable(*temp), naga::Span::UNDEFINED);
        target.push(naga::Statement::Store { pointer, value: *value }, naga::Span::UNDEFINED);
    }
    let mut snapshots = Vec::with_capacity(transfers.len());
    for (local, temp, _) in transfers {
        let pointer = func.expressions.append(naga::Expression::LocalVariable(temp), naga::Span::UNDEFINED);
        let loaded = func.expressions.append(naga::Expression::Load { pointer }, naga::Span::UNDEFINED);
        target.push(naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)), naga::Span::UNDEFINED);
        snapshots.push((local, loaded));
    }
    for (local, value) in snapshots {
        let pointer = func.expressions.append(naga::Expression::LocalVariable(local), naga::Span::UNDEFINED);
        target.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
    }
    Ok(())
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
) -> Result<(), String> {
    use sonatina_ir::InstDowncast;
    let crate::structurize::Region::IfThenElse { header, then_branch, else_branch, merge } = region else {
        return Err("spirv structurize: expected if region".to_string());
    };
    if let Some(merge) = merge {
        ensure_phi_locals(function, inst_set, *merge, word_type, f32_type, bool_type, func, phi_locals);
    }
    emit_phi_loads_for_block(
        function, inst_set, *header, func, target, value_map, phi_locals,
    );
    let mut ignored_result = None;
    emit_block_to_target(function, inst_set, word, *header, func, target, value_map, phi_locals, &mut ignored_result);
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
    emit_non_loop_regions(function, inst_set, word, then_branch, word_type, f32_type, bool_type, func, &mut accept, &mut then_values, phi_locals, transport, *merge, &mut then_returns, &mut then_edge_emitted)?;
    emit_non_loop_regions(function, inst_set, word, else_branch, word_type, f32_type, bool_type, func, &mut reject, &mut else_values, phi_locals, transport, *merge, &mut else_returns, &mut else_edge_emitted)?;
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
            let returns = function.layout.iter_inst(*block).any(|iid|
                <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(inst_set, function.dfg.inst(iid)).is_some()
            );
            if returns { ArmOutcome::Terminal } else { ArmOutcome::Predecessor(*block) }
        }
        crate::structurize::Region::IfThenElse { merge: Some(_), .. } => ArmOutcome::AlreadyAtMerge,
        crate::structurize::Region::IfThenElse { merge: None, .. }
        | crate::structurize::Region::Loop { .. }
        | crate::structurize::Region::LoopExit { .. } => ArmOutcome::Terminal,
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
) -> Result<(), String> {
    let mut region_idx = 0;
    while region_idx < regions.len() {
        let region = &regions[region_idx];
        match region {
            crate::structurize::Region::Block(block) => {
                emit_phi_loads_for_block(function, inst_set, *block, func, target, value_map, phi_locals);
                let mut block_result = None;
                emit_block_to_target(function, inst_set, word, *block, func, target, value_map, phi_locals, &mut block_result);
                if find_block_return_value(*block, function, inst_set).is_some() {
                    let value = block_result.ok_or_else(|| format!("spirv: unresolved structured return in {block:?}"))?;
                    let pointer = func.expressions.append(
                        naga::Expression::LocalVariable(transport.result), naga::Span::UNDEFINED,
                    );
                    target.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
                    let returned = func.expressions.append(
                        naga::Expression::Literal(naga::Literal::Bool(true)), naga::Span::UNDEFINED,
                    );
                    let pointer = func.expressions.append(
                        naga::Expression::LocalVariable(transport.did_return), naga::Span::UNDEFINED,
                    );
                    target.push(naga::Statement::Store { pointer, value: returned }, naga::Span::UNDEFINED);
                    *may_return = true;
                }
                region_idx += 1;
            }
            crate::structurize::Region::IfThenElse { header, merge: nested_merge, .. } => {
                let mut nested_returns = false;
                emit_if_region(
                    function, inst_set, word, region, word_type, f32_type, bool_type,
                    func, target, value_map, phi_locals, transport, &mut nested_returns,
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
                // A loop nested inside this conditional arm: emit it into the arm's
                // `target` block via the parameterized loop emitter, so the whole
                // Naga Loop statement lands inside the `if`. dec's inlined operator
                // loops carry no loop-internal Return (the kernel's single Return is
                // at the end), so a returning nested loop is not needed yet and
                // fails closed below.
                let mut no_result = None;
                let loop_return = emit_recursive_loop_region(
                    function, inst_set, word, *header, body, word_type, f32_type, bool_type,
                    func, value_map, phi_locals, &mut no_result, target,
                )?;
                if loop_return.is_some() {
                    return Err(format!(
                        "spirv structurize: a loop nested inside a conditional with an \
                         internal return is not supported yet (loop header {header:?})"
                    ));
                }
                region_idx += 1;
            }
            crate::structurize::Region::LoopExit { from, target } => return Err(format!(
                "spirv: loop exit edge {from:?}->{target:?} appeared outside its loop"
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
}

#[cfg(feature = "spirv-backend")]
fn allocate_return_transport(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
) -> Result<StructuredReturnTransport, String> {
    let return_ty = function.layout.iter_block().find_map(|block| {
        find_block_return_value(block, function, inst_set).map(|value| function.dfg.value_ty(value))
    }).ok_or_else(|| "spirv: value kernel has no return value".to_string())?;
    let return_naga_ty = match return_ty {
        sonatina_ir::Type::F32 => f32_type,
        sonatina_ir::Type::I1 => bool_type,
        _ => word_type,
    };
    let result = func.local_variables.append(
        naga::LocalVariable { name: Some("structured_result".into()), ty: return_naga_ty, init: None },
        naga::Span::UNDEFINED,
    );
    let returned_false = func.expressions.append(
        naga::Expression::Literal(naga::Literal::Bool(false)),
        naga::Span::UNDEFINED,
    );
    let did_return = func.local_variables.append(
        naga::LocalVariable {
            name: Some("structured_did_return".into()),
            ty: bool_type,
            init: Some(returned_false),
        },
        naga::Span::UNDEFINED,
    );
    Ok(StructuredReturnTransport { result, did_return })
}

#[cfg(feature = "spirv-backend")]
fn emit_regions_in_loop(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    regions: &[crate::structurize::Region],
    loop_header: sonatina_ir::BlockId,
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
) -> Result<RegionOutcome, String> {
    use sonatina_ir::InstDowncast;
    let mut outcome = None;
    for region in regions {
        match region {
            crate::structurize::Region::Block(block) => {
                emit_phi_loads_for_block(function, inst_set, *block, func, target, value_map, phi_locals);
                emit_block_to_target(function, inst_set, word, *block, func, target, value_map, phi_locals, result_expr);
                for inst_id in function.layout.iter_inst(*block) {
                    let inst = function.dfg.inst(inst_id);
                    if let Some(jump) = <&sonatina_ir::inst::control_flow::Jump as InstDowncast>::downcast(inst_set, inst) {
                        if *jump.dest() == loop_header {
                            emit_exact_phi_edge(function, inst_set, word, *block, loop_header, func, target, value_map, phi_locals)?;
                            target.push(naga::Statement::Continue, naga::Span::UNDEFINED);
                            return Ok(RegionOutcome::Terminal);
                        }
                    } else if let Some(ret) = <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(inst_set, inst) {
                        let value_id = ret.args().as_slice().first().ok_or_else(|| "spirv: unit return in value kernel".to_string())?;
                        let value = resolve_naga_value(*value_id, function, word, value_map, phi_locals, func)
                            .ok_or_else(|| format!("spirv: unresolved return in {block:?}"))?;
                        let pointer = func.expressions.append(naga::Expression::LocalVariable(return_local), naga::Span::UNDEFINED);
                        target.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
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
                    }
                }
                outcome = Some(RegionOutcome::Fallthrough(*block));
            }
            crate::structurize::Region::IfThenElse { header, then_branch, else_branch, merge } => {
                if let Some(merge) = merge {
                    ensure_phi_locals(function, inst_set, *merge, word_type, f32_type, bool_type, func, phi_locals);
                }
                emit_phi_loads_for_block(
                    function, inst_set, *header, func, target, value_map, phi_locals,
                );
                emit_block_to_target(function, inst_set, word, *header, func, target, value_map, phi_locals, result_expr);
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
                    emit_regions_in_loop(function, inst_set, word, then_branch, loop_header, word_type, f32_type, bool_type, return_local, did_return_local, may_return, func, &mut accept, &mut accept_values, phi_locals, result_expr)?
                };
                let else_outcome = if else_branch.is_empty() { RegionOutcome::Fallthrough(*header) } else {
                    emit_regions_in_loop(function, inst_set, word, else_branch, loop_header, word_type, f32_type, bool_type, return_local, did_return_local, may_return, func, &mut reject, &mut reject_values, phi_locals, result_expr)?
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
                // A loop nested inside this loop's body: emit it into `target` via
                // the (already parameterized) loop emitter, so the inner Naga Loop
                // lands inside the outer loop body. dec's inlined operator loops
                // carry no loop-internal Return, so a returning inner loop is not
                // needed yet and fails closed (the Break-cascade through the outer
                // loop is a follow-on).
                let mut nested_result = None;
                let inner_return = emit_recursive_loop_region(
                    function, inst_set, word, *header, body, word_type, f32_type, bool_type,
                    func, value_map, phi_locals, &mut nested_result, target,
                )?;
                if inner_return.is_some() {
                    return Err(format!(
                        "spirv structurize: a loop nested inside a loop with an internal \
                         return is not supported yet (inner loop header {header:?})"
                    ));
                }
                // Control resumes at the inner loop's exit successor.
                let mut inner_blocks = std::collections::HashSet::new();
                inner_blocks.insert(*header);
                region_blocks(body, &mut inner_blocks);
                let inner_branch = function.layout.iter_inst(*header).find_map(|iid|
                    <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, function.dfg.inst(iid))
                ).ok_or_else(|| format!("spirv: nested loop header {header:?} has no branch"))?;
                let inner_exit = if inner_blocks.contains(inner_branch.nz_dest()) {
                    *inner_branch.z_dest()
                } else {
                    *inner_branch.nz_dest()
                };
                outcome = Some(RegionOutcome::Fallthrough(inner_exit));
            }
            crate::structurize::Region::LoopExit { from, target: exit } => {
                ensure_phi_locals(
                    function, inst_set, *exit, word_type, f32_type, bool_type, func, phi_locals,
                );
                emit_exact_phi_edge(
                    function, inst_set, word, *from, *exit, func, target, value_map, phi_locals,
                )?;
                if let Some(ret) = find_block_return_value(*exit, function, inst_set) {
                    emit_phi_loads_for_block(
                        function, inst_set, *exit, func, target, value_map, phi_locals,
                    );
                    emit_block_to_target(
                        function, inst_set, word, *exit, func, target, value_map, phi_locals,
                        result_expr,
                    );
                    let value = resolve_naga_value(
                        ret, function, word, value_map, phi_locals, func,
                    )
                    .ok_or_else(|| format!("spirv: unresolved loop exit return in {exit:?}"))?;
                    let pointer = func.expressions.append(
                        naga::Expression::LocalVariable(return_local),
                        naga::Span::UNDEFINED,
                    );
                    target.push(
                        naga::Statement::Store { pointer, value },
                        naga::Span::UNDEFINED,
                    );
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
    // directly — it would alias the `&mut func` this fn also needs.)
    target: &mut naga::Block,
) -> Result<Option<StructuredReturnTransport>, String> {
    use sonatina_ir::InstDowncast;
    let mut loop_blocks = std::collections::HashSet::new();
    loop_blocks.insert(header);
    region_blocks(body_regions, &mut loop_blocks);
    ensure_phi_locals(function, inst_set, header, word_type, f32_type, bool_type, func, phi_locals);

    let outside_pred = function.layout.iter_inst(header).find_map(|iid| {
        let phi = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, function.dfg.inst(iid))?;
        phi.args().iter().find_map(|(_, pred)| (!loop_blocks.contains(pred)).then_some(*pred))
    });
    if let Some(outside_pred) = outside_pred {
        let mut init = naga::Block::new();
        emit_exact_phi_edge(function, inst_set, word, outside_pred, header, func, &mut init, value_map, phi_locals)?;
        target.extend_block(init);
    }

    let return_ty = function.layout.iter_block().find_map(|block| {
        find_block_return_value(block, function, inst_set).map(|value| function.dfg.value_ty(value))
    }).ok_or_else(|| "spirv: value kernel has no return value".to_string())?;
    let return_naga_ty = match return_ty {
        sonatina_ir::Type::F32 => f32_type,
        sonatina_ir::Type::I1 => bool_type,
        _ => word_type,
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
    emit_block_to_target(
        function, inst_set, word, header, func, &mut loop_body, value_map,
        phi_locals, &mut nested_result_expr,
    );
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
    ensure_phi_locals(function, inst_set, exit, word_type, f32_type, bool_type, func, phi_locals);
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
            function, inst_set, word, body_regions, header, word_type, f32_type, bool_type,
            return_local, did_return_local, &mut may_return, func, &mut continue_arm,
            &mut continue_values, phi_locals, &mut nested_result_expr,
        )?;
        // Fallthrough(header) is a trailing nested loop whose exit IS this
        // loop's back-edge: its exit arm already stored this header's phi
        // locals, and falling off the Naga loop body is an implicit continue.
        // Any other fallthrough would silently continue instead of breaking;
        // fail closed until the break-cascade lands.
        if let RegionOutcome::Fallthrough(resume) = body_outcome && resume != header {
            return Err(format!(
                "spirv structurize: loop {header:?} body falls through to {resume:?} \
                 instead of terminating (break-cascade not supported yet)"
            ));
        }
    }
    let mut exit_arm = naga::Block::new();
    let mut exit_values = value_map.clone();
    emit_exact_phi_edge(
        function, inst_set, word, header, exit, func, &mut exit_arm, &mut exit_values, phi_locals,
    )?;
    let exit_return = find_block_return_value(exit, function, inst_set);
    if let Some(ret) = exit_return {
        emit_phi_loads_for_block(
            function, inst_set, exit, func, &mut exit_arm, &mut exit_values, phi_locals,
        );
        emit_block_to_target(
            function, inst_set, word, exit, func, &mut exit_arm, &mut exit_values,
            phi_locals, &mut nested_result_expr,
        );
        let value = resolve_naga_value(ret, function, word, &mut exit_values, phi_locals, func)
            .ok_or_else(|| format!("spirv: unresolved loop exit return in {exit:?}"))?;
        let pointer = func.expressions.append(naga::Expression::LocalVariable(return_local), naga::Span::UNDEFINED);
        exit_arm.push(naga::Statement::Store { pointer, value }, naga::Span::UNDEFINED);
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
        || <&arith::Add as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Sub as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Mul as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fneg as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fadd as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fsub as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fmul as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fdiv as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fsqrt as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fabs as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fmin as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fmax as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Fclamp as InstDowncast>::downcast(is, inst).is_some()
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
        || <&data::ObjAlloc as InstDowncast>::downcast(is, inst).is_some()
        || <&data::ObjStore as InstDowncast>::downcast(is, inst).is_some()
        || <&data::ObjLoad as InstDowncast>::downcast(is, inst).is_some()
        || <&data::ObjIndex as InstDowncast>::downcast(is, inst).is_some()
}

#[cfg(feature = "spirv-backend")]
fn translate_to_naga(
    module: &Module,
    workgroup_size: [u32; 3],
    grid: bool,
    render: bool,
) -> Result<(naga::Module, SpirvLayout), String> {
    use std::collections::HashMap;

    // Content-derived word scalar: the kernel's own Sonatina return type is the
    // word-width SSOT (no hardwire, no config knob). I32 -> u32 (browser profile,
    // no SHADER_INT64); I64 -> i64 (the original path, bit-for-bit). Anything else,
    // a missing return, or a mixed-width argument fails closed.
    let funcs_peek = module.funcs();
    let first_func = *funcs_peek
        .first()
        .ok_or_else(|| "spirv: module has no functions to translate".to_string())?;

    let sig = module
        .ctx
        .get_sig(first_func)
        .ok_or_else(|| "spirv: first function has no declared signature".to_string())?;

    let word = match sig.single_ret_ty() {
        Some(sonatina_ir::Type::I32) => WordKind::U32,
        Some(sonatina_ir::Type::I64) => WordKind::I64,
        Some(other) => {
            return Err(format!(
                "spirv: unsupported kernel return type {other:?}; only i32 (u32 word) \
                 and i64 words are supported"
            ));
        }
        None => {
            return Err(
                "spirv: kernel has no single return value; the word width cannot be derived"
                    .to_string(),
            );
        }
    };

    for (i, &arg_ty) in sig.args().iter().enumerate() {
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

    // Scan the first function for ObjAlloc (output mode). Under a u32 word, also
    // fail closed on any signedness-sensitive op (Sar / signed compares / signed
    // div|mod): Sonatina integers are signless, so u32 is exact for wrapping
    // Add/Sub/Mul but WRONG for these until a sign mapping is designed. We never
    // silently emit the signed WGSL operator.
    let (param_count, has_obj_alloc) = module
        .func_store
        .try_view(first_func, |f| -> Result<(usize, bool), String> {
            let pc = f.arg_values.len();
            let is = f.inst_set();
            let mut has_alloc = false;
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
            // Diagnose unsupported f32 object storage before the general lowering
            // whitelist. Address construction such as `obj.proj` may itself be
            // outside that whitelist, but must not mask the more specific error.
            for bid in f.layout.iter_block() {
                for iid in f.layout.iter_inst(bid) {
                    let inst_data = f.dfg.inst(iid);
                    if let Some(store) = <&sonatina_ir::inst::data::ObjStore as sonatina_ir::InstDowncast>::downcast(is, inst_data) {
                        if f.dfg.value_ty(*store.value()) == sonatina_ir::Type::F32 {
                            return Err("spirv: f32 object storage is unsupported".to_string());
                        }
                    }
                    if <&sonatina_ir::inst::data::ObjLoad as sonatina_ir::InstDowncast>::downcast(is, inst_data).is_some()
                        && f.dfg.inst_result(iid).is_some_and(|value| f.dfg.value_ty(value) == sonatina_ir::Type::F32)
                    {
                        return Err("spirv: f32 object storage is unsupported".to_string());
                    }
                }
            }
            for bid in f.layout.iter_block() {
                for iid in f.layout.iter_inst(bid) {
                    let inst_data = f.dfg.inst(iid);
                    if let Some(result) = f.dfg.inst_result(iid) {
                        let result_ty = f.dfg.value_ty(result);
                        let carrier_ty = match word {
                            WordKind::U32 => sonatina_ir::Type::I32,
                            WordKind::I64 => sonatina_ir::Type::I64,
                        };
                        if result_ty.is_integral()
                            && result_ty != sonatina_ir::Type::I1
                            && result_ty != carrier_ty
                        {
                            return Err(format!(
                                "spirv: narrow or mixed integer instruction result {result:?} has unsupported type {result_ty:?}; expected {carrier_ty:?} carrier"
                            ));
                        }
                    }
                    if !spirv_instruction_is_lowered(is, inst_data) {
                        return Err(format!(
                            "spirv: instruction `{}` is unsupported by the SPIR-V translator",
                            inst_data.as_text()
                        ));
                    }
                    if <&sonatina_ir::inst::data::ObjAlloc as sonatina_ir::InstDowncast>::downcast(
                        is, inst_data,
                    )
                    .is_some()
                    {
                        has_alloc = true;
                    }
                    if word == WordKind::U32 {
                        if let Some(op) = unsupported_signed_op_under_u32(is, inst_data) {
                            return Err(format!(
                                "spirv u32: signedness-sensitive op `{op}` is unsupported under \
                                 a u32 word (Sonatina integers are signless; a sign mapping is \
                                 not yet designed). Fail closed."
                            ));
                        }
                        // `Sar` is now word-aware (i32-bitcast arithmetic shift), but the
                        // u32 arm materializes the shift amount as a `Literal::U32`, so the
                        // `bits` operand must be an immediate. A non-immediate shift amount
                        // fails closed here with a named error rather than silently reading 0.
                        if let Some(sar) =
                            <&sonatina_ir::inst::arith::Sar as sonatina_ir::InstDowncast>::downcast(
                                is, inst_data,
                            )
                        {
                            if f.dfg.value_imm(*sar.bits()).is_none() {
                                return Err(
                                    "spirv u32: sar with a non-immediate shift amount is \
                                     unsupported (the u32 arithmetic shift materializes the \
                                     amount as a WGSL u32 literal). Fail closed."
                                        .to_string(),
                                );
                            }
                        }
                        // `Shr` (logical shift) has the SAME immediate-only rule as
                        // `Sar` under u32: the amount is materialized as a WGSL u32
                        // literal. A non-immediate amount fails closed rather than
                        // silently reading 0.
                        if let Some(shr) =
                            <&sonatina_ir::inst::arith::Shr as sonatina_ir::InstDowncast>::downcast(
                                is, inst_data,
                            )
                        {
                            if f.dfg.value_imm(*shr.bits()).is_none() {
                                return Err(
                                    "spirv u32: shr with a non-immediate shift amount is \
                                     unsupported (the u32 logical shift materializes the \
                                     amount as a WGSL u32 literal). Fail closed."
                                        .to_string(),
                                );
                            }
                        }
                        if let Some(shl) =
                            <&sonatina_ir::inst::arith::Shl as sonatina_ir::InstDowncast>::downcast(
                                is, inst_data,
                            )
                        {
                            if f.dfg.value_imm(*shl.bits()).is_none() {
                                return Err(
                                    "spirv u32: shl with a non-immediate shift amount is \
                                     unsupported (the u32 shift-left materializes the \
                                     amount as a WGSL u32 literal). Fail closed."
                                        .to_string(),
                                );
                            }
                        }
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
            Ok((pc, has_alloc))
        })
        .ok_or_else(|| "spirv: first function body is unavailable".to_string())??;

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
        if has_obj_alloc {
            return Err(
                "spirv render: render and batch (ObjAlloc) modes are mutually exclusive"
                    .to_string(),
            );
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
        let broadcast = param_count - 2;
        let effective_params = broadcast.max(1);
        let mut input_members = Vec::with_capacity(effective_params);
        let mut layout_input_members = Vec::with_capacity(broadcast);
        let mut input_span = 0;
        let mut input_align = 1;
        for (i, ty) in sig.args().iter().skip(2).copied().enumerate() {
            let (naga_ty, width, scalar) = match ty {
                sonatina_ir::Type::I32 => (word_type, 4, SpirvScalarKind::I32),
                sonatina_ir::Type::F32 => (f32_type, 4, SpirvScalarKind::F32),
                _ => return Err(format!("spirv render: broadcast arg {} has unsupported storage type {ty:?}", i + 2)),
            };
            input_members.push(naga::StructMember { name: Some(format!("p{i}")), ty: naga_ty, binding: None, offset: input_span });
            layout_input_members.push(SpirvBindingMember { arg_index: (i + 2) as u32, offset: input_span, width, scalar });
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
                inner: naga::TypeInner::Struct { members: input_members, span: input_span },
            },
            naga::Span::UNDEFINED,
        );
        let input_var = naga_mod.global_variables.append(
            naga::GlobalVariable {
                name: Some("input".into()),
                space: naga::AddressSpace::Storage { access: naga::StorageAccess::LOAD },
                binding: Some(naga::ResourceBinding { group: 0, binding: 1 }),
                ty: input_struct,
                init: None,
                memory_decorations: naga::ir::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );

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

        let funcs = module.funcs();
        let mut result_expr = None;
        let mut body_error = None;
        if let Some(&func_ref) = funcs.first() {
            module.func_store.try_view(func_ref, |function| {
                let inst_set = function.inst_set();
                let mut value_map: HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>> =
                    HashMap::new();
                let mut phi_locals: HashMap<
                    sonatina_ir::ValueId,
                    naga::Handle<naga::LocalVariable>,
                > = HashMap::new();

                // The broadcast input global (unused if there are no args 2..; an
                // unused GlobalVariable is legal, as in Grid's gradient kernel).
                let input_expr = fs.expressions.append(
                    naga::Expression::GlobalVariable(input_var),
                    naga::Span::UNDEFINED,
                );

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

                // Args 2.. load from the broadcast input struct at member idx - 2,
                // the SAME castless path Grid uses.
                for (idx, &arg_val) in function.arg_values.iter().enumerate().skip(2) {
                    let field = fs.expressions.append(
                        naga::Expression::AccessIndex { base: input_expr, index: (idx - 2) as u32 },
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
                    value_map.insert(arg_val, loaded);
                }

                // The mode-blind body: SAME structurizer + region emission Grid and
                // Scalar use (zero changes to emit_naga_regions / emit_single_inst).
                let scfg = match crate::structurize::structurize_function(function) {
                    Ok(scfg) => scfg,
                    Err(err) => { body_error = Some(err); return; }
                };
                if let Err(err) = emit_naga_regions(
                    function, inst_set, word, &scfg.regions, word_type, f32_type, bool_type,
                    &mut fs, &mut value_map, &mut phi_locals, &mut result_expr,
                ) {
                    body_error = Some(err);
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
            bindings: vec![SpirvBinding {
                group: 0,
                binding: 1,
                name: "input".to_string(),
                access: Access::Read,
                role: Role::Input,
                stride: input_span,
                span: input_span,
                members: layout_input_members,
            }],
            builtin_inputs: vec![
                SpirvBuiltinInput { arg_index: 0, source: SpirvBuiltinSource::FragmentPositionX, scalar: SpirvScalarKind::I32 },
                SpirvBuiltinInput { arg_index: 1, source: SpirvBuiltinSource::FragmentPositionY, scalar: SpirvScalarKind::I32 },
            ],
            result: None,
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

    // u32 vec3 type for global_invocation_id
    let u32_type = naga_mod.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Uint,
                width: 4,
            }),
        },
        naga::Span::UNDEFINED,
    );
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

    // Translate the first Sonatina function
    let funcs = module.funcs();
    let mut result_expr = None;
    // In grid mode, the gid.x / gid.y expressions bound to args 0,1 are emitted
    // inside the body closure but reused by the per-pixel store that follows it,
    // so their handles flow out here.
    let mut grid_gid: Option<(naga::Handle<naga::Expression>, naga::Handle<naga::Expression>)> =
        None;

    let mut body_error = None;
    if let Some(&func_ref) = funcs.first() {
        module.func_store.try_view(func_ref, |function| {
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
                Err(err) => { body_error = Some(err); return; }
            };
            if let Err(err) = emit_naga_regions(
                function, inst_set, word, &scfg.regions, word_type, f32_type, bool_type,
                &mut func, &mut value_map, &mut phi_locals, &mut result_expr,
            ) {
                body_error = Some(err);
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
        // Access returns a pointer — no Emit needed.
        let ptr = func.expressions.append(
            naga::Expression::Access { base: output_expr, index: idx_i32 },
            naga::Span::UNDEFINED,
        );
        func.body.push(
            naga::Statement::Store { pointer: ptr, value: result_val },
            naga::Span::UNDEFINED,
        );
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
        bindings: vec![
            SpirvBinding {
                group: 0,
                binding: 0,
                name: "output".to_string(),
                access: Access::ReadWrite,
                role: Role::Output,
                stride: word_width,
                span: word_width,
                members: Vec::new(),
            },
            SpirvBinding {
                group: 0,
                binding: 1,
                name: "input".to_string(),
                access: Access::Read,
                role: Role::Input,
                stride: input_span,
                span: input_span,
                members: layout_input_members,
            },
        ],
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
        // Compute modes (Scalar/Grid/Batch) have no vertex/fragment stages and no
        // color target.
        vertex_entry: None,
        fragment_entry: None,
        color_target_format: None,
    };

    Ok((naga_mod, layout))
}

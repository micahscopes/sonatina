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

/// One storage-buffer binding, as the compiler actually emitted it.
#[derive(Debug, Clone)]
pub struct SpirvBinding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub access: Access,
    pub role: Role,
    /// Element stride: word width for the output, per-invocation input span for
    /// the input buffer.
    pub stride: u32,
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
                if body.iter().any(|region| matches!(region, crate::structurize::Region::IfThenElse { .. })) {
                    return Err(format!(
                        "spirv structurize: loop {header:?} containing a conditional is not supported yet"
                    ));
                }
                if body.iter().any(|region| matches!(region, crate::structurize::Region::Loop { .. })) {
                    return Err(format!("spirv structurize: loop {header:?} nested inside a loop is not supported yet"));
                }
                validate_legacy_loop_shape(function, inst_set, *header, body)?;
                region_idx += 1;
                emit_loop_region(
                    function, inst_set, word, *header, body, &regions[region_idx..],
                    &mut region_idx, word_type, func, value_map, phi_locals, result_expr,
                );
            }
            crate::structurize::Region::IfThenElse { .. } => {
                let mut target = naga::Block::new();
                emit_if_region(
                    function, inst_set, word, region, word_type, f32_type, bool_type, func, &mut target,
                    value_map, phi_locals, result_expr,
                )?;
                func.body.extend_block(target);
                region_idx += 1;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "spirv-backend")]
fn validate_legacy_loop_shape(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    header: sonatina_ir::BlockId,
    body: &[crate::structurize::Region],
) -> Result<(), String> {
    use sonatina_ir::InstDowncast;

    let mut loop_blocks = std::collections::HashSet::new();
    loop_blocks.insert(header);
    region_blocks(body, &mut loop_blocks);
    let branch = function.layout.iter_inst(header).find_map(|inst_id| {
        <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(
            inst_set,
            function.dfg.inst(inst_id),
        )
    }).ok_or_else(|| {
        format!("spirv structurize: legacy loop {header:?} requires a conditional header")
    })?;
    if !loop_blocks.contains(branch.nz_dest()) || loop_blocks.contains(branch.z_dest()) {
        return Err(format!(
            "spirv structurize: loop {header:?} has unsupported branch polarity"
        ));
    }

    for inst_id in function.layout.iter_inst(header) {
        let inst = function.dfg.inst(inst_id);
        let Some(phi) =
            <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst)
        else {
            break;
        };
        let outside_count = phi.args().iter()
            .filter(|(_, pred)| !loop_blocks.contains(pred))
            .count();
        if outside_count != 1 {
            return Err(format!(
                "spirv structurize: loop {header:?} phi requires exactly one preheader input, found {outside_count}"
            ));
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
        }
    }
}

#[cfg(feature = "spirv-backend")]
fn emit_phi_edge_stores(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    merge: sonatina_ir::BlockId,
    predecessors: &std::collections::HashSet<sonatina_ir::BlockId>,
    func: &mut naga::Function,
    target: &mut naga::Block,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
) -> Result<(), String> {
    use sonatina_ir::InstDowncast;
    for inst_id in function.layout.iter_inst(merge) {
        let inst = function.dfg.inst(inst_id);
        let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst) else { break };
        let result = function.dfg.inst_result(inst_id).ok_or_else(|| "spirv structurize: phi has no result".to_string())?;
        let local = *phi_locals.get(&result).ok_or_else(|| "spirv structurize: merge phi has no local".to_string())?;
        let matching_inputs = phi.args().iter().filter(|(_, pred)| predecessors.contains(pred)).collect::<Vec<_>>();
        let [(value, _)] = matching_inputs.as_slice() else {
            return Err(format!("spirv structurize: merge phi {result:?} has {} inputs for one arm", matching_inputs.len()));
        };
        let value = resolve_naga_value(*value, function, word, value_map, phi_locals, func)
            .ok_or_else(|| format!("spirv structurize: unresolved phi input {value:?}"))?;
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
    result_expr: &mut Option<naga::Handle<naga::Expression>>,
) -> Result<(), String> {
    use sonatina_ir::InstDowncast;
    let crate::structurize::Region::IfThenElse { header, then_branch, else_branch, merge } = region else {
        return Err("spirv structurize: expected if region".to_string());
    };
    let Some(merge) = merge else {
        return Err(format!("spirv structurize: conditional {header:?} without a merge is not supported yet"));
    };
    ensure_phi_locals(function, inst_set, *merge, word_type, f32_type, bool_type, func, phi_locals);
    emit_block_to_target(function, inst_set, word, *header, func, target, value_map, phi_locals, result_expr);
    let branch = function.layout.iter_inst(*header).find_map(|inst_id|
        <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, function.dfg.inst(inst_id))
    ).ok_or_else(|| format!("spirv structurize: if header {header:?} has no branch"))?;
    let condition = resolve_naga_value(*branch.cond(), function, word, value_map, phi_locals, func)
        .ok_or_else(|| format!("spirv structurize: unresolved condition in {header:?}"))?;

    let mut accept = naga::Block::new();
    let mut reject = naga::Block::new();
    emit_non_loop_regions(function, inst_set, word, then_branch, word_type, f32_type, bool_type, func, &mut accept, value_map, phi_locals, result_expr)?;
    emit_non_loop_regions(function, inst_set, word, else_branch, word_type, f32_type, bool_type, func, &mut reject, value_map, phi_locals, result_expr)?;
    let mut then_preds = std::collections::HashSet::new();
    region_blocks(then_branch, &mut then_preds);
    if then_preds.is_empty() { then_preds.insert(*header); }
    emit_phi_edge_stores(function, inst_set, word, *merge, &then_preds, func, &mut accept, value_map, phi_locals)?;
    let mut else_preds = std::collections::HashSet::new();
    region_blocks(else_branch, &mut else_preds);
    if else_preds.is_empty() { else_preds.insert(*header); }
    emit_phi_edge_stores(function, inst_set, word, *merge, &else_preds, func, &mut reject, value_map, phi_locals)?;
    target.push(naga::Statement::If { condition, accept, reject }, naga::Span::UNDEFINED);
    Ok(())
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
    result_expr: &mut Option<naga::Handle<naga::Expression>>,
) -> Result<(), String> {
    for region in regions {
        match region {
            crate::structurize::Region::Block(block) => {
                emit_phi_loads_for_block(function, inst_set, *block, func, target, value_map, phi_locals);
                emit_block_to_target(function, inst_set, word, *block, func, target, value_map, phi_locals, result_expr);
            }
            crate::structurize::Region::IfThenElse { .. } => emit_if_region(
                function, inst_set, word, region, word_type, f32_type, bool_type, func, target, value_map, phi_locals, result_expr,
            )?,
            crate::structurize::Region::Loop { .. } => return Err(
                "spirv structurize: loop nested inside conditional is not supported yet".to_string()
            ),
        }
    }
    Ok(())
}

/// Emit a Loop region, handling inner conditional branches and post-loop exit blocks.
#[cfg(feature = "spirv-backend")]
fn emit_loop_region(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    header: sonatina_ir::BlockId,
    body: &[crate::structurize::Region],
    remaining_regions: &[crate::structurize::Region],
    region_idx: &mut usize,
    word_type: naga::Handle<naga::Type>,
    func: &mut naga::Function,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    result_expr: &mut Option<naga::Handle<naga::Expression>>,
) {
    use sonatina_ir::InstDowncast;

    let mut loop_blocks = std::collections::HashSet::new();
    loop_blocks.insert(header);
    region_blocks(body, &mut loop_blocks);

    // Create LocalVariables for phi nodes in the loop header
    for inst_id in function.layout.iter_inst(header) {
        let inst_data = function.dfg.inst(inst_id);
        if let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst_data) {
            if let Some(result) = function.dfg.inst_result(inst_id) {
                let local = func.local_variables.append(
                    naga::LocalVariable {
                        name: Some(format!("phi_{}", result.0)),
                        ty: word_type,
                        init: None,
                    },
                    naga::Span::UNDEFINED,
                );
                phi_locals.insert(result, local);

                // Initialize from the unique predecessor outside the loop.
                if let Some(&(init_val, _)) = phi.args().iter()
                    .find(|(_, pred)| !loop_blocks.contains(pred))
                {
                    if let Some(init) = resolve_naga_value(init_val, function, word, value_map, phi_locals, func) {
                        let ptr = func.expressions.append(
                            naga::Expression::LocalVariable(local),
                            naga::Span::UNDEFINED,
                        );
                        func.body.push(
                            naga::Statement::Store { pointer: ptr, value: init },
                            naga::Span::UNDEFINED,
                        );
                    }
                }
            }
        } else {
            break;
        }
    }

    // Detect if any non-header body block has a Br (inner conditional)
    let has_inner_br = body.iter().any(|inner| {
        if let crate::structurize::Region::Block(bid) = inner {
            if *bid == header { return false; }
            for inst_id in function.layout.iter_inst(*bid) {
                let inst_data = function.dfg.inst(inst_id);
                if <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                    return true;
                }
            }
        }
        false
    });

    // If there's an inner Br, create a result_local for the function result.
    // Both exit paths (header exit, inner escape) store their return value here.
    let result_local = if has_inner_br {
        Some(func.local_variables.append(
            naga::LocalVariable {
                name: Some("loop_result".into()),
                ty: word_type,
                init: None,
            },
            naga::Span::UNDEFINED,
        ))
    } else {
        None
    };

    // Build the loop body
    let mut loop_body = naga::Block::new();
    let loop_continuing = naga::Block::new();

    // Load phi values at top of each iteration
    for inst_id in function.layout.iter_inst(header) {
        let inst_data = function.dfg.inst(inst_id);
        if <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst_data).is_some() {
            if let Some(result) = function.dfg.inst_result(inst_id) {
                if let Some(&local) = phi_locals.get(&result) {
                    let ptr = func.expressions.append(naga::Expression::LocalVariable(local), naga::Span::UNDEFINED);
                    let loaded = func.expressions.append(naga::Expression::Load { pointer: ptr }, naga::Span::UNDEFINED);
                    // Only emit Load; LocalVariable is a const expression (always in scope)
                    loop_body.push(naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)), naga::Span::UNDEFINED);
                    value_map.insert(result, loaded);
                }
            }
        } else { break; }
    }

    // Header comparison (non-phi, non-Br instructions)
    for inst_id in function.layout.iter_inst(header) {
        let inst_data = function.dfg.inst(inst_id);
        if <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst_data).is_some() { continue; }
        if <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, inst_data).is_some() { continue; }
        emit_single_inst(inst_id, function, inst_set, word, func, &mut loop_body, value_map, phi_locals, result_expr);
    }

    // Header Br: find exit block and its return value, then emit if NOT(cond) { store exit result; break; }
    let mut header_exit_block = None;
    for inst_id in function.layout.iter_inst(header) {
        let inst_data = function.dfg.inst(inst_id);
        if let Some(br) = <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, inst_data) {
            header_exit_block = Some(*br.z_dest());
            if let Some(c) = resolve_naga_value(*br.cond(), function, word, value_map, phi_locals, func) {
                let not_c = func.expressions.append(naga::Expression::Unary { op: naga::UnaryOperator::LogicalNot, expr: c }, naga::Span::UNDEFINED);
                loop_body.push(naga::Statement::Emit(naga::Range::new_from_bounds(not_c, not_c)), naga::Span::UNDEFINED);
                let mut break_block = naga::Block::new();
                // Emit side effects (ObjStore etc.) from exit block before break
                emit_obj_ops_from_block(function, inst_set, word, *br.z_dest(), func, &mut break_block, value_map, phi_locals);
                if let Some(res_local) = result_local {
                    if let Some(ret_val) = find_block_return_value(*br.z_dest(), function, inst_set) {
                        let expr_count_before = func.expressions.len();
                        if let Some(v) = resolve_naga_value(ret_val, function, word, value_map, phi_locals, func) {
                            if v.index() >= expr_count_before {
                                if matches!(func.expressions[v], naga::Expression::Load { .. }
                                    | naga::Expression::Binary { .. }
                                    | naga::Expression::Unary { .. }) {
                                    break_block.push(naga::Statement::Emit(naga::Range::new_from_bounds(v, v)), naga::Span::UNDEFINED);
                                }
                            }
                            let ptr = func.expressions.append(naga::Expression::LocalVariable(res_local), naga::Span::UNDEFINED);
                            break_block.push(naga::Statement::Store { pointer: ptr, value: v }, naga::Span::UNDEFINED);
                        }
                    }
                }
                break_block.push(naga::Statement::Break, naga::Span::UNDEFINED);
                loop_body.push(naga::Statement::If { condition: not_c, accept: break_block, reject: naga::Block::new() }, naga::Span::UNDEFINED);
            }
            break;
        }
    }

    // Collect non-header body blocks
    let non_header_blocks: Vec<sonatina_ir::BlockId> = body.iter().filter_map(|inner| {
        if let crate::structurize::Region::Block(bid) = inner {
            if *bid != header { return Some(*bid); }
        }
        None
    }).collect();

    if has_inner_br {
        // Find the block with the inner Br
        let mut br_block_idx = None;
        for (idx, &bid) in non_header_blocks.iter().enumerate() {
            for inst_id in function.layout.iter_inst(bid) {
                let inst_data = function.dfg.inst(inst_id);
                if <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                    br_block_idx = Some(idx);
                    break;
                }
            }
            if br_block_idx.is_some() { break; }
        }

        if let Some(br_idx) = br_block_idx {
            let br_bid = non_header_blocks[br_idx];

            // Emit compute instructions from the Br block into loop_body
            let mut inner_cond_handle = None;
            let mut inner_escape_block = None;
            for inst_id in function.layout.iter_inst(br_bid) {
                let inst_data = function.dfg.inst(inst_id);
                if let Some(br) = <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, inst_data) {
                    inner_cond_handle = resolve_naga_value(*br.cond(), function, word, value_map, phi_locals, func);
                    inner_escape_block = Some(*br.z_dest());
                    continue;
                }
                emit_single_inst(inst_id, function, inst_set, word, func, &mut loop_body, value_map, phi_locals, result_expr);
            }

            if let Some(cond) = inner_cond_handle {
                // Accept branch (condition true): continue blocks + phi updates
                let mut accept_block = naga::Block::new();
                for &bid in &non_header_blocks[br_idx + 1..] {
                    emit_block_to_target(function, inst_set, word, bid, func, &mut accept_block, value_map, phi_locals, result_expr);

                    // Phi updates for blocks that jump back to header
                    for inst_id in function.layout.iter_inst(bid) {
                        let inst_data = function.dfg.inst(inst_id);
                        if <&sonatina_ir::inst::control_flow::Jump as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                            for target_inst_id in function.layout.iter_inst(header) {
                                let target_inst = function.dfg.inst(target_inst_id);
                                if let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, target_inst) {
                                    if let Some(phi_result) = function.dfg.inst_result(target_inst_id) {
                                        if let Some(&local) = phi_locals.get(&phi_result) {
                                            for &(val, from_block) in phi.args() {
                                                if from_block == bid {
                                                    if let Some(v) = resolve_naga_value(val, function, word, value_map, phi_locals, func) {
                                                        let ptr = func.expressions.append(naga::Expression::LocalVariable(local), naga::Span::UNDEFINED);
                                                        accept_block.push(naga::Statement::Store { pointer: ptr, value: v }, naga::Span::UNDEFINED);
                                                    }
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                } else { break; }
                            }
                        }
                    }
                }

                // Reject branch (condition false, escape): emit side effects, store result, break
                let mut reject_block = naga::Block::new();
                if let Some(esc_bid) = inner_escape_block {
                    emit_obj_ops_from_block(function, inst_set, word, esc_bid, func, &mut reject_block, value_map, phi_locals);
                    if let Some(res_local) = result_local {
                        if let Some(ret_val) = find_block_return_value(esc_bid, function, inst_set) {
                            let expr_count_before = func.expressions.len();
                            if let Some(v) = resolve_naga_value(ret_val, function, word, value_map, phi_locals, func) {
                                if v.index() >= expr_count_before {
                                    if matches!(func.expressions[v], naga::Expression::Load { .. }
                                        | naga::Expression::Binary { .. }
                                        | naga::Expression::Unary { .. }) {
                                        reject_block.push(naga::Statement::Emit(naga::Range::new_from_bounds(v, v)), naga::Span::UNDEFINED);
                                    }
                                }
                                let ptr = func.expressions.append(naga::Expression::LocalVariable(res_local), naga::Span::UNDEFINED);
                                reject_block.push(naga::Statement::Store { pointer: ptr, value: v }, naga::Span::UNDEFINED);
                            }
                        }
                    }
                }
                reject_block.push(naga::Statement::Break, naga::Span::UNDEFINED);

                loop_body.push(
                    naga::Statement::If { condition: cond, accept: accept_block, reject: reject_block },
                    naga::Span::UNDEFINED,
                );
            }
        }
    } else {
        // Simple loop body (no inner Br)
        for &bid in &non_header_blocks {
            emit_block_to_target(function, inst_set, word, bid, func, &mut loop_body, value_map, phi_locals, result_expr);

            // Phi updates
            for inst_id in function.layout.iter_inst(bid) {
                let inst_data = function.dfg.inst(inst_id);
                if <&sonatina_ir::inst::control_flow::Jump as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                    for target_inst_id in function.layout.iter_inst(header) {
                        let target_inst = function.dfg.inst(target_inst_id);
                        if let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, target_inst) {
                            if let Some(phi_result) = function.dfg.inst_result(target_inst_id) {
                                if let Some(&local) = phi_locals.get(&phi_result) {
                                    for &(val, from_block) in phi.args() {
                                        if from_block == bid {
                                            if let Some(v) = resolve_naga_value(val, function, word, value_map, phi_locals, func) {
                                                let ptr = func.expressions.append(naga::Expression::LocalVariable(local), naga::Span::UNDEFINED);
                                                loop_body.push(naga::Statement::Store { pointer: ptr, value: v }, naga::Span::UNDEFINED);
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        } else { break; }
                    }
                }
            }
        }
    }

    // Emit the Naga Loop statement
    func.body.push(
        naga::Statement::Loop {
            body: loop_body,
            continuing: loop_continuing,
            break_if: None,
        },
        naga::Span::UNDEFINED,
    );

    // After the loop, phi values from the loop body are out of scope.
    for inst_id in function.layout.iter_inst(header) {
        let inst_data = function.dfg.inst(inst_id);
        if <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst_data).is_some() {
            if let Some(result) = function.dfg.inst_result(inst_id) {
                value_map.remove(&result);
            }
        } else {
            break;
        }
    }

    // If we used a result_local, load from it and set result_expr,
    // then skip the post-loop return blocks
    if let Some(res_local) = result_local {
        let ptr = func.expressions.append(naga::Expression::LocalVariable(res_local), naga::Span::UNDEFINED);
        let loaded = func.expressions.append(naga::Expression::Load { pointer: ptr }, naga::Span::UNDEFINED);
        func.body.push(naga::Statement::Emit(naga::Range::new_from_bounds(loaded, loaded)), naga::Span::UNDEFINED);
        *result_expr = Some(loaded);

        // Skip the post-loop return blocks (they are the exit targets
        // whose values we already captured into result_local)
        let mut post_blocks_to_skip = std::collections::HashSet::new();
        if let Some(exit_bid) = header_exit_block {
            post_blocks_to_skip.insert(exit_bid);
        }
        // Also find the inner escape block
        for inner in body {
            if let crate::structurize::Region::Block(bid) = inner {
                if *bid == header { continue; }
                for inst_id in function.layout.iter_inst(*bid) {
                    let inst_data = function.dfg.inst(inst_id);
                    if let Some(br) = <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, inst_data) {
                        post_blocks_to_skip.insert(*br.z_dest());
                    }
                }
            }
        }

        // Skip remaining regions that are post-loop return blocks
        // we've already captured. remaining_regions starts at the
        // current region_idx position, so offset 0 = next unprocessed region.
        let mut skip_offset = 0;
        while skip_offset < remaining_regions.len() {
            if let crate::structurize::Region::Block(bid) = &remaining_regions[skip_offset] {
                if post_blocks_to_skip.contains(bid) {
                    *region_idx += 1;
                    skip_offset += 1;
                    continue;
                }
            }
            break;
        }
    }
}

/// Emit only ObjIndex/ObjStore instructions from a block, creating fresh expressions.
/// Used in break/escape paths inside loops where the full block can't be re-emitted.
#[cfg(feature = "spirv-backend")]
fn emit_obj_ops_from_block(
    function: &sonatina_ir::Function,
    inst_set: &dyn sonatina_ir::InstSetBase,
    word: WordKind,
    block: sonatina_ir::BlockId,
    func: &mut naga::Function,
    target: &mut naga::Block,
    value_map: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phi_locals: &mut std::collections::HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
) {
    use sonatina_ir::InstDowncast;
    for inst_id in function.layout.iter_inst(block) {
        let inst_data = function.dfg.inst(inst_id);

        if let Some(obj_index) = <&sonatina_ir::inst::data::ObjIndex as InstDowncast>::downcast(inst_set, inst_data) {
            if let Some(result) = function.dfg.inst_result(inst_id) {
                let base = resolve_naga_value(*obj_index.object(), function, word, value_map, phi_locals, func).unwrap();
                let index = resolve_naga_value(*obj_index.index(), function, word, value_map, phi_locals, func).unwrap();
                let i32_idx = func.expressions.append(
                    naga::Expression::As { expr: index, kind: naga::ScalarKind::Sint, convert: Some(4) },
                    naga::Span::UNDEFINED,
                );
                target.push(naga::Statement::Emit(naga::Range::new_from_bounds(i32_idx, i32_idx)), naga::Span::UNDEFINED);
                let h = func.expressions.append(
                    naga::Expression::Access { base, index: i32_idx },
                    naga::Span::UNDEFINED,
                );
                value_map.insert(result, h);
            }
        } else if let Some(obj_store) = <&sonatina_ir::inst::data::ObjStore as InstDowncast>::downcast(inst_set, inst_data) {
            let dest = resolve_naga_value(*obj_store.object(), function, word, value_map, phi_locals, func).unwrap();
            let val = resolve_naga_value(*obj_store.value(), function, word, value_map, phi_locals, func).unwrap();
            target.push(naga::Statement::Store { pointer: dest, value: val }, naga::Span::UNDEFINED);
        }
    }
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
    use sonatina_ir::{InstDowncast, inst::{arith, cmp, control_flow, data}};

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
        || <&arith::Sar as InstDowncast>::downcast(is, inst).is_some()
        || <&arith::Shr as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::Lt as InstDowncast>::downcast(is, inst).is_some()
        || <&cmp::Slt as InstDowncast>::downcast(is, inst).is_some()
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
            let mut cfg = sonatina_ir::cfg::ControlFlowGraph::default();
            cfg.compute(f);
            let mut domtree = crate::domtree::DomTree::new();
            domtree.compute(&cfg);
            let mut loop_tree = crate::loop_analysis::LoopTree::new();
            loop_tree.compute(&cfg, &domtree);
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
                    if <&sonatina_ir::inst::control_flow::Phi as sonatina_ir::InstDowncast>::downcast(is, inst_data).is_some() {
                        let phi_ty = f.dfg.inst_result(iid).map(|value| f.dfg.value_ty(value));
                        let is_loop_header = loop_tree
                            .loop_of_block(bid)
                            .is_some_and(|lp| loop_tree.loop_header(lp) == bid);
                        if is_loop_header && matches!(phi_ty, Some(sonatina_ir::Type::F32 | sonatina_ir::Type::I1)) {
                            return Err(format!(
                                "spirv: loop-header phi of type {:?} is unsupported until loop phi locals are per-value typed",
                                phi_ty.unwrap()
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
        let mut input_span = 0;
        let mut input_align = 1;
        for (i, ty) in sig.args().iter().skip(2).copied().enumerate() {
            let (naga_ty, width) = match ty {
                sonatina_ir::Type::I32 => (word_type, 4),
                sonatina_ir::Type::F32 => (f32_type, 4),
                _ => return Err(format!("spirv render: broadcast arg {} has unsupported storage type {ty:?}", i + 2)),
            };
            input_members.push(naga::StructMember { name: Some(format!("p{i}")), ty: naga_ty, binding: None, offset: input_span });
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
            }],
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
    let mut input_span = 0;
    let mut input_align = 1;
    for (i, ty) in sig.args().iter().skip(if grid { 2 } else { 0 }).copied().enumerate() {
        let (naga_ty, width) = match ty {
            sonatina_ir::Type::I32 => (word_type, 4),
            sonatina_ir::Type::I64 if word == WordKind::I64 => (word_type, 8),
            sonatina_ir::Type::F32 => (f32_type, 4),
            _ => return Err(format!("spirv: input arg {i} has unsupported storage type {ty:?}")),
        };
        input_span = (input_span + width - 1) & !(width - 1);
        input_members.push(naga::StructMember { name: Some(format!("p{i}")), ty: naga_ty, binding: None, offset: input_span });
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
            },
            SpirvBinding {
                group: 0,
                binding: 1,
                name: "input".to_string(),
                access: Access::Read,
                role: Role::Input,
                stride: input_span,
            },
        ],
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

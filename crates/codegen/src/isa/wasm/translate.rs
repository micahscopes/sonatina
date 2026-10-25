//! Translation of a wasm32 Sonatina module to a WAFFLE module.
//!
//! WAFFLE takes SSA with block parameters, so Sonatina phi nodes become block
//! parameters, and recovers structured control flow when the module is
//! encoded. The integer lowerings live in [`super::arith`].
//!
//! Every instruction of the native instruction set is matched explicitly. An
//! instruction that is not translated returns an error naming it; no
//! instruction is skipped.

use rustc_hash::{FxHashMap, FxHashSet};
use waffle::{
    BlockTarget, ExportKind, Func, FuncDecl, FunctionBody, MemoryArg, Module as WaffleModule,
    Operator, SignatureData, Terminator, Type as WType, ValueDef,
};

use sonatina_ir::{
    BlockId, ControlFlowGraph, Function, Immediate, Inst, Linkage, Module, Type, Value, ValueId,
    inst::{inst_set::InstSetExt, native::inst_set::NativeInstKind},
    isa::native::inst_set as native_inst_set,
    module::{FuncRef, ModuleCtx},
    types::CompoundType,
};

use super::{
    WasmError,
    arith::{Arith, Cmp, DivRem, Emit, Int, Shift},
};

type V = waffle::Value;

pub(super) struct Translated {
    pub(super) module: WaffleModule<'static>,
    /// Exported function names, in module order.
    pub(super) exports: Vec<String>,
}

fn unsupported(message: String) -> WasmError {
    WasmError::Unsupported(message)
}

fn invalid(message: String) -> WasmError {
    WasmError::Invalid(message)
}

/// A function the module can call: its WAFFLE index and its Sonatina signature.
struct Callee {
    func: Func,
    params: Vec<Type>,
    rets: Vec<Type>,
    /// Results of an import come from the host and are not known to be
    /// canonical.
    is_import: bool,
}

pub(super) fn translate_module(
    module: &Module,
    import_module: &str,
) -> Result<Translated, WasmError> {
    if let Some(symbol) = module
        .ctx
        .with_gv_store(|store| store.all_gv_data().next().map(|data| data.symbol.clone()))
    {
        return Err(unsupported(format!(
            "global variable `{symbol}`: global variables are not translated"
        )));
    }

    let mut wmod = WaffleModule::empty();
    // Linear memory is always present and exported, so scalar loads and stores
    // and host access share one contract. It starts at one page and may grow.
    let memory = wmod.memories.push(waffle::MemoryData {
        initial_pages: 1,
        maximum_pages: None,
        segments: vec![],
    });
    wmod.exports.push(waffle::Export {
        name: "memory".to_string(),
        kind: ExportKind::Memory(memory),
    });

    let funcs = module.funcs();
    let mut callees: FxHashMap<FuncRef, Callee> = FxHashMap::default();
    let mut exports = Vec::new();
    let mut export_names: FxHashSet<String> = FxHashSet::default();
    export_names.insert("memory".to_string());

    // WAFFLE requires imported functions to take the lowest function indexes,
    // so declare every import before any defined function.
    let mut definitions = Vec::new();
    for &func_ref in &funcs {
        let name = module.ctx.func_sig(func_ref, |sig| sig.name().to_string());
        let linkage = module.ctx.func_linkage(func_ref);
        let has_body = module
            .func_store
            .try_view(func_ref, |function| function.layout.entry_block().is_some())
            .unwrap_or(false);

        match (linkage, has_body) {
            (Linkage::External, false) => {
                let signature = declared_signature(module, func_ref, &name)?;
                let wsig = wmod.signatures.push(signature.wasm);
                let func = wmod.funcs.push(FuncDecl::Import(wsig, name.clone()));
                wmod.imports.push(waffle::Import {
                    module: import_module.to_string(),
                    name,
                    kind: waffle::ImportKind::Func(func),
                });
                callees.insert(
                    func_ref,
                    Callee {
                        func,
                        params: signature.params,
                        rets: signature.rets,
                        is_import: true,
                    },
                );
            }
            (Linkage::External, true) => {
                return Err(invalid(format!(
                    "function `{name}` is external but has a body"
                )));
            }
            (Linkage::Public | Linkage::Private, false) => {
                return Err(invalid(format!(
                    "function `{name}` is defined in this module but has no body"
                )));
            }
            (Linkage::Public | Linkage::Private, true) => {
                definitions.push((func_ref, name, linkage))
            }
        }
    }

    // Declare every defined function before translating bodies so calls
    // resolve regardless of definition order.
    let mut pending = Vec::new();
    for (func_ref, name, linkage) in definitions {
        let signature = declared_signature(module, func_ref, &name)?;
        let wsig = wmod.signatures.push(signature.wasm);
        let placeholder = FunctionBody::new(&wmod, wsig);
        let func = wmod
            .funcs
            .push(FuncDecl::Body(wsig, name.clone(), placeholder));
        let public = linkage == Linkage::Public;
        if public {
            if !export_names.insert(name.clone()) {
                return Err(invalid(format!(
                    "export name `{name}` is used by more than one function or by the memory"
                )));
            }
            wmod.exports.push(waffle::Export {
                name: name.clone(),
                kind: ExportKind::Func(func),
            });
            exports.push(name.clone());
        }
        callees.insert(
            func_ref,
            Callee {
                func,
                params: signature.params,
                rets: signature.rets,
                is_import: false,
            },
        );
        pending.push((func_ref, func, wsig, name, public));
    }

    for (func_ref, func, wsig, name, public) in pending {
        let body = module
            .func_store
            .try_view(func_ref, |function| {
                FunctionTranslator::new(
                    module, function, func_ref, &name, &wmod, wsig, memory, &callees,
                )
                .translate(public)
            })
            .ok_or_else(|| invalid(format!("function `{name}` has no body")))??;
        wmod.funcs[func] = FuncDecl::Body(wsig, name, body);
    }

    Ok(Translated {
        module: wmod,
        exports,
    })
}

struct DeclaredSignature {
    wasm: SignatureData,
    params: Vec<Type>,
    rets: Vec<Type>,
}

fn declared_signature(
    module: &Module,
    func_ref: FuncRef,
    name: &str,
) -> Result<DeclaredSignature, WasmError> {
    module.ctx.func_sig(func_ref, |sig| {
        let carrier = |ty: Type, role: String| {
            value_carrier(&module.ctx, ty).ok_or_else(|| {
                unsupported(format!(
                    "function `{name}`: {role} has type {}, which the wasm32 backend does not support",
                    type_name(&module.ctx, ty)
                ))
            })
        };
        let params = sig
            .args()
            .iter()
            .enumerate()
            .map(|(index, ty)| carrier(*ty, format!("parameter {index}")))
            .collect::<Result<Vec<_>, _>>()?;
        let returns = sig
            .ret_tys()
            .iter()
            .enumerate()
            .map(|(index, ty)| carrier(*ty, format!("result {index}")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DeclaredSignature {
            wasm: SignatureData { params, returns },
            params: sig.args().to_vec(),
            rets: sig.ret_tys().to_vec(),
        })
    })
}

/// The WebAssembly value type that carries a Sonatina value, if there is one.
fn value_carrier(ctx: &ModuleCtx, ty: Type) -> Option<WType> {
    if let Some(int) = Int::from_type(ty) {
        return Some(int.carrier());
    }
    match ty.resolve_compound(ctx) {
        // wasm32 pointers are 32-bit linear-memory offsets.
        Some(CompoundType::Ptr(_)) => Some(WType::I32),
        _ => None,
    }
}

fn type_name(ctx: &ModuleCtx, ty: Type) -> String {
    match ty {
        Type::Unit => "unit".to_string(),
        Type::I1 => "i1".to_string(),
        Type::I8 => "i8".to_string(),
        Type::I16 => "i16".to_string(),
        Type::I32 => "i32".to_string(),
        Type::I64 => "i64".to_string(),
        Type::I128 => "i128".to_string(),
        Type::I256 => "i256".to_string(),
        Type::EnumTag(_) => "an enum tag".to_string(),
        Type::Compound(_) => match ty.resolve_compound(ctx) {
            Some(CompoundType::Ptr(_)) => "a pointer",
            Some(CompoundType::ObjRef(_)) => "an object reference",
            Some(CompoundType::ConstRef(_)) => "a constant reference",
            Some(CompoundType::Array { .. }) => "an array",
            Some(CompoundType::Struct(_)) => "a struct",
            Some(CompoundType::Enum(_)) => "an enum",
            Some(CompoundType::Func { .. }) => "a function type",
            None => "an unresolved compound type",
        }
        .to_string(),
    }
}

/// How a scalar moves between linear memory and a value.
#[derive(Clone, Copy)]
enum Access {
    Byte,
    Half,
    Word,
    Double,
}

impl Access {
    fn align(self) -> u32 {
        match self {
            Self::Byte => 0,
            Self::Half => 1,
            Self::Word => 2,
            Self::Double => 3,
        }
    }
}

struct FunctionTranslator<'a> {
    module: &'a Module,
    function: &'a Function,
    name: &'a str,
    body: FunctionBody,
    memory: waffle::Memory,
    callees: &'a FxHashMap<FuncRef, Callee>,
    return_tys: Vec<Type>,
    /// Sonatina values defined so far, in canonical WAFFLE form.
    values: FxHashMap<ValueId, V>,
    blocks: FxHashMap<BlockId, waffle::Block>,
}

impl<'a> FunctionTranslator<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        module: &'a Module,
        function: &'a Function,
        func_ref: FuncRef,
        name: &'a str,
        wmod: &WaffleModule,
        wsig: waffle::Signature,
        memory: waffle::Memory,
        callees: &'a FxHashMap<FuncRef, Callee>,
    ) -> Self {
        Self {
            module,
            function,
            name,
            body: FunctionBody::new(wmod, wsig),
            memory,
            callees,
            return_tys: callees[&func_ref].rets.clone(),
            values: FxHashMap::default(),
            blocks: FxHashMap::default(),
        }
    }

    fn translate(mut self, public: bool) -> Result<FunctionBody, WasmError> {
        let function = self.function;
        let entry = function
            .layout
            .entry_block()
            .ok_or_else(|| invalid(format!("function `{}` has no entry block", self.name)))?;

        // Blocks unreachable from the entry never run and are not translated.
        let mut cfg = ControlFlowGraph::default();
        cfg.compute(function);
        let mut order: Vec<BlockId> = cfg.post_order().collect();
        order.reverse();

        // A Sonatina entry block may be a loop header, but WebAssembly cannot
        // branch to the start of a function body. Enter such a block through
        // a separate prologue.
        let needs_prologue = cfg.preds_of(entry).next().is_some();
        for &block in &order {
            let wb = if block == entry && !needs_prologue {
                self.body.entry
            } else {
                self.body.add_block()
            };
            self.blocks.insert(block, wb);
        }

        let params = self.body.blocks[self.body.entry].params.clone();
        for (index, &arg) in function.arg_values.iter().enumerate() {
            let mut value = params[index].1;
            if public && let Some(int) = Int::from_type(function.dfg.value_ty(arg)) {
                // The host may pass any bits above the width of a narrow
                // parameter.
                let block = self.body.entry;
                value = self.emit(block).canonicalize(int, value);
            }
            self.values.insert(arg, value);
        }
        if needs_prologue {
            let target = self.blocks[&entry];
            self.body.set_terminator(
                self.body.entry,
                Terminator::Br {
                    target: BlockTarget {
                        block: target,
                        args: vec![],
                    },
                },
            );
        }

        for &block in &order {
            for inst_id in function.layout.iter_inst(block) {
                let inst = function.dfg.inst(inst_id);
                let NativeInstKind::Phi(_) = self.resolve(inst)? else {
                    break;
                };
                if block == entry {
                    return Err(invalid(format!(
                        "function `{}`: the entry block contains a phi",
                        self.name
                    )));
                }
                let result = function.dfg.inst_result(inst_id).ok_or_else(|| {
                    invalid(format!("function `{}`: a phi has no result", self.name))
                })?;
                let ty = function.dfg.value_ty(result);
                let carrier = value_carrier(&self.module.ctx, ty).ok_or_else(|| {
                    unsupported(format!(
                        "function `{}`: `phi` of type {}, which the wasm32 backend does not support",
                        self.name,
                        type_name(&self.module.ctx, ty)
                    ))
                })?;
                let param = self.body.add_blockparam(self.blocks[&block], carrier);
                self.values.insert(result, param);
            }
        }

        for &block in &order {
            self.translate_block(block)?;
        }

        // WAFFLE may recompute a pure value at each use in another block.
        // Sonatina values can depend on memory, so make every cross-block use
        // an explicit block parameter to keep the definition-time snapshot.
        self.body.convert_to_max_ssa(None);
        Ok(self.body)
    }

    fn emit(&mut self, block: waffle::Block) -> Emit<'_> {
        Emit {
            body: &mut self.body,
            block,
        }
    }

    fn resolve(&self, inst: &'a dyn Inst) -> Result<NativeInstKind<'a>, WasmError> {
        let inst_set = native_inst_set();
        if !inst_set.contains_inst(inst) {
            return Err(unsupported(format!(
                "function `{}`: instruction `{}` is not in the wasm32 instruction set",
                self.name,
                inst.as_text()
            )));
        }
        Ok(inst_set.resolve_inst(inst))
    }

    fn unsupported_inst(&self, op: &str) -> WasmError {
        unsupported(format!(
            "function `{}`: instruction `{op}` is not supported by the wasm32 backend",
            self.name
        ))
    }

    /// The integer type of an operand, or an error naming the type and the
    /// instruction that needs it.
    fn int_ty(&self, op: &str, ty: Type) -> Result<Int, WasmError> {
        Int::from_type(ty).ok_or_else(|| {
            unsupported(format!(
                "function `{}`: `{op}` on {}, which the wasm32 backend does not support \
                 (integers of 1, 8, 16, 32 and 64 bits are supported)",
                self.name,
                type_name(&self.module.ctx, ty)
            ))
        })
    }

    fn is_pointer(&self, ty: Type) -> bool {
        matches!(
            ty.resolve_compound(&self.module.ctx),
            Some(CompoundType::Ptr(_))
        )
    }

    fn operand(&mut self, wb: waffle::Block, id: ValueId) -> Result<V, WasmError> {
        if let Some(&value) = self.values.get(&id) {
            return Ok(value);
        }
        let function = self.function;
        match function.dfg.value(id) {
            Value::Immediate { imm, .. } => {
                let mut emit = self.emit(wb);
                match *imm {
                    Immediate::I1(value) => Ok(emit.c32(u32::from(value))),
                    Immediate::I8(value) => Ok(emit.c32(u32::from(value as u8))),
                    Immediate::I16(value) => Ok(emit.c32(u32::from(value as u16))),
                    Immediate::I32(value) => Ok(emit.c32(value as u32)),
                    Immediate::I64(value) => Ok(emit.c64(value as u64)),
                    other => Err(self.unsupported_type("an immediate", other.ty())),
                }
            }
            Value::Undef { ty } => {
                // An undefined value is materialized as zero, as the native
                // backend does.
                if let Some(int) = Int::from_type(*ty) {
                    Ok(self.emit(wb).zero(int))
                } else if self.is_pointer(*ty) {
                    Ok(self.emit(wb).c32(0))
                } else {
                    Err(self.unsupported_type("an undefined value", *ty))
                }
            }
            Value::Global { .. } => Err(unsupported(format!(
                "function `{}`: global variable operands are not supported",
                self.name
            ))),
            Value::Arg { .. } | Value::Inst { .. } => Err(invalid(format!(
                "function `{}`: value v{} is used before it is defined",
                self.name, id.0
            ))),
        }
    }

    fn unsupported_type(&self, what: &str, ty: Type) -> WasmError {
        unsupported(format!(
            "function `{}`: {what} of type {}, which the wasm32 backend does not support",
            self.name,
            type_name(&self.module.ctx, ty)
        ))
    }

    /// Record result `index` of an instruction. `expected` is the type the
    /// lowering produced, which must be the declared type of the result.
    fn define(
        &mut self,
        inst_id: sonatina_ir::InstId,
        index: usize,
        value: V,
        expected: Type,
    ) -> Result<(), WasmError> {
        let function = self.function;
        let result = function
            .dfg
            .inst_results(inst_id)
            .get(index)
            .copied()
            .ok_or_else(|| {
                invalid(format!(
                    "function `{}`: `{}` has no result {index}",
                    self.name,
                    function.dfg.inst(inst_id).as_text()
                ))
            })?;
        let actual = function.dfg.value_ty(result);
        if actual != expected {
            return Err(invalid(format!(
                "function `{}`: `{}` produces {} but its result is declared as {}",
                self.name,
                function.dfg.inst(inst_id).as_text(),
                type_name(&self.module.ctx, expected),
                type_name(&self.module.ctx, actual)
            )));
        }
        self.values.insert(result, value);
        Ok(())
    }

    /// Operands of a two-operand integer instruction, which must share a type.
    fn binary_operands(
        &mut self,
        wb: waffle::Block,
        op: &str,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(Int, Type, V, V), WasmError> {
        let ty = self.function.dfg.value_ty(lhs);
        let int = self.int_ty(op, ty)?;
        if self.function.dfg.value_ty(rhs) != ty {
            return Err(invalid(format!(
                "function `{}`: `{op}` operands have different types",
                self.name
            )));
        }
        let lhs = self.operand(wb, lhs)?;
        let rhs = self.operand(wb, rhs)?;
        Ok((int, ty, lhs, rhs))
    }

    fn unary_operand(
        &mut self,
        wb: waffle::Block,
        op: &str,
        arg: ValueId,
    ) -> Result<(Int, Type, V), WasmError> {
        let ty = self.function.dfg.value_ty(arg);
        let int = self.int_ty(op, ty)?;
        let arg = self.operand(wb, arg)?;
        Ok((int, ty, arg))
    }

    /// An integer instruction with one result of the operand type.
    fn binary(
        &mut self,
        wb: waffle::Block,
        inst_id: sonatina_ir::InstId,
        (lhs, rhs): (ValueId, ValueId),
        lower: impl FnOnce(&mut Emit<'_>, Int, V, V) -> V,
    ) -> Result<(), WasmError> {
        let op = self.function.dfg.inst(inst_id).as_text();
        let (int, ty, lhs, rhs) = self.binary_operands(wb, op, lhs, rhs)?;
        let value = lower(&mut self.emit(wb), int, lhs, rhs);
        self.define(inst_id, 0, value, ty)
    }

    fn unary(
        &mut self,
        wb: waffle::Block,
        inst_id: sonatina_ir::InstId,
        arg: ValueId,
        lower: impl FnOnce(&mut Emit<'_>, Int, V) -> V,
    ) -> Result<(), WasmError> {
        let op = self.function.dfg.inst(inst_id).as_text();
        let (int, ty, arg) = self.unary_operand(wb, op, arg)?;
        let value = lower(&mut self.emit(wb), int, arg);
        self.define(inst_id, 0, value, ty)
    }

    fn compare(
        &mut self,
        wb: waffle::Block,
        inst_id: sonatina_ir::InstId,
        (lhs, rhs): (ValueId, ValueId),
        cmp: Cmp,
        signed: bool,
    ) -> Result<(), WasmError> {
        let op = self.function.dfg.inst(inst_id).as_text();
        let (int, _, lhs, rhs) = self.binary_operands(wb, op, lhs, rhs)?;
        let value = self.emit(wb).compare(int, cmp, signed, lhs, rhs);
        self.define(inst_id, 0, value, Type::I1)
    }

    /// An integer instruction with a result and an overflow flag.
    fn overflow(
        &mut self,
        wb: waffle::Block,
        inst_id: sonatina_ir::InstId,
        (lhs, rhs): (ValueId, ValueId),
        arith: Arith,
        signed: bool,
    ) -> Result<(), WasmError> {
        let op = self.function.dfg.inst(inst_id).as_text();
        let (int, ty, lhs, rhs) = self.binary_operands(wb, op, lhs, rhs)?;
        let (value, overflow) = self.emit(wb).overflow(int, arith, signed, lhs, rhs);
        self.define(inst_id, 0, value, ty)?;
        self.define(inst_id, 1, overflow, Type::I1)
    }

    fn shift(
        &mut self,
        wb: waffle::Block,
        inst_id: sonatina_ir::InstId,
        (value, bits): (ValueId, ValueId),
        kind: Shift,
    ) -> Result<(), WasmError> {
        // The count has the type of the value.
        self.binary(wb, inst_id, (value, bits), |emit, int, value, bits| {
            emit.shift(int, kind, value, bits)
        })
    }

    fn address(&mut self, wb: waffle::Block, op: &str, addr: ValueId) -> Result<V, WasmError> {
        let ty = self.function.dfg.value_ty(addr);
        if ty != Type::I32 && !self.is_pointer(ty) {
            return Err(unsupported(format!(
                "function `{}`: `{op}` address of type {}; wasm32 addresses are pointers or i32",
                self.name,
                type_name(&self.module.ctx, ty)
            )));
        }
        self.operand(wb, addr)
    }

    fn access(&self, op: &str, ty: Type) -> Result<Access, WasmError> {
        match ty {
            Type::I1 | Type::I8 => Ok(Access::Byte),
            Type::I16 => Ok(Access::Half),
            Type::I32 => Ok(Access::Word),
            Type::I64 => Ok(Access::Double),
            _ if self.is_pointer(ty) => Ok(Access::Word),
            _ => Err(unsupported(format!(
                "function `{}`: `{op}` of {}; only scalar integers and pointers are stored in memory",
                self.name,
                type_name(&self.module.ctx, ty)
            ))),
        }
    }

    fn memarg(&self, access: Access) -> MemoryArg {
        MemoryArg {
            align: access.align(),
            offset: 0,
            memory: self.memory,
        }
    }

    /// The branch target of the edge `from` to `to`, with the phi inputs that
    /// `from` supplies. Inputs are materialized in the WAFFLE block `wb`.
    fn edge(
        &mut self,
        from: BlockId,
        to: BlockId,
        wb: waffle::Block,
    ) -> Result<BlockTarget, WasmError> {
        let function = self.function;
        let block = *self.blocks.get(&to).ok_or_else(|| {
            invalid(format!(
                "function `{}`: branch to block{} which is not reachable from the entry",
                self.name,
                to.as_u32()
            ))
        })?;
        let mut args = Vec::new();
        for inst_id in function.layout.iter_inst(to) {
            let NativeInstKind::Phi(phi) = self.resolve(function.dfg.inst(inst_id))? else {
                break;
            };
            let input = phi
                .args()
                .iter()
                .find(|(_, pred)| *pred == from)
                .map(|(value, _)| *value)
                .ok_or_else(|| {
                    invalid(format!(
                        "function `{}`: a phi in block{} has no input for block{}",
                        self.name,
                        to.as_u32(),
                        from.as_u32()
                    ))
                })?;
            let result = function
                .dfg
                .inst_result(inst_id)
                .ok_or_else(|| invalid(format!("function `{}`: a phi has no result", self.name)))?;
            if function.dfg.value_ty(input) != function.dfg.value_ty(result) {
                return Err(invalid(format!(
                    "function `{}`: a phi input from block{} has the wrong type",
                    self.name,
                    from.as_u32()
                )));
            }
            args.push(self.operand(wb, input)?);
        }
        Ok(BlockTarget { block, args })
    }

    fn trap_block(&mut self) -> waffle::Block {
        let block = self.body.add_block();
        self.body.set_terminator(block, Terminator::Unreachable);
        block
    }

    fn translate_block(&mut self, block: BlockId) -> Result<(), WasmError> {
        let wb = self.blocks[&block];
        let function = self.function;
        for inst_id in function.layout.iter_inst(block) {
            let inst = function.dfg.inst(inst_id);
            self.translate_inst(block, wb, inst_id, inst)?;
            if inst.is_terminator() {
                return Ok(());
            }
        }
        Err(invalid(format!(
            "function `{}`: block{} does not end in a terminator",
            self.name,
            block.as_u32()
        )))
    }

    fn translate_inst(
        &mut self,
        block: BlockId,
        wb: waffle::Block,
        inst_id: sonatina_ir::InstId,
        inst: &'a dyn Inst,
    ) -> Result<(), WasmError> {
        let function = self.function;
        let op = inst.as_text();
        match self.resolve(inst)? {
            // Phi inputs are passed as block arguments by the predecessors.
            NativeInstKind::Phi(_) => {}

            NativeInstKind::Add(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.add_sub_mul(int, Arith::Add, l, r)
                })?
            }
            NativeInstKind::Sub(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.add_sub_mul(int, Arith::Sub, l, r)
                })?
            }
            NativeInstKind::Mul(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.add_sub_mul(int, Arith::Mul, l, r)
                })?
            }
            NativeInstKind::Neg(i) => {
                self.unary(wb, inst_id, *i.arg(), |e, int, v| e.neg(int, v))?
            }
            NativeInstKind::Not(i) => {
                self.unary(wb, inst_id, *i.arg(), |e, int, v| e.not(int, v))?
            }
            NativeInstKind::Bswap(i) => {
                if function.dfg.value_ty(*i.arg()) == Type::I1 {
                    return Err(self.unsupported_type("`bswap`", Type::I1));
                }
                self.unary(wb, inst_id, *i.arg(), |e, int, v| e.bswap(int, v))?
            }
            NativeInstKind::And(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.bitwise(int, Operator::I32And, Operator::I64And, l, r)
                })?
            }
            NativeInstKind::Or(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.bitwise(int, Operator::I32Or, Operator::I64Or, l, r)
                })?
            }
            NativeInstKind::Xor(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.bitwise(int, Operator::I32Xor, Operator::I64Xor, l, r)
                })?
            }

            NativeInstKind::Udiv(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.div_rem(int, DivRem::Udiv, l, r)
                })?
            }
            NativeInstKind::Sdiv(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.div_rem(int, DivRem::Sdiv, l, r)
                })?
            }
            NativeInstKind::Umod(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.div_rem(int, DivRem::Umod, l, r)
                })?
            }
            NativeInstKind::Smod(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.div_rem(int, DivRem::Smod, l, r)
                })?
            }

            NativeInstKind::Shl(i) => {
                self.shift(wb, inst_id, (*i.value(), *i.bits()), Shift::Shl)?
            }
            NativeInstKind::Shr(i) => {
                self.shift(wb, inst_id, (*i.value(), *i.bits()), Shift::Shr)?
            }
            NativeInstKind::Sar(i) => {
                self.shift(wb, inst_id, (*i.value(), *i.bits()), Shift::Sar)?
            }

            NativeInstKind::Uaddo(i) => {
                self.overflow(wb, inst_id, (*i.lhs(), *i.rhs()), Arith::Add, false)?
            }
            NativeInstKind::Usubo(i) => {
                self.overflow(wb, inst_id, (*i.lhs(), *i.rhs()), Arith::Sub, false)?
            }
            NativeInstKind::Umulo(i) => {
                self.overflow(wb, inst_id, (*i.lhs(), *i.rhs()), Arith::Mul, false)?
            }
            NativeInstKind::Saddo(i) => {
                self.overflow(wb, inst_id, (*i.lhs(), *i.rhs()), Arith::Add, true)?
            }
            NativeInstKind::Ssubo(i) => {
                self.overflow(wb, inst_id, (*i.lhs(), *i.rhs()), Arith::Sub, true)?
            }
            NativeInstKind::Smulo(i) => {
                self.overflow(wb, inst_id, (*i.lhs(), *i.rhs()), Arith::Mul, true)?
            }
            NativeInstKind::Snego(i) => {
                let (int, ty, arg) = self.unary_operand(wb, op, *i.arg())?;
                let (value, overflow) = self.emit(wb).signed_neg_overflow(int, arg);
                self.define(inst_id, 0, value, ty)?;
                self.define(inst_id, 1, overflow, Type::I1)?;
            }
            NativeInstKind::Uaddsat(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.saturating(int, Arith::Add, false, l, r)
                })?
            }
            NativeInstKind::Usubsat(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.saturating(int, Arith::Sub, false, l, r)
                })?
            }
            NativeInstKind::Umulsat(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.saturating(int, Arith::Mul, false, l, r)
                })?
            }
            NativeInstKind::Saddsat(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.saturating(int, Arith::Add, true, l, r)
                })?
            }
            NativeInstKind::Ssubsat(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.saturating(int, Arith::Sub, true, l, r)
                })?
            }
            NativeInstKind::Smulsat(i) => {
                self.binary(wb, inst_id, (*i.lhs(), *i.rhs()), |e, int, l, r| {
                    e.saturating(int, Arith::Mul, true, l, r)
                })?
            }

            NativeInstKind::Eq(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Eq, false)?
            }
            NativeInstKind::Ne(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Ne, false)?
            }
            NativeInstKind::Lt(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Lt, false)?
            }
            NativeInstKind::Gt(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Gt, false)?
            }
            NativeInstKind::Le(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Le, false)?
            }
            NativeInstKind::Ge(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Ge, false)?
            }
            NativeInstKind::Slt(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Lt, true)?
            }
            NativeInstKind::Sgt(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Gt, true)?
            }
            NativeInstKind::Sle(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Le, true)?
            }
            NativeInstKind::Sge(i) => {
                self.compare(wb, inst_id, (*i.lhs(), *i.rhs()), Cmp::Ge, true)?
            }
            NativeInstKind::IsZero(i) => {
                let (int, _, arg) = self.unary_operand(wb, op, *i.lhs())?;
                let value = self.emit(wb).is_zero(int, arg);
                self.define(inst_id, 0, value, Type::I1)?;
            }

            NativeInstKind::Zext(i) => self.convert(wb, inst_id, op, *i.from(), *i.ty(), false)?,
            NativeInstKind::Sext(i) => self.convert(wb, inst_id, op, *i.from(), *i.ty(), true)?,
            NativeInstKind::Trunc(i) => self.convert(wb, inst_id, op, *i.from(), *i.ty(), false)?,
            NativeInstKind::Bitcast(i) => {
                let from_ty = function.dfg.value_ty(*i.from());
                let to_ty = *i.ty();
                let width = |ty: Type| match ty {
                    Type::I1 => Some(1),
                    Type::I8 => Some(8),
                    Type::I16 => Some(16),
                    Type::I32 => Some(32),
                    Type::I64 => Some(64),
                    _ if self.is_pointer(ty) => Some(32),
                    _ => None,
                };
                let (Some(from_width), Some(to_width)) = (width(from_ty), width(to_ty)) else {
                    let unsupported_ty = if width(from_ty).is_none() {
                        from_ty
                    } else {
                        to_ty
                    };
                    return Err(self.unsupported_type("`bitcast`", unsupported_ty));
                };
                if from_width != to_width {
                    return Err(invalid(format!(
                        "function `{}`: `bitcast` between types of different widths",
                        self.name
                    )));
                }
                // Same-width scalars and pointers share one representation.
                let value = self.operand(wb, *i.from())?;
                self.define(inst_id, 0, value, to_ty)?;
            }
            NativeInstKind::IntToPtr(i) => {
                let to_ty = *i.ty();
                if !self.is_pointer(to_ty) {
                    return Err(self.unsupported_type("`int_to_ptr` result", to_ty));
                }
                let (int, _, arg) = self.unary_operand(wb, op, *i.from())?;
                let value = self.emit(wb).resize(int, Int::I32, false, arg);
                self.define(inst_id, 0, value, to_ty)?;
            }
            NativeInstKind::PtrToInt(i) => {
                let from_ty = function.dfg.value_ty(*i.from());
                if !self.is_pointer(from_ty) {
                    return Err(self.unsupported_type("`ptr_to_int` operand", from_ty));
                }
                let to_ty = *i.ty();
                let to = self.int_ty(op, to_ty)?;
                let arg = self.operand(wb, *i.from())?;
                let value = self.emit(wb).resize(Int::I32, to, false, arg);
                self.define(inst_id, 0, value, to_ty)?;
            }

            NativeInstKind::Mload(i) => {
                let ty = *i.ty();
                let access = self.access(op, ty)?;
                let addr = self.address(wb, op, *i.addr())?;
                let memory = self.memarg(access);
                let (load, carrier) = match access {
                    Access::Byte => (Operator::I32Load8U { memory }, WType::I32),
                    Access::Half => (Operator::I32Load16U { memory }, WType::I32),
                    Access::Word => (Operator::I32Load { memory }, WType::I32),
                    Access::Double => (Operator::I64Load { memory }, WType::I64),
                };
                let mut value = self.body.add_op(wb, load, &[addr], &[carrier]);
                if ty == Type::I1 {
                    // Only the low bit of a stored byte is the value.
                    value = self.emit(wb).canonicalize(Int::I1, value);
                }
                self.define(inst_id, 0, value, ty)?;
            }
            NativeInstKind::Mstore(i) => {
                let ty = *i.ty();
                let access = self.access(op, ty)?;
                if function.dfg.value_ty(*i.value()) != ty {
                    return Err(invalid(format!(
                        "function `{}`: `mstore` value does not have the stored type",
                        self.name
                    )));
                }
                let addr = self.address(wb, op, *i.addr())?;
                let value = self.operand(wb, *i.value())?;
                let memory = self.memarg(access);
                let store = match access {
                    Access::Byte => Operator::I32Store8 { memory },
                    Access::Half => Operator::I32Store16 { memory },
                    Access::Word => Operator::I32Store { memory },
                    Access::Double => Operator::I64Store { memory },
                };
                self.body.add_op(wb, store, &[addr, value], &[]);
            }

            NativeInstKind::Call(call) => {
                self.call(wb, inst_id, op, *call.callee(), call.args())?
            }

            NativeInstKind::Jump(jump) => {
                let target = self.edge(block, *jump.dest(), wb)?;
                self.body.set_terminator(wb, Terminator::Br { target });
            }
            NativeInstKind::Br(br) => {
                let cond_ty = function.dfg.value_ty(*br.cond());
                let int = self.int_ty(op, cond_ty)?;
                let cond = self.operand(wb, *br.cond())?;
                let nonzero = self.edge(block, *br.nz_dest(), wb)?;
                let zero = self.edge(block, *br.z_dest(), wb)?;
                let terminator = if int.is_wide() {
                    // `br_if` takes an i32, so branch on the i64 being zero.
                    let is_zero = self.emit(wb).is_zero(int, cond);
                    Terminator::CondBr {
                        cond: is_zero,
                        if_true: zero,
                        if_false: nonzero,
                    }
                } else {
                    Terminator::CondBr {
                        cond,
                        if_true: nonzero,
                        if_false: zero,
                    }
                };
                self.body.set_terminator(wb, terminator);
            }
            NativeInstKind::BrTable(table) => {
                let scrutinee_ty = function.dfg.value_ty(*table.scrutinee());
                let int = self.int_ty(op, scrutinee_ty)?;
                let scrutinee = self.operand(wb, *table.scrutinee())?;
                let default = *table.default();

                // Each case is tested in turn; the edge arguments are
                // materialized in the block that takes the edge.
                let mut current = wb;
                for (index, &(case, dest)) in table.table().iter().enumerate() {
                    if function.dfg.value_ty(case) != scrutinee_ty {
                        return Err(invalid(format!(
                            "function `{}`: a `br_table` case has the wrong type",
                            self.name
                        )));
                    }
                    let case = self.operand(current, case)?;
                    let matched = self
                        .emit(current)
                        .compare(int, Cmp::Eq, false, scrutinee, case);
                    let if_true = self.edge(block, dest, current)?;
                    let if_false = if index + 1 < table.table().len() {
                        let next = self.body.add_block();
                        BlockTarget {
                            block: next,
                            args: vec![],
                        }
                    } else if let Some(default) = default {
                        self.edge(block, default, current)?
                    } else {
                        BlockTarget {
                            block: self.trap_block(),
                            args: vec![],
                        }
                    };
                    let next = if_false.block;
                    self.body.set_terminator(
                        current,
                        Terminator::CondBr {
                            cond: matched,
                            if_true,
                            if_false,
                        },
                    );
                    current = next;
                }
                if table.table().is_empty() {
                    let default = default.ok_or_else(|| {
                        invalid(format!(
                            "function `{}`: an empty `br_table` has no default",
                            self.name
                        ))
                    })?;
                    let target = self.edge(block, default, wb)?;
                    self.body.set_terminator(wb, Terminator::Br { target });
                }
            }
            NativeInstKind::Return(ret) => {
                let args = ret.args().as_slice();
                if args.len() != self.return_tys.len() {
                    return Err(invalid(format!(
                        "function `{}`: `return` has {} values but the signature has {}",
                        self.name,
                        args.len(),
                        self.return_tys.len()
                    )));
                }
                let mut values = Vec::with_capacity(args.len());
                for (index, &arg) in args.iter().enumerate() {
                    if function.dfg.value_ty(arg) != self.return_tys[index] {
                        return Err(invalid(format!(
                            "function `{}`: `return` value {index} does not have the result type",
                            self.name
                        )));
                    }
                    values.push(self.operand(wb, arg)?);
                }
                self.body.set_terminator(wb, Terminator::Return { values });
            }
            NativeInstKind::Unreachable(_) => {
                self.body.set_terminator(wb, Terminator::Unreachable);
            }

            // Not part of the integer subset: aggregates, objects, enums,
            // references, symbols, function pointers and bulk memory.
            NativeInstKind::Memzero(_)
            | NativeInstKind::Gep(_)
            | NativeInstKind::GetFunctionPtr(_)
            | NativeInstKind::SymAddr(_)
            | NativeInstKind::SymSize(_)
            | NativeInstKind::Alloca(_)
            | NativeInstKind::ObjAlloc(_)
            | NativeInstKind::ConstRef(_)
            | NativeInstKind::ConstProj(_)
            | NativeInstKind::ConstIndex(_)
            | NativeInstKind::ConstLoad(_)
            | NativeInstKind::ObjProj(_)
            | NativeInstKind::ObjIndex(_)
            | NativeInstKind::ObjLoad(_)
            | NativeInstKind::ObjStore(_)
            | NativeInstKind::ObjInitConst(_)
            | NativeInstKind::EnumMake(_)
            | NativeInstKind::EnumTag(_)
            | NativeInstKind::EnumIsVariant(_)
            | NativeInstKind::EnumAssertVariant(_)
            | NativeInstKind::EnumAssertVariantRef(_)
            | NativeInstKind::EnumExtract(_)
            | NativeInstKind::EnumSetTag(_)
            | NativeInstKind::EnumWriteVariant(_)
            | NativeInstKind::EnumGetTag(_)
            | NativeInstKind::EnumProj(_)
            | NativeInstKind::ObjMaterializeStack(_)
            | NativeInstKind::ObjMaterializeHeap(_)
            | NativeInstKind::MemAllocDynamic(_)
            | NativeInstKind::InsertValue(_)
            | NativeInstKind::ExtractValue(_) => return Err(self.unsupported_inst(op)),
        }
        Ok(())
    }

    /// `zext`, `sext` and `trunc`: change the width of an integer.
    fn convert(
        &mut self,
        wb: waffle::Block,
        inst_id: sonatina_ir::InstId,
        op: &str,
        from: ValueId,
        to_ty: Type,
        signed: bool,
    ) -> Result<(), WasmError> {
        let (from_int, _, value) = self.unary_operand(wb, op, from)?;
        let to_int = self.int_ty(op, to_ty)?;
        let value = self.emit(wb).resize(from_int, to_int, signed, value);
        self.define(inst_id, 0, value, to_ty)
    }

    fn call(
        &mut self,
        wb: waffle::Block,
        inst_id: sonatina_ir::InstId,
        op: &str,
        callee: FuncRef,
        call_args: &[ValueId],
    ) -> Result<(), WasmError> {
        let function = self.function;
        let callees = self.callees;
        let callee = callees.get(&callee).ok_or_else(|| {
            invalid(format!(
                "function `{}`: `{op}` of a function that is not declared in the module",
                self.name
            ))
        })?;
        if call_args.len() != callee.params.len() {
            return Err(invalid(format!(
                "function `{}`: `{op}` passes {} arguments but the callee takes {}",
                self.name,
                call_args.len(),
                callee.params.len()
            )));
        }
        let mut args = Vec::with_capacity(call_args.len());
        for (index, &arg) in call_args.iter().enumerate() {
            if function.dfg.value_ty(arg) != callee.params[index] {
                return Err(invalid(format!(
                    "function `{}`: `{op}` argument {index} does not have the parameter type",
                    self.name
                )));
            }
            args.push(self.operand(wb, arg)?);
        }
        let results = function.dfg.inst_results(inst_id);
        if results.len() != callee.rets.len() {
            return Err(invalid(format!(
                "function `{}`: `{op}` has {} results but the callee returns {}",
                self.name,
                results.len(),
                callee.rets.len()
            )));
        }

        let call = Operator::Call {
            function_index: callee.func,
        };
        if results.is_empty() {
            self.body.add_op(wb, call, &args, &[]);
            return Ok(());
        }
        let carriers: Vec<WType> = callee
            .rets
            .iter()
            .map(|ty| value_carrier(&self.module.ctx, *ty).expect("signature types were checked"))
            .collect();
        // WAFFLE stores a multi-value operator's results in forward order
        // while the stack holds them in reverse, so declare the physical order
        // and pick each result from the end.
        let physical: Vec<WType> = carriers.iter().rev().copied().collect();
        let call_value = self.body.add_op(wb, call, &args, &physical);
        let count = carriers.len();
        for (index, carrier) in carriers.into_iter().enumerate() {
            let mut value = if count == 1 {
                call_value
            } else {
                let picked = self.body.add_value(ValueDef::PickOutput(
                    call_value,
                    (count - 1 - index) as u32,
                    carrier,
                ));
                self.body.append_to_block(wb, picked);
                picked
            };
            if callee.is_import
                && let Some(int) = Int::from_type(callee.rets[index])
            {
                // The host may return any bits above the width of a narrow
                // result.
                value = self.emit(wb).canonicalize(int, value);
            }
            self.define(inst_id, index, value, callee.rets[index])?;
        }
        Ok(())
    }
}

use std::collections::HashMap;

use cranelift_codegen::ir::{self as clif, InstBuilder, instructions::BlockArg};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{FuncId, Linkage, Module as ClifModule};
use cranelift_jit::JITModule;

use sonatina_ir::{
    BlockId, Function, Immediate, InstSetExt, Module, Signature, Type, Value, ValueId,
    inst::native::inst_set::{NativeInstKind, NativeInstSet},
    module::FuncRef,
};

pub(super) fn translate_module(
    module: &Module,
    jit: &mut JITModule,
) -> Result<HashMap<String, FuncId>, String> {
    let mut func_map: HashMap<String, FuncId> = HashMap::new();
    let mut func_id_map: HashMap<FuncRef, FuncId> = HashMap::new();
    let mut runtime_declarations = HashMap::new();

    let funcs = module.funcs();

    for &func_ref in &funcs {
        let has_body = module
            .func_store
            .try_view(func_ref, |function| function.layout.entry_block().is_some())
            .unwrap_or(false);
        if !has_body {
            let runtime = module.ctx.func_sig(func_ref, |sig| {
                let runtime = match sig.name() {
                    "addmod" | "__addmod" | "__u256_addmod" => "__u256_addmod",
                    "mulmod" | "__mulmod" | "__u256_mulmod" => "__u256_mulmod",
                    name => return Err(format!("unsupported native runtime declaration `{name}`")),
                };
                if sig.args() != [Type::I256; 3] || sig.ret_tys() != [Type::I256] {
                    return Err(format!(
                        "native runtime declaration `{}` requires (i256, i256, i256) -> i256",
                        sig.name()
                    ));
                }
                Ok(runtime)
            })?;
            runtime_declarations.insert(func_ref, runtime);
            continue;
        }
        let (name, sig) = module.ctx.func_sig(func_ref, |sig| {
            let name = sig.name().to_string();
            let clif_sig = sonatina_sig_to_clif(sig, jit);
            (name, clif_sig)
        });

        let func_id = jit
            .declare_function(&name, Linkage::Export, &sig)
            .map_err(|e| format!("failed to declare function {name}: {e}"))?;

        func_map.insert(name, func_id);
        func_id_map.insert(func_ref, func_id);
    }

    for &func_ref in &funcs {
        let name = module.ctx.func_sig(func_ref, |sig| sig.name().to_string());
        if runtime_declarations.contains_key(&func_ref) {
            continue;
        }

        let translated = module.func_store.try_view(func_ref, |function| {
            if function.layout.entry_block().is_none() {
                return Err("authored function has no entry block".to_string());
            }
            let func_id = func_id_map[&func_ref];
            translate_function(
                module,
                function,
                func_ref,
                func_id,
                &func_id_map,
                &runtime_declarations,
                jit,
            )
        });
        translated
            .ok_or_else(|| format!("missing authored function body {name}"))?
            .map_err(|e| format!("failed to translate function {name}: {e}"))?;
    }

    Ok(func_map)
}

fn returns_struct(sig: &Signature) -> bool {
    sig.ret_tys()
        .iter()
        .any(|ty| matches!(ty, Type::Compound(_)))
}

/// Cranelift's host ABIs do not admit an arbitrary number of direct scalar
/// results. Keep the Sonatina multi-result IR intact, but lower result lists
/// wider than the common two-register envelope through one caller-owned
/// buffer. Every scalar occupies one sixteen-byte slot, preserving its own
/// bit-width in the low-address bytes. This is a private JIT ABI shared by
/// declarations, definitions, and calls below; it is not exposed as a C
/// struct layout.
fn returns_many_scalars(sig: &Signature) -> bool {
    sig.ret_tys().len() > 2 && !returns_struct(sig)
}

fn returns_indirectly(sig: &Signature) -> bool {
    returns_struct(sig) || returns_many_scalars(sig)
}

fn scalar_return_buffer_size(arity: usize) -> u32 {
    u32::try_from(arity)
        .expect("native scalar return arity exceeds u32")
        .checked_mul(16)
        .expect("native scalar return buffer size overflow")
}

fn sonatina_sig_to_clif(sig: &Signature, jit: &JITModule) -> clif::Signature {
    let mut clif_sig = jit.make_signature();

    // If returning a struct, add hidden sret pointer as first param
    if returns_indirectly(sig) {
        clif_sig.params.push(clif::AbiParam::new(clif::types::I64));
    }

    for &arg_ty in sig.args() {
        if let Some(clif_ty) = sonatina_type_to_clif(arg_ty) {
            clif_sig.params.push(clif::AbiParam::new(clif_ty));
        }
    }

    if returns_indirectly(sig) {
        // Aggregate or wide scalar-list return via sret pointer.
    } else {
        for &ret_ty in sig.ret_tys() {
            if let Some(clif_ty) = sonatina_type_to_clif(ret_ty) {
                clif_sig.returns.push(clif::AbiParam::new(clif_ty));
            }
        }
    }
    clif_sig
}

fn sonatina_type_to_clif(ty: Type) -> Option<clif::Type> {
    match ty {
        Type::Unit => None,
        Type::I1 => Some(clif::types::I8),
        Type::I8 => Some(clif::types::I8),
        Type::I16 => Some(clif::types::I16),
        Type::I32 => Some(clif::types::I32),
        Type::I64 => Some(clif::types::I64),
        Type::I128 => Some(clif::types::I128),
        // I256: represent as pointer to 32 bytes on stack
        Type::I256 => Some(clif::types::I64),
        // Compound types (objref, constref, ptr) → native pointer
        Type::Compound(_) => Some(clif::types::I64),
        Type::F32 => Some(clif::types::F32),
        _ => None,
    }
}

fn sonatina_type_to_clif_or_err(ty: Type) -> Result<clif::Type, String> {
    sonatina_type_to_clif(ty).ok_or_else(|| format!("unsupported type for cranelift: {ty:?}"))
}

fn translate_function(
    module: &Module,
    function: &Function,
    func_ref: FuncRef,
    func_id: FuncId,
    func_id_map: &HashMap<FuncRef, FuncId>,
    runtime_declarations: &HashMap<FuncRef, &'static str>,
    jit: &mut JITModule,
) -> Result<(), String> {
    let mut ctx = jit.make_context();
    let sig = module
        .ctx
        .func_sig(func_ref, |sig| sonatina_sig_to_clif(sig, jit));
    ctx.func.signature = sig;

    let mut builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    let mut block_map: HashMap<BlockId, clif::Block> = HashMap::new();
    let mut value_map: HashMap<ValueId, clif::Value> = HashMap::new();
    let mut var_map: HashMap<ValueId, Variable> = HashMap::new();
    let mut next_var: u32 = 0;

    for block in function.layout.iter_block() {
        let clif_block = builder.create_block();
        block_map.insert(block, clif_block);
    }

    let (has_sret, has_struct_sret, scalar_sret_tys) = module.ctx.func_sig(func_ref, |sig| {
        (
            returns_indirectly(sig),
            returns_struct(sig),
            returns_many_scalars(sig).then(|| sig.ret_tys().to_vec()),
        )
    });

    let entry = function.layout.entry_block().ok_or("no entry block")?;
    let clif_entry = block_map[&entry];
    builder.append_block_params_for_function_params(clif_entry);
    builder.switch_to_block(clif_entry);

    let sret_ptr = if has_sret {
        Some(builder.block_params(clif_entry)[0])
    } else {
        None
    };

    let arg_offset = if has_sret { 1 } else { 0 };
    for (idx, &arg_value) in function.arg_values.iter().enumerate() {
        let param = builder.block_params(clif_entry)[idx + arg_offset];
        value_map.insert(arg_value, param);
    }

    let inst_set = function.inst_set();

    // No blanket ISA rejection — the translator handles each instruction
    // individually, emitting intrinsic calls for EVM-specific operations
    // (addmod, mulmod) and errors for truly unsupported ones.

    // Rung 3 STEP 2 (native leg): loop-membership analysis for
    // `MemAllocDynamic`, mirroring the SAME pre-scan the SPIR-V private-heap
    // translator already runs. Cranelift stack slots are sized once, at
    // IR-construction time (this loop), not re-allocated per invocation of
    // the code that references them -- so a `MemAllocDynamic` inside a
    // Sonatina loop would silently lower to "the SAME stack slot, reused
    // every iteration" rather than wasm's growing-arena semantics (a fresh
    // region per iteration). That is memory-safe on native (no aliasing
    // hazard the way an exhausted SPIR-V emulated heap had), but it is a
    // SILENT SEMANTIC DIVERGENCE from wasm/SPIR-V for a hypothetical future
    // kernel that relies on per-iteration freshness. Fail closed for
    // cross-backend consistency, the same reason the SPIR-V pre-scan does,
    // not because native's OWN memory model is unsafe here.
    let mut cfg = sonatina_ir::cfg::ControlFlowGraph::default();
    cfg.compute(function);
    let mut domtree = crate::domtree::DomTree::new();
    domtree.compute(&cfg);
    let mut loop_tree = crate::loop_analysis::LoopTree::new();
    loop_tree.compute(&cfg, &domtree);

    for block in function.layout.iter_block() {
        let clif_block = block_map[&block];
        if block != entry {
            builder.switch_to_block(clif_block);

            for inst_id in function.layout.iter_inst(block) {
                let inst_data = function.dfg.inst(inst_id);
                if let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                    let result = function.dfg.inst_result(inst_id)
                        .ok_or("phi has no result")?;
                    let ty = function.dfg.value_ty(result);
                    let clif_ty = sonatina_type_to_clif_or_err(ty)?;
                    let param = builder.append_block_param(clif_block, clif_ty);
                    value_map.insert(result, param);
                } else {
                    break;
                }
            }
        }

        for inst_id in function.layout.iter_inst(block) {
            let inst_data = function.dfg.inst(inst_id);

            if <&sonatina_ir::inst::control_flow::Phi as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                continue;
            }

            if let Some(add) = <&sonatina_ir::inst::arith::Add as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *add.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *add.rhs(), &value_map, &mut builder)?;
                let (lhs, rhs) = widen_to_match(&mut builder, lhs, rhs);
                let result_val = builder.ins().iadd(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(sub) = <&sonatina_ir::inst::arith::Sub as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *sub.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *sub.rhs(), &value_map, &mut builder)?;
                let (lhs, rhs) = widen_to_match(&mut builder, lhs, rhs);
                let result_val = builder.ins().isub(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(mul) = <&sonatina_ir::inst::arith::Mul as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *mul.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *mul.rhs(), &value_map, &mut builder)?;
                let (lhs, rhs) = widen_to_match(&mut builder, lhs, rhs);
                let result_val = builder.ins().imul(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(neg) = <&sonatina_ir::inst::arith::Neg as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *neg.arg(), &value_map, &mut builder)?;
                let result_val = builder.ins().ineg(val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            // Float arithmetic. `Type::F32` -> `clif::types::F32` (added above)
            // makes these native cranelift float instructions instead of the
            // previous "unsupported type for cranelift" translation error that
            // silently skipped any function using `Fadd`/`Fsqrt`/etc (the
            // `NativeInstSet` already declared them; nothing lowered them). This
            // is the sqrt-and-friends TEMPLATE that `Fabs`/`Fmin`/`Fmax`/`Fclamp`
            // below mirror.
            } else if let Some(fneg) = <&sonatina_ir::inst::arith::Fneg as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *fneg.arg(), &value_map, &mut builder)?;
                let result_val = builder.ins().fneg(val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fadd) = <&sonatina_ir::inst::arith::Fadd as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *fadd.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *fadd.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().fadd(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fsub) = <&sonatina_ir::inst::arith::Fsub as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *fsub.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *fsub.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().fsub(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fmul) = <&sonatina_ir::inst::arith::Fmul as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *fmul.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *fmul.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().fmul(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fdiv) = <&sonatina_ir::inst::arith::Fdiv as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *fdiv.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *fdiv.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().fdiv(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fsqrt) = <&sonatina_ir::inst::arith::Fsqrt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *fsqrt.arg(), &value_map, &mut builder)?;
                let result_val = builder.ins().sqrt(val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fabs) = <&sonatina_ir::inst::arith::Fabs as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *fabs.arg(), &value_map, &mut builder)?;
                // Pure bitwise sign-clear, per cranelift's own `fabs` doc comment.
                let result_val = builder.ins().fabs(val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fmin) = <&sonatina_ir::inst::arith::Fmin as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *fmin.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *fmin.rhs(), &value_map, &mut builder)?;
                // cranelift's `fmin` doc: "propagating NaNs using the WebAssembly
                // rules" -- exactly the semantics pinned for `Fmin` (matches
                // wasm's `f32.min` bit-for-bit on non-NaN inputs).
                let result_val = builder.ins().fmin(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fmax) = <&sonatina_ir::inst::arith::Fmax as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *fmax.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *fmax.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().fmax(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fmin) = <&sonatina_ir::inst::arith::FminRelaxed as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *fmin.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *fmin.rhs(), &value_map, &mut builder)?;
                // Relaxed contract: cranelift's native `fmin` already IS the
                // WebAssembly-rules-exact semantics, so it trivially
                // conforms to the weaker relaxed latitude too. Same
                // instruction as `Fmin` above -- zero new backend surface.
                let result_val = builder.ins().fmin(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fmax) = <&sonatina_ir::inst::arith::FmaxRelaxed as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *fmax.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *fmax.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().fmax(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(ffloor) = <&sonatina_ir::inst::arith::Ffloor as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *ffloor.arg(), &value_map, &mut builder)?;
                // Rounding family: single native instruction each, no
                // bit-twiddling, no NaN/-0 subtlety.
                let result_val = builder.ins().floor(val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fceil) = <&sonatina_ir::inst::arith::Fceil as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *fceil.arg(), &value_map, &mut builder)?;
                let result_val = builder.ins().ceil(val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(ftrunc) = <&sonatina_ir::inst::arith::Ftrunc as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *ftrunc.arg(), &value_map, &mut builder)?;
                let result_val = builder.ins().trunc(val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fround) = <&sonatina_ir::inst::arith::Fround as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *fround.arg(), &value_map, &mut builder)?;
                // cranelift's `nearest` doc: "Round floating point round to
                // integral, towards nearest with ties to even" -- exactly
                // `Fround`'s pinned `roundTiesToEven` semantics.
                let result_val = builder.ins().nearest(val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(fclamp) = <&sonatina_ir::inst::arith::Fclamp as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let arg = resolve_value(function, *fclamp.arg(), &value_map, &mut builder)?;
                let lo = resolve_value(function, *fclamp.lo(), &value_map, &mut builder)?;
                let hi = resolve_value(function, *fclamp.hi(), &value_map, &mut builder)?;
                // No native cranelift `fclamp`; compose as `min(max(arg, lo), hi)`,
                // matching the wasm backend's composition exactly.
                let maxed = builder.ins().fmax(arg, lo);
                let result_val = builder.ins().fmin(maxed, hi);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(div) = <&sonatina_ir::inst::arith::Udiv as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *div.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *div.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().udiv(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(div) = <&sonatina_ir::inst::arith::Sdiv as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *div.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *div.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().sdiv(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(shl) = <&sonatina_ir::inst::arith::Shl as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *shl.value(), &value_map, &mut builder)?;
                let bits = resolve_value(function, *shl.bits(), &value_map, &mut builder)?;
                let result_val = builder.ins().ishl(val, bits);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(shr) = <&sonatina_ir::inst::arith::Shr as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *shr.value(), &value_map, &mut builder)?;
                let bits = resolve_value(function, *shr.bits(), &value_map, &mut builder)?;
                let result_val = builder.ins().ushr(val, bits);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(sar) = <&sonatina_ir::inst::arith::Sar as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *sar.value(), &value_map, &mut builder)?;
                let bits = resolve_value(function, *sar.bits(), &value_map, &mut builder)?;
                let result_val = builder.ins().sshr(val, bits);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(and) = <&sonatina_ir::inst::logic::And as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *and.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *and.rhs(), &value_map, &mut builder)?;
                // See widen_to_match_bitwise's doc comment: the SAME
                // I32-Sonatina-typed-but-I64-cranelift-actual pointer issue
                // Add/Sub/Mul needed widening for also hits And -- confirmed
                // live via the pointer-round-up idiom
                // `(base + align-1) & ~(align-1)` (wasm_lower.rs's
                // lower_alloc_object), which lowers to Sonatina Add then
                // And. `band` requires exact operand-width match, same as
                // `iadd`. Deliberately SIGN-extend here, not
                // `widen_to_match`'s zero-extend: `~(align-1)` is a negative
                // i32 mask (e.g. `-8`), and zero-extending it truncates the
                // upper 32 bits of a real 64-bit stack address instead of
                // just the low align bits (caught via SIGSEGV in
                // `cranelift_mem_alloc_dynamic_pointer_round_up_idiom_
                // executes`, not by the verifier).
                let (lhs, rhs) = widen_to_match_bitwise(&mut builder, lhs, rhs);
                let result_val = builder.ins().band(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(or) = <&sonatina_ir::inst::logic::Or as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *or.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *or.rhs(), &value_map, &mut builder)?;
                // Same width-coercion as And above (sign-extend, see
                // widen_to_match_bitwise's doc comment); not observed to
                // fire in the 4 target kernels' address arithmetic today,
                // but Or is an equally plausible bitwise-address idiom for a
                // future kernel, and this is a no-op for every
                // already-matching case.
                let (lhs, rhs) = widen_to_match_bitwise(&mut builder, lhs, rhs);
                let result_val = builder.ins().bor(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(xor) = <&sonatina_ir::inst::logic::Xor as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *xor.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *xor.rhs(), &value_map, &mut builder)?;
                // Same width-coercion as And/Or above (sign-extend, see
                // widen_to_match_bitwise's doc comment).
                let (lhs, rhs) = widen_to_match_bitwise(&mut builder, lhs, rhs);
                let result_val = builder.ins().bxor(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(not) = <&sonatina_ir::inst::logic::Not as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *not.arg(), &value_map, &mut builder)?;
                let result_val = builder.ins().bnot(val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(lt) = <&sonatina_ir::inst::cmp::Lt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedLessThan, *lt.lhs(), *lt.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(gt) = <&sonatina_ir::inst::cmp::Gt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedGreaterThan, *gt.lhs(), *gt.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(le) = <&sonatina_ir::inst::cmp::Le as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedLessThanOrEqual, *le.lhs(), *le.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(ge) = <&sonatina_ir::inst::cmp::Ge as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedGreaterThanOrEqual, *ge.lhs(), *ge.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(slt) = <&sonatina_ir::inst::cmp::Slt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedLessThan, *slt.lhs(), *slt.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(sgt) = <&sonatina_ir::inst::cmp::Sgt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedGreaterThan, *sgt.lhs(), *sgt.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(eq) = <&sonatina_ir::inst::cmp::Eq as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs_ty = function.dfg.value_ty(*eq.lhs());
                if lhs_ty == Type::I256 {
                    let lhs = resolve_value(function, *eq.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *eq.rhs(), &value_map, &mut builder)?;
                    let mut sig = jit.make_signature();
                    sig.params.push(clif::AbiParam::new(clif::types::I64));
                    sig.params.push(clif::AbiParam::new(clif::types::I64));
                    sig.returns.push(clif::AbiParam::new(clif::types::I64));
                    let func_id = jit.declare_function("__u256_eq", Linkage::Import, &sig)
                        .map_err(|e| format!("failed to declare __u256_eq: {e}"))?;
                    let func_ref = jit.declare_func_in_func(func_id, builder.func);
                    let clif_call = builder.ins().call(func_ref, &[lhs, rhs]);
                    let raw_result = builder.inst_results(clif_call)[0];
                    let bool_result = builder.ins().ireduce(clif::types::I8, raw_result);
                    if let Some(result) = function.dfg.inst_result(inst_id) {
                        value_map.insert(result, bool_result);
                    }
                } else {
                    translate_icmp(IntCC::Equal, *eq.lhs(), *eq.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
                }
            } else if let Some(ne) = <&sonatina_ir::inst::cmp::Ne as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::NotEqual, *ne.lhs(), *ne.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(is_zero) = <&sonatina_ir::inst::cmp::IsZero as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *is_zero.lhs(), &value_map, &mut builder)?;
                let val_ty = function.dfg.value_ty(*is_zero.lhs());
                let clif_ty = sonatina_type_to_clif(val_ty).unwrap_or(clif::types::I64);
                let zero = builder.ins().iconst(clif_ty, 0);
                let result_val = builder.ins().icmp(IntCC::Equal, val, zero);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(sle) = <&sonatina_ir::inst::cmp::Sle as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedLessThanOrEqual, *sle.lhs(), *sle.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(sge) = <&sonatina_ir::inst::cmp::Sge as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedGreaterThanOrEqual, *sge.lhs(), *sge.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(feq) = <&sonatina_ir::inst::cmp::Feq as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_fcmp(FloatCC::Equal, *feq.lhs(), *feq.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(flt) = <&sonatina_ir::inst::cmp::Flt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_fcmp(FloatCC::LessThan, *flt.lhs(), *flt.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(fle) = <&sonatina_ir::inst::cmp::Fle as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_fcmp(FloatCC::LessThanOrEqual, *fle.lhs(), *fle.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(sext) = <&sonatina_ir::inst::cast::Sext as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *sext.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*sext.ty())?;
                let result_val = builder.ins().sextend(to_ty, val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(zext) = <&sonatina_ir::inst::cast::Zext as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *zext.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*zext.ty())?;
                let result_val = builder.ins().uextend(to_ty, val);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(trunc) = <&sonatina_ir::inst::cast::Trunc as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let from_ty = function.dfg.value_ty(*trunc.from());
                let val = resolve_value(function, *trunc.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*trunc.ty())?;
                let result_val = if from_ty == Type::I256 {
                    // i256 values are pointers — load the target-sized value from the pointer
                    builder.ins().load(to_ty, cranelift_codegen::ir::MemFlags::new(), val, 0)
                } else {
                    builder.ins().ireduce(to_ty, val)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(jump) = <&sonatina_ir::inst::control_flow::Jump as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let dest = block_map[jump.dest()];
                let phi_args = collect_phi_args_for_block(function, *jump.dest(), block, inst_set, &value_map, &mut builder)?;
                builder.ins().jump(dest, &phi_args);
            } else if let Some(br) = <&sonatina_ir::inst::control_flow::Br as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let cond = resolve_value(function, *br.cond(), &value_map, &mut builder)?;
                let nz_block = block_map[br.nz_dest()];
                let z_block = block_map[br.z_dest()];
                let nz_args = collect_phi_args_for_block(function, *br.nz_dest(), block, inst_set, &value_map, &mut builder)?;
                let z_args = collect_phi_args_for_block(function, *br.z_dest(), block, inst_set, &value_map, &mut builder)?;
                builder.ins().brif(cond, nz_block, &nz_args, z_block, &z_args);
            } else if let Some(ret) = <&sonatina_ir::inst::control_flow::Return as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if let Some(sret) = sret_ptr {
                    if has_struct_sret {
                        // Write the existing pointer-represented aggregate return.
                        for &val_id in ret.args().as_slice() {
                            let val = resolve_value(function, val_id, &value_map, &mut builder)?;
                            // Copy 32 bytes from val (pointer) to sret (pointer)
                            for i in 0..4 {
                                let limb = builder.ins().load(
                                    clif::types::I64,
                                    cranelift_codegen::ir::MemFlags::new(),
                                    val, (i * 8) as i32,
                                );
                                builder.ins().store(
                                    cranelift_codegen::ir::MemFlags::new(),
                                    limb, sret, (i * 8) as i32,
                                );
                            }
                        }
                    } else {
                        let return_tys = scalar_sret_tys
                            .as_deref()
                            .ok_or("missing native scalar indirect-return layout")?;
                        if ret.args().len() != return_tys.len() {
                            return Err(format!(
                                "native scalar indirect return has {} values for {} result types",
                                ret.args().len(),
                                return_tys.len()
                            ));
                        }
                        for (index, (&val_id, &ty)) in ret
                            .args()
                            .as_slice()
                            .iter()
                            .zip(return_tys)
                            .enumerate()
                        {
                            let val = resolve_value(function, val_id, &value_map, &mut builder)?;
                            sonatina_type_to_clif_or_err(ty)?;
                            builder.ins().store(
                                cranelift_codegen::ir::MemFlags::new(),
                                val,
                                sret,
                                (index * 16) as i32,
                            );
                        }
                    }
                    builder.ins().return_(&[]);
                } else {
                    let args: Vec<clif::Value> = ret.args().as_slice()
                        .iter()
                        .filter_map(|v| resolve_value(function, *v, &value_map, &mut builder).ok())
                        .collect();
                    builder.ins().return_(&args);
                }
            } else if <&sonatina_ir::inst::control_flow::CallIndirect as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                return Err("cranelift translation: call_indirect is not lowered yet".to_string());
            } else if let Some(call) = <&sonatina_ir::inst::control_flow::Call as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let callee = *call.callee();
                if let Some(&intrinsic_name) = runtime_declarations.get(&callee) {
                    let args: Result<Vec<_>, _> = call.args()
                        .iter()
                        .map(|v| resolve_value(function, *v, &value_map, &mut builder))
                        .collect();
                    // Args are already pointers to 32-byte u256 buffers
                    // (from obj.load passthrough or emit_i256_immediate stack slots)
                    let result_val = emit_u256_intrinsic_call(
                        jit, &mut builder, intrinsic_name, &args?, true,
                    )?;
                    let ir_results = function.dfg.inst_results(inst_id);
                    if !ir_results.is_empty() {
                        value_map.insert(ir_results[0], result_val);
                    }
                } else {
                    let clif_func_id = func_id_map.get(&callee)
                        .ok_or_else(|| format!("unknown callee {:?}", callee))?;
                    let clif_func_ref = jit.declare_func_in_func(*clif_func_id, builder.func);
                    let ir_results = function.dfg.inst_results(inst_id);
                    let (callee_returns_struct, callee_returns_many_scalars, callee_ret_tys) =
                        module.ctx.func_sig(callee, |sig| {
                            (
                                returns_struct(sig),
                                returns_many_scalars(sig),
                                sig.ret_tys().to_vec(),
                            )
                        });
                    let callee_returns_indirectly =
                        callee_returns_struct || callee_returns_many_scalars;

                    let mut call_args: Vec<clif::Value> = Vec::new();
                    let sret_slot = if callee_returns_indirectly {
                        // Allocate caller-owned space for aggregate or wide
                        // scalar-list return.
                        let bytes = if callee_returns_struct {
                            32
                        } else {
                            scalar_return_buffer_size(callee_ret_tys.len())
                        };
                        let slot = builder.create_sized_stack_slot(
                            cranelift_codegen::ir::StackSlotData::new(
                                cranelift_codegen::ir::StackSlotKind::ExplicitSlot, bytes, 0,
                            ),
                        );
                        let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
                        call_args.push(addr); // hidden sret param
                        Some(addr)
                    } else {
                        None
                    };

                    let args: Result<Vec<_>, _> = call.args()
                        .iter()
                        .map(|v| resolve_value(function, *v, &value_map, &mut builder))
                        .collect();
                    call_args.extend(args?);

                    let clif_call = builder.ins().call(clif_func_ref, &call_args);

                    if let Some(sret_addr) = sret_slot {
                        if callee_returns_struct {
                            // Aggregate result is represented by the address.
                            if !ir_results.is_empty() {
                                value_map.insert(ir_results[0], sret_addr);
                            }
                        } else {
                            if ir_results.len() != callee_ret_tys.len() {
                                return Err(format!(
                                    "native scalar indirect call has {} IR results for {} result types",
                                    ir_results.len(),
                                    callee_ret_tys.len()
                                ));
                            }
                            for (index, (&ir_result, &ty)) in ir_results
                                .iter()
                                .zip(&callee_ret_tys)
                                .enumerate()
                            {
                                let clif_ty = sonatina_type_to_clif_or_err(ty)?;
                                let value = builder.ins().load(
                                    clif_ty,
                                    cranelift_codegen::ir::MemFlags::new(),
                                    sret_addr,
                                    (index * 16) as i32,
                                );
                                value_map.insert(ir_result, value);
                            }
                        }
                    } else {
                        let results = builder.inst_results(clif_call).to_vec();
                        for (ir_result, clif_result) in ir_results.iter().zip(results.iter()) {
                            value_map.insert(*ir_result, *clif_result);
                        }
                    }
                }
            } else if let Some((op, signed, lhs_id, rhs_id)) = crate::isa::overflow::overflow_operands(inst_set, inst_data) {
                use crate::isa::overflow::OverflowArithmetic;
                let ty = function.dfg.value_ty(lhs_id);
                if !matches!(ty, Type::I1 | Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128) {
                    return Err(format!("unsupported native overflow arithmetic type {ty:?}"));
                }
                let lhs = resolve_value(function, lhs_id, &value_map, &mut builder)?;
                let rhs = resolve_value(function, rhs_id, &value_map, &mut builder)?;
                let (value, overflow) = if ty == Type::I1 {
                    // Cranelift's minimum integer carrier is i8. Compute the
                    // exact one-bit operation in that carrier, then apply the
                    // semantic [-1, 0] or [0, 1] range explicitly.
                    let (lhs, rhs) = if signed {
                        (builder.ins().ineg(lhs), builder.ins().ineg(rhs))
                    } else { (lhs, rhs) };
                    let exact = match op {
                        OverflowArithmetic::Add => builder.ins().iadd(lhs, rhs),
                        OverflowArithmetic::Sub => builder.ins().isub(lhs, rhs),
                        OverflowArithmetic::Mul => builder.ins().imul(lhs, rhs),
                    };
                    let below = builder.ins().icmp_imm(IntCC::SignedLessThan, exact, if signed { -1 } else { 0 });
                    let above = builder.ins().icmp_imm(IntCC::SignedGreaterThan, exact, if signed { 0 } else { 1 });
                    let overflow = builder.ins().bor(below, above);
                    (builder.ins().band_imm(exact, 1), overflow)
                } else if ty == Type::I128 && matches!(op, OverflowArithmetic::Mul) {
                    i128_overflow_mul(&mut builder, lhs, rhs, signed)
                } else {
                    match (op, signed) {
                        (OverflowArithmetic::Add, false) => builder.ins().uadd_overflow(lhs, rhs),
                        (OverflowArithmetic::Sub, false) => builder.ins().usub_overflow(lhs, rhs),
                        (OverflowArithmetic::Mul, false) => builder.ins().umul_overflow(lhs, rhs),
                        (OverflowArithmetic::Add, true) => builder.ins().sadd_overflow(lhs, rhs),
                        (OverflowArithmetic::Sub, true) => builder.ins().ssub_overflow(lhs, rhs),
                        (OverflowArithmetic::Mul, true) => builder.ins().smul_overflow(lhs, rhs),
                    }
                };
                for (result, value) in function.dfg.inst_results(inst_id).iter().zip([value, overflow]) {
                    value_map.insert(*result, value);
                }
            } else if let Some(obj_load) = <&sonatina_ir::inst::data::ObjLoad as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let addr = resolve_value(function, *obj_load.object(), &value_map, &mut builder)?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    if result_ty == Type::I256 || matches!(result_ty, Type::Compound(_)) {
                        // For i256/struct: passthrough the pointer
                        value_map.insert(result, addr);
                    } else {
                        let clif_ty = sonatina_type_to_clif_or_err(result_ty)?;
                        let loaded = builder.ins().load(clif_ty, cranelift_codegen::ir::MemFlags::new(), addr, 0);
                        value_map.insert(result, loaded);
                    }
                }
            } else if let Some(extract) = <&sonatina_ir::inst::data::ExtractValue as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let base = resolve_value(function, *extract.dest(), &value_map, &mut builder)?;
                let idx_val = function.dfg.value_imm(*extract.idx())
                    .map(|imm| match imm {
                        Immediate::I8(v) => v as i32,
                        Immediate::I32(v) => v,
                        Immediate::I64(v) => v as i32,
                        Immediate::I256(v) => v.to_u256().low_u64() as i32,
                        _ => 0,
                    })
                    .unwrap_or(0);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    let elem_size = module.ctx.size_of_unchecked(result_ty) as i32;
                    let offset = idx_val * elem_size;
                    if result_ty == Type::I256 || matches!(result_ty, Type::Compound(_)) {
                        let addr = builder.ins().iadd_imm(base, offset as i64);
                        value_map.insert(result, addr);
                    } else {
                        let clif_ty = sonatina_type_to_clif_or_err(result_ty)?;
                        let loaded = builder.ins().load(clif_ty, cranelift_codegen::ir::MemFlags::new(), base, offset);
                        value_map.insert(result, loaded);
                    }
                }
            } else if let Some(alloc) = <&sonatina_ir::inst::data::MemAllocDynamic as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                // Rung 3 STEP 2 (native leg): function-local `[u32; N]`
                // arrays lower to `MemAllocDynamic` + `Mload`/`Mstore`
                // (the SAME backend-neutral op set the SPIR-V private-heap
                // emulation consumes). `Mload`/`Mstore` already lower to
                // plain native loads/stores above/below (pre-existing,
                // used by the object model), and `Unreachable` already
                // lowers to a real `trap` instruction further down -- both
                // untouched by this rung. The ONLY missing piece was
                // `MemAllocDynamic` itself ("unsupported instruction for
                // CraneliftBackend: Opaque").
                //
                // Cranelift has REAL memory: unlike SPIR-V's emulated
                // shared heap + bump pointer (needed because a GPU storage
                // buffer has no general allocator), each MemAllocDynamic
                // site gets its OWN correctly-sized stack slot here --
                // simpler AND stronger than the SPIR-V scheme, since
                // distinct arrays can never alias each other by
                // construction (no shared heap to exhaust). This is the
                // exact same "one stack slot per instruction site" idiom
                // the pre-existing Alloca/ObjAlloc arms already use.
                //
                // Codex bug 1's analog (heap-exhaustion aliasing): closed
                // by construction (see above), but the size must still be
                // a compile-time constant (stack slots are sized at
                // IR-construction time) and the instruction must not sit
                // inside a loop (see the pre-scan comment above `cfg`).
                // Both fail closed with a named `Err`, which
                // `translate_module`'s existing skip-and-report convention
                // (and `native.rs::compile_and_verify_definitions`'s
                // missing-definition check, on the fe-codegen side) already
                // turns into a hard, non-silent failure -- never a wrong
                // compile.
                let Some(size_imm) = function.dfg.value_imm(*alloc.size()) else {
                    return Err(
                        "cranelift: MemAllocDynamic with a non-constant size is unsupported \
                         (stack slots are sized at compile time). Fail closed."
                            .to_string(),
                    );
                };
                let size_bytes: u32 = match size_imm {
                    Immediate::I1(v) => v as u32,
                    Immediate::I8(v) => v as u8 as u32,
                    Immediate::I32(v) => v as u32,
                    Immediate::I64(v) => u32::try_from(v).map_err(|_| {
                        "cranelift: MemAllocDynamic size does not fit a u32 stack-slot size. \
                         Fail closed."
                            .to_string()
                    })?,
                    _ => {
                        return Err(
                            "cranelift: MemAllocDynamic size has an unsupported immediate \
                             kind. Fail closed."
                                .to_string(),
                        );
                    }
                };
                if loop_tree.loop_of_block(block).is_some() {
                    return Err(
                        "cranelift: MemAllocDynamic inside a loop is unsupported (the same \
                         stack slot would be silently reused every iteration instead of \
                         wasm's fresh-per-iteration arena semantics). Fail closed."
                            .to_string(),
                    );
                }
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    // 8-byte-aligned, matching Fe's own array-base alignment
                    // convention (`layout_utils.rs`) and the SPIR-V private
                    // heap's word alignment; not load-bearing for native
                    // (real byte-addressed memory tolerates any alignment
                    // on x86_64/aarch64) but keeps the three backends'
                    // provenance assumptions identical.
                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, size_bytes, 3,
                        ),
                    );
                    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
                    // Explicit zero-init: cranelift stack slots are NOT
                    // zero-initialized by the OS/runtime (unlike wasm's
                    // linear memory, which starts zeroed). Matches the
                    // SPIR-V `fe_heap`'s load-bearing `ZeroValue` init for
                    // the same reason -- keep all three backends' "freshly
                    // allocated array reads as zero before any store"
                    // contract identical.
                    //
                    // `buffer_align` is passed as 1 (no claimed alignment),
                    // NOT the slot's real 8-byte base alignment: real Fe
                    // sizes are NOT always multiples of 8 (`lower_alloc_object`
                    // over-allocates by up to ALIGN-1 bytes so the RETURNED
                    // POINTER can be rounded up after the fact, e.g. a
                    // requested N*8-byte array can arrive here as N*8+7) --
                    // confirmed empirically: `emit_small_memset`'s own
                    // internal invariant (`greatest_divisible_power_of_two(size)
                    // >= buffer_align`) panicked ("size is smaller than
                    // dest's alignment value") on the real
                    // poseidon_merkle_root_loop.fe kernel with a hardcoded
                    // `buffer_align: 8`. `buffer_align` only hints an
                    // optimization (whether the emitted zero-fill stores get
                    // `.set_aligned()`); it is never a correctness lever --
                    // the slot's actual base address is still 8-aligned via
                    // `align_shift` above regardless of this value.
                    if size_bytes > 0 {
                        builder.emit_small_memset(
                            jit.target_config(),
                            addr,
                            0,
                            size_bytes as u64,
                            1,
                            cranelift_codegen::ir::MemFlags::new(),
                        );
                    }
                    value_map.insert(result, addr);
                }
            } else if let Some(alloca) = <&sonatina_ir::inst::data::Alloca as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 32, 0,
                        ),
                    );
                    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
                    value_map.insert(result, addr);
                }
            } else if let Some(mstore) = <&sonatina_ir::inst::data::Mstore as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let addr = resolve_value(function, *mstore.addr(), &value_map, &mut builder)?;
                let val = resolve_value(function, *mstore.value(), &value_map, &mut builder)?;
                let store_ty = function.dfg.value_ty(*mstore.value());
                if store_ty == Type::I256 {
                    for i in 0..4 {
                        let limb = builder.ins().load(clif::types::I64, cranelift_codegen::ir::MemFlags::new(), val, (i * 8) as i32);
                        builder.ins().store(cranelift_codegen::ir::MemFlags::new(), limb, addr, (i * 8) as i32);
                    }
                } else {
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), val, addr, 0);
                }
            } else if let Some(mload) = <&sonatina_ir::inst::data::Mload as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let addr = resolve_value(function, *mload.addr(), &value_map, &mut builder)?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    if result_ty == Type::I256 {
                        value_map.insert(result, addr);
                    } else {
                        let clif_ty = sonatina_type_to_clif_or_err(result_ty)?;
                        let loaded = builder.ins().load(clif_ty, cranelift_codegen::ir::MemFlags::new(), addr, 0);
                        value_map.insert(result, loaded);
                    }
                }
            } else if let Some(addmod) = <&sonatina_ir::inst::evm::EvmAddMod as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let a = resolve_value(function, *addmod.lhs(), &value_map, &mut builder)?;
                let b = resolve_value(function, *addmod.rhs(), &value_map, &mut builder)?;
                let m = resolve_value(function, *addmod.modulus(), &value_map, &mut builder)?;
                let result_val = emit_u256_intrinsic_call(
                    jit, &mut builder, "__u256_addmod",
                    &[a, b, m], true,
                )?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(mulmod) = <&sonatina_ir::inst::evm::EvmMulMod as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let a = resolve_value(function, *mulmod.lhs(), &value_map, &mut builder)?;
                let b = resolve_value(function, *mulmod.rhs(), &value_map, &mut builder)?;
                let m = resolve_value(function, *mulmod.modulus(), &value_map, &mut builder)?;
                let result_val = emit_u256_intrinsic_call(
                    jit, &mut builder, "__u256_mulmod",
                    &[a, b, m], true,
                )?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(obj_store) = <&sonatina_ir::inst::data::ObjStore as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let dest = resolve_value(function, *obj_store.object(), &value_map, &mut builder)?;
                let val = resolve_value(function, *obj_store.value(), &value_map, &mut builder)?;
                let val_ty = function.dfg.value_ty(*obj_store.value());
                if val_ty == Type::I256 {
                    // i256 store: copy 32 bytes from val (pointer) to dest (pointer)
                    for i in 0..4 {
                        let limb = builder.ins().load(clif::types::I64, cranelift_codegen::ir::MemFlags::new(), val, (i * 8) as i32);
                        builder.ins().store(cranelift_codegen::ir::MemFlags::new(), limb, dest, (i * 8) as i32);
                    }
                } else {
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), val, dest, 0);
                }
            } else if let Some(obj_alloc) = <&sonatina_ir::inst::data::ObjAlloc as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    let alloc_size = compute_alloc_size(result_ty, &module.ctx);
                    let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot, alloc_size, 0,
                    ));
                    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
                    value_map.insert(result, addr);
                }
            } else if let Some(obj_proj) = <&sonatina_ir::inst::data::ObjProj as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let vals = obj_proj.values();
                let base = resolve_value(function, vals[0], &value_map, &mut builder)?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, base);
                }
            } else if let Some(obj_index) = <&sonatina_ir::inst::data::ObjIndex as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let base = resolve_value(function, *obj_index.object(), &value_map, &mut builder)?;
                let index_val_id = *obj_index.index();
                let index_ty = function.dfg.value_ty(index_val_id);
                let index = if index_ty == Type::I256 {
                    if let Some(imm) = function.dfg.value_imm(index_val_id) {
                        let idx_i64 = match imm {
                            Immediate::I256(v) => {
                                let u = v.to_u256();
                                u.low_u64() as i64
                            }
                            _ => 0,
                        };
                        builder.ins().iconst(clif::types::I64, idx_i64)
                    } else {
                        let raw = resolve_value(function, index_val_id, &value_map, &mut builder)?;
                        builder.ins().load(clif::types::I64, cranelift_codegen::ir::MemFlags::new(), raw, 0)
                    }
                } else {
                    resolve_scalar_value(module, function, index_val_id, &value_map, &mut builder)?
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let obj_ty = function.dfg.value_ty(*obj_index.object());
                    let elem_size = crate::isa::compute_element_size(obj_ty, &module.ctx);
                    let stride = builder.ins().iconst(clif::types::I64, elem_size as i64);
                    let offset = builder.ins().imul(index, stride);
                    let addr = builder.ins().iadd(base, offset);
                    value_map.insert(result, addr);
                }
            } else if let Some(evm_umod) = <&sonatina_ir::inst::evm::EvmUmod as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *evm_umod.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *evm_umod.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().urem(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if <&sonatina_ir::inst::evm::EvmRevert as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                builder.ins().trap(cranelift_codegen::ir::TrapCode::user(2).unwrap());
            } else if <&sonatina_ir::inst::evm::EvmStop as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                builder.ins().return_(&[]);
            } else if let Some(const_ref) = <&sonatina_ir::inst::data::ConstRef as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let gv_ref = const_ref.global().gv();
                    let result_ty = function.dfg.value_ty(result);
                    let data_size = compute_alloc_size(result_ty, &module.ctx);
                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, data_size, 0,
                        ),
                    );
                    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
                    let init_data = module.ctx.with_gv_store(|store| store.init_data(gv_ref).cloned());
                    if let Some(init) = init_data {
                        let gv_ty = module.ctx.with_gv_store(|store| store.ty(gv_ref));
                        materialize_gv_initializer(&init, gv_ty, addr, 0, &module.ctx, &mut builder);
                    }
                    value_map.insert(result, addr);
                }
            } else if let Some(const_index) = <&sonatina_ir::inst::data::ConstIndex as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let base = resolve_value(function, *const_index.object(), &value_map, &mut builder)?;
                let index_val_id = *const_index.index();
                let index_ty = function.dfg.value_ty(index_val_id);
                let index = if index_ty == Type::I256 {
                    if let Some(imm) = function.dfg.value_imm(index_val_id) {
                        let idx = match imm {
                            Immediate::I256(v) => v.to_u256().low_u64() as i64,
                            _ => 0,
                        };
                        builder.ins().iconst(clif::types::I64, idx)
                    } else {
                        let raw = resolve_value(function, index_val_id, &value_map, &mut builder)?;
                        builder.ins().load(clif::types::I64, cranelift_codegen::ir::MemFlags::new(), raw, 0)
                    }
                } else {
                    resolve_scalar_value(module, function, index_val_id, &value_map, &mut builder)?
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let obj_ty = function.dfg.value_ty(*const_index.object());
                    let elem_size = crate::isa::compute_element_size(obj_ty, &module.ctx);
                    let stride = builder.ins().iconst(clif::types::I64, elem_size as i64);
                    let offset = builder.ins().imul(index, stride);
                    let ptr = builder.ins().iadd(base, offset);
                    value_map.insert(result, ptr);
                }
            } else if let Some(const_load) = <&sonatina_ir::inst::data::ConstLoad as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let addr = resolve_value(function, *const_load.object(), &value_map, &mut builder)?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    if result_ty == Type::I256 || matches!(result_ty, Type::Compound(_)) {
                        value_map.insert(result, addr);
                    } else {
                        let clif_ty = sonatina_type_to_clif_or_err(result_ty)?;
                        let loaded = builder.ins().load(clif_ty, cranelift_codegen::ir::MemFlags::new(), addr, 0);
                        value_map.insert(result, loaded);
                    }
                }
            } else if <&sonatina_ir::inst::control_flow::Unreachable as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                builder.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
            } else {
                return Err(format!(
                    "unsupported instruction for CraneliftBackend: {:?}",
                    inst_data.kind()
                ));
            }
        }
    }

    builder.seal_all_blocks();
    builder.finalize();

    if std::env::var("DUMP_CLIF").is_ok() {
        let name = module.ctx.func_sig(func_ref, |sig| sig.name().to_string());
        eprintln!("[cranelift] CLIF IR for {name}:\n{}", ctx.func.display());
    }

    if let Err(e) = jit.define_function(func_id, &mut ctx) {
        eprintln!("[cranelift] CLIF IR (error):\n{}", ctx.func.display());
        // `ModuleError`'s `Display` only prints the collapsed "Compilation
        // error: Verifier errors" wrapper; `{e:#?}` walks down to the
        // itemized `VerifierError` list (instruction + value IDs for each
        // individual violation), essential for diagnosing which specific
        // operand/type mismatch tripped the verifier rather than just
        // knowing that SOMETHING did.
        eprintln!("[cranelift] verifier detail: {e:#?}");
        return Err(format!("cranelift define_function failed: {e}"));
    }

    Ok(())
}

/// Zero-extend the narrower of `a`/`b` to match the wider one's cranelift
/// type, if they differ; a no-op (returns both unchanged) when they already
/// match, which is every pre-existing call site's shape.
///
/// Rung 3 STEP 2 (native leg): needed for address arithmetic on a
/// `MemAllocDynamic` result. Sonatina types that result `Type::I32`
/// (matching wasm's 32-bit linear-memory address space, the SAME
/// backend-neutral IR the SPIR-V translator also consumes unchanged), but
/// the cranelift lowering of `MemAllocDynamic` produces a REAL, native
/// pointer-width (`I64`) `stack_addr` value -- Sonatina's own type system
/// has no notion of "logical i32 address, physically wider on this target"
/// the way wasm's actual 32-bit address space does. Plain `Add`/`Sub`/`Mul`
/// on a stack address and an i32-typed offset would otherwise be a raw
/// cranelift width mismatch (a verifier error, not a silent miscompile:
/// cranelift's own binary opcodes require matching operand widths, so this
/// fails LOUD, never wrong, without this fix).
///
/// Zero-extension (not sign-extension) is correct HERE specifically:
/// Add/Sub/Mul's operands in this IR are byte offsets, array indices, or
/// allocation sizes -- non-negative quantities by construction.
///
/// Used by Add/Sub/Mul only (confirmed via
/// `cranelift_mem_alloc_dynamic_array_executes`). And/Or/Xor need a
/// DIFFERENT (sign-extending) policy for their bitwise-mask operands -- see
/// `widen_to_match_bitwise` below, and its doc comment for why zero-
/// extension is wrong there specifically (it looked like the same fix at
/// first, per `CRANELIFT_HUNCH.md`, but zero-extending a negative alignment
/// mask like `-8i32` truncates a real 64-bit stack pointer's upper bits
/// instead of preserving them -- a verifier-invisible bug caught only by
/// actually executing the JIT'd code).
fn widen_to_match(
    builder: &mut FunctionBuilder,
    a: clif::Value,
    b: clif::Value,
) -> (clif::Value, clif::Value) {
    let ta = builder.func.dfg.value_type(a);
    let tb = builder.func.dfg.value_type(b);
    if ta == tb {
        return (a, b);
    }
    if ta.bits() < tb.bits() {
        (builder.ins().uextend(tb, a), b)
    } else {
        (a, builder.ins().uextend(ta, b))
    }
}

/// Widen the narrower of `a`/`b` to the wider's width via SIGN-extension,
/// for bitwise (`And`/`Or`/`Xor`) operands specifically.
///
/// This is deliberately a *different* policy from `widen_to_match` above.
/// `widen_to_match` zero-extends because Add/Sub/Mul's operands in this IR
/// are byte offsets / sizes / indices -- non-negative quantities where
/// zero-extension is the width-correct choice.
///
/// Bitwise masks are not offsets: the pointer-round-up idiom
/// `(base + align-1) & ~(align-1)` (`wasm_lower.rs`'s `lower_alloc_object`)
/// materializes `~(align-1)` as a small NEGATIVE i32 immediate (e.g.
/// `-8i32` = `0xFFFF_FFF8`, wasm/Sonatina's 32-bit-address-space spelling
/// of "clear the low 3 bits"). Zero-extending that i32 pattern to i64 gives
/// `0x0000_0000_FFFF_FFF8` -- which, ANDed against a REAL 64-bit stack
/// pointer (always far above 2^32 on a real process, e.g. `0x00007ffX_...`
/// on Linux/x86_64), clears the ENTIRE upper 32 bits of the address instead
/// of just the low 3, producing a bogus, almost-certainly-unmapped address.
/// This was caught by the `cranelift_mem_alloc_dynamic_pointer_round_up_
/// idiom_executes` regression test, which SIGSEGV'd at runtime (the
/// verifier accepts either extension -- widths match either way -- so this
/// class of bug is invisible to the verifier and only shows up by actually
/// executing the JIT'd code, not merely compiling it).
///
/// Sign-extension is the width-correct choice for a bitwise mask: it
/// replicates the i32 pattern's top bit, so `0xFFFF_FFF8` (top bit 1)
/// widens to `0xFFFF_FFFF_FFFF_FFF8` (all upper bits also 1, correctly
/// clearing only the low 3 bits of a 64-bit value, same as the i32 pattern
/// clears only the low 3 bits of a 32-bit value). For a small POSITIVE
/// mask (top bit 0, e.g. an alignment-check constant like `7`),
/// sign-extension and zero-extension produce an IDENTICAL result, so this
/// is strictly a superset fix with no regression risk for that case.
fn widen_to_match_bitwise(
    builder: &mut FunctionBuilder,
    a: clif::Value,
    b: clif::Value,
) -> (clif::Value, clif::Value) {
    let ta = builder.func.dfg.value_type(a);
    let tb = builder.func.dfg.value_type(b);
    if ta == tb {
        return (a, b);
    }
    if ta.bits() < tb.bits() {
        (builder.ins().sextend(tb, a), b)
    } else {
        (a, builder.ins().sextend(ta, b))
    }
}

fn resolve_scalar_value(
    module: &Module,
    function: &Function,
    value_id: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let ty = function.dfg.value_ty(value_id);
    let val = resolve_value(function, value_id, value_map, builder)?;
    if ty.is_obj_ref(&module.ctx) {
        if let Some(inner) = ty.resolve_compound(&module.ctx) {
            if let sonatina_ir::types::CompoundType::ObjRef(elem) = inner {
                if let Some(clif_ty) = sonatina_type_to_clif(elem) {
                    return Ok(builder.ins().load(clif_ty, cranelift_codegen::ir::MemFlags::new(), val, 0));
                }
            }
        }
    }
    Ok(val)
}

fn resolve_value(
    function: &Function,
    value_id: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    if let Some(&clif_val) = value_map.get(&value_id) {
        return Ok(clif_val);
    }
    // Check if there's a Variable for this (phi values in loops)
    // Variables are looked up via the FunctionBuilder's SSA system

    let value = function.dfg.value(value_id);
    match value {
        Value::Immediate { imm, ty } => {
            if let Immediate::F32(bits) = imm {
                let val = builder
                    .ins()
                    .f32const(cranelift_codegen::ir::immediates::Ieee32::with_bits(*bits));
                Ok(val)
            } else if let Immediate::I128(bits) = imm {
                // Preserve both words rather than sign-extending a single
                // immediate through Cranelift's i64 constant operand.
                let lo = builder.ins().iconst(clif::types::I64, *bits as i64);
                let hi = builder.ins().iconst(clif::types::I64, (*bits >> 64) as i64);
                Ok(builder.ins().iconcat(lo, hi))
            } else if let Immediate::I256(value) = imm {
                Ok(emit_i256_immediate(value, builder))
            } else {
                let clif_ty = sonatina_type_to_clif_or_err(*ty)?;
                let i64_val = imm_to_i64(imm)?;
                let val = builder.ins().iconst(clif_ty, i64_val);
                Ok(val)
            }
        }
        _ => Err(format!("unresolved value v{}", value_id.0)),
    }
}

/// Cranelift's overflow multiply stops at i64. Compute the full product
/// using four 64-bit limb products, preserving the i128 semantic width.
fn i128_overflow_mul(
    builder: &mut FunctionBuilder,
    lhs: clif::Value,
    rhs: clif::Value,
    signed: bool,
) -> (clif::Value, clif::Value) {
    let negative = if signed {
        let signs = builder.ins().bxor(lhs, rhs);
        Some(builder.ins().icmp_imm(IntCC::SignedLessThan, signs, 0))
    } else { None };
    let magnitude = |builder: &mut FunctionBuilder, value| {
        if signed {
            let is_negative = builder.ins().icmp_imm(IntCC::SignedLessThan, value, 0);
            let negated = builder.ins().ineg(value);
            builder.ins().select(is_negative, negated, value)
        } else { value }
    };
    let lhs = magnitude(builder, lhs);
    let rhs = magnitude(builder, rhs);
    let (a0, a1) = builder.ins().isplit(lhs);
    let (b0, b1) = builder.ins().isplit(rhs);
    let low = builder.ins().imul(a0, b0);
    let carry = builder.ins().umulhi(a0, b0);
    let cross0 = builder.ins().imul(a0, b1);
    let cross0_hi = builder.ins().umulhi(a0, b1);
    let cross1 = builder.ins().imul(a1, b0);
    let cross1_hi = builder.ins().umulhi(a1, b0);
    let (high, carry0) = builder.ins().uadd_overflow(carry, cross0);
    let (high, carry1) = builder.ins().uadd_overflow(high, cross1);
    let a1_nonzero = builder.ins().icmp_imm(IntCC::NotEqual, a1, 0);
    let b1_nonzero = builder.ins().icmp_imm(IntCC::NotEqual, b1, 0);
    let high_product = builder.ins().band(a1_nonzero, b1_nonzero);
    let cross_high = builder.ins().bor(cross0_hi, cross1_hi);
    let cross_overflow = builder.ins().icmp_imm(IntCC::NotEqual, cross_high, 0);
    let carries = builder.ins().bor(carry0, carry1);
    let overflow = builder.ins().bor(high_product, cross_overflow);
    let overflow = builder.ins().bor(overflow, carries);
    let product = builder.ins().iconcat(low, high);
    if let Some(negative) = negative {
        let max_low = builder.ins().iconst(clif::types::I64, -1);
        let max_high = builder.ins().iconst(clif::types::I64, i64::MAX);
        let max_positive = builder.ins().iconcat(max_low, max_high);
        let min_magnitude = builder.ins().iadd_imm(max_positive, 1);
        let limit = builder.ins().select(negative, min_magnitude, max_positive);
        let beyond_limit = builder.ins().icmp(IntCC::UnsignedGreaterThan, product, limit);
        let overflow = builder.ins().bor(overflow, beyond_limit);
        let negated = builder.ins().ineg(product);
        (builder.ins().select(negative, negated, product), overflow)
    } else {
        (product, overflow)
    }
}

fn imm_to_i64(imm: &Immediate) -> Result<i64, String> {
    match imm {
        Immediate::I1(b) => Ok(*b as i64),
        Immediate::I8(v) => Ok(*v as i64),
        Immediate::I16(v) => Ok(*v as i64),
        Immediate::I32(v) => Ok(*v as i64),
        Immediate::I64(v) => Ok(*v),
        _ => Err(format!("unsupported immediate type for cranelift: {imm:?}")),
    }
}

fn emit_i256_immediate(
    imm: &sonatina_ir::I256,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let slot = builder.create_sized_stack_slot(
        cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 32, 0,
        ),
    );
    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);

    let u256 = imm.to_u256();
    let bytes = u256.to_little_endian();
    for i in 0..4 {
        let limb = u64::from_le_bytes(bytes[i*8..(i+1)*8].try_into().unwrap());
        let val = builder.ins().iconst(clif::types::I64, limb as i64);
        builder.ins().store(
            cranelift_codegen::ir::MemFlags::new(),
            val, addr, (i * 8) as i32,
        );
    }

    addr
}

fn translate_icmp(
    cc: IntCC,
    lhs: ValueId,
    rhs: ValueId,
    inst_id: sonatina_ir::inst::InstId,
    module: &Module,
    function: &Function,
    value_map: &mut HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(), String> {
    let lhs_val = resolve_scalar_value(module, function, lhs, value_map, builder)?;
    let rhs_val = resolve_scalar_value(module, function, rhs, value_map, builder)?;
    let signed = matches!(
        cc,
        IntCC::SignedLessThan
            | IntCC::SignedGreaterThan
            | IntCC::SignedLessThanOrEqual
            | IntCC::SignedGreaterThanOrEqual
    );
    let (lhs_val, rhs_val) = widen_integer_comparison_operands(builder, lhs_val, rhs_val, signed);
    let result_val = builder.ins().icmp(cc, lhs_val, rhs_val);
    if let Some(result) = function.dfg.inst_result(inst_id) {
        value_map.insert(result, result_val);
    }
    Ok(())
}

fn widen_integer_comparison_operands(
    builder: &mut FunctionBuilder,
    lhs: clif::Value,
    rhs: clif::Value,
    signed: bool,
) -> (clif::Value, clif::Value) {
    let lhs_ty = builder.func.dfg.value_type(lhs);
    let rhs_ty = builder.func.dfg.value_type(rhs);
    if lhs_ty == rhs_ty {
        return (lhs, rhs);
    }
    if lhs_ty.bits() < rhs_ty.bits() {
        let lhs = if signed {
            builder.ins().sextend(rhs_ty, lhs)
        } else {
            builder.ins().uextend(rhs_ty, lhs)
        };
        (lhs, rhs)
    } else {
        let rhs = if signed {
            builder.ins().sextend(lhs_ty, rhs)
        } else {
            builder.ins().uextend(lhs_ty, rhs)
        };
        (lhs, rhs)
    }
}

fn translate_fcmp(
    cc: FloatCC,
    lhs: ValueId,
    rhs: ValueId,
    inst_id: sonatina_ir::inst::InstId,
    module: &Module,
    function: &Function,
    value_map: &mut HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(), String> {
    let lhs_val = resolve_scalar_value(module, function, lhs, value_map, builder)?;
    let rhs_val = resolve_scalar_value(module, function, rhs, value_map, builder)?;
    let result_val = builder.ins().fcmp(cc, lhs_val, rhs_val);
    if let Some(result) = function.dfg.inst_result(inst_id) {
        value_map.insert(result, result_val);
    }
    Ok(())
}

/// If `val` is a raw i64 (loaded from obj.load of i256), write it to a
/// 32-byte stack slot and return the slot's address. If `val` is already
/// a stack address (from emit_i256_immediate), return it as-is.
///
/// This ensures u256 intrinsics always receive valid pointers to 32-byte buffers.
fn ensure_u256_on_stack(val: clif::Value, builder: &mut FunctionBuilder) -> clif::Value {
    // Always write to a fresh stack slot — safe for any value
    let slot = builder.create_sized_stack_slot(
        cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 32, 0,
        ),
    );
    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
    // Store the i64 value at offset 0, zero the rest
    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), val, addr, 0);
    let zero = builder.ins().iconst(clif::types::I64, 0);
    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, addr, 8);
    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, addr, 16);
    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, addr, 24);
    addr
}

fn emit_u256_intrinsic_call(
    jit: &mut JITModule,
    builder: &mut FunctionBuilder,
    name: &str,
    args: &[clif::Value],
    has_result: bool,
) -> Result<clif::Value, String> {
    let ptr_ty = clif::types::I64;

    // Build the intrinsic signature: all args are pointers, optional result pointer
    let mut sig = jit.make_signature();
    for _ in args {
        sig.params.push(clif::AbiParam::new(ptr_ty));
    }
    if has_result {
        sig.params.push(clif::AbiParam::new(ptr_ty)); // result pointer
    }

    let func_id = jit
        .declare_function(name, Linkage::Import, &sig)
        .map_err(|e| format!("failed to declare {name}: {e}"))?;
    let func_ref = jit.declare_func_in_func(func_id, builder.func);

    if has_result {
        // Allocate 32-byte stack slot for the result
        let result_slot = builder.create_sized_stack_slot(
            cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                32, 0,
            ),
        );
        let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);

        let mut call_args: Vec<clif::Value> = args.to_vec();
        call_args.push(result_addr);
        builder.ins().call(func_ref, &call_args);

        Ok(result_addr)
    } else {
        builder.ins().call(func_ref, args);
        Ok(builder.ins().iconst(ptr_ty, 0))
    }
}

fn materialize_gv_initializer(
    init: &sonatina_ir::global_variable::GvInitializer,
    ty: Type,
    base: clif::Value,
    offset: i32,
    ctx: &sonatina_ir::module::ModuleCtx,
    builder: &mut FunctionBuilder,
) {
    use sonatina_ir::global_variable::GvInitializer;
    match init {
        GvInitializer::Immediate(imm) => {
            match imm {
                Immediate::I8(v) => {
                    let val = builder.ins().iconst(clif::types::I8, *v as i64);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), val, base, offset);
                }
                Immediate::I16(v) => {
                    let val = builder.ins().iconst(clif::types::I16, *v as i64);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), val, base, offset);
                }
                Immediate::I32(v) => {
                    let val = builder.ins().iconst(clif::types::I32, *v as i64);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), val, base, offset);
                }
                Immediate::I64(v) => {
                    let val = builder.ins().iconst(clif::types::I64, *v);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), val, base, offset);
                }
                Immediate::I256(v) => {
                    let u = v.to_u256();
                    let bytes = u.to_little_endian();
                    for i in 0..4 {
                        let limb = u64::from_le_bytes(bytes[i*8..(i+1)*8].try_into().unwrap());
                        let val = builder.ins().iconst(clif::types::I64, limb as i64);
                        builder.ins().store(cranelift_codegen::ir::MemFlags::new(), val, base, offset + (i * 8) as i32);
                    }
                }
                _ => {}
            }
        }
        GvInitializer::Array(elems) => {
            if let Some(cmpd) = ty.resolve_compound(ctx) {
                if let sonatina_ir::types::CompoundType::Array { elem, .. }
                    | sonatina_ir::types::CompoundType::ConstRef(elem) = cmpd
                {
                    let elem_size = ctx.size_of_unchecked(elem) as i32;
                    for (i, elem_init) in elems.iter().enumerate() {
                        materialize_gv_initializer(elem_init, elem, base, offset + i as i32 * elem_size, ctx, builder);
                    }
                }
            }
        }
        GvInitializer::Struct(fields) => {
            if let Some(cmpd) = ty.resolve_compound(ctx) {
                if let sonatina_ir::types::CompoundType::Struct(s) = cmpd {
                    let mut field_offset = offset;
                    for (i, (field_init, &field_ty)) in fields.iter().zip(s.fields.iter()).enumerate() {
                        materialize_gv_initializer(field_init, field_ty, base, field_offset, ctx, builder);
                        field_offset += ctx.size_of_unchecked(field_ty) as i32;
                    }
                }
            }
        }
    }
}

fn compute_alloc_size(ty: Type, ctx: &sonatina_ir::module::ModuleCtx) -> u32 {
    if let Type::Compound(_) = ty {
        if let Some(cmpd) = ty.resolve_compound(ctx) {
            match cmpd {
                sonatina_ir::types::CompoundType::Array { elem, len } => {
                    let elem_size = ctx.size_of_unchecked(elem);
                    return (elem_size * len).max(8) as u32;
                }
                sonatina_ir::types::CompoundType::ObjRef(inner)
                | sonatina_ir::types::CompoundType::ConstRef(inner) => {
                    return compute_alloc_size(inner, ctx);
                }
                sonatina_ir::types::CompoundType::Struct(s) => {
                    let total: usize = s.fields.iter().map(|f| ctx.size_of_unchecked(*f)).sum();
                    return total.max(8) as u32;
                }
                _ => {}
            }
        }
    }
    let size = ctx.size_of_unchecked(ty);
    size.max(8) as u32
}

fn collect_phi_args_for_block(
    function: &Function,
    target_block: BlockId,
    source_block: BlockId,
    inst_set: &dyn sonatina_ir::InstSetBase,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<Vec<BlockArg>, String> {
    let mut args = Vec::new();
    for inst_id in function.layout.iter_inst(target_block) {
        let inst_data = function.dfg.inst(inst_id);
        if let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
            for &(value, from_block) in phi.args() {
                if from_block == source_block {
                    let clif_val = resolve_value(function, value, value_map, builder)?;
                    args.push(BlockArg::Value(clif_val));
                    break;
                }
            }
        } else {
            break;
        }
    }
    Ok(args)
}

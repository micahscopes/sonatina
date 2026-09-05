//! Sonatina IR → WAFFLE IR translation.
//!
//! Translates Sonatina's SSA IR (phi nodes, arbitrary CFG) to WAFFLE's
//! SSA IR (block params, structured terminators). WAFFLE then handles
//! control flow recovery (Ramsey's algorithm) and WASM emission.

use std::collections::HashMap;

use waffle::{
    BlockTarget, ExportKind, Func, FuncDecl, FunctionBody, GlobalData, Module as WaffleModule,
    Operator, SignatureData, TableData, Terminator, Type as WType, ValueDef,
    entity::EntityRef,
};

use sonatina_ir::{
    BlockId, Function, GlobalVariableRef, Immediate, Inst, InstDowncast, InstSetBase, Linkage,
    Module, Signature, Type, Value, ValueId,
    cfg::ControlFlowGraph,
    global_variable::GvInitializer,
    module::{FuncRef, ModuleCtx},
    types::CompoundType,
};
use super::CanonicalStackMemoryManifest;

use crate::domtree::DomTree;
use crate::isa::overflow::{OverflowArithmetic, overflow_operands};

fn sonatina_to_waffle_type(ty: Type) -> Option<WType> {
    match ty {
        Type::Unit => None,
        Type::I1 | Type::I8 | Type::I16 | Type::I32 => Some(WType::I32),
        Type::F32 => Some(WType::F32),
        Type::I64 => Some(WType::I64),
        // objref<T> / constref<T> — use the inner type's WASM representation
        Type::Compound(_) => Some(WType::I64),
        _ => None,
    }
}

/// Preserve both results of unsigned overflow arithmetic. Narrow integers
/// use their semantic width, not merely their i32 Wasm carrier.
fn unsigned_overflow_arithmetic(
    body: &mut FunctionBody,
    block: waffle::Block,
    ty: Type,
    lhs: waffle::Value,
    rhs: waffle::Value,
    arithmetic: OverflowArithmetic,
) -> Result<(waffle::Value, waffle::Value), String> {
    let bits = match ty {
        Type::I1 => 1,
        Type::I8 => 8,
        Type::I16 => 16,
        Type::I32 => 32,
        Type::I64 => 64,
        other => return Err(format!("unsupported Wasm unsigned overflow arithmetic type {other:?}")),
    };
    let wide = bits == 64;
    let carrier = if wide { WType::I64 } else { WType::I32 };
    let op = match (wide, arithmetic) {
        (false, OverflowArithmetic::Add) => Operator::I32Add,
        (false, OverflowArithmetic::Sub) => Operator::I32Sub,
        (false, OverflowArithmetic::Mul) => Operator::I32Mul,
        (true, OverflowArithmetic::Add) => Operator::I64Add,
        (true, OverflowArithmetic::Sub) => Operator::I64Sub,
        (true, OverflowArithmetic::Mul) => Operator::I64Mul,
    };
    let raw = body.add_op(block, op, &[lhs, rhs], &[carrier]);
    if matches!(arithmetic, OverflowArithmetic::Sub) {
        let compare = if wide { Operator::I64LtU } else { Operator::I32LtU };
        let overflow = body.add_op(block, compare, &[lhs, rhs], &[WType::I32]);
        let result = if bits < 32 {
            let limit = body.add_op(block, Operator::I32Const { value: (1 << bits) - 1 }, &[], &[WType::I32]);
            body.add_op(block, Operator::I32And, &[raw, limit], &[WType::I32])
        } else { raw };
        return Ok((result, overflow));
    }
    if bits < 32 {
        // Inputs are typed unsigned values. Their full sum/product fits i32
        // for every admitted narrow width, including the i16 product.
        let limit = body.add_op(block, Operator::I32Const { value: (1 << bits) - 1 }, &[], &[WType::I32]);
        let result = body.add_op(block, Operator::I32And, &[raw, limit], &[WType::I32]);
        let overflow = body.add_op(block, Operator::I32LtU, &[limit, raw], &[WType::I32]);
        return Ok((result, overflow));
    }
    if matches!(arithmetic, OverflowArithmetic::Add) {
        let compare = if wide { Operator::I64LtU } else { Operator::I32LtU };
        let overflow = body.add_op(block, compare, &[raw, lhs], &[WType::I32]);
        return Ok((raw, overflow));
    }
    // Product overflow iff lhs != 0 and wrapped_product / lhs != rhs.
    // Substitute one for a zero divisor without introducing control flow or
    // a trap in an otherwise total operation.
    let eqz = if wide { Operator::I64Eqz } else { Operator::I32Eqz };
    let lhs_zero = body.add_op(block, eqz, &[lhs], &[WType::I32]);
    let zero_word = if wide {
        body.add_op(block, Operator::I64ExtendI32U, &[lhs_zero], &[WType::I64])
    } else {
        lhs_zero
    };
    let or = if wide { Operator::I64Or } else { Operator::I32Or };
    let divisor = body.add_op(block, or, &[lhs, zero_word], &[carrier]);
    let div = if wide { Operator::I64DivU } else { Operator::I32DivU };
    let quotient = body.add_op(block, div, &[raw, divisor], &[carrier]);
    let ne = if wide { Operator::I64Ne } else { Operator::I32Ne };
    let mismatch = body.add_op(block, ne, &[quotient, rhs], &[WType::I32]);
    let lhs_nonzero = body.add_op(block, Operator::I32Eqz, &[lhs_zero], &[WType::I32]);
    let overflow = body.add_op(block, Operator::I32And, &[mismatch, lhs_nonzero], &[WType::I32]);
    Ok((raw, overflow))
}

fn signed_overflow_arithmetic(
    body: &mut FunctionBody,
    block: waffle::Block,
    ty: Type,
    lhs: waffle::Value,
    rhs: waffle::Value,
    arithmetic: OverflowArithmetic,
) -> Result<(waffle::Value, waffle::Value), String> {
    let bits = match ty {
        Type::I1 => 1,
        Type::I8 => 8,
        Type::I16 => 16,
        Type::I32 => 32,
        Type::I64 => 64,
        other => return Err(format!("unsupported Wasm signed overflow arithmetic type {other:?}")),
    };
    let op = match arithmetic {
        OverflowArithmetic::Add => Operator::I64Add,
        OverflowArithmetic::Sub => Operator::I64Sub,
        OverflowArithmetic::Mul => Operator::I64Mul,
    };
    if bits < 64 {
        // All signed i32 products fit i64. Sign-extend from the semantic
        // width before widening, then check bounds before narrowing.
        let widen = |body: &mut FunctionBody, value| {
            let value = if bits < 32 {
                let shift = body.add_op(block, Operator::I32Const { value: 32 - bits }, &[], &[WType::I32]);
                let shifted = body.add_op(block, Operator::I32Shl, &[value, shift], &[WType::I32]);
                body.add_op(block, Operator::I32ShrS, &[shifted, shift], &[WType::I32])
            } else { value };
            body.add_op(block, Operator::I64ExtendI32S, &[value], &[WType::I64])
        };
        let lhs = widen(body, lhs);
        let rhs = widen(body, rhs);
        let exact = body.add_op(block, op, &[lhs, rhs], &[WType::I64]);
        let min = body.add_op(block, Operator::I64Const { value: (-(1i64 << (bits - 1))) as u64 }, &[], &[WType::I64]);
        let max = body.add_op(block, Operator::I64Const { value: (1u64 << (bits - 1)) - 1 }, &[], &[WType::I64]);
        let below = body.add_op(block, Operator::I64LtS, &[exact, min], &[WType::I32]);
        let above = body.add_op(block, Operator::I64LtS, &[max, exact], &[WType::I32]);
        let overflow = body.add_op(block, Operator::I32Or, &[below, above], &[WType::I32]);
        let raw = body.add_op(block, Operator::I32WrapI64, &[exact], &[WType::I32]);
        let result = if bits < 32 {
            let mask = body.add_op(block, Operator::I32Const { value: (1 << bits) - 1 }, &[], &[WType::I32]);
            body.add_op(block, Operator::I32And, &[raw, mask], &[WType::I32])
        } else { raw };
        return Ok((result, overflow));
    }
    let raw = body.add_op(block, op, &[lhs, rhs], &[WType::I64]);
    let zero = body.add_op(block, Operator::I64Const { value: 0 }, &[], &[WType::I64]);
    let lhs_changed = body.add_op(block, Operator::I64Xor, &[lhs, raw], &[WType::I64]);
    if !matches!(arithmetic, OverflowArithmetic::Mul) {
        let other = match arithmetic {
            OverflowArithmetic::Add => body.add_op(block, Operator::I64Xor, &[rhs, raw], &[WType::I64]),
            OverflowArithmetic::Sub => body.add_op(block, Operator::I64Xor, &[lhs, rhs], &[WType::I64]),
            OverflowArithmetic::Mul => unreachable!(),
        };
        let sign_change = body.add_op(block, Operator::I64And, &[lhs_changed, other], &[WType::I64]);
        let overflow = body.add_op(block, Operator::I64LtS, &[sign_change, zero], &[WType::I32]);
        return Ok((raw, overflow));
    }
    // Compare unsigned magnitudes before multiplying. abs(MIN) remains
    // representable as an unsigned word, and no signed MIN / -1 occurs.
    let shift = body.add_op(block, Operator::I64Const { value: 63 }, &[], &[WType::I64]);
    let magnitude = |body: &mut FunctionBody, value| {
        let sign = body.add_op(block, Operator::I64ShrS, &[value, shift], &[WType::I64]);
        let flipped = body.add_op(block, Operator::I64Xor, &[value, sign], &[WType::I64]);
        body.add_op(block, Operator::I64Sub, &[flipped, sign], &[WType::I64])
    };
    let lhs_abs = magnitude(body, lhs);
    let rhs_abs = magnitude(body, rhs);
    let signs = body.add_op(block, Operator::I64Xor, &[lhs, rhs], &[WType::I64]);
    let negative = body.add_op(block, Operator::I64LtS, &[signs, zero], &[WType::I32]);
    let negative_word = body.add_op(block, Operator::I64ExtendI32U, &[negative], &[WType::I64]);
    let max = body.add_op(block, Operator::I64Const { value: i64::MAX as u64 }, &[], &[WType::I64]);
    let limit = body.add_op(block, Operator::I64Add, &[max, negative_word], &[WType::I64]);
    let lhs_zero = body.add_op(block, Operator::I64Eqz, &[lhs_abs], &[WType::I32]);
    let zero_word = body.add_op(block, Operator::I64ExtendI32U, &[lhs_zero], &[WType::I64]);
    let divisor = body.add_op(block, Operator::I64Or, &[lhs_abs, zero_word], &[WType::I64]);
    let allowed = body.add_op(block, Operator::I64DivU, &[limit, divisor], &[WType::I64]);
    let exceeds = body.add_op(block, Operator::I64LtU, &[allowed, rhs_abs], &[WType::I32]);
    let lhs_nonzero = body.add_op(block, Operator::I32Eqz, &[lhs_zero], &[WType::I32]);
    let overflow = body.add_op(block, Operator::I32And, &[exceeds, lhs_nonzero], &[WType::I32]);
    Ok((raw, overflow))
}

fn sonatina_to_waffle_type_in_ctx(ctx: &ModuleCtx, ty: Type) -> Option<WType> {
    if matches!(
        ty.resolve_compound(ctx),
        Some(CompoundType::Ptr(pointee))
            if matches!(pointee.resolve_compound(ctx), Some(CompoundType::Func { .. }))
    ) {
        Some(WType::I32)
    } else {
        sonatina_to_waffle_type(ty)
    }
}

fn indirect_signature(ctx: &ModuleCtx, ty: Type) -> Result<SignatureData, String> {
    let Some(CompoundType::Ptr(pointee)) = ty.resolve_compound(ctx) else {
        return Err(format!("wasm call_indirect signature `{ty:?}` is not a pointer"));
    };
    let Some(CompoundType::Func { args, ret_tys }) = pointee.resolve_compound(ctx) else {
        return Err(format!(
            "wasm call_indirect signature `{ty:?}` does not point to a function"
        ));
    };
    let map = |types: &[Type], position: &str| -> Result<Vec<WType>, String> {
        types
            .iter()
            .map(|ty| {
                sonatina_to_waffle_type_in_ctx(ctx, *ty).ok_or_else(|| {
                    format!(
                        "wasm call_indirect {position} type `{ty:?}` is not representable in wasm"
                    )
                })
            })
            .collect()
    };
    Ok(SignatureData {
        params: map(&args, "parameter")?,
        returns: map(&ret_tys, "result")?,
    })
}

fn scalar_memory_arg(memory: waffle::Memory, ty: Type) -> Result<waffle::MemoryArg, String> {
    // WebAssembly encodes alignment as log2(bytes). These are natural
    // alignments, but unaligned runtime addresses remain legal in Wasm.
    let align = match ty {
        Type::I1 | Type::I8 => 0,
        Type::I16 => 1,
        Type::I32 | Type::F32 => 2,
        Type::I64 => 3,
        _ => return Err(format!("unsupported wasm scalar memory type `{ty:?}`")),
    };
    Ok(waffle::MemoryArg {
        align,
        offset: 0,
        memory,
    })
}

#[derive(Clone, Copy)]
struct CanonicalArenaFunctions {
    alloc: Func,
    checkpoint: Func,
    rewind: Func,
}

pub(super) fn translate_module(
    module: &Module,
    import_modules: &HashMap<String, String>,
    canonical_arena: bool,
    canonical_memory: Option<&CanonicalStackMemoryManifest>,
) -> Result<(WaffleModule<'static>, Vec<String>), String> {
    translate_module_inner(
        module,
        import_modules,
        canonical_arena,
        canonical_memory,
        false,
    )
}

/// Translate an owned Sonatina module while releasing each source function
/// after its WAFFLE body has been derived. Function declarations remain in the
/// module context so later callers retain the same signatures and indexes, but
/// the full source and target body graphs do not have to coexist.
pub(super) fn translate_owned_module(
    module: Module,
    import_modules: &HashMap<String, String>,
    canonical_arena: bool,
    canonical_memory: Option<&CanonicalStackMemoryManifest>,
) -> Result<(WaffleModule<'static>, Vec<String>), String> {
    translate_module_inner(
        &module,
        import_modules,
        canonical_arena,
        canonical_memory,
        true,
    )
}

fn translate_module_inner(
    module: &Module,
    import_modules: &HashMap<String, String>,
    canonical_arena: bool,
    canonical_memory: Option<&CanonicalStackMemoryManifest>,
    release_translated_bodies: bool,
) -> Result<(WaffleModule<'static>, Vec<String>), String> {
    let mut wmod = WaffleModule::empty();
    let mut func_names = Vec::new();

    // Add linear memory (1 page = 64KB, growable). Do not impose an
    // implementation-policy maximum on generated programs. The canonical
    // allocator retains its wasm32 overflow checks and traps when the host
    // rejects `memory.grow`, while valid workloads may grow beyond the former
    // arbitrary 16 MiB ceiling.
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
    let intrinsic_names: std::collections::HashSet<&str> =
        ["addmod", "mulmod"].into_iter().collect();

    let mut func_map: HashMap<FuncRef, Func> = HashMap::new();
    let mut global_map = HashMap::new();
    let mut pending: Vec<(FuncRef, Func, waffle::Signature, String)> = Vec::new();

    // Sonatina scalar globals have value semantics at `mload`/`mstore` sites.
    // Represent them as genuine Wasm globals so state persists across exported
    // calls without consuming or aliasing linear memory.
    module.ctx.with_gv_store(|store| -> Result<(), String> {
        for gv in store.all_gv_refs() {
            let data = store.gv_data(gv);
            let Some(ty) = sonatina_to_waffle_type(data.ty) else {
                // Aggregate globals retain their existing data-address
                // semantics and are not silently reinterpreted as Wasm globals.
                continue;
            };
            let value = wasm_global_initializer(data.ty, data.initializer.as_ref())?;
            let wglobal = wmod.globals.push(GlobalData {
                ty,
                value: Some(value),
                mutable: !data.is_const,
            });
            if data.linkage == Linkage::Public {
                wmod.exports.push(waffle::Export {
                    name: data.symbol.clone(),
                    kind: ExportKind::Global(wglobal),
                });
            }
            global_map.insert(gv, wglobal);
        }
        Ok(())
    })?;

    // Pass 0: emit a wasm import for every external declaration (a function
    // with `Linkage::External` and no body). WAFFLE requires imported functions
    // to occupy the lowest slots of the `funcs` arena, so this MUST run before
    // pass 1 pushes any defined body: the wasm function index space then lays
    // imports out before defined functions, and the existing `Call` arm resolves
    // an imported callee through `func_map` with no special case.
    for &func_ref in &funcs {
        let has_body = module
            .func_store
            .try_view(func_ref, |f| f.layout.entry_block().is_some())
            .unwrap_or(false);
        if has_body {
            continue;
        }
        if module.ctx.func_linkage(func_ref) != Linkage::External {
            // A bodyless non-external declaration is not an import; pass 1's
            // has-body gate skips it, exactly as before this pass existed.
            continue;
        }

        let name = module.ctx.func_sig(func_ref, |sig| sig.name().to_string());
        // Do not race the addmod/mulmod intrinsic handling (op-matrix / R2
        // territory): leave those names skipped exactly as pass 1 does.
        if intrinsic_names.contains(name.as_str()) {
            continue;
        }

        // Fail closed: an external signature that is not representable in wasm
        // (e.g. i256) is an error, never a silently dropped param/result.
        let sig_data = module.ctx.func_sig(func_ref, |sig| {
            let mut params = Vec::with_capacity(sig.args().len());
            for ty in sig.args() {
                match sonatina_to_waffle_type_in_ctx(&module.ctx, *ty) {
                    Some(wty) => params.push(wty),
                    None => {
                        return Err(format!(
                            "wasm import `{name}`: parameter type `{ty:?}` is \
                             not representable in wasm"
                        ));
                    }
                }
            }
            let mut returns = Vec::with_capacity(sig.ret_tys().len());
            for ty in sig.ret_tys() {
                match sonatina_to_waffle_type_in_ctx(&module.ctx, *ty) {
                    Some(wty) => returns.push(wty),
                    None => {
                        return Err(format!(
                            "wasm import `{name}`: result type `{ty:?}` is not \
                             representable in wasm"
                        ));
                    }
                }
            }
            Ok(SignatureData { params, returns })
        })?;

        let wsig = wmod.signatures.push(sig_data);
        let wfunc = wmod.funcs.push(FuncDecl::Import(wsig, name.clone()));
        // The import module: the frontend-supplied name for this symbol, or the
        // flat `"fe"` v0 convention when the symbol is not in the side table (an
        // attribute-less `extern`). The import FIELD name stays the symbol.
        let import_module = import_modules
            .get(&name)
            .cloned()
            .unwrap_or_else(|| "fe".to_string());
        wmod.imports.push(waffle::Import {
            module: import_module,
            name,
            kind: waffle::ImportKind::Func(wfunc),
        });

        func_map.insert(func_ref, wfunc);
    }

    // Imported functions must occupy the lowest Wasm function indexes.
    let canonical_arena = canonical_arena
        .then(|| synthesize_canonical_arena(&mut wmod, memory, &mut func_names));
    let canonical_memory_arena = canonical_memory
        .map(|manifest| synthesize_canonical_memory(
            &mut wmod,
            memory,
            &mut func_names,
            manifest,
        ))
        .transpose()?;
    let canonical_arena = canonical_arena.or(canonical_memory_arena);

    // Pass 1: declare every translatable defined function up front (placeholder
    // bodies), recording the Sonatina `FuncRef` -> WAFFLE `Func` mapping. Doing
    // this before any body is translated is what lets `Call` resolve its
    // callee's WAFFLE function index regardless of definition order.
    for &func_ref in &funcs {
        let has_body = module
            .func_store
            .try_view(func_ref, |f| f.layout.entry_block().is_some())
            .unwrap_or(false);
        if !has_body {
            continue;
        }

        let name = module.ctx.func_sig(func_ref, |sig| sig.name().to_string());
        if intrinsic_names.contains(name.as_str()) {
            continue;
        }

        let (params, results) = module.ctx.func_sig(func_ref, |sig| {
            let params: Vec<WType> = sig
                .args()
                .iter()
                .filter_map(|ty| sonatina_to_waffle_type_in_ctx(&module.ctx, *ty))
                .collect();
            let results: Vec<WType> = sig
                .ret_tys()
                .iter()
                .filter_map(|ty| sonatina_to_waffle_type_in_ctx(&module.ctx, *ty))
                .collect();
            (params, results)
        });

        let sig_data = SignatureData {
            params,
            returns: results,
        };
        let wsig = wmod.signatures.push(sig_data);
        let placeholder = FunctionBody::new(&wmod, wsig);
        let wfunc = wmod.funcs.push(FuncDecl::Body(wsig, name.clone(), placeholder));

        // Linkage is the compiler-owned Wasm export boundary. Private helper
        // bodies remain callable inside the module without becoming host ABI.
        if module.ctx.func_linkage(func_ref) == Linkage::Public {
            wmod.exports.push(waffle::Export {
                name: name.clone(),
                kind: ExportKind::Func(wfunc),
            });
            func_names.push(name.clone());
        }

        func_map.insert(func_ref, wfunc);
        pending.push((func_ref, wfunc, wsig, name));
    }

    // Assign deterministic, non-zero table indexes to address-taken functions.
    // Slot zero is null, preserving WebAssembly's native null-pointer trap.
    let mut address_taken = Vec::new();
    let mut signature_types = Vec::new();
    for &func_ref in &funcs {
        module.func_store.try_view(func_ref, |function| {
            for block in function.layout.iter_block() {
                for inst_id in function.layout.iter_inst(block) {
                    let inst = function.dfg.inst(inst_id);
                    if let Some(ptr) =
                        <&sonatina_ir::inst::data::GetFunctionPtr as InstDowncast>::downcast(
                            function.inst_set(),
                            inst,
                        )
                    {
                        address_taken.push(*ptr.func());
                    }
                    if let Some(call) =
                        <&sonatina_ir::inst::control_flow::CallIndirect as InstDowncast>::downcast(
                            function.inst_set(),
                            inst,
                        )
                    {
                        signature_types.push(*call.signature());
                    }
                }
            }
        });
    }
    address_taken.sort_by_key(|func| func.as_u32());
    address_taken.dedup();
    signature_types.sort_by_key(|ty| format!("{ty:?}"));
    signature_types.dedup();

    let mut target_slots = HashMap::new();
    let needs_table = !address_taken.is_empty() || !signature_types.is_empty();
    let table = if !needs_table {
        None
    } else {
        let mut elements = vec![Func::invalid()];
        for target in address_taken {
            let wfunc = func_map.get(&target).copied().ok_or_else(|| {
                format!(
                    "wasm translation: address-taken function %{} has no wasm body or import",
                    target.as_u32()
                )
            })?;
            let slot = elements.len() as u32;
            target_slots.insert(target, slot);
            elements.push(wfunc);
        }
        let len = elements.len() as u64;
        Some(wmod.tables.push(TableData {
            ty: WType::FuncRef,
            initial: len,
            max: Some(len),
            func_elements: Some(elements),
        }))
    };

    let mut indirect_signatures = HashMap::new();
    for ty in signature_types {
        let signature = wmod.signatures.push(indirect_signature(&module.ctx, ty)?);
        indirect_signatures.insert(ty, signature);
    }

    // Pass 2: translate each body, now that every callee has a WAFFLE `Func`.
    for (func_ref, wfunc, wsig, name) in pending {
        let body = translate_function(
            module,
            func_ref,
            &wmod,
            wsig,
            memory,
            &func_map,
            &global_map,
            canonical_arena,
            table,
            &target_slots,
            &indirect_signatures,
        )?;
        if release_translated_bodies {
            let removed = module.func_store.remove(func_ref);
            debug_assert!(
                removed.is_some(),
                "owned Wasm translation must consume each translated Sonatina body",
            );
        }
        wmod.funcs[wfunc] = FuncDecl::Body(wsig, name, body);
    }

    Ok((wmod, func_names))
}

fn wasm_global_initializer(
    ty: Type,
    initializer: Option<&GvInitializer>,
) -> Result<u64, String> {
    let Some(initializer) = initializer else {
        return Ok(0);
    };
    let GvInitializer::Immediate(immediate) = initializer else {
        return Err(format!(
            "wasm scalar global `{ty:?}` requires a scalar initializer"
        ));
    };
    match (ty, immediate) {
        (Type::I1, Immediate::I1(value)) => Ok(u64::from(*value)),
        (Type::I8, Immediate::I8(value)) => Ok(*value as u8 as u64),
        (Type::I16, Immediate::I16(value)) => Ok(*value as u16 as u64),
        (Type::I32, Immediate::I32(value)) => Ok(*value as u32 as u64),
        (Type::I64, Immediate::I64(value)) => Ok(*value as u64),
        (Type::F32, Immediate::F32(bits)) => Ok(u64::from(*bits)),
        _ => Err(format!(
            "wasm global initializer `{immediate:?}` does not match `{ty:?}`"
        )),
    }
}

fn synthesize_canonical_arena(
    module: &mut WaffleModule<'static>,
    memory: waffle::Memory,
    func_names: &mut Vec<String>,
) -> CanonicalArenaFunctions {
    const HEAP_BASE: u32 = 1024;
    const PAGE_SHIFT: u32 = 16;
    const PAGE_MASK: u32 = (1 << PAGE_SHIFT) - 1;

    let cursor = module.globals.push(GlobalData {
        ty: WType::I32,
        value: Some(HEAP_BASE as u64),
        mutable: true,
    });
    let alloc_sig = module.signatures.push(SignatureData {
        params: vec![WType::I32, WType::I32],
        returns: vec![WType::I32],
    });
    let mut body = FunctionBody::new(module, alloc_sig);
    let entry = body.entry;
    let valid = body.add_block();
    let check_grow = body.add_block();
    let grow = body.add_block();
    let success = body.add_block();
    let trap = body.add_block();
    let size = body.blocks[entry].params[0].1;
    let align = body.blocks[entry].params[1].1;
    let zero = body.add_op(entry, Operator::I32Const { value: 0 }, &[], &[WType::I32]);
    let one = body.add_op(entry, Operator::I32Const { value: 1 }, &[], &[WType::I32]);
    let align_nonzero = body.add_op(entry, Operator::I32Ne, &[align, zero], &[WType::I32]);
    let align_minus_one = body.add_op(entry, Operator::I32Sub, &[align, one], &[WType::I32]);
    let align_bits = body.add_op(entry, Operator::I32And, &[align, align_minus_one], &[WType::I32]);
    let align_power_two = body.add_op(entry, Operator::I32Eq, &[align_bits, zero], &[WType::I32]);
    let alignment_valid =
        body.add_op(entry, Operator::I32And, &[align_nonzero, align_power_two], &[WType::I32]);
    body.set_terminator(entry, Terminator::CondBr {
        cond: alignment_valid,
        if_true: BlockTarget { block: valid, args: vec![] },
        if_false: BlockTarget { block: trap, args: vec![] },
    });

    let current =
        body.add_op(valid, Operator::GlobalGet { global_index: cursor }, &[], &[WType::I32]);
    let biased = body.add_op(valid, Operator::I32Add, &[current, align_minus_one], &[WType::I32]);
    let bias_overflow = body.add_op(valid, Operator::I32LtU, &[biased, current], &[WType::I32]);
    let negative_align = body.add_op(valid, Operator::I32Sub, &[zero, align], &[WType::I32]);
    let aligned = body.add_op(valid, Operator::I32And, &[biased, negative_align], &[WType::I32]);
    let end = body.add_op(valid, Operator::I32Add, &[aligned, size], &[WType::I32]);
    let end_overflow = body.add_op(valid, Operator::I32LtU, &[end, aligned], &[WType::I32]);
    let overflow = body.add_op(valid, Operator::I32Or, &[bias_overflow, end_overflow], &[WType::I32]);
    body.set_terminator(valid, Terminator::CondBr {
        cond: overflow,
        if_true: BlockTarget { block: trap, args: vec![] },
        if_false: BlockTarget { block: check_grow, args: vec![] },
    });

    let shift =
        body.add_op(check_grow, Operator::I32Const { value: PAGE_SHIFT }, &[], &[WType::I32]);
    let mask = body.add_op(check_grow, Operator::I32Const { value: PAGE_MASK }, &[], &[WType::I32]);
    let whole_pages = body.add_op(check_grow, Operator::I32ShrU, &[end, shift], &[WType::I32]);
    let remainder = body.add_op(check_grow, Operator::I32And, &[end, mask], &[WType::I32]);
    let partial = body.add_op(check_grow, Operator::I32Ne, &[remainder, zero], &[WType::I32]);
    let needed_pages =
        body.add_op(check_grow, Operator::I32Add, &[whole_pages, partial], &[WType::I32]);
    let current_pages =
        body.add_op(check_grow, Operator::MemorySize { mem: memory }, &[], &[WType::I32]);
    let needs_grow =
        body.add_op(check_grow, Operator::I32LtU, &[current_pages, needed_pages], &[WType::I32]);
    body.set_terminator(check_grow, Terminator::CondBr {
        cond: needs_grow,
        if_true: BlockTarget { block: grow, args: vec![] },
        if_false: BlockTarget { block: success, args: vec![] },
    });

    let delta = body.add_op(grow, Operator::I32Sub, &[needed_pages, current_pages], &[WType::I32]);
    let previous = body.add_op(grow, Operator::MemoryGrow { mem: memory }, &[delta], &[WType::I32]);
    let failed = body.add_op(grow, Operator::I32Const { value: u32::MAX }, &[], &[WType::I32]);
    let grew = body.add_op(grow, Operator::I32Ne, &[previous, failed], &[WType::I32]);
    body.set_terminator(grow, Terminator::CondBr {
        cond: grew,
        if_true: BlockTarget { block: success, args: vec![] },
        if_false: BlockTarget { block: trap, args: vec![] },
    });

    body.add_op(success, Operator::GlobalSet { global_index: cursor }, &[end], &[]);
    body.set_terminator(success, Terminator::Return { values: vec![aligned] });
    body.set_terminator(trap, Terminator::Unreachable);
    let alloc = module.funcs.push(FuncDecl::Body(
        alloc_sig, "fe_cabi_alloc".to_string(), body,
    ));
    module.exports.push(waffle::Export {
        name: "fe_cabi_alloc".to_string(),
        kind: ExportKind::Func(alloc),
    });
    func_names.push("fe_cabi_alloc".to_string());

    let reset_sig = module.signatures.push(SignatureData { params: vec![], returns: vec![] });
    let mut reset = FunctionBody::new(module, reset_sig);
    let base = reset.add_op(
        reset.entry, Operator::I32Const { value: HEAP_BASE }, &[], &[WType::I32],
    );
    reset.add_op(
        reset.entry, Operator::GlobalSet { global_index: cursor }, &[base], &[],
    );
    reset.set_terminator(reset.entry, Terminator::Return { values: vec![] });
    let reset_func = module.funcs.push(FuncDecl::Body(
        reset_sig, "fe_cabi_reset".to_string(), reset,
    ));
    module.exports.push(waffle::Export {
        name: "fe_cabi_reset".to_string(),
        kind: ExportKind::Func(reset_func),
    });
    func_names.push("fe_cabi_reset".to_string());
    let checkpoint = synthesize_arena_checkpoint(module, cursor, func_names);
    let rewind = synthesize_arena_rewind(module, cursor, HEAP_BASE, func_names);
    CanonicalArenaFunctions {
        alloc,
        checkpoint,
        rewind,
    }
}

fn synthesize_arena_checkpoint(
    module: &mut WaffleModule<'static>,
    cursor: waffle::Global,
    func_names: &mut Vec<String>,
) -> Func {
    let sig = module.signatures.push(SignatureData {
        params: vec![],
        returns: vec![WType::I32],
    });
    let mut body = FunctionBody::new(module, sig);
    let current = body.add_op(
        body.entry,
        Operator::GlobalGet {
            global_index: cursor,
        },
        &[],
        &[WType::I32],
    );
    body.set_terminator(body.entry, Terminator::Return { values: vec![current] });
    let name = "__fe_cabi_checkpoint_internal".to_string();
    let checkpoint = module.funcs.push(FuncDecl::Body(sig, name.clone(), body));
    func_names.push(name);
    checkpoint
}

fn synthesize_arena_rewind(
    module: &mut WaffleModule<'static>,
    cursor: waffle::Global,
    heap_base: u32,
    func_names: &mut Vec<String>,
) -> Func {
    let sig = module.signatures.push(SignatureData {
        params: vec![WType::I32],
        returns: vec![],
    });
    let mut body = FunctionBody::new(module, sig);
    let entry = body.entry;
    let valid = body.add_block();
    let trap = body.add_block();
    let checkpoint = body.blocks[entry].params[0].1;
    let base = body.add_op(
        entry,
        Operator::I32Const { value: heap_base },
        &[],
        &[WType::I32],
    );
    let current = body.add_op(
        entry,
        Operator::GlobalGet {
            global_index: cursor,
        },
        &[],
        &[WType::I32],
    );
    let at_or_above_base = body.add_op(
        entry,
        Operator::I32GeU,
        &[checkpoint, base],
        &[WType::I32],
    );
    let at_or_below_cursor = body.add_op(
        entry,
        Operator::I32LeU,
        &[checkpoint, current],
        &[WType::I32],
    );
    let checkpoint_valid = body.add_op(
        entry,
        Operator::I32And,
        &[at_or_above_base, at_or_below_cursor],
        &[WType::I32],
    );
    body.set_terminator(entry, Terminator::CondBr {
        cond: checkpoint_valid,
        if_true: BlockTarget {
            block: valid,
            args: vec![],
        },
        if_false: BlockTarget {
            block: trap,
            args: vec![],
        },
    });
    body.add_op(
        valid,
        Operator::GlobalSet {
            global_index: cursor,
        },
        &[checkpoint],
        &[],
    );
    body.set_terminator(valid, Terminator::Return { values: vec![] });
    body.set_terminator(trap, Terminator::Unreachable);
    let name = "__fe_cabi_rewind_internal".to_string();
    let rewind = module.funcs.push(FuncDecl::Body(sig, name.clone(), body));
    func_names.push(name);
    rewind
}

fn synthesize_canonical_memory(
    module: &mut WaffleModule<'static>,
    memory: waffle::Memory,
    func_names: &mut Vec<String>,
    manifest: &CanonicalStackMemoryManifest,
) -> Result<CanonicalArenaFunctions, String> {
    const HEAP_BASE: u32 = 1024;
    const HEADER_SIZE: u32 = 16;
    const MAGIC: u32 = 0x0fec_ab1e;
    const PAGE_SHIFT: u32 = 16;
    const PAGE_MASK: u32 = (1 << PAGE_SHIFT) - 1;

    let mut names = std::collections::HashSet::new();
    names.insert("cabi_realloc");
    if manifest.scoped_host_borrows {
        names.insert("fe_cabi_checkpoint");
        names.insert("fe_cabi_rewind");
    }
    for name in &manifest.post_return_exports {
        if name.is_empty() || !names.insert(name.as_str()) {
            return Err(format!("duplicate or empty canonical-memory export `{name}`"));
        }
        if module.exports.iter().any(|export| export.name == *name) {
            return Err(format!("canonical-memory export `{name}` collides with module export"));
        }
    }
    if module.exports.iter().any(|export| export.name == "cabi_realloc") {
        return Err("canonical-memory export `cabi_realloc` collides with module export".into());
    }
    if manifest.scoped_host_borrows {
        for name in ["fe_cabi_checkpoint", "fe_cabi_rewind"] {
            if module.exports.iter().any(|export| export.name == name) {
                return Err(format!(
                    "canonical-memory export `{name}` collides with module export"
                ));
            }
        }
    }

    let cursor = module.globals.push(GlobalData {
        ty: WType::I32,
        value: Some(HEAP_BASE as u64),
        mutable: true,
    });
    let sig = module.signatures.push(SignatureData {
        params: vec![WType::I32, WType::I32, WType::I32, WType::I32],
        returns: vec![WType::I32],
    });
    let mut body = FunctionBody::new(module, sig);
    let entry = body.entry;
    let allocate = body.add_block();
    let validate_old = body.add_block();
    let resize = body.add_block();
    let release = body.add_block();
    let ensure_capacity = body.add_block();
    let grow = body.add_block();
    let commit = body.add_block();
    let return_zero = body.add_block();
    let trap = body.add_block();
    let old_ptr = body.blocks[entry].params[0].1;
    let old_size = body.blocks[entry].params[1].1;
    let align = body.blocks[entry].params[2].1;
    let new_size = body.blocks[entry].params[3].1;
    let zero = body.add_op(entry, Operator::I32Const { value: 0 }, &[], &[WType::I32]);
    let one = body.add_op(entry, Operator::I32Const { value: 1 }, &[], &[WType::I32]);
    let old_is_zero = body.add_op(entry, Operator::I32Eq, &[old_ptr, zero], &[WType::I32]);
    let align_nonzero = body.add_op(entry, Operator::I32Ne, &[align, zero], &[WType::I32]);
    let align_minus_one = body.add_op(entry, Operator::I32Sub, &[align, one], &[WType::I32]);
    let align_bits = body.add_op(entry, Operator::I32And, &[align, align_minus_one], &[WType::I32]);
    let align_power_two = body.add_op(entry, Operator::I32Eq, &[align_bits, zero], &[WType::I32]);
    let alignment_valid =
        body.add_op(entry, Operator::I32And, &[align_nonzero, align_power_two], &[WType::I32]);
    let dispatch = body.add_block();
    body.set_terminator(entry, Terminator::CondBr {
        cond: alignment_valid,
        if_true: BlockTarget { block: dispatch, args: vec![] },
        if_false: BlockTarget { block: trap, args: vec![] },
    });
    body.set_terminator(dispatch, Terminator::CondBr {
        cond: old_is_zero,
        if_true: BlockTarget { block: allocate, args: vec![] },
        if_false: BlockTarget { block: validate_old, args: vec![] },
    });

    let old_size_zero = body.add_op(allocate, Operator::I32Eq, &[old_size, zero], &[WType::I32]);
    let allocate_valid = body.add_block();
    body.set_terminator(allocate, Terminator::CondBr {
        cond: old_size_zero,
        if_true: BlockTarget { block: allocate_valid, args: vec![] },
        if_false: BlockTarget { block: trap, args: vec![] },
    });
    let new_is_zero =
        body.add_op(allocate_valid, Operator::I32Eq, &[new_size, zero], &[WType::I32]);
    let allocate_nonzero = body.add_block();
    body.set_terminator(allocate_valid, Terminator::CondBr {
        cond: new_is_zero,
        if_true: BlockTarget { block: return_zero, args: vec![] },
        if_false: BlockTarget { block: allocate_nonzero, args: vec![] },
    });
    let current = body.add_op(
        allocate_nonzero,
        Operator::GlobalGet { global_index: cursor },
        &[],
        &[WType::I32],
    );
    let header = body.add_op(
        allocate_nonzero,
        Operator::I32Const { value: HEADER_SIZE },
        &[],
        &[WType::I32],
    );
    let after_header =
        body.add_op(allocate_nonzero, Operator::I32Add, &[current, header], &[WType::I32]);
    let biased =
        body.add_op(allocate_nonzero, Operator::I32Add, &[after_header, align_minus_one], &[WType::I32]);
    let negative_align =
        body.add_op(allocate_nonzero, Operator::I32Sub, &[zero, align], &[WType::I32]);
    let new_ptr =
        body.add_op(allocate_nonzero, Operator::I32And, &[biased, negative_align], &[WType::I32]);
    let new_end =
        body.add_op(allocate_nonzero, Operator::I32Add, &[new_ptr, new_size], &[WType::I32]);
    let header_overflow =
        body.add_op(allocate_nonzero, Operator::I32LtU, &[after_header, current], &[WType::I32]);
    let bias_overflow =
        body.add_op(allocate_nonzero, Operator::I32LtU, &[biased, after_header], &[WType::I32]);
    let end_overflow =
        body.add_op(allocate_nonzero, Operator::I32LtU, &[new_end, new_ptr], &[WType::I32]);
    let overflow =
        body.add_op(allocate_nonzero, Operator::I32Or, &[header_overflow, bias_overflow], &[WType::I32]);
    let overflow = body.add_op(allocate_nonzero, Operator::I32Or, &[overflow, end_overflow], &[WType::I32]);
    let allocation_checked = body.add_block();
    body.set_terminator(allocate_nonzero, Terminator::CondBr {
        cond: overflow,
        if_true: BlockTarget { block: trap, args: vec![] },
        if_false: BlockTarget { block: allocation_checked, args: vec![] },
    });

    let min_ptr = body.add_op(
        validate_old,
        Operator::I32Const { value: HEAP_BASE + HEADER_SIZE },
        &[],
        &[WType::I32],
    );
    let ptr_too_low =
        body.add_op(validate_old, Operator::I32LtU, &[old_ptr, min_ptr], &[WType::I32]);
    let validate_header = body.add_block();
    body.set_terminator(validate_old, Terminator::CondBr {
        cond: ptr_too_low,
        if_true: BlockTarget { block: trap, args: vec![] },
        if_false: BlockTarget { block: validate_header, args: vec![] },
    });
    let current_live = body.add_op(
        validate_header,
        Operator::GlobalGet { global_index: cursor },
        &[],
        &[WType::I32],
    );
    let ptr_past_cursor =
        body.add_op(validate_header, Operator::I32LtU, &[current_live, old_ptr], &[WType::I32]);
    let validate_header_size = body.add_op(
        validate_header,
        Operator::I32Const { value: HEADER_SIZE },
        &[],
        &[WType::I32],
    );
    let header_addr = body.add_op(
        validate_header,
        Operator::I32Sub,
        &[old_ptr, validate_header_size],
        &[WType::I32],
    );
    let memarg = waffle::MemoryArg { align: 2, offset: 0, memory };
    let previous = body.add_op(
        validate_header,
        Operator::I32Load { memory: memarg },
        &[header_addr],
        &[WType::I32],
    );
    let four = body.add_op(validate_header, Operator::I32Const { value: 4 }, &[], &[WType::I32]);
    let eight = body.add_op(validate_header, Operator::I32Const { value: 8 }, &[], &[WType::I32]);
    let twelve = body.add_op(validate_header, Operator::I32Const { value: 12 }, &[], &[WType::I32]);
    let size_addr = body.add_op(validate_header, Operator::I32Add, &[header_addr, four], &[WType::I32]);
    let align_addr = body.add_op(validate_header, Operator::I32Add, &[header_addr, eight], &[WType::I32]);
    let magic_addr = body.add_op(validate_header, Operator::I32Add, &[header_addr, twelve], &[WType::I32]);
    let stored_size = body.add_op(validate_header, Operator::I32Load { memory: memarg }, &[size_addr], &[WType::I32]);
    let stored_align = body.add_op(validate_header, Operator::I32Load { memory: memarg }, &[align_addr], &[WType::I32]);
    let stored_magic = body.add_op(validate_header, Operator::I32Load { memory: memarg }, &[magic_addr], &[WType::I32]);
    let expected_magic =
        body.add_op(validate_header, Operator::I32Const { value: MAGIC }, &[], &[WType::I32]);
    let magic_ok = body.add_op(validate_header, Operator::I32Eq, &[stored_magic, expected_magic], &[WType::I32]);
    let size_ok = body.add_op(validate_header, Operator::I32Eq, &[stored_size, old_size], &[WType::I32]);
    let align_ok = body.add_op(validate_header, Operator::I32Eq, &[stored_align, align], &[WType::I32]);
    let stored_end = body.add_op(validate_header, Operator::I32Add, &[old_ptr, stored_size], &[WType::I32]);
    let is_top = body.add_op(validate_header, Operator::I32Eq, &[stored_end, current_live], &[WType::I32]);
    let valid = body.add_op(validate_header, Operator::I32And, &[magic_ok, size_ok], &[WType::I32]);
    let valid = body.add_op(validate_header, Operator::I32And, &[valid, align_ok], &[WType::I32]);
    let valid = body.add_op(validate_header, Operator::I32And, &[valid, is_top], &[WType::I32]);
    let ptr_bounds_ok = body.add_op(validate_header, Operator::I32Eq, &[ptr_past_cursor, zero], &[WType::I32]);
    let valid = body.add_op(validate_header, Operator::I32And, &[valid, ptr_bounds_ok], &[WType::I32]);
    let old_valid = body.add_block();
    body.set_terminator(validate_header, Terminator::CondBr {
        cond: valid,
        if_true: BlockTarget { block: old_valid, args: vec![] },
        if_false: BlockTarget { block: trap, args: vec![] },
    });
    let release_requested =
        body.add_op(old_valid, Operator::I32Eq, &[new_size, zero], &[WType::I32]);
    body.set_terminator(old_valid, Terminator::CondBr {
        cond: release_requested,
        if_true: BlockTarget { block: release, args: vec![] },
        if_false: BlockTarget { block: resize, args: vec![] },
    });
    body.add_op(release, Operator::I32Store { memory: memarg }, &[magic_addr, zero], &[]);
    body.add_op(release, Operator::GlobalSet { global_index: cursor }, &[previous], &[]);
    body.set_terminator(release, Terminator::Return { values: vec![zero] });

    let resize_end = body.add_op(resize, Operator::I32Add, &[old_ptr, new_size], &[WType::I32]);
    let resize_overflow =
        body.add_op(resize, Operator::I32LtU, &[resize_end, old_ptr], &[WType::I32]);
    let resize_checked = body.add_block();
    body.set_terminator(resize, Terminator::CondBr {
        cond: resize_overflow,
        if_true: BlockTarget { block: trap, args: vec![] },
        if_false: BlockTarget { block: resize_checked, args: vec![] },
    });

    for index in 0..4 {
        let value =
            body.values.push(ValueDef::BlockParam(ensure_capacity, index, WType::I32));
        body.blocks[ensure_capacity].params.push((WType::I32, value));
    }
    let target_end = body.blocks[ensure_capacity].params[0].1;
    let target_ptr = body.blocks[ensure_capacity].params[1].1;
    let target_previous = body.blocks[ensure_capacity].params[2].1;
    let allocation_mode = body.blocks[ensure_capacity].params[3].1;
    body.set_terminator(allocation_checked, Terminator::Br {
        target: BlockTarget {
            block: ensure_capacity,
            args: vec![new_end, new_ptr, current, one],
        },
    });
    body.set_terminator(resize_checked, Terminator::Br {
        target: BlockTarget {
            block: ensure_capacity,
            args: vec![resize_end, old_ptr, zero, zero],
        },
    });
    let shift = body.add_op(ensure_capacity, Operator::I32Const { value: PAGE_SHIFT }, &[], &[WType::I32]);
    let mask = body.add_op(ensure_capacity, Operator::I32Const { value: PAGE_MASK }, &[], &[WType::I32]);
    let whole_pages = body.add_op(ensure_capacity, Operator::I32ShrU, &[target_end, shift], &[WType::I32]);
    let remainder = body.add_op(ensure_capacity, Operator::I32And, &[target_end, mask], &[WType::I32]);
    let partial = body.add_op(ensure_capacity, Operator::I32Ne, &[remainder, zero], &[WType::I32]);
    let needed_pages = body.add_op(ensure_capacity, Operator::I32Add, &[whole_pages, partial], &[WType::I32]);
    let current_pages = body.add_op(ensure_capacity, Operator::MemorySize { mem: memory }, &[], &[WType::I32]);
    let needs_grow = body.add_op(ensure_capacity, Operator::I32LtU, &[current_pages, needed_pages], &[WType::I32]);
    body.set_terminator(ensure_capacity, Terminator::CondBr {
        cond: needs_grow,
        if_true: BlockTarget {
            block: grow,
            args: vec![target_end, target_ptr, target_previous, allocation_mode],
        },
        if_false: BlockTarget {
            block: commit,
            args: vec![target_end, target_ptr, target_previous, allocation_mode],
        },
    });
    for index in 0..4 {
        let value = body.values.push(ValueDef::BlockParam(grow, index, WType::I32));
        body.blocks[grow].params.push((WType::I32, value));
        let value = body.values.push(ValueDef::BlockParam(commit, index, WType::I32));
        body.blocks[commit].params.push((WType::I32, value));
    }
    let grow_end = body.blocks[grow].params[0].1;
    let grow_ptr = body.blocks[grow].params[1].1;
    let grow_previous = body.blocks[grow].params[2].1;
    let grow_mode = body.blocks[grow].params[3].1;
    let delta = body.add_op(grow, Operator::I32Sub, &[needed_pages, current_pages], &[WType::I32]);
    let prior_pages = body.add_op(grow, Operator::MemoryGrow { mem: memory }, &[delta], &[WType::I32]);
    let failed = body.add_op(grow, Operator::I32Const { value: u32::MAX }, &[], &[WType::I32]);
    let grew = body.add_op(grow, Operator::I32Ne, &[prior_pages, failed], &[WType::I32]);
    body.set_terminator(grow, Terminator::CondBr {
        cond: grew,
        if_true: BlockTarget {
            block: commit,
            args: vec![grow_end, grow_ptr, grow_previous, grow_mode],
        },
        if_false: BlockTarget { block: trap, args: vec![] },
    });
    let committed_end = body.blocks[commit].params[0].1;
    let committed_ptr = body.blocks[commit].params[1].1;
    let committed_previous = body.blocks[commit].params[2].1;
    let committed_mode = body.blocks[commit].params[3].1;
    let allocating = body.add_op(commit, Operator::I32Ne, &[committed_mode, zero], &[WType::I32]);
    let commit_allocate = body.add_block();
    let commit_resize = body.add_block();
    body.set_terminator(commit, Terminator::CondBr {
        cond: allocating,
        if_true: BlockTarget { block: commit_allocate, args: vec![] },
        if_false: BlockTarget { block: commit_resize, args: vec![] },
    });
    let commit_header = body.add_op(
        commit_allocate,
        Operator::I32Const { value: HEADER_SIZE },
        &[],
        &[WType::I32],
    );
    let commit_four =
        body.add_op(commit_allocate, Operator::I32Const { value: 4 }, &[], &[WType::I32]);
    let commit_eight =
        body.add_op(commit_allocate, Operator::I32Const { value: 8 }, &[], &[WType::I32]);
    let commit_twelve =
        body.add_op(commit_allocate, Operator::I32Const { value: 12 }, &[], &[WType::I32]);
    let commit_magic =
        body.add_op(commit_allocate, Operator::I32Const { value: MAGIC }, &[], &[WType::I32]);
    let new_header_addr = body.add_op(
        commit_allocate,
        Operator::I32Sub,
        &[committed_ptr, commit_header],
        &[WType::I32],
    );
    let new_size_addr =
        body.add_op(commit_allocate, Operator::I32Add, &[new_header_addr, commit_four], &[WType::I32]);
    let new_align_addr =
        body.add_op(commit_allocate, Operator::I32Add, &[new_header_addr, commit_eight], &[WType::I32]);
    let new_magic_addr =
        body.add_op(commit_allocate, Operator::I32Add, &[new_header_addr, commit_twelve], &[WType::I32]);
    body.add_op(commit_allocate, Operator::I32Store { memory: memarg }, &[new_header_addr, committed_previous], &[]);
    body.add_op(commit_allocate, Operator::I32Store { memory: memarg }, &[new_size_addr, new_size], &[]);
    body.add_op(commit_allocate, Operator::I32Store { memory: memarg }, &[new_align_addr, align], &[]);
    body.add_op(commit_allocate, Operator::I32Store { memory: memarg }, &[new_magic_addr, commit_magic], &[]);
    body.add_op(commit_allocate, Operator::GlobalSet { global_index: cursor }, &[committed_end], &[]);
    body.set_terminator(commit_allocate, Terminator::Return { values: vec![committed_ptr] });
    let resize_header = body.add_op(
        commit_resize,
        Operator::I32Const { value: HEADER_SIZE - 4 },
        &[],
        &[WType::I32],
    );
    let resize_size_addr = body.add_op(
        commit_resize,
        Operator::I32Sub,
        &[committed_ptr, resize_header],
        &[WType::I32],
    );
    body.add_op(
        commit_resize,
        Operator::I32Store { memory: memarg },
        &[resize_size_addr, new_size],
        &[],
    );
    body.add_op(commit_resize, Operator::GlobalSet { global_index: cursor }, &[committed_end], &[]);
    body.set_terminator(commit_resize, Terminator::Return { values: vec![committed_ptr] });
    body.set_terminator(return_zero, Terminator::Return { values: vec![zero] });
    body.set_terminator(trap, Terminator::Unreachable);

    let realloc = module.funcs.push(FuncDecl::Body(sig, "cabi_realloc".into(), body));
    module.exports.push(waffle::Export {
        name: "cabi_realloc".into(),
        kind: ExportKind::Func(realloc),
    });
    func_names.push("cabi_realloc".into());

    let post_sig = module.signatures.push(SignatureData {
        params: vec![WType::I32, WType::I32, WType::I32],
        returns: vec![],
    });
    for name in &manifest.post_return_exports {
        let mut post = FunctionBody::new(module, post_sig);
        let ptr = post.blocks[post.entry].params[0].1;
        let size = post.blocks[post.entry].params[1].1;
        let alignment = post.blocks[post.entry].params[2].1;
        let zero = post.add_op(post.entry, Operator::I32Const { value: 0 }, &[], &[WType::I32]);
        post.add_op(
            post.entry,
            Operator::Call { function_index: realloc },
            &[ptr, size, alignment, zero],
            &[WType::I32],
        );
        post.set_terminator(post.entry, Terminator::Return { values: vec![] });
        let function = module.funcs.push(FuncDecl::Body(post_sig, name.clone(), post));
        module.exports.push(waffle::Export {
            name: name.clone(),
            kind: ExportKind::Func(function),
        });
        func_names.push(name.clone());
    }
    let internal_sig = module.signatures.push(SignatureData {
        params: vec![WType::I32, WType::I32],
        returns: vec![WType::I32],
    });
    let mut internal = FunctionBody::new(module, internal_sig);
    let size = internal.blocks[internal.entry].params[0].1;
    let alignment = internal.blocks[internal.entry].params[1].1;
    let zero =
        internal.add_op(internal.entry, Operator::I32Const { value: 0 }, &[], &[WType::I32]);
    let result = internal.add_op(
        internal.entry,
        Operator::Call { function_index: realloc },
        &[zero, zero, alignment, size],
        &[WType::I32],
    );
    internal.set_terminator(internal.entry, Terminator::Return { values: vec![result] });
    let alloc = module.funcs.push(FuncDecl::Body(
        internal_sig,
        "__fe_cabi_alloc_internal".into(),
        internal,
    ));
    let checkpoint = synthesize_arena_checkpoint(module, cursor, func_names);
    let rewind = synthesize_arena_rewind(module, cursor, HEAP_BASE, func_names);
    if manifest.scoped_host_borrows {
        module.exports.push(waffle::Export {
            name: "fe_cabi_checkpoint".to_string(),
            kind: ExportKind::Func(checkpoint),
        });
        module.exports.push(waffle::Export {
            name: "fe_cabi_rewind".to_string(),
            kind: ExportKind::Func(rewind),
        });
    }
    Ok(CanonicalArenaFunctions {
        alloc,
        checkpoint,
        rewind,
    })
}

fn translate_function(
    module: &Module,
    func_ref: FuncRef,
    wmod: &WaffleModule,
    wsig: waffle::Signature,
    memory: waffle::Memory,
    func_map: &HashMap<FuncRef, Func>,
    global_map: &HashMap<GlobalVariableRef, waffle::Global>,
    canonical_arena: Option<CanonicalArenaFunctions>,
    table: Option<waffle::Table>,
    target_slots: &HashMap<FuncRef, u32>,
    indirect_signatures: &HashMap<Type, waffle::Signature>,
) -> Result<FunctionBody, String> {
    let mut body = FunctionBody::new(wmod, wsig);
    // Stack pointer for bump allocation in linear memory (starts at 1024 to leave space)
    let mut stack_ptr: u32 = 1024;

    module
        .func_store
        .try_view(func_ref, |function| {
            let inst_set = function.inst_set();

            // Map Sonatina blocks → WAFFLE blocks
            let mut block_map: HashMap<BlockId, waffle::Block> = HashMap::new();
            let entry_block = function.layout.entry_block().ok_or("no entry block")?;

            // The entry block in WAFFLE is already created by FunctionBody::new
            block_map.insert(entry_block, body.entry);

            for block in function.layout.iter_block() {
                if block != entry_block {
                    let wb = body.add_block();
                    block_map.insert(block, wb);
                }
            }

            // Map Sonatina values → WAFFLE values
            let mut value_map: HashMap<ValueId, waffle::Value> = HashMap::new();

            // Map function args (entry block params in WAFFLE)
            for (idx, &arg_value) in function.arg_values.iter().enumerate() {
                let entry_params = body.blocks[body.entry].params.clone();
                if idx < entry_params.len() {
                    value_map.insert(arg_value, entry_params[idx].1);
                }
            }

            // First pass: create block params for phi nodes
            for block in function.layout.iter_block() {
                if block == entry_block {
                    continue;
                }
                let wb = block_map[&block];
                for inst_id in function.layout.iter_inst(block) {
                    let inst_data = function.dfg.inst(inst_id);
                    if let Some(_phi) = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let ty = function.dfg.value_ty(result);
                            let wty =
                                sonatina_to_waffle_type_in_ctx(function.ctx(), ty)
                                    .unwrap_or(WType::I64);
                            let param = body.add_blockparam(wb, wty);
                            value_map.insert(result, param);
                        }
                    } else {
                        break;
                    }
                }
            }

            // Second pass: translate instructions and set terminators.
            //
            // Iterate blocks in reverse post-order (dominators before the blocks
            // they dominate), NOT `layout` order. `resolve_value` is a single
            // forward pass keyed on a `value_map` populated as instructions are
            // translated; a value defined in a block laid out AFTER one that uses
            // it (a non-RPO layout, which Fe's MIR block numbering can produce for
            // e.g. loop pre-headers) would otherwise be `unresolved`. RPO makes
            // every non-phi definition precede its uses; loop back-edge values are
            // carried by phi results, which the first pass already pre-seeds. This
            // mirrors the EVM machine path, which already orders by `DomTree::rpo`
            // (isa/evm/machine/prepare.rs).
            let block_order = {
                let mut cfg = ControlFlowGraph::new();
                cfg.compute(function);
                let mut dom = DomTree::new();
                dom.compute(&cfg);
                let rpo = dom.rpo().to_owned();
                let in_rpo: std::collections::HashSet<BlockId> = rpo.iter().copied().collect();
                let mut order = rpo;
                // Keep processing any block the RPO walk did not reach (an
                // unreachable block still in layout), so its WAFFLE block still
                // receives a terminator exactly as under the old layout walk.
                for block in function.layout.iter_block() {
                    if !in_rpo.contains(&block) {
                        order.push(block);
                    }
                }
                order
            };
            for block in block_order.iter().copied() {
                let wb = block_map[&block];

                for inst_id in function.layout.iter_inst(block) {
                    let inst_data = function.dfg.inst(inst_id);

                    // Skip phi nodes (handled as block params above)
                    if <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                        continue;
                    }

                    // Return
                    if let Some(ret) = <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(inst_set, inst_data) {
                        let values: Vec<waffle::Value> = ret
                            .args()
                            .as_slice()
                            .iter()
                            .filter_map(|v| resolve_value(function, *v, &value_map, &mut body, wb))
                            .collect();
                        body.set_terminator(wb, Terminator::Return { values });
                    }
                    // Jump
                    else if let Some(jump) = <&sonatina_ir::inst::control_flow::Jump as InstDowncast>::downcast(inst_set, inst_data) {
                        let target_block = block_map[jump.dest()];
                        let args = collect_phi_args(function, *jump.dest(), block, inst_set, &value_map, &mut body, wb);
                        body.set_terminator(wb, Terminator::Br {
                            target: waffle::BlockTarget {
                                block: target_block,
                                args,
                            },
                        });
                    }
                    // Conditional branch
                    else if let Some(br) = <&sonatina_ir::inst::control_flow::Br as InstDowncast>::downcast(inst_set, inst_data) {
                        let cond = resolve_value(function, *br.cond(), &value_map, &mut body, wb)
                            .ok_or("unresolved branch condition")?;
                        let nz_block = block_map[br.nz_dest()];
                        let z_block = block_map[br.z_dest()];
                        let nz_args = collect_phi_args(function, *br.nz_dest(), block, inst_set, &value_map, &mut body, wb);
                        let z_args = collect_phi_args(function, *br.z_dest(), block, inst_set, &value_map, &mut body, wb);
                        body.set_terminator(wb, Terminator::CondBr {
                            cond,
                            if_true: waffle::BlockTarget {
                                block: nz_block,
                                args: nz_args,
                            },
                            if_false: waffle::BlockTarget {
                                block: z_block,
                                args: z_args,
                            },
                        });
                    }
                    // Unreachable
                    else if <&sonatina_ir::inst::control_flow::Unreachable as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                        body.set_terminator(wb, Terminator::Unreachable);
                    }
                    // Add
                    else if let Some(add) = <&sonatina_ir::inst::arith::Add as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *add.lhs(), &value_map, &mut body, wb)
                                .ok_or("unresolved add lhs")?;
                            let rhs = resolve_value(function, *add.rhs(), &value_map, &mut body, wb)
                                .ok_or("unresolved add rhs")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32Add } else { Operator::I64Add };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Sub
                    else if let Some(sub) = <&sonatina_ir::inst::arith::Sub as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *sub.lhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let rhs = resolve_value(function, *sub.rhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32Sub } else { Operator::I64Sub };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Mul
                    else if let Some(mul) = <&sonatina_ir::inst::arith::Mul as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *mul.lhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let rhs = resolve_value(function, *mul.rhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32Mul } else { Operator::I64Mul };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Integer division/remainder preserve Sonatina's explicit
                    // signedness in the selected WebAssembly opcode. Wasm
                    // provides the required divide-by-zero trap directly; its
                    // signed division also traps MIN / -1, while signed
                    // remainder returns zero for that pair as specified.
                    else if let Some(div) = <&sonatina_ir::inst::arith::Udiv as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *div.lhs(), &value_map, &mut body, wb).ok_or("unresolved udiv lhs")?;
                            let rhs = resolve_value(function, *div.rhs(), &value_map, &mut body, wb).ok_or("unresolved udiv rhs")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32DivU } else { Operator::I64DivU };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(div) = <&sonatina_ir::inst::arith::Sdiv as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *div.lhs(), &value_map, &mut body, wb).ok_or("unresolved sdiv lhs")?;
                            let rhs = resolve_value(function, *div.rhs(), &value_map, &mut body, wb).ok_or("unresolved sdiv rhs")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32DivS } else { Operator::I64DivS };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(rem) = <&sonatina_ir::inst::arith::Umod as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *rem.lhs(), &value_map, &mut body, wb).ok_or("unresolved umod lhs")?;
                            let rhs = resolve_value(function, *rem.rhs(), &value_map, &mut body, wb).ok_or("unresolved umod rhs")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32RemU } else { Operator::I64RemU };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(rem) = <&sonatina_ir::inst::arith::Smod as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *rem.lhs(), &value_map, &mut body, wb).ok_or("unresolved smod lhs")?;
                            let rhs = resolve_value(function, *rem.rhs(), &value_map, &mut body, wb).ok_or("unresolved smod rhs")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32RemS } else { Operator::I64RemS };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Bitwise and
                    else if let Some(and) = <&sonatina_ir::inst::logic::And as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *and.lhs(), &value_map, &mut body, wb).ok_or("unresolved and lhs")?;
                            let rhs = resolve_value(function, *and.rhs(), &value_map, &mut body, wb).ok_or("unresolved and rhs")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32And } else { Operator::I64And };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Bitwise or. Sign-agnostic per-bit op, type-keyed like `And`.
                    else if let Some(or) = <&sonatina_ir::inst::logic::Or as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *or.lhs(), &value_map, &mut body, wb).ok_or("unresolved or lhs")?;
                            let rhs = resolve_value(function, *or.rhs(), &value_map, &mut body, wb).ok_or("unresolved or rhs")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32Or } else { Operator::I64Or };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Bitwise xor. Sign-agnostic per-bit op, type-keyed like `And`.
                    else if let Some(xor) = <&sonatina_ir::inst::logic::Xor as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *xor.lhs(), &value_map, &mut body, wb).ok_or("unresolved xor lhs")?;
                            let rhs = resolve_value(function, *xor.rhs(), &value_map, &mut body, wb).ok_or("unresolved xor rhs")?;
                            let ty = result_waffle_type(function, result);
                            let op = if ty == WType::I32 { Operator::I32Xor } else { Operator::I64Xor };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fneg) = <&sonatina_ir::inst::arith::Fneg as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let arg = resolve_value(function, *fneg.arg(), &value_map, &mut body, wb)
                                .ok_or("unresolved fneg argument")?;
                            let wval = body.add_op(wb, Operator::F32Neg, &[arg], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fsqrt) = <&sonatina_ir::inst::arith::Fsqrt as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let arg = resolve_value(function, *fsqrt.arg(), &value_map, &mut body, wb)
                                .ok_or("unresolved fsqrt argument")?;
                            let wval = body.add_op(wb, Operator::F32Sqrt, &[arg], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fabs) = <&sonatina_ir::inst::arith::Fabs as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let arg = resolve_value(function, *fabs.arg(), &value_map, &mut body, wb)
                                .ok_or("unresolved fabs argument")?;
                            let wval = body.add_op(wb, Operator::F32Abs, &[arg], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // `f32.min`/`f32.max` are wasm's NaN-propagating, -0.0 < +0.0
                    // "WebAssembly rules" -- this IS the pinned cross-backend
                    // semantics (see arith::Fmin/Fmax doc comments), so this is a
                    // direct, unconditional native lowering.
                    else if let Some(fmin) = <&sonatina_ir::inst::arith::Fmin as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *fmin.lhs(), &value_map, &mut body, wb).ok_or("unresolved fmin lhs")?;
                            let rhs = resolve_value(function, *fmin.rhs(), &value_map, &mut body, wb).ok_or("unresolved fmin rhs")?;
                            let wval = body.add_op(wb, Operator::F32Min, &[lhs, rhs], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fmax) = <&sonatina_ir::inst::arith::Fmax as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *fmax.lhs(), &value_map, &mut body, wb).ok_or("unresolved fmax lhs")?;
                            let rhs = resolve_value(function, *fmax.rhs(), &value_map, &mut body, wb).ok_or("unresolved fmax rhs")?;
                            let wval = body.add_op(wb, Operator::F32Max, &[lhs, rhs], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Relaxed min/max: same `f32.min`/`f32.max` native
                    // instruction as the exact ops above -- wasm's native
                    // instruction already IS the pinned "WebAssembly rules"
                    // exact semantics, so it is trivially a conforming
                    // implementation of the weaker relaxed contract too.
                    // Zero new backend surface.
                    else if let Some(fmin) = <&sonatina_ir::inst::arith::FminRelaxed as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *fmin.lhs(), &value_map, &mut body, wb).ok_or("unresolved fmin_relaxed lhs")?;
                            let rhs = resolve_value(function, *fmin.rhs(), &value_map, &mut body, wb).ok_or("unresolved fmin_relaxed rhs")?;
                            let wval = body.add_op(wb, Operator::F32Min, &[lhs, rhs], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fmax) = <&sonatina_ir::inst::arith::FmaxRelaxed as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *fmax.lhs(), &value_map, &mut body, wb).ok_or("unresolved fmax_relaxed lhs")?;
                            let rhs = resolve_value(function, *fmax.rhs(), &value_map, &mut body, wb).ok_or("unresolved fmax_relaxed rhs")?;
                            let wval = body.add_op(wb, Operator::F32Max, &[lhs, rhs], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Rounding family: single native instruction each, no
                    // bit-twiddling, no NaN/-0 subtlety.
                    else if let Some(ffloor) = <&sonatina_ir::inst::arith::Ffloor as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let arg = resolve_value(function, *ffloor.arg(), &value_map, &mut body, wb)
                                .ok_or("unresolved ffloor argument")?;
                            let wval = body.add_op(wb, Operator::F32Floor, &[arg], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fceil) = <&sonatina_ir::inst::arith::Fceil as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let arg = resolve_value(function, *fceil.arg(), &value_map, &mut body, wb)
                                .ok_or("unresolved fceil argument")?;
                            let wval = body.add_op(wb, Operator::F32Ceil, &[arg], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(ftrunc) = <&sonatina_ir::inst::arith::Ftrunc as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let arg = resolve_value(function, *ftrunc.arg(), &value_map, &mut body, wb)
                                .ok_or("unresolved ftrunc argument")?;
                            let wval = body.add_op(wb, Operator::F32Trunc, &[arg], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // `f32.nearest` is wasm's ties-to-even rounding (per the wasm
                    // spec), matching `Fround`'s pinned `roundTiesToEven` semantics
                    // exactly -- a direct, unconditional native lowering.
                    else if let Some(fround) = <&sonatina_ir::inst::arith::Fround as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let arg = resolve_value(function, *fround.arg(), &value_map, &mut body, wb)
                                .ok_or("unresolved fround argument")?;
                            let wval = body.add_op(wb, Operator::F32Nearest, &[arg], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // wasm has no native `f32.clamp`; compose it from the two native
                    // ops above as `min(max(arg, lo), hi)` (the textbook clamp
                    // order), matching the cranelift lowering's composition exactly.
                    else if let Some(fclamp) = <&sonatina_ir::inst::arith::Fclamp as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let arg = resolve_value(function, *fclamp.arg(), &value_map, &mut body, wb).ok_or("unresolved fclamp arg")?;
                            let lo = resolve_value(function, *fclamp.lo(), &value_map, &mut body, wb).ok_or("unresolved fclamp lo")?;
                            let hi = resolve_value(function, *fclamp.hi(), &value_map, &mut body, wb).ok_or("unresolved fclamp hi")?;
                            let maxed = body.add_op(wb, Operator::F32Max, &[arg, lo], &[WType::F32]);
                            let wval = body.add_op(wb, Operator::F32Min, &[maxed, hi], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fadd) = <&sonatina_ir::inst::arith::Fadd as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *fadd.lhs(), &value_map, &mut body, wb).ok_or("unresolved fadd lhs")?;
                            let rhs = resolve_value(function, *fadd.rhs(), &value_map, &mut body, wb).ok_or("unresolved fadd rhs")?;
                            let wval = body.add_op(wb, Operator::F32Add, &[lhs, rhs], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fsub) = <&sonatina_ir::inst::arith::Fsub as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *fsub.lhs(), &value_map, &mut body, wb).ok_or("unresolved fsub lhs")?;
                            let rhs = resolve_value(function, *fsub.rhs(), &value_map, &mut body, wb).ok_or("unresolved fsub rhs")?;
                            let wval = body.add_op(wb, Operator::F32Sub, &[lhs, rhs], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fmul) = <&sonatina_ir::inst::arith::Fmul as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *fmul.lhs(), &value_map, &mut body, wb).ok_or("unresolved fmul lhs")?;
                            let rhs = resolve_value(function, *fmul.rhs(), &value_map, &mut body, wb).ok_or("unresolved fmul rhs")?;
                            let wval = body.add_op(wb, Operator::F32Mul, &[lhs, rhs], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fdiv) = <&sonatina_ir::inst::arith::Fdiv as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *fdiv.lhs(), &value_map, &mut body, wb).ok_or("unresolved fdiv lhs")?;
                            let rhs = resolve_value(function, *fdiv.rhs(), &value_map, &mut body, wb).ok_or("unresolved fdiv rhs")?;
                            let wval = body.add_op(wb, Operator::F32Div, &[lhs, rhs], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(feq) = <&sonatina_ir::inst::cmp::Feq as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *feq.lhs(), &value_map, &mut body, wb).ok_or("unresolved feq lhs")?;
                            let rhs = resolve_value(function, *feq.rhs(), &value_map, &mut body, wb).ok_or("unresolved feq rhs")?;
                            let wval = body.add_op(wb, Operator::F32Eq, &[lhs, rhs], &[WType::I32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(flt) = <&sonatina_ir::inst::cmp::Flt as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *flt.lhs(), &value_map, &mut body, wb).ok_or("unresolved flt lhs")?;
                            let rhs = resolve_value(function, *flt.rhs(), &value_map, &mut body, wb).ok_or("unresolved flt rhs")?;
                            let wval = body.add_op(wb, Operator::F32Lt, &[lhs, rhs], &[WType::I32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(fle) = <&sonatina_ir::inst::cmp::Fle as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *fle.lhs(), &value_map, &mut body, wb).ok_or("unresolved fle lhs")?;
                            let rhs = resolve_value(function, *fle.rhs(), &value_map, &mut body, wb).ok_or("unresolved fle rhs")?;
                            let wval = body.add_op(wb, Operator::F32Le, &[lhs, rhs], &[WType::I32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Lt (unsigned). Key on the OPERAND type, not the result (the
                    // result is the I32 bool): an i32-operand compare must use
                    // `i32.lt_u` or wasmtime rejects the module at validation.
                    else if let Some(lt) = <&sonatina_ir::inst::cmp::Lt as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *lt.lhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let rhs = resolve_value(function, *lt.rhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let ty = sonatina_to_waffle_type(function.dfg.value_ty(*lt.lhs())).unwrap_or(WType::I64);
                            let op = if ty == WType::I32 { Operator::I32LtU } else { Operator::I64LtU };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[WType::I32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Eq. Keyed on the operand type, same as the compares above.
                    else if let Some(eq) = <&sonatina_ir::inst::cmp::Eq as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *eq.lhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let rhs = resolve_value(function, *eq.rhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let ty = sonatina_to_waffle_type(function.dfg.value_ty(*eq.lhs())).unwrap_or(WType::I64);
                            let op = if ty == WType::I32 { Operator::I32Eq } else { Operator::I64Eq };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[WType::I32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Slt (signed less-than). Keyed on the operand type: i32 -> I32LtS,
                    // else I64LtS. The signed vs unsigned choice is load-bearing for
                    // 0x80000000-class operands.
                    else if let Some(slt) = <&sonatina_ir::inst::cmp::Slt as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let lhs = resolve_value(function, *slt.lhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let rhs = resolve_value(function, *slt.rhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let ty = sonatina_to_waffle_type(function.dfg.value_ty(*slt.lhs())).unwrap_or(WType::I64);
                            let op = if ty == WType::I32 { Operator::I32LtS } else { Operator::I64LtS };
                            let wval = body.add_op(wb, op, &[lhs, rhs], &[WType::I32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // All overflow operations share result transport. Their
                    // signedness and semantic width select the exact checks.
                    else if let Some((op, signed, lhs_id, rhs_id)) = overflow_operands(inst_set, inst_data) {
                        let ir_results = function.dfg.inst_results(inst_id);
                        if !ir_results.is_empty() {
                            let lhs = resolve_value(function, lhs_id, &value_map, &mut body, wb).ok_or("unresolved overflow lhs")?;
                            let rhs = resolve_value(function, rhs_id, &value_map, &mut body, wb).ok_or("unresolved overflow rhs")?;
                            let lower = if signed { signed_overflow_arithmetic } else { unsigned_overflow_arithmetic };
                            let (wval, overflow) = lower(&mut body, wb, function.dfg.value_ty(lhs_id), lhs, rhs, op)?;
                            value_map.insert(ir_results[0], wval);
                            if ir_results.len() >= 2 {
                                value_map.insert(ir_results[1], overflow);
                            }
                        }
                    }
                    // Sar (arithmetic shift right). Keyed on the VALUE operand's type
                    // (the `bits` immediate is created at the value's type by fe, so
                    // both inputs agree): i32 -> I32ShrS, else I64ShrS. Output type
                    // matches the value operand.
                    else if let Some(sar) = <&sonatina_ir::inst::arith::Sar as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let val = resolve_value(function, *sar.value(), &value_map, &mut body, wb).ok_or("unresolved sar val")?;
                            let bits = resolve_value(function, *sar.bits(), &value_map, &mut body, wb).ok_or("unresolved sar bits")?;
                            let ty = sonatina_to_waffle_type(function.dfg.value_ty(*sar.value())).unwrap_or(WType::I64);
                            let op = if ty == WType::I32 { Operator::I32ShrS } else { Operator::I64ShrS };
                            let wval = body.add_op(wb, op, &[val, bits], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Shr (logical shift right). Keyed on the VALUE operand's type,
                    // mirroring `Sar`: i32 -> I32ShrU, else I64ShrU. Output type
                    // matches the value operand. The pre-word-aware emission always
                    // used I64ShrU, so i64 values stay byte-identical.
                    else if let Some(shr) = <&sonatina_ir::inst::arith::Shr as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let val = resolve_value(function, *shr.value(), &value_map, &mut body, wb).ok_or("unresolved shr val")?;
                            let bits = resolve_value(function, *shr.bits(), &value_map, &mut body, wb).ok_or("unresolved shr bits")?;
                            let ty = sonatina_to_waffle_type(function.dfg.value_ty(*shr.value())).unwrap_or(WType::I64);
                            let op = if ty == WType::I32 { Operator::I32ShrU } else { Operator::I64ShrU };
                            let wval = body.add_op(wb, op, &[val, bits], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Shl. Keyed on the VALUE operand's type (as `Sar`/`Shr`):
                    // i32 -> I32Shl, else I64Shl. i64 values stay byte-identical.
                    else if let Some(shl) = <&sonatina_ir::inst::arith::Shl as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let val = resolve_value(function, *shl.value(), &value_map, &mut body, wb).ok_or("unresolved shl val")?;
                            let bits = resolve_value(function, *shl.bits(), &value_map, &mut body, wb).ok_or("unresolved shl bits")?;
                            let ty = sonatina_to_waffle_type(function.dfg.value_ty(*shl.value())).unwrap_or(WType::I64);
                            let op = if ty == WType::I32 { Operator::I32Shl } else { Operator::I64Shl };
                            let wval = body.add_op(wb, op, &[val, bits], &[ty]);
                            value_map.insert(result, wval);
                        }
                    }
                    // ObjLoad — load i64 from linear memory at address
                    else if let Some(obj_load) = <&sonatina_ir::inst::data::ObjLoad as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let addr = resolve_value(function, *obj_load.object(), &value_map, &mut body, wb);
                            if let Some(v) = addr {
                                let result_ty = function.dfg.value_ty(result);
                                if result_ty == Type::I256 || matches!(result_ty, Type::Compound(_)) {
                                    // For compound types / i256: pass through address
                                    value_map.insert(result, v);
                                } else {
                                    // Load scalar from linear memory
                                    let mem_arg = waffle::MemoryArg { align: 8, offset: 0, memory };
                                    let loaded = body.add_op(wb, Operator::I64Load { memory: mem_arg }, &[v], &[WType::I64]);
                                    value_map.insert(result, loaded);
                                }
                            }
                        }
                    }
                    // ObjStore — store i64 to linear memory
                    else if let Some(obj_store) = <&sonatina_ir::inst::data::ObjStore as InstDowncast>::downcast(inst_set, inst_data) {
                        let dest = resolve_value(function, *obj_store.object(), &value_map, &mut body, wb);
                        let val = resolve_value(function, *obj_store.value(), &value_map, &mut body, wb);
                        if let (Some(d), Some(v)) = (dest, val) {
                            let mem_arg = waffle::MemoryArg { align: 8, offset: 0, memory };
                            body.add_op(wb, Operator::I64Store { memory: mem_arg }, &[d, v], &[]);
                        }
                    }
                    // ObjAlloc — bump allocate in linear memory
                    else if <&sonatina_ir::inst::data::ObjAlloc as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let result_ty = function.dfg.value_ty(result);
                            let alloc_size = module.ctx.size_of_unchecked(result_ty).max(8) as u32;
                            let addr = body.add_op(wb, Operator::I32Const { value: stack_ptr }, &[], &[WType::I32]);
                            stack_ptr += alloc_size;
                            value_map.insert(result, addr);
                        }
                    }
                    // ObjIndex — pointer arithmetic: base + index * elem_size
                    else if let Some(obj_index) = <&sonatina_ir::inst::data::ObjIndex as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let base = resolve_value(function, *obj_index.object(), &value_map, &mut body, wb).ok_or("unresolved obj_index base")?;
                            let index_val_id = *obj_index.index();
                            let index_ty = function.dfg.value_ty(index_val_id);
                            let index = if index_ty == Type::I256 {
                                if let Some(imm) = function.dfg.value_imm(index_val_id) {
                                    let idx = match imm {
                                        sonatina_ir::Immediate::I256(v) => v.to_u256().low_u64() as u32,
                                        _ => 0,
                                    };
                                    body.add_op(wb, Operator::I32Const { value: idx }, &[], &[WType::I32])
                                } else {
                                    resolve_value(function, index_val_id, &value_map, &mut body, wb).ok_or("unresolved index")?
                                }
                            } else {
                                resolve_value(function, index_val_id, &value_map, &mut body, wb).ok_or("unresolved index")?
                            };
                            let obj_ty = function.dfg.value_ty(*obj_index.object());
                            let elem_size = crate::isa::compute_element_size(obj_ty, &module.ctx) as u32;
                            let stride = body.add_op(wb, Operator::I32Const { value: elem_size }, &[], &[WType::I32]);
                            let offset = body.add_op(wb, Operator::I32Mul, &[index, stride], &[WType::I32]);
                            let addr = body.add_op(wb, Operator::I32Add, &[base, offset], &[WType::I32]);
                            value_map.insert(result, addr);
                        }
                    }
                    // MemAllocDynamic has one operand, byte size, and no
                    // alignment operand in Sonatina IR. Its portable contract
                    // therefore cannot promise more than byte alignment. Route
                    // it through the opt-in canonical arena with align=1 so it
                    // shares growth, overflow checks, and the one cursor.
                    else if let Some(alloc) = <&sonatina_ir::inst::data::MemAllocDynamic as InstDowncast>::downcast(inst_set, inst_data) {
                        let canonical_arena = canonical_arena.ok_or(
                            "wasm translation: mem.alloc_dynamic requires the opt-in canonical arena",
                        )?;
                        let size_ty = function.dfg.value_ty(*alloc.size());
                        if sonatina_to_waffle_type(size_ty) != Some(WType::I32) {
                            return Err(format!(
                                "wasm translation: mem.alloc_dynamic size `{size_ty:?}` is not representable by the canonical wasm32 allocator"
                            ));
                        }
                        let size = resolve_value(function, *alloc.size(), &value_map, &mut body, wb)
                            .ok_or("unresolved mem.alloc_dynamic size")?;
                        let align = body.add_op(
                            wb,
                            Operator::I32Const { value: 1 },
                            &[],
                            &[WType::I32],
                        );
                        let result = function
                            .dfg
                            .inst_result(inst_id)
                            .ok_or("mem.alloc_dynamic has no pointer result")?;
                        let address = body.add_op(
                            wb,
                            Operator::Call {
                                function_index: canonical_arena.alloc,
                            },
                            &[size, align],
                            &[WType::I32],
                        );
                        value_map.insert(result, address);
                    }
                    // MemCheckpoint observes the current arena cursor without
                    // allocating. It exists specifically to bracket a
                    // compiler-proven non-escaping function frame.
                    else if <&sonatina_ir::inst::data::MemCheckpoint as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                        let canonical_arena = canonical_arena.ok_or(
                            "wasm translation: mem.checkpoint requires the opt-in canonical arena",
                        )?;
                        let result = function
                            .dfg
                            .inst_result(inst_id)
                            .ok_or("mem.checkpoint has no pointer result")?;
                        let checkpoint = body.add_op(
                            wb,
                            Operator::Call {
                                function_index: canonical_arena.checkpoint,
                            },
                            &[],
                            &[WType::I32],
                        );
                        value_map.insert(result, checkpoint);
                    }
                    // MemRewind restores a compiler-proven non-escaping
                    // function frame. The synthesized helper validates that
                    // the checkpoint belongs to the live arena prefix before
                    // moving the cursor backwards.
                    else if let Some(rewind) = <&sonatina_ir::inst::data::MemRewind as InstDowncast>::downcast(inst_set, inst_data) {
                        let canonical_arena = canonical_arena.ok_or(
                            "wasm translation: mem.rewind requires the opt-in canonical arena",
                        )?;
                        let checkpoint = resolve_value(
                            function,
                            *rewind.checkpoint(),
                            &value_map,
                            &mut body,
                            wb,
                        )
                        .ok_or("unresolved mem.rewind checkpoint")?;
                        body.add_op(
                            wb,
                            Operator::Call {
                                function_index: canonical_arena.rewind,
                            },
                            &[checkpoint],
                            &[],
                        );
                    }
                    // Alloca — allocate a local
                    else if <&sonatina_ir::inst::data::Alloca as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let zero = body.add_op(wb, Operator::I64Const { value: 0 }, &[], &[WType::I64]);
                            value_map.insert(result, zero);
                        }
                    }
                    // Mstore — typed scalar store to linear memory.
                    else if let Some(mstore) = <&sonatina_ir::inst::data::Mstore as InstDowncast>::downcast(inst_set, inst_data) {
                        let value = resolve_value(function, *mstore.value(), &value_map, &mut body, wb)
                            .ok_or("unresolved mstore value")?;
                        if let Value::Global { gv, .. } = function.dfg.value(*mstore.addr()) {
                            let global = global_map.get(gv).ok_or_else(|| {
                                format!("wasm mstore global `{gv:?}` is not a scalar global")
                            })?;
                            let (global_ty, is_const) = module.ctx.with_gv_store(|store| {
                                (store.ty(*gv), store.is_const(*gv))
                            });
                            if global_ty != *mstore.ty() {
                                return Err(format!(
                                    "wasm mstore type `{:?}` does not match global `{gv:?}` type `{global_ty:?}`",
                                    mstore.ty()
                                ));
                            }
                            if is_const {
                                return Err(format!(
                                    "wasm mstore cannot write immutable global `{gv:?}`"
                                ));
                            }
                            body.add_op(
                                wb,
                                Operator::GlobalSet {
                                    global_index: *global,
                                },
                                &[value],
                                &[],
                            );
                            continue;
                        }
                        let addr = resolve_value(function, *mstore.addr(), &value_map, &mut body, wb)
                            .ok_or("unresolved mstore address")?;
                        let memarg = scalar_memory_arg(memory, *mstore.ty())?;
                        let op = match mstore.ty() {
                            Type::I1 | Type::I8 => Operator::I32Store8 { memory: memarg },
                            Type::I16 => Operator::I32Store16 { memory: memarg },
                            Type::I32 => Operator::I32Store { memory: memarg },
                            Type::I64 => Operator::I64Store { memory: memarg },
                            Type::F32 => Operator::F32Store { memory: memarg },
                            ty => return Err(format!("unsupported wasm mstore type `{ty:?}`")),
                        };
                        body.add_op(wb, op, &[addr, value], &[]);
                    }
                    // Mload — typed scalar load from linear memory. Narrow
                    // integer values are zero-extended into Wasm's i32 carrier.
                    else if let Some(mload) = <&sonatina_ir::inst::data::Mload as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            if let Value::Global { gv, .. } = function.dfg.value(*mload.addr()) {
                                let global = global_map.get(gv).ok_or_else(|| {
                                    format!("wasm mload global `{gv:?}` is not a scalar global")
                                })?;
                                let global_ty =
                                    module.ctx.with_gv_store(|store| store.ty(*gv));
                                if global_ty != *mload.ty() {
                                    return Err(format!(
                                        "wasm mload type `{:?}` does not match global `{gv:?}` type `{global_ty:?}`",
                                        mload.ty()
                                    ));
                                }
                                let result_ty = sonatina_to_waffle_type(*mload.ty())
                                    .ok_or_else(|| format!(
                                        "unsupported wasm global mload type `{:?}`",
                                        mload.ty()
                                    ))?;
                                let loaded = body.add_op(
                                    wb,
                                    Operator::GlobalGet {
                                        global_index: *global,
                                    },
                                    &[],
                                    &[result_ty],
                                );
                                value_map.insert(result, loaded);
                                continue;
                            }
                            let addr = resolve_value(function, *mload.addr(), &value_map, &mut body, wb);
                            let addr = addr.ok_or("unresolved mload address")?;
                            let memarg = scalar_memory_arg(memory, *mload.ty())?;
                            let (op, result_ty) = match mload.ty() {
                                Type::I1 | Type::I8 => (Operator::I32Load8U { memory: memarg }, WType::I32),
                                Type::I16 => (Operator::I32Load16U { memory: memarg }, WType::I32),
                                Type::I32 => (Operator::I32Load { memory: memarg }, WType::I32),
                                Type::I64 => (Operator::I64Load { memory: memarg }, WType::I64),
                                Type::F32 => (Operator::F32Load { memory: memarg }, WType::F32),
                                ty => return Err(format!("unsupported wasm mload type `{ty:?}`")),
                            };
                            let loaded = body.add_op(wb, op, &[addr], &[result_ty]);
                            value_map.insert(result, loaded);
                        }
                    }
                    // Memzero has the exact WebAssembly memory.fill semantics:
                    // write `len` zero bytes beginning at `dest`.
                    else if let Some(memzero) = <&sonatina_ir::inst::data::Memzero as InstDowncast>::downcast(inst_set, inst_data) {
                        let dest = resolve_value(function, *memzero.dest(), &value_map, &mut body, wb)
                            .ok_or("unresolved memzero destination")?;
                        let len = resolve_value(function, *memzero.len(), &value_map, &mut body, wb)
                            .ok_or("unresolved memzero length")?;
                        let zero = body.add_op(wb, Operator::I32Const { value: 0 }, &[], &[WType::I32]);
                        body.add_op(wb, Operator::MemoryFill { mem: memory }, &[dest, zero, len], &[]);
                    }
                    // Memcopy has the exact WebAssembly memory.copy semantics,
                    // including overlap-safe copying within the same memory.
                    else if let Some(memcopy) =
                        <&sonatina_ir::inst::data::Memcopy as InstDowncast>::downcast(
                            inst_set, inst_data,
                        )
                    {
                        let dest =
                            resolve_value(function, *memcopy.dest(), &value_map, &mut body, wb)
                                .ok_or("unresolved memcopy destination")?;
                        let src =
                            resolve_value(function, *memcopy.src(), &value_map, &mut body, wb)
                                .ok_or("unresolved memcopy source")?;
                        let len =
                            resolve_value(function, *memcopy.len(), &value_map, &mut body, wb)
                                .ok_or("unresolved memcopy length")?;
                        body.add_op(
                            wb,
                            Operator::MemoryCopy { dst_mem: memory, src_mem: memory },
                            &[dest, src, len],
                            &[],
                        );
                    }
                    // ExtractValue — load at field offset from base address
                    else if let Some(extract) = <&sonatina_ir::inst::data::ExtractValue as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let base = resolve_value(function, *extract.dest(), &value_map, &mut body, wb).ok_or("unresolved extract base")?;
                            let idx_val = function.dfg.value_imm(*extract.idx())
                                .map(|imm| match imm {
                                    sonatina_ir::Immediate::I8(v) => v as u32,
                                    sonatina_ir::Immediate::I32(v) => v as u32,
                                    sonatina_ir::Immediate::I64(v) => v as u32,
                                    sonatina_ir::Immediate::I256(v) => v.to_u256().low_u64() as u32,
                                    _ => 0,
                                })
                                .unwrap_or(0);
                            let result_ty = function.dfg.value_ty(result);
                            let elem_size = module.ctx.size_of_unchecked(result_ty) as u32;
                            let offset = idx_val * elem_size;
                            let mem_arg = waffle::MemoryArg { align: 8, offset, memory };
                            let loaded = body.add_op(wb, Operator::I64Load { memory: mem_arg }, &[base], &[WType::I64]);
                            value_map.insert(result, loaded);
                        }
                    }
                    // Trunc — wrap i64 to i32 if needed
                    else if let Some(conv) = <&sonatina_ir::inst::cast::I32ToF32 as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let from = resolve_value(function, *conv.from(), &value_map, &mut body, wb).ok_or("unresolved i32_to_f32 source")?;
                            let wval = body.add_op(wb, Operator::F32ConvertI32S, &[from], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(conv) = <&sonatina_ir::inst::cast::U32ToF32 as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let from = resolve_value(function, *conv.from(), &value_map, &mut body, wb).ok_or("unresolved u32_to_f32 source")?;
                            let wval = body.add_op(wb, Operator::F32ConvertI32U, &[from], &[WType::F32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(conv) = <&sonatina_ir::inst::cast::F32ToI32 as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let from = resolve_value(function, *conv.from(), &value_map, &mut body, wb).ok_or("unresolved f32_to_i32 source")?;
                            let wval = body.add_op(wb, Operator::I32TruncSatF32S, &[from], &[WType::I32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(conv) = <&sonatina_ir::inst::cast::F32ToU32 as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let from = resolve_value(function, *conv.from(), &value_map, &mut body, wb).ok_or("unresolved f32_to_u32 source")?;
                            let wval = body.add_op(wb, Operator::I32TruncSatF32U, &[from], &[WType::I32]);
                            value_map.insert(result, wval);
                        }
                    }
                    // Representation-preserving scalar reinterpretation. Wasm
                    // exposes the two 32-bit directions directly; integer
                    // signedness is not represented in its i32 carrier.
                    else if let Some(bitcast) = <&sonatina_ir::inst::cast::Bitcast as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let from = resolve_value(function, *bitcast.from(), &value_map, &mut body, wb)
                                .ok_or("unresolved bitcast source")?;
                            let from_ty = function.dfg.value_ty(*bitcast.from());
                            let to_ty = *bitcast.ty();
                            let (op, result_ty) = match (from_ty, to_ty) {
                                (Type::I32, Type::F32) => (Operator::F32ReinterpretI32, WType::F32),
                                (Type::F32, Type::I32) => (Operator::I32ReinterpretF32, WType::I32),
                                _ if from_ty == to_ty => {
                                    value_map.insert(result, from);
                                    continue;
                                }
                                _ => return Err(format!(
                                    "unsupported wasm bitcast `{from_ty:?}` -> `{to_ty:?}`"
                                )),
                            };
                            let value = body.add_op(wb, op, &[from], &[result_ty]);
                            value_map.insert(result, value);
                        }
                    }
                    else if let Some(trunc) = <&sonatina_ir::inst::cast::Trunc as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let val = resolve_value(function, *trunc.from(), &value_map, &mut body, wb).ok_or("unresolved trunc")?;
                            let from_ty = function.dfg.value_ty(*trunc.from());
                            let to_ty = *trunc.ty();
                            let narrowed = if from_ty == Type::I64
                                && matches!(to_ty, Type::I32 | Type::I16 | Type::I8 | Type::I1)
                            {
                                body.add_op(wb, Operator::I32WrapI64, &[val], &[WType::I32])
                            } else if matches!(from_ty, Type::I32 | Type::I16 | Type::I8 | Type::I1)
                                && matches!(to_ty, Type::I16 | Type::I8 | Type::I1)
                            {
                                val
                            } else {
                                return Err(format!(
                                    "unsupported wasm trunc `{from_ty:?}` -> `{to_ty:?}`"
                                ));
                            };
                            let narrowed = match to_ty {
                                Type::I16 | Type::I8 | Type::I1 => {
                                    let mask = match to_ty {
                                        Type::I16 => 0xffff,
                                        Type::I8 => 0xff,
                                        Type::I1 => 1,
                                        _ => unreachable!(),
                                    };
                                    let mask = body.add_op(
                                        wb,
                                        Operator::I32Const { value: mask },
                                        &[],
                                        &[WType::I32],
                                    );
                                    body.add_op(wb, Operator::I32And, &[narrowed, mask], &[WType::I32])
                                }
                                Type::I32 => narrowed,
                                _ => unreachable!(),
                            };
                            value_map.insert(result, narrowed);
                        }
                    }
                    else if let Some(ext) = <&sonatina_ir::inst::cast::Zext as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let val = resolve_value(function, *ext.from(), &value_map, &mut body, wb).ok_or("unresolved zext")?;
                            let from_ty = function.dfg.value_ty(*ext.from());
                            let to_ty = *ext.ty();
                            let normalized = match from_ty {
                                Type::I1 | Type::I8 | Type::I16 => {
                                    let mask = match from_ty {
                                        Type::I1 => 1,
                                        Type::I8 => 0xff,
                                        Type::I16 => 0xffff,
                                        _ => unreachable!(),
                                    };
                                    let mask = body.add_op(
                                        wb,
                                        Operator::I32Const { value: mask },
                                        &[],
                                        &[WType::I32],
                                    );
                                    body.add_op(wb, Operator::I32And, &[val, mask], &[WType::I32])
                                }
                                Type::I32 => val,
                                _ => return Err(format!("unsupported wasm zext source `{from_ty:?}`")),
                            };
                            let extended = match to_ty {
                                Type::I8 | Type::I16 | Type::I32 => normalized,
                                Type::I64 => body.add_op(
                                    wb,
                                    Operator::I64ExtendI32U,
                                    &[normalized],
                                    &[WType::I64],
                                ),
                                _ => return Err(format!("unsupported wasm zext target `{to_ty:?}`")),
                            };
                            value_map.insert(result, extended);
                        }
                    }
                    else if let Some(ext) = <&sonatina_ir::inst::cast::Sext as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let val = resolve_value(function, *ext.from(), &value_map, &mut body, wb).ok_or("unresolved sext")?;
                            let from_ty = function.dfg.value_ty(*ext.from());
                            let to_ty = *ext.ty();
                            let normalized = match from_ty {
                                Type::I1 => {
                                    let amount = body.add_op(
                                        wb,
                                        Operator::I32Const { value: 31 },
                                        &[],
                                        &[WType::I32],
                                    );
                                    let shifted = body.add_op(
                                        wb,
                                        Operator::I32Shl,
                                        &[val, amount],
                                        &[WType::I32],
                                    );
                                    body.add_op(
                                        wb,
                                        Operator::I32ShrS,
                                        &[shifted, amount],
                                        &[WType::I32],
                                    )
                                }
                                Type::I8 => body.add_op(wb, Operator::I32Extend8S, &[val], &[WType::I32]),
                                Type::I16 => body.add_op(wb, Operator::I32Extend16S, &[val], &[WType::I32]),
                                Type::I32 => val,
                                _ => return Err(format!("unsupported wasm sext source `{from_ty:?}`")),
                            };
                            let extended = match to_ty {
                                Type::I8 | Type::I16 | Type::I32 => normalized,
                                Type::I64 => body.add_op(
                                    wb,
                                    Operator::I64ExtendI32S,
                                    &[normalized],
                                    &[WType::I64],
                                ),
                                _ => return Err(format!("unsupported wasm sext target `{to_ty:?}`")),
                            };
                            value_map.insert(result, extended);
                        }
                    }
                    // EvmRevert/EvmStop → unreachable
                    else if <&sonatina_ir::inst::evm::EvmRevert as InstDowncast>::downcast(inst_set, inst_data).is_some()
                        || <&sonatina_ir::inst::evm::EvmStop as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                        body.set_terminator(wb, Terminator::Unreachable);
                    }
                    // IsZero
                    else if let Some(is_zero) = <&sonatina_ir::inst::cmp::IsZero as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let val = resolve_value(function, *is_zero.lhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            // eqz width must match the operand's WASM type. Select it via
                            // `sonatina_to_waffle_type` — the same mapping that declared the
                            // value — so it is validation-consistent by construction:
                            // I1/I8/I16/I32 -> i32; I64 and `Compound` handles -> i64. (A raw
                            // `Type` match would mishandle `Compound(_)`, whose handle is i64,
                            // e.g. an IsZero null-check on a ref, which the old hardcoded
                            // I64Eqz got accidentally right.)
                            let eqz = match sonatina_to_waffle_type(function.dfg.value_ty(*is_zero.lhs())) {
                                Some(WType::I32) => Operator::I32Eqz,
                                Some(WType::I64) => Operator::I64Eqz,
                                other => return Err(format!("unsupported wasm iszero operand `{other:?}`")),
                            };
                            let wval = body.add_op(wb, eqz, &[val], &[WType::I32]);
                            value_map.insert(result, wval);
                        }
                    }
                    else if let Some(get_fn) = <&sonatina_ir::inst::data::GetFunctionPtr as InstDowncast>::downcast(inst_set, inst_data) {
                        let result = function
                            .dfg
                            .inst_result(inst_id)
                            .ok_or("get_function_ptr has no result")?;
                        let slot = target_slots.get(get_fn.func()).copied().ok_or_else(|| {
                            format!(
                                "wasm translation: function %{} has no table slot",
                                get_fn.func().as_u32()
                            )
                        })?;
                        let value = body.add_op(
                            wb,
                            Operator::I32Const { value: slot },
                            &[],
                            &[WType::I32],
                        );
                        value_map.insert(result, value);
                    }
                    else if let Some(call) = <&sonatina_ir::inst::control_flow::CallIndirect as InstDowncast>::downcast(inst_set, inst_data) {
                        let table = table.ok_or(
                            "wasm translation: call_indirect has no function table",
                        )?;
                        let sig_index = indirect_signatures
                            .get(call.signature())
                            .copied()
                            .ok_or("wasm translation: call_indirect signature was not declared")?;
                        let mut args = Vec::with_capacity(call.args().len() + 1);
                        for arg in call.args() {
                            args.push(
                                resolve_value(function, *arg, &value_map, &mut body, wb)
                                    .ok_or("wasm translation: unresolved call_indirect argument")?,
                            );
                        }
                        // WAFFLE follows Wasm stack order: parameters, then table index.
                        args.push(
                            resolve_value(function, *call.callee(), &value_map, &mut body, wb)
                                .ok_or("wasm translation: unresolved call_indirect callee")?,
                        );
                        let op = Operator::CallIndirect {
                            sig_index,
                            table_index: table,
                        };
                        let results = function.dfg.inst_results(inst_id);
                        if results.is_empty() {
                            body.add_op(wb, op, &args, &[]);
                        } else {
                            let result_tys: Vec<WType> = results
                                .iter()
                                .map(|result| result_waffle_type(function, *result))
                                .collect();
                            let physical_tys: Vec<WType> =
                                result_tys.iter().copied().rev().collect();
                            let call_value = body.add_op(wb, op, &args, &physical_tys);
                            if results.len() == 1 {
                                value_map.insert(results[0], call_value);
                            } else {
                                let result_count = results.len();
                                for (index, (result, ty)) in
                                    results.iter().zip(result_tys).enumerate()
                                {
                                    let picked = body.add_value(ValueDef::PickOutput(
                                        call_value,
                                        (result_count - 1 - index) as u32,
                                        ty,
                                    ));
                                    body.append_to_block(wb, picked);
                                    value_map.insert(*result, picked);
                                }
                            }
                        }
                    }
                    // Call — direct call to another translated function.
                    else if let Some(call) = <&sonatina_ir::inst::control_flow::Call as InstDowncast>::downcast(inst_set, inst_data) {
                        let callee = *call.callee();
                        let wfunc = func_map.get(&callee).copied().ok_or_else(|| {
                            "wasm translation: call to a callee that was neither translated as a \
                             defined function nor emitted as an import (e.g. an intrinsic such as \
                             addmod/mulmod, still handled by the op matrix)"
                                .to_string()
                        })?;
                        let args: Vec<waffle::Value> = call
                            .args()
                            .iter()
                            .filter_map(|v| resolve_value(function, *v, &value_map, &mut body, wb))
                            .collect();
                        let op = Operator::Call { function_index: wfunc };
                        let results = function.dfg.inst_results(inst_id);
                        if results.is_empty() {
                            body.add_op(wb, op, &args, &[]);
                        } else {
                            let result_tys: Vec<WType> = results
                                .iter()
                                .map(|result| result_waffle_type(function, *result))
                                .collect();
                            // WAFFLE 0.2 stores a multi-value operator's stack
                            // results into its local vector in forward order.
                            // Because `local.set` pops the stack, that reverses
                            // the logical result order. Describe the physical
                            // local order here, then pick it in reverse, so the
                            // Sonatina result slots retain the callee signature's
                            // order (including when adjacent result types differ).
                            let physical_tys: Vec<WType> =
                                result_tys.iter().copied().rev().collect();
                            let call_value = body.add_op(wb, op, &args, &physical_tys);
                            if results.len() == 1 {
                                value_map.insert(results[0], call_value);
                            } else {
                                let result_count = results.len();
                                for (index, (result, ty)) in
                                    results.iter().zip(result_tys).enumerate()
                                {
                                    let picked = body.add_value(ValueDef::PickOutput(
                                        call_value,
                                        (result_count - 1 - index) as u32,
                                        ty,
                                    ));
                                    body.append_to_block(wb, picked);
                                    value_map.insert(*result, picked);
                                }
                            }
                        }
                    }
                    // Fail closed: an unhandled instruction must be an error, never
                    // a silent drop (which would miscompile). Real op coverage is R2.
                    else {
                        return Err(format!(
                            "wasm translation: unsupported instruction `{}`",
                            inst_data.as_text()
                        ));
                    }
                }

                // If no terminator was set, add an implicit return
                if matches!(body.blocks[wb].terminator, Terminator::None) {
                    body.set_terminator(wb, Terminator::Return { values: vec![] });
                }
            }

            // WAFFLE's stackifier may inline a cross-block SSA definition at
            // each use. That is only semantics-preserving for pure values.
            // Sonatina values can depend on memory (for example `1 / m[pivot]`)
            // and must retain their definition-time snapshot even if a loop
            // mutates that memory before later uses. Make every cross-block
            // use explicit before structurization; redundant parameters are a
            // later optimization concern, while reloading mutable memory is a
            // miscompile.
            body.convert_to_max_ssa(None);

            Ok::<(), String>(())
        })
        .ok_or_else(|| "function has no body".to_string())??;

    Ok(body)
}

fn resolve_value(
    function: &Function,
    value_id: ValueId,
    value_map: &HashMap<ValueId, waffle::Value>,
    body: &mut FunctionBody,
    block: waffle::Block,
) -> Option<waffle::Value> {
    if let Some(&wval) = value_map.get(&value_id) {
        return Some(wval);
    }

    let value = function.dfg.value(value_id);
    match value {
        Value::Immediate { imm, ty } => {
            let wval = match imm {
                Immediate::I1(b) => {
                    body.add_op(block, Operator::I32Const { value: *b as u32 }, &[], &[WType::I32])
                }
                Immediate::I8(v) => {
                    body.add_op(block, Operator::I32Const { value: *v as u32 }, &[], &[WType::I32])
                }
                Immediate::I16(v) => {
                    body.add_op(block, Operator::I32Const { value: *v as u32 }, &[], &[WType::I32])
                }
                Immediate::I32(v) => {
                    body.add_op(block, Operator::I32Const { value: *v as u32 }, &[], &[WType::I32])
                }
                Immediate::I64(v) => {
                    body.add_op(block, Operator::I64Const { value: *v as u64 }, &[], &[WType::I64])
                }
                Immediate::F32(bits) => {
                    body.add_op(block, Operator::F32Const { value: *bits }, &[], &[WType::F32])
                }
                _ => return None,
            };
            Some(wval)
        }
        _ => None,
    }
}

fn result_waffle_type(function: &Function, result: ValueId) -> WType {
    let ty = function.dfg.value_ty(result);
    sonatina_to_waffle_type_in_ctx(function.ctx(), ty).unwrap_or(WType::I64)
}

fn collect_phi_args(
    function: &Function,
    target_block: BlockId,
    source_block: BlockId,
    inst_set: &dyn InstSetBase,
    value_map: &HashMap<ValueId, waffle::Value>,
    body: &mut FunctionBody,
    wb: waffle::Block,
) -> Vec<waffle::Value> {
    let mut args = Vec::new();
    for inst_id in function.layout.iter_inst(target_block) {
        let inst_data = function.dfg.inst(inst_id);
        if let Some(phi) = <&sonatina_ir::inst::control_flow::Phi as InstDowncast>::downcast(
            inst_set, inst_data,
        ) {
            for &(value, from_block) in phi.args() {
                if from_block == source_block {
                    if let Some(wval) = resolve_value(function, value, value_map, body, wb) {
                        args.push(wval);
                    }
                    break;
                }
            }
        } else {
            break;
        }
    }
    args
}

//! Sonatina IR → WAFFLE IR translation.
//!
//! Translates Sonatina's SSA IR (phi nodes, arbitrary CFG) to WAFFLE's
//! SSA IR (block params, structured terminators). WAFFLE then handles
//! control flow recovery (Ramsey's algorithm) and WASM emission.

use std::collections::HashMap;

use waffle::{
    BlockTarget, ExportKind, Func, FuncDecl, FunctionBody, GlobalData, Module as WaffleModule,
    Operator, SignatureData, Terminator, Type as WType, ValueDef,
};

use sonatina_ir::{
    BlockId, Function, Immediate, Inst, InstDowncast, InstSetBase, Linkage, Module, Signature, Type,
    Value, ValueId,
    module::FuncRef,
};

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

pub(super) fn translate_module(
    module: &Module,
    import_modules: &HashMap<String, String>,
    canonical_arena: bool,
) -> Result<(WaffleModule<'static>, Vec<String>), String> {
    let mut wmod = WaffleModule::empty();
    let mut func_names = Vec::new();

    // Add linear memory (1 page = 64KB, growable)
    let memory = wmod.memories.push(waffle::MemoryData {
        initial_pages: 1,
        maximum_pages: Some(256),
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
    let mut pending: Vec<(FuncRef, Func, waffle::Signature, String)> = Vec::new();

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
                match sonatina_to_waffle_type(*ty) {
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
                match sonatina_to_waffle_type(*ty) {
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
    let canonical_alloc = canonical_arena
        .then(|| synthesize_canonical_arena(&mut wmod, memory, &mut func_names));

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
                .filter_map(|ty| sonatina_to_waffle_type(*ty))
                .collect();
            let results: Vec<WType> = sig
                .ret_tys()
                .iter()
                .filter_map(|ty| sonatina_to_waffle_type(*ty))
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

    // Pass 2: translate each body, now that every callee has a WAFFLE `Func`.
    for (func_ref, wfunc, wsig, name) in pending {
        let body = translate_function(
            module,
            func_ref,
            &wmod,
            wsig,
            memory,
            &func_map,
            canonical_alloc,
        )?;
        wmod.funcs[wfunc] = FuncDecl::Body(wsig, name, body);
    }

    Ok((wmod, func_names))
}

fn synthesize_canonical_arena(
    module: &mut WaffleModule<'static>,
    memory: waffle::Memory,
    func_names: &mut Vec<String>,
) -> Func {
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
    alloc
}

fn translate_function(
    module: &Module,
    func_ref: FuncRef,
    wmod: &WaffleModule,
    wsig: waffle::Signature,
    memory: waffle::Memory,
    func_map: &HashMap<FuncRef, Func>,
    canonical_alloc: Option<Func>,
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
                            let wty = sonatina_to_waffle_type(ty).unwrap_or(WType::I64);
                            let param = body.add_blockparam(wb, wty);
                            value_map.insert(result, param);
                        }
                    } else {
                        break;
                    }
                }
            }

            // Second pass: translate instructions and set terminators
            for block in function.layout.iter_block() {
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
                    // Uaddo (just add, ignore overflow for WASM)
                    else if let Some(uaddo) = <&sonatina_ir::inst::arith::Uaddo as InstDowncast>::downcast(inst_set, inst_data) {
                        let ir_results = function.dfg.inst_results(inst_id);
                        if !ir_results.is_empty() {
                            let lhs = resolve_value(function, *uaddo.lhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let rhs = resolve_value(function, *uaddo.rhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let wval = body.add_op(wb, Operator::I64Add, &[lhs, rhs], &[WType::I64]);
                            value_map.insert(ir_results[0], wval);
                            if ir_results.len() >= 2 {
                                let zero = body.add_op(wb, Operator::I32Const { value: 0 }, &[], &[WType::I32]);
                                value_map.insert(ir_results[1], zero);
                            }
                        }
                    }
                    // Umulo (just mul, ignore overflow)
                    else if let Some(umulo) = <&sonatina_ir::inst::arith::Umulo as InstDowncast>::downcast(inst_set, inst_data) {
                        let ir_results = function.dfg.inst_results(inst_id);
                        if !ir_results.is_empty() {
                            let lhs = resolve_value(function, *umulo.lhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let rhs = resolve_value(function, *umulo.rhs(), &value_map, &mut body, wb).ok_or("unresolved")?;
                            let wval = body.add_op(wb, Operator::I64Mul, &[lhs, rhs], &[WType::I64]);
                            value_map.insert(ir_results[0], wval);
                            if ir_results.len() >= 2 {
                                let zero = body.add_op(wb, Operator::I32Const { value: 0 }, &[], &[WType::I32]);
                                value_map.insert(ir_results[1], zero);
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
                        let canonical_alloc = canonical_alloc.ok_or(
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
                                function_index: canonical_alloc,
                            },
                            &[size, align],
                            &[WType::I32],
                        );
                        value_map.insert(result, address);
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
                            let wval = body.add_op(wb, Operator::I64Eqz, &[val], &[WType::I32]);
                            value_map.insert(result, wval);
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
    sonatina_to_waffle_type(ty).unwrap_or(WType::I64)
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

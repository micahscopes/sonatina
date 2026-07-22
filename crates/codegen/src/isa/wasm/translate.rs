//! Sonatina IR → WAFFLE IR translation.
//!
//! Translates Sonatina's SSA IR (phi nodes, arbitrary CFG) to WAFFLE's
//! SSA IR (block params, structured terminators). WAFFLE then handles
//! control flow recovery (Ramsey's algorithm) and WASM emission.

use std::collections::HashMap;

use waffle::{
    ExportKind, Func, FuncDecl, FunctionBody, Module as WaffleModule, Operator, SignatureData,
    Terminator, Type as WType, ValueDef,
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

pub(super) fn translate_module(
    module: &Module,
    import_modules: &HashMap<String, String>,
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

        wmod.exports.push(waffle::Export {
            name: name.clone(),
            kind: ExportKind::Func(wfunc),
        });

        func_map.insert(func_ref, wfunc);
        func_names.push(name.clone());
        pending.push((func_ref, wfunc, wsig, name));
    }

    // Pass 2: translate each body, now that every callee has a WAFFLE `Func`.
    for (func_ref, wfunc, wsig, name) in pending {
        let body = translate_function(module, func_ref, &wmod, wsig, memory, &func_map)?;
        wmod.funcs[wfunc] = FuncDecl::Body(wsig, name, body);
    }

    Ok((wmod, func_names))
}

fn translate_function(
    module: &Module,
    func_ref: FuncRef,
    wmod: &WaffleModule,
    wsig: waffle::Signature,
    memory: waffle::Memory,
    func_map: &HashMap<FuncRef, Func>,
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
                    // Alloca — allocate a local
                    else if <&sonatina_ir::inst::data::Alloca as InstDowncast>::downcast(inst_set, inst_data).is_some() {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let zero = body.add_op(wb, Operator::I64Const { value: 0 }, &[], &[WType::I64]);
                            value_map.insert(result, zero);
                        }
                    }
                    // Mstore — store to local (just map the value)
                    else if let Some(mstore) = <&sonatina_ir::inst::data::Mstore as InstDowncast>::downcast(inst_set, inst_data) {
                        let val = resolve_value(function, *mstore.value(), &value_map, &mut body, wb);
                        let addr = resolve_value(function, *mstore.addr(), &value_map, &mut body, wb);
                        // In WASM, mstore updates the "local" that addr represents
                        // For now, just track the value
                    }
                    // Mload — load from local
                    else if let Some(mload) = <&sonatina_ir::inst::data::Mload as InstDowncast>::downcast(inst_set, inst_data) {
                        if let Some(result) = function.dfg.inst_result(inst_id) {
                            let addr = resolve_value(function, *mload.addr(), &value_map, &mut body, wb);
                            if let Some(v) = addr {
                                value_map.insert(result, v);
                            }
                        }
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
                            let to_ty = *trunc.ty();
                            if matches!(to_ty, Type::I32 | Type::I16 | Type::I8 | Type::I1) {
                                let wrapped = body.add_op(wb, Operator::I32WrapI64, &[val], &[WType::I32]);
                                value_map.insert(result, wrapped);
                            } else {
                                value_map.insert(result, val);
                            }
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

use std::collections::HashMap;

use cranelift_codegen::ir::{self as clif, InstBuilder, instructions::BlockArg};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
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

    let funcs = module.funcs();

    for &func_ref in &funcs {
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
        let translated = module.func_store.try_view(func_ref, |function| {
            if function.layout.entry_block().is_none() {
                return Ok(());
            }
            let func_id = func_id_map[&func_ref];
            translate_function(module, function, func_ref, func_id, &func_id_map, jit)
        });
        if let Some(result) = translated {
            let name = module.ctx.func_sig(func_ref, |sig| sig.name().to_string());
            result.map_err(|e| format!("error translating function {name}: {e}"))?;
        }
    }

    Ok(func_map)
}

fn sonatina_sig_to_clif(sig: &Signature, jit: &JITModule) -> clif::Signature {
    let mut clif_sig = jit.make_signature();
    for &arg_ty in sig.args() {
        if let Some(clif_ty) = sonatina_type_to_clif(arg_ty) {
            clif_sig.params.push(clif::AbiParam::new(clif_ty));
        }
    }
    for &ret_ty in sig.ret_tys() {
        if let Some(clif_ty) = sonatina_type_to_clif(ret_ty) {
            clif_sig.returns.push(clif::AbiParam::new(clif_ty));
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
    jit: &mut JITModule,
) -> Result<(), String> {
    let mut ctx = jit.make_context();
    let sig = module.ctx.func_sig(func_ref, |sig| sonatina_sig_to_clif(sig, jit));
    ctx.func.signature = sig;

    let mut builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    let mut block_map: HashMap<BlockId, clif::Block> = HashMap::new();
    let mut value_map: HashMap<ValueId, clif::Value> = HashMap::new();

    for block in function.layout.iter_block() {
        let clif_block = builder.create_block();
        block_map.insert(block, clif_block);
    }

    let entry = function.layout.entry_block().ok_or("no entry block")?;
    let clif_entry = block_map[&entry];
    builder.append_block_params_for_function_params(clif_entry);
    builder.switch_to_block(clif_entry);
    builder.seal_block(clif_entry);

    for (idx, &arg_value) in function.arg_values.iter().enumerate() {
        let param = builder.block_params(clif_entry)[idx];
        value_map.insert(arg_value, param);
    }

    let inst_set = function.inst_set();

    if inst_set.has_evm_stop().is_some() {
        return Err("CraneliftBackend requires a native (non-EVM) instruction set".to_string());
    }

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
                let result_val = builder.ins().iadd(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(sub) = <&sonatina_ir::inst::arith::Sub as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *sub.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *sub.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().isub(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(mul) = <&sonatina_ir::inst::arith::Mul as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *mul.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *mul.rhs(), &value_map, &mut builder)?;
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
                let result_val = builder.ins().band(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(or) = <&sonatina_ir::inst::logic::Or as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *or.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *or.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().bor(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(xor) = <&sonatina_ir::inst::logic::Xor as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *xor.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *xor.rhs(), &value_map, &mut builder)?;
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
                translate_icmp(IntCC::UnsignedLessThan, *lt.lhs(), *lt.rhs(), inst_id, function, &mut value_map, &mut builder)?;
            } else if let Some(gt) = <&sonatina_ir::inst::cmp::Gt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedGreaterThan, *gt.lhs(), *gt.rhs(), inst_id, function, &mut value_map, &mut builder)?;
            } else if let Some(le) = <&sonatina_ir::inst::cmp::Le as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedLessThanOrEqual, *le.lhs(), *le.rhs(), inst_id, function, &mut value_map, &mut builder)?;
            } else if let Some(ge) = <&sonatina_ir::inst::cmp::Ge as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedGreaterThanOrEqual, *ge.lhs(), *ge.rhs(), inst_id, function, &mut value_map, &mut builder)?;
            } else if let Some(slt) = <&sonatina_ir::inst::cmp::Slt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedLessThan, *slt.lhs(), *slt.rhs(), inst_id, function, &mut value_map, &mut builder)?;
            } else if let Some(sgt) = <&sonatina_ir::inst::cmp::Sgt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedGreaterThan, *sgt.lhs(), *sgt.rhs(), inst_id, function, &mut value_map, &mut builder)?;
            } else if let Some(eq) = <&sonatina_ir::inst::cmp::Eq as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::Equal, *eq.lhs(), *eq.rhs(), inst_id, function, &mut value_map, &mut builder)?;
            } else if let Some(ne) = <&sonatina_ir::inst::cmp::Ne as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::NotEqual, *ne.lhs(), *ne.rhs(), inst_id, function, &mut value_map, &mut builder)?;
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
                let val = resolve_value(function, *trunc.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*trunc.ty())?;
                let result_val = builder.ins().ireduce(to_ty, val);
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
                let args: Vec<clif::Value> = ret.args().as_slice()
                    .iter()
                    .filter_map(|v| resolve_value(function, *v, &value_map, &mut builder).ok())
                    .collect();
                builder.ins().return_(&args);
            } else if let Some(call) = <&sonatina_ir::inst::control_flow::Call as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let callee = *call.callee();
                let clif_func_id = func_id_map.get(&callee)
                    .ok_or_else(|| format!("unknown callee {:?}", callee))?;
                let clif_func_ref = jit.declare_func_in_func(*clif_func_id, builder.func);
                let args: Result<Vec<_>, _> = call.args()
                    .iter()
                    .map(|v| resolve_value(function, *v, &value_map, &mut builder))
                    .collect();
                let clif_call = builder.ins().call(clif_func_ref, &args?);
                let results = builder.inst_results(clif_call).to_vec();
                let ir_results = function.dfg.inst_results(inst_id);
                for (ir_result, clif_result) in ir_results.iter().zip(results.iter()) {
                    value_map.insert(*ir_result, *clif_result);
                }
            } else if <&sonatina_ir::inst::control_flow::Unreachable as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                builder.ins().trap(cranelift_codegen::ir::TrapCode::user(0).unwrap());
            } else {
                return Err(format!(
                    "unsupported instruction for CraneliftBackend: {:?}",
                    inst_data.kind()
                ));
            }
        }
    }

    for block in function.layout.iter_block() {
        if block != entry {
            builder.seal_block(block_map[&block]);
        }
    }

    builder.finalize();

    jit.define_function(func_id, &mut ctx)
        .map_err(|e| format!("cranelift define_function failed: {e}"))?;

    Ok(())
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

    let value = function.dfg.value(value_id);
    match value {
        Value::Immediate { imm, ty } => {
            let clif_ty = sonatina_type_to_clif_or_err(*ty)?;
            let i64_val = imm_to_i64(imm)?;
            let val = builder.ins().iconst(clif_ty, i64_val);
            Ok(val)
        }
        _ => Err(format!("unresolved value v{}", value_id.0)),
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

fn translate_icmp(
    cc: IntCC,
    lhs: ValueId,
    rhs: ValueId,
    inst_id: sonatina_ir::inst::InstId,
    function: &Function,
    value_map: &mut HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(), String> {
    let lhs_val = resolve_value(function, lhs, value_map, builder)?;
    let rhs_val = resolve_value(function, rhs, value_map, builder)?;
    let result_val = builder.ins().icmp(cc, lhs_val, rhs_val);
    if let Some(result) = function.dfg.inst_result(inst_id) {
        value_map.insert(result, result_val);
    }
    Ok(())
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

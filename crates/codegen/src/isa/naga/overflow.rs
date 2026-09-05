//! Two-result arithmetic in the shader's unsigned word carrier.
//! Semantic widths remain explicit; no wider shader feature is required.

use super::{WordKind, emit_expr};
use crate::isa::overflow::OverflowArithmetic;
use naga::{BinaryOperator as B, Expression as E, Handle};

pub(super) fn emit(
    func: &mut naga::Function,
    block: &mut naga::Block,
    word: WordKind,
    ty: sonatina_ir::Type,
    op: OverflowArithmetic,
    signed: bool,
    lhs: Handle<E>,
    rhs: Handle<E>,
) -> Result<(Handle<E>, Handle<E>), String> {
    let bits = match ty {
        sonatina_ir::Type::I1 => 1,
        sonatina_ir::Type::I32 => 32,
        sonatina_ir::Type::I64 if word == WordKind::I64 => 64,
        _ => {
            return Err(format!(
                "unsupported Naga overflow arithmetic type {ty:?} for {word:?}"
            ));
        }
    };
    let literal = |func: &mut naga::Function, value: u64| {
        let value = match word {
            WordKind::U32 => naga::Literal::U32(value as u32),
            WordKind::I64 => naga::Literal::U64(value),
        };
        func.expressions
            .append(E::Literal(value), naga::Span::UNDEFINED)
    };
    let zero = literal(func, 0);
    let one = literal(func, 1);
    let mask = literal(func, u64::MAX >> (64 - bits));
    let sign = literal(func, 1u64 << (bits - 1));
    macro_rules! binary {
        ($op:ident, $lhs:expr, $rhs:expr) => {
            emit_expr(
                func,
                block,
                E::Binary {
                    op: B::$op,
                    left: $lhs,
                    right: $rhs,
                },
            )
        };
    }
    macro_rules! select {
        ($condition:expr, $yes:expr, $no:expr) => {
            emit_expr(
                func,
                block,
                E::Select {
                    condition: $condition,
                    accept: $yes,
                    reject: $no,
                },
            )
        };
    }
    let mut unsigned = |value| {
        let value = if bits == 1 {
            select!(value, one, zero)
        } else if word == WordKind::I64 {
            emit_expr(
                func,
                block,
                E::As {
                    expr: value,
                    kind: naga::ScalarKind::Uint,
                    convert: None,
                },
            )
        } else {
            value
        };
        binary!(And, value, mask)
    };
    let lhs = unsigned(lhs);
    let rhs = unsigned(rhs);
    let raw = match op {
        OverflowArithmetic::Add => binary!(Add, lhs, rhs),
        OverflowArithmetic::Sub => binary!(Subtract, lhs, rhs),
        OverflowArithmetic::Mul => binary!(Multiply, lhs, rhs),
    };
    let result = binary!(And, raw, mask);
    let flag = match (op, signed) {
        (OverflowArithmetic::Add, false) => {
            let room = binary!(Subtract, mask, rhs);
            binary!(Greater, lhs, room)
        }
        (OverflowArithmetic::Sub, false) => binary!(Less, lhs, rhs),
        (OverflowArithmetic::Add | OverflowArithmetic::Sub, true) => {
            let lhs_changed = binary!(ExclusiveOr, lhs, result);
            let other = match op {
                OverflowArithmetic::Add => binary!(ExclusiveOr, rhs, result),
                _ => binary!(ExclusiveOr, lhs, rhs),
            };
            let changed = binary!(And, lhs_changed, other);
            let changed_sign = binary!(And, changed, sign);
            binary!(NotEqual, changed_sign, zero)
        }
        (OverflowArithmetic::Mul, _) => {
            let (a, b, limit) = if signed {
                let lhs_sign = binary!(And, lhs, sign);
                let rhs_sign = binary!(And, rhs, sign);
                let lhs_negative = binary!(NotEqual, lhs_sign, zero);
                let rhs_negative = binary!(NotEqual, rhs_sign, zero);
                let negative = binary!(NotEqual, lhs_negative, rhs_negative);
                let lhs_negated = binary!(Subtract, zero, lhs);
                let lhs_negated = binary!(And, lhs_negated, mask);
                let rhs_negated = binary!(Subtract, zero, rhs);
                let rhs_negated = binary!(And, rhs_negated, mask);
                let a = select!(lhs_negative, lhs_negated, lhs);
                let b = select!(rhs_negative, rhs_negated, rhs);
                let positive_limit = binary!(Subtract, sign, one);
                (a, b, select!(negative, sign, positive_limit))
            } else {
                (lhs, rhs, mask)
            };
            let nonzero = binary!(NotEqual, a, zero);
            // Both select arms may be evaluated. Make the divisor safe before
            // division rather than relying on a surrounding boolean guard.
            let divisor = select!(nonzero, a, one);
            let room = binary!(Divide, limit, divisor);
            let beyond = binary!(Greater, b, room);
            binary!(LogicalAnd, nonzero, beyond)
        }
    };
    let result = if bits == 1 {
        binary!(NotEqual, result, zero)
    } else if word == WordKind::I64 {
        emit_expr(
            func,
            block,
            E::As {
                expr: result,
                kind: naga::ScalarKind::Sint,
                convert: None,
            },
        )
    } else {
        result
    };
    Ok((result, flag))
}

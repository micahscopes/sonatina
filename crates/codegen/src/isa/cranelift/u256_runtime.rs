//! Runtime intrinsics for u256 (256-bit integer) arithmetic.
//!
//! These functions are registered as imported symbols in the Cranelift JIT,
//! enabling Sonatina IR with I256 operations to execute natively.
//!
//! Representation: u256 as `[u64; 4]` in little-endian limb order
//! (limb[0] is least significant).

/// u256 addition: result = a + b (wrapping)
#[unsafe(no_mangle)]
pub extern "C" fn __u256_add(a: *const [u64; 4], b: *const [u64; 4], result: *mut [u64; 4]) {
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    let r = unsafe { &mut *result };
    let mut carry = 0u64;
    for i in 0..4 {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        r[i] = s2;
        carry = (c1 as u64) + (c2 as u64);
    }
}

/// u256 subtraction: result = a - b (wrapping)
#[unsafe(no_mangle)]
pub extern "C" fn __u256_sub(a: *const [u64; 4], b: *const [u64; 4], result: *mut [u64; 4]) {
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    let r = unsafe { &mut *result };
    let mut borrow = 0u64;
    for i in 0..4 {
        let (s1, c1) = a[i].overflowing_sub(b[i]);
        let (s2, c2) = s1.overflowing_sub(borrow);
        r[i] = s2;
        borrow = (c1 as u64) + (c2 as u64);
    }
}

/// u256 equality: returns 1 if a == b, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn __u256_eq(a: *const [u64; 4], b: *const [u64; 4]) -> u64 {
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    (a == b) as u64
}

/// u256 less-than (unsigned): returns 1 if a < b, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn __u256_lt(a: *const [u64; 4], b: *const [u64; 4]) -> u64 {
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    for i in (0..4).rev() {
        if a[i] < b[i] { return 1; }
        if a[i] > b[i] { return 0; }
    }
    0
}

/// u256 is-zero: returns 1 if a == 0, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn __u256_is_zero(a: *const [u64; 4]) -> u64 {
    let a = unsafe { &*a };
    (a[0] | a[1] | a[2] | a[3] == 0) as u64
}

/// u256 multiplication: result = a * b (wrapping, lower 256 bits)
#[unsafe(no_mangle)]
pub extern "C" fn __u256_mul(a: *const [u64; 4], b: *const [u64; 4], result: *mut [u64; 4]) {
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    let r = unsafe { &mut *result };
    *r = [0u64; 4];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            if i + j >= 4 { break; }
            let prod = (a[i] as u128) * (b[j] as u128) + (r[i + j] as u128) + carry;
            r[i + j] = prod as u64;
            carry = prod >> 64;
        }
    }
}

fn from_limbs(limbs: [u64; 4]) -> sonatina_ir::U256 {
    let mut value = sonatina_ir::U256::zero();
    value.0 = limbs;
    value
}

/// u256 addmod with an untruncated sum; zero modulus returns zero.
#[unsafe(no_mangle)]
pub extern "C" fn __u256_addmod(
    a: *const [u64; 4],
    b: *const [u64; 4],
    m: *const [u64; 4],
    result: *mut [u64; 4],
) {
    use sonatina_ir::U256;
    // Copy inputs before writing the destination, which may alias an input.
    let a = from_limbs(unsafe { *a });
    let b = from_limbs(unsafe { *b });
    let m = from_limbs(unsafe { *m });
    let value = if m.is_zero() {
        [0; 4]
    } else {
        let wide_m = m.full_mul(U256::one());
        let remainder = (a.full_mul(U256::one()) + b.full_mul(U256::one())) % wide_m;
        remainder.0[..4].try_into().unwrap()
    };
    unsafe { *result = value };
}

/// u256 mulmod with a full 512-bit product; zero modulus returns zero.
#[unsafe(no_mangle)]
pub extern "C" fn __u256_mulmod(
    a: *const [u64; 4],
    b: *const [u64; 4],
    m: *const [u64; 4],
    result: *mut [u64; 4],
) {
    use sonatina_ir::U256;
    let a = from_limbs(unsafe { *a });
    let b = from_limbs(unsafe { *b });
    let m = from_limbs(unsafe { *m });
    let value = if m.is_zero() {
        [0; 4]
    } else {
        let remainder = a.full_mul(b) % m.full_mul(U256::one());
        remainder.0[..4].try_into().unwrap()
    };
    unsafe { *result = value };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modular_arithmetic_matches_independent_ruint_oracle() {
        use revm::primitives::U256;
        let mut seed = 0x6a09e667f3bcc909u64;
        let mut next = || {
            std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            })
        };
        for index in 0..512 {
            let a = next();
            let b = next();
            let m = match index % 8 {
                0 => [0; 4],
                1 => [1, 0, 0, 0],
                2 => [13, 0, 0, 0],
                _ => next(),
            };
            let aa = U256::from_limbs(a);
            let bb = U256::from_limbs(b);
            let mm = U256::from_limbs(m);
            let mut out = [0; 4];
            __u256_addmod(&a, &b, &m, &mut out);
            assert_eq!(out, aa.add_mod(bb, mm).into_limbs(), "add case {index}");
            __u256_mulmod(&a, &b, &m, &mut out);
            assert_eq!(out, aa.mul_mod(bb, mm).into_limbs(), "mul case {index}");
            let mut aliased = a;
            let ptr = &mut aliased as *mut [u64; 4];
            __u256_mulmod(ptr, &b, &m, ptr);
            assert_eq!(aliased, out, "alias case {index}");
        }
    }

    #[test]
    fn modular_full_width_boundaries() {
        let max = [u64::MAX; 4];
        let zero = [0; 4];
        let one = [1, 0, 0, 0];
        let two = [2, 0, 0, 0];
        let mut out = [99; 4];
        for modulus in [zero, one, max] {
            __u256_addmod(&max, &max, &modulus, &mut out);
            assert_eq!(out, zero);
            __u256_mulmod(&max, &max, &modulus, &mut out);
            assert_eq!(out, zero);
        }
        __u256_addmod(&max, &one, &max, &mut out);
        assert_eq!(out, one);
        __u256_mulmod(&max, &max, &two, &mut out);
        assert_eq!(out, one);
        let almost_max = [u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX];
        __u256_mulmod(&almost_max, &almost_max, &max, &mut out);
        assert_eq!(out, one);
        __u256_addmod(&almost_max, &almost_max, &max, &mut out);
        assert_eq!(out, [u64::MAX - 2, u64::MAX, u64::MAX, u64::MAX]);
    }

    #[test]
    fn test_u256_add() {
        let a = [3u64, 0, 0, 0];
        let b = [4u64, 0, 0, 0];
        let mut r = [0u64; 4];
        __u256_add(&a, &b, &mut r);
        assert_eq!(r, [7, 0, 0, 0]);
    }

    #[test]
    fn test_u256_mul() {
        let a = [7u64, 0, 0, 0];
        let b = [3u64, 0, 0, 0];
        let mut r = [0u64; 4];
        __u256_mul(&a, &b, &mut r);
        assert_eq!(r, [21, 0, 0, 0]);
    }

    #[test]
    fn test_u256_addmod_small() {
        let a = [42u64, 0, 0, 0];
        let b = [17u64, 0, 0, 0];
        let m = [100u64, 0, 0, 0];
        let mut r = [0u64; 4];
        __u256_addmod(&a, &b, &m, &mut r);
        assert_eq!(r, [59, 0, 0, 0]); // 42 + 17 = 59 < 100
    }

    #[test]
    fn test_u256_addmod_with_reduction() {
        let a = [90u64, 0, 0, 0];
        let b = [20u64, 0, 0, 0];
        let m = [100u64, 0, 0, 0];
        let mut r = [0u64; 4];
        __u256_addmod(&a, &b, &m, &mut r);
        assert_eq!(r, [10, 0, 0, 0]); // (90 + 20) % 100 = 10
    }

    #[test]
    fn test_u256_mulmod_small() {
        let a = [7u64, 0, 0, 0];
        let b = [3u64, 0, 0, 0];
        let m = [100u64, 0, 0, 0];
        let mut r = [0u64; 4];
        __u256_mulmod(&a, &b, &m, &mut r);
        assert_eq!(r, [21, 0, 0, 0]); // 7 * 3 = 21 < 100
    }

    #[test]
    fn test_u256_eq() {
        let a = [42u64, 0, 0, 0];
        let b = [42u64, 0, 0, 0];
        let c = [43u64, 0, 0, 0];
        assert_eq!(__u256_eq(&a, &b), 1);
        assert_eq!(__u256_eq(&a, &c), 0);
    }

    #[test]
    fn test_u256_pow5() {
        // 3^5 = 243 via mulmod chain
        let three = [3u64, 0, 0, 0];
        let m = [u64::MAX, u64::MAX, u64::MAX, u64::MAX]; // huge modulus
        let mut x2 = [0u64; 4];
        let mut x4 = [0u64; 4];
        let mut x5 = [0u64; 4];
        __u256_mulmod(&three, &three, &m, &mut x2);   // 9
        __u256_mulmod(&x2, &x2, &m, &mut x4);          // 81
        __u256_mulmod(&x4, &three, &m, &mut x5);       // 243
        assert_eq!(x5, [243, 0, 0, 0]);
    }
}

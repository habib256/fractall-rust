use num_complex::Complex64;
use rug::{Complex, Float};
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatExp {
    pub mantissa: f64,
    pub exponent: i32,
}

impl FloatExp {
    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            mantissa: 0.0,
            exponent: 0,
        }
    }

    #[inline(always)]
    pub fn from_f64(value: f64) -> Self {
        if value == 0.0 {
            return Self::zero();
        }
        let (mantissa, exponent) = frexp(value);
        Self { mantissa, exponent }
    }

    #[inline(always)]
    pub fn new(mantissa: f64, exponent: i32) -> Self {
        if mantissa == 0.0 {
            return Self::zero();
        }
        let (m, e) = frexp(mantissa);
        Self {
            mantissa: m,
            exponent: exponent + e,
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            exponent: self.exponent,
        }
    }

    /// Create a FloatExp from a GMP Float, preserving the extended exponent range.
    /// This allows storing values that would overflow or underflow f64.
    pub fn from_gmp(value: &Float) -> Self {
        if value.is_zero() {
            return Self::zero();
        }
        // Get exponent directly from GMP (avoids precision loss)
        let exp = value.get_exp().unwrap_or(0);
        // Get mantissa by scaling: mantissa = value * 2^(-exp)
        // For normalized floats, mantissa is in [0.5, 1.0)
        let mut mantissa_float = value.clone();
        // Scale to get mantissa in [0.5, 1.0)
        if exp != 0 {
            mantissa_float >>= exp;
        }
        let mantissa = mantissa_float.to_f64();
        Self { mantissa, exponent: exp }
    }

    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        if self.mantissa == 0.0 {
            return 0.0;
        }
        if self.exponent > 1023 {
            return f64::INFINITY.copysign(self.mantissa);
        }
        if self.exponent < -1022 {
            return 0.0;
        }
        self.mantissa * pow2i(self.exponent)
    }

    /// Normalize the mantissa so that it lies in [0.5, 1.0) and adjust the exponent.
    /// Inspired by rust-fractal-core's `reduce()` method which periodically
    /// re-normalizes the mantissa to prevent gradual precision loss during
    /// long iteration sequences.
    #[inline(always)]
    pub fn reduce(&mut self) {
        if self.mantissa == 0.0 {
            self.exponent = 0;
            return;
        }
        let (m, e) = frexp(self.mantissa);
        self.mantissa = m;
        self.exponent += e;
    }

    /// Squared value: (mantissa^2, exponent*2)
    #[inline(always)]
    pub fn sqr(self) -> Self {
        Self::new(self.mantissa * self.mantissa, self.exponent * 2)
    }
}

impl Add for FloatExp {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        if self.mantissa == 0.0 {
            return rhs;
        }
        if rhs.mantissa == 0.0 {
            return self;
        }
        let diff = self.exponent - rhs.exponent;
        if diff >= 54 {
            return self;
        }
        if diff <= -54 {
            return rhs;
        }
        if diff >= 0 {
            let scaled_rhs = rhs.mantissa * pow2i(-diff);
            Self::new(self.mantissa + scaled_rhs, self.exponent)
        } else {
            let scaled_self = self.mantissa * pow2i(diff);
            Self::new(scaled_self + rhs.mantissa, rhs.exponent)
        }
    }
}

impl Sub for FloatExp {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self + Self::new(-rhs.mantissa, rhs.exponent)
    }
}

impl Mul for FloatExp {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.mantissa * rhs.mantissa, self.exponent + rhs.exponent)
    }
}

impl Mul<f64> for FloatExp {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.mantissa * rhs, self.exponent)
    }
}

impl Mul<FloatExp> for f64 {
    type Output = FloatExp;

    #[inline(always)]
    fn mul(self, rhs: FloatExp) -> Self::Output {
        rhs * self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexExp {
    pub re: FloatExp,
    pub im: FloatExp,
}

impl ComplexExp {
    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            re: FloatExp::zero(),
            im: FloatExp::zero(),
        }
    }

    #[inline(always)]
    pub fn from_complex64(value: Complex64) -> Self {
        Self {
            re: FloatExp::from_f64(value.re),
            im: FloatExp::from_f64(value.im),
        }
    }

    /// Create a ComplexExp from a GMP Complex, preserving the extended exponent range.
    /// This allows storing values that would overflow or underflow f64.
    #[inline]
    pub fn from_gmp(value: &Complex) -> Self {
        Self {
            re: FloatExp::from_gmp(value.real()),
            im: FloatExp::from_gmp(value.imag()),
        }
    }

    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        Self { re, im }
    }

    #[inline(always)]
    pub fn mul_complex64(self, rhs: Complex64) -> Self {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        Self { re, im }
    }

    #[inline(always)]
    pub fn norm_sqr_approx(self) -> f64 {
        let re = self.re.to_f64();
        let im = self.im.to_f64();
        re * re + im * im
    }

    /// Norme² en FloatExp — ne underflow pas pour exp < -1022 (contrairement
    /// à `norm_sqr_approx` qui collapse à 0.0 pour les zooms > 1e308).
    ///
    /// Utilisé par les checks de validité BLA via `BlaTableUnified::lookup_fexp`
    /// pour éviter d'appliquer le skip max à tous les pixels en deep zoom.
    #[inline(always)]
    pub fn norm_sqr_fexp(self) -> FloatExp {
        self.re.sqr() + self.im.sqr()
    }

    #[inline(always)]
    pub fn to_complex64_approx(self) -> Complex64 {
        Complex64::new(self.re.to_f64(), self.im.to_f64())
    }

    /// Multiply with sign adjustment for Burning Ship perturbation.
    /// Used when the quadrant is stable and we can apply signed perturbation.
    /// result.re = sign_re * self.re
    /// result.im = sign_im * self.im
    #[inline(always)]
    #[allow(dead_code)]
    pub fn mul_signed(self, sign_re: f64, sign_im: f64) -> Self {
        Self {
            re: FloatExp::new(self.re.mantissa * sign_re, self.re.exponent),
            im: FloatExp::new(self.im.mantissa * sign_im, self.im.exponent),
        }
    }

    /// Normalize both components. Inspired by rust-fractal-core's `reduce()` method
    /// which periodically re-normalizes mantissas to prevent gradual precision loss
    /// during long iteration sequences (called e.g. every 250 iterations).
    #[inline(always)]
    pub fn reduce(&mut self) {
        self.re.reduce();
        self.im.reduce();
    }

}

#[inline(always)]
fn frexp(value: f64) -> (f64, i32) {
    if value == 0.0 {
        return (0.0, 0);
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 { 1.0 } else { -1.0 };
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let mant = bits & 0x000f_ffff_ffff_ffff;
    if exp == 0 {
        if mant == 0 {
            return (0.0, 0);
        }
        let mut m = mant;
        let mut e = -1022;
        while (m & (1u64 << 52)) == 0 {
            m <<= 1;
            e -= 1;
        }
        let mantissa = sign * (m as f64) / (1u64 << 53) as f64;
        (mantissa, e + 1)
    } else {
        let mantissa = sign * ((1u64 << 52 | mant) as f64) / (1u64 << 53) as f64;
        (mantissa, exp - 1022)
    }
}

#[inline(always)]
pub fn pow2i(exp: i32) -> f64 {
    if exp < -1022 {
        return 0.0;
    }
    if exp > 1023 {
        return f64::INFINITY;
    }
    f64::from_bits(((exp + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    use super::{ComplexExp, FloatExp};
    use num_complex::Complex64;

    #[test]
    fn floatexp_roundtrip() {
        let value = 1.5f64;
        let fx = FloatExp::from_f64(value);
        let back = fx.to_f64();
        let diff = (back - value).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn floatexp_add_ignores_tiny() {
        let big = FloatExp::new(1.0, 40);
        let tiny = FloatExp::new(1.0, -80);
        let sum = big + tiny;
        let diff = (sum.to_f64() - big.to_f64()).abs();
        assert!(diff < 1e-9);
    }

    #[test]
    fn complexexp_mul_matches_f64() {
        let a = ComplexExp::from_complex64(Complex64::new(1.0, 2.0));
        let b = ComplexExp::from_complex64(Complex64::new(3.0, 4.0));
        let out = a.mul(b).to_complex64_approx();
        assert!((out.re + 5.0).abs() < 1e-9);
        assert!((out.im - 10.0).abs() < 1e-9);
    }

    #[test]
    fn floatexp_reduce_normalizes() {
        let mut fx = FloatExp { mantissa: 4.0, exponent: 5 };
        fx.reduce();
        // After reduce, mantissa should be in [0.5, 1.0)
        assert!(fx.mantissa.abs() >= 0.5 && fx.mantissa.abs() < 1.0,
            "mantissa should be normalized, got {}", fx.mantissa);
        // Value should be preserved
        let value = fx.to_f64();
        assert!((value - 128.0).abs() < 1e-10, "value should be 4*2^5=128, got {}", value);
    }

    #[test]
    fn floatexp_reduce_zero() {
        let mut fx = FloatExp::zero();
        fx.reduce();
        assert_eq!(fx.mantissa, 0.0);
        assert_eq!(fx.exponent, 0);
    }

    #[test]
    fn complexexp_reduce_preserves_value() {
        let original = ComplexExp::from_complex64(Complex64::new(3.14, -2.71));
        let mut reduced = original;
        reduced.reduce();
        let orig_f64 = original.to_complex64_approx();
        let reduced_f64 = reduced.to_complex64_approx();
        assert!((orig_f64.re - reduced_f64.re).abs() < 1e-12);
        assert!((orig_f64.im - reduced_f64.im).abs() < 1e-12);
    }

    #[test]
    fn floatexp_sqr() {
        let fx = FloatExp::from_f64(3.0);
        let sq = fx.sqr();
        assert!((sq.to_f64() - 9.0).abs() < 1e-10);
    }
}

use num_complex::Complex64;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatExp {
    pub mantissa: f64,
    pub exponent: i32,
}

impl FloatExp {
    pub fn zero() -> Self {
        Self {
            mantissa: 0.0,
            exponent: 0,
        }
    }

    pub fn from_f64(value: f64) -> Self {
        if value == 0.0 {
            return Self::zero();
        }
        let (mantissa, exponent) = frexp(value);
        Self { mantissa, exponent }
    }

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

    pub fn abs(self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            exponent: self.exponent,
        }
    }

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
}

impl Add for FloatExp {
    type Output = Self;

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

    fn sub(self, rhs: Self) -> Self::Output {
        self + Self::new(-rhs.mantissa, rhs.exponent)
    }
}

impl Mul for FloatExp {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.mantissa * rhs.mantissa, self.exponent + rhs.exponent)
    }
}

impl Mul<f64> for FloatExp {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.mantissa * rhs, self.exponent)
    }
}

impl Mul<FloatExp> for f64 {
    type Output = FloatExp;

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
    pub fn zero() -> Self {
        Self {
            re: FloatExp::zero(),
            im: FloatExp::zero(),
        }
    }

    pub fn from_complex64(value: Complex64) -> Self {
        Self {
            re: FloatExp::from_f64(value.re),
            im: FloatExp::from_f64(value.im),
        }
    }

    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    pub fn mul(self, rhs: Self) -> Self {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        Self { re, im }
    }

    pub fn mul_complex64(self, rhs: Complex64) -> Self {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        Self { re, im }
    }

    pub fn norm_sqr_approx(self) -> f64 {
        let re = self.re.to_f64();
        let im = self.im.to_f64();
        re * re + im * im
    }

    pub fn to_complex64_approx(self) -> Complex64 {
        Complex64::new(self.re.to_f64(), self.im.to_f64())
    }
}

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

fn pow2i(exp: i32) -> f64 {
    if exp < -1022 {
        return 0.0;
    }
    if exp > 1023 {
        return f64::INFINITY;
    }
    f64::from_bits(((exp + 1023) as u64) << 52)
}

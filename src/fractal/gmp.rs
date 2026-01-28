use num_complex::Complex64;
use rug::Complex;
use rug::Float;
use rug::ops::Pow;

use crate::fractal::{FractalParams, FractalType};

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ComplexF {
    pub re: Float,
    pub im: Float,
}

#[allow(dead_code)]
impl ComplexF {
    pub fn with_val(prec: u32, re: f64, im: f64) -> Self {
        Self {
            re: Float::with_val(prec, re),
            im: Float::with_val(prec, im),
        }
    }

    pub fn new(re: Float, im: Float) -> Self {
        Self { re, im }
    }

    pub fn norm_sqr(&self) -> Float {
        let mut re2 = self.re.clone();
        re2 *= &self.re;
        let mut im2 = self.im.clone();
        im2 *= &self.im;
        re2 += im2;
        re2
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut re = self.re.clone();
        re += &other.re;
        let mut im = self.im.clone();
        im += &other.im;
        Self { re, im }
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut re = self.re.clone();
        re -= &other.re;
        let mut im = self.im.clone();
        im -= &other.im;
        Self { re, im }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut re = self.re.clone();
        re *= &other.re;
        let mut re_sub = self.im.clone();
        re_sub *= &other.im;
        re -= re_sub;

        let mut im = self.re.clone();
        im *= &other.im;
        let mut im_add = self.im.clone();
        im_add *= &other.re;
        im += im_add;

        Self { re, im }
    }

    pub fn div(&self, other: &Self) -> Self {
        let mut denom = other.re.clone();
        denom *= &other.re;
        let mut denom_add = other.im.clone();
        denom_add *= &other.im;
        denom += denom_add;

        let mut re = self.re.clone();
        re *= &other.re;
        let mut re_add = self.im.clone();
        re_add *= &other.im;
        re += re_add;
        re /= &denom;

        let mut im = self.im.clone();
        im *= &other.re;
        let mut im_sub = self.re.clone();
        im_sub *= &other.im;
        im -= im_sub;
        im /= &denom;

        Self { re, im }
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re.clone(),
            im: -self.im.clone(),
        }
    }

    pub fn sin(&self, prec: u32) -> Self {
        let sin_re = self.re.clone().sin();
        let cos_re = self.re.clone().cos();
        let sinh_im = self.im.clone().sinh();
        let cosh_im = self.im.clone().cosh();
        Self {
            re: sin_re * cosh_im,
            im: cos_re * sinh_im,
        }
        .with_prec(prec)
    }

    pub fn pow_u32(&self, mut exp: u32, prec: u32) -> Self {
        let mut base = self.clone();
        let mut result = ComplexF::with_val(prec, 1.0, 0.0);
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    pub fn pow_f64(&self, exp: f64, prec: u32) -> Self {
        let r = self.norm_sqr().sqrt();
        if r == 0 {
            return ComplexF::with_val(prec, 0.0, 0.0);
        }
        let theta = self.im.clone().atan2(&self.re);
        let exp_f = Float::with_val(prec, exp);
        let r_pow = r.pow(&exp_f);
        let angle = theta * exp_f;
        let cos_a = angle.clone().cos();
        let sin_a = angle.sin();
        Self {
            re: &r_pow * cos_a,
            im: r_pow * sin_a,
        }
    }

    pub fn to_complex64(&self) -> Complex64 {
        Complex64::new(self.re.to_f64(), self.im.to_f64())
    }

    fn with_prec(self, prec: u32) -> Self {
        Self {
            re: Float::with_val(prec, self.re),
            im: Float::with_val(prec, self.im),
        }
    }
}

#[allow(dead_code)]
pub struct GmpParams {
    pub prec: u32,
    pub iteration_max: u32,
    pub bailout_sqr: Float,
    pub seed: ComplexF,
    pub fractal_type: FractalType,
    pub multibrot_power: f64,
}

#[allow(dead_code)]
impl GmpParams {
    pub fn from_params(params: &FractalParams) -> Self {
        let prec = params.precision_bits.max(64);
        let bailout = Float::with_val(prec, params.bailout);
        let mut bailout_sqr = bailout.clone();
        bailout_sqr *= &bailout;
        Self {
            prec,
            iteration_max: params.iteration_max,
            bailout_sqr,
            seed: ComplexF::with_val(prec, params.seed.re, params.seed.im),
            fractal_type: params.fractal_type,
            multibrot_power: params.multibrot_power,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MpcParams {
    pub prec: u32,
    pub iteration_max: u32,
    pub bailout_sqr: Float,
    pub seed: Complex,
    pub fractal_type: FractalType,
    pub multibrot_power: f64,
}

impl MpcParams {
    pub fn from_params(params: &FractalParams) -> Self {
        // Pour perturbation, utiliser la précision calculée si disponible
        // Sinon utiliser params.precision_bits
        use crate::fractal::perturbation::compute_perturbation_precision_bits;
        let prec = if params.algorithm_mode == crate::fractal::AlgorithmMode::Perturbation 
            || params.algorithm_mode == crate::fractal::AlgorithmMode::ReferenceGmp {
            compute_perturbation_precision_bits(params)
        } else {
            params.precision_bits.max(64)
        };
        let bailout = Float::with_val(prec, params.bailout);
        let mut bailout_sqr = bailout.clone();
        bailout_sqr *= &bailout;
        Self {
            prec,
            iteration_max: params.iteration_max,
            bailout_sqr,
            seed: Complex::with_val(prec, (params.seed.re, params.seed.im)),
            fractal_type: params.fractal_type,
            multibrot_power: params.multibrot_power,
        }
    }
}

pub fn complex_from_xy(prec: u32, re: Float, im: Float) -> Complex {
    Complex::with_val(prec, (re, im))
}

#[allow(dead_code)]
pub fn complex_from_f64(prec: u32, re: f64, im: f64) -> Complex {
    Complex::with_val(prec, (re, im))
}

pub fn complex_to_complex64(value: &Complex) -> Complex64 {
    let re = value.real().to_f64();
    let im = value.imag().to_f64();
    // S'assurer que les valeurs sont finies, sinon utiliser 0.0
    let re_finite = if re.is_finite() { re } else { 0.0 };
    let im_finite = if im.is_finite() { im } else { 0.0 };
    Complex64::new(re_finite, im_finite)
}

pub fn complex_norm_sqr(value: &Complex, prec: u32) -> Float {
    let mut re2 = value.real().clone();
    re2 *= value.real();
    let mut im2 = value.imag().clone();
    im2 *= value.imag();
    let mut sum = Float::with_val(prec, re2);
    sum += im2;
    sum
}

fn pow_u32_mpc(value: &Complex, mut exp: u32, prec: u32) -> Complex {
    let mut result = Complex::with_val(prec, (1, 0));
    let mut base = value.clone();
    while exp > 0 {
        if exp & 1 == 1 {
            result *= &base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base.clone();
        }
    }
    result
}

pub fn pow_f64_mpc(value: &Complex, exp: f64, prec: u32) -> Complex {
    let r = complex_norm_sqr(value, prec).sqrt();
    if r == 0 {
        return Complex::with_val(prec, (0, 0));
    }
    let theta = value.imag().clone().atan2(value.real());
    let exp_f = Float::with_val(prec, exp);
    let r_pow = r.pow(&exp_f);
    let angle = theta * exp_f;
    let cos_a = angle.clone().cos();
    let sin_a = angle.sin();
    Complex::with_val(prec, (r_pow.clone() * cos_a, r_pow * sin_a))
}

pub fn iterate_point_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    match g.fractal_type {
        FractalType::VonKoch | FractalType::Dragon => {
            panic!("Les fractales vectorielles doivent être rendues via render_von_koch/render_dragon")
        }
        FractalType::Mandelbrot => mandelbrot_mpc(g, z_pixel),
        FractalType::Julia => julia_mpc(g, z_pixel),
        FractalType::JuliaSin => julia_sin_mpc(g, z_pixel),
        FractalType::Newton => newton_mpc(g, z_pixel),
        FractalType::Phoenix => phoenix_mpc(g, z_pixel),
        FractalType::Buffalo => buffalo_mpc(g, z_pixel),
        FractalType::BarnsleyJulia => barnsley_julia_mpc(g, z_pixel),
        FractalType::BarnsleyMandelbrot => barnsley_mandelbrot_mpc(g, z_pixel),
        FractalType::MagnetJulia => magnet_julia_mpc(g, z_pixel),
        FractalType::MagnetMandelbrot => magnet_mandelbrot_mpc(g, z_pixel),
        FractalType::BurningShip => burning_ship_mpc(g, z_pixel),
        FractalType::Tricorn => tricorn_mpc(g, z_pixel),
        FractalType::Mandelbulb => mandelbulb_mpc(g, z_pixel),
        FractalType::PerpendicularBurningShip => perpendicular_burning_ship_mpc(g, z_pixel),
        FractalType::Celtic => celtic_mpc(g, z_pixel),
        FractalType::AlphaMandelbrot => alpha_mandelbrot_mpc(g, z_pixel),
        FractalType::PickoverStalks => pickover_stalks_mpc(g, z_pixel),
        FractalType::Nova => nova_mpc(g, z_pixel),
        FractalType::Multibrot => multibrot_mpc(g, z_pixel),
        FractalType::Buddhabrot => {
            panic!("Buddhabrot doit être rendu via render_buddhabrot(), pas iterate_point_mpc()")
        }
        FractalType::Lyapunov => {
            panic!("Lyapunov doit être rendu via render_lyapunov(), pas iterate_point_mpc()")
        }
        FractalType::Nebulabrot => {
            panic!("Nebulabrot doit être rendu via render_nebulabrot(), pas iterate_point_mpc()")
        }
    }
}

#[allow(dead_code)]
pub fn iterate_point_gmp(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    match g.fractal_type {
        FractalType::VonKoch | FractalType::Dragon => {
            panic!("Les fractales vectorielles doivent être rendues via render_von_koch/render_dragon")
        }
        FractalType::Mandelbrot => mandelbrot(g, z_pixel),
        FractalType::Julia => julia(g, z_pixel),
        FractalType::JuliaSin => julia_sin(g, z_pixel),
        FractalType::Newton => newton(g, z_pixel),
        FractalType::Phoenix => phoenix(g, z_pixel),
        FractalType::Buffalo => buffalo(g, z_pixel),
        FractalType::BarnsleyJulia => barnsley_julia(g, z_pixel),
        FractalType::BarnsleyMandelbrot => barnsley_mandelbrot(g, z_pixel),
        FractalType::MagnetJulia => magnet_julia(g, z_pixel),
        FractalType::MagnetMandelbrot => magnet_mandelbrot(g, z_pixel),
        FractalType::BurningShip => burning_ship(g, z_pixel),
        FractalType::Tricorn => tricorn(g, z_pixel),
        FractalType::Mandelbulb => mandelbulb(g, z_pixel),
        FractalType::PerpendicularBurningShip => perpendicular_burning_ship(g, z_pixel),
        FractalType::Celtic => celtic(g, z_pixel),
        FractalType::AlphaMandelbrot => alpha_mandelbrot(g, z_pixel),
        FractalType::PickoverStalks => pickover_stalks(g, z_pixel),
        FractalType::Nova => nova(g, z_pixel),
        FractalType::Multibrot => multibrot(g, z_pixel),
        FractalType::Buddhabrot => {
            panic!("Buddhabrot doit être rendu via render_buddhabrot(), pas iterate_point_gmp()")
        }
        FractalType::Lyapunov => {
            panic!("Lyapunov doit être rendu via render_lyapunov(), pas iterate_point_gmp()")
        }
        FractalType::Nebulabrot => {
            panic!("Nebulabrot doit être rendu via render_nebulabrot(), pas iterate_point_gmp()")
        }
    }
}

#[allow(dead_code)]
fn mandelbrot(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        z = z.mul(&z).add(z_pixel);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        z = z.mul(&z).add(&g.seed);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn julia_sin(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let sin_z = z.sin(g.prec);
        z = g.seed.mul(&sin_z);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn newton(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let degree = g.seed.re.to_f64().round() as i32;
    let degree = if degree <= 0 { 3 } else { degree } as u32;
    let degree_f = Float::with_val(g.prec, degree);
    let epsilon = Float::with_val(g.prec, 1e-12f64);
    let mut epsilon_sqr = epsilon.clone();
    epsilon_sqr *= &epsilon;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_pow = z.pow_u32(degree, g.prec);
        let z_pow_deriv = z.pow_u32(degree - 1, g.prec);
        let numerator = z_pow.sub(&ComplexF::with_val(g.prec, 1.0, 0.0));
        let mut denom_re = z_pow_deriv.re.clone();
        denom_re *= &degree_f;
        let mut denom_im = z_pow_deriv.im.clone();
        denom_im *= &degree_f;
        let denominator = ComplexF::new(denom_re, denom_im);
        if denominator.norm_sqr() < epsilon_sqr {
            break;
        }
        let z_quot = numerator.div(&denominator);
        z = z.sub(&z_quot);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn phoenix(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut y = ComplexF::with_val(g.prec, 0.0, 0.0);
    let mut i = 0u32;
    let p1 = Float::with_val(g.prec, 0.56667);
    let p2 = Float::with_val(g.prec, -0.5);
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_sq = z.mul(&z);
        let mut z_temp = ComplexF::new({
            let mut re = z_sq.re.clone();
            re += &p1;
            re
        }, z_sq.im);
        let mut zp_re = y.re.clone();
        zp_re *= &p2;
        let mut zp_im = y.im.clone();
        zp_im *= &p2;
        let zp_temp = ComplexF::new(zp_re, zp_im);
        z_temp = z_temp.add(&zp_temp);
        y = z.clone();
        z = z_temp;
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn barnsley_julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let one = ComplexF::with_val(g.prec, 1.0, 0.0);
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        if z.re >= 0 {
            z = z.sub(&one).mul(&g.seed);
        } else {
            z = z.add(&one).mul(&g.seed);
        }
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn barnsley_mandelbrot(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let c = z_pixel;
    let mut i = 0u32;
    let one = ComplexF::with_val(g.prec, 1.0, 0.0);
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        if z.re >= 0 {
            z = z.sub(&one).mul(c);
        } else {
            z = z.add(&one).mul(c);
        }
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn magnet_julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let mut seed_minus_one_re = g.seed.re.clone();
    seed_minus_one_re -= Float::with_val(g.prec, 1.0);
    let seed_minus_one = ComplexF::new(seed_minus_one_re, g.seed.im.clone());
    let mut seed_minus_two_re = g.seed.re.clone();
    seed_minus_two_re -= Float::with_val(g.prec, 2.0);
    let seed_minus_two = ComplexF::new(seed_minus_two_re, g.seed.im.clone());
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let mut n = z.mul(&z).add(&seed_minus_one);
        n = n.mul(&n);
        let q = z.mul(&ComplexF::with_val(g.prec, 2.0, 0.0)).add(&seed_minus_two);
        z = n.div(&q);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn magnet_mandelbrot(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let c = z_pixel;
    let mut z = ComplexF::with_val(g.prec, 0.0, 0.0);
    let mut i = 0u32;
    let mut c_minus_one_re = c.re.clone();
    c_minus_one_re -= Float::with_val(g.prec, 1.0);
    let c_minus_one = ComplexF::new(c_minus_one_re, c.im.clone());
    let mut c_minus_two_re = c.re.clone();
    c_minus_two_re -= Float::with_val(g.prec, 2.0);
    let c_minus_two = ComplexF::new(c_minus_two_re, c.im.clone());
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let mut n = z.mul(&z).add(&c_minus_one);
        n = n.mul(&n);
        let q = z.mul(&ComplexF::with_val(g.prec, 2.0, 0.0)).add(&c_minus_two);
        z = n.div(&q);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn burning_ship(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let re = z.re.clone().abs();
        let im = z.im.clone().abs();
        let mut z_temp = ComplexF::new(re, im);
        z_temp = z_temp.mul(&z_temp);
        z = z_temp.add(z_pixel);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn buffalo(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_sq = z.mul(&z);
        let re_sq = z_sq.re.clone().abs();
        let im_sq = z_sq.im.clone().abs();
        z = ComplexF::new(re_sq + &z_pixel.re, im_sq + &z_pixel.im);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn tricorn(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_conj = z.conj();
        let z_temp = z_conj.mul(&z_conj);
        z = z_temp.add(z_pixel);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn mandelbulb(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_pow = z.pow_u32(8, g.prec);
        z = z_pow.add(z_pixel);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn perpendicular_burning_ship(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let x = z.re.clone();
        let y = z.im.clone();
        let y_abs = y.clone().abs();
        let mut x2 = x.clone();
        x2 *= &x;
        let mut y2 = y_abs.clone();
        y2 *= &y_abs;
        let mut re = x2;
        re -= y2;
        re += &z_pixel.re;

        let mut im = x;
        im *= y_abs;
        im *= Float::with_val(g.prec, -2.0);
        im += &z_pixel.im;

        z = ComplexF::new(re, im);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn celtic(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let x = z.re.clone();
        let y = z.im.clone();
        let mut u = x.clone();
        u *= &x;
        let mut y2 = y.clone();
        y2 *= &y;
        u -= y2;
        let mut v = x;
        v *= y;
        v *= Float::with_val(g.prec, 2.0);
        let mut re = u.abs();
        re += &z_pixel.re;
        let mut im = v;
        im += &z_pixel.im;
        z = ComplexF::new(re, im);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn alpha_mandelbrot(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_sq = z.mul(&z);
        let m = z_sq.add(z_pixel);
        let m_sq = m.mul(&m);
        z = z_sq.add(&m_sq).add(z_pixel);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn pickover_stalks(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    let mut trap_min = Float::with_val(g.prec, 1e10f64);
    let trap_divisor = Float::with_val(g.prec, 0.03f64);
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        z = z.mul(&z).add(z_pixel);
        let re_abs = z.re.clone().abs();
        let im_abs = z.im.clone().abs();
        let trap_distance = if re_abs < im_abs { re_abs } else { im_abs };
        if trap_distance < trap_min {
            trap_min = trap_distance;
        }
        i += 1;
    }
    let mut iter_value: u32;
    if trap_min > Float::with_val(g.prec, 1e-10f64) {
        let log_trap = -(&trap_min / trap_divisor).ln();
        iter_value = (log_trap.to_f64() * 100.0) as u32;
        if iter_value >= g.iteration_max {
            iter_value = g.iteration_max.saturating_sub(1);
        }
    } else {
        iter_value = g.iteration_max.saturating_sub(1);
    }
    (iter_value, z)
}

#[allow(dead_code)]
fn nova(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = ComplexF::with_val(g.prec, 1.0, 0.0);
    let mut i = 0u32;
    let a_relax = ComplexF::with_val(g.prec, 1.0, 0.0);
    let p_poly = 3u32;
    let conv_epsilon = Float::with_val(g.prec, 1e-7f64);
    let mut conv_epsilon_sq = conv_epsilon.clone();
    conv_epsilon_sq *= &conv_epsilon;
    while i < g.iteration_max {
        let z_pow = z.pow_u32(p_poly, g.prec);
        let z_pow_deriv = z.pow_u32(p_poly - 1, g.prec);
        let numerator = z_pow.sub(&ComplexF::with_val(g.prec, 1.0, 0.0));
        let p_poly_f = Float::with_val(g.prec, p_poly);
        let mut denom_re = z_pow_deriv.re.clone();
        denom_re *= &p_poly_f;
        let mut denom_im = z_pow_deriv.im.clone();
        denom_im *= &p_poly_f;
        let denominator = ComplexF::new(denom_re, denom_im);
        let denom_epsilon = Float::with_val(g.prec, 1e-10f64);
        let mut denom_epsilon_sqr = denom_epsilon.clone();
        denom_epsilon_sqr *= &denom_epsilon;
        if denominator.norm_sqr() < denom_epsilon_sqr {
            break;
        }
        let mut newton_step = numerator.div(&denominator);
        newton_step = newton_step.mul(&a_relax);
        let z_prev = z.clone();
        z = z.sub(&newton_step).add(z_pixel);
        let diff = z.sub(&z_prev);
        let diff_sq = diff.norm_sqr();
        let z_sq = z.norm_sqr();
        let one = Float::with_val(g.prec, 1.0);
        let denom = if z_sq < one { one } else { z_sq };
        let mut ratio = diff_sq;
        ratio /= &denom;
        if ratio < conv_epsilon_sq {
            break;
        }
        if z.norm_sqr() > g.bailout_sqr {
            break;
        }
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn multibrot(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    let d = g.multibrot_power;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_pow = z.pow_f64(d, g.prec);
        z = z_pow.add(z_pixel);
        i += 1;
    }
    (i, z)
}

fn mandelbrot_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut z_sq = z.clone();
        z_sq *= &z;
        z_sq += z_pixel;
        z = z_sq;
        i += 1;
    }
    (i, z)
}

fn julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut z_sq = z.clone();
        z_sq *= &z;
        z_sq += &g.seed;
        z = z_sq;
        i += 1;
    }
    (i, z)
}

fn julia_sin_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut sin_z = z.clone().sin();
        sin_z *= &g.seed;
        z = sin_z;
        i += 1;
    }
    (i, z)
}

fn newton_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let degree = g.seed.real().to_f64().round() as i32;
    let degree = if degree <= 0 { 3 } else { degree } as u32;
    let degree_f = Float::with_val(g.prec, degree);
    let epsilon = Float::with_val(g.prec, 1e-12f64);
    let mut epsilon_sqr = epsilon.clone();
    epsilon_sqr *= &epsilon;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let z_pow = pow_u32_mpc(&z, degree, g.prec);
        let z_pow_deriv = pow_u32_mpc(&z, degree - 1, g.prec);
        let mut numerator = z_pow.clone();
        numerator -= Complex::with_val(g.prec, (1, 0));
        let mut denom_re = z_pow_deriv.real().clone();
        denom_re *= &degree_f;
        let mut denom_im = z_pow_deriv.imag().clone();
        denom_im *= &degree_f;
        let denominator = complex_from_xy(g.prec, denom_re, denom_im);
        if complex_norm_sqr(&denominator, g.prec) < epsilon_sqr {
            break;
        }
        let z_quot = numerator / denominator;
        z -= &z_quot;
        i += 1;
    }
    (i, z)
}

fn phoenix_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut y = Complex::with_val(g.prec, (0, 0));
    let mut i = 0u32;
    let p1 = Float::with_val(g.prec, 0.56667);
    let p2 = Float::with_val(g.prec, -0.5);
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut z_sq = z.clone();
        z_sq *= &z;
        let mut re = z_sq.real().clone();
        re += &p1;
        let im = z_sq.imag().clone();
        let mut z_temp = complex_from_xy(g.prec, re, im);
        let mut zp_re = y.real().clone();
        zp_re *= &p2;
        let mut zp_im = y.imag().clone();
        zp_im *= &p2;
        let zp_temp = complex_from_xy(g.prec, zp_re, zp_im);
        z_temp += zp_temp;
        y = z;
        z = z_temp;
        i += 1;
    }
    (i, z)
}

fn barnsley_julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let one = Float::with_val(g.prec, 1.0);
    let zero = Float::with_val(g.prec, 0.0);
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut re = z.real().clone();
        if re >= zero {
            re -= &one;
        } else {
            re += &one;
        }
        let im = z.imag().clone();
        let mut z_adj = complex_from_xy(g.prec, re, im);
        z_adj *= &g.seed;
        z = z_adj;
        i += 1;
    }
    (i, z)
}

fn barnsley_mandelbrot_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let c = z_pixel;
    let mut i = 0u32;
    let one = Float::with_val(g.prec, 1.0);
    let zero = Float::with_val(g.prec, 0.0);
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut re = z.real().clone();
        if re >= zero {
            re -= &one;
        } else {
            re += &one;
        }
        let im = z.imag().clone();
        let mut z_adj = complex_from_xy(g.prec, re, im);
        z_adj *= c;
        z = z_adj;
        i += 1;
    }
    (i, z)
}

fn magnet_julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let mut seed_minus_one_re = g.seed.real().clone();
    seed_minus_one_re -= Float::with_val(g.prec, 1.0);
    let seed_minus_one = complex_from_xy(g.prec, seed_minus_one_re, g.seed.imag().clone());
    let mut seed_minus_two_re = g.seed.real().clone();
    seed_minus_two_re -= Float::with_val(g.prec, 2.0);
    let seed_minus_two = complex_from_xy(g.prec, seed_minus_two_re, g.seed.imag().clone());
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut n = z.clone();
        n *= &z;
        n += &seed_minus_one;
        let mut n_sq = n.clone();
        n_sq *= &n;
        let mut q = z.clone();
        q *= Complex::with_val(g.prec, (2, 0));
        q += &seed_minus_two;
        z = n_sq / q;
        i += 1;
    }
    (i, z)
}

fn magnet_mandelbrot_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let c = z_pixel;
    let mut z = Complex::with_val(g.prec, (0, 0));
    let mut i = 0u32;
    let mut c_minus_one_re = c.real().clone();
    c_minus_one_re -= Float::with_val(g.prec, 1.0);
    let c_minus_one = complex_from_xy(g.prec, c_minus_one_re, c.imag().clone());
    let mut c_minus_two_re = c.real().clone();
    c_minus_two_re -= Float::with_val(g.prec, 2.0);
    let c_minus_two = complex_from_xy(g.prec, c_minus_two_re, c.imag().clone());
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut n = z.clone();
        n *= &z;
        n += &c_minus_one;
        let mut n_sq = n.clone();
        n_sq *= &n;
        let mut q = z.clone();
        q *= Complex::with_val(g.prec, (2, 0));
        q += &c_minus_two;
        z = n_sq / q;
        i += 1;
    }
    (i, z)
}

fn burning_ship_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let re = z.real().clone().abs();
        let im = z.imag().clone().abs();
        let mut z_temp = complex_from_xy(g.prec, re, im);
        z_temp *= z_temp.clone();
        z_temp += z_pixel;
        z = z_temp;
        i += 1;
    }
    (i, z)
}

fn buffalo_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut z_sq = z.clone();
        z_sq *= &z;
        let re_sq = z_sq.real().clone().abs();
        let im_sq = z_sq.imag().clone().abs();
        let mut re = re_sq;
        re += z_pixel.real();
        let mut im = im_sq;
        im += z_pixel.imag();
        z = complex_from_xy(g.prec, re, im);
        i += 1;
    }
    (i, z)
}

fn tricorn_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let z_conj = z.clone().conj();
        let mut z_temp = z_conj.clone();
        z_temp *= &z_conj;
        z_temp += z_pixel;
        z = z_temp;
        i += 1;
    }
    (i, z)
}

fn mandelbulb_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let z_pow = pow_u32_mpc(&z, 8, g.prec);
        let mut z_next = z_pow;
        z_next += z_pixel;
        z = z_next;
        i += 1;
    }
    (i, z)
}

fn perpendicular_burning_ship_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let x = z.real().clone();
        let y = z.imag().clone();
        let y_abs = y.clone().abs();
        let mut x2 = x.clone();
        x2 *= &x;
        let mut y2 = y_abs.clone();
        y2 *= &y_abs;
        let mut re = x2;
        re -= y2;
        re += z_pixel.real();

        let mut im = x;
        im *= y_abs;
        im *= Float::with_val(g.prec, -2.0);
        im += z_pixel.imag();

        z = complex_from_xy(g.prec, re, im);
        i += 1;
    }
    (i, z)
}

fn celtic_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let x = z.real().clone();
        let y = z.imag().clone();
        let mut u = x.clone();
        u *= &x;
        let mut y2 = y.clone();
        y2 *= &y;
        u -= y2;
        let mut v = x;
        v *= y;
        v *= Float::with_val(g.prec, 2.0);
        let mut re = u.abs();
        re += z_pixel.real();
        let mut im = v;
        im += z_pixel.imag();
        z = complex_from_xy(g.prec, re, im);
        i += 1;
    }
    (i, z)
}

fn alpha_mandelbrot_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut z_sq = z.clone();
        z_sq *= &z;
        let mut m = z_sq.clone();
        m += z_pixel;
        let mut m_sq = m.clone();
        m_sq *= &m;
        let mut z_next = z_sq;
        z_next += &m_sq;
        z_next += z_pixel;
        z = z_next;
        i += 1;
    }
    (i, z)
}

fn pickover_stalks_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    let mut trap_min = Float::with_val(g.prec, 1e10f64);
    let trap_divisor = Float::with_val(g.prec, 0.03f64);
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let mut z_next = z.clone();
        z_next *= &z;
        z_next += z_pixel;
        z = z_next;
        let re_abs = z.real().clone().abs();
        let im_abs = z.imag().clone().abs();
        let trap_distance = if re_abs < im_abs { re_abs } else { im_abs };
        if trap_distance < trap_min {
            trap_min = trap_distance;
        }
        i += 1;
    }
    let mut iter_value: u32;
    if trap_min > Float::with_val(g.prec, 1e-10f64) {
        let log_trap = -(&trap_min / trap_divisor).ln();
        iter_value = (log_trap.to_f64() * 100.0) as u32;
        if iter_value >= g.iteration_max {
            iter_value = g.iteration_max.saturating_sub(1);
        }
    } else {
        iter_value = g.iteration_max.saturating_sub(1);
    }
    (iter_value, z)
}

fn nova_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = Complex::with_val(g.prec, (1.0, 0.0));
    let mut i = 0u32;
    let a_relax = Complex::with_val(g.prec, (1.0, 0.0));
    let p_poly = 3u32;
    let conv_epsilon = Float::with_val(g.prec, 1e-7f64);
    let mut conv_epsilon_sq = conv_epsilon.clone();
    conv_epsilon_sq *= &conv_epsilon;
    while i < g.iteration_max {
        let z_pow = pow_u32_mpc(&z, p_poly, g.prec);
        let z_pow_deriv = pow_u32_mpc(&z, p_poly - 1, g.prec);
        let mut numerator = z_pow.clone();
        numerator -= Complex::with_val(g.prec, (1.0, 0.0));
        let p_poly_f = Float::with_val(g.prec, p_poly);
        let mut denom_re = z_pow_deriv.real().clone();
        denom_re *= &p_poly_f;
        let mut denom_im = z_pow_deriv.imag().clone();
        denom_im *= &p_poly_f;
        let denominator = complex_from_xy(g.prec, denom_re, denom_im);
        let denom_epsilon = Float::with_val(g.prec, 1e-10f64);
        let mut denom_epsilon_sqr = denom_epsilon.clone();
        denom_epsilon_sqr *= &denom_epsilon;
        if complex_norm_sqr(&denominator, g.prec) < denom_epsilon_sqr {
            break;
        }
        let mut newton_step = numerator / denominator;
        newton_step *= &a_relax;
        let z_prev = z.clone();
        let mut z_next = z - &newton_step;
        z_next += z_pixel;
        z = z_next;
        let diff = z.clone() - z_prev;
        let diff_sq = complex_norm_sqr(&diff, g.prec);
        let z_sq = complex_norm_sqr(&z, g.prec);
        let one = Float::with_val(g.prec, 1.0);
        let denom = if z_sq < one { one } else { z_sq };
        let mut ratio = diff_sq;
        ratio /= &denom;
        if ratio < conv_epsilon_sq {
            break;
        }
        if complex_norm_sqr(&z, g.prec) > g.bailout_sqr {
            break;
        }
        i += 1;
    }
    (i, z)
}

fn multibrot_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    let d = g.multibrot_power;
    while i < g.iteration_max && complex_norm_sqr(&z, g.prec) < g.bailout_sqr {
        let z_pow = pow_f64_mpc(&z, d, g.prec);
        let mut z_next = z_pow;
        z_next += z_pixel;
        z = z_next;
        i += 1;
    }
    (i, z)
}

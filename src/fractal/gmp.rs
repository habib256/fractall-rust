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
        let prec = params.engine.precision_bits.max(64);
        let bailout = Float::with_val(prec, params.bailout);
        let mut bailout_sqr = bailout.clone();
        bailout_sqr *= &bailout;
        Self {
            prec,
            iteration_max: params.iteration_max,
            bailout_sqr,
            seed: ComplexF::with_val(prec, params.seed.re, params.seed.im),
            fractal_type: params.fractal_type,
            multibrot_power: params.formula.multibrot_power,
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
        let prec = if params.engine.algorithm_mode == crate::fractal::AlgorithmMode::Perturbation 
            || params.engine.algorithm_mode == crate::fractal::AlgorithmMode::ReferenceGmp {
            compute_perturbation_precision_bits(params)
        } else {
            params.engine.precision_bits.max(64)
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
            multibrot_power: params.formula.multibrot_power,
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

pub fn complex_norm_sqr(value: &Complex, _prec: u32) -> Float {
    let mut re2 = value.real().clone();
    re2 *= value.real();
    let mut im2 = value.imag().clone();
    im2 *= value.imag();
    re2 += im2;
    re2
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
    // IMPORTANT: Créer des copies avec la précision explicite pour éviter la perte de précision
    let im_prec = Float::with_val(prec, value.imag());
    let re_prec = Float::with_val(prec, value.real());
    let theta = im_prec.atan2(&re_prec);
    let exp_f = Float::with_val(prec, exp);
    let r_pow = r.pow(&exp_f);
    let angle = theta * &exp_f;
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
        FractalType::BurningShipJulia => burning_ship_julia_mpc(g, z_pixel),
        FractalType::TricornJulia => tricorn_julia_mpc(g, z_pixel),
        FractalType::CelticJulia => celtic_julia_mpc(g, z_pixel),
        FractalType::BuffaloJulia => buffalo_julia_mpc(g, z_pixel),
        FractalType::MultibrotJulia => multibrot_julia_mpc(g, z_pixel),
        FractalType::PerpendicularBurningShipJulia => perpendicular_burning_ship_julia_mpc(g, z_pixel),
        FractalType::AlphaMandelbrotJulia => alpha_mandelbrot_julia_mpc(g, z_pixel),
        FractalType::MandelbrotSin => mandelbrot_sin_mpc(g, z_pixel),
        special @ (FractalType::Buddhabrot
        | FractalType::Lyapunov
        | FractalType::Nebulabrot
        | FractalType::AntiBuddhabrot) => panic!(
            "{special:?} doit être rendu via le dispatcher cancellable, pas iterate_point_mpc()"
        ),
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
        FractalType::BurningShipJulia => burning_ship_julia(g, z_pixel),
        FractalType::TricornJulia => tricorn_julia(g, z_pixel),
        FractalType::CelticJulia => celtic_julia(g, z_pixel),
        FractalType::BuffaloJulia => buffalo_julia(g, z_pixel),
        FractalType::MultibrotJulia => multibrot_julia(g, z_pixel),
        FractalType::PerpendicularBurningShipJulia => perpendicular_burning_ship_julia(g, z_pixel),
        FractalType::AlphaMandelbrotJulia => alpha_mandelbrot_julia(g, z_pixel),
        FractalType::MandelbrotSin => mandelbrot_sin(g, z_pixel),
        special @ (FractalType::Buddhabrot
        | FractalType::Lyapunov
        | FractalType::Nebulabrot
        | FractalType::AntiBuddhabrot) => panic!(
            "{special:?} doit être rendu via le dispatcher cancellable, pas iterate_point_gmp()"
        ),
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
fn mandelbrot_sin(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let sin_z = z.sin(g.prec);
        z = z_pixel.mul(&sin_z);
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
    // Magnet1j: z_{n+1} = ((z_n² + c - 1) / (2*z_n + c - 2))²
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let mut seed_minus_one_re = g.seed.re.clone();
    seed_minus_one_re -= Float::with_val(g.prec, 1.0);
    let seed_minus_one = ComplexF::new(seed_minus_one_re, g.seed.im.clone());
    let mut seed_minus_two_re = g.seed.re.clone();
    seed_minus_two_re -= Float::with_val(g.prec, 2.0);
    let seed_minus_two = ComplexF::new(seed_minus_two_re, g.seed.im.clone());
    let threshold = Float::with_val(g.prec, 1e-24);
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let n = z.mul(&z).add(&seed_minus_one);
        let q = z.mul(&ComplexF::with_val(g.prec, 2.0, 0.0)).add(&seed_minus_two);
        if q.norm_sqr() < threshold {
            break;
        }
        let ratio = n.div(&q);
        z = ratio.mul(&ratio);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn magnet_mandelbrot(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    // Magnet1m: z_{n+1} = ((z_n² + c - 1) / (2*z_n + c - 2))²
    let c = z_pixel;
    let mut z = ComplexF::with_val(g.prec, 0.0, 0.0);
    let mut i = 0u32;
    let mut c_minus_one_re = c.re.clone();
    c_minus_one_re -= Float::with_val(g.prec, 1.0);
    let c_minus_one = ComplexF::new(c_minus_one_re, c.im.clone());
    let mut c_minus_two_re = c.re.clone();
    c_minus_two_re -= Float::with_val(g.prec, 2.0);
    let c_minus_two = ComplexF::new(c_minus_two_re, c.im.clone());
    let threshold = Float::with_val(g.prec, 1e-24);
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let n = z.mul(&z).add(&c_minus_one);
        let q = z.mul(&ComplexF::with_val(g.prec, 2.0, 0.0)).add(&c_minus_two);
        if q.norm_sqr() < threshold {
            break;
        }
        let ratio = n.div(&q);
        z = ratio.mul(&ratio);
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

// Julia variants (z_0 = z_pixel, c = g.seed)

#[allow(dead_code)]
fn burning_ship_julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let re = z.re.clone().abs();
        let im = z.im.clone().abs();
        let mut z_temp = ComplexF::new(re, im);
        z_temp = z_temp.mul(&z_temp);
        z = z_temp.add(&g.seed);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn tricorn_julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_conj = z.conj();
        let z_temp = z_conj.mul(&z_conj);
        z = z_temp.add(&g.seed);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn celtic_julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
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
        re += &g.seed.re;
        let mut im = v;
        im += &g.seed.im;
        z = ComplexF::new(re, im);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn buffalo_julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_sq = z.mul(&z);
        let re_sq = z_sq.re.clone().abs();
        let im_sq = z_sq.im.clone().abs();
        z = ComplexF::new(re_sq + &g.seed.re, im_sq + &g.seed.im);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn multibrot_julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let d = g.multibrot_power;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_pow = z.pow_f64(d, g.prec);
        z = z_pow.add(&g.seed);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn perpendicular_burning_ship_julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
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
        re += &g.seed.re;

        let mut im = x;
        im *= y_abs;
        im *= Float::with_val(g.prec, -2.0);
        im += &g.seed.im;

        z = ComplexF::new(re, im);
        i += 1;
    }
    (i, z)
}

#[allow(dead_code)]
fn alpha_mandelbrot_julia(g: &GmpParams, z_pixel: &ComplexF) -> (u32, ComplexF) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && z.norm_sqr() < g.bailout_sqr {
        let z_sq = z.mul(&z);
        let m = z_sq.add(&g.seed);
        let m_sq = m.mul(&m);
        z = z_sq.add(&m_sq).add(&g.seed);
        i += 1;
    }
    (i, z)
}

/// Test de boucle `|z|² < bailout²`, décision IDENTIQUE à
/// `complex_norm_sqr(z) < g.bailout_sqr` mais sans payer, à chaque
/// itération, deux multiplications pleine précision et deux allocations :
/// la norme est d'abord estimée en f64 (erreur relative ~2⁻⁵¹), et tant
/// qu'elle reste à plus de 1 % sous le rayon la norme exacte l'est aussi.
/// Près du rayon, au-delà, ou hors de la plage f64 (NaN, infini, dépassement
/// du carré), on retombe sur la comparaison exacte. Même procédé que le test
/// d'évasion de l'orbite de référence (`orbit.rs`).
#[inline]
fn mpc_below_bailout(z: &Complex, g: &MpcParams) -> bool {
    let (re, im) = (z.real().to_f64(), z.imag().to_f64());
    if re * re + im * im < g.bailout_sqr.to_f64() * 0.99 {
        return true;
    }
    complex_norm_sqr(z, g.prec) < g.bailout_sqr
}

/// Carré complexe EN PLACE, arrondi EXACTEMENT comme `mpc_sqr` / `mpc_mul`
/// (chaque composante est l'arrondi au plus près du résultat exact), sans
/// aucune allocation : le profil d'un rendu GMP pur montrait 60 % du temps
/// dans `malloc`/`free` — MPC alloue ses temporaires à CHAQUE appel.
///
/// - `im' = 2·x·y` : un produit MPFR arrondi une fois, doublement exact ;
/// - `re' = x² − y²` : les deux carrés sont calculés EXACTS dans des tampons
///   à précision double (le carré d'un nombre de p bits tient en 2p bits),
///   puis UNE soustraction arrondie à la précision de `z`. Jamais
///   `fl(x²) − fl(y²)` (double arrondi, faux dès que x ≈ y).
///
/// `sq` : deux tampons créés par [`sqr_scratch`] pour la précision de `z`.
/// Valeurs finies seulement (c'est le cas dans les boucles : `z` est sous le
/// rayon d'évasion à l'entrée). Verrou `in_place_square_matches_mpc_sqr`.
#[inline]
fn sqr_in_place(z: &mut Complex, sq: &mut (Float, Float)) {
    use rug::Assign;
    let (re, im) = z.as_mut_real_imag();
    sq.0.assign(re.square_ref());
    sq.1.assign(im.square_ref());
    *im *= &*re;
    *im <<= 1;
    re.assign(&sq.0 - &sq.1);
}

/// Tampons de [`sqr_in_place`] pour un `z` donné (précision double de la
/// plus grande de ses deux composantes).
#[inline]
fn sqr_scratch(z: &Complex) -> (Float, Float) {
    let (pr, pi) = z.prec();
    let p = pr.max(pi).saturating_mul(2);
    (Float::new(p), Float::new(p))
}

/// Raccourci des cycles EXACTS pour les boucles GMP (même principe que
/// `fractal/cycle.rs`, cf. sa doc) : dès que `z` reprend, à sa précision, la
/// valeur exacte d'un point de contrôle (Brent, test tous les 8 pas), l'orbite
/// est périodique pour toujours — le compteur saute un nombre entier de
/// cycles et la boucle finit les itérations restantes. Même compte et même z
/// final que l'itération complète. Égalité de VALEUR MPFR plus signe des
/// zéros (seule liberté de représentation à précision égale) ; `NaN` ne
/// s'égale jamais. Sert aux rendus GMP qui montrent l'intérieur d'un ensemble
/// (juge de la QA, replis GMP plein cadre) ; la convergence à p bits demande
/// ~p/53 fois plus d'itérations qu'en f64.
struct MpcCycle {
    ck: Complex,
    armed: bool,
    ck_iter: u32,
    ck_next: u32,
}

impl MpcCycle {
    fn new(z: &Complex) -> Self {
        Self {
            ck: Complex::new(z.prec()),
            armed: false,
            ck_iter: 0,
            ck_next: if crate::fractal::cycle::shortcut_enabled() { 8 } else { u32::MAX },
        }
    }

    /// À appeler juste après l'incrément de `i`, avec l'état que lira le pas
    /// suivant. Peut avancer `i` (jamais au-delà de `iteration_max - 1`).
    #[inline]
    fn observe(&mut self, i: &mut u32, iteration_max: u32, z: &Complex) {
        if *i % 8 != 0 || self.ck_next == u32::MAX {
            return;
        }
        let same = |a: &Float, b: &Float| a == b && a.is_sign_negative() == b.is_sign_negative();
        if self.armed && same(z.real(), self.ck.real()) && same(z.imag(), self.ck.imag()) {
            if *i < iteration_max {
                let len = *i - self.ck_iter;
                *i += (iteration_max - 1 - *i) / len * len;
            }
            self.ck_next = u32::MAX;
        } else if *i >= self.ck_next {
            use rug::Assign;
            self.ck.assign(z);
            self.armed = true;
            self.ck_iter = *i;
            self.ck_next = i.saturating_mul(2);
        }
    }
}

// `z ← z² + c` EN PLACE, sans allocation par itération (cf. `sqr_in_place`) :
// même valeur au bit près que `clone` + `mpc_mul`. Ces boucles servent de JUGE
// (`fractall-quality`), de correction par pixel (pixels au-delà du qd, pixels
// Julia signalés) et de rendu GMP plein cadre.
fn mandelbrot_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut sq = sqr_scratch(&z);
    let mut cyc = MpcCycle::new(&z);
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        sqr_in_place(&mut z, &mut sq);
        z += z_pixel;
        i += 1;
        cyc.observe(&mut i, g.iteration_max, &z);
    }
    (i, z)
}

fn julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut sq = sqr_scratch(&z);
    let mut cyc = MpcCycle::new(&z);
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        sqr_in_place(&mut z, &mut sq);
        z += &g.seed;
        i += 1;
        cyc.observe(&mut i, g.iteration_max, &z);
    }
    (i, z)
}

fn julia_sin_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let mut sin_z = z.clone().sin();
        sin_z *= &g.seed;
        z = sin_z;
        i += 1;
    }
    (i, z)
}

fn mandelbrot_sin_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let mut sin_z = z.clone().sin();
        sin_z *= z_pixel;
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    // Magnet1j: z_{n+1} = ((z_n² + c - 1) / (2*z_n + c - 2))²
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let mut seed_minus_one_re = g.seed.real().clone();
    seed_minus_one_re -= Float::with_val(g.prec, 1.0);
    let seed_minus_one = complex_from_xy(g.prec, seed_minus_one_re, g.seed.imag().clone());
    let mut seed_minus_two_re = g.seed.real().clone();
    seed_minus_two_re -= Float::with_val(g.prec, 2.0);
    let seed_minus_two = complex_from_xy(g.prec, seed_minus_two_re, g.seed.imag().clone());
    let threshold = Float::with_val(g.prec, 1e-24);
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let mut n = z.clone();
        n *= &z;
        n += &seed_minus_one;
        let mut q = z.clone();
        q *= Complex::with_val(g.prec, (2, 0));
        q += &seed_minus_two;
        if complex_norm_sqr(&q, g.prec) < threshold {
            break;
        }
        let ratio = n / q;
        z = ratio.clone();
        z *= &ratio;
        i += 1;
    }
    (i, z)
}

fn magnet_mandelbrot_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    // Magnet1m: z_{n+1} = ((z_n² + c - 1) / (2*z_n + c - 2))²
    let c = z_pixel;
    let mut z = Complex::with_val(g.prec, (0, 0));
    let mut i = 0u32;
    let mut c_minus_one_re = c.real().clone();
    c_minus_one_re -= Float::with_val(g.prec, 1.0);
    let c_minus_one = complex_from_xy(g.prec, c_minus_one_re, c.imag().clone());
    let mut c_minus_two_re = c.real().clone();
    c_minus_two_re -= Float::with_val(g.prec, 2.0);
    let c_minus_two = complex_from_xy(g.prec, c_minus_two_re, c.imag().clone());
    let threshold = Float::with_val(g.prec, 1e-24);
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let mut n = z.clone();
        n *= &z;
        n += &c_minus_one;
        let mut q = z.clone();
        q *= Complex::with_val(g.prec, (2, 0));
        q += &c_minus_two;
        if complex_norm_sqr(&q, g.prec) < threshold {
            break;
        }
        let ratio = n / q;
        z = ratio.clone();
        z *= &ratio;
        i += 1;
    }
    (i, z)
}

fn burning_ship_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    // En place (cf. `mandelbrot_mpc`) quand `z` est à la précision `g.prec`
    // — sinon la boucle d'origine, qui ré-arrondit à `g.prec` à chaque tour.
    if z.prec() == (g.prec, g.prec) {
        let mut sq = sqr_scratch(&z);
        let mut cyc = MpcCycle::new(&z);
        while i < g.iteration_max && mpc_below_bailout(&z, g) {
            z.mut_real().abs_mut();
            z.mut_imag().abs_mut();
            sqr_in_place(&mut z, &mut sq);
            z += z_pixel;
            i += 1;
            cyc.observe(&mut i, g.iteration_max, &z);
        }
        return (i, z);
    }
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    let mut sq = sqr_scratch(&z);
    let mut cyc = MpcCycle::new(&z);
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        // En place (cf. `mandelbrot_mpc`) : conj exact puis carré.
        z.conj_mut();
        sqr_in_place(&mut z, &mut sq);
        z += z_pixel;
        i += 1;
        cyc.observe(&mut i, g.iteration_max, &z);
    }
    (i, z)
}

fn mandelbulb_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = g.seed.clone();
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let z_pow = pow_f64_mpc(&z, d, g.prec);
        let mut z_next = z_pow;
        z_next += z_pixel;
        z = z_next;
        i += 1;
    }
    (i, z)
}

fn burning_ship_julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let re = z.real().clone().abs();
        let im = z.imag().clone().abs();
        let mut z_temp = complex_from_xy(g.prec, re, im);
        z_temp *= z_temp.clone();
        z_temp += &g.seed;
        z = z_temp;
        i += 1;
    }
    (i, z)
}

fn tricorn_julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let z_conj = z.clone().conj();
        let mut z_temp = z_conj.clone();
        z_temp *= &z_conj;
        z_temp += &g.seed;
        z = z_temp;
        i += 1;
    }
    (i, z)
}

fn celtic_julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
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
        re += g.seed.real();
        let mut im = v;
        im += g.seed.imag();
        z = complex_from_xy(g.prec, re, im);
        i += 1;
    }
    (i, z)
}

fn buffalo_julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let mut z_sq = z.clone();
        z_sq *= &z;
        let re_sq = z_sq.real().clone().abs();
        let im_sq = z_sq.imag().clone().abs();
        let mut re = re_sq;
        re += g.seed.real();
        let mut im = im_sq;
        im += g.seed.imag();
        z = complex_from_xy(g.prec, re, im);
        i += 1;
    }
    (i, z)
}

fn multibrot_julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    let d = g.multibrot_power;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let z_pow = pow_f64_mpc(&z, d, g.prec);
        let mut z_next = z_pow;
        z_next += &g.seed;
        z = z_next;
        i += 1;
    }
    (i, z)
}

fn perpendicular_burning_ship_julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let x = z.real().clone();
        let y = z.imag().clone();
        let y_abs = y.clone().abs();
        let mut x2 = x.clone();
        x2 *= &x;
        let mut y2 = y_abs.clone();
        y2 *= &y_abs;
        let mut re = x2;
        re -= y2;
        re += g.seed.real();

        let mut im = x;
        im *= y_abs;
        im *= Float::with_val(g.prec, -2.0);
        im += g.seed.imag();

        z = complex_from_xy(g.prec, re, im);
        i += 1;
    }
    (i, z)
}

fn alpha_mandelbrot_julia_mpc(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
    let mut z = z_pixel.clone();
    let mut i = 0u32;
    while i < g.iteration_max && mpc_below_bailout(&z, g) {
        let mut z_sq = z.clone();
        z_sq *= &z;
        let mut m = z_sq.clone();
        m += &g.seed;
        let mut m_sq = m.clone();
        m_sq *= &m;
        let mut z_next = z_sq;
        z_next += &m_sq;
        z_next += &g.seed;
        z = z_next;
        i += 1;
    }
    (i, z)
}

#[cfg(test)]
mod mpc_loop_tests {
    use super::*;
    use crate::fractal::definitions::default_params_for_type;

    fn same(a: &Complex, b: &Complex) -> bool {
        let f = |x: &Float, y: &Float| (x.is_nan() && y.is_nan()) || (x == y && x.prec() == y.prec());
        f(a.real(), b.real()) && f(a.imag(), b.imag())
    }

    /// Les boucles d'avant le 2026-11-06, mot pour mot : norme GMP exacte à
    /// chaque tour, `clone` + `mpc_mul`.
    fn reference(g: &MpcParams, z_pixel: &Complex) -> (u32, Complex) {
        let exact = |z: &Complex| complex_norm_sqr(z, g.prec) < g.bailout_sqr;
        let mut i = 0u32;
        match g.fractal_type {
            FractalType::Mandelbrot => {
                let mut z = g.seed.clone();
                while i < g.iteration_max && exact(&z) {
                    let mut z_sq = z.clone();
                    z_sq *= &z;
                    z_sq += z_pixel;
                    z = z_sq;
                    i += 1;
                }
                (i, z)
            }
            FractalType::Julia => {
                let mut z = z_pixel.clone();
                while i < g.iteration_max && exact(&z) {
                    let mut z_sq = z.clone();
                    z_sq *= &z;
                    z_sq += &g.seed;
                    z = z_sq;
                    i += 1;
                }
                (i, z)
            }
            FractalType::Tricorn => {
                let mut z = g.seed.clone();
                while i < g.iteration_max && exact(&z) {
                    let z_conj = z.clone().conj();
                    let mut z_temp = z_conj.clone();
                    z_temp *= &z_conj;
                    z_temp += z_pixel;
                    z = z_temp;
                    i += 1;
                }
                (i, z)
            }
            FractalType::BurningShip => {
                let mut z = g.seed.clone();
                while i < g.iteration_max && exact(&z) {
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
            other => panic!("pas de référence pour {other:?}"),
        }
    }

    /// Verrou : les boucles GMP allégées (test d'évasion en f64 d'abord,
    /// carré en place) rendent le MÊME compte d'itérations et le MÊME z final,
    /// au bit près, que les boucles d'origine — elles sont le juge de la QA.
    #[test]
    fn in_place_mpc_loops_match_original_loops_bit_for_bit() {
        for ty in [FractalType::Mandelbrot, FractalType::Julia, FractalType::Tricorn, FractalType::BurningShip] {
            for prec in [64u32, 128, 300, 1024] {
                // Plafonds élevés à basse précision : des points intérieurs y
                // atteignent un cycle exact, le raccourci saute.
                let maxes: &[u32] = if prec <= 128 { &[0, 1, 2, 50, 400, 3001, 20_000] } else { &[0, 1, 2, 50, 400] };
                for &max in maxes {
                    let mut params = default_params_for_type(ty, 16, 12);
                    params.iteration_max = max;
                    params.engine.precision_bits = prec;
                    params.engine.use_gmp = true;
                    params.engine.algorithm_mode = crate::fractal::AlgorithmMode::ReferenceGmp;
                    let g = MpcParams::from_params(&params);
                    for j in 0..12 {
                        for i in 0..16 {
                            let x = params.center_x + params.span_x * ((i as f64 + 0.5) / 16.0 - 0.5);
                            let y = params.center_y + params.span_y * ((j as f64 + 0.5) / 12.0 - 0.5);
                            let p = Complex::with_val(g.prec, (x, y));
                            let (ia, za) = iterate_point_mpc(&g, &p);
                            let (ib, zb) = reference(&g, &p);
                            assert!(
                                ia == ib && same(&za, &zb),
                                "{ty:?} prec={prec} max={max} point=({x},{y}) : {ia} ≠ {ib} ou z ≠"
                            );
                        }
                    }
                }
            }
        }
    }

    /// `sqr_in_place` rend, composante par composante, EXACTEMENT ce que
    /// rend `mpc_sqr` (et `mpc_mul(z, z)`) : valeurs aléatoires, précisions
    /// variées, composantes presque égales (annulation de x² − y²), zéros
    /// signés, exposants très différents, précisions re ≠ im.
    #[test]
    fn in_place_square_matches_mpc_sqr() {
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut checked = 0u32;
        for prec in [53u32, 64, 100, 128, 256, 300, 1000, 2048] {
            for k in 0..400u32 {
                let mut z = Complex::with_val(prec, (rnd() * 4.0 - 2.0, rnd() * 4.0 - 2.0));
                // Remplit toute la mantisse (pas seulement 53 bits).
                let t = Complex::with_val(prec, (rnd(), rnd()));
                z *= &t;
                z += Complex::with_val(prec, (rnd() - 0.5, rnd() - 0.5));
                match k % 8 {
                    1 => {
                        // x ≈ y à un souffle près : annulation massive.
                        let x = z.real().clone();
                        let mut y = x.clone();
                        y += Float::with_val(prec, rnd()) >> (prec - 3);
                        z = Complex::with_val(prec, (x, y));
                    }
                    2 => z = Complex::with_val(prec, (z.real(), -z.real().clone())),
                    3 => z = Complex::with_val(prec, (0.0, z.imag())),
                    4 => z = Complex::with_val(prec, (z.real(), -0.0)),
                    5 => {
                        let small = z.imag().clone() >> 500u32;
                        z = Complex::with_val(prec, (z.real(), small));
                    }
                    6 => z = Complex::with_val((prec, prec + 37), (z.real(), z.imag())),
                    _ => {}
                }
                let mut a = z.clone();
                let mut sq = sqr_scratch(&a);
                sqr_in_place(&mut a, &mut sq);
                let mut b = z.clone();
                b.square_mut();
                let mut c = z.clone();
                c *= &z;
                let eq = |u: &Complex, v: &Complex| {
                    same(u, v)
                        && u.real().is_sign_negative() == v.real().is_sign_negative()
                        && u.imag().is_sign_negative() == v.imag().is_sign_negative()
                };
                assert!(eq(&a, &b), "prec={prec} k={k} z={z:?} : en place ≠ mpc_sqr");
                assert!(same(&a, &c), "prec={prec} k={k} z={z:?} : en place ≠ mpc_mul");
                checked += 1;
            }
        }
        assert_eq!(checked, 8 * 400);
    }

    /// Le test d'évasion en deux temps décide comme la norme exacte, y
    /// compris au ras du rayon, sur le rayon, et hors de la plage f64.
    #[test]
    fn two_stage_bailout_test_decides_like_the_exact_norm() {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 8, 8);
        params.engine.precision_bits = 256;
        params.engine.use_gmp = true;
        params.engine.algorithm_mode = crate::fractal::AlgorithmMode::ReferenceGmp;
        let g = MpcParams::from_params(&params);
        let r = params.bailout;
        let mut cases: Vec<Complex> = Vec::new();
        let eps = Float::with_val(g.prec, 1) >> 200u32;
        for (a, b) in [(0.0, 0.0), (r, 0.0), (0.0, r), (r * 0.6, r * 0.8), (-r, 0.0), (r * 0.995, 0.0), (r * 1.004, 0.0), (1e200, 1.0), (1e-300, 1e-310), (3.0, 4.0)] {
            let base = Complex::with_val(g.prec, (a, b));
            cases.push(base.clone());
            // Juste sous / juste au-dessus, d'un écart invisible en f64.
            let mut lo = base.clone();
            *lo.mut_real() -= &eps;
            cases.push(lo);
            let mut hi = base;
            *hi.mut_real() += &eps;
            cases.push(hi);
        }
        cases.push(Complex::with_val(g.prec, (f64::NAN, 0.0)));
        cases.push(Complex::with_val(g.prec, (f64::INFINITY, 0.0)));
        cases.push(Complex::with_val(g.prec, (0.0, f64::NEG_INFINITY)));
        for z in &cases {
            assert_eq!(
                mpc_below_bailout(z, &g),
                complex_norm_sqr(z, g.prec) < g.bailout_sqr,
                "décision ≠ pour z = {z:?}"
            );
        }
    }
}

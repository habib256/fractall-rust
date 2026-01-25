use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rug::{Complex, Float};

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::gmp::complex_to_complex64;
use crate::fractal::perturbation::bla::{BlaTable, build_bla_table};

#[derive(Clone, Debug)]
pub struct ReferenceOrbit {
    pub cref: Complex64,
    pub z_ref: Vec<Complex64>,
}

/// Cache for reference orbit and BLA table to avoid recomputation between frames.
#[derive(Clone, Debug)]
pub struct ReferenceOrbitCache {
    pub orbit: ReferenceOrbit,
    pub bla_table: BlaTable,
    /// Center X as string for arbitrary precision comparison
    pub center_x: String,
    /// Center Y as string for arbitrary precision comparison
    pub center_y: String,
    pub fractal_type: FractalType,
    pub precision_bits: u32,
    pub iteration_max: u32,
    /// Julia seed (for Julia-type fractals)
    pub seed_re: f64,
    pub seed_im: f64,
    /// BLA threshold used when building the table
    pub bla_threshold: f64,
}

impl ReferenceOrbitCache {
    /// Check if the cache is valid for the given parameters.
    /// The cache is valid if: same center, same type, same precision, iteration_max <= cached.
    pub fn is_valid_for(&self, params: &FractalParams) -> bool {
        let center_x = (params.xmin + params.xmax) / 2.0;
        let center_y = (params.ymin + params.ymax) / 2.0;

        // Compare center as strings for exact matching
        let cx_str = format!("{:.16e}", center_x);
        let cy_str = format!("{:.16e}", center_y);

        self.fractal_type == params.fractal_type
            && self.center_x == cx_str
            && self.center_y == cy_str
            && self.precision_bits >= params.precision_bits
            && self.iteration_max >= params.iteration_max
            && (self.seed_re - params.seed.re).abs() < 1e-15
            && (self.seed_im - params.seed.im).abs() < 1e-15
            && (self.bla_threshold - params.bla_threshold).abs() < 1e-20
    }

    /// Create a new cache from computed orbit and BLA table.
    pub fn new(
        orbit: ReferenceOrbit,
        bla_table: BlaTable,
        params: &FractalParams,
    ) -> Self {
        let center_x = (params.xmin + params.xmax) / 2.0;
        let center_y = (params.ymin + params.ymax) / 2.0;

        Self {
            orbit,
            bla_table,
            center_x: format!("{:.16e}", center_x),
            center_y: format!("{:.16e}", center_y),
            fractal_type: params.fractal_type,
            precision_bits: params.precision_bits,
            iteration_max: params.iteration_max,
            seed_re: params.seed.re,
            seed_im: params.seed.im,
            bla_threshold: params.bla_threshold,
        }
    }
}

/// Compute the reference orbit and BLA table, using cache if available.
pub fn compute_reference_orbit_cached(
    params: &FractalParams,
    cancel: Option<&AtomicBool>,
    cache: Option<&Arc<ReferenceOrbitCache>>,
) -> Option<Arc<ReferenceOrbitCache>> {
    // Check if cache is valid
    if let Some(cached) = cache {
        if cached.is_valid_for(params) {
            return Some(Arc::clone(cached));
        }
    }

    // Compute fresh orbit and BLA table
    let orbit = compute_reference_orbit(params, cancel)?;
    let bla_table = build_bla_table(&orbit.z_ref, params);

    Some(Arc::new(ReferenceOrbitCache::new(orbit, bla_table, params)))
}

pub fn compute_reference_orbit(
    params: &FractalParams,
    cancel: Option<&AtomicBool>,
) -> Option<ReferenceOrbit> {
    let prec = params.precision_bits.max(64);
    let center_x = (params.xmin + params.xmax) / 2.0;
    let center_y = (params.ymin + params.ymax) / 2.0;
    let cref = Complex::with_val(prec, (center_x, center_y));
    let cref_f64 = Complex64::new(center_x, center_y);
    let mut z = match params.fractal_type {
        FractalType::Mandelbrot | FractalType::BurningShip => {
            Complex::with_val(prec, (params.seed.re, params.seed.im))
        }
        FractalType::Julia => Complex::with_val(prec, (center_x, center_y)),
        _ => return None,
    };
    let seed = Complex::with_val(prec, (params.seed.re, params.seed.im));

    let bailout = Float::with_val(prec, params.bailout);
    let mut bailout_sqr = bailout.clone();
    bailout_sqr *= &bailout;

    let mut z_ref = Vec::with_capacity(params.iteration_max as usize + 1);
    z_ref.push(complex_to_complex64(&z));

    for i in 0..params.iteration_max {
        if let Some(cancel) = cancel {
            if i % 256 == 0 && cancel.load(Ordering::Relaxed) {
                return None;
            }
        }
        if complex_norm_sqr(&z, prec) > bailout_sqr {
            break;
        }
        z = match params.fractal_type {
            FractalType::Mandelbrot => {
                let mut z_sq = z.clone();
                z_sq *= &z;
                z_sq += &cref;
                z_sq
            }
            FractalType::Julia => {
                let mut z_sq = z.clone();
                z_sq *= &z;
                z_sq += &seed;
                z_sq
            }
            FractalType::BurningShip => {
                let re_abs = z.real().clone().abs();
                let im_abs = z.imag().clone().abs();
                let mut z_abs = Complex::with_val(prec, (re_abs, im_abs));
                z_abs *= z_abs.clone();
                z_abs += &cref;
                z_abs
            }
            _ => return None,
        };
        z_ref.push(complex_to_complex64(&z));
    }

    Some(ReferenceOrbit { cref: cref_f64, z_ref })
}

fn complex_norm_sqr(value: &Complex, prec: u32) -> Float {
    let mut re2 = value.real().clone();
    re2 *= value.real();
    let mut im2 = value.imag().clone();
    im2 *= value.imag();
    let mut sum = Float::with_val(prec, re2);
    sum += im2;
    sum
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rug::{Complex, Float};

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::gmp::{complex_to_complex64, pow_f64_mpc};
use crate::fractal::perturbation::bla::{BlaTable, build_bla_table};
use crate::fractal::perturbation::types::ComplexExp;
use crate::fractal::perturbation::series::{SeriesTable, build_series_table};

#[derive(Clone, Debug)]
pub struct ReferenceOrbit {
    pub cref: Complex64,
    /// High precision reference orbit (extended exponent range via FloatExp)
    pub z_ref: Vec<ComplexExp>,
    /// Fast path: f64 version of reference orbit for shallow zooms
    pub z_ref_f64: Vec<Complex64>,
}

/// Cache for reference orbit and BLA table to avoid recomputation between frames.
#[derive(Clone, Debug)]
pub struct ReferenceOrbitCache {
    pub orbit: ReferenceOrbit,
    pub bla_table: BlaTable,
    /// Standalone series table for iteration skipping (optional)
    pub series_table: Option<SeriesTable>,
    /// Center X in GMP precision (stored as string for Clone/Debug)
    pub center_x_gmp: String,
    /// Center Y in GMP precision (stored as string for Clone/Debug)
    pub center_y_gmp: String,
    pub fractal_type: FractalType,
    pub precision_bits: u32,
    pub iteration_max: u32,
    /// Julia seed (for Julia-type fractals)
    pub seed_re: f64,
    pub seed_im: f64,
    /// BLA threshold used when building the table
    pub bla_threshold: f64,
    /// BLA validity scale used when building the table
    pub bla_validity_scale: f64,
}

impl ReferenceOrbitCache {
    /// Check if the cache is valid for the given parameters.
    /// The cache is valid if: same center (GMP precision), same type, precision >= required, iteration_max >= required.
    pub fn is_valid_for(&self, params: &FractalParams) -> bool {
        // Compute center in GMP precision for exact comparison
        let prec = params.precision_bits.max(self.precision_bits).max(128);
        let center_x = Float::with_val(prec, params.center_x);
        let center_y = Float::with_val(prec, params.center_y);

        // Compare as GMP strings with full precision
        let cx_str = center_x.to_string_radix(10, None);
        let cy_str = center_y.to_string_radix(10, None);

        self.fractal_type == params.fractal_type
            && self.center_x_gmp == cx_str
            && self.center_y_gmp == cy_str
            && self.precision_bits >= params.precision_bits
            && self.iteration_max >= params.iteration_max
            && (self.seed_re - params.seed.re).abs() < 1e-15
            && (self.seed_im - params.seed.im).abs() < 1e-15
            && (self.bla_threshold - params.bla_threshold).abs() < 1e-20
            && (self.bla_validity_scale - params.bla_validity_scale).abs() < 1e-10
    }

    /// Create a new cache from computed orbit and BLA table.
    pub fn new(
        orbit: ReferenceOrbit,
        bla_table: BlaTable,
        series_table: Option<SeriesTable>,
        params: &FractalParams,
        center_x_gmp: String,
        center_y_gmp: String,
    ) -> Self {
        Self {
            orbit,
            bla_table,
            series_table,
            center_x_gmp,
            center_y_gmp,
            fractal_type: params.fractal_type,
            precision_bits: params.precision_bits,
            iteration_max: params.iteration_max,
            seed_re: params.seed.re,
            seed_im: params.seed.im,
            bla_threshold: params.bla_threshold,
            bla_validity_scale: params.bla_validity_scale,
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
    let (orbit, center_x_gmp, center_y_gmp) = compute_reference_orbit(params, cancel)?;
    // Use z_ref_f64 for BLA table building (BLA works with f64 coefficients)
    let bla_table = build_bla_table(&orbit.z_ref_f64, params);

    // Build series table for standalone series approximation (if enabled)
    // Only for Mandelbrot and Julia; Burning Ship has abs() which breaks series
    let series_table = if params.series_standalone
        && matches!(params.fractal_type, FractalType::Mandelbrot | FractalType::Julia)
    {
        Some(build_series_table(&orbit.z_ref_f64))
    } else {
        None
    };

    Some(Arc::new(ReferenceOrbitCache::new(
        orbit,
        bla_table,
        series_table,
        params,
        center_x_gmp,
        center_y_gmp,
    )))
}

/// Compute reference orbit in GMP precision.
/// Returns (orbit, center_x_gmp_string, center_y_gmp_string).
pub fn compute_reference_orbit(
    params: &FractalParams,
    cancel: Option<&AtomicBool>,
) -> Option<(ReferenceOrbit, String, String)> {
    let prec = params.precision_bits.max(128);

    // Use center directly (no need to compute from xmin/xmax anymore)
    let center_x_gmp = Float::with_val(prec, params.center_x);
    let center_y_gmp = Float::with_val(prec, params.center_y);

    // Store GMP strings for cache validation
    let cx_str = center_x_gmp.to_string_radix(10, None);
    let cy_str = center_y_gmp.to_string_radix(10, None);

    let cref = Complex::with_val(prec, (&center_x_gmp, &center_y_gmp));
    let cref_f64 = Complex64::new(center_x_gmp.to_f64(), center_y_gmp.to_f64());

    let mut z = match params.fractal_type {
        FractalType::Mandelbrot | FractalType::BurningShip | FractalType::Multibrot | FractalType::Tricorn => {
            Complex::with_val(prec, (params.seed.re, params.seed.im))
        }
        FractalType::Julia => cref.clone(),
        _ => return None,
    };
    let seed = Complex::with_val(prec, (params.seed.re, params.seed.im));

    let bailout = Float::with_val(prec, params.bailout);
    let mut bailout_sqr = bailout.clone();
    bailout_sqr *= &bailout;

    let mut z_ref = Vec::with_capacity(params.iteration_max as usize + 1);
    let mut z_ref_f64 = Vec::with_capacity(params.iteration_max as usize + 1);
    // Store both high-precision and f64 versions
    z_ref.push(ComplexExp::from_gmp(&z));
    z_ref_f64.push(complex_to_complex64(&z));

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
            FractalType::Multibrot => {
                let mut z_pow = pow_f64_mpc(&z, params.multibrot_power, prec);
                z_pow += &cref;
                z_pow
            }
            FractalType::Tricorn => {
                // Tricorn: z' = conj(z)² + c
                let z_conj = z.clone().conj();
                let mut z_temp = z_conj.clone();
                z_temp *= &z_conj;
                z_temp += &cref;
                z_temp
            }
            _ => return None,
        };
        // Store both high-precision and f64 versions
        z_ref.push(ComplexExp::from_gmp(&z));
        z_ref_f64.push(complex_to_complex64(&z));
    }

    Some((ReferenceOrbit { cref: cref_f64, z_ref, z_ref_f64 }, cx_str, cy_str))
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

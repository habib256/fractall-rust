use std::sync::atomic::{AtomicBool, Ordering};

use num_complex::Complex64;
use rug::{Complex, Float};

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::gmp::complex_to_complex64;

#[derive(Clone, Debug)]
pub struct ReferenceOrbit {
    pub cref: Complex64,
    pub z_ref: Vec<Complex64>,
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

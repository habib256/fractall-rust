use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::FractalParams;
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::perturbation::bla::build_bla_table;
use crate::fractal::perturbation::delta::iterate_pixel;
use crate::fractal::perturbation::orbit::compute_reference_orbit;
use crate::fractal::perturbation::types::ComplexExp;

pub mod types;
pub mod orbit;
pub mod bla;
pub mod delta;

#[allow(dead_code)]
pub fn render_mandelbrot_perturbation(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let cancel = Arc::new(AtomicBool::new(false));
    render_mandelbrot_perturbation_cancellable_with_reuse(params, &cancel, None)
        .unwrap_or_else(|| (Vec::new(), Vec::new()))
}

pub fn render_mandelbrot_perturbation_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    _reuse: Option<(&[u32], &[Complex64], u32, u32)>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    let ref_orbit = compute_reference_orbit(params, Some(cancel.as_ref()))?;
    let bla_table = build_bla_table(&ref_orbit.z_ref, params);
    let gmp_params = MpcParams::from_params(params);
    let prec = gmp_params.prec;

    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return Some((iterations, zs));
    }

    let x_range = params.xmax - params.xmin;
    let y_range = params.ymax - params.ymin;
    let x_step = x_range / params.width as f64;
    let y_step = y_range / params.height as f64;

    let cancelled = AtomicBool::new(false);
    let ref_orbit = Arc::new(ref_orbit);
    let bla_table = Arc::new(bla_table);

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            if j % 16 == 0 && cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }

            let yg = y_step * j as f64 + params.ymin;
            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let xg = x_step * i as f64 + params.xmin;
                let z_pixel = Complex64::new(xg, yg);
                let dc = ComplexExp::from_complex64(z_pixel - ref_orbit.cref);

                let result = iterate_pixel(params, &ref_orbit, &bla_table, dc);
                if result.glitched {
                    let z_pixel_mpc = complex_from_xy(
                        prec,
                        Float::with_val(prec, xg),
                        Float::with_val(prec, yg),
                    );
                    let (iter_val, z_final) = iterate_point_mpc(&gmp_params, &z_pixel_mpc);
                    *iter = iter_val;
                    *z = complex_to_complex64(&z_final);
                } else {
                    *iter = result.iteration;
                    *z = result.z_final;
                }
            }
        });

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        Some((iterations, zs))
    }
}

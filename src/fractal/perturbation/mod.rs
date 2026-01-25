use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::perturbation::delta::iterate_pixel;
use crate::fractal::perturbation::orbit::compute_reference_orbit_cached;
use crate::fractal::perturbation::types::ComplexExp;

pub mod types;
pub mod orbit;
pub mod bla;
pub mod delta;

pub use orbit::ReferenceOrbitCache;

struct ReuseData<'a> {
    iterations: &'a [u32],
    zs: &'a [Complex64],
    width: u32,
    ratio: u32,
}

fn build_reuse<'a>(
    params: &FractalParams,
    reuse: Option<(&'a [u32], &'a [Complex64], u32, u32)>,
) -> Option<ReuseData<'a>> {
    let (iterations, zs, width, height) = reuse?;
    if width == 0 || height == 0 {
        return None;
    }
    let expected_len = (width * height) as usize;
    if iterations.len() != expected_len || zs.len() != expected_len {
        return None;
    }
    if params.width % width != 0 || params.height % height != 0 {
        return None;
    }
    let ratio_x = params.width / width;
    let ratio_y = params.height / height;
    if ratio_x < 2 || ratio_y < 2 || ratio_x != ratio_y {
        return None;
    }
    Some(ReuseData {
        iterations,
        zs,
        width,
        ratio: ratio_x,
    })
}

pub(crate) fn compute_perturbation_precision_bits(params: &FractalParams) -> u32 {
    if params.width == 0 || params.height == 0 {
        return params.precision_bits.max(64);
    }
    let span_x = (params.xmax - params.xmin).abs();
    let span_y = (params.ymax - params.ymin).abs();
    let pixel_size = span_x.max(span_y) / params.width as f64;
    if !pixel_size.is_finite() || pixel_size <= 0.0 {
        return params.precision_bits.max(64);
    }
    let center_x = (params.xmin + params.xmax) / 2.0;
    let center_y = (params.ymin + params.ymax) / 2.0;
    let scale = center_x.abs().max(center_y.abs()).max(1.0);
    let zoom = (scale / pixel_size).abs();
    let zoom_threshold = 1e13;
    if !zoom.is_finite() || zoom <= zoom_threshold {
        return params.precision_bits.max(64);
    }
    let needed_bits = if zoom > 0.0 {
        zoom.log2().ceil() as i32 + 32
    } else {
        0
    };
    let needed_bits = needed_bits.max(64) as u32;
    params
        .precision_bits
        .max(needed_bits)
        .min(4096)
}

#[allow(dead_code)]
pub fn render_mandelbrot_perturbation(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let cancel = Arc::new(AtomicBool::new(false));
    render_perturbation_cancellable_with_reuse(params, &cancel, None)
        .unwrap_or_else(|| (Vec::new(), Vec::new()))
}

pub fn render_perturbation_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<(&[u32], &[Complex64], u32, u32)>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let (result, _cache) = render_perturbation_with_cache(params, cancel, reuse, None)?;
    Some(result)
}

/// Render perturbation with optional orbit cache support.
/// Returns the result and the updated cache for reuse in subsequent frames.
pub fn render_perturbation_with_cache(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<(&[u32], &[Complex64], u32, u32)>,
    orbit_cache: Option<&Arc<ReferenceOrbitCache>>,
) -> Option<((Vec<u32>, Vec<Complex64>), Arc<ReferenceOrbitCache>)> {
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let supports = matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
    );
    if !supports {
        return None;
    }

    let mut orbit_params = params.clone();
    orbit_params.precision_bits = compute_perturbation_precision_bits(params);

    // Use cached orbit/BLA or compute fresh
    let cache =
        compute_reference_orbit_cached(&orbit_params, Some(cancel.as_ref()), orbit_cache)?;
    let gmp_params = MpcParams::from_params(&orbit_params);
    let prec = gmp_params.prec;

    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return Some(((iterations, zs), cache));
    }

    let x_range = params.xmax - params.xmin;
    let y_range = params.ymax - params.ymin;
    let x_step = x_range / params.width as f64;
    let y_step = y_range / params.height as f64;

    let cancelled = AtomicBool::new(false);
    let reuse = build_reuse(params, reuse);

    // Clone cache for use in parallel iteration
    let cache_ref = Arc::clone(&cache);

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let reuse_row = reuse.as_ref();
            if j % 16 == 0 && cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }

            let yg = y_step * j as f64 + params.ymin;
            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                if let Some(reuse) = reuse_row {
                    let ratio = reuse.ratio as usize;
                    if j % ratio == 0 && i % ratio == 0 {
                        let src_x = i / ratio;
                        let src_y = j / ratio;
                        let src_idx = (src_y * reuse.width as usize + src_x) as usize;
                        if src_idx < reuse.iterations.len() {
                            *iter = reuse.iterations[src_idx];
                            *z = reuse.zs[src_idx];
                            continue;
                        }
                    }
                }
                let xg = x_step * i as f64 + params.xmin;
                let z_pixel = Complex64::new(xg, yg);
                let dc = ComplexExp::from_complex64(z_pixel - cache_ref.orbit.cref);
                let (delta0, dc_term) = if params.fractal_type == FractalType::Julia {
                    (dc, ComplexExp::zero())
                } else {
                    (ComplexExp::zero(), dc)
                };

                let result = iterate_pixel(params, &cache_ref.orbit, &cache_ref.bla_table, delta0, dc_term);
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
        Some(((iterations, zs), cache))
    }
}

#[cfg(test)]
mod tests {
    use super::render_perturbation_cancellable_with_reuse;
    use crate::fractal::iterations::iterate_point;
    use crate::fractal::{AlgorithmMode, FractalParams, FractalType};
    use num_complex::Complex64;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn base_params(fractal_type: FractalType) -> FractalParams {
        FractalParams {
            width: 5,
            height: 5,
            xmin: -2.0,
            xmax: 2.0,
            ymin: -1.5,
            ymax: 1.5,
            seed: Complex64::new(0.0, 0.0),
            iteration_max: 64,
            bailout: 4.0,
            fractal_type,
            color_mode: 0,
            color_repeat: 2,
            use_gmp: false,
            precision_bits: 192,
            algorithm_mode: AlgorithmMode::Perturbation,
            bla_threshold: 1e-6,
            glitch_tolerance: 1e-4,
            lyapunov_preset: Default::default(),
            lyapunov_sequence: Vec::new(),
        }
    }

    fn assert_close_iterations(params: &FractalParams, indices: &[(u32, u32)]) {
        let cancel = Arc::new(AtomicBool::new(false));
        let (iters, _) =
            render_perturbation_cancellable_with_reuse(params, &cancel, None).unwrap();
        let x_step = (params.xmax - params.xmin) / params.width as f64;
        let y_step = (params.ymax - params.ymin) / params.height as f64;
        for &(x, y) in indices {
            let idx = (y * params.width + x) as usize;
            let xg = x_step * x as f64 + params.xmin;
            let yg = y_step * y as f64 + params.ymin;
            let z_pixel = Complex64::new(xg, yg);
            let ref_iter = iterate_point(params, z_pixel).iteration;
            let got = iters[idx];
            let diff = (got as i32 - ref_iter as i32).abs();
            assert!(diff <= 1, "iter mismatch: got {got}, ref {ref_iter}");
        }
    }

    #[test]
    fn perturbation_matches_f64_mandelbrot() {
        let mut params = base_params(FractalType::Mandelbrot);
        params.xmin = -2.5;
        params.xmax = 1.5;
        assert_close_iterations(&params, &[(0, 0), (2, 2), (4, 4)]);
    }

    #[test]
    fn perturbation_matches_f64_julia() {
        let mut params = base_params(FractalType::Julia);
        params.seed = Complex64::new(0.36228, -0.0777);
        assert_close_iterations(&params, &[(1, 1), (2, 2), (3, 3)]);
    }

    #[test]
    fn perturbation_matches_f64_burning_ship() {
        let mut params = base_params(FractalType::BurningShip);
        params.xmin = -2.5;
        params.xmax = 1.5;
        params.ymin = -2.0;
        params.ymax = 2.0;
        assert_close_iterations(&params, &[(0, 4), (2, 2), (4, 0)]);
    }
}

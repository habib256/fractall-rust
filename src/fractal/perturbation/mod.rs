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
pub mod series;

pub use orbit::ReferenceOrbitCache;

pub fn mark_neighbor_glitches(
    iterations: &[u32],
    width: u32,
    height: u32,
    threshold: u32,
) -> Vec<bool> {
    let size = (width * height) as usize;
    let mut mask = vec![false; size];
    if width < 3 || height < 3 || iterations.len() != size {
        return mask;
    }
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = (y * width + x) as usize;
            let center = iterations[idx];
            let left = iterations[(y * width + (x - 1)) as usize];
            let right = iterations[(y * width + (x + 1)) as usize];
            let up = iterations[((y - 1) * width + x) as usize];
            let down = iterations[((y + 1) * width + x) as usize];
            let mut max_diff = center.abs_diff(left);
            max_diff = max_diff.max(center.abs_diff(right));
            max_diff = max_diff.max(center.abs_diff(up));
            max_diff = max_diff.max(center.abs_diff(down));
            if max_diff > threshold {
                mask[idx] = true;
            }
        }
    }
    mask
}

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
        return params.precision_bits.max(128);
    }
    let span_x = (params.xmax - params.xmin).abs();
    let span_y = (params.ymax - params.ymin).abs();
    let pixel_size = span_x.max(span_y) / params.width as f64;
    if !pixel_size.is_finite() || pixel_size <= 0.0 {
        return params.precision_bits.max(128);
    }

    // Compute zoom level from pixel size (base range ~4.0 for Mandelbrot)
    let base_range = 4.0;
    let zoom = base_range / pixel_size;
    if !zoom.is_finite() || zoom <= 1.0 {
        return params.precision_bits.max(128);
    }

    // Bits needed = log2(zoom) + safety margin
    // For deep zooms: need ~3.32 bits per decimal digit of zoom
    // Add 64 bits safety margin for intermediate calculations
    let zoom_bits = zoom.log2().ceil() as i32;
    let needed_bits = (zoom_bits + 64).max(128) as u32;

    // Clamp to reasonable range: 128 minimum, 8192 maximum
    params
        .precision_bits
        .max(needed_bits)
        .clamp(128, 8192)
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
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];
    let glitch_mask: Vec<AtomicBool> = (0..width * height)
        .map(|_| AtomicBool::new(false))
        .collect();

    if width == 0 || height == 0 {
        return Some(((iterations, zs), cache));
    }

    let x_range = params.xmax - params.xmin;
    let y_range = params.ymax - params.ymin;
    let x_step = x_range / params.width as f64;
    let y_step = y_range / params.height as f64;
    let x_half = x_range * 0.5;
    let y_half = y_range * 0.5;

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

            let dy = y_step * j as f64 - y_half;
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
                let dx = x_step * i as f64 - x_half;
                let dc = ComplexExp::from_complex64(Complex64::new(dx, dy));
                let (delta0, dc_term) = if params.fractal_type == FractalType::Julia {
                    (dc, ComplexExp::zero())
                } else {
                    (ComplexExp::zero(), dc)
                };

                let result = iterate_pixel(
                    params,
                    &cache_ref.orbit,
                    &cache_ref.bla_table,
                    delta0,
                    dc_term,
                );
                *iter = result.iteration;
                *z = result.z_final;
                if result.glitched || result.suspect {
                    glitch_mask[j * width + i].store(true, Ordering::Relaxed);
                }
            }
        });

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        let mut glitch_mask: Vec<bool> = glitch_mask
            .iter()
            .map(|flag| flag.load(Ordering::Relaxed))
            .collect();

        if params.glitch_neighbor_pass {
            let neighbor_threshold = (params.iteration_max / 50).max(8);
            let neighbor_mask = mark_neighbor_glitches(
                &iterations,
                params.width,
                params.height,
                neighbor_threshold,
            );
            for (idx, flagged) in neighbor_mask.into_iter().enumerate() {
                if flagged {
                    glitch_mask[idx] = true;
                }
            }
        }

        let glitched_indices: Vec<usize> = glitch_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, flagged)| if *flagged { Some(idx) } else { None })
            .collect();
        if !glitched_indices.is_empty() {
            let gmp_params = MpcParams::from_params(&orbit_params);
            let prec = gmp_params.prec;
            let width_u32 = params.width;
            let x_range = params.xmax - params.xmin;
            let y_range = params.ymax - params.ymin;
            let x_step = x_range / params.width as f64;
            let y_step = y_range / params.height as f64;
            let x_half = x_range * 0.5;
            let y_half = y_range * 0.5;
            let center_x = (params.xmin + params.xmax) * 0.5;
            let center_y = (params.ymin + params.ymax) * 0.5;

            let corrections: Vec<_> = glitched_indices
                .par_iter()
                .map(|&idx| {
                    let x = (idx as u32 % width_u32) as f64;
                    let y = (idx as u32 / width_u32) as f64;
                    let dx = x_step * x - x_half;
                    let dy = y_step * y - y_half;
                    let xg = center_x + dx;
                    let yg = center_y + dy;
                    let z_pixel = complex_from_xy(
                        prec,
                        Float::with_val(prec, xg),
                        Float::with_val(prec, yg),
                    );
                    let (iter_val, z_final) = iterate_point_mpc(&gmp_params, &z_pixel);
                    (idx, iter_val, complex_to_complex64(&z_final))
                })
                .collect();

            for (idx, iter_val, z_final) in corrections {
                iterations[idx] = iter_val;
                zs[idx] = z_final;
            }
        }

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
            series_order: 2,
            series_threshold: 1e-6,
            series_error_tolerance: 1e-9,
            glitch_neighbor_pass: false,
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

    #[test]
    fn neighbor_glitch_detection_marks_outliers() {
        let width = 5;
        let height = 5;
        let mut iterations = vec![10u32; (width * height) as usize];
        iterations[(2 * width + 2) as usize] = 200;
        let mask = super::mark_neighbor_glitches(&iterations, width, height, 50);
        assert!(mask[(2 * width + 2) as usize]);
    }
}

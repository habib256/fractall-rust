use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::{AlgorithmMode, FractalParams, FractalResult, FractalType, PlaneTransform};
use crate::fractal::iterations::iterate_point;
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::{render_lyapunov, render_von_koch, render_dragon, render_buddhabrot, render_nebulabrot};
use crate::fractal::lyapunov::{render_lyapunov_cancellable, render_lyapunov_mpc_cancellable, render_lyapunov_mpc};
use crate::fractal::buddhabrot::{
    render_buddhabrot_cancellable,
    render_nebulabrot_cancellable,
    render_buddhabrot_mpc,
    render_nebulabrot_mpc,
    render_buddhabrot_mpc_cancellable,
    render_nebulabrot_mpc_cancellable,
};
use crate::fractal::perturbation::render_perturbation_cancellable_with_reuse;

/// Calcule la matrice d'itérations et la matrice des valeurs finales de z
/// pour une fractale escape-time (ou algorithme spécial).
///
/// Retourne un tuple (iterations, zs) où :
/// - `iterations.len() == width * height`
/// - `zs.len() == width * height`
///
/// Le calcul est parallélisé sur plusieurs cœurs CPU avec rayon.
#[allow(dead_code)]
pub fn render_escape_time(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    // Dispatch vers les algorithmes spéciaux
    match params.fractal_type {
        FractalType::VonKoch => return render_von_koch(params),
        FractalType::Dragon => return render_dragon(params),
        FractalType::Buddhabrot => {
            return if params.use_gmp {
                render_buddhabrot_mpc(params)
            } else {
                render_buddhabrot(params)
            };
        }
        FractalType::Lyapunov => {
            return if params.use_gmp {
                render_lyapunov_mpc(params)
            } else {
                render_lyapunov(params)
            };
        }
        FractalType::Nebulabrot => {
            return if params.use_gmp {
                render_nebulabrot_mpc(params)
            } else {
                render_nebulabrot(params)
            };
        }
        _ => {}
    }

    if matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
    ) {
        // Perturbation ne supporte pas les transformations de plan (delta-based).
        // Si l'utilisateur force Perturbation avec un plan != Mu, on retombe sur standard.
        if params.plane_transform != PlaneTransform::Mu
            && params.algorithm_mode == AlgorithmMode::Perturbation
        {
            if params.use_gmp {
                return render_escape_time_gmp(params);
            }
            return render_escape_time_f64(params);
        }
        match params.algorithm_mode {
            AlgorithmMode::ReferenceGmp => return render_escape_time_gmp(params),
            AlgorithmMode::StandardF64 | AlgorithmMode::StandardDS => {
                // StandardDS n'existe que sur GPU, sur CPU on utilise f64
                return render_escape_time_f64(params);
            }
            AlgorithmMode::Perturbation => {
                return render_perturbation_cancellable_with_reuse(
                    params,
                    &Arc::new(AtomicBool::new(false)),
                    None,
                )
                .unwrap_or_else(|| (Vec::new(), Vec::new()));
            }
            AlgorithmMode::Auto => {
                if should_use_perturbation(params, false) {
                    return render_perturbation_cancellable_with_reuse(
                        params,
                        &Arc::new(AtomicBool::new(false)),
                        None,
                    )
                    .unwrap_or_else(|| (Vec::new(), Vec::new()));
                }
            }
        }
    }

    if params.use_gmp {
        return render_escape_time_gmp(params);
    }
    render_escape_time_f64(params)
}

#[allow(dead_code)]
fn render_escape_time_f64(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    // Même mapping que Fractal_CalculateMatrix en C :
    // xg = ((xmax - xmin) / xpixel) * i + xmin
    // yg = ((ymax - ymin) / ypixel) * j + ymin
    if width == 0 || height == 0 {
        return (iterations, zs);
    }

    // Utiliser center+span directement pour éviter les problèmes de précision
    // xg = center_x + (i/width - 0.5) * span_x
    // yg = center_y + (j/height - 0.5) * span_y

    // Parallélisation par lignes avec rayon (beaucoup plus élégant que std::thread)
    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let y_ratio = j as f64 / params.height as f64;
            let yg = params.center_y + (y_ratio - 0.5) * params.span_y;
            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let x_ratio = i as f64 / params.width as f64;
                let xg = params.center_x + (x_ratio - 0.5) * params.span_x;
                let z_pixel = Complex64::new(xg, yg);
                let z_pixel = params.plane_transform.transform(z_pixel);
                let FractalResult { iteration, z: z_final, orbit: _ } = iterate_point(params, z_pixel);
                *iter = iteration;
                *z = z_final;
            }
        });

    (iterations, zs)
}

#[allow(dead_code)]
fn render_escape_time_gmp(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return (iterations, zs);
    }

    let gmp = MpcParams::from_params(params);
    let prec = gmp.prec;

    // Utiliser center+span directement pour éviter les problèmes de précision
    let center_x = Float::with_val(prec, params.center_x);
    let center_y = Float::with_val(prec, params.center_y);
    let span_x = Float::with_val(prec, params.span_x);
    let span_y = Float::with_val(prec, params.span_y);

    let width_f = Float::with_val(prec, params.width);
    let height_f = Float::with_val(prec, params.height);
    let half = Float::with_val(prec, 0.5);

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let j_f = Float::with_val(prec, j as u32);
            let mut y_ratio = j_f.clone();
            y_ratio /= &height_f;
            y_ratio -= &half;
            let mut yg = span_y.clone();
            yg *= &y_ratio;
            yg += &center_y;
            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let i_f = Float::with_val(prec, i as u32);
                let mut x_ratio = i_f;
                x_ratio /= &width_f;
                x_ratio -= &half;
                let mut xg = span_x.clone();
                xg *= &x_ratio;
                xg += &center_x;
                // Apply plane transformation using f64 approximation (transform doesn't need GMP precision)
                let z_approx = Complex64::new(xg.to_f64(), yg.to_f64());
                let z_transformed = params.plane_transform.transform(z_approx);
                let z_pixel = complex_from_xy(
                    prec,
                    Float::with_val(prec, z_transformed.re),
                    Float::with_val(prec, z_transformed.im),
                );
                let (iter_val, z_final) = iterate_point_mpc(&gmp, &z_pixel);
                *iter = iter_val;
                *z = complex_to_complex64(&z_final);
            }
        });

    (iterations, zs)
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

/// Version annulable du rendu escape-time.
/// Retourne None si annulé, Some(...) sinon.
#[allow(dead_code)]
pub fn render_escape_time_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_escape_time_cancellable_with_reuse(params, cancel, None)
}

/// Version annulable du rendu escape-time avec réutilisation d'une passe précédente.
/// Les points déjà calculés sont réutilisés quand les résolutions s'alignent.
pub fn render_escape_time_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<(&[u32], &[Complex64], u32, u32)>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    // Vérifier l'annulation avant de commencer
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    // Dispatch vers les algorithmes spéciaux
    match params.fractal_type {
        FractalType::VonKoch => return Some(render_von_koch(params)),
        FractalType::Dragon => return Some(render_dragon(params)),
        FractalType::Buddhabrot => {
            return if params.use_gmp {
                render_buddhabrot_mpc_cancellable(params, cancel)
            } else {
                render_buddhabrot_cancellable(params, cancel)
            };
        }
        FractalType::Lyapunov => {
            return if params.use_gmp {
                render_lyapunov_mpc_cancellable(params, cancel)
            } else {
                render_lyapunov_cancellable(params, cancel)
            };
        }
        FractalType::Nebulabrot => {
            return if params.use_gmp {
                render_nebulabrot_mpc_cancellable(params, cancel)
            } else {
                render_nebulabrot_cancellable(params, cancel)
            };
        }
        _ => {}
    }

    if matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
    ) {
        // Perturbation ne supporte pas les transformations de plan.
        // Si forcé, fallback sur standard (f64 ou GMP suivant use_gmp).
        if params.plane_transform != PlaneTransform::Mu
            && params.algorithm_mode == AlgorithmMode::Perturbation
        {
            let reuse = build_reuse(params, reuse);
            if params.use_gmp {
                return render_escape_time_gmp_cancellable_with_reuse(params, cancel, reuse);
            }
            return render_escape_time_f64_cancellable_with_reuse(params, cancel, reuse);
        }
        match params.algorithm_mode {
            AlgorithmMode::ReferenceGmp => {
                let reuse = build_reuse(params, reuse);
                return render_escape_time_gmp_cancellable_with_reuse(params, cancel, reuse);
            }
            AlgorithmMode::StandardF64 | AlgorithmMode::StandardDS => {
                // StandardDS n'existe que sur GPU, sur CPU on utilise f64
                let reuse = build_reuse(params, reuse);
                return render_escape_time_f64_cancellable_with_reuse(params, cancel, reuse);
            }
            AlgorithmMode::Perturbation => {
                return render_perturbation_cancellable_with_reuse(params, cancel, reuse);
            }
            AlgorithmMode::Auto => {
                if should_use_perturbation(params, false) {
                    return render_perturbation_cancellable_with_reuse(params, cancel, reuse);
                }
            }
        }
    }

    let reuse = build_reuse(params, reuse);
    if params.use_gmp {
        return render_escape_time_gmp_cancellable_with_reuse(params, cancel, reuse);
    }
    render_escape_time_f64_cancellable_with_reuse(params, cancel, reuse)
}

pub fn should_use_perturbation(params: &FractalParams, gpu_f32: bool) -> bool {
    if params.width == 0 || params.height == 0 {
        return false;
    }
    // Disable perturbation for non-Mu plane transforms
    // Perturbation relies on delta-based calculations that don't work correctly with plane transforms
    if params.plane_transform != PlaneTransform::Mu {
        return false;
    }
    if !matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
    ) {
        return false;
    }
    let pixel_size = params.span_x.abs().max(params.span_y.abs()) / params.width as f64;
    
    // Seuil maximum: au-delà de zoom ~4e15, la précision de ComplexExp (mantisse f64)
    // n'est plus suffisante. Forcer GMP pour ces zooms extrêmes.
    const MAX_ZOOM_THRESHOLD: f64 = 1e-15;
    if pixel_size < MAX_ZOOM_THRESHOLD {
        return false;
    }
    
    if gpu_f32 {
        // En mode GPU fp32: basculer sur perturbation pour zoom > e5 (pixel_size < 1e-5)
        const GPU_PERTURBATION_THRESHOLD: f64 = 1e-5;
        return pixel_size < GPU_PERTURBATION_THRESHOLD;
    } else {
        // En mode CPU fp64: basculer sur perturbation pour zoom > e13 (pixel_size < 1e-13)
        const CPU_PERTURBATION_THRESHOLD: f64 = 1e-13;
        return pixel_size < CPU_PERTURBATION_THRESHOLD;
    }
}

#[allow(dead_code)]
fn render_escape_time_f64_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_escape_time_f64_cancellable_with_reuse(params, cancel, None)
}

fn render_escape_time_f64_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<ReuseData<'_>>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return Some((iterations, zs));
    }

    // Utiliser center+span directement pour éviter les problèmes de précision
    // xg = center_x + (i/width - 0.5) * span_x
    // yg = center_y + (j/height - 0.5) * span_y

    // Flag interne pour propager l'annulation aux threads Rayon
    let cancelled = AtomicBool::new(false);

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let reuse_row = reuse.as_ref();
            // Vérifier l'annulation toutes les 16 lignes
            if j % 16 == 0 {
                if cancel.load(Ordering::Relaxed) {
                    cancelled.store(true, Ordering::Relaxed);
                    return;
                }
            }
            // Si déjà annulé, sortir
            if cancelled.load(Ordering::Relaxed) {
                return;
            }

            // Utiliser center+span directement
            let y_ratio = j as f64 / params.height as f64;
            let yg = params.center_y + (y_ratio - 0.5) * params.span_y;
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
                let x_ratio = i as f64 / params.width as f64;
                let xg = params.center_x + (x_ratio - 0.5) * params.span_x;
                let z_pixel = Complex64::new(xg, yg);
                let z_pixel = params.plane_transform.transform(z_pixel);
                let FractalResult { iteration, z: z_final, orbit: _ } = iterate_point(params, z_pixel);
                *iter = iteration;
                *z = z_final;
            }
        });

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        Some((iterations, zs))
    }
}

#[allow(dead_code)]
fn render_escape_time_gmp_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_escape_time_gmp_cancellable_with_reuse(params, cancel, None)
}

fn render_escape_time_gmp_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<ReuseData<'_>>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return Some((iterations, zs));
    }

    let gmp = MpcParams::from_params(params);
    let prec = gmp.prec;

    // Utiliser center+span directement pour éviter les problèmes de précision
    let center_x = Float::with_val(prec, params.center_x);
    let center_y = Float::with_val(prec, params.center_y);
    let span_x = Float::with_val(prec, params.span_x);
    let span_y = Float::with_val(prec, params.span_y);

    let width_f = Float::with_val(prec, params.width);
    let height_f = Float::with_val(prec, params.height);
    let half = Float::with_val(prec, 0.5);

    let cancelled = AtomicBool::new(false);

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let reuse_row = reuse.as_ref();
            if j % 8 == 0 {
                if cancel.load(Ordering::Relaxed) {
                    cancelled.store(true, Ordering::Relaxed);
                    return;
                }
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }

            let j_f = Float::with_val(prec, j as u32);
            let mut y_ratio = j_f.clone();
            y_ratio /= &height_f;
            y_ratio -= &half;
            let mut yg = span_y.clone();
            yg *= &y_ratio;
            yg += &center_y;
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
                let i_f = Float::with_val(prec, i as u32);
                let mut x_ratio = i_f;
                x_ratio /= &width_f;
                x_ratio -= &half;
                let mut xg = span_x.clone();
                xg *= &x_ratio;
                xg += &center_x;
                // Apply plane transformation using f64 approximation (transform doesn't need GMP precision)
                let z_approx = Complex64::new(xg.to_f64(), yg.to_f64());
                let z_transformed = params.plane_transform.transform(z_approx);
                let z_pixel = complex_from_xy(prec, Float::with_val(prec, z_transformed.re), Float::with_val(prec, z_transformed.im));
                let (iter_val, z_final) = iterate_point_mpc(&gmp, &z_pixel);
                *iter = iter_val;
                *z = complex_to_complex64(&z_final);
            }
        });

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        Some((iterations, zs))
    }
}


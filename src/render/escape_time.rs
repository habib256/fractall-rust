use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::{FractalParams, FractalResult, FractalType};
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
    let x_range = params.xmax - params.xmin;
    let y_range = params.ymax - params.ymin;

    if width == 0 || height == 0 {
        return (iterations, zs);
    }

    let x_step = x_range / params.width as f64;
    let y_step = y_range / params.height as f64;

    // Parallélisation par lignes avec rayon (beaucoup plus élégant que std::thread)
    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let yg = y_step * j as f64 + params.ymin;
            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let xg = x_step * i as f64 + params.xmin;
                let z_pixel = Complex64::new(xg, yg);
                let FractalResult { iteration, z: z_final } = iterate_point(params, z_pixel);
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

    let xmin = Float::with_val(prec, params.xmin);
    let xmax = Float::with_val(prec, params.xmax);
    let ymin = Float::with_val(prec, params.ymin);
    let ymax = Float::with_val(prec, params.ymax);

    let mut x_range = xmax.clone();
    x_range -= &xmin;
    let mut y_range = ymax.clone();
    y_range -= &ymin;
    let width_f = Float::with_val(prec, params.width);
    let height_f = Float::with_val(prec, params.height);
    let mut x_step = x_range;
    x_step /= &width_f;
    let mut y_step = y_range;
    y_step /= &height_f;

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let mut yg = y_step.clone();
            yg *= j as u32;
            yg += &ymin;
            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let mut xg = x_step.clone();
                xg *= i as u32;
                xg += &xmin;
                let z_pixel = complex_from_xy(prec, xg, yg.clone());
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

    let reuse = build_reuse(params, reuse);
    if params.use_gmp {
        return render_escape_time_gmp_cancellable_with_reuse(params, cancel, reuse);
    }
    render_escape_time_f64_cancellable_with_reuse(params, cancel, reuse)
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

    let x_range = params.xmax - params.xmin;
    let y_range = params.ymax - params.ymin;

    if width == 0 || height == 0 {
        return Some((iterations, zs));
    }

    let x_step = x_range / params.width as f64;
    let y_step = y_range / params.height as f64;

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
                let FractalResult { iteration, z: z_final } = iterate_point(params, z_pixel);
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

    let xmin = Float::with_val(prec, params.xmin);
    let xmax = Float::with_val(prec, params.xmax);
    let ymin = Float::with_val(prec, params.ymin);
    let ymax = Float::with_val(prec, params.ymax);

    let mut x_range = xmax.clone();
    x_range -= &xmin;
    let mut y_range = ymax.clone();
    y_range -= &ymin;
    let width_f = Float::with_val(prec, params.width);
    let height_f = Float::with_val(prec, params.height);
    let mut x_step = x_range;
    x_step /= &width_f;
    let mut y_step = y_range;
    y_step /= &height_f;

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

            let mut yg = y_step.clone();
            yg *= j as u32;
            yg += &ymin;
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
                let mut xg = x_step.clone();
                xg *= i as u32;
                xg += &xmin;
                let z_pixel = complex_from_xy(prec, xg, yg.clone());
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


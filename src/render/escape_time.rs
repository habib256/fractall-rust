use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::{AlgorithmMode, FractalParams, FractalResult, FractalType, PlaneTransform};
use crate::fractal::iterations::iterate_point;
use crate::fractal::orbit_traps::OrbitData;
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::{render_lyapunov, render_von_koch, render_dragon, render_buddhabrot, render_nebulabrot, render_antibuddhabrot};
use crate::fractal::lyapunov::{render_lyapunov_cancellable, render_lyapunov_mpc_cancellable, render_lyapunov_mpc};
use crate::fractal::buddhabrot::{
    render_buddhabrot_cancellable,
    render_nebulabrot_cancellable,
    render_buddhabrot_mpc,
    render_nebulabrot_mpc,
    render_buddhabrot_mpc_cancellable,
    render_nebulabrot_mpc_cancellable,
    render_antibuddhabrot_cancellable,
    render_antibuddhabrot_mpc,
    render_antibuddhabrot_mpc_cancellable,
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
        FractalType::AntiBuddhabrot => {
            return if params.use_gmp {
                render_antibuddhabrot_mpc(params)
            } else {
                render_antibuddhabrot(params)
            };
        }
        _ => {}
    }

    if matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip | FractalType::Tricorn
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
            AlgorithmMode::StandardF64 => {
                return render_escape_time_f64(params);
            }
            AlgorithmMode::Perturbation => {
                return render_perturbation_cancellable_with_reuse(
                    params,
                    &Arc::new(AtomicBool::new(false)),
                    None,
                )
                .map(|(i, z, _d)| (i, z))
                .unwrap_or_else(|| (Vec::new(), Vec::new()));
            }
            AlgorithmMode::Auto => {
                // Auto mode dispatch:
                // 1. For very deep zooms (>10^15), use perturbation with GMP reference orbit
                // 2. For moderate zooms (10^13 to 10^15), use perturbation for performance
                // 3. For shallow zooms (<10^13), use standard f64 (faster, sufficient precision)
                // 4. For extremely deep zooms (>10^16), use GMP reference if perturbation is not suitable
                if should_use_perturbation(params, false) {
                    return render_perturbation_cancellable_with_reuse(
                        params,
                        &Arc::new(AtomicBool::new(false)),
                        None,
                    )
                    .map(|(i, z, _d)| (i, z))
                    .unwrap_or_else(|| (Vec::new(), Vec::new()));
                }
                if should_use_gmp_reference(params) {
                    return render_escape_time_gmp(params);
                }
                return render_escape_time_f64(params);
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
                let FractalResult { iteration, z: z_final, orbit: _, distance: _ } = iterate_point(params, z_pixel);
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

    // IMPORTANT: Utiliser les String haute précision si disponibles pour préserver la précision GMP
    // aux zooms profonds (>e16). Sinon fallback sur f64 pour compatibilité.
    let center_x = if let Some(ref cx_hp) = params.center_x_hp {
        match Float::parse(cx_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse center_x_hp, using f64 fallback");
                Float::with_val(prec, params.center_x)
            }
        }
    } else {
        Float::with_val(prec, params.center_x)
    };
    
    let center_y = if let Some(ref cy_hp) = params.center_y_hp {
        match Float::parse(cy_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse center_y_hp, using f64 fallback");
                Float::with_val(prec, params.center_y)
            }
        }
    } else {
        Float::with_val(prec, params.center_y)
    };
    
    let span_x = if let Some(ref sx_hp) = params.span_x_hp {
        match Float::parse(sx_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse span_x_hp, using f64 fallback");
                Float::with_val(prec, params.span_x)
            }
        }
    } else {
        Float::with_val(prec, params.span_x)
    };
    
    let span_y = if let Some(ref sy_hp) = params.span_y_hp {
        match Float::parse(sy_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse span_y_hp, using f64 fallback");
                Float::with_val(prec, params.span_y)
            }
        }
    } else {
        Float::with_val(prec, params.span_y)
    };

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
                // IMPORTANT: Utiliser la version GMP de plane_transform pour éviter la perte de précision
                // aux zooms profonds (>e16). La conversion GMP → f64 → GMP perdait toute la précision.
                let z_gmp = complex_from_xy(prec, xg, yg.clone());
                let z_transformed = params.plane_transform.transform_gmp(&z_gmp, prec);
                let z_pixel = z_transformed;
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
/// Retourne None si annulé, Some(iterations, zs) sinon (orbites/distances ignorés pour compatibilité).
#[allow(dead_code)]
pub fn render_escape_time_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_escape_time_cancellable_with_reuse(params, cancel, None)
        .map(|(i, z, _o, _d)| (i, z))
}

/// Résultat du rendu escape-time (iterations, zs, orbites optionnelles, distances optionnelles).
pub type EscapeTimeResult = (
    Vec<u32>,
    Vec<Complex64>,
    Vec<Option<OrbitData>>,
    Vec<f64>,
);

/// Version annulable du rendu escape-time avec réutilisation d'une passe précédente.
/// Les points déjà calculés sont réutilisés quand les résolutions s'alignent.
/// Retourne (iterations, zs, orbits, distances) : orbits remplies pour f64 si enable_orbit_traps,
/// distances remplies pour perturbation si enable_distance_estimation.
pub fn render_escape_time_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<(&[u32], &[Complex64], u32, u32)>,
) -> Option<EscapeTimeResult> {
    // Vérifier l'annulation avant de commencer
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    // Dispatch vers les algorithmes spéciaux (pas d'orbites ni distances)
    let empty_orbits_distances = |n: usize| -> (Vec<Option<OrbitData>>, Vec<f64>) {
        (vec![None; n], vec![])
    };
    match params.fractal_type {
        FractalType::VonKoch => {
            let (i, z) = render_von_koch(params);
            let n = i.len();
            return Some((i, z, empty_orbits_distances(n).0, vec![]));
        }
        FractalType::Dragon => {
            let (i, z) = render_dragon(params);
            let n = i.len();
            return Some((i, z, empty_orbits_distances(n).0, vec![]));
        }
        FractalType::Buddhabrot => {
            return if params.use_gmp {
                render_buddhabrot_mpc_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            } else {
                render_buddhabrot_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            };
        }
        FractalType::Lyapunov => {
            return if params.use_gmp {
                render_lyapunov_mpc_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            } else {
                render_lyapunov_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            };
        }
        FractalType::Nebulabrot => {
            return if params.use_gmp {
                render_nebulabrot_mpc_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            } else {
                render_nebulabrot_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            };
        }
        FractalType::AntiBuddhabrot => {
            return if params.use_gmp {
                render_antibuddhabrot_mpc_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            } else {
                render_antibuddhabrot_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            };
        }
        _ => {}
    }

    if matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip | FractalType::Tricorn
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
            AlgorithmMode::StandardF64 => {
                let reuse = build_reuse(params, reuse);
                return render_escape_time_f64_cancellable_with_reuse(params, cancel, reuse);
            }
            AlgorithmMode::Perturbation => {
                return render_perturbation_cancellable_with_reuse(params, cancel, reuse)
                    .map(|(i, z, d)| (i, z, vec![], d));
            }
            AlgorithmMode::Auto => {
                // Auto mode dispatch for cancellable version:
                // Use perturbation for moderate to deep zooms (10^13 to 10^15)
                if should_use_perturbation(params, false) {
                    return render_perturbation_cancellable_with_reuse(params, cancel, reuse)
                        .map(|(i, z, d)| (i, z, vec![], d));
                }
                let reuse = build_reuse(params, reuse);
                if should_use_gmp_reference(params) {
                    return render_escape_time_gmp_cancellable_with_reuse(params, cancel, reuse);
                }
                return render_escape_time_f64_cancellable_with_reuse(params, cancel, reuse);
            }
        }
    }

    let reuse = build_reuse(params, reuse);
    if params.use_gmp {
        return render_escape_time_gmp_cancellable_with_reuse(params, cancel, reuse);
    }
    render_escape_time_f64_cancellable_with_reuse(params, cancel, reuse)
}

/// Calcule le niveau de zoom à partir des paramètres.
/// Le zoom est calculé comme base_range / pixel_size où base_range = 4.0 (plage standard Mandelbrot).
fn compute_zoom(params: &FractalParams) -> Option<f64> {
    if params.width == 0 || params.height == 0 {
        return None;
    }
    let pixel_size = params.span_x.abs().max(params.span_y.abs()) / params.width as f64;
    if !pixel_size.is_finite() || pixel_size <= 0.0 {
        return None;
    }
    let base_range = 4.0;
    let zoom = base_range / pixel_size;
    if !zoom.is_finite() || zoom <= 1.0 {
        return None;
    }
    Some(zoom)
}

/// Détermine si on doit utiliser GMP reference basé sur le niveau de zoom.
/// Pour des zooms de e1 à e16 (10^1 à 10^16), on utilise CPU f64.
/// Au-delà de 10^16, on bascule sur GMP reference.
pub fn should_use_gmp_reference(params: &FractalParams) -> bool {
    let zoom = match compute_zoom(params) {
        Some(z) => z,
        None => return false,
    };
    // Seuil: zoom > 10^16 → GMP reference
    zoom > 1e16
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
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip | FractalType::Tricorn
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
) -> Option<EscapeTimeResult> {
    render_escape_time_f64_cancellable_with_reuse(params, cancel, None)
}

fn render_escape_time_f64_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<ReuseData<'_>>,
) -> Option<EscapeTimeResult> {
    let width = params.width as usize;
    let height = params.height as usize;
    let n = width * height;
    let mut iterations = vec![0u32; n];
    let mut zs = vec![Complex64::new(0.0, 0.0); n];
    let mut orbits: Vec<Option<OrbitData>> = vec![None; n];
    let mut distances: Vec<f64> = vec![f64::INFINITY; n]; // INFINITY = pas d'estimation

    if width == 0 || height == 0 {
        return Some((iterations, zs, orbits, distances));
    }

    let cancelled = AtomicBool::new(false);
    let need_orbits = params.enable_orbit_traps;
    let need_distances = params.enable_distance_estimation;

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .zip(orbits.par_chunks_mut(width))
        .zip(distances.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (((iter_row, z_row), orbit_row), dist_row))| {
            let reuse_row = reuse.as_ref();
            if j % 16 == 0 && cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }

            let y_ratio = j as f64 / params.height as f64;
            let yg = params.center_y + (y_ratio - 0.5) * params.span_y;
            for (i, (((iter, z), orbit_cell), dist_cell)) in iter_row
                .iter_mut()
                .zip(z_row.iter_mut())
                .zip(orbit_row.iter_mut())
                .zip(dist_row.iter_mut())
                .enumerate()
            {
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
                let FractalResult {
                    iteration,
                    z: z_final,
                    orbit,
                    distance,
                } = iterate_point(params, z_pixel);
                *iter = iteration;
                *z = z_final;
                if need_orbits {
                    *orbit_cell = orbit;
                }
                if need_distances {
                    *dist_cell = distance.unwrap_or(f64::INFINITY);
                }
            }
        });

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        Some((iterations, zs, orbits, distances))
    }
}

#[allow(dead_code)]
fn render_escape_time_gmp_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<EscapeTimeResult> {
    render_escape_time_gmp_cancellable_with_reuse(params, cancel, None)
}

fn render_escape_time_gmp_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<ReuseData<'_>>,
) -> Option<EscapeTimeResult> {
    let width = params.width as usize;
    let height = params.height as usize;
    let n = width * height;
    let mut iterations = vec![0u32; n];
    let mut zs = vec![Complex64::new(0.0, 0.0); n];

    if width == 0 || height == 0 {
        return Some((iterations, zs, vec![], vec![]));
    }

    let gmp = MpcParams::from_params(params);
    let prec = gmp.prec;

    // IMPORTANT: Utiliser les String haute précision si disponibles pour préserver la précision GMP
    // aux zooms profonds (>e16). Sinon fallback sur f64 pour compatibilité.
    let center_x = if let Some(ref cx_hp) = params.center_x_hp {
        match Float::parse(cx_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse center_x_hp, using f64 fallback");
                Float::with_val(prec, params.center_x)
            }
        }
    } else {
        Float::with_val(prec, params.center_x)
    };
    
    let center_y = if let Some(ref cy_hp) = params.center_y_hp {
        match Float::parse(cy_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse center_y_hp, using f64 fallback");
                Float::with_val(prec, params.center_y)
            }
        }
    } else {
        Float::with_val(prec, params.center_y)
    };
    
    let span_x = if let Some(ref sx_hp) = params.span_x_hp {
        match Float::parse(sx_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse span_x_hp, using f64 fallback");
                Float::with_val(prec, params.span_x)
            }
        }
    } else {
        Float::with_val(prec, params.span_x)
    };
    
    let span_y = if let Some(ref sy_hp) = params.span_y_hp {
        match Float::parse(sy_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse span_y_hp, using f64 fallback");
                Float::with_val(prec, params.span_y)
            }
        }
    } else {
        Float::with_val(prec, params.span_y)
    };

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
                // IMPORTANT: Utiliser la version GMP de plane_transform pour éviter la perte de précision
                // aux zooms profonds (>e16). La conversion GMP → f64 → GMP perdait toute la précision.
                let z_gmp = complex_from_xy(prec, xg, yg.clone());
                let z_transformed = params.plane_transform.transform_gmp(&z_gmp, prec);
                let z_pixel = z_transformed;
                let (iter_val, z_final) = iterate_point_mpc(&gmp, &z_pixel);
                *iter = iter_val;
                *z = complex_to_complex64(&z_final);
            }
        });

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        Some((iterations, zs, vec![None; n], vec![]))
    }
}

//! Module de perturbation pour les zooms profonds (Section 2 de la théorie des zooms profonds).
//!
//! # Section 2: Perturbation
//!
//! Low precision deltas relative to high precision orbit.
//!
//! Pour les zooms très profonds (>1e13), la précision standard f64 devient insuffisante.
//! La méthode de perturbation permet de calculer les pixels avec une précision relative
//! en utilisant:
//!
//! 1. **Orbite de référence haute précision** `Z_m` calculée au centre de l'image (GMP)
//! 2. **Deltas de perturbation basse précision** `z_n` relatifs à cette orbite (f64)
//!
//! ## Formules mathématiques
//!
//! - **Pixel orbit**: `Z_m + z_n` où:
//!   - `Z_m` est l'orbite de référence haute précision à l'itération `m`
//!   - `z_n` est le delta de perturbation basse précision à l'itération `n`
//!
//! - **Point C du pixel**: `C + c` où:
//!   - `C` est le point de référence (centre de l'image)
//!   - `c` est l'offset du pixel par rapport au centre
//!
//! - **Formule de perturbation**: `z_{n+1} = 2·Z_m·z_n + z_n² + c`
//!
//!   Cette formule découle du développement de Taylor de `(Z_m + z_n)² + (C + c)`:
//!   ```
//!   (Z_m + z_n)² + (C + c) = Z_m² + 2·Z_m·z_n + z_n² + C + c
//!                           = (Z_m² + C) + (2·Z_m·z_n + z_n² + c)
//!                           = Z_{m+1} + z_{n+1}
//!   ```
//!
//! ## Initialisation
//!
//! - `m` et `n` commencent à 0 (`m = 0`, `n = 0`)
//! - `z_0 = 0` (delta initial = 0 pour Mandelbrot)
//!
//! **Note**: Dans le code, `m` et `n` sont représentés par une seule variable `n` qui est toujours
//! synchronisée (`m = n`). Pour Julia, l'initialisation diffère: `z_0 = c` (delta initial = offset du pixel).
//!
//! ## Avantages
//!
//! - Calcul de l'orbite de référence une seule fois (au centre)
//! - Calcul des pixels en f64 (rapide) au lieu de GMP (lent)
//! - Permet les zooms jusqu'à ~1e15 avant de nécessiter GMP complet
//!
//! ## Rebasing
//!
//! Rebasing to avoid glitches: when `|Z_m + z_n| < |z_n|`, replace `z_n` with `Z_m + z_n`
//! and reset the reference iteration count `m` to 0.
//!
//! **Dans le code**: Comme `m = n` (une seule variable `n`), réinitialiser `n` à 0 équivaut
//! à réinitialiser `m` à 0.
//!
//! ## Optimisations
//!
//! - **Bivariate Linear Approximation (BLA)**: Sometimes, `l` iterations starting at `n` can be
//!   approximated by bivariate linear function: `z_{n+l} = A_{n,l}·z_n + B_{n,l}·c`. This is valid
//!   when the non-linear part of the full perturbation iterations is so small that omitting it would
//!   cause fewer problems than the rounding error of the low precision data type.
//!
//! - **ABS Variation BLA**: The only problem with the Mandelbrot set is the non-linearity, but some
//!   other formulas have other problems, for example the Burning Ship, defined by:
//!   `X + iY → (|X| + i|Y|)² + C`. The absolute value folds the plane when `X` or `Y` are near 0,
//!   so the single step BLA radius becomes the minimum of the non-linearity radius and the folding
//!   radii: `R = max{0, min{ε·inf|A| - sup|B|·|c| / inf|A|, |X|, |Y|}}`. Currently Fraktaler 3 uses
//!   a fudge factor for paranoia, dividing `|X|` and `|Y|` by 2. The merged BLA step radius is unchanged.
//!
//! - **Non-Conformal BLA**: The Mandelbrot set is conformal (angles are preserved). This means
//!   complex numbers can be used for derivatives. Some other formulas are not conformal, for
//!   example the Tricorn aka Mandelbar, defined by: `X + iY → (X - iY)² + C`. For non-conformal
//!   formulas, replace complex numbers by 2×2 real matrices for `A`, `B`. Be careful finding norms:
//!   define `sup|M|` and `inf|M|` as the largest and smallest singular values of `M`. Then:
//!   - Single step BLA radius: `R = ε·inf|A| - sup|B|·|c| / inf|A|`
//!   - Merging BLA steps radius: `R_z = max{0, min{R_x, R_y - sup|B_x|·|c| / sup|A_x|}}`
//! - **Séries**: Approximation par séries de Taylor pour les termes d'ordre supérieur
//! - **Rebasing**: Quand `|Z_m + z_n| < |z_n|`, remplace `z_n` par `Z_m + z_n` et réinitialise `m` à 0
//! - **Hybrid BLA**: For a hybrid loop with multiple phases, you need multiple references, one
//!   starting at each phase in the loop. Rebasing switches to the reference for the current phase.
//!   You need one BLA table per reference. Current implementation uses secondary references for
//!   glitch correction, where each reference has its own orbit and BLA table.
//! - **Détection de glitches**: Recalcule en GMP les pixels suspects

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use num_complex::Complex64;
use rayon::prelude::*;

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::perturbation::delta::{iterate_pixel, iterate_pixel_gmp};
use crate::fractal::perturbation::orbit::{compute_reference_orbit_cached, compute_reference_orbit};
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};
use rug::{Complex, Float};

pub mod types;
pub mod orbit;
pub mod bla;
pub mod delta;
pub mod series;
pub mod glitch;
pub mod nonconformal;
pub mod distance;
pub mod interior;

pub use orbit::{ReferenceOrbitCache, HybridBlaReferences};
pub use glitch::{detect_glitch_clusters, select_secondary_reference_points};

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

/// Active l'instrumentation de performance (timings + compteurs) via env var.
/// Exemple: `FRACTALL_PERTURB_STATS=1`.
pub(crate) fn perf_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("FRACTALL_PERTURB_STATS"))
}

/// Iterate a pixel using Hybrid BLA with multiple references (one per phase).
///
/// For a hybrid loop with multiple phases, you need multiple references, one starting at
/// each phase in the loop. Rebasing switches to the reference for the current phase.
/// You need one BLA table per reference.
///
/// This function manages phase switching during rebasing: when rebasing occurs (reaching
/// end of effective orbit), it switches to the reference for the next phase in the cycle.
fn iterate_pixel_hybrid_bla(
    params: &FractalParams,
    hybrid_refs: &HybridBlaReferences,
    series_table: Option<&crate::fractal::perturbation::series::SeriesTable>,
    delta0: crate::fractal::perturbation::types::ComplexExp,
    dc: crate::fractal::perturbation::types::ComplexExp,
) -> crate::fractal::perturbation::delta::DeltaResult {
    use crate::fractal::perturbation::types::ComplexExp;
    
    if hybrid_refs.cycle_period == 0 {
        // No cycle detected: use primary reference (single reference)
        return iterate_pixel(
            params,
            &hybrid_refs.primary,
            &hybrid_refs.primary_bla,
            series_table,
            delta0,
            dc,
            None, // No phase change for single reference
            None, // No hybrid refs for single reference
        );
    }
    
    // Hybrid BLA: iterate through phases, switching references on rebasing
    // For Hybrid BLA: rebasing switches to the reference for the current phase
    // The current phase is determined by the total iteration count: phase = (iteration - cycle_start) % cycle_period
    let mut delta = delta0;
    let mut total_iterations = 0u32;
    let mut current_phase = hybrid_refs.get_current_phase(total_iterations);
    
    // Iterate through phases until bailout or max iterations
    // Rebasing switches to the reference for the current phase (determined by iteration count)
    while total_iterations < params.iteration_max {
        // Get reference and BLA table for current phase
        let ref_orbit = hybrid_refs.get_reference(current_phase);
        let bla_table = hybrid_refs.get_bla_table(current_phase);
        let effective_len = ref_orbit.effective_len() as u32;
        
        // Create a modified params with reduced iteration_max for this phase
        let mut phase_params = params.clone();
        phase_params.iteration_max = (params.iteration_max - total_iterations).min(effective_len);
        
        // Iterate with current phase reference
        // Pass current_phase and hybrid_refs to iterate_pixel() so it can update phase on rebasing
        let result = iterate_pixel(
            &phase_params,
            ref_orbit,
            bla_table,
            series_table,
            delta,
            dc,
            Some(&mut current_phase),
            Some(hybrid_refs),
        );
        
        total_iterations += result.iteration;
        
        // Check if we escaped, glitched, or reached max iterations
        if result.z_final.norm_sqr() > params.bailout * params.bailout 
            || result.glitched 
            || total_iterations >= params.iteration_max {
            return crate::fractal::perturbation::delta::DeltaResult {
                iteration: total_iterations.min(params.iteration_max),
                z_final: result.z_final,
                glitched: result.glitched,
                suspect: result.suspect,
                distance: result.distance,
                is_interior: result.is_interior,
                phase_changed: result.phase_changed,
            };
        }
        
        // Check if phase changed during rebasing
        if result.phase_changed {
            // Phase changed: update delta and continue with new phase
            // The current_phase has already been updated by iterate_pixel()
            delta = ComplexExp::from_complex64(result.z_final);
            continue;
        }
        
        // Check if rebasing occurred (completed effective orbit for this phase)
        // If result.iteration == effective_len, we reached the end and rebased
        // In Hybrid BLA, rebasing switches to the reference for the current phase
        // (which will be determined by the new total_iterations count)
        if result.iteration >= effective_len.saturating_sub(1) {
            // Rebasing occurred: update delta and continue with next iteration
            // The phase will be recalculated based on the new total_iterations count
            delta = ComplexExp::from_complex64(result.z_final);
            // Recalculate phase based on new total_iterations (in case iterate_pixel didn't update it)
            current_phase = hybrid_refs.get_current_phase(total_iterations);
            continue;
        } else {
            // Normal iteration: return result with accumulated iterations
            return crate::fractal::perturbation::delta::DeltaResult {
                iteration: total_iterations,
                z_final: result.z_final,
                glitched: result.glitched,
                suspect: result.suspect,
                distance: result.distance,
                is_interior: result.is_interior,
                phase_changed: result.phase_changed,
            };
        }
    }
    
    // Fallback: use primary reference
    iterate_pixel(
        params,
        &hybrid_refs.primary,
        &hybrid_refs.primary_bla,
        series_table,
        delta0,
        dc,
        None, // No phase change for fallback
        None, // No hybrid refs for fallback
    )
}

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
    
    // Calculer le centre de l'image pour éviter de marquer les pixels au centre comme glitched
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;
    // Rayon autour du centre où on désactive la détection de glitch par voisinage
    // pour éviter les artefacts circulaires (rayon de 20 pixels)
    let center_radius_sqr = 20.0 * 20.0;
    
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = (y * width + x) as usize;
            
            // Vérifier si le pixel est proche du centre
            let dx = x as f64 - center_x;
            let dy = y as f64 - center_y;
            let dist_sqr = dx * dx + dy * dy;
            let is_near_center = dist_sqr < center_radius_sqr;
            
            // Ne pas marquer comme glitched si proche du centre pour éviter les artefacts circulaires
            if is_near_center {
                continue;
            }
            
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

/// Calcule la précision GMP en bits pour l'orbite de référence et le cache perturbation.
///
/// Deux politiques (référence C++ Fraktaler-3 vs conservative Rust):
/// - **Formule référence C++** (`use_reference_precision_formula = true`):  
///   `prec = max(24, 24 + exp(zoom * height))` (param.cc), équivalent bits  
///   `bits = max(24, 24 + floor(log2(zoom * height)))`, puis clamp 128..8192.
/// - **Politique conservative** (défaut): `bits = log2(zoom) + marge_par_palier`, 128..8192.  
///   Choix délibéré plus conservateur pour éviter les glitches aux zooms extrêmes.
pub(crate) fn compute_perturbation_precision_bits(params: &FractalParams) -> u32 {
    if params.width == 0 || params.height == 0 {
        return params.precision_bits.max(128);
    }
    let pixel_size = params.span_x.abs().max(params.span_y.abs()) / params.width as f64;
    if !pixel_size.is_finite() || pixel_size <= 0.0 {
        return params.precision_bits.max(128);
    }

    // Compute zoom level from pixel size (base range ~4.0 for Mandelbrot)
    let base_range = 4.0;
    let zoom = base_range / pixel_size;
    if !zoom.is_finite() || zoom <= 1.0 {
        return params.precision_bits.max(128);
    }

    let final_bits = if params.use_reference_precision_formula {
        // Formule référence C++ Fraktaler-3: prec = max(24, 24 + (par.zoom * par.p.image.height).exp)
        // .exp est l'exposant binaire du floatexp, donc équivalent à floor(log2(zoom * height))
        let zoom_height = zoom * params.height as f64;
        let exp = if zoom_height > 0.0 && zoom_height.is_finite() {
            zoom_height.log2().floor() as i32
        } else {
            0
        };
        let bits = (24 + exp).max(24) as u32;
        bits.clamp(128, 8192)
    } else {
        // Politique conservative Rust: log2(zoom) + marge par palier (choix délibéré)
        let zoom_bits = zoom.log2().ceil() as i32;
        let safety_margin = if zoom > 1e30 {
            200  // Marge très grande pour les zooms extrêmes (>10^30)
        } else if zoom > 1e20 {
            160  // Marge grande pour les zooms très profonds (>10^20)
        } else if zoom > 1e15 {
            128  // Marge importante pour zooms profonds (>10^15) - CRITIQUE pour éviter bugs
        } else if zoom > 1e10 {
            96   // Marge plus grande pour les très grands zooms (>10^10)
        } else if zoom > 1e6 {
            80   // Marge moyenne pour les zooms moyens
        } else {
            64   // Marge standard pour les zooms faibles
        };
        let needed_bits = (zoom_bits + safety_margin).max(128) as u32;
        needed_bits.clamp(128, 8192)
    };

    // Log de diagnostic pour zoom profond (une seule fois par appel avec un cache statique)
    if zoom > 1e15 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST_LOGGED_ZOOM: AtomicU64 = AtomicU64::new(0);
        let zoom_bits_u64 = (final_bits as u64) << 32 | (zoom.to_bits() >> 32);
        let last_logged = LAST_LOGGED_ZOOM.load(Ordering::Relaxed);
        if zoom_bits_u64 != last_logged {
            LAST_LOGGED_ZOOM.store(zoom_bits_u64, Ordering::Relaxed);
            eprintln!("[PRECISION DEBUG] zoom={:.2e}, pixel_size={:.2e}, final_bits={}, preset_bits={}, reference_formula={}",
                zoom, pixel_size, final_bits, params.precision_bits, params.use_reference_precision_formula);
        }
    }

    final_bits
}

/// Détermine si le zoom est trop profond pour utiliser la perturbation standard (f64/ComplexExp).
/// Pour les zooms très profonds (>10^15), il faut utiliser GMP complet pour tous les calculs.
///
/// # Arguments
/// * `params` - Paramètres de la fractale
///
/// # Returns
/// `true` si GMP complet doit être utilisé, `false` sinon
pub fn should_use_full_gmp_perturbation(params: &FractalParams) -> bool {
    if params.width == 0 || params.height == 0 {
        return false;
    }
    let pixel_size = params.span_x.abs().max(params.span_y.abs()) / params.width as f64;
    if !pixel_size.is_finite() || pixel_size <= 0.0 {
        return false;
    }
    
    // Seuil : pixel_size < 1e-15 correspond à un zoom > 10^15
    // À ce niveau, la précision f64 est insuffisante même avec ComplexExp
    // Il faut utiliser GMP complet pour tous les calculs
    pixel_size < 1e-15
}

/// Calcule l'offset dc (pixel offset from center) en GMP pour les zooms profonds.
/// Cette fonction préserve la précision GMP en calculant directement l'offset
/// sans utiliser params.center_x/center_y qui sont en f64.
///
/// # Arguments
/// * `i` - Coordonnée X du pixel (0..width)
/// * `j` - Coordonnée Y du pixel (0..height)
/// * `params` - Paramètres de la fractale
/// * `prec` - Précision GMP à utiliser
///
/// # Returns
/// Le complexe dc représentant l'offset du pixel par rapport au centre, en GMP
pub fn compute_dc_gmp(
    i: usize,
    j: usize,
    params: &FractalParams,
    _center_x_gmp: &Float,
    _center_y_gmp: &Float,
    prec: u32,
) -> Complex {
    let inv_width = Float::with_val(prec, 1.0) / Float::with_val(prec, params.width as f64);
    let inv_height = Float::with_val(prec, 1.0) / Float::with_val(prec, params.height as f64);
    
    // Utiliser les String haute précision si disponibles, sinon fallback sur f64
    let x_range = if let Some(ref sx_hp) = params.span_x_hp {
        match Float::parse(sx_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => Float::with_val(prec, params.span_x)
        }
    } else {
        Float::with_val(prec, params.span_x)
    };
    
    let y_range = if let Some(ref sy_hp) = params.span_y_hp {
        match Float::parse(sy_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => Float::with_val(prec, params.span_y)
        }
    } else {
        Float::with_val(prec, params.span_y)
    };
    
    let half = Float::with_val(prec, 0.5);
    
    // Log de diagnostic pour quelques pixels (coin supérieur gauche) - une seule fois
    let pixel_size = params.span_x.abs().max(params.span_y.abs()) / params.width as f64;
    if pixel_size < 1e-15 && i == 0 && j == 0 {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED_DC: AtomicBool = AtomicBool::new(false);
        if !LOGGED_DC.swap(true, Ordering::Relaxed) {
            eprintln!("[PRECISION DEBUG] compute_dc_gmp: pixel (0,0), using_hp={}, span_x f64={:.20e}, span_y f64={:.20e}, x_range_gmp={}, y_range_gmp={}",
                params.span_x_hp.is_some(), params.span_x, params.span_y,
                x_range.to_string_radix(10, Some(30)), y_range.to_string_radix(10, Some(30)));
        }
    }
    
    // dc_re = (i/width - 0.5) * x_range
    let i_float = Float::with_val(prec, i as f64);
    let j_float = Float::with_val(prec, j as f64);
    let mut x_ratio = Float::with_val(prec, &i_float * &inv_width);
    let mut y_ratio = Float::with_val(prec, &j_float * &inv_height);
    x_ratio -= &half;
    y_ratio -= &half;
    let x_offset = Float::with_val(prec, &x_ratio * &x_range);
    let y_offset = Float::with_val(prec, &y_ratio * &y_range);
    
    // Log de diagnostic pour quelques pixels - une seule fois
    if pixel_size < 1e-15 && i == 0 && j == 0 {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED_DC_OFFSET: AtomicBool = AtomicBool::new(false);
        if !LOGGED_DC_OFFSET.swap(true, Ordering::Relaxed) {
            eprintln!("[PRECISION DEBUG] dc_gmp: x_offset={}, y_offset={}",
                x_offset.to_string_radix(10, Some(30)), y_offset.to_string_radix(10, Some(30)));
        }
    }
    
    // Le point complexe du pixel est center + dc
    // Mais dc seul est juste l'offset, donc on retourne juste l'offset
    Complex::with_val(prec, (x_offset, y_offset))
}

#[allow(dead_code)]
pub fn render_mandelbrot_perturbation(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>, Vec<f64>) {
    let cancel = Arc::new(AtomicBool::new(false));
    render_perturbation_cancellable_with_reuse(params, &cancel, None)
        .unwrap_or_else(|| (Vec::new(), Vec::new(), Vec::new()))
}

pub fn render_perturbation_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<(&[u32], &[Complex64], u32, u32)>,
) -> Option<(Vec<u32>, Vec<Complex64>, Vec<f64>)> {
    let (result, _cache) = render_perturbation_with_cache(params, cancel, reuse, None)?;
    Some(result)
}

/// Rend une fractale en utilisant la méthode de perturbation.
///
/// Cette fonction calcule l'orbite de référence haute précision au centre de l'image,
/// puis itère chaque pixel en utilisant la formule de perturbation:
/// `z_{n+1} = 2·Z_m·z_n + z_n² + c`
///
/// # Pipeline de rendu
///
/// 1. Calcul de l'orbite de référence `Z_m` au centre (GMP, haute précision)
/// 2. Construction de la table BLA pour sauter des itérations
/// 3. Pour chaque pixel:
///    - Calcul de `dc` (offset par rapport au centre)
///    - Itération avec perturbation (`iterate_pixel`)
///    - Détection de glitches
/// 4. Correction des glitches détectés (recalcul en GMP ou références secondaires)
///
/// # Arguments
///
/// * `params` - Paramètres de la fractale (dimensions, centre, zoom, etc.)
/// * `cancel` - Flag d'annulation pour interrompre le calcul
/// * `reuse` - Données de réutilisation d'un rendu précédent (optionnel)
/// * `orbit_cache` - Cache de l'orbite de référence pour éviter le recalcul (optionnel)
///
/// # Retour
///
/// Retourne `Some((iterations, zs), cache)` si le calcul réussit, `None` si annulé.
/// - `iterations`: Nombre d'itérations avant divergence pour chaque pixel
/// - `zs`: Valeur finale de z pour chaque pixel (pour le coloriage)
/// - `cache`: Cache de l'orbite de référence pour réutilisation
pub fn render_perturbation_with_cache(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    _reuse: Option<(&[u32], &[Complex64], u32, u32)>,
    orbit_cache: Option<&Arc<ReferenceOrbitCache>>,
) -> Option<((Vec<u32>, Vec<Complex64>, Vec<f64>), Arc<ReferenceOrbitCache>)> {
    let perf = perf_enabled();
    let t_all_start = Instant::now();
    let t_orbit_start = Instant::now();
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let supports = matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip | FractalType::Tricorn
    );
    if !supports {
        return None;
    }
    
    // IMPORTANT: Ne pas réutiliser les résultats de pixels entre passes progressives pour la perturbation.
    // Chaque pixel doit être recalculé avec le bon dc (offset) pour sa position exacte dans la nouvelle résolution.
    // La réutilisation des pixels cause des artefacts (comme un cercle au centre) car les pixels réutilisés
    // ont été calculés avec un dc incorrect pour leur nouvelle position.
    // Seule la référence et la BLA sont réutilisées (via orbit_cache), comme dans fraktaler-3.
    // Référence: fraktaler-3 réutilise seulement reference_can_be_reused() et bla_can_be_reused(),
    // mais recalcule toujours tous les pixels à chaque passe.
    // Désactiver la réutilisation des pixels pour la perturbation
    let reuse_for_pixels: Option<(&[u32], &[Complex64], u32, u32)> = None;

    let mut orbit_params = params.clone();
    orbit_params.precision_bits = compute_perturbation_precision_bits(params);

    // Check if we need full GMP perturbation (very deep zooms >10^15)
    let use_full_gmp = should_use_full_gmp_perturbation(params);

    // Use cached orbit/BLA or compute fresh
    let cache =
        compute_reference_orbit_cached(&orbit_params, Some(cancel.as_ref()), orbit_cache)?;
    let t_orbit = t_orbit_start.elapsed();
    let width = params.width as usize;
    let height = params.height as usize;
    let pixel_count = width.saturating_mul(height);
    // Heuristique "petite image": privilégier le coût/pixel (moins de post-traitements).
    let small_image = params.width.max(params.height) <= 512;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];
    let mut distances = vec![f64::INFINITY; width * height];
    let glitch_mask: Vec<AtomicBool> = (0..width * height)
        .map(|_| AtomicBool::new(false))
        .collect();

    if width == 0 || height == 0 {
        return Some(((iterations, zs, distances), cache));
    }

    // For very deep zooms, use full GMP perturbation path
    if use_full_gmp {
        // Ne pas réutiliser les pixels pour GMP non plus
        return render_perturbation_gmp_path(params, cancel, None, &cache, iterations, zs, distances);
    }

    // Compute dc (pixel offset from center) directly to avoid precision loss.
    //
    // Formule: dc = (pixel_index/dimension - 0.5) * range
    //
    // Cette méthode évite la soustraction de grands nombres proches (xmin vs center_x)
    // qui causerait des erreurs de précision lors de zooms profonds.
    //
    // Pour un pixel à la position (i, j):
    // - dc_re = (i/width - 0.5) * span_x
    // - dc_im = (j/height - 0.5) * span_y
    //
    // Le point complexe du pixel est alors: C + dc où C = (center_x, center_y)
    let x_range = params.span_x;
    let y_range = params.span_y;
    let inv_width = 1.0 / params.width as f64;
    let inv_height = 1.0 / params.height as f64;

    let cancelled = AtomicBool::new(false);
    // Ne pas réutiliser les pixels pour la perturbation (voir commentaire ci-dessus)
    let reuse = build_reuse(params, reuse_for_pixels);

    // Clone cache for use in parallel iteration
    let cache_ref = Arc::clone(&cache);

    // Pré-calcul dc pour amortir le coût par pixel (surtout utile sur petites images).
    // Stocker directement en FloatExp pour éviter les conversions via Complex64.
    let dc_re_fexp: Vec<FloatExp> = (0..width)
        .map(|i| FloatExp::from_f64((i as f64 * inv_width - 0.5) * x_range))
        .collect();
    let dc_im_fexp: Vec<FloatExp> = (0..height)
        .map(|j| FloatExp::from_f64((j as f64 * inv_height - 0.5) * y_range))
        .collect();

    let t_pixels_start = Instant::now();
    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .zip(distances.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, ((iter_row, z_row), dist_row))| {
            let reuse_row = reuse.as_ref();
            if j % 16 == 0 && cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }
            let dc_im = dc_im_fexp[j];

            for (i, ((iter, z), dist)) in iter_row.iter_mut().zip(z_row.iter_mut()).zip(dist_row.iter_mut()).enumerate() {
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

                let dc = ComplexExp {
                    re: dc_re_fexp[i],
                    im: dc_im,
                };
                
                // Initialisation du delta selon le type de fractale:
                // - Mandelbrot: z_0 = 0, donc delta0 = 0, et c = dc dans la formule
                // - Julia: z_0 = c (le point C du pixel), donc delta0 = dc, et pas de terme c
                let (delta0, dc_term) = if params.fractal_type == FractalType::Julia {
                    // Julia: delta initial = dc (car z_0 = C + c pour Julia)
                    (dc, ComplexExp::zero())
                } else {
                    // Mandelbrot: delta initial = 0 (car z_0 = 0), terme c = dc
                    (ComplexExp::zero(), dc)
                };

                // Hybrid BLA: use the appropriate reference for the current phase
                // For a hybrid loop with multiple phases, you need multiple references, one starting at
                // each phase in the loop. Rebasing switches to the reference for the current phase.
                // You need one BLA table per reference.
                let result = if let Some(ref hybrid) = cache_ref.hybrid_refs {
                    // Hybrid BLA: iterate with phase-aware reference switching
                    iterate_pixel_hybrid_bla(
                        params,
                        hybrid,
                        cache_ref.series_table.as_ref(),
                        delta0,
                        dc_term,
                    )
                } else {
                    // Single reference (no cycle detected)
                    iterate_pixel(
                        params,
                        &cache_ref.orbit,
                        &cache_ref.bla_table,
                        cache_ref.series_table.as_ref(),
                        delta0,
                        dc_term,
                        None, // No phase change for single reference
                        None, // No hybrid refs for single reference
                    )
                };
                
                // Use distance estimation and interior detection results
                // Encode is_interior in z.im sign: negative = interior point
                // Encode distance in z.re when available (for distance-based coloring)
                let mut z_value = result.z_final;
                
                if result.is_interior {
                    // Interior point: encode flag in z.im sign (negative = interior)
                    // This allows color_for_pixel to detect and color interior points black
                    z_value = Complex64::new(z_value.re, -z_value.im.abs());
                } else if result.distance.is_finite() && result.distance != f64::INFINITY && result.distance > 0.0 {
                    // Distance estimation available: can be used for distance field coloring
                    // For now, we keep z as-is to preserve smooth_iteration calculation
                    // Distance can be accessed via result.distance if needed in the future
                    // Optionally encode distance in z.re for special distance-based coloring modes
                    // z_value = Complex64::new(result.distance, z_value.im);
                }
                
                *iter = result.iteration;
                *z = z_value;
                *dist = result.distance;

                // Fast-path petites images: corriger seulement les vrais glitches (pas "suspect")
                if small_image {
                    if result.glitched {
                        glitch_mask[j * width + i].store(true, Ordering::Relaxed);
                    }
                } else if result.glitched || result.suspect {
                    glitch_mask[j * width + i].store(true, Ordering::Relaxed);
                }
            }
        });
    let t_pixels = t_pixels_start.elapsed();

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        let t_post_start = Instant::now();
        let mut glitch_mask: Vec<bool> = glitch_mask
            .iter()
            .map(|flag| flag.load(Ordering::Relaxed))
            .collect();
        let glitched_initial = glitch_mask.iter().filter(|v| **v).count();

        // Ne pas marquer automatiquement les pixels avec itération <= 1 comme suspects.
        // Ces pixels peuvent être corrects (divergence immédiate réelle).
        // La détection de glitch basée sur la tolérance et le voisinage est suffisante.
        // Marquer seulement si déjà détecté comme glitched/suspect par iterate_pixel.

        // Fast-path petites images: éviter le post-traitement voisinage (coût fixe non négligeable)
        if !small_image && params.glitch_neighbor_pass {
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

        // Hybrid BLA: Multi-reference glitch correction
        //
        // For a hybrid loop with multiple phases, you need multiple references, one starting at
        // each phase in the loop. Rebasing switches to the reference for the current phase.
        // You need one BLA table per reference.
        //
        // Current implementation: Use secondary reference points to fix glitch clusters.
        // Each secondary reference has its own orbit and BLA table. When a pixel is recalculated
        // with a secondary reference, it uses that reference's orbit and BLA table.
        //
        // Note: The current rebasing implementation (in iterate_pixel) resets n to 0 with the
        // same reference. A full Hybrid BLA implementation would switch to a different reference
        // corresponding to the current phase when rebasing.
        // Fast-path petites images: désactiver références secondaires (coût fixe + peu de pixels)
        if !small_image && params.max_secondary_refs > 0 {
            let clusters = detect_glitch_clusters(
                &glitch_mask,
                params.width,
                params.height,
                params,
                params.min_glitch_cluster_size as usize,
            );

            let secondary_refs = select_secondary_reference_points(
                &clusters,
                params.max_secondary_refs as usize,
            );

            // Process each secondary reference
            // Each reference corresponds to a different phase/starting point in the hybrid loop.
            // One BLA table per reference is computed.
            for cluster in secondary_refs {
                // Create params with new center for secondary orbit (different phase)
                let mut sec_params = orbit_params.clone();
                sec_params.center_x = cluster.center_x;
                sec_params.center_y = cluster.center_y;

                // Compute secondary reference orbit (one reference per phase)
                if let Some((sec_orbit, _, _)) = compute_reference_orbit(&sec_params, Some(cancel.as_ref())) {
                    // Build BLA table for this reference (one BLA table per reference)
                    let sec_bla = bla::build_bla_table(&sec_orbit.z_ref_f64, &sec_params, sec_orbit.cref);
                    let sec_series = if params.series_standalone
                        && matches!(params.fractal_type, FractalType::Mandelbrot | FractalType::Julia)
                    {
                        Some(series::build_series_table(&sec_orbit.z_ref_f64))
                    } else {
                        None
                    };

                    // Re-render cluster pixels with secondary reference
                    for &idx in &cluster.pixel_indices {
                        let px = idx % width;
                        let py = idx / width;

                        // Compute dc relative to secondary center
                        let dc_re = (px as f64 * inv_width - 0.5) * x_range
                            - (cluster.center_x - params.center_x);
                        let dc_im = (py as f64 * inv_height - 0.5) * y_range
                            - (cluster.center_y - params.center_y);

                        let dc = ComplexExp::from_complex64(Complex64::new(dc_re, dc_im));
                        let (delta0, dc_term) = if params.fractal_type == FractalType::Julia {
                            (dc, ComplexExp::zero())
                        } else {
                            (ComplexExp::zero(), dc)
                        };

                        let result = iterate_pixel(
                            params,
                            &sec_orbit,
                            &sec_bla,
                            sec_series.as_ref(),
                            delta0,
                            dc_term,
                            None, // No phase change for secondary reference
                            None, // No hybrid refs for secondary reference
                        );

                        // Only update if the secondary reference gave a good result
                        if !result.glitched && !result.suspect {
                            iterations[idx] = result.iteration;
                            zs[idx] = result.z_final;
                            glitch_mask[idx] = false;
                        }
                    }
                }
            }
        }

        let glitched_indices: Vec<usize> = glitch_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, flagged)| if *flagged { Some(idx) } else { None })
            .collect();
        let corrections_requested = glitched_indices.len();

        // Fallback complet vers GMP si trop de glitches (>30% des pixels)
        // Augmenté de 10% à 30% pour éviter de recalculer toute l'image trop souvent.
        // La correction individuelle avec perturbation GMP est maintenant plus efficace.
        let total_pixels = width * height;
        let glitch_ratio = glitched_indices.len() as f64 / total_pixels as f64;
        const GLITCH_FALLBACK_THRESHOLD: f64 = 0.30; // 30% (augmenté de 10%)

        if glitch_ratio > GLITCH_FALLBACK_THRESHOLD {
            // Trop de glitches: recalculer tous les pixels en GMP
            let gmp_params = MpcParams::from_params(&orbit_params);
            let prec = compute_perturbation_precision_bits(params);
            let width_u32 = params.width;

            // Utiliser compute_dc_gmp pour calculer directement en GMP
            let center_x_gmp = if let Some(ref cx_hp) = params.center_x_hp {
                match Float::parse(cx_hp) {
                    Ok(parse_result) => Float::with_val(prec, parse_result),
                    Err(_) => Float::with_val(prec, params.center_x),
                }
            } else {
                Float::with_val(prec, params.center_x)
            };
            let center_y_gmp = if let Some(ref cy_hp) = params.center_y_hp {
                match Float::parse(cy_hp) {
                    Ok(parse_result) => Float::with_val(prec, parse_result),
                    Err(_) => Float::with_val(prec, params.center_y),
                }
            } else {
                Float::with_val(prec, params.center_y)
            };
            
            let all_corrections: Vec<_> = (0..total_pixels)
                .into_par_iter()
                .map(|idx| {
                    let i = (idx as u32 % width_u32) as usize;
                    let j = (idx as u32 / width_u32) as usize;
                    
                    // Calculer dc en GMP directement
                    let dc_gmp = compute_dc_gmp(i, j, params, &center_x_gmp, &center_y_gmp, prec);
                    
                    // Calculer le point pixel = center + dc en GMP
                    let mut z_pixel_re = center_x_gmp.clone();
                    z_pixel_re += dc_gmp.real();
                    let mut z_pixel_im = center_y_gmp.clone();
                    z_pixel_im += dc_gmp.imag();
                    let z_pixel = complex_from_xy(prec, z_pixel_re, z_pixel_im);
                    
                    let (iter_val, z_final) = iterate_point_mpc(&gmp_params, &z_pixel);
                    (idx, iter_val, complex_to_complex64(&z_final))
                })
                .collect();

            for (idx, iter_val, z_final) in all_corrections {
                iterations[idx] = iter_val;
                zs[idx] = z_final;
            }

            return Some(((iterations, zs, distances), cache));
        }

        if !glitched_indices.is_empty() {
            // Utiliser perturbation GMP au lieu de recalcul complet pour corriger les glitches.
            // C'est beaucoup plus rapide car on réutilise l'orbite de référence déjà calculée.
            let prec = compute_perturbation_precision_bits(params);
            let width_u32 = params.width;
            
            // Utiliser les String haute précision si disponibles pour le calcul de dc
            let center_x_gmp = if let Some(ref cx_hp) = params.center_x_hp {
                match Float::parse(cx_hp) {
                    Ok(parse_result) => Float::with_val(prec, parse_result),
                    Err(_) => Float::with_val(prec, params.center_x),
                }
            } else {
                Float::with_val(prec, params.center_x)
            };
            let center_y_gmp = if let Some(ref cy_hp) = params.center_y_hp {
                match Float::parse(cy_hp) {
                    Ok(parse_result) => Float::with_val(prec, parse_result),
                    Err(_) => Float::with_val(prec, params.center_y),
                }
            } else {
                Float::with_val(prec, params.center_y)
            };
            
            // Utiliser l'orbite de référence GMP déjà calculée (plus rapide que recalcul complet)
            use crate::fractal::perturbation::delta::iterate_pixel_gmp;
            let corrections: Vec<_> = glitched_indices
                .par_iter()
                .map(|&idx| {
                    let i = (idx as u32 % width_u32) as usize;
                    let j = (idx as u32 / width_u32) as usize;
                    
                    // Calculer dc en GMP directement
                    let dc_gmp = compute_dc_gmp(i, j, params, &center_x_gmp, &center_y_gmp, prec);
                    
                    // Utiliser perturbation GMP avec l'orbite de référence (beaucoup plus rapide)
                    let result = iterate_pixel_gmp(
                        params,
                        &cache.orbit,
                        &dc_gmp,
                        prec,
                    );
                    
                    (idx, result.iteration, result.z_final)
                })
                .collect();

            for (idx, iter_val, z_final) in corrections {
                iterations[idx] = iter_val;
                zs[idx] = z_final;
            }
        }
        let t_post = t_post_start.elapsed();

        if perf {
            let pixel_size = params.span_x.abs().max(params.span_y.abs()) / params.width.max(1) as f64;
            let zoom = if pixel_size.is_finite() && pixel_size > 0.0 {
                4.0 / pixel_size
            } else {
                0.0
            };
            eprintln!(
                "[PERTURB PERF] {}x{} pixels={} zoom={:.2e} small_image={} orbit={:.3}s pixels={:.3}s post={:.3}s total={:.3}s glitched_initial={} corrections={} fallback_ratio={:.3}",
                params.width,
                params.height,
                pixel_count,
                zoom,
                small_image,
                t_orbit.as_secs_f64(),
                t_pixels.as_secs_f64(),
                t_post.as_secs_f64(),
                t_all_start.elapsed().as_secs_f64(),
                glitched_initial,
                corrections_requested,
                glitch_ratio,
            );
        }

        Some(((iterations, zs, distances), cache))
    }
}

/// Rendu avec chemin GMP complet pour les zooms très profonds (>10^15).
/// Cette fonction utilise GMP pour tous les calculs de perturbation.
fn render_perturbation_gmp_path(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<(&[u32], &[Complex64], u32, u32)>,
    cache: &Arc<ReferenceOrbitCache>,
    mut iterations: Vec<u32>,
    mut zs: Vec<Complex64>,
    distances: Vec<f64>,
) -> Option<((Vec<u32>, Vec<Complex64>, Vec<f64>), Arc<ReferenceOrbitCache>)> {
    // Utiliser la précision calculée au lieu du preset
    let prec = compute_perturbation_precision_bits(params);
    let width = params.width as usize;
    
    // IMPORTANT: Vérifier que la précision du cache correspond à la précision calculée
    // Si la précision du cache est inférieure, cela peut causer des erreurs de précision
    if cache.precision_bits < prec {
        eprintln!("[PRECISION WARNING] Cache precision ({}) < required precision ({}) for zoom {:.2e}. Cache may need recomputation.",
            cache.precision_bits, prec, params.span_x.abs().max(params.span_y.abs()) / params.width as f64);
    }
    
    // Log de diagnostic - une seule fois
    let pixel_size = params.span_x.abs().max(params.span_y.abs()) / params.width as f64;
    if pixel_size < 1e-15 {
        use std::sync::atomic::{AtomicU32, Ordering};
        static LAST_LOGGED_GMP_PATH: AtomicU32 = AtomicU32::new(0);
        let last_logged = LAST_LOGGED_GMP_PATH.load(Ordering::Relaxed);
        if prec != last_logged {
            LAST_LOGGED_GMP_PATH.store(prec, Ordering::Relaxed);
            eprintln!("[PRECISION DEBUG] render_perturbation_gmp_path: prec={}, preset_bits={}, cache.precision_bits={}, cache.center_x_gmp={}, cache.center_y_gmp={}",
                prec, params.precision_bits, cache.precision_bits, cache.center_x_gmp, cache.center_y_gmp);
        }
    }
    
    // Parse center from GMP strings stored in cache
    let center_x_gmp = match Float::parse(&cache.center_x_gmp) {
        Ok(parse_result) => Float::with_val(prec, parse_result),
        Err(_) => {
            eprintln!("[PRECISION ERROR] Failed to parse center_x_gmp: {}", cache.center_x_gmp);
            return None;
        },
    };
    let center_y_gmp = match Float::parse(&cache.center_y_gmp) {
        Ok(parse_result) => Float::with_val(prec, parse_result),
        Err(_) => {
            eprintln!("[PRECISION ERROR] Failed to parse center_y_gmp: {}", cache.center_y_gmp);
            return None;
        },
    };
    
    // Log de diagnostic pour vérifier la conversion String → GMP - une seule fois
    if pixel_size < 1e-15 {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED_PARSED: AtomicBool = AtomicBool::new(false);
        if !LOGGED_PARSED.swap(true, Ordering::Relaxed) {
            eprintln!("[PRECISION DEBUG] Parsed center_x_gmp={}, center_y_gmp={}",
                center_x_gmp.to_string_radix(10, Some(30)), center_y_gmp.to_string_radix(10, Some(30)));
        }
    }
    
    let cancelled = AtomicBool::new(false);
    // IMPORTANT: Ne pas réutiliser les résultats de pixels entre passes progressives pour la perturbation.
    // Chaque pixel doit être recalculé avec le bon dc (offset) pour sa position exacte.
    // La réutilisation cause des artefacts (comme un cercle au centre) car les pixels réutilisés
    // ont été calculés avec un dc incorrect pour leur nouvelle position.
    // Note: reuse est déjà None (passé depuis render_perturbation_with_cache), donc reuse_data sera None.
    let reuse_data = build_reuse(params, reuse);
    
    // Clone cache for use in parallel iteration
    let cache_ref = Arc::clone(cache);
    
    // Collect glitched pixels for correction
    let glitch_mask: Vec<AtomicBool> = (0..width * params.height as usize)
        .map(|_| AtomicBool::new(false))
        .collect();
    
    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let reuse_row = reuse_data.as_ref();
            if j % 16 == 0 && cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }
            
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
                
                // Compute dc in GMP precision
                let dc_gmp = compute_dc_gmp(
                    i,
                    j,
                    params,
                    &center_x_gmp,
                    &center_y_gmp,
                    prec,
                );
                
                // Iterate pixel with full GMP precision
                let result = iterate_pixel_gmp(
                    params,
                    &cache_ref.orbit,
                    &dc_gmp,
                    prec,
                );
                
                *iter = result.iteration;
                *z = result.z_final;
                
                // Mark glitched or suspect pixels for correction
                if result.glitched || result.suspect || !result.z_final.re.is_finite() || !result.z_final.im.is_finite() {
                    let idx = j * width + i;
                    glitch_mask[idx].store(true, Ordering::Relaxed);
                }
            }
        });
    
    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        // Correct glitched pixels using direct GMP iteration (fallback)
        let glitched_indices: Vec<usize> = glitch_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| if flag.load(Ordering::Relaxed) { Some(idx) } else { None })
            .collect();
        
        if !glitched_indices.is_empty() {
            // Use direct GMP iteration as fallback for glitched pixels
            let mut orbit_params = params.clone();
            orbit_params.precision_bits = prec;
            let gmp_params = MpcParams::from_params(&orbit_params);
            let width_u32 = params.width;
            
            let corrections: Vec<_> = glitched_indices
                .par_iter()
                .map(|&idx| {
                    let i = (idx as u32 % width_u32) as usize;
                    let j = (idx as u32 / width_u32) as usize;
                    
                    // Calculate pixel point directly in GMP: center + dc
                    let dc_gmp = compute_dc_gmp(i, j, params, &center_x_gmp, &center_y_gmp, prec);
                    let mut z_pixel_re = center_x_gmp.clone();
                    z_pixel_re += dc_gmp.real();
                    let mut z_pixel_im = center_y_gmp.clone();
                    z_pixel_im += dc_gmp.imag();
                    let z_pixel = complex_from_xy(prec, z_pixel_re, z_pixel_im);
                    
                    // Use direct GMP iteration (no perturbation)
                    let (iter_val, z_final) = iterate_point_mpc(&gmp_params, &z_pixel);
                    (idx, iter_val, complex_to_complex64(&z_final))
                })
                .collect();
            
            for (idx, iter_val, z_final) in corrections {
                iterations[idx] = iter_val;
                zs[idx] = z_final;
            }
        }
        
        Some(((iterations, zs, distances), Arc::clone(cache)))
    }
}

#[cfg(test)]
mod tests {
    use super::render_perturbation_cancellable_with_reuse;
    use crate::fractal::definitions::default_params_for_type;
    use crate::fractal::iterations::iterate_point;
    use crate::fractal::{AlgorithmMode, FractalParams, FractalType};
    use num_complex::Complex64;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn base_params(fractal_type: FractalType) -> FractalParams {
        // center=(0,0), span=(4,3) -> xmin=-2, xmax=2, ymin=-1.5, ymax=1.5
        let mut p = default_params_for_type(fractal_type, 5, 5);
        p.span_x = 4.0;
        p.span_y = 3.0;
        p.iteration_max = 64;
        p.precision_bits = 192;
        p.algorithm_mode = AlgorithmMode::Perturbation;
        p.bla_threshold = 1e-6;
        p.glitch_neighbor_pass = false;
        p
    }

    fn assert_close_iterations(params: &FractalParams, indices: &[(u32, u32)]) {
        let cancel = Arc::new(AtomicBool::new(false));
        let (iters, _, _) =
            render_perturbation_cancellable_with_reuse(params, &cancel, None).unwrap();
        for &(x, y) in indices {
            let idx = (y * params.width + x) as usize;
            // Utiliser center+span directement pour éviter les problèmes de précision
            let x_ratio = x as f64 / params.width as f64;
            let y_ratio = y as f64 / params.height as f64;
            let xg = params.center_x + (x_ratio - 0.5) * params.span_x;
            let yg = params.center_y + (y_ratio - 0.5) * params.span_y;
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
        // xmin=-2.5, xmax=1.5 -> center=-0.5, span=4.0
        params.center_x = -0.5;
        params.span_x = 4.0;
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
        // xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0 -> center=(-0.5, 0), span=(4, 4)
        params.center_x = -0.5;
        params.center_y = 0.0;
        params.span_x = 4.0;
        params.span_y = 4.0;
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

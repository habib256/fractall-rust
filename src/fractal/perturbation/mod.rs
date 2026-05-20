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

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use num_complex::Complex64;
use rayon::prelude::*;

use crate::fractal::{FractalParams, FractalType, OutColoringMode};
use crate::fractal::bytecode::compile_formula;
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::perturbation::delta::{bytecode_path_label, iterate_pixel, iterate_pixel_gmp};
use crate::fractal::perturbation::orbit::{compute_reference_orbit_cached, compute_reference_orbit};
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};
use rug::{Complex, Float};

/// Le pixel passe par le path bytecode/F3 (BLA mat2 + rebasing F3 strict) ?
/// Si oui, `iterate_pixel` retourne toujours `glitched: false` et le post-traitement
/// (neighbor pass Pauldelbrot + secondary references) n'est qu'overhead + source
/// de pixels divergents (corrigés via GMP avec résultat ≠ fexp).
fn uses_bytecode_path(params: &FractalParams) -> bool {
    params.use_bytecode_engine
        && compile_formula(params.fractal_type, params.multibrot_power).is_some()
}

pub mod types;
pub mod dd;
pub mod orbit;
pub mod bla;
pub mod delta;
pub mod series;
pub mod glitch;
pub mod nonconformal;
pub mod nucleus;
#[cfg(test)]
pub mod debug_pure_f3;
pub use orbit::{ReferenceOrbitCache, HybridBlaReferences};
pub use glitch::{detect_glitch_clusters, select_secondary_reference_points, segregate_glitches_by_iteration};

fn env_flag_off(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"),
        Err(_) => false,
    }
}

/// Affiche le breakdown timing perturbation par défaut sur stderr.
/// Opt-out : `FRACTALL_PERTURB_STATS=0`.
pub(crate) fn perf_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_flag_off("FRACTALL_PERTURB_STATS"))
}

/// Compteurs partagés pour le reporter live façon Fraktaler-3
/// (`Frame[NN%] Ref[NN%] BLA[NN%] Tile[NN%]`). Stockés en pourcentages 0..100.
#[derive(Default)]
pub(crate) struct ProgressState {
    pub r#ref: AtomicU32,
    pub bla: AtomicU32,
    pub tile: AtomicU32,
    pub done: AtomicBool,
}

impl ProgressState {
    fn snapshot_line(&self) -> String {
        format!(
            "Frame[100%] Ref[{:>3}%] BLA[{:>3}%] Tile[{:>3}%]",
            self.r#ref.load(Ordering::Relaxed).min(100),
            self.bla.load(Ordering::Relaxed).min(100),
            self.tile.load(Ordering::Relaxed).min(100),
        )
    }
}

/// Lance un thread qui affiche `Frame[NN%] Ref[NN%] BLA[NN%] Tile[NN%]\r` toutes
/// les 500 ms tant que `state.done == false`. Mirror de F3 `batch.cc::batch()`.
/// Retourne le handle à joindre une fois le rendu terminé (qui imprime la
/// ligne finale avec retour à la ligne).
pub(crate) fn spawn_progress_reporter(state: Arc<ProgressState>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        // Premier tick immédiat pour montrer 0/0/0 au démarrage.
        let mut last = String::new();
        loop {
            let line = state.snapshot_line();
            if line != last {
                eprint!("\r{} ", line);
                let _ = std::io::Write::flush(&mut std::io::stderr());
                last = line;
            }
            if state.done.load(Ordering::Relaxed) {
                // Ligne finale avec retour à la ligne.
                eprintln!("\r{} ", state.snapshot_line());
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    })
}

/// Imprime la ligne de résumé finale `[FRACTALL]` au format compact, alignée
/// sur la sortie F3 pour faciliter les comparaisons côte-à-côte.
#[allow(clippy::too_many_arguments)]
pub(crate) fn print_fractall_summary(
    path: &'static str,
    fractal_type: FractalType,
    prec_bits: u32,
    iter_max: u32,
    iterations: &[u32],
    pixel_count: usize,
    t_pixels: Duration,
    t_total: Duration,
) {
    let total_iters: u64 = iterations.iter().map(|&n| n as u64).sum();
    let max_iter = iterations.iter().copied().max().unwrap_or(0);
    let avg_iter = if pixel_count > 0 {
        total_iters as f64 / pixel_count as f64
    } else {
        0.0
    };
    let ns_per_iter = if total_iters > 0 {
        t_pixels.as_secs_f64() * 1e9 / total_iters as f64
    } else {
        0.0
    };
    eprintln!(
        "[FRACTALL] type={:?} path={} prec={}b iter_max={} avg_iter/px={:.0} max_iter/px={} ns/iter={:.1} pixels={:.3}s total={:.3}s",
        fractal_type, path, prec_bits, iter_max, avg_iter, max_iter, ns_per_iter,
        t_pixels.as_secs_f64(), t_total.as_secs_f64(),
    );
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
                smooth_iteration: result.smooth_iteration,
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
                smooth_iteration: result.smooth_iteration,
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
    // Disable pixel reuse for coloring modes that need per-pixel orbit/distance data,
    // since reused pixels don't carry this data and would create checkerboard artifacts.
    let needs_extra_data = matches!(
        params.out_coloring_mode,
        OutColoringMode::Distance | OutColoringMode::DistanceAO | OutColoringMode::Distance3D
        | OutColoringMode::OrbitTraps | OutColoringMode::Wings
    );
    if needs_extra_data {
        return None;
    }
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

/// Spans (re, im) extraits depuis les coordonnées HP en FloatExp pour préserver
/// les magnitudes < f64::MIN_POSITIVE.
///
/// À zoom > 1e308, `params.span_x / span_y` (f64) underflow à 0, ce qui force
/// le dc per-pixel à zéro et produit une image uniforme. Ce helper construit
/// FloatExp directement depuis les chaînes HP (parsing GMP → mantisse + exp
/// arbitraire) puis convertit en FloatExp (mantisse f64 + exposant i64) qui
/// est ensuite utilisable dans tous les arithmétiques par-pixel sans précision
/// loss tant que les operations restent en FloatExp.
pub(crate) fn effective_spans_fexp(params: &FractalParams) -> (crate::fractal::perturbation::types::FloatExp, crate::fractal::perturbation::types::FloatExp) {
    use crate::fractal::perturbation::types::FloatExp;
    let from_hp_or_f64 = |hp: Option<&str>, fallback: f64| -> FloatExp {
        if fallback.is_finite() && fallback != 0.0 {
            return FloatExp::from_f64(fallback);
        }
        let Some(hp_str) = hp else {
            return FloatExp::from_f64(fallback);
        };
        let Ok(raw) = Float::parse(hp_str) else {
            return FloatExp::from_f64(fallback);
        };
        // Précision suffisante pour capturer un exposant arbitraire.
        let v = Float::with_val(1024, raw);
        if v.is_zero() || !v.is_finite() {
            return FloatExp::from_f64(fallback);
        }
        // Décomposer en (mantisse f64, exposant binaire). On utilise log2 ensuite
        // pow2 pour reconstruire — ldexp ferait pareil mais sans risque d'overflow
        // car FloatExp::new normalise.
        let log2 = v.clone().abs().ln() / Float::with_val(1024, 2.0f64.ln());
        let exp_int = log2.to_f64().floor();
        // mantisse = v / 2^exp_int ∈ [1, 2)
        let two_pow = Float::with_val(1024, 2.0f64.powf(exp_int.min(1023.0).max(-1023.0)));
        // Si exp_int hors range f64 (très probable à zoom 1e1000), on construit
        // 2^exp_int via une boucle. Pour simplicité ici on accepte un mantisse
        // possiblement déformé et on laisse FloatExp::new() normaliser.
        let mantissa_float = if exp_int.abs() < 1023.0 {
            v.clone() / &two_pow
        } else {
            // Décomposition explicite : v * 2^(-exp_int) via successive halvings
            // sur Float (1024 bits) — sûr pour any exp_int.
            let mut m = v.clone();
            let mut remaining = -exp_int;
            // Apply by 1000-step chunks
            while remaining.abs() >= 1000.0 {
                let step = if remaining > 0.0 { 1000.0 } else { -1000.0 };
                let chunk = Float::with_val(1024, 2.0f64.powf(step));
                m *= &chunk;
                remaining -= step;
            }
            if remaining != 0.0 {
                let chunk = Float::with_val(1024, 2.0f64.powf(remaining));
                m *= &chunk;
            }
            m
        };
        let mantissa = mantissa_float.to_f64();
        if !mantissa.is_finite() || mantissa == 0.0 {
            return FloatExp::from_f64(fallback);
        }
        // FloatExp exposant = i32 ; les span en zoom corpus restent dans
        // ±(quelques milliers) — bien dans le range i32 (±2^31).
        let exp_clamped = exp_int.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        FloatExp::new(mantissa, exp_clamped)
    };
    let sx = from_hp_or_f64(params.span_x_hp.as_deref(), params.span_x);
    let sy = from_hp_or_f64(params.span_y_hp.as_deref(), params.span_y);
    (sx, sy)
}

/// Calcule un pixel_size effectif tenant compte des coordonnées HP quand le calcul
/// f64 underflow.
///
/// À zoom > ~1e308, `span_x/width` calculé en f64 produit 0 (denormal underflow),
/// ce qui fait croire au dispatcher (`should_use_full_gmp_perturbation`,
/// `bytecode_path_label`, `try_bytecode_unified_path`) que pixel_size est nul
/// → retombe sur le path legacy `iterate_pixel` qui cape la boucle à
/// `effective_len-1` et produit une image uniforme sur orbite référence escapée.
///
/// Retourne `0.0` si pixel_size est vraiment 0 (width=0 ou tout span nul même
/// en HP), sinon une valeur positive représentant le pixel_size effectif. La
/// valeur peut être en dessous du normal range f64 (denormal), c'est OK pour
/// les comparaisons `< seuil`.
pub(crate) fn effective_pixel_size(params: &FractalParams) -> f64 {
    if params.width == 0 || params.height == 0 {
        return 0.0;
    }
    let pixel_size_f64 = (params.span_x.abs() / params.width as f64)
        .max(params.span_y.abs() / params.height as f64);
    if pixel_size_f64.is_finite() && pixel_size_f64 > 0.0 {
        return pixel_size_f64;
    }
    // f64 underflow / non-finite : reconstruire via HP si dispo.
    let parse = |s: &str| -> Option<Float> {
        let raw = Float::parse(s).ok()?;
        Some(Float::with_val(1024, raw))
    };
    let sx_str = params.span_x_hp.as_deref();
    let sy_str = params.span_y_hp.as_deref().or(sx_str);
    let sx = sx_str.and_then(parse);
    let sy = sy_str.and_then(parse);
    let (Some(sx), Some(sy)) = (sx, sy) else {
        return 0.0;
    };
    let w = Float::with_val(1024, params.width as f64);
    let h = Float::with_val(1024, params.height as f64);
    let mut a = sx.abs();
    a /= &w;
    let mut b = sy.abs();
    b /= &h;
    let pixel = if a > b { a } else { b };
    if pixel.is_zero() || !pixel.is_finite() {
        return 0.0;
    }
    // log2 first then to_f64 — sinon to_f64 sature les très petits à 0.
    let log2 = pixel.ln() / Float::with_val(1024, 2.0f64.ln());
    let log2_f64 = log2.to_f64();
    if !log2_f64.is_finite() {
        return 0.0;
    }
    // Reconstruit pixel_size = 2^log2 ; pour log2 < f64::MIN_EXP (≈-1074), retourne
    // f64::MIN_POSITIVE comme sentinelle « extrêmement petit mais > 0 ».
    let result = 2.0_f64.powf(log2_f64);
    if result > 0.0 && result.is_finite() {
        result
    } else {
        f64::MIN_POSITIVE
    }
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
    // log2(zoom) where zoom = max(|span_x|/width, |span_y|/height) -> base_range/pixel_size.
    //
    // For super-deep zooms (zoom > 1e308) the f64 pixel_size underflows to 0, which
    // previously made this function fall through to params.precision_bits (typically
    // 256). At zoom 1e1000 that yields garbage reference orbits because the center
    // is rounded to ~75 decimal digits when we actually need ~1000. Compute log2(zoom)
    // from the HP span strings when available to keep precision correct down to any
    // depth the corpus throws at us (corpus has zoom 1e8000).
    let log2_zoom = {
        let log2_from_f64 = || -> Option<f64> {
            let pixel_size = (params.span_x.abs() / params.width as f64)
                .max(params.span_y.abs() / params.height as f64);
            if !pixel_size.is_finite() || pixel_size <= 0.0 {
                return None;
            }
            let zoom = 4.0 / pixel_size;
            if !zoom.is_finite() || zoom <= 1.0 {
                return None;
            }
            Some(zoom.log2())
        };
        let log2_from_hp = || -> Option<f64> {
            // span_x_hp stores the span as a decimal string; parse with enough precision
            // to capture the exponent regardless of magnitude (1024 bits = ~308 decimal digits,
            // which is sufficient for zoom up to ~10^10^4 — anything deeper than the corpus).
            let sx = params.span_x_hp.as_deref()?;
            let sy = params.span_y_hp.as_deref().unwrap_or(sx);
            let parse = |s: &str| -> Option<Float> {
                let raw = Float::parse(s).ok()?;
                Some(Float::with_val(1024, raw))
            };
            let px_x = parse(sx)?;
            let px_y = parse(sy)?;
            // pixel_size in HP = max(|sx|/w, |sy|/h) ; we want log2(4/pixel_size).
            let w = Float::with_val(1024, params.width as f64);
            let h = Float::with_val(1024, params.height as f64);
            let mut a = px_x.clone().abs();
            a /= &w;
            let mut b = px_y.clone().abs();
            b /= &h;
            let pixel = if a > b { a } else { b };
            if pixel.is_zero() || !pixel.is_finite() {
                return None;
            }
            let mut zoom = Float::with_val(1024, 4.0);
            zoom /= &pixel;
            if zoom <= 1.0 {
                return None;
            }
            // Float::to_f64 saturates to ±inf for values outside f64 range, so use log2
            // first then convert — log2(1e8000) ≈ 26575 fits comfortably in f64.
            let lz = zoom.ln() / Float::with_val(1024, 2.0f64.ln());
            Some(lz.to_f64())
        };
        log2_from_hp().or_else(log2_from_f64)
    };
    let log2_zoom = match log2_zoom {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => return params.precision_bits.max(128),
    };

    let final_bits = if params.use_reference_precision_formula {
        // Formule référence C++ Fraktaler-3: prec = max(24, 24 + (par.zoom * par.p.image.height).exp)
        // .exp est l'exposant binaire du floatexp, donc équivalent à floor(log2(zoom * height))
        let log2_height = (params.height as f64).max(1.0).log2();
        let exp = (log2_zoom + log2_height).floor() as i64;
        // 1 + exp matches max(24, 24 + exp) when exp grows; clamp negative to 0.
        let bits = if exp >= 0 { (24 + exp) as i64 } else { 24 } as u64;
        bits.clamp(128, 65536) as u32
    } else {
        // Politique conservative Rust: log2(zoom) + marge par palier (choix délibéré)
        let zoom_bits = log2_zoom.ceil() as i64;
        let safety_margin: i64 = if log2_zoom > 100.0 {
            200  // > 10^30
        } else if log2_zoom > 66.0 {
            160  // > 10^20
        } else if log2_zoom > 50.0 {
            128  // > 10^15
        } else if log2_zoom > 33.0 {
            96   // > 10^10
        } else if log2_zoom > 20.0 {
            80   // > 10^6
        } else {
            64
        };
        let needed_bits = (zoom_bits + safety_margin).max(128) as u64;
        needed_bits.clamp(128, 65536) as u32
    };

    // Respect params.precision_bits as a floor: if the user (or a preset) explicitly
    // requests higher precision than the auto-formula, honor it. This keeps GMP pure
    // and perturbation aligned under MpcParams::from_params and avoids precision-
    // mismatch divergences at extreme zooms (seen at e50 with 170k+ iterations).
    final_bits.max(params.precision_bits.clamp(128, 65536))
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
    // Override possible via FRACTALL_FORCE_GMP_PERTURB=1 si jamais un cas pathologique
    // se présente.
    if matches!(std::env::var("FRACTALL_FORCE_GMP_PERTURB").as_deref(), Ok("1" | "true")) {
        return true;
    }
    // Si le path bytecode (f64 ou exp) peut s'en charger, il est strictement
    // plus rapide que GMP-per-pixel et tout aussi précis (réf. orbite en GMP,
    // delta en ComplexExp couvrant des magnitudes 2^±2^31). On laisse donc
    // le pipeline routes via try_bytecode_unified_path → bytecode_exp pour
    // les zooms > 1e100 sur Mandelbrot/Julia/BS/Tricorn/Multibrot. GMP-per-pixel
    // ne reste utile que comme garde-fou si bytecode_path_label retourne None
    // (multi-phase, type non-bytecode, ou pixel_size truly < bytecode_gmp_threshold).
    if crate::fractal::perturbation::delta::bytecode_path_label(params).is_some() {
        return false;
    }
    let pixel_size = effective_pixel_size(params);
    if pixel_size <= 0.0 {
        return false;
    }

    // FloatExp = (f64 mantissa, i32 exponent) → couvre des magnitudes jusqu'à
    // ~2^(2^31). Le GMP-per-pixel n'est nécessaire que si la précision
    // 53-bit du mantissa ne suffit plus à représenter un pixel — ce qui ne
    // se produit qu'à des zooms extrêmes (>1e300) où l'erreur accumulée
    // dans la boucle delta dépasse la taille d'un pixel. À zoom <1e15 le
    // path bytecode/perturbation_fexp est ~10-50× plus rapide que GMP-per-pixel
    // pour un résultat identique au pixel près (cf. comparaison avec Fraktaler-3
    // dans `docs/`).
    //
    // Ancien seuil : 1e-15 — trop conservateur, déclenchait à zoom > 1e15
    // alors que ComplexExp gère parfaitement ce régime.
    pixel_size < 1e-300
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

/// Pre-computed constants for GMP dc computation.
/// Avoids re-allocating shared GMP values for every pixel.
pub struct DcGmpContext {
    pub inv_width: Float,
    pub inv_height: Float,
    pub half: Float,
    pub x_range: Float,
    pub y_range: Float,
    pub prec: u32,
    /// Matrice de rotation appliquée au delta (None si rotation == 0).
    /// Cf. `FractalParams::rotation_matrix` et F3 hybrid.cc:265.
    pub rot: Option<(f64, f64, f64, f64)>,
}

impl DcGmpContext {
    pub fn new(params: &FractalParams, prec: u32) -> Self {
        let inv_width = Float::with_val(prec, 1.0) / Float::with_val(prec, params.width as f64);
        let inv_height = Float::with_val(prec, 1.0) / Float::with_val(prec, params.height as f64);

        let x_range = if let Some(ref sx_hp) = params.span_x_hp {
            match Float::parse(sx_hp) {
                Ok(parse_result) => Float::with_val(prec, parse_result),
                Err(_) => Float::with_val(prec, params.span_x),
            }
        } else {
            Float::with_val(prec, params.span_x)
        };

        let y_range = if let Some(ref sy_hp) = params.span_y_hp {
            match Float::parse(sy_hp) {
                Ok(parse_result) => Float::with_val(prec, parse_result),
                Err(_) => Float::with_val(prec, params.span_y),
            }
        } else {
            Float::with_val(prec, params.span_y)
        };

        let half = Float::with_val(prec, 0.5);

        DcGmpContext { inv_width, inv_height, half, x_range, y_range, prec, rot: params.transform_matrix() }
    }

    /// Compute dc for a pixel using pre-computed constants.
    /// Only 2-3 GMP allocations per pixel instead of ~13.
    pub fn compute_dc(&self, i: usize, j: usize) -> Complex {
        let mut i_float = Float::with_val(self.prec, i as f64);
        i_float += &self.half;
        let mut j_float = Float::with_val(self.prec, j as f64);
        j_float += &self.half;
        let mut x_ratio = Float::with_val(self.prec, &i_float * &self.inv_width);
        let mut y_ratio = Float::with_val(self.prec, &j_float * &self.inv_height);
        x_ratio -= &self.half;
        y_ratio -= &self.half;
        let dx = Float::with_val(self.prec, &x_ratio * &self.x_range);
        let dy = Float::with_val(self.prec, &y_ratio * &self.y_range);
        match self.rot {
            Some((a, b, c, d)) => {
                let dx_r = Float::with_val(self.prec, &dx * a) + Float::with_val(self.prec, &dy * b);
                let dy_r = Float::with_val(self.prec, &dx * c) + Float::with_val(self.prec, &dy * d);
                Complex::with_val(self.prec, (dx_r, dy_r))
            }
            None => Complex::with_val(self.prec, (dx, dy)),
        }
    }
}

#[allow(dead_code)]
pub fn compute_dc_gmp(
    i: usize,
    j: usize,
    params: &FractalParams,
    _center_x_gmp: &Float,
    _center_y_gmp: &Float,
    prec: u32,
) -> Complex {
    let ctx = DcGmpContext::new(params, prec);
    ctx.compute_dc(i, j)
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
    reuse: Option<(&[u32], &[Complex64], u32, u32)>,
    orbit_cache: Option<&Arc<ReferenceOrbitCache>>,
) -> Option<((Vec<u32>, Vec<Complex64>, Vec<f64>), Arc<ReferenceOrbitCache>)> {
    let perf = perf_enabled();
    let t_all_start = Instant::now();
    let t_orbit_start = Instant::now();
    // Reporter live façon Fraktaler-3 (Frame[NN%] Ref[NN%] BLA[NN%] Tile[NN%]).
    let progress = Arc::new(ProgressState::default());
    let reporter = spawn_progress_reporter(Arc::clone(&progress));
    if cancel.load(Ordering::Relaxed) {
        progress.done.store(true, Ordering::Relaxed);
        let _ = reporter.join();
        return None;
    }
    let supports = matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip | FractalType::Tricorn | FractalType::Multibrot
    );
    if !supports {
        progress.done.store(true, Ordering::Relaxed);
        let _ = reporter.join();
        return None;
    }

    // Réutiliser les pixels de la passe précédente quand les résolutions s'alignent.
    // Les pixels réutilisés sont à des positions alignées avec le même dc (à un sous-pixel près).
    // La fonction build_reuse() désactive automatiquement le reuse pour les modes de colorisation
    // nécessitant des données supplémentaires (Distance, OrbitTraps, Wings).
    // Les artefacts historiques ("cercle au centre") étaient causés par des bugs corrigés depuis
    // (BLA off-by-one, centrage pixels, glitch tolerance scaling, GMP z_ref stale).
    let reuse_for_pixels = reuse;

    let mut orbit_params = params.clone();
    orbit_params.precision_bits = compute_perturbation_precision_bits(params);

    // Check if we need full GMP perturbation (very deep zooms >10^15)
    let use_full_gmp = should_use_full_gmp_perturbation(params);

    // Use cached orbit/BLA or compute fresh
    let cache =
        compute_reference_orbit_cached(&orbit_params, Some(cancel.as_ref()), orbit_cache)?;
    let t_orbit = t_orbit_start.elapsed();
    // Ref + BLA + series complétés en bloc dans compute_reference_orbit_cached.
    progress.r#ref.store(100, Ordering::Relaxed);
    progress.bla.store(100, Ordering::Relaxed);

    // Use the cache's iteration_max if it was auto-adjusted upward by series skip ratio.
    // This ensures iterate_pixel uses the adjusted value to reveal detail that would
    // otherwise be hidden behind an insufficient iteration count.
    let params = if cache.iteration_max > params.iteration_max {
        let mut adjusted = params.clone();
        adjusted.iteration_max = cache.iteration_max;
        std::borrow::Cow::Owned(adjusted)
    } else {
        std::borrow::Cow::Borrowed(params)
    };
    let params = params.as_ref();

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
        progress.tile.store(100, Ordering::Relaxed);
        progress.done.store(true, Ordering::Relaxed);
        let _ = reporter.join();
        return Some(((iterations, zs, distances), cache));
    }

    // For very deep zooms, use full GMP perturbation path
    if use_full_gmp {
        let prec_gmp = compute_perturbation_precision_bits(params);
        let t_gmp_pixels_start = Instant::now();
        let result = render_perturbation_gmp_path(
            params, cancel, reuse_for_pixels, &cache, iterations, zs, distances,
            Arc::clone(&progress), t_all_start,
        );
        let t_gmp_pixels = t_gmp_pixels_start.elapsed();
        progress.done.store(true, Ordering::Relaxed);
        let _ = reporter.join();
        if let Some(((ref iters_ref, _, _), _)) = result.as_ref() {
            print_fractall_summary(
                "full_gmp",
                params.fractal_type,
                prec_gmp,
                params.iteration_max,
                iters_ref,
                pixel_count,
                t_gmp_pixels,
                t_all_start.elapsed(),
            );
        }
        return result;
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
    // x_range/y_range HP-aware en FloatExp pour les zooms > 1e308 où le f64
    // span underflow à 0 (cf. e1000 zoom 1e1000 → dc = 0 partout → image
    // uniforme avant ce fix).
    let (x_range_fexp, y_range_fexp) = effective_spans_fexp(params);
    let inv_width = 1.0 / params.width as f64;
    let inv_height = 1.0 / params.height as f64;

    let cancelled = AtomicBool::new(false);
    // Ne pas réutiliser les pixels pour la perturbation (voir commentaire ci-dessus)
    let reuse = build_reuse(params, reuse_for_pixels);

    // Clone cache for use in parallel iteration
    let cache_ref = Arc::clone(&cache);

    // Offset sous-pixel AA « per-frame » (en unités de pixel), constant sur
    // tout le frame : décale la grille pour le sample courant. Replié dans le
    // précalcul dc ci-dessous (la moyenne des frames colorés est faite par
    // l'appelant, CLI/GUI). Remplace l'ancien jitter per-pixel non-moyenné.
    let [aa_dx, aa_dy] = params.aa_subpixel_offset;

    // Pré-calcul dc pour amortir le coût par pixel (surtout utile sur petites images).
    // Stocker directement en FloatExp pour éviter les conversions via Complex64.
    // dx/dy sont précalculés sans rotation ; K (rot) est appliqué par pixel dans
    // la boucle (mélange re/im) — cf. plus bas.
    let rot = params.transform_matrix();
    let dc_re_fexp: Vec<FloatExp> = (0..width)
        .map(|i| x_range_fexp * ((i as f64 + 0.5 + aa_dx) * inv_width - 0.5))
        .collect();
    let dc_im_fexp: Vec<FloatExp> = (0..height)
        .map(|j| y_range_fexp * ((j as f64 + 0.5 + aa_dy) * inv_height - 0.5))
        .collect();

    let t_pixels_start = Instant::now();
    // Finer chunk granularity to improve rayon work-stealing. Whole rows
    // (width pixels) caused load imbalance when a row straddles fast-escaping
    // exterior pixels and slow interior/glitched ones. Target ~16 chunks per
    // thread so stragglers are redistributed, with a floor of 64 pixels to
    // avoid per-chunk overhead dominating on small images.
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = {
        let target = pixel_count / (num_threads * 16).max(1);
        target.max(64).min(width.saturating_mul(4).max(64))
    };
    let chunks_done = Arc::new(AtomicU32::new(0));
    let total_chunks = ((pixel_count + chunk_size - 1) / chunk_size).max(1) as u32;
    // Rayon ne garantit pas que `enumerate()` sur ce parallèle suit l'ordre des
    // pixels dans le tampon : dériver l'index du début de tranche depuis l'adresse.
    let iterations_base_addr = iterations.as_ptr() as usize;
    iterations
        .par_chunks_mut(chunk_size)
        .zip(zs.par_chunks_mut(chunk_size))
        .zip(distances.par_chunks_mut(chunk_size))
        .for_each(|((iter_chunk, z_chunk), dist_chunk)| {
            let reuse_row = reuse.as_ref();
            let chunk_start = (iter_chunk.as_ptr() as usize - iterations_base_addr)
                / std::mem::size_of::<u32>();
            // Cooperative cancel: poll once per chunk rather than per row.
            if cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }

            for (local_idx, ((iter, z), dist)) in iter_chunk.iter_mut().zip(z_chunk.iter_mut()).zip(dist_chunk.iter_mut()).enumerate() {
                let pixel_idx = chunk_start + local_idx;
                let i = pixel_idx % width;
                let j = pixel_idx / width;
                let dc_im = dc_im_fexp[j];
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

                // dc précalculé (inclut l'offset sous-pixel AA per-frame
                // `params.aa_subpixel_offset`, replié dans le précalcul plus haut).
                let dc = ComplexExp {
                    re: dc_re_fexp[i],
                    im: dc_im,
                };

                // Rotation : dc' = K * dc (aligné F3 hybrid.cc:265).
                // Cas dominant rot=None : no-op. Sinon, mélange re/im en restant
                // sur FloatExp pour préserver l'exposant étendu en deep zoom.
                let dc = match rot {
                    Some((a, b, c, d)) => ComplexExp {
                        re: dc.re * a + dc.im * b,
                        im: dc.re * c + dc.im * d,
                    },
                    None => dc,
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
                    iterate_pixel_hybrid_bla(
                        params,
                        hybrid,
                        cache_ref.series_table.as_ref(),
                        delta0,
                        dc_term,
                    )
                } else {
                    iterate_pixel(
                        params,
                        &cache_ref.orbit,
                        &cache_ref.bla_table,
                        cache_ref.series_table.as_ref(),
                        delta0,
                        dc_term,
                        None,
                        None,
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
            // Progression Tile[%] : une unité de parallélisme = un chunk (pas une ligne).
            let done = chunks_done.fetch_add(1, Ordering::Relaxed) + 1;
            progress
                .tile
                .store((done * 100 / total_chunks).min(100), Ordering::Relaxed);
        });
    let t_pixels = t_pixels_start.elapsed();

    if cancelled.load(Ordering::Relaxed) {
        progress.done.store(true, Ordering::Relaxed);
        let _ = reporter.join();
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

        // Neighbor pass (heuristique Pauldelbrot legacy) : flag les pixels dont
        // l'itération diffère fortement des voisins. Inutile + nuisible quand le
        // path bytecode/F3 est utilisé car (a) le rebasing F3 prévient les vrais
        // glitches structurellement, (b) sur le détail fractal fin les sauts
        // d'itération entre pixels adjacents sont réels, pas des glitches, et
        // (c) les pixels flaggés sont re-rendus via GMP (path secondary refs)
        // dont le résultat diverge légèrement du fexp → diff visuelle artificielle.
        let bytecode_path = uses_bytecode_path(params);
        if !small_image && params.glitch_neighbor_pass && !bytecode_path {
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
        // Skip secondary references entirely when bytecode/F3 path is used.
        // Le bytecode pixel_loop flag `glitched: true` UNIQUEMENT pour les
        // pixels en orbite référence exhaustée (centres escape-time), qui
        // sont resolus par `iterate_pixel_gmp` (per-pixel GMP) en aval.
        // Les "vrais" glitches Pauldelbrot ne sont pas produits par le
        // bytecode (rebasing F3 strict les prévient structurellement), donc
        // les références secondaires (overhead lourd) restent inutiles ici.
        if !small_image && params.max_secondary_refs > 0 && !bytecode_path {
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
            //
            // Improvement inspired by rust-fractal-core: parallelize pixel re-rendering
            // within each cluster using rayon. The orbit computation is sequential (must be),
            // but once the orbit + BLA table are ready, all pixels in the cluster can be
            // re-rendered in parallel.
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
                        let is_julia = params.fractal_type == FractalType::Julia;
                        let pixel_size = (params.span_x.abs() / params.width.max(1) as f64)
                            .max(params.span_y.abs() / params.height.max(1) as f64);
                        let adaptive_order = series::compute_adaptive_series_order(
                            pixel_size,
                            params.iteration_max,
                            params.series_order,
                        ).max(4);
                        let interval = if sec_orbit.z_ref_f64.len() > 100_000 { 10 } else { 1 };
                        Some(series::build_series_table_ho(&sec_orbit.z_ref_f64, is_julia, adaptive_order, interval))
                    } else {
                        None
                    };

                    // Re-render cluster pixels with secondary reference (parallelized).
                    // Inspired by rust-fractal-core: parallelize pixel iteration within each
                    // glitch cluster for significant speedup on large clusters.
                    let results: Vec<(usize, delta::DeltaResult)> = cluster.pixel_indices
                        .par_iter()
                        .map(|&idx| {
                            let px = idx % width;
                            let py = idx / width;

                            // Compute dc relative to secondary center (pixel center = (px+0.5)/width)
                            let dc_re = ((px as f64 + 0.5) * inv_width - 0.5) * x_range
                                - (cluster.center_x - params.center_x);
                            let dc_im = ((py as f64 + 0.5) * inv_height - 0.5) * y_range
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
                                None,
                                None,
                            );
                            (idx, result)
                        })
                        .collect();

                    // Apply results (sequential write to avoid race conditions)
                    for (idx, result) in results {
                        if !result.glitched && !result.suspect {
                            iterations[idx] = result.iteration;
                            zs[idx] = result.z_final;
                            glitch_mask[idx] = false;
                        }
                    }
                }
            }
        }

        // Second pass: recursive iteration-based glitch resolution (inspired by rust-fractal-core).
        //
        // rust-fractal-core's `resolve_glitches()` groups glitched pixels by iteration depth,
        // creates delta-based references using the pixel with smallest |z| as center,
        // and recursively resolves remaining glitches. This is more effective than spatial
        // clustering for glitches at similar iterations but spatially dispersed.
        //
        // Key improvements over previous approach:
        // 1. Delta-based reference: uses existing orbit + delta offset (faster than full recompute)
        // 2. Recursive: after resolving one level, remaining glitches are re-resolved
        // 3. Selects optimal reference pixel (smallest |z| norm in each group)
        if !small_image && params.max_secondary_refs > 0 {
            let max_resolution_rounds = 3; // Limit recursion depth to avoid infinite loops
            for _round in 0..max_resolution_rounds {
                let remaining_glitches: usize = glitch_mask.iter().filter(|v| **v).count();
                let remaining_ratio = remaining_glitches as f64 / (width * height) as f64;
                // Only apply if >0.5% pixels still glitched (lowered from 1% for more thorough resolution)
                if remaining_ratio < 0.005 {
                    break;
                }

                if cancel.load(Ordering::Relaxed) {
                    break;
                }

                let iter_clusters = segregate_glitches_by_iteration(
                    &glitch_mask,
                    &iterations,
                    &zs,
                    params.width,
                    params.height,
                    params,
                    params.min_glitch_cluster_size as usize,
                );

                if iter_clusters.is_empty() {
                    break;
                }

                let max_iter_refs = (params.max_secondary_refs as usize).min(iter_clusters.len());
                let mut resolved_any = false;

                for cluster in iter_clusters.iter().take(max_iter_refs) {
                    // Try delta-based reference creation from existing orbit first
                    // (inspired by rust-fractal-core's get_glitch_resolving_reference).
                    // This is much faster than computing a full new orbit from scratch.
                    let best_idx = cluster.pixel_indices.iter()
                        .min_by(|&&a, &&b| {
                            let na = zs[a].norm_sqr();
                            let nb = zs[b].norm_sqr();
                            na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .copied()
                        .unwrap_or(cluster.pixel_indices[0]);
                    let best_iter = iterations[best_idx];

                    // Compute deltas for the best pixel (reference center for this group)
                    let bpx = best_idx % width;
                    let bpy = best_idx / width;
                    let best_dc_re = ((bpx as f64 + 0.5) * inv_width - 0.5) * x_range;
                    let best_dc_im = ((bpy as f64 + 0.5) * inv_height - 0.5) * y_range;
                    let best_z_final = zs[best_idx];

                    // Try delta-based reference from existing orbit (fast path)
                    let sec_orbit_opt = cache_ref.orbit.create_glitch_reference(
                        best_iter,
                        best_dc_re,
                        best_dc_im,
                        best_z_final.re - cache_ref.orbit.z_ref_f64.get(best_iter as usize)
                            .map_or(0.0, |z| z.re),
                        best_z_final.im - cache_ref.orbit.z_ref_f64.get(best_iter as usize)
                            .map_or(0.0, |z| z.im),
                        params,
                        Some(cancel.as_ref()),
                    );

                    // Fall back to full orbit computation if delta-based fails
                    let sec_orbit = match sec_orbit_opt {
                        Some(orbit) => orbit,
                        None => {
                            let mut sec_params = orbit_params.clone();
                            sec_params.center_x = cluster.center_x;
                            sec_params.center_y = cluster.center_y;
                            match compute_reference_orbit(&sec_params, Some(cancel.as_ref())) {
                                Some((orbit, _, _)) => orbit,
                                None => continue,
                            }
                        }
                    };

                    let sec_bla = bla::build_bla_table(&sec_orbit.z_ref_f64, &orbit_params, sec_orbit.cref);

                    // Parallelize pixel iteration within each cluster (inspired by rust-fractal-core).
                    let results: Vec<(usize, delta::DeltaResult)> = cluster.pixel_indices
                        .par_iter()
                        .map(|&idx| {
                            let px = idx % width;
                            let py = idx / width;

                            let dc_re = ((px as f64 + 0.5) * inv_width - 0.5) * x_range
                                - (cluster.center_x - params.center_x);
                            let dc_im = ((py as f64 + 0.5) * inv_height - 0.5) * y_range
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
                                None,
                                delta0,
                                dc_term,
                                None,
                                None,
                            );
                            (idx, result)
                        })
                        .collect();

                    for (idx, result) in results {
                        if !result.glitched && !result.suspect {
                            iterations[idx] = result.iteration;
                            zs[idx] = result.z_final;
                            glitch_mask[idx] = false;
                            resolved_any = true;
                        }
                    }
                }

                // If no pixels were resolved in this round, stop recursing
                if !resolved_any {
                    break;
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

        // Le bytecode flag `glitched: true` UNIQUEMENT en exhaustion d'orbite
        // référence (centres escape-time non-périodiques). Dans ce cas, l'orbite
        // GMP est elle aussi tronquée — `iterate_pixel_gmp` (perturbation GMP)
        // cape au même iter que pixel_loop_exp et donne le même mauvais résultat
        // uniforme (cf. e113.toml). Seul `iterate_point_mpc` (full GMP per pixel,
        // sans dépendance à l'orbite référence) produit le bon iter d'escape.
        // On autorise donc le full recalc même en bytecode_path. Les "vrais"
        // glitches Pauldelbrot ne sont jamais flaggés par le bytecode (rebasing
        // F3 strict les prévient), donc tout pixel glitched ici est ref_exhausted.
        let allow_full_gmp_fallback = true;

        if allow_full_gmp_fallback && glitch_ratio > GLITCH_FALLBACK_THRESHOLD {
            // Trop de glitches: recalculer tous les pixels en GMP
            let gmp_params = MpcParams::from_params(&orbit_params);
            let prec = compute_perturbation_precision_bits(params);
            let width_u32 = params.width;

            // Pre-compute shared GMP constants for dc computation
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
            let dc_ctx = DcGmpContext::new(params, prec);

            let all_corrections: Vec<_> = (0..total_pixels)
                .into_par_iter()
                .map(|idx| {
                    let i = (idx as u32 % width_u32) as usize;
                    let j = (idx as u32 / width_u32) as usize;

                    // Calculer dc en GMP directement
                    let dc_gmp = dc_ctx.compute_dc(i, j);
                    
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
            let prec = compute_perturbation_precision_bits(params);
            let width_u32 = params.width;
            let dc_ctx = DcGmpContext::new(params, prec);

            // Stratégie deux-passes : (a) perturbation GMP per-pixel
            // (`iterate_pixel_gmp`) — rapide (réutilise l'orbite référence,
            // ~10³× plus rapide que le full GMP). (b) si la majorité des
            // pixels saturent à `effective_len-1` (signe que l'orbite
            // référence est trop courte), on bascule sur `iterate_point_mpc`
            // (full GMP per-pixel, lent mais sans dépendance à l'orbite) pour
            // récupérer le vrai escape iter. Le seuil 30 % est aligné sur
            // GLITCH_FALLBACK_THRESHOLD plus haut.
            use crate::fractal::perturbation::delta::iterate_pixel_gmp;
            let effective_len = cache.orbit.effective_len() as u32;
            let cap_iter = params.iteration_max.min(effective_len.saturating_sub(1));
            let corrections: Vec<_> = glitched_indices
                .par_iter()
                .map(|&idx| {
                    let i = (idx as u32 % width_u32) as usize;
                    let j = (idx as u32 / width_u32) as usize;
                    let dc_gmp = dc_ctx.compute_dc(i, j);
                    let result = iterate_pixel_gmp(
                        params,
                        &cache.orbit,
                        &dc_gmp,
                        prec,
                    );
                    (idx, result.iteration, result.z_final)
                })
                .collect();

            // Détection de saturation : si la grande majorité des pixels glitchés
            // sont coincés à `cap_iter` (saturation à la fin de l'orbite référence),
            // l'orbite référence est inutilisable — on refait ces pixels en pure GMP.
            let saturated_count = corrections.iter().filter(|&&(_, it, _)| it >= cap_iter).count();
            let need_pure_gmp = bytecode_path
                && cap_iter < params.iteration_max
                && corrections.len() > 0
                && saturated_count as f64 / corrections.len() as f64 > 0.30;

            if need_pure_gmp {
                let gmp_params = MpcParams::from_params(&orbit_params);
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
                let pure_corrections: Vec<_> = glitched_indices
                    .par_iter()
                    .map(|&idx| {
                        let i = (idx as u32 % width_u32) as usize;
                        let j = (idx as u32 / width_u32) as usize;
                        let dc_gmp = dc_ctx.compute_dc(i, j);
                        let mut z_pixel_re = center_x_gmp.clone();
                        z_pixel_re += dc_gmp.real();
                        let mut z_pixel_im = center_y_gmp.clone();
                        z_pixel_im += dc_gmp.imag();
                        let z_pixel = complex_from_xy(prec, z_pixel_re, z_pixel_im);
                        let (iter_val, z_final) = iterate_point_mpc(&gmp_params, &z_pixel);
                        (idx, iter_val, complex_to_complex64(&z_final))
                    })
                    .collect();
                for (idx, iter_val, z_final) in pure_corrections {
                    iterations[idx] = iter_val;
                    zs[idx] = z_final;
                }
            } else {
                for (idx, iter_val, z_final) in corrections {
                    iterations[idx] = iter_val;
                    zs[idx] = z_final;
                }
            }
        }
        let t_post = t_post_start.elapsed();

        if perf {
            let pixel_size = (params.span_x.abs() / params.width.max(1) as f64)
            .max(params.span_y.abs() / params.height.max(1) as f64);
            let zoom = if pixel_size.is_finite() && pixel_size > 0.0 {
                4.0 / pixel_size
            } else {
                0.0
            };
            // Effective work per pixel = smoking gun for BLA / rebasing efficiency.
            // avg ≪ params.iteration_max → BLA + rebasing skipping correctly.
            // avg ≈ params.iteration_max → BLA not helping, the pixel loop is
            // doing the full iteration count per pixel and the cost scales linearly
            // with iteration_max regardless of zoom depth.
            let total_iters: u64 = iterations.iter().map(|&n| n as u64).sum();
            let max_iter = iterations.iter().copied().max().unwrap_or(0);
            let avg_iter = if pixel_count > 0 {
                total_iters as f64 / pixel_count as f64
            } else {
                0.0
            };
            let total = t_all_start.elapsed().as_secs_f64();
            let ns_per_iter = if total_iters > 0 {
                t_pixels.as_secs_f64() * 1e9 / total_iters as f64
            } else {
                0.0
            };
            eprintln!(
                "[PERTURB PERF] {}x{} pixels={} zoom={:.2e} small_image={} orbit={:.3}s pixels={:.3}s post={:.3}s total={:.3}s avg_iter/px={:.0} max_iter/px={} ns/iter={:.1} glitched_initial={} corrections={} fallback_ratio={:.3}",
                params.width,
                params.height,
                pixel_count,
                zoom,
                small_image,
                t_orbit.as_secs_f64(),
                t_pixels.as_secs_f64(),
                t_post.as_secs_f64(),
                total,
                avg_iter,
                max_iter,
                ns_per_iter,
                glitched_initial,
                corrections_requested,
                glitch_ratio,
            );
        }

        // Reporter live + ligne finale [FRACTALL] (format aligné F3 pour
        // comparaison directe avec sa sortie batch).
        progress.tile.store(100, Ordering::Relaxed);
        progress.done.store(true, Ordering::Relaxed);
        let _ = reporter.join();
        let path_label = bytecode_path_label(params).unwrap_or("legacy_fexp");
        print_fractall_summary(
            path_label,
            params.fractal_type,
            orbit_params.precision_bits,
            params.iteration_max,
            &iterations,
            pixel_count,
            t_pixels,
            t_all_start.elapsed(),
        );
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
    progress: Arc<ProgressState>,
    t_all_start: Instant,
) -> Option<((Vec<u32>, Vec<Complex64>, Vec<f64>), Arc<ReferenceOrbitCache>)> {
    // Utiliser la précision calculée au lieu du preset
    let prec = compute_perturbation_precision_bits(params);
    let width = params.width as usize;
    let height = params.height as usize;
    let _pixel_count = width.saturating_mul(height);
    let t_pixels_start = Instant::now();
    
    // IMPORTANT: Vérifier que la précision du cache correspond à la précision calculée
    // Si la précision du cache est inférieure, cela peut causer des erreurs de précision
    if cache.precision_bits < prec {
        let ps = (params.span_x.abs() / params.width as f64)
            .max(params.span_y.abs() / params.height as f64);
        eprintln!("[PRECISION WARNING] Cache precision ({}) < required precision ({}) for zoom {:.2e}. Cache may need recomputation.",
            cache.precision_bits, prec, ps);
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
    
    let cancelled = AtomicBool::new(false);
    // Réutiliser les pixels alignés de la passe précédente (même logique que le chemin f64).
    // build_reuse() valide l'alignement et désactive le reuse pour les modes distance/orbit.
    let reuse_data = build_reuse(params, reuse);
    
    // Clone cache for use in parallel iteration
    let cache_ref = Arc::clone(cache);
    
    // Collect glitched pixels for correction
    let glitch_mask: Vec<AtomicBool> = (0..width * params.height as usize)
        .map(|_| AtomicBool::new(false))
        .collect();

    // Pre-compute shared GMP constants for dc computation
    let dc_ctx = DcGmpContext::new(params, prec);

    let rows_done = Arc::new(AtomicU32::new(0));
    let total_rows_f = height.max(1) as u32;
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
                let dc_gmp = dc_ctx.compute_dc(i, j);

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
            let done = rows_done.fetch_add(1, Ordering::Relaxed) + 1;
            progress.tile.store((done * 100 / total_rows_f).min(100), Ordering::Relaxed);
        });
    let t_pixels = t_pixels_start.elapsed();

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
            
            // Pre-compute shared GMP constants for dc computation
            let dc_ctx = DcGmpContext::new(params, prec);

            let corrections: Vec<_> = glitched_indices
                .par_iter()
                .map(|&idx| {
                    let i = (idx as u32 % width_u32) as usize;
                    let j = (idx as u32 / width_u32) as usize;

                    // Calculate pixel point directly in GMP: center + dc
                    let dc_gmp = dc_ctx.compute_dc(i, j);
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

        progress.tile.store(100, Ordering::Relaxed);
        // Le summary [FRACTALL] est imprimé par le caller, après join du reporter,
        // pour que la ligne finale `Frame[100%] ...` apparaisse AVANT [FRACTALL].
        let _ = t_pixels;
        let _ = t_all_start;
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

    #[test]
    fn dbg_effective_spans_extreme_zoom() {
        use super::effective_spans_fexp;
        for (label, span_str, expected_log2) in [
            ("e50", "4e-50", -161.0),
            ("e1000", "4e-1000", -3320.0),
            ("e1121", "9.68e-1122", -3725.0),
        ] {
            let mut p = default_params_for_type(FractalType::Mandelbrot, 200, 200);
            p.span_x = 0.0;
            p.span_y = 0.0;
            p.span_x_hp = Some(span_str.to_string());
            p.span_y_hp = Some(span_str.to_string());
            let (sx, _sy) = effective_spans_fexp(&p);
            let actual_log2 = (sx.mantissa.abs().ln() / 2.0f64.ln()) + sx.exponent as f64;
            eprintln!(
                "{}: span_str={} sx=(mant={:.4}, exp={}) log2={:.2} expected_log2={}",
                label, span_str, sx.mantissa, sx.exponent, actual_log2, expected_log2
            );
            assert!(sx.mantissa != 0.0, "{} mantissa zero!", label);
            assert!(
                (actual_log2 - expected_log2).abs() < 5.0,
                "{} log2 mismatch: got {} expected {}",
                label, actual_log2, expected_log2
            );
        }
    }

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

    fn assert_close_iterations(params: &FractalParams, indices: &[(u32, u32)], tolerance: i32) {
        let cancel = Arc::new(AtomicBool::new(false));
        let (iters, _, _) =
            render_perturbation_cancellable_with_reuse(params, &cancel, None).unwrap();
        for &(x, y) in indices {
            let idx = (y * params.width + x) as usize;
            // Utiliser center+span directement pour éviter les problèmes de précision
            // +0.5 pour centrer sur le pixel (même convention que le rendu)
            let x_ratio = (x as f64 + 0.5) / params.width as f64;
            let y_ratio = (y as f64 + 0.5) / params.height as f64;
            let xg = params.center_x + (x_ratio - 0.5) * params.span_x;
            let yg = params.center_y + (y_ratio - 0.5) * params.span_y;
            let z_pixel = Complex64::new(xg, yg);
            let ref_iter = iterate_point(params, z_pixel).iteration;
            let got = iters[idx];
            let diff = (got as i32 - ref_iter as i32).abs();
            assert!(
                diff <= tolerance,
                "iter mismatch: got {got}, ref {ref_iter}, diff {diff} > tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn perturbation_matches_f64_mandelbrot() {
        let mut params = base_params(FractalType::Mandelbrot);
        // xmin=-2.5, xmax=1.5 -> center=-0.5, span=4.0
        params.center_x = -0.5;
        params.span_x = 4.0;
        assert_close_iterations(&params, &[(0, 0), (2, 2), (4, 4)], 1);
    }

    #[test]
    fn perturbation_matches_f64_julia() {
        let mut params = base_params(FractalType::Julia);
        params.seed = Complex64::new(0.36228, -0.0777);
        // Tolérance plus large que pour Mandelbrot : l'orbite de référence Julia
        // n'a pas un point critique 0 stable, donc avec REFERENCE_BAILOUT_SQR=1e10
        // (F3-aligned, cf. orbit.rs:243) elle peut accumuler |z|² largement au-delà
        // de bailout pixel = 16 avant de bailer. Le bruit numérique sur les grandes
        // valeurs z_ref peut décaler la détection d'escape de quelques itérations
        // côté perturbation vs f64 pur. Le rendu visuel reste correct, c'est une
        // marge attendue. Si la tolérance doit monter au-delà de 5 → investiguer.
        assert_close_iterations(&params, &[(1, 1), (2, 2), (3, 3)], 5);
    }

    #[test]
    fn perturbation_matches_f64_burning_ship() {
        let mut params = base_params(FractalType::BurningShip);
        // xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0 -> center=(-0.5, 0), span=(4, 4)
        params.center_x = -0.5;
        params.center_y = 0.0;
        params.span_x = 4.0;
        params.span_y = 4.0;
        assert_close_iterations(&params, &[(0, 4), (2, 2), (4, 0)], 1);
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

    #[test]
    fn should_rebase_hysteresis() {
        use super::delta::should_rebase;

        // Defaut: hysteresis=1.0 (F3-strict), rebase si z_curr < delta.
        // L'hysteresis <1.0 est opt-in via FRACTALL_REBASE_HYSTERESIS env var.

        // Standard rebase: z_curr < delta
        assert!(should_rebase(0.1, 1.0, 0.5));
        // Rebase aussi quand z_curr est proche mais inferieur (sans hysteresis)
        assert!(should_rebase(0.8, 1.0, 0.5));
        // Pas de rebase quand z_curr >= delta
        assert!(!should_rebase(1.0, 1.0, 0.5));
        assert!(!should_rebase(1.2, 1.0, 0.5));

        // No rebase: z_ref est minuscule (pres d'un zero de l'orbite)
        assert!(!should_rebase(0.1, 1.0, 1e-25));

        // No rebase: valeurs nulles
        assert!(!should_rebase(0.0, 1.0, 0.5));
        assert!(!should_rebase(0.1, 0.0, 0.5));
    }
}

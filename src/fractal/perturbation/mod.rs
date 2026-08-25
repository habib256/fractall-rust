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
//!   ```text
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
use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;
use rayon::prelude::*;

use crate::fractal::bytecode::compile_formula;
use crate::fractal::bytecode::pixel_loop_gmp::iterate_pixel_gmp;
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::perturbation::compress::strip_orbit_arrays_for_compress;
use crate::fractal::perturbation::delta::{bytecode_path_label, iterate_pixel_with_dd};
use crate::fractal::perturbation::orbit::{
    compute_reference_orbit, compute_reference_orbit_cached,
};
use crate::fractal::perturbation::types::{ComplexExp, DeltaResult, FloatExp};
use crate::fractal::{FractalParams, FractalType};
use rug::Float;

/// Le pixel passe par le path bytecode/F3 (BLA mat2 + rebasing F3 strict) ?
/// Si oui, `iterate_pixel` retourne toujours `glitched: false` et le post-traitement
/// (neighbor pass Pauldelbrot + secondary references) n'est qu'overhead + source
/// de pixels divergents (corrigés via GMP avec résultat ≠ fexp).
fn uses_bytecode_path(params: &FractalParams) -> bool {
    params.engine.use_bytecode_engine
        && compile_formula(params.fractal_type, params.formula.multibrot_power).is_some()
}

pub mod bla;
pub mod compress;
pub mod counter;
pub mod dd;
#[cfg(test)]
pub mod debug_pure_f3;
pub mod delta;
pub mod nonconformal;
pub mod nucleus;
pub mod orbit;
pub mod pixel_math;
mod precision;
mod progress;
mod reuse;
pub mod sampling;
pub mod series;
pub mod types;
pub use orbit::{HybridBlaReferences, ReferenceOrbitCache};
pub use precision::should_use_full_gmp_perturbation;
pub(crate) use precision::{
    compute_perturbation_precision_bits, effective_pixel_size, effective_spans_dd,
    effective_spans_fexp, log2_zoom,
};
pub(crate) use progress::{
    perf_enabled, print_fractall_summary, spawn_progress_reporter, ProgressState,
};
use reuse::build_reuse;
pub use sampling::{DcGmpContext, ResolvedSamplingPlan};

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
) -> DeltaResult {
    use crate::fractal::perturbation::types::ComplexExp;

    if hybrid_refs.cycle_period == 0 {
        // No cycle detected: use primary reference (single reference)
        return iterate_pixel_with_dd(
            delta::PerturbPixelRequest::new(
                params,
                &hybrid_refs.primary,
                &hybrid_refs.primary_bla,
                delta0,
                dc,
            )
            .with_series(series_table),
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
        let result = iterate_pixel_with_dd(
            delta::PerturbPixelRequest::new(&phase_params, ref_orbit, bla_table, delta, dc)
                .with_series(series_table)
                .with_hybrid_state(&mut current_phase, hybrid_refs),
        );

        total_iterations += result.iteration;

        // Check if we escaped, glitched, or reached max iterations
        if result.z_final.norm_sqr() > params.bailout * params.bailout
            || result.glitched
            || total_iterations >= params.iteration_max
        {
            return DeltaResult {
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
            return DeltaResult {
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
    iterate_pixel_with_dd(
        delta::PerturbPixelRequest::new(
            params,
            &hybrid_refs.primary,
            &hybrid_refs.primary_bla,
            delta0,
            dc,
        )
        .with_series(series_table),
    )
}

pub fn render_perturbation_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<(&[u32], &[Complex64], u32, u32)>,
) -> Option<(Vec<u32>, Vec<Complex64>, Vec<f64>)> {
    let (result, _cache) = render_perturbation_with_cache(params, cancel, reuse, None, None, None)?;
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
    xaos: Option<&crate::fractal::xaos::XaosMap>,
    tiles: Option<&crate::render::tiles::TileOpts>,
) -> Option<(
    (Vec<u32>, Vec<Complex64>, Vec<f64>),
    Arc<ReferenceOrbitCache>,
)> {
    // Garde-fou (miroir du dispatcher, pour les appels directs) : mapping XaoS
    // ignoré si ses dimensions ne correspondent pas aux params (indexation
    // hors bornes sinon).
    let xaos = xaos.filter(|m| {
        m.src_col.len() == params.width as usize && m.src_row.len() == params.height as usize
    });
    // INVARIANT G10.4b (miroir du dispatcher) : écho XaoS ⊃ pas de reuse
    // basse-résolution — les pixels frais du map doivent être réellement
    // calculés (le reuse copie des centres décalés de (ratio−1)/2 px, ce qui
    // contaminerait les axes déclarés exacts, consommés par le refine union).
    let reuse = if xaos.is_some() { None } else { reuse };
    // Fix G3 (anneaux concentriques) : `max_perturb_iterations` / `max_bla_steps`
    // ne doivent JAMAIS plafonner sous `iteration_max`. Comme `iters_ptb ≤ n <
    // iteration_max`, un cap < iteration_max tronque les pixels qui ont besoin de
    // beaucoup de pas directs → ils sortent tôt avec un compte d'itération
    // ~radial → anneaux (cf. cusp -0.75, défaut 1024 < iter requis ~1700). F3 met
    // `maximum_perturb_iterations = iterations` ; on s'aligne. Le loader TOML le
    // faisait déjà ; ici on couvre GUI + CLI non-TOML (chemin commun).
    let params = &{
        let mut p = params.clone();
        p.perturbation.max_perturb_iterations = p.perturbation.max_perturb_iterations.max(p.iteration_max);
        p.perturbation.max_bla_steps = p.perturbation.max_bla_steps.max(p.iteration_max);
        p
    };
    let perf = perf_enabled();
    let t_all_start = Instant::now();
    let t_orbit_start = Instant::now();
    // Reporter live façon Fraktaler-3 (Frame[NN%] Ref[NN%] BLA[NN%] Tile[NN%]).
    let progress = Arc::new(ProgressState::default());
    let reporter = spawn_progress_reporter(Arc::clone(&progress));
    if cancel.load(Ordering::Relaxed) {
        progress.finish();
        let _ = reporter.join();
        return None;
    }
    let supports = matches!(
        params.fractal_type,
        FractalType::Mandelbrot
            | FractalType::Julia
            | FractalType::BurningShip
            | FractalType::Tricorn
            | FractalType::Multibrot
    );
    if !supports {
        progress.finish();
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
    orbit_params.engine.precision_bits = compute_perturbation_precision_bits(params);

    // Check if we need full GMP perturbation (very deep zooms >10^15)
    let use_full_gmp = should_use_full_gmp_perturbation(params);

    // Use cached orbit/BLA or compute fresh
    let cache = compute_reference_orbit_cached(
        &orbit_params,
        Some(cancel.as_ref()),
        orbit_cache,
        Some(&progress.r#ref),
        true, // G10.2 : chemin CPU FloatExp → réutilisation off-center autorisée (offset dc propagé plus bas)
    )?;
    let t_orbit = t_orbit_start.elapsed();
    // Ref + BLA + series complétés en bloc dans compute_reference_orbit_cached.
    progress.r#ref.store(100, Ordering::Relaxed);
    progress.bla.store(100, Ordering::Relaxed);

    // Use the cache's iteration_max if it was auto-adjusted upward by series skip ratio.
    // This ensures iterate_pixel uses the adjusted value to reveal detail that would
    // otherwise be hidden behind an insufficient iteration count.
    // Idem pour la matrice K du nucleus finder : F3 rend la vue dans le frame
    // du minibrot (`out.transform = K`, `engine.cc:208`). K vit dans le cache
    // (calculée avec l'orbite) et s'applique ici au mapping pixel→c —
    // `transform_matrix()` (rot) ET `transform_sigma1()` (rayon BLA).
    let nucleus_k = if params.engine.find_nucleus {
        cache.nucleus_transform
    } else {
        None
    };
    let params = if cache.iteration_max > params.iteration_max || nucleus_k.is_some() {
        let mut adjusted = params.clone();
        adjusted.iteration_max = adjusted.iteration_max.max(cache.iteration_max);
        if let Some((k, rot_deg)) = nucleus_k {
            adjusted.transform_k = Some(k);
            adjusted.rotation = rot_deg;
        }
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
        progress.finish();
        let _ = reporter.join();
        return Some(((iterations, zs, distances), cache));
    }

    // For very deep zooms, use full GMP perturbation path
    if use_full_gmp {
        let prec_gmp = compute_perturbation_precision_bits(params);
        let t_gmp_pixels_start = Instant::now();
        let result = render_perturbation_gmp_path(
            params,
            cancel,
            reuse_for_pixels,
            xaos,
            tiles,
            &cache,
            iterations,
            zs,
            distances,
            Arc::clone(&progress),
            t_all_start,
        );
        let t_gmp_pixels = t_gmp_pixels_start.elapsed();
        progress.finish();
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
    // Spans HP-aware en FloatExp pour les zooms > 1e308 où le f64
    // span underflow à 0 (cf. e1000 zoom 1e1000 → dc = 0 partout → image
    // uniforme avant ce fix).
    let (x_range_fexp, y_range_fexp) = effective_spans_fexp(params);
    let inv_width = 1.0 / params.width as f64;
    let inv_height = 1.0 / params.height as f64;

    let cancelled = AtomicBool::new(false);
    // Ne pas réutiliser les pixels pour la perturbation (voir commentaire ci-dessus)
    let reuse = build_reuse(params, reuse_for_pixels);

    // Offset sous-pixel AA « per-frame » (en unités de pixel), constant sur
    // tout le frame : décale la grille pour le sample courant. Replié dans le
    // précalcul dc ci-dessous (la moyenne des frames colorés est faite par
    // l'appelant, CLI/GUI). Remplace l'ancien jitter per-pixel non-moyenné.
    let resolved_sampling =
        ResolvedSamplingPlan::with_reference(params, cache.precision_bits.max(256), &cache);
    let sampling = resolved_sampling.sampling;
    let [aa_dx, aa_dy] = sampling.aa_uniform;
    // AA **par pixel** (Cranley-Patterson F3, `jitter::pixel_offset`) :
    // prioritaire sur l'offset uniforme. Quand actif, `aa_dx/aa_dy` valent 0
    // (le précalcul dc reste la base non-jitterée) et la boucle ajoute l'offset
    // décorrélé du pixel `jx·(span/width)`, `jy·(span/height)` en c-space avant
    // rotation. `None` hors AA → aucun surcoût (branche non prise).
    let aa_jit = sampling.aa_jitter;
    let img_w = params.width as usize;
    // Échelles pixel→c en FloatExp (survivent l'underflow deep zoom).
    let jit_re_fexp = x_range_fexp * inv_width;
    let jit_im_fexp = y_range_fexp * inv_height;

    // Pré-calcul dc pour amortir le coût par pixel (surtout utile sur petites images).
    // Stocker directement en FloatExp pour éviter les conversions via Complex64.
    // dx/dy sont précalculés sans rotation ; K (rot) est appliqué par pixel dans
    // la boucle (mélange re/im) — cf. plus bas.
    let rot = sampling.transform;
    // G10.2 : offset dc = centre_vue − centre_référence. La boucle calcule
    // `c = cref + dc` ; pour que `c` soit la coordonnée réelle du pixel il faut
    // `dc = grille + (centre_vue − cref)`. NUL quand la référence est centrée
    // (exact-reuse / rebuild, tous les chemins actuels) ; non-nul UNIQUEMENT en
    // réutilisation subset off-center (G10.2), qui est gated sans rotation → on
    // peut l'ajouter à la grille pré-rotation (séparable) sans risque.
    let (off_re_fexp, off_im_fexp) = resolved_sampling
        .ref_offset
        .as_ref()
        .map(|(re, im)| (FloatExp::from_gmp(re), FloatExp::from_gmp(im)))
        .unwrap_or((FloatExp::from_f64(0.0), FloatExp::from_f64(0.0)));
    let dc_re_fexp: Vec<FloatExp> = (0..width)
        .map(|i| x_range_fexp * ((i as f64 + 0.5 + aa_dx) * inv_width - 0.5) + off_re_fexp)
        .collect();
    let dc_im_fexp: Vec<FloatExp> = (0..height)
        .map(|j| y_range_fexp * ((j as f64 + 0.5 + aa_dy) * inv_height - 0.5) + off_im_fexp)
        .collect();

    // Tier dd : précompute `dc` en double-double (~106 b) quand le tier est
    // actif (Mandelbrot, sans rotation — K non appliqué ici). Le `dc` ComplexExp
    // ci-dessus est 53 b (span f64 × fraction f64) → plancher résiduel sur les
    // pixels de bord (grand |dc|). En dd : span depuis la string HP (106 b) et
    // fraction pixel via division dd. Vide sinon (path ComplexExp inchangé).
    let build_dc_dd = params.engine.use_dd_tier
        && matches!(params.fractal_type, FractalType::Mandelbrot)
        && rot.is_none();
    #[allow(clippy::type_complexity)]
    let (dc_re_dd, dc_im_dd, jit_re_dd, jit_im_dd): (
        Vec<crate::fractal::perturbation::dd::DoubleDoubleExp>,
        Vec<crate::fractal::perturbation::dd::DoubleDoubleExp>,
        crate::fractal::perturbation::dd::DoubleDoubleExp,
        crate::fractal::perturbation::dd::DoubleDoubleExp,
    ) = if build_dc_dd {
        use crate::fractal::perturbation::dd::DoubleDoubleExp as DdE;
        let (x_range_dd, y_range_dd) = effective_spans_dd(params);
        let width_dd = DdE::from_f64(width as f64);
        let height_dd = DdE::from_f64(height as f64);
        let half = DdE::from_f64(0.5);
        let re = (0..width)
            .map(|i| {
                let frac = DdE::from_f64(i as f64 + 0.5 + aa_dx)
                    .div(width_dd)
                    .sub(half);
                x_range_dd.mul(frac)
            })
            .collect();
        let im = (0..height)
            .map(|j| {
                let frac = DdE::from_f64(j as f64 + 0.5 + aa_dy)
                    .div(height_dd)
                    .sub(half);
                y_range_dd.mul(frac)
            })
            .collect();
        // Échelle pixel→c en dd pour l'AA par pixel (jx·span/width).
        (re, im, x_range_dd.div(width_dd), y_range_dd.div(height_dd))
    } else {
        use crate::fractal::perturbation::dd::DoubleDoubleExp as DdE;
        (
            Vec::new(),
            Vec::new(),
            DdE::from_f64(0.0),
            DdE::from_f64(0.0),
        )
    };

    // Pré-construit la table BLA partagée pendant que le pool rayon est libre :
    // son build (parallélisé) serait sinon serial sous le lock au premier pixel.
    // `cache.orbit` est l'orbite que `iterate_pixel` utilisera (branche
    // non-hybride, même ptr → hit cache). Sur les hybrides multi-phase, le pixel
    // loop consomme d'autres orbites → on saute (le build se fera comme avant).
    if cache.hybrid_refs.is_none() {
        crate::fractal::perturbation::delta::prewarm_bla_entry(params, &cache.orbit);
    }

    // ── COMPRESS phase 2 (FRACTALL_COMPRESS_REF=1) : la boucle pixel lit la
    // réf via le décompresseur à waypoints (routage delta.rs) → les tableaux
    // pleins `z_ref_f64` (16 o/iter) + `z_ref` (32 o/iter) deviennent du poids
    // mort. On les libère APRÈS le prewarm BLA (le build lit le tableau plein ;
    // la clé de cache = identité COMPRESSÉE, stable à travers la libération,
    // cf. `delta::orbit_identity`). ⚠️ Doit précéder le clone `cache_ref`
    // (Arc::try_unwrap exige refcount 1).
    let cache = strip_orbit_arrays_for_compress(cache, params);

    // Clone cache for use in parallel iteration
    let cache_ref = Arc::clone(&cache);

    let t_pixels_start = Instant::now();
    // G10.5 : file de tuiles priorité-centre (remplace le chunking linéaire).
    // La granularité {64,32,16} de TileGrid vise ≥ 8 tuiles/thread — même
    // objectif d'équilibrage que l'ancien chunking (les lignes entières
    // créaient un déséquilibre extérieur-rapide / intérieur-lent), plus
    // l'ordre curseur-d'abord et le streaming intra-passe (sink GUI).
    let tile_priority = tiles.map(|t| t.priority).unwrap_or((0.5, 0.5));
    let tile_sink = tiles.and_then(|t| t.sink);
    let grid = crate::render::tiles::TileGrid::new(width, height, tile_priority);
    let tiles_done = Arc::new(AtomicU32::new(0));
    let total_tiles = grid.order.len().max(1) as u32;
    let it_grid = grid.split(&mut iterations);
    let zs_grid = grid.split(&mut zs);
    let dist_grid = grid.split(&mut distances);
    let slots: Vec<_> = it_grid
        .into_iter()
        .zip(zs_grid)
        .zip(dist_grid)
        .map(|((it, z), di)| std::sync::Mutex::new(Some((it, z, di))))
        .collect();
    let completed = crate::render::tiles::run_prioritized(
        &grid.order,
        &slots,
        cancel.as_ref(),
        &|tile_id, tile_bufs| {
            let (mut it_rows, mut z_rows, mut dist_rows) = tile_bufs;
            let reuse_row = reuse.as_ref();
            let (x0, y0, tw, th) = grid.rect(tile_id);
            for dj in 0..th {
                // Les pixels perturbation sont lourds en deep zoom : poll
                // d'annulation par ligne de tuile (en plus du poll par tuile
                // de l'exécuteur).
                if cancel.load(Ordering::Relaxed) {
                    cancelled.store(true, Ordering::Relaxed);
                    return;
                }
                let j = y0 + dj;
                for di in 0..tw {
                    let i = x0 + di;
                    let iter = &mut it_rows[dj][di];
                    let z = &mut z_rows[dj][di];
                    let dist = &mut dist_rows[dj][di];
                    // G10.4 : copie inter-frame XaoS (écho produit ou refine
                    // union). Les pixels copiés sont absolus (iterations + z à
                    // l'échappement) : valides quelle que soit l'orbite référence.
                    if let Some(x) = xaos {
                        if let Some(sidx) = x.source_index(i, j) {
                            if let Some(&it) = x.iterations.get(sidx) {
                                *iter = it;
                                *z = x.zs[sidx];
                                continue;
                            }
                        }
                    }
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

                    // AA par pixel (Cranley-Patterson F3) : offset décorrélé du
                    // pixel, en unités de pixel. `(0,0)` hors AA per-pixel.
                    let (jx, jy) = match aa_jit {
                        Some((k, scale)) => {
                            crate::fractal::jitter::pixel_offset(img_w, i, j, k, scale)
                        }
                        None => (0.0, 0.0),
                    };

                    // dc précalculé (base séparable non-jitterée). L'offset AA
                    // par pixel (converti en c-space via l'échelle pixel→c) n'est
                    // ajouté QUE si l'AA per-pixel est actif → hors AA, `dc` est
                    // l'expression d'origine bit-identique (goldens verrouillés).
                    let mut dc = ComplexExp {
                        re: dc_re_fexp[i],
                        im: dc_im,
                    };
                    if aa_jit.is_some() {
                        dc.re = dc.re + jit_re_fexp * jx;
                        dc.im = dc.im + jit_im_fexp * jy;
                    }

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
                    // Tier dd : `dc` du pixel en ~106 b (Mandelbrot, précompute plus
                    // haut). `None` → le dispatch retombe sur le dc ComplexExp 53 b.
                    let dc_dd = if build_dc_dd {
                        use crate::fractal::perturbation::dd::DoubleDoubleExp as DdE;
                        let (mut re, mut im) = (dc_re_dd[i], dc_im_dd[j]);
                        // AA par pixel en dd : ajoute jx·(span/width), jy·(span/height).
                        // Gaté sur l'AA per-pixel → dd non-AA bit-identique.
                        if aa_jit.is_some() {
                            re = re.add(jit_re_dd.mul(DdE::from_f64(jx)));
                            im = im.add(jit_im_dd.mul(DdE::from_f64(jy)));
                        }
                        Some(crate::fractal::perturbation::dd::ComplexDDExp { re, im })
                    } else {
                        None
                    };
                    let result = if let Some(ref hybrid) = cache_ref.hybrid_refs {
                        iterate_pixel_hybrid_bla(
                            params,
                            hybrid,
                            cache_ref.series_table.as_ref(),
                            delta0,
                            dc_term,
                        )
                    } else {
                        iterate_pixel_with_dd(delta::PerturbPixelRequest {
                            params,
                            ref_orbit: &cache_ref.orbit,
                            bla_table: &cache_ref.bla_table,
                            series_table: cache_ref.series_table.as_ref(),
                            delta0,
                            dc: dc_term,
                            dc_dd,
                            current_phase: None,
                            hybrid_refs: None,
                        })
                    };

                    // Use distance estimation and interior detection results
                    // Encode is_interior in z.im sign: negative = interior point
                    // Encode distance in z.re when available (for distance-based coloring)
                    let mut z_value = result.z_final;

                    if result.is_interior {
                        // Interior point: encode flag in z.im sign (negative = interior)
                        // This allows color_for_pixel to detect and color interior points black
                        z_value = Complex64::new(z_value.re, -z_value.im.abs());
                    } else if result.distance.is_finite()
                        && result.distance != f64::INFINITY
                        && result.distance > 0.0
                    {
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
            }
            // G10.5 : streaming intra-passe — tuile terminée livrée au sink
            // (copies locales ; les buffers de sortie restent la vérité).
            if let Some(sink) = tile_sink {
                sink(crate::render::tiles::TileUpdate {
                    x0,
                    y0,
                    w: tw,
                    h: th,
                    iterations: crate::render::tiles::collect_rows(&it_rows),
                    zs: crate::render::tiles::collect_rows(&z_rows),
                    distances: crate::render::tiles::collect_rows(&dist_rows),
                });
            }
            // Progression Tile[%] : une unité de parallélisme = une tuile.
            let done = tiles_done.fetch_add(1, Ordering::Relaxed) + 1;
            progress
                .tile
                .store((done * 100 / total_tiles).min(100), Ordering::Relaxed);
        },
    );
    drop(slots);
    // Re-check : un cancel observé DANS une tuile (early-return par ligne)
    // peut ne pas être vu par l'exécuteur si la file était déjà vide.
    if !completed || cancel.load(Ordering::Relaxed) {
        cancelled.store(true, Ordering::Relaxed);
    }
    let t_pixels = t_pixels_start.elapsed();

    if cancelled.load(Ordering::Relaxed) {
        progress.finish();
        let _ = reporter.join();
        None
    } else {
        let t_post_start = Instant::now();
        let glitch_mask: Vec<bool> = glitch_mask
            .iter()
            .map(|flag| flag.load(Ordering::Relaxed))
            .collect();
        let glitched_initial = glitch_mask.iter().filter(|v| **v).count();

        // Les pixels signalés le sont par la boucle pixel elle-même (référence
        // épuisée, ou critère de fiabilité du chemin legacy) : ils sont corrigés
        // pixel par pixel en GMP plus bas. Il n'y a plus de passe d'inférence
        // par voisinage ni de références secondaires — voir l'en-tête du module.

        // Neighbor pass (heuristique Pauldelbrot legacy) : flag les pixels dont
        // l'itération diffère fortement des voisins. Inutile + nuisible quand le
        // path bytecode/F3 est utilisé car (a) le rebasing F3 prévient les vrais
        // glitches structurellement, (b) sur le détail fractal fin les sauts
        // d'itération entre pixels adjacents sont réels, pas des glitches, et
        // (c) les pixels flaggés sont re-rendus via GMP (path secondary refs)
        // dont le résultat diverge légèrement du fexp → diff visuelle artificielle.
        let bytecode_path = uses_bytecode_path(params);
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
        //
        // ⚠️ `!bytecode_path` (miroir du bloc secondary-refs ci-dessus, l.1534) :
        // sur le path bytecode/F3 les pixels flaggés sont des ref_exhausted /
        // fausses évasions de PRÉCISION (réf INTÉRIEURE frôlant zéro → annulation
        // f64 au rebase), PAS des glitches Pauldelbrot. Les re-rendre via le
        // `iterate_pixel` LEGACY (référence secondaire) leur donne une valeur
        // TOUJOURS fausse (même imprécision f64) mais NON-flaggée → un-flag
        // (glitch_mask=false) qui les retire du set ET fait tomber le glitch_ratio
        // sous `GLITCH_FALLBACK_THRESHOLD` (0.30) → le fallback full-GMP ne se
        // déclenche plus. C'était la cause du bug « réf intérieure » : PASS à
        // ≤512² (bloc sauté, small_image) mais 3.4 % de structure spurious à
        // 800×547 (bloc actif). Gate → le ratio reste haut → fallback GMP → correct.
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
        // Ratio sur les pixels réellement CALCULÉS : les pixels copiés par
        // l'écho XaoS ne sont jamais flaggés et diluaient le ratio (régime
        // réf-intérieure ~36 % flaggés → < 0.30 après écho → escalade dd
        // sautée → chaque pixel frais passait en GMP par-pixel, ordres de
        // grandeur plus lent, bug 2026-08-23).
        let computed_pixels = total_pixels
            .saturating_sub(xaos.map_or(0, |m| m.copied))
            .max(1);
        let glitch_ratio = glitched_indices.len() as f64 / computed_pixels as f64;
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
            // Escalade tier **dd** (perf) AVANT le fallback full-GMP. Le régime
            // qui déclenche ce fallback (réf INTÉRIEURE frôlant zéro → annulation
            // f64 au rebase → ~36 % de pixels ref_exhausted flaggés) est
            // exactement celui que le tier dd (~106 b) résout : le pixel loop dd
            // NE flagge PAS ces pixels (`glitched_initial=0`, mesuré) → il rend
            // toute la frame proprement, pixel-exact GMP (vérifié 800²), sans
            // re-déclencher ce fallback. Coût ~25× moindre que `iterate_point_mpc`
            // (full GMP per-pixel, ~1 µs/iter) qui suit. Mandelbrot bytecode
            // seulement (tier dd) ; garde `!use_dd_tier` = pas de récursion (si
            // le rendu dd flaggait quand même >30 %, la ré-entrée retomberait sur
            // le full-GMP ci-dessous = backstop). `tiles=None` : la frame dd est
            // blittée en entier par l'appelant (pas de double-stream du sink).
            if bytecode_path
                && !params.engine.use_dd_tier
                && matches!(params.fractal_type, FractalType::Mandelbrot)
            {
                let mut dd_params = params.clone();
                dd_params.engine.use_dd_tier = true;
                if let Some(dd_result) =
                    render_perturbation_with_cache(&dd_params, cancel, None, None, None, None)
                {
                    if perf {
                        eprintln!(
                            "[DD-ESCALATION] glitch_ratio={:.3} > {:.2} → re-render tier dd (backstop full-GMP évité)",
                            glitch_ratio, GLITCH_FALLBACK_THRESHOLD
                        );
                    }
                    return Some(dd_result);
                }
            }
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
            // Relatif à la référence pour `iterate_pixel_gmp` (compute_dc_ref) ;
            // les escalades full-GMP ci-dessous restent absolues (compute_dc).
            let dc_ctx = DcGmpContext::with_reference(params, prec, &cache);

            // `iterate_pixel_gmp` lit `z_ref_gmp` (orbite GMP dense), qui peut
            // être vide sur le path bytecode (stockage sauté à la construction,
            // cf. `compute_reference_orbit` force_dense_gmp). Le recompute ici,
            // une seule fois, avec le MÊME chemin bytecode + force_dense_gmp=true
            // → valeurs GMP bit-identiques à l'ancien stockage eager. Orbite
            // courte sur les cas qui glitchent (cusp ~2500 iters → ms) ; jamais
            // atteint sur un deep zoom nominal sans glitch (dragon).
            let rebuilt_gmp_orbit = if cache.orbit.z_ref_gmp.is_empty() {
                compute_reference_orbit(params, Some(cancel.as_ref()), true).map(|r| r.0)
            } else {
                None
            };
            let gmp_orbit = rebuilt_gmp_orbit.as_ref().unwrap_or(&cache.orbit);

            // Stratégie deux-passes : (a) perturbation GMP per-pixel
            // (`iterate_pixel_gmp`) — rapide (réutilise l'orbite référence,
            // ~10³× plus rapide que le full GMP). (b) si la majorité des
            // pixels saturent à `effective_len-1` (signe que l'orbite
            // référence est trop courte), on bascule sur `iterate_point_mpc`
            // (full GMP per-pixel, lent mais sans dépendance à l'orbite) pour
            // récupérer le vrai escape iter. Le seuil 30 % est aligné sur
            // GLITCH_FALLBACK_THRESHOLD plus haut.
            let effective_len = gmp_orbit.effective_len() as u32;
            let cap_iter = params.iteration_max.min(effective_len.saturating_sub(1));
            let corrections: Vec<_> = glitched_indices
                .par_iter()
                .map(|&idx| {
                    let i = (idx as u32 % width_u32) as usize;
                    let j = (idx as u32 / width_u32) as usize;
                    let dc_gmp = dc_ctx.compute_dc_ref(i, j);
                    let result = iterate_pixel_gmp(
                        crate::fractal::bytecode::pixel_loop_gmp::GmpPixelRequest {
                            params,
                            ref_orbit: gmp_orbit,
                            dc: &dc_gmp,
                            precision: prec,
                        },
                    );
                    (idx, result.iteration, result.z_final, result.glitched)
                })
                .collect();

            // Détection de saturation : si la grande majorité des pixels glitchés
            // sont coincés à `cap_iter` (saturation à la fin de l'orbite référence),
            // l'orbite référence est inutilisable — on refait ces pixels en pure GMP.
            let saturated_count = corrections
                .iter()
                .filter(|&&(_, it, _, _)| it >= cap_iter)
                .count();
            let need_pure_gmp = bytecode_path
                && cap_iter < params.iteration_max
                && corrections.len() > 0
                && saturated_count as f64 / corrections.len() as f64 > 0.30;

            // Pixels à escalader vers le full GMP par-pixel (`iterate_point_mpc`,
            // sans dépendance à l'orbite référence) :
            //  - `need_pure_gmp` : orbite référence trop courte (ref-exhausted,
            //    saturation à `cap_iter`) → tout le cluster.
            //  - sinon : les pixels TOUJOURS glitchés après la correction GMP-delta.
            //    C'est le **glitch de référence unique** — δ a décorrélé du vrai
            //    orbit sans que le rebase (`|Z+δ|²<|δ|²`) ni la validité BLA ne le
            //    captent. La précision GMP sur la MÊME référence ne corrige rien
            //    (structural, pas numérique) ; seul un rendu GMP indépendant de la
            //    référence résout ces pixels (cf. fuzz mandelbrot -0.615+0.401i
            //    zoom 6e7 : blob intérieur au z_pert bit-identique, faussement
            //    évadé à iter 304 vs 2048 réel).
            let escalate: Vec<usize> = if need_pure_gmp {
                glitched_indices.clone()
            } else {
                corrections
                    .iter()
                    .filter_map(|&(idx, _, _, g)| if g { Some(idx) } else { None })
                    .collect()
            };
            let escalate_set: std::collections::HashSet<usize> = escalate.iter().copied().collect();
            // Applique les corrections GMP-delta rapides pour les pixels résolus.
            for (idx, iter_val, z_final, _) in &corrections {
                if !escalate_set.contains(idx) {
                    iterations[*idx] = *iter_val;
                    zs[*idx] = *z_final;
                }
            }
            if !escalate.is_empty() {
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
                let pure_corrections: Vec<_> = escalate
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
        progress.finish();
        let _ = reporter.join();
        crate::fractal::wisdom::log_plan_if_enabled(params);
        // Marqueur du path compressé (vérifiable dans la ligne [FRACTALL]) :
        // le routage delta.rs a envoyé chaque pixel Mandelbrot f64 vers le
        // décompresseur à waypoints.
        let path_label = if delta::compressed_ref_route_active(params, &cache.orbit) {
            "bytecode_f64_compressed"
        } else if delta::harmonic_entry_active(params, &cache.orbit) {
            // Marqueur du path Harmonic LA (routé wisdom G9.3, forcé via
            // FRACTALL_HARMONIC_LA) : reflète la décision RÉELLE (table
            // construite dans l'entrée cache), pas le seul candidat.
            match crate::fractal::bytecode::harmonic_mla::harmonic_variant() {
                Some(crate::fractal::bytecode::harmonic_mla::HarmonicVariant::Mla) => {
                    "bytecode_f64_harmonic_mla"
                }
                _ => "bytecode_f64_harmonic_lla",
            }
        } else {
            bytecode_path_label(params).unwrap_or("legacy_fexp")
        };
        print_fractall_summary(
            path_label,
            params.fractal_type,
            orbit_params.engine.precision_bits,
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
    xaos: Option<&crate::fractal::xaos::XaosMap>,
    tiles: Option<&crate::render::tiles::TileOpts>,
    cache: &Arc<ReferenceOrbitCache>,
    mut iterations: Vec<u32>,
    mut zs: Vec<Complex64>,
    distances: Vec<f64>,
    progress: Arc<ProgressState>,
    t_all_start: Instant,
) -> Option<(
    (Vec<u32>, Vec<Complex64>, Vec<f64>),
    Arc<ReferenceOrbitCache>,
)> {
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
            eprintln!(
                "[PRECISION ERROR] Failed to parse center_x_gmp: {}",
                cache.center_x_gmp
            );
            return None;
        }
    };
    let center_y_gmp = match Float::parse(&cache.center_y_gmp) {
        Ok(parse_result) => Float::with_val(prec, parse_result),
        Err(_) => {
            eprintln!(
                "[PRECISION ERROR] Failed to parse center_y_gmp: {}",
                cache.center_y_gmp
            );
            return None;
        }
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

    // Pre-compute shared GMP constants for dc computation (relatif à la
    // référence `cache` : offset off-center G10.2 inclus).
    let dc_ctx = DcGmpContext::with_reference(params, prec, cache);

    // G10.5 : file de tuiles priorité-centre. Pixels GMP très lourds → poll
    // d'annulation par ligne de tuile.
    let tile_priority = tiles.map(|t| t.priority).unwrap_or((0.5, 0.5));
    let tile_sink = tiles.and_then(|t| t.sink);
    let grid = crate::render::tiles::TileGrid::new(width, height, tile_priority);
    let tiles_done = Arc::new(AtomicU32::new(0));
    let total_tiles = grid.order.len().max(1) as u32;
    let it_grid = grid.split(&mut iterations);
    let zs_grid = grid.split(&mut zs);
    let slots: Vec<_> = it_grid
        .into_iter()
        .zip(zs_grid)
        .map(|(it, z)| std::sync::Mutex::new(Some((it, z))))
        .collect();
    let completed = crate::render::tiles::run_prioritized(
        &grid.order,
        &slots,
        cancel.as_ref(),
        &|tile_id, tile_bufs| {
            let (mut it_rows, mut z_rows) = tile_bufs;
            let reuse_row = reuse_data.as_ref();
            let (x0, y0, tw, th) = grid.rect(tile_id);
            for dj in 0..th {
                if cancel.load(Ordering::Relaxed) {
                    cancelled.store(true, Ordering::Relaxed);
                    return;
                }
                let j = y0 + dj;
                for di in 0..tw {
                    let i = x0 + di;
                    let iter = &mut it_rows[dj][di];
                    let z = &mut z_rows[dj][di];
                    // G10.4 : copie inter-frame XaoS (écho produit ou refine union).
                    if let Some(x) = xaos {
                        if let Some(sidx) = x.source_index(i, j) {
                            if let Some(&it) = x.iterations.get(sidx) {
                                *iter = it;
                                *z = x.zs[sidx];
                                continue;
                            }
                        }
                    }
                    if let Some(reuse) = reuse_row {
                        let ratio = reuse.ratio as usize;
                        if j % ratio == 0 && i % ratio == 0 {
                            let src_x = i / ratio;
                            let src_y = j / ratio;
                            let src_idx = src_y * reuse.width as usize + src_x;
                            if src_idx < reuse.iterations.len() {
                                *iter = reuse.iterations[src_idx];
                                *z = reuse.zs[src_idx];
                                continue;
                            }
                        }
                    }

                    // Compute dc in GMP precision (relatif à la référence)
                    let dc_gmp = dc_ctx.compute_dc_ref(i, j);

                    // Iterate pixel with full GMP precision
                    let result = iterate_pixel_gmp(
                        crate::fractal::bytecode::pixel_loop_gmp::GmpPixelRequest {
                            params,
                            ref_orbit: &cache_ref.orbit,
                            dc: &dc_gmp,
                            precision: prec,
                        },
                    );

                    *iter = result.iteration;
                    *z = result.z_final;

                    // Mark glitched or suspect pixels for correction
                    if result.glitched
                        || result.suspect
                        || !result.z_final.re.is_finite()
                        || !result.z_final.im.is_finite()
                    {
                        let idx = j * width + i;
                        glitch_mask[idx].store(true, Ordering::Relaxed);
                    }
                }
            }
            if let Some(sink) = tile_sink {
                sink(crate::render::tiles::TileUpdate {
                    x0,
                    y0,
                    w: tw,
                    h: th,
                    iterations: crate::render::tiles::collect_rows(&it_rows),
                    zs: crate::render::tiles::collect_rows(&z_rows),
                    distances: Vec::new(),
                });
            }
            let done = tiles_done.fetch_add(1, Ordering::Relaxed) + 1;
            progress
                .tile
                .store((done * 100 / total_tiles).min(100), Ordering::Relaxed);
        },
    );
    drop(slots);
    if !completed || cancel.load(Ordering::Relaxed) {
        cancelled.store(true, Ordering::Relaxed);
    }
    let t_pixels = t_pixels_start.elapsed();

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        // Correct glitched pixels using direct GMP iteration (fallback)
        let glitched_indices: Vec<usize> = glitch_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| {
                if flag.load(Ordering::Relaxed) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        if !glitched_indices.is_empty() {
            // Use direct GMP iteration as fallback for glitched pixels
            let mut orbit_params = params.clone();
            orbit_params.engine.precision_bits = prec;
            let gmp_params = MpcParams::from_params(&orbit_params);
            let width_u32 = params.width;

            // Pre-compute shared GMP constants for dc computation. La base
            // `center_*_gmp` est ici le centre de la RÉFÉRENCE (cache) →
            // dc relatif à la référence (offset off-center inclus).
            let dc_ctx = DcGmpContext::with_reference(params, prec, cache);

            let corrections: Vec<_> = glitched_indices
                .par_iter()
                .map(|&idx| {
                    let i = (idx as u32 % width_u32) as usize;
                    let j = (idx as u32 / width_u32) as usize;

                    // Calculate pixel point directly in GMP: cref + dc_ref
                    let dc_gmp = dc_ctx.compute_dc_ref(i, j);
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

    /// G10.2 : la réutilisation OFF-CENTER d'une référence pour une vue CONTENUE
    /// dans son empreinte produit exactement le même rendu qu'une référence
    /// fraîche (même `c = cref + dc` pour chaque pixel), et réutilise bien (pas
    /// de rebuild). Verrou du chemin offset≠0 (les goldens ne testent que offset=0).
    #[test]
    fn subset_reuse_offcenter_matches_fresh_and_reuses() {
        use super::render_perturbation_with_cache;
        // Vue « large » : référence centrée en A=(-0.5, 0), empreinte 3×3.
        // -0.5 est dans la cardioïde → orbite bornée (référence pleine longueur).
        let mut big = default_params_for_type(FractalType::Mandelbrot, 64, 64);
        big.engine.algorithm_mode = AlgorithmMode::Perturbation; // force le path même en shallow
        big.center_x = -0.5;
        big.center_y = 0.0;
        big.center_x_hp = None;
        big.center_y_hp = None;
        big.span_x = 3.0;
        big.span_y = 3.0;
        big.span_x_hp = None;
        big.span_y_hp = None;
        big.engine.precision_bits = 256;
        big.iteration_max = 400;
        let cancel = Arc::new(AtomicBool::new(false));
        let (_r, cache_big) = render_perturbation_with_cache(&big, &cancel, None, None, None, None)
            .expect("render big");

        // Vue CONTENUE : zoom-in ×2 (span 1.5) + pan (0.3, 0.1).
        // x: |0.3|+0.75=1.05 ≤ 1.5 ; y: |0.1|+0.75=0.85 ≤ 1.5 → sous-ensemble.
        let mut view = big.clone();
        view.center_x = -0.2;
        view.center_y = 0.1;
        view.span_x = 1.5;
        view.span_y = 1.5;

        assert!(
            cache_big.can_subset_reuse(&view),
            "la vue devrait être contenue dans l'empreinte de la référence"
        );
        // Verrou 2026-08-23 : un changement de FORMULE au même centre
        // invalide la réutilisation subset (comme is_valid_for).
        let mut hybrid = view.clone();
        hybrid.formula.hybrid_phases = Some(vec![FractalType::Mandelbrot, FractalType::BurningShip]);
        assert!(!cache_big.can_subset_reuse(&hybrid), "hybride ≠ z²+c");
        assert!(!cache_big.is_valid_for(&hybrid));
        let mut opcodes = view.clone();
        opcodes.formula.hybrid_opcodes = Some("sqr rot{30} add".into());
        assert!(!cache_big.can_subset_reuse(&opcodes), "opcodes ≠ z²+c");
        let mut power = view.clone();
        power.formula.multibrot_power = 3.0;
        assert!(!cache_big.can_subset_reuse(&power), "puissance ≠");
        assert!(!cache_big.is_valid_for(&power));

        // Rendu avec réutilisation off-center (offset dc = view.center - big.center).
        let (res_reuse, cache_after) =
            render_perturbation_with_cache(&view, &cancel, None, Some(&cache_big), None, None)
                .expect("reuse");
        // Preuve de RÉUTILISATION : la référence est restée en A (-0.5), pas
        // recalculée au centre de la vue (-0.2).
        assert_eq!(
            cache_after.center_x_gmp, cache_big.center_x_gmp,
            "doit réutiliser la référence off-center, pas rebuild"
        );

        // Rendu FRAIS : référence recalculée au centre de la vue.
        let (res_fresh, _c) =
            render_perturbation_with_cache(&view, &cancel, None, None, None, None).expect("fresh");

        // Correctness : même vue → même c par pixel → même compte d'itération
        // (à ~±1 iter de bord près sur quelques pixels dus au path delta f64).
        let (it_reuse, _z1, _d1) = res_reuse;
        let (it_fresh, _z2, _d2) = res_fresh;
        assert_eq!(it_reuse.len(), it_fresh.len());
        let ndiff = it_reuse
            .iter()
            .zip(&it_fresh)
            .filter(|(a, b)| a != b)
            .count();
        let total = it_reuse.len();
        assert!(
            ndiff * 100 <= total, // ≤ 1 % de pixels de bord tolérés
            "subset-reuse diverge du rendu frais : {ndiff}/{total} px"
        );

        use super::{
            compute_reference_orbit_cached, render_perturbation_gmp_path, DcGmpContext,
            ProgressState, ReferenceOrbitCache, ResolvedSamplingPlan,
        };
        use std::time::Instant;
        // Verrou #24 (2026-08-23) : le path GMP legacy (iterate_pixel_gmp)
        // propage aussi l'offset off-center. Références LEGACY (GMP dense,
        // `use_bytecode_engine = false`) : big en A, réutilisée off-center pour
        // `view`, vs référence fraîche au centre de la vue — même image
        // (avant : frame décalée du pan). Juge : le rendu f64 frais.
        let mut big_legacy = big.clone();
        big_legacy.engine.use_bytecode_engine = false;
        let mut view_legacy = view.clone();
        view_legacy.engine.use_bytecode_engine = false;
        let cache_big_legacy =
            compute_reference_orbit_cached(&big_legacy, Some(&cancel), None, None, false)
                .expect("réf legacy A");
        assert!(
            !cache_big_legacy.orbit.z_ref_gmp.is_empty(),
            "GMP dense requis"
        );
        assert!(cache_big_legacy.can_subset_reuse(&view_legacy));
        let ctx = DcGmpContext::with_reference(&view_legacy, 256, &cache_big_legacy);
        let resolved = ResolvedSamplingPlan::with_reference(&view_legacy, 256, &cache_big_legacy);
        let (ox, oy) = ctx
            .ref_offset
            .clone()
            .expect("offset non nul (pan 0.3, 0.1)");
        let (rx, ry) = resolved.ref_offset.expect("plan résolu off-center");
        assert_eq!((ox.clone(), oy.clone()), (rx, ry));
        assert!((ox.to_f64() - 0.3).abs() < 1e-12 && (oy.to_f64() - 0.1).abs() < 1e-12);
        let n = (view.width * view.height) as usize;
        let gmp_render = |cache: &Arc<ReferenceOrbitCache>| {
            let progress = Arc::new(ProgressState::default());
            let (r, _) = render_perturbation_gmp_path(
                &view_legacy,
                &cancel,
                None,
                None,
                None,
                cache,
                vec![0u32; n],
                vec![Complex64::new(0.0, 0.0); n],
                vec![f64::INFINITY; n],
                progress,
                Instant::now(),
            )
            .expect("gmp path");
            r.0
        };
        let it_gmp_offcenter = gmp_render(&cache_big_legacy);
        let nd = it_gmp_offcenter
            .iter()
            .zip(&it_fresh)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            nd * 100 <= total,
            "path GMP off-center diverge du rendu frais : {nd}/{total} px (offset non propagé ?)"
        );
        // Prouver que l'offset compte : sans lui (ancien comportement) l'image
        // serait celle de la vue centrée en A.
        let mut no_off = ctx.clone();
        no_off.ref_offset = None;
        let d_ref = ctx.compute_dc_ref(3, 5);
        let d_plain = no_off.compute_dc_ref(3, 5);
        assert!((d_ref.real().to_f64() - d_plain.real().to_f64() - 0.3).abs() < 1e-12);
    }

    /// Verrou #25 (2026-08-23) : les corrections GMP suivent le jitter AA
    /// (par pixel prioritaire, sinon uniforme) ; hors AA, bit-identique.
    #[test]
    fn gmp_dc_context_applies_aa_jitter() {
        use super::DcGmpContext;
        let mut p = default_params_for_type(FractalType::Mandelbrot, 16, 16);
        p.center_x_hp = None;
        p.center_y_hp = None;
        p.span_x_hp = None;
        p.span_y_hp = None;
        let base = DcGmpContext::new(&p, 128);
        let d0 = base.compute_dc(4, 7);
        // Uniforme legacy : +0.25 px en x → +0.25·span/width.
        let mut pu = p.clone();
        pu.sampling.aa_subpixel_offset = [0.25, 0.0];
        let du = DcGmpContext::new(&pu, 128).compute_dc(4, 7);
        let expect = 0.25 * p.span_x / 16.0;
        assert!((du.real().to_f64() - d0.real().to_f64() - expect).abs() < 1e-15);
        assert_eq!(du.imag().to_f64(), d0.imag().to_f64());
        // Par pixel (prioritaire sur l'uniforme) : = pixel_offset en c-space.
        let mut pj = pu.clone();
        pj.sampling.aa_jitter = Some((2, 1.0));
        let dj = DcGmpContext::new(&pj, 128).compute_dc(4, 7);
        let (jx, jy) = crate::fractal::jitter::pixel_offset(16, 4, 7, 2, 1.0);
        assert!((dj.real().to_f64() - d0.real().to_f64() - jx * p.span_x / 16.0).abs() < 1e-15);
        assert!((dj.imag().to_f64() - d0.imag().to_f64() - jy * p.span_y / 16.0).abs() < 1e-15);
        // Deux pixels voisins → jitters décorrélés.
        let dj2 = DcGmpContext::new(&pj, 128).compute_dc(5, 7);
        let dx_plain = d0.real().to_f64() + p.span_x / 16.0;
        assert!((dj2.real().to_f64() - dx_plain) != (dj.real().to_f64() - d0.real().to_f64()));
    }

    /// Régression (bug « zones d'erreur » 2026-07-16, centre d'image 1
    /// -1.3719…, -0.0860…) : un zoom CONTINU qui thread le cache d'orbite de
    /// frame en frame (comme la GUI post-G10.2, qui ne jette plus `orbit_cache`
    /// au zoom) doit produire la frame profonde finale PIXEL-EXACTE vs un rendu
    /// FRAIS (`cache=None`). Avant le fix, une référence bâtie hors régime
    /// atom-domain (span large) était réutilisée jusqu'à 4e25 (précision
    /// suffisante ⇒ jamais invalidée) : sa troncature ne correspondait pas à
    /// celle d'un build frais → ~1.7 % de pixels faux (bruit sel-et-poivre).
    /// Le fix (`atom_regime_scale_mismatch`) rebuild quand l'échelle change dans
    /// le régime atom-domain. Verrou : divergence EXACTEMENT nulle.
    #[test]
    fn continuous_zoom_cache_reuse_matches_fresh() {
        use super::{render_perturbation_with_cache, ReferenceOrbitCache};
        let cx =
            "-1.371894034497786177276218629447827355152810566065208011938331009089941256128969";
        let cy =
            "-8.596946447921205325727880816022532777086452667518370092652771867971556457154214e-2";
        let final_span = 9.462012871361855867690684674541606291146700322899874668115416663170852644098815e-26_f64;
        let (w, h) = (128u32, 96u32);
        let cancel = Arc::new(AtomicBool::new(false));

        let make = |span_x: f64| -> FractalParams {
            let mut p = default_params_for_type(FractalType::Mandelbrot, w, h);
            p.engine.algorithm_mode = AlgorithmMode::Perturbation;
            p.iteration_max = 2500;
            p.center_x = -1.3718940344977861;
            p.center_y = -0.08596946447921205;
            p.center_x_hp = Some(cx.to_string());
            p.center_y_hp = Some(cy.to_string());
            let span_y = span_x * h as f64 / w as f64;
            p.span_x = span_x;
            p.span_y = span_y;
            p.span_x_hp = Some(format!("{span_x:.17e}"));
            p.span_y_hp = Some(format!("{span_y:.17e}"));
            p
        };

        // Zoom continu ×2 depuis span 1.0 (régime peu profond, réf non-atom)
        // jusqu'au span final (régime atom-domain profond), cache threadé.
        let mut cache: Option<Arc<ReferenceOrbitCache>> = None;
        let mut span = 1.0_f64;
        while span > final_span {
            let p = make(span);
            let (_r, c) =
                render_perturbation_with_cache(&p, &cancel, None, cache.as_ref(), None, None)
                    .expect("zoom step");
            cache = Some(c);
            span /= 2.0;
        }
        let p_final = make(final_span);
        let (res_reuse, _c1) =
            render_perturbation_with_cache(&p_final, &cancel, None, cache.as_ref(), None, None)
                .expect("final reuse");
        let (res_fresh, _c2) =
            render_perturbation_with_cache(&p_final, &cancel, None, None, None, None)
                .expect("final fresh");

        let (it_reuse, _z1, _d1) = res_reuse;
        let (it_fresh, _z2, _d2) = res_fresh;
        let ndiff = it_reuse
            .iter()
            .zip(&it_fresh)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            ndiff,
            0,
            "zoom continu (cache réutilisé) diverge du rendu frais : {ndiff}/{} px",
            it_reuse.len()
        );
    }

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
                label,
                actual_log2,
                expected_log2
            );
        }
    }

    /// VERROU précision ultra-deep (2026-07-12) : le plafond 65 536 b rendait
    /// e22522 (74 855 b requis) et e52465 (174 350 b) en image UNIFORME fausse
    /// (réf sous-précise). F3 n'a AUCUN plafond (`param.cc:132`) ; le nôtre est
    /// SUPPRIMÉ (u32::MAX = borne du type rug, décision utilisateur 2026-07-12) —
    /// le cas 1e300000 verrouille l'absence de plafond de design. Régression =
    /// ce test casse.
    #[test]
    fn precision_bits_covers_ultra_deep_corpus() {
        use super::compute_perturbation_precision_bits;
        for (label, span_str, min_bits) in [
            ("e22522", "1.38e-22522", 74_000u32),
            ("e52465", "3.88e-52465", 174_000u32),
            ("no-design-cap-1e300000", "1e-300000", 996_000u32),
        ] {
            let mut p = default_params_for_type(FractalType::Mandelbrot, 256, 256);
            p.span_x = 0.0;
            p.span_y = 0.0;
            p.span_x_hp = Some(span_str.to_string());
            p.span_y_hp = Some(span_str.to_string());
            let bits = compute_perturbation_precision_bits(&p);
            assert!(
                bits >= min_bits,
                "{label}: {bits} bits < {min_bits} requis — plafond MAX_PERTURB_PRECISION_BITS \
                 trop bas (réf sous-précise → image uniforme fausse)"
            );
        }
    }

    fn base_params(fractal_type: FractalType) -> FractalParams {
        // center=(0,0), span=(4,3) -> xmin=-2, xmax=2, ymin=-1.5, ymax=1.5
        let mut p = default_params_for_type(fractal_type, 5, 5);
        p.span_x = 4.0;
        p.span_y = 3.0;
        p.iteration_max = 64;
        p.engine.precision_bits = 192;
        p.engine.algorithm_mode = AlgorithmMode::Perturbation;
        p.perturbation.bla_threshold = 1e-6;
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

    // VERROU (2026-07-13) : la perturbation des types non-conformes Celtic /
    // Buffalo / PerpendicularBurningShip doit égaler l'itération f64 directe
    // (`iterate_point`), comme Mandelbrot/Julia/BurningShip ci-dessus. Ces types
    // partagent le pixel-loop bytecode unifié (opcodes AbsX/AbsY/NegY après/avant
    // Sqr) mais n'avaient AUCUN verrou perturbation↔f64. Diagnostic 2026-07-13 :
    // à un zoom modéré (span 4) f64 est ample, donc pert == f64 exactement ; la
    // divergence vs GMP observée à l'antenne -1.75 (zoom 1e6) est une
    // sensibilité de PRÉCISION (GMP-128 lui-même non convergé, il faut 256+ b —
    // même classe que e13/dd-sensibilité), PAS un bug de perturbation. Ces
    // verrous protègent la correction de la boucle pixel pour ces 3 types.
    // Ces 3 types passent par le pixel-loop BYTECODE perturbation (via le
    // dispatcher unique), pas par le `render_perturbation_cancellable_with_reuse`
    // legacy (réservé Mandelbrot/Julia/BurningShip/Tricorn — renvoie None sinon).
    // On vérifie donc le chemin de PRODUCTION (dispatcher, mode Perturbation forcé)
    // vs l'itération f64 directe `iterate_point`.
    fn assert_pert_dispatch_matches_f64(base: &FractalParams, indices: &[(u32, u32)], tol: i32) {
        use crate::render::escape_time::render_escape_time;
        let mut params = base.clone();
        params.engine.algorithm_mode = AlgorithmMode::Perturbation;
        let iters = render_escape_time(&params).iterations;
        for &(x, y) in indices {
            let idx = (y * params.width + x) as usize;
            let x_ratio = (x as f64 + 0.5) / params.width as f64;
            let y_ratio = (y as f64 + 0.5) / params.height as f64;
            let xg = params.center_x + (x_ratio - 0.5) * params.span_x;
            let yg = params.center_y + (y_ratio - 0.5) * params.span_y;
            let z_pixel = Complex64::new(xg, yg);
            let ref_iter = iterate_point(&params, z_pixel).iteration;
            let got = iters[idx];
            let diff = (got as i32 - ref_iter as i32).abs();
            assert!(
                diff <= tol,
                "{:?} pixel ({x},{y}): pert {got}, f64 {ref_iter}, diff {diff} > {tol}",
                params.fractal_type
            );
        }
    }

    fn celtic_family_params(fractal_type: FractalType) -> FractalParams {
        let mut p = base_params(fractal_type);
        p.center_x = -0.5;
        p.center_y = 0.0;
        p.span_x = 4.0;
        p.span_y = 4.0;
        p
    }

    #[test]
    fn perturbation_matches_f64_celtic() {
        assert_pert_dispatch_matches_f64(
            &celtic_family_params(FractalType::Celtic),
            &[(0, 4), (2, 2), (4, 0)],
            1,
        );
    }

    #[test]
    fn perturbation_matches_f64_buffalo() {
        assert_pert_dispatch_matches_f64(
            &celtic_family_params(FractalType::Buffalo),
            &[(0, 4), (2, 2), (4, 0)],
            1,
        );
    }

    #[test]
    fn perturbation_matches_f64_perpendicular_burning_ship() {
        assert_pert_dispatch_matches_f64(
            &celtic_family_params(FractalType::PerpendicularBurningShip),
            &[(0, 4), (2, 2), (4, 0)],
            1,
        );
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

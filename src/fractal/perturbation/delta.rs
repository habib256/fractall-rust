use num_complex::Complex64;
use rug::{Complex, Float};
use std::sync::OnceLock;

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::perturbation::bla::BlaTable;
use crate::fractal::perturbation::orbit::{ReferenceOrbit, HybridBlaReferences};
use crate::fractal::perturbation::types::ComplexExp;
use crate::fractal::gmp::complex_norm_sqr;
use crate::fractal::perturbation::series::{
    SeriesConfig, SeriesTable, should_use_series, estimate_series_error,
    compute_series_skip,
};
use crate::fractal::perturbation::distance::{DualComplex, compute_distance_estimate, transform_pixel_to_complex};
use crate::fractal::perturbation::interior::{ExtendedDualComplex, is_interior};
use crate::fractal::perturbation::nonconformal::Matrix2x2;

/// Applies a 2×2 real matrix to all components (value + duals) of an ExtendedDualComplex.
/// Used for non-conformal fractal formulas (Tricorn) where the Jacobian is a real 2×2 matrix.
fn apply_nonconformal_matrix(m: Matrix2x2, dual: ExtendedDualComplex) -> ExtendedDualComplex {
    let (v_re, v_im) = m.mul_vector(dual.value.re, dual.value.im);
    let (dre_re, dre_im) = m.mul_vector(dual.dual_re.re, dual.dual_re.im);
    let (dim_re, dim_im) = m.mul_vector(dual.dual_im.re, dual.dual_im.im);
    let (dz1re_re, dz1re_im) = m.mul_vector(dual.dual_z1_re.re, dual.dual_z1_re.im);
    let (dz1im_re, dz1im_im) = m.mul_vector(dual.dual_z1_im.re, dual.dual_z1_im.im);
    ExtendedDualComplex {
        value: Complex64::new(v_re, v_im),
        dual_re: Complex64::new(dre_re, dre_im),
        dual_im: Complex64::new(dim_re, dim_im),
        dual_z1_re: Complex64::new(dz1re_re, dz1re_im),
        dual_z1_im: Complex64::new(dz1im_re, dz1im_im),
    }
}

fn rebase_stride() -> u32 {
    static STRIDE: OnceLock<u32> = OnceLock::new();
    *STRIDE.get_or_init(|| {
        let raw = std::env::var("FRACTALL_PERTURB_REBASE_STRIDE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(1);
        raw.clamp(1, 64)
    })
}

pub struct DeltaResult {
    pub iteration: u32,
    pub z_final: Complex64,
    pub glitched: bool,
    pub suspect: bool,
    /// Distance estimation (if computed). f64::INFINITY if not computed or invalid.
    /// Can be used for distance-based coloring or 3D rendering.
    pub distance: f64,
    /// Whether the point is in the interior of the set. Only valid if interior detection was enabled.
    /// Can be used for interior coloring (e.g., black for interior points).
    pub is_interior: bool,
    /// Whether the phase changed during rebasing (for Hybrid BLA).
    /// If true, the caller should switch to the new phase reference.
    pub phase_changed: bool,
}

/// Calcule diffabs(c, d) = |c + d| - |c|
///
/// Cette fonction calcule la différence |c + d| - |c| de manière stable.
/// Utilisée pour Burning Ship où on a besoin de la variation de |x|.
///
/// # Arguments
/// * `c` - Valeur de référence (haute précision)
/// * `d` - Delta (basse précision)
///
/// # Returns
/// La différence |c + d| - |c|
/// 
/// # Formule
/// ```
/// diffabs(c, d) = {
///   d        si c >= 0 et c + d >= 0
///   -(2c + d) si c >= 0 et c + d < 0
///   2c + d    si c < 0 et c + d > 0
///   -d       si c < 0 et c + d <= 0
/// }
/// ```
pub fn diffabs(c: f64, d: f64) -> f64 {
    let cd = c + d;
    let c2d = 2.0 * c + d;
    if c >= 0.0 {
        if cd >= 0.0 {
            d
        } else {
            -c2d
        }
    } else {
        if cd > 0.0 {
            c2d
        } else {
            -d
        }
    }
}

/// Calcule la tolérance de glitch adaptative basée sur le niveau de zoom.
///
/// Plus le zoom est profond, plus la tolérance peut être relaxée car les erreurs
/// numériques sont plus importantes mais moins visibles à grande échelle.
///
/// # Arguments
/// * `pixel_size` - Taille d'un pixel dans l'espace complexe
/// * `user_tolerance` - Tolérance définie par l'utilisateur (1e-4 par défaut)
///
/// # Returns
/// La tolérance adaptative à utiliser pour la détection des glitches.
pub fn compute_adaptive_glitch_tolerance(pixel_size: f64, user_tolerance: f64) -> f64 {
    // Si l'utilisateur a défini une tolérance personnalisée (différente de 1e-4),
    // respecter son choix
    const DEFAULT_TOLERANCE: f64 = 1e-4;
    if (user_tolerance - DEFAULT_TOLERANCE).abs() > 1e-10 {
        return user_tolerance;
    }

    // Calculer le niveau de zoom: log10(4 / pixel_size)
    // À pixel_size = 4.0 (vue complète), zoom_level ≈ 0
    // À pixel_size = 4e-14, zoom_level ≈ 14
    let zoom_level = if pixel_size > 0.0 && pixel_size.is_finite() {
        (4.0 / pixel_size).log10().max(0.0)
    } else {
        0.0
    };

    // Tolérance adaptative selon le niveau de zoom
    // Ajustée pour réduire les faux positifs et améliorer les performances
    // La perturbation devrait être rapide, donc on évite de marquer trop de pixels comme glitched
    match zoom_level as u32 {
        0..=6 => 1e-5,    // Zoom peu profond : modéré (relaxé de 1e-6)
        7..=14 => 1e-4,   // Moyen : standard (relaxé de 1e-5)
        15..=30 => 1e-3,  // Profond : relaxé (relaxé de 1e-4 pour éviter trop de recalculs)
        31..=50 => 1e-2,  // Très profond : très relaxé (relaxé de 1e-3)
        _ => 1e-1,        // Extrême (>50) : très tolérant (relaxé de 1e-2)
    }
}

/// Iterate a pixel using perturbation theory (Section 2 of deep zoom theory).
///
/// # Section 2: Perturbation
///
/// Low precision deltas relative to high precision orbit.
///
/// ## Formules mathématiques
///
/// - **Pixel orbit**: `Z_m + z_n` (où `Z_m` est l'orbite de référence haute précision, `z_n` le delta basse précision)
/// - **Point C du pixel**: `C + c` (où `C` est le centre, `c` l'offset du pixel)
/// - **Formule de perturbation**: `z_{n+1} = 2·Z_m·z_n + z_n² + c`
///
/// ## Initialisation
///
/// - `m` et `n` commencent à 0 (`m = 0`, `n = 0`)
/// - `z_0 = 0` (delta initial = 0 pour Mandelbrot)
///
/// **Note**: Dans le code, `m` et `n` sont représentés par une seule variable `n` qui est toujours
/// synchronisée (`m = n`). Pour Julia, l'initialisation diffère: `z_0 = c` (delta initial = offset du pixel).
///
/// ## Rebasing
///
/// Rebasing to avoid glitches: when `|Z_m + z_n| < |z_n|`, replace `z_n` with `Z_m + z_n`
/// and reset the reference iteration count `m` to 0.
///
/// ## Notation dans le code
///
/// **Important**: Dans le code, il n'y a qu'une seule variable `n` qui représente à la fois
/// `m` (l'index de l'orbite de référence) et `n` (l'itération du delta). Ils sont toujours
/// synchronisés (`m = n` dans notre implémentation).
///
/// - `Z_m` ↔ `ref_orbit.z_ref_f64[n]` (orbite de référence haute précision à l'itération `n`)
/// - `z_n` ↔ `delta` (delta de perturbation basse précision, type `ComplexExp`)
/// - `C` ↔ `ref_orbit.cref` (point de référence, centre de l'image)
/// - `c` ↔ `dc` (offset du pixel par rapport au centre, type `ComplexExp`)
/// - `m` et `n` ↔ `n` (une seule variable dans le code, toujours synchronisée)
///
/// # Arguments
///
/// * `params` - Paramètres de la fractale
/// * `ref_orbit` - Orbite de référence haute précision calculée au centre (ou référence de phase pour Hybrid BLA)
/// * `bla_table` - Table BLA (Bivariate Linear Approximation) pour sauter des itérations.
///   Approxime `l` itérations par `z_{n+l} = A_{n,l}·z_n + B_{n,l}·c` quand valide.
///   For Hybrid BLA: one BLA table per reference (one per phase).
/// * `series_table` - Table de séries pour approximation (optionnelle)
/// * `delta0` - Delta initial (`z_0`): `ComplexExp::zero()` pour Mandelbrot, `dc` pour Julia
/// * `dc` - Offset du pixel par rapport au centre (`c` dans la formule)
///
/// # Hybrid BLA
///
/// For a hybrid loop with multiple phases, this function is called with the reference
/// for the current phase. Rebasing switches to the reference for the current phase.
///
/// # Parameters
///
/// * `current_phase` - Current phase (for Hybrid BLA). Updated when rebasing occurs.
/// * `hybrid_refs` - Hybrid BLA references (for Hybrid BLA). Used to calculate new phase on rebasing.
pub fn iterate_pixel(
    params: &FractalParams,
    ref_orbit: &ReferenceOrbit,
    bla_table: &BlaTable,
    series_table: Option<&SeriesTable>,
    delta0: ComplexExp,
    dc: ComplexExp,
    mut current_phase: Option<&mut u32>,
    hybrid_refs: Option<&HybridBlaReferences>,
) -> DeltaResult {
    let rebase_stride = rebase_stride();
    // If distance estimation or interior detection is enabled, use dual numbers version
    // Note: pixel coordinates need to be passed from caller - for now, estimate from dc
    // This is a limitation: ideally iterate_pixel() should accept pixel coordinates
    if params.enable_distance_estimation || params.enable_interior_detection {
        // Estimate pixel coordinates from dc
        // dc_re = (i/width - 0.5) * span_x, so i ≈ (dc_re / span_x + 0.5) * width
        // dc_im = (j/height - 0.5) * span_y, so j ≈ (dc_im / span_y + 0.5) * height
        let dc_approx = dc.to_complex64_approx();
        let pixel_x = if params.span_x != 0.0 && params.span_x.is_finite() {
            ((dc_approx.re / params.span_x) + 0.5) * params.width as f64
        } else {
            params.width as f64 * 0.5
        };
        let pixel_y = if params.span_y != 0.0 && params.span_y.is_finite() {
            ((dc_approx.im / params.span_y) + 0.5) * params.height as f64
        } else {
            params.height as f64 * 0.5
        };
        
        return iterate_pixel_with_duals(
            params,
            ref_orbit,
            bla_table,
            series_table,
            delta0,
            dc,
            pixel_x.max(0.0).min(params.width as f64),
            pixel_y.max(0.0).min(params.height as f64),
            params.enable_distance_estimation,
            params.enable_interior_detection,
            current_phase,
            hybrid_refs,
        );
    }
    
    // Standard path without dual numbers
    // Compteurs alignés C++ Fraktaler-3: iters_ptb (itérations perturbation), steps_bla (pas BLA).
    // Limites séparées max_perturb_iterations / max_bla_steps (0 = illimité).
    let mut n = 0u32;
    let mut iters_ptb = 1u32;  // C++: iters_ptb = 1 au départ
    let mut steps_bla = 0u32;  // C++: steps_bla = 0 au départ
    let mut delta = delta0;
    // Use z_ref_f64 for fast path iteration (z_ref is high-precision Vec<ComplexExp>)
    // Hybrid BLA: account for phase offset in effective length
    let effective_len = ref_orbit.effective_len() as u32;
    let max_iter = params.iteration_max.min(effective_len.saturating_sub(1));
    let bailout_sqr = params.bailout * params.bailout;
    let mut phase_changed = false;

    // Calculer le pixel_size pour la tolérance adaptative
    let pixel_size = params.span_x / params.width as f64;
    
    // DÉSACTIVÉ: Optimisation pour le centre exact qui causait des artefacts circulaires visibles.
    // La perturbation standard fonctionne correctement même au centre exact.
    // Variables conservées pour référence future mais non utilisées:
    // let dc_norm_sqr = dc.norm_sqr_approx();
    // let delta_norm_sqr_initial = delta.norm_sqr_approx();
    // let pixel_size_sqr = pixel_size * pixel_size;
    // let center_threshold = pixel_size_sqr * 0.01;
    // let delta_threshold = pixel_size_sqr * 1e-8;
    // let is_center_like = dc_norm_sqr < center_threshold && delta_norm_sqr_initial < delta_threshold;
    let adaptive_tolerance = compute_adaptive_glitch_tolerance(pixel_size, params.glitch_tolerance);
    let glitch_tolerance_sqr = adaptive_tolerance * adaptive_tolerance;
    let is_julia = params.fractal_type == FractalType::Julia;
    let is_burning_ship = params.fractal_type == FractalType::BurningShip;
    let is_multibrot = params.fractal_type == FractalType::Multibrot;
    let is_tricorn = params.fractal_type == FractalType::Tricorn;
    let multibrot_power = params.multibrot_power;
    let series_config = SeriesConfig::from_params(params);
    let mut suspect = false;

    // Use high-precision orbit when pixel size is very small (deep zoom)
    // This helps avoid precision loss when z_ref values are at extreme ranges
    let use_high_precision = pixel_size < 1e-14;

    // Try standalone series skip BEFORE BLA (if enabled and series table is available).
    // For Mandelbrot: the series variable is dc (pixel offset), since delta_0 = 0.
    // For Julia: the series variable is also dc (= delta_0), since delta_0 = dc.
    // In both cases, dc is the "small parameter" that the series is expanded in.
    if let Some(table) = series_table {
        if params.series_standalone && !is_burning_ship && !is_multibrot && !is_tricorn {
            if let Some(skip_result) = compute_series_skip(
                table,
                dc,
                series_config.error_tolerance,
            ) {
                // Skip to the computed iteration
                n = skip_result.skip_to as u32;
                delta = skip_result.delta;
                if skip_result.estimated_error > series_config.error_tolerance * 0.5 {
                    suspect = true;
                }
            }
        }
    }

    // Main iteration loop: BLA Table Lookup algorithm
    // Based on fraktaler-3 implementation with nested BLA loop for consecutive steps
    // For each iteration:
    // 1. Nested BLA loop: apply consecutive BLA steps until no more valid BLA is found
    //    - Check rebasing at each BLA iteration
    // 2. If no BLA was applied, do a perturbation iteration
    // 3. Check for rebasing opportunities after perturbation step
    // Limites séparées (C++: iters_ptb < PerturbIterations && steps_bla < BLASteps)
    let limit_ptb = params.max_perturb_iterations;
    let limit_bla = params.max_bla_steps;
    while n < max_iter
        && (limit_ptb == 0 || iters_ptb < limit_ptb)
        && (limit_bla == 0 || steps_bla < limit_bla)
    {
        // Nested BLA loop: apply consecutive BLA steps (like C++ do...while(b))
        // This allows multiple BLA steps to be applied in sequence, improving performance
        let mut bla_applied = false;
        // Cache de la norme carrée pour éviter les recalculs (optimisation 2)
        let mut delta_norm_sqr_cached = delta.norm_sqr_approx();
        // Cache de la conversion ComplexExp → Complex64 (optimisation 1)
        let mut delta_approx_cached = delta.to_complex64_approx();
        loop {
            // C++: break if limits exceeded (lines 293-294)
            if limit_ptb != 0 && iters_ptb >= limit_ptb {
                break;
            }
            if limit_bla != 0 && steps_bla >= limit_bla {
                break;
            }
            // Optimisation 4: Vérifier rebasing seulement si nécessaire (fin d'orbite effective)
            // Les autres vérifications de rebasing sont faites après application BLA pour améliorer les performances
            if n >= effective_len {
                // Reached end of effective orbit: rebase
                let last_idx = effective_len.saturating_sub(1) as usize;
                let z_ref = ref_orbit.get_z_ref_f64(last_idx as u32).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                // Utiliser le cache de delta_approx (optimisation 1)
                let z_curr = z_ref + delta_approx_cached;
                delta = ComplexExp::from_complex64(z_curr);
                // Mettre à jour les caches après rebasing
                delta_approx_cached = delta.to_complex64_approx();
                delta_norm_sqr_cached = delta.norm_sqr_approx();
                
                // Hybrid BLA: change phase on rebasing: phase = (phase + n) % cycle_period
                if let Some(ref mut phase) = current_phase {
                    if let Some(refs) = hybrid_refs {
                        if refs.cycle_period > 0 {
                            **phase = (**phase + n) % refs.cycle_period;
                            phase_changed = true;
                        }
                    }
                }
                n = 0;
            }
            
            // Try to find and apply a BLA step
            let mut stepped = false;

            // Try non-conformal BLA first for Tricorn
            if is_tricorn {
                if let Some(ref nonconformal_levels) = bla_table.nonconformal_levels {
                    if !nonconformal_levels.is_empty() {
                        // Utiliser le cache de delta_approx (optimisation 1)
                        let delta_vec = (delta_approx_cached.re, delta_approx_cached.im);
                        let delta_norm_sqr_check = delta_vec.0 * delta_vec.0 + delta_vec.1 * delta_vec.1;
                        
                        for level in (0..nonconformal_levels.len()).rev() {
                            let level_nodes = &nonconformal_levels[level];
                            if (n as usize) >= level_nodes.len() {
                                continue;
                            }
                            let node = &level_nodes[n as usize];
                            
                            if delta_norm_sqr_check < node.validity_radius * node.validity_radius {
                                // Apply non-conformal BLA: z_{n+l} = A_{n,l}·z_n + B_{n,l}·c
                                // For non-conformal fractals (Tricorn), A and B are 2×2 real matrices
                                // instead of complex numbers, as angles are not preserved.
                                let dc_approx = dc.to_complex64_approx();
                                let dc_vec = (dc_approx.re, dc_approx.im);
                                
                                // Linear term: A_{n,l}·z_n
                                let linear_vec = node.a.mul_vector(delta_vec.0, delta_vec.1);
                                
                                // Add dc term: B_{n,l}·c
                                let dc_term_vec = node.b.mul_vector(dc_vec.0, dc_vec.1);
                                
                                let next_vec = (linear_vec.0 + dc_term_vec.0, linear_vec.1 + dc_term_vec.1);
                                
                                // Convert back to ComplexExp
                                delta = ComplexExp::from_complex64(Complex64::new(next_vec.0, next_vec.1));
                                n += 1u32 << level;  // Skip l = 2^level iterations
                                stepped = true;
                                bla_applied = true;
                                steps_bla += 1;  // C++: steps_bla++ après chaque pas BLA
                                // Mettre à jour les caches après application BLA
                                delta_approx_cached = delta.to_complex64_approx();
                                delta_norm_sqr_cached = delta.norm_sqr_approx();
                                
                                // Optimisation 4: Vérifier rebasing après application BLA
                                if n < effective_len && (rebase_stride == 1 || (n % rebase_stride) == 0) {
                                    let z_ref_check = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                                        ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                                    });
                                    let z_curr_check = z_ref_check + delta_approx_cached;
                                    let z_curr_norm_sqr_check = z_curr_check.norm_sqr();
                                    
                                    if z_curr_norm_sqr_check > 0.0 && delta_norm_sqr_cached > 0.0 && z_curr_norm_sqr_check < delta_norm_sqr_cached {
                                        // Rebasing: replace z_n with Z_m + z_n and reset m to 0
                                        delta = ComplexExp::from_complex64(z_curr_check);
                                        // Mettre à jour les caches après rebasing
                                        delta_approx_cached = delta.to_complex64_approx();
                                        delta_norm_sqr_cached = delta.norm_sqr_approx();
                                        
                                        // Hybrid BLA: change phase on rebasing: phase = (phase + n) % cycle_period
                                        if let Some(ref mut phase) = current_phase {
                                            if let Some(refs) = hybrid_refs {
                                                if refs.cycle_period > 0 {
                                                    **phase = (**phase + n) % refs.cycle_period;
                                                    phase_changed = true;
                                                }
                                            }
                                        }
                                        n = 0;
                                    }
                                }
                                break; // Found a BLA step, continue nested loop to try another
                            }
                        }
                    }
                }
            }

            // BLA Table Lookup:
            // Find the BLA starting from iteration m (n in our code) that has the largest skip l
            // satisfying |z| < R. If there is none, do a perturbation iteration.
            // Check for rebasing opportunities after each BLA application or perturbation step.
            //
            // Bivariate Linear Approximation (BLA):
            // Sometimes, l iterations starting at n can be approximated by bivariate linear function:
            // z_{n+l} = A_{n,l}·z_n + B_{n,l}·c
            // This is valid when the non-linear part of the full perturbation iterations is so small
            // that omitting it would cause fewer problems than the rounding error of the low precision data type.
            //
            // Note: BLA is automatically disabled for very deep zooms (>10^15) where f64 precision
            // is insufficient. In such cases, the BLA table will be empty and full GMP perturbation is used.
            //
            // Search strategy: iterate levels in reverse order (largest skip first) to find the largest
            // valid skip l = 2^level satisfying |z_n| < R_{n,l}.
            if !stepped && !bla_table.levels.is_empty() {
                // Utiliser le cache de la norme au lieu de recalculer (optimisation 2)
                let delta_norm_sqr = delta_norm_sqr_cached;
                // Search from highest level (largest skip) to lowest level (smallest skip)
                for level in (0..bla_table.levels.len()).rev() {
                    let level_nodes = &bla_table.levels[level];
                    if (n as usize) >= level_nodes.len() {
                        continue;
                    }
                    let node = &level_nodes[n as usize];
                    // Pour Burning Ship, vérifier que le BLA est valide (quadrant stable)
                    if is_burning_ship && !node.burning_ship_valid {
                        continue;
                    }
                    // Single Step BLA validity condition: |z_n| < R_{n,l}
                    // Derived from: |z_n²| << |2·Z_n·z_n + c|
                    // Assuming negligibility of c: |z_n| << |2·Z_n| = |A_{n,1}|
                    // Therefore: |z_n| < R_{n,l} where R_{n,l} = ε·|A_{n,l}|
                    if delta_norm_sqr < node.validity_radius * node.validity_radius {
                        // For Burning Ship, apply sign transformation to delta
                        let work_delta = if is_burning_ship && node.burning_ship_valid {
                            delta.mul_signed(node.sign_re, node.sign_im)
                        } else {
                            delta
                        };

                        if should_use_series(series_config, delta_norm_sqr, node.validity_radius) {
                            // Use series approximation with higher-order terms
                            let delta_sq = work_delta.mul(work_delta);
                            let mut next_delta = work_delta.mul_complex64(node.a);  // A_{n,l}·z_n
                            if !is_julia {
                                next_delta = next_delta.add(dc.mul_complex64(node.b));  // + B_{n,l}·c
                            }
                            // Terme quadratique (ordre 2): C·z_n²
                            next_delta = next_delta.add(delta_sq.mul_complex64(node.c));

                            // Termes d'ordre supérieur pour Multibrot (ordre 4-6)
                            if series_config.order >= 4 && is_multibrot {
                                // Terme cubique δ³
                                let delta_cube = delta_sq.mul(work_delta);
                                if node.d.norm() > 1e-20 {
                                    next_delta = next_delta.add(delta_cube.mul_complex64(node.d));
                                }

                                // Terme quartique δ⁴ (ordre 5-6)
                                if series_config.order >= 5 && node.e.norm() > 1e-20 {
                                    let delta_4 = delta_sq.mul(delta_sq);
                                    next_delta = next_delta.add(delta_4.mul_complex64(node.e));
                                }
                            }

                            let coeff_a_norm = node.a.norm();
                            let err = estimate_series_error(
                                delta_norm_sqr,
                                series_config.order,
                                level,
                                coeff_a_norm,
                            );
                            if err > series_config.error_tolerance {
                                suspect = true;
                            }
                            delta = next_delta;
                        } else {
                            // Apply BLA: z_{n+l} = A_{n,l}·z_n + B_{n,l}·c
                            let mut next_delta = work_delta.mul_complex64(node.a);  // A_{n,l}·z_n
                            if !is_julia {
                                next_delta = next_delta.add(dc.mul_complex64(node.b));  // + B_{n,l}·c
                            }
                            delta = next_delta;
                        }
                        // Skip l = 2^level iterations: z_{n+l} has been computed
                        n += 1u32 << level;
                        stepped = true;
                        bla_applied = true;
                        steps_bla += 1;  // C++: steps_bla++ après chaque pas BLA
                        // Mettre à jour les caches après application BLA
                        delta_approx_cached = delta.to_complex64_approx();
                        delta_norm_sqr_cached = delta.norm_sqr_approx();
                        
                        // Optimisation 4: Vérifier rebasing après application BLA (plus efficace que à chaque itération)
                        if n < effective_len && (rebase_stride == 1 || (n % rebase_stride) == 0) {
                            let z_ref_check = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                                ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                            });
                            let z_curr_check = z_ref_check + delta_approx_cached;
                            let z_curr_norm_sqr_check = z_curr_check.norm_sqr();
                            
                            if z_curr_norm_sqr_check > 0.0 && delta_norm_sqr_cached > 0.0 && z_curr_norm_sqr_check < delta_norm_sqr_cached {
                                // Rebasing: replace z_n with Z_m + z_n and reset m to 0
                                delta = ComplexExp::from_complex64(z_curr_check);
                                // Mettre à jour les caches après rebasing
                                delta_approx_cached = delta.to_complex64_approx();
                                delta_norm_sqr_cached = delta.norm_sqr_approx();
                                
                                // Hybrid BLA: change phase on rebasing: phase = (phase + n) % cycle_period
                                if let Some(ref mut phase) = current_phase {
                                    if let Some(refs) = hybrid_refs {
                                        if refs.cycle_period > 0 {
                                            **phase = (**phase + n) % refs.cycle_period;
                                            phase_changed = true;
                                        }
                                    }
                                }
                                n = 0;
                            }
                        }
                        break; // Found a BLA step, continue nested loop to try another
                    }
                }
            }
            
            // If no BLA step was found, exit the nested loop
            if !stepped {
                break;
            }
        }

        // If no BLA was applied, do a perturbation iteration
        if !bla_applied {
            // Cache de z_ref pour éviter les accès répétés (optimisation 3)
            let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
            });
            
            // DÉSACTIVÉ: Optimisation pour le centre exact qui causait des artefacts circulaires visibles.
            // Même avec des seuils stricts, cette optimisation créait un cercle au centre.
            // La perturbation standard fonctionne correctement même au centre exact.
            // if is_center_like && !is_burning_ship && !is_multibrot && !is_tricorn {
            //     // Au centre exact, delta reste ≈ 0, donc on peut le forcer à 0 pour éviter les erreurs d'arrondi
            //     // qui pourraient s'accumuler. Cela garantit que le pixel suit exactement l'orbite de référence.
            //     delta = ComplexExp::zero();
            //     n += 1;
            //     // Skip to rebasing check after perturbation
            // } else 
            if is_burning_ship {
                // Utiliser le cache de delta_approx (optimisation 1)
                let delta_approx = delta_approx_cached;
                let z_curr = z_ref + delta_approx;

                // Check if quadrant is stable (same signs)
                let ref_sign_re = if z_ref.re >= 0.0 { 1.0 } else { -1.0 };
                let ref_sign_im = if z_ref.im >= 0.0 { 1.0 } else { -1.0 };
                let curr_sign_re = if z_curr.re >= 0.0 { 1.0 } else { -1.0 };
                let curr_sign_im = if z_curr.im >= 0.0 { 1.0 } else { -1.0 };

                let quadrant_stable = ref_sign_re == curr_sign_re && ref_sign_im == curr_sign_im;

                if quadrant_stable {
                    // Optimized Burning Ship perturbation when quadrant is stable:
                    // Use diffabs for correct derivative calculation: d/dδ |z_ref + δ|
                    // δ' = diffabs(Re(z_ref), δ_re) + i·diffabs(Im(z_ref), δ_im)
                    // Then: δ' = 2·z_abs·δ_diffabs + δ_diffabs² + dc
                    // where z_abs = (|Re(z_ref)|, |Im(z_ref)|) and δ_diffabs uses diffabs
                    // Réutiliser delta_approx calculé précédemment (optimisation 5)
                    
                    // Calculate diffabs for real and imaginary parts
                    let delta_diffabs_re = diffabs(z_ref.re, delta_approx.re);
                    let delta_diffabs_im = diffabs(z_ref.im, delta_approx.im);
                    let delta_diffabs = ComplexExp::from_complex64(Complex64::new(delta_diffabs_re, delta_diffabs_im));

                    // z_abs = (|Re(z_ref)|, |Im(z_ref)|)
                    let z_abs = Complex64::new(z_ref.re.abs(), z_ref.im.abs());

                    // Linear term: 2·z_abs·δ_diffabs
                    // For Burning Ship: d/dδ of (|z_re + δ_re|, |z_im + δ_im|)² = 2·(diffabs(z_re, δ_re), diffabs(z_im, δ_im))
                    let linear = delta_diffabs.mul_complex64(z_abs * 2.0);

                    // Quadratic term: δ_diffabs²
                    let nonlinear = delta_diffabs.mul(delta_diffabs);

                    delta = linear.add(nonlinear).add(dc);
                } else {
                    // Quadrant changed - use full computation (fallback)
                    let re = z_curr.re.abs();
                    let im = z_curr.im.abs();
                    let mut z_temp = Complex64::new(re, im);
                    z_temp = z_temp * z_temp;
                    let c_pixel = ref_orbit.cref + dc.to_complex64_approx();
                    let z_next = z_temp + c_pixel;
                    let next_index = n + 1;
                    let z_ref_next = match ref_orbit.get_z_ref_f64(next_index) {
                        Some(z) => z,
                        None => break, // End of effective orbit for this phase
                    };
                    delta = ComplexExp::from_complex64(z_next - z_ref_next);
                }
                n += 1;
            } else if is_multibrot {
                // Multibrot: z^d + c
                // δ' ≈ d·z_ref^(d-1)·δ + d(d-1)/2·z_ref^(d-2)·δ²
                // z_ref déjà en cache (optimisation 3)
                let d = multibrot_power;
                let z_norm = z_ref.norm();
                if z_norm > 1e-15 {
                    let a = z_ref.powf(d - 1.0) * d;
                    let linear = delta.mul_complex64(a);
                    let c_coeff = d * (d - 1.0) / 2.0;
                    let c = z_ref.powf(d - 2.0) * c_coeff;
                    let nonlinear = delta.mul(delta).mul_complex64(c);
                    delta = linear.add(nonlinear).add(dc);
                } else {
                    // Near origin, use simpler formula
                    delta = dc;
                }
                n += 1;
            } else if is_tricorn {
                // Tricorn: z' = conj(z)² + c
                // Use non-conformal matrices for perturbation
                // z_ref déjà en cache (optimisation 3)
                let coeffs = crate::fractal::perturbation::nonconformal::compute_tricorn_bla_coefficients(z_ref);
                
                // Utiliser le cache de delta_approx (optimisation 1)
                let delta_vec = (delta_approx_cached.re, delta_approx_cached.im);
                
                // Linear term: A·delta_vec where A = [[2X, -2Y], [-2Y, -2X]]
                let linear_vec = coeffs.a.mul_vector(delta_vec.0, delta_vec.1);
                
                // Nonlinear term: conj(δ)² = (δ_re - i·δ_im)² = (δ_re² - δ_im²) - i·2·δ_re·δ_im
                let delta_conj_sq_re = delta_vec.0 * delta_vec.0 - delta_vec.1 * delta_vec.1;
                let delta_conj_sq_im = -2.0 * delta_vec.0 * delta_vec.1;
                
                // Add dc term: B·dc_vec (B is identity)
                let dc_approx = dc.to_complex64_approx();
                let dc_vec = (dc_approx.re, dc_approx.im);
                let dc_term_vec = coeffs.b.mul_vector(dc_vec.0, dc_vec.1);
                
                // Combine: next_vec = A·delta_vec + conj(δ)² + B·dc_vec
                let next_vec = (
                    linear_vec.0 + delta_conj_sq_re + dc_term_vec.0,
                    linear_vec.1 + delta_conj_sq_im + dc_term_vec.1,
                );
                
                // Convert back to ComplexExp
                delta = ComplexExp::from_complex64(Complex64::new(next_vec.0, next_vec.1));
                n += 1;
            } else if use_high_precision {
                // High-precision path: use ComplexExp for z_ref multiplication
                // δ' = 2·z_ref·δ + δ²
                let z_ref_hp = ref_orbit.get_z_ref(n).unwrap_or_else(|| {
                    ref_orbit.z_ref[ref_orbit.z_ref.len().saturating_sub(1)]
                });
                // Scale z_ref by 2 for the linear term (optimized: multiply mantissa directly)
                let z_ref_2 = ComplexExp {
                    re: crate::fractal::perturbation::types::FloatExp::new(
                        z_ref_hp.re.mantissa * 2.0,
                        z_ref_hp.re.exponent,
                    ),
                    im: crate::fractal::perturbation::types::FloatExp::new(
                        z_ref_hp.im.mantissa * 2.0,
                        z_ref_hp.im.exponent,
                    ),
                };
                let linear = delta.mul(z_ref_2);
                let nonlinear = delta.mul(delta);
                delta = if is_julia {
                    linear.add(nonlinear)
                } else {
                    linear.add(nonlinear).add(dc)
                };
                n += 1;
            } else {
                // Standard perturbation: z_{n+1} = 2·Z_m·z_n + z_n² + c
                // Pre-compute 2·z_ref once (optimization)
                let z_ref_2 = Complex64::new(z_ref.re * 2.0, z_ref.im * 2.0);
                let linear = delta.mul_complex64(z_ref_2);  // 2·Z_m·z_n
                let nonlinear = delta.mul(delta);  // z_n²
                delta = if is_julia {
                    linear.add(nonlinear)
                } else {
                    linear.add(nonlinear).add(dc)
                };
                n += 1;
            }
            // Mettre à jour les caches après itération de perturbation
            delta_approx_cached = delta.to_complex64_approx();
            delta_norm_sqr_cached = delta.norm_sqr_approx();
            iters_ptb += 1;  // C++: iters_ptb++ après chaque itération de perturbation
        }

        // Check rebasing after perturbation iteration
        // Note: Rebasing is also checked at the start of the BLA loop, but we check again here
        // after a perturbation step to ensure we catch all rebasing opportunities
        // For high-precision path, use ComplexExp for z_curr calculation
        let (z_curr, z_ref_norm_sqr) = if use_high_precision && !is_burning_ship && !is_multibrot && !is_tricorn {
            let z_ref_hp = ref_orbit.get_z_ref(n).unwrap_or_else(|| {
                ref_orbit.z_ref[ref_orbit.z_ref.len().saturating_sub(1)]
            });
            let z_curr_hp = z_ref_hp.add(delta);
            let z_curr = z_curr_hp.to_complex64_approx();
            let z_ref_norm = z_ref_hp.norm_sqr_approx();
            (z_curr, z_ref_norm)
        } else {
            let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
            });
            // Utiliser le cache de delta_approx (optimisation 1)
            let z_curr = z_ref + delta_approx_cached;
            (z_curr, z_ref.norm_sqr())
        };

        // ====================================================================================
        // Check for rebasing opportunities after each BLA application or perturbation step
        // ====================================================================================
        //
        // Rebasing to avoid glitches: when |Z_m + z_n| < |z_n|, replace z_n with Z_m + z_n
        // and reset the reference iteration count m to 0.
        //
        // This check is performed after each BLA application or perturbation iteration step
        // as specified in the BLA Table Lookup algorithm.
        //
        // Implémentation:
        // - z_curr = Z_m + z_n (où Z_m = z_ref[n] et z_n = delta)
        //   Note: Dans le code, m et n sont représentés par la même variable n (m = n toujours)
        // - Condition: |z_curr| < |delta| (équivalent à |Z_m + z_n| < |z_n|)
        // - Si la condition est vraie:
        //   * delta ← z_curr (replace z_n with Z_m + z_n)
        //   * n ← 0 (reset m to 0, car m = n dans notre implémentation)
        let z_curr_norm_sqr = z_curr.norm_sqr();
        // Recalculer delta_norm_sqr pour la vérification de rebasing
        let delta_norm_sqr_check = delta.norm_sqr_approx();
        
        // Vérifier la condition de rebasing: |Z_m + z_n| < |z_n|
        // Équivalent à: |z_curr| < |delta|
        // Note: z_curr = Z_m + z_n où Z_m = z_ref[n] (m = n dans notre implémentation)
        if z_curr_norm_sqr > 0.0 && delta_norm_sqr_check > 0.0 && z_curr_norm_sqr < delta_norm_sqr_check {
            // Rebasing: replace z_n with Z_m + z_n and reset m to 0
            delta = ComplexExp::from_complex64(z_curr);  // replace z_n with Z_m + z_n
            
            // Hybrid BLA: change phase on rebasing: phase = (phase + n) % cycle_period
            // This must be done for all rebasing, not just in the BLA loop
            if let Some(ref mut phase) = current_phase {
                if let Some(refs) = hybrid_refs {
                    if refs.cycle_period > 0 {
                        **phase = (**phase + n) % refs.cycle_period;
                        phase_changed = true;
                    }
                }
            }
            n = 0;  // reset m to 0 (car m = n)
            continue;
        }

        if !z_curr.re.is_finite() || !z_curr.im.is_finite() {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
                suspect,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed,
            };
        }
        if z_curr.norm_sqr() > bailout_sqr {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: false,
                suspect,
                distance: f64::INFINITY, // Distance estimation not computed for escaped points
                is_interior: false,
                phase_changed,
            };
        }

        let glitch_scale = z_ref_norm_sqr + 1.0;
        if !delta_norm_sqr_check.is_finite() || delta_norm_sqr_check > glitch_tolerance_sqr * glitch_scale {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
                suspect,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed,
            };
        }
    }

    let effective_len = ref_orbit.effective_len() as u32;
    let final_index = n.min(effective_len.saturating_sub(1));
    let z_curr = if use_high_precision && !is_burning_ship && !is_multibrot && !is_tricorn {
        let z_ref_hp = ref_orbit.get_z_ref(final_index).unwrap_or_else(|| {
            ref_orbit.z_ref[ref_orbit.z_ref.len().saturating_sub(1)]
        });
        z_ref_hp.add(delta).to_complex64_approx()
    } else {
        let z_ref = ref_orbit.get_z_ref_f64(final_index).unwrap_or_else(|| {
            ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
        });
        z_ref + delta.to_complex64_approx()
    };
    DeltaResult {
        iteration: final_index,
        z_final: z_curr,
        glitched: false,
        suspect,
        distance: f64::INFINITY, // Distance estimation not computed by default
        is_interior: false, // Interior detection not computed by default
        phase_changed,
    }
}

/// Iterate a pixel using perturbation theory with full GMP precision.
/// This function is used for very deep zooms (>10^15) where f64/ComplexExp precision is insufficient.
///
/// # Arguments
///
/// * `params` - Paramètres de la fractale
/// * `ref_orbit` - Orbite de référence haute précision calculée au centre (avec z_ref_gmp)
/// * `dc_gmp` - Offset du pixel par rapport au centre en GMP (`c` dans la formule)
/// * `prec` - Précision GMP à utiliser
///
/// # Returns
///
/// DeltaResult avec le nombre d'itérations et la valeur finale de z
pub fn iterate_pixel_gmp(
    params: &FractalParams,
    ref_orbit: &ReferenceOrbit,
    dc_gmp: &Complex,
    prec: u32,
) -> DeltaResult {
    let mut n = 0u32;
    let effective_len = ref_orbit.effective_len() as u32;
    let max_iter = params.iteration_max.min(effective_len.saturating_sub(1));
    
    let bailout = Float::with_val(prec, params.bailout);
    let mut bailout_sqr = bailout.clone();
    bailout_sqr *= &bailout;
    
    // Initialisation selon le type de fractale
    // IMPORTANT: S'assurer que delta utilise la même précision que prec
    let mut delta = match params.fractal_type {
        FractalType::Julia => {
            // Julia: delta initial = dc (car z_0 = C + c pour Julia)
            // Créer une nouvelle valeur avec la précision explicite
            Complex::with_val(prec, (dc_gmp.real(), dc_gmp.imag()))
        }
        _ => {
            // Mandelbrot/BurningShip: delta initial = 0 (car z_0 = seed)
            Complex::with_val(prec, (0, 0))
        }
    };
    
    let is_julia = params.fractal_type == FractalType::Julia;
    let is_burning_ship = params.fractal_type == FractalType::BurningShip;
    let is_tricorn = params.fractal_type == FractalType::Tricorn;
    
    // Main iteration loop with full GMP precision
    while n < max_iter {
        // Get reference point at iteration n
        let z_ref = match ref_orbit.get_z_ref_gmp(n) {
            Some(z) => z,
            None => break, // End of effective orbit
        };
        
        // Apply perturbation formula depending on fractal type.
        // Burning Ship and Tricorn have their own formulas and must NOT use the
        // standard Mandelbrot perturbation (which would corrupt delta before
        // their type-specific handling).
        if is_burning_ship {
            // Burning Ship: z' = (|Re(z)|, |Im(z)|)² + c
            // For deep zooms, compute full orbit: z_curr = z_ref + delta
            // IMPORTANT: S'assurer que toutes les opérations utilisent la même précision prec
            let z_ref_prec_bs = Complex::with_val(prec, (z_ref.real(), z_ref.imag()));
            let delta_prec_bs = Complex::with_val(prec, (delta.real(), delta.imag()));
            let mut z_curr = z_ref_prec_bs;
            z_curr += &delta_prec_bs;
            
            // Apply abs() to real and imaginary parts
            let re_abs = z_curr.real().clone().abs();
            let im_abs = z_curr.imag().clone().abs();
            let mut z_abs = Complex::with_val(prec, (re_abs, im_abs));
            z_abs *= z_abs.clone();
            
            // Add cref + dc
            // IMPORTANT: S'assurer que toutes les valeurs utilisent la même précision
            let mut z_next = Complex::with_val(prec, (z_abs.real(), z_abs.imag()));
            let cref_prec = Complex::with_val(prec, (ref_orbit.cref_gmp.real(), ref_orbit.cref_gmp.imag()));
            z_next += &cref_prec;
            if !is_julia {
                let dc_gmp_prec = Complex::with_val(prec, (dc_gmp.real(), dc_gmp.imag()));
                z_next += &dc_gmp_prec;
            }
            
            // Calculate delta for next iteration: z_next - z_ref_next
            if (n + 1) >= effective_len {
                // Rebase: delta = z_next, reset n to 0
                // IMPORTANT: S'assurer que delta utilise la bonne précision
                delta = Complex::with_val(prec, (z_next.real(), z_next.imag()));
                n = 0;
                continue;
            }
            
            let z_ref_next = match ref_orbit.get_z_ref_gmp(n + 1) {
                Some(z) => z,
                None => break,
            };
            // IMPORTANT: Créer de nouvelles valeurs avec la précision explicite pour la soustraction
            let z_next_prec = Complex::with_val(prec, (z_next.real(), z_next.imag()));
            let z_ref_next_prec = Complex::with_val(prec, (z_ref_next.real(), z_ref_next.imag()));
            delta = z_next_prec - z_ref_next_prec;
        } else if is_tricorn {
            // Tricorn: z' = conj(z)² + c
            // IMPORTANT: S'assurer que toutes les opérations utilisent la même précision prec
            let z_ref_prec_tc = Complex::with_val(prec, (z_ref.real(), z_ref.imag()));
            let delta_prec_tc = Complex::with_val(prec, (delta.real(), delta.imag()));
            let mut z_curr = z_ref_prec_tc;
            z_curr += &delta_prec_tc;
            let z_conj = z_curr.clone().conj();
            let mut z_temp = Complex::with_val(prec, (z_conj.real(), z_conj.imag()));
            z_temp *= &z_conj;
            // IMPORTANT: S'assurer que cref_gmp utilise la même précision
            let cref_prec = Complex::with_val(prec, (ref_orbit.cref_gmp.real(), ref_orbit.cref_gmp.imag()));
            z_temp += &cref_prec;
            if !is_julia {
                let dc_gmp_prec = Complex::with_val(prec, (dc_gmp.real(), dc_gmp.imag()));
                z_temp += &dc_gmp_prec;
            }
            
            if (n + 1) >= effective_len {
                // IMPORTANT: S'assurer que delta utilise la bonne précision
                delta = Complex::with_val(prec, (z_temp.real(), z_temp.imag()));
                n = 0;
                continue;
            }
            
            let z_ref_next = match ref_orbit.get_z_ref_gmp(n + 1) {
                Some(z) => z,
                None => break,
            };
            // IMPORTANT: Créer de nouvelles valeurs avec la précision explicite pour la soustraction
            let z_temp_prec = Complex::with_val(prec, (z_temp.real(), z_temp.imag()));
            let z_ref_next_prec = Complex::with_val(prec, (z_ref_next.real(), z_ref_next.imag()));
            delta = z_temp_prec - z_ref_next_prec;
        } else {
            // Standard Mandelbrot/Julia: delta_{n+1} = 2·z_ref·delta + delta² + dc
            let delta_prec = Complex::with_val(prec, (delta.real(), delta.imag()));
            let z_ref_prec = Complex::with_val(prec, (z_ref.real(), z_ref.imag()));

            let mut delta_sq = delta_prec.clone();
            delta_sq *= &delta_prec;

            let mut linear_term = z_ref_prec.clone();
            linear_term *= &delta_prec;
            let two = Float::with_val(prec, 2.0);
            linear_term *= &two;

            let mut next_delta = Complex::with_val(prec, (linear_term.real(), linear_term.imag()));
            next_delta += &delta_sq;

            if !is_julia {
                let dc_gmp_prec = Complex::with_val(prec, (dc_gmp.real(), dc_gmp.imag()));
                next_delta += &dc_gmp_prec;
            }

            delta = next_delta;
        }
        
        // Advance iteration counter: delta now holds delta_{n+1}
        n += 1;

        // For Mandelbrot standard path, handle orbit end (BS/Tricorn already handled above)
        if !is_burning_ship && !is_tricorn && n >= effective_len {
            delta = Complex::with_val(prec, (delta.real(), delta.imag()));
            // Orbit end reached, just continue to exit via while condition
            break;
        }

        // Check bailout using z_ref[n] (the NEW n, i.e. the next reference point)
        // IMPORTANT: After computing delta_{n+1}, the correct full z is z_ref[n+1] + delta_{n+1}
        let z_ref_next = match ref_orbit.get_z_ref_gmp(n) {
            Some(z) => z,
            None => break,
        };
        let z_ref_next_prec = Complex::with_val(prec, (z_ref_next.real(), z_ref_next.imag()));
        let delta_prec = Complex::with_val(prec, (delta.real(), delta.imag()));
        let mut z_curr = z_ref_next_prec.clone();
        z_curr += &delta_prec;
        let z_curr_norm_sqr = complex_norm_sqr(&z_curr, prec);

        if !z_curr.real().is_finite() || !z_curr.imag().is_finite() {
            return DeltaResult {
                iteration: n,
                z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
                glitched: true,
                suspect: false,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed: false,
            };
        }

        if z_curr_norm_sqr > bailout_sqr {
            return DeltaResult {
                iteration: n,
                z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
                glitched: false,
                suspect: false,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed: false,
            };
        }

        // Check for rebasing: when |Z_m + z_n| < |z_n|
        let delta_norm_sqr = complex_norm_sqr(&delta_prec, prec);
        if z_curr_norm_sqr > Float::with_val(prec, 0.0)
            && delta_norm_sqr > Float::with_val(prec, 0.0)
            && z_curr_norm_sqr < delta_norm_sqr {
            // Rebasing: replace z_n with Z_m + z_n and reset m to 0
            delta = z_curr;
            n = 0;
            continue;
        }

        // Check for glitch: delta is too large relative to z_ref at current iteration
        let pixel_size = params.span_x / params.width as f64;
        let adaptive_tolerance = compute_adaptive_glitch_tolerance(pixel_size, params.glitch_tolerance);
        let glitch_tolerance_sqr = Float::with_val(prec, adaptive_tolerance * adaptive_tolerance);
        let z_ref_norm_sqr = complex_norm_sqr(&z_ref_next_prec, prec);
        let mut glitch_scale = z_ref_norm_sqr.clone();
        glitch_scale += Float::with_val(prec, 1.0);
        let mut glitch_threshold = glitch_tolerance_sqr.clone();
        glitch_threshold *= &glitch_scale;

        // Check if delta_norm_sqr is too large (glitch detected)
        if !delta_norm_sqr.is_finite() || delta_norm_sqr > glitch_threshold {
            return DeltaResult {
                iteration: n,
                z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
                glitched: true,
                suspect: false,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed: false,
            };
        }
    }
    
    // Final result
    // IMPORTANT: S'assurer que toutes les opérations utilisent la même précision prec
    let final_index = n.min(effective_len.saturating_sub(1));
    let z_ref = match ref_orbit.get_z_ref_gmp(final_index) {
        Some(z) => z,
        None => match ref_orbit.z_ref_gmp.last() {
            Some(z) => z,
            None => {
                // Vecteur vide - retourner un résultat glitch
                return DeltaResult {
                    iteration: 0,
                    z_final: Complex64::new(0.0, 0.0),
                    glitched: true,
                    suspect: false,
                    distance: f64::INFINITY,
                    is_interior: false,
                    phase_changed: false,
                };
            }
        },
    };
    let z_ref_prec = Complex::with_val(prec, (z_ref.real(), z_ref.imag()));
    let delta_prec = Complex::with_val(prec, (delta.real(), delta.imag()));
    let mut z_curr = z_ref_prec.clone();
    z_curr += &delta_prec;
    
    // Final glitch check: verify delta is reasonable
    let pixel_size = params.span_x / params.width as f64;
    let adaptive_tolerance = compute_adaptive_glitch_tolerance(pixel_size, params.glitch_tolerance);
    let glitch_tolerance_sqr = Float::with_val(prec, adaptive_tolerance * adaptive_tolerance);
    let z_ref_norm_sqr = complex_norm_sqr(&z_ref_prec, prec);
    let delta_norm_sqr = complex_norm_sqr(&delta_prec, prec);
    let mut glitch_scale = z_ref_norm_sqr.clone();
    glitch_scale += Float::with_val(prec, 1.0);
    let mut glitch_threshold = glitch_tolerance_sqr.clone();
    glitch_threshold *= &glitch_scale;
    let is_glitched = !delta_norm_sqr.is_finite() || delta_norm_sqr > glitch_threshold;
    
    DeltaResult {
        iteration: final_index,
        z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
        glitched: is_glitched,
        suspect: false,
        distance: f64::INFINITY,
        is_interior: false,
        phase_changed: false,
    }
}

/// Version de iterate_pixel() utilisant ExtendedDualComplex pour distance estimation et interior detection.
/// 
/// # Arguments
/// * `params` - Paramètres de la fractale
/// * `ref_orbit` - Orbite de référence
/// * `bla_table` - Table BLA
/// * `series_table` - Table de séries (optionnelle)
/// * `delta0` - Delta initial
/// * `dc` - Offset pixel par rapport au centre
/// * `pixel_x` - Coordonnée X du pixel (0..width)
/// * `pixel_y` - Coordonnée Y du pixel (0..height)
/// * `enable_distance` - Activer distance estimation
/// * `enable_interior` - Activer interior detection
/// * `current_phase` - Current phase (for Hybrid BLA, not used in this function)
/// * `hybrid_refs` - Hybrid BLA references (for Hybrid BLA, not used in this function)
pub(crate) fn iterate_pixel_with_duals(
    params: &FractalParams,
    ref_orbit: &ReferenceOrbit,
    bla_table: &BlaTable,
    _series_table: Option<&SeriesTable>,
    delta0: ComplexExp,
    dc: ComplexExp,
    pixel_x: f64,
    pixel_y: f64,
    enable_distance: bool,
    enable_interior: bool,
    _current_phase: Option<&mut u32>,
    _hybrid_refs: Option<&HybridBlaReferences>,
) -> DeltaResult {
    let mut n = 0u32;
    // Hybrid BLA: account for phase offset in effective length
    let effective_len = ref_orbit.effective_len() as u32;
    let max_iter = params.iteration_max.min(effective_len.saturating_sub(1));
    let bailout_sqr = params.bailout * params.bailout;
    
    let pixel_size = params.span_x / params.width as f64;
    let adaptive_tolerance = compute_adaptive_glitch_tolerance(pixel_size, params.glitch_tolerance);
    let glitch_tolerance_sqr = adaptive_tolerance * adaptive_tolerance;
    let is_julia = params.fractal_type == FractalType::Julia;
    let is_burning_ship = params.fractal_type == FractalType::BurningShip;
    let is_multibrot = params.fractal_type == FractalType::Multibrot;
    let is_tricorn = params.fractal_type == FractalType::Tricorn;
    let multibrot_power = params.multibrot_power;
    let _series_config = SeriesConfig::from_params(params);
    let suspect = false;
    
    // Initialize ExtendedDualComplex
    // Transform pixel coordinates to complex plane with derivatives
    let pixel_dual = transform_pixel_to_complex(
        pixel_x,
        pixel_y,
        params.center_x,
        params.center_y,
        params.span_x,
        params.span_y,
        params.width as f64,
        params.height as f64,
    );
    
    // For Mandelbrot: dc has derivatives, for Julia: dc is constant (no derivatives)
    let dc_dual = if is_julia {
        ExtendedDualComplex::from_complex(dc.to_complex64_approx())
    } else {
        // dc_dual should have same derivatives as pixel transformation
        ExtendedDualComplex {
            value: dc.to_complex64_approx(),
            dual_re: pixel_dual.dual_re,
            dual_im: pixel_dual.dual_im,
            dual_z1_re: Complex64::new(0.0, 0.0),
            dual_z1_im: Complex64::new(0.0, 0.0),
        }
    };
    
    // Initialize delta_dual: start with delta0, add derivatives for interior detection if enabled
    let mut delta_dual = ExtendedDualComplex::from_complex(delta0.to_complex64_approx());
    if enable_interior {
        // Initialize interior derivative: dzdz1 = 1+0i at critical point
        delta_dual.dual_z1_re = Complex64::new(1.0, 0.0);
    }
    
    // Try standalone series skip (simplified - doesn't propagate duals fully)
    // For now, skip series when using duals to avoid complexity
    
    while n < max_iter {
        let mut stepped = false;

        // Rebase if we've reached the end of the effective orbit
        if n >= effective_len {
            let last_idx = effective_len.saturating_sub(1);
            let z_ref = ref_orbit.get_z_ref_f64(last_idx).unwrap_or_else(|| {
                ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
            });
            // Rebase: update value to full z_curr, duals are preserved (z_ref is constant)
            delta_dual.value = z_ref + delta_dual.value;
            n = 0;
        }

        // Check interior detection
        if enable_interior {
            if is_interior(delta_dual, params.interior_threshold) {
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let z_curr = z_ref + delta_dual.value;
                return DeltaResult {
                    iteration: n,
                    z_final: z_curr,
                    glitched: false,
                    suspect,
                    distance: f64::INFINITY,
                    is_interior: true,
                    phase_changed: false,
                };
            }
        }

        // Try BLA for conformal fractals
        if !is_tricorn && !bla_table.levels.is_empty() {
            let delta_norm_sqr = delta_dual.norm_sqr();
            for level in (0..bla_table.levels.len()).rev() {
                let level_nodes = &bla_table.levels[level];
                if (n as usize) >= level_nodes.len() {
                    continue;
                }
                let node = &level_nodes[n as usize];
                if is_burning_ship && !node.burning_ship_valid {
                    continue;
                }
                if delta_norm_sqr < node.validity_radius * node.validity_radius {
                    // Apply BLA with dual propagation
                    let work_delta = if is_burning_ship && node.burning_ship_valid {
                        delta_dual.mul_signed(node.sign_re, node.sign_im)
                    } else {
                        delta_dual
                    };
                    
                    // Linear term: A·delta (with dual propagation)
                    let a_dual = ExtendedDualComplex::from_complex(node.a);
                    let mut next_dual = work_delta.mul(a_dual);
                    
                    // Add dc term if not Julia
                    if !is_julia {
                        let b_dual = ExtendedDualComplex::from_complex(node.b);
                        let dc_scaled = dc_dual.mul(b_dual);
                        next_dual = next_dual.add(dc_scaled);
                    }
                    
                    delta_dual = next_dual;
                    n += 1u32 << level;
                    stepped = true;
                    break;
                }
            }
        }
        
        // Try non-conformal BLA for Tricorn
        if is_tricorn {
            if let Some(ref nonconformal_levels) = bla_table.nonconformal_levels {
                if !nonconformal_levels.is_empty() {
                    let delta_vec = (delta_dual.value.re, delta_dual.value.im);
                    let delta_norm_sqr_check = delta_vec.0 * delta_vec.0 + delta_vec.1 * delta_vec.1;
                    
                    for level in (0..nonconformal_levels.len()).rev() {
                        let level_nodes = &nonconformal_levels[level];
                        if (n as usize) >= level_nodes.len() {
                            continue;
                        }
                        let node = &level_nodes[n as usize];
                        
                        if delta_norm_sqr_check < node.validity_radius * node.validity_radius {
                            // Apply non-conformal BLA with full dual propagation
                            // z_{n+l} = A·z_n + B·dc (2×2 real matrices applied to all components)
                            let linear_dual = apply_nonconformal_matrix(node.a, delta_dual);
                            let dc_term_dual = apply_nonconformal_matrix(node.b, dc_dual);
                            delta_dual = linear_dual.add(dc_term_dual);
                            n += 1u32 << level;
                            stepped = true;
                            break;
                        }
                    }
                }
            }
        }
        
        if !stepped {
            // Standard perturbation iteration with dual propagation
            if is_burning_ship {
                // Burning Ship: z' = (|Re(z)| + i|Im(z)|)² + c
                // Perturbation: delta' = 2·z_abs·delta_diffabs + delta_diffabs² + dc
                // where z_abs = (|Re(z_ref)|, |Im(z_ref)|)
                //       delta_diffabs = (diffabs(Re(z_ref), Re(delta)), diffabs(Im(z_ref), Im(delta)))
                //
                // Dual propagation through abs(): d|x|/dk = sign(x) · dx/dk
                // Applied via sign of z_curr = z_ref + delta (not z_ref)
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let z_curr_value = z_ref + delta_dual.value;
                let sign_re = if z_curr_value.re >= 0.0 { 1.0 } else { -1.0 };
                let sign_im = if z_curr_value.im >= 0.0 { 1.0 } else { -1.0 };

                // delta_diffabs: value via diffabs, duals via sign(z_curr)
                let delta_diffabs = ExtendedDualComplex {
                    value: Complex64::new(
                        diffabs(z_ref.re, delta_dual.value.re),
                        diffabs(z_ref.im, delta_dual.value.im),
                    ),
                    dual_re: Complex64::new(delta_dual.dual_re.re * sign_re, delta_dual.dual_re.im * sign_im),
                    dual_im: Complex64::new(delta_dual.dual_im.re * sign_re, delta_dual.dual_im.im * sign_im),
                    dual_z1_re: Complex64::new(delta_dual.dual_z1_re.re * sign_re, delta_dual.dual_z1_re.im * sign_im),
                    dual_z1_im: Complex64::new(delta_dual.dual_z1_im.re * sign_re, delta_dual.dual_z1_im.im * sign_im),
                };

                // z_abs = (|Re(z_ref)|, |Im(z_ref)|) — constant, zero derivatives
                let z_abs = Complex64::new(z_ref.re.abs(), z_ref.im.abs());
                let z_abs_2_dual = ExtendedDualComplex::from_complex(z_abs * 2.0);

                // delta' = 2·z_abs·delta_diffabs + delta_diffabs² + dc
                let linear = z_abs_2_dual.mul(delta_diffabs);
                let nonlinear = delta_diffabs.square();
                delta_dual = if is_julia {
                    linear.add(nonlinear)
                } else {
                    linear.add(nonlinear).add(dc_dual)
                };
                n += 1;
            } else if is_multibrot {
                // Multibrot: z^d + c
                // Perturbation: delta' ≈ d·Z^(d-1)·delta + d(d-1)/2·Z^(d-2)·delta² + dc
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let d = multibrot_power;
                let z_norm = z_ref.norm();
                if z_norm > 1e-15 {
                    // Linear coefficient A = d·Z^(d-1) — constant, zero derivatives
                    let a = z_ref.powf(d - 1.0) * d;
                    let a_dual = ExtendedDualComplex::from_complex(a);
                    let linear = a_dual.mul(delta_dual);

                    // Quadratic coefficient C = d(d-1)/2·Z^(d-2) — constant, zero derivatives
                    let c_coeff = d * (d - 1.0) / 2.0;
                    let c_val = z_ref.powf(d - 2.0) * c_coeff;
                    let c_dual = ExtendedDualComplex::from_complex(c_val);
                    let nonlinear = c_dual.mul(delta_dual.square());

                    delta_dual = if is_julia {
                        linear.add(nonlinear)
                    } else {
                        linear.add(nonlinear).add(dc_dual)
                    };
                } else {
                    delta_dual = dc_dual;
                }
                n += 1;
            } else if is_tricorn {
                // Tricorn: z' = conj(z)² + c
                // Perturbation: delta' = A·delta + conj(delta)² + dc
                // A = [[2X, -2Y], [-2Y, -2X]] where X=Re(z_ref), Y=Im(z_ref)
                // Duals propagated through real 2×2 Jacobian matrices
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let coeffs = crate::fractal::perturbation::nonconformal::compute_tricorn_bla_coefficients(z_ref);

                // Linear term: A·delta (with full dual propagation through 2×2 matrix)
                let linear_dual = apply_nonconformal_matrix(coeffs.a, delta_dual);

                // Nonlinear term: conj(δ)² = (d_re²-d_im², -2·d_re·d_im)
                // Jacobian of conj(δ)²: J = [[2·d_re, -2·d_im], [-2·d_im, -2·d_re]]
                let d_re = delta_dual.value.re;
                let d_im = delta_dual.value.im;
                let nonlin_jacobian = Matrix2x2 {
                    m00: 2.0 * d_re, m01: -2.0 * d_im,
                    m10: -2.0 * d_im, m11: -2.0 * d_re,
                };
                let mut nonlin_dual = apply_nonconformal_matrix(nonlin_jacobian, delta_dual);
                // Override value: Jacobian gives 2×(conj(δ)²), but actual value is conj(δ)²
                nonlin_dual.value = Complex64::new(d_re * d_re - d_im * d_im, -2.0 * d_re * d_im);

                // dc term: B·dc (B = identity, propagated through matrix)
                delta_dual = if is_julia {
                    linear_dual.add(nonlin_dual)
                } else {
                    let dc_term_dual = apply_nonconformal_matrix(coeffs.b, dc_dual);
                    linear_dual.add(nonlin_dual).add(dc_term_dual)
                };
                n += 1;
            } else {
                // Mandelbrot/Julia: delta_{n+1} = 2·Z_n·delta_n + delta_n² + dc
                // Standard perturbation formula avoids precision-losing subtraction z_next - z_ref_next
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                // 2·Z_n is a constant (zero derivatives)
                let z_ref_2_dual = ExtendedDualComplex::from_complex(z_ref * 2.0);
                // Linear term: 2·Z_n·delta (propagates duals: 2·Z_n·d(delta)/dk)
                let linear = z_ref_2_dual.mul(delta_dual);
                // Nonlinear term: delta² (propagates duals: 2·delta·d(delta)/dk)
                let nonlinear = delta_dual.square();
                // delta_{n+1} = linear + nonlinear [+ dc]
                delta_dual = if is_julia {
                    linear.add(nonlinear)
                } else {
                    linear.add(nonlinear).add(dc_dual)
                };
                n += 1;
            }
        }
        
        // Check bailout
        let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
            ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
        });
        let z_curr = z_ref + delta_dual.value;
        
        if !z_curr.re.is_finite() || !z_curr.im.is_finite() {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
                suspect,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed: false,
            };
        }
        
        if z_curr.norm_sqr() > bailout_sqr {
            // Calculate distance estimation if enabled
            let distance = if enable_distance {
                // Extract DualComplex from ExtendedDualComplex
                let dual = DualComplex {
                    value: delta_dual.value,
                    dual_re: delta_dual.dual_re,
                    dual_im: delta_dual.dual_im,
                };
                compute_distance_estimate(dual)
            } else {
                f64::INFINITY
            };
            
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: false,
                suspect,
                distance,
                is_interior: false,
                phase_changed: false,
            };
        }
        
        // Check glitch
        let z_ref_norm_sqr = z_ref.norm_sqr();
        let delta_norm_sqr = delta_dual.norm_sqr();
        let glitch_scale = z_ref_norm_sqr + 1.0;
        if !delta_norm_sqr.is_finite() || delta_norm_sqr > glitch_tolerance_sqr * glitch_scale {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
                suspect,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed: false,
            };
        }

        // Rebasing: when |z_curr| < |delta|, replace delta with z_curr and reset n
        // Duals are preserved because z_ref is constant (no dependence on pixel coords or z1)
        let z_curr_norm_sqr = z_curr.norm_sqr();
        if z_curr_norm_sqr > 0.0 && delta_norm_sqr > 0.0 && z_curr_norm_sqr < delta_norm_sqr {
            delta_dual.value = z_curr;
            n = 0;
            continue;
        }
    }
    
    // Final result
    let effective_len = ref_orbit.effective_len() as u32;
    let final_index = n.min(effective_len.saturating_sub(1));
    let z_ref = ref_orbit.get_z_ref_f64(final_index).unwrap_or_else(|| {
        ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
    });
    let z_curr = z_ref + delta_dual.value;
    
    let distance = if enable_distance {
        let dual = DualComplex {
            value: delta_dual.value,
            dual_re: delta_dual.dual_re,
            dual_im: delta_dual.dual_im,
        };
        compute_distance_estimate(dual)
    } else {
        f64::INFINITY
    };
    
    let is_interior_result = if enable_interior {
        is_interior(delta_dual, params.interior_threshold)
    } else {
        false
    };
    
    DeltaResult {
        iteration: final_index,
        z_final: z_curr,
        glitched: false,
        suspect,
        distance,
        is_interior: is_interior_result,
        phase_changed: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::definitions::default_params_for_type;
    use crate::fractal::{AlgorithmMode, FractalParams, FractalType};

    fn test_params() -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 100, 100);
        p.span_x = 4.0;
        p.span_y = 4.0;
        p.iteration_max = 100;
        p.precision_bits = 192;
        p.algorithm_mode = AlgorithmMode::Perturbation;
        p.bla_threshold = 1e-6;
        p.glitch_neighbor_pass = false;
        p
    }

    #[test]
    fn delta_result_has_new_fields() {
        let result = DeltaResult {
            iteration: 10,
            z_final: Complex64::new(1.0, 2.0),
            glitched: false,
            suspect: false,
            distance: f64::INFINITY,
            is_interior: false,
            phase_changed: false,
        };
        assert_eq!(result.iteration, 10);
        assert_eq!(result.distance, f64::INFINITY);
        assert_eq!(result.is_interior, false);
        assert_eq!(result.phase_changed, false);
    }
}

#[cfg(test)]
mod dual_tests {
    use super::*;
    use crate::fractal::perturbation::distance::{DualComplex, compute_distance_estimate};
    use crate::fractal::perturbation::interior::{ExtendedDualComplex, is_interior};
    
    #[test]
    fn dual_complex_propagation() {
        let z1 = DualComplex {
            value: Complex64::new(1.0, 2.0),
            dual_re: Complex64::new(1.0, 0.0),
            dual_im: Complex64::new(0.0, 1.0),
        };
        let z2 = DualComplex {
            value: Complex64::new(3.0, 4.0),
            dual_re: Complex64::new(1.0, 0.0),
            dual_im: Complex64::new(0.0, 1.0),
        };
        let prod = z1.mul(z2);
        // (1+2i)*(3+4i) = -5+10i
        assert!((prod.value.re - (-5.0)).abs() < 1e-10);
        assert!((prod.value.im - 10.0).abs() < 1e-10);
    }
    
    #[test]
    fn distance_estimation_calculation() {
        let dual = DualComplex {
            value: Complex64::new(2.0, 0.0),
            dual_re: Complex64::new(1.0, 0.0),
            dual_im: Complex64::new(0.0, 1.0),
        };
        let distance = compute_distance_estimate(dual);
        // distance = |z|·ln|z| / |dz/dk| = 2·ln(2) / sqrt(2) ≈ 0.9803
        assert!(distance > 0.0 && distance.is_finite());
    }
    
    #[test]
    fn interior_detection() {
        // Point with small derivative (interior)
        let interior = ExtendedDualComplex {
            value: Complex64::new(0.1, 0.1),
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
            dual_z1_re: Complex64::new(0.0005, 0.0),
            dual_z1_im: Complex64::new(0.0, 0.0),
        };
        assert!(is_interior(interior, 0.001));
        
        // Point with large derivative (exterior)
        let exterior = ExtendedDualComplex {
            value: Complex64::new(2.0, 2.0),
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
            dual_z1_re: Complex64::new(10.0, 0.0),
            dual_z1_im: Complex64::new(0.0, 0.0),
        };
        assert!(!is_interior(exterior, 0.001));
    }
}

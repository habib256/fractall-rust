use num_complex::Complex64;
use rug::{Complex, Float};

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::perturbation::bla::BlaTable;
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::types::ComplexExp;
use crate::fractal::gmp::complex_norm_sqr;
use crate::fractal::perturbation::series::{
    SeriesConfig, SeriesTable, should_use_series, estimate_series_error, compute_series_skip,
};
use crate::fractal::perturbation::distance::{DualComplex, compute_distance_estimate, transform_pixel_to_complex};
use crate::fractal::perturbation::interior::{ExtendedDualComplex, is_interior};

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
    // Tolerances resserrées pour mieux détecter les glitches aux zooms profonds
    // et les recalculer en GMP plutôt que d'accumuler des erreurs numériques
    match zoom_level as u32 {
        0..=6 => 1e-6,    // Zoom peu profond : très strict
        7..=14 => 1e-5,   // Moyen : strict
        15..=30 => 1e-4,  // Profond : standard (resserré de 1e-3)
        31..=50 => 1e-3,  // Très profond : relaxé (resserré de 1e-2)
        _ => 1e-2,        // Extrême (>50) : tolérant mais contrôlé (resserré de 1e-1)
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
pub fn iterate_pixel(
    params: &FractalParams,
    ref_orbit: &ReferenceOrbit,
    bla_table: &BlaTable,
    series_table: Option<&SeriesTable>,
    delta0: ComplexExp,
    dc: ComplexExp,
) -> DeltaResult {
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
        );
    }
    
    // Standard path without dual numbers
    let mut n = 0u32;
    let mut delta = delta0;
    // Use z_ref_f64 for fast path iteration (z_ref is high-precision Vec<ComplexExp>)
    // Hybrid BLA: account for phase offset in effective length
    let effective_len = ref_orbit.effective_len() as u32;
    let max_iter = params.iteration_max.min(effective_len.saturating_sub(1));
    let bailout_sqr = params.bailout * params.bailout;

    // Calculer le pixel_size pour la tolérance adaptative
    let pixel_size = params.span_x / params.width as f64;
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

    // Try standalone series skip BEFORE BLA (if enabled and series table is available)
    if let Some(table) = series_table {
        if params.series_standalone && !is_burning_ship && !is_multibrot {
            if let Some(skip_result) = compute_series_skip(
                table,
                delta,
                dc,
                series_config.error_tolerance,
                is_julia,
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
    // For each iteration:
    // 1. Find the BLA starting from iteration m (n) that has the largest skip l satisfying |z| < R
    // 2. If there is none, do a perturbation iteration
    // 3. Check for rebasing opportunities after each BLA application or perturbation step
    while n < max_iter {
        let mut stepped = false;  // Track if a BLA was applied
        let mut delta_norm_sqr = 0.0;

        // Try non-conformal BLA first for Tricorn
        if is_tricorn {
            if let Some(ref nonconformal_levels) = bla_table.nonconformal_levels {
                if !nonconformal_levels.is_empty() {
                    // Convert delta to (re, im) vector
                    let delta_approx = delta.to_complex64_approx();
                    let delta_vec = (delta_approx.re, delta_approx.im);
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
                            delta_norm_sqr = delta.norm_sqr_approx();
                            n += 1u32 << level;  // Skip l = 2^level iterations
                            stepped = true;
                            break;
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
            delta_norm_sqr = delta.norm_sqr_approx();
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
                    delta_norm_sqr = delta.norm_sqr_approx();
                    // Skip l = 2^level iterations: z_{n+l} has been computed
                    n += 1u32 << level;
                    stepped = true;
                    break;
                }
            }
        }

        // If no BLA was found (stepped == false), do a perturbation iteration
        if !stepped {
            if is_burning_ship {
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let delta_approx = delta.to_complex64_approx();
                let z_curr = z_ref + delta_approx;

                // Check if quadrant is stable (same signs)
                let ref_sign_re = if z_ref.re >= 0.0 { 1.0 } else { -1.0 };
                let ref_sign_im = if z_ref.im >= 0.0 { 1.0 } else { -1.0 };
                let curr_sign_re = if z_curr.re >= 0.0 { 1.0 } else { -1.0 };
                let curr_sign_im = if z_curr.im >= 0.0 { 1.0 } else { -1.0 };

                let quadrant_stable = ref_sign_re == curr_sign_re && ref_sign_im == curr_sign_im;

                if quadrant_stable {
                    // Optimized Burning Ship perturbation when quadrant is stable:
                    // δ' = 2·(s_re·|Re(z_ref)|, s_im·|Im(z_ref)|)·δ_signed + δ_signed² + dc
                    // where δ_signed = (s_re·δ_re, s_im·δ_im)

                    // Apply signs to delta
                    let delta_signed = delta.mul_signed(ref_sign_re, ref_sign_im);

                    // z_abs = (|Re(z_ref)|, |Im(z_ref)|)
                    let z_abs = Complex64::new(z_ref.re.abs(), z_ref.im.abs());

                    // Linear term: 2·z_abs·δ_signed (component-wise for Burning Ship)
                    // For Burning Ship: d/dδ of (|z_re + δ_re|, |z_im + δ_im|)² = 2·(s_re·|z_re|, s_im·|z_im|)
                    let linear = delta_signed.mul_complex64(z_abs * 2.0);

                    // Quadratic term: δ_signed²
                    let nonlinear = delta_signed.mul(delta_signed);

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
                delta_norm_sqr = delta.norm_sqr_approx();
                n += 1;
            } else if is_multibrot {
                // Multibrot: z^d + c
                // δ' ≈ d·z_ref^(d-1)·δ + d(d-1)/2·z_ref^(d-2)·δ²
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
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
                delta_norm_sqr = delta.norm_sqr_approx();
                n += 1;
            } else if is_tricorn {
                // Tricorn: z' = conj(z)² + c
                // Use non-conformal matrices for perturbation
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let coeffs = crate::fractal::perturbation::nonconformal::compute_tricorn_bla_coefficients(z_ref);
                
                // Convert delta to vector
                let delta_approx = delta.to_complex64_approx();
                let delta_vec = (delta_approx.re, delta_approx.im);
                
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
                delta_norm_sqr = delta.norm_sqr_approx();
                n += 1;
            } else if use_high_precision {
                // High-precision path: use ComplexExp for z_ref multiplication
                // δ' = 2·z_ref·δ + δ²
                let z_ref_hp = ref_orbit.get_z_ref(n).unwrap_or_else(|| {
                    ref_orbit.z_ref[ref_orbit.z_ref.len().saturating_sub(1)]
                });
                // Scale z_ref by 2 for the linear term
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
                if is_julia {
                    delta = linear.add(nonlinear);
                } else {
                    delta = linear.add(nonlinear).add(dc);
                }
                delta_norm_sqr = delta.norm_sqr_approx();
                n += 1;
            } else {
                // Section 2: Perturbation
                //
                // Formule de perturbation: z_{n+1} = 2·Z_m·z_n + z_n² + c
                //
                // Décomposition selon la documentation:
                // - Terme linéaire: 2·Z_m·z_n
                // - Terme quadratique: z_n²
                // - Terme constant: c (offset du pixel)
                //
                // Notation mathématique → code:
                // - Z_m ↔ z_ref[n] (orbite de référence haute précision, où m = n dans notre implémentation)
                // - z_n ↔ delta (delta de perturbation basse précision)
                // - c ↔ dc (offset du pixel par rapport au centre)
                //
                // Pour Julia, le point C est fixe (seed), donc pas de terme c dans la perturbation.
                // Hybrid BLA: use get_z_ref_f64 to account for phase offset
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });  // Z_m (où m = n, avec phase offset pour Hybrid BLA)
                let linear = delta.mul_complex64(z_ref * 2.0);  // 2·Z_m·z_n
                let nonlinear = delta.mul(delta);  // z_n²
                if is_julia {
                    // Julia: z_{n+1} = 2·Z_m·z_n + z_n² (pas de terme c car C est fixe)
                    delta = linear.add(nonlinear);
                } else {
                    // Mandelbrot: z_{n+1} = 2·Z_m·z_n + z_n² + c
                    delta = linear.add(nonlinear).add(dc);
                }
                delta_norm_sqr = delta.norm_sqr_approx();
                n += 1;  // Incrémenter m et n simultanément (m = n dans notre implémentation)
            }
        }

        // Rebasing quand on atteint la fin de l'orbite de référence (amélioration de Zhuoran)
        //
        // Hybrid BLA: For a hybrid loop with multiple phases, rebasing switches to the reference
        // for the current phase. You need one BLA table per reference.
        //
        // When reaching the end of the effective orbit:
        // - If Hybrid BLA is enabled and a cycle is detected, switch to the reference for the next phase
        // - Otherwise, reset delta and reuse the same orbit in a loop (Zhuoran's method)
        let effective_len = ref_orbit.effective_len() as u32;
        if n >= effective_len {
            // Calculer z_curr en utilisant le dernier point de référence valide
            let last_idx = effective_len.saturating_sub(1) as usize;
            let z_ref = ref_orbit.get_z_ref_f64(last_idx as u32).unwrap_or_else(|| {
                ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
            });
            let delta_approx = delta.to_complex64_approx();
            let z_curr = z_ref + delta_approx;
            
            // Rebasing: delta = z_curr
            // In Hybrid BLA, if hybrid_refs is provided, we would switch to the reference
            // for the next phase. For now, we reset n to 0 and continue with the same reference.
            // Full Hybrid BLA implementation would switch references here.
            delta = ComplexExp::from_complex64(z_curr);
            n = 0;
            continue;
        }

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
            let delta_approx = delta.to_complex64_approx();
            let z_curr = z_ref + delta_approx;
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
        // Recalculer delta_norm_sqr si pas déjà calculé (ex: après un saut BLA)
        if !stepped {
            delta_norm_sqr = delta.norm_sqr_approx();
        }
        let delta_norm_sqr_check = delta_norm_sqr;
        
        // Vérifier la condition de rebasing: |Z_m + z_n| < |z_n|
        // Équivalent à: |z_curr| < |delta|
        // Note: z_curr = Z_m + z_n où Z_m = z_ref[n] (m = n dans notre implémentation)
        if z_curr_norm_sqr > 0.0 && delta_norm_sqr_check > 0.0 && z_curr_norm_sqr < delta_norm_sqr_check {
            // Rebasing: replace z_n with Z_m + z_n and reset m to 0
            delta = ComplexExp::from_complex64(z_curr);  // replace z_n with Z_m + z_n
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
            };
        }

        let glitch_scale = z_ref_norm_sqr + 1.0;
        if !delta_norm_sqr.is_finite() || delta_norm_sqr > glitch_tolerance_sqr * glitch_scale {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
                suspect,
                distance: f64::INFINITY,
                is_interior: false,
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
    let mut delta = match params.fractal_type {
        FractalType::Julia => {
            // Julia: delta initial = dc (car z_0 = C + c pour Julia)
            dc_gmp.clone()
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
        
        // Apply perturbation formula: z_{n+1} = 2·Z_m·z_n + z_n² + c
        // In GMP: delta_{n+1} = 2·z_ref·delta + delta² + dc
        let mut delta_sq = delta.clone();
        delta_sq *= &delta;
        
        let mut linear_term = z_ref.clone();
        linear_term *= &delta;
        linear_term *= Float::with_val(prec, 2.0);
        
        let mut next_delta = linear_term;
        next_delta += &delta_sq;
        
        if !is_julia {
            // Mandelbrot: add dc term
            next_delta += dc_gmp;
        }
        // Julia: dc is already incorporated in initial delta
        
        // Handle special cases
        if is_burning_ship {
            // Burning Ship: z' = (|Re(z)|, |Im(z)|)² + c
            // For deep zooms, compute full orbit: z_curr = z_ref + delta
            let mut z_curr = z_ref.clone();
            z_curr += &delta;
            
            // Apply abs() to real and imaginary parts
            let re_abs = z_curr.real().clone().abs();
            let im_abs = z_curr.imag().clone().abs();
            let mut z_abs = Complex::with_val(prec, (re_abs, im_abs));
            z_abs *= z_abs.clone();
            
            // Add cref + dc
            let mut z_next = z_abs;
            z_next += &ref_orbit.cref_gmp;
            if !is_julia {
                z_next += dc_gmp;
            }
            
            // Calculate delta for next iteration: z_next - z_ref_next
            if (n + 1) >= effective_len {
                // Rebase: delta = z_next, reset n to 0
                delta = z_next;
                n = 0;
                continue;
            }
            
            let z_ref_next = match ref_orbit.get_z_ref_gmp(n + 1) {
                Some(z) => z,
                None => break,
            };
            delta = z_next - z_ref_next;
        } else if is_tricorn {
            // Tricorn: z' = conj(z)² + c
            let mut z_curr = z_ref.clone();
            z_curr += &delta;
            let z_conj = z_curr.clone().conj();
            let mut z_temp = z_conj.clone();
            z_temp *= &z_conj;
            z_temp += &ref_orbit.cref_gmp;
            if !is_julia {
                z_temp += dc_gmp;
            }
            
            if (n + 1) >= effective_len {
                delta = z_temp;
                n = 0;
                continue;
            }
            
            let z_ref_next = match ref_orbit.get_z_ref_gmp(n + 1) {
                Some(z) => z,
                None => break,
            };
            delta = z_temp - z_ref_next;
        } else {
            // Standard Mandelbrot/Julia: use computed next_delta
            delta = next_delta;
        }
        
        // Check bailout
        let mut z_curr = z_ref.clone();
        z_curr += &delta;
        let z_curr_norm_sqr = complex_norm_sqr(&z_curr, prec);
        
        if !z_curr.real().is_finite() || !z_curr.imag().is_finite() {
            return DeltaResult {
                iteration: n,
                z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
                glitched: true,
                suspect: false,
                distance: f64::INFINITY,
                is_interior: false,
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
            };
        }
        
        // Check for rebasing: when |Z_m + z_n| < |z_n|
        let delta_norm_sqr = complex_norm_sqr(&delta, prec);
        if z_curr_norm_sqr > Float::with_val(prec, 0.0) 
            && delta_norm_sqr > Float::with_val(prec, 0.0) 
            && z_curr_norm_sqr < delta_norm_sqr {
            // Rebasing: replace z_n with Z_m + z_n and reset m to 0
            delta = z_curr;
            n = 0;
            continue;
        }
        
        n += 1;
    }
    
    // Final result
    let final_index = n.min(effective_len.saturating_sub(1));
    let z_ref = match ref_orbit.get_z_ref_gmp(final_index) {
        Some(z) => z,
        None => ref_orbit.z_ref_gmp.last().unwrap(),
    };
    let mut z_curr = z_ref.clone();
    z_curr += &delta;
    
    DeltaResult {
        iteration: final_index,
        z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
        glitched: false,
        suspect: false,
        distance: f64::INFINITY,
        is_interior: false,
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
                        // For Burning Ship, we'd need to handle sign transformation
                        delta_dual
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
                            // Apply non-conformal BLA
                            let dc_vec = (dc_dual.value.re, dc_dual.value.im);
                            let linear_vec = node.a.mul_vector(delta_vec.0, delta_vec.1);
                            let dc_term_vec = node.b.mul_vector(dc_vec.0, dc_vec.1);
                            let next_vec = (linear_vec.0 + dc_term_vec.0, linear_vec.1 + dc_term_vec.1);
                            
                            // Update value (simplified - full dual propagation through matrices would be more complex)
                            delta_dual.value = Complex64::new(next_vec.0, next_vec.1);
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
                //
                // For non-complex-analytic formulas (like Burning Ship), you can use dual numbers
                // with two dual parts, for each of the real and imaginary components. At the end they
                // can be combined into a Jacobian matrix and used in the (directional) distance estimate
                // formula for general iterations.
                //
                // Note: Current implementation is simplified - dual propagation for Burning Ship does not
                // fully implement the Jacobian matrix approach. The absolute value operations create
                // discontinuities that require special handling in the dual propagation. A full implementation
                // would require propagating duals through the absolute value operations and combining into
                // a Jacobian matrix for distance estimation.
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let z_ref_dual = ExtendedDualComplex::from_complex(z_ref);
                let z_curr_dual = z_ref_dual.add(delta_dual);
                // For Burning Ship, we'd compute |z|² with duals
                // Simplified: use standard path (duals not fully propagated through abs() operations)
                let z_curr = z_curr_dual.value;
                let re_abs = z_curr.re.abs();
                let im_abs = z_curr.im.abs();
                let z_abs_sq = Complex64::new(re_abs * re_abs - im_abs * im_abs, 2.0 * re_abs * im_abs);
                let z_next = z_abs_sq + ref_orbit.cref + dc_dual.value;
                // Hybrid BLA: rebase when reaching end of effective orbit for current phase
                let effective_len = ref_orbit.effective_len() as u32;
                if (n + 1) >= effective_len {
                    // Rebase: delta_dual = z_next (which is z_curr for next iteration), reset n to 0
                    // In Hybrid BLA, this switches to the reference for the current phase
                    delta_dual = ExtendedDualComplex::from_complex(z_next);
                    n = 0;
                    continue;
                }
                let z_ref_next = ref_orbit.get_z_ref_f64(n + 1).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                delta_dual = ExtendedDualComplex::from_complex(z_next - z_ref_next);
                n += 1;
            } else if is_multibrot {
                // Multibrot: z^d + c
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let z_ref_dual = ExtendedDualComplex::from_complex(z_ref);
                let z_curr_dual = z_ref_dual.add(delta_dual);
                let d = multibrot_power;
                let z_norm = z_curr_dual.value.norm();
                if z_norm > 1e-15 {
                    // Simplified: use standard computation
                    let z_pow = z_curr_dual.value.powf(d);
                    let z_next = z_pow + ref_orbit.cref + dc_dual.value;
                    // Hybrid BLA: rebase when reaching end of effective orbit for current phase
                    let effective_len = ref_orbit.effective_len() as u32;
                    if (n + 1) >= effective_len {
                        // Rebase: switch to reference for current phase
                        delta_dual = ExtendedDualComplex::from_complex(z_next);
                        n = 0;
                        continue;
                    }
                    let z_ref_next = ref_orbit.get_z_ref_f64(n + 1).unwrap_or_else(|| {
                        ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                    });
                    delta_dual = ExtendedDualComplex::from_complex(z_next - z_ref_next);
                } else {
                    delta_dual = dc_dual;
                }
                n += 1;
            } else if is_tricorn {
                // Tricorn: z' = conj(z)² + c
                // Use non-conformal matrices for perturbation
                //
                // For non-complex-analytic formulas (like Mandelbar/Tricorn), you can use dual numbers
                // with two dual parts, for each of the real and imaginary components. At the end they
                // can be combined into a Jacobian matrix and used in the (directional) distance estimate
                // formula for general iterations.
                //
                // Note: Current implementation is simplified - dual propagation for Tricorn does not fully
                // implement the Jacobian matrix approach. The duals are not correctly propagated through
                // the non-conformal matrix operations. A full implementation would require:
                // 1. Propagating dual_re and dual_im through 2×2 real matrices (Jacobian)
                // 2. Combining the results into a proper distance estimate using the Jacobian norm
                let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let coeffs = crate::fractal::perturbation::nonconformal::compute_tricorn_bla_coefficients(z_ref);
                let delta_vec = (delta_dual.value.re, delta_dual.value.im);
                let linear_vec = coeffs.a.mul_vector(delta_vec.0, delta_vec.1);
                // Nonlinear term: conj(δ)² = (δ_re - i·δ_im)² = (δ_re² - δ_im²) - i·2·δ_re·δ_im
                let delta_conj_sq_re = delta_vec.0 * delta_vec.0 - delta_vec.1 * delta_vec.1;
                let delta_conj_sq_im = -2.0 * delta_vec.0 * delta_vec.1;
                let dc_vec = (dc_dual.value.re, dc_dual.value.im);
                let dc_term_vec = coeffs.b.mul_vector(dc_vec.0, dc_vec.1);
                let next_vec = (
                    linear_vec.0 + delta_conj_sq_re + dc_term_vec.0,
                    linear_vec.1 + delta_conj_sq_im + dc_term_vec.1,
                );
                // Update value, keep duals (simplified - full implementation would propagate duals
                // through 2×2 real matrices and combine into Jacobian for distance estimation)
                delta_dual.value = Complex64::new(next_vec.0, next_vec.1);
                n += 1;
            } else {
                // Mandelbrot/Julia: z² + c
                // Use dual propagation: (z_ref + delta)² = z_ref² + 2·z_ref·delta + delta²
                let z_ref = ref_orbit.z_ref_f64[n as usize];
                let z_ref_dual = ExtendedDualComplex::from_complex(z_ref);
                let z_curr_dual = z_ref_dual.add(delta_dual);
                
                // z² with dual propagation
                let z_sq_dual = z_curr_dual.square();
                
                // Add c (cref + dc)
                let c_dual = if is_julia {
                    ExtendedDualComplex::from_complex(params.seed)
                } else {
                    let cref_dual = ExtendedDualComplex::from_complex(ref_orbit.cref);
                    cref_dual.add(dc_dual)
                };
                
                let z_next_dual = z_sq_dual.add(c_dual);
                
                // Calculate delta for next iteration: z_next - z_ref_next
                // Hybrid BLA: if we reach the end of the effective orbit, rebase instead of breaking
                let effective_len = ref_orbit.effective_len() as u32;
                if (n + 1) >= effective_len {
                    // Rebase: delta_dual = z_next_dual (which is z_curr for next iteration), reset n to 0
                    // In Hybrid BLA, this switches to the reference for the current phase
                    delta_dual = z_next_dual;
                    n = 0;
                    continue;
                }
                let z_ref_next = ref_orbit.get_z_ref_f64(n + 1).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                let z_ref_next_dual = ExtendedDualComplex::from_complex(z_ref_next);
                delta_dual = z_next_dual.add(z_ref_next_dual.scale(-1.0));
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
            };
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{AlgorithmMode, FractalParams, FractalType};
    use crate::fractal::perturbation::orbit::ReferenceOrbit;
    use crate::fractal::perturbation::bla::BlaTable;

    fn test_params() -> FractalParams {
        FractalParams {
            width: 100,
            height: 100,
            center_x: 0.0,
            center_y: 0.0,
            span_x: 4.0,
            span_y: 4.0,
            seed: num_complex::Complex64::new(0.0, 0.0),
            iteration_max: 100,
            bailout: 4.0,
            fractal_type: FractalType::Mandelbrot,
            color_mode: 0,
            color_repeat: 2,
            use_gmp: false,
            precision_bits: 192,
            algorithm_mode: AlgorithmMode::Perturbation,
            bla_threshold: 1e-6,
            bla_validity_scale: 1.0,
            glitch_tolerance: 1e-4,
            series_order: 2,
            series_threshold: 1e-6,
            series_error_tolerance: 1e-9,
            glitch_neighbor_pass: false,
            series_standalone: false,
            max_secondary_refs: 3,
            min_glitch_cluster_size: 100,
            multibrot_power: 2.5,
            lyapunov_preset: Default::default(),
            lyapunov_sequence: Vec::new(),
        }
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
        };
        assert_eq!(result.iteration, 10);
        assert_eq!(result.distance, f64::INFINITY);
        assert_eq!(result.is_interior, false);
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

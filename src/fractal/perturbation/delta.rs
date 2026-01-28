use num_complex::Complex64;

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::perturbation::bla::BlaTable;
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::types::ComplexExp;
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
/// Uses high precision reference orbit Zm and low precision deltas zn relative to it.
/// Pixel orbit is Zm+zn, C+c where c is the pixel offset from center.
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
    let max_iter = params.iteration_max.min(ref_orbit.z_ref_f64.len().saturating_sub(1) as u32);
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

    while n < max_iter {
        let mut stepped = false;
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
                            // Apply non-conformal BLA: next_vec = A·delta_vec + B·dc_vec
                            let dc_approx = dc.to_complex64_approx();
                            let dc_vec = (dc_approx.re, dc_approx.im);
                            
                            // Linear term: A·delta_vec
                            let linear_vec = node.a.mul_vector(delta_vec.0, delta_vec.1);
                            
                            // Add dc term: B·dc_vec (B is identity for Tricorn)
                            let dc_term_vec = node.b.mul_vector(dc_vec.0, dc_vec.1);
                            
                            let next_vec = (linear_vec.0 + dc_term_vec.0, linear_vec.1 + dc_term_vec.1);
                            
                            // Convert back to ComplexExp
                            delta = ComplexExp::from_complex64(Complex64::new(next_vec.0, next_vec.1));
                            delta_norm_sqr = delta.norm_sqr_approx();
                            n += 1u32 << level;
                            stepped = true;
                            break;
                        }
                    }
                }
            }
        }

        // BLA table lookup (Section 3.4 of deep zoom theory):
        // Find the BLA starting from iteration m that has the largest skip l satisfying |z| < R.
        // Le BLA est maintenant supporté pour Burning Ship quand le quadrant est stable
        if !stepped && !bla_table.levels.is_empty() {
            delta_norm_sqr = delta.norm_sqr_approx();
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
                if delta_norm_sqr < node.validity_radius * node.validity_radius {
                    // For Burning Ship, apply sign transformation to delta
                    let work_delta = if is_burning_ship && node.burning_ship_valid {
                        delta.mul_signed(node.sign_re, node.sign_im)
                    } else {
                        delta
                    };

                    if should_use_series(series_config, delta_norm_sqr, node.validity_radius) {
                        let delta_sq = work_delta.mul(work_delta);
                        let mut next_delta = work_delta.mul_complex64(node.a);
                        if !is_julia {
                            next_delta = next_delta.add(dc.mul_complex64(node.b));
                        }
                        // Terme quadratique (ordre 2)
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
                        let mut next_delta = work_delta.mul_complex64(node.a);
                        if !is_julia {
                            next_delta = next_delta.add(dc.mul_complex64(node.b));
                        }
                        delta = next_delta;
                    }
                    delta_norm_sqr = delta.norm_sqr_approx();
                    n += 1u32 << level;
                    stepped = true;
                    break;
                }
            }
        }

        if !stepped {
            if is_burning_ship {
                let z_ref = ref_orbit.z_ref_f64[n as usize];
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
                    let next_index = (n + 1) as usize;
                    if next_index >= ref_orbit.z_ref_f64.len() {
                        break;
                    }
                    let z_ref_next = ref_orbit.z_ref_f64[next_index];
                    delta = ComplexExp::from_complex64(z_next - z_ref_next);
                }
                delta_norm_sqr = delta.norm_sqr_approx();
                n += 1;
            } else if is_multibrot {
                // Multibrot: z^d + c
                // δ' ≈ d·z_ref^(d-1)·δ + d(d-1)/2·z_ref^(d-2)·δ²
                let z_ref = ref_orbit.z_ref_f64[n as usize];
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
                let z_ref = ref_orbit.z_ref_f64[n as usize];
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
                let z_ref_hp = ref_orbit.z_ref[n as usize];
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
                // Standard perturbation iteration (Section 2 of deep zoom theory):
                // Pixel orbit Zm+zn, C+c. Perturbation formula: zn+1 = 2·Zm·zn + zn² + c
                // For Mandelbrot/Julia: z² + c
                // δ' = 2·z_ref·δ + δ² + dc (for Mandelbrot) or δ' = 2·z_ref·δ + δ² (for Julia)
                let z_ref = ref_orbit.z_ref_f64[n as usize];
                let linear = delta.mul_complex64(z_ref * 2.0);
                let nonlinear = delta.mul(delta);
                if is_julia {
                    delta = linear.add(nonlinear);
                } else {
                    delta = linear.add(nonlinear).add(dc);
                }
                delta_norm_sqr = delta.norm_sqr_approx();
                n += 1;
            }
        }

        // Note: We no longer break here when reaching the end of the reference orbit.
        // Instead, we rebase (see rebasing logic below) to reuse the same reference orbit in a loop.
        // This is Zhuoran's improvement that allows using a single reference orbit for very deep zooms.

        // Check if we've reached the end of the reference orbit (Zhuoran's improvement)
        // If so, rebase to reuse the same reference orbit in a loop
        if n >= ref_orbit.z_ref_f64.len() as u32 {
            // Calculate z_curr using the last valid reference point
            let last_idx = ref_orbit.z_ref_f64.len().saturating_sub(1);
            let z_ref = ref_orbit.z_ref_f64[last_idx];
            let delta_approx = delta.to_complex64_approx();
            let z_curr = z_ref + delta_approx;
            
            // Rebase: delta = z_curr, reset n to 0 to reuse the reference orbit
            // This is Zhuoran's method that allows using a single reference orbit for very deep zooms
            delta = ComplexExp::from_complex64(z_curr);
            n = 0;
            continue;
        }

        // For high-precision path, use ComplexExp for z_curr calculation
        let (z_curr, z_ref_norm_sqr) = if use_high_precision && !is_burning_ship && !is_multibrot && !is_tricorn {
            let z_ref_hp = ref_orbit.z_ref[n as usize];
            let z_curr_hp = z_ref_hp.add(delta);
            let z_curr = z_curr_hp.to_complex64_approx();
            let z_ref_norm = z_ref_hp.norm_sqr_approx();
            (z_curr, z_ref_norm)
        } else {
            let z_ref = ref_orbit.z_ref_f64[n as usize];
            let delta_approx = delta.to_complex64_approx();
            let z_curr = z_ref + delta_approx;
            (z_curr, z_ref.norm_sqr())
        };

        // Rebasing (Section 2.1 of deep zoom theory):
        // When |Zm + zn| < |zn|, replace zn with Zm + zn and reset the reference iteration count m to 0.
        // This helps maintain delta small relative to z_ref and reduces glitches.
        // In our implementation: z_curr = Zm + zn (where Zm is z_ref and zn is delta),
        // so we check if |z_curr| < |delta|, and if so, rebase delta = z_curr and reset n (m) to 0.
        let z_curr_norm_sqr = z_curr.norm_sqr();
        // Recalculate delta_norm_sqr if not already computed (e.g., after BLA step)
        if !stepped {
            delta_norm_sqr = delta.norm_sqr_approx();
        }
        let delta_norm_sqr_check = delta_norm_sqr;
        if z_curr_norm_sqr > 0.0 && delta_norm_sqr_check > 0.0 && z_curr_norm_sqr < delta_norm_sqr_check {
            // Perform rebasing: delta = z_curr (which is Zm + zn), reset n (m) to 0
            // This allows reusing the same reference orbit in a loop (Zhuoran's method)
            delta = ComplexExp::from_complex64(z_curr);
            n = 0;
            // Continue to next iteration with rebased delta
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

    let final_index = n.min(ref_orbit.z_ref_f64.len().saturating_sub(1) as u32);
    let z_curr = if use_high_precision && !is_burning_ship && !is_multibrot && !is_tricorn {
        let z_ref_hp = ref_orbit.z_ref[final_index as usize];
        z_ref_hp.add(delta).to_complex64_approx()
    } else {
        let z_ref = ref_orbit.z_ref_f64[final_index as usize];
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
    let max_iter = params.iteration_max.min(ref_orbit.z_ref_f64.len().saturating_sub(1) as u32);
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
                let z_ref = ref_orbit.z_ref_f64[n as usize];
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
                // Burning Ship: simplified (full implementation would need quadrant handling)
                let z_ref = ref_orbit.z_ref_f64[n as usize];
                let z_ref_dual = ExtendedDualComplex::from_complex(z_ref);
                let z_curr_dual = z_ref_dual.add(delta_dual);
                // For Burning Ship, we'd compute |z|² with duals
                // Simplified: use standard path
                let z_curr = z_curr_dual.value;
                let re_abs = z_curr.re.abs();
                let im_abs = z_curr.im.abs();
                let z_abs_sq = Complex64::new(re_abs * re_abs - im_abs * im_abs, 2.0 * re_abs * im_abs);
                let z_next = z_abs_sq + ref_orbit.cref + dc_dual.value;
                // Zhuoran's improvement: rebase when reaching end of reference orbit
                if ((n + 1) as usize) >= ref_orbit.z_ref_f64.len() {
                    // Rebase: delta_dual = z_next (which is z_curr for next iteration), reset n to 0
                    delta_dual = ExtendedDualComplex::from_complex(z_next);
                    n = 0;
                    continue;
                }
                let z_ref_next = ref_orbit.z_ref_f64[(n + 1) as usize];
                delta_dual = ExtendedDualComplex::from_complex(z_next - z_ref_next);
                n += 1;
            } else if is_multibrot {
                // Multibrot: z^d + c
                let z_ref = ref_orbit.z_ref_f64[n as usize];
                let z_ref_dual = ExtendedDualComplex::from_complex(z_ref);
                let z_curr_dual = z_ref_dual.add(delta_dual);
                let d = multibrot_power;
                let z_norm = z_curr_dual.value.norm();
                if z_norm > 1e-15 {
                    // Simplified: use standard computation
                    let z_pow = z_curr_dual.value.powf(d);
                    let z_next = z_pow + ref_orbit.cref + dc_dual.value;
                    // Zhuoran's improvement: rebase when reaching end of reference orbit
                    if ((n + 1) as usize) >= ref_orbit.z_ref_f64.len() {
                        delta_dual = ExtendedDualComplex::from_complex(z_next);
                        n = 0;
                        continue;
                    }
                    let z_ref_next = ref_orbit.z_ref_f64[(n + 1) as usize];
                    delta_dual = ExtendedDualComplex::from_complex(z_next - z_ref_next);
                } else {
                    delta_dual = dc_dual;
                }
                n += 1;
            } else if is_tricorn {
                // Tricorn: z' = conj(z)² + c
                // Use non-conformal matrices for perturbation
                // Note: Dual propagation for Tricorn is simplified - full implementation would
                // require propagating duals through matrix operations
                let z_ref = ref_orbit.z_ref_f64[n as usize];
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
                // Update value, keep duals (simplified - full propagation would update duals through matrices)
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
                // Zhuoran's improvement: if we reach the end of the reference orbit, rebase instead of breaking
                if ((n + 1) as usize) >= ref_orbit.z_ref_f64.len() {
                    // Rebase: delta_dual = z_next_dual (which is z_curr for next iteration), reset n to 0
                    // This allows reusing the same reference orbit in a loop
                    delta_dual = z_next_dual;
                    n = 0;
                    continue;
                }
                let z_ref_next = ref_orbit.z_ref_f64[(n + 1) as usize];
                let z_ref_next_dual = ExtendedDualComplex::from_complex(z_ref_next);
                delta_dual = z_next_dual.add(z_ref_next_dual.scale(-1.0));
                n += 1;
            }
        }
        
        // Check bailout
        let z_ref = ref_orbit.z_ref_f64[n as usize];
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
    let final_index = n.min(ref_orbit.z_ref_f64.len().saturating_sub(1) as u32);
    let z_ref = ref_orbit.z_ref_f64[final_index as usize];
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

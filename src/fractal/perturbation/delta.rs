use num_complex::Complex64;

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::perturbation::bla::BlaTable;
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::types::ComplexExp;
use crate::fractal::perturbation::series::{SeriesConfig, should_use_series, estimate_series_error};

pub struct DeltaResult {
    pub iteration: u32,
    pub z_final: Complex64,
    pub glitched: bool,
    pub suspect: bool,
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

pub fn iterate_pixel(
    params: &FractalParams,
    ref_orbit: &ReferenceOrbit,
    bla_table: &BlaTable,
    delta0: ComplexExp,
    dc: ComplexExp,
) -> DeltaResult {
    let mut n = 0u32;
    let mut delta = delta0;
    let max_iter = params.iteration_max.min(ref_orbit.z_ref.len().saturating_sub(1) as u32);
    let bailout_sqr = params.bailout * params.bailout;
    
    // Calculer le pixel_size pour la tolérance adaptative
    let pixel_size = params.span_x / params.width as f64;
    let adaptive_tolerance = compute_adaptive_glitch_tolerance(pixel_size, params.glitch_tolerance);
    let glitch_tolerance_sqr = adaptive_tolerance * adaptive_tolerance;
    let is_julia = params.fractal_type == FractalType::Julia;
    let is_burning_ship = params.fractal_type == FractalType::BurningShip;
    let is_multibrot = params.fractal_type == FractalType::Multibrot;
    let multibrot_power = params.multibrot_power;
    let series_config = SeriesConfig::from_params(params);
    let mut suspect = false;

    while n < max_iter {
        let mut stepped = false;
        let mut delta_norm_sqr = 0.0;

        // Le BLA est maintenant supporté pour Burning Ship quand le quadrant est stable
        if !bla_table.levels.is_empty() {
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
                    if should_use_series(series_config, delta_norm_sqr, node.validity_radius) {
                        let delta_sq = delta.mul(delta);
                        let mut next_delta = delta.mul_complex64(node.a);
                        if !is_julia {
                            next_delta = next_delta.add(dc.mul_complex64(node.b));
                        }
                        // Terme quadratique (ordre 2)
                        next_delta = next_delta.add(delta_sq.mul_complex64(node.c));
                        
                        // Termes d'ordre supérieur pour Multibrot (ordre 4-6)
                        if series_config.order >= 4 && is_multibrot {
                            // Terme cubique δ³
                            let delta_cube = delta_sq.mul(delta);
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
                        let mut next_delta = delta.mul_complex64(node.a);
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
            let z_ref = ref_orbit.z_ref[n as usize];
            if is_burning_ship {
                let z_curr = z_ref + delta.to_complex64_approx();
                let re = z_curr.re.abs();
                let im = z_curr.im.abs();
                let mut z_temp = Complex64::new(re, im);
                z_temp = z_temp * z_temp;
                let c_pixel = ref_orbit.cref + dc.to_complex64_approx();
                let z_next = z_temp + c_pixel;
                let next_index = (n + 1) as usize;
                if next_index >= ref_orbit.z_ref.len() {
                    break;
                }
                let z_ref_next = ref_orbit.z_ref[next_index];
                delta = ComplexExp::from_complex64(z_next - z_ref_next);
                n += 1;
            } else if is_multibrot {
                // Multibrot: z^d + c
                // δ' ≈ d·z_ref^(d-1)·δ + d(d-1)/2·z_ref^(d-2)·δ²
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
            } else {
                // Standard Mandelbrot/Julia: z² + c
                // δ' = 2·z_ref·δ + δ²
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

        if n >= ref_orbit.z_ref.len() as u32 {
            break;
        }
        let z_ref = ref_orbit.z_ref[n as usize];
        let delta_approx = delta.to_complex64_approx();
        let z_curr = z_ref + delta_approx;
        if !z_curr.re.is_finite() || !z_curr.im.is_finite() {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
                suspect,
            };
        }
        if z_curr.norm_sqr() > bailout_sqr {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: false,
                suspect,
            };
        }

        let z_ref_norm_sqr = z_ref.norm_sqr();
        let glitch_scale = z_ref_norm_sqr + 1.0;
        if !delta_norm_sqr.is_finite() || delta_norm_sqr > glitch_tolerance_sqr * glitch_scale {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
                suspect,
            };
        }
    }

    let final_index = n.min(ref_orbit.z_ref.len().saturating_sub(1) as u32);
    let z_ref = ref_orbit.z_ref[final_index as usize];
    let z_curr = z_ref + delta.to_complex64_approx();
    DeltaResult {
        iteration: final_index,
        z_final: z_curr,
        glitched: false,
        suspect,
    }
}

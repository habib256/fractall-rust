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
    let glitch_tolerance_sqr = params.glitch_tolerance * params.glitch_tolerance;
    let is_julia = params.fractal_type == FractalType::Julia;
    let is_burning_ship = params.fractal_type == FractalType::BurningShip;
    let series_config = SeriesConfig::from_params(params);
    let mut suspect = false;

    while n < max_iter {
        let mut stepped = false;
        let mut delta_norm_sqr = 0.0;

        if !bla_table.levels.is_empty() && !is_burning_ship {
            delta_norm_sqr = delta.norm_sqr_approx();
            for level in (0..bla_table.levels.len()).rev() {
                let level_nodes = &bla_table.levels[level];
                if (n as usize) >= level_nodes.len() {
                    continue;
                }
                let node = &level_nodes[n as usize];
                if delta_norm_sqr < node.validity_radius * node.validity_radius {
                    if should_use_series(series_config, delta_norm_sqr, node.validity_radius) {
                        let delta_sq = delta.mul(delta);
                        let mut next_delta = delta.mul_complex64(node.a);
                        if !is_julia {
                            next_delta = next_delta.add(dc.mul_complex64(node.b));
                        }
                        next_delta = next_delta.add(delta_sq.mul_complex64(node.c));
                        let err = estimate_series_error(delta_norm_sqr, series_config.order);
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
            } else {
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

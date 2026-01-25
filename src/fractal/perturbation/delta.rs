use num_complex::Complex64;

use crate::fractal::FractalParams;
use crate::fractal::perturbation::bla::BlaTable;
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::types::ComplexExp;

pub struct DeltaResult {
    pub iteration: u32,
    pub z_final: Complex64,
    pub glitched: bool,
}

pub fn iterate_pixel(
    params: &FractalParams,
    ref_orbit: &ReferenceOrbit,
    bla_table: &BlaTable,
    dc: ComplexExp,
) -> DeltaResult {
    let mut n = 0u32;
    let mut delta = ComplexExp::zero();
    let max_iter = params.iteration_max.min(ref_orbit.z_ref.len().saturating_sub(1) as u32);
    let bailout_sqr = params.bailout * params.bailout;

    while n < max_iter {
        let mut stepped = false;

        if !bla_table.levels.is_empty() {
            for level in (0..bla_table.levels.len()).rev() {
                let level_nodes = &bla_table.levels[level];
                if (n as usize) >= level_nodes.len() {
                    continue;
                }
                let node = &level_nodes[n as usize];
                if delta.norm_sqr_approx().sqrt() < node.validity_radius {
                    delta = delta.mul_complex64(node.a).add(dc.mul_complex64(node.b));
                    n += 1u32 << level;
                    stepped = true;
                    break;
                }
            }
        }

        if !stepped {
            let z_ref = ref_orbit.z_ref[n as usize];
            let linear = delta.mul_complex64(z_ref * 2.0);
            let nonlinear = delta.mul(delta);
            delta = linear.add(nonlinear).add(dc);
            n += 1;
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
            };
        }
        if z_curr.norm_sqr() > bailout_sqr {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: false,
            };
        }

        let z_ref_norm = z_ref.norm();
        if z_ref_norm > 0.0 && delta.norm_sqr_approx().sqrt() > params.glitch_tolerance * z_ref_norm {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
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
    }
}

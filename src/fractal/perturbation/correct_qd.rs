//! Correction **quad-double** des pixels au-delà du double-double (TODO
//! G14.1 h) : maillon entre le dd sans BLA et le GMP par pixel.
//!
//! dc calculé EXACTEMENT en GMP (`DcGmpContext::compute_dc_ref`, le même que
//! les corrections GMP : rotation/K, jitter AA, `centre_vue − cref` compris),
//! mis à l'échelle 2^s par décalage exact, puis arrondi en qd. Orbite qd
//! rejouée depuis les points de contrôle GMP (`orbit_qd`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use num_complex::Complex64;
use rayon::prelude::*;

use super::orbit::ReferenceOrbitCache;
use super::orbit_qd::build_qd_orbit;
use super::qd::ComplexQd;
use super::sampling::DcGmpContext;
use crate::fractal::bytecode::pixel_loop::PixelLoopLimits;
use crate::fractal::bytecode::pixel_loop_qd::{iterate_pixel_qd_mandelbrot, qd_scale_for};
use crate::fractal::FractalParams;

/// Résultat qd d'un pixel : `(index, itérations, z_final, fiable)`. Un pixel
/// non fiable (borne qd emballée, ou hors domaine d'échelle du qd) n'a pas de
/// valeur exploitable : l'appelant le passe au GMP (sous plafond).
pub(super) type QdFix = (usize, u32, Complex64, bool);

/// Corrige `indices` en qd. `None` si annulé ou si l'orbite qd n'a pu être
/// bâtie.
pub(super) fn correct_pixels_qd(
    params: &FractalParams,
    cache: &ReferenceOrbitCache,
    cancel: &AtomicBool,
    indices: &[usize],
) -> Option<Vec<QdFix>> {
    let t0 = Instant::now();
    let orbit = &cache.orbit;
    let n = if !orbit.z_ref.is_empty() {
        orbit.z_ref.len()
    } else {
        orbit.z_ref_dd.len()
    };
    let zq = build_qd_orbit(orbit, n)?;
    let t_orbit = t0.elapsed();
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let t1 = Instant::now();
    let prec = super::compute_perturbation_precision_bits(params);
    let dc_ctx = DcGmpContext::with_reference(params, prec, cache);
    let limits = PixelLoopLimits::from(params);
    let runaway = crate::fractal::bytecode::reliability::reliability_runaway() as f32;
    let width = params.width.max(1) as usize;
    let res: Vec<QdFix> = indices
        .par_iter()
        .map(|&idx| {
            let fail = (idx, 0, Complex64::new(0.0, 0.0), false);
            if cancel.load(Ordering::Relaxed) {
                return fail;
            }
            let dc = dc_ctx.compute_dc_ref(idx % width, idx / width);
            let e = [dc.real(), dc.imag()]
                .iter()
                .filter_map(|f| f.get_exp())
                .max();
            let Some(s) = e.and_then(|e| qd_scale_for(e as i64)) else {
                return fail;
            };
            let mut dc_s = dc;
            dc_s <<= s as u32;
            let mut scratch = rug::Float::new(prec.max(64));
            let dc_q = ComplexQd::from_complex(&dc_s, &mut scratch);
            let (r, bound) =
                iterate_pixel_qd_mandelbrot(orbit, &zq.z, dc_q, s, zq.prec, limits, true);
            // Même critère d'évasion relatif (opt-in) que le dd.
            let bound = if r.iteration < limits.iteration_max {
                crate::fractal::bytecode::pixel_loop_dd::dd_escape_bound(bound, r.z_final)
            } else {
                bound
            };
            (idx, r.iteration, r.z_final, bound < runaway)
        })
        .collect();
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    if super::perf_enabled() {
        eprintln!(
            "[RELIABILITY-QD] px={} qd_runaway={} orbit_prec={} orbit={:.3}s pixels={:.3}s",
            indices.len(),
            res.iter().filter(|r| !r.3).count(),
            zq.prec,
            t_orbit.as_secs_f64(),
            t1.elapsed().as_secs_f64()
        );
    }
    Some(res)
}

/// Étape qd de l'escalade : `unreliable` = pixels dont la borne dd s'emballe
/// (second essai sans BLA compris). Si le tier qd est actif
/// (`FRACTALL_QD`) et que leur nombre passe le plafond qd, les pixels que le
/// qd rend fiables reçoivent leur valeur ; renvoie les pixels RESTANTS (pour
/// la politique GMP). `None` = annulation.
pub(super) fn qd_stage(
    params: &FractalParams,
    cache: &ReferenceOrbitCache,
    cancel: &AtomicBool,
    unreliable: Vec<usize>,
    frame_pixels: usize,
    iterations: &mut [u32],
    zs: &mut [Complex64],
) -> Option<Vec<usize>> {
    use crate::fractal::bytecode::pixel_loop_qd as q;
    // Graine Mandelbrot non nulle (z₀ ≠ 0, métadonnées éditées à la main) :
    // l'orbite qd de repli et le rebase de la boucle supposent Z[0] = 0.
    if unreliable.is_empty()
        || params.seed.re != 0.0
        || params.seed.im != 0.0
        || !q::qd_enabled()
        || !q::qd_escalation_allowed(unreliable.len(), frame_pixels, q::qd_max_fraction())
    {
        return Some(unreliable);
    }
    let Some(fixed) = correct_pixels_qd(params, cache, cancel, &unreliable) else {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        return Some(unreliable);
    };
    let mut rest = Vec::new();
    for (idx, it, z, ok) in fixed {
        if ok {
            iterations[idx] = it;
            zs[idx] = z;
        } else {
            rest.push(idx);
        }
    }
    Some(rest)
}

//! Pixel loop unifié extended-precision (ComplexExp) pour deep zoom > 1e15.
//!
//! Mirror de `pixel_loop.rs::iterate_pixel_unified` mais avec `delta` en
//! `ComplexExp` (mantissa + exponent) au lieu de `Complex64`. La reference
//! orbit reste en `Complex64` (les valeurs sont O(1), pas d'underflow).
//!
//! Choix d'implémentation pour la précision :
//! - `delta` et `dc` : `ComplexExp` (préservent les magnitudes < 1e-308)
//! - `z_ref` : `Complex64` (valeurs bornées, suffit)
//! - `BLA mat2 A, B` : `Mat2<f64>` (coefficients O(1) à profondeur raisonnable)
//! - Bailout check : convertit `z_ref + delta_approx` en `f64` (Z domine)
//! - Rebase check : compare via `f64` (les magnitudes après cancellation
//!   restent dans la range f64 normale dès qu'on est en zone d'évasion)
//!
//! Limitation : la BLA est construite en f64 ; à très deep zoom (> ~1e150),
//! les coefficients A peuvent devenir mal conditionnés. Pour ces zooms,
//! l'orbite référence f64 underflow aussi → fallback sur le path GMP legacy
//! reste actif (gated par `pixel_size < 1e-150` dans le dispatcher).

use num_complex::Complex64;

use super::bla_dual::BlaTableUnified;
use super::delta_form::DeltaStateExp;
use super::{Formula, Op, Phase};
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};

/// Phase Mandelbrot = exactement [Sqr, Add]. Détection au runtime pour
/// activer le path spécialisé qui évite le dispatch opcode et l'update
/// wasted de `z_ref` dans DeltaStateExp.
#[inline]
fn phase_is_mandelbrot(phase: &Phase) -> bool {
    phase.ops.len() == 2
        && matches!(phase.ops[0], Op::Sqr)
        && matches!(phase.ops[1], Op::Add)
}

/// Résultat du pixel loop extended-precision.
pub struct UnifiedPixelResultExp {
    pub iteration: u32,
    pub z_final: Complex64,
    #[allow(dead_code)]
    pub rebase_count: u32,
    #[allow(dead_code)]
    pub bla_steps: u32,
}

/// Pixel loop unifié extended-precision. Signature mirror de
/// `iterate_pixel_unified` (pixel_loop.rs) mais accepte `dc` et
/// `delta_initial` en `ComplexExp`.
pub fn iterate_pixel_unified_exp(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    formula: &Formula,
    c_ref: Complex64,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
) -> UnifiedPixelResultExp {
    assert_eq!(
        formula.phases.len(),
        1,
        "Multi-phase pas encore supporté dans pixel_loop_exp"
    );
    let phase = &formula.phases[0];
    iterate_pixel_unified_exp_single_phase(
        ref_orbit,
        bla,
        phase,
        c_ref,
        dc,
        delta_initial,
        iteration_max,
        bailout,
    )
}

fn iterate_pixel_unified_exp_single_phase(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    phase: &Phase,
    c_ref: Complex64,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
) -> UnifiedPixelResultExp {
    // Hot path Mandelbrot : évite DeltaStateExp::step (dispatch opcode + update
    // wasted de z_ref) en inlinant `δ' = 2·Z·δ + δ² + dc`.
    if phase_is_mandelbrot(phase) {
        return iterate_pixel_unified_exp_mandelbrot(
            ref_orbit, bla, dc, delta_initial, iteration_max, bailout,
        );
    }
    let _ = c_ref;
    iterate_pixel_unified_exp_generic(
        ref_orbit, bla, phase, c_ref, dc, delta_initial, iteration_max, bailout,
    )
}

/// Mirror exact de la boucle générique mais avec le pas perturbation
/// hardcoded Mandelbrot. Reste à l'identique : BLA lookup, bailout check,
/// rebase F3 strict, period clamp.
fn iterate_pixel_unified_exp_mandelbrot(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
) -> UnifiedPixelResultExp {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return UnifiedPixelResultExp {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
        };
    }

    let mut delta = delta_initial;
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;
    const REDUCE_INTERVAL: u32 = 250;

    while n < iteration_max {
        let z_m = ref_orbit.z_ref_f64[m as usize];
        let delta_approx = delta.to_complex64_approx();
        let z_abs = z_m + delta_approx;
        if z_abs.norm_sqr() >= bailout_sqr {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: z_abs,
                rebase_count,
                bla_steps,
            };
        }

        let delta_norm_sqr = delta.norm_sqr_approx();
        if let Some(node) = bla.lookup(m as usize, delta_norm_sqr) {
            let new_n = n.saturating_add(node.l);
            let new_m = m.saturating_add(node.l);
            if new_n <= iteration_max && (new_m as usize) < ref_len {
                let a = node.a;
                let b = node.b;
                let new_re = (a.m00 * delta.re) + (a.m01 * delta.im)
                    + (b.m00 * dc.re) + (b.m01 * dc.im);
                let new_im = (a.m10 * delta.re) + (a.m11 * delta.im)
                    + (b.m10 * dc.re) + (b.m11 * dc.im);
                delta = ComplexExp { re: new_re, im: new_im };
                n = new_n;
                m = new_m;
                bla_steps += 1;
                if n % REDUCE_INTERVAL == 0 {
                    delta.reduce();
                }
                continue;
            }
        }

        // Pas perturbation Mandelbrot : δ' = 2·Z·δ + δ² + dc
        // Inline complet, pas de DeltaStateExp à allouer/lire/écrire.
        let two_z = z_m + z_m;
        let two_z_delta = delta.mul_complex64(two_z);
        let delta_sq = delta.mul(delta);
        delta = two_z_delta.add(delta_sq).add(dc);
        n += 1;
        m += 1;

        let delta_approx = delta.to_complex64_approx();
        if !delta_approx.re.is_finite() || !delta_approx.im.is_finite() {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: ref_orbit.z_ref_f64[m.min((ref_len - 1) as u32) as usize] + delta_approx,
                rebase_count,
                bla_steps,
            };
        }

        if n % REDUCE_INTERVAL == 0 {
            delta.reduce();
        }

        // Rebase F3 : |Z[m+1] + δ|² < |δ|² OR fin d'orbite. Pour les orbites
        // périodiques (centre intérieur), on cycle m via modulo au lieu de
        // rebaser — évite l'uniformisation pixel sur ces cas.
        let end_of_ref = (m as usize) + 1 >= ref_len;
        if end_of_ref {
            if let Some(m_wrapped) = ref_orbit.wrap_periodic(m) {
                m = m_wrapped;
            } else {
                let z_m_new = ref_orbit.z_ref_f64[(m as usize).min(ref_len - 1)];
                let new_re = FloatExp::from_f64(z_m_new.re) + delta.re;
                let new_im = FloatExp::from_f64(z_m_new.im) + delta.im;
                delta = ComplexExp { re: new_re, im: new_im };
                m = 0;
                rebase_count += 1;
            }
        } else {
            let z_m_new = ref_orbit.z_ref_f64[m as usize];
            let z_curr_re = FloatExp::from_f64(z_m_new.re) + delta.re;
            let z_curr_im = FloatExp::from_f64(z_m_new.im) + delta.im;
            let z_curr_norm_sqr = {
                let re = z_curr_re.to_f64();
                let im = z_curr_im.to_f64();
                re * re + im * im
            };
            let delta_norm_sqr_f64 = delta_approx.norm_sqr();
            if z_curr_norm_sqr < delta_norm_sqr_f64 && delta_norm_sqr_f64 > 0.0 {
                delta = ComplexExp { re: z_curr_re, im: z_curr_im };
                m = 0;
                rebase_count += 1;
            }
        }
    }

    let final_m = m.min((ref_len - 1) as u32);
    let delta_approx = delta.to_complex64_approx();
    UnifiedPixelResultExp {
        iteration: n,
        z_final: ref_orbit.z_ref_f64[final_m as usize] + delta_approx,
        rebase_count,
        bla_steps,
    }
}

fn iterate_pixel_unified_exp_generic(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    phase: &Phase,
    c_ref: Complex64,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
) -> UnifiedPixelResultExp {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return UnifiedPixelResultExp {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
        };
    }

    let mut delta = delta_initial;
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;

    // Reduce périodique (cf. rust-fractal-core) pour éviter la perte de
    // précision graduelle sur les mantissas après beaucoup d'itérations.
    const REDUCE_INTERVAL: u32 = 250;

    while n < iteration_max {
        let z_m = ref_orbit.z_ref_f64[m as usize];
        // Bailout : |Z + δ|² ≥ bailout² (les magnitudes après évasion sont
        // dans la range f64 normale donc to_complex64_approx est OK).
        let delta_approx = delta.to_complex64_approx();
        let z_abs = z_m + delta_approx;
        if z_abs.norm_sqr() >= bailout_sqr {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: z_abs,
                rebase_count,
                bla_steps,
            };
        }

        // BLA lookup. La norme du delta est calculée en FloatExp pour ne
        // pas underflow ; on prend l'approx f64 SEULEMENT pour comparer au
        // r² (qui est f64). Si delta est minuscule (exp très négatif), son
        // norm_sqr_approx vaudra 0.0 → toujours < r² → BLA toujours valide.
        // C'est cohérent : un delta très petit DOIT pouvoir être absorbé
        // par n'importe quel BLA.
        let delta_norm_sqr = delta.norm_sqr_approx();
        if let Some(node) = bla.lookup(m as usize, delta_norm_sqr) {
            let new_n = n.saturating_add(node.l);
            let new_m = m.saturating_add(node.l);
            if new_n <= iteration_max && (new_m as usize) < ref_len {
                // δ := A·δ + B·dc
                // A est mat2<f64>. δ peut être ComplexExp (très petit). On
                // multiplie chaque composante FloatExp par les coefs f64.
                let a = node.a;
                let b = node.b;
                let new_re = (a.m00 * delta.re) + (a.m01 * delta.im)
                    + (b.m00 * dc.re) + (b.m01 * dc.im);
                let new_im = (a.m10 * delta.re) + (a.m11 * delta.im)
                    + (b.m10 * dc.re) + (b.m11 * dc.im);
                delta = ComplexExp {
                    re: new_re,
                    im: new_im,
                };
                n = new_n;
                m = new_m;
                bla_steps += 1;
                if n % REDUCE_INTERVAL == 0 {
                    delta.reduce();
                }
                continue;
            }
        }

        // Pas perturbation via DeltaStateExp.
        let mut state = DeltaStateExp::new(z_m, delta);
        state.step(phase, c_ref, dc);
        delta = state.delta;
        n += 1;
        m += 1;

        // NaN / Inf check sur l'approx f64 (suffit pour détecter explosion).
        let delta_approx = delta.to_complex64_approx();
        if !delta_approx.re.is_finite() || !delta_approx.im.is_finite() {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: ref_orbit.z_ref_f64[m.min((ref_len - 1) as u32) as usize] + delta_approx,
                rebase_count,
                bla_steps,
            };
        }

        if n % REDUCE_INTERVAL == 0 {
            delta.reduce();
        }

        // Rebase F3 : |Z[m+1] + δ|² < |δ|² OR fin d'orbite. Pour orbites
        // périodiques, on cycle m via modulo (évite l'uniformisation pixel).
        let end_of_ref = (m as usize) + 1 >= ref_len;
        if end_of_ref {
            if let Some(m_wrapped) = ref_orbit.wrap_periodic(m) {
                m = m_wrapped;
            } else {
                // Clamp m si l'orbite est tronquée (escape, pas périodicité).
                let z_m_new = ref_orbit.z_ref_f64[(m as usize).min(ref_len - 1)];
                // δ_new = Z[m] + δ (promotion delta → absolu)
                let new_re = FloatExp::from_f64(z_m_new.re) + delta.re;
                let new_im = FloatExp::from_f64(z_m_new.im) + delta.im;
                delta = ComplexExp { re: new_re, im: new_im };
                m = 0;
                rebase_count += 1;
            }
        } else {
            let z_m_new = ref_orbit.z_ref_f64[m as usize];
            // z_curr = Z[m] + δ
            let z_curr_re = FloatExp::from_f64(z_m_new.re) + delta.re;
            let z_curr_im = FloatExp::from_f64(z_m_new.im) + delta.im;
            // Comparaison via to_f64 (les magnitudes en zone "interesting"
            // sont dans la range f64 normale pour le rebase).
            let z_curr_norm_sqr = {
                let re = z_curr_re.to_f64();
                let im = z_curr_im.to_f64();
                re * re + im * im
            };
            // Le delta peut être minuscule (exp < -300), to_f64 → 0.
            // Dans ce cas la condition `0 < 0` est false et on ne rebase pas
            // — ce qui est correct quand delta est trop petit pour avoir
            // dépassé z_curr.
            let delta_norm_sqr_f64 = delta_approx.norm_sqr();
            if z_curr_norm_sqr < delta_norm_sqr_f64 && delta_norm_sqr_f64 > 0.0 {
                delta = ComplexExp {
                    re: z_curr_re,
                    im: z_curr_im,
                };
                m = 0;
                rebase_count += 1;
            }
        }
    }

    let final_m = m.min((ref_len - 1) as u32);
    let delta_approx = delta.to_complex64_approx();
    UnifiedPixelResultExp {
        iteration: n,
        z_final: ref_orbit.z_ref_f64[final_m as usize] + delta_approx,
        rebase_count,
        bla_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::{build_bla_table_for_formula, compile_formula};
    use crate::fractal::perturbation::orbit::compute_reference_orbit;
    use crate::fractal::{default_params_for_type, AlgorithmMode, FractalType};

    fn make_ref_orbit(
        center_x: f64,
        center_y: f64,
        zoom: f64,
        iter_max: u32,
    ) -> ReferenceOrbit {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 100);
        params.center_x = center_x;
        params.center_y = center_y;
        let span_x = 4.0 / zoom;
        params.span_x = span_x;
        params.span_y = span_x * 100.0 / 160.0;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        let (orbit, _, _) = compute_reference_orbit(&params, None).expect("ref orbit");
        orbit
    }

    /// À zoom 1e6, le pixel loop ComplexExp doit donner les mêmes
    /// classifications que la version f64 (iterate_pixel_unified) et que
    /// l'itération directe.
    #[test]
    fn unified_exp_matches_f64_at_zoom_1e6() {
        let iter_max = 1500u32;
        let cx = -1.7693831791955;
        let cy = 0.004236847918736;
        let zoom = 1e6;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;
        let orbit = make_ref_orbit(cx, cy, zoom, iter_max);
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables =
            build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8).expect("BLA");
        let bla = &tables[0];

        let mut mismatches = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let dc_exp = ComplexExp::from_complex64(Complex64::new(dx, dy));
                let res = iterate_pixel_unified_exp(
                    &orbit,
                    bla,
                    &formula,
                    orbit.cref,
                    dc_exp,
                    ComplexExp::zero(),
                    iter_max,
                    4.0,
                );
                let escaped_exp = res.iteration < iter_max;

                let c_abs = Complex64::new(cx + dx, cy + dy);
                let mut z = Complex64::new(0.0, 0.0);
                let mut nn = 0u32;
                while nn < iter_max && z.norm_sqr() < 16.0 {
                    z = z * z + c_abs;
                    nn += 1;
                }
                let escaped_direct = nn < iter_max;
                if escaped_exp != escaped_direct {
                    mismatches += 1;
                }
                total += 1;
            }
        }
        assert_eq!(mismatches, 0, "exp doit classifier comme direct à zoom 1e6");
        let _ = total;
    }

    /// Test deep zoom : zoom 1e20. À cette profondeur, dc_f64 underflowerait
    /// (~ 1e-20 / 160 = 1e-22, encore OK f64 mais le delta après quelques
    /// itérations en δ² serait ~1e-44 puis 1e-88 etc.). L'exp version
    /// préserve la magnitude via l'exponent séparé.
    ///
    /// On ne peut pas comparer à f64 direct (qui pète à ce zoom), donc on
    /// vérifie juste que le pixel loop termine sans NaN/Inf et produit un
    /// résultat (escape ou intérieur) plausible.
    #[test]
    fn unified_exp_zoom_1e20_terminates() {
        let iter_max = 2000u32;
        let cx = -1.7693831791955;
        let cy = 0.004236847918736;
        // Pour zoom 1e20, on a besoin d'une vraie référence HP. Sans ça
        // le ref center fait perdre toute la précision avant même de
        // commencer. Skip le test si la fonction make_ref_orbit ne peut
        // pas générer une orbite suffisamment précise.
        let zoom = 1e20;
        let orbit = make_ref_orbit(cx, cy, zoom, iter_max);
        if orbit.z_ref_f64.len() < 10 {
            eprintln!("orbite trop courte à zoom 1e20, skip");
            return;
        }
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA");
        let bla = &tables[0];

        let dc_exp = ComplexExp {
            re: FloatExp::new(1.0, -70),
            im: FloatExp::new(2.0, -70),
        };
        let res = iterate_pixel_unified_exp(
            &orbit,
            bla,
            &formula,
            orbit.cref,
            dc_exp,
            ComplexExp::zero(),
            iter_max,
            4.0,
        );
        // Pas de NaN, pas d'Inf.
        assert!(res.z_final.re.is_finite() && res.z_final.im.is_finite());
        // Le résultat est soit dans le set (n == iter_max) soit échappé
        // à un n < iter_max. Pas besoin de valeur précise — juste vivacité.
        eprintln!(
            "exp zoom 1e20 : n={} (max {}), z={:?}, rebases={}",
            res.iteration, iter_max, res.z_final, res.rebase_count
        );
    }
}

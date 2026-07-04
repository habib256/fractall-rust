//! Pixel loop perturbation **double-double** (~106 bits de mantisse) pour le
//! tier haute précision deep-zoom Mandelbrot (`params.use_dd_tier`).
//!
//! Équivalent pur-Rust du path `float128` de Fraktaler-3 (sélectionné par son
//! wisdom file). Mirror de `pixel_loop_exp.rs::iterate_pixel_unified_exp_mandelbrot`
//! mais :
//! - `delta` et `dc` : `ComplexDDExp` (mantisse double-double + exposant `i32`)
//!   au lieu de `ComplexExp` (mantisse f64 53 bits).
//! - la **référence** est lue depuis `ref_orbit.z_ref_dd` (Z à 106 bits) et non
//!   `z_ref_f64` : indispensable car `Z` entre non-arrondi dans `2·Z·δ` — un `Z`
//!   f64 (2⁻⁵²·|Z|) capperait la précision quel que soit le delta.
//! - **pas de BLA** : un pas BLA linéarisé avec des coefficients `A` f64
//!   réintroduirait 2⁻⁵² tôt dans l'orbite (les pixels sensibles BLA-ent quand
//!   δ est minuscule), annulant le gain dd. On itère donc en pas directs purs.
//!   (Une dd-BLA — coefficients `A` double-double — est le chantier perf suivant.)
//!
//! Motivation : les spirales ultra-sensibles profondes (e30/e50) escapent après
//! ~25000 itérations ; l'amplification de Lyapunov transforme le 2⁻⁵² d'arrondi
//! f64 en écart d'itération O(50-200) vs GMP (cf. TODO G2). Le tier dd repousse
//! ce plancher à ~2⁻¹⁰⁵.

use num_complex::Complex64;

use super::pixel_loop_exp::UnifiedPixelResultExp;
use crate::fractal::perturbation::dd::{ComplexDDExp, DoubleDoubleExp};
use crate::fractal::perturbation::orbit::ReferenceOrbit;

/// Pixel loop Mandelbrot double-double : pas directs `δ' = 2·Z·δ + δ² + dc` en
/// ~106 bits + rebasing F3, sans BLA. Utilise `ref_orbit.z_ref_dd`.
///
/// `dc` et `delta_initial` sont en `ComplexDDExp`. Renvoie la même shape que le
/// path exp (`UnifiedPixelResultExp`) pour un câblage transparent au dispatch.
#[allow(clippy::too_many_arguments)]
pub fn iterate_pixel_unified_ddexp_mandelbrot(
    ref_orbit: &ReferenceOrbit,
    dc: ComplexDDExp,
    delta_initial: ComplexDDExp,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
) -> UnifiedPixelResultExp {
    let ref_len = ref_orbit.z_ref_dd.len();
    // Garde : le tier dd exige une orbite dd non vide et de longueur cohérente.
    if ref_len < 2 {
        return UnifiedPixelResultExp {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            ref_exhausted: false,
        };
    }
    let z_ref_dd = &ref_orbit.z_ref_dd;
    let bailout_sqr = DoubleDoubleExp::from_f64(bailout * bailout);

    let mut delta = delta_initial;
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;
    let mut iters_ptb = 0u32;
    const REDUCE_INTERVAL: u32 = 250;

    while n < iteration_max
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
    {
        let z_m = z_ref_dd[m as usize];
        // Bailout : |Z[m] + δ|² ≥ bailout² (tout en dd, pas de saturation f64).
        let z_abs = z_m.add(delta);
        if !(z_abs.norm_sqr() < bailout_sqr) {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: z_abs.to_complex64_approx(),
                rebase_count,
                bla_steps: 0,
                ref_exhausted: false,
            };
        }

        // Pas perturbation Mandelbrot en dd : δ' = 2·Z·δ + δ² + dc.
        let two_z = ComplexDDExp {
            re: z_m.re.mul_f64(2.0),
            im: z_m.im.mul_f64(2.0),
        };
        delta = delta.mul(two_z).add(delta.sqr()).add(dc);
        n += 1;
        m += 1;
        iters_ptb += 1;

        if !delta.is_finite() {
            let m_read = (m as usize).min(ref_len - 1);
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: z_ref_dd[m_read].add(delta).to_complex64_approx(),
                rebase_count,
                bla_steps: 0,
                ref_exhausted: false,
            };
        }

        if n % REDUCE_INTERVAL == 0 {
            delta.reduce();
        }

        // Rebase F3 (`hybrid.cc:296-307`) : `|Z[m]+δ|² < |δ|²` OU bout de
        // référence ⇒ z := Z[m]+δ, m := 0. Périodique : `wrap_periodic`.
        // Escape-time au bout : rebase à 0 (garde le pixel sur le path dd).
        // ⚠️ `m` peut valoir `ref_len` après le pas → clamp pour la lecture.
        let end_of_ref = (m as usize) + 1 >= ref_len;
        let m_read = (m as usize).min(ref_len - 1);
        let z_curr = z_ref_dd[m_read].add(delta);
        if end_of_ref {
            if let Some(m_wrapped) = ref_orbit.wrap_periodic(m) {
                m = m_wrapped;
            } else {
                delta = z_curr;
                m = 0;
                rebase_count += 1;
            }
        } else if z_curr.norm_sqr() < delta.norm_sqr() {
            delta = z_curr;
            m = 0;
            rebase_count += 1;
        }
    }

    let final_m = (m as usize).min(ref_len - 1);
    UnifiedPixelResultExp {
        iteration: n,
        z_final: z_ref_dd[final_m].add(delta).to_complex64_approx(),
        rebase_count,
        bla_steps: 0,
        ref_exhausted: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::perturbation::dd::DoubleDouble;

    /// Le pixel loop dd doit rester fini et cohérent sur une orbite courte
    /// synthétique (référence = quelques Z bornés). Sanity de vivacité.
    #[test]
    fn ddexp_loop_terminates_finite() {
        // Orbite référence dd triviale : Z borné O(1).
        let z_ref_dd: Vec<ComplexDDExp> = (0..64)
            .map(|k| {
                let v = 0.1 * (k as f64 % 5.0);
                ComplexDDExp {
                    re: DoubleDoubleExp::from_dd(DoubleDouble::from_f64(v)),
                    im: DoubleDoubleExp::from_dd(DoubleDouble::from_f64(-v)),
                }
            })
            .collect();
        let orbit = ReferenceOrbit {
            cref: Complex64::new(0.0, 0.0),
            z_ref: Vec::new(),
            z_ref_f64: vec![Complex64::new(0.0, 0.0); z_ref_dd.len()],
            z_ref_gmp: Vec::new(),
            cref_gmp: rug::Complex::with_val(53, (0, 0)),
            phase_offset: 0,
            extended_iterations: Vec::new(),
            high_precision_data: Vec::new(),
            data_storage_interval: 1,
            cycle_period: 0,
            cycle_start: 0,
            z_ref_dd,
        };
        let dc = ComplexDDExp::from_complex64(Complex64::new(1e-30, 1e-30));
        let res = iterate_pixel_unified_ddexp_mandelbrot(
            &orbit,
            dc,
            ComplexDDExp::ZERO,
            500,
            25.0,
            0,
        );
        assert!(res.z_final.re.is_finite() && res.z_final.im.is_finite());
        assert!(res.iteration <= 500);
    }
}

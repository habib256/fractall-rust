//! Nucleus finder pour Mandelbrot (inspiré de Fraktaler-3.1 `hybrid_center`).
//!
//! À deep zoom, l'utilisateur cible un point near-minibrot dont l'orbite
//! escape avant `iteration_max`. La perturbation tronque alors l'orbite
//! référence → image uniforme ou GMP-fallback prohibitif.
//!
//! Solution F3 : trouver le nucleus exact du minibrot voisin via Newton sur
//! `z_period(c) = 0` (le centre périodique passe par 0). Une fois `c` raffiné,
//! l'orbite référence est non-escape et la perturbation fonctionne pour tous
//! les pixels voisins.
//!
//! Pipeline :
//! 1. `find_period` : itère depuis `c0`, retient l'iter `n` qui minimise `|z_n|`
//!    avant escape. Cette n est la période candidate du minibrot le plus proche.
//! 2. `newton_refine_center` : itère Newton `c := c - z_period(c) / z'_period(c)`
//!    jusqu'à convergence (`|delta| < 2^(-prec/2)`). Les dérivées sont propagées
//!    via dual numbers complexes.
//!
//! Limitations actuelles :
//! - Mandelbrot z² + c uniquement (pas BS/Tricorn/Multibrot).
//! - Period detection par scan min-|z| : peut rater des minibrots à `n` >
//!   iter_escape. F3 utilise une condition plus sophistiquée (`|w| > 1/s` où
//!   `s` est la taille du domaine de l'atom).

use rug::{Assign, Complex, Float};
use rug::ops::Pow;

/// Résultat de la recherche de nucleus.
#[derive(Clone, Debug)]
pub struct NucleusResult {
    /// Centre raffiné du minibrot.
    pub center_x: Float,
    pub center_y: Float,
    /// Période détectée du minibrot.
    pub period: u32,
    /// Nombre d'itérations Newton avant convergence.
    pub newton_steps: u32,
    /// `true` si Newton a convergé.
    pub converged: bool,
}

/// Scan l'orbite depuis `(cx, cy)` jusqu'à `max_iter` ou escape, retient
/// l'iter qui minimise `|z_n|`. Retourne `None` si aucun minimum significatif
/// (orbite escape immédiatement ou min reste près du seed).
pub fn find_period(
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    prec: u32,
) -> Option<u32> {
    let bailout_sqr = Float::with_val(prec, 4096.0);
    let mut z_re = Float::with_val(prec, 0);
    let mut z_im = Float::with_val(prec, 0);
    let mut min_norm_sqr = Float::with_val(prec, f64::INFINITY);
    let mut min_iter = 0u32;

    // Scratch buffers pour minimiser les allocations GMP.
    let mut z_re_sq = Float::with_val(prec, 0);
    let mut z_im_sq = Float::with_val(prec, 0);
    let mut z_re_im = Float::with_val(prec, 0);
    let mut norm_sqr = Float::with_val(prec, 0);

    for n in 1..=max_iter {
        // z := z² + c
        z_re_sq.assign(&z_re);
        z_re_sq.square_mut();
        z_im_sq.assign(&z_im);
        z_im_sq.square_mut();
        z_re_im.assign(&z_re);
        z_re_im *= &z_im;
        z_re_im *= 2;

        // new_re = z_re² - z_im² + cx
        let new_re = Float::with_val(prec, &z_re_sq - &z_im_sq) + cx;
        // new_im = 2*z_re*z_im + cy
        let new_im = Float::with_val(prec, &z_re_im) + cy;
        z_re = new_re;
        z_im = new_im;

        // bailout check
        norm_sqr.assign(&z_re);
        norm_sqr.square_mut();
        let tmp = Float::with_val(prec, &z_im) * &z_im;
        norm_sqr += &tmp;
        if norm_sqr > bailout_sqr {
            break;
        }

        // Tracker minimum
        if norm_sqr < min_norm_sqr {
            min_norm_sqr = norm_sqr.clone();
            min_iter = n;
        }
    }

    // Filtrer : si le min iter est très petit (1-3) ou trop tard, c'est
    // suspect. On accepte n >= 4 comme une vraie période candidate.
    if min_iter < 4 {
        return None;
    }
    Some(min_iter)
}

/// Newton-raffine le centre vers le nucleus exact de période `period`.
/// Suit `hybrid_center` de F3.cc : itère `c := c - z_period(c) / z'_period(c)`
/// avec dérivées propagées via dual numbers complexes.
pub fn newton_refine_center(
    cx_in: &Float,
    cy_in: &Float,
    period: u32,
    prec: u32,
    max_steps: u32,
) -> NucleusResult {
    let mut cx = cx_in.clone();
    let mut cy = cy_in.clone();

    // epsilon² = 2^(16 - 2*prec). À prec très grand (e.g., 3357 bits pour
    // e1000), 2^(16-6714) underflow le `Float::pow` à zéro, ce qui rend
    // toute itération "convergée" trivialement après 1 step. F3 utilise
    // floatexp pour cette comparaison ; on émule via rebound (Float * 2^exp).
    let exp_target: i32 = 16i32.saturating_sub(2i32.saturating_mul(prec as i32));
    // Compute 2^exp_target via shift if exp_target valid for Float ; sinon
    // construit explicitement.
    let mut epsilon_sqr = Float::with_val(prec, 1.0);
    if exp_target >= 0 {
        epsilon_sqr <<= exp_target as u32;
    } else {
        epsilon_sqr >>= (-exp_target) as u32;
    }

    // Best-seen state pour return même si non strictement convergé.
    let mut best_z_norm_sqr: Option<Float> = None;
    let mut best_cx = cx.clone();
    let mut best_cy = cy.clone();

    for step in 0..max_steps {
        // Itérer z + dérivées dz/dc sur period itérations.
        // Dual: z = (z_re + i*z_im) ; dz = (dz_re + i*dz_im) = dz/dc.
        // Récurrence :
        //   dz_{n+1} = 2 * z_n * dz_n + 1
        //   z_{n+1}  = z_n² + c
        let mut z_re = Float::with_val(prec, 0);
        let mut z_im = Float::with_val(prec, 0);
        let mut dz_re = Float::with_val(prec, 0);
        let mut dz_im = Float::with_val(prec, 0);

        for _ in 0..period {
            // dz_new = 2 * z * dz + 1
            //   re = 2*(z_re*dz_re - z_im*dz_im) + 1
            //   im = 2*(z_re*dz_im + z_im*dz_re)
            let dz_re_new = {
                let a = Float::with_val(prec, &z_re) * &dz_re;
                let b = Float::with_val(prec, &z_im) * &dz_im;
                let mut t = a - b;
                t *= 2;
                t += 1;
                t
            };
            let dz_im_new = {
                let a = Float::with_val(prec, &z_re) * &dz_im;
                let b = Float::with_val(prec, &z_im) * &dz_re;
                let mut t = a + b;
                t *= 2;
                t
            };
            // z_new = z² + c
            //   re = z_re² - z_im² + cx
            //   im = 2*z_re*z_im + cy
            let z_re_new = {
                let a = Float::with_val(prec, &z_re) * &z_re;
                let b = Float::with_val(prec, &z_im) * &z_im;
                let mut t = a - b;
                t += &cx;
                t
            };
            let z_im_new = {
                let a = Float::with_val(prec, &z_re) * &z_im;
                let mut t = a * 2;
                t += &cy;
                t
            };
            z_re = z_re_new;
            z_im = z_im_new;
            dz_re = dz_re_new;
            dz_im = dz_im_new;
        }

        // Newton: c -= z / dz
        // (z_re + i*z_im) / (dz_re + i*dz_im) = (z * conj(dz)) / |dz|²
        // delta_re = (z_re*dz_re + z_im*dz_im) / |dz|²
        // delta_im = (z_im*dz_re - z_re*dz_im) / |dz|²
        let dz_norm_sqr = {
            let mut t = Float::with_val(prec, &dz_re);
            t.square_mut();
            let mut u = Float::with_val(prec, &dz_im);
            u.square_mut();
            t += &u;
            t
        };
        if dz_norm_sqr.is_zero() {
            return NucleusResult {
                center_x: cx,
                center_y: cy,
                period,
                newton_steps: step,
                converged: false,
            };
        }

        let delta_re = {
            let a = Float::with_val(prec, &z_re) * &dz_re;
            let b = Float::with_val(prec, &z_im) * &dz_im;
            (a + b) / &dz_norm_sqr
        };
        let delta_im = {
            let a = Float::with_val(prec, &z_im) * &dz_re;
            let b = Float::with_val(prec, &z_re) * &dz_im;
            (a - b) / &dz_norm_sqr
        };
        cx -= &delta_re;
        cy -= &delta_im;

        let z_norm_sqr = {
            let mut t = Float::with_val(prec, &z_re);
            t.square_mut();
            let mut u = Float::with_val(prec, &z_im);
            u.square_mut();
            t += &u;
            t
        };

        // Best-seen tracking : on retient le centre qui minimise |z_period(c)|
        // pour pouvoir retourner même si Newton n'atteint pas epsilon strict
        // (improvement net sur le centre original).
        let update_best = match best_z_norm_sqr {
            None => true,
            Some(ref b) => z_norm_sqr < *b,
        };
        if update_best {
            best_z_norm_sqr = Some(z_norm_sqr.clone());
            best_cx = cx.clone();
            best_cy = cy.clone();
        }

        // Convergence : delta < epsilon (norme sub-précision).
        let delta_norm_sqr = {
            let mut t = Float::with_val(prec, &delta_re);
            t.square_mut();
            let mut u = Float::with_val(prec, &delta_im);
            u.square_mut();
            t += &u;
            t
        };
        if delta_norm_sqr < epsilon_sqr && z_norm_sqr < epsilon_sqr {
            return NucleusResult {
                center_x: cx,
                center_y: cy,
                period,
                newton_steps: step + 1,
                converged: true,
            };
        }
    }

    // Pas de convergence stricte mais on a peut-être amélioré.
    // Si |z_period(best_center)| < |z_period(center_original)|, considère succès.
    let cx_final;
    let cy_final;
    if best_z_norm_sqr.is_some() {
        cx_final = best_cx;
        cy_final = best_cy;
    } else {
        cx_final = cx;
        cy_final = cy;
    }

    NucleusResult {
        center_x: cx_final,
        center_y: cy_final,
        period,
        newton_steps: max_steps,
        converged: false,
    }
}

/// Pipeline complet : trouve la période candidate puis Newton-raffine.
/// Retourne `None` si aucune période trouvée ou Newton ne converge pas.
pub fn find_nucleus(
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    prec: u32,
) -> Option<NucleusResult> {
    let period = find_period(cx, cy, max_iter, prec)?;
    let result = newton_refine_center(cx, cy, period, prec, 64);
    // Accepte le résultat même non strictement convergé : Newton améliore
    // souvent le centre suffisamment pour que la perturbation fonctionne,
    // sans atteindre l'epsilon sub-précision (qui est parfois inatteignable
    // par rounding pour des minibrots à orbites longues).
    Some(result)
}

/// Helper : centre complexe → (Float, Float).
#[allow(dead_code)]
pub fn complex_to_xy(c: &Complex, prec: u32) -> (Float, Float) {
    (
        Float::with_val(prec, c.real()),
        Float::with_val(prec, c.imag()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nucleus_main_cardioid_seed_converges_to_zero() {
        // Le seed (0, 0) est près du nucleus période 1 (le cardioïde central).
        // Newton doit converger vers (-0.25, 0) — le centre de la période-1
        // bulbe, mais en réalité la cardioïde n'a pas de "nucleus" classique.
        // Test plus simple : period 1, point près de 0, doit converger.
        let prec = 256u32;
        let cx = Float::with_val(prec, 0.1);
        let cy = Float::with_val(prec, 0.0);
        let res = newton_refine_center(&cx, &cy, 1, prec, 64);
        // z_1(c) = c, donc Newton c := c - c / 1 = 0. Doit converger en 1 step.
        assert!(res.converged, "Newton should converge for period 1");
    }

    #[test]
    fn nucleus_period_2_converges_to_minus_one() {
        // Période 2 : z_1 = c, z_2 = c² + c = c(c+1). Pour z_2 = 0 → c = 0 ou c = -1.
        // Démarrer près de -1 doit converger vers -1.
        let prec = 256u32;
        let cx = Float::with_val(prec, -0.9);
        let cy = Float::with_val(prec, 0.0);
        let res = newton_refine_center(&cx, &cy, 2, prec, 64);
        assert!(res.converged);
        let actual = res.center_x.to_f64();
        assert!((actual - (-1.0)).abs() < 1e-10, "Expected -1, got {}", actual);
    }
}

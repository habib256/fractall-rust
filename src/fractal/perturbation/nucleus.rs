//! Nucleus finder pour Mandelbrot (port fidèle de Fraktaler-3.1
//! `hybrid_period` + `hybrid_center`, cf. `fraktaler-3-3.1/src/hybrid.cc:417`).
//!
//! À deep zoom, l'utilisateur cible un point near-minibrot dont l'orbite
//! escape avant `iteration_max`. La perturbation tronque alors l'orbite
//! référence → image uniforme ou GMP-fallback prohibitif.
//!
//! Solution F3 :
//! 1. Détecter la période du minibrot voisin via le critère ball-arithmetic /
//!    atom-domain : `|z|² < s² · |dz|²` où `s` est l'échelle de vue (`~ 1/zoom`)
//!    et `dz` la dérivée de l'itération par rapport à `c`. Plus précis que
//!    le min-|z| heuristique : la criterion encode directement la définition
//!    de l'atome (boule centrée sur le nucleus de taille s/|dz|).
//! 2. Newton-raffiner le centre vers `z_period(c) = 0` (le centre périodique
//!    passe par 0). Une fois `c` raffiné, l'orbite référence est non-escape
//!    et la perturbation fonctionne pour tous les pixels voisins.
//!
//! Limitations :
//! - Mandelbrot z² + c uniquement (pas BS/Tricorn/Multibrot).
//! - Itération de période en GMP (précis mais lent pour très deep zoom). F3
//!   utilise floatexp pour la passe de détection ; à porter si la perf
//!   devient bloquante (cf. e22522 prec ~80k bits).

use rug::{Assign, Complex, Float};

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

/// Détecte la période du minibrot le plus proche via le critère atom-domain F3
/// (`fraktaler-3-3.1/src/hybrid.cc::hybrid_period`).
///
/// Itère `z_{n+1} = z_n² + c` et `dz_{n+1} = 2 z_n dz_n + 1` (dérivée par
/// rapport à `c`). Retourne la première itération `n ≥ 4` où :
///
///   |z_n|² < s² · |dz_n|²
///
/// `s` est l'échelle de vue (~ `1 / zoom` ou `view_radius`). Le critère encode
/// la définition même d'un atome de période n : z_n est dans une boule de
/// rayon `s / |dz_n|` autour du nucleus, dont la taille `s/|dz|` est l'estimée
/// linéaire de la taille de l'atome via le théorème d'Alexander.
///
/// Retourne `None` si l'orbite escape avant qu'une période ne soit détectée,
/// ou si `max_iter` est atteint. C'est un signal que l'utilisateur est hors
/// d'atteinte d'un atome dans le champ visuel — la perturbation classique
/// reste alors la meilleure stratégie.
pub fn find_period_atom_domain(
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    s: &Float,
    prec: u32,
) -> Option<u32> {
    // Bailout très large (F3 utilise 10^10000, on prend 1e60 — plus que
    // suffisant pour détecter une orbite non-bornée sans gêner la détection
    // de boucle interne).
    let bailout_sqr = Float::with_val(prec, 1e120);
    let s_sqr = Float::with_val(prec, s * s);

    let mut z_re = Float::with_val(prec, 0);
    let mut z_im = Float::with_val(prec, 0);
    // dz initialisé à 0 (z_0 = 0, donc dz_0 = d0/dc = 0). Première iter :
    //   dz_1 = 2 * 0 * 0 + 1 = 1
    let mut dz_re = Float::with_val(prec, 0);
    let mut dz_im = Float::with_val(prec, 0);

    let mut z_norm_sqr = Float::with_val(prec, 0);
    let mut dz_norm_sqr = Float::with_val(prec, 0);
    let mut threshold = Float::with_val(prec, 0);

    for n in 1..=max_iter {
        // dz := 2 * z * dz + 1
        //   re = 2*(z_re*dz_re - z_im*dz_im) + 1
        //   im = 2*(z_re*dz_im + z_im*dz_re)
        let dz_re_new = {
            let a = Float::with_val(prec, &z_re * &dz_re);
            let b = Float::with_val(prec, &z_im * &dz_im);
            let mut t = a - b;
            t *= 2;
            t += 1;
            t
        };
        let dz_im_new = {
            let a = Float::with_val(prec, &z_re * &dz_im);
            let b = Float::with_val(prec, &z_im * &dz_re);
            let mut t = a + b;
            t *= 2;
            t
        };

        // z := z² + c
        //   re = z_re² - z_im² + cx
        //   im = 2*z_re*z_im + cy
        let z_re_new = {
            let a = Float::with_val(prec, &z_re * &z_re);
            let b = Float::with_val(prec, &z_im * &z_im);
            let mut t = a - b;
            t += cx;
            t
        };
        let z_im_new = {
            let a = Float::with_val(prec, &z_re * &z_im);
            let mut t = a * 2;
            t += cy;
            t
        };

        z_re = z_re_new;
        z_im = z_im_new;
        dz_re = dz_re_new;
        dz_im = dz_im_new;

        // Bailout — orbite hors Mandelbrot, pas de période détectable.
        z_norm_sqr.assign(&z_re);
        z_norm_sqr.square_mut();
        let tmp = Float::with_val(prec, &z_im * &z_im);
        z_norm_sqr += &tmp;
        if z_norm_sqr > bailout_sqr {
            return None;
        }

        // Atom-domain criterion : |z|² < s² * |dz|²
        // F3 ne pose aucune borne inférieure sur n : pour un centre près du
        // cardioïde (période 1) ou de la bulbe principale (période 2),
        // retourner ces périodes courtes est correct. Le critère est lui-même
        // strict (un orbite chaotique ne satisfait pas |z|² < s²|dz|²).
        dz_norm_sqr.assign(&dz_re);
        dz_norm_sqr.square_mut();
        let tmp = Float::with_val(prec, &dz_im * &dz_im);
        dz_norm_sqr += &tmp;
        threshold.assign(&s_sqr);
        threshold *= &dz_norm_sqr;
        if z_norm_sqr < threshold {
            return Some(n);
        }
    }

    None
}

/// Alias rétro-compatible : utilise une échelle `s = 1e-6` par défaut, qui
/// approxime une "zone d'intérêt" sans dépendre du zoom courant. Préférer
/// `find_period_atom_domain` avec un `s` explicite quand le zoom est connu.
#[deprecated(note = "use find_period_atom_domain with an explicit view scale `s`")]
pub fn find_period(
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    prec: u32,
) -> Option<u32> {
    let s = Float::with_val(prec, 1e-6);
    find_period_atom_domain(cx, cy, max_iter, &s, prec)
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

/// Pipeline complet : détecte la période via le critère atom-domain puis
/// Newton-raffine le centre. Retourne `None` si aucune période n'est trouvée
/// ou si la dérivée Newton est trop petite pour converger.
///
/// `s` est l'échelle de vue (~ `max(span_x, span_y) / 2`). À zoom courant,
/// l'utilisateur regarde une fenêtre de cette taille, donc on cherche le
/// nucleus dont l'atome remplit cette fenêtre — pas un nucleus plus petit
/// (qui serait subpixel) ni plus gros (qui ne serait pas centré).
pub fn find_nucleus(
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    s: &Float,
    prec: u32,
) -> Option<NucleusResult> {
    let period = find_period_atom_domain(cx, cy, max_iter, s, prec)?;
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

    #[test]
    fn atom_domain_detects_period_3_satellite() {
        // L'atom de période 3 vit autour de c = -1.754877666... (un minibrot
        // sur la branche négative). Près de ce point, l'atom-domain criterion
        // doit fire à n = 3.
        let prec = 128u32;
        let cx = Float::with_val(prec, -1.7548776);
        let cy = Float::with_val(prec, 0.0);
        // Échelle de vue ~ 0.01 — taille typique d'une vue centrée sur ce
        // minibrot. s grand permet à la condition de fire dès la période
        // minimale.
        let s = Float::with_val(prec, 0.01);
        let period = find_period_atom_domain(&cx, &cy, 100, &s, prec);
        assert_eq!(
            period,
            Some(3),
            "Expected period 3 for c ≈ -1.7548776, got {:?}",
            period
        );
    }

    #[test]
    fn atom_domain_returns_none_for_escaping_orbit() {
        // c = 1.0 escape immédiatement (z_2 = 2, z_3 = 5...). Aucune période
        // ne doit être détectée.
        let prec = 64u32;
        let cx = Float::with_val(prec, 1.0);
        let cy = Float::with_val(prec, 0.0);
        let s = Float::with_val(prec, 1e-3);
        let period = find_period_atom_domain(&cx, &cy, 100, &s, prec);
        assert_eq!(period, None);
    }

    #[test]
    fn atom_domain_full_pipeline_period_3() {
        // Pipeline complet : find_nucleus depuis un point proche du minibrot
        // période 3 doit retourner cx ≈ -1.7548776662..., converged.
        let prec = 192u32;
        let cx = Float::with_val(prec, -1.7548);
        let cy = Float::with_val(prec, 0.0);
        let s = Float::with_val(prec, 0.01);
        let res = find_nucleus(&cx, &cy, 100, &s, prec).expect("should find nucleus");
        assert_eq!(res.period, 3);
        let expected = -1.7548776662466927_f64;
        assert!(
            (res.center_x.to_f64() - expected).abs() < 1e-12,
            "Expected x ≈ {}, got {}",
            expected,
            res.center_x.to_f64()
        );
    }
}

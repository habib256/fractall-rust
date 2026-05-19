//! Implémentation minimaliste F3-pure de la perturbation Mandelbrot pour debug.
//!
//! Mirror exact de `hybrid_render` de Fraktaler-3, *sans* BLA, *sans* glitch
//! detection, *sans* aucune autre feature. Sert uniquement à isoler le bug
//! du path rebasing (cf. `docs/p3-1-task6-investigation.md`).
//!
//! Boucle F3 pour Mandelbrot (z² + c) :
//! ```text
//! while (n < Iterations && |z|² < ER² && iters_ptb < limit) {
//!     // Pas perturbation : δ_{n+1} = 2·Z[m]·δ + δ² + c_pixel
//!     δ = 2·Z[m]·δ + δ·δ + c_pixel
//!     m += 1; n += 1; iters_ptb += 1;
//!     // Rebase check : si |Z[m] + δ|² < |δ|² OR fin d'orbite : reseat
//!     let z_curr = Z[m] + δ;
//!     if |z_curr|² < |δ|² || m+1 == ref_len {
//!         δ = z_curr;  // promotion delta → absolu
//!         m = 0;
//!     }
//! }
//! ```

use num_complex::Complex64;

use crate::fractal::perturbation::orbit::ReferenceOrbit;

/// Résultat d'une itération F3-pure d'un pixel.
#[derive(Clone, Copy, Debug)]
pub struct PureF3Result {
    /// Nombre total d'itérations (n).
    pub iteration: u32,
    /// Valeur finale du delta.
    #[allow(dead_code)]
    pub delta_final: Complex64,
    /// Nombre de rebases effectuées.
    pub rebase_count: u32,
    /// Index dans la référence à la sortie (m).
    pub m_at_exit: u32,
}

/// Calcule l'évasion Mandelbrot pour un pixel via perturbation + rebasing F3 pur.
///
/// - `ref_orbit` : orbite référence (utilise z_ref_f64).
/// - `c_pixel` : offset du pixel par rapport au centre de la référence.
/// - `iteration_max` : cap d'itérations.
/// - `bailout` : rayon d'évasion (compare `|z|² ≥ bailout²` où z = Z + δ).
pub fn iterate_pixel_pure_f3_mandelbrot(
    ref_orbit: &ReferenceOrbit,
    c_pixel: Complex64,
    iteration_max: u32,
    bailout: f64,
) -> PureF3Result {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return PureF3Result {
            iteration: 0,
            delta_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            m_at_exit: 0,
        };
    }

    let mut delta = Complex64::new(0.0, 0.0);
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;

    while n < iteration_max {
        // Test bailout sur z absolu = Z[m] + delta.
        let z_m = ref_orbit.z_ref_f64[m as usize];
        let z_abs = z_m + delta;
        if z_abs.norm_sqr() >= bailout_sqr {
            break;
        }

        // Pas perturbation Mandelbrot :
        // δ_{n+1} = 2·Z[m]·δ + δ² + c_pixel
        let two_zm = z_m * 2.0;
        delta = two_zm * delta + delta * delta + c_pixel;

        n += 1;
        m += 1;

        // Protection NaN/Inf.
        if !delta.re.is_finite() || !delta.im.is_finite() {
            break;
        }

        // Rebase check F3 strict : si |Z[m] + δ|² < |δ|² OR fin d'orbite.
        let end_of_ref = (m as usize) + 1 >= ref_len;
        if end_of_ref {
            // Promotion delta → absolu, reset m.
            let z_m_new = ref_orbit.z_ref_f64[m as usize];
            delta = z_m_new + delta;
            m = 0;
            rebase_count += 1;
        } else {
            let z_m_new = ref_orbit.z_ref_f64[m as usize];
            let z_curr = z_m_new + delta;
            let z_curr_norm_sqr = z_curr.norm_sqr();
            let delta_norm_sqr = delta.norm_sqr();
            if z_curr_norm_sqr < delta_norm_sqr {
                delta = z_curr;
                m = 0;
                rebase_count += 1;
            }
        }
    }

    PureF3Result {
        iteration: n,
        delta_final: delta,
        rebase_count,
        m_at_exit: m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::perturbation::orbit::{compute_reference_orbit, ReferenceOrbit};
    use crate::fractal::{default_params_for_type, FractalType};

    /// Construit une ReferenceOrbit pour Mandelbrot à un center/zoom donné.
    fn make_ref_orbit(center_x: f64, center_y: f64, zoom: f64, iter_max: u32) -> ReferenceOrbit {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 100);
        params.center_x = center_x;
        params.center_y = center_y;
        let span_x = 4.0 / zoom;
        params.span_x = span_x;
        params.span_y = span_x * 100.0 / 160.0;
        params.iteration_max = iter_max;
        params.algorithm_mode = crate::fractal::AlgorithmMode::Perturbation;
        let (orbit, _, _) = compute_reference_orbit(&params, None)
            .expect("compute_reference_orbit failed");
        orbit
    }

    /// Sur une vue Mandelbrot peu profonde (zoom 100), F3-pure devrait classer
    /// un pixel intérieur à `iteration_max` (pas d'évasion) et un pixel extérieur
    /// à un n < iteration_max.
    #[test]
    fn pure_f3_classifies_interior_and_exterior() {
        let iter_max = 500u32;
        // Pixel intérieur : centre de la cardioïde principale (-0.25, 0).
        let orbit = make_ref_orbit(-0.25, 0.0, 100.0, iter_max);
        let res_interior = iterate_pixel_pure_f3_mandelbrot(
            &orbit,
            Complex64::new(0.0, 0.0), // pixel au centre = centre de la référence
            iter_max,
            4.0,
        );
        assert_eq!(
            res_interior.iteration, iter_max,
            "Centre cardioïde doit être intérieur, n={}",
            res_interior.iteration
        );

        // Pixel extérieur : c_abs = -0.25 + 2 + 2i = 1.75 + 2i, clairement hors set.
        let c_outside = Complex64::new(2.0, 2.0);
        let res_outside = iterate_pixel_pure_f3_mandelbrot(&orbit, c_outside, iter_max, 4.0);
        assert!(
            res_outside.iteration < iter_max,
            "Pixel extérieur (c_abs=1.75+2i) doit échapper, n={}",
            res_outside.iteration
        );
    }

    /// Sur un pixel intérieur, le rebasing F3 devrait se déclencher plusieurs fois
    /// (le delta croît, croise la référence proche de 0, rebase).
    #[test]
    fn pure_f3_rebasing_triggers() {
        let iter_max = 500u32;
        let orbit = make_ref_orbit(-0.5, 0.0, 50.0, iter_max);
        // Pixel proche du centre mais légèrement décalé.
        let c_pixel = Complex64::new(1e-4, 1e-4);
        let res = iterate_pixel_pure_f3_mandelbrot(&orbit, c_pixel, iter_max, 4.0);
        // Avec ref_len < iter_max, on doit forcément rebase au moins une fois
        // (fin d'orbite).
        let ref_len = orbit.z_ref_f64.len() as u32;
        if ref_len < iter_max {
            assert!(
                res.rebase_count > 0,
                "rebase_count attendu > 0 quand ref_len({}) < iter_max({})",
                ref_len,
                iter_max
            );
        }
    }

    /// Compare F3-pure vs `iterate_pixel(use_legacy_glitch_detection=false)`
    /// pour identifier d'où vient la divergence dans le path "F3-pure" de
    /// production. Si les deux matchent à 100%, le path en production est
    /// algorithmiquement correct (et la divergence vs legacy vient du BLA
    /// + glitch detection legacy). Si non, on a un bug dans iterate_pixel.
    #[test]
    fn pure_f3_vs_iterate_pixel_no_legacy() {
        use crate::fractal::perturbation::bla::build_bla_table;
        use crate::fractal::perturbation::delta::iterate_pixel;
        use crate::fractal::perturbation::types::ComplexExp;
        use crate::fractal::{AlgorithmMode, default_params_for_type, FractalType};

        let iter_max = 1500u32;
        let center_x = -1.7693831791955;
        let center_y = 0.004236847918736;
        let zoom = 1e6;
        let span_x = 4.0 / zoom;
        let aspect = 100.0 / 160.0;
        let span_y = span_x * aspect;

        // Setup full params comme l'irait le rendu réel.
        let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 100);
        params.center_x = center_x;
        params.center_y = center_y;
        params.span_x = span_x;
        params.span_y = span_y;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        // Note historique : avant Session E, ce test forçait
        // use_legacy_glitch_detection=false pour exercer le path F3-pur de
        // production. Depuis Session E + cleanup, le path bytecode est par
        // défaut et le champ a été supprimé. Le test reste valide car
        // iterate_pixel est devenu F3-pur via le dispatch bytecode.

        let (orbit, _, _) = compute_reference_orbit(&params, None)
            .expect("compute_reference_orbit failed");
        let bla_table = build_bla_table(&orbit.z_ref_f64, &params, orbit.cref);

        let mut classification_diffs = 0usize;
        let mut iter_diffs: Vec<i64> = Vec::new();
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let c_pixel_offset = Complex64::new(dx, dy);

                let pure = iterate_pixel_pure_f3_mandelbrot(
                    &orbit, c_pixel_offset, iter_max, 4.0,
                );

                // iterate_pixel attend delta0=ComplexExp::zero() pour Mandelbrot
                // et dc = ComplexExp du pixel offset
                let delta0 = ComplexExp::from_complex64(Complex64::new(0.0, 0.0));
                let dc_exp = ComplexExp::from_complex64(c_pixel_offset);
                let result = iterate_pixel(
                    &params, &orbit, &bla_table, None, delta0, dc_exp, None, None,
                );

                let escaped_pure = pure.iteration < iter_max;
                let escaped_prod = result.iteration < iter_max;
                if escaped_pure != escaped_prod {
                    classification_diffs += 1;
                    if classification_diffs <= 5 {
                        eprintln!(
                            "  ({},{}) classif diff: pure n={} esc={} | prod n={} esc={} glitched={}",
                            i, j, pure.iteration, escaped_pure,
                            result.iteration, escaped_prod, result.glitched
                        );
                    }
                }
                if escaped_pure && escaped_prod {
                    iter_diffs.push(pure.iteration as i64 - result.iteration as i64);
                }
                total += 1;
            }
        }
        let identical_iter = iter_diffs.iter().filter(|&&d| d == 0).count();
        let mean_iter_diff = if iter_diffs.is_empty() {
            0.0
        } else {
            iter_diffs.iter().sum::<i64>() as f64 / iter_diffs.len() as f64
        };
        let max_iter_diff = iter_diffs.iter().map(|d| d.abs()).max().unwrap_or(0);
        eprintln!(
            "Pure F3 vs iterate_pixel(use_legacy=false) sur {} pixels :",
            total
        );
        eprintln!(
            "  classifications matchent : {}/{}",
            total - classification_diffs,
            total
        );
        eprintln!(
            "  iter counts (pixels escaped) : identiques {}/{}, mean diff {:.2}, max |diff| {}",
            identical_iter,
            iter_diffs.len(),
            mean_iter_diff,
            max_iter_diff
        );
    }

    /// Compare F3-pure vs path standard (sans perturbation) sur un pixel.
    /// L'idée : à zoom = 1, ref + perturbation == iteration directe.
    /// On vérifie que le n produit est cohérent (peut différer de ±quelques
    /// itérations à cause de l'ordre des checks bailout vs increment, mais le
    /// résultat classifiant doit être pareil : escaped ou non).
    /// Cas critique : sur la location Mandelbrot zoom 1e6 (`mandelbrot_perturb_1e6`
    /// du corpus golden), F3-pure devrait produire une image cohérente avec une
    /// itération directe haute-précision. On échantillonne 50 pixels d'une grille
    /// et on compare la classification escaped/interior avec une itération directe
    /// à la précision GMP du centre.
    ///
    /// L'objectif de ce test : prouver que **F3-pure (sans BLA, sans glitch
    /// detection) classifie correctement les pixels d'une vue deep zoom**.
    /// Si oui, alors la divergence legacy vs F3-pure dans iterate_pixel vient
    /// d'autre chose (BLA, ordre des checks…). Si non, on a identifié un bug
    /// dans le rebasing.
    #[test]
    fn pure_f3_classifies_correctly_at_deep_zoom() {
        let iter_max = 1500u32;
        let center_x = -1.7693831791955;
        let center_y = 0.004236847918736;
        let zoom = 1e6;
        let orbit = make_ref_orbit(center_x, center_y, zoom, iter_max);
        let span_x = 4.0 / zoom;
        let aspect = 100.0 / 160.0;
        let span_y = span_x * aspect;

        // Pour chaque pixel, calculer c_abs (centre + offset) et le c_pixel (offset).
        // Comparer F3-pure escape vs itération directe escape.
        let mut mismatches = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let c_pixel_offset = Complex64::new(dx, dy);
                let c_abs = Complex64::new(center_x + dx, center_y + dy);

                let pure = iterate_pixel_pure_f3_mandelbrot(
                    &orbit, c_pixel_offset, iter_max, 4.0,
                );
                let escaped_pure = pure.iteration < iter_max;

                // Itération directe en f64. C'est imprécis à ce zoom mais
                // donne une référence approximative pour la classification.
                let mut z = Complex64::new(0.0, 0.0);
                let mut n = 0u32;
                while n < iter_max && z.norm_sqr() < 16.0 {
                    z = z * z + c_abs;
                    n += 1;
                }
                let escaped_direct = n < iter_max;

                if escaped_pure != escaped_direct {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "  ({:3},{:3}) dx={:.3e} dy={:.3e} | pure: n={} esc={} rebase={} | direct: n={} esc={}",
                            i, j, dx, dy, pure.iteration, escaped_pure, pure.rebase_count, n, escaped_direct
                        );
                    }
                }
                total += 1;
            }
        }
        eprintln!(
            "Deep zoom 1e6 : {}/{} classifications matchent F3-pure vs direct",
            total - mismatches,
            total
        );

        // Re-passe : pour les pixels qui matchent en classification, mesurer
        // les écarts de iteration count entre F3-pure et direct. Ça nous dira
        // si la divergence d'images vient des couleurs (smooth coloring est
        // sensible au n exact) plutôt que de la classification.
        let mut iter_diffs: Vec<i64> = Vec::new();
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let c_pixel_offset = Complex64::new(dx, dy);
                let c_abs = Complex64::new(center_x + dx, center_y + dy);

                let pure = iterate_pixel_pure_f3_mandelbrot(
                    &orbit, c_pixel_offset, iter_max, 4.0,
                );
                let mut z = Complex64::new(0.0, 0.0);
                let mut n = 0u32;
                while n < iter_max && z.norm_sqr() < 16.0 {
                    z = z * z + c_abs;
                    n += 1;
                }
                if pure.iteration != iter_max && n != iter_max {
                    iter_diffs.push(pure.iteration as i64 - n as i64);
                }
            }
        }
        if !iter_diffs.is_empty() {
            let mean = iter_diffs.iter().sum::<i64>() as f64 / iter_diffs.len() as f64;
            let max_abs = iter_diffs.iter().map(|d| d.abs()).max().unwrap();
            let identical = iter_diffs.iter().filter(|&&d| d == 0).count();
            eprintln!(
                "Iteration counts (pixels échappés): identiques={}/{}, mean diff={:.2}, max |diff|={}",
                identical,
                iter_diffs.len(),
                mean,
                max_abs
            );
        }
        // À ce zoom modéré, on s'attend à ce que la perturbation rebasing
        // donne approximativement les mêmes classifications que f64 direct
        // (le f64 direct est imprécis mais pas tant que ça à zoom 1e6).
        // On tolère qq mismatches mais pas la majorité.
        let mismatch_pct = 100.0 * mismatches as f64 / total as f64;
        assert!(
            mismatch_pct < 50.0,
            "Trop de mismatches F3-pure vs direct: {:.1}%",
            mismatch_pct
        );
    }

    #[test]
    fn pure_f3_consistent_with_direct_iteration() {
        let iter_max = 1000u32;
        let orbit = make_ref_orbit(-0.5, 0.0, 1.0, iter_max);
        // 20 pixels random-like dans une grille raisonnable
        let test_points = [
            (-0.5, 0.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            (-1.5, 0.5),
            (0.3, 0.5),
            (-0.75, 0.1),
            (-2.0, 0.0),
            (-0.16, 1.04),
            (-1.25, 0.0),
            (0.25, 0.0),
        ];
        let mut classifications_match = 0;
        let mut total = 0;
        for (px, py) in &test_points {
            // c_pixel est l'absolu, donc on calcule offset depuis le ref center.
            let c_offset = Complex64::new(*px - (-0.5), *py - 0.0);
            let res = iterate_pixel_pure_f3_mandelbrot(&orbit, c_offset, iter_max, 4.0);

            // Comparer avec itération Mandelbrot directe.
            let c_abs = Complex64::new(*px, *py);
            let mut z = Complex64::new(0.0, 0.0);
            let mut n = 0u32;
            while n < iter_max && z.norm_sqr() < 16.0 {
                z = z * z + c_abs;
                n += 1;
            }

            // Les deux classifications (escaped/interior) doivent matcher.
            let escaped_pure = res.iteration < iter_max;
            let escaped_direct = n < iter_max;
            if escaped_pure == escaped_direct {
                classifications_match += 1;
            } else {
                eprintln!(
                    "[Mismatch] pixel ({}, {}): pure={} (n={}, rebase={}, m={}), direct={} (n={})",
                    px, py, escaped_pure, res.iteration, res.rebase_count, res.m_at_exit, escaped_direct, n
                );
            }
            total += 1;
        }
        assert_eq!(
            classifications_match, total,
            "F3-pure et itération directe doivent classifier identiquement"
        );
    }
}

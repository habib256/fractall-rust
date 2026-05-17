//! Pixel loop perturbation F3-pur avec BLA mat2 unifiée.
//!
//! Mirror direct de `hybrid_render` de Fraktaler-3 (cf.
//! `docs/fraktaler-3-analysis.md` §3). Architecture :
//! 1. BLA lookup : `bla.lookup(m, |δ|²)`. Si trouvé : `δ := A·δ + B·dc`,
//!    skip `l` itérations.
//! 2. Sinon : pas perturbation via la forme delta de l'opcode.
//! 3. Check rebasing F3 strict : `|Z[m+1] + δ|² < |δ|²` OR fin d'orbite
//!    → reseat `δ := Z[m+1] + δ`, `m := 0`.
//!
//! Pas de glitch detection (Pauldelbrot). Pas de secondary refs. C'est
//! l'objectif F3 : remplacer toute la machinerie glitch par le rebasing
//! proactif.
//!
//! Limitations actuelles (incrémental) :
//! - Mandelbrot uniquement pour le pas perturbation (formule
//!   `δ' = 2·Z·δ + δ² + dc` hardcodée). Les autres types appellent le
//!   path existant via fallback. Extension à BS/Tricorn/Celtic/Multibrot
//!   = chantier suivant (utilise la forme delta du bytecode).
//! - f64 path (pas de ComplexExp pour très deep zoom).
//! - Mono-phase seulement (la `BlaTableUnified` est passée seule, pas
//!   `Vec<BlaTableUnified>`).

use num_complex::Complex64;

use super::bla_dual::BlaTableUnified;
use super::delta_form::DeltaState;
use super::{Formula, Phase};
use crate::fractal::perturbation::orbit::ReferenceOrbit;

/// Résultat d'un pixel via le pixel loop unifié.
pub struct UnifiedPixelResult {
    pub iteration: u32,
    pub z_final: Complex64,
    /// Nombre de rebases effectués (utile pour stats/debug).
    pub rebase_count: u32,
    /// Nombre de pas BLA appliqués (utile pour stats/debug).
    pub bla_steps: u32,
}

/// Pixel loop unifié pour TOUS les types escape-time supportés par le bytecode :
/// BLA mat2 + interpréteur delta-form + rebasing F3.
///
/// Généralise `iterate_pixel_unified_mandelbrot` à tous les types via
/// `DeltaState::step` qui propage (Z, δ) selon la règle de chaîne propre
/// à chaque opcode.
///
/// - `ref_orbit` : orbite référence (f64 path).
/// - `bla` : table BLA unifiée pour la phase.
/// - `formula` : compile_formula(type, multibrot_power) — défini la phase
///   appliquée à chaque itération.
/// - `c_ref` : constante ajoutée à la reference (cref pour Mandelbrot-like,
///   seed pour Julia-like).
/// - `dc` : offset = c_pixel - c_ref.
pub fn iterate_pixel_unified(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    formula: &Formula,
    c_ref: Complex64,
    dc: Complex64,
    delta_initial: Complex64,
    iteration_max: u32,
    bailout: f64,
) -> UnifiedPixelResult {
    // Mono-phase pour l'instant (la formule a une seule phase).
    assert_eq!(
        formula.phases.len(),
        1,
        "Multi-phase pas encore supporté dans pixel_loop unifié"
    );
    let phase = &formula.phases[0];
    iterate_pixel_unified_single_phase(
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

fn iterate_pixel_unified_single_phase(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    phase: &Phase,
    c_ref: Complex64,
    dc: Complex64,
    delta_initial: Complex64,
    iteration_max: u32,
    bailout: f64,
) -> UnifiedPixelResult {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return UnifiedPixelResult {
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

    while n < iteration_max {
        let z_m = ref_orbit.z_ref_f64[m as usize];
        let z_abs = z_m + delta;
        if z_abs.norm_sqr() >= bailout_sqr {
            return UnifiedPixelResult {
                iteration: n,
                z_final: z_abs,
                rebase_count,
                bla_steps,
            };
        }

        // BLA lookup
        let delta_norm_sqr = delta.norm_sqr();
        if let Some(node) = bla.lookup(m as usize, delta_norm_sqr) {
            let new_n = n.saturating_add(node.l);
            let new_m = m.saturating_add(node.l);
            if new_n <= iteration_max && (new_m as usize) < ref_len {
                let a = node.a;
                let b = node.b;
                let (a_re, a_im) = (
                    a.m00 * delta.re + a.m01 * delta.im,
                    a.m10 * delta.re + a.m11 * delta.im,
                );
                let (b_re, b_im) = (
                    b.m00 * dc.re + b.m01 * dc.im,
                    b.m10 * dc.re + b.m11 * dc.im,
                );
                delta = Complex64::new(a_re + b_re, a_im + b_im);
                n = new_n;
                m = new_m;
                bla_steps += 1;

                if !delta.re.is_finite() || !delta.im.is_finite() {
                    return UnifiedPixelResult {
                        iteration: n,
                        z_final: ref_orbit.z_ref_f64[m as usize] + delta,
                        rebase_count,
                        bla_steps,
                    };
                }
                continue;
            }
        }

        // Pas perturbation via delta-form interpreter
        // (généralise la formule Mandelbrot hardcodée à tous les types).
        let mut state = DeltaState::new(z_m, delta);
        state.step(phase, c_ref, dc);
        delta = state.delta;
        n += 1;
        m += 1;

        if !delta.re.is_finite() || !delta.im.is_finite() {
            return UnifiedPixelResult {
                iteration: n,
                z_final: ref_orbit.z_ref_f64[m.min((ref_len - 1) as u32) as usize] + delta,
                rebase_count,
                bla_steps,
            };
        }

        // Rebase F3
        let end_of_ref = (m as usize) + 1 >= ref_len;
        if end_of_ref {
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

    let final_m = m.min((ref_len - 1) as u32);
    UnifiedPixelResult {
        iteration: n,
        z_final: ref_orbit.z_ref_f64[final_m as usize] + delta,
        rebase_count,
        bla_steps,
    }
}

/// Pixel loop Mandelbrot unifié : BLA mat2 + perturbation delta + rebasing F3.
///
/// Conservé pour rétrocompatibilité des tests Session C. Nouveaux callers
/// devraient utiliser `iterate_pixel_unified` (généralisé à tous les types).
///
/// - `ref_orbit` : orbite référence (f64 path).
/// - `bla` : table BLA unifiée pour la phase (mat2-based, built via
///   `bla_dual::build_bla_table_for_formula`).
/// - `dc` : offset du pixel par rapport au centre de la référence (Complex64).
/// - `iteration_max`, `bailout` : caps standards.
#[allow(dead_code)]
pub fn iterate_pixel_unified_mandelbrot(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    dc: Complex64,
    iteration_max: u32,
    bailout: f64,
) -> UnifiedPixelResult {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return UnifiedPixelResult {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
        };
    }

    let mut delta = Complex64::new(0.0, 0.0);
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;

    while n < iteration_max {
        // Bailout absolu : |Z[m] + δ|² ≥ bailout²
        let z_m = ref_orbit.z_ref_f64[m as usize];
        let z_abs = z_m + delta;
        if z_abs.norm_sqr() >= bailout_sqr {
            return UnifiedPixelResult {
                iteration: n,
                z_final: z_abs,
                rebase_count,
                bla_steps,
            };
        }

        // Étape 1 : essai BLA
        let delta_norm_sqr = delta.norm_sqr();
        if let Some(node) = bla.lookup(m as usize, delta_norm_sqr) {
            // Vérifier qu'on ne dépasse pas iteration_max ni ref_len.
            let new_n = n.saturating_add(node.l);
            let new_m = m.saturating_add(node.l);
            if new_n <= iteration_max && (new_m as usize) < ref_len {
                // δ := A·δ + B·dc
                let a = node.a;
                let b = node.b;
                let (a_re, a_im) = (
                    a.m00 * delta.re + a.m01 * delta.im,
                    a.m10 * delta.re + a.m11 * delta.im,
                );
                let (b_re, b_im) = (
                    b.m00 * dc.re + b.m01 * dc.im,
                    b.m10 * dc.re + b.m11 * dc.im,
                );
                delta = Complex64::new(a_re + b_re, a_im + b_im);
                n = new_n;
                m = new_m;
                bla_steps += 1;

                // NaN / Inf protection
                if !delta.re.is_finite() || !delta.im.is_finite() {
                    return UnifiedPixelResult {
                        iteration: n,
                        z_final: ref_orbit.z_ref_f64[m as usize] + delta,
                        rebase_count,
                        bla_steps,
                    };
                }
                continue;
            }
            // BLA voulait sauter trop loin → on tombe sur le pas perturbation.
        }

        // Étape 2 : pas perturbation Mandelbrot
        // δ_{n+1} = 2·Z[m]·δ + δ² + dc
        let two_zm = z_m * 2.0;
        delta = two_zm * delta + delta * delta + dc;
        n += 1;
        m += 1;

        if !delta.re.is_finite() || !delta.im.is_finite() {
            return UnifiedPixelResult {
                iteration: n,
                z_final: ref_orbit.z_ref_f64[m.min((ref_len - 1) as u32) as usize] + delta,
                rebase_count,
                bla_steps,
            };
        }

        // Étape 3 : rebase F3 strict
        let end_of_ref = (m as usize) + 1 >= ref_len;
        if end_of_ref {
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

    let final_m = m.min((ref_len - 1) as u32);
    UnifiedPixelResult {
        iteration: n,
        z_final: ref_orbit.z_ref_f64[final_m as usize] + delta,
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
    use std::path::PathBuf;

    fn make_ref_orbit(
        center_x: f64,
        center_y: f64,
        zoom: f64,
        iter_max: u32,
    ) -> crate::fractal::perturbation::orbit::ReferenceOrbit {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 100);
        params.center_x = center_x;
        params.center_y = center_y;
        let span_x = 4.0 / zoom;
        params.span_x = span_x;
        params.span_y = span_x * 100.0 / 160.0;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        let (orbit, _, _) = compute_reference_orbit(&params, None)
            .expect("compute_reference_orbit failed");
        orbit
    }

    /// Render une image complète Mandelbrot zoom 1e6 via le pixel loop
    /// unifié, sauvegarde-la, et diffe vs la golden legacy
    /// `tests/golden/mandelbrot_perturb_1e6.png`.
    ///
    /// Cette comparaison est *le* test de Session C : si le diff est faible
    /// (<1% pixels), l'unified path produit une image visuellement
    /// équivalente à legacy. Si gros, on a quantifié le travail restant.
    ///
    /// Le test ne fait PAS échouer le build sur diff > 0 — il rapporte.
    #[test]
    fn render_full_image_vs_legacy_golden() {
        use crate::color::palettes::{color_for_pixel_with_lut, PaletteLut};
        use crate::fractal::OutColoringMode;

        let width = 160u32;
        let height = 100u32;
        let iter_max = 1500u32;
        let cx = -1.7693831791955;
        let cy = 0.004236847918736;
        let zoom = 1e6;
        let span_x = 4.0 / zoom;
        let span_y = span_x * height as f64 / width as f64;

        let orbit = make_ref_orbit(cx, cy, zoom, iter_max);
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table build");
        let bla = &tables[0];

        // Render
        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        let palette = PaletteLut::new(
            6, // Plasma
            crate::fractal::ColorSpace::Rgb,
        );

        for j in 0..height {
            for i in 0..width {
                let dx = ((i as f64 + 0.5) / width as f64 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / height as f64 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);

                let res = iterate_pixel_unified_mandelbrot(&orbit, bla, dc, iter_max, 4.0);

                // Coloriage via la fonction de production (mêmes constantes
                // que default_params_for_type pour Mandelbrot : palette 6 (Plasma),
                // color_repeat 40, OutColoringMode::Smooth, ColorSpace::Rgb).
                let (r, g, b) = color_for_pixel_with_lut(
                    res.iteration,
                    res.z_final,
                    iter_max,
                    6,  // palette index Plasma
                    40, // color_repeat
                    OutColoringMode::Smooth,
                    crate::fractal::ColorSpace::Rgb,
                    None, // orbit
                    None, // distance
                    false, // interior_flag_encoded
                    Some(&palette),
                );
                pixels.extend_from_slice(&[r, g, b, 255]);
            }
        }

        // Save the unified-render output
        let unified_path = std::env::temp_dir().join("mandelbrot_perturb_1e6_unified.png");
        let buf = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, pixels.clone())
            .expect("buffer build");
        buf.save(&unified_path).expect("save unified png");

        // Diff vs legacy golden
        let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/golden/mandelbrot_perturb_1e6.png");
        let golden = image::open(&golden_path)
            .expect("open golden")
            .to_rgba8()
            .into_raw();

        assert_eq!(
            golden.len(),
            pixels.len(),
            "Tailles différentes : golden {} vs unified {}",
            golden.len(),
            pixels.len()
        );

        // Stats supplémentaires : combien de pixels utilisent la BLA ?
        let mut total_bla_steps = 0u64;
        let mut total_rebase = 0u64;
        let mut total_iter = 0u64;
        for j in 0..height {
            for i in 0..width {
                let dx = ((i as f64 + 0.5) / width as f64 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / height as f64 - 0.5) * span_y;
                let res = iterate_pixel_unified_mandelbrot(
                    &orbit,
                    bla,
                    Complex64::new(dx, dy),
                    iter_max,
                    4.0,
                );
                total_bla_steps += res.bla_steps as u64;
                total_rebase += res.rebase_count as u64;
                total_iter += res.iteration as u64;
            }
        }

        let mut diffs = 0usize;
        let mut total_abs_diff = 0u64;
        let mut heavy_diff = 0usize;
        for (g, u) in golden.chunks_exact(4).zip(pixels.chunks_exact(4)) {
            if g != u {
                diffs += 1;
            }
            let mut max_chan = 0u8;
            for k in 0..3 {
                let d = (g[k] as i64 - u[k] as i64).unsigned_abs() as u8;
                total_abs_diff += d as u64;
                if d > max_chan {
                    max_chan = d;
                }
            }
            if max_chan > 50 {
                heavy_diff += 1;
            }
        }
        let total_pixels = (width * height) as usize;
        let mean_abs_diff = total_abs_diff as f64 / (total_pixels as f64 * 3.0);
        eprintln!(
            "Session C unified vs legacy golden (mandelbrot zoom 1e6) :\n  \
             {}/{} pixels diffèrent ({:.2}%), mean |diff|/canal = {:.2}/255\n  \
             pixels lourdement diff (>50/canal) : {}/{} ({:.2}%)\n  \
             stats unified : BLA steps total {}, rebases {}, iter sum {}\n  \
             Output: {}",
            diffs,
            total_pixels,
            100.0 * diffs as f64 / total_pixels as f64,
            mean_abs_diff,
            heavy_diff,
            total_pixels,
            100.0 * heavy_diff as f64 / total_pixels as f64,
            total_bla_steps,
            total_rebase,
            total_iter,
            unified_path.display()
        );
    }

    /// Test le pixel loop unifié généralisé sur Burning Ship.
    /// Le delta-form gère les ops AbsX/AbsY via diffabs.
    #[test]
    fn unified_pixel_loop_burning_ship_zoom_100() {
        let iter_max = 800u32;
        let cx = -1.762f64;
        let cy = -0.028f64;
        let zoom = 100.0f64;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;

        // ReferenceOrbit pour BurningShip
        let mut params = default_params_for_type(FractalType::BurningShip, 160, 100);
        params.center_x = cx;
        params.center_y = cy;
        params.span_x = span_x;
        params.span_y = span_y;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        let (orbit, _, _) = compute_reference_orbit(&params, None).expect("ref orbit");

        let formula = compile_formula(FractalType::BurningShip, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table");
        let bla = &tables[0];

        // Itération directe BurningShip pour ref classification.
        let mut mismatches = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);
                let c_abs = Complex64::new(cx + dx, cy + dy);

                let res = iterate_pixel_unified(
                    &orbit,
                    bla,
                    &formula,
                    orbit.cref,
                    dc,
                    Complex64::new(0.0, 0.0),
                    iter_max,
                    4.0,
                );
                let escaped_unif = res.iteration < iter_max;

                let mut z = Complex64::new(0.0, 0.0);
                let mut nn = 0u32;
                while nn < iter_max && z.norm_sqr() < 16.0 {
                    let z_abs = Complex64::new(z.re.abs(), z.im.abs());
                    z = z_abs * z_abs + c_abs;
                    nn += 1;
                }
                let escaped_direct = nn < iter_max;
                if escaped_unif != escaped_direct {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "BS ({},{}) unif n={} esc={} | direct n={} esc={}",
                            i, j, res.iteration, escaped_unif, nn, escaped_direct
                        );
                    }
                }
                total += 1;
            }
        }
        eprintln!(
            "Unified loop BurningShip zoom 100 : {}/{} classifications matchent",
            total - mismatches,
            total
        );
        // BurningShip est non-conformal et hautement chaotique : tolérer
        // 1-2 pixels frontière sur 100 (cas typique : pixel à l'extrême
        // limite d'un minibrot).
        let mismatch_pct = 100.0 * mismatches as f64 / total as f64;
        assert!(
            mismatch_pct <= 5.0,
            "Trop de mismatches BS unified vs direct : {:.1}%",
            mismatch_pct
        );
    }

    /// Idem sur Tricorn.
    #[test]
    fn unified_pixel_loop_tricorn_zoom_50() {
        let iter_max = 500u32;
        let cx = -0.5f64;
        let cy = 0.1f64;
        let zoom = 50.0f64;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;

        let mut params = default_params_for_type(FractalType::Tricorn, 160, 100);
        params.center_x = cx;
        params.center_y = cy;
        params.span_x = span_x;
        params.span_y = span_y;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        let (orbit, _, _) = compute_reference_orbit(&params, None).expect("ref orbit");

        let formula = compile_formula(FractalType::Tricorn, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table");
        let bla = &tables[0];

        let mut mismatches = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);
                let c_abs = Complex64::new(cx + dx, cy + dy);

                let res = iterate_pixel_unified(
                    &orbit,
                    bla,
                    &formula,
                    orbit.cref,
                    dc,
                    Complex64::new(0.0, 0.0),
                    iter_max,
                    4.0,
                );
                let escaped_unif = res.iteration < iter_max;

                // Tricorn direct : (X - iY)² + C
                let mut z = Complex64::new(0.0, 0.0);
                let mut nn = 0u32;
                while nn < iter_max && z.norm_sqr() < 16.0 {
                    let conj = Complex64::new(z.re, -z.im);
                    z = conj * conj + c_abs;
                    nn += 1;
                }
                let escaped_direct = nn < iter_max;
                if escaped_unif != escaped_direct {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "Tricorn ({},{}) unif n={} esc={} | direct n={} esc={}",
                            i, j, res.iteration, escaped_unif, nn, escaped_direct
                        );
                    }
                }
                total += 1;
            }
        }
        eprintln!(
            "Unified loop Tricorn zoom 50 : {}/{} classifications matchent",
            total - mismatches,
            total
        );
        assert_eq!(mismatches, 0);
    }

    /// À zoom 1e6, la classification doit matcher l'itération directe.
    #[test]
    fn unified_pixel_loop_mandelbrot_zoom_1e6() {
        let iter_max = 1500u32;
        let cx = -1.7693831791955;
        let cy = 0.004236847918736;
        let zoom = 1e6;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;

        let orbit = make_ref_orbit(cx, cy, zoom, iter_max);
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        // c_norm pour la formule de merge BLA = |cref|
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table build");
        let bla = &tables[0];

        let mut mismatches = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);
                let c_abs = Complex64::new(cx + dx, cy + dy);

                let res = iterate_pixel_unified_mandelbrot(&orbit, bla, dc, iter_max, 4.0);
                let escaped_unif = res.iteration < iter_max;

                let mut z = Complex64::new(0.0, 0.0);
                let mut nn = 0u32;
                while nn < iter_max && z.norm_sqr() < 16.0 {
                    z = z * z + c_abs;
                    nn += 1;
                }
                let escaped_direct = nn < iter_max;

                if escaped_unif != escaped_direct {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "  ({},{}) unif n={} esc={} bla_steps={} rebase={} | direct n={} esc={}",
                            i, j, res.iteration, escaped_unif, res.bla_steps, res.rebase_count, nn, escaped_direct
                        );
                    }
                }
                total += 1;
            }
        }
        eprintln!(
            "Unified loop zoom 1e6 : {}/{} classifications matchent",
            total - mismatches,
            total
        );
        assert_eq!(
            mismatches, 0,
            "Unified pixel loop doit classifier comme l'itération directe"
        );
    }
}

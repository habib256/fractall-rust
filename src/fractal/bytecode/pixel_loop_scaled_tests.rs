use super::*;
use crate::fractal::bytecode::compile_formula;
use crate::fractal::bytecode::pixel_loop::PixelLoopLimits;
use crate::fractal::bytecode::pixel_loop_exp::iterate_pixel_unified_exp;
use crate::fractal::perturbation::orbit::compute_reference_orbit;
use crate::fractal::view::ViewHp;
use crate::fractal::{default_params_for_type, FractalType};

#[test]
fn scaled_cexp_round_trip_is_exact() {
    let vals = [
        ComplexExp {
            re: FloatExp::new(0.75, -3000),
            im: FloatExp::new(-0.6, -3010),
        },
        ComplexExp {
            re: FloatExp::zero(),
            im: FloatExp::new(0.5, -100_000),
        },
        ComplexExp {
            re: FloatExp::new(-0.9, 7),
            im: FloatExp::zero(),
        },
        ComplexExp {
            re: FloatExp::new(0.51234567, -1500),
            im: FloatExp::new(0.99999, -2399), // écart 899 binades : accepté
        },
        ComplexExp::from_complex64(Complex64::new(1.2345e-300, -6.789e-290)),
    ];
    for v in vals {
        let s = Scaled::from_cexp(v).expect("représentable");
        assert_eq!(s.to_cexp(), v, "aller-retour exact pour {v:?}");
        let mx = s.d.re.abs().max(s.d.im.abs());
        assert!(
            (0.5..1.0).contains(&mx),
            "composante max normalisée : {s:?}"
        );
    }
    // δ = 0 et écart de composantes > 900 binades : non représentables.
    assert!(Scaled::from_cexp(ComplexExp::zero()).is_none());
    let far = ComplexExp {
        re: FloatExp::new(0.5, 0),
        im: FloatExp::new(0.5, -1000),
    };
    assert!(Scaled::from_cexp(far).is_none());
}

#[test]
fn renorm_is_exact_scaling() {
    let d = Complex64::new(3.0e30, -1.5e25);
    let s = renorm(d, -4000).expect("renorm");
    assert!(s.d.re.abs() >= D_LO && s.d.re.abs() <= D_HI);
    let a = Scaled { d, e: -4000 }.to_cexp();
    assert_eq!(s.to_cexp(), a, "renormaliser ne change pas la valeur");
    assert!(renorm(Complex64::new(0.0, 0.0), 0).is_none());
    assert!(renorm(Complex64::new(f64::NAN, 1e100), 0).is_none());
}

fn toml_field(text: &str, key: &str) -> String {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with(&format!("{key} ")))
        .unwrap_or_else(|| panic!("clé {key}"));
    line.split_once('=')
        .unwrap()
        .1
        .trim()
        .trim_matches('"')
        .to_string()
}

/// Verrou principal : sur une grille de pixels d'un cas PROFOND du corpus
/// (e401, 1e401, tier exp, réf atom-tronquée + BLA FloatExp), la boucle
/// scaled-double rend EXACTEMENT ce que rend la boucle HP `ComplexExp`
/// (itération ET z_final bit-à-bit, rebases et sauts BLA compris), et emprunte
/// bien les deux chemins (rapide et lent).
#[test]
fn scaled_matches_floatexp_loop_on_deep_grid() {
    let (n, escaped, _, total) = check_grid("e401", 64, 5);
    // Anti-vacuité : le cas exerce les trois régimes et des pixels qui
    // s'échappent.
    assert!(
        total.fast_direct > 0
            && total.fast_jump > 0
            && total.slow_iter > 0
            && total.fast_rebase > 0,
        "{total:?}"
    );
    assert!(escaped > 0 && n > 0);
}

/// Verrou du drapeau de fiabilité : olbaid2 (216 pixels non fiables / 65 536
/// à 256²). La grille DOIT contenir des pixels flaggés, que la boucle scaled
/// tracée flagge à l'identique (borne au bit près).
#[test]
fn scaled_tracked_flags_same_pixels_as_floatexp_on_olbaid2() {
    let (_, _, flagged, _) = check_grid("olbaid2", 256, 4);
    assert!(flagged > 0, "grille sans pixel non fiable : verrou vide");
}

/// Diagnostic de coût du traqueur (`FRACTALL_SCALED_TIMING=1`, cas
/// `FRACTALL_SCALED_CASE`, défaut e318) : boucle scaled nue vs tracée.
#[test]
#[ignore]
fn scaled_tracker_cost_diagnostic() {
    let stem = std::env::var("FRACTALL_SCALED_CASE").unwrap_or_else(|_| "e318".into());
    check_grid(&stem, 256, 4);
}

/// Grille (pas `step`) d'une vue `w`×`w` du cas `stem` : boucle scaled vs
/// FloatExp, non tracée ET tracée. Renvoie (n, évadés, flaggés, stats).
fn check_grid(stem: &str, w: u32, step: usize) -> (u32, u32, u32, ScaledStats) {
    let path = format!("{}/toml/{stem}.toml", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).expect("toml");
    let (re, im, zoom) = (
        toml_field(&text, "real"),
        toml_field(&text, "imag"),
        toml_field(&text, "zoom"),
    );
    let iters: u32 = toml_field(&text, "iterations").parse().unwrap();
    let h = w;
    let mut params = default_params_for_type(FractalType::Mandelbrot, w, h);
    let view = ViewHp::from_center_and_zoom(&re, &im, &zoom, w, h, 1024).expect("vue");
    view.write_to_params(&mut params);
    params.center_x_hp = Some(re);
    params.center_y_hp = Some(im);
    params.iteration_max = iters;
    params.perturbation.max_perturb_iterations = iters;
    params.perturbation.max_bla_steps = iters;
    let (orbit, _, _) = compute_reference_orbit(&params, None, false).expect("orbite");
    let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
    let (sx, sy) = crate::fractal::perturbation::effective_spans_fexp(&params);
    let px = sx
        .abs()
        .div(FloatExp::from_f64(w as f64))
        .max(sy.abs().div(FloatExp::from_f64(h as f64)));
    let diag = ((w * w + h * h) as f64).sqrt();
    let c_norm = px * diag;
    let eps = FloatExp::from_f64(1.0 / (1u64 << 53) as f64);
    let bla = BlaTableUnifiedExp::build(&orbit.z_ref, &formula.phases[0], c_norm, eps);
    // Table f64 factice : le path HP ne la lit pas.
    let bla_f64 = crate::fractal::bytecode::bla_dual::BlaTableUnified { levels: Vec::new() };
    let limits = PixelLoopLimits::from(&params);

    let mut total = ScaledStats::default();
    let mut escaped = 0;
    let mut flagged = 0u32;
    let eps_f64 = 1.0 / (1u64 << 53) as f64;
    let mut n = 0;
    for j in (0..h).step_by(step) {
        for i in (0..w).step_by(step) {
            let fx = (i as f64 + 0.5) / w as f64 - 0.5;
            let fy = (j as f64 + 0.5) / h as f64 - 0.5;
            let dc = ComplexExp {
                re: sx * fx,
                im: sy * fy,
            };
            let reference = iterate_pixel_unified_exp(
                &orbit,
                &bla_f64,
                Some(&bla),
                &formula,
                orbit.cref,
                dc,
                ComplexExp::zero(),
                limits,
            );
            let (got, st) =
                iterate_pixel_scaled_with_stats(&orbit, Some(&bla), dc, ComplexExp::zero(), limits);
            assert_eq!(
                got.iteration, reference.iteration,
                "pixel ({i},{j}) itération"
            );
            assert_eq!(
                (got.z_final.re.to_bits(), got.z_final.im.to_bits()),
                (
                    reference.z_final.re.to_bits(),
                    reference.z_final.im.to_bits()
                ),
                "pixel ({i},{j}) z_final"
            );
            assert_eq!(
                got.rebase_count, reference.rebase_count,
                "pixel ({i},{j}) rebases"
            );
            assert_eq!(got.bla_steps, reference.bla_steps, "pixel ({i},{j}) sauts");

            // Variante TRACÉE (détecteur de fiabilité, défaut) : même résultat
            // ET même borne au bit près ⇒ même drapeau fiable / non fiable.
            let (rt, tr) = crate::fractal::bytecode::pixel_loop_exp::iterate_pixel_unified_exp_mandelbrot_hp_tracked(
                &orbit, Some(&bla), dc, ComplexExp::zero(), limits, eps_f64,
            );
            let (gt, tg) = iterate_pixel_scaled_exp_mandelbrot_tracked(
                &orbit,
                Some(&bla),
                dc,
                ComplexExp::zero(),
                limits,
                eps_f64,
            );
            assert_eq!(gt.iteration, rt.iteration, "tracé ({i},{j}) itération");
            assert_eq!(
                (gt.z_final.re.to_bits(), gt.z_final.im.to_bits()),
                (rt.z_final.re.to_bits(), rt.z_final.im.to_bits()),
                "tracé ({i},{j}) z_final"
            );
            // Traqueur léger : borne à quelques ulp relatifs près (pas au bit),
            // décision identique (vérifiée ci-dessous).
            let (bg, br) = (tg.err.to_f64(), tr.err.to_f64());
            let close = tg.err.exponent.abs_diff(tr.err.exponent) <= 1
                && ((bg - br).abs() <= 1e-9 * br.abs() || tg.err.exponent > 900);
            assert!(close, "tracé ({i},{j}) borne {:?} vs {:?}", tg.err, tr.err);
            let flag = |b: f32, z| {
                crate::fractal::perturbation::delta::escape_undetermined(b, z, limits.bailout)
                    >= crate::fractal::bytecode::reliability::reliability_runaway() as f32
            };
            let f_ref = flag(tr.error_bound(), rt.z_final);
            assert_eq!(
                flag(tg.error_bound(), gt.z_final),
                f_ref,
                "tracé ({i},{j}) drapeau"
            );
            flagged += f_ref as u32;
            total.fast_direct += st.fast_direct;
            total.fast_jump += st.fast_jump;
            total.slow_iter += st.slow_iter;
            total.fast_rebase += st.fast_rebase;
            escaped += (got.iteration < iters) as u32;
            n += 1;
        }
    }
    if std::env::var_os("FRACTALL_SCALED_TIMING").is_some() {
        let dcs: Vec<ComplexExp> = (0..h)
            .step_by(step)
            .flat_map(|j| (0..w).step_by(step).map(move |i| (i, j)))
            .map(|(i, j)| ComplexExp {
                re: sx * ((i as f64 + 0.5) / w as f64 - 0.5),
                im: sy * ((j as f64 + 0.5) / h as f64 - 0.5),
            })
            .collect();
        let z0 = ComplexExp::zero();
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let mut acc = 0u64;
            for &dc in &dcs {
                acc += iterate_pixel_scaled_exp_mandelbrot(&orbit, Some(&bla), dc, z0, limits)
                    .iteration as u64;
            }
            let t_plain = t.elapsed();
            let t = std::time::Instant::now();
            for &dc in &dcs {
                acc += iterate_pixel_scaled_exp_mandelbrot_tracked(
                    &orbit,
                    Some(&bla),
                    dc,
                    z0,
                    limits,
                    eps_f64,
                )
                .0
                .iteration as u64;
            }
            let t_trk = t.elapsed();
            eprintln!("timing {stem}: plain {t_plain:?} tracked {t_trk:?} ({acc})");
        }
    }
    eprintln!(
        "scaled grid {stem}: n={n} escaped={escaped} flagged={flagged} ref_len={} {total:?}",
        orbit.z_ref.len()
    );
    (n, escaped, flagged, total)
}

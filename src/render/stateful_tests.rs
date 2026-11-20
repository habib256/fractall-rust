//! Tests DIFFÉRENTIELS des chemins interactifs À ÉTAT (cache d'orbite threadé
//! entre requêtes, comme la GUI) : une séquence de rendus passe par le
//! dispatcher unique avec le MÊME `orbit_cache`, puis la dernière frame est
//! comparée à un rendu frais (cache `None`) de la même vue. CLI / quality /
//! harness rendent toujours sans cache : ces défauts leur sont invisibles.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use rug::Float;

use crate::fractal::definitions::default_params_for_type;
use crate::fractal::gmp::{complex_from_xy, iterate_point_mpc, MpcParams};
use crate::fractal::perturbation::ReferenceOrbitCache;
use crate::fractal::{AlgorithmMode, FractalParams, FractalType};
use crate::render::{render_request, RenderOutput, RenderRequest};

type Cache = Option<Arc<ReferenceOrbitCache>>;

const SEAHORSE_X: &str = "-0.743643887037158704752191506114774";
const SEAHORSE_Y: &str = "0.131825904205311970493132056385139";

/// Vue de type `ty`, centre HP `(cx, cy)` décalé de `(dx_px, dy_px)` pixels,
/// span horizontal `span_x` (chaîne HP), `w×h`, plafond `iters`.
#[allow(clippy::too_many_arguments)]
fn view(
    ty: FractalType,
    cx: &str,
    cy: &str,
    dx_px: f64,
    dy_px: f64,
    span_x: &str,
    (w, h): (u32, u32),
    iters: u32,
) -> FractalParams {
    let prec = 4096;
    let mut p = default_params_for_type(ty, w, h);
    let sx = Float::with_val(prec, Float::parse(span_x).unwrap());
    let sy = Float::with_val(prec, &sx * h) / w;
    let px = Float::with_val(prec, &sx / w);
    let cxf = Float::with_val(prec, Float::parse(cx).unwrap()) + Float::with_val(prec, &px * dx_px);
    let cyf = Float::with_val(prec, Float::parse(cy).unwrap()) + Float::with_val(prec, &px * dy_px);
    p.iteration_max = iters;
    p.center_x = cxf.to_f64();
    p.center_y = cyf.to_f64();
    p.center_x_hp = Some(cxf.to_string_radix(10, Some(400)));
    p.center_y_hp = Some(cyf.to_string_radix(10, Some(400)));
    p.span_x = sx.to_f64();
    p.span_y = sy.to_f64();
    p.span_x_hp = Some(sx.to_string_radix(10, Some(40)));
    p.span_y_hp = Some(sy.to_string_radix(10, Some(40)));
    p
}

fn render(p: &FractalParams, cache: &mut Cache) -> RenderOutput {
    let cancel = Arc::new(AtomicBool::new(false));
    render_request(RenderRequest::new(p, &cancel), cache).expect("rendu")
}

fn fresh(p: &FractalParams) -> RenderOutput {
    render(p, &mut None)
}

fn diff_indices(a: &RenderOutput, b: &RenderOutput) -> Vec<usize> {
    assert_eq!(a.iterations.len(), b.iterations.len(), "tailles différentes");
    a.iterations
        .iter()
        .zip(&b.iterations)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect()
}

/// Compte du GMP pur (juge) au pixel `idx` de la vue `p` (sans rotation).
fn gmp_count(p: &FractalParams, idx: usize) -> u32 {
    let mut q = p.clone();
    q.engine.algorithm_mode = AlgorithmMode::Perturbation; // précision perturbation
    let g = MpcParams::from_params(&q);
    let prec = g.prec.max(256) + 64;
    let g = MpcParams { prec, ..g };
    let (w, h) = (p.width as usize, p.height as usize);
    let (i, j) = (idx % w, idx / w);
    let parse = |s: &Option<String>, f: f64| match s {
        Some(s) => Float::with_val(prec, Float::parse(s).unwrap()),
        None => Float::with_val(prec, f),
    };
    let sx = parse(&p.span_x_hp, p.span_x);
    let sy = parse(&p.span_y_hp, p.span_y);
    let fx = (i as f64 + 0.5) / w as f64 - 0.5;
    let fy = (j as f64 + 0.5) / h as f64 - 0.5;
    let re = parse(&p.center_x_hp, p.center_x) + Float::with_val(prec, &sx * fx);
    let im = parse(&p.center_y_hp, p.center_y) + Float::with_val(prec, &sy * fy);
    let g = MpcParams {
        seed: rug::Complex::with_val(prec, (p.seed.re, p.seed.im)),
        ..g
    };
    iterate_point_mpc(&g, &complex_from_xy(prec, re, im)).0
}

/// Juge les pixels où `stateful` ≠ `fresh` contre le GMP pur : renvoie le
/// nombre de pixels où le chemin à état est FAUX alors que le frais est juste.
fn stateful_only_wrong(p: &FractalParams, stateful: &RenderOutput, fresh: &RenderOutput) -> usize {
    diff_indices(stateful, fresh)
        .into_iter()
        .filter(|&idx| {
            let truth = gmp_count(p, idx);
            stateful.iterations[idx] != truth && fresh.iterations[idx] == truth
        })
        .count()
}

/// Anti-vacuité : la vue a de la structure (assez de comptes distincts).
fn assert_structured(o: &RenderOutput, label: &str) {
    let mut v = o.iterations.clone();
    v.sort_unstable();
    v.dedup();
    assert!(v.len() >= 20, "{label} : vue sans structure ({} comptes distincts)", v.len());
}

const JULIA_X: &str = "1.42279534109440317773666336881783446850630340e-1";
const JULIA_Y: &str = "5.17774645170123477959332321568757233266599329e-1";

const E30_X: &str = "-0.0494700290631040937516922267273536301187457124882248793181049402326421947726869034279915499747594190000000000000000000";
const E30_Y: &str = "-0.6747875758446753640113920531305976563347707068224034806979997947909941983454845111514208499540310299999999999999999880";

fn e30(dx: f64, span: &str, size: (u32, u32), iters: u32) -> FractalParams {
    let mut p = view(FractalType::Mandelbrot, E30_X, E30_Y, dx, 0.0, span, size, iters);
    p.engine.algorithm_mode = AlgorithmMode::Perturbation;
    p
}

/// Vue profonde structurée : e30 (span ≥ 1e-280) ou e318 (tier exp).
fn deep_scene(span: &str, size: (u32, u32), iters: u32) -> FractalParams {
    let exp = Float::with_val(64, Float::parse(span).unwrap()) < 1e-280;
    let (x, y) = if exp { (E318_X, E318_Y) } else { (E30_X, E30_Y) };
    let mut p = view(FractalType::Mandelbrot, x, y, 0.0, 0.0, span, size, iters);
    p.engine.algorithm_mode = AlgorithmMode::Perturbation;
    p
}

fn seahorse(dx: f64, span: &str, size: (u32, u32), iters: u32) -> FractalParams {
    let mut p = view(FractalType::Mandelbrot, SEAHORSE_X, SEAHORSE_Y, dx, 0.0, span, size, iters);
    p.engine.algorithm_mode = AlgorithmMode::Perturbation;
    p
}

// ── 1. Plafond monté / redescendu, orbite prolongée ─────────────────────────

fn cap_sequence(span: &str, caps: &[u32]) {
    let size = (64, 40);
    let mut cache: Cache = None;
    let mut last = None;
    for &c in caps {
        let p = deep_scene(span, size, c);
        last = Some((render(&p, &mut cache), p));
    }
    let (stateful, p) = last.unwrap();
    let f = fresh(&p);
    assert_structured(&f, "frais");
    assert!(
        stateful.iterations.iter().all(|&i| i <= p.iteration_max),
        "compte au-delà du plafond demandé ({caps:?}, span {span})"
    );
    let d = diff_indices(&stateful, &f);
    assert!(d.len() <= 2, "séquence de plafonds {caps:?} span {span} : {} pixels ≠ frais", d.len());
}

#[test]
fn cap_up_then_down_deep_matches_fresh() {
    cap_sequence("4e-25", &[16000, 30000, 17000]);
}

#[test]
fn cap_down_then_up_deep_matches_fresh() {
    cap_sequence("4e-25", &[30000, 15000, 18000]);
}

#[test]
fn cap_up_then_down_exp_tier_matches_fresh() {
    cap_sequence("4e-300", &[88000, 120000, 88200]);
}

#[test]
fn cap_down_then_up_exp_tier_matches_fresh() {
    cap_sequence("4e-300", &[120000, 87800, 88300]);
}

// ── 2. Pan à profondeur fixe puis changement de plafond ─────────────────────

#[test]
fn pan_then_cap_change_with_reused_orbit_is_correct() {
    let size = (64, 40);
    let span = "4e-12";
    let mut cache: Cache = None;
    render(&seahorse(0.0, span, size, 3000), &mut cache);
    // Zoom ×2 ancré à côté du centre (sous-ensemble), puis plafond baissé et
    // remonté sans bouger.
    let half = "2e-12";
    let p1 = seahorse(6.0, half, size, 3000);
    render(&p1, &mut cache);
    render(&seahorse(6.0, half, size, 1500), &mut cache);
    let p = seahorse(6.0, half, size, 2500);
    let s = render(&p, &mut cache);
    let f = fresh(&p);
    assert_structured(&f, "frais");
    assert!(s.iterations.iter().all(|&i| i <= p.iteration_max));
    let bad = stateful_only_wrong(&p, &s, &f);
    assert!(bad <= 1, "pan + plafond, orbite réutilisée : {bad} pixels faux (frais juste)");
}

// ── 3. Taille / rapport de fenêtre changés, cache gardé ─────────────────────

#[test]
fn resize_with_kept_cache_matches_fresh() {
    let mut cache: Cache = None;
    render(&e30(0.0, "4e-25", (64, 40), 20000), &mut cache);
    // Même span vertical, autre largeur (rapport) puis autre taille.
    let p0 = e30(0.0, "4e-25", (64, 40), 20000);
    let sy = p0.span_y_hp.clone().unwrap();
    for (w, h) in [(80u32, 40u32), (96, 60)] {
        let sy_f = Float::with_val(512, Float::parse(&sy).unwrap());
        let sx = Float::with_val(512, &sy_f * w) / h;
        let sx = sx.to_string_radix(10, Some(40));
        let p = e30(0.0, &sx, (w, h), 20000);
        let s = render(&p, &mut cache);
        let f = fresh(&p);
        assert_structured(&f, "frais");
        let d = diff_indices(&s, &f);
        assert!(d.len() <= 2, "redimensionnement {w}×{h} : {} pixels ≠ frais", d.len());
    }
}

// ── 4. Réglages moteur basculés à vue constante, cache gardé ────────────────

fn toggle_case(label: &str, edit: impl Fn(&mut FractalParams)) {
    let base = e30(0.0, "4e-25", (64, 40), 20000);
    let mut cache: Cache = None;
    let b = render(&base, &mut cache);
    let before = cache.clone().unwrap();
    let mut p = base.clone();
    edit(&mut p);
    crate::render::ensure_required_channels(&mut p);
    let s = render(&p, &mut cache);
    let f = fresh(&p);
    assert_structured(&f, "frais");
    let d = diff_indices(&s, &f);
    let after = cache.clone().unwrap();
    eprintln!(
        "[STATEFUL] {label}: reused={} has_dd={} fresh_vs_base={} stateful_vs_fresh={} stateful_vs_base={}",
        Arc::ptr_eq(&before, &after),
        after.orbit.has_dd(),
        diff_indices(&f, &b).len(),
        d.len(),
        diff_indices(&s, &b).len()
    );
    assert!(d.len() <= 2, "{label} basculé, cache gardé : {} pixels ≠ frais", d.len());
    // Et la distance, quand elle est produite.
    assert_eq!(s.distances.len(), f.distances.len(), "{label} : canal distances");
    // Tier dd demandé : la frame doit être rendue avec une orbite dd (sinon
    // le réglage est ignoré sans avertissement, même si cette scène-ci rend
    // les mêmes comptes en f64).
    if crate::fractal::wisdom::dd_requested(&p) {
        assert!(after.orbit.has_dd(), "{label} : orbite sans dd réutilisée pour une frame dd");
    }
}

#[test]
fn toggling_rotation_with_kept_cache_matches_fresh() {
    toggle_case("rotation", |p| p.rotation = 30.0);
}

#[test]
fn toggling_transform_k_with_kept_cache_matches_fresh() {
    toggle_case("transform_k", |p| p.transform_k = Some([1.0, 0.3, 0.0, 1.0]));
}

#[test]
fn toggling_find_nucleus_with_kept_cache_matches_fresh() {
    toggle_case("find_nucleus", |p| p.engine.find_nucleus = true);
}

/// Même vue avec `find_nucleus`, rendue deux fois avec le cache threadé (passes
/// GUI, sonde puis rendu final) : le centre du cache est le NUCLEUS, pas celui
/// de la vue — il était rejeté à chaque passe (recherche + orbite refaites).
#[test]
fn same_view_with_find_nucleus_reuses_orbit_and_matches_fresh() {
    let mut p = e30(0.0, "4e-25", (64, 40), 20000);
    p.engine.find_nucleus = true;
    let mut cache: Cache = None;
    let t = std::time::Instant::now();
    let first = render(&p, &mut cache);
    let t_first = t.elapsed();
    let built = cache.clone().unwrap();
    let key = built.nucleus_search.clone().expect("recherche du nucleus enregistrée");
    assert!(key.period.is_some(), "scène sans nucleus trouvé : test vide");
    let t = std::time::Instant::now();
    let second = render(&p, &mut cache);
    eprintln!(
        "[STATEFUL] find_nucleus même vue : 1er rendu {:.3}s, 2e {:.3}s (période {:?})",
        t_first.as_secs_f64(),
        t.elapsed().as_secs_f64(),
        key.period
    );
    assert!(
        Arc::ptr_eq(&built, cache.as_ref().unwrap()),
        "même vue + find_nucleus : orbite recalculée au lieu d'être réutilisée"
    );
    let f = fresh(&p);
    assert_structured(&f, "frais");
    assert_eq!(first.iterations, second.iterations, "second rendu ≠ premier");
    assert_eq!(second.iterations, f.iterations, "rendu sur cache ≠ frais");
}

/// Pan / zoom après un rendu `find_nucleus` : la recherche dépend du centre
/// et de l'échelle de vue — le cache ne doit pas être resservi tel quel.
#[test]
fn pan_or_zoom_after_find_nucleus_matches_fresh() {
    let mut base = e30(0.0, "4e-25", (64, 40), 20000);
    base.engine.find_nucleus = true;
    for (label, mut p) in [
        ("pan 5 px", e30(5.0, "4e-25", (64, 40), 20000)),
        ("zoom ×2", e30(0.0, "2e-25", (64, 40), 20000)),
    ] {
        p.engine.find_nucleus = true;
        let mut cache: Cache = None;
        render(&base, &mut cache);
        let built = cache.clone().unwrap();
        let s = render(&p, &mut cache);
        assert!(!Arc::ptr_eq(&built, cache.as_ref().unwrap()), "{label} : cache resservi");
        let f = fresh(&p);
        assert_structured(&f, label);
        assert_eq!(s.iterations, f.iterations, "{label} : ≠ frais");
    }
}

#[test]
fn toggling_dd_tier_with_kept_cache_matches_fresh() {
    toggle_case("use_dd_tier", |p| p.engine.use_dd_tier = true);
}

#[test]
fn toggling_distance_estimation_with_kept_cache_matches_fresh() {
    toggle_case("distance", |p| p.channels.enable_distance_estimation = true);
}

// ── 5. Sonde d'itérations puis rendu final avec le cache de la sonde ────────

fn probe_then_render(p0: &FractalParams, label: &str) {
    use crate::fractal::wisdom::{auto_iterations_with, AutoIterOptions};
    let cancel = Arc::new(AtomicBool::new(false));
    let mut cache: Cache = None;
    let r = auto_iterations_with(p0, &cancel, AutoIterOptions::default(), &mut cache)
        .expect("sonde");
    let mut p = p0.clone();
    p.iteration_max = r.iteration_max;
    let s = render(&p, &mut cache);
    let f = fresh(&p);
    assert_structured(&f, label);
    let d = diff_indices(&s, &f);
    let bad = stateful_only_wrong(&p, &s, &f);
    eprintln!("[STATEFUL] probe {label}: N={} diff={} stateful_only_wrong={bad}", p.iteration_max, d.len());
    assert!(bad == 0 && d.len() <= 2, "{label} : sonde puis rendu ≠ frais ({} px, {bad} faux)", d.len());
}

#[test]
fn probe_cache_then_final_render_matches_fresh_deep() {
    let mut p = e30(0.0, "4e-25", (128, 80), 1000);
    p.engine.algorithm_mode = AlgorithmMode::Auto;
    probe_then_render(&p, "e30 4e-25");
}

#[test]
fn probe_cache_then_final_render_matches_fresh_std_path() {
    // Seahorse 1e8 : path f64 standard + correction par perturbation dont la
    // référence passe par le cache d'orbite.
    let mut p = seahorse(0.0, "4e-8", (128, 80), 1000);
    p.engine.algorithm_mode = AlgorithmMode::Auto;
    probe_then_render(&p, "seahorse 4e-8 std");
}

// ── 5b. Zoom arrière d'une vue profonde (perturbation) vers le path f64 ─────

#[test]
fn zoom_out_from_perturbation_to_std_path_with_kept_cache_matches_fresh() {
    let size = (64, 40);
    let mut cache: Cache = None;
    // Vue profonde (régime atom-domain), même centre, puis zooms arrière
    // jusqu'au path standard, plafond constant.
    for span in ["4e-13", "4e-11", "4e-9"] {
        let mut p = seahorse(0.0, span, size, 4000);
        p.engine.algorithm_mode = AlgorithmMode::Auto;
        render(&p, &mut cache);
    }
    let mut p = seahorse(0.0, "4e-8", size, 4000);
    p.engine.algorithm_mode = AlgorithmMode::Auto;
    let s = render(&p, &mut cache);
    let f = fresh(&p);
    assert_structured(&f, "seahorse 4e-8");
    let d = diff_indices(&s, &f);
    let bad = stateful_only_wrong(&p, &s, &f);
    assert!(bad == 0 && d.len() <= 2, "zoom arrière vers std : {} px ≠ frais, {bad} faux", d.len());
}

// ── 5c. Écho XaoS (pan entier + zoom ×2 ancré) avec cache d'orbite ──────────

fn xaos_sequence(mk: impl Fn(f64, &str) -> FractalParams, span: &str, half: &str, label: &str) {
    use crate::fractal::xaos::{build_map, build_refine_map, params_fingerprint, view_strings, XaosSourceFrame};
    let cancel = Arc::new(AtomicBool::new(false));
    let mut cache: Cache = None;
    let src_frame = |p: &FractalParams, o: &RenderOutput| {
        XaosSourceFrame::exact(
            Arc::new(o.iterations.clone()),
            Arc::new(o.zs.clone()),
            p.width,
            p.height,
            view_strings(p),
            params_fingerprint(p),
        )
    };
    let a = mk(0.0, span);
    let oa = render(&a, &mut cache);
    // Pan de 7 px exactement : copie des colonnes communes.
    let b = mk(7.0, span);
    let map = build_map(&src_frame(&a, &oa), &b).expect("map de pan");
    assert!(map.any_reuse());
    let ob = render_request(RenderRequest::new(&b, &cancel).with_xaos(&map), &mut cache).unwrap();
    let fb = fresh(&b);
    assert_structured(&fb, label);
    let bad = stateful_only_wrong(&b, &ob, &fb);
    let d = diff_indices(&ob, &fb).len();
    assert!(bad == 0 && d <= 2, "{label} pan XaoS : {d} px ≠ frais, {bad} faux");
    // Zoom ×2 ancré (écho) puis raffinement exact.
    let c = mk(7.0 + 5.0, half);
    let map = build_map(&src_frame(&b, &ob), &c).expect("map de zoom");
    let oc = render_request(RenderRequest::new(&c, &cancel).with_xaos(&map), &mut cache).unwrap();
    let src = XaosSourceFrame::from_map(
        Arc::new(oc.iterations.clone()),
        Arc::new(oc.zs.clone()),
        c.width,
        c.height,
        view_strings(&c),
        params_fingerprint(&c),
        &map,
    );
    let refined = match build_refine_map(&src, &c) {
        Some(m) => render_request(RenderRequest::new(&c, &cancel).with_xaos(&m), &mut cache).unwrap(),
        None => oc,
    };
    let fc = fresh(&c);
    let bad = stateful_only_wrong(&c, &refined, &fc);
    let d = diff_indices(&refined, &fc).len();
    eprintln!("[STATEFUL] xaos {label}: zoom+refine diff={d} bad={bad}");
    assert!(bad == 0 && d <= 2, "{label} zoom XaoS + raffinement : {d} px ≠ frais, {bad} faux");
}

#[test]
fn xaos_pan_and_zoom_with_orbit_cache_std_path() {
    // `mk(dx, span)` : `dx` en pixels de la vue `span` ; zoom ancré traité
    // comme un recentrage (centre décalé) — la carte XaoS gère la géométrie.
    let mk = |dx: f64, span: &str| {
        let mut p = seahorse(dx, span, (64, 40), 4000);
        p.engine.algorithm_mode = AlgorithmMode::Auto;
        p
    };
    xaos_sequence(mk, "4e-8", "2e-8", "seahorse std");
}

#[test]
fn xaos_pan_and_zoom_with_orbit_cache_deep() {
    let mk = |dx: f64, span: &str| e30(dx, span, (64, 40), 20000);
    xaos_sequence(mk, "4e-25", "2e-25", "e30 profond");
}

// ── 6. Julia (Z[0] ≠ 0) : pan avec orbite réutilisée ────────────────────────

#[test]
fn julia_pan_with_reused_orbit_is_correct() {
    let size = (64, 40);
    let mk = |dx: f64, span: &str, iters: u32| {
        let mut p = view(FractalType::Julia, JULIA_X, JULIA_Y, dx, 0.0, span, size, iters);
        p.seed.re = -0.8;
        p.seed.im = 0.156;
        p.engine.algorithm_mode = AlgorithmMode::Perturbation;
        p
    };
    let mut cache: Cache = None;
    render(&mk(0.0, "4e-12", 8192), &mut cache);
    let p = mk(5.0, "2e-12", 8192);
    let s = render(&p, &mut cache);
    let f = fresh(&p);
    assert_structured(&f, "frais");
    let bad = stateful_only_wrong(&p, &s, &f);
    assert!(bad <= 1, "Julia pan, orbite réutilisée : {bad} pixels faux (frais juste)");
}

/// Graine Julia changée de quelques ulp à vue profonde, cache gardé : la
/// tolérance de 1e-15 du cache tenait les deux graines pour égales alors que
/// la vue (span 4e-20) les distingue.
#[test]
fn julia_seed_nudge_with_kept_cache_matches_fresh() {
    let mk = |seed_re: f64| {
        let mut p = view(FractalType::Julia, JULIA_X, JULIA_Y, 0.0, 0.0, "4e-15", (64, 40), 8192);
        p.seed.re = seed_re;
        p.seed.im = 0.156;
        p.engine.algorithm_mode = AlgorithmMode::Perturbation;
        p
    };
    let a = mk(-0.8);
    let mut cache: Cache = None;
    let oa = render(&a, &mut cache);
    let nudged = f64::from_bits((-0.8f64).to_bits() + 4); // ~4e-16 plus loin
    let b = mk(nudged);
    let s = render(&b, &mut cache);
    let f = fresh(&b);
    assert_structured(&f, "Julia 4e-15");
    assert!(diff_indices(&f, &oa).len() > 100, "anti-vacuité : la graine décalée doit changer la vue");
    let d = diff_indices(&s, &f).len();
    assert!(d <= 2, "graine Julia décalée, cache gardé : {d} px ≠ frais");
}

/// Hybrides en perturbation FORCÉE (menu Tech de la GUI) hors régime
/// atom-domain : zoom ×2 ancré hors centre ⇒ réutilisation subset de la
/// référence (décalage `centre − cref` dans les boucles multi-phase).
#[test]
fn hybrid_offcentre_subset_reuse_matches_fresh() {
    for (k, phases) in [
        vec![FractalType::Mandelbrot, FractalType::Mandelbrot],
        vec![FractalType::Mandelbrot, FractalType::BurningShip],
    ]
    .into_iter()
    .enumerate()
    {
        let mk = |dx: f64, span: &str| {
            let mut p = seahorse(dx, span, (64, 40), 3000);
            p.formula.hybrid_phases = Some(phases.clone());
            p
        };
        let mut cache: Cache = None;
        render(&mk(0.0, "3.2e-11"), &mut cache);
        let before = cache.clone().unwrap();
        let p = mk(10.0, "1.6e-11");
        let s = render(&p, &mut cache);
        assert!(Arc::ptr_eq(&before, cache.as_ref().unwrap()), "référence non réutilisée (k={k})");
        let f = fresh(&p);
        if k == 0 {
            assert_structured(&f, "[M,M] seahorse");
        }
        let d = diff_indices(&s, &f).len();
        // Hybrides non suivis (pas de traqueur) : la dépendance à la
        // référence au plancher f64 fait différer quelques pixels de bord.
        // [M,M] == z²+c : juge GMP pur ; [M,BS] : borne sur l'écart seul.
        // Symétrie : le rendu frais a autant de pixels faux que le rendu à
        // état ⇒ bruit de référence, pas un défaut de l'état.
        let (bad, rev) = if k == 0 {
            (stateful_only_wrong(&p, &s, &f), stateful_only_wrong(&p, &f, &s))
        } else {
            (0, 0)
        };
        eprintln!("[STATEFUL] hybrid subset k={k}: diff={d} stateful_only_wrong={bad} fresh_only_wrong={rev}");
        assert!(bad <= rev + 1 && d <= 20, "hybride {phases:?} subset hors centre : {d} px ≠ frais, {bad} faux");
    }
}

// ── 7. Hybrides / opcodes : changement de séquence, cache gardé ─────────────

#[test]
fn changing_hybrid_sequence_with_kept_cache_matches_fresh() {
    let base = e30(0.0, "4e-25", (64, 40), 20000);
    let mut cache: Cache = None;
    let mut seqs: Vec<FractalParams> = Vec::new();
    let mut a = base.clone();
    a.formula.hybrid_phases = Some(vec![FractalType::Mandelbrot, FractalType::BurningShip]);
    seqs.push(a);
    let mut b = base.clone();
    b.formula.hybrid_phases = Some(vec![FractalType::Mandelbrot, FractalType::Mandelbrot]);
    seqs.push(b);
    let mut c = base.clone();
    c.formula.hybrid_opcodes = Some("sqr rot{30} add".into());
    seqs.push(c);
    seqs.push(base.clone());
    for (k, p) in seqs.iter().enumerate() {
        let s = render(p, &mut cache);
        let f = fresh(p);
        if k == 1 || k == 3 {
            assert_structured(&f, "frais"); // [M,BS] et rot30 y sont uniformes
        }
        let d = diff_indices(&s, &f);
        assert!(d.len() <= 2, "séquence hybride n° {k} : {} pixels ≠ frais", d.len());
    }
}

const E318_X: &str = "-0.160671574544376235234299806562325714463713365698322466075294476769843023909577654790769037756572078750948504637062800069784283006963539070396938476002191397367130415091653379715775734728331195648418196184720437315573495294645574770752080340338379190322084091695814872716225514834195509029187461868448533384966001341735332278497598401959";
const E318_Y: &str = "1.0369425716116291171688106641088455988538567298124323080597903890054138172127708543024909014955338074002178250818737097944716605548909996311034876508441714228244078222420638636408073445850546946976971476583353238902945581912155462907695065826318848338789529151847288915828593504988849176913597129353591778847015303039106385284417320876575";

#[test]
#[ignore]
fn scan_scenes() {
    let size = (64, 40);
    let mut cands: Vec<(&str, FractalParams)> = Vec::new();
    for (span, it) in [("4e-25", 20000u32), ("4e-30", 50000), ("4e-30", 131072)] {
        cands.push(("e30", e30(0.0, span, size, it)));
    }
    for (span, it) in [("4e-300", 100000u32), ("4e-300", 220000)] {
        let mut p = view(FractalType::Mandelbrot, E318_X, E318_Y, 0.0, 0.0, span, size, it);
        p.engine.algorithm_mode = AlgorithmMode::Perturbation;
        cands.push(("e318", p));
    }
    for span in ["4e-12", "4e-20"] {
        let mut p = view(FractalType::Julia, "1.42279534109440317773666336881783446850630340e-1", "5.17774645170123477959332321568757233266599329e-1", 0.0, 0.0, span, size, 8192);
        p.seed.re = -0.8;
        p.seed.im = 0.156;
        p.engine.algorithm_mode = AlgorithmMode::Perturbation;
        cands.push(("julia", p));
    }
    for span in ["4.3e-13", "8.6e-13", "4.3e-11", "4.3e-9"] {
        let mut h = view(FractalType::Mandelbrot, "-0.474404914780100630144032921810699588477366255144032921810700", "-0.632738206821761938443072702331961591220850480109739368998629", 0.0, 0.0, span, size, 6000);
        h.formula.hybrid_phases = Some(vec![FractalType::Mandelbrot, FractalType::BurningShip]);
        h.engine.algorithm_mode = AlgorithmMode::Perturbation;
        cands.push(("hyb", h));
    }
    for (n, p) in cands {
        let t = std::time::Instant::now();
        let o = fresh(&p);
        let mut v = o.iterations.clone();
        v.sort_unstable();
        let capped = v.iter().filter(|&&x| x >= p.iteration_max).count();
        let maxesc = v.iter().filter(|&&x| x < p.iteration_max).max().copied().unwrap_or(0);
        let minesc = v[0];
        v.dedup();
        eprintln!("[SCAN] {n} span={:?} it={} distinct={} capped={} min={} maxesc={} t={:.2}s", p.span_x_hp, p.iteration_max, v.len(), capped, minesc, maxesc, t.elapsed().as_secs_f64());
    }
}

/// Auto-ajustement de la série (`ReferenceOrbitCache::requested_iteration_max`
/// < `iteration_max`) : scène où il se déclenche réellement — point de
/// Misiurewicz c = i, span 4e-10, 640×32 (image « non petite »), plafond 40
/// relevé à 160 (skip validé 21 > 25 % de 40). Un rendu frais à 64 s'arrête,
/// lui, à 128 : le cache ne connaît l'issue de l'ajustement QUE pour le plafond
/// demandé à sa construction. Séquence : même plafond, intermédiaires,
/// supérieur — chaque rendu sur cache == frais.
#[test]
fn auto_adjusted_cache_serves_only_its_requested_cap() {
    let mk = |it: u32| {
        let mut p = view(FractalType::Mandelbrot, "0", "1", 0.0, 0.0, "4e-10", (640, 32), it);
        p.engine.algorithm_mode = AlgorithmMode::Perturbation;
        p
    };
    let mut cache: Cache = None;
    render(&mk(40), &mut cache);
    let c = cache.clone().unwrap();
    assert!(
        c.iteration_max > c.requested_iteration_max,
        "scène sans auto-ajustement : test vide ({} / {})",
        c.iteration_max,
        c.requested_iteration_max
    );
    for it in [40u32, 64, 100, 200, 40] {
        let p = mk(it);
        let s = render(&p, &mut cache);
        let f = fresh(&p);
        assert_structured(&f, "frais");
        let d = diff_indices(&s, &f);
        assert!(
            d.is_empty(),
            "plafond {it} après cache auto-ajusté : {} pixels ≠ frais (max sur cache {:?}, frais {:?}, cache {}/{})",
            d.len(),
            s.iterations.iter().max(),
            f.iterations.iter().max(),
            cache.as_ref().unwrap().requested_iteration_max,
            cache.as_ref().unwrap().iteration_max
        );
    }
}

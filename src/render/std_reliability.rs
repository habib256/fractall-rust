//! Fiabilité du path f64 STANDARD (chantier de correction n° 1, 2026-11-18).
//!
//! Le mode Auto rend les vues dont le pixel dépasse ~1e-12 par l'itération
//! directe en f64 (`interp_batch`). Sur un bord chaotique à fort plafond,
//! l'arrondi y est amplifié par l'orbite : seahorse 1e10, 2500 it., 256×160,
//! 7,8 % de pixels faux contre le GMP pur — sans aucun signal. Le noyau par
//! lots porte désormais un traqueur de borne d'erreur
//! ([`iterate_bytecode_f64_batch_tracked`]) ; ce module décide QUAND le
//! brancher et recalcule les SEULS pixels signalés :
//!
//! - **Mandelbrot / Julia** (mapping sans K ni offset AA uniforme) :
//!   [`recompute_pixels_cpu`] — perturbation f64 tracée autour d'une orbite
//!   de référence au centre de la vue (≈ 1 ms à ces profondeurs), puis la
//!   chaîne existante dd → qd → GMP pour ce qu'elle juge encore non fiable.
//!   Même code que la correction des pixels du kernel GPU f32.
//! - **autres familles suivables** (Burning Ship, Tricorn, Celtic, Buffalo,
//!   PerpBS et leurs Julia, opt-in `FRACTALL_RELIABILITY_STD=all`) : GMP pur
//!   par pixel ([`iterate_point_mpc`], le juge de la QA) sous le plafond des
//!   familles sans dd de la perturbation — 0,1 % des pixels OU 5 % des
//!   itérations de la frame — sinon valeur f64 gardée et avertissement.
//!
//! Interrupteurs (lus une fois) : `FRACTALL_RELIABILITY=off` coupe tout,
//! `observe` suit et rapporte sans corriger ; `FRACTALL_RELIABILITY_STD=0`
//! coupe ce seul path, `all` y ajoute les familles hors Mandelbrot/Julia.
//!
//! [`iterate_bytecode_f64_batch_tracked`]: crate::fractal::bytecode::interp_batch::iterate_bytecode_f64_batch_tracked
//! [`recompute_pixels_cpu`]: crate::fractal::perturbation::recompute_pixels_cpu

use std::sync::atomic::{AtomicBool, Ordering};

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::bytecode::interp_batch::phase_trackable;
use crate::fractal::bytecode::reliability::{self as rel, ReliabilityMode};
use crate::fractal::bytecode::Phase;
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::perturbation::{
    compute_perturbation_precision_bits, perf_enabled, recompute_pixels_cpu,
    DcGmpContext,
};
use crate::fractal::{FractalParams, FractalType, PlaneTransform};

/// Réglage `FRACTALL_RELIABILITY_STD`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StdScope {
    /// Pas de suivi du path standard.
    Off,
    /// Mandelbrot et Julia (correction par perturbation) — défaut.
    MandelbrotJulia,
    /// + toutes les familles suivables (correction GMP sous plafond).
    All,
}

fn parse_scope(raw: Option<&str>) -> StdScope {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("0") | Some("off") | Some("false") => StdScope::Off,
        Some("all") => StdScope::All,
        _ => StdScope::MandelbrotJulia,
    }
}

fn std_scope() -> StdScope {
    static SCOPE: std::sync::OnceLock<StdScope> = std::sync::OnceLock::new();
    *SCOPE.get_or_init(|| parse_scope(std::env::var("FRACTALL_RELIABILITY_STD").ok().as_deref()))
}

/// Correction des pixels signalés : par perturbation (Mandelbrot / Julia) ou
/// par GMP pur sous plafond (autres familles).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StdCorrection {
    Perturbation,
    GmpCapped,
}

/// Plan de suivi du path standard pour ce rendu : `Some((correction,
/// escalade))` si le noyau par lots doit tourner avec le traqueur.
/// `escalade == false` : mode `observe` (rapport seulement).
pub fn std_tracking_plan(
    params: &FractalParams,
    phase: &Phase,
) -> Option<(StdCorrection, bool)> {
    plan_for(params, phase, rel::reliability_mode(), std_scope())
}

fn plan_for(
    params: &FractalParams,
    phase: &Phase,
    mode: ReliabilityMode,
    scope: StdScope,
) -> Option<(StdCorrection, bool)> {
    if !mode.tracks() || scope == StdScope::Off || !phase_trackable(phase) {
        return None;
    }
    // La correction recalcule c depuis le centre HP : le plan doit être Mu
    // (la transformation de plan n'est appliquée que par le path f64).
    if params.plane_transform != PlaneTransform::Mu || params.is_hybrid_formula() {
        return None;
    }
    let mandelbrot_like = !crate::fractal::bytecode::Formula::is_julia_for(params.fractal_type);
    // Graine non nulle : ni la perturbation ni les boucles GMP pures ne la
    // portent (cf. `seeded_mandelbrot_is_never_rendered_by_perturbation`).
    if mandelbrot_like && (params.seed.re != 0.0 || params.seed.im != 0.0) {
        return None;
    }
    let escalate = mode == ReliabilityMode::Escalate;
    let perturbable = matches!(params.fractal_type, FractalType::Mandelbrot | FractalType::Julia)
        && params.engine.use_bytecode_engine
        && params.transform_matrix().is_none()
        && params.sampling.aa_subpixel_offset == [0.0, 0.0];
    if perturbable {
        return Some((StdCorrection::Perturbation, escalate));
    }
    // Mandelbrot / Julia tournés : GMP pur sous plafond (DcGmpContext porte K).
    if matches!(params.fractal_type, FractalType::Mandelbrot | FractalType::Julia)
        || scope == StdScope::All
    {
        return Some((StdCorrection::GmpCapped, escalate));
    }
    None
}

/// Recalcule les pixels `flagged` (indices plein cadre) et écrit les valeurs
/// corrigées dans `iterations` / `zs`. Renvoie le nombre de pixels corrigés
/// (`None` si annulé).
pub fn correct_std_pixels(
    params: &FractalParams,
    correction: StdCorrection,
    cancel: &AtomicBool,
    flagged: &[usize],
    iterations: &mut [u32],
    zs: &mut [Complex64],
) -> Option<usize> {
    if flagged.is_empty() {
        return Some(0);
    }
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let t = std::time::Instant::now();
    let n_pixels = iterations.len();
    let fixed: Vec<(usize, u32, Complex64)> = match correction {
        StdCorrection::Perturbation => {
            let mut orbit_params = params.clone();
            orbit_params.engine.precision_bits = compute_perturbation_precision_bits(params);
            let cache = crate::fractal::perturbation::orbit::compute_reference_orbit_cached(
                &orbit_params,
                Some(cancel),
                None,
                None,
                false,
            )?;
            recompute_pixels_cpu(&orbit_params, &cache, flagged)
        }
        StdCorrection::GmpCapped => {
            let flagged_iters: f64 = flagged.iter().map(|&i| iterations[i] as f64).sum();
            let frame_iters: f64 = iterations.iter().map(|&n| n as f64).sum();
            if !(rel::gmp_escalation_allowed(
                flagged.len(),
                n_pixels,
                rel::reliability_gmp_max_fraction(),
            ) || flagged_iters <= STD_GMP_ITER_BUDGET * frame_iters)
            {
                eprintln!(
                    "[RELIABILITY-STD] {} unreliable pixels ({:.2} %) left at their f64 value (type {:?}) — GMP recomputation over budget; set FRACTALL_RELIABILITY_GMP_MAX_FRACTION=1 to correct them all",
                    flagged.len(),
                    100.0 * flagged.len() as f64 / n_pixels.max(1) as f64,
                    params.fractal_type
                );
                return Some(0);
            }
            gmp_pixels(params, cancel, flagged)?
        }
    };
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let n = fixed.len();
    for (idx, it, z) in fixed {
        iterations[idx] = it;
        zs[idx] = z;
    }
    if perf_enabled() {
        eprintln!(
            "[RELIABILITY-STD] flagged={} ({:.3} %) corrected={} via={:?} time={:.3}s",
            flagged.len(),
            100.0 * flagged.len() as f64 / n_pixels.max(1) as f64,
            n,
            correction,
            t.elapsed().as_secs_f64()
        );
    }
    Some(n)
}

/// Budget du GMP pur des familles sans perturbation : Σ itérations des pixels
/// signalés ≤ cette fraction des itérations de la frame (même valeur que
/// `FAMILY_GMP_ITER_BUDGET` en perturbation).
const STD_GMP_ITER_BUDGET: f64 = 0.05;

fn gmp_pixels(
    params: &FractalParams,
    cancel: &AtomicBool,
    flagged: &[usize],
) -> Option<Vec<(usize, u32, Complex64)>> {
    let prec = compute_perturbation_precision_bits(params);
    let mut gp = params.clone();
    gp.engine.precision_bits = prec;
    let gmp_params = MpcParams::from_params(&gp);
    let parse = |hp: &Option<String>, f: f64| match hp.as_deref().map(Float::parse) {
        Some(Ok(v)) => Float::with_val(prec, v),
        _ => Float::with_val(prec, f),
    };
    let cx = parse(&params.center_x_hp, params.center_x);
    let cy = parse(&params.center_y_hp, params.center_y);
    let dc_ctx = DcGmpContext::new(params, prec);
    let width = params.width.max(1) as usize;
    let out: Vec<_> = flagged
        .par_iter()
        .map(|&idx| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let dc = dc_ctx.compute_dc(idx % width, idx / width);
            let mut re = cx.clone();
            re += dc.real();
            let mut im = cy.clone();
            im += dc.imag();
            let (it, z) = iterate_point_mpc(&gmp_params, &complex_from_xy(prec, re, im));
            Some((idx, it, complex_to_complex64(&z)))
        })
        .collect();
    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::compile_formula;
    use crate::fractal::default_params_for_type;

    fn phase(ty: FractalType) -> Phase {
        compile_formula(ty, 2.0).unwrap().phases.remove(0)
    }

    /// E2E par le dispatcher unique : seahorse 1e10, 2500 it., rendu Auto
    /// (path f64 standard à cette taille de pixel) contre le GMP pur. Sans
    /// suivi le path standard rend ~8 % de pixels faux ; avec, au plus un
    /// résidu de quelques pixels.
    #[test]
    fn std_path_matches_gmp_on_a_chaotic_edge() {
        use crate::fractal::AlgorithmMode;
        let (w, h) = (96u32, 60u32);
        let mut p = default_params_for_type(FractalType::Mandelbrot, w, h);
        p.center_x_hp = Some("-0.743643887037158704752191506114774".into());
        p.center_y_hp = Some("0.131825904205311970493132056385139".into());
        p.center_x = -0.743_643_887_037_158_7;
        p.center_y = 0.131_825_904_205_311_97;
        p.span_x = 4e-10;
        p.span_y = 4e-10 * h as f64 / w as f64;
        p.span_x_hp = None;
        p.span_y_hp = None;
        p.iteration_max = 2500;
        assert_eq!(
            crate::fractal::wisdom::select_algorithm(&p, crate::fractal::wisdom::Device::Cpu),
            crate::fractal::wisdom::Algorithm::StandardF64,
            "la scène doit passer par le path f64 standard"
        );
        let prod = crate::render::render_escape_time(&p);
        let mut g = p.clone();
        g.engine.algorithm_mode = AlgorithmMode::ReferenceGmp;
        let gmp = crate::render::render_escape_time(&g);
        // Référence sans correction : le noyau par lots nu.
        let phase = crate::fractal::bytecode::formula_for_params(&p).unwrap().phases.remove(0);
        let pts: Vec<Complex64> = (0..(w * h) as usize)
            .map(|k| {
                let (i, j) = (k % w as usize, k / w as usize);
                Complex64::new(
                    p.center_x + ((i as f64 + 0.5) / w as f64 - 0.5) * p.span_x,
                    p.center_y + ((j as f64 + 0.5) / h as f64 - 0.5) * p.span_y,
                )
            })
            .collect();
        let mut raw = vec![0u32; pts.len()];
        let mut zs = vec![Complex64::new(0.0, 0.0); pts.len()];
        crate::fractal::bytecode::interp_batch::iterate_bytecode_f64_batch(
            &phase, &pts, p.seed, false, 2500, p.bailout, &mut raw, &mut zs,
        );
        let wrong = |a: &[u32]| a.iter().zip(&gmp.iterations).filter(|(x, y)| x != y).count();
        let before = wrong(&raw);
        let after = wrong(&prod.iterations);
        assert!(before * 100 > pts.len() * 3, "sans suivi : {before} pixels faux attendus en masse");
        assert!(after <= 3, "path standard corrigé : {after} pixels ≠ GMP (avant : {before})");
    }

    #[test]
    fn scope_parsing() {
        assert_eq!(parse_scope(None), StdScope::MandelbrotJulia);
        assert_eq!(parse_scope(Some("0")), StdScope::Off);
        assert_eq!(parse_scope(Some(" ALL ")), StdScope::All);
        assert_eq!(parse_scope(Some("1")), StdScope::MandelbrotJulia);
    }

    /// Qui est suivi, et comment il est corrigé.
    #[test]
    fn tracking_plan_routes_families() {
        let esc = ReliabilityMode::Escalate;
        let m = default_params_for_type(FractalType::Mandelbrot, 64, 64);
        let mp = phase(FractalType::Mandelbrot);
        assert_eq!(
            plan_for(&m, &mp, esc, StdScope::MandelbrotJulia),
            Some((StdCorrection::Perturbation, true))
        );
        assert_eq!(plan_for(&m, &mp, esc, StdScope::Off), None);
        assert_eq!(plan_for(&m, &mp, ReliabilityMode::Off, StdScope::All), None);
        assert_eq!(
            plan_for(&m, &mp, ReliabilityMode::Observe, StdScope::MandelbrotJulia),
            Some((StdCorrection::Perturbation, false))
        );
        // Rotation : la perturbation par pixel ne porte pas K → GMP.
        let mut r = m.clone();
        r.rotation = 30.0;
        assert_eq!(
            plan_for(&r, &mp, esc, StdScope::MandelbrotJulia),
            Some((StdCorrection::GmpCapped, true))
        );
        // Graine non nulle, plan transformé : non suivis.
        let mut s = m.clone();
        s.seed = Complex64::new(0.1, 0.0);
        assert_eq!(plan_for(&s, &mp, esc, StdScope::All), None);
        let mut pl = m.clone();
        pl.plane_transform = PlaneTransform::Inversion;
        assert_eq!(plan_for(&pl, &mp, esc, StdScope::All), None);
        // Burning Ship : opt-in `all`, GMP sous plafond.
        let bs = default_params_for_type(FractalType::BurningShip, 64, 64);
        let bp = phase(FractalType::BurningShip);
        assert_eq!(plan_for(&bs, &bp, esc, StdScope::MandelbrotJulia), None);
        assert_eq!(
            plan_for(&bs, &bp, esc, StdScope::All),
            Some((StdCorrection::GmpCapped, true))
        );
        // Julia : perturbation.
        let j = default_params_for_type(FractalType::Julia, 64, 64);
        assert_eq!(
            plan_for(&j, &mp, esc, StdScope::MandelbrotJulia),
            Some((StdCorrection::Perturbation, true))
        );
    }
}

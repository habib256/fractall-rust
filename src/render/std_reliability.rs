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
//! - **Mandelbrot / Julia / Tricorn** (mapping sans K ni offset AA uniforme) :
//!   [`recompute_pixels_cpu`] — perturbation f64 tracée autour d'une orbite
//!   de référence au centre de la vue (≈ 1 ms à ces profondeurs), puis la
//!   chaîne existante dd → qd → GMP pour ce qu'elle juge encore non fiable.
//!   Même code que la correction des pixels du kernel GPU f32.
//! - **autres familles suivables** — Mandelbrot / Julia tournés et, par
//!   défaut, les variantes Julia de Burning Ship / Tricorn / Buffalo / Perp. BS
//!   (2026-11-19) ; Burning Ship, Celtic, Buffalo et Celtic Julia en opt-in
//!   `FRACTALL_RELIABILITY_STD=all` : itération DIRECTE en double-double
//!   tracée (`bytecode::direct_dd`, ~10-15 ns/itération), puis GMP pur par
//!   pixel ([`iterate_point_mpc`], le juge de la QA) pour ce que le dd ne
//!   garantit pas, sous le plafond des familles sans dd de la perturbation —
//!   0,1 % des pixels OU 5 % des itérations de la frame — sinon valeur f64
//!   gardée et avertissement.
//! - **Perp. Burning Ship** (Mandelbrot-like, bord `CHAOS`) : même chaîne en
//!   mode [`StdCorrection::Budgeted`] — rien n'est tenté au-delà de 2 % de
//!   pixels signalés, et le reste hors budget GMP est gardé sans bruit.
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
    compute_perturbation_precision_bits, direct_dd_phase, direct_dd_pixels, perf_enabled,
    recompute_pixels_cpu, DcGmpContext,
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

/// Le suivi du path standard est-il coupé (`FRACTALL_RELIABILITY_STD=0`) ?
/// Partagé avec le suivi des boucles scalaires (`scalar_reliability`).
pub(crate) fn std_scope_is_off() -> bool {
    std_scope() == StdScope::Off
}

/// Correction des pixels signalés : par perturbation (Mandelbrot / Julia) ou
/// par double-double direct puis GMP pur sous plafond (autres familles).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StdCorrection {
    Perturbation,
    GmpCapped,
    /// Comme `GmpCapped`, mais SILENCIEUX et borné en amont : au-delà de
    /// [`FOLD_FLAG_BUDGET`] pixels signalés la frame garde ses valeurs f64
    /// sans rien payer, et ce que ni le double-double ni le budget GMP ne
    /// couvrent reste en f64 sans avertissement. Familles à pliage
    /// Mandelbrot-like par défaut, dont le bord est `CHAOS` (~98 % signalés).
    Budgeted,
}

/// Fraction de pixels signalés au-delà de laquelle la correction
/// [`StdCorrection::Budgeted`] renonce (vue d'accueil Perp. BS 128×80 :
/// 1,3 % ; Burning Ship / Celtic ~5 % ; Buffalo 9,6 % ; bords ×4096 : 97-99 %).
pub const FOLD_FLAG_BUDGET: f64 = 0.02;

/// Variantes Julia des familles à pliage suivies par défaut (correction
/// double-double puis GMP sous plafond). Leurs bords sont CONVERGÉS (juge
/// 256 b ≡ 512 b), contrairement aux versions Mandelbrot-like. Celtic Julia
/// n'y est PAS : son intérieur est un attracteur chaotique borné — la borne
/// s'emballe (52 % des pixels signalés sur le bord ×4096) sans que le compte
/// (le plafond) soit jamais faux, et aucun de ces pixels n'est garanti par le
/// double-double ; le suivre coûterait ×7 pour un avertissement.
fn fold_julia_tracked_by_default(ft: FractalType) -> bool {
    matches!(
        ft,
        FractalType::BurningShipJulia
            | FractalType::TricornJulia
            | FractalType::BuffaloJulia
            | FractalType::PerpendicularBurningShipJulia
    )
}

/// Familles à pliage Mandelbrot-like suivies par défaut en mode
/// [`StdCorrection::Budgeted`].
fn fold_mandelbrot_budgeted_by_default(ft: FractalType) -> bool {
    matches!(ft, FractalType::PerpendicularBurningShip)
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
    // Tricorn (2026-11-19) : même correction que Mandelbrot / Julia — ses
    // pixels signalés sont recalculés par la perturbation tracée de la
    // famille, puis en GMP pur pour ce qu'elle signale encore (pas de dd).
    let perturbable = matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::Tricorn
    ) && params.engine.use_bytecode_engine
        && params.transform_matrix().is_none()
        && params.sampling.aa_subpixel_offset == [0.0, 0.0];
    if perturbable {
        return Some((StdCorrection::Perturbation, escalate));
    }
    // Mandelbrot / Julia tournés : GMP pur sous plafond (DcGmpContext porte K).
    if matches!(params.fractal_type, FractalType::Mandelbrot | FractalType::Julia)
        || fold_julia_tracked_by_default(params.fractal_type)
        || scope == StdScope::All
    {
        return Some((StdCorrection::GmpCapped, escalate));
    }
    if fold_mandelbrot_budgeted_by_default(params.fractal_type) {
        return Some((StdCorrection::Budgeted, escalate));
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
    let quiet_over_budget = correction == StdCorrection::Budgeted;
    if quiet_over_budget && flagged.len() as f64 > FOLD_FLAG_BUDGET * n_pixels as f64 {
        if perf_enabled() {
            eprintln!(
                "[RELIABILITY-STD] flagged={} ({:.3} %) over the {:.0} % budget — f64 kept",
                flagged.len(),
                100.0 * flagged.len() as f64 / n_pixels.max(1) as f64,
                100.0 * FOLD_FLAG_BUDGET
            );
        }
        return Some(0);
    }
    // Mode Budgeted : les pixels signalés restés au PLAFOND sont, sur ces
    // familles, des orbites bornées chaotiques (intérieur « hirsute ») que le
    // double-double ne garantit pas non plus — Perp. BS accueil 720p : 86 %
    // des signalés, ×3 sur le rendu pour 0 pixel garanti. Seuls les pixels
    // évadés sont tentés (meilleur effort, sans avertissement).
    let escaped_only: Vec<usize>;
    let flagged: &[usize] = if quiet_over_budget {
        let cap = params.iteration_max;
        escaped_only = flagged.iter().copied().filter(|&i| iterations[i] < cap).collect();
        &escaped_only
    } else {
        flagged
    };
    let mut dd_fixed = 0usize;
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
        StdCorrection::GmpCapped | StdCorrection::Budgeted => {
            let ctx = GmpPointContext::new(params);
            // Étage double-double DIRECT d'abord (~10-15 ns/itération contre
            // ~150 en GMP pur) : il ne rend que des pixels garantis, le reste
            // seul passe au GMP sous plafond.
            let (mut fixed, rest) = match direct_dd_phase(params, true) {
                Some(phase) => {
                    direct_dd_pixels(params, &phase, &ctx.cx, &ctx.cy, &ctx.dc_ctx, flagged)
                }
                None => (Vec::new(), flagged.to_vec()),
            };
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            dd_fixed = fixed.len();
            let rest_iters: f64 = rest.iter().map(|&i| iterations[i] as f64).sum();
            let frame_iters: f64 = iterations.iter().map(|&n| n as f64).sum();
            // Budgeted : le plafond en PIXELS seulement — le budget de 5 % des
            // itérations y coûtait ×4 (Perp. BS accueil 720p : 1 056 pixels
            // évadés au-delà du dd, 1,25 s de GMP pour un rendu de 0,28 s).
            if rest.is_empty()
                || rel::gmp_escalation_allowed(
                    rest.len(),
                    n_pixels,
                    rel::reliability_gmp_max_fraction(),
                )
                || (!quiet_over_budget && rest_iters <= STD_GMP_ITER_BUDGET * frame_iters)
            {
                fixed.extend(gmp_pixels(params, &ctx, cancel, &rest)?);
            } else if !quiet_over_budget {
                eprintln!(
                    "[RELIABILITY-STD] {} unreliable pixels ({:.2} %) left at their f64 value (type {:?}) — GMP recomputation over budget; set FRACTALL_RELIABILITY_GMP_MAX_FRACTION=1 to correct them all",
                    rest.len(),
                    100.0 * rest.len() as f64 / n_pixels.max(1) as f64,
                    params.fractal_type
                );
            }
            fixed
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
            "[RELIABILITY-STD] flagged={} ({:.3} %) corrected={} dd_direct={} via={:?} time={:.3}s",
            flagged.len(),
            100.0 * flagged.len() as f64 / n_pixels.max(1) as f64,
            n,
            dd_fixed,
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

/// Point exact d'un pixel : centre HP + dc GMP (K compris), à la précision
/// de la perturbation. Partagé par l'étage double-double et le GMP pur.
struct GmpPointContext {
    prec: u32,
    gmp_params: MpcParams,
    cx: Float,
    cy: Float,
    dc_ctx: DcGmpContext,
}

impl GmpPointContext {
    fn new(params: &FractalParams) -> Self {
        let prec = compute_perturbation_precision_bits(params);
        let mut gp = params.clone();
        gp.engine.precision_bits = prec;
        let parse = |hp: &Option<String>, f: f64| match hp.as_deref().map(Float::parse) {
            Some(Ok(v)) => Float::with_val(prec, v),
            _ => Float::with_val(prec, f),
        };
        Self {
            prec,
            gmp_params: MpcParams::from_params(&gp),
            cx: parse(&params.center_x_hp, params.center_x),
            cy: parse(&params.center_y_hp, params.center_y),
            dc_ctx: DcGmpContext::new(params, prec),
        }
    }
}

fn gmp_pixels(
    params: &FractalParams,
    ctx: &GmpPointContext,
    cancel: &AtomicBool,
    flagged: &[usize],
) -> Option<Vec<(usize, u32, Complex64)>> {
    let width = params.width.max(1) as usize;
    let prec = ctx.prec;
    let out: Vec<_> = flagged
        .par_iter()
        .map(|&idx| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let dc = ctx.dc_ctx.compute_dc(idx % width, idx / width);
            let mut re = ctx.cx.clone();
            re += dc.real();
            let mut im = ctx.cy.clone();
            im += dc.imag();
            let (it, z) = iterate_point_mpc(&ctx.gmp_params, &complex_from_xy(prec, re, im));
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

    /// Bord PARABOLIQUE au plafond que choisit la sonde d'itérations
    /// (vallée des éléphants 1e7, 60 000 it.) : la borne dd des pixels
    /// corrigés ne s'emballe pas — l'orbite s'évade juste après le goulot —
    /// mais elle est comparable à |z|. Avant le test d'évasion relatif des
    /// vues peu profondes (`shallow_dd_escape_rel`) : 1,2 % de pixels faux
    /// contre GMP pur à 128×80, invisibles au plafond 2500.
    #[test]
    fn parabolic_edge_at_auto_cap_matches_gmp() {
        let (w, h) = (64u32, 40u32);
        let mut p = default_params_for_type(FractalType::Mandelbrot, w, h);
        p.center_x_hp = Some("0.27631692275258796".into());
        p.center_y_hp = Some("0.0085".into());
        p.center_x = 0.276_316_922_752_587_96;
        p.center_y = 0.0085;
        p.span_x = 4e-7;
        p.span_y = 4e-7 * h as f64 / w as f64;
        p.span_x_hp = None;
        p.span_y_hp = None;
        p.iteration_max = 60_000;
        assert_eq!(
            crate::fractal::wisdom::select_algorithm(&p, crate::fractal::wisdom::Device::Cpu),
            crate::fractal::wisdom::Algorithm::StandardF64,
        );
        let prod = crate::render::render_escape_time(&p);
        let mut g = p.clone();
        g.engine.algorithm_mode = crate::fractal::AlgorithmMode::ReferenceGmp;
        let gmp = crate::render::render_escape_time(&g);
        let wrong = prod.iterations.iter().zip(&gmp.iterations).filter(|(x, y)| x != y).count();
        // Anti-vacuité : la vue a bien un bord (ni tout évadé tôt, ni intérieur).
        let late = gmp.iterations.iter().filter(|&&n| n > 5_000).count();
        assert!(late * 20 > (w * h) as usize, "scène sans bord tardif ({late})");
        assert!(wrong * 500 <= (w * h) as usize, "{wrong} pixels ≠ GMP pur sur {}", w * h);
    }

    /// Vue de l'axe `types` (zoom = 4/span_x, plafond figé).
    fn scene(ty: FractalType, cx: f64, cy: f64, zoom: f64, w: u32, h: u32, n: u32) -> FractalParams {
        let mut p = default_params_for_type(ty, w, h);
        p.center_x = cx;
        p.center_y = cy;
        p.center_x_hp = Some(format!("{cx:?}"));
        p.center_y_hp = Some(format!("{cy:?}"));
        p.span_x = 4.0 / zoom;
        p.span_y = p.span_x * h as f64 / w as f64;
        p.span_x_hp = None;
        p.span_y_hp = None;
        p.iteration_max = n;
        p
    }

    /// L'étage double-double DIRECT de la correction `GmpCapped` ne rend que
    /// des valeurs que le GMP pur confirme : sur des vues de bord de familles
    /// à PLIAGE (abs), Mandelbrot-like comme Julia, chaque pixel que le dd
    /// déclare fiable a le MÊME compte que le chemin tout-GMP. Au pli, la
    /// borne absolue reste valable (|abs(x+e) − abs(x)| ≤ |e|) : aucun test
    /// « |Re z| ≤ E » n'est nécessaire, et l'anti-vacuité vérifie que le dd
    /// garantit bien des pixels ET en refuse.
    #[test]
    fn direct_dd_stage_agrees_with_gmp_on_fold_families() {
        let cancel = AtomicBool::new(false);
        let cases = [
            (FractalType::PerpendicularBurningShipJulia, 0.321075439453125, -0.36871948242187497),
            (FractalType::BurningShipJulia, -0.476837158203125, -1.077777099609375),
            (FractalType::BurningShip, -0.857635498046875, -0.591802978515625),
            (FractalType::Celtic, -1.2645797729492188, -0.14105072021484374),
            (FractalType::Buffalo, -1.524261474609375, -0.731927490234375),
        ];
        let (mut total_ok, mut total_refused) = (0usize, 0usize);
        for (ty, cx, cy) in cases {
            let p = scene(ty, cx, cy, 4096.0, 24, 16, 1500);
            let phase = direct_dd_phase(&p, true).expect("phase dd");
            let ctx = GmpPointContext::new(&p);
            let all: Vec<usize> = (0..(24 * 16)).collect();
            let (dd, refused) = direct_dd_pixels(&p, &phase, &ctx.cx, &ctx.cy, &ctx.dc_ctx, &all);
            let gmp = gmp_pixels(&p, &ctx, &cancel, &all).unwrap();
            for &(idx, it, z) in &dd {
                let (_, git, gz) = gmp[idx];
                assert_eq!(it, git, "{ty:?} pixel {idx} : dd {it} ≠ GMP {git}");
                // z final : garanti à la borne d'évasion du dd seulement
                // (E < max(E_MAX, 0,01·|z|)) — mesuré jusqu'à 1e-8 relatif
                // d'écart au GMP sur Celtic, invisible en couleur lissée.
                if it < p.iteration_max {
                    assert!((z - gz).norm() <= 0.01 * gz.norm(), "{ty:?} pixel {idx} : z {z} ≠ {gz}");
                }
            }
            total_ok += dd.len();
            total_refused += refused.len();
        }
        assert!(total_ok > 500 && total_refused > 50, "dd fiable {total_ok}, refusé {total_refused}");
    }

    /// Perp. Burning Ship Julia, bord ×4096 de l'axe `types` : 9 pixels ≠ GMP
    /// pur (0,35 %, juge convergé) avant le suivi par défaut des variantes
    /// Julia à pliage ; 0 après — corrigés par le double-double direct.
    #[test]
    fn perp_burning_ship_julia_edge_matches_gmp() {
        let p = scene(
            FractalType::PerpendicularBurningShipJulia,
            0.321075439453125,
            -0.36871948242187497,
            4096.0,
            64,
            40,
            5000,
        );
        let prod = crate::render::render_escape_time(&p);
        let mut g = p.clone();
        g.engine.algorithm_mode = crate::fractal::AlgorithmMode::ReferenceGmp;
        let gmp = crate::render::render_escape_time(&g);
        let wrong = prod.iterations.iter().zip(&gmp.iterations).filter(|(x, y)| x != y).count();
        assert_eq!(wrong, 0, "{wrong} pixels ≠ GMP pur");
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
        // Variantes Julia à pliage : par défaut (dd puis GMP sous plafond),
        // sauf Celtic Julia (intérieur chaotique) — opt-in `all`.
        for ty in [
            FractalType::BurningShipJulia,
            FractalType::TricornJulia,
            FractalType::BuffaloJulia,
            FractalType::PerpendicularBurningShipJulia,
        ] {
            let q = default_params_for_type(ty, 64, 64);
            assert_eq!(
                plan_for(&q, &phase(ty), esc, StdScope::MandelbrotJulia),
                Some((StdCorrection::GmpCapped, true)),
                "{ty:?}"
            );
        }
        let cj = default_params_for_type(FractalType::CelticJulia, 64, 64);
        let cjp = phase(FractalType::CelticJulia);
        assert_eq!(plan_for(&cj, &cjp, esc, StdScope::MandelbrotJulia), None);
        assert_eq!(
            plan_for(&cj, &cjp, esc, StdScope::All),
            Some((StdCorrection::GmpCapped, true))
        );
        // Perp. BS : budgété par défaut, GMP sous plafond en `all`.
        let pb = default_params_for_type(FractalType::PerpendicularBurningShip, 64, 64);
        let pbp = phase(FractalType::PerpendicularBurningShip);
        assert_eq!(
            plan_for(&pb, &pbp, esc, StdScope::MandelbrotJulia),
            Some((StdCorrection::Budgeted, true))
        );
        assert_eq!(
            plan_for(&pb, &pbp, esc, StdScope::All),
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

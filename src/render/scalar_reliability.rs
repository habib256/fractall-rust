//! Fiabilité des types à boucle SCALAIRE (hors bytecode) — 2026-11-19.
//!
//! Le traqueur du path standard (`std_reliability`) ne couvre que le noyau
//! bytecode par lots. Les types à boucle dédiée (Barnsley, Magnet, Phoenix,
//! Alpha Mandelbrot, Mandelbulb…) n'avaient aucun suivi : l'axe `types` du
//! harness y a mesuré, à juge GMP convergé (256 b ≡ 512 b), 5 % de pixels
//! faux sur un bord Barnsley Julia ×2048 et 1 % sur sa vue d'accueil.
//!
//! Premier membre : **Barnsley** (Julia et Mandelbrot),
//! `z' = (z ∓ 1)·c` selon le signe de `Re z`. La carte dilate de `|c|` à
//! chaque pas (≈ 1,25 pour la graine par défaut) : l'arrondi f64 atteint
//! l'ordre de `z` vers 130 itérations, et le test de branche `Re z ≥ 0`
//! bascule sur un bit. Trois étages, même schéma que les familles bytecode :
//!
//! 1. boucle f64 **tracée**, bit-identique à la boucle de production
//!    (`iterations::barnsley_*`, verrou) : borne `E' = |c|·E + u·|z'|`
//!    (+ `|z ∓ 1|·E_c` quand `c` est le pixel), pixel signalé si la borne
//!    dépasse [`E_MAX`], si `|Re z| ≤ E` au test de branche, ou si l'évasion
//!    n'est pas déterminée (`||z| − R| ≤ E`) ;
//! 2. pixels signalés : même boucle en **double-double** (point exact
//!    recalculé depuis le centre HP), ~10× moins chère que le GMP ;
//! 3. ce que le double-double ne garantit pas : GMP pur sous le budget du
//!    path standard (`std_reliability::correct_std_pixels`, `GmpCapped`).
//!
//! Étendre à un autre type = sa boucle tracée f64 + double-double et une
//! entrée dans [`plan`] ; le juge est la scène du type dans l'axe `types`.

use std::sync::atomic::{AtomicBool, Ordering};

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::bytecode::direct_dd::dd_from_gmp;
use crate::fractal::bytecode::reliability::{self as rel, ReliabilityMode};
use crate::fractal::perturbation::dd::DoubleDouble as Dd;
use crate::fractal::perturbation::{compute_perturbation_precision_bits, perf_enabled, DcGmpContext};
use crate::fractal::{FractalParams, FractalType, PlaneTransform};

/// Borne absolue (espace z) au-delà de laquelle le pixel n'est plus déterminé.
pub const E_MAX: f64 = 1e-3;
/// Unité d'arrondi f64, avec un bit de garde.
const U64: f64 = 1.0 / (1u64 << 52) as f64;
/// Unité d'arrondi double-double, avec deux bits de garde.
const UDD: f64 = crate::fractal::bytecode::direct_dd::DD_UNIT;

/// Ce rendu doit-il suivre sa boucle scalaire ? `Some(escalade)` si oui
/// (`false` = mode `observe` : signaler sans corriger).
pub fn plan(params: &FractalParams) -> Option<bool> {
    plan_for(params, rel::reliability_mode(), crate::render::std_reliability::std_scope_is_off())
}

fn plan_for(params: &FractalParams, mode: ReliabilityMode, scope_off: bool) -> Option<bool> {
    if !mode.tracks() || scope_off {
        return None;
    }
    if !matches!(
        params.fractal_type,
        FractalType::BarnsleyJulia | FractalType::BarnsleyMandelbrot
    ) {
        return None;
    }
    // La correction recalcule le point depuis le centre HP (plan Mu) et ne
    // produit aucun canal annexe.
    if params.plane_transform != PlaneTransform::Mu
        || params.channels.enable_orbit_traps
        || params.channels.enable_distance_estimation
        || params.channels.enable_interior_detection
    {
        return None;
    }
    Some(mode == ReliabilityMode::Escalate)
}

/// Itère le pixel par la boucle f64 tracée du type. Renvoie
/// `(itérations, z final, non_fiable)` ; compte et z identiques au bit près
/// à `iterations::iterate_point`.
#[inline]
pub fn iterate_tracked(params: &FractalParams, z_pixel: Complex64) -> (u32, Complex64, bool) {
    match params.fractal_type {
        FractalType::BarnsleyJulia => {
            barnsley_f64(z_pixel, params.seed, false, params.iteration_max, params.bailout)
        }
        _ => barnsley_f64(z_pixel, z_pixel, true, params.iteration_max, params.bailout),
    }
}

/// Barnsley f64 tracé. `c_is_pixel` : `c` porte l'arrondi du pixel (variante
/// Mandelbrot) ; sinon `c` est la graine, exacte.
fn barnsley_f64(
    z0: Complex64,
    c: Complex64,
    c_is_pixel: bool,
    iteration_max: u32,
    bailout: f64,
) -> (u32, Complex64, bool) {
    let b2 = bailout * bailout;
    // Majorants L1 (|z| ≤ |re| + |im|) : aucune racine dans la boucle.
    let c_abs = c.norm();
    let e_c = if c_is_pixel { U64 * c_abs } else { 0.0 };
    let mut z = z0;
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<2>::new();
    // Arrondi du pixel lui-même (le juge part du point exact).
    let mut e = U64 * (z0.re.abs() + z0.im.abs());
    let mut bad = false;
    while i < iteration_max && z.norm_sqr() < b2 {
        let re = z.re;
        let w = if re >= 0.0 {
            z - Complex64::new(1.0, 0.0)
        } else {
            z + Complex64::new(1.0, 0.0)
        };
        z = w * c;
        i += 1;
        // Suivi inerte une fois le pixel signalé (il sera recalculé) : les
        // pixels chaotiques retrouvent le débit de la boucle nue.
        if !bad {
            if re.abs() <= e {
                bad = true;
            }
            e = c_abs * e + (w.re.abs() + w.im.abs()) * e_c + U64 * (z.re.abs() + z.im.abs());
            // Test de sortie de CE pas déterminé ssi |z| ∉ [R − E, R + E].
            let n2 = z.norm_sqr();
            let lo = (bailout - e).max(0.0);
            let hi = bailout + e;
            if (n2 >= lo * lo && n2 <= hi * hi) || !(e < E_MAX) {
                bad = true;
            }
        }
        cyc.observe(&mut i, iteration_max, [z.re, z.im]);
    }
    (i, z, bad)
}

/// Barnsley double-double tracé. Renvoie `(itérations, z, fiable)`.
fn barnsley_dd(
    z0: (Dd, Dd),
    c: (Dd, Dd),
    c_is_pixel: bool,
    iteration_max: u32,
    bailout: f64,
) -> (u32, Complex64, bool) {
    let b2 = bailout * bailout;
    let c_abs = c.0.hi.hypot(c.1.hi);
    let e_c = if c_is_pixel { UDD * c_abs } else { 0.0 };
    let one = Dd::from_f64(1.0);
    let (mut x, mut y) = z0;
    let mut i = 0u32;
    let mut e = UDD * x.hi.hypot(y.hi);
    while i < iteration_max && x.hi * x.hi + y.hi * y.hi < b2 {
        if x.hi.abs() <= e {
            return (i, Complex64::new(x.hi, y.hi), false);
        }
        let wx = if x.hi >= 0.0 { x.sub(one) } else { x.add(one) };
        let w_abs = wx.hi.hypot(y.hi);
        // (wx + i·y)·(c.re + i·c.im)
        let nx = wx.mul(c.0).sub(y.mul(c.1));
        let ny = wx.mul(c.1).add(y.mul(c.0));
        x = nx;
        y = ny;
        let n = x.hi.hypot(y.hi);
        e = c_abs * e + w_abs * e_c + UDD * n;
        i += 1;
        if !n.is_finite() || (n - bailout).abs() <= e || !(e < E_MAX) {
            return (i, Complex64::new(x.hi, y.hi), false);
        }
    }
    (i, Complex64::new(x.hi, y.hi), true)
}

/// Recalcule les pixels `flagged` : double-double, puis GMP pur sous budget
/// pour ceux que le double-double ne garantit pas. `None` si annulé.
pub fn correct(
    params: &FractalParams,
    cancel: &AtomicBool,
    flagged: &[usize],
    iterations: &mut [u32],
    zs: &mut [Complex64],
) -> Option<()> {
    if flagged.is_empty() {
        return Some(());
    }
    let t = std::time::Instant::now();
    let prec = compute_perturbation_precision_bits(params);
    let parse = |hp: &Option<String>, f: f64| match hp.as_deref().map(Float::parse) {
        Some(Ok(v)) => Float::with_val(prec, v),
        _ => Float::with_val(prec, f),
    };
    let cx = parse(&params.center_x_hp, params.center_x);
    let cy = parse(&params.center_y_hp, params.center_y);
    let dc_ctx = DcGmpContext::new(params, prec);
    let width = params.width.max(1) as usize;
    let is_julia = params.fractal_type == FractalType::BarnsleyJulia;
    let seed = (Dd::from_f64(params.seed.re), Dd::from_f64(params.seed.im));
    let tried: Vec<(usize, u32, Complex64, bool)> = flagged
        .par_iter()
        .map(|&idx| {
            let dc = dc_ctx.compute_dc(idx % width, idx / width);
            let mut re = cx.clone();
            re += dc.real();
            let mut im = cy.clone();
            im += dc.imag();
            let point = (dd_from_gmp(&re), dd_from_gmp(&im));
            let (it, z, ok) = if is_julia {
                barnsley_dd(point, seed, false, params.iteration_max, params.bailout)
            } else {
                barnsley_dd(point, point, true, params.iteration_max, params.bailout)
            };
            (idx, it, z, ok)
        })
        .collect();
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let mut rest = Vec::new();
    for (idx, it, z, ok) in tried {
        if ok {
            iterations[idx] = it;
            zs[idx] = z;
        } else {
            rest.push(idx);
        }
    }
    if perf_enabled() {
        eprintln!(
            "[RELIABILITY-SCALAR] flagged={} ({:.3} %) dd_fixed={} beyond_dd={} time={:.3}s",
            flagged.len(),
            100.0 * flagged.len() as f64 / iterations.len().max(1) as f64,
            flagged.len() - rest.len(),
            rest.len(),
            t.elapsed().as_secs_f64()
        );
    }
    if !rest.is_empty() {
        crate::render::std_reliability::correct_std_pixels(
            params,
            crate::render::std_reliability::StdCorrection::GmpCapped,
            cancel,
            &rest,
            iterations,
            zs,
        )?;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::iterations::iterate_point;
    use crate::fractal::{default_params_for_type, AlgorithmMode};

    /// La boucle tracée rend le compte et le z de la boucle de production,
    /// au bit près, pour les deux variantes.
    #[test]
    fn tracked_barnsley_is_bit_identical_to_production_loop() {
        for ft in [FractalType::BarnsleyJulia, FractalType::BarnsleyMandelbrot] {
            let p = default_params_for_type(ft, 48, 36);
            let mut flagged = 0;
            for j in 0..36 {
                for i in 0..48 {
                    let zp = Complex64::new(
                        p.center_x + ((i as f64 + 0.5) / 48.0 - 0.5) * p.span_x,
                        p.center_y + ((j as f64 + 0.5) / 36.0 - 0.5) * p.span_y,
                    );
                    let r = iterate_point(&p, zp);
                    let (it, z, bad) = iterate_tracked(&p, zp);
                    assert_eq!(it, r.iteration, "{ft:?} ({i},{j})");
                    assert_eq!(z.re.to_bits(), r.z.re.to_bits(), "{ft:?} ({i},{j})");
                    assert_eq!(z.im.to_bits(), r.z.im.to_bits(), "{ft:?} ({i},{j})");
                    flagged += bad as usize;
                }
            }
            assert!(flagged < 48 * 36, "{ft:?} : tout est signalé");
        }
    }

    /// Vue d'accueil Barnsley Julia : 1 % de pixels faux contre GMP pur
    /// avant le suivi (axe `types` du harness) ; plus aucun après.
    #[test]
    fn barnsley_julia_home_matches_gmp() {
        let (w, h) = (64u32, 40u32);
        let mut p = default_params_for_type(FractalType::BarnsleyJulia, w, h);
        p.iteration_max = 5000;
        assert_eq!(plan_for(&p, ReliabilityMode::Escalate, false), Some(true));
        let prod = crate::render::render_escape_time(&p);
        let mut g = p.clone();
        g.engine.algorithm_mode = AlgorithmMode::ReferenceGmp;
        let gmp = crate::render::render_escape_time(&g);
        let wrong = prod.iterations.iter().zip(&gmp.iterations).filter(|(a, b)| a != b).count();
        // Sans suivi : la boucle de production nue.
        let raw_wrong = (0..(w * h) as usize)
            .filter(|&k| {
                let (i, j) = (k % w as usize, k / w as usize);
                let zp = Complex64::new(
                    p.center_x + ((i as f64 + 0.5) / w as f64 - 0.5) * p.span_x,
                    p.center_y + ((j as f64 + 0.5) / h as f64 - 0.5) * p.span_y,
                );
                iterate_point(&p, zp).iteration != gmp.iterations[k]
            })
            .count();
        assert!(raw_wrong >= 5, "scène sans pixel faux à corriger ({raw_wrong})");
        assert!(wrong <= 1, "{wrong} pixels ≠ GMP pur (sans suivi : {raw_wrong})");
    }

    #[test]
    fn plan_scope() {
        let p = default_params_for_type(FractalType::BarnsleyJulia, 8, 8);
        assert_eq!(plan_for(&p, ReliabilityMode::Off, false), None);
        assert_eq!(plan_for(&p, ReliabilityMode::Escalate, true), None);
        assert_eq!(plan_for(&p, ReliabilityMode::Observe, false), Some(false));
        let m = default_params_for_type(FractalType::Mandelbrot, 8, 8);
        assert_eq!(plan_for(&m, ReliabilityMode::Escalate, false), None);
        let mut d = p.clone();
        d.channels.enable_distance_estimation = true;
        assert_eq!(plan_for(&d, ReliabilityMode::Escalate, false), None);
    }
}

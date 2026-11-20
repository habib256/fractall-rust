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
//!
//! Membres suivants (même soir) : **Alpha Mandelbrot**, **Mandelbulb**
//! (2D, puissance 8), **Magnet Julia** — bords ×4096 : 116 / 101 / 8 pixels
//! faux sur 2560 → 0. Leçons, toutes payées par une mesure :
//! - le multiplicateur d'un pas doit être la DÉRIVÉE EXACTE en module
//!   (`|1 + 2m|` pour Alpha, `|z − r|/|q|` pour Magnet) : l'inégalité
//!   triangulaire (`1 + 2|m|`) signalait 100 % des pixels ;
//! - aucun majorant de `|z|` sans racine élevé à une puissance : la borne
//!   est suivie AU CARRÉ (`|z|² = norm_sqr` exact), sommes par
//!   `(x + y)² ≤ (1 + δ)x² + (1 + 1/δ)y²`, δ = 2⁻¹⁰ ;
//! - borne NON linéarisée (termes en `E²`, `(|z| + E)ⁿ`) ⇒ rigoureuse à
//!   grande erreur ⇒ plafond [`E_MAX_NL`] = 0,25 au lieu de 1e-3 (31 → 17 %
//!   de pixels signalés sur le bord Alpha, tous les faux toujours pris) ;
//! - la chaîne de dépendance de `g = E²` ne doit pas dépasser celle de `z`
//!   (Alpha : +114 % → +40 % en sortant de la chaîne ce qui ne dépend pas de
//!   `g`).
//!
//! Puis **Magnet Mandelbrot**, **Mandelbrot Sin** et **Julia Sin**
//! (2026-11-19, juges GMP enfin assez rapides pour les voir) — bords ×3277 /
//! ×2608 à 64×40 : 273 / 294 pixels faux sur 2560 → 0 (Julia Sin : 0 faux à
//! 64×40, 1 à 128×80, suivi pour +1-4 %). Leçons :
//! - Magnet Mandelbrot = la carte de Magnet Julia avec `s` = le pixel :
//!   l'erreur du point est ADDITIVE dans `s − 1` et `s − 2`, le
//!   multiplicateur reste `|z − r|/|q|` (même boucle, `e_s` en plus) ;
//! - sinus : dérivée exacte `|cos ẑ|² = cos²x + sinh²y`, reste de Taylor
//!   `½E²·cosh(|y| + E)` (pas de majorant `cosh y` du premier ordre : il
//!   s'annule près de `Re z = ±π/2`) ; les composantes `sin/cos/sinh/cosh`
//!   viennent de `iterations::csin_parts`, mêmes bits que `csin` (verrou) ;
//! - vers le point fixe `z = 0` du sinus, `|sin z|` et la borne décroissent
//!   jusqu'aux SOUS-NORMAUX : +165 % sur la vue d'accueil, +26 % avec un
//!   plancher (majorant, donc sûr) [`SIN_FLOOR`] ;
//! - le double-double du sinus exige `sin/cos/exp` dd ([`dd_trig`], tables
//!   GMP + Taylor, ~340 ns/itération contre ~15 µs en GMP pur 128 b) : sans
//!   lui, 29 % de pixels signalés sur le bord iraient au GMP, hors budget.
//!   Coût réel du bord Mandelbrot Sin 256×160 : ~2 % des pixels sont au-delà
//!   du dd (vraiment : le dd y diverge du GMP) et le GMP pur de ces pixels
//!   domine la correction.
//!
//! Barnsley aligné sur les autres (même soir) : étage dd à voies entrelacées
//! avec raccourci de cycle (plus de `hypot`), erreur du pixel span compris
//! ([`pixel_error`]), arrondi du produit relatif aux OPÉRANDES `|w|₁·|c|₁`.
//! Son bord Mandelbrot est CHAOS (juge 256 b ≠ 512 b) : 99 % de pixels
//! signalés, c'est la bonne réponse.
//!
//! **Phoenix** (2026-11-20, [`scalar_phoenix`](crate::render::scalar_phoenix)) :
//! son erreur vit dans un état à deux composantes `(z, y)` de jacobien
//! `[[2z, p2], [1, 0]]` ; une borne en modules (rayon spectral
//! `|z| + √(|z|² + ½) > 1` dès `|z| > ¼`) s'emballait sur 100 % du bord. Il
//! est suivi par une borne ELLIPSOÏDALE (matrice hermitienne 2×2 transportée
//! par le jacobien exact, phases comprises) : bord ×4096 2 → 0 pixel faux,
//! 7,6 % signalés, étage dd dédié (état `(z, y)`).

use std::sync::atomic::{AtomicBool, Ordering};

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::bytecode::direct_dd::dd_from_gmp;
use crate::render::dd_trig::{self, dd_add_fast, DdTrig};
use crate::fractal::bytecode::reliability::{self as rel, ReliabilityMode};
use crate::fractal::perturbation::dd::DoubleDouble as Dd;
use crate::fractal::perturbation::{compute_perturbation_precision_bits, perf_enabled, DcGmpContext};
use crate::fractal::{FractalParams, FractalType, PlaneTransform};

/// Borne absolue (espace z) au-delà de laquelle le pixel n'est plus déterminé.
pub const E_MAX: f64 = 1e-3;
/// Plafond de la borne des boucles à borne NON linéarisée (Alpha,
/// Mandelbulb, Magnet Julia, et leurs étages double-double) : leurs bornes gardent les termes en `E²` / `(|z| + E)ⁿ`, donc
/// restent rigoureuses à grande erreur ; seul le test d'évasion juge. Un
/// plafond à 1e-3 signalait 31 % des pixels d'un bord dont 4,5 % sont faux.
const E_MAX_NL: f64 = 0.25;
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
        FractalType::BarnsleyJulia
            | FractalType::BarnsleyMandelbrot
            | FractalType::AlphaMandelbrot
            | FractalType::Mandelbulb
            | FractalType::MagnetJulia
            | FractalType::MagnetMandelbrot
            | FractalType::MandelbrotSin
            | FractalType::JuliaSin
            | FractalType::Phoenix
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
        FractalType::BarnsleyJulia => barnsley_f64(
            z_pixel,
            params.seed,
            pixel_error(params, z_pixel),
            0.0,
            params.iteration_max,
            params.bailout,
        ),
        FractalType::AlphaMandelbrot => alpha_f64(
            params.seed,
            z_pixel,
            pixel_error(params, z_pixel),
            params.iteration_max,
            params.bailout,
        ),
        FractalType::Mandelbulb => mandelbulb_f64(
            params.seed,
            z_pixel,
            pixel_error(params, z_pixel),
            params.iteration_max,
            params.bailout,
        ),
        FractalType::MagnetJulia => magnet_f64(
            z_pixel,
            params.seed,
            pixel_error(params, z_pixel),
            0.0,
            params.iteration_max,
            params.bailout,
        ),
        FractalType::MagnetMandelbrot => magnet_f64(
            Complex64::new(0.0, 0.0),
            z_pixel,
            0.0,
            pixel_error(params, z_pixel),
            params.iteration_max,
            params.bailout,
        ),
        FractalType::MandelbrotSin => sin_f64(
            params.seed,
            z_pixel,
            0.0,
            pixel_error(params, z_pixel),
            params.iteration_max,
            params.bailout,
        ),
        FractalType::JuliaSin => sin_f64(
            z_pixel,
            params.seed,
            pixel_error(params, z_pixel),
            0.0,
            params.iteration_max,
            params.bailout,
        ),
        FractalType::Phoenix => crate::render::scalar_phoenix::phoenix_f64(
            z_pixel,
            pixel_error(params, z_pixel),
            params.iteration_max,
            params.bailout,
        ),
        _ => {
            let e = pixel_error(params, z_pixel);
            barnsley_f64(z_pixel, z_pixel, e, e, params.iteration_max, params.bailout)
        }
    }
}

/// Borne de l'arrondi du point f64 du pixel par rapport au point exact du
/// juge (centre HP + dc exact) : `centre + dx` arrondi, et `dx` lui-même
/// calculé en f64 (ordre du span — domine quand le centre est proche de 0).
#[inline(always)]
fn pixel_error(params: &FractalParams, z_pixel: Complex64) -> f64 {
    U64 * (z_pixel.re.abs() + z_pixel.im.abs() + params.span_x.abs() + params.span_y.abs())
}

// --- Bornes au CARRÉ (Alpha, Mandelbulb, Magnet Julia) ----------------------
//
// Le multiplicateur d'un pas (`8|z|⁷`, `2|z|·|1 + 2m|`) doit être EXACT : un
// majorant de `|z|` sans racine (L1 : ×√2, `max + 0,414·min` : ×1,083) élevé
// à la puissance 7 transforme le multiplicateur d'un cycle attractif en
// expansion, et la borne de tout l'intérieur s'emballe (mesuré : 100 % de
// pixels signalés sur le bord Mandelbulb). On suit donc `G = E²`, où les
// carrés des multiplicateurs sont exacts (`|z|² = norm_sqr`), et l'on somme
// par `(x + y)² ≤ (1 + δ)·x² + (1 + 1/δ)·y²` : le terme propagé paie
// `1 + δ` par pas (×1,0005 sur E, ×11 en 5000 pas d'un cycle neutre),
// l'arrondi local `1 + 1/δ` (5 bits). Aucune racine.

/// δ du découpage `(x + y)² ≤ (1 + δ)x² + (1 + 1/δ)y²`.
const SPLIT: f64 = 1.0 / 1024.0;
const SPLIT_X: f64 = 1.0 + SPLIT;
const SPLIT_Y: f64 = 1.0 + 1.0 / SPLIT;

/// Le test de sortie du pas courant est-il indéterminé, ou la borne
/// trop grande ? `n2 = |z|²`, `g = E²`. `|z| ∈ [R − E, R + E]` implique
/// `|n2 − R²| ≤ E·(2R + E)` ; comme `E < E_MAX_NL` sinon, on teste
/// `(n2 − R²)² ≤ g·(2R + E_MAX_NL)²` — sans racine.
#[inline(always)]
fn sq_bound_unreliable(n2: f64, b2: f64, two_r_emax_sq: f64, g: f64) -> bool {
    let d = n2 - b2;
    d * d <= g * two_r_emax_sq || !(g < E_MAX_NL * E_MAX_NL)
}

/// Alpha Mandelbrot f64 tracé : `s = z²`, `m = s + c`, `z' = s + m² + c`
/// (opérations et ordre de `iterations::alpha_mandelbrot`). `e_c` : erreur
/// du point `c` ; la graine est exacte.
///
/// Propagation : avec `Δm = Δs + δ` (`δ` = erreur de `c` + arrondi de `m`),
/// `Δz' = Δs·(1 + 2m + Δm) + δ·(2m + Δm) + Δc` — le facteur `|1 + 2m|`
/// (et non `1 + 2|m|`) : la dérivée s'annule près de `m = −½`, et l'inégalité
/// triangulaire y faisait exploser la borne (100 % de pixels signalés).
fn alpha_f64(
    seed: Complex64,
    c: Complex64,
    e_c: f64,
    iteration_max: u32,
    bailout: f64,
) -> (u32, Complex64, bool) {
    let b2 = bailout * bailout;
    let two_r_emax_sq = (2.0 * bailout + E_MAX_NL) * (2.0 * bailout + E_MAX_NL);
    let c1 = c.re.abs() + c.im.abs();
    // Constantes hors boucle : rs² = (2u·|z|²)², arrondi de m ≤ u·|m| ≤ ½u(1 + |m|²).
    let k_rs = SPLIT_Y * 4.0 * U64 * U64;
    let half_u = 0.5 * U64;
    let k_g = 4.0 * SPLIT_X * SPLIT_Y;
    let k_m = 2.0 * SPLIT_X * SPLIT_Y;
    let dm_max = E_MAX_NL * (2.0 * bailout + E_MAX_NL) + 1.0;
    let mut z = seed;
    let mut n2 = z.norm_sqr();
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<2>::new();
    let mut g = 0.0f64; // E²
    let mut bad = false;
    while i < iteration_max && n2 < b2 {
        let z_sq = z * z;
        let m = z_sq + c;
        let m_sq = m * m;
        z = z_sq + m_sq + c;
        i += 1;
        let n2p = n2;
        n2 = z.norm_sqr();
        if !bad {
            // Chaîne de dépendance de `g` réduite au minimum (elle bornait
            // la boucle par sa LATENCE, +114 % — celle de `z` en a 28 cycles) :
            // tout ce qui ne dépend pas de `g` est calculé à côté.
            //
            // Δs = Δz·(2z + Δz) + arrondi du carré (≤ 2u·|z|²) :
            // |2z + Δz|² ≤ 4(|z| + E)² ≤ 4((1+δ)|z|² + (1+1/δ)E²), donc
            // |Δs|² ≤ (1+δ)·4E²·(…) + (1+1/δ)(2u|z|²)².
            let gs = g * ((4.0 * SPLIT_X * SPLIT_X) * n2p + k_g * g) + k_rs * n2p * n2p;
            // δ = e_c + arrondi de m (≤ u|m| ≤ ½u(1 + |m|²)).
            let mm = m.norm_sqr();
            let dm = e_c + half_u * (1.0 + mm);
            // Δz' = Δs·(1 + 2m + Δm) + δ·(2m + Δm) + Δc + arrondis, avec
            // |1 + 2m + Δm|² ≤ (1+δ)|1 + 2m|² + (1+1/δ)·2(|Δs|² + δ²) et
            // |1 + 2m|² = 1 + 4(Re m + |m|²) ; développé en |Δs|² :
            let b = (SPLIT_X * SPLIT_X) * (1.0 + 4.0 * (m.re + mm)) + k_m * dm * dm;
            // Additifs : δ·(2|m| + |Δm|) avec 2|m| ≤ 1 + |m|² et |Δm| ≤ dm_max
            // (pixel non signalé : E < E_MAX_NL, |z| < R), e_c, arrondi de m²
            // et des deux sommes.
            let add = dm * (1.0 + mm + dm_max) + e_c + 2.0 * U64 * (n2p + mm + c1);
            g = gs * (b + k_m * gs) + SPLIT_Y * add * add;
            if sq_bound_unreliable(n2, b2, two_r_emax_sq, g) {
                bad = true;
            }
        }
        cyc.observe(&mut i, iteration_max, [z.re, z.im]);
    }
    (i, z, bad)
}

/// Barnsley f64 tracé. `e0` : erreur du point de départ, `e_c` : celle de
/// `c` (le pixel pour la variante Mandelbrot — [`pixel_error`], span compris
/// — ; 0 pour la graine exacte de Julia).
fn barnsley_f64(
    z0: Complex64,
    c: Complex64,
    e0: f64,
    e_c: f64,
    iteration_max: u32,
    bailout: f64,
) -> (u32, Complex64, bool) {
    let b2 = bailout * bailout;
    // Majorants L1 (|z| ≤ |re| + |im|) : aucune racine dans la boucle.
    let c_abs = c.norm();
    // Arrondis de `z ∓ 1` (propagé par |c|) et du produit complexe (≤ u·|w|₁|c|₁
    // par composante, relatif aux OPÉRANDES : |z'| seul le sous-estime quand
    // les deux termes d'une composante se compensent).
    let k_w = e_c + 2.0 * U64 * (c.re.abs() + c.im.abs());
    let mut z = z0;
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<2>::new();
    let mut e = e0;
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
            e = c_abs * e + (w.re.abs() + w.im.abs()) * k_w;
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

/// Mandelbulb 2D (puissance 8) f64 tracé : `z' = ((z²)²)² + c` (opérations
/// et ordre de `iterations::mandelbulb`).
fn mandelbulb_f64(
    seed: Complex64,
    c: Complex64,
    e_c: f64,
    iteration_max: u32,
    bailout: f64,
) -> (u32, Complex64, bool) {
    let b2 = bailout * bailout;
    let two_r_emax_sq = (2.0 * bailout + E_MAX_NL) * (2.0 * bailout + E_MAX_NL);
    let mut z = seed;
    let mut n2 = z.norm_sqr();
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<2>::new();
    let mut g = 0.0f64; // E²
    let mut bad = false;
    while i < iteration_max && n2 < b2 {
        let mut z_temp = z * z;
        z_temp = z_temp * z_temp;
        z_temp = z_temp * z_temp;
        z = z_temp + c;
        i += 1;
        let n2p = n2;
        n2 = z.norm_sqr();
        if !bad {
            // |(z+Δ)⁸ − z⁸| ≤ 8·Δ·(|z| + Δ)⁷, au carré : 64·E²·((|z|+E)²)⁷
            // avec (|z|+E)² ≤ (1+δ)|z|² + (1+1/δ)E². Arrondi des trois carrés
            // (erreur relative doublée à chaque carré : ≤ 7·2u·|z|⁸) + somme.
            let q = SPLIT_X * n2p + SPLIT_Y * g;
            let q2 = q * q;
            let q7 = q2 * q2 * q2 * q;
            let n4 = n2p * n2p;
            let add = 16.0 * U64 * n4 * n4 + e_c + U64 * (z.re.abs() + z.im.abs());
            g = SPLIT_X * 64.0 * g * q7 + SPLIT_Y * add * add;
            if sq_bound_unreliable(n2, b2, two_r_emax_sq, g) {
                bad = true;
            }
        }
        cyc.observe(&mut i, iteration_max, [z.re, z.im]);
    }
    (i, z, bad)
}

/// Magnet f64 tracé : `n = z² + (s − 1)`, `q = 2z + (s − 2)`,
/// `r = n / q`, `z' = r²` (opérations, ordre, sortie sur `|q|² < 1e-24` et
/// raccourci de cycle de `iterations::magnet_julia` / `magnet_mandelbrot`).
///
/// Propagation sans inégalité triangulaire sur le quotient : avec `Δq = 2Δ`
/// et `Δn = Δ(2z + Δ)`, `n* q̂ − n̂ q* = Δ·(2(z − r) + Δ)·q̂ + arrondis`, donc
/// `|Δr| ≤ [E(2|z − r| + E) + ρ_n + |r|ρ_q + …] / (|q| − 2E − ρ_q)` — le
/// facteur `|z − r|` est la dérivée exacte ; `1 + |r|` à la place
/// signalerait tout l'intérieur. Bornes au carré, sans racine.
///
/// Magnet MANDELBROT (même carte, `s` = le pixel, `z₀ = 0`) : l'erreur `e_s`
/// du point entre dans les deux constantes `s − 1` et `s − 2` — additive,
/// donc comptée avec leur arrondi ; le multiplicateur reste `|z − r|/|q|`.
fn magnet_f64(
    z0: Complex64,
    seed: Complex64,
    e0: f64,
    e_s: f64,
    iteration_max: u32,
    bailout: f64,
) -> (u32, Complex64, bool) {
    let b2 = bailout * bailout;
    let two_r_emax_sq = (2.0 * bailout + E_MAX_NL) * (2.0 * bailout + E_MAX_NL);
    let seed_minus_one = Complex64::new(seed.re - 1.0, seed.im);
    let seed_minus_two = Complex64::new(seed.re - 2.0, seed.im);
    // Arrondi des constantes (le juge les forme exactes depuis la graine ou
    // le point exact du pixel), plus l'erreur du point lui-même.
    let s1 = U64 * (seed_minus_one.re.abs() + seed_minus_one.im.abs()) + e_s;
    let s2 = U64 * (seed_minus_two.re.abs() + seed_minus_two.im.abs()) + e_s;
    let one_m_split = 1.0 - SPLIT;
    let mut z = z0;
    let mut n2 = z.norm_sqr();
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<2>::new();
    let mut g = e0 * e0; // E²
    let mut bad = false;
    while i < iteration_max && n2 < b2 {
        let n = z * z + seed_minus_one;
        let q = Complex64::new(2.0, 0.0) * z + seed_minus_two;
        let qq = q.norm_sqr();
        // Eviter division par zero
        if qq < 1e-24 {
            // Le vrai |q|² est-il aussi sous le seuil ? |Δq| ≤ d = 2E + ρ_q,
            // donc |q*|² ≤ (|q̂| + d)² ≤ (1+δ)|q̂|² + (1+1/δ)d² — c'est CETTE
            // somme qu'il faut comparer au seuil, pas d² seul : un |q̂| juste
            // sous 1e-12 avec une borne minuscule passait pour déterminé alors
            // que le point exact était au-dessus (verrou
            // `magnet_pole_exit_is_judged_on_q_plus_error`). Vaut pour les
            // deux variantes (Magnet Mandelbrot : ρ_q porte l'erreur du pixel).
            let rho_q = U64 * (q.re.abs() + q.im.abs()) + s2;
            if SPLIT_X * qq + SPLIT_Y * (8.0 * g + 2.0 * rho_q * rho_q) >= 1e-24 {
                bad = true;
            }
            break;
        }
        let ratio = n / q;
        let zp = z;
        let n2p = n2;
        z = ratio * ratio;
        i += 1;
        n2 = z.norm_sqr();
        if !bad {
            let rr = ratio.norm_sqr();
            let r1 = ratio.re.abs() + ratio.im.abs();
            // Arrondis de n (carré + somme) et de q (somme ; 2z exact).
            let rho_n = 2.0 * U64 * n2p + U64 * (n.re.abs() + n.im.abs()) + s1;
            let rho_q = U64 * (q.re.abs() + q.im.abs()) + s2;
            // |q*|² ≥ (1−δ)|q̂|² − (1/δ)(2E + ρ_q)², (2E + ρ_q)² ≤ 8E² + 2ρ_q².
            let den = one_m_split * qq - SPLIT_Y * (8.0 * g + 2.0 * rho_q * rho_q);
            // Erreur de la division (normwise ≤ 6u) sur 2E + ρ_q, E < E_MAX_NL.
            let b = rho_n + r1 * rho_q + 4.0 * U64 * r1 * (2.0 * E_MAX_NL + rho_q);
            let (wr, wi) = (zp.re - ratio.re, zp.im - ratio.im);
            let a2 = g * (SPLIT_X * 4.0 * (wr * wr + wi * wi) + SPLIT_Y * g);
            let er_div = 4.0 * U64 * r1;
            let gr = SPLIT_X * (SPLIT_X * a2 + SPLIT_Y * b * b) / den + SPLIT_Y * er_div * er_div;
            // z' = r² : Δz' = Δr(2r + Δr) + arrondi du carré.
            let rq = 2.0 * U64 * rr;
            g = SPLIT_X * gr * (4.0 * SPLIT_X * rr + SPLIT_Y * gr) + SPLIT_Y * rq * rq;
            if !(den > 1e-20) || sq_bound_unreliable(n2, b2, two_r_emax_sq, g) {
                bad = true;
            }
        }
        cyc.observe_each(&mut i, iteration_max, [z.re, z.im]);
    }
    (i, z, bad)
}

/// Plancher de `|sin ẑ|₁` dans les bornes Sin (majorant : sûr).
const SIN_FLOOR: f64 = 1e-60;

/// Mandelbrot Sin / Julia Sin f64 tracé : `z' = k·sin z` (opérations et
/// raccourci de cycle de `iterations::mandelbrot_sin` — `k` = pixel, `z₀` =
/// graine — et `julia_sin` — `k` = graine, `z₀` = pixel), `sin` par les
/// composantes de `iterations::csin_parts` (mêmes bits que `csin`).
///
/// Propagation autour du `ẑ` CALCULÉ : `Δ' = k*·(sin z* − sin ẑ) +
/// (k* − k̂)·sin ẑ + arrondis`, et par Taylor à reste intégral
/// `|sin(ẑ + Δ) − sin ẑ| ≤ E·|cos ẑ| + ½E²·cosh(|Im ẑ| + E)` — le facteur
/// `|cos ẑ|` est la dérivée EXACTE (`|cos z|² = cos²x + sinh²y`, sans racine
/// ni majorant) : il s'annule près de `Re z = ±π/2`, où un majorant `cosh y`
/// emballerait la borne de tout l'intérieur. `cosh(|y| + E) ≤ cosh y·e^E`,
/// `e^E ≤ 1,2841` sous [`E_MAX_NL`]. Arrondis : `csin` ≤ 8u par composante
/// (sincos 1 ulp, `expm1` et ses deux roundings), produit complexe 2u·|k|₁.
fn sin_f64(
    z0: Complex64,
    k: Complex64,
    e0: f64,
    e_k: f64,
    iteration_max: u32,
    bailout: f64,
) -> (u32, Complex64, bool) {
    use crate::fractal::iterations::csin_parts;
    let b2 = bailout * bailout;
    let two_r_emax_sq = (2.0 * bailout + E_MAX_NL) * (2.0 * bailout + E_MAX_NL);
    let k_abs = k.norm();
    let k1 = k.re.abs() + k.im.abs();
    // Marges relatives (1e-12) : arrondis du calcul de la borne elle-même.
    let kk = SPLIT_X * (k_abs + e_k) * (k_abs + e_k) * (1.0 + 1e-12);
    let a_coef = (e_k + 10.0 * U64 * k1) * (1.0 + 1e-12);
    let kh = SPLIT_Y * 0.25 * 1.29 * 1.29;
    let cos_margin = SPLIT_X * (1.0 + 16.0 * U64);
    let mut z = z0;
    let mut n2 = z.norm_sqr();
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<2>::new();
    let mut g = e0 * e0; // E²
    let mut bad = false;
    while i < iteration_max && n2 < b2 {
        let (s, c, sh, ch) = csin_parts(z);
        z = k * Complex64::new(s * ch, c * sh);
        i += 1;
        n2 = z.norm_sqr();
        if !bad {
            // |Δ'|² ≤ (1+δ)|k*|²·E²·((1+δ)|cos ẑ|² + (1+1/δ)·E²·(1,29·ch/2)²)
            //        + (1+1/δ)·(a·|sin ẑ|₁)².
            let cos2 = cos_margin * (c * c + sh * sh);
            // Plancher (majorant, donc sûr) : vers le point fixe z = 0, |sin z|
            // et la borne décroissent géométriquement jusqu'aux SOUS-NORMAUX,
            // dont chaque opération coûte une assistance micro-code (+165 % sur
            // la vue d'accueil Mandelbrot Sin).
            let add = a_coef * (s.abs() * ch + c.abs() * sh.abs()).max(SIN_FLOOR);
            g = kk * g * (cos2 + kh * ch * ch * g) + SPLIT_Y * add * add;
            if sq_bound_unreliable(n2, b2, two_r_emax_sq, g) {
                bad = true;
            }
        }
        cyc.observe_each(&mut i, iteration_max, [z.re, z.im]);
    }
    (i, z, bad)
}

/// Complexe double-double minimal des boucles de correction.
#[derive(Clone, Copy)]
struct Cdd {
    re: Dd,
    im: Dd,
}

impl Cdd {
    #[inline(always)]
    fn sqr(self) -> Self {
        Cdd {
            re: self.re.sqr().sub(self.im.sqr()),
            im: self.re.mul(self.im).mul_f64(2.0),
        }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Cdd { re: self.re.add(o.re), im: self.im.add(o.im) }
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Cdd {
            re: self.re.mul(o.re).sub(self.im.mul(o.im)),
            im: self.re.mul(o.im).add(self.im.mul(o.re)),
        }
    }
    /// Carré avec la soustraction « rapide » ([`dd_add_fast`]).
    #[inline(always)]
    fn sqr_fast(self) -> Self {
        Cdd {
            re: dd_add_fast(self.re.sqr(), self.im.sqr().neg()),
            im: self.re.mul(self.im).mul_f64(2.0),
        }
    }
    #[inline(always)]
    fn add_fast(self, o: Self) -> Self {
        Cdd { re: dd_add_fast(self.re, o.re), im: dd_add_fast(self.im, o.im) }
    }
    #[inline(always)]
    fn abs_hi(self) -> f64 {
        // |z| < R ≪ 1e150 dans ces boucles : pas de débordement, pas de
        // `hypot` (bien plus lent qu'une racine).
        (self.re.hi * self.re.hi + self.im.hi * self.im.hi).sqrt()
    }
    #[inline(always)]
    fn hi(self) -> Complex64 {
        Complex64::new(self.re.hi, self.im.hi)
    }
}

/// Fin d'un pas double-double : `(continuer, fiable)` — la borne `e` doit
/// rester petite et l'évasion déterminée.
#[inline(always)]
fn dd_step_ok(n: f64, bailout: f64, e: f64) -> bool {
    n.is_finite() && (n - bailout).abs() > e && e < E_MAX_NL
}

/// Un pas Alpha Mandelbrot double-double tracé : `(z', E', |z'|)`.
#[inline(always)]
fn alpha_dd_step(z: Cdd, e: f64, c: Cdd) -> (Cdd, f64, f64) {
    let c_abs = c.abs_hi();
    let e_c = UDD * c_abs;
    let a = z.abs_hi();
    let s = z.sqr_fast();
    let m = s.add_fast(c);
    let m2 = m.sqr_fast();
    let z = s.add_fast(m2).add_fast(c);
    // Même propagation que `alpha_f64` : Δs·(1 + 2m + Δm) + δ·(2m + Δm) + Δc.
    // Sommes rapides : arrondi relatif aux opérandes (m : |s| + |c|).
    let es = e * (2.0 * a + e) + 2.0 * UDD * a * a;
    let bm = m.abs_hi();
    let dm = e_c + UDD * (a * a + c_abs);
    let em = es + dm;
    let (pr, pi) = (1.0 + 2.0 * m.re.hi, 2.0 * m.im.hi);
    let f = (pr * pr + pi * pi).sqrt() + em;
    let e = es * f + dm * (2.0 * bm + em) + e_c + 2.0 * UDD * (a * a + bm * bm + c_abs);
    (z, e, z.abs_hi())
}

/// Un pas Mandelbulb (puissance 8) double-double tracé : `(z', E', |z'|)`.
#[inline(always)]
fn mandelbulb_dd_step(z: Cdd, e: f64, c: Cdd) -> (Cdd, f64, f64) {
    let c_abs = c.abs_hi();
    let e_c = UDD * c_abs;
    let a = z.abs_hi();
    let z = z.sqr_fast().sqr_fast().sqr_fast().add_fast(c);
    let t = a + e;
    let t2 = t * t;
    let t7 = t2 * t2 * t2 * t;
    let a2 = a * a;
    let a8 = a2 * a2 * a2 * a2;
    let n = z.abs_hi();
    // Somme rapide : arrondi relatif aux opérandes |z⁸| + |c|.
    (z, 8.0 * e * t7 + 16.0 * UDD * a8 + e_c + UDD * (a8 + c_abs + n), n)
}

/// Un pas Magnet Julia double-double tracé : `(z', E', |z'|)` — même
/// propagation que [`magnet_julia_f64`]. `E' = ∞` si `|q|` n'est pas
/// séparé de 0 (le pixel part au GMP, qui décide la sortie `|q|² < 1e-24`).
#[inline(always)]
fn magnet_dd_step(z: Cdd, e: f64, s1: Cdd, s2: Cdd, e_s: f64) -> (Cdd, f64, f64) {
    let n = z.sqr().add(s1);
    let q = Cdd { re: z.re.mul_f64(2.0), im: z.im.mul_f64(2.0) }.add(s2);
    let qq = q.re.sqr().add(q.im.sqr());
    // r = n·conj(q) / |q|²
    let nr = n.re.mul(q.re).add(n.im.mul(q.im));
    let ni = n.im.mul(q.re).sub(n.re.mul(q.im));
    let r = Cdd { re: nr.div(qq), im: ni.div(qq) };
    let zn = r.sqr();
    let q_abs = qq.hi.sqrt();
    let r_abs = r.abs_hi();
    let rho_n =
        UDD * (2.0 * (z.re.hi * z.re.hi + z.im.hi * z.im.hi) + n.abs_hi() + s1.abs_hi()) + e_s;
    let rho_q = UDD * (q_abs + s2.abs_hi()) + e_s;
    let den = q_abs - 2.0 * e - rho_q;
    let (wr, wi) = (z.re.hi - r.re.hi, z.im.hi - r.im.hi);
    let w = (wr * wr + wi * wi).sqrt();
    let er = (e * (2.0 * w + e) + rho_n + r_abs * rho_q + 8.0 * UDD * r_abs * (2.0 * e + rho_q))
        / den
        + 8.0 * UDD * r_abs;
    let e2 = if den > 1e-10 && qq.hi >= 1e-20 {
        er * (2.0 * r_abs + er) + 2.0 * UDD * r_abs * r_abs
    } else {
        f64::INFINITY
    };
    (zn, e2, zn.abs_hi())
}

/// Un pas Sin double-double tracé : `z' = k·sin z`, `(z', E', |z'|)` — même
/// propagation que [`sin_f64`], erreurs de `sin`/`cos`/`sinh`/`cosh` dd
/// bornées par [`dd_trig`](crate::render::dd_trig) (absolues, ∝ `cosh y`).
/// `E' = ∞` hors du domaine des fonctions dd (pixel au GMP).
#[inline(always)]
fn sin_dd_step(trig: &DdTrig, z: Cdd, e: f64, k: Cdd, e_k: f64) -> (Cdd, f64, f64) {
    let (Some((sx, cx)), Some((sh, ch))) = (trig.sin_cos(z.re), trig.sinh_cosh(z.im)) else {
        return (z, f64::INFINITY, f64::NAN);
    };
    let zn = k.mul(Cdd { re: sx.mul(ch), im: cx.mul(sh) });
    let (s, c, shf, chf) = (sx.hi, cx.hi, sh.hi, ch.hi);
    let cos_abs = (c * c + shf * shf).sqrt();
    // Plancher anti-sous-normaux (cf. `sin_f64`).
    let sin1 = (s.abs() * chf + c.abs() * shf.abs()).max(SIN_FLOOR);
    // Erreur absolue de sin ẑ (deux composantes, produits dd compris).
    let eps_sin = 2.0
        * (dd_trig::SIN_ERR * (1.0 + z.re.hi.abs()) + dd_trig::HYP_ERR + 2.0)
        * UDD
        * chf;
    let k_abs = k.abs_hi();
    let k1 = k.re.hi.abs() + k.im.hi.abs();
    let e2 = ((k_abs + e_k) * e * (cos_abs + 0.5 * e * chf * (1.0 + 2.0 * e))
        + (e_k + 4.0 * UDD * k1) * (sin1 + eps_sin)
        + k_abs * eps_sin)
        * (1.0 + 1e-12);
    (zn, e2, zn.abs_hi())
}

/// État complet d'un pas dd pour le raccourci des cycles exacts : `z` (les
/// deux mots de chaque composante) ET la borne `E` — le pas suivant lit les
/// deux, donc un état répété bit pour bit répète tout l'avenir, borne
/// comprise (sauter des cycles ne cache aucune croissance de la borne).
/// L'intérieur d'un Magnet signalé coûtait `iteration_max` pas dd.
#[inline(always)]
fn dd_cycle_state(z: Cdd, e: f64) -> [f64; 5] {
    [z.re.hi, z.re.lo, z.im.hi, z.im.lo, e]
}

/// Un pixel double-double tracé (référence des voies de [`dd_lanes`]).
/// Renvoie `(itérations, z, fiable)`.
#[cfg(test)]
fn dd_pixel(
    seed: Cdd,
    c: Cdd,
    iteration_max: u32,
    bailout: f64,
    step: impl Fn(Cdd, f64, Cdd) -> (Cdd, f64, f64),
) -> (u32, Complex64, bool) {
    let b2 = bailout * bailout;
    let mut z = seed;
    let mut i = 0u32;
    let mut e = 0.0f64;
    let mut cyc = crate::fractal::cycle::CycleCut::<5>::new();
    while i < iteration_max && z.re.hi * z.re.hi + z.im.hi * z.im.hi < b2 {
        let (nz, ne, n) = step(z, e, c);
        z = nz;
        e = ne;
        i += 1;
        if !dd_step_ok(n, bailout, e) {
            return (i, z.hi(), false);
        }
        cyc.observe(&mut i, iteration_max, dd_cycle_state(z, e));
    }
    (i, z.hi(), true)
}

#[cfg(test)]
fn alpha_dd(seed: Cdd, c: Cdd, iteration_max: u32, bailout: f64) -> (u32, Complex64, bool) {
    dd_pixel(seed, c, iteration_max, bailout, alpha_dd_step)
}

#[cfg(test)]
fn mandelbulb_dd(seed: Cdd, c: Cdd, iteration_max: u32, bailout: f64) -> (u32, Complex64, bool) {
    dd_pixel(seed, c, iteration_max, bailout, mandelbulb_dd_step)
}

/// Voies entrelacées de l'étage double-double : le pas dd est une longue
/// chaîne dépendante de `two_sum`/`two_prod`, bornée par sa LATENCE —
/// [`DD_LANES`] pixels avancent de front, chaque voie rechargée dès qu'elle
/// finit. Mesuré (un fil, bords de l'axe `types`) : ns/itération dd 60 → 44
/// (Alpha), 64 → 41 (Mandelbulb), 4,0 → 1,3 (Magnet, cycles raccourcis). Chaque voie fait exactement la suite
/// d'opérations de [`dd_pixel`] : résultats identiques (verrou
/// `dd_lanes_match_single_pixel_loop`).
const DD_LANES: usize = 4;

/// [`dd_lanes`] en parallèle par paquets (monomorphisé sur le pas : un
/// appel indirect empêcherait l'entrelacement des voies).
fn dd_lanes_par<F: Fn(Cdd, f64, Cdd) -> (Cdd, f64, f64) + Sync + Copy>(
    points: &[Cdd],
    seed: Cdd,
    julia: bool,
    iteration_max: u32,
    bailout: f64,
    step: F,
) -> Vec<(u32, Complex64, bool)> {
    const CHUNK: usize = 64;
    points
        .par_chunks(CHUNK)
        .flat_map_iter(|ch| dd_lanes(ch, seed, julia, iteration_max, bailout, step))
        .collect()
}

///
/// `julia` : le point du pixel est le `z` de départ (borne initiale = son
/// arrondi en dd) et le pas ignore son troisième argument ; sinon `z` part de
/// `seed` et le point est le `c` du pas.
fn dd_lanes(
    points: &[Cdd],
    seed: Cdd,
    julia: bool,
    iteration_max: u32,
    bailout: f64,
    step: impl Fn(Cdd, f64, Cdd) -> (Cdd, f64, f64),
) -> Vec<(u32, Complex64, bool)> {
    const FREE: usize = usize::MAX;
    let b2 = bailout * bailout;
    let mut out = vec![(0u32, Complex64::new(0.0, 0.0), false); points.len()];
    let mut next = 0usize;
    let mut idx = [FREE; DD_LANES];
    let mut z = [seed; DD_LANES];
    let mut c = [seed; DD_LANES];
    let mut e = [0.0f64; DD_LANES];
    let mut n = [0.0f64; DD_LANES];
    let mut it = [0u32; DD_LANES];
    let mut cyc: [crate::fractal::cycle::CycleCut<5>; DD_LANES] =
        std::array::from_fn(|_| crate::fractal::cycle::CycleCut::new());
    loop {
        // Sorties par la condition de boucle, puis recharge des voies libres.
        for l in 0..DD_LANES {
            loop {
                if idx[l] != FREE {
                    let zl = z[l];
                    if it[l] < iteration_max && zl.re.hi * zl.re.hi + zl.im.hi * zl.im.hi < b2 {
                        break;
                    }
                    out[idx[l]] = (it[l], zl.hi(), true);
                    idx[l] = FREE;
                }
                if next == points.len() {
                    break;
                }
                idx[l] = next;
                c[l] = points[next];
                cyc[l] = crate::fractal::cycle::CycleCut::new();
                if julia {
                    z[l] = points[next];
                    e[l] = UDD * points[next].abs_hi();
                } else {
                    z[l] = seed;
                    e[l] = 0.0;
                }
                it[l] = 0;
                next += 1;
            }
        }
        if idx.iter().all(|&k| k == FREE) {
            break;
        }
        // Un pas sur toutes les voies (une voie libre calcule pour rien).
        for l in 0..DD_LANES {
            let (nz, ne, nn) = step(z[l], e[l], c[l]);
            z[l] = nz;
            e[l] = ne;
            n[l] = nn;
        }
        for l in 0..DD_LANES {
            if idx[l] != FREE {
                it[l] += 1;
                if !dd_step_ok(n[l], bailout, e[l]) {
                    out[idx[l]] = (it[l], z[l].hi(), false);
                    idx[l] = FREE;
                } else {
                    cyc[l].observe(&mut it[l], iteration_max, dd_cycle_state(z[l], e[l]));
                }
            }
        }
    }
    out
}

/// Un pas Barnsley double-double tracé : `z' = (z ∓ 1)·c`, `(z', E', |z'|)`.
/// Borne LINÉARISÉE (plafond [`E_MAX`], comme `barnsley_f64`) ; `E' = ∞` si
/// la branche `Re z ≥ 0` n'est pas déterminée ou si la borne dépasse
/// [`E_MAX`] (le pixel part au GMP). Arrondis relatifs aux opérandes
/// (`|w|₁·|c|₁`), sans `hypot`.
#[inline(always)]
fn barnsley_dd_step(z: Cdd, e: f64, c: Cdd, e_c: f64) -> (Cdd, f64, f64) {
    if z.re.hi.abs() <= e {
        return (z, f64::INFINITY, f64::NAN);
    }
    let one = Dd::from_f64(1.0);
    let wx = if z.re.hi >= 0.0 { z.re.sub(one) } else { z.re.add(one) };
    let w = Cdd { re: wx, im: z.im };
    let zn = w.mul(c);
    let w1 = w.re.hi.abs() + w.im.hi.abs();
    let c1 = c.re.hi.abs() + c.im.hi.abs();
    let e2 = c.abs_hi() * e + w1 * (e_c + 4.0 * UDD * c1);
    (zn, if e2 < E_MAX { e2 } else { f64::INFINITY }, zn.abs_hi())
}

/// Étage double-double (voies entrelacées, tous les types suivis).
/// `points` : point exact de chaque pixel, arrondi en dd.
fn dd_stage(params: &FractalParams, points: &[(Dd, Dd)]) -> Option<Vec<(u32, Complex64, bool)>> {
    let (n, r) = (params.iteration_max, params.bailout);
    let seed = (Dd::from_f64(params.seed.re), Dd::from_f64(params.seed.im));
    let seed_c = Cdd { re: seed.0, im: seed.1 };
    let pts = || -> Vec<Cdd> { points.iter().map(|&(re, im)| Cdd { re, im }).collect() };
    match params.fractal_type {
        // Barnsley : `z₀` = le point (julia = true) ; la variante Mandelbrot
        // reçoit aussi le point comme `c`, la Julia prend la graine exacte.
        FractalType::BarnsleyJulia => {
            let step = move |z: Cdd, e: f64, _: Cdd| barnsley_dd_step(z, e, seed_c, 0.0);
            Some(dd_lanes_par(&pts(), seed_c, true, n, r, step))
        }
        FractalType::BarnsleyMandelbrot => {
            let step = |z: Cdd, e: f64, c: Cdd| barnsley_dd_step(z, e, c, UDD * c.abs_hi());
            Some(dd_lanes_par(&pts(), seed_c, true, n, r, step))
        }
        FractalType::AlphaMandelbrot => {
            Some(dd_lanes_par(&pts(), seed_c, false, n, r, alpha_dd_step))
        }
        FractalType::Mandelbulb => Some(dd_lanes_par(&pts(), seed_c, false, n, r, mandelbulb_dd_step)),
        FractalType::MagnetJulia => {
            let s1 = Cdd { re: seed.0.sub(Dd::from_f64(1.0)), im: seed.1 };
            let s2 = Cdd { re: seed.0.sub(Dd::from_f64(2.0)), im: seed.1 };
            let step = move |z: Cdd, e: f64, _: Cdd| magnet_dd_step(z, e, s1, s2, 0.0);
            Some(dd_lanes_par(&pts(), seed_c, true, n, r, step))
        }
        FractalType::MandelbrotSin => {
            let trig = DdTrig::get();
            let step = move |z: Cdd, e: f64, c: Cdd| sin_dd_step(trig, z, e, c, UDD * c.abs_hi());
            Some(dd_lanes_par(&pts(), seed_c, false, n, r, step))
        }
        FractalType::JuliaSin => {
            let trig = DdTrig::get();
            let step = move |z: Cdd, e: f64, _: Cdd| sin_dd_step(trig, z, e, seed_c, 0.0);
            Some(dd_lanes_par(&pts(), seed_c, true, n, r, step))
        }
        FractalType::MagnetMandelbrot => {
            // z₀ = 0 exact ; le point (arrondi en dd) entre dans s − 1, s − 2.
            let zero = Cdd { re: Dd::ZERO, im: Dd::ZERO };
            let step = |z: Cdd, e: f64, c: Cdd| {
                let s1 = Cdd { re: c.re.sub(Dd::from_f64(1.0)), im: c.im };
                let s2 = Cdd { re: c.re.sub(Dd::from_f64(2.0)), im: c.im };
                magnet_dd_step(z, e, s1, s2, UDD * c.abs_hi())
            };
            Some(dd_lanes_par(&pts(), zero, false, n, r, step))
        }
        // État à deux composantes (z, y) : étage dédié (`scalar_phoenix`).
        FractalType::Phoenix => Some(crate::render::scalar_phoenix::dd_stage(points, n, r)),
        _ => None,
    }
}

/// Précision minimale de l'étage GMP qui suit le double-double.
const GMP_FLOOR_BITS: u32 = 256;

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
    // Point exact de chaque pixel (centre HP + dc en GMP), arrondi en dd.
    let points: Vec<(Dd, Dd)> = flagged
        .par_iter()
        .map(|&idx| {
            let dc = dc_ctx.compute_dc(idx % width, idx / width);
            let mut re = cx.clone();
            re += dc.real();
            let mut im = cy.clone();
            im += dc.imag();
            (dd_from_gmp(&re), dd_from_gmp(&im))
        })
        .collect();
    // Tout type suivi a son étage dd ; sinon tout part au GMP.
    let res = dd_stage(params, &points)
        .unwrap_or_else(|| vec![(0, Complex64::new(0.0, 0.0), false); points.len()]);
    let tried: Vec<(usize, u32, Complex64, bool)> =
        flagged.iter().zip(res).map(|(&idx, (it, z, ok))| (idx, it, z, ok)).collect();
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
        // Le GMP arbitre des pixels que le double-double (~106 b) n'a PAS su
        // déterminer : à la précision de la vue (128 b dès que
        // `precision_bits` < 128 sur une vue peu profonde) il n'a que ~20 bits
        // de plus et rend une valeur arbitraire présentée comme corrigée
        // (verrou `gmp_stage_runs_well_above_dd_precision`). Plancher
        // GMP_FLOOR_BITS ≈ 2,4 × dd — la précision par défaut.
        let mut gp;
        let gmp_params = if params.engine.precision_bits < GMP_FLOOR_BITS {
            gp = params.clone();
            gp.engine.precision_bits = GMP_FLOOR_BITS;
            &gp
        } else {
            params
        };
        crate::render::std_reliability::correct_std_pixels(
            gmp_params,
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

    /// Vues d'accueil (`None`) et de bord (scènes de l'axe `types`).
    const SCENES: &[(FractalType, Option<(f64, f64, f64)>)] = &[
        (FractalType::BarnsleyJulia, Some((-0.50238037109375, -1.9626342773437502, 2048.0))),
        (FractalType::BarnsleyMandelbrot, None),
        (FractalType::BarnsleyMandelbrot, Some((-0.1156768798828125, -1.3585296630859376, 2730.666666666666))),
        (FractalType::AlphaMandelbrot, None),
        (FractalType::AlphaMandelbrot, Some((-1.260833740234375, -0.350164794921875, 4096.0))),
        (FractalType::Mandelbulb, None),
        (FractalType::Mandelbulb, Some((-0.7341842651367188, -0.23336334228515626, 5461.333))),
        (FractalType::MagnetJulia, None),
        (FractalType::MagnetJulia, Some((0.233306884765625, -1.676715087890625, 4096.0))),
        (FractalType::MagnetMandelbrot, None),
        (FractalType::MagnetMandelbrot, Some((0.6216079711914062, -1.4390701293945312, 3276.8))),
        (FractalType::MandelbrotSin, None),
        (FractalType::MandelbrotSin, Some((-1.9899086401850268, -0.5337651978513115, 2607.5945876176133))),
        (FractalType::JuliaSin, None),
        (FractalType::JuliaSin, Some((4.7936899621426774e-05, -0.00012437648618453408, 2607.5945876176133))),
        (FractalType::Phoenix, None),
        // Bord Phoenix plus dur que la scène de l'axe (×4096 : 2 pixels faux
        // à 64×40 seulement) : 5 pixels faux sans suivi.
        (FractalType::Phoenix, Some((0.629583796356445, -0.523261146274227, 30907.369550505096))),
    ];

    fn scene_params(ft: FractalType, view: Option<(f64, f64, f64)>, w: usize, h: usize) -> FractalParams {
        let mut p = default_params_for_type(ft, w as u32, h as u32);
        if let Some((cx, cy, zoom)) = view {
            p.center_x = cx;
            p.center_y = cy;
            p.span_x = 4.0 / zoom;
            p.span_y = p.span_x * h as f64 / w as f64;
        }
        p
    }

    fn scene_points(p: &FractalParams, w: usize, h: usize) -> Vec<Complex64> {
        (0..w * h)
            .map(|k| {
                let (i, j) = (k % w, k / w);
                Complex64::new(
                    p.center_x + ((i as f64 + 0.5) / w as f64 - 0.5) * p.span_x,
                    p.center_y + ((j as f64 + 0.5) / h as f64 - 0.5) * p.span_y,
                )
            })
            .collect()
    }

    fn dd_points(pts: &[Complex64]) -> Vec<(Dd, Dd)> {
        pts.iter().map(|z| (Dd::from_f64(z.re), Dd::from_f64(z.im))).collect()
    }

    /// Mêmes verrous pour les autres types suivis : vue d'accueil ET scène de
    /// bord de l'axe `types` (là où les bornes s'emballent), seed non nulle
    /// comprise pour les types à graine.
    #[test]
    fn tracked_scalar_loops_are_bit_identical_to_production_loops() {
        for &(ft, view) in SCENES {
            for seeded in [false, true] {
                let (w, h) = (40usize, 30usize);
                let mut p = scene_params(ft, view, w, h);
                p.iteration_max = 3000;
                if seeded {
                    p.seed = if ft == FractalType::MagnetJulia {
                        Complex64::new(1.3, 0.2)
                    } else {
                        Complex64::new(0.01, -0.02)
                    };
                }
                let mut flagged = 0;
                for (k, &zp) in scene_points(&p, w, h).iter().enumerate() {
                    let r = iterate_point(&p, zp);
                    let (it, z, bad) = iterate_tracked(&p, zp);
                    assert_eq!(it, r.iteration, "{ft:?} {view:?} pixel {k}");
                    assert_eq!(z.re.to_bits(), r.z.re.to_bits(), "{ft:?} pixel {k}");
                    assert_eq!(z.im.to_bits(), r.z.im.to_bits(), "{ft:?} pixel {k}");
                    flagged += bad as usize;
                }
                // Bord Barnsley Mandelbrot : CHAOS (juge 256 b ≠ 512 b sur
                // 73 % des pixels) — tout signaler y est la bonne réponse.
                if !(ft == FractalType::BarnsleyMandelbrot && view.is_some()) {
                    assert!(flagged < w * h / 2, "{ft:?} {view:?} : {flagged} signalés");
                }
            }
        }
    }

    /// Scènes de bord de l'axe `types` (64×40, 5000 itérations, juge GMP pur
    /// convergé) : 116 / 101 / 8 pixels faux sans suivi pour Alpha
    /// Mandelbrot / Mandelbulb / Magnet Julia ; plus aucun avec.
    #[test]
    fn tracked_scalar_edges_match_gmp() {
        // Julia Sin : aucun pixel faux à cette taille (1 à 128×80, corrigé —
        // `diag_julia_sin_wrong`) : rien à verrouiller contre le GMP. Bord
        // Barnsley Mandelbrot : juge non convergé (CHAOS), pas un juge.
        let judged = |ft: FractalType| !matches!(ft, FractalType::JuliaSin | FractalType::BarnsleyMandelbrot);
        for &(ft, view) in SCENES.iter().filter(|s| s.1.is_some() && judged(s.0)) {
            let (w, h) = (64usize, 40usize);
            let mut p = scene_params(ft, view, w, h);
            p.iteration_max = 5000;
            assert_eq!(plan_for(&p, ReliabilityMode::Escalate, false), Some(true));
            let prod = crate::render::render_escape_time(&p);
            let mut g = p.clone();
            g.engine.algorithm_mode = AlgorithmMode::ReferenceGmp;
            // Sans `use_gmp`, un type hors famille perturbation retombait sur
            // le path f64 standard — suivi et corrigé : le « juge » était la
            // production elle-même.
            g.engine.use_gmp = true;
            let t = std::time::Instant::now();
            let gmp = crate::render::render_escape_time(&g);
            eprintln!("{ft:?} juge GMP : {:.1} s", t.elapsed().as_secs_f64());
            let wrong = prod.iterations.iter().zip(&gmp.iterations).filter(|(a, b)| a != b).count();
            let raw_wrong = scene_points(&p, w, h)
                .iter()
                .zip(&gmp.iterations)
                .filter(|(&zp, &gi)| iterate_point(&p, zp).iteration != gi)
                .count();
            eprintln!("{ft:?} bord : {raw_wrong} pixels faux sans suivi, {wrong} avec");
            assert!(raw_wrong >= 5, "{ft:?} : scène sans pixel faux à corriger ({raw_wrong})");
            assert!(wrong <= 1, "{ft:?} : {wrong} pixels ≠ GMP pur (sans suivi : {raw_wrong})");
        }
    }

    /// Coût du traqueur seul (mono-fil, min de 7 passes alternées) :
    /// `cargo test --release --lib bench_tracked_loops -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_tracked_loops() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        for &(ft, view) in SCENES {
            let (w, h) = (128usize, 80usize);
            let mut p = scene_params(ft, view, w, h);
            p.iteration_max = 5000;
            let pts = scene_points(&p, w, h);
            let (mut tp, mut tt) = (f64::MAX, f64::MAX);
            let mut sink = 0u64;
            for _ in 0..7 {
                let t = std::time::Instant::now();
                for &zp in &pts {
                    sink += iterate_point(&p, zp).iteration as u64;
                }
                tp = tp.min(t.elapsed().as_secs_f64());
                let t = std::time::Instant::now();
                for &zp in &pts {
                    let (it, _, bad) = iterate_tracked(&p, zp);
                    sink += it as u64 + bad as u64;
                }
                tt = tt.min(t.elapsed().as_secs_f64());
            }
            // Étage double-double (voies entrelacées, un fil) sur les signalés.
            let flagged: Vec<Complex64> =
                pts.iter().copied().filter(|&zp| iterate_tracked(&p, zp).2).collect();
            let fp = dd_points(&flagged);
            let (mut td, mut dd_iters, mut beyond) = (f64::MAX, 0u64, 0usize);
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let r = pool.install(|| dd_stage(&p, &fp)).unwrap();
                td = td.min(t.elapsed().as_secs_f64());
                dd_iters = r.iter().map(|x| x.0 as u64).sum();
                beyond = r.iter().filter(|x| !x.2).count();
            }
            eprintln!(
                "{ft:?} {}: prod {:.4}s tracked {:.4}s (+{:.0} %) ; dd : {} signalés ({:.1} %), {} au-delà, {:.4}s, {:.1} ns/iter [{sink}]",
                if view.is_some() { "edge" } else { "home" },
                tp,
                tt,
                100.0 * (tt / tp - 1.0),
                flagged.len(),
                100.0 * flagged.len() as f64 / pts.len() as f64,
                beyond,
                td,
                1e9 * td / dd_iters.max(1) as f64
            );
        }
    }

    /// Les voies entrelacées rendent exactement ce que rend la boucle dd
    /// d'un pixel (compte, z, fiabilité), recharges comprises (nombre de
    /// points non multiple du nombre de voies, pixels courts et plafonnés).
    #[test]
    fn dd_lanes_match_single_pixel_loop() {
        let s0 = Cdd { re: Dd::ZERO, im: Dd::ZERO };
        for (ft, cx, cy, span) in [
            (FractalType::AlphaMandelbrot, -1.260833740234375, -0.350164794921875, 2e-3),
            (FractalType::Mandelbulb, -0.7341842651367188, -0.23336334228515626, 2e-3),
        ] {
            let pts: Vec<Cdd> = (0..103)
                .map(|k| {
                    let (i, j) = ((k % 11) as f64, (k / 11) as f64);
                    let re = Dd::from_f64(cx + span * (i / 10.0 - 0.5)).add(Dd::new(1e-20 * i, 0.0));
                    Cdd { re, im: Dd::from_f64(cy + span * (j / 9.0 - 0.5)) }
                })
                .collect();
            let lanes = if ft == FractalType::Mandelbulb {
                dd_lanes(&pts, s0, false, 2000, 4.0, mandelbulb_dd_step)
            } else {
                dd_lanes(&pts, s0, false, 2000, 4.0, alpha_dd_step)
            };
            let mut capped = 0;
            for (k, (&pc, &(it, z, ok))) in pts.iter().zip(&lanes).enumerate() {
                let (it1, z1, ok1) = if ft == FractalType::Mandelbulb {
                    mandelbulb_dd(s0, pc, 2000, 4.0)
                } else {
                    alpha_dd(s0, pc, 2000, 4.0)
                };
                assert_eq!((it, ok), (it1, ok1), "{ft:?} point {k}");
                assert_eq!(z.re.to_bits(), z1.re.to_bits(), "{ft:?} point {k}");
                assert_eq!(z.im.to_bits(), z1.im.to_bits(), "{ft:?} point {k}");
                capped += (it == 2000) as usize;
            }
            assert!(capped > 0 && capped < pts.len(), "{ft:?} : grille sans mélange ({capped})");
        }
        // Mandelbrot Sin (graine 1) : même verrou, pas transcendant compris.
        let trig = DdTrig::get();
        let step = |z: Cdd, e: f64, c: Cdd| sin_dd_step(trig, z, e, c, UDD * c.abs_hi());
        let seed = Cdd { re: Dd::from_f64(1.0), im: Dd::ZERO };
        let (cx, cy, span) = (-1.9899086401850268, -0.5337651978513115, 4e-3);
        let pts: Vec<Cdd> = (0..103)
            .map(|k| {
                let (i, j) = ((k % 11) as f64, (k / 11) as f64);
                let re = Dd::from_f64(cx + span * (i / 10.0 - 0.5)).add(Dd::new(1e-20 * i, 0.0));
                Cdd { re, im: Dd::from_f64(cy + span * (j / 9.0 - 0.5)) }
            })
            .collect();
        let lanes = dd_lanes(&pts, seed, false, 2000, 4.0, step);
        let mut capped = 0;
        for (k, (&pc, &(it, z, ok))) in pts.iter().zip(&lanes).enumerate() {
            let (it1, z1, ok1) = dd_pixel(seed, pc, 2000, 4.0, step);
            assert_eq!((it, ok), (it1, ok1), "Sin point {k}");
            assert_eq!(z.re.to_bits(), z1.re.to_bits(), "Sin point {k}");
            assert_eq!(z.im.to_bits(), z1.im.to_bits(), "Sin point {k}");
            capped += (it == 2000) as usize;
        }
        assert!(capped > 0 && capped < pts.len(), "Sin : grille sans mélange ({capped})");
    }

    /// Compte dd d'un point SANS borne (pas suivi, `E` remis à 0).
    fn dd_untracked(p: &FractalParams, zp: Complex64) -> u32 {
        let c = Cdd { re: Dd::from_f64(zp.re), im: Dd::from_f64(zp.im) };
        let seed = Cdd { re: Dd::from_f64(p.seed.re), im: Dd::from_f64(p.seed.im) };
        let zero = Cdd { re: Dd::ZERO, im: Dd::ZERO };
        let (n, r) = (p.iteration_max, p.bailout);
        let trig = DdTrig::get();
        let (z0, step): (Cdd, Box<dyn Fn(Cdd, f64, Cdd) -> (Cdd, f64, f64)>) = match p.fractal_type {
            FractalType::MandelbrotSin => (seed, Box::new(|z, _, c| sin_dd_step(trig, z, 0.0, c, 0.0))),
            FractalType::MagnetMandelbrot => (
                zero,
                Box::new(|z, _, c: Cdd| {
                    let s1 = Cdd { re: c.re.sub(Dd::from_f64(1.0)), im: c.im };
                    let s2 = Cdd { re: c.re.sub(Dd::from_f64(2.0)), im: c.im };
                    magnet_dd_step(z, 0.0, s1, s2, 0.0)
                }),
            ),
            FractalType::AlphaMandelbrot => (seed, Box::new(|z, _, c| alpha_dd_step(z, 0.0, c))),
            FractalType::Mandelbulb => (seed, Box::new(|z, _, c| mandelbulb_dd_step(z, 0.0, c))),
            _ => return u32::MAX,
        };
        let step = |z: Cdd, _e: f64, c: Cdd| {
            let (z, _, nn) = step(z, 0.0, c);
            (z, 0.0, nn)
        };
        dd_pixel(z0, c, n, r, step).0
    }

    #[test]
    #[ignore]
    fn diag_flag_breakdown() {
        for &(ft, view) in SCENES.iter().filter(|s| s.1.is_some() && s.0 != FractalType::BarnsleyMandelbrot) {
            let (w, h) = (128usize, 80usize);
            let mut p = scene_params(ft, view, w, h);
            p.iteration_max = 5000;
            let (mut fc, mut fe, mut nc, mut ne) = (0, 0, 0, 0);
            let mut hist = [0usize; 6];
            let mut agree = [[0usize; 2]; 2];
            let mut beyond = [0usize; 2];
            for j in 0..h {
                for i in 0..w {
                    let zp = Complex64::new(
                        p.center_x + ((i as f64 + 0.5) / w as f64 - 0.5) * p.span_x,
                        p.center_y + ((j as f64 + 0.5) / h as f64 - 0.5) * p.span_y,
                    );
                    let (it, _, bad) = iterate_tracked(&p, zp);
                    if bad {
                        let (dit, _, ok) = dd_stage(&p, &dd_points(&[zp])).unwrap()[0];
                        agree[(dit == it) as usize][ok as usize] += 1;
                        if !ok {
                            let mut gp = p.clone();
                            gp.engine.precision_bits = 256;
                            let g = crate::fractal::gmp::MpcParams::from_params(&gp);
                            let (git, _) = crate::fractal::gmp::iterate_point_mpc(
                                &g,
                                &crate::fractal::gmp::complex_from_xy(
                                    256,
                                    Float::with_val(256, zp.re),
                                    Float::with_val(256, zp.im),
                                ),
                            );
                            // dd mené à son terme SANS borne : le signalement
                            // dd était-il nécessaire ?
                            beyond[(git == dd_untracked(&p, zp)) as usize] += 1;
                        }
                    }
                    let capped = it >= p.iteration_max;
                    match (bad, capped) {
                        (true, true) => fc += 1,
                        (true, false) => {
                            fe += 1;
                            hist[(it as f64).log10().floor().clamp(0.0, 5.0) as usize] += 1
                        }
                        (false, true) => nc += 1,
                        (false, false) => ne += 1,
                    }
                }
            }
            eprintln!("{ft:?}: flagged capped={fc} escaped={fe} (log10 it hist {hist:?}) ; ok capped={nc} escaped={ne} ; flagged [diff,same]x[dd_bad,dd_ok] {agree:?} ; dd_bad vs gmp [diff,same] {beyond:?}");
        }
    }

    /// Julia Sin, vues d'accueil et de bord : pixels faux sans / avec suivi.
    #[test]
    #[ignore]
    fn diag_julia_sin_wrong() {
        for &(ft, view) in SCENES.iter().filter(|s| s.0 == FractalType::JuliaSin) {
            for (w, h) in [(64usize, 40usize), (128, 80)] {
                let mut p = scene_params(ft, view, w, h);
                p.iteration_max = 5000;
                let prod = crate::render::render_escape_time(&p);
                let mut g = p.clone();
                g.engine.algorithm_mode = AlgorithmMode::ReferenceGmp;
                let gmp = crate::render::render_escape_time(&g);
                let wrong = prod.iterations.iter().zip(&gmp.iterations).filter(|(a, b)| a != b).count();
                let raw_wrong = scene_points(&p, w, h)
                    .iter()
                    .zip(&gmp.iterations)
                    .filter(|(&zp, &gi)| iterate_point(&p, zp).iteration != gi)
                    .count();
                eprintln!("JuliaSin {view:?} {w}x{h} : {raw_wrong} faux sans suivi, {wrong} avec");
            }
        }
    }

    /// Sortie « pôle » de Magnet Julia (`|q|² < 1e-24`) : point f64 EXACT
    /// (vue 1×1) dont le 2e pas tombe en f64 à |q|² = 0,99956e-24 mais à
    /// 1,00003e-24 en exact. La boucle f64 s'arrêtait à 1 itération et se
    /// déclarait fiable (elle comparait la seule borne d'erreur au seuil) ;
    /// le GMP pur continue.
    #[test]
    fn magnet_pole_exit_is_judged_on_q_plus_error() {
        let z0 = Complex64::new(f64::from_bits(0x3fe426b473df3572), f64::from_bits(0x3fedd7afc17c751f));
        let mut p = default_params_for_type(FractalType::MagnetJulia, 1, 1);
        p.center_x = z0.re;
        p.center_y = z0.im;
        p.span_x = 1e-12;
        p.span_y = 1e-12;
        p.iteration_max = 200;
        let g = crate::fractal::gmp::MpcParams::from_params(&p);
        let (truth, _) = crate::fractal::gmp::iterate_point_mpc(
            &g,
            &crate::fractal::gmp::complex_from_xy(g.prec, Float::with_val(g.prec, z0.re), Float::with_val(g.prec, z0.im)),
        );
        let raw = iterate_point(&p, z0).iteration;
        assert_eq!(raw, 1, "scène : la boucle f64 doit sortir au pôle");
        assert_ne!(truth, raw, "scène : le point exact ne doit pas sortir au pôle");
        let (_, _, bad) = iterate_tracked(&p, z0);
        assert!(bad, "sortie au pôle non déterminée déclarée fiable");
        let prod = crate::render::render_escape_time(&p);
        assert_eq!(prod.iterations[0], truth, "rendu de production ≠ GMP pur");
    }

    /// Pixel de bord Mandelbulb ×6e9 que le double-double ne détermine pas :
    /// avec `precision_bits` = 64 la vue réclame 128 b, et le GMP à 128 b
    /// rendait 1667 itérations au lieu de 1492 (GMP 256 b ≡ 512 b).
    #[test]
    fn gmp_stage_runs_well_above_dd_precision() {
        let (w, h) = (24u32, 16u32);
        let mut p = default_params_for_type(FractalType::Mandelbulb, w, h);
        let (cx, cy) = ("-6.88218023928699463034588e-1", "-2.67570936992955699034588e-1");
        p.center_x_hp = Some(cx.into());
        p.center_y_hp = Some(cy.into());
        p.center_x = cx.parse().unwrap();
        p.center_y = cy.parse().unwrap();
        p.span_x = 6.738679960241853e-10;
        p.span_y = 6.738679960241853e-10;
        p.iteration_max = 20_000;
        p.bailout = 2.0;
        p.engine.precision_bits = 64;
        let n = (w * h) as usize;
        let (mut it, mut zs) = (vec![0u32; n], vec![Complex64::new(0.0, 0.0); n]);
        correct(&p, &AtomicBool::new(false), &[161], &mut it, &mut zs).unwrap();
        assert_eq!(it[161], 1492, "pixel au-delà du dd mal arbitré par le GMP");
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

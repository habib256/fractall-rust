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
//! 3. ce que le double-double ne garantit pas : **quad-double** tracé
//!    ([`scalar_qd`](crate::render::scalar_qd), 2026-11-20 — Sin et Magnet en
//!    voies vectorielles, [`scalar_lanes`](crate::render::scalar_lanes)),
//!    seulement si le GMP aurait tourné ([`gmp_would_run`]) ;
//! 4. le reste : GMP pur sous le budget du path standard
//!    (`std_reliability::correct_std_pixels`, `GmpCapped`).
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
//! Puis (2026-11-20) bornes f64 **LINÉAIRES** pour Magnet et Sin (`E` et
//! non `E²`, une à trois racines par pas hors de la chaîne de `E`) : la
//! forme au carré payait `(1 + δ)ᵏ` par pas — ×2·10⁶ sur `E` en 5000 pas
//! d'un cycle presque neutre — et, pour Magnet, minorait `|q*|²` par
//! `(1 − δ)|q̂|² − (1/δ)(2E + ρ_q)²`, qui exige `|q̂| > 45·(2E + ρ_q)` :
//! 91 % des signalements du bord Magnet Mandelbrot venaient de ce seul
//! dénominateur (`diag_magnet_confusion`, juge GMP 256 b ≡ 512 b). Bords
//! 128×80 : Magnet Mandelbrot 38,5 → 28,2 % de pixels signalés, Magnet
//! Julia 3,8 → 2,0 %, Mandelbrot Sin 29 → 24 % ; comptes de sortie
//! identiques, 0 pixel ≠ GMP. Ce qui reste signalé l'est par l'emballement
//! de la borne (E ≥ 0,25), dont 39 % sont réellement faux en f64. Essayé et
//! rejeté : une sonde « disque invariant » (borne propagée depuis un disque
//! B(c, ρ) ; si `B(ẑ_{k+m}, P) ⊂ B(c, ρ)` l'orbite ne sort jamais) dans les
//! étages dd et qd — rigoureuse, mais les pixels plafonnés au-delà du dd
//! sont des orbites bornées CHAOTIQUES (pas de cycle attractif : écart
//! d'une période à l'autre ~1e-2 au pas 5000) et ceux du dd sont déjà
//! raccourcis par le cycle exact ; gain mesuré 0-5 %.
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

use crate::fractal::bytecode::direct_dd::{dd_from_gmp, qd_from_gmp};
use crate::render::scalar_qd::{qd_stage, Cqd};
use crate::render::scalar_lanes::{self, LaneKind};
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

/// Le test de sortie AVANT le premier pas est-il indéterminé ? `n2` = `|ẑ₀|²`
/// tel que la boucle le compare à `R²` (arrondi ≤ 2u relatif), `e0` = erreur
/// du point de départ. Un pixel Julia à portée d'erreur du rayon sortait au
/// pas 0 « fiable » sans qu'aucune borne ne soit jamais lue (verrou
/// `julia_start_on_the_bailout_circle_is_not_declared_reliable`).
#[inline]
pub(crate) fn start_undetermined(n2: f64, e0: f64, bailout: f64) -> bool {
    let lo = (bailout - e0).max(0.0);
    let hi = bailout + e0;
    n2 >= lo * lo * (1.0 - 4.0 * U64) && n2 <= hi * hi * (1.0 + 4.0 * U64)
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
    // Point de départ (le pixel, à `e0` près) : sa sortie au pas 0 doit être
    // déterminée elle aussi (la boucle ne lit `bad` qu'après un pas).
    let mut bad = start_undetermined(z0.norm_sqr(), e0, bailout);
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
/// signalerait tout l'intérieur.
///
/// Borne LINÉAIRE `E` (2026-11-20), comme les étages dd/qd, et non au carré
/// comme Alpha / Mandelbulb : la version au carré payait `(1 + δ)⁵/(1 − δ)`
/// sur `E²` à chaque pas (×2·10⁶ sur `E` en 5000 pas d'un cycle presque
/// neutre) et minorait `|q*|²` par `(1 − δ)|q̂|² − (1/δ)(2E + ρ_q)²`, qui
/// exige `|q̂| > 45·(2E + ρ_q)` : 94 % des pixels signalés du bord Magnet
/// Mandelbrot avaient un compte f64 juste, signalés à 91 % par ce seul
/// dénominateur (`diag_magnet_confusion`). Trois racines par pas, hors de
/// la chaîne de `E` (elles ne dépendent que de `z`).
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
    let mut z = z0;
    let mut n2 = z.norm_sqr();
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<2>::new();
    let mut e = e0;
    // Point de départ (le pixel, à `e0` près) : sa sortie au pas 0 doit être
    // déterminée elle aussi (la boucle ne lit `bad` qu'après un pas).
    let mut bad = start_undetermined(z0.norm_sqr(), e0, bailout);
    while i < iteration_max && n2 < b2 {
        let n = z * z + seed_minus_one;
        let q = Complex64::new(2.0, 0.0) * z + seed_minus_two;
        let qq = q.norm_sqr();
        // Eviter division par zero
        if qq < 1e-24 {
            // Le vrai |q|² est-il aussi sous le seuil ? |Δq| ≤ d = 2E + ρ_q,
            // donc |q*| ≤ |q̂| + d — c'est CETTE somme qu'il faut comparer au
            // seuil, pas d seul : un |q̂| juste sous 1e-12 avec une borne
            // minuscule passait pour déterminé alors que le point exact était
            // au-dessus (verrou `magnet_pole_exit_is_judged_on_q_plus_error`).
            // Vaut pour les deux variantes (Magnet Mandelbrot : ρ_q porte
            // l'erreur du pixel). Marge relative : arrondis de la somme.
            let rho_q = U64 * (q.re.abs() + q.im.abs()) + s2;
            if (qq.sqrt() + 2.0 * e + rho_q) * (1.0 + 8.0 * U64) >= 1e-12 {
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
            // |q*| ≥ |q̂| − (2E + ρ_q) : minorant LINÉAIRE (cf. doc — la
            // version au carré exigeait |q̂| > 45·(2E + ρ_q)). Arrondis du
            // calcul : |q̂| minoré de 4u, la soustraction de 2u relatifs
            // (opérandes positifs, résultat positif quand il compte).
            let den = (qq.sqrt() * (1.0 - 4.0 * U64) - (2.0 * e + rho_q) * (1.0 + 4.0 * U64)) * (1.0 - 2.0 * U64);
            // Erreur de la division (normwise ≤ 6u) sur 2E + ρ_q, E < E_MAX_NL.
            let b = rho_n + r1 * rho_q + 4.0 * U64 * r1 * (2.0 * E_MAX_NL + rho_q);
            let (wr, wi) = (zp.re - ratio.re, zp.im - ratio.im);
            let w = (wr * wr + wi * wi).sqrt();
            let er_div = 4.0 * U64 * r1;
            // |Δr| ≤ (E(2|z − r| + E) + b) / |q*| + erreur de la division.
            let er = (e * (2.0 * w + e) + b) / den + er_div;
            // z' = r² : Δz' = Δr(2r + Δr) + arrondi du carré. Facteur final :
            // arrondis de ce calcul de borne (~20 opérations sur des
            // quantités positives, chacune ≤ u relatif ; racines ≤ u).
            e = (er * (2.0 * rr.sqrt() + er) + 2.0 * U64 * rr) * (1.0 + 64.0 * U64);
            if !(den > 1e-10) || sq_bound_unreliable(n2, b2, two_r_emax_sq, e * e) {
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
///
/// Borne LINÉAIRE `E` depuis le 2026-11-20 (une racine par pas, `|cos ẑ|`,
/// à côté d'un `sincos` et d'un `expm1`) : au carré, le découpage
/// `(x + y)² ≤ (1 + δ)x² + (1 + 1/δ)y²` coûtait `(1 + δ)²` par pas et
/// pesait 32× le terme en `E²` — bord Mandelbrot Sin 29 → 24 % de pixels
/// signalés, comptes inchangés (juge GMP pur).
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
    let kk = (k_abs + e_k) * (1.0 + 1e-12);
    let a_coef = (e_k + 10.0 * U64 * k1) * (1.0 + 1e-12);
    let kh = 0.5 * 1.29;
    let cos_margin = 1.0 + 16.0 * U64;
    let mut z = z0;
    let mut n2 = z.norm_sqr();
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<2>::new();
    let mut e = e0;
    // Point de départ (le pixel, à `e0` près) : sa sortie au pas 0 doit être
    // déterminée elle aussi (la boucle ne lit `bad` qu'après un pas).
    let mut bad = start_undetermined(z0.norm_sqr(), e0, bailout);
    while i < iteration_max && n2 < b2 {
        let (s, c, sh, ch) = csin_parts(z);
        z = k * Complex64::new(s * ch, c * sh);
        i += 1;
        n2 = z.norm_sqr();
        if !bad {
            // |Δ'| ≤ |k*|·E·(|cos ẑ| + E·1,29·ch/2) + a·|sin ẑ|₁ (borne
            // LINÉAIRE en E depuis 2026-11-20 : la version au carré payait
            // (1+δ)² par pas et 32× le terme en E²).
            let cos_abs = cos_margin * (c * c + sh * sh).sqrt();
            // Plancher (majorant, donc sûr) : vers le point fixe z = 0, |sin z|
            // et la borne décroissent géométriquement jusqu'aux SOUS-NORMAUX,
            // dont chaque opération coûte une assistance micro-code (+165 % sur
            // la vue d'accueil Mandelbrot Sin).
            let add = a_coef * (s.abs() * ch + c.abs() * sh.abs()).max(SIN_FLOOR);
            e = (kk * e * (cos_abs + kh * ch * e) + add) * (1.0 + 16.0 * U64);
            if sq_bound_unreliable(n2, b2, two_r_emax_sq, e * e) {
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
        if !crate::render::scalar_qd::step_ok(n, bailout, e) {
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

impl crate::render::scalar_qd::LaneZ<5> for Cdd {
    #[inline(always)]
    fn hi_c(self) -> Complex64 {
        self.hi()
    }
    #[inline(always)]
    fn abs_h(self) -> f64 {
        self.abs_hi()
    }
    #[inline(always)]
    fn state(self, e: f64) -> [f64; 5] {
        dd_cycle_state(self, e)
    }
}

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

/// Voies entrelacées dd ([`scalar_qd::lanes`](crate::render::scalar_qd::lanes)
/// générique, partagé avec l'étage qd).
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
    crate::render::scalar_qd::lanes::<Cdd, 5, DD_LANES>(points, seed, julia, UDD, iteration_max, bailout, step)
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
        // Sin : voies vectorielles (`scalar_lanes`, ×3 sur l'étage dd). PAS
        // Magnet : son pas dd est court (~30 opérations dd) et la borne par
        // voie (racines, division) domine — mesuré 26 → 46-52 ns/itération
        // en voies ; il reste sur les voies entrelacées scalaires.
        FractalType::MandelbrotSin | FractalType::JuliaSin if scalar_lanes::enabled() => {
            use crate::fractal::bytecode::direct_simd::DdV;
            let julia = params.fractal_type == FractalType::JuliaSin;
            Some(scalar_lanes::lanes_par::<DdV<8>, 8>(LaneKind::Sin, points, seed, julia, n, r))
        }
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

/// Coordonnée HP du centre (repli f64) à `prec` bits.
fn parse_at(prec: u32, hp: &Option<String>, f: f64) -> Float {
    match hp.as_deref().map(Float::parse) {
        Some(Ok(v)) => Float::with_val(prec, v),
        _ => Float::with_val(prec, f),
    }
}

/// La correction `GmpCapped` de `std_reliability::correct_std_pixels`
/// recalculerait-elle ces pixels ? (Plafond en pixels OU budget de 5 % des
/// itérations de la frame — mêmes règles, sur les comptes f64 courants.)
fn gmp_would_run(rest: &[usize], iterations: &[u32]) -> bool {
    let rest_iters: f64 = rest.iter().map(|&i| iterations[i] as f64).sum();
    let frame_iters: f64 = iterations.iter().map(|&n| n as f64).sum();
    rel::gmp_escalation_allowed(rest.len(), iterations.len(), rel::reliability_gmp_max_fraction())
        || rest_iters <= STD_GMP_ITER_BUDGET * frame_iters
}

/// Copie de `std_reliability::STD_GMP_ITER_BUDGET` (privée là-bas) : à
/// garder égale.
const STD_GMP_ITER_BUDGET: f64 = 0.05;

/// `FRACTALL_SCALAR_QD=0` : pas d'étage quad-double (pixels au-delà du dd
/// droit au GMP pur — A/B). Lu une fois.
fn scalar_qd_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FRACTALL_SCALAR_QD").map_or(true, |v| v != "0"))
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
    let cx = parse_at(prec, &params.center_x_hp, params.center_x);
    let cy = parse_at(prec, &params.center_y_hp, params.center_y);
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
    let t_dd = t.elapsed().as_secs_f64();
    let beyond_dd = rest.len();
    // Étage quad-double : ce que le dd ne détermine pas (cf. `scalar_qd`).
    // Point exact recalculé à la précision de l'étage GMP qui suit. Il ne
    // remplace QUE du GMP : il ne tourne que si le GMP aurait tourné (même
    // plafond que `correct_std_pixels`, `GmpCapped`). Bord Barnsley
    // Mandelbrot (CHAOS, 97 % de pixels au-delà du dd, GMP hors budget) :
    // sans ce garde le qd coûtait 0,8 → 2,5 s pour des pixels qui restent
    // ensuite en f64 et ne changeait rien de garanti.
    if !rest.is_empty() && scalar_qd_enabled() && gmp_would_run(&rest, iterations) {
        let qprec = prec.max(GMP_FLOOR_BITS);
        let qcx = parse_at(qprec, &params.center_x_hp, params.center_x);
        let qcy = parse_at(qprec, &params.center_y_hp, params.center_y);
        let qctx = DcGmpContext::new(params, qprec);
        let qpoints: Vec<Cqd> = rest
            .par_iter()
            .map(|&idx| {
                let dc = qctx.compute_dc(idx % width, idx / width);
                let mut re = qcx.clone();
                re += dc.real();
                let mut im = qcy.clone();
                im += dc.imag();
                Cqd { re: qd_from_gmp(&re), im: qd_from_gmp(&im) }
            })
            .collect();
        if let Some(res) = qd_stage(params, &qpoints) {
            let mut still = Vec::new();
            for (&idx, (it, z, ok)) in rest.iter().zip(res) {
                if ok {
                    iterations[idx] = it;
                    zs[idx] = z;
                } else {
                    still.push(idx);
                }
            }
            rest = still;
        }
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
    }
    if perf_enabled() {
        eprintln!(
            "[RELIABILITY-SCALAR] flagged={} ({:.3} %) dd_fixed={} beyond_dd={} qd_fixed={} beyond_qd={} dd={:.3}s qd={:.3}s",
            flagged.len(),
            100.0 * flagged.len() as f64 / iterations.len().max(1) as f64,
            flagged.len() - beyond_dd,
            beyond_dd,
            beyond_dd - rest.len(),
            rest.len(),
            t_dd,
            t.elapsed().as_secs_f64() - t_dd
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

    /// Étage quad-double : sur les pixels de bord que le double-double ne
    /// détermine pas, tout pixel que le qd déclare fiable a le compte du GMP
    /// pur 256 b (juge convergé de l'axe `types`), et le qd en règle la
    /// majorité (sinon l'étage ne sert à rien).
    #[test]
    fn qd_stage_agrees_with_gmp_beyond_dd() {
        let judged = |ft: FractalType| !matches!(ft, FractalType::JuliaSin | FractalType::BarnsleyMandelbrot);
        for &(ft, view) in SCENES.iter().filter(|s| s.1.is_some() && judged(s.0)) {
            let (w, h) = (96usize, 60usize);
            let mut p = scene_params(ft, view, w, h);
            p.iteration_max = 5000;
            let flagged: Vec<Complex64> =
                scene_points(&p, w, h).into_iter().filter(|&zp| iterate_tracked(&p, zp).2).collect();
            let dd = dd_stage(&p, &dd_points(&flagged)).unwrap();
            let beyond: Vec<Complex64> =
                flagged.iter().zip(&dd).filter(|(_, r)| !r.2).map(|(&zp, _)| zp).collect();
            let qpts: Vec<Cqd> = beyond
                .iter()
                .map(|z| Cqd {
                    re: crate::fractal::perturbation::qd::Qd::from_f64(z.re),
                    im: crate::fractal::perturbation::qd::Qd::from_f64(z.im),
                })
                .collect();
            // Phoenix n'a pas d'étage qd (dd ellipsoïdal puis GMP).
            let Some(qd) = qd_stage(&p, &qpts) else { continue };
            let mut gp = p.clone();
            gp.engine.precision_bits = 256;
            let g = crate::fractal::gmp::MpcParams::from_params(&gp);
            let (mut ok, mut wrong) = (0, 0);
            for (&zp, &(it, _, rel)) in beyond.iter().zip(&qd) {
                if !rel {
                    continue;
                }
                ok += 1;
                let (git, _) = crate::fractal::gmp::iterate_point_mpc(
                    &g,
                    &crate::fractal::gmp::complex_from_xy(256, Float::with_val(256, zp.re), Float::with_val(256, zp.im)),
                );
                wrong += (git != it) as usize;
            }
            eprintln!("{ft:?} : {} au-delà du dd, {ok} réglés par le qd, {wrong} ≠ GMP", beyond.len());
            assert_eq!(wrong, 0, "{ft:?} : pixels qd « fiables » ≠ GMP");
            assert!(2 * ok >= beyond.len(), "{ft:?} : le qd n'en règle que {ok}/{}", beyond.len());
        }
    }

    /// Coût des étages dd et qd (mono-fil, min de 5) sur les bords :
    /// `cargo test --release --lib bench_correction_stages -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_correction_stages() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let only = std::env::var("BENCH_TYPES").ok();
        for &(ft, view) in SCENES.iter().filter(|s| s.1.is_some()) {
            if only.as_deref().is_some_and(|o| !o.split(',').any(|t| format!("{ft:?}") == t)) {
                continue;
            }
            let (w, h) = (128usize, 80usize);
            let mut p = scene_params(ft, view, w, h);
            p.iteration_max = 5000;
            let flagged: Vec<Complex64> =
                scene_points(&p, w, h).into_iter().filter(|&zp| iterate_tracked(&p, zp).2).collect();
            let fp = dd_points(&flagged);
            let (mut td, mut tq) = (f64::MAX, f64::MAX);
            let (mut dd_it, mut qd_it, mut nb, mut nq) = (0u64, 0u64, 0usize, 0usize);
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let r = pool.install(|| dd_stage(&p, &fp)).unwrap();
                td = td.min(t.elapsed().as_secs_f64());
                dd_it = r.iter().map(|x| x.0 as u64).sum();
                let beyond: Vec<Cqd> = flagged
                    .iter()
                    .zip(&r)
                    .filter(|(_, x)| !x.2)
                    .map(|(z, _)| Cqd {
                        re: crate::fractal::perturbation::qd::Qd::from_f64(z.re),
                        im: crate::fractal::perturbation::qd::Qd::from_f64(z.im),
                    })
                    .collect();
                nb = beyond.len();
                let t = std::time::Instant::now();
                let Some(q) = pool.install(|| qd_stage(&p, &beyond)) else { break };
                tq = tq.min(t.elapsed().as_secs_f64());
                qd_it = q.iter().map(|x| x.0 as u64).sum();
                nq = q.iter().filter(|x| x.2).count();
            }
            eprintln!(
                "{ft:?}: dd {} px {:.3}s {:.1} ns/it ; qd {nb} px ({nq} réglés) {:.3}s {:.1} ns/it",
                flagged.len(),
                td,
                1e9 * td / dd_it.max(1) as f64,
                tq,
                1e9 * tq / qd_it.max(1) as f64
            );
        }
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

    /// Matrice de confusion du traqueur Magnet (bords types 11 et 12, 128×80,
    /// 5000 itérations) : signalé × réellement faux (f64 brut contre GMP pur
    /// 256 b, convergence vérifiée à 512 b), par classe de sortie (évadé /
    /// plafonné / pôle) et par motif du signalement (borne emballée, évasion
    /// indéterminée, dénominateur). `MAGNET_DIAG_W=64` pour 64×40.
    /// `cargo test --release --lib diag_magnet_confusion -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn diag_magnet_confusion() {
        let w: usize = std::env::var("MAGNET_DIAG_W").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let h = w * 5 / 8;
        for &(ft, view) in SCENES
            .iter()
            .filter(|s| matches!(s.0, FractalType::MagnetJulia | FractalType::MagnetMandelbrot))
        {
            let mut p = scene_params(ft, view, w, h);
            p.iteration_max = 5000;
            let pts = scene_points(&p, w, h);
            let gmp_at = |prec: u32, zp: Complex64| {
                let mut gp = p.clone();
                gp.engine.precision_bits = prec;
                let g = crate::fractal::gmp::MpcParams::from_params(&gp);
                crate::fractal::gmp::iterate_point_mpc(
                    &g,
                    &crate::fractal::gmp::complex_from_xy(prec, Float::with_val(prec, zp.re), Float::with_val(prec, zp.im)),
                )
                .0
            };
            // (classe, motif) -> [signalé&juste, signalé&faux, chaos]
            let rows: Vec<(u32, bool, u8, u32, f64, bool, bool)> = pts
                .par_iter()
                .map(|&zp| {
                    let (it, z, bad) = iterate_tracked(&p, zp);
                    let (_, _, _, fstep, why) = magnet_f64_diag(&p, zp);
                    let g256 = gmp_at(256, zp);
                    let chaos = gmp_at(512, zp) != g256;
                    let class = if it >= p.iteration_max {
                        0u8
                    } else if z.norm_sqr() >= p.bailout * p.bailout {
                        1
                    } else {
                        2
                    };
                    (it, bad, class, fstep, why, it != g256, chaos)
                })
                .collect();
            let names = ["plafonné", "évadé", "pôle"];
            let mut tab = std::collections::BTreeMap::<(u8, i64), [usize; 4]>::new();
            let (mut flagged, mut wrong_unflagged, mut wrong) = (0, 0, 0);
            let mut steps: Vec<u32> = Vec::new();
            for &(_, bad, class, fstep, why, wr, chaos) in &rows {
                wrong += wr as usize;
                if bad {
                    flagged += 1;
                    steps.push(fstep);
                    let e = tab.entry((class, why as i64)).or_default();
                    e[wr as usize] += 1;
                    e[2] += chaos as usize;
                } else if wr {
                    wrong_unflagged += 1;
                }
                let e = tab.entry((class, -1)).or_default();
                e[3] += 1;
            }
            steps.sort_unstable();
            let q = |f: f64| steps.get(((steps.len() as f64 - 1.0) * f) as usize).copied().unwrap_or(0);
            eprintln!(
                "{ft:?} {} {w}x{h} : {flagged} signalés ({:.1} %), {wrong} faux en f64 brut, {wrong_unflagged} faux NON signalés ; pas du signalement p10/p50/p90 = {}/{}/{}",
                if view.is_some() { "edge" } else { "home" },
                100.0 * flagged as f64 / rows.len() as f64,
                q(0.1),
                q(0.5),
                q(0.9)
            );
            for ((class, why), v) in &tab {
                if *why < 0 {
                    eprintln!("   {:9} total {}", names[*class as usize], v[3]);
                } else {
                    let motif = ["emballée", "évasion indét.", "dénominateur", "pôle"][*why as usize];
                    eprintln!("   {:9} {:15} juste {:5} faux {:5} (chaos {})", names[*class as usize], motif, v[0], v[1], v[2]);
                }
            }
        }
    }

    /// Banc + juge des scènes Magnet (types 11/12, accueil et bord) : rendu
    /// de production à 256×160 (min de 3), empreinte des comptes à 128×80 et
    /// 256×160, pixels ≠ GMP pur à 64×40 et 128×80. Écrit les empreintes dans
    /// `$MAGNET_BENCH_OUT` pour comparer deux versions.
    /// `cargo test --release --lib bench_magnet_scenes -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_magnet_scenes() {
        use std::hash::{Hash, Hasher};
        let skip_gmp = std::env::var("MAGNET_SKIP_GMP").is_ok();
        let mut out = String::new();
        for &(ft, view) in SCENES
            .iter()
            .filter(|s| match std::env::var("MAGNET_BENCH_TYPES") {
                Ok(t) => t.split(',').any(|x| format!("{:?}", s.0) == x),
                Err(_) => matches!(s.0, FractalType::MagnetJulia | FractalType::MagnetMandelbrot),
            })
        {
            let tag = format!("{ft:?} {}", if view.is_some() { "edge" } else { "home" });
            for (w, h) in [(64usize, 40usize), (128, 80), (256, 160)] {
                let mut p = scene_params(ft, view, w, h);
                p.iteration_max = 5000;
                let mut best = f64::MAX;
                let mut prod = None;
                let runs = if w == 256 { 3 } else { 1 };
                for _ in 0..runs {
                    let t = std::time::Instant::now();
                    let r = crate::render::render_escape_time(&p);
                    best = best.min(t.elapsed().as_secs_f64());
                    prod = Some(r);
                }
                let prod = prod.unwrap();
                let mut hs = std::collections::hash_map::DefaultHasher::new();
                prod.iterations.hash(&mut hs);
                let flagged = scene_points(&p, w, h).iter().filter(|&&zp| iterate_tracked(&p, zp).2).count();
                let mut line = format!(
                    "{tag} {w}x{h}: {best:.3}s signalés {flagged} ({:.1} %) empreinte {:016x}",
                    100.0 * flagged as f64 / (w * h) as f64,
                    hs.finish()
                );
                if w < 256 && !skip_gmp {
                    let mut g = p.clone();
                    g.engine.algorithm_mode = AlgorithmMode::ReferenceGmp;
                    g.engine.use_gmp = true;
                    let gmp = crate::render::render_escape_time(&g);
                    let wrong = prod.iterations.iter().zip(&gmp.iterations).filter(|(a, b)| a != b).count();
                    line += &format!(" ≠GMP {wrong}");
                }
                eprintln!("{line}");
                out += &line;
                out.push('\n');
            }
        }
        if let Ok(f) = std::env::var("MAGNET_BENCH_OUT") {
            std::fs::write(f, out).unwrap();
        }
    }

    /// Pixels du bord Magnet que l'étage qd mène au plafond : à quel pas une
    /// sonde « disque invariant » (rayons 1e-3..1e-12) les certifierait.
    /// `cargo test --release --lib diag_magnet_qd_probe -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn diag_magnet_qd_probe() {
        use crate::fractal::perturbation::qd::Qd;
        for &(ft, view) in SCENES
            .iter()
            .filter(|s| matches!(s.0, FractalType::MagnetJulia | FractalType::MagnetMandelbrot) && s.1.is_some())
        {
            let (w, h) = (128usize, 80usize);
            let mut p = scene_params(ft, view, w, h);
            p.iteration_max = 5000;
            let flagged: Vec<Complex64> =
                scene_points(&p, w, h).into_iter().filter(|&zp| iterate_tracked(&p, zp).2).collect();
            let dd = dd_stage(&p, &dd_points(&flagged)).unwrap();
            let dd_capped = dd.iter().filter(|r| r.2 && r.0 >= p.iteration_max).count();
            let beyond: Vec<Complex64> = if std::env::var("PROBE_SET").as_deref() == Ok("dd") {
                flagged.iter().zip(&dd).filter(|(_, r)| r.2 && r.0 >= p.iteration_max).map(|(&zp, _)| zp).collect()
            } else {
                flagged.iter().zip(&dd).filter(|(_, r)| !r.2).map(|(&zp, _)| zp).collect()
            };
            let julia = ft == FractalType::MagnetJulia;
            let radii = [1e-2, 1e-3, 1e-4, 1e-6, 1e-9];
            let period: u32 = std::env::var("PROBE_K").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
            let rows: Vec<(u32, bool, Option<(u32, usize)>)> = beyond
                .par_iter()
                .map(|&zp| {
                    let pt = Cqd { re: Qd::from_f64(zp.re), im: Qd::from_f64(zp.im) };
                    let seed = Cqd { re: Qd::from_f64(p.seed.re), im: Qd::from_f64(p.seed.im) };
                    let (s, e_s) = if julia { (seed, 0.0) } else { (pt, crate::render::scalar_qd::UQ * pt.abs_hi()) };
                    let one = Qd::from_f64(1.0);
                    let s1 = Cqd { re: s.re.sub(one), im: s.im };
                    let s2 = Cqd { re: s.re.sub(Qd::from_f64(2.0)), im: s.im };
                    let mut z = if julia { pt } else { Cqd::ZERO };
                    let mut e = if julia { crate::render::scalar_qd::UQ * pt.abs_hi() } else { 0.0 };
                    let mut i = 0u32;
                    // sondes : (centre, rayon, P)
                    let mut probes: Vec<(Complex64, f64, f64, usize)> = Vec::new();
                    let mut next_ck = period;
                    let mut cert = None;
                    while i < p.iteration_max && z.abs_hi() < p.bailout {
                        let (nz, ne, nn) = crate::render::scalar_qd::magnet_qd_step(z, e, s1, s2, e_s);
                        let mut np = Vec::new();
                        for &(c0, rho, pb, ri) in &probes {
                            let (_, pp, _) = crate::render::scalar_qd::magnet_qd_step(z, pb, s1, s2, e_s);
                            if crate::render::scalar_qd::step_ok(nn, p.bailout, pp) {
                                np.push((c0, rho, pp, ri));
                            }
                        }
                        z = nz;
                        e = ne;
                        i += 1;
                        if !crate::render::scalar_qd::step_ok(nn, p.bailout, e) {
                            break;
                        }
                        probes = np;
                        for &(c0, rho, pb, ri) in &probes {
                            let d = (z.hi() - c0).norm() * (1.0 + 1e-12) + 1e-300;
                            if d + pb * (1.0 + 1e-12) <= rho {
                                cert = Some((i, ri));
                                break;
                            }
                        }
                        if cert.is_some() {
                            break;
                        }
                        if i == next_ck {
                            next_ck += period;
                            probes.clear();
                            let c0 = z.hi();
                            for (ri, &rho) in radii.iter().enumerate() {
                                if e < rho * 0.5 {
                                    probes.push((c0, rho, rho * (1.0 + 1e-12) + 1e-30, ri));
                                }
                            }
                        }
                    }
                    (i, e.is_finite(), cert)
                })
                .collect();
            let capped = rows.iter().filter(|r| r.0 >= p.iteration_max || r.2.is_some()).count();
            let certs: Vec<_> = rows.iter().filter_map(|r| r.2).collect();
            let mut by_r = [0usize; 5];
            for c in &certs {
                by_r[c.1] += 1;
            }
            let mut st: Vec<u32> = certs.iter().map(|c| c.0).collect();
            st.sort_unstable();
            // Pixels qd plafonnés : période, multiplicateur, distance au cycle.
            if std::env::var("PROBE_ORBITS").is_ok() {
                for (&zp, r) in beyond.iter().zip(&rows).take(400) {
                    if r.0 < p.iteration_max {
                        continue;
                    }
                    let pt = Cqd { re: Qd::from_f64(zp.re), im: Qd::from_f64(zp.im) };
                    let one = Qd::from_f64(1.0);
                    let s1 = Cqd { re: pt.re.sub(one), im: pt.im };
                    let s2 = Cqd { re: pt.re.sub(Qd::from_f64(2.0)), im: pt.im };
                    let mut z = if julia { pt } else { Cqd::ZERO };
                    let mut hist = Vec::new();
                    let mut first_near1 = None;
                    for i in 0..p.iteration_max {
                        let (nz, _, _) = crate::render::scalar_qd::magnet_qd_step(z, 0.0, s1, s2, 0.0);
                        z = nz;
                        let h = z.hi();
                        if first_near1.is_none() && (h - Complex64::new(1.0, 0.0)).norm() < 1e-3 {
                            first_near1 = Some(i);
                        }
                        hist.push(h);
                    }
                    let n = hist.len() - 1;
                    let per = (1..200).min_by(|&a, &b| {
                        (hist[n] - hist[n - a]).norm().partial_cmp(&(hist[n] - hist[n - b]).norm()).unwrap()
                    });
                    let per = per.unwrap();
                    let d0 = (hist[n] - hist[n - per]).norm();
                    let d1 = (hist[n - per] - hist[n - 2 * per]).norm();
                    eprintln!(
                        "   plafonné : z_fin {:.4} {:+.4} période {per} d0 {d0:.2e} λ≈{:.5} près de 1 au pas {first_near1:?}",
                        hist[n].re,
                        hist[n].im,
                        d0 / d1
                    );
                }
            }
            let saved: u64 = certs.iter().map(|c| (p.iteration_max - c.0) as u64).sum();
            let total: u64 = rows.iter().map(|r| r.0 as u64).sum::<u64>() + saved;
            eprintln!("   pas qd économisés {saved} / {total}");
            eprintln!(
                "{ft:?}: {} signalés, dd plafonnés {dd_capped}, {} au-delà du dd, {capped} plafonnés/certifiés en qd, {} certifiés (par rayon {by_r:?}), pas médian {}",
                flagged.len(),
                beyond.len(),
                certs.len(),
                st.get(st.len() / 2).copied().unwrap_or(0)
            );
        }
    }

    /// Copie instrumentée de [`magnet_f64`] : `(it, z, bad, pas du
    /// signalement, motif)` — motif 0 = borne emballée, 1 = évasion
    /// indéterminée, 2 = dénominateur, 3 = sortie au pôle.
    fn magnet_f64_diag(p: &FractalParams, zp: Complex64) -> (u32, Complex64, bool, u32, f64) {
        let (z0, seed, e0, e_s) = if p.fractal_type == FractalType::MagnetJulia {
            (zp, p.seed, pixel_error(p, zp), 0.0)
        } else {
            (Complex64::new(0.0, 0.0), zp, 0.0, pixel_error(p, zp))
        };
        let (iteration_max, bailout) = (p.iteration_max, p.bailout);
        let b2 = bailout * bailout;
        let two_r_emax_sq = (2.0 * bailout + E_MAX_NL) * (2.0 * bailout + E_MAX_NL);
        let seed_minus_one = Complex64::new(seed.re - 1.0, seed.im);
        let seed_minus_two = Complex64::new(seed.re - 2.0, seed.im);
        let s1 = U64 * (seed_minus_one.re.abs() + seed_minus_one.im.abs()) + e_s;
        let s2 = U64 * (seed_minus_two.re.abs() + seed_minus_two.im.abs()) + e_s;
        let mut z = z0;
        let mut n2 = z.norm_sqr();
        let mut i = 0u32;
        let mut cyc = crate::fractal::cycle::CycleCut::<2>::new();
        let mut e = e0;
        let (mut bad, mut fstep, mut why) = (false, 0u32, 0.0);
        while i < iteration_max && n2 < b2 {
            let n = z * z + seed_minus_one;
            let q = Complex64::new(2.0, 0.0) * z + seed_minus_two;
            let qq = q.norm_sqr();
            if qq < 1e-24 {
                let rho_q = U64 * (q.re.abs() + q.im.abs()) + s2;
                if (qq.sqrt() + 2.0 * e + rho_q) * (1.0 + 8.0 * U64) >= 1e-12 && !bad {
                    bad = true;
                    fstep = i;
                    why = 3.0;
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
                let rho_n = 2.0 * U64 * n2p + U64 * (n.re.abs() + n.im.abs()) + s1;
                let rho_q = U64 * (q.re.abs() + q.im.abs()) + s2;
                let den = (qq.sqrt() * (1.0 - 4.0 * U64) - (2.0 * e + rho_q) * (1.0 + 4.0 * U64)) * (1.0 - 2.0 * U64);
                let b = rho_n + r1 * rho_q + 4.0 * U64 * r1 * (2.0 * E_MAX_NL + rho_q);
                let (wr, wi) = (zp.re - ratio.re, zp.im - ratio.im);
                let w = (wr * wr + wi * wi).sqrt();
                let er_div = 4.0 * U64 * r1;
                let er = (e * (2.0 * w + e) + b) / den + er_div;
                e = (er * (2.0 * rr.sqrt() + er) + 2.0 * U64 * rr) * (1.0 + 64.0 * U64);
                let (d_bad, runaway) = (!(den > 1e-10), !(e < E_MAX_NL));
                if d_bad || sq_bound_unreliable(n2, b2, two_r_emax_sq, e * e) {
                    bad = true;
                    fstep = i;
                    why = if d_bad { 2.0 } else if runaway { 0.0 } else { 1.0 };
                }
            }
            cyc.observe_each(&mut i, iteration_max, [z.re, z.im]);
        }
        (i, z, bad, fstep, why)
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

    /// Un pixel dont le point exact est JUSTE sous le rayon d'évasion (à
    /// 2⁻⁷⁰ relatif) mais dont les parties hautes f64 donnent une norme
    /// arrondie au-dessus : l'étage dd (et qd) jugeait l'évasion sur cette
    /// norme f64 avec une marge `e` (~2⁻¹⁰³·R) qui ignore son arrondi
    /// (~2⁻⁵²·R) — évasion déclarée FIABLE au mauvais pas.
    #[test]
    fn dd_stage_escape_test_covers_f64_norm_rounding() {
        const P: u32 = 400;
        let r = 3.7f64;
        let mut found = None;
        for k in 1..400_000u32 {
            let th = k as f64 * 0.000_731;
            let mut cr = Float::with_val(P, th).cos();
            let mut ci = Float::with_val(P, th).sin();
            let shrink = Float::with_val(P, 1.0) - Float::with_val(P, Float::i_exp(1, -70));
            cr *= r;
            cr *= &shrink;
            ci *= r;
            ci *= &shrink;
            let (h, v) = (cr.to_f64(), ci.to_f64());
            let n2 = h * h + v * v;
            if n2 >= r * r && n2.sqrt() > r {
                found = Some((cr, ci));
                break;
            }
        }
        let (cr, ci) = found.expect("aucun point trouvé");
        // Anti-vacuité : l'ancien test de sortie (`|n − R| > e` sans l'arrondi
        // de la norme f64) acceptait ce pas — la borne dd y vaut ~2⁻¹⁰³·R.
        let (h, v) = (cr.to_f64(), ci.to_f64());
        let n = (h * h + v * v).sqrt();
        let e_dd = UDD * (r + r + n);
        assert!((n - r).abs() > e_dd, "scène : l'ancien test aurait refusé ce pas");
        let mut p = default_params_for_type(FractalType::Mandelbulb, 1, 1);
        p.center_x_hp = Some(cr.to_string_radix(10, None));
        p.center_y_hp = Some(ci.to_string_radix(10, None));
        p.center_x = cr.to_f64();
        p.center_y = ci.to_f64();
        p.span_x = 1e-10;
        p.span_y = 1e-10;
        p.bailout = r;
        p.iteration_max = 50;
        // Vérité : |c| < R, donc pas d'évasion au premier pas.
        let pts = [(dd_from_gmp(&cr), dd_from_gmp(&ci))];
        let res = dd_stage(&p, &pts).unwrap();
        let mut g = p.clone();
        g.engine.precision_bits = P;
        let gp = crate::fractal::gmp::MpcParams::from_params(&g);
        let (truth, _) = crate::fractal::gmp::iterate_point_mpc(
            &gp,
            &crate::fractal::gmp::complex_from_xy(P, cr.clone(), ci.clone()),
        );
        assert!(truth >= 2, "scène : le point exact ne doit pas s'évader au premier pas ({truth})");
        assert!(!(res[0].2 && res[0].0 != truth), "dd : évasion {} déclarée fiable, vérité {truth}", res[0].0);
        let q = [Cqd { re: qd_from_gmp(&cr), im: qd_from_gmp(&ci) }];
        let rq = qd_stage(&p, &q).unwrap();
        assert!(!(rq[0].2 && rq[0].0 != truth), "qd : évasion {} déclarée fiable, vérité {truth}", rq[0].0);
        let prod = crate::render::render_escape_time(&p);
        assert_eq!(prod.iterations[0], truth, "rendu de production ≠ GMP pur");
    }

    /// Pixel Julia Sin dont le point exact est JUSTE sous le rayon (2⁻⁷⁰
    /// relatif) mais dont la norme f64 arrondit au rayon : la boucle f64 sort
    /// au pas 0, et la boucle tracée — qui ne juge la sortie qu'APRÈS un pas —
    /// la déclarait fiable (0 itération, vérité ≥ 1).
    #[test]
    fn julia_start_on_the_bailout_circle_is_not_declared_reliable() {
        const P: u32 = 400;
        let mut p = default_params_for_type(FractalType::JuliaSin, 1, 1);
        let r = p.bailout;
        let mut found = None;
        for k in 1..400_000u32 {
            let th = Float::with_val(P, k) * 0.000_731f64;
            let shrink = Float::with_val(P, 1.0) - Float::with_val(P, Float::i_exp(1, -70));
            let mut zr = Float::with_val(P, th.cos_ref());
            let mut zi = Float::with_val(P, th.sin_ref());
            zr *= r;
            zr *= &shrink;
            zi *= r;
            zi *= &shrink;
            let (h, v) = (zr.to_f64(), zi.to_f64());
            if h * h + v * v >= r * r {
                found = Some((zr, zi));
                break;
            }
        }
        let (zr, zi) = found.expect("aucun point trouvé");
        p.center_x_hp = Some(zr.to_string_radix(10, None));
        p.center_y_hp = Some(zi.to_string_radix(10, None));
        p.center_x = zr.to_f64();
        p.center_y = zi.to_f64();
        p.span_x = 1e-10;
        p.span_y = 1e-10;
        p.iteration_max = 50;
        let zp = Complex64::new(p.center_x, p.center_y);
        assert_eq!(iterate_point(&p, zp).iteration, 0, "scène : la boucle f64 doit sortir au pas 0");
        let mut g = p.clone();
        g.engine.precision_bits = P;
        let gp = crate::fractal::gmp::MpcParams::from_params(&g);
        let (truth, _) = crate::fractal::gmp::iterate_point_mpc(
            &gp,
            &crate::fractal::gmp::complex_from_xy(P, zr.clone(), zi.clone()),
        );
        assert!(truth >= 1, "scène : le point exact ne doit pas sortir au pas 0");
        let (_, _, bad) = iterate_tracked(&p, zp);
        assert!(bad, "sortie au pas 0 indéterminée déclarée fiable");
        let prod = crate::render::render_escape_time(&p);
        assert_eq!(prod.iterations[0], truth, "rendu de production ≠ GMP pur");
    }
}

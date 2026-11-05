//! Détecteur de fiabilité par pixel (G9.6) — borne d'erreur propagée +
//! test de shadowing, d'après « Reliable Mandelbrot » (Claude Heiland-Allen,
//! <https://mathr.co.uk/web/m-reliable.html>).
//!
//! **Problème.** La boucle perturbation f64 rend parfois un compte d'itération
//! FAUX sans rien signaler : plancher de précision du δ sur les scènes à forte
//! sensibilité (cancellation au rebase près de |Z|≈0, amplification de
//! Lyapunov). Cas mesurés : `mandelbrot-e13` (2 px à +210 iters à 256²), presets
//! dd seahorse/misiurewicz/minibrot en f64. Le tier dd les corrige, mais il est
//! opt-in : aucun proxy par frame ne sépare les pixels faux des justes (proxy
//! `cbits` réfuté, cf. TODO G3).
//!
//! **Idée.** Propager une borne d'erreur absolue `E` sur `z = Z + δ`
//! (`u = 2⁻⁵³`) sur les trois événements de la boucle :
//! - pas direct `δ' = 2Zδ + δ² + dc` : `E ← E·(2|z| + E) + γ·(|2Zδ| + |δ|² + |dc|)` ;
//! - saut BLA `δ' = Aδ + B·dc` : `E ← σ₁(A)·E + (ε_bla + γ)·(|Aδ| + |B·dc|)` ;
//! - rebase `δ ← Z + δ` : `E ← E + u·(|Z| + |z|)` — l'arrondi f64 de la
//!   référence, inoffensif tant que δ est relatif à Z, devient une erreur
//!   ABSOLUE de l'état.
//!
//! **Critère : l'EMBALLEMENT de la borne** (`E ≥ 1e30` ou non finie). Tant
//! que `E ≪ |z|`, la borne croît comme l'orbite elle-même ; dès qu'elle
//! dépasse |z|, le terme `E²` la fait doubler d'exposant à chaque pas : le
//! compte d'itération au centre exact n'est plus déterminé par
//! l'arithmétique f64. Historique de calibration (2026-10-02, étude vs GMP
//! pur `quality::reliability_study`) :
//! - critère de shadowing d'origine `E/|dz/dc| > κ·pixel` : distribution
//!   bimodale (≈1e-11 pixel ou ∞), et TOUS les pixels faux à ∞ (seahorse
//!   33/33, e30 5/5, e50 17/18) — la dérivée ne départageait qu'une poignée
//!   de pixels à ratio fini, aucun faux ;
//! - critère z-space à seuil MODÉRÉ (`E ≥ τ`, τ ∈ [1e-3, 1e2]) : PAS bimodal,
//!   0,8-8 % de flags sur seahorse → rejeté ;
//! - propagation LINÉARISÉE (`E' = 2|z|·E + local`) : 0/33 détecté → les
//!   pixels faux ont une erreur de premier ordre ~1e-11 pixel ; c'est le
//!   terme E² qui révèle la perte de détermination. Ne pas linéariser.
//! D'où le critère retenu : emballement seul, SANS dérivée (une
//! multiplication complexe et une chaîne de dépendance de moins par
//! itération).
//!
//! Le traqueur s'injecte par monomorphisation ([`StepObserver`]) : la boucle de
//! production instancie [`NoObserve`] (ZST, hooks vides) → code identique à
//! l'avant-détecteur, bit-identique (verrous goldens).

use num_complex::Complex64;

use super::bla_dual::Mat2;

/// Epsilon machine f64 (u = 2⁻⁵³).
pub const U: f64 = 1.0 / 9_007_199_254_740_992.0;
/// Constante d'arrondi locale d'un pas complexe (quelques ulp par composante).
const GAMMA: f64 = 4.0 * U;

/// Hooks appelés par la boucle pixel sur ses trois événements. Les
/// implémentations par défaut sont vides : [`NoObserve`] monomorphise en la
/// boucle d'origine.
pub trait StepObserver: Copy {
    /// Pas direct : `z_norm_sqr` = `|Z[m] + δ_old|²` (état avant le pas),
    /// `two_zm_delta` = `2·Z[m]·δ_old` (terme dominant calculé par la boucle),
    /// `delta_norm_sqr` = `|δ_old|²`. Les deux normes sont celles que la
    /// boucle garde déjà en cache de tête : le hook ne les recalcule pas.
    #[inline(always)]
    fn direct(&mut self, _z_norm_sqr: f64, _two_zm_delta: Complex64, _delta_norm_sqr: f64) {}

    /// Saut BLA accepté de `l` itérations : `delta_new = A·δ_old + B·dc`.
    #[inline(always)]
    fn bla(&mut self, _a: &Mat2, _b: &Mat2, _a_delta: Complex64, _b_dc: Complex64, _l: u32) {}

    /// Rebase `δ ← Z + δ` : `z_ref` = la valeur f64 de la référence absorbée,
    /// `z_new` = nouvel état (== nouveau δ).
    #[inline(always)]
    fn rebase(&mut self, _z_ref: Complex64, _z_new: Complex64) {}
}

/// Observateur nul : la boucle de production.
#[derive(Clone, Copy)]
pub struct NoObserve;
impl StepObserver for NoObserve {}

/// Traqueur de fiabilité : borne d'erreur absolue `E` sur `z`.
#[derive(Clone, Copy, Debug)]
pub struct ShadowTracker {
    /// Borne d'erreur absolue sur `z`.
    pub err: f64,
    /// `|dc|` du pixel (terme d'arrondi constant du pas direct).
    dc_abs: f64,
    /// Epsilon de validité BLA de la table (erreur de troncature par saut).
    bla_epsilon: f64,
}

impl ShadowTracker {
    pub fn new(dc: Complex64, bla_epsilon: f64) -> Self {
        Self {
            err: 0.0,
            dc_abs: dc.norm(),
            bla_epsilon,
        }
    }

    /// Borne finale saturée en `f32` (`+∞` si emballée au-delà de f32 ou
    /// NaN). Le pixel est non fiable ssi `bound ≥ seuil d'emballement`.
    pub fn error_bound(&self) -> f32 {
        if self.err.is_nan() {
            f32::INFINITY
        } else {
            self.err as f32
        }
    }
}

/// Borne L1 du module (`|re| + |im| ≥ |z|`), sans racine carrée.
#[inline(always)]
fn l1(z: Complex64) -> f64 {
    z.re.abs() + z.im.abs()
}

/// Norme spectrale (plus grande valeur singulière) d'une mat2. Fermée : pour
/// `M`, `σ₁² = (F² + √(F⁴ − 4·det²)) / 2` avec `F` la norme de Frobenius. La
/// Frobenius surestimerait de √2 une matrice conforme — facteur qui se
/// COMPOSE de saut en saut et ferait exploser la borne.
///
/// Cas CONFORME (Mandelbrot : `A` représente un complexe `a`, m00 = m11 et
/// m01 = −m10) : `σ₁ = |a|`, une seule racine au lieu de deux — le hook BLA
/// est sur le chemin chaud des scènes à longs sauts (e50).
#[inline]
fn spectral_norm(m: &Mat2) -> f64 {
    if m.m00 == m.m11 && m.m01 == -m.m10 {
        return (m.m00 * m.m00 + m.m10 * m.m10).sqrt();
    }
    let f2 = m.m00 * m.m00 + m.m01 * m.m01 + m.m10 * m.m10 + m.m11 * m.m11;
    let det = m.m00 * m.m11 - m.m01 * m.m10;
    let disc = (f2 * f2 - 4.0 * det * det).max(0.0);
    ((f2 + disc.sqrt()) * 0.5).sqrt()
}

impl StepObserver for ShadowTracker {
    #[inline(always)]
    fn direct(&mut self, z_norm_sqr: f64, two_zm_delta: Complex64, delta_norm_sqr: f64) {
        // Propagation : |(z+e)² − z²| ≤ |e|·(2|z| + |e|).
        // Seul le facteur de PROPAGATION exige le module exact (une
        // surestimation s'y composerait d'itération en itération) ; les termes
        // d'arrondi locaux, additifs, prennent la borne L1 (≤ √2·module) sans
        // racine carrée.
        let z_abs = z_norm_sqr.sqrt();
        let local = GAMMA * (l1(two_zm_delta) + delta_norm_sqr + self.dc_abs);
        self.err = self.err * (2.0 * z_abs + self.err) + local;
    }

    #[inline(always)]
    fn bla(&mut self, a: &Mat2, b: &Mat2, a_delta: Complex64, b_dc: Complex64, l: u32) {
        let _ = b;
        // Troncature cumulée des `l` pas fusionnés (≈ l·ε), cf. le traqueur exp.
        let local = (self.bla_epsilon * l as f64 + GAMMA) * (l1(a_delta) + l1(b_dc));
        self.err = spectral_norm(a) * self.err + local;
    }

    #[inline(always)]
    fn rebase(&mut self, z_ref: Complex64, z_new: Complex64) {
        self.err += U * (l1(z_ref) + l1(z_new));
    }
}

impl ShadowTracker {
    /// Saut LA CONFORME de `l` pas (chemin Harmonic LA : `A` complexe) :
    /// même modèle que le hook `bla` (troncature `l·ε`), mais `σ₁(A) = |A|`
    /// est passé par l'appelant, calculé SANS carré (`hypot`). Les segments
    /// harmonic sont longs : à zoom > 1e150 `|A|` dépasse 1e154 et
    /// `spectral_norm` (m00² + m10²) déborde à +∞ — mesuré mitosis 1e270 :
    /// 100 % des pixels flaggés (borne infinie) au lieu de 0.
    #[inline(always)]
    pub fn la_jump(&mut self, a_abs: f64, a_delta: Complex64, b_dc: Complex64, l: u32) {
        let local = (self.bla_epsilon * l as f64 + GAMMA) * (l1(a_delta) + l1(b_dc));
        self.err = a_abs * self.err + local;
    }
}

// ─── Tier exp (> 1e280) ─────────────────────────────────────────────────────
//
// Même récurrence que [`ShadowTracker`], mais la borne est en `FloatExp` :
// à cette profondeur z frôle 0 à ~1e-1000 près (grazes) et les erreurs
// d'arrondi locales sont du même ordre — une borne f64 sous-dépasserait à 0
// et le traqueur deviendrait aveugle. Mesuré (2026-11-05, rendu exp vs tier
// dd, lui-même == GMP pur sur e318) : 24/35 cas profonds du corpus avaient
// des pixels faux non signalés, jusqu'à 4,8 % (e318).

use super::bla_dual_exp::Mat2Exp;
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};

/// Hooks du pas de la boucle exp haute précision (mirror de [`StepObserver`]).
pub trait StepObserverExp: Copy {
    /// Pas direct : `z` = `Z[m] + δ_old`, `two_z_delta` = `2·Z[m]·δ_old`.
    #[inline(always)]
    fn direct(&mut self, _z: ComplexExp, _two_z_delta: ComplexExp, _delta_old: ComplexExp) {}
    /// Saut BLA accepté `δ' = A·δ + B·dc` de `l` itérations (matrices et
    /// opérandes bruts : la boucle de production ne calcule rien de plus pour
    /// l'observateur nul).
    #[inline(always)]
    fn bla(
        &mut self,
        _a: &Mat2Exp,
        _b: &Mat2Exp,
        _delta_old: ComplexExp,
        _dc: ComplexExp,
        _l: u32,
    ) {
    }
    /// Rebase `δ ← Z + δ`.
    #[inline(always)]
    fn rebase(&mut self, _z_ref: ComplexExp, _z_new: ComplexExp) {}
}

impl StepObserverExp for NoObserve {}

#[inline(always)]
fn l1_exp(z: ComplexExp) -> FloatExp {
    z.re.abs() + z.im.abs()
}

/// Traqueur de fiabilité du tier exp : borne d'erreur absolue `E` en FloatExp.
#[derive(Clone, Copy, Debug)]
pub struct ShadowTrackerExp {
    pub err: FloatExp,
    dc_l1: FloatExp,
    bla_epsilon: f64,
}

impl ShadowTrackerExp {
    pub fn new(dc: ComplexExp, bla_epsilon: f64) -> Self {
        Self {
            err: FloatExp::zero(),
            dc_l1: l1_exp(dc),
            bla_epsilon,
        }
    }

    /// Borne finale saturée en f32 (`+∞` au-delà) — même canal et même seuil
    /// d'emballement que le tier f64 (`DeltaResult::error_bound`).
    pub fn error_bound(&self) -> f32 {
        let v = self.err.to_f64();
        if v.is_nan() {
            f32::INFINITY
        } else {
            v as f32
        }
    }
}

/// Plafond de l'exposant (base 2) de la borne exp : bien au-delà de
/// l'emballement (1e30 ≈ 2¹⁰⁰), bien en deçà du débordement `i32`.
///
/// ⚠️ Sans ce plafond la borne emballée REVENAIT à « fiable » : dans le
/// régime emballé le terme `E²` double l'exposant de `E` à chaque pas
/// (`FloatExp::sqr` = `exponent * 2`) ; ~22 pas après 1e196 l'exposant `i32`
/// déborde, s'enroule (arithmétique enveloppante en release) et la borne
/// retombe à ~1e-14. Mesuré olbaid2 96² : 7/9 pixels faux (jusqu'à 531
/// itérations d'écart vs GMP) avaient une borne emballée puis effondrée.
const EXP_ERR_CAP: i32 = 1 << 16;

impl ShadowTrackerExp {
    /// Borne emballée = COLLANTE : plafonnée à 2^EXP_ERR_CAP, elle reste
    /// au-dessus de tout seuil d'emballement jusqu'à la fin du pixel.
    #[inline(always)]
    fn saturate(&mut self) {
        if self.err.exponent > EXP_ERR_CAP {
            self.err = FloatExp {
                mantissa: 0.5,
                exponent: EXP_ERR_CAP,
            };
        }
    }
}

impl StepObserverExp for ShadowTrackerExp {
    #[inline(always)]
    fn direct(&mut self, z: ComplexExp, two_z_delta: ComplexExp, delta_old: ComplexExp) {
        let z_abs = z.norm_sqr_fexp().sqrt();
        let local = (l1_exp(two_z_delta) + delta_old.norm_sqr_fexp() + self.dc_l1) * GAMMA;
        self.err = self.err * (z_abs * 2.0 + self.err) + local;
        self.saturate();
    }

    #[inline(always)]
    fn bla(&mut self, a: &Mat2Exp, b: &Mat2Exp, d: ComplexExp, dc: ComplexExp, l: u32) {
        let a_delta = ComplexExp {
            re: a.m00 * d.re + a.m01 * d.im,
            im: a.m10 * d.re + a.m11 * d.im,
        };
        let b_dc = ComplexExp {
            re: b.m00 * dc.re + b.m01 * dc.im,
            im: b.m10 * dc.re + b.m11 * dc.im,
        };
        // σ₁(A) : |a| pour une A conforme (Mandelbrot), sinon la borne
        // Frobenius (surestime au plus √2 — cas non conforme hors fast-path).
        let sigma = if a.m00 == a.m11 && a.m01 == FloatExp::zero() - a.m10 {
            (a.m00.sqr() + a.m10.sqr()).sqrt()
        } else {
            (a.m00.sqr() + a.m01.sqr() + a.m10.sqr() + a.m11.sqr()).sqrt()
        };
        // Troncature : CHACUN des `l` pas fusionnés abandonne un terme δ² ≤ ε
        // fois son terme linéaire (rayon de validité F3), transporté jusqu'au
        // bout du saut par les pas suivants ⇒ ≈ l·ε relatif à l'arrivée, pas
        // ε. Avec `ε` seul la borne SOUS-estimait l'erreur réelle d'un
        // facteur ~40 sur les orbites à longs sauts (olbaid2 (12,25), tracé
        // contre GMP pur) : elle ne s'emballait jamais avant l'évasion et les
        // pixels faux passaient pour fiables.
        let trunc = self.bla_epsilon * l as f64 + GAMMA;
        let local = (l1_exp(a_delta) + l1_exp(b_dc)) * trunc;
        self.err = sigma * self.err + local;
        self.saturate();
    }

    #[inline(always)]
    fn rebase(&mut self, z_ref: ComplexExp, z_new: ComplexExp) {
        self.err = self.err + (l1_exp(z_ref) + l1_exp(z_new)) * U;
    }
}

// ─── Tier dd (~106 b) ───────────────────────────────────────────────────────
//
// Même récurrence et même critère (emballement) que [`ShadowTracker`], à
// l'unité d'arrondi double-double. La borne elle-même est calculée en f64 à
// partir des parties hautes (une magnitude n'a pas besoin de 106 bits) : la
// boucle dd fournit des normes/modules f64. Prouvé nécessaire : seahorse 1e6,
// pixel (117,17) — le dd rendait 1838 itérations, GMP convergé 1598, sans
// rien signaler (sensibilité au-delà de 106 bits).
//
// Limite connue : f64 sous-déborde sous ~1e-308 ; au-delà de ~1e290 de zoom
// (tier dd demandé en exp) la borne peut s'annuler et le traqueur devenir
// aveugle — le tier f64 (seul consommateur de l'escalade) s'arrête à 1e280.

/// Unité d'arrondi du double-double (≈ 2⁻¹⁰⁴, quelques ulp de marge sur 2⁻¹⁰⁶).
pub const U_DD: f64 = 4.930_380_657_631_324e-32; // 2^-104
/// Constante d'arrondi locale d'un pas complexe dd.
const GAMMA_DD: f64 = 4.0 * U_DD;

/// Traqueur de fiabilité du tier dd : borne d'erreur absolue `E` sur `z`
/// (f64), unité `U_DD`, ε BLA de la table dd (`DD_BLA_EPSILON`).
#[derive(Clone, Copy, Debug)]
pub struct ShadowTrackerDd {
    pub err: f64,
    dc_abs: f64,
    bla_epsilon: f64,
    /// Seuil d'emballement (borne collante au-delà).
    cap: f64,
}

impl ShadowTrackerDd {
    pub fn new(dc_abs: f64, bla_epsilon: f64) -> Self {
        Self {
            err: 0.0,
            dc_abs,
            bla_epsilon,
            cap: reliability_runaway(),
        }
    }

    /// Pas direct : `z_abs` = |Z[m] + δ_old|, `two_z_delta_l1` ≥ |2·Z·δ_old|,
    /// `delta_norm_sqr` = |δ_old|². Mêmes termes que [`ShadowTracker`].
    #[inline(always)]
    pub fn direct(&mut self, z_abs: f64, two_z_delta_l1: f64, delta_norm_sqr: f64) {
        let local = GAMMA_DD * (two_z_delta_l1 + delta_norm_sqr + self.dc_abs);
        self.err = self.err * (2.0 * z_abs + self.err) + local;
        self.stick();
    }

    /// Saut BLA de `l` pas : `sigma_a` = σ₁(A), `a_delta_l1` ≥ |A·δ|,
    /// `b_dc_l1` ≥ |B·dc|. Troncature cumulée `l·ε` (cf. [`ShadowTrackerExp`]).
    #[inline(always)]
    pub fn bla(&mut self, sigma_a: f64, a_delta_l1: f64, b_dc_l1: f64, l: u32) {
        let local = (self.bla_epsilon * l as f64 + GAMMA_DD) * (a_delta_l1 + b_dc_l1);
        self.err = sigma_a * self.err + local;
        self.stick();
    }

    /// Borne emballée COLLANTE : au-delà du seuil d'emballement elle passe à
    /// `+∞` et y reste (un saut BLA à σ₁ < 1 ne peut plus la faire
    /// redescendre sous le seuil ; `∞·0` donne NaN, lui aussi « emballé »).
    #[inline(always)]
    fn stick(&mut self) {
        if !(self.err < self.cap) {
            self.err = f64::INFINITY;
        }
    }

    /// Rebase `δ ← Z + δ` (arrondi dd de la référence devenu absolu).
    #[inline(always)]
    pub fn rebase(&mut self, z_ref_l1: f64, z_new_l1: f64) {
        self.err += U_DD * (z_ref_l1 + z_new_l1);
        self.stick();
    }

    /// Emballée (≥ seuil ou NaN) ?
    #[inline(always)]
    pub fn runaway(&self, threshold: f64) -> bool {
        !(self.err < threshold)
    }

    /// Borne finale saturée en f32 (`+∞` si NaN), même canal que le f64.
    pub fn error_bound(&self) -> f32 {
        if self.err.is_nan() {
            f32::INFINITY
        } else {
            self.err as f32
        }
    }
}

/// Suivi du tier **dd** : ACTIF par défaut dès que le détecteur l'est ;
/// `FRACTALL_RELIABILITY_DD=0` le coupe (A/B).
pub fn reliability_tracks_dd() -> bool {
    reliability_tracks_dd_inner()
}

/// Fraction par défaut du plafond de l'escalade dd → GMP (cf.
/// [`gmp_escalation_allowed`]).
pub const DEFAULT_GMP_MAX_FRACTION: f64 = 0.01;
/// Plancher absolu du plafond : une poignée de pixels part toujours au GMP.
pub const GMP_MIN_PIXELS: usize = 16;

/// Plafond de l'escalade dd → GMP en FRACTION de pixels de la frame
/// (`FRACTALL_RELIABILITY_GMP_MAX_FRACTION`, ≥ 1 = illimité). Jamais en
/// temps : un budget temps rendrait la sortie dépendante de la machine.
pub fn reliability_gmp_max_fraction() -> f64 {
    use std::sync::OnceLock;
    static F: OnceLock<f64> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("FRACTALL_RELIABILITY_GMP_MAX_FRACTION")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|f| f.is_finite() && *f >= 0.0)
            .unwrap_or(DEFAULT_GMP_MAX_FRACTION)
    })
}

/// Décision TOUT OU RIEN de l'escalade dd → GMP : `n_unreliable` pixels dont
/// la borne dd s'emballe partent au GMP ssi `n ≤ max(16, ⌊f·N⌋)` (ou
/// `f ≥ 1`). Au-delà, aucun n'y part (l'image ne dépend d'aucun ordre de
/// traitement) : ils gardent leur valeur dd de meilleur effort.
/// Motif : le GMP par pixel coûte ~1 µs/itération — lethal_weapon (1,6 M
/// itérations) 48² : 406 px (17,6 %) = 73 s, des heures en 1080p.
pub fn gmp_escalation_allowed(n_unreliable: usize, frame_pixels: usize, max_fraction: f64) -> bool {
    if max_fraction >= 1.0 {
        return true;
    }
    let cap = ((max_fraction * frame_pixels as f64).floor() as usize).max(GMP_MIN_PIXELS);
    n_unreliable <= cap
}

fn reliability_tracks_dd_inner() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        reliability_mode().tracks()
            && !matches!(
                std::env::var("FRACTALL_RELIABILITY_DD").ok().as_deref().map(str::trim),
                Some("0") | Some("off") | Some("false")
            )
    })
}

impl ShadowTrackerExp {
    /// Hook `direct` pour la boucle scaled-double (`pixel_loop_scaled`, δ =
    /// d·2^e, `2·Z·δ = tzd·2^e`, `Z[m]` = `z_ref` f64) : MÊMES opérations, même
    /// ordre, que [`StepObserverExp::direct`], mais en f64 dans le cadre 2^e —
    /// donc la MÊME borne au bit près tant que chaque addition a un partenaire
    /// ≥ 2⁻⁹⁶⁰ (un opérande sous-débordé, < 2⁻¹⁰²², est alors abandonné à
    /// l'identique par FloatExp et par l'arrondi f64) et qu'aucun produit
    /// significatif ne sous-déborde. Gardes non tenues ⇒ `false`, rien n'est
    /// modifié : le caller appelle le hook ComplexExp.
    #[inline(always)]
    pub fn direct_scaled(
        &mut self,
        z_ref: Complex64,
        d: Complex64,
        e: i32,
        tzd: Complex64,
    ) -> bool {
        #[inline(always)]
        fn p2(k: i32) -> f64 {
            if k < -1022 {
                0.0
            } else if k > 1023 {
                f64::INFINITY
            } else {
                f64::from_bits(((k + 1023) as u64) << 52)
            }
        }
        const T2: f64 = f64::from_bits(((-960 + 1023) as u64) << 52);
        const ZBIG: f64 = f64::from_bits(((-700 + 1023) as u64) << 52);
        const MINN: f64 = f64::from_bits(((-1000 + 1023) as u64) << 52);
        #[inline(always)]
        fn big(a: f64, b: f64) -> bool {
            a.abs() >= T2 || b.abs() >= T2
        }
        let (em, ee) = (self.err.mantissa, self.err.exponent);
        let (cm, ce) = (self.dc_l1.mantissa, self.dc_l1.exponent);
        if e > 100 || (em != 0.0 && (ee > 900 || ee - e > 900)) || (cm != 0.0 && ce - e > 900) {
            return false;
        }
        // z = Z + δ (FloatExp, absolu). Pour e < −900, |δ| < 2⁻⁸³⁶ est
        // abandonné devant des composantes de Z ≥ 2⁻⁷⁰⁰.
        let (zr, zi, ok_z) = if e >= -900 {
            let se = p2(e);
            let (dr, di) = (d.re * se, d.im * se);
            (
                z_ref.re + dr,
                z_ref.im + di,
                big(z_ref.re, dr) & big(z_ref.im, di),
            )
        } else {
            let ok = (z_ref.re.abs() >= ZBIG) & (z_ref.im.abs() >= ZBIG);
            (z_ref.re, z_ref.im, ok)
        };
        let (zr2, zi2) = (zr * zr, zi * zi);
        let z_abs = (zr2 + zi2).sqrt();
        // local = (|2Zδ|₁ + |δ|² + |dc|₁)·γ, dans le cadre 2^e.
        let l1 = tzd.re.abs() + tzd.im.abs();
        let nd = (d.re * d.re + d.im * d.im) * p2(e);
        let dcf = if cm == 0.0 { 0.0 } else { cm * p2(ce - e) };
        let s1 = l1 + nd;
        let s2 = s1 + dcf;
        let local = s2 * GAMMA;
        // E' = E·(2|z| + E) + local (E absolu dans le facteur, cadre 2^e sinon).
        let err_abs = if em == 0.0 { 0.0 } else { em * p2(ee) };
        let w = z_abs * 2.0 + err_abs;
        let err_f = if em == 0.0 { 0.0 } else { em * p2(ee - e) };
        let prod = err_f * w;
        let r = prod + local;
        let ok = ok_z
            & big(zr2, zi2)
            & big(l1, nd)
            & big(s1, dcf)
            & big(z_abs, err_abs)
            & (big(prod, local) || (em == 0.0 && local.abs() >= MINN));
        if !ok {
            return false;
        }
        self.err = FloatExp::new(r, e);
        self.saturate();
        true
    }

    /// Hook `bla` pour la boucle scaled-double : `ad` = A·δ (sommes `t1+t2`,
    /// `v1+v2` du saut, exactes sous ses gardes) et `bt` = les quatre termes
    /// `b00·dc.re, b01·dc.im, b10·dc.re, b11·dc.im`, tous dans le cadre 2^ep.
    /// Mêmes opérations que [`StepObserverExp::bla`] ⇒ même borne au bit près
    /// sous gardes ; sinon `false`, rien n'est modifié.
    #[inline(always)]
    pub fn bla_scaled(
        &mut self,
        a: &Mat2Exp,
        ad: Complex64,
        bt: [f64; 4],
        ep: i32,
        l: u32,
    ) -> bool {
        #[inline(always)]
        fn p2(k: i32) -> f64 {
            if k < -1022 {
                0.0
            } else if k > 1023 {
                f64::INFINITY
            } else {
                f64::from_bits(((k + 1023) as u64) << 52)
            }
        }
        const T2: f64 = f64::from_bits(((-960 + 1023) as u64) << 52);
        const MINN: f64 = f64::from_bits(((-1000 + 1023) as u64) << 52);
        const SUB: f64 = f64::MIN_POSITIVE;
        #[inline(always)]
        fn big(a: f64, b: f64) -> bool {
            a.abs() >= T2 || b.abs() >= T2
        }
        #[inline(always)]
        fn ex(x: FloatExp) -> i32 {
            if x.mantissa == 0.0 {
                -(1 << 28)
            } else {
                x.exponent
            }
        }
        // σ₁(A) dans le cadre 2^es (es = plus grand exposant des coefficients
        // sommés) : la plus grande composante y vaut ≥ ½, son carré ≥ ¼ ; les
        // termes sous-débordés (< 2⁻¹⁰²²) sont abandonnés à l'identique.
        let conformal = a.m00 == a.m11 && a.m01 == FloatExp::zero() - a.m10;
        let es = if conformal {
            ex(a.m00).max(ex(a.m10))
        } else {
            ex(a.m00).max(ex(a.m01)).max(ex(a.m10)).max(ex(a.m11))
        };
        if es < -(1 << 27) {
            return false;
        }
        let f = |x: FloatExp| x.mantissa * p2(ex(x) - es);
        let n = if conformal {
            let (x0, x1) = (f(a.m00), f(a.m10));
            x0 * x0 + x1 * x1
        } else {
            let (x0, x1, x2, x3) = (f(a.m00), f(a.m01), f(a.m10), f(a.m11));
            x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
        };
        let sf = n.sqrt();
        // local = (|Aδ|₁ + |B·dc|₁)·(l·ε + γ), cadre 2^ep.
        let l1a = ad.re.abs() + ad.im.abs();
        let (b_re, b_im) = (bt[0] + bt[1], bt[2] + bt[3]);
        let l1b = b_re.abs() + b_im.abs();
        let exact_b = big(bt[0], bt[1]) & big(bt[2], bt[3]) & big(l1a, l1b);
        let negligible_b = (bt[0].abs() < SUB)
            & (bt[1].abs() < SUB)
            & (bt[2].abs() < SUB)
            & (bt[3].abs() < SUB)
            & (l1a >= T2);
        let s = l1a + l1b;
        let trunc = self.bla_epsilon * l as f64 + GAMMA;
        let local = s * trunc;
        // E' = σ·E + local.
        let (em, ee) = (self.err.mantissa, self.err.exponent);
        if em != 0.0 && ee + es - ep > 900 {
            return false;
        }
        let prod = if em == 0.0 {
            0.0
        } else {
            sf * (em * p2(ee + es - ep))
        };
        let r = prod + local;
        let ok =
            (exact_b | negligible_b) & (big(prod, local) || (em == 0.0 && local.abs() >= MINN));
        if !ok {
            return false;
        }
        self.err = FloatExp::new(r, ep);
        self.saturate();
        true
    }
}

/// Mode du détecteur, lu une fois depuis `FRACTALL_RELIABILITY` :
/// - `off` / `0` : détecteur inactif (boucle de production intacte) ;
/// - `observe` : détecte et rapporte (`[RELIABILITY]`), ne corrige rien ;
/// - défaut (`escalate`, `auto`, `1`, unset) : détecte et CORRIGE — pixels
///   non fiables recalculés au tier dd, frame entière en dd au-delà de
///   [`UNRELIABLE_FRAME_THRESHOLD`].
///
/// En échange du coût du suivi et de la correction dd : seahorse 1e8 192²
/// WARN (max_diff 437) → PASS pixel-exact vs GMP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReliabilityMode {
    Off,
    Observe,
    Escalate,
}

impl ReliabilityMode {
    pub fn tracks(self) -> bool {
        !matches!(self, ReliabilityMode::Off)
    }
}

/// Suivi du tier **exp** (> 1e280) — ACTIF PAR DÉFAUT depuis 2026-11-06
/// (`FRACTALL_RELIABILITY_EXP=0|off|false` le coupe ; `FRACTALL_RELIABILITY=off`
/// coupe tous les tiers).
///
/// Historique : opt-in le 2026-11-05 car « +25-45 % pour 0-2 pixels
/// détectés, les résidus (3-9 px vs dd) ont une borne qui ne s'emballe pas ».
/// Ces résidus venaient de TROIS défauts du traqueur : débordement d'exposant
/// qui blanchissait la borne emballée, troncature BLA comptée ε au lieu de
/// l·ε, et pas de test de détermination à l'évasion (`escape_undetermined`).
/// Corrigés : 31 cas profonds 96² pixel-exacts vs dd (12 avaient 1-9 pixels
/// faux) ; à 256², sans traqueur e401 6, olbaid2 48, e1298 18 pixels faux →
/// 0 avec, pour +24-28 % de temps (e1000 inchangé).
pub fn reliability_tracks_exp() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        reliability_mode().tracks()
            && !matches!(
                std::env::var("FRACTALL_RELIABILITY_EXP")
                    .ok()
                    .map(|v| v.trim().to_ascii_lowercase())
                    .as_deref(),
                Some("0") | Some("off") | Some("false")
            )
    })
}

pub fn reliability_mode() -> ReliabilityMode {
    use std::sync::OnceLock;
    static MODE: OnceLock<ReliabilityMode> = OnceLock::new();
    *MODE.get_or_init(|| parse_mode(std::env::var("FRACTALL_RELIABILITY").ok().as_deref()))
}

fn parse_mode(raw: Option<&str>) -> ReliabilityMode {
    match raw.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if s == "0" || s == "off" => ReliabilityMode::Off,
        Some(s) if s == "observe" || s == "obs" => ReliabilityMode::Observe,
        _ => ReliabilityMode::Escalate,
    }
}

/// Seuil d'emballement de la borne (espace-z). Override
/// `FRACTALL_RELIABILITY_RUNAWAY` (calibration). Le régime emballé double
/// l'exposant de `E` à chaque pas : entre 1e30 et le débordement f64 il n'y a
/// que ~3 itérations, le seuil exact est sans effet mesurable — 1e30 tient
/// dans un f32 (canal `error_bound`, kernel GPU f32).
pub fn reliability_runaway() -> f64 {
    use std::sync::OnceLock;
    static RUNAWAY: OnceLock<f64> = OnceLock::new();
    *RUNAWAY.get_or_init(|| {
        std::env::var("FRACTALL_RELIABILITY_RUNAWAY")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|k| k.is_finite() && *k > 0.0)
            .unwrap_or(DEFAULT_RUNAWAY)
    })
}

/// Seuil d'emballement par défaut.
pub const DEFAULT_RUNAWAY: f64 = 1e30;

/// Fraction de pixels non fiables au-delà de laquelle la frame entière est
/// re-rendue au tier dd plutôt que corrigée pixel par pixel.
///
/// Depuis que la correction pixel par pixel est elle-même au tier dd
/// (`correct_pixels_dd`, orbite dd reconstruite sans re-calcul), corriger une
/// fraction f coûte ≈ f × (frame dd) + un surcoût fixe faible : le pixel par
/// pixel est TOUJOURS moins cher, sauf f → 1. L'ancien seuil de 5 % (hérité
/// de la correction GMP, ~25× le dd par itération) aurait déclenché une frame
/// dd jusqu'à 45× plus lente que le f64. NB mesuré : aucun cas du corpus ne
/// tombe entre 5 et 50 % (e50 : 0,46 % en 48×32, 0,28 % en 96×64, 0,38 % en
/// 256² — la fraction ne croît pas à basse résolution sur cette scène) ; le
/// changement est de principe. Le seuil ne reste qu'un filet : au-delà de la moitié de l'image, le
/// décrochage est général et le tier dd couvre aussi les pixels non flaggés.
pub const UNRELIABLE_FRAME_THRESHOLD: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_norm_of_conformal_matrix_is_modulus() {
        // a = 3 + 4i comme mat2 conforme : σ₁ = |a| = 5 (Frobenius = 5√2).
        let m = Mat2 {
            m00: 3.0,
            m01: -4.0,
            m10: 4.0,
            m11: 3.0,
        };
        assert!((spectral_norm(&m) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn spectral_norm_of_diagonal_matrix_is_max_entry() {
        let m = Mat2 {
            m00: 2.0,
            m01: 0.0,
            m10: 0.0,
            m11: -7.0,
        };
        assert!((spectral_norm(&m) - 7.0).abs() < 1e-12);
    }

    /// Verrou du plafond de l'escalade dd → GMP : plafond en fraction,
    /// plancher 16, illimité à f ≥ 1, décision binaire sur le TOTAL.
    #[test]
    fn gmp_escalation_cap_decision() {
        // 256² à 1 % : plafond 655.
        assert!(gmp_escalation_allowed(655, 65536, 0.01));
        assert!(!gmp_escalation_allowed(656, 65536, 0.01));
        // Petite frame : le plancher 16 domine (48² à 1 % = 23 ; 16² → 2).
        assert!(gmp_escalation_allowed(16, 256, 0.01));
        assert!(!gmp_escalation_allowed(17, 256, 0.01));
        assert!(gmp_escalation_allowed(23, 2304, 0.01));
        assert!(!gmp_escalation_allowed(406, 2304, 0.01));
        // f ≥ 1 : illimité ; f = 0 : seul le plancher.
        assert!(gmp_escalation_allowed(1_000_000, 10, 1.0));
        assert!(gmp_escalation_allowed(16, 1 << 20, 0.0));
        assert!(!gmp_escalation_allowed(17, 1 << 20, 0.0));
        assert!(gmp_escalation_allowed(0, 0, 0.01));
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(parse_mode(None), ReliabilityMode::Escalate);
        assert_eq!(parse_mode(Some("auto")), ReliabilityMode::Escalate);
        assert_eq!(parse_mode(Some("escalate")), ReliabilityMode::Escalate);
        assert_eq!(parse_mode(Some("OFF")), ReliabilityMode::Off);
        assert_eq!(parse_mode(Some("0")), ReliabilityMode::Off);
        assert_eq!(parse_mode(Some("observe")), ReliabilityMode::Observe);
    }

    /// Orbite intérieure attractive (c = -0.1+0.1i, cycle stable) : la
    /// borne reste bornée et minuscule — pas d'emballement sur 10⁴ pas.
    #[test]
    fn bound_stays_small_on_attracting_orbit() {
        let c = Complex64::new(-0.1, 0.1);
        let mut t = ShadowTracker::new(c, 0.0);
        let mut z = Complex64::new(0.0, 0.0);
        for _ in 0..10_000 {
            // Référence nulle : δ = z, Z = 0 → 2Zδ = 0.
            t.direct(z.norm_sqr(), Complex64::new(0.0, 0.0), z.norm_sqr());
            z = z * z + c;
        }
        assert!(t.err < 1e-12, "E = {}", t.err);
    }

    /// Une erreur initiale comparable à |z| s'emballe au-delà du seuil.
    #[test]
    fn bound_runs_away_once_comparable_to_z() {
        let c = Complex64::new(-0.1, 0.1);
        let mut t = ShadowTracker::new(c, 0.0);
        t.err = 1.0;
        let mut z = Complex64::new(0.0, 0.0);
        for _ in 0..20 {
            t.direct(z.norm_sqr(), Complex64::new(0.0, 0.0), z.norm_sqr());
            z = z * z + c;
        }
        assert!(t.error_bound() as f64 >= DEFAULT_RUNAWAY);
    }

    /// Tier f64 : même règle l·ε que le tier exp. Sans elle, à 256² le
    /// rendu de production avait des pixels faux NON flaggés vs GMP pur
    /// (e227 384, golden_spider 169, rug 86, e50 10).
    #[test]
    fn f64_bla_truncation_scales_with_jump_length() {
        let eps = U;
        let id = Mat2 {
            m00: 1.0,
            m01: 0.0,
            m10: 0.0,
            m11: 1.0,
        };
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let mut t = ShadowTracker::new(zero, eps);
        t.bla(&id, &id, one, zero, 1024);
        assert!(t.err >= 1024.0 * eps, "borne {:e} < l·ε", t.err);
        let mut t1 = ShadowTracker::new(zero, eps);
        t1.bla(&id, &id, one, zero, 1);
        assert!(t1.err < 8.0 * eps);
    }

    /// Tier exp : un saut BLA de `l` itérations cumule la troncature de ses
    /// `l` pas (≈ l·ε relatif), pas d'un seul. Avec ε seul, la borne sous-
    /// estimait l'erreur réelle ~40× (olbaid2 (12,25) tracé vs GMP pur).
    #[test]
    fn exp_bla_truncation_scales_with_jump_length() {
        let eps = 1.0 / (1u64 << 53) as f64;
        let one = ComplexExp::from_complex64(Complex64::new(1.0, 0.0));
        let zero = ComplexExp::zero();
        let a = Mat2Exp::identity();
        let mut t = ShadowTrackerExp::new(zero, eps);
        t.bla(&a, &a, one, zero, 1024);
        let e = t.err.to_f64();
        assert!(e >= 1024.0 * eps, "borne {e:e} < l·ε = {:e}", 1024.0 * eps);
        let mut t1 = ShadowTrackerExp::new(zero, eps);
        t1.bla(&a, &a, one, zero, 1);
        assert!(t1.err.to_f64() < 8.0 * eps);
    }

    /// Tier exp : une borne emballée doit RESTER emballée. Le régime `E²`
    /// double l'exposant FloatExp à chaque pas ; sans plafond, l'exposant
    /// `i32` déborde après ~30 pas, s'enroule, et la borne retombe sous le
    /// seuil — le pixel faux était alors déclaré fiable (olbaid2 : 7/9 pixels
    /// faux ainsi blanchis).
    #[test]
    fn exp_bound_stays_runaway_through_exponent_overflow() {
        let dc = ComplexExp::from_complex64(Complex64::new(1e-300, 0.0));
        let mut t = ShadowTrackerExp::new(dc, 0.0);
        t.err = FloatExp::from_f64(2.0);
        let z = ComplexExp::from_complex64(Complex64::new(0.5, 0.0));
        let zero = ComplexExp::zero();
        for step in 0..200 {
            t.direct(z, zero, zero);
            assert!(
                t.error_bound() as f64 >= DEFAULT_RUNAWAY || step < 8,
                "pas {step} : borne retombée à {:e} (exposant {})",
                t.error_bound(),
                t.err.exponent
            );
        }
        // Les autres hooks ne la font pas retomber non plus.
        let a = Mat2Exp::identity();
        t.bla(&a, &a, zero, zero, 1);
        t.rebase(z, z);
        assert!(t.error_bound() as f64 >= DEFAULT_RUNAWAY);
    }
}

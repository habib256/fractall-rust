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

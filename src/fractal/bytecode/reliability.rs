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
    /// Pas direct : `z` = `Z[m] + δ_old` (état avant le pas), `two_zm_delta` =
    /// `2·Z[m]·δ_old` (terme dominant calculé par la boucle), `delta_old`.
    #[inline(always)]
    fn direct(&mut self, _z: Complex64, _two_zm_delta: Complex64, _delta_old: Complex64) {}

    /// Saut BLA accepté : `delta_new = A·δ_old + B·dc`.
    #[inline(always)]
    fn bla(&mut self, _a: &Mat2, _b: &Mat2, _a_delta: Complex64, _b_dc: Complex64) {}

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
#[inline]
fn spectral_norm(m: &Mat2) -> f64 {
    let f2 = m.m00 * m.m00 + m.m01 * m.m01 + m.m10 * m.m10 + m.m11 * m.m11;
    let det = m.m00 * m.m11 - m.m01 * m.m10;
    let disc = (f2 * f2 - 4.0 * det * det).max(0.0);
    ((f2 + disc.sqrt()) * 0.5).sqrt()
}

impl StepObserver for ShadowTracker {
    #[inline(always)]
    fn direct(&mut self, z: Complex64, two_zm_delta: Complex64, delta_old: Complex64) {
        // Propagation : |(z+e)² − z²| ≤ |e|·(2|z| + |e|).
        // Seul le facteur de PROPAGATION exige le module exact (une
        // surestimation s'y composerait d'itération en itération) ; les termes
        // d'arrondi locaux, additifs, prennent la borne L1 (≤ √2·module) sans
        // racine carrée.
        let z_abs = z.norm_sqr().sqrt();
        let local = GAMMA
            * (l1(two_zm_delta) + delta_old.norm_sqr() + self.dc_abs);
        self.err = self.err * (2.0 * z_abs + self.err) + local;
    }

    #[inline(always)]
    fn bla(&mut self, a: &Mat2, b: &Mat2, a_delta: Complex64, b_dc: Complex64) {
        let _ = b;
        let local = (self.bla_epsilon + GAMMA) * (l1(a_delta) + l1(b_dc));
        self.err = spectral_norm(a) * self.err + local;
    }

    #[inline(always)]
    fn rebase(&mut self, z_ref: Complex64, z_new: Complex64) {
        self.err += U * (l1(z_ref) + l1(z_new));
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
/// re-rendue au tier dd plutôt que corrigée pixel par pixel en GMP.
pub const UNRELIABLE_FRAME_THRESHOLD: f64 = 0.05;

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
            t.direct(z, Complex64::new(0.0, 0.0), z);
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
            t.direct(z, Complex64::new(0.0, 0.0), z);
            z = z * z + c;
        }
        assert!(t.error_bound() as f64 >= DEFAULT_RUNAWAY);
    }
}

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
//! **Idée.** Une orbite bruitée est correcte au sens pixel tant qu'elle
//! « ombre » une vraie orbite d'un `c` voisin : son erreur, ramenée en espace-c
//! par la dérivée, doit rester sous une fraction du pixel :
//!
//! ```text
//! fiable ⟺ E_n < κ · pixel_size · |dz_n/dc|
//! ```
//!
//! `E` = borne d'erreur absolue sur `z = Z + δ`, propagée par récurrence
//! (`u = 2⁻⁵³`) sur les trois événements de la boucle :
//! - pas direct `δ' = 2Zδ + δ² + dc` : `E ← E·(2|z| + E) + γ·(|2Zδ| + |δ|² + |dc|)` ;
//! - saut BLA `δ' = Aδ + B·dc` : `E ← σ₁(A)·E + (ε_bla + γ)·(|Aδ| + |B·dc|)` ;
//! - rebase `δ ← Z + δ` : `E ← E + u·(|Z| + |z|)` — l'arrondi f64 de la
//!   référence, inoffensif tant que δ est relatif à Z, devient une erreur
//!   ABSOLUE de l'état (c'est exactement le mécanisme d'e13).
//!
//! La dérivée suit `D' = 2zD + 1` (pas direct), `D' = A·D + B` (BLA), et est
//! invariante au rebase (`d(Z+δ)/dc = dδ/dc`).
//!
//! **Ce qui sépare réellement (mesuré 2026-10-02)** : le terme NON LINÉAIRE
//! `E²` de la propagation. Les pixels faux vs GMP ont une erreur de premier
//! ordre minuscule en espace-c (~5e-11 pixel — en propagation linéarisée
//! `E' = 2|z|·E + local`, AUCUN n'est détecté) ; ils sont faux parce que z
//! varie si vite à l'intérieur du pixel (`pixel·|dz/dc| ≫ |z|`) qu'un
//! décalage de 1e-11 pixel change déjà le compte d'itération. La borne y
//! dépasse |z|, le terme E² l'emballe jusqu'à ∞ : le résultat au centre
//! exact n'est plus DÉTERMINÉ par l'arithmétique. D'où la distribution
//! bimodale (≈1e-11 pixel ou ∞) et un κ insensible sur 5 ordres de grandeur.
//! Ne pas « simplifier » la propagation en linéaire.
//!
//! **Pourquoi ça sépare là où cbits échouait** : la borne est propagée
//! multiplicativement ET normalisée par la dérivée. Une cancellation au début
//! d'une orbite très expansive (deep zoom, |D| énorme) ne pèse rien en
//! espace-c ; la même cancellation sur une orbite faiblement expansive pèse un
//! pixel entier.
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

/// Traqueur de shadowing : borne d'erreur `E` + dérivée `D = dz/dc`.
#[derive(Clone, Copy, Debug)]
pub struct ShadowTracker {
    /// Borne d'erreur absolue sur `z`.
    pub err: f64,
    /// Dérivée `dz/dc` (Mandelbrot-like : `D₀ = 0`).
    pub deriv: Complex64,
    /// `|dc|` du pixel (terme d'arrondi constant du pas direct).
    dc_abs: f64,
    /// Epsilon de validité BLA de la table (erreur de troncature par saut).
    bla_epsilon: f64,
}

impl ShadowTracker {
    pub fn new(dc: Complex64, bla_epsilon: f64) -> Self {
        Self {
            err: 0.0,
            deriv: Complex64::new(0.0, 0.0),
            dc_abs: dc.norm(),
            bla_epsilon,
        }
    }

    /// Erreur ramenée en espace-c : `E / |dz/dc|`. `+∞` si la borne a
    /// débordé ou si la dérivée est nulle avec une erreur non nulle.
    pub fn c_space_error(&self) -> f64 {
        if !self.err.is_finite() {
            return f64::INFINITY;
        }
        if self.err == 0.0 {
            return 0.0;
        }
        let d = self.deriv.norm();
        if d.is_finite() {
            if d > 0.0 {
                self.err / d
            } else {
                f64::INFINITY
            }
        } else {
            // Dérivée débordée mais erreur finie : expansion énorme, l'erreur
            // est négligeable en espace-c.
            0.0
        }
    }

    /// Erreur en espace-c exprimée en pixels (`E / |D| / pixel_size`),
    /// saturée en `f32` (`+∞` si non bornée). Le pixel est fiable ssi
    /// `ratio ≤ κ`.
    pub fn shadow_ratio(&self, pixel_size: f64) -> f32 {
        // Diagnostic de calibration (`STUDY_ZSPACE=1`, lu par l'étude
        // `quality::reliability_study`) : renvoie la borne BRUTE en espace-z,
        // pour tester un critère sans dérivée (`E ≥ τ`).
        let r = if study_zspace() {
            self.err
        } else {
            self.c_space_error() / pixel_size
        };
        if r.is_nan() {
            f32::INFINITY
        } else {
            r as f32
        }
    }
}

/// Borne L1 du module (`|re| + |im| ≥ |z|`), sans racine carrée.
#[inline(always)]
fn l1(z: Complex64) -> f64 {
    z.re.abs() + z.im.abs()
}

fn study_zspace() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("STUDY_ZSPACE").is_some())
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
        // Dérivée : D' = 2·z·D + 1.
        let two_z = z * 2.0;
        self.deriv = two_z * self.deriv + Complex64::new(1.0, 0.0);
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
        // Dérivée : D' = A·D + B·1 (colonne 0 de B = image de dc = 1).
        let d = self.deriv;
        self.deriv = Complex64::new(
            a.m00 * d.re + a.m01 * d.im + b.m00,
            a.m10 * d.re + a.m11 * d.im + b.m10,
        );
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
/// Coût mesuré (seahorse 1e8, 512²) : suivi +35 % sur la boucle f64
/// (2,2 → 2,9 ns/iter), correction dd des 0,45 % de pixels flaggés ≈ +0,3 s.
/// En échange : WARN (max_diff 437) → PASS pixel-exact vs GMP.
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

/// κ du test de shadowing (fraction de pixel tolérée en espace-c). Override
/// `FRACTALL_RELIABILITY_KAPPA` (calibration).
pub fn reliability_kappa() -> f64 {
    use std::sync::OnceLock;
    static KAPPA: OnceLock<f64> = OnceLock::new();
    *KAPPA.get_or_init(|| {
        std::env::var("FRACTALL_RELIABILITY_KAPPA")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|k| k.is_finite() && *k > 0.0)
            .unwrap_or(DEFAULT_KAPPA)
    })
}

/// Valeur par défaut de κ (calibrée sur les presets QA, cf. tests).
pub const DEFAULT_KAPPA: f64 = 0.5;

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

    /// Le traqueur reproduit la dérivée exacte d'une orbite Mandelbrot sans
    /// rebase : D_n = dz_n/dc (différences finies sur c).
    #[test]
    fn derivative_matches_finite_difference() {
        let c = Complex64::new(-0.1, 0.65);
        let h = 1e-7;
        let orbit = |c: Complex64, n: usize| {
            let mut z = Complex64::new(0.0, 0.0);
            for _ in 0..n {
                z = z * z + c;
            }
            z
        };
        let mut t = ShadowTracker::new(c, 0.0);
        let mut z = Complex64::new(0.0, 0.0);
        for _ in 0..20 {
            // Référence nulle : δ = z, Z = 0 → 2Zδ = 0.
            t.direct(z, Complex64::new(0.0, 0.0), z);
            z = z * z + c;
        }
        let fd = (orbit(c + h, 20) - orbit(c - h, 20)) / (2.0 * h);
        let rel = (t.deriv - fd).norm() / fd.norm();
        assert!(rel < 1e-5, "D={:?} fd={:?}", t.deriv, fd);
    }
}

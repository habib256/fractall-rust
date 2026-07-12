//! Wisdom auto-dispatch — sélection automatique de l'algorithme et du **tier
//! numérique** (f64 / ComplexExp / double-double) à partir des paramètres de
//! frame, sur le modèle du `wisdom_lookup` de Fraktaler-3 (`wisdom.cc:240`).
//!
//! # Modèle F3
//!
//! F3 filtre les types numériques par **viabilité** (statique) puis choisit le
//! plus rapide (mesuré). Un type de mantisse `M` bits et d'exposant `E` bits est
//! viable si :
//!
//! - `required_exponent + 16 < 2^E / 2` — la plage d'exposant couvre le zoom ;
//! - `required_precision < M` — la mantisse couvre la précision pixel.
//!
//! où (`render.cc:219`) `pixel_spacing = 4 / (zoom·height)`, et
//! `required_precision = max(|offset|/pixel_spacing, hypot(w,h))`. Pour un rendu
//! **centré** (notre cas : l'orbite référence EST le centre de vue) l'offset est
//! ~0, donc `required_precision ≈ log2(hypot(w,h))` — une dizaine de bits. La
//! conséquence, capitale, est que **la mantisse f64 (53 b) suffit toujours pour
//! une frame centrée** : F3 n'escalade que sur l'**exposant** (profondeur de
//! zoom), jamais sur la mantisse. C'est exactement ce que fait fractall.
//!
//! # Ce que ce module centralise
//!
//! Auparavant la sélection de tier était dupliquée (`bytecode_path_label` ET
//! `try_bytecode_unified_path`, `delta.rs`). Ce module en est désormais la
//! **source unique** ([`number_tier`], [`dd_requested`], [`wants_exp`]) et
//! calcule le [`WisdomPlan`] inspectable (exposant/précision requis, tier,
//! précision GMP de l'orbite) pour le log `[WISDOM]` (env `FRACTALL_WISDOM=1`).
//!
//! # Ce que ce module ne fait PAS (encore)
//!
//! Le tier **dd (~106 b, « float128 »)** reste piloté à la demande
//! (`params.use_dd_tier` / `--dd-tier`). La viabilité F3 ne le réclame jamais
//! pour une frame centrée (mantisse requise ~11 b) : le besoin dd vient d'une
//! **sensibilité/conditionnement par pixel** (amplification de Lyapunov sur des
//! spirales, cf. e13/e30/e50, TODO G3) qu'aucun détecteur cheap fiable ne capte
//! aujourd'hui (le proxy `cbits` a été réfuté sur données). L'auto-firing du dd
//! est donc explicitement hors périmètre ; ce module fournit le point d'accroche
//! (`required_precision` vs `tier.mantissa_bits()`) pour le brancher plus tard.

use crate::fractal::types::{FractalParams, FractalType, PlaneTransform};
use crate::fractal::perturbation::{
    compute_perturbation_precision_bits, effective_pixel_size, log2_zoom,
};
use crate::fractal::perturbation::delta::pixel_size_exp_threshold;
use crate::render::escape_time::{should_use_gmp_reference, should_use_perturbation};

/// Algorithme de rendu escape-time sélectionné pour la frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// f64 standard (zooms faibles, pas de perturbation).
    StandardF64,
    /// Perturbation + BLA (tier numérique dans [`WisdomPlan::tier`]).
    Perturbation,
    /// GMP complet par pixel (garde-fou / types non perturbés profonds).
    ReferenceGmp,
}

/// Tier numérique de la boucle pixel perturbation. Analogue du type numérique
/// choisi par `wisdom_lookup` de F3 (double → floatexp → float128).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberTier {
    /// `Complex64` — 53 b mantisse, exposant f64 (~2^±1023). Le plus rapide.
    F64,
    /// `ComplexExp` — 53 b mantisse, exposant i32 (~2^±2^31). Deep zoom.
    Exp,
    /// `ComplexDDExp` double-double — ~106 b mantisse. Opt-in (sensibilité).
    Dd,
}

impl NumberTier {
    /// Label du path bytecode (aligné sur la sortie `[FRACTALL] path=…`).
    pub fn path_label(self) -> &'static str {
        match self {
            NumberTier::F64 => "bytecode_f64",
            NumberTier::Exp => "bytecode_exp",
            NumberTier::Dd => "bytecode_dd",
        }
    }

    /// Bits de mantisse effectifs du tier (viabilité F3 : `required_precision < M`).
    pub fn mantissa_bits(self) -> u32 {
        match self {
            NumberTier::F64 | NumberTier::Exp => 53,
            NumberTier::Dd => 106,
        }
    }
}

/// Plan complet inspectable pour une frame : algorithme + tier + les grandeurs
/// F3 qui les justifient. Produit par [`plan`]. Purement descriptif — n'altère
/// aucun rendu (les call-sites consomment [`number_tier`]/[`dd_requested`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WisdomPlan {
    pub algorithm: Algorithm,
    /// Tier numérique perturbation, `None` si le path bytecode ne s'applique pas
    /// (type non compilable, multi-phase, `--no-bytecode`) ou hors perturbation.
    pub tier: Option<NumberTier>,
    /// `pixel_spacing` effectif en espace-c (HP-aware ; peut être dénormal).
    pub pixel_size: f64,
    /// `~log2(zoom)` = exposant binaire requis (magnitude). `None` si dégénéré.
    pub required_exponent: Option<f64>,
    /// Mantisse requise F3 (frame centrée ≈ `log2(hypot(w,h))`).
    pub required_precision: u32,
    /// Précision GMP de l'orbite référence (`compute_perturbation_precision_bits`).
    pub reference_precision_bits: u32,
}

/// Le tier dd est-il **demandé** ? (`use_dd_tier` + Mandelbrot escape-time).
///
/// Prédicat statique partagé par [`number_tier`] et le dispatch d'exécution
/// (`try_bytecode_unified_path`). L'exécution effective exige EN PLUS que
/// l'orbite ait la référence dd calculée (`ReferenceOrbit::has_dd`) ; sinon on
/// retombe sur le split exp/f64 par `pixel_size`.
#[inline]
pub fn dd_requested(params: &FractalParams) -> bool {
    params.use_dd_tier && matches!(params.fractal_type, FractalType::Mandelbrot)
}

/// Faut-il le tier ComplexExp plutôt que f64 pour ce `pixel_size` ?
///
/// Seuil `pixel_size < 1e-280` (`PIXEL_SIZE_EXP_THRESHOLD`, override
/// `FRACTALL_EXP_THRESHOLD`). Correspond à la borne de **viabilité d'exposant**
/// F3 pour f64 : en deçà, `δ ~ pixel_size` devient sous-normal (< 2.2e-308) et
/// la mantisse f64 se dégrade (casse observée à safari 1e307). Empiriquement
/// plus conservateur que la borne théorique F3 (~2^1008 ≈ zoom 1e301) — on garde
/// le seuil calibré sur corpus.
#[inline]
pub fn wants_exp(pixel_size: f64) -> bool {
    pixel_size < pixel_size_exp_threshold()
}

/// Tier numérique du path bytecode pour cette frame, `None` si le path bytecode
/// ne s'applique pas (mêmes conditions que l'ancien `bytecode_path_label` :
/// `use_bytecode_engine`, formule compilable, single-phase, `pixel_size > 0`).
///
/// **Source unique** de la sélection de tier. Ordre (identique à F3
/// double→floatexp→float128, hoisté sur dd) : dd demandé > exp (deep) > f64.
pub fn number_tier(params: &FractalParams) -> Option<NumberTier> {
    if !params.use_bytecode_engine {
        return None;
    }
    let formula =
        crate::fractal::bytecode::compile_formula(params.fractal_type, params.multibrot_power)?;
    if formula.phases.len() != 1 {
        // Multi-phase pas encore porté sur le pixel_loop unifié.
        return None;
    }
    let pixel_size = effective_pixel_size(params);
    if pixel_size <= 0.0 {
        return None;
    }
    // Tier dd (~106 b) opt-in : couvre tout le range (le dd loop tourne sans
    // gating de zoom), donc hoisté avant le split exp/f64.
    if dd_requested(params) {
        return Some(NumberTier::Dd);
    }
    if wants_exp(pixel_size) {
        Some(NumberTier::Exp)
    } else {
        Some(NumberTier::F64)
    }
}

/// Mantisse requise F3-style pour une frame centrée : `ceil(log2(hypot(w,h)))`.
/// L'offset centre→réf est nul (l'orbite est construite au centre de vue), donc
/// le terme dominant est la diagonale image. Sert au log `[WISDOM]` et de point
/// d'accroche futur pour l'escalade dd (`required_precision > tier.mantissa`).
fn required_precision_bits(params: &FractalParams) -> u32 {
    let diag = ((params.width as f64).powi(2) + (params.height as f64).powi(2)).sqrt();
    if diag <= 1.0 {
        return 1;
    }
    diag.log2().ceil() as u32
}

/// Sélectionne l'algorithme escape-time (Auto). Compose les gardes existantes
/// `should_use_perturbation`/`should_use_gmp_reference` — même ordre que le
/// dispatcher (`render/escape_time.rs`) : perturbation d'abord, puis GMP, sinon
/// f64. Les types spéciaux (non escape-time) ne passent pas par ici.
fn select_algorithm(params: &FractalParams) -> Algorithm {
    use crate::fractal::types::AlgorithmMode;
    // Perturbation forcée incompatible plane_transform ≠ Mu → fallback (miroir
    // exact de `render/escape_time.rs`). En Auto, `should_use_perturbation`
    // retourne déjà false pour un plane ≠ Mu, donc pas de garde ici.
    if params.plane_transform != PlaneTransform::Mu
        && params.algorithm_mode == AlgorithmMode::Perturbation
    {
        return if params.use_gmp { Algorithm::ReferenceGmp } else { Algorithm::StandardF64 };
    }
    match params.algorithm_mode {
        AlgorithmMode::ReferenceGmp => Algorithm::ReferenceGmp,
        AlgorithmMode::StandardF64 => Algorithm::StandardF64,
        AlgorithmMode::Perturbation => Algorithm::Perturbation,
        AlgorithmMode::Auto => {
            if should_use_perturbation(params, false) {
                Algorithm::Perturbation
            } else if should_use_gmp_reference(params) {
                Algorithm::ReferenceGmp
            } else {
                Algorithm::StandardF64
            }
        }
    }
}

/// Calcule le [`WisdomPlan`] complet pour une frame — algorithme, tier et les
/// grandeurs F3 (exposant/précision requis, précision GMP orbite). Descriptif :
/// aucun effet de bord sur le rendu.
pub fn plan(params: &FractalParams) -> WisdomPlan {
    let algorithm = select_algorithm(params);
    // Le tier n'a de sens qu'en perturbation.
    let tier = if algorithm == Algorithm::Perturbation {
        number_tier(params)
    } else {
        None
    };
    WisdomPlan {
        algorithm,
        tier,
        pixel_size: effective_pixel_size(params),
        required_exponent: log2_zoom(params),
        required_precision: required_precision_bits(params),
        reference_precision_bits: compute_perturbation_precision_bits(params),
    }
}

/// Imprime le plan wisdom sur stderr si `FRACTALL_WISDOM=1` (diagnostic ; ne
/// remplace pas la ligne `[FRACTALL]` finale). Appelé sur le path perturbation.
pub fn log_plan_if_enabled(params: &FractalParams) {
    if !matches!(std::env::var("FRACTALL_WISDOM").as_deref(), Ok("1" | "true")) {
        return;
    }
    let p = plan(params);
    let tier = p.tier.map(|t| t.path_label()).unwrap_or("legacy");
    eprintln!(
        "[WISDOM] type={:?} algo={:?} tier={} req_exp={} req_prec={}b tier_mantissa={}b ref_prec={}b pixel_size={:.3e}",
        params.fractal_type,
        p.algorithm,
        tier,
        p.required_exponent.map(|e| format!("{:.1}", e)).unwrap_or_else(|| "-".into()),
        p.required_precision,
        p.tier.map(|t| t.mantissa_bits()).unwrap_or(0),
        p.reference_precision_bits,
        p.pixel_size,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::default_params_for_type;
    use crate::fractal::types::AlgorithmMode;

    /// Frame Mandelbrot centrée à un zoom donné (span = 4/zoom sur les deux axes).
    fn frame(zoom: f64) -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 256, 256);
        p.center_x = -0.75;
        p.center_y = 0.0;
        p.span_x = 4.0 / zoom;
        p.span_y = 4.0 / zoom;
        p.algorithm_mode = AlgorithmMode::Auto;
        p
    }

    #[test]
    fn tier_matches_legacy_thresholds_across_zoom_sweep() {
        // La sélection de tier via `number_tier` doit reproduire EXACTEMENT
        // l'ancienne logique dupliquée : dd demandé > exp (pixel<1e-280) > f64.
        for exp in [2, 6, 12, 13, 20, 50, 100, 200, 279] {
            let z = 10f64.powi(exp);
            let p = frame(z);
            let pixel_size = effective_pixel_size(&p);
            let expected = if pixel_size < pixel_size_exp_threshold() {
                NumberTier::Exp
            } else {
                NumberTier::F64
            };
            assert_eq!(
                number_tier(&p),
                Some(expected),
                "zoom 1e{exp} pixel_size={pixel_size:e}"
            );
        }
    }

    #[test]
    fn dd_requested_hoists_over_exp_and_f64() {
        // `use_dd_tier` sur Mandelbrot force le tier dd à tout zoom (hoisté).
        for exp in [8, 12, 50] {
            let mut p = frame(10f64.powi(exp));
            p.use_dd_tier = true;
            assert_eq!(number_tier(&p), Some(NumberTier::Dd), "zoom 1e{exp}");
        }
        // dd non réclamé pour un type non-Mandelbrot même si use_dd_tier.
        let mut p = frame(1e8);
        p.fractal_type = FractalType::BurningShip;
        p.use_dd_tier = true;
        assert!(!dd_requested(&p));
        assert_eq!(number_tier(&p), Some(NumberTier::F64));
    }

    #[test]
    fn no_bytecode_engine_yields_no_tier() {
        let mut p = frame(1e20);
        p.use_bytecode_engine = false;
        assert_eq!(number_tier(&p), None);
    }

    #[test]
    fn algorithm_follows_perturbation_gate() {
        // Shallow → f64 ; deep (> ~1e12) → perturbation.
        assert_eq!(plan(&frame(1e3)).algorithm, Algorithm::StandardF64);
        assert_eq!(plan(&frame(1e14)).algorithm, Algorithm::Perturbation);
        // Forcé.
        let mut p = frame(1e3);
        p.algorithm_mode = AlgorithmMode::ReferenceGmp;
        assert_eq!(plan(&p).algorithm, Algorithm::ReferenceGmp);
    }

    #[test]
    fn required_precision_is_small_for_centered_frames() {
        // La conséquence F3 clé : mantisse requise ≪ 53 b → f64 toujours viable
        // sur la mantisse ; l'escalade se fait sur l'exposant (exp), pas dd.
        let p = frame(1e50);
        let plan = plan(&p);
        assert!(
            plan.required_precision < 20,
            "required_precision={} devrait être ~log2(diag) ≪ 53",
            plan.required_precision
        );
    }
}
